"""
Monitoring Alert Webhook Handler
Receives Prometheus Alertmanager webhooks and triggers GitHub Actions / CI/CD.
Add this endpoint to your FastAPI app or run as a standalone service.
"""

import json
import os
import time
import logging
import subprocess
from typing import Optional
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alert Webhook Handler", version="1.0.0")

# ── Config from environment ───────────────────────────────────────────────────
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN",   "")
GITHUB_OWNER   = os.getenv("GITHUB_OWNER",   "your-username")
GITHUB_REPO    = os.getenv("GITHUB_REPO",    "fraud-detection")
KFP_ENDPOINT   = os.getenv("KFP_ENDPOINT",   "http://localhost:8888")

ALERT_LOG_PATH = os.getenv("ALERT_LOG_PATH", "/tmp/alert_log.jsonl")

# Thresholds that should trigger retraining
RETRAIN_ALERTS = {
    "FraudRecallLow",
    "PredictionDistributionShift",
    "HighFalsePositiveRate",
}


class AlertmanagerPayload(BaseModel):
    """Prometheus Alertmanager webhook payload format."""
    version:           str = "4"
    groupKey:          Optional[str] = None
    status:            str = "firing"
    receiver:          str = ""
    groupLabels:       dict = {}
    commonLabels:      dict = {}
    commonAnnotations: dict = {}
    externalURL:       str = ""
    alerts:            list = []


def log_alert(alert_name: str, status: str, labels: dict, annotations: dict):
    entry = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "alert_name":  alert_name,
        "status":      status,
        "labels":      labels,
        "annotations": annotations,
    }
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Alert logged: {alert_name} [{status}]")


def trigger_github_actions(trigger_reason: str, recall_value: str = ""):
    """Trigger GitHub Actions workflow_dispatch via API."""
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — skipping GitHub Actions trigger")
        return False

    url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/actions/workflows/fraud_detection_cicd.yml/dispatches"
    )
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
    }
    payload = {
        "ref": "main",
        "inputs": {
            "trigger_reason": trigger_reason,
            "recall_value":   recall_value,
        },
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 204:
            logger.info(f"✅ GitHub Actions triggered: reason={trigger_reason}")
            return True
        else:
            logger.error(f"GitHub Actions trigger failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"GitHub Actions trigger error: {e}")
        return False


def trigger_local_retraining(trigger_reason: str, recall_value: str = ""):
    """Trigger local retraining pipeline if GitHub not available."""
    logger.info(f"Triggering local retraining: {trigger_reason}")
    try:
        subprocess.Popen([
            "python", "scripts/trigger_pipeline.py",
            "--endpoint",       KFP_ENDPOINT,
            "--pipeline_file",  "fraud_detection_pipeline.yaml",
            "--run_name",       f"alert-retrain-{int(time.time())}",
            "--trigger_reason", trigger_reason,
        ])
        return True
    except Exception as e:
        logger.error(f"Local retraining trigger failed: {e}")
        return False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/alerts/webhook")
async def handle_alert_webhook(request: Request):
    """
    Receives Prometheus Alertmanager webhook.
    Fires when alert rules in alert_rules.yml are triggered.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    alerts = body.get("alerts", [])
    actions_taken = []

    for alert in alerts:
        alert_name = alert.get("labels", {}).get("alertname", "unknown")
        status     = alert.get("status", "firing")
        labels     = alert.get("labels", {})
        annotations = alert.get("annotations", {})

        logger.info(f"Received alert: {alert_name} [{status}]")
        log_alert(alert_name, status, labels, annotations)

        if status == "firing" and alert_name in RETRAIN_ALERTS:
            # Determine trigger reason
            if alert_name == "FraudRecallLow":
                reason       = "recall_drop"
                recall_value = labels.get("current_recall", "")
            elif alert_name == "PredictionDistributionShift":
                reason       = "data_drift"
                recall_value = ""
            else:
                reason       = "model_drift"
                recall_value = ""

            logger.warning(
                f"🚨 ALERT TRIGGERED: {alert_name} → initiating retraining "
                f"(reason={reason})"
            )

            # Try GitHub Actions first, fall back to local
            triggered = trigger_github_actions(reason, recall_value)
            if not triggered:
                triggered = trigger_local_retraining(reason, recall_value)

            actions_taken.append({
                "alert":     alert_name,
                "action":    "retrain_triggered",
                "reason":    reason,
                "triggered": triggered,
            })
        else:
            actions_taken.append({
                "alert":  alert_name,
                "status": status,
                "action": "logged_only",
            })

    return {
        "received":     len(alerts),
        "actions_taken": actions_taken,
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.post("/alerts/drift")
async def handle_drift_alert(request: Request):
    """Specific endpoint for drift alerts."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.warning(f"Drift alert received: {json.dumps(body)[:200]}")
    log_alert("DataDrift", "firing", body.get("labels", {}), {})

    triggered = trigger_github_actions("data_drift")
    if not triggered:
        triggered = trigger_local_retraining("data_drift")

    return {
        "status":    "drift_alert_processed",
        "triggered": triggered,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.get("/alerts/log")
async def get_alert_log(last_n: int = 20):
    """Return last N alert log entries."""
    if not os.path.exists(ALERT_LOG_PATH):
        return {"entries": [], "total": 0}
    with open(ALERT_LOG_PATH) as f:
        lines = f.readlines()
    entries = [json.loads(l) for l in lines[-last_n:]]
    return {"entries": entries, "total": len(lines)}


@app.get("/health")
async def health():
    return {"status": "ok", "github_configured": bool(GITHUB_TOKEN)}


if __name__ == "__main__":
    uvicorn.run("monitoring.webhook_handler:app", host="0.0.0.0", port=8001, reload=False)
