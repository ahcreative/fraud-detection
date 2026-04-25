"""
Fraud Detection Inference API
FastAPI server with:
- /predict endpoint
- /health endpoint  
- Prometheus metrics exposure
- Request logging for drift detection
"""

import json
import os
import pickle
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API request latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total predictions",
    ["prediction"]  # "fraud" | "legitimate"
)
FRAUD_RECALL_GAUGE = Gauge(
    "fraud_recall_current",
    "Current fraud recall estimate"
)
PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence",
    "Distribution of fraud probability scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
FALSE_POSITIVE_RATE = Gauge(
    "fraud_false_positive_rate",
    "Current estimated false positive rate"
)
MODEL_VERSION = Gauge(
    "fraud_model_version",
    "Deployed model version"
)

# ── Global model state ────────────────────────────────────────────────────────
MODEL = None
PREPROCESSOR = None
FEATURE_CONFIG = None
SERVING_DIR = os.getenv("SERVING_DIR", "/app/serving")
PREDICTION_LOG = []  # In-memory log (use Redis/DB in production)


# ── Schemas ───────────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    TransactionDT: Optional[float] = None
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    # C, D, V fields — accept anything
    class Config:
        extra = "allow"


class PredictionResponse(BaseModel):
    transaction_id: Optional[str] = None
    fraud_probability: float
    is_fraud: bool
    threshold: float
    confidence: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]
    serving_dir: str


# ── Model loading ─────────────────────────────────────────────────────────────
def load_serving_artifacts():
    global MODEL, PREPROCESSOR, FEATURE_CONFIG
    try:
        model_path        = os.path.join(SERVING_DIR, "model.pkl")
        preprocessor_path = os.path.join(SERVING_DIR, "preprocessor.pkl")
        config_path       = os.path.join(SERVING_DIR, "feature_config.json")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                MODEL = pickle.load(f)
            logger.info(f"✅ Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}")

        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                PREPROCESSOR = pickle.load(f)
            logger.info("✅ Preprocessor loaded")

        if os.path.exists(config_path):
            with open(config_path) as f:
                FEATURE_CONFIG = json.load(f)
            logger.info(f"✅ Feature config loaded — {len(FEATURE_CONFIG.get('final_features', []))} features")

        MODEL_VERSION.set(1.0)

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_serving_artifacts()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="IEEE CIS Fraud Detection - Real-time Inference",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    meta_path = os.path.join(SERVING_DIR, "deployment_metadata.json")
    model_name = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_name = json.load(f).get("model_name")
    return HealthResponse(
        status="ok" if MODEL is not None else "model_not_loaded",
        model_loaded=MODEL is not None,
        model_name=model_name,
        serving_dir=SERVING_DIR,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    try:
        # Convert request to DataFrame
        data = request.dict()
        df = pd.DataFrame([data])

        # Apply preprocessing if available
        if PREPROCESSOR is not None:
            from pipeline.components.data_preprocessing.preprocess import (
                impute_missing, encode_categoricals, apply_feature_engineering
            )
            df = apply_feature_engineering(df)
            num_cols = PREPROCESSOR["num_cols"]
            cat_cols = PREPROCESSOR["cat_cols"]
            df, _ = impute_missing(df, num_cols, cat_cols,
                                   imputers=PREPROCESSOR["imputers"], fit=False)
            df, _ = encode_categoricals(df, cat_cols,
                                        encoders=PREPROCESSOR["encoders"], fit=False)
            present_num = [c for c in num_cols if c in df.columns]
            df[present_num] = PREPROCESSOR["scaler"].transform(df[present_num])

        # Align features
        if FEATURE_CONFIG and "final_features" in FEATURE_CONFIG:
            for col in FEATURE_CONFIG["final_features"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[FEATURE_CONFIG["final_features"]]

        df = df.fillna(0)

        # Predict
        if isinstance(MODEL, dict) and "xgb" in MODEL:
            X = MODEL["selector"].transform(df)
            proba = float(MODEL["xgb"].predict_proba(X)[0, 1])
        else:
            proba = float(MODEL.predict_proba(df)[0, 1])

        threshold = 0.5
        is_fraud  = proba >= threshold
        confidence = "high" if abs(proba - 0.5) > 0.3 else "medium" if abs(proba - 0.5) > 0.1 else "low"
        latency_ms = (time.time() - t0) * 1000

        # Update metrics
        FRAUD_PREDICTIONS.labels("fraud" if is_fraud else "legitimate").inc()
        PREDICTION_CONFIDENCE.observe(proba)

        # Log prediction for drift monitoring
        PREDICTION_LOG.append({
            "timestamp": time.time(),
            "proba": proba,
            "is_fraud": is_fraud,
            "transaction_amt": data.get("TransactionAmt"),
        })

        return PredictionResponse(
            fraud_probability=round(proba, 4),
            is_fraud=is_fraud,
            threshold=threshold,
            confidence=confidence,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
def model_info():
    """Return model metadata."""
    meta_path = os.path.join(SERVING_DIR, "deployment_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"status": "no model deployed"}


@app.get("/predictions/stats")
def prediction_stats():
    """Recent prediction statistics for monitoring."""
    if not PREDICTION_LOG:
        return {"total": 0}
    recent = PREDICTION_LOG[-1000:]
    probas = [p["proba"] for p in recent]
    frauds = [p["is_fraud"] for p in recent]
    return {
        "total_recent": len(recent),
        "fraud_rate": sum(frauds) / len(frauds),
        "avg_proba": sum(probas) / len(probas),
        "high_confidence_fraud": sum(1 for p in probas if p > 0.8),
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
