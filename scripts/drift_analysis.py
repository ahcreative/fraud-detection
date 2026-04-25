"""
Drift analysis script — called by CI/CD Stage 4 (intelligent trigger).
Reads monitoring metrics and generates a drift summary report.
"""

import argparse
import json
import os
import sys
import time


def run_drift_analysis(trigger_reason: str, recall_value: str, output_report: str):
    print(f"\n{'='*60}")
    print("DRIFT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Trigger reason : {trigger_reason}")
    print(f"  Recall value   : {recall_value}")

    recall = None
    if recall_value:
        try:
            recall = float(recall_value)
        except ValueError:
            pass

    severity = "LOW"
    actions  = []
    details  = {}

    if trigger_reason == "recall_drop":
        severity = "HIGH" if recall and recall < 0.75 else "MEDIUM"
        actions  = ["retrain_model", "update_threshold"]
        details  = {
            "current_recall":    recall,
            "recall_threshold":  0.80,
            "gap":               round(0.80 - recall, 4) if recall else None,
        }
        print(f"  ⚠️  Recall drop detected: {recall}")

    elif trigger_reason == "model_drift":
        severity = "HIGH"
        actions  = ["retrain_model", "run_drift_simulation"]
        details  = {"drift_type": "model_performance", "psi_exceeded": True}
        print("  ⚠️  Model drift detected")

    elif trigger_reason == "data_drift":
        severity = "MEDIUM"
        actions  = ["run_drift_simulation", "retrain_model"]
        details  = {"drift_type": "data_distribution", "features_affected": "unknown"}
        print("  ⚠️  Data drift detected")

    else:
        severity = "LOW"
        actions  = ["monitor"]
        print(f"  ℹ️  Manual trigger: {trigger_reason}")

    report = {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger_reason": trigger_reason,
        "severity":       severity,
        "actions":        actions,
        "details":        details,
        "retrain_recommended": severity in ("HIGH", "MEDIUM"),
    }

    os.makedirs(os.path.dirname(output_report) if os.path.dirname(output_report) else ".", exist_ok=True)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Severity : {severity}")
    print(f"  Actions  : {actions}")
    print(f"  Report   : {output_report}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger_reason", default="manual")
    parser.add_argument("--recall_value",   default="")
    parser.add_argument("--output_report",  default="drift_report.json")
    args = parser.parse_args()
    run_drift_analysis(args.trigger_reason, args.recall_value, args.output_report)


if __name__ == "__main__":
    main()
