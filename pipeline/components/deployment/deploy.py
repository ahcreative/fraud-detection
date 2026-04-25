"""
Component 7: Conditional Deployment
- Reads deploy decision from evaluation
- Registers model if thresholds met
- Copies model to serving directory
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime


def deploy_model(
    deploy_threshold_path: str,
    best_model_path: str,
    preprocessor_path: str,
    feature_config_path: str,
    serving_dir: str,
):
    print(f"\n{'='*60}")
    print("CONDITIONAL DEPLOYMENT")
    print(f"{'='*60}")

    # Read deploy decision
    with open(deploy_threshold_path) as f:
        decision = json.load(f)

    print(f"  Model    : {decision['best_model']}")
    print(f"  AUC-ROC  : {decision['auc_roc']:.4f} (threshold={decision['auc_roc_threshold']})")
    print(f"  Recall   : {decision['recall']:.4f} (threshold={decision['recall_threshold']})")
    print(f"  Decision : {decision['reason']}")

    if not decision["should_deploy"]:
        print("\n❌ Model does NOT meet deployment thresholds. Skipping deployment.")
        sys.exit(0)  # Non-error exit, pipeline continues

    print("\n✅ Model meets thresholds — deploying ...")

    os.makedirs(serving_dir, exist_ok=True)

    # Copy model artifacts
    shutil.copy(best_model_path,     os.path.join(serving_dir, "model.pkl"))
    shutil.copy(preprocessor_path,   os.path.join(serving_dir, "preprocessor.pkl"))
    shutil.copy(feature_config_path, os.path.join(serving_dir, "feature_config.json"))

    # Write deployment metadata
    metadata = {
        "deployed_at":  datetime.utcnow().isoformat(),
        "model_name":   decision["best_model"],
        "auc_roc":      decision["auc_roc"],
        "recall":       decision["recall"],
        "model_path":   os.path.join(serving_dir, "model.pkl"),
        "status":       "active",
    }
    with open(os.path.join(serving_dir, "deployment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Serving artifacts deployed to: {serving_dir}")
    print(f"  Metadata: {metadata}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_threshold_path", required=True)
    parser.add_argument("--best_model_path",       required=True)
    parser.add_argument("--preprocessor_path",     required=True)
    parser.add_argument("--feature_config_path",   required=True)
    parser.add_argument("--serving_dir",           required=True)
    args = parser.parse_args()

    deploy_model(
        deploy_threshold_path=args.deploy_threshold_path,
        best_model_path=args.best_model_path,
        preprocessor_path=args.preprocessor_path,
        feature_config_path=args.feature_config_path,
        serving_dir=args.serving_dir,
    )


if __name__ == "__main__":
    main()
