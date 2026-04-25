"""
Script to submit a compiled Kubeflow pipeline run via the KFP SDK.
Used by CI/CD (GitHub Actions / Jenkins) and monitoring-triggered retraining.
"""

import argparse
import json
import sys
import time

try:
    import kfp
    from kfp.client import Client
    KFP_AVAILABLE = True
except ImportError:
    KFP_AVAILABLE = False


def trigger_pipeline(
    endpoint: str,
    pipeline_file: str,
    run_name: str,
    experiment_name: str = "fraud-detection",
    trigger_reason: str = "manual",
    namespace: str = "kubeflow",
    params: dict = None,
):
    print(f"\n{'='*60}")
    print("KUBEFLOW PIPELINE TRIGGER")
    print(f"{'='*60}")
    print(f"  Endpoint       : {endpoint}")
    print(f"  Pipeline file  : {pipeline_file}")
    print(f"  Run name       : {run_name}")
    print(f"  Trigger reason : {trigger_reason}")
    print(f"  Namespace      : {namespace}")

    if not KFP_AVAILABLE:
        print("\n  ⚠️  kfp not installed — simulating trigger")
        print(f"  Would submit: {pipeline_file} as run '{run_name}'")
        return {"status": "simulated", "run_name": run_name}

    try:
        client = Client(host=endpoint, namespace=namespace)

        # Get or create experiment
        try:
            experiment = client.get_experiment(experiment_name=experiment_name)
        except Exception:
            experiment = client.create_experiment(
                name=experiment_name,
                description="IEEE CIS Fraud Detection Experiments",
            )

        default_params = {
            "transaction_path":     "/data/train_transaction.csv",
            "identity_path":        "/data/train_identity.csv",
            "imbalance_strategy":   "smote",
            "accuracy_threshold":   0.90,
            "recall_threshold":     0.80,
        }
        if params:
            default_params.update(params)

        run = client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=default_params,
            run_name=run_name,
            experiment_name=experiment_name,
            enable_caching=True,
        )

        print(f"\n  ✅ Pipeline run submitted!")
        print(f"  Run ID : {run.run_id}")
        print(f"  URL    : {endpoint}/#/runs/details/{run.run_id}")

        # Log trigger to file
        log_entry = {
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id":         run.run_id,
            "run_name":       run_name,
            "trigger_reason": trigger_reason,
            "endpoint":       endpoint,
        }
        with open("pipeline_trigger_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {"status": "submitted", "run_id": run.run_id, "run_name": run_name}

    except Exception as e:
        print(f"\n  ❌ Failed to submit pipeline: {e}")
        print("  This is expected if Kubeflow is not reachable from CI environment.")
        print("  In local setup, use: python scripts/trigger_pipeline.py --endpoint http://localhost:8888")
        return {"status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint",       default="http://localhost:8888")
    parser.add_argument("--pipeline_file",  default="fraud_detection_pipeline.yaml")
    parser.add_argument("--run_name",       default=f"run-{int(time.time())}")
    parser.add_argument("--experiment",     default="fraud-detection")
    parser.add_argument("--trigger_reason", default="manual")
    parser.add_argument("--namespace",      default="kubeflow")
    args = parser.parse_args()

    result = trigger_pipeline(
        endpoint=args.endpoint,
        pipeline_file=args.pipeline_file,
        run_name=args.run_name,
        experiment_name=args.experiment,
        trigger_reason=args.trigger_reason,
        namespace=args.namespace,
    )
    print(f"\n  Result: {result}")
    sys.exit(0 if result["status"] in ("submitted", "simulated") else 1)


if __name__ == "__main__":
    main()
