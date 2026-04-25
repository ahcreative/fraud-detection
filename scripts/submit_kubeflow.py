"""
Submit pipeline to Kubeflow running in Minikube.
Run this AFTER starting port-forwarding:
  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def submit_to_kubeflow(
    endpoint: str = "http://localhost:8888",
    data_dir: str = "/data",
    experiment_name: str = "fraud-detection",
    imbalance_strategy: str = "smote",
):
    print(f"\n{'='*60}")
    print("SUBMITTING PIPELINE TO KUBEFLOW (MINIKUBE)")
    print(f"{'='*60}")
    print(f"  KFP Endpoint: {endpoint}")

    try:
        import kfp
        from kfp.client import Client
    except ImportError:
        print("  ❌ kfp not installed. Run: pip install kfp==2.6.0")
        sys.exit(1)

    # Step 1: Compile pipeline
    print("\n  Step 1: Compiling pipeline ...")
    from pipeline.pipeline import fraud_detection_pipeline
    from kfp import compiler

    pipeline_file = "fraud_detection_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path=pipeline_file,
    )
    print(f"  ✅ Compiled: {pipeline_file}")

    # Step 2: Connect to KFP
    print(f"\n  Step 2: Connecting to KFP at {endpoint} ...")
    try:
        client = Client(host=endpoint)
        print("  ✅ Connected")
    except Exception as e:
        print(f"  ❌ Cannot connect to KFP: {e}")
        print("\n  Make sure port-forwarding is running:")
        print("    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80")
        sys.exit(1)

    # Step 3: Create/get experiment
    print(f"\n  Step 3: Setting up experiment '{experiment_name}' ...")
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
        print(f"  Found existing experiment: {experiment.experiment_id}")
    except Exception:
        experiment = client.create_experiment(
            name=experiment_name,
            description="IEEE CIS Fraud Detection — Full MLOps Pipeline",
        )
        print(f"  Created new experiment: {experiment.experiment_id}")

    # Step 4: Submit run
    run_name = f"fraud-run-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"\n  Step 4: Submitting run '{run_name}' ...")

    run = client.create_run_from_pipeline_package(
        pipeline_file=pipeline_file,
        arguments={
            "transaction_path":  os.path.join(data_dir, "train_transaction.csv"),
            "identity_path":     os.path.join(data_dir, "train_identity.csv"),
            "imbalance_strategy": imbalance_strategy,
            "accuracy_threshold": 0.90,
            "recall_threshold":   0.80,
        },
        run_name=run_name,
        experiment_name=experiment_name,
        enable_caching=True,
    )

    print(f"\n  ✅ Run submitted!")
    print(f"  Run ID  : {run.run_id}")
    print(f"  UI URL  : {endpoint}/#/runs/details/{run.run_id}")
    print(f"\n  Monitor progress at: {endpoint}")

    return run.run_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint",           default="http://localhost:8888")
    parser.add_argument("--data_dir",           default="/data")
    parser.add_argument("--experiment",         default="fraud-detection")
    parser.add_argument("--imbalance_strategy", default="smote")
    args = parser.parse_args()

    submit_to_kubeflow(
        endpoint=args.endpoint,
        data_dir=args.data_dir,
        experiment_name=args.experiment,
        imbalance_strategy=args.imbalance_strategy,
    )
