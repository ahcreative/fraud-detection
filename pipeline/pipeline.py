"""
Kubeflow Pipeline Definition
Full fraud detection pipeline with 7 components + conditional deployment.
Uses KFP v2 SDK (kfp==2.6.0).
"""

import os
from kfp import dsl, compiler
from kfp.dsl import Dataset, Model, Metrics, Input, Output, component, pipeline


# ── Base image — all components share same Python environment ─────────────────
BASE_IMAGE = "fraud-detection:latest"


# ════════════════════════════════════════════════════════════════════════════════
# COMPONENT DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════

@component(base_image=BASE_IMAGE, packages_to_install=[])
def data_ingestion_op(
    transaction_path: str,
    identity_path: str,
    merged_data: Output[Dataset],
    is_train: bool = True,
) -> dict:
    """Load and merge transaction + identity CSVs."""
    import sys, json
    sys.path.insert(0, "/app")
    from pipeline.components.data_ingestion.ingest import load_and_merge
    stats = load_and_merge(transaction_path, identity_path, merged_data.path, is_train)
    return stats


@component(base_image=BASE_IMAGE, packages_to_install=[])
def data_validation_op(
    input_data: Input[Dataset],
    validation_report: Output[Dataset],
    is_train: bool = True,
):
    """Validate schema, missing values, target distribution."""
    import sys
    sys.path.insert(0, "/app")
    from pipeline.components.data_validation.validate import validate_data
    validate_data(input_data.path, validation_report.path, is_train)


@component(base_image=BASE_IMAGE, packages_to_install=[])
def preprocessing_op(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    preprocessor: Output[Model],
    imbalance_strategy: str = "smote",
    is_train: bool = True,
):
    """Imputation, encoding, scaling, imbalance handling."""
    import sys, os, pickle
    sys.path.insert(0, "/app")
    from pipeline.components.data_preprocessing.preprocess import preprocess

    output_dir    = os.path.dirname(train_data.path)
    artifacts_dir = os.path.dirname(preprocessor.path)

    preprocess(
        input_path=input_data.path,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        imbalance_strategy=imbalance_strategy,
        is_train=is_train,
    )


@component(base_image=BASE_IMAGE, packages_to_install=[])
def feature_engineering_op(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    engineered_train: Output[Dataset],
    engineered_val: Output[Dataset],
    feature_config: Output[Dataset],
):
    """Feature engineering: aggregations, selection."""
    import sys, os
    sys.path.insert(0, "/app")
    from pipeline.components.feature_engineering.engineer import engineer_features

    engineer_features(
        input_path=train_data.path,
        output_path=engineered_train.path,
        feature_config_path=feature_config.path,
        is_train=True,
    )
    engineer_features(
        input_path=val_data.path,
        output_path=engineered_val.path,
        feature_config_path=feature_config.path,
        is_train=False,
        dropped_cols_path=feature_config.path,
    )


@component(base_image=BASE_IMAGE, packages_to_install=[])
def model_training_op(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    models_dir: Output[Model],
    training_config: Output[Dataset],
):
    """Train XGBoost, LightGBM, and Hybrid models."""
    import sys, os
    sys.path.insert(0, "/app")
    from pipeline.components.model_training.train import train_all_models

    train_all_models(
        train_path=train_data.path,
        val_path=val_data.path,
        models_dir=models_dir.path,
        training_config_path=training_config.path,
    )


@component(base_image=BASE_IMAGE, packages_to_install=[])
def model_evaluation_op(
    val_data: Input[Dataset],
    models_dir: Input[Model],
    eval_metrics: Output[Metrics],
    evaluation_report: Output[Dataset],
    best_model: Output[Model],
    deploy_decision: Output[Dataset],
    accuracy_threshold: float = 0.90,
    recall_threshold: float = 0.80,
):
    """Evaluate all models, select best, make deploy decision."""
    import sys, os
    sys.path.insert(0, "/app")
    from pipeline.components.model_evaluation.evaluate import evaluate_all_models

    output_dir = os.path.join(os.path.dirname(evaluation_report.path), "eval_plots")

    report = evaluate_all_models(
        val_path=val_data.path,
        models_dir=models_dir.path,
        output_dir=output_dir,
        evaluation_report_path=evaluation_report.path,
        best_model_path=best_model.path,
        deploy_threshold_path=deploy_decision.path,
        accuracy_threshold=accuracy_threshold,
        recall_threshold=recall_threshold,
    )

    # Log metrics to Kubeflow UI
    best = report["all_models"][report["best_model"]]
    eval_metrics.log_metric("auc_roc",   best["auc_roc"])
    eval_metrics.log_metric("recall",    best["recall"])
    eval_metrics.log_metric("precision", best["precision"])
    eval_metrics.log_metric("f1",        best["f1"])
    eval_metrics.log_metric("auc_pr",    best["auc_pr"])


@component(base_image=BASE_IMAGE, packages_to_install=[])
def deployment_op(
    deploy_decision: Input[Dataset],
    best_model: Input[Model],
    preprocessor: Input[Model],
    feature_config: Input[Dataset],
    serving_dir: str = "/app/serving",
):
    """Conditional deployment — only runs if thresholds met."""
    import sys
    sys.path.insert(0, "/app")
    from pipeline.components.deployment.deploy import deploy_model

    deploy_model(
        deploy_threshold_path=deploy_decision.path,
        best_model_path=best_model.path,
        preprocessor_path=preprocessor.path,
        feature_config_path=feature_config.path,
        serving_dir=serving_dir,
    )


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE DEFINITION
# ════════════════════════════════════════════════════════════════════════════════

@pipeline(
    name="fraud-detection-pipeline",
    description="IEEE CIS Fraud Detection — Full MLOps Pipeline",
)
def fraud_detection_pipeline(
    transaction_path: str = "/data/train_transaction.csv",
    identity_path:    str = "/data/train_identity.csv",
    imbalance_strategy: str = "smote",
    accuracy_threshold: float = 0.90,
    recall_threshold:   float = 0.80,
):
    # ── Step 1: Data Ingestion ────────────────────────────────────────────────
    ingest = data_ingestion_op(
        transaction_path=transaction_path,
        identity_path=identity_path,
        is_train=True,
    )
    ingest.set_retry(num_retries=2, backoff_duration="30s")
    ingest.set_cpu_limit("1")
    ingest.set_memory_limit("4G")

    # ── Step 2: Data Validation ───────────────────────────────────────────────
    validate = data_validation_op(
        input_data=ingest.outputs["merged_data"],
        is_train=True,
    )
    validate.set_retry(num_retries=1)
    validate.set_cpu_limit("500m")
    validate.set_memory_limit("2G")

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    preprocess = preprocessing_op(
        input_data=ingest.outputs["merged_data"],
        imbalance_strategy=imbalance_strategy,
        is_train=True,
    )
    preprocess.after(validate)
    preprocess.set_retry(num_retries=1)
    preprocess.set_cpu_limit("2")
    preprocess.set_memory_limit("6G")

    # ── Step 4: Feature Engineering ───────────────────────────────────────────
    feat_eng = feature_engineering_op(
        train_data=preprocess.outputs["train_data"],
        val_data=preprocess.outputs["val_data"],
    )
    feat_eng.set_cpu_limit("1")
    feat_eng.set_memory_limit("4G")

    # ── Step 5: Model Training ────────────────────────────────────────────────
    train = model_training_op(
        train_data=feat_eng.outputs["engineered_train"],
        val_data=feat_eng.outputs["engineered_val"],
    )
    train.set_retry(num_retries=1, backoff_duration="60s")
    train.set_cpu_limit("2")
    train.set_memory_limit("8G")

    # ── Step 6: Model Evaluation ──────────────────────────────────────────────
    evaluate = model_evaluation_op(
        val_data=feat_eng.outputs["engineered_val"],
        models_dir=train.outputs["models_dir"],
        accuracy_threshold=accuracy_threshold,
        recall_threshold=recall_threshold,
    )
    evaluate.set_cpu_limit("2")
    evaluate.set_memory_limit("4G")

    # ── Step 7: Conditional Deployment ───────────────────────────────────────
    deploy = deployment_op(
        deploy_decision=evaluate.outputs["deploy_decision"],
        best_model=evaluate.outputs["best_model"],
        preprocessor=preprocess.outputs["preprocessor"],
        feature_config=feat_eng.outputs["feature_config"],
    )
    deploy.set_cpu_limit("500m")
    deploy.set_memory_limit("1G")


# ── Compile pipeline ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.yaml",
    )
    print("✅ Pipeline compiled: fraud_detection_pipeline.yaml")
