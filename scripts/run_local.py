"""
Master Local Run Script
Runs all pipeline components locally without Kubeflow.
Use this to test the full pipeline before deploying to Kubeflow.

Usage:
    python scripts/run_local.py \
        --data_dir ./data \
        --output_dir ./outputs \
        --imbalance_strategy smote \
        --skip_shap false
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def banner(title: str):
    print(f"\n{'#'*60}")
    print(f"#  {title}")
    print(f"{'#'*60}")


def run_local_pipeline(
    data_dir:           str,
    output_dir:         str,
    imbalance_strategy: str = "smote",
    skip_shap:          bool = False,
    compare_strategies: bool = True,
    sample_frac:        float = 0.3,   # Use 30% of data for laptop speed
):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    dirs = {
        "merged":       os.path.join(output_dir, "01_merged"),
        "validated":    os.path.join(output_dir, "02_validated"),
        "preprocessed": os.path.join(output_dir, "03_preprocessed"),
        "engineered":   os.path.join(output_dir, "04_engineered"),
        "models":       os.path.join(output_dir, "05_models"),
        "evaluation":   os.path.join(output_dir, "06_evaluation"),
        "serving":      os.path.join(output_dir, "07_serving"),
        "drift":        os.path.join(output_dir, "08_drift"),
        "retraining":   os.path.join(output_dir, "09_retraining"),
        "explainability": os.path.join(output_dir, "10_explainability"),
        "artifacts":    os.path.join(output_dir, "artifacts"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ── STEP 1: Data Ingestion ────────────────────────────────────────────────
    banner("STEP 1: Data Ingestion")
    from pipeline.components.data_ingestion.ingest import load_and_merge
    import pandas as pd

    merged_path = os.path.join(dirs["merged"], "merged_train.csv")
    trans_path  = os.path.join(data_dir, "train_transaction.csv")
    ident_path  = os.path.join(data_dir, "train_identity.csv")

    # Validate data files exist
    for p, name in [(trans_path, "train_transaction.csv"), (ident_path, "train_identity.csv")]:
        if not os.path.exists(p):
            print(f"\n❌ Missing: {p}")
            print(f"   Please place {name} in {data_dir}/")
            sys.exit(1)

    stats = load_and_merge(trans_path, ident_path, merged_path, is_train=True)

    # Sample for laptop speed
    if sample_frac < 1.0:
        print(f"\n  Sampling {sample_frac*100:.0f}% of data for speed ...")
        df = pd.read_csv(merged_path)
        df_sample = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        df_sample.to_csv(merged_path, index=False)
        print(f"  Sampled rows: {len(df_sample):,}")

    # ── STEP 2: Data Validation ───────────────────────────────────────────────
    banner("STEP 2: Data Validation")
    from pipeline.components.data_validation.validate import validate_data
    report_path = os.path.join(dirs["validated"], "validation_report.json")
    validate_data(merged_path, report_path, is_train=True)

    # ── STEP 3: Preprocessing (SMOTE + Class Weight for comparison) ───────────
    banner("STEP 3: Preprocessing")
    from pipeline.components.data_preprocessing.preprocess import preprocess

    strategies_to_run = [imbalance_strategy]
    if compare_strategies and imbalance_strategy != "class_weight":
        strategies_to_run.append("class_weight")

    train_paths = {}
    val_path    = None

    for strategy in strategies_to_run:
        print(f"\n  --- Strategy: {strategy} ---")
        train_out, val_out, _ = preprocess(
            input_path=merged_path,
            output_dir=dirs["preprocessed"],
            artifacts_dir=dirs["artifacts"],
            imbalance_strategy=strategy,
            is_train=True,
        )
        train_paths[strategy] = train_out
        if val_path is None:
            val_path = val_out

    primary_train = train_paths[imbalance_strategy]

    # ── STEP 4: Feature Engineering ───────────────────────────────────────────
    banner("STEP 4: Feature Engineering")
    from pipeline.components.feature_engineering.engineer import engineer_features

    eng_train_path  = os.path.join(dirs["engineered"], "train_engineered.csv")
    eng_val_path    = os.path.join(dirs["engineered"], "val_engineered.csv")
    feat_config_path = os.path.join(dirs["artifacts"], "feature_config.json")

    engineer_features(primary_train,  eng_train_path, feat_config_path, is_train=True)
    engineer_features(val_path, eng_val_path,   feat_config_path, is_train=False,
                      dropped_cols_path=feat_config_path)

    # ── STEP 5: Model Training ────────────────────────────────────────────────
    banner("STEP 5: Model Training")
    from pipeline.components.model_training.train import train_all_models
    training_config_path = os.path.join(dirs["models"], "training_config.json")
    train_all_models(
        train_path=eng_train_path,
        val_path=eng_val_path,
        models_dir=dirs["models"],
        training_config_path=training_config_path,
    )

    # ── STEP 6: Model Evaluation ──────────────────────────────────────────────
    banner("STEP 6: Model Evaluation")
    from pipeline.components.model_evaluation.evaluate import evaluate_all_models
    eval_report_path   = os.path.join(dirs["evaluation"], "evaluation_report.json")
    best_model_path    = os.path.join(dirs["serving"],    "model.pkl")
    deploy_thresh_path = os.path.join(dirs["evaluation"], "deploy_decision.json")

    report = evaluate_all_models(
        val_path=eng_val_path,
        models_dir=dirs["models"],
        output_dir=dirs["evaluation"],
        evaluation_report_path=eval_report_path,
        best_model_path=best_model_path,
        deploy_threshold_path=deploy_thresh_path,
        accuracy_threshold=0.90,
        recall_threshold=0.80,
    )

    # ── STEP 7: Deployment ────────────────────────────────────────────────────
    banner("STEP 7: Conditional Deployment")
    import shutil
    preprocessor_path = os.path.join(dirs["artifacts"], "preprocessor.pkl")
    if os.path.exists(preprocessor_path):
        shutil.copy(preprocessor_path, os.path.join(dirs["serving"], "preprocessor.pkl"))
    if os.path.exists(feat_config_path):
        shutil.copy(feat_config_path, os.path.join(dirs["serving"], "feature_config.json"))

    from pipeline.components.deployment.deploy import deploy_model
    deploy_model(
        deploy_threshold_path=deploy_thresh_path,
        best_model_path=best_model_path,
        preprocessor_path=os.path.join(dirs["serving"], "preprocessor.pkl"),
        feature_config_path=os.path.join(dirs["serving"], "feature_config.json"),
        serving_dir=dirs["serving"],
    )

    # ── STEP 8: Drift Simulation ──────────────────────────────────────────────
    banner("STEP 8: Drift Simulation")
    from drift_simulation.simulate_drift import simulate_drift
    drift_report_path = os.path.join(dirs["drift"], "drift_report.json")
    simulate_drift(
        merged_data_path=merged_path,
        output_dir=dirs["drift"],
        drift_report_path=drift_report_path,
        train_frac=0.7,
        inject_patterns=True,
    )

    # ── STEP 9: Retraining Strategy Comparison ────────────────────────────────
    banner("STEP 9: Retraining Strategy Comparison")
    from drift_simulation.retraining_strategy import compare_retraining_strategies
    retrain_report_path = os.path.join(dirs["retraining"], "retraining_report.json")
    compare_retraining_strategies(
        output_dir=dirs["retraining"],
        report_path=retrain_report_path,
        n_periods=60,
    )

    # ── STEP 10: SHAP Explainability ──────────────────────────────────────────
    if not skip_shap:
        banner("STEP 10: SHAP Explainability")
        from explainability.shap_analysis import run_explainability
        shap_report_path = os.path.join(dirs["explainability"], "shap_report.json")
        run_explainability(
            best_model_path=best_model_path,
            val_path=eng_val_path,
            models_dir=dirs["models"],
            output_dir=dirs["explainability"],
            report_path=shap_report_path,
            sample_size=1000,
        )
    else:
        print("\n  [SHAP skipped — pass --skip_shap false to enable]")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner("PIPELINE COMPLETE")
    print(f"\n  Total time : {elapsed/60:.1f} minutes")
    print(f"  Output dir : {output_dir}")
    print(f"\n  Key outputs:")
    print(f"    Model          : {best_model_path}")
    print(f"    Evaluation     : {eval_report_path}")
    print(f"    Drift report   : {drift_report_path}")
    print(f"    Retrain report : {retrain_report_path}")
    if not skip_shap:
        print(f"    SHAP report    : {dirs['explainability']}/shap_report.json")

    # Print best model metrics
    best = report["best_model"]
    best_m = report["all_models"][best]
    print(f"\n  Best model : {best}")
    print(f"    AUC-ROC  : {best_m['auc_roc']:.4f}")
    print(f"    Recall   : {best_m['recall']:.4f}")
    print(f"    Precision: {best_m['precision']:.4f}")
    print(f"    F1       : {best_m['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run full fraud detection pipeline locally")
    parser.add_argument("--data_dir",           default="./data",
                        help="Directory containing train_transaction.csv and train_identity.csv")
    parser.add_argument("--output_dir",         default="./outputs")
    parser.add_argument("--imbalance_strategy", default="smote",
                        choices=["smote", "class_weight", "undersample"])
    parser.add_argument("--skip_shap",          default="false")
    parser.add_argument("--compare_strategies", default="true")
    parser.add_argument("--sample_frac",        type=float, default=0.3,
                        help="Fraction of data to use (0.3 = 30%% for laptop speed)")
    args = parser.parse_args()

    run_local_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        imbalance_strategy=args.imbalance_strategy,
        skip_shap=args.skip_shap.lower() == "true",
        compare_strategies=args.compare_strategies.lower() == "true",
        sample_frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()
