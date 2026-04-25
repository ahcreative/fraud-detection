"""
quick_demo.py — Quick demonstration of each task
Runs lightweight versions of all tasks for fast verification.
Designed for Core i5 / 16GB RAM — finishes in ~10 minutes.

Usage:
    python scripts/quick_demo.py --data_dir ./data
"""

import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEMO_ROWS = 30000   # Small sample for quick demo


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def quick_demo(data_dir: str, output_dir: str = "./demo_outputs"):
    t_total = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  FRAUD DETECTION SYSTEM — QUICK DEMO")
    print("  All 9 Tasks | ~10 minute run")
    print("="*60)

    # ── Verify data ─────────────────────────────────────────────────────────
    for fname in ["train_transaction.csv", "train_identity.csv"]:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"\n❌ Missing: {fpath}")
            print(f"   Download from Kaggle and place in {data_dir}/")
            sys.exit(1)
    print(f"\n  ✅ Data files found in {data_dir}/")

    # ── TASK 1 — Data Ingestion + Kubeflow pipeline compile ─────────────────
    section("TASK 1 — Data Ingestion + Kubeflow Pipeline")
    import pandas as pd, numpy as np
    from pipeline.components.data_ingestion.ingest import load_and_merge

    merged_path = os.path.join(output_dir, "merged.csv")
    stats = load_and_merge(
        os.path.join(data_dir, "train_transaction.csv"),
        os.path.join(data_dir, "train_identity.csv"),
        merged_path, is_train=True
    )
    # Sample down for demo speed
    df = pd.read_csv(merged_path)
    df = df.sample(n=min(DEMO_ROWS, len(df)), random_state=42)
    df.to_csv(merged_path, index=False)
    print(f"  Rows used for demo : {len(df):,}")
    print(f"  Fraud rate         : {stats['fraud_rate']:.4f}")
    print(f"  Columns            : {stats['n_cols']}")

    try:
        from pipeline.pipeline import fraud_detection_pipeline
        from kfp import compiler
        compiler.Compiler().compile(
            pipeline_func=fraud_detection_pipeline,
            package_path=os.path.join(output_dir, "pipeline.yaml"),
        )
        print("  ✅ KFP Pipeline compiled → demo_outputs/pipeline.yaml")
    except Exception as e:
        print(f"  ⚠️  KFP compile (non-fatal): {e}")

    # ── TASK 2 — Imbalance Strategy Comparison ───────────────────────────────
    section("TASK 2 — Imbalance Strategy Comparison (SMOTE vs Class-Weight)")
    from scripts.compare_imbalance import compare_imbalance_strategies
    imbalance_report = os.path.join(output_dir, "imbalance_report.json")
    report = compare_imbalance_strategies(
        merged_data_path=merged_path,
        output_dir=os.path.join(output_dir, "imbalance"),
        report_path=imbalance_report,
        sample_rows=DEMO_ROWS,
    )
    print(f"  Best recall strategy : {report['best_recall_strategy']}")
    print(f"  Lowest cost strategy : {report['lowest_cost_strategy']}")

    # ── TASK 3+4 — Train all models + cost-sensitive ─────────────────────────
    section("TASK 3+4 — Model Training (XGBoost + LightGBM + Hybrid + Cost-Sensitive)")
    from pipeline.components.data_preprocessing.preprocess import preprocess
    from pipeline.components.feature_engineering.engineer import engineer_features
    from pipeline.components.model_training.train import train_all_models
    from pipeline.components.model_evaluation.evaluate import evaluate_all_models

    pre_dir   = os.path.join(output_dir, "preprocessed")
    art_dir   = os.path.join(output_dir, "artifacts")
    eng_dir   = os.path.join(output_dir, "engineered")
    model_dir = os.path.join(output_dir, "models")
    eval_dir  = os.path.join(output_dir, "evaluation")
    serve_dir = os.path.join(output_dir, "serving")

    train_path, val_path, _ = preprocess(
        input_path=merged_path, output_dir=pre_dir,
        artifacts_dir=art_dir, imbalance_strategy="class_weight", is_train=True
    )
    eng_train = os.path.join(eng_dir, "train_eng.csv")
    eng_val   = os.path.join(eng_dir, "val_eng.csv")
    feat_cfg  = os.path.join(art_dir, "feature_config.json")
    os.makedirs(eng_dir, exist_ok=True)
    engineer_features(train_path, eng_train, feat_cfg, is_train=True)
    engineer_features(val_path,   eng_val,   feat_cfg, is_train=False, dropped_cols_path=feat_cfg)

    os.makedirs(model_dir, exist_ok=True)
    train_all_models(eng_train, eng_val, model_dir, os.path.join(model_dir, "config.json"))

    best_model_path    = os.path.join(serve_dir, "model.pkl")
    deploy_thresh_path = os.path.join(eval_dir,  "deploy_decision.json")
    eval_report_path   = os.path.join(eval_dir,  "eval_report.json")
    os.makedirs(serve_dir, exist_ok=True)

    eval_report = evaluate_all_models(
        val_path=eng_val, models_dir=model_dir, output_dir=eval_dir,
        evaluation_report_path=eval_report_path,
        best_model_path=best_model_path,
        deploy_threshold_path=deploy_thresh_path,
        accuracy_threshold=0.85, recall_threshold=0.70,   # relaxed for small demo sample
    )
    best = eval_report["best_model"]
    m    = eval_report["all_models"][best]
    print(f"  Best model  : {best}")
    print(f"  AUC-ROC     : {m['auc_roc']:.4f}")
    print(f"  Recall      : {m['recall']:.4f}")
    print(f"  Precision   : {m['precision']:.4f}")
    print(f"  Biz Cost    : ${m['business_cost']:,.0f}")

    # ── TASK 5 — CI/CD (show files) ─────────────────────────────────────────
    section("TASK 5 — CI/CD Pipeline")
    cicd_files = [
        "cicd/.github/workflows/fraud_detection_cicd.yml",
        "cicd/Jenkinsfile",
        "scripts/trigger_pipeline.py",
    ]
    for f in cicd_files:
        exists = "✅" if os.path.exists(f) else "❌"
        print(f"  {exists} {f}")
    print("  → Push to GitHub to activate CI/CD automatically")

    # ── TASK 6 — Monitoring (show config) ────────────────────────────────────
    section("TASK 6 — Monitoring (Prometheus + Grafana)")
    monitoring_files = [
        "monitoring/prometheus/prometheus.yml",
        "monitoring/prometheus/alert_rules.yml",
        "monitoring/grafana/dashboards/system_health.json",
        "monitoring/grafana/dashboards/model_performance.json",
        "monitoring/grafana/dashboards/data_drift.json",
    ]
    for f in monitoring_files:
        exists = "✅" if os.path.exists(f) else "❌"
        print(f"  {exists} {f}")
    print("  → Start monitoring: docker-compose up -d")
    print("  → Grafana: http://localhost:3000  (admin/admin123)")

    # ── TASK 7 — Drift Simulation ────────────────────────────────────────────
    section("TASK 7 — Drift Simulation (Time-based)")
    from drift_simulation.simulate_drift import simulate_drift
    drift_dir    = os.path.join(output_dir, "drift")
    drift_report = os.path.join(drift_dir, "drift_report.json")
    dr = simulate_drift(
        merged_data_path=merged_path,
        output_dir=drift_dir,
        drift_report_path=drift_report,
        train_frac=0.7,
        inject_patterns=True,
    )
    print(f"  Features drifted  : {dr['n_features_drifted']} / {dr['n_features_checked']}")
    print(f"  High drift cols   : {dr['n_features_high_drift']}")
    print(f"  Fraud rate A→B    : {dr['fraud_rate_a']:.4f} → {dr['fraud_rate_b']:.4f}")
    print(f"  Retrain needed    : {dr['retrain_recommended']}")

    # ── TASK 8 — Retraining Strategy ─────────────────────────────────────────
    section("TASK 8 — Retraining Strategy Comparison")
    from drift_simulation.retraining_strategy import compare_retraining_strategies
    retrain_dir    = os.path.join(output_dir, "retraining")
    retrain_report = os.path.join(retrain_dir, "report.json")
    rr = compare_retraining_strategies(
        output_dir=retrain_dir, report_path=retrain_report, n_periods=60
    )
    print(f"  Recommended strategy : {rr['recommended_strategy']}")
    for name, data in rr["strategies"].items():
        print(f"    {name:20s} → retrains={data['n_retrains']:2d}  "
              f"avg_recall={data['avg_recall']:.4f}  "
              f"cost={data['compute_cost']:.1f}")

    # ── TASK 9 — SHAP Explainability ─────────────────────────────────────────
    section("TASK 9 — SHAP Explainability")
    if os.path.exists(best_model_path):
        from explainability.shap_analysis import run_explainability
        shap_dir    = os.path.join(output_dir, "shap")
        shap_report = os.path.join(shap_dir, "report.json")
        sr = run_explainability(
            best_model_path=best_model_path,
            val_path=eng_val,
            models_dir=model_dir,
            output_dir=shap_dir,
            report_path=shap_report,
            sample_size=500,
        )
        print(f"  Top feature : {sr['top_10_features'][0]['feature']}")
        print(f"  SHAP value  : {sr['top_10_features'][0]['mean_abs_shap']:.4f}")
        print(f"  Plots saved : {shap_dir}/")
    else:
        print("  ⚠️  Model not found — run full pipeline first")

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print("="*60)
    print(f"\n  Total time  : {elapsed/60:.1f} minutes")
    print(f"  All outputs : {output_dir}/")
    print(f"\n  Key files generated:")
    for dirn, desc in [
        ("imbalance",   "Imbalance strategy comparison"),
        ("evaluation",  "Model metrics + ROC/PR curves"),
        ("drift",       "Feature drift report + plots"),
        ("retraining",  "Retraining strategy comparison"),
        ("shap",        "SHAP feature importance plots"),
    ]:
        full = os.path.join(output_dir, dirn)
        if os.path.exists(full):
            files = [f for f in os.listdir(full) if f.endswith((".png",".json"))]
            print(f"    {full}/  ({len(files)} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--output_dir", default="./demo_outputs")
    args = parser.parse_args()
    quick_demo(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
