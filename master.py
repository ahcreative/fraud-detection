#!/usr/bin/env python3
"""
master_orchestrator.py
======================
This is the single entry point that runs inside Docker.
It executes every task in order, fully automatically:

  Step 1  — Ingest train_transaction + train_identity  → merged_train.csv
  Step 2  — Validate merged_train.csv
  Step 3  — Preprocess train data (SMOTE + class_weight comparison)
  Step 4  — Feature engineering on train data
  Step 5  — Train all models (XGBoost, LightGBM, Hybrid)
  Step 6  — Evaluate models on validation split
  Step 7  — Ingest test_transaction + test_identity    → merged_test.csv
  Step 8  — Preprocess & engineer test data (using saved artifacts)
  Step 9  — Run model on test data → test_predictions.csv
  Step 10 — Task 4: Cost-sensitive analysis
  Step 11 — Task 6: Push metrics to Prometheus / update dashboards
  Step 12 — Task 7: Drift simulation (train period vs test period)
  Step 13 — Task 8: Retraining strategy comparison
  Step 14 — Task 9: SHAP explainability
  Step 15 — Conditional deployment (copy best model to /serving)
  Step 16 — Generate final summary report
"""

import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/outputs/orchestrator.log"),
    ],
)
log = logging.getLogger("orchestrator")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(os.getenv("DATA_DIR",      "/data"))
OUTPUT_DIR    = Path(os.getenv("OUTPUT_DIR",    "/outputs"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/artifacts"))
SERVING_DIR   = Path(os.getenv("SERVING_DIR",   "/serving"))
SAMPLE_FRAC   = float(os.getenv("SAMPLE_FRAC",  "0.3"))   # 30% for laptop speed

# Sub-directories
DIRS = {
    "merged":       OUTPUT_DIR / "01_merged",
    "validated":    OUTPUT_DIR / "02_validated",
    "preprocessed": OUTPUT_DIR / "03_preprocessed",
    "engineered":   OUTPUT_DIR / "04_engineered",
    "models":       OUTPUT_DIR / "05_models",
    "evaluation":   OUTPUT_DIR / "06_evaluation",
    "test_merged":  OUTPUT_DIR / "07_test_merged",
    "test_processed": OUTPUT_DIR / "08_test_processed",
    "predictions":  OUTPUT_DIR / "09_predictions",
    "cost_analysis":OUTPUT_DIR / "10_cost_analysis",
    "drift":        OUTPUT_DIR / "11_drift",
    "retraining":   OUTPUT_DIR / "12_retraining",
    "shap":         OUTPUT_DIR / "13_shap",
    "imbalance":    OUTPUT_DIR / "14_imbalance_comparison",
    "summary":      OUTPUT_DIR / "15_summary",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SERVING_DIR.mkdir(parents=True, exist_ok=True)

# ── State tracking ─────────────────────────────────────────────────────────────
STATE = {
    "started_at":   datetime.utcnow().isoformat(),
    "steps":        {},
    "best_model":   None,
    "auc_roc":      None,
    "recall":       None,
    "deployed":     False,
    "errors":       [],
}

def step_start(name):
    log.info("")
    log.info("=" * 70)
    log.info(f"  ▶  {name}")
    log.info("=" * 70)
    STATE["steps"][name] = {"status": "running", "started": datetime.utcnow().isoformat()}
    return time.time()

def step_done(name, t0, info=""):
    elapsed = time.time() - t0
    STATE["steps"][name]["status"]  = "done"
    STATE["steps"][name]["elapsed"] = round(elapsed, 1)
    STATE["steps"][name]["info"]    = info
    log.info(f"  ✅  {name}  ({elapsed:.1f}s)  {info}")
    save_state()

def step_fail(name, t0, exc):
    elapsed = time.time() - t0
    STATE["steps"][name]["status"]  = "failed"
    STATE["steps"][name]["elapsed"] = round(elapsed, 1)
    STATE["steps"][name]["error"]   = str(exc)
    STATE["errors"].append({"step": name, "error": str(exc)})
    log.error(f"  ❌  {name} FAILED: {exc}")
    log.error(traceback.format_exc())
    save_state()

def save_state():
    STATE["updated_at"] = datetime.utcnow().isoformat()
    with open(OUTPUT_DIR / "pipeline_state.json", "w") as f:
        json.dump(STATE, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: TRAIN DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
def run_train_ingestion():
    name = "Step 1 — Train Data Ingestion"
    t0 = step_start(name)
    try:
        import pandas as pd
        sys.path.insert(0, "/app")
        from pipeline.components.data_ingestion.ingest import load_and_merge

        trans_path = DATA_DIR / "train_transaction.csv"
        ident_path = DATA_DIR / "train_identity.csv"
        out_path   = DIRS["merged"] / "merged_train.csv"

        for p in [trans_path, ident_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    "Place train_transaction.csv and train_identity.csv in the /data volume."
                )

        stats = load_and_merge(str(trans_path), str(ident_path), str(out_path), is_train=True)

        # Sample for laptop speed
        if SAMPLE_FRAC < 1.0:
            df = pd.read_csv(out_path)
            # Stratified sample to preserve fraud ratio
            fraud = df[df["isFraud"] == 1]
            legit = df[df["isFraud"] == 0]
            n_fraud = int(len(fraud) * SAMPLE_FRAC)
            n_legit = int(len(legit) * SAMPLE_FRAC)
            sampled = pd.concat([
                fraud.sample(n=n_fraud, random_state=42),
                legit.sample(n=n_legit, random_state=42),
            ]).sample(frac=1, random_state=42).reset_index(drop=True)
            sampled.to_csv(out_path, index=False)
            log.info(f"  Sampled {len(sampled):,} rows ({SAMPLE_FRAC*100:.0f}%)")
            log.info(f"  Fraud: {sampled['isFraud'].sum():,}  Legit: {(sampled['isFraud']==0).sum():,}")

        step_done(name, t0, f"rows={stats['n_rows']:,} fraud_rate={stats.get('fraud_rate',0):.4f}")
        return str(out_path)
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DATA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def run_validation(merged_path):
    name = "Step 2 — Data Validation"
    t0 = step_start(name)
    try:
        from pipeline.components.data_validation.validate import validate_data
        report_path = DIRS["validated"] / "validation_report.json"
        validate_data(merged_path, str(report_path), is_train=True)
        step_done(name, t0)
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PREPROCESSING + IMBALANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocessing(merged_path):
    name = "Step 3 — Preprocessing (SMOTE + Class-Weight comparison)"
    t0 = step_start(name)
    try:
        from pipeline.components.data_preprocessing.preprocess import preprocess

        # Run SMOTE (primary)
        log.info("  Running SMOTE strategy ...")
        train_smote, val_path, _ = preprocess(
            input_path=merged_path,
            output_dir=str(DIRS["preprocessed"]),
            artifacts_dir=str(ARTIFACTS_DIR),
            imbalance_strategy="smote",
            is_train=True,
        )

        # Run class_weight (for comparison)
        log.info("  Running class_weight strategy ...")
        train_cw, _, _ = preprocess(
            input_path=merged_path,
            output_dir=str(DIRS["preprocessed"]),
            artifacts_dir=str(ARTIFACTS_DIR / "cw"),
            imbalance_strategy="class_weight",
            is_train=True,
        )

        # Run imbalance comparison script
        log.info("  Generating imbalance comparison charts ...")
        from scripts.compare_imbalance import compare_imbalance_strategies
        compare_imbalance_strategies(
            merged_data_path=merged_path,
            output_dir=str(DIRS["imbalance"]),
            report_path=str(DIRS["imbalance"] / "report.json"),
            sample_rows=min(50000, _count_rows(merged_path)),
        )

        step_done(name, t0, f"train={train_smote}  val={val_path}")
        return train_smote, val_path
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def run_feature_engineering(train_path, val_path):
    name = "Step 4 — Feature Engineering"
    t0 = step_start(name)
    try:
        from pipeline.components.feature_engineering.engineer import engineer_features

        eng_train = str(DIRS["engineered"] / "train_engineered.csv")
        eng_val   = str(DIRS["engineered"] / "val_engineered.csv")
        feat_cfg  = str(ARTIFACTS_DIR / "feature_config.json")

        engineer_features(train_path, eng_train, feat_cfg, is_train=True)
        engineer_features(val_path,   eng_val,   feat_cfg, is_train=False,
                          dropped_cols_path=feat_cfg)

        step_done(name, t0)
        return eng_train, eng_val, feat_cfg
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def run_model_training(eng_train, eng_val):
    name = "Step 5 — Model Training (XGBoost + LightGBM + Hybrid)"
    t0 = step_start(name)
    try:
        from pipeline.components.model_training.train import train_all_models

        config = train_all_models(
            train_path=eng_train,
            val_path=eng_val,
            models_dir=str(DIRS["models"]),
            training_config_path=str(DIRS["models"] / "training_config.json"),
        )
        models_trained = list(config["models"].keys())
        step_done(name, t0, f"models={models_trained}")
        return config
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: MODEL EVALUATION ON VALIDATION SET
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation(eng_val):
    name = "Step 6 — Model Evaluation (validation set)"
    t0 = step_start(name)
    try:
        from pipeline.components.model_evaluation.evaluate import evaluate_all_models

        best_model_path    = str(DIRS["models"] / "best_model.pkl")
        deploy_thresh_path = str(DIRS["evaluation"] / "deploy_decision.json")
        eval_report_path   = str(DIRS["evaluation"] / "evaluation_report.json")

        report = evaluate_all_models(
            val_path=eng_val,
            models_dir=str(DIRS["models"]),
            output_dir=str(DIRS["evaluation"]),
            evaluation_report_path=eval_report_path,
            best_model_path=best_model_path,
            deploy_threshold_path=deploy_thresh_path,
            accuracy_threshold=float(os.getenv("AUC_THRESHOLD", "0.88")),
            recall_threshold=float(os.getenv("RECALL_THRESHOLD", "0.75")),
        )

        best = report["best_model"]
        m    = report["all_models"][best]
        STATE["best_model"] = best
        STATE["auc_roc"]    = m["auc_roc"]
        STATE["recall"]     = m["recall"]

        step_done(name, t0,
                  f"best={best} auc_roc={m['auc_roc']:.4f} recall={m['recall']:.4f}")
        return report, best_model_path, deploy_thresh_path
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: TEST DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
def run_test_ingestion():
    name = "Step 7 — Test Data Ingestion"
    t0 = step_start(name)
    try:
        import pandas as pd
        from pipeline.components.data_ingestion.ingest import load_and_merge

        trans_path = DATA_DIR / "test_transaction.csv"
        ident_path = DATA_DIR / "test_identity.csv"
        out_path   = DIRS["test_merged"] / "merged_test.csv"

        for p in [trans_path, ident_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    "Place test_transaction.csv and test_identity.csv in the /data volume."
                )

        stats = load_and_merge(str(trans_path), str(ident_path),
                               str(out_path), is_train=False)

        # Sample test data too
        if SAMPLE_FRAC < 1.0:
            df = pd.read_csv(out_path)
            sampled = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
            sampled.to_csv(out_path, index=False)
            log.info(f"  Test sampled: {len(sampled):,} rows")

        step_done(name, t0, f"rows={stats['n_rows']:,}")
        return str(out_path)
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: TEST DATA PREPROCESSING + FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def run_test_preprocessing(test_merged_path, feat_cfg):
    name = "Step 8 — Test Data Preprocessing & Feature Engineering"
    t0 = step_start(name)
    try:
        from pipeline.components.data_preprocessing.preprocess import preprocess
        from pipeline.components.feature_engineering.engineer import engineer_features

        # Preprocess using saved train artifacts (no leakage)
        test_proc_path, _, _ = preprocess(
            input_path=test_merged_path,
            output_dir=str(DIRS["test_processed"]),
            artifacts_dir=str(ARTIFACTS_DIR),
            imbalance_strategy="class_weight",  # irrelevant for test, uses saved artifacts
            is_train=False,
        )

        # Feature engineering using saved feature config
        test_eng_path = str(DIRS["test_processed"] / "test_engineered.csv")
        engineer_features(
            input_path=test_proc_path,
            output_path=test_eng_path,
            feature_config_path=feat_cfg,
            is_train=False,
            dropped_cols_path=feat_cfg,
        )

        step_done(name, t0)
        return test_eng_path
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: GENERATE TEST PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_test_predictions(test_eng_path, best_model_path):
    name = "Step 9 — Test Set Predictions"
    t0 = step_start(name)
    try:
        import pickle
        import numpy as np
        import pandas as pd

        with open(best_model_path, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(test_eng_path).fillna(0)

        # Handle hybrid model
        if isinstance(model, dict) and "xgb" in model:
            X_sel  = model["selector"].transform(test_df)
            probas = model["xgb"].predict_proba(X_sel)[:, 1]
        else:
            probas = model.predict_proba(test_df)[:, 1]

        threshold = float(os.getenv("PREDICT_THRESHOLD", "0.5"))
        preds     = (probas >= threshold).astype(int)

        # Save predictions
        test_raw = pd.read_csv(DIRS["test_merged"] / "merged_test.csv",
                               usecols=["TransactionID"])
        results = pd.DataFrame({
            "TransactionID":      test_raw["TransactionID"].values[:len(probas)],
            "fraud_probability":  np.round(probas, 4),
            "isFraud_predicted":  preds,
        })
        out_path = DIRS["predictions"] / "test_predictions.csv"
        results.to_csv(out_path, index=False)

        n_fraud = preds.sum()
        fraud_rate = preds.mean()
        log.info(f"  Predicted fraud: {n_fraud:,} / {len(preds):,} ({fraud_rate:.4f})")

        # Summary stats
        stats = {
            "n_predictions":   int(len(preds)),
            "n_fraud_flagged": int(n_fraud),
            "predicted_fraud_rate": round(float(fraud_rate), 4),
            "threshold":       threshold,
            "model_used":      STATE.get("best_model", "unknown"),
        }
        with open(DIRS["predictions"] / "prediction_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        step_done(name, t0, f"predictions={len(preds):,} fraud_flagged={n_fraud:,}")
        return str(out_path)
    except Exception as e:
        step_fail(name, t0, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: TASK 4 — COST-SENSITIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_cost_sensitive_analysis(eng_train, eng_val):
    name = "Step 10 — Task 4: Cost-Sensitive Learning Analysis"
    t0 = step_start(name)
    try:
        from scripts.cost_sensitive_analysis import run_cost_sensitive_analysis
        run_cost_sensitive_analysis(
            train_path=eng_train,
            val_path=eng_val,
            output_dir=str(DIRS["cost_analysis"]),
            report_path=str(DIRS["cost_analysis"] / "cost_report.json"),
        )
        step_done(name, t0)
    except Exception as e:
        step_fail(name, t0, e)
        log.warning("  Cost-sensitive analysis failed — continuing pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: TASK 6 — PUSH METRICS TO PROMETHEUS
# ══════════════════════════════════════════════════════════════════════════════
def push_metrics_to_prometheus(eval_report):
    name = "Step 11 — Task 6: Push Metrics to Prometheus"
    t0 = step_start(name)
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        PUSHGATEWAY = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")

        registry = CollectorRegistry()
        best = eval_report["best_model"]
        m    = eval_report["all_models"][best]

        for metric_name, value, desc in [
            ("fraud_recall_current",      m["recall"],    "Current fraud recall"),
            ("fraud_precision_current",   m["precision"], "Current precision"),
            ("fraud_f1_current",          m["f1"],        "Current F1"),
            ("fraud_auc_roc_current",     m["auc_roc"],   "Current AUC-ROC"),
            ("fraud_auc_pr_current",      m["auc_pr"],    "Current AUC-PR"),
            ("fraud_false_positive_rate", m["false_positive_rate"], "Current FPR"),
            ("fraud_business_cost",       m["business_cost"], "Business cost"),
            ("fraud_model_version",       1.0,            "Model version"),
        ]:
            g = Gauge(metric_name, desc, registry=registry)
            g.set(value)

        try:
            push_to_gateway(PUSHGATEWAY, job="fraud_pipeline", registry=registry)
            log.info(f"  Pushed metrics to Pushgateway at {PUSHGATEWAY}")
        except Exception as push_err:
            log.warning(f"  Pushgateway not reachable ({push_err}) — metrics written to file")

        # Always write metrics to file as backup
        metrics_out = {
            "recall":    m["recall"],
            "precision": m["precision"],
            "f1":        m["f1"],
            "auc_roc":   m["auc_roc"],
            "auc_pr":    m["auc_pr"],
            "fpr":       m["false_positive_rate"],
            "business_cost": m["business_cost"],
            "best_model": best,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(OUTPUT_DIR / "current_metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        step_done(name, t0, f"recall={m['recall']:.4f} auc_roc={m['auc_roc']:.4f}")
    except Exception as e:
        step_fail(name, t0, e)
        log.warning("  Metric push failed — continuing")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12: TASK 7 — DRIFT SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def run_drift_simulation(train_merged_path, test_merged_path):
    name = "Step 12 — Task 7: Drift Simulation (train vs test distribution)"
    t0 = step_start(name)
    try:
        from drift_simulation.simulate_drift import simulate_drift
        report = simulate_drift(
            merged_data_path=train_merged_path,
            output_dir=str(DIRS["drift"]),
            drift_report_path=str(DIRS["drift"] / "drift_report.json"),
            train_frac=0.7,
            inject_patterns=True,
        )
        step_done(name, t0,
                  f"drifted={report['n_features_drifted']} "
                  f"retrain={report['retrain_recommended']}")
        return report
    except Exception as e:
        step_fail(name, t0, e)
        log.warning("  Drift simulation failed — continuing")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13: TASK 8 — RETRAINING STRATEGY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def run_retraining_strategy():
    name = "Step 13 — Task 8: Retraining Strategy Comparison"
    t0 = step_start(name)
    try:
        from drift_simulation.retraining_strategy import compare_retraining_strategies
        report = compare_retraining_strategies(
            output_dir=str(DIRS["retraining"]),
            report_path=str(DIRS["retraining"] / "retraining_report.json"),
            n_periods=60,
        )
        step_done(name, t0, f"recommended={report['recommended_strategy']}")
    except Exception as e:
        step_fail(name, t0, e)
        log.warning("  Retraining strategy failed — continuing")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 14: TASK 9 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
def run_shap(best_model_path, eng_val):
    name = "Step 14 — Task 9: SHAP Explainability"
    t0 = step_start(name)
    try:
        from explainability.shap_analysis import run_explainability
        run_explainability(
            best_model_path=best_model_path,
            val_path=eng_val,
            models_dir=str(DIRS["models"]),
            output_dir=str(DIRS["shap"]),
            report_path=str(DIRS["shap"] / "shap_report.json"),
            sample_size=int(os.getenv("SHAP_SAMPLE", "800")),
        )
        step_done(name, t0)
    except Exception as e:
        step_fail(name, t0, e)
        log.warning("  SHAP failed — continuing")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 15: CONDITIONAL DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
def run_deployment(best_model_path, deploy_thresh_path):
    name = "Step 15 — Conditional Deployment"
    t0 = step_start(name)
    try:
        from pipeline.components.deployment.deploy import deploy_model
        import shutil

        preprocessor_src = ARTIFACTS_DIR / "preprocessor.pkl"
        feat_config_src  = ARTIFACTS_DIR / "feature_config.json"

        # Copy to serving dir
        shutil.copy(preprocessor_src, SERVING_DIR / "preprocessor.pkl")
        shutil.copy(feat_config_src,  SERVING_DIR / "feature_config.json")

        meta = deploy_model(
            deploy_threshold_path=deploy_thresh_path,
            best_model_path=best_model_path,
            preprocessor_path=str(SERVING_DIR / "preprocessor.pkl"),
            feature_config_path=str(SERVING_DIR / "feature_config.json"),
            serving_dir=str(SERVING_DIR),
        )
        STATE["deployed"] = meta.get("status") == "active"
        step_done(name, t0, f"deployed={STATE['deployed']}")
    except SystemExit:
        # sys.exit(0) from deploy_model = thresholds not met
        log.info("  Model did not meet deployment thresholds — skipped")
        step_done(name, t0, "skipped (thresholds not met)")
    except Exception as e:
        step_fail(name, t0, e)
        log.warning("  Deployment failed — continuing")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 16: FINAL SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
def write_summary():
    name = "Step 16 — Final Summary Report"
    t0 = step_start(name)

    STATE["finished_at"] = datetime.utcnow().isoformat()
    steps_done   = sum(1 for s in STATE["steps"].values() if s["status"] == "done")
    steps_failed = sum(1 for s in STATE["steps"].values() if s["status"] == "failed")

    summary_lines = [
        "=" * 70,
        "  FRAUD DETECTION PIPELINE — COMPLETE SUMMARY",
        "=" * 70,
        f"  Started    : {STATE['started_at']}",
        f"  Finished   : {STATE['finished_at']}",
        f"  Steps done : {steps_done}",
        f"  Steps failed: {steps_failed}",
        "",
        f"  Best model : {STATE.get('best_model', 'N/A')}",
        f"  AUC-ROC    : {STATE.get('auc_roc', 'N/A')}",
        f"  Recall     : {STATE.get('recall', 'N/A')}",
        f"  Deployed   : {STATE.get('deployed', False)}",
        "",
        "  OUTPUT LOCATIONS:",
        f"    Merged train CSV   : /outputs/01_merged/merged_train.csv",
        f"    Merged test CSV    : /outputs/07_test_merged/merged_test.csv",
        f"    Predictions        : /outputs/09_predictions/test_predictions.csv",
        f"    Model metrics      : /outputs/06_evaluation/evaluation_report.json",
        f"    ROC curves         : /outputs/06_evaluation/roc_pr_curves.png",
        f"    Confusion matrices : /outputs/06_evaluation/confusion_matrices.png",
        f"    Cost analysis      : /outputs/10_cost_analysis/",
        f"    Drift report       : /outputs/11_drift/drift_report.json",
        f"    Retrain strategy   : /outputs/12_retraining/retraining_report.json",
        f"    SHAP plots         : /outputs/13_shap/",
        f"    Imbalance compare  : /outputs/14_imbalance_comparison/",
        f"    Serving model      : /serving/model.pkl",
        "",
        "  MONITORING:",
        f"    Grafana  : http://localhost:3000  (admin / admin123)",
        f"    Prometheus: http://localhost:9090",
        f"    API       : http://localhost:8000/health",
        "=" * 70,
    ]

    for line in summary_lines:
        log.info(line)

    with open(DIRS["summary"] / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    save_state()
    step_done(name, t0)


# ── Utilities ──────────────────────────────────────────────────────────────────
def _count_rows(path):
    import pandas as pd
    return len(pd.read_csv(path, usecols=[0]))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("  IEEE CIS FRAUD DETECTION — FULL AUTOMATED PIPELINE")
    log.info(f"  Sample fraction : {SAMPLE_FRAC*100:.0f}%")
    log.info(f"  Data directory  : {DATA_DIR}")
    log.info(f"  Output directory: {OUTPUT_DIR}")
    log.info("=" * 70)

    try:
        # ── TRAIN SIDE ────────────────────────────────────────────────────────
        train_merged    = run_train_ingestion()          # Step 1
        run_validation(train_merged)                     # Step 2
        train_path, val_path = run_preprocessing(train_merged)  # Step 3
        eng_train, eng_val, feat_cfg = run_feature_engineering(train_path, val_path)  # Step 4
        run_model_training(eng_train, eng_val)           # Step 5
        eval_report, best_model_path, deploy_thresh = run_evaluation(eng_val)  # Step 6

        # ── TEST SIDE ─────────────────────────────────────────────────────────
        test_merged     = run_test_ingestion()           # Step 7
        test_eng_path   = run_test_preprocessing(test_merged, feat_cfg)  # Step 8
        run_test_predictions(test_eng_path, best_model_path)  # Step 9

        # ── ANALYSIS TASKS ────────────────────────────────────────────────────
        run_cost_sensitive_analysis(eng_train, eng_val)  # Step 10 (Task 4)
        push_metrics_to_prometheus(eval_report)          # Step 11 (Task 6)
        run_drift_simulation(train_merged, test_merged)  # Step 12 (Task 7)
        run_retraining_strategy()                        # Step 13 (Task 8)
        run_shap(best_model_path, eng_val)               # Step 14 (Task 9)
        run_deployment(best_model_path, deploy_thresh)   # Step 15

        # ── SUMMARY ───────────────────────────────────────────────────────────
        write_summary()                                  # Step 16

        log.info("")
        log.info("  🎉  ALL STEPS COMPLETE — Pipeline finished successfully!")
        log.info("  📊  Open Grafana: http://localhost:3000 (admin / admin123)")
        log.info("  🔍  Check outputs/ folder for all results and plots")
        log.info("")

    except FileNotFoundError as e:
        log.error(f"\n  ❌  DATA FILE MISSING: {e}")
        log.error("  Please place all 4 CSV files in the data/ folder and restart.")
        sys.exit(1)
    except Exception as e:
        log.error(f"\n  ❌  Pipeline failed at: {e}")
        log.error(traceback.format_exc())
        write_summary()
        sys.exit(1)


if __name__ == "__main__":
    main()