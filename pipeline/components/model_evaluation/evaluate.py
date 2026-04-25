"""
Component 6: Model Evaluation
- Precision, Recall, F1, AUC-ROC, AUC-PR
- Confusion matrix (fraud class focus)
- Cost-sensitive analysis (false negatives vs false positives)
- SHAP explainability
- Imbalance strategy comparison
"""

import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
)


# ── Business cost parameters ──────────────────────────────────────────────────
# False Negative (missed fraud): bank absorbs full transaction loss
# False Positive (false alarm) : customer friction, investigation cost
COST_FALSE_NEGATIVE = 200.0   # avg fraudulent transaction amount ($)
COST_FALSE_POSITIVE = 5.0     # investigation / customer service cost ($)


def predict_with_model(model, X, model_name: str):
    """Unified prediction interface for all model types."""
    if isinstance(model, dict) and "xgb" in model:
        # Hybrid model
        selector = model["selector"]
        xgb_model = model["xgb"]
        X_sel = selector.transform(X)
        proba = xgb_model.predict_proba(X_sel)[:, 1]
    else:
        proba = model.predict_proba(X)[:, 1]
    return proba


def get_xgb_model(model):
    """Get underlying XGB model for SHAP (handles hybrid)."""
    if isinstance(model, dict) and "xgb" in model:
        return model["xgb"], model["selector"]
    return model, None


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series,
                   model_name: str, threshold: float = 0.5) -> dict:
    """Compute all evaluation metrics for a single model."""
    proba = predict_with_model(model, X_val, model_name)
    preds = (proba >= threshold).astype(int)

    # Core metrics
    metrics = {
        "model": model_name,
        "threshold": threshold,
        "accuracy":  float(np.mean(preds == y_val)),
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "recall":    float(recall_score(y_val, preds, zero_division=0)),
        "f1":        float(f1_score(y_val, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_val, proba)),
        "auc_pr":    float(average_precision_score(y_val, proba)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    })

    # Business cost analysis
    total_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    metrics["business_cost"] = float(total_cost)
    metrics["fraud_loss"]    = float(fn * COST_FALSE_NEGATIVE)
    metrics["false_alarm_cost"] = float(fp * COST_FALSE_POSITIVE)

    return metrics, proba


def find_optimal_threshold(y_val, proba, target_recall: float = 0.85) -> float:
    """Find threshold that achieves target_recall with max precision."""
    prec, rec, thresholds = precision_recall_curve(y_val, proba)
    # Among thresholds achieving target_recall, pick highest precision
    valid = [(p, t) for p, r, t in zip(prec[:-1], rec[:-1], thresholds) if r >= target_recall]
    if not valid:
        return 0.5
    best_p, best_t = max(valid, key=lambda x: x[0])
    return float(best_t)


def plot_roc_curves(results: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, data in results.items():
        y_val  = data["y_val"]
        proba  = data["proba"]
        fpr, tpr, _ = roc_curve(y_val, proba)
        prec, rec, _ = precision_recall_curve(y_val, proba)
        auc = data["metrics"]["auc_roc"]
        ap  = data["metrics"]["auc_pr"]

        axes[0].plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        axes[1].plot(rec, prec, label=f"{model_name} (AP={ap:.3f})")

    axes[0].plot([0,1],[0,1],"k--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(results: dict, output_dir: str):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        m = data["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Fraud"])
        ax.set_yticklabels(["True Normal", "True Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{model_name}\nRecall={m['recall']:.3f}")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def plot_cost_comparison(results: dict, output_dir: str):
    names  = list(results.keys())
    costs  = [r["metrics"]["business_cost"] for r in results.values()]
    losses = [r["metrics"]["fraud_loss"] for r in results.values()]
    alarms = [r["metrics"]["false_alarm_cost"] for r in results.values()]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, costs,  width, label="Total Cost",       color="red",    alpha=0.7)
    ax.bar(x,         losses, width, label="Fraud Loss (FN)",  color="orange", alpha=0.7)
    ax.bar(x + width, alarms, width, label="False Alarm (FP)", color="blue",   alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Business Cost ($)")
    ax.set_title("Cost-Sensitive Model Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "cost_comparison.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def run_shap_analysis(model, X_val: pd.DataFrame, model_name: str, output_dir: str,
                       max_display: int = 20, sample_size: int = 1000):
    """Compute and plot SHAP values for the best model."""
    print(f"  Computing SHAP for {model_name} ...")

    # Sample for speed
    if len(X_val) > sample_size:
        X_sample = X_val.sample(sample_size, random_state=42)
    else:
        X_sample = X_val.copy()

    try:
        xgb_model, selector = get_xgb_model(model)

        if selector is not None:
            X_sample_sel = selector.transform(X_sample)
            selected_features = X_sample.columns[selector.get_support()].tolist()
            X_shap = pd.DataFrame(X_sample_sel, columns=selected_features)
        else:
            X_shap = X_sample

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_shap)

        # Summary plot
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, max_display=max_display,
                          show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        path = os.path.join(output_dir, f"shap_importance_{model_name}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

        # Beeswarm plot
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, max_display=max_display, show=False)
        plt.title(f"SHAP Beeswarm — {model_name}")
        plt.tight_layout()
        path2 = os.path.join(output_dir, f"shap_beeswarm_{model_name}.png")
        plt.savefig(path2, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path2}")

        # Top features
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_features = pd.DataFrame({
            "feature": X_shap.columns,
            "mean_abs_shap": mean_shap
        }).sort_values("mean_abs_shap", ascending=False).head(20)

        return top_features.to_dict(orient="records")

    except Exception as e:
        print(f"  ⚠️  SHAP failed: {e}")
        return []


def evaluate_all_models(
    val_path: str,
    models_dir: str,
    output_dir: str,
    evaluation_report_path: str,
    best_model_path: str,
    deploy_threshold_path: str,
    accuracy_threshold: float = 0.90,
    recall_threshold: float = 0.80,
):
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load validation data
    val = pd.read_csv(val_path)
    y_val = val["isFraud"].astype(int)
    X_val = val.drop(columns=["isFraud"]).fillna(0)

    print(f"  Val data: {X_val.shape}, fraud={y_val.sum()}")

    # Load all models
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    all_results = {}

    for mf in model_files:
        model_name = mf.replace(".pkl", "")
        with open(os.path.join(models_dir, mf), "rb") as f:
            model = pickle.load(f)

        print(f"\n  Evaluating: {model_name}")

        # Find optimal threshold for fraud recall
        proba_tmp = predict_with_model(model, X_val, model_name)
        opt_thresh = find_optimal_threshold(y_val, proba_tmp, target_recall=0.82)
        print(f"    Optimal threshold (recall≥82%): {opt_thresh:.3f}")

        metrics, proba = evaluate_model(model, X_val, y_val, model_name, threshold=opt_thresh)

        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall   : {metrics['recall']:.4f}")
        print(f"    F1       : {metrics['f1']:.4f}")
        print(f"    AUC-ROC  : {metrics['auc_roc']:.4f}")
        print(f"    AUC-PR   : {metrics['auc_pr']:.4f}")
        print(f"    Business cost: ${metrics['business_cost']:,.0f}")

        all_results[model_name] = {
            "metrics": metrics,
            "proba": proba,
            "y_val": y_val,
            "model": model,
        }

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_curves(all_results, output_dir)
    plot_confusion_matrices(all_results, output_dir)
    plot_cost_comparison(all_results, output_dir)

    # ── Select best model (by AUC-ROC) ────────────────────────────────────────
    best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["auc_roc"])
    best_metrics = all_results[best_name]["metrics"]
    best_model   = all_results[best_name]["model"]
    print(f"\n  Best model: {best_name}  (AUC-ROC={best_metrics['auc_roc']:.4f})")

    # ── SHAP for best model ────────────────────────────────────────────────────
    shap_results = run_shap_analysis(best_model, X_val, best_name, output_dir)

    # ── Deployment decision ───────────────────────────────────────────────────
    should_deploy = (
        best_metrics["auc_roc"] >= accuracy_threshold and
        best_metrics["recall"]  >= recall_threshold
    )
    deploy_decision = {
        "should_deploy": should_deploy,
        "best_model":    best_name,
        "auc_roc":       best_metrics["auc_roc"],
        "recall":        best_metrics["recall"],
        "auc_roc_threshold":  accuracy_threshold,
        "recall_threshold":   recall_threshold,
        "reason": "PASS" if should_deploy else
                  f"FAIL: auc_roc={best_metrics['auc_roc']:.3f} (need {accuracy_threshold}) | "
                  f"recall={best_metrics['recall']:.3f} (need {recall_threshold})",
    }
    print(f"\n  Deploy decision: {deploy_decision['reason']}")

    with open(deploy_threshold_path, "w") as f:
        json.dump(deploy_decision, f, indent=2)

    # ── Save best model ───────────────────────────────────────────────────────
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Best model saved: {best_model_path}")

    # ── Full evaluation report ────────────────────────────────────────────────
    report = {
        "all_models": {k: v["metrics"] for k, v in all_results.items()},
        "best_model": best_name,
        "deploy_decision": deploy_decision,
        "shap_top_features": shap_results,
    }
    with open(evaluation_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report: {evaluation_report_path}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path",               required=True)
    parser.add_argument("--models_dir",             required=True)
    parser.add_argument("--output_dir",             required=True)
    parser.add_argument("--evaluation_report_path", required=True)
    parser.add_argument("--best_model_path",        required=True)
    parser.add_argument("--deploy_threshold_path",  required=True)
    parser.add_argument("--accuracy_threshold",     type=float, default=0.90)
    parser.add_argument("--recall_threshold",       type=float, default=0.80)
    args = parser.parse_args()

    evaluate_all_models(
        val_path=args.val_path,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        evaluation_report_path=args.evaluation_report_path,
        best_model_path=args.best_model_path,
        deploy_threshold_path=args.deploy_threshold_path,
        accuracy_threshold=args.accuracy_threshold,
        recall_threshold=args.recall_threshold,
    )


if __name__ == "__main__":
    main()
