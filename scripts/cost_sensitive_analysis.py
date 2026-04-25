"""
Task 4: Cost-Sensitive Learning — Detailed Analysis
Compares standard vs cost-sensitive training across all models.
Produces business impact report: fraud loss vs false alarm cost.
"""

import argparse
import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Business cost constants ────────────────────────────────────────────────────
# These represent realistic banking costs
COST_FALSE_NEGATIVE = 200.0   # avg fraud transaction loss absorbed by bank
COST_FALSE_POSITIVE = 5.0     # investigation cost + customer friction
COST_TRUE_POSITIVE  = -50.0   # value recovered by catching fraud (negative = benefit)

XGB_BASE = dict(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    random_state=42, n_jobs=2, tree_method="hist",
    eval_metric="aucpr", verbosity=0,
)
LGB_BASE = dict(
    n_estimators=200, max_depth=5, num_leaves=31, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    random_state=42, n_jobs=2, verbose=-1,
)


def load_data(train_path: str, val_path: str):
    train = pd.read_csv(train_path).fillna(0)
    val   = pd.read_csv(val_path).fillna(0)
    target = "isFraud"
    X_tr, y_tr = train.drop(columns=[target]), train[target].astype(int)
    X_val, y_val = val.drop(columns=[target]), val[target].astype(int)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"  Train: {X_tr.shape}  fraud={y_tr.sum()}")
    print(f"  Val  : {X_val.shape}  fraud={y_val.sum()}")
    print(f"  Scale pos weight: {spw:.1f}")
    return X_tr, y_tr, X_val, y_val, spw


def get_metrics(model, X_val, y_val, label: str, threshold: float = 0.5) -> dict:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val)[:, 1]
    else:
        proba = model.predict(X_val)

    preds = (proba >= threshold).astype(int)
    cm    = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()

    # Business impact
    total_cost    = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    fraud_loss    = fn * COST_FALSE_NEGATIVE
    false_alarms  = fp * COST_FALSE_POSITIVE
    value_recovered = tp * abs(COST_TRUE_POSITIVE)

    return {
        "model":              label,
        "threshold":          threshold,
        "precision":          round(precision_score(y_val, preds, zero_division=0), 4),
        "recall":             round(recall_score(y_val, preds, zero_division=0), 4),
        "f1":                 round(f1_score(y_val, preds, zero_division=0), 4),
        "auc_roc":            round(roc_auc_score(y_val, proba), 4),
        "auc_pr":             round(average_precision_score(y_val, proba), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "total_business_cost":  round(total_cost, 2),
        "fraud_loss_fn":       round(fraud_loss, 2),
        "false_alarm_cost_fp": round(false_alarms, 2),
        "value_recovered_tp":  round(value_recovered, 2),
        "net_cost":            round(total_cost - value_recovered, 2),
    }


def find_optimal_threshold_for_recall(y_val, proba, target_recall=0.85):
    prec, rec, thresholds = precision_recall_curve(y_val, proba)
    valid = [(p, t) for p, r, t in zip(prec[:-1], rec[:-1], thresholds) if r >= target_recall]
    if not valid:
        return 0.5
    return float(max(valid, key=lambda x: x[0])[1])


def train_pair(name: str, standard_model, cost_model,
               X_tr, y_tr, X_val, y_val,
               X_val_lgb=None, y_val_lgb=None) -> tuple:
    """Train one standard + one cost-sensitive model, return both metrics."""
    print(f"\n  [{name}] Training standard ...")
    standard_model.fit(X_tr, y_tr,
                       eval_set=[(X_val, y_val)],
                       verbose=False)

    print(f"  [{name}] Training cost-sensitive ...")
    cost_model.fit(X_tr, y_tr,
                   eval_set=[(X_val, y_val)],
                   verbose=False)

    proba_std = standard_model.predict_proba(X_val)[:, 1]
    proba_cs  = cost_model.predict_proba(X_val)[:, 1]

    thresh_std = find_optimal_threshold_for_recall(y_val, proba_std, 0.82)
    thresh_cs  = find_optimal_threshold_for_recall(y_val, proba_cs,  0.82)

    m_std = get_metrics(standard_model, X_val, y_val, f"{name}_standard",    thresh_std)
    m_cs  = get_metrics(cost_model,     X_val, y_val, f"{name}_cost_sensitive", thresh_cs)
    return m_std, m_cs


def plot_cost_analysis(all_metrics: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_metrics)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Cost-Sensitive Learning: Standard vs Cost-Sensitive Models\n"
                 f"(FN cost=${COST_FALSE_NEGATIVE:.0f}, FP cost=${COST_FALSE_POSITIVE:.0f})",
                 fontsize=13, fontweight="bold")

    labels  = df["model"].tolist()
    x       = np.arange(len(labels))
    colors  = ["#2196F3" if "standard" in m else "#F44336" for m in labels]

    def bar_plot(ax, col, title, fmt=".3f", unit=""):
        vals = df[col].tolist()
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{v:{fmt}}{unit}", ha="center", va="bottom", fontsize=7)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#2196F3", label="Standard"),
                            Patch(color="#F44336", label="Cost-Sensitive")],
                  fontsize=8, loc="upper right")

    bar_plot(axes[0, 0], "recall",    "Recall ↑ (fraud caught)")
    bar_plot(axes[0, 1], "precision", "Precision ↑")
    bar_plot(axes[0, 2], "auc_roc",   "AUC-ROC ↑")
    bar_plot(axes[1, 0], "fraud_loss_fn",
             f"Fraud Loss $ (FN×${COST_FALSE_NEGATIVE:.0f}) ↓", fmt=",.0f", unit="")
    bar_plot(axes[1, 1], "false_alarm_cost_fp",
             f"False Alarm Cost $ (FP×${COST_FALSE_POSITIVE:.0f}) ↓", fmt=",.0f", unit="")
    bar_plot(axes[1, 2], "net_cost",
             "Net Business Cost $ ↓", fmt=",.0f", unit="")

    plt.tight_layout()
    path = os.path.join(output_dir, "cost_sensitive_comparison.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_threshold_analysis(y_val, probas: dict, output_dir: str):
    """Show how threshold choice affects recall vs precision for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    thresholds = np.linspace(0.1, 0.9, 50)

    for label, proba in probas.items():
        recalls    = [recall_score(y_val, (proba >= t).astype(int), zero_division=0) for t in thresholds]
        precisions = [precision_score(y_val, (proba >= t).astype(int), zero_division=0) for t in thresholds]
        costs      = [(y_val[proba < t].sum() * COST_FALSE_NEGATIVE +
                       ((proba >= t).astype(int) - y_val).clip(0).sum() * COST_FALSE_POSITIVE)
                      for t in thresholds]
        ls = "--" if "cost" in label else "-"
        axes[0].plot(thresholds, recalls, linestyle=ls, label=label, linewidth=1.5)
        axes[1].plot(thresholds, costs,   linestyle=ls, label=label, linewidth=1.5)

    axes[0].axhline(0.82, color="black", linestyle=":", label="Target recall=0.82")
    axes[0].set_xlabel("Decision Threshold")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Recall vs Threshold")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Decision Threshold")
    axes[1].set_ylabel("Business Cost ($)")
    axes[1].set_title("Business Cost vs Threshold")
    axes[1].legend(fontsize=7)

    plt.suptitle("Threshold Sensitivity Analysis", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_analysis.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run_cost_sensitive_analysis(
    train_path: str,
    val_path:   str,
    output_dir: str,
    report_path: str,
):
    print(f"\n{'='*60}")
    print("COST-SENSITIVE LEARNING ANALYSIS")
    print(f"  FN penalty: ${COST_FALSE_NEGATIVE:.0f}  |  FP penalty: ${COST_FALSE_POSITIVE:.0f}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    X_tr, y_tr, X_val, y_val, spw = load_data(train_path, val_path)

    all_metrics = []
    all_probas  = {}

    # ── XGBoost pair ──────────────────────────────────────────────────────────
    xgb_std = xgb.XGBClassifier(**XGB_BASE)
    xgb_cs  = xgb.XGBClassifier(**{**XGB_BASE, "scale_pos_weight": spw})
    m_std, m_cs = train_pair("XGBoost", xgb_std, xgb_cs, X_tr, y_tr, X_val, y_val)
    all_metrics += [m_std, m_cs]
    all_probas["xgb_standard"]      = xgb_std.predict_proba(X_val)[:, 1]
    all_probas["xgb_cost_sensitive"] = xgb_cs.predict_proba(X_val)[:, 1]

    # ── LightGBM pair ─────────────────────────────────────────────────────────
    lgb_std = lgb.LGBMClassifier(**LGB_BASE)
    lgb_cs  = lgb.LGBMClassifier(**{**LGB_BASE, "scale_pos_weight": spw})

    lgb_std.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
    lgb_cs.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])

    proba_lgb_std = lgb_std.predict_proba(X_val)[:, 1]
    proba_lgb_cs  = lgb_cs.predict_proba(X_val)[:, 1]

    t_std = find_optimal_threshold_for_recall(y_val, proba_lgb_std, 0.82)
    t_cs  = find_optimal_threshold_for_recall(y_val, proba_lgb_cs,  0.82)

    all_metrics += [
        get_metrics(lgb_std, X_val, y_val, "LightGBM_standard",      t_std),
        get_metrics(lgb_cs,  X_val, y_val, "LightGBM_cost_sensitive", t_cs),
    ]
    all_probas["lgb_standard"]      = proba_lgb_std
    all_probas["lgb_cost_sensitive"] = proba_lgb_cs

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n  {'Model':<30} {'Recall':>8} {'Prec':>8} {'AUC':>8} {'FraudLoss':>12} {'FPCost':>10} {'NetCost':>10}")
    print(f"  {'─'*90}")
    for m in all_metrics:
        print(f"  {m['model']:<30} {m['recall']:>8.4f} {m['precision']:>8.4f} "
              f"{m['auc_roc']:>8.4f} {m['fraud_loss_fn']:>12,.0f} "
              f"{m['false_alarm_cost_fp']:>10,.0f} {m['net_cost']:>10,.0f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_cost_analysis(all_metrics, output_dir)
    plot_threshold_analysis(y_val, all_probas, output_dir)

    # ── Business impact summary ────────────────────────────────────────────────
    best_recall = max(all_metrics, key=lambda m: m["recall"])
    best_cost   = min(all_metrics, key=lambda m: m["net_cost"])
    best_auc    = max(all_metrics, key=lambda m: m["auc_roc"])

    # Standard vs cost-sensitive gain
    std_models = [m for m in all_metrics if "standard" in m["model"]]
    cs_models  = [m for m in all_metrics if "cost_sensitive" in m["model"]]
    avg_recall_std = np.mean([m["recall"] for m in std_models])
    avg_recall_cs  = np.mean([m["recall"] for m in cs_models])
    avg_cost_std   = np.mean([m["net_cost"] for m in std_models])
    avg_cost_cs    = np.mean([m["net_cost"] for m in cs_models])

    print(f"\n  Standard models    → avg recall={avg_recall_std:.4f}  avg cost=${avg_cost_std:,.0f}")
    print(f"  Cost-sensitive     → avg recall={avg_recall_cs:.4f}  avg cost=${avg_cost_cs:,.0f}")
    print(f"  Recall improvement : +{(avg_recall_cs - avg_recall_std)*100:.2f}%")
    print(f"  Cost reduction     : ${avg_cost_std - avg_cost_cs:,.0f} saved")

    report = {
        "cost_parameters": {
            "false_negative_cost": COST_FALSE_NEGATIVE,
            "false_positive_cost": COST_FALSE_POSITIVE,
            "true_positive_value": abs(COST_TRUE_POSITIVE),
        },
        "all_models":             all_metrics,
        "best_recall_model":      best_recall["model"],
        "best_cost_model":        best_cost["model"],
        "best_auc_model":         best_auc["model"],
        "standard_avg_recall":    round(avg_recall_std, 4),
        "cost_sensitive_avg_recall": round(avg_recall_cs, 4),
        "recall_improvement_pct": round((avg_recall_cs - avg_recall_std) * 100, 2),
        "cost_savings":           round(avg_cost_std - avg_cost_cs, 2),
        "recommendation": (
            f"Use '{best_cost['model']}' for minimum business cost. "
            f"Cost-sensitive training improves recall by "
            f"{(avg_recall_cs - avg_recall_std)*100:.1f}% on average."
        ),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",  required=True)
    parser.add_argument("--val_path",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--report_path", required=True)
    args = parser.parse_args()
    run_cost_sensitive_analysis(
        args.train_path, args.val_path, args.output_dir, args.report_path
    )


if __name__ == "__main__":
    main()
