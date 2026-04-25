"""
Task 2 Requirement: Compare at least 2 imbalance handling strategies.
This script runs both SMOTE and Class-Weight strategies on the same data
and produces a detailed side-by-side comparison report.
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, recall_score,
    precision_score, f1_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Simple fast XGB for comparison (lighter than full pipeline)
XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 42, "n_jobs": 2,
    "tree_method": "hist", "verbosity": 0,
    "use_label_encoder": False,
}

COST_FN = 200.0  # Cost of false negative (missed fraud)
COST_FP = 5.0    # Cost of false positive (false alarm)


def prepare_simple_features(df: pd.DataFrame):
    """Quick feature prep for comparison — numeric only, no deep preprocessing."""
    target = "isFraud"
    drop = ["TransactionID", "isFraud"]

    y = df[target].astype(int) if target in df.columns else None
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    # Keep only numeric
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    return X, y


def evaluate_model(model, X_val, y_val, strategy_name):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    business_cost = fn * COST_FN + fp * COST_FP

    return {
        "strategy":       strategy_name,
        "precision":      round(precision_score(y_val, preds, zero_division=0), 4),
        "recall":         round(recall_score(y_val, preds, zero_division=0), 4),
        "f1":             round(f1_score(y_val, preds, zero_division=0), 4),
        "auc_roc":        round(roc_auc_score(y_val, proba), 4),
        "auc_pr":         round(average_precision_score(y_val, proba), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "business_cost":  round(business_cost, 2),
        "fraud_loss":     round(fn * COST_FN, 2),
        "false_alarm_cost": round(fp * COST_FP, 2),
    }


def run_strategy(X_train, y_train, X_val, y_val, strategy: str):
    print(f"\n  --- Strategy: {strategy.upper()} ---")

    params = XGB_PARAMS.copy()

    if strategy == "smote":
        n_min = y_train.sum()
        k = min(5, n_min - 1)
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"    After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

    elif strategy == "class_weight":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / n_pos
        params["scale_pos_weight"] = spw
        X_res, y_res = X_train.copy(), y_train.copy()
        print(f"    scale_pos_weight = {spw:.1f}")

    elif strategy == "undersample":
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print(f"    After undersampling: {pd.Series(y_res).value_counts().to_dict()}")

    elif strategy == "no_handling":
        X_res, y_res = X_train.copy(), y_train.copy()
        print(f"    No imbalance handling (baseline)")

    model = xgb.XGBClassifier(**params)
    model.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=False)
    metrics = evaluate_model(model, X_val, y_val, strategy)

    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"    Biz Cost : ${metrics['business_cost']:,.0f}")
    return metrics, model


def plot_comparison(all_metrics: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    strategies = df["strategy"].tolist()
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics_to_plot = [
        ("recall",         "Recall (↑ better)",        "steelblue"),
        ("precision",      "Precision (↑ better)",     "darkorange"),
        ("f1",             "F1-Score (↑ better)",      "green"),
        ("auc_roc",        "AUC-ROC (↑ better)",       "purple"),
        ("auc_pr",         "AUC-PR (↑ better)",        "crimson"),
        ("business_cost",  "Business Cost $ (↓ better)", "brown"),
    ]

    for ax, (metric, title, color) in zip(axes.flatten(), metrics_to_plot):
        values = df[metric].tolist()
        bars = ax.bar(strategies, values, color=color, alpha=0.8, edgecolor="black")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f"{val:.3f}" if metric != "business_cost" else f"${val:,.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(values) * 1.15)

    plt.suptitle("Imbalance Handling Strategy Comparison\n(IEEE CIS Fraud Detection)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "imbalance_strategy_comparison.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Confusion matrix comparison
    fig, axes = plt.subplots(1, len(all_metrics), figsize=(5 * len(all_metrics), 4))
    if len(all_metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, all_metrics):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{m['strategy']}\nRecall={m['recall']:.3f}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Legit", "Pred Fraud"], fontsize=7)
        ax.set_yticklabels(["True Legit", "True Fraud"], fontsize=7)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.suptitle("Confusion Matrices by Strategy", fontweight="bold")
    plt.tight_layout()
    path2 = os.path.join(output_dir, "imbalance_confusion_matrices.png")
    plt.savefig(path2, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")


def compare_imbalance_strategies(
    merged_data_path: str,
    output_dir:       str,
    report_path:      str,
    sample_rows:      int = 50000,
):
    print(f"\n{'='*60}")
    print("IMBALANCE STRATEGY COMPARISON")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(merged_data_path, nrows=sample_rows)
    print(f"  Loaded {len(df):,} rows (fraud rate: {df['isFraud'].mean():.4f})")

    X, y = prepare_simple_features(df)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"  Train fraud: {y_train.sum()}  Val fraud: {y_val.sum()}")

    strategies = ["no_handling", "smote", "class_weight", "undersample"]
    all_metrics = []
    all_models  = {}

    for strategy in strategies:
        metrics, model = run_strategy(X_train, y_train, X_val, y_val, strategy)
        all_metrics.append(metrics)
        all_models[strategy] = model

    plot_comparison(all_metrics, output_dir)

    # Best by recall (most important for fraud)
    best_recall = max(all_metrics, key=lambda m: m["recall"])
    best_auc    = max(all_metrics, key=lambda m: m["auc_roc"])
    best_cost   = min(all_metrics, key=lambda m: m["business_cost"])

    print(f"\n  Best recall   : {best_recall['strategy']} ({best_recall['recall']:.4f})")
    print(f"  Best AUC-ROC  : {best_auc['strategy']} ({best_auc['auc_roc']:.4f})")
    print(f"  Lowest cost   : {best_cost['strategy']} (${best_cost['business_cost']:,.0f})")

    report = {
        "n_samples": int(len(df)),
        "fraud_rate": float(y.mean()),
        "strategies": all_metrics,
        "best_recall_strategy":   best_recall["strategy"],
        "best_auc_strategy":      best_auc["strategy"],
        "lowest_cost_strategy":   best_cost["strategy"],
        "recommendation": (
            f"Use '{best_recall['strategy']}' for maximum fraud detection recall. "
            f"Use '{best_cost['strategy']}' to minimize business cost."
        ),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_data_path", required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--report_path",      required=True)
    parser.add_argument("--sample_rows",      type=int, default=50000)
    args = parser.parse_args()

    compare_imbalance_strategies(
        merged_data_path=args.merged_data_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        sample_rows=args.sample_rows,
    )


if __name__ == "__main__":
    main()
