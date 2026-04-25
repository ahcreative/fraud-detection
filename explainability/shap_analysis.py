"""
Task 9: Explainability — SHAP Analysis
- Global feature importance (bar + beeswarm)
- Local explanation for individual transactions (waterfall)
- Force plot for a fraud vs legitimate example
- Feature interaction analysis
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


def load_model_and_data(model_path: str, val_path: str, sample_size: int = 2000):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    val = pd.read_csv(val_path).fillna(0)
    y   = val["isFraud"].astype(int) if "isFraud" in val.columns else None
    X   = val.drop(columns=["isFraud"], errors="ignore")

    # Stratified sample
    if y is not None and len(X) > sample_size:
        fraud_idx = y[y == 1].index
        legit_idx = y[y == 0].index
        n_fraud = min(len(fraud_idx), sample_size // 4)
        n_legit = sample_size - n_fraud
        sampled_idx = (
            list(np.random.choice(fraud_idx, n_fraud, replace=False)) +
            list(np.random.choice(legit_idx, min(n_legit, len(legit_idx)), replace=False))
        )
        X = X.iloc[sampled_idx].reset_index(drop=True)
        y = y.iloc[sampled_idx].reset_index(drop=True) if y is not None else None

    print(f"  Sample: {X.shape}  fraud={y.sum() if y is not None else 'N/A'}")
    return model, X, y


def get_underlying_model_and_features(model, X: pd.DataFrame):
    """Handle hybrid model (RF+XGB selector)."""
    if isinstance(model, dict) and "xgb" in model:
        selector = model["selector"]
        xgb_m    = model["xgb"]
        X_sel = pd.DataFrame(
            selector.transform(X),
            columns=X.columns[selector.get_support()]
        )
        return xgb_m, X_sel
    return model, X


def run_shap_global(model, X: pd.DataFrame, output_dir: str, model_name: str):
    """Global feature importance — summary plots."""
    print(f"\n  Computing SHAP values (global) ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs(output_dir, exist_ok=True)

    # ── Bar chart (mean |SHAP|) ────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=25, show=False)
    plt.title(f"Global Feature Importance (Mean |SHAP|)\n{model_name}", pad=15)
    plt.tight_layout()
    p = os.path.join(output_dir, "shap_global_bar.png")
    plt.savefig(p, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {p}")

    # ── Beeswarm ───────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X, max_display=25, show=False)
    plt.title(f"SHAP Beeswarm Plot\n{model_name}", pad=15)
    plt.tight_layout()
    p = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(p, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {p}")

    # ── Top features table ─────────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).head(30)

    top.to_csv(os.path.join(output_dir, "shap_top_features.csv"), index=False)
    print(f"    Top feature: {top.iloc[0]['feature']}  (SHAP={top.iloc[0]['mean_abs_shap']:.4f})")

    return shap_values, top


def run_shap_local_waterfall(explainer, shap_values, X: pd.DataFrame, y: pd.Series,
                              output_dir: str, n_examples: int = 4):
    """Local explanation — waterfall plots for individual predictions."""
    fraud_indices = y[y == 1].index.tolist()[:n_examples // 2]
    legit_indices = y[y == 0].index.tolist()[:n_examples // 2]
    examples = [(i, "FRAUD") for i in fraud_indices] + [(i, "LEGIT") for i in legit_indices]

    for idx, label in examples:
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X.iloc[idx].values,
                feature_names=X.columns.tolist(),
            ),
            max_display=15,
            show=False,
        )
        plt.title(f"Local Explanation — Transaction #{idx} ({label})", pad=10)
        plt.tight_layout()
        p = os.path.join(output_dir, f"shap_waterfall_{label.lower()}_{idx}.png")
        plt.savefig(p, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"    Saved waterfall: {p}")


def run_shap_dependence_plots(shap_values, X: pd.DataFrame, top_features: pd.DataFrame,
                               output_dir: str, n_plots: int = 4):
    """Dependence plots for top features — shows interaction effects."""
    top_cols = top_features["feature"].head(n_plots).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_cols[:4]):
        if feat not in X.columns:
            continue
        feat_idx = X.columns.tolist().index(feat)
        shap.dependence_plot(
            feat_idx, shap_values, X,
            ax=axes[i], show=False, alpha=0.5,
        )
        axes[i].set_title(f"SHAP Dependence: {feat}")

    plt.suptitle("Feature Dependence Plots (SHAP)", y=1.01)
    plt.tight_layout()
    p = os.path.join(output_dir, "shap_dependence_plots.png")
    plt.savefig(p, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {p}")


def run_cost_sensitive_shap_comparison(model_std_path: str, model_cs_path: str,
                                        X: pd.DataFrame, output_dir: str):
    """Compare SHAP importance between standard and cost-sensitive models."""
    print("\n  Comparing standard vs cost-sensitive SHAP ...")

    results = {}
    for label, path in [("standard", model_std_path), ("cost_sensitive", model_cs_path)]:
        if not os.path.exists(path):
            print(f"    Skipping {label}: file not found")
            continue
        with open(path, "rb") as f:
            m = pickle.load(f)
        m_inner, X_inner = get_underlying_model_and_features(m, X)
        exp = shap.TreeExplainer(m_inner)
        sv  = exp.shap_values(X_inner)
        mean_abs = np.abs(sv).mean(axis=0)
        top = pd.DataFrame({
            "feature": X_inner.columns,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(20)
        results[label] = top

    if len(results) < 2:
        return

    std_top = results["standard"].set_index("feature")["mean_abs_shap"]
    cs_top  = results["cost_sensitive"].set_index("feature")["mean_abs_shap"]
    common  = std_top.index.intersection(cs_top.index)[:15]

    x = np.arange(len(common))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, std_top.loc[common], w, label="Standard",       color="steelblue", alpha=0.8)
    ax.bar(x + w/2, cs_top.loc[common],  w, label="Cost-Sensitive", color="coral",     alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Feature Importance Shift: Standard vs Cost-Sensitive Model")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(output_dir, "shap_standard_vs_cost_sensitive.png")
    plt.savefig(p, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {p}")


def run_explainability(
    best_model_path: str,
    val_path: str,
    models_dir: str,
    output_dir: str,
    report_path: str,
    sample_size: int = 1500,
):
    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSIS (SHAP)")
    print(f"{'='*60}")

    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    model, X, y = load_model_and_data(best_model_path, val_path, sample_size)
    model_name  = os.path.basename(best_model_path).replace(".pkl", "")
    model_inner, X_inner = get_underlying_model_and_features(model, X)

    print(f"  Model: {model_name}")
    print(f"  Features: {X_inner.shape[1]}")

    # ── Global analysis ────────────────────────────────────────────────────────
    shap_values, top_features = run_shap_global(model_inner, X_inner, output_dir, model_name)
    explainer = shap.TreeExplainer(model_inner)

    # ── Local waterfall plots ──────────────────────────────────────────────────
    if y is not None:
        run_shap_local_waterfall(explainer, shap_values, X_inner, y, output_dir)

    # ── Dependence plots ───────────────────────────────────────────────────────
    run_shap_dependence_plots(shap_values, X_inner, top_features, output_dir)

    # ── Standard vs cost-sensitive comparison ─────────────────────────────────
    std_path = os.path.join(models_dir, "xgb_standard.pkl")
    cs_path  = os.path.join(models_dir, "xgb_cost_sensitive.pkl")
    run_cost_sensitive_shap_comparison(std_path, cs_path, X_inner, output_dir)

    # ── Business interpretation ────────────────────────────────────────────────
    top10 = top_features.head(10).to_dict(orient="records")
    interpretations = []
    feature_meanings = {
        "TransactionAmt": "Higher transaction amounts increase fraud risk",
        "card1":          "Specific card numbers are associated with fraud clusters",
        "addr1":          "Billing address anomalies indicate stolen card usage",
        "dist1":          "Distance between billing and shipping address is a strong fraud signal",
        "D1":             "Days since last transaction — unusual gaps indicate account takeover",
        "C1":             "Count of payment addresses linked to card — high count = fraud ring",
        "V258":           "Vesta-engineered velocity feature — high value = suspicious pattern",
        "TransactionDT":  "Unusual transaction timing (e.g., 3am) correlates with fraud",
        "P_emaildomain":  "Free email providers (gmail/yahoo) are more fraud-prone",
        "card2":          "Card BIN (bank identification number) fraud patterns",
    }
    for feat_dict in top10:
        feat = feat_dict["feature"]
        interp = feature_meanings.get(feat, f"{feat}: important discriminating feature")
        interpretations.append({"feature": feat, "mean_abs_shap": feat_dict["mean_abs_shap"],
                                  "interpretation": interp})

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "model_name":       model_name,
        "n_samples_analyzed": int(len(X_inner)),
        "n_features":       int(X_inner.shape[1]),
        "top_10_features":  top10,
        "interpretations":  interpretations,
        "plots_generated":  [f for f in os.listdir(output_dir) if f.endswith(".png")],
        "shap_library_version": shap.__version__,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Explainability report: {report_path}")
    print(f"  Plots saved to: {output_dir}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model_path", required=True)
    parser.add_argument("--val_path",        required=True)
    parser.add_argument("--models_dir",      required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--report_path",     required=True)
    parser.add_argument("--sample_size",     type=int, default=1500)
    args = parser.parse_args()

    run_explainability(
        best_model_path=args.best_model_path,
        val_path=args.val_path,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
