"""
Task 7: Drift Simulation
- Time-based drift: train on earlier transactions, test on later ones
- New fraud patterns introduced in later period
- Feature importance shifts detected via SHAP comparison
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def split_temporal(df: pd.DataFrame, train_frac: float = 0.7):
    """
    Split dataset by time (TransactionDT).
    Earlier transactions = train (Period A)
    Later transactions   = test/drift (Period B)
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    period_a = df.iloc[:split_idx].copy()
    period_b = df.iloc[split_idx:].copy()
    print(f"  Period A (train) : {len(period_a):,} rows  "
          f"(DT {period_a['TransactionDT'].min():.0f} – {period_a['TransactionDT'].max():.0f})")
    print(f"  Period B (drift) : {len(period_b):,} rows  "
          f"(DT {period_b['TransactionDT'].min():.0f} – {period_b['TransactionDT'].max():.0f})")
    return period_a, period_b


def inject_new_fraud_patterns(df_b: pd.DataFrame, fraud_multiplier: float = 2.5,
                               random_state: int = 42):
    """
    Simulate new fraud patterns in Period B:
    1. Higher transaction amounts for fraudulent transactions
    2. New email domain patterns (spoofed domains)
    3. Increased fraud rate
    """
    rng = np.random.default_rng(random_state)
    df = df_b.copy()

    if "isFraud" not in df.columns:
        return df

    fraud_idx = df[df["isFraud"] == 1].index

    # Pattern 1: New high-value fraud (amount spike)
    if "TransactionAmt" in df.columns and len(fraud_idx) > 0:
        spike_idx = rng.choice(fraud_idx, size=min(len(fraud_idx) // 3, 500), replace=False)
        df.loc[spike_idx, "TransactionAmt"] *= rng.uniform(3, 8, size=len(spike_idx))
        print(f"  Injected high-value fraud pattern: {len(spike_idx)} transactions")

    # Pattern 2: Flip some legitimate transactions to fraud (increase fraud rate)
    legit_idx = df[df["isFraud"] == 0].index
    n_new_fraud = int(len(fraud_idx) * (fraud_multiplier - 1))
    n_new_fraud = min(n_new_fraud, len(legit_idx) // 10)  # cap at 10%
    if n_new_fraud > 0:
        new_fraud_idx = rng.choice(legit_idx, size=n_new_fraud, replace=False)
        df.loc[new_fraud_idx, "isFraud"] = 1
        print(f"  Increased fraud rate: +{n_new_fraud} new fraud cases")

    # Pattern 3: Simulate C-feature shifts (velocity features change behavior)
    c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    if c_cols and len(fraud_idx) > 0:
        for col in c_cols[:3]:
            df.loc[fraud_idx, col] = df.loc[fraud_idx, col] * rng.uniform(1.5, 3.0, size=len(fraud_idx))
        print(f"  Shifted velocity features (C-cols) for fraud transactions")

    new_fraud_rate = df["isFraud"].mean()
    print(f"  New fraud rate in Period B: {new_fraud_rate:.4f} (was {df_b['isFraud'].mean():.4f})")
    return df


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI):
    PSI < 0.1  → No significant change
    PSI 0.1–0.2 → Moderate change
    PSI > 0.2  → Significant change
    """
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Avoid log(0)
    exp_pct = np.where(exp_pct == 0, 0.0001, exp_pct)
    act_pct = np.where(act_pct == 0, 0.0001, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_ks_statistic(a: np.ndarray, b: np.ndarray) -> tuple:
    """Kolmogorov-Smirnov test for distribution drift."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    ks_stat, p_value = stats.ks_2samp(a, b)
    return float(ks_stat), float(p_value)


def detect_feature_drift(period_a: pd.DataFrame, period_b: pd.DataFrame,
                          feature_cols: list, output_dir: str) -> pd.DataFrame:
    """
    Detect drift for each feature using PSI and KS test.
    Returns DataFrame with drift scores per feature.
    """
    results = []
    for col in feature_cols:
        if col not in period_a.columns or col not in period_b.columns:
            continue
        if not pd.api.types.is_numeric_dtype(period_a[col]):
            continue

        a_vals = period_a[col].dropna().values
        b_vals = period_b[col].dropna().values

        if len(a_vals) < 10 or len(b_vals) < 10:
            continue

        psi = compute_psi(a_vals, b_vals)
        ks, p = compute_ks_statistic(a_vals, b_vals)

        results.append({
            "feature":     col,
            "psi":         round(psi, 4),
            "ks_stat":     round(ks, 4),
            "ks_p_value":  round(p, 4),
            "drift_level": "HIGH" if psi > 0.2 else "MEDIUM" if psi > 0.1 else "LOW",
            "drifted":     psi > 0.1 or p < 0.05,
        })

    drift_df = pd.DataFrame(results).sort_values("psi", ascending=False)
    return drift_df


def plot_drift_report(drift_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Top drifted features bar chart
    top_n = min(20, len(drift_df))
    top_drift = drift_df.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["red" if d == "HIGH" else "orange" if d == "MEDIUM" else "green"
              for d in top_drift["drift_level"]]
    axes[0].barh(top_drift["feature"], top_drift["psi"], color=colors)
    axes[0].axvline(0.1, color="orange", linestyle="--", label="Moderate threshold (0.1)")
    axes[0].axvline(0.2, color="red",    linestyle="--", label="High threshold (0.2)")
    axes[0].set_xlabel("PSI (Population Stability Index)")
    axes[0].set_title("Feature Drift — PSI Scores")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    drift_counts = drift_df["drift_level"].value_counts()
    axes[1].pie(drift_counts.values, labels=drift_counts.index,
                colors=["red", "orange", "green"][:len(drift_counts)],
                autopct="%1.0f%%")
    axes[1].set_title("Drift Level Distribution")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_drift_report.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_distributions(period_a: pd.DataFrame, period_b: pd.DataFrame,
                                top_features: list, output_dir: str):
    """Plot distribution comparison for top drifted features."""
    n = min(6, len(top_features))
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, feat in enumerate(top_features[:n]):
        if feat not in period_a.columns:
            continue
        a_vals = period_a[feat].dropna()
        b_vals = period_b[feat].dropna()

        # Clip to 1st-99th percentile for visibility
        p1, p99 = np.percentile(np.concatenate([a_vals, b_vals]), [1, 99])
        a_clip = a_vals.clip(p1, p99)
        b_clip = b_vals.clip(p1, p99)

        axes[i].hist(a_clip, bins=50, alpha=0.6, label="Period A (train)", color="blue", density=True)
        axes[i].hist(b_clip, bins=50, alpha=0.6, label="Period B (drift)", color="red",  density=True)
        axes[i].set_title(f"{feat}")
        axes[i].legend(fontsize=7)

    plt.suptitle("Distribution Shift: Period A vs Period B (Top Drifted Features)")
    plt.tight_layout()
    path = os.path.join(output_dir, "distribution_comparison.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def simulate_drift(
    merged_data_path: str,
    output_dir: str,
    drift_report_path: str,
    train_frac: float = 0.7,
    inject_patterns: bool = True,
):
    print(f"\n{'='*60}")
    print("DRIFT SIMULATION (Time-based)")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(merged_data_path, nrows=200000)  # Cap for laptop memory
    print(f"  Loaded {len(df):,} rows")

    # Temporal split
    period_a, period_b = split_temporal(df, train_frac)

    # Inject new fraud patterns into period B
    if inject_patterns and "isFraud" in df.columns:
        period_b_drifted = inject_new_fraud_patterns(period_b)
    else:
        period_b_drifted = period_b.copy()

    # Save split datasets
    period_a.to_csv(os.path.join(output_dir, "period_a_train.csv"), index=False)
    period_b_drifted.to_csv(os.path.join(output_dir, "period_b_drifted.csv"), index=False)

    # Feature drift detection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    important_cols = (
        ["TransactionAmt", "card1", "addr1", "dist1", "dist2"]
        + [f"C{i}" for i in range(1, 10)]
        + [f"D{i}" for i in range(1, 10)]
        + [f"V{i}" for i in range(1, 50)]
    )
    check_cols = [c for c in important_cols if c in numeric_cols]

    print(f"\n  Running drift detection on {len(check_cols)} features ...")
    drift_df = detect_feature_drift(period_a, period_b_drifted, check_cols, output_dir)

    n_drifted = drift_df["drifted"].sum()
    high_drift = (drift_df["drift_level"] == "HIGH").sum()
    print(f"  Features drifted : {n_drifted} / {len(drift_df)}")
    print(f"  High drift       : {high_drift}")

    # Plots
    plot_drift_report(drift_df, output_dir)
    top_drifted = drift_df[drift_df["drifted"]]["feature"].head(6).tolist()
    plot_feature_distributions(period_a, period_b_drifted, top_drifted, output_dir)

    # Fraud rate comparison
    report = {
        "n_period_a": int(len(period_a)),
        "n_period_b": int(len(period_b_drifted)),
        "fraud_rate_a": float(period_a["isFraud"].mean()) if "isFraud" in period_a.columns else None,
        "fraud_rate_b": float(period_b_drifted["isFraud"].mean()) if "isFraud" in period_b_drifted.columns else None,
        "n_features_checked": int(len(drift_df)),
        "n_features_drifted": int(n_drifted),
        "n_features_high_drift": int(high_drift),
        "top_drifted_features": drift_df.head(10).to_dict(orient="records"),
        "drift_threshold_psi": 0.1,
        "retrain_recommended": bool(high_drift > 3 or n_drifted > len(drift_df) * 0.3),
    }

    with open(drift_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Drift report: {drift_report_path}")
    print(f"  Retrain recommended: {report['retrain_recommended']}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_data_path",  required=True)
    parser.add_argument("--output_dir",        required=True)
    parser.add_argument("--drift_report_path", required=True)
    parser.add_argument("--train_frac",        type=float, default=0.7)
    parser.add_argument("--inject_patterns",   default="true")
    args = parser.parse_args()

    simulate_drift(
        merged_data_path=args.merged_data_path,
        output_dir=args.output_dir,
        drift_report_path=args.drift_report_path,
        train_frac=args.train_frac,
        inject_patterns=args.inject_patterns.lower() == "true",
    )


if __name__ == "__main__":
    main()
