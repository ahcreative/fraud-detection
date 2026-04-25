"""
Component 4: Feature Engineering
- Aggregation features (transaction velocity per card/email)
- Interaction features
- Feature selection (removes near-zero variance and highly correlated cols)
- SHAP-based feature importance (post-training hook)
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def build_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build velocity and aggregation features:
    - Count, mean, std of TransactionAmt per card1 / addr1 / P_emaildomain
    These capture 'how unusual is this transaction for this card/address'.
    """
    df = df.copy()

    group_features = []

    # card1 aggregations
    if "card1" in df.columns and "TransactionAmt" in df.columns:
        g = df.groupby("card1")["TransactionAmt"].agg(["count", "mean", "std"]).reset_index()
        g.columns = ["card1", "card1_txn_count", "card1_txn_mean", "card1_txn_std"]
        df = df.merge(g, on="card1", how="left")
        group_features += ["card1_txn_count", "card1_txn_mean", "card1_txn_std"]

    # addr1 aggregations
    if "addr1" in df.columns and "TransactionAmt" in df.columns:
        g = df.groupby("addr1")["TransactionAmt"].agg(["count", "mean"]).reset_index()
        g.columns = ["addr1", "addr1_txn_count", "addr1_txn_mean"]
        df = df.merge(g, on="addr1", how="left")
        group_features += ["addr1_txn_count", "addr1_txn_mean"]

    # Amount deviation from card mean
    if "card1_txn_mean" in df.columns and "TransactionAmt" in df.columns:
        df["amt_deviation_from_card_mean"] = (
            df["TransactionAmt"] - df["card1_txn_mean"]
        ).abs()
        group_features.append("amt_deviation_from_card_mean")

    print(f"  Created {len(group_features)} aggregation features")
    return df


def remove_low_variance_features(df: pd.DataFrame, target_col: str = "isFraud",
                                  threshold: float = 0.01) -> tuple:
    """Remove near-constant features (variance < threshold)."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[numeric_cols].fillna(0))
    kept = [c for c, v in zip(numeric_cols, selector.get_support()) if v]
    removed = [c for c in numeric_cols if c not in kept]

    print(f"  Low-variance removal: dropped {len(removed)} cols, kept {len(kept)}")
    return df.drop(columns=removed, errors="ignore"), removed


def remove_highly_correlated(df: pd.DataFrame, target_col: str = "isFraud",
                              threshold: float = 0.98) -> tuple:
    """Remove highly correlated feature pairs (keep first of pair)."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Only compute on a sample for speed
    sample = df[numeric_cols].fillna(0)
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=42)

    corr_matrix = sample.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"  Correlation removal: dropped {len(to_drop)} highly correlated cols (threshold={threshold})")
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def engineer_features(
    input_path: str,
    output_path: str,
    feature_config_path: str,
    is_train: bool = True,
    dropped_cols_path: str = None,
):
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING")
    print(f"{'='*60}")

    df = pd.read_csv(input_path)
    print(f"  Input shape: {df.shape}")

    target_col = "isFraud" if "isFraud" in df.columns else None
    
    # Separate target column (if present) before any feature transformations
    target_series = None
    if target_col in df.columns:
        target_series = df[target_col].copy()
        df = df.drop(columns=[target_col])
    
    # ── Aggregation features ────────────────────────────────────────────────
    df = build_aggregation_features(df)

    # ── Remove low-variance and correlated features (train only) ────────────
    dropped_low_var = []
    dropped_corr = []

    if is_train:
        df, dropped_low_var = remove_low_variance_features(df , "")  # target_col already removed
        df, dropped_corr    = remove_highly_correlated(df)

        feature_config = {
            "final_features": df.columns.tolist(),
            "dropped_low_variance": dropped_low_var,
            "dropped_correlated": dropped_corr,
            "n_features": df.shape[1],
        }
        os.makedirs(os.path.dirname(feature_config_path), exist_ok=True)
        with open(feature_config_path, "w") as f:
            json.dump(feature_config, f, indent=2)
        print(f"  Feature config saved: {feature_config_path}")
    else:
        # Apply same column dropping (target already separated)
        if dropped_cols_path and os.path.exists(dropped_cols_path):
            with open(dropped_cols_path) as f:
                feature_config = json.load(f)
            cols_to_drop = feature_config["dropped_low_variance"] + feature_config["dropped_correlated"]
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            # Align columns to training features
            for col in feature_config["final_features"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_config["final_features"]]

    # ── Reattach target column if it existed ────────────────────────────────
    if target_series is not None:
        df[target_col] = target_series

    print(f"  Output shape: {df.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",          required=True)
    parser.add_argument("--output_path",         required=True)
    parser.add_argument("--feature_config_path", required=True)
    parser.add_argument("--is_train",            default="true")
    parser.add_argument("--dropped_cols_path",   default=None)
    args = parser.parse_args()

    engineer_features(
        input_path=args.input_path,
        output_path=args.output_path,
        feature_config_path=args.feature_config_path,
        is_train=args.is_train.lower() == "true",
        dropped_cols_path=args.dropped_cols_path,
    )


if __name__ == "__main__":
    main()
