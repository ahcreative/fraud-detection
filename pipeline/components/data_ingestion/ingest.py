"""
Component 1: Data Ingestion
Loads and merges train_transaction + train_identity CSVs.
Handles column name normalization (test_identity uses id-XX vs id_XX).
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize id column names: id-01 -> id_01"""
    rename_map = {}
    for col in df.columns:
        if col.startswith("id-"):
            rename_map[col] = col.replace("id-", "id_")
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed {len(rename_map)} identity columns (id-XX -> id_XX)")
    return df


def load_and_merge(
    transaction_path: str,
    identity_path: str,
    output_path: str,
    is_train: bool = True,
) -> dict:
    print(f"\n{'='*60}")
    print(f"DATA INGESTION - {'TRAIN' if is_train else 'TEST'}")
    print(f"{'='*60}")

    # ── Load transaction data ──────────────────────────────────────
    print(f"\nLoading transaction file: {transaction_path}")
    trans = pd.read_csv(transaction_path)
    print(f"  Shape: {trans.shape}")
    print(f"  Memory: {trans.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ── Load identity data ──────────────────────────────────────────
    print(f"\nLoading identity file: {identity_path}")
    ident = pd.read_csv(identity_path)
    ident = normalize_identity_columns(ident)
    print(f"  Shape: {ident.shape}")

    # ── Merge on TransactionID ──────────────────────────────────────
    print("\nMerging on TransactionID (left join) ...")
    merged = trans.merge(ident, on="TransactionID", how="left")
    print(f"  Merged shape: {merged.shape}")
    print(f"  Transactions with identity info: "
          f"{ident['TransactionID'].isin(trans['TransactionID']).sum()} / {len(trans)}")

    # ── Basic stats ─────────────────────────────────────────────────
    stats: dict = {
        "n_rows": int(merged.shape[0]),
        "n_cols": int(merged.shape[1]),
        "missing_pct": float(merged.isnull().mean().mean() * 100),
    }

    if is_train and "isFraud" in merged.columns:
        fraud_rate = merged["isFraud"].mean()
        stats["fraud_rate"] = float(fraud_rate)
        stats["n_fraud"] = int(merged["isFraud"].sum())
        stats["n_legit"] = int((merged["isFraud"] == 0).sum())
        print(f"\n  Fraud rate : {fraud_rate:.4f} ({stats['n_fraud']:,} fraud / {stats['n_legit']:,} legit)")

    # ── Column type catalogue ────────────────────────────────────────
    numerical_cols = merged.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = merged.select_dtypes(include=["object", "category"]).columns.tolist()

    # Known categorical columns stored as ints/floats in the dataset
    known_cat_cols = [
        "ProductCD", "card4", "card6",
        "P_emaildomain", "R_emaildomain",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
        "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
        "id_29", "id_30", "id_31", "id_33", "id_34", "id_35",
        "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo",
    ]
    # Only keep those that actually exist in the merged frame
    known_cat_cols = [c for c in known_cat_cols if c in merged.columns]

    stats["numerical_cols"] = numerical_cols
    stats["categorical_cols"] = categorical_cols
    stats["known_cat_cols"] = known_cat_cols

    print(f"\n  Numerical cols : {len(numerical_cols)}")
    print(f"  Text/object cols: {len(categorical_cols)}")
    print(f"  Known categorical (semantic): {len(known_cat_cols)}")

    # ── Save merged CSV ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")

    # ── Save stats JSON ──────────────────────────────────────────────
    stats_path = output_path.replace(".csv", "_stats.json")
    # Remove large lists before saving to keep JSON small
    json_stats = {k: v for k, v in stats.items()
                  if k not in ("numerical_cols", "categorical_cols", "known_cat_cols")}
    with open(stats_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"  Stats  saved : {stats_path}")

    return stats


# ── KFP-compatible entry-point ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction_path", required=True)
    parser.add_argument("--identity_path",    required=True)
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--is_train",         default="true")
    args = parser.parse_args()

    is_train = args.is_train.lower() == "true"
    load_and_merge(args.transaction_path, args.identity_path, args.output_path, is_train)


if __name__ == "__main__":
    main()
