"""
Component 2: Data Validation
Checks schema, missing value thresholds, class balance, and data types.
Fails pipeline if critical checks don't pass.
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


# ── Column schema expected in the merged training data ────────────────────────
REQUIRED_COLS = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"]
REQUIRED_COLS_TEST = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"]

NUMERIC_COLS = [
    "TransactionDT", "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "D11", "D12", "D13", "D14", "D15",
]

CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
]

MAX_MISSING_PCT = 80.0   # drop columns with > 80% missing
MIN_ROWS = 1000


def validate_data(input_path: str, output_report_path: str, is_train: bool = True) -> dict:
    print(f"\n{'='*60}")
    print("DATA VALIDATION")
    print(f"{'='*60}")

    df = pd.read_csv(input_path)
    print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")

    report = {
        "input_path": input_path,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    # ── Check 1: Minimum row count ────────────────────────────────────────────
    if df.shape[0] < MIN_ROWS:
        msg = f"Too few rows: {df.shape[0]} < {MIN_ROWS}"
        report["errors"].append(msg)
        print(f"  ❌ {msg}")
    else:
        report["checks"]["min_rows"] = "PASS"
        print(f"  ✅ Row count OK: {df.shape[0]:,}")

    # ── Check 2: Required columns ─────────────────────────────────────────────
    required = REQUIRED_COLS if is_train else REQUIRED_COLS_TEST
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        msg = f"Missing required columns: {missing_required}"
        report["errors"].append(msg)
        print(f"  ❌ {msg}")
    else:
        report["checks"]["required_cols"] = "PASS"
        print(f"  ✅ All required columns present")

    # ── Check 3: Missing value analysis ──────────────────────────────────────
    miss_pct = df.isnull().mean() * 100
    high_miss_cols = miss_pct[miss_pct > MAX_MISSING_PCT].index.tolist()
    report["high_missing_cols"] = high_miss_cols
    report["avg_missing_pct"] = float(miss_pct.mean())

    if high_miss_cols:
        msg = f"{len(high_miss_cols)} cols exceed {MAX_MISSING_PCT}% missing: {high_miss_cols[:5]}..."
        report["warnings"].append(msg)
        print(f"  ⚠️  {msg}")
    else:
        report["checks"]["missing_values"] = "PASS"
        print(f"  ✅ No col exceeds {MAX_MISSING_PCT}% missing  (avg={miss_pct.mean():.1f}%)")

    # ── Check 4: Target column (train only) ───────────────────────────────────
    if is_train and "isFraud" in df.columns:
        invalid_target = df["isFraud"].isin([0, 1])
        n_invalid = (~invalid_target).sum()
        if n_invalid > 0:
            report["errors"].append(f"isFraud has {n_invalid} non-binary values")
        else:
            report["checks"]["target_binary"] = "PASS"

        fraud_rate = df["isFraud"].mean()
        report["fraud_rate"] = float(fraud_rate)
        print(f"  ✅ Target column OK — fraud rate: {fraud_rate:.4f}")

        if fraud_rate < 0.001:
            report["warnings"].append("Extreme class imbalance (fraud_rate < 0.1%)")

    # ── Check 5: Numeric column dtypes ───────────────────────────────────────
    type_issues = []
    for col in NUMERIC_COLS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                type_issues.append(col)
    if type_issues:
        report["warnings"].append(f"Expected numeric but got other dtype: {type_issues}")
        print(f"  ⚠️  Type issues in: {type_issues}")
    else:
        report["checks"]["numeric_dtypes"] = "PASS"
        print(f"  ✅ Numeric column dtypes OK")

    # ── Check 6: TransactionAmt sanity ───────────────────────────────────────
    if "TransactionAmt" in df.columns:
        n_negative = (df["TransactionAmt"] < 0).sum()
        if n_negative > 0:
            report["warnings"].append(f"TransactionAmt has {n_negative} negative values")
        else:
            report["checks"]["transaction_amt"] = "PASS"
            print(f"  ✅ TransactionAmt sanity OK")

    # ── Summary ───────────────────────────────────────────────────────────────
    report["status"] = "FAIL" if report["errors"] else "PASS"
    print(f"\n  Status: {report['status']}")
    print(f"  Errors  : {len(report['errors'])}")
    print(f"  Warnings: {len(report['warnings'])}")

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {output_report_path}")

    if report["status"] == "FAIL":
        print("\n❌ Validation FAILED — aborting pipeline")
        sys.exit(1)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",          required=True)
    parser.add_argument("--output_report_path",  required=True)
    parser.add_argument("--is_train",            default="true")
    args = parser.parse_args()
    validate_data(args.input_path, args.output_report_path, args.is_train.lower() == "true")


if __name__ == "__main__":
    main()
