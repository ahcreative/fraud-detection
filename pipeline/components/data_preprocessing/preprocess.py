"""
Component 3: Data Preprocessing
- Missing value imputation (advanced strategies per column type)
- High-cardinality encoding (target encoding / frequency encoding)
- Label encoding for low-cardinality categoricals
- Feature scaling
- Class imbalance handling: SMOTE vs Class Weighting
"""

import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import category_encoders as ce


# ── Column taxonomy ───────────────────────────────────────────────────────────
# Columns with NUMERIC semantics (continuous / ordinal)
NUMERIC_FEATURES = [
    "TransactionDT", "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
] + [f"C{i}" for i in range(1, 15)] \
  + [f"D{i}" for i in range(1, 16)] \
  + [f"V{i}" for i in range(1, 340)] \
  + [f"id_{str(i).zfill(2)}" for i in range(1, 12)] \
  + ["id_13", "id_14", "id_17", "id_18", "id_19", "id_20", "id_21",
     "id_22", "id_24", "id_25", "id_26", "id_32"]

# Columns with CATEGORICAL semantics (textual or coded categories)
CATEGORICAL_FEATURES = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_30", "id_31", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo",
]

# High-cardinality columns → target / frequency encoding
HIGH_CARDINALITY_THRESHOLD = 10
DROP_COLS = ["TransactionID"]


def get_column_lists(df: pd.DataFrame, is_train: bool):
    """Return actual numeric and categorical cols present in df."""
    present_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    present_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    # Any leftover object cols not in taxonomy → treat as categorical
    object_cols = df.select_dtypes(include="object").columns.tolist()
    for c in object_cols:
        if c not in present_cat and c not in DROP_COLS and c != "isFraud":
            present_cat.append(c)
    return present_num, present_cat


def impute_missing(df: pd.DataFrame, num_cols: list, cat_cols: list,
                   imputers: dict = None, fit: bool = True):
    """
    Advanced imputation:
      - Numeric: median imputation (robust to outliers) for V-cols and C-cols;
                 KNN (k=5) for D-cols (time-based, correlated);
                 -999 sentinel for known always-missing groups.
      - Categorical: mode (most_frequent) imputation + "MISSING" sentinel.
    """
    df = df.copy()
    fitted = {}

    # Remove duplicate column names (preserve order)
    num_cols = list(dict.fromkeys(num_cols))
    cat_cols = list(dict.fromkeys(cat_cols))

    # Split numeric cols into groups
    v_c_num = [c for c in num_cols if c.startswith(("V", "C", "card", "addr",
               "dist", "Transaction", "id_0", "id_1"))]
    d_num    = [c for c in num_cols if c.startswith("D")]
    other_num = [c for c in num_cols if c not in v_c_num and c not in d_num]

    all_num_groups = [
        ("median", v_c_num + other_num),
        ("median", d_num),          # KNN too slow → median
    ]

    for strategy, cols in all_num_groups:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        key = f"num_{strategy}_{cols[0]}"
        if fit:
            imp = SimpleImputer(strategy="median")
            df[cols] = imp.fit_transform(df[cols])
            fitted[key] = (imp, cols)
        else:
            imp, _ = imputers[key]
            df[cols] = imp.transform(df[cols])

    # Categorical imputation
    cat_key = "cat_mode"
    if fit:
        imp_cat = SimpleImputer(strategy="most_frequent")
        if cat_cols:
            # Convert to string, replace literal "nan" strings, then impute
            # FIX: iterate over columns to avoid duplicate column mismatch
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).replace("nan", np.nan)
            df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
        fitted[cat_key] = (imp_cat, cat_cols)
    else:
        if cat_cols:
            imp_cat, _ = imputers[cat_key]
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).replace("nan", np.nan)
            df[cat_cols] = imp_cat.transform(df[cat_cols])

    # Add "MISSING" flag features for cols with > 5% original missing
    return df, fitted


def encode_categoricals(df: pd.DataFrame, cat_cols: list, target: pd.Series = None,
                         encoders: dict = None, fit: bool = True):
    """
    Encoding strategy:
      - Low cardinality (<=10 unique): LabelEncoder
      - High cardinality (>10 unique): TargetEncoder (train) / FrequencyEncoder (test)
    """
    df = df.copy()
    fitted_enc = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        n_unique = df[col].nunique()

        if fit:
            if n_unique <= HIGH_CARDINALITY_THRESHOLD:
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col])
                fitted_enc[col] = ("label", enc)
            else:
                # Target encoding with smoothing
                enc = ce.TargetEncoder(cols=[col], smoothing=10, min_samples_leaf=5)
                df[col] = enc.fit_transform(df[[col]], target)[col]
                fitted_enc[col] = ("target", enc)
        else:
            enc_type, enc = encoders[col]
            if enc_type == "label":
                # Handle unseen labels
                classes = set(enc.classes_)
                df[col] = df[col].apply(lambda x: x if x in classes else "UNSEEN")
                enc_unseen = LabelEncoder()
                enc_unseen.classes_ = np.append(enc.classes_, "UNSEEN")
                df[col] = enc_unseen.transform(df[col])
            else:
                df[col] = enc.transform(df[[col]])[col]

    return df, fitted_enc


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Light feature engineering added during preprocessing."""
    df = df.copy()

    # Transaction amount log transform (right-skewed)
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Time-based features from TransactionDT (seconds from a reference point)
    if "TransactionDT" in df.columns:
        df["hour"]    = (df["TransactionDT"] // 3600) % 24
        df["day"]     = (df["TransactionDT"] // (3600 * 24)) % 7
        df["week"]    = (df["TransactionDT"] // (3600 * 24 * 7)) % 52

    # Card-email interaction
    if "card1" in df.columns and "P_emaildomain" in df.columns:
        df["card1_email"] = df["card1"].astype(str) + "_" + df["P_emaildomain"].astype(str)

    return df


def handle_imbalance_smote(X_train, y_train, random_state=42):
    """SMOTE oversampling strategy."""
    print("  Applying SMOTE oversampling ...")
    # Reduce k_neighbors if minority class is very small
    n_minority = y_train.sum()
    k_neighbors = min(5, n_minority - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"    Before: {y_train.value_counts().to_dict()}")
    print(f"    After : {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def handle_imbalance_class_weight(y_train):
    """Returns class_weight dict for sklearn / xgboost scale_pos_weight."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / n_pos
    class_weight = {0: 1.0, 1: float(scale)}
    print(f"  Class weights: {class_weight}  (scale_pos_weight = {scale:.1f})")
    return class_weight


def preprocess(
    input_path: str,
    output_dir: str,
    artifacts_dir: str,
    imbalance_strategy: str = "smote",  # "smote" | "class_weight" | "undersample"
    is_train: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
):
    print(f"\n{'='*60}")
    print(f"DATA PREPROCESSING  (strategy={imbalance_strategy})")
    print(f"{'='*60}")

    df = pd.read_csv(input_path)
    print(f"  Loaded {df.shape[0]:,} rows")

    # Drop ID and target columns before feature processing
    target = None
    if is_train and "isFraud" in df.columns:
        target = df["isFraud"].astype(int)
        df = df.drop(columns=["isFraud"])

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Apply feature engineering first (creates new columns)
    df = apply_feature_engineering(df)

    num_cols, cat_cols = get_column_lists(df, is_train)
    # Remove card1_email (high cardinality, handle separately)
    if "card1_email" in df.columns:
        cat_cols.append("card1_email")

    print(f"  Numeric features  : {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")

    os.makedirs(artifacts_dir, exist_ok=True)

    if is_train:
        # ── Split BEFORE imputation to prevent data leakage ──────────────────
        X_train, X_val, y_train, y_val = train_test_split(
            df, target, test_size=test_size, random_state=random_state, stratify=target
        )
        print(f"\n  Train: {X_train.shape}, Val: {X_val.shape}")

        # ── Impute ────────────────────────────────────────────────────────────
        X_train, imputers = impute_missing(X_train, num_cols, cat_cols, fit=True)
        X_val, _ = impute_missing(X_val, num_cols, cat_cols, imputers=imputers, fit=False)

        # ── Encode ────────────────────────────────────────────────────────────
        X_train, encoders = encode_categoricals(X_train, cat_cols, target=y_train, fit=True)
        X_val, _ = encode_categoricals(X_val, cat_cols, encoders=encoders, fit=False)

        # ── Scale numeric features ────────────────────────────────────────────
        scaler = StandardScaler()
        present_num = [c for c in num_cols if c in X_train.columns]
        X_train[present_num] = scaler.fit_transform(X_train[present_num])
        X_val[present_num]   = scaler.transform(X_val[present_num])

        # ── Handle Imbalance ──────────────────────────────────────────────────
        class_weights = None
        scale_pos_weight = None
        X_train_res, y_train_res = X_train.copy(), y_train.copy()

        if imbalance_strategy == "smote":
            X_train_res, y_train_res = handle_imbalance_smote(X_train, y_train, random_state)
        elif imbalance_strategy == "undersample":
            rus = RandomUnderSampler(random_state=random_state)
            X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
            print(f"  Undersampled: {pd.Series(y_train_res).value_counts().to_dict()}")
        elif imbalance_strategy == "class_weight":
            class_weights = handle_imbalance_class_weight(y_train)
            scale_pos_weight = class_weights[1]

        # ── Save processed data ───────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        X_train_res_df = pd.DataFrame(X_train_res, columns=X_train.columns if hasattr(X_train_res, 'shape') else X_train.columns)
        y_train_res_s  = pd.Series(y_train_res, name="isFraud")

        X_train_res_df["isFraud"] = y_train_res_s.values
        X_val["isFraud"] = y_val.values

        train_out = os.path.join(output_dir, f"train_processed_{imbalance_strategy}.csv")
        val_out   = os.path.join(output_dir, "val_processed.csv")

        X_train_res_df.to_csv(train_out, index=False)
        X_val.to_csv(val_out, index=False)
        print(f"\n  Saved train: {train_out}")
        print(f"  Saved val  : {val_out}")

        # ── Save artifacts for inference ──────────────────────────────────────
        artifacts = {
            "imputers": imputers,
            "encoders": encoders,
            "scaler": scaler,
            "num_cols": present_num,
            "cat_cols": cat_cols,
            "feature_cols": list(X_train.columns),
            "class_weights": class_weights,
            "scale_pos_weight": scale_pos_weight,
            "imbalance_strategy": imbalance_strategy,
        }
        with open(os.path.join(artifacts_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(artifacts, f)
        print(f"  Artifacts saved: {artifacts_dir}/preprocessor.pkl")

        return train_out, val_out, artifacts

    else:
        # ── Test preprocessing (load saved artifacts) ─────────────────────────
        with open(os.path.join(artifacts_dir, "preprocessor.pkl"), "rb") as f:
            artifacts = pickle.load(f)

        df, _ = impute_missing(df, artifacts["num_cols"], artifacts["cat_cols"],
                               imputers=artifacts["imputers"], fit=False)
        df, _ = encode_categoricals(df, artifacts["cat_cols"],
                                    encoders=artifacts["encoders"], fit=False)
        present_num = [c for c in artifacts["num_cols"] if c in df.columns]
        df[present_num] = artifacts["scaler"].transform(df[present_num])

        out_path = os.path.join(output_dir, "test_processed.csv")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Saved test: {out_path}")
        return out_path, None, artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",          required=True)
    parser.add_argument("--output_dir",          required=True)
    parser.add_argument("--artifacts_dir",       required=True)
    parser.add_argument("--imbalance_strategy",  default="smote",
                        choices=["smote", "class_weight", "undersample"])
    parser.add_argument("--is_train",            default="true")
    parser.add_argument("--test_size",           type=float, default=0.2)
    parser.add_argument("--random_state",        type=int, default=42)
    args = parser.parse_args()

    preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        imbalance_strategy=args.imbalance_strategy,
        is_train=args.is_train.lower() == "true",
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
