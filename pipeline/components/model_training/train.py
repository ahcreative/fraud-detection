"""
Component 5: Model Training
Models:
  1. XGBoost (standard)
  2. XGBoost (cost-sensitive)
  3. LightGBM (standard)
  4. LightGBM (cost-sensitive)
  5. Hybrid: RandomForest feature selection + XGBoost (RF+XGB)

Resource-aware hyperparameters tuned for 16GB RAM / Core i5 laptop.
"""

import argparse
import json
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb


# ── Laptop-friendly hyperparameters ──────────────────────────────────────────
XGB_BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 2,          # limit CPU for laptop
    "tree_method": "hist", # fast histogram method
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "verbosity": 0,
}

LGB_BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 2,
    "verbose": -1,
    "metric": "average_precision",
}

RF_SELECTOR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "random_state": 42,
    "n_jobs": 2,
    "class_weight": "balanced",
}


def load_data(train_path: str, val_path: str):
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)

    target = "isFraud"
    X_train = train.drop(columns=[target])
    y_train = train[target].astype(int)
    X_val   = val.drop(columns=[target])
    y_val   = val[target].astype(int)

    # Fill any remaining NaN
    X_train = X_train.fillna(0)
    X_val   = X_val.fillna(0)

    print(f"  Train: {X_train.shape}, fraud={y_train.sum()}")
    print(f"  Val  : {X_val.shape},   fraud={y_val.sum()}")
    return X_train, y_train, X_val, y_val


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg / n_pos)


def train_xgboost(X_train, y_train, X_val, y_val,
                   cost_sensitive: bool = False, scale_pos_weight: float = None):
    params = XGB_BASE_PARAMS.copy()
    label = "xgb_cost_sensitive" if cost_sensitive else "xgb_standard"

    if cost_sensitive:
        spw = scale_pos_weight or compute_scale_pos_weight(y_train)
        params["scale_pos_weight"] = spw
        print(f"  [XGB cost-sensitive] scale_pos_weight={spw:.1f}")
    else:
        print(f"  [XGB standard]")

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model, label


def train_lightgbm(X_train, y_train, X_val, y_val,
                    cost_sensitive: bool = False, scale_pos_weight: float = None):
    params = LGB_BASE_PARAMS.copy()
    label = "lgb_cost_sensitive" if cost_sensitive else "lgb_standard"

    if cost_sensitive:
        spw = scale_pos_weight or compute_scale_pos_weight(y_train)
        params["scale_pos_weight"] = spw
        print(f"  [LGB cost-sensitive] scale_pos_weight={spw:.1f}")
    else:
        print(f"  [LGB standard]")

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return model, label


def train_hybrid_rf_xgb(X_train, y_train, X_val, y_val):
    """
    Hybrid: RandomForest selects top features → XGBoost trains on them.
    RF provides feature importance; XGB provides final predictions.
    """
    print("  [Hybrid RF+XGB] Step 1: RF feature selection ...")
    rf = RandomForestClassifier(**RF_SELECTOR_PARAMS)
    rf.fit(X_train, y_train)

    selector = SelectFromModel(rf, threshold="median", prefit=True)
    X_train_sel = selector.transform(X_train)
    X_val_sel   = selector.transform(X_val)
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"    RF selected {len(selected_features)} / {X_train.shape[1]} features")

    print("  [Hybrid RF+XGB] Step 2: XGBoost on selected features ...")
    spw = compute_scale_pos_weight(y_train)
    params = XGB_BASE_PARAMS.copy()
    params["scale_pos_weight"] = spw

    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        X_train_sel, y_train,
        eval_set=[(X_val_sel, y_val)],
        verbose=False,
    )

    hybrid = {
        "rf": rf,
        "selector": selector,
        "xgb": xgb_model,
        "selected_features": selected_features,
    }
    return hybrid, "hybrid_rf_xgb"


def save_model(model, label: str, output_dir: str):
    path = os.path.join(output_dir, f"{label}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved: {path}")
    return path


def train_all_models(
    train_path: str,
    val_path: str,
    models_dir: str,
    training_config_path: str,
):
    print(f"\n{'='*60}")
    print("MODEL TRAINING")
    print(f"{'='*60}")

    os.makedirs(models_dir, exist_ok=True)
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)

    spw = compute_scale_pos_weight(y_train)
    results = {}

    models_to_train = [
        ("xgb_standard",       lambda: train_xgboost(X_train, y_train, X_val, y_val, cost_sensitive=False)),
        ("xgb_cost_sensitive", lambda: train_xgboost(X_train, y_train, X_val, y_val, cost_sensitive=True, scale_pos_weight=spw)),
        ("lgb_standard",       lambda: train_lightgbm(X_train, y_train, X_val, y_val, cost_sensitive=False)),
        ("lgb_cost_sensitive", lambda: train_lightgbm(X_train, y_train, X_val, y_val, cost_sensitive=True, scale_pos_weight=spw)),
        ("hybrid_rf_xgb",      lambda: train_hybrid_rf_xgb(X_train, y_train, X_val, y_val)),
    ]

    for name, train_fn in models_to_train:
        print(f"\nTraining: {name}")
        t0 = time.time()
        model, label = train_fn()
        elapsed = time.time() - t0
        path = save_model(model, label, models_dir)
        results[label] = {
            "path": path,
            "train_time_sec": round(elapsed, 1),
            "n_train": int(len(y_train)),
            "n_fraud_train": int(y_train.sum()),
        }
        print(f"    Training time: {elapsed:.1f}s")

    # Save feature columns for inference
    feature_cols_path = os.path.join(models_dir, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump({"feature_cols": X_train.columns.tolist()}, f)

    # Save training config
    config = {
        "models": results,
        "feature_columns_path": feature_cols_path,
        "scale_pos_weight": spw,
        "train_path": train_path,
        "val_path": val_path,
    }
    os.makedirs(os.path.dirname(training_config_path), exist_ok=True)
    with open(training_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Training config saved: {training_config_path}")

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",           required=True)
    parser.add_argument("--val_path",             required=True)
    parser.add_argument("--models_dir",           required=True)
    parser.add_argument("--training_config_path", required=True)
    args = parser.parse_args()

    train_all_models(
        train_path=args.train_path,
        val_path=args.val_path,
        models_dir=args.models_dir,
        training_config_path=args.training_config_path,
    )


if __name__ == "__main__":
    main()
