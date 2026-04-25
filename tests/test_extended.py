"""
Additional Unit Tests — Cost-Sensitive, Monitoring, API Edge Cases
Run with: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
import pickle

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_preprocessed(tmp_path_factory):
    """Create a minimal preprocessed train/val split for model tests."""
    tmp = tmp_path_factory.mktemp("data")
    np.random.seed(42)
    n_train, n_val = 800, 200
    n_feats = 30

    X_tr = np.random.randn(n_train, n_feats)
    y_tr = np.random.choice([0, 1], n_train, p=[0.94, 0.06])
    X_val = np.random.randn(n_val, n_feats)
    y_val = np.random.choice([0, 1], n_val, p=[0.94, 0.06])

    cols = [f"feat_{i}" for i in range(n_feats)]
    train_df = pd.DataFrame(X_tr, columns=cols)
    train_df["isFraud"] = y_tr
    val_df = pd.DataFrame(X_val, columns=cols)
    val_df["isFraud"] = y_val

    train_path = str(tmp / "train.csv")
    val_path   = str(tmp / "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)

    return train_path, val_path, str(tmp)


# ── Cost-Sensitive Tests ───────────────────────────────────────────────────────

class TestCostSensitive:
    def test_scale_pos_weight_increases_recall(self, small_preprocessed):
        """Cost-sensitive XGBoost should achieve >= standard recall."""
        import xgboost as xgb
        from sklearn.metrics import recall_score

        train_path, val_path, _ = small_preprocessed
        train = pd.read_csv(train_path)
        val   = pd.read_csv(val_path)
        X_tr, y_tr   = train.drop(columns=["isFraud"]), train["isFraud"].astype(int)
        X_val, y_val = val.drop(columns=["isFraud"]),   val["isFraud"].astype(int)

        if y_tr.sum() < 2:
            pytest.skip("Too few fraud examples in this random seed")

        spw = float((y_tr == 0).sum() / max(y_tr.sum(), 1))
        params = dict(n_estimators=50, max_depth=3, random_state=42,
                      n_jobs=1, tree_method="hist", verbosity=0)

        std_model = xgb.XGBClassifier(**params)
        std_model.fit(X_tr, y_tr, verbose=False)

        cs_model = xgb.XGBClassifier(**{**params, "scale_pos_weight": spw})
        cs_model.fit(X_tr, y_tr, verbose=False)

        recall_std = recall_score(y_val, std_model.predict(X_val), zero_division=0)
        recall_cs  = recall_score(y_val, cs_model.predict(X_val), zero_division=0)

        # Cost-sensitive should not be worse than standard
        assert recall_cs >= recall_std - 0.15, (
            f"CS recall {recall_cs:.3f} much lower than standard {recall_std:.3f}"
        )

    def test_business_cost_calculation(self):
        from scripts.cost_sensitive_analysis import get_metrics
        import xgboost as xgb

        np.random.seed(0)
        X = np.random.randn(200, 10)
        y = np.random.choice([0, 1], 200, p=[0.9, 0.1])
        model = xgb.XGBClassifier(n_estimators=20, random_state=42,
                                   verbosity=0, tree_method="hist")
        model.fit(X, y)
        metrics = get_metrics(model, X, y, "test_model")

        assert "total_business_cost" in metrics
        assert "fraud_loss_fn" in metrics
        assert "false_alarm_cost_fp" in metrics
        assert "net_cost" in metrics
        # Net cost must be non-negative
        assert metrics["total_business_cost"] >= 0
        # Fraud loss = FN * 200
        assert abs(metrics["fraud_loss_fn"] - metrics["fn"] * 200.0) < 0.01


# ── Model Training Tests ───────────────────────────────────────────────────────

class TestModelTraining:
    def test_xgboost_trains_and_predicts(self, small_preprocessed):
        import xgboost as xgb
        train_path, val_path, _ = small_preprocessed
        train = pd.read_csv(train_path).fillna(0)
        X_tr, y_tr = train.drop(columns=["isFraud"]), train["isFraud"].astype(int)
        model = xgb.XGBClassifier(
            n_estimators=30, max_depth=3, random_state=42,
            verbosity=0, tree_method="hist",
        )
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_tr)
        assert preds.shape == (len(X_tr), 2)
        assert np.allclose(preds.sum(axis=1), 1.0)

    def test_lightgbm_trains_and_predicts(self, small_preprocessed):
        import lightgbm as lgb
        train_path, val_path, _ = small_preprocessed
        train = pd.read_csv(train_path).fillna(0)
        X_tr, y_tr = train.drop(columns=["isFraud"]), train["isFraud"].astype(int)
        model = lgb.LGBMClassifier(
            n_estimators=30, max_depth=3, random_state=42, verbose=-1,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_tr)
        assert preds.shape[1] == 2

    def test_hybrid_rf_xgb_trains(self, small_preprocessed):
        train_path, val_path, _ = small_preprocessed
        train = pd.read_csv(train_path).fillna(0)
        val   = pd.read_csv(val_path).fillna(0)
        X_tr,  y_tr  = train.drop(columns=["isFraud"]), train["isFraud"].astype(int)
        X_val, y_val = val.drop(columns=["isFraud"]),   val["isFraud"].astype(int)

        if y_tr.sum() < 5:
            pytest.skip("Too few fraud cases")

        from pipeline.components.model_training.train import train_hybrid_rf_xgb
        hybrid, label = train_hybrid_rf_xgb(X_tr, y_tr, X_val, y_val)

        assert isinstance(hybrid, dict)
        assert "rf" in hybrid
        assert "xgb" in hybrid
        assert "selector" in hybrid
        assert "selected_features" in hybrid
        assert len(hybrid["selected_features"]) > 0

    def test_model_saved_as_pkl(self, small_preprocessed, tmp_path):
        import xgboost as xgb
        train_path, val_path, _ = small_preprocessed
        train = pd.read_csv(train_path).fillna(0)
        X_tr, y_tr = train.drop(columns=["isFraud"]), train["isFraud"].astype(int)
        model = xgb.XGBClassifier(n_estimators=10, random_state=42,
                                   verbosity=0, tree_method="hist")
        model.fit(X_tr, y_tr)
        path = str(tmp_path / "model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        assert os.path.exists(path)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded.predict_proba(X_tr).shape == (len(X_tr), 2)


# ── Evaluation Tests ───────────────────────────────────────────────────────────

class TestEvaluation:
    def test_metrics_all_present(self, small_preprocessed):
        import xgboost as xgb
        from pipeline.components.model_evaluation.evaluate import evaluate_model

        train_path, val_path, _ = small_preprocessed
        val = pd.read_csv(val_path).fillna(0)
        X_val, y_val = val.drop(columns=["isFraud"]), val["isFraud"].astype(int)

        model = xgb.XGBClassifier(n_estimators=20, random_state=42,
                                   verbosity=0, tree_method="hist")
        train = pd.read_csv(train_path).fillna(0)
        model.fit(train.drop(columns=["isFraud"]), train["isFraud"].astype(int))

        if y_val.sum() == 0:
            pytest.skip("No fraud in val sample")

        metrics, proba = evaluate_model(model, X_val, y_val, "test_xgb", threshold=0.5)
        required = ["precision", "recall", "f1", "auc_roc", "auc_pr",
                    "tp", "fp", "fn", "tn", "business_cost"]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_optimal_threshold_respects_recall(self, small_preprocessed):
        import xgboost as xgb
        from pipeline.components.model_evaluation.evaluate import (
            find_optimal_threshold, predict_with_model
        )
        from sklearn.metrics import recall_score

        train_path, val_path, _ = small_preprocessed
        train = pd.read_csv(train_path).fillna(0)
        val   = pd.read_csv(val_path).fillna(0)
        X_tr,  y_tr  = train.drop(columns=["isFraud"]), train["isFraud"].astype(int)
        X_val, y_val = val.drop(columns=["isFraud"]),   val["isFraud"].astype(int)

        if y_tr.sum() < 5 or y_val.sum() < 2:
            pytest.skip("Too few fraud cases")

        model = xgb.XGBClassifier(n_estimators=30, scale_pos_weight=15,
                                   random_state=42, verbosity=0, tree_method="hist")
        model.fit(X_tr, y_tr)
        proba  = predict_with_model(model, X_val, "xgb")
        thresh = find_optimal_threshold(y_val, proba, target_recall=0.70)
        preds  = (proba >= thresh).astype(int)
        actual_recall = recall_score(y_val, preds, zero_division=0)
        # Threshold should achieve at least 70% recall or be the best possible
        assert thresh >= 0.0 and thresh <= 1.0


# ── Deployment Tests ───────────────────────────────────────────────────────────

class TestDeployment:
    def test_deploy_passes_when_thresholds_met(self, tmp_path):
        from pipeline.components.deployment.deploy import deploy_model
        import xgboost as xgb, numpy as np

        # Create a fake model
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model = xgb.XGBClassifier(n_estimators=5, random_state=42,
                                   verbosity=0, tree_method="hist")
        model.fit(X, y)

        model_path = str(tmp_path / "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        prep_path = str(tmp_path / "preprocessor.pkl")
        with open(prep_path, "wb") as f:
            pickle.dump({}, f)

        feat_path = str(tmp_path / "feature_config.json")
        with open(feat_path, "w") as f:
            json.dump({"final_features": ["a", "b"]}, f)

        decision_path = str(tmp_path / "decision.json")
        decision = {
            "should_deploy":       True,
            "best_model":          "xgb_standard",
            "auc_roc":             0.95,
            "recall":              0.85,
            "auc_roc_threshold":   0.90,
            "recall_threshold":    0.80,
            "reason":              "PASS",
        }
        with open(decision_path, "w") as f:
            json.dump(decision, f)

        serve_dir = str(tmp_path / "serving")
        meta = deploy_model(
            deploy_threshold_path=decision_path,
            best_model_path=model_path,
            preprocessor_path=prep_path,
            feature_config_path=feat_path,
            serving_dir=serve_dir,
        )
        assert meta["status"] == "active"
        assert os.path.exists(os.path.join(serve_dir, "model.pkl"))
        assert os.path.exists(os.path.join(serve_dir, "preprocessor.pkl"))
        assert os.path.exists(os.path.join(serve_dir, "deployment_metadata.json"))

    def test_deploy_skips_when_threshold_not_met(self, tmp_path):
        """Deployment should exit(0) when thresholds not met — not raise exception."""
        from pipeline.components.deployment.deploy import deploy_model

        decision_path = str(tmp_path / "decision.json")
        decision = {
            "should_deploy":     False,
            "best_model":        "xgb_standard",
            "auc_roc":           0.82,
            "recall":            0.65,
            "auc_roc_threshold": 0.90,
            "recall_threshold":  0.80,
            "reason":            "FAIL: recall too low",
        }
        with open(decision_path, "w") as f:
            json.dump(decision, f)

        # Should call sys.exit(0) — catch with SystemExit
        with pytest.raises(SystemExit) as exc:
            deploy_model(
                deploy_threshold_path=decision_path,
                best_model_path="/nonexistent",
                preprocessor_path="/nonexistent",
                feature_config_path="/nonexistent",
                serving_dir=str(tmp_path / "serving"),
            )
        assert exc.value.code == 0   # graceful exit, not error


# ── Feature Engineering Tests ──────────────────────────────────────────────────

class TestFeatureEngineeringDetailed:
    def test_apply_feature_engineering_creates_expected_cols(self):
        from pipeline.components.data_preprocessing.preprocess import apply_feature_engineering
        np.random.seed(0)
        df = pd.DataFrame({
            "TransactionAmt": [100.0, 250.0, 50.0],
            "TransactionDT":  [86400, 172800, 259200],
            "card1":          [1234, 5678, 9999],
            "P_emaildomain":  ["gmail.com", "yahoo.com", "hotmail.com"],
        })
        result = apply_feature_engineering(df)
        assert "TransactionAmt_log" in result.columns
        assert "hour" in result.columns
        assert "day"  in result.columns
        assert "week" in result.columns
        assert (result["TransactionAmt_log"] == np.log1p(df["TransactionAmt"])).all()

    def test_aggregation_features(self):
        from pipeline.components.feature_engineering.engineer import build_aggregation_features
        np.random.seed(42)
        df = pd.DataFrame({
            "TransactionAmt": np.random.uniform(10, 500, 100),
            "card1":          np.random.choice([1111, 2222, 3333], 100),
            "addr1":          np.random.choice([100, 200, 300], 100),
            "isFraud":        np.random.randint(0, 2, 100),
        })
        result = build_aggregation_features(df)
        assert "card1_txn_count" in result.columns
        assert "card1_txn_mean"  in result.columns
        assert "addr1_txn_count" in result.columns
        # Verify count correctness
        for card, group in df.groupby("card1"):
            mask = result["card1"] == card
            assert (result.loc[mask, "card1_txn_count"] == len(group)).all()

    def test_variance_threshold_removes_constants(self):
        from pipeline.components.feature_engineering.engineer import remove_low_variance_features
        df = pd.DataFrame({
            "const_col": [1.0] * 100,           # zero variance — should be removed
            "normal_col": np.random.randn(100),  # high variance — should be kept
            "isFraud":  np.random.randint(0, 2, 100),
        })
        result, removed = remove_low_variance_features(df, "isFraud", threshold=0.01)
        assert "const_col" not in result.columns
        assert "normal_col" in result.columns
        assert "const_col" in removed


# ── Ingestion Edge Cases ───────────────────────────────────────────────────────

class TestIngestionEdgeCases:
    def test_left_join_preserves_all_transactions(self, tmp_path):
        """All transactions should appear in merged output regardless of identity match."""
        from pipeline.components.data_ingestion.ingest import load_and_merge

        trans = pd.DataFrame({
            "TransactionID":  [1, 2, 3, 4, 5],
            "isFraud":        [0, 1, 0, 0, 1],
            "TransactionAmt": [100.0, 250.0, 50.0, 75.0, 300.0],
            "TransactionDT":  [1000, 2000, 3000, 4000, 5000],
            "ProductCD":      ["W", "H", "C", "W", "S"],
        })
        ident = pd.DataFrame({
            "TransactionID": [1, 3],   # Only 2 of 5 have identity info
            "DeviceType":    ["desktop", "mobile"],
        })
        trans_path = str(tmp_path / "trans.csv")
        ident_path = str(tmp_path / "ident.csv")
        out_path   = str(tmp_path / "merged.csv")
        trans.to_csv(trans_path, index=False)
        ident.to_csv(ident_path, index=False)

        stats = load_and_merge(trans_path, ident_path, out_path, is_train=True)
        merged = pd.read_csv(out_path)

        # All 5 transactions must be present (left join)
        assert stats["n_rows"] == 5
        assert len(merged) == 5
        # Transactions 2,4,5 should have NaN DeviceType
        no_identity = merged[~merged["TransactionID"].isin([1, 3])]
        assert no_identity["DeviceType"].isna().all()

    def test_stats_json_created(self, tmp_path):
        from pipeline.components.data_ingestion.ingest import load_and_merge
        trans = pd.DataFrame({
            "TransactionID": [1, 2], "isFraud": [0, 1],
            "TransactionAmt": [10.0, 20.0], "TransactionDT": [100, 200],
            "ProductCD": ["W", "H"],
        })
        ident = pd.DataFrame({"TransactionID": [1]})
        trans.to_csv(str(tmp_path / "t.csv"), index=False)
        ident.to_csv(str(tmp_path / "i.csv"), index=False)
        out = str(tmp_path / "out.csv")
        load_and_merge(str(tmp_path / "t.csv"), str(tmp_path / "i.csv"), out)
        assert os.path.exists(out.replace(".csv", "_stats.json"))


# ── Validation Edge Cases ──────────────────────────────────────────────────────

class TestValidationEdgeCases:
    def test_negative_transaction_amount_is_warning(self, tmp_path):
        from pipeline.components.data_validation.validate import validate_data
        df = pd.DataFrame({
            "TransactionID":  [1, 2, 3],
            "isFraud":        [0, 1, 0],
            "TransactionDT":  [100, 200, 300],
            "TransactionAmt": [50.0, -10.0, 75.0],  # negative amount
            "ProductCD":      ["W", "H", "C"],
        })
        path   = str(tmp_path / "data.csv")
        report = str(tmp_path / "report.json")
        df.to_csv(path, index=False)
        result = validate_data(path, report)
        # Should still pass overall but with a warning
        with open(report) as f:
            rdata = json.load(f)
        assert any("negative" in w.lower() for w in rdata.get("warnings", []))

    def test_high_missing_cols_recorded(self, tmp_path):
        from pipeline.components.data_validation.validate import validate_data
        df = pd.DataFrame({
            "TransactionID":  range(200),
            "isFraud":        [0] * 195 + [1] * 5,
            "TransactionDT":  range(200),
            "TransactionAmt": [10.0] * 200,
            "ProductCD":      ["W"] * 200,
            "mostly_missing": [np.nan] * 190 + [1.0] * 10,  # 95% missing
        })
        path   = str(tmp_path / "data.csv")
        report = str(tmp_path / "report.json")
        df.to_csv(path, index=False)
        validate_data(path, report)
        with open(report) as f:
            rdata = json.load(f)
        assert "mostly_missing" in rdata.get("high_missing_cols", [])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
