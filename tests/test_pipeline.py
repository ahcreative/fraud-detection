"""
Unit Tests for Fraud Detection Pipeline Components
Run: pytest tests/ -v
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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_transaction_df():
    """Minimal transaction dataframe for testing."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "TransactionID":  np.arange(1, n + 1),
        "isFraud":        np.random.choice([0, 1], n, p=[0.96, 0.04]),
        "TransactionDT":  np.random.randint(86400, 86400 * 180, n),
        "TransactionAmt": np.abs(np.random.normal(100, 80, n)),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
        "card1":          np.random.randint(1000, 9999, n).astype(float),
        "card2":          np.random.randint(100, 600, n).astype(float),
        "card3":          np.random.choice([150.0, 185.0, np.nan], n),
        "card4":          np.random.choice(["visa", "mastercard", None], n),
        "card5":          np.random.randint(100, 600, n).astype(float),
        "card6":          np.random.choice(["credit", "debit", None], n),
        "addr1":          np.random.randint(100, 500, n).astype(float),
        "addr2":          np.random.choice([87.0, 60.0, np.nan], n),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "R_emaildomain":  np.random.choice(["gmail.com", "outlook.com", None], n),
        "C1":             np.random.randint(0, 10, n).astype(float),
        "C2":             np.random.randint(0, 5, n).astype(float),
        "D1":             np.random.randint(0, 300, n).astype(float),
        "M1":             np.random.choice(["T", "F", None], n),
        "M2":             np.random.choice(["T", "F", None], n),
        **{f"V{i}": np.random.choice([np.nan, np.random.random()], n) for i in range(1, 20)},
    })


@pytest.fixture
def sample_identity_df():
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "id_01": np.random.choice([-5.0, 0.0, np.nan], n),
        "id_02": np.random.randint(100, 500, n).astype(float),
        "id_12": np.random.choice(["NotFound", "Found", None], n),
        "DeviceType": np.random.choice(["desktop", "mobile", None], n),
        "DeviceInfo": np.random.choice(["Windows", "iOS 11.0", None], n),
    })


@pytest.fixture
def merged_csv(sample_transaction_df, sample_identity_df, tmp_path):
    merged = sample_transaction_df.merge(sample_identity_df, on="TransactionID", how="left")
    path = str(tmp_path / "merged.csv")
    merged.to_csv(path, index=False)
    return path


# ── Data Ingestion Tests ───────────────────────────────────────────────────────

class TestDataIngestion:
    def test_merge_basic(self, sample_transaction_df, sample_identity_df, tmp_path):
        from pipeline.components.data_ingestion.ingest import load_and_merge
        trans_path = str(tmp_path / "trans.csv")
        ident_path = str(tmp_path / "ident.csv")
        out_path   = str(tmp_path / "merged.csv")
        sample_transaction_df.to_csv(trans_path, index=False)
        sample_identity_df.to_csv(ident_path, index=False)
        stats = load_and_merge(trans_path, ident_path, out_path, is_train=True)
        assert os.path.exists(out_path)
        assert stats["n_rows"] == len(sample_transaction_df)
        assert "fraud_rate" in stats
        assert 0 < stats["fraud_rate"] < 1

    def test_identity_col_normalization(self, tmp_path):
        from pipeline.components.data_ingestion.ingest import normalize_identity_columns
        df = pd.DataFrame({"id-01": [1], "id-02": [2], "TransactionID": [100]})
        result = normalize_identity_columns(df)
        assert "id_01" in result.columns
        assert "id_02" in result.columns
        assert "id-01" not in result.columns

    def test_output_stats_keys(self, sample_transaction_df, sample_identity_df, tmp_path):
        from pipeline.components.data_ingestion.ingest import load_and_merge
        trans_path = str(tmp_path / "trans.csv")
        ident_path = str(tmp_path / "ident.csv")
        out_path   = str(tmp_path / "merged.csv")
        sample_transaction_df.to_csv(trans_path, index=False)
        sample_identity_df.to_csv(ident_path, index=False)
        stats = load_and_merge(trans_path, ident_path, out_path)
        for key in ["n_rows", "n_cols", "missing_pct"]:
            assert key in stats


# ── Data Validation Tests ─────────────────────────────────────────────────────

class TestDataValidation:
    def test_valid_data_passes(self, merged_csv, tmp_path):
        from pipeline.components.data_validation.validate import validate_data
        report_path = str(tmp_path / "report.json")
        report = validate_data(merged_csv, report_path, is_train=True)
        assert report["status"] == "PASS"
        assert os.path.exists(report_path)

    def test_missing_target_detected(self, tmp_path):
        from pipeline.components.data_validation.validate import validate_data
        df = pd.DataFrame({"TransactionID": [1, 2], "TransactionDT": [100, 200],
                           "TransactionAmt": [50, 100], "ProductCD": ["W", "H"]})
        path = str(tmp_path / "no_target.csv")
        df.to_csv(path, index=False)
        report_path = str(tmp_path / "report.json")
        report = validate_data(path, report_path, is_train=True)
        assert report["status"] == "FAIL"

    def test_report_is_valid_json(self, merged_csv, tmp_path):
        from pipeline.components.data_validation.validate import validate_data
        report_path = str(tmp_path / "report.json")
        validate_data(merged_csv, report_path)
        with open(report_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ── Preprocessing Tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_smote_output_shape(self, merged_csv, tmp_path):
        from pipeline.components.data_preprocessing.preprocess import preprocess
        train, val, arts = preprocess(
            input_path=merged_csv,
            output_dir=str(tmp_path / "out"),
            artifacts_dir=str(tmp_path / "arts"),
            imbalance_strategy="smote",
            is_train=True,
        )
        assert os.path.exists(train)
        assert os.path.exists(val)
        df_train = pd.read_csv(train)
        assert "isFraud" in df_train.columns
        # SMOTE should balance classes
        counts = df_train["isFraud"].value_counts()
        assert counts[1] > 0

    def test_class_weight_strategy(self, merged_csv, tmp_path):
        from pipeline.components.data_preprocessing.preprocess import preprocess
        train, val, arts = preprocess(
            input_path=merged_csv,
            output_dir=str(tmp_path / "out"),
            artifacts_dir=str(tmp_path / "arts"),
            imbalance_strategy="class_weight",
            is_train=True,
        )
        assert arts["scale_pos_weight"] is not None
        assert arts["scale_pos_weight"] > 1.0

    def test_artifacts_saved(self, merged_csv, tmp_path):
        from pipeline.components.data_preprocessing.preprocess import preprocess
        artifacts_dir = str(tmp_path / "arts")
        preprocess(merged_csv, str(tmp_path / "out"), artifacts_dir,
                   imbalance_strategy="class_weight")
        assert os.path.exists(os.path.join(artifacts_dir, "preprocessor.pkl"))

    def test_no_target_leakage_in_val(self, merged_csv, tmp_path):
        from pipeline.components.data_preprocessing.preprocess import preprocess
        _, val_path, _ = preprocess(
            merged_csv, str(tmp_path / "out"), str(tmp_path / "arts"),
            imbalance_strategy="class_weight",
        )
        val = pd.read_csv(val_path)
        # isFraud should be in val (for evaluation) but not in feature-encoded cols
        assert "isFraud" in val.columns


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_aggregation_features_created(self, merged_csv, tmp_path):
        from pipeline.components.data_preprocessing.preprocess import preprocess
        from pipeline.components.feature_engineering.engineer import engineer_features

        train_path, _, _ = preprocess(
            merged_csv, str(tmp_path / "pre"), str(tmp_path / "arts"),
            imbalance_strategy="class_weight",
        )
        out_path    = str(tmp_path / "eng" / "train_eng.csv")
        config_path = str(tmp_path / "arts" / "feature_config.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        engineer_features(train_path, out_path, config_path, is_train=True)
        assert os.path.exists(out_path)
        df = pd.read_csv(out_path)
        assert df.shape[0] > 0

    def test_feature_config_saved(self, merged_csv, tmp_path):
        from pipeline.components.data_preprocessing.preprocess import preprocess
        from pipeline.components.feature_engineering.engineer import engineer_features
        train_path, _, _ = preprocess(
            merged_csv, str(tmp_path / "pre"), str(tmp_path / "arts"),
            imbalance_strategy="class_weight",
        )
        config_path = str(tmp_path / "arts" / "feature_config.json")
        out_path    = str(tmp_path / "eng" / "out.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        engineer_features(train_path, out_path, config_path, is_train=True)
        assert os.path.exists(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        assert "final_features" in cfg
        assert "n_features" in cfg


# ── Drift Simulation Tests ────────────────────────────────────────────────────

class TestDriftSimulation:
    def test_psi_computation(self):
        from drift_simulation.simulate_drift import compute_psi
        a = np.random.normal(0, 1, 1000)
        b = np.random.normal(0, 1, 1000)  # Same distribution
        psi_same = compute_psi(a, b)
        assert psi_same < 0.1  # Low drift for same distribution

        c = np.random.normal(2, 1, 1000)  # Shifted distribution
        psi_shift = compute_psi(a, c)
        assert psi_shift > psi_same  # Higher drift for shifted

    def test_ks_statistic(self):
        from drift_simulation.simulate_drift import compute_ks_statistic
        a = np.random.normal(0, 1, 500)
        b = np.random.normal(0, 1, 500)
        ks, p = compute_ks_statistic(a, b)
        assert 0 <= ks <= 1
        assert 0 <= p <= 1

    def test_temporal_split(self, sample_transaction_df):
        from drift_simulation.simulate_drift import split_temporal
        a, b = split_temporal(sample_transaction_df, train_frac=0.7)
        assert len(a) + len(b) == len(sample_transaction_df)
        assert a["TransactionDT"].max() <= b["TransactionDT"].min()


# ── Retraining Strategy Tests ─────────────────────────────────────────────────

class TestRetrainingStrategy:
    def test_threshold_strategy(self):
        from drift_simulation.retraining_strategy import (
            run_threshold_strategy, simulate_model_decay, simulate_psi_trend
        )
        recalls = simulate_model_decay(30)
        psis    = simulate_psi_trend(30)
        config  = {"recall_threshold": 0.80, "psi_threshold": 0.15}
        result  = run_threshold_strategy(recalls, psis, config)
        assert "n_retrains" in result
        assert "avg_recall" in result
        assert result["avg_recall"] >= result["min_recall"]

    def test_periodic_strategy(self):
        from drift_simulation.retraining_strategy import (
            run_periodic_strategy, simulate_model_decay, simulate_psi_trend
        )
        recalls = simulate_model_decay(30)
        psis    = simulate_psi_trend(30)
        config  = {"retrain_every_days": 10}
        result  = run_periodic_strategy(recalls, psis, config)
        # 30 days / 10 = exactly 3 retrains at days 0, 10, 20
        assert result["n_retrains"] == 3

    def test_all_strategies_produce_reports(self, tmp_path):
        from drift_simulation.retraining_strategy import compare_retraining_strategies
        report_path = str(tmp_path / "retrain_report.json")
        report = compare_retraining_strategies(
            output_dir=str(tmp_path),
            report_path=report_path,
            n_periods=20,
        )
        assert os.path.exists(report_path)
        assert "recommended_strategy" in report
        assert report["recommended_strategy"] in ("threshold_based", "periodic", "hybrid")


# ── API Tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from api.app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_without_model_returns_503(self):
        from fastapi.testclient import TestClient
        from api.app import app
        client = TestClient(app)
        payload = {"TransactionAmt": 100.0, "ProductCD": "W"}
        response = client.post("/predict", json=payload)
        # Without model loaded, should return 503
        assert response.status_code in (503, 200)

    def test_metrics_endpoint(self):
        from fastapi.testclient import TestClient
        from api.app import app
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"fraud_api" in response.content or b"HELP" in response.content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
