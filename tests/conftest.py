"""
Shared pytest fixtures for APS Failure Classification tests.
All fixtures that require external services (MongoDB, AWS) are mocked here
so tests run fully offline in CI/CD.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


# ── Schema fixture (mirrors config/schema.yaml structure) ──────────────────
@pytest.fixture
def schema_config():
    """Minimal schema config with 5 columns for fast unit tests."""
    return {
        "columns": {f"col_{i}": "float64" for i in range(5)},
        "numerical_columns": [f"col_{i}" for i in range(5)],
        "drop_columns": [],
    }


# ── DataFrame fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def sample_dataframe():
    """Clean 5-column dataframe, no missing values."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f"col_{i}" for i in range(5)],
    )


@pytest.fixture
def dataframe_with_na():
    """Dataframe where ~20 % of values are NaN."""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f"col_{i}" for i in range(5)],
    )
    for col in df.columns:
        df.loc[df.sample(frac=0.2).index, col] = np.nan
    return df


@pytest.fixture
def dataframe_with_zero_std():
    """Dataframe where col_0 has zero standard deviation (constant)."""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f"col_{i}" for i in range(5)],
    )
    df["col_0"] = 5.0  # constant → std = 0
    return df


# ── ML array fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def binary_classification_arrays():
    """Small imbalanced arrays suitable for XGBoost training."""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10).astype(np.float32)
    # ~10 % positive class (imbalanced)
    y = (np.random.rand(n_samples) < 0.1).astype(int)
    split = int(n_samples * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


# ── Config / Artifact mocks ────────────────────────────────────────────────
@pytest.fixture
def mock_data_ingestion_artifact():
    artifact = MagicMock()
    artifact.trained_file_path = "dummy_train.csv"
    artifact.test_file_path = "dummy_test.csv"
    return artifact


@pytest.fixture
def mock_data_validation_config():
    config = MagicMock()
    config.drift_report_file_path = "dummy_drift.yaml"
    config.invalid_train_file_path = "dummy_invalid_train.csv"
    config.invalid_test_file_path = "dummy_invalid_test.csv"
    return config


@pytest.fixture
def mock_model_trainer_config():
    config = MagicMock()
    config.expected_accuracy = 0.0          # low threshold so tests always pass
    config.overfitting_underfitting_threshold = 1.0  # wide so never fails
    config.trained_model_fie_path = "dummy_model.pkl"
    return config


@pytest.fixture
def mock_data_transformation_artifact(tmp_path):
    artifact = MagicMock()
    artifact.transformed_train_file_path = str(tmp_path / "train.npy")
    artifact.transformed_test_file_path = str(tmp_path / "test.npy")
    artifact.transformed_object_file_path = str(tmp_path / "preprocessor.pkl")
    return artifact
