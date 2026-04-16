"""
Tests for sensor/components/data_validation.py
All MongoDB and filesystem calls are mocked — no external dependencies needed.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from sensor.components.data_validation import DataValidation


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_validator(schema_config, ingestion_artifact, validation_config):
    """Build a DataValidation instance with a patched schema."""
    with patch(
        "sensor.components.data_validation.read_yaml_file",
        return_value=schema_config,
    ):
        return DataValidation(ingestion_artifact, validation_config)


# ── validate_number_of_columns ─────────────────────────────────────────────

class TestValidateNumberOfColumns:
    def test_returns_true_when_column_count_matches(
        self, schema_config, mock_data_ingestion_artifact, mock_data_validation_config, sample_dataframe
    ):
        validator = _make_validator(schema_config, mock_data_ingestion_artifact, mock_data_validation_config)
        assert validator.validate_number_of_columns(sample_dataframe) is True

    def test_returns_false_when_column_count_is_less(
        self, schema_config, mock_data_ingestion_artifact, mock_data_validation_config
    ):
        validator = _make_validator(schema_config, mock_data_ingestion_artifact, mock_data_validation_config)
        short_df = pd.DataFrame(np.random.randn(10, 3), columns=[f"col_{i}" for i in range(3)])
        assert validator.validate_number_of_columns(short_df) is False

    def test_returns_false_when_column_count_is_more(
        self, schema_config, mock_data_ingestion_artifact, mock_data_validation_config
    ):
        validator = _make_validator(schema_config, mock_data_ingestion_artifact, mock_data_validation_config)
        wide_df = pd.DataFrame(np.random.randn(10, 10), columns=[f"col_{i}" for i in range(10)])
        assert validator.validate_number_of_columns(wide_df) is False


# ── is_numerical_column_exist ──────────────────────────────────────────────

class TestIsNumericalColumnExist:
    def test_returns_true_when_all_numerical_columns_present(
        self, schema_config, mock_data_ingestion_artifact, mock_data_validation_config, sample_dataframe
    ):
        validator = _make_validator(schema_config, mock_data_ingestion_artifact, mock_data_validation_config)
        assert validator.is_numerical_column_exist(sample_dataframe) is True

    def test_returns_false_when_numerical_column_missing(
        self, schema_config, mock_data_ingestion_artifact, mock_data_validation_config, sample_dataframe
    ):
        validator = _make_validator(schema_config, mock_data_ingestion_artifact, mock_data_validation_config)
        df_missing = sample_dataframe.drop(columns=["col_2"])
        assert validator.is_numerical_column_exist(df_missing) is False
