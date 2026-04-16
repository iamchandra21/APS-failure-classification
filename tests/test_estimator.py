"""
Tests for sensor/ml/model/estimator.py
No I/O or external dependencies.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import os

from sensor.ml.model.estimator import TargetValueMapping, SensorModel, ModelResolver


# ── TargetValueMapping ─────────────────────────────────────────────────────

class TestTargetValueMapping:
    def test_to_dict_returns_correct_mapping(self):
        mapping = TargetValueMapping()
        d = mapping.to_dict()
        assert d["neg"] == 0
        assert d["pos"] == 1

    def test_reverse_mapping_inverts_correctly(self):
        mapping = TargetValueMapping()
        rev = mapping.reverse_mapping()
        assert rev[0] == "neg"
        assert rev[1] == "pos"

    def test_round_trip(self):
        mapping = TargetValueMapping()
        original = mapping.to_dict()
        reversed_map = mapping.reverse_mapping()
        for label, value in original.items():
            assert reversed_map[value] == label


# ── SensorModel ────────────────────────────────────────────────────────────

class TestSensorModel:
    def _make_sensor_model(self):
        preprocessor = MagicMock()
        preprocessor.transform.side_effect = lambda X: X  # identity transform

        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0, 1])

        return SensorModel(preprocessor=preprocessor, model=model)

    def test_predict_calls_preprocessor_then_model(self):
        sensor_model = self._make_sensor_model()
        X = np.random.randn(4, 5)
        result = sensor_model.predict(X)
        sensor_model.preprocessor.transform.assert_called_once_with(X)
        assert len(result) == 4

    def test_predict_output_shape_matches_input_rows(self):
        sensor_model = self._make_sensor_model()
        X = np.random.randn(4, 5)
        result = sensor_model.predict(X)
        assert result.shape == (4,)

    def test_predict_returns_binary_labels(self):
        sensor_model = self._make_sensor_model()
        X = np.random.randn(4, 5)
        result = sensor_model.predict(X)
        assert set(result).issubset({0, 1})


# ── ModelResolver ──────────────────────────────────────────────────────────

class TestModelResolver:
    def test_is_model_exists_returns_false_when_dir_missing(self, tmp_path):
        resolver = ModelResolver(model_dir=str(tmp_path / "nonexistent"))
        assert resolver.is_model_exists() is False

    def test_is_model_exists_returns_false_when_dir_empty(self, tmp_path):
        model_dir = tmp_path / "saved_models"
        model_dir.mkdir()
        resolver = ModelResolver(model_dir=str(model_dir))
        assert resolver.is_model_exists() is False

    def test_get_best_model_path_returns_latest_timestamp(self, tmp_path):
        model_dir = tmp_path / "saved_models"
        # Create two timestamp directories with a dummy model file
        for ts in ["1700000000", "1700000100"]:
            ts_dir = model_dir / ts
            ts_dir.mkdir(parents=True)
            (ts_dir / "model.pkl").touch()

        resolver = ModelResolver(model_dir=str(model_dir))
        best_path = resolver.get_best_model_path()
        # Should pick the later timestamp
        assert "1700000100" in best_path

    def test_is_model_exists_returns_true_when_model_file_present(self, tmp_path):
        model_dir = tmp_path / "saved_models"
        ts_dir = model_dir / "1700000000"
        ts_dir.mkdir(parents=True)
        (ts_dir / "model.pkl").touch()

        resolver = ModelResolver(model_dir=str(model_dir))
        assert resolver.is_model_exists() is True
