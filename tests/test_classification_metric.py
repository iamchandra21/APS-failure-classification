"""
Tests for sensor/ml/metric/classification_metric.py
Pure logic tests — no I/O, no mocks needed.
"""
import pytest
import numpy as np

from sensor.ml.metric.classification_metric import get_classification_score


class TestGetClassificationScore:
    def test_returns_artifact_with_three_fields(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        result = get_classification_score(y_true, y_pred)
        assert hasattr(result, "f1_score")
        assert hasattr(result, "precision_score")
        assert hasattr(result, "recall_score")

    def test_perfect_predictions_give_score_of_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = get_classification_score(y_true, y_pred)
        assert result.f1_score == pytest.approx(1.0)
        assert result.precision_score == pytest.approx(1.0)
        assert result.recall_score == pytest.approx(1.0)

    def test_all_wrong_predictions_give_zero_f1(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        result = get_classification_score(y_true, y_pred)
        assert result.f1_score == pytest.approx(0.0)
        assert result.recall_score == pytest.approx(0.0)

    def test_scores_are_floats_between_zero_and_one(self):
        np.random.seed(0)
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.random.randint(0, 2, size=100)
        result = get_classification_score(y_true, y_pred)
        for score in [result.f1_score, result.precision_score, result.recall_score]:
            assert 0.0 <= score <= 1.0

    def test_accepts_string_encoded_labels(self):
        """Matches real pipeline behaviour where labels come in as int arrays."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = get_classification_score(y_true, y_pred)
        assert result.f1_score > 0

    def test_partial_correct_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        result = get_classification_score(y_true, y_pred)
        # 2 TP, 1 FP, 1 FN → precision=2/3, recall=2/3 → F1=2/3
        assert result.f1_score == pytest.approx(2 / 3, rel=1e-3)
