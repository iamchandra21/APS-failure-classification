"""
Tests for sensor/components/model_trainer.py
Uses tiny synthetic arrays so tests run in seconds with no I/O.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from sensor.components.model_trainer import ModelTrainer


class TestTrainModel:
    """train_model() should return a fitted XGBClassifier."""

    def test_returns_fitted_model(
        self, mock_model_trainer_config, mock_data_transformation_artifact, binary_classification_arrays
    ):
        x_train, y_train, _, _ = binary_classification_arrays
        trainer = ModelTrainer(mock_model_trainer_config, mock_data_transformation_artifact)
        model = trainer.train_model(x_train, y_train)
        preds = model.predict(x_train)
        assert preds.shape == (len(x_train),)

    def test_predictions_are_binary(
        self, mock_model_trainer_config, mock_data_transformation_artifact, binary_classification_arrays
    ):
        x_train, y_train, x_test, _ = binary_classification_arrays
        trainer = ModelTrainer(mock_model_trainer_config, mock_data_transformation_artifact)
        model = trainer.train_model(x_train, y_train)
        preds = model.predict(x_test)
        assert set(preds).issubset({0, 1})

    def test_model_has_feature_importances(
        self, mock_model_trainer_config, mock_data_transformation_artifact, binary_classification_arrays
    ):
        x_train, y_train, _, _ = binary_classification_arrays
        trainer = ModelTrainer(mock_model_trainer_config, mock_data_transformation_artifact)
        model = trainer.train_model(x_train, y_train)
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == x_train.shape[1]


class TestInitiateModelTrainer:
    """initiate_model_trainer() wires up the full I/O flow."""

    def test_raises_when_f1_below_threshold(
        self, mock_model_trainer_config, mock_data_transformation_artifact, binary_classification_arrays, tmp_path
    ):
        """If model F1 is below expected_accuracy, an exception must be raised."""
        x_train, y_train, x_test, y_test = binary_classification_arrays

        # Stack into arrays matching the .npy format (features + target as last column)
        train_arr = np.c_[x_train, y_train]
        test_arr = np.c_[x_test, y_test]

        # Set a threshold that is impossible to meet
        mock_model_trainer_config.expected_accuracy = 1.1

        preprocessor_mock = MagicMock()
        preprocessor_mock.get_feature_names_out.return_value = [f"col_{i}" for i in range(10)]

        with patch("sensor.components.model_trainer.load_numpy_array_data", side_effect=[train_arr, test_arr]), \
             patch("sensor.components.model_trainer.load_object", return_value=preprocessor_mock), \
             patch("sensor.components.model_trainer.save_object"):
            trainer = ModelTrainer(mock_model_trainer_config, mock_data_transformation_artifact)
            with pytest.raises(Exception, match="not good"):
                trainer.initiate_model_trainer()
