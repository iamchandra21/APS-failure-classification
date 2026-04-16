"""
Tests for sensor/components/model_trainer.py
Uses tiny synthetic arrays so tests run in seconds with no I/O.
"""
import numpy as np

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
