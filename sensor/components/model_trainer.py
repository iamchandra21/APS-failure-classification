from sensor.utils.main_utils import load_numpy_array_data
from sensor.exception import SensorException
from sensor.logger import logging
import os, sys
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object, load_object
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server environments
import matplotlib.pyplot as plt
from xgboost import plot_importance


def _init_dagshub():
    """Initialise DagsHub MLflow tracking once per process."""
    dagshub.init(
        repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME", ""),
        repo_name="APS-failure-classification",
        mlflow=True,
    )


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(str(e))

    def perform_hyperparameter_tuning(self, x_train, y_train):
        try:
            from sklearn.model_selection import GridSearchCV

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
                "scale_pos_weight": [1, 10, 50],  # critical for imbalanced data
            }

            xgb = XGBClassifier(eval_metric="logloss")
            grid_search = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                scoring="f1",
                cv=3,
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(x_train, y_train)
            logging.info(f"Best hyperparameters: {grid_search.best_params_}")
            logging.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
            return grid_search.best_estimator_
        except Exception as e:
            raise SensorException(str(e))

    def train_model(self, x_train, y_train):
        try:
            model = self.perform_hyperparameter_tuning(x_train, y_train)
            return model
        except Exception as e:
            raise SensorException(str(e))

    def _log_feature_importance(self, model) -> str:
        """Save feature importance bar chart and return the file path."""
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_importance(model, ax=ax, max_num_features=20, importance_type="gain",
                        title="Top 20 Features by Gain")
        plt.tight_layout()
        fig_path = os.path.join(
            self.model_trainer_config.model_trainer_dir, "feature_importance.png"
        )
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # --- MLflow run --------------------------------------------------
            _init_dagshub()
            with mlflow.start_run(run_name="APS-XGBoost-Training") as run:
                logging.info(f"MLflow run started — run_id: {run.info.run_id}")

                model = self.train_model(x_train, y_train)

                # --- metrics -------------------------------------------------
                y_train_pred = model.predict(x_train)
                classification_train_metric = get_classification_score(
                    y_true=y_train, y_pred=y_train_pred
                )

                if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                    raise Exception("Trained model is not good to provide expected accuracy")

                y_test_pred = model.predict(x_test)
                classification_test_metric = get_classification_score(
                    y_true=y_test, y_pred=y_test_pred
                )

                # Overfitting / Underfitting check
                diff = abs(
                    classification_train_metric.f1_score - classification_test_metric.f1_score
                )
                if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                    raise Exception("Model is not good — try more experimentation")

                # --- log hyperparameters -------------------------------------
                mlflow.log_params(model.get_params())

                # --- log train metrics ---------------------------------------
                mlflow.log_metrics({
                    "train_f1":        classification_train_metric.f1_score,
                    "train_precision": classification_train_metric.precision_score,
                    "train_recall":    classification_train_metric.recall_score,
                })

                # --- log test metrics ----------------------------------------
                mlflow.log_metrics({
                    "test_f1":        classification_test_metric.f1_score,
                    "test_precision": classification_test_metric.precision_score,
                    "test_recall":    classification_test_metric.recall_score,
                    "train_test_f1_diff": diff,
                })

                logging.info(
                    f"Logged metrics — train_f1={classification_train_metric.f1_score:.4f}, "
                    f"test_f1={classification_test_metric.f1_score:.4f}"
                )

                # --- log feature importance chart ----------------------------
                fig_path = self._log_feature_importance(model)
                mlflow.log_artifact(fig_path, artifact_path="plots")
                logging.info("Logged feature importance chart to MLflow")

                # --- save model locally + log to MLflow registry -------------
                preprocessor = load_object(
                    file_path=self.data_transformation_artifact.transformed_object_file_path
                )
                model_dir_path = os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                )
                os.makedirs(model_dir_path, exist_ok=True)
                sensor_model = SensorModel(preprocessor=preprocessor, model=model)
                save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)

                mlflow.sklearn.log_model(
                    sensor_model,
                    artifact_path="sensor-model",
                    registered_model_name="APS-SensorModel",
                )
                logging.info("Registered SensorModel in MLflow Model Registry")

            # -----------------------------------------------------------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            logging.info(f"Model trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(str(e))
