import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from sensor.entity.artifact_entity import ClassificationMetricArtifact
from sensor.exception import SensorException


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)

        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric
    except Exception as e:
        raise SensorException(str(e))
