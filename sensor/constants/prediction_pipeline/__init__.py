import os
from sensor.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

PREDICTION_PIPELINE_DIR_NAME: str = "predictions"
PREDICTION_INPUT_FILE_NAME: str = "input.csv"
PREDICTION_OUTPUT_FILE_NAME: str = "predictions.csv"

# Reuse model path constants from training pipeline
PREDICTION_MODEL_DIR: str = SAVED_MODEL_DIR
PREDICTION_MODEL_FILE_NAME: str = MODEL_FILE_NAME
