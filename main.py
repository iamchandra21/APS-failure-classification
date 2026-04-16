from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from starlette.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import uvicorn
import pandas as pd
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import os
import datetime
import time
from starlette.middleware.base import BaseHTTPMiddleware

# Sensor-related imports
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.constants.training_pipeline import SAVED_MODEL_DIR
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object
from sensor.logger import logging

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = round(time.time() - start_time, 4)
        logging.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} [{duration}s]"
        )
        return response

app.add_middleware(RequestLoggingMiddleware)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            logging.info("training pipeline is already running.")
            return Response("Training pipeline is already running.")
        logging.info("Starting training pipeline.")
        train_pipeline.run_pipeline()
        logging.info("Training pipeline is successfully completed.")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


# Directory to save predictions
PREDICTIONS_DIR = "predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Load trained Model
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return JSONResponse(content={"error": "Model is not available"}, status_code=400)

        best_model_path = model_resolver.get_best_model_path()
        sensor_model = load_object(file_path=best_model_path)  # Load Model object

        # Extract preprocessor from SensorModel
        preprocessor = sensor_model.preprocessor
        expected_features = preprocessor.get_feature_names_out()

        # Validate & preprocess input data
        df = df[expected_features]
        df.replace("na", float("nan"), inplace=True)
        transformed_data = preprocessor.transform(df)

        # Make predictions
        y_pred = sensor_model.model.predict(transformed_data)
        y_proba = sensor_model.model.predict_proba(transformed_data)
        df["predicted_column"] = y_pred
        df["predicted_column"] = df["predicted_column"].replace(TargetValueMapping().reverse_mapping())
        df["confidence_score"] = y_proba[:, 1].round(4)

        # Save predictions with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.csv"
        file_path = os.path.join(PREDICTIONS_DIR, filename)
        df.to_csv(file_path, index=False)

        return JSONResponse(content={"file_path": file_path})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    # aws
