# APS Failure Classification

**Predicting Air Pressure System failures in heavy-duty trucks before they happen — saving $50 per missed fault.**

[![CI/CD](https://github.com/iamchandra21/APS-failure-classification/actions/workflows/main.yml/badge.svg)](https://github.com/YOUR_USERNAME/APS-failure-classification/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECR%20%7C%20S3-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

## Description

This project builds an end-to-end machine learning pipeline to classify Air Pressure System (APS) failures in Scania heavy-duty trucks. The model identifies whether a fault originates from the APS or another unrelated component, enabling maintenance teams to avoid $50 in unnecessary repair costs for every false negative (missed APS fault).

The system ingests raw sensor data from MongoDB, validates schema integrity, handles class imbalance with SMOTETomek resampling, trains an XGBoost classifier, evaluates performance against the current production model, and deploys via a Dockerized FastAPI service on AWS EC2 — all orchestrated through a single training pipeline with GitHub Actions CI/CD.

---

## Demo

> 📸 _Screenshots of the deployed application are in the [`Screenshots/`](Screenshots/) folder._

| Pipeline Flow | Deployment Architecture |
|---|---|
| ![Pipeline](flowcharts/APS%20Failure%20Classification%20High%20Level%20Code%20Flow.png) | ![Deployment](flowcharts/Deployment%20Architecture.png) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML / Data** | XGBoost, scikit-learn, imbalanced-learn (SMOTETomek), pandas, NumPy |
| **API** | FastAPI, Uvicorn, Starlette |
| **UI** | Streamlit |
| **Database** | MongoDB Atlas (pymongo) |
| **Cloud Storage** | AWS S3 |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions → AWS ECR → EC2 self-hosted runner |
| **Config** | YAML, python-dotenv |
| **Serialization** | dill, NumPy `.npy` |

---

## Features

- **Automated 6-stage ML pipeline** — Ingestion → Validation → Transformation → Training → Evaluation → Deployment, all triggered via a single API call
- **Cost-aware evaluation** — False negatives carry 50× the cost of false positives; model acceptance uses F1-score with improvement thresholds
- **Dataset drift detection** — Kolmogorov-Smirnov tests across all 171 features generate per-feature drift reports before training
- **Imbalanced dataset handling** — SMOTETomek resampling on a 98.3% negative class dataset (59,000 negative / 1,000 positive)
- **Model versioning** — Each training run saves timestamped artifacts; `ModelResolver` automatically selects the best available model for inference
- **Cloud-backed artifacts** — All pipeline artifacts and trained models are synced to AWS S3 after each run
- **REST API for training and prediction** — `GET /train` triggers the full pipeline; `POST /predict` accepts CSV input and returns per-row predictions
- **Streamlit UI** — Alternative frontend supporting CSV file upload or direct MongoDB data fetch for batch prediction
- **Dockerized deployment** — Reproducible container with AWS CLI; deployed to EC2 via GitHub Actions self-hosted runner

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                           │
│                                                                     │
│  MongoDB ──► DataIngestion ──► DataValidation ──► DataTransform    │
│                (CSV export)      (KS drift,         (Imputer +      │
│                (80/20 split)      schema check)      RobustScaler + │
│                                                       SMOTETomek)   │
│                                                            │         │
│  S3 Sync ◄── ModelPusher ◄── ModelEvaluation ◄── ModelTrainer     │
│  (artifacts)   (timestamp       (F1 delta          (XGBClassifier  │
│               versioning)        >= 0.02)           F1 >= 0.6)     │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐     ┌──────────────────────────────┐
│       INFERENCE API          │     │       CI/CD PIPELINE         │
│                              │     │                              │
│  POST /predict               │     │  push → GitHub Actions       │
│    CSV input                 │     │    → Docker build            │
│    → SensorModel.predict()   │     │    → push to ECR             │
│    → JSON output             │     │    → EC2 self-hosted runner  │
│                              │     │    → docker run              │
└──────────────────────────────┘     └──────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized run)
- MongoDB Atlas account (or local MongoDB)
- AWS account with S3, ECR, EC2 access (for cloud deployment)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/iamchandra21/APS-failure-classification.git
cd APS-failure-classification

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the sensor package in editable mode
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```env
MONGO_DB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

### Run Locally

```bash
# Start the FastAPI server
python main.py
# → Runs on http://0.0.0.0:8080
# → API docs at http://localhost:8080/docs

# Trigger training via API
curl http://localhost:8080/train

# Predict via API (CSV file)
curl -X POST http://localhost:8080/predict \
  -F "file=@path/to/input.csv"

# Or run the Streamlit UI
streamlit run streamlit_app.py
```

### Run with Docker

```bash
# Build the image
docker build -t aps-failure-classification .

# Run the container
docker run -p 8080:8080 \
  -e MONGO_DB_URL=$MONGO_DB_URL \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION=$AWS_REGION \
  aps-failure-classification
```

---

## Folder Structure

```
APS-failure-classification/
├── .github/
│   └── workflows/
│       └── main.yml            # GitHub Actions CI/CD (lint → build → ECR → EC2)
├── config/
│   └── schema.yaml             # Data schema: 171 columns, drop list, numerical cols
├── flowcharts/                 # Architecture and pipeline diagrams
├── notebooks/
│   └── Scania_APS_failure_prediction.ipynb  # EDA and baseline experiments
├── Screenshots/                # Deployment documentation screenshots
├── sensor/                     # Main application package
│   ├── cloud_storage/          # AWS S3 sync utilities
│   ├── components/             # Core pipeline stages (6 components)
│   ├── configuration/          # MongoDB connection setup
│   ├── constants/              # All magic numbers and path constants
│   ├── data_access/            # MongoDB → DataFrame export layer
│   ├── entity/                 # Config & artifact dataclasses
│   ├── ml/
│   │   ├── metric/             # F1, precision, recall computation
│   │   └── model/              # SensorModel wrapper, ModelResolver
│   ├── pipeline/
│   │   └── training_pipeline.py  # Orchestrates all 6 pipeline stages
│   ├── utils/                  # YAML, numpy, dill serialization helpers
│   ├── exception.py            # Custom SensorException with traceback detail
│   └── logger.py               # Timestamped log files in logs/
├── main.py                     # FastAPI entrypoint
├── streamlit_app.py            # Streamlit UI frontend
├── Dockerfile                  # Production container definition
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── runtime.txt                 # Deployment Python version (3.11.13)
```

---

## API Reference

### `GET /train`

Triggers the full 6-stage training pipeline.

**Response:**
```json
{
  "status": "Training successful"
}
```

---

### `POST /predict`

Accepts a CSV file and returns binary predictions for each row.

**Request:** `multipart/form-data` with field `file` (`.csv`)

**Response:**
```json
{
  "status": "success",
  "predictions": [0, 1, 0, 0, 1]
}
```

**Label mapping:**
- `0` → `neg` — No APS failure detected
- `1` → `pos` — APS failure detected

---

## Dataset

The project uses the **Scania APS Failure Dataset** (UCI ML Repository).

| Property | Value |
|---|---|
| Training examples | 60,000 |
| Test examples | 16,000 |
| Features | 171 anonymized sensor attributes |
| Positive class (APS failure) | ~1.7% |
| Missing values | Encoded as `"na"` string |
| Imputation strategy | Constant fill (0) |
| Resampling | SMOTETomek |

**Cost Matrix:**
- False Positive (unnecessary repair): **Cost₁**
- False Negative (missed APS fault): **Cost₂ = 50 × Cost₁**

---

## CI/CD Pipeline

```
Developer Push
     │
     ▼
GitHub Actions
     ├── Lint (flake8)
     ├── Unit Tests (pytest)
     ├── Docker Build
     ├── Push to AWS ECR
     └── Deploy to EC2 (self-hosted runner)
              └── docker pull + docker run -p 80:8080
```

**Required GitHub Secrets:**
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPOSITORY_NAME
AWS_ECR_LOGIN_URI
MONGO_DB_URL
```

---
---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Author

**Tiyyagura Chandra Reddy**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github)](https://github.com/iamchandra21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/iamchandra21)
[![Email](https://img.shields.io/badge/Email-chandra.tiyyagura%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:chandra.tiyyagura@gmail.com)

---

