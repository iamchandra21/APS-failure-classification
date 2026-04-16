# APS Failure Classification

**Predicting Air Pressure System failures in heavy-duty trucks before they happen — saving $50 per missed fault.**

[![CI/CD](https://github.com/iamchandra21/APS-failure-classification/actions/workflows/main.yml/badge.svg)](https://github.com/iamchandra21/APS-failure-classification/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.50-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECR%20%7C%20S3-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

## Description

This project builds an end-to-end machine learning pipeline to classify Air Pressure System (APS) failures in Scania heavy-duty trucks. The model identifies whether a fault originates from the APS or another unrelated component, enabling maintenance teams to avoid $50 in unnecessary repair costs for every false negative (missed APS fault).

The system ingests raw sensor data from MongoDB, validates schema integrity, handles class imbalance with SMOTETomek resampling, trains an XGBoost classifier with GridSearchCV hyperparameter tuning, tracks every experiment with MLflow on DagsHub, versions data with DVC, evaluates performance against the current production model, and deploys via a Dockerized FastAPI service on AWS EC2 — all orchestrated through a single training pipeline with GitHub Actions CI/CD.

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
| **ML / Data** | XGBoost 2.1, scikit-learn, imbalanced-learn (SMOTETomek), pandas, NumPy |
| **Experiment Tracking** | MLflow 2.16 + DagsHub |
| **Data Versioning** | DVC 3.50 + DagsHub remote |
| **API** | FastAPI, Uvicorn, Starlette |
| **UI** | Streamlit |
| **Database** | MongoDB Atlas (pymongo) |
| **Cloud Storage** | AWS S3 |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions → AWS ECR → EC2 self-hosted runner |
| **Package Manager** | uv + pyproject.toml |
| **Config** | YAML, python-dotenv |
| **Serialization** | dill, NumPy `.npy` |

---

## Features

- **Automated 6-stage ML pipeline** — Ingestion → Validation → Transformation → Training → Evaluation → Deployment, all triggered via a single API call
- **MLflow experiment tracking** — Logs GridSearchCV params, train/test F1, precision, recall, feature importance chart, and registers the model in the MLflow Model Registry on DagsHub
- **DVC data & pipeline versioning** — Tracks raw CSV data, defines reproducible pipeline stages in `dvc.yaml`, and pushes artifacts to DagsHub remote
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
│               versioning)        >= 0.02)        + GridSearchCV)   │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐     ┌──────────────────────────────┐
│       EXPERIMENT TRACKING    │     │       CI/CD PIPELINE         │
│                              │     │                              │
│  MLflow → DagsHub            │     │  push → GitHub Actions       │
│    params, metrics           │     │    → lint + pytest           │
│    feature importance        │     │    → Docker build            │
│    model registry            │     │    → push to ECR             │
│                              │     │    → EC2 self-hosted runner  │
│  DVC → DagsHub remote        │     │    → docker run              │
│    data versioning           │     │                              │
│    pipeline reproducibility  │     │                              │
└──────────────────────────────┘     └──────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Docker (optional, for containerized run)
- MongoDB Atlas account with sensor data loaded
- AWS account with S3, ECR, EC2 access (for cloud deployment)
- DagsHub account (for MLflow tracking + DVC remote)

### 1. Clone the Repository

```bash
git clone https://github.com/iamchandra21/APS-failure-classification.git
cd APS-failure-classification
```

### 2. Install Dependencies

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install all dependencies
uv venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

uv pip install -r requirements.txt
uv pip install -e .
```

### 3. Configure Environment Variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# MongoDB
MONGO_DB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/

# AWS
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# DagsHub (MLflow + DVC)
DAGSHUB_USER_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/APS-failure-classification.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

---

## Execution Steps

### Run the Full Pipeline (Local)

```bash
# Starts FastAPI server on http://0.0.0.0:8080
python main.py

# Trigger the full 6-stage training pipeline
curl http://localhost:8080/train

# Run batch prediction (CSV file)
curl -X POST http://localhost:8080/predict \
  -F "file=@path/to/input.csv"

# Interactive API docs
open http://localhost:8080/docs
```

### Run the Streamlit UI

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Run with Docker

```bash
# Build image
docker build -t aps-failure-classification .

# Run container
docker run -p 8080:8080 \
  -e MONGO_DB_URL=$MONGO_DB_URL \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION=$AWS_REGION \
  -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
  -e MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME \
  -e MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD \
  aps-failure-classification
```

### Run Tests

```bash
# Run all unit tests with coverage
pytest tests/ -v --cov=sensor --cov-report=term-missing

# Lint check
flake8 sensor/ --count --max-line-length=120 --statistics
```

---

## MLflow Setup (DagsHub)

MLflow tracks every training run — hyperparameters, metrics, feature importance charts, and the registered model — all visible in your DagsHub UI.

### Step 1 — Create DagsHub Account & Connect Repo

1. Go to [dagshub.com](https://dagshub.com) → sign up / log in with GitHub
2. Click **Create Repo → Connect a repo** → select `APS-failure-classification`
3. Your MLflow tracking URI will be: `https://dagshub.com/<username>/APS-failure-classification.mlflow`

### Step 2 — Get DagsHub Token

1. Go to **https://dagshub.com/user/settings/tokens**
2. Click **Generate new token** → copy it
3. Add it to your `.env` as `DAGSHUB_USER_TOKEN` and `MLFLOW_TRACKING_PASSWORD`

### Step 3 — Run Pipeline & View Results

```bash
python main.py
# Then trigger training: curl http://localhost:8080/train
```

View your experiments at:
```
https://dagshub.com/<your-username>/APS-failure-classification/experiments
```

### What Gets Logged Per Run

| Logged Item | Details |
|---|---|
| **Parameters** | All XGBoost hyperparameters from GridSearchCV best estimator |
| **Train metrics** | F1 score, precision, recall on training set |
| **Test metrics** | F1 score, precision, recall on test set, train/test F1 diff |
| **Feature importance** | Bar chart of top 20 features by gain (saved as `plots/feature_importance.png`) |
| **Model artifact** | `SensorModel` (preprocessor + XGBoost) registered as `APS-SensorModel` in Model Registry |

### Add GitHub Secret

Add `DAGSHUB_USER_TOKEN`, `MLFLOW_TRACKING_URI`, and `MLFLOW_TRACKING_USERNAME` to **GitHub → Settings → Secrets and variables → Actions** so CI/CD runs also log to DagsHub.

---

## DVC Setup (Data & Pipeline Versioning)

DVC versions your raw data and makes every pipeline stage reproducible with `dvc repro`.

### Step 1 — Initialise DVC

```bash
dvc init
```

### Step 2 — Add DagsHub as DVC Remote

```bash
dvc remote add origin https://dagshub.com/<your-username>/APS-failure-classification.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-username>
dvc remote modify origin --local password <your-dagshub-token>
dvc remote default origin
```

### Step 3 — Track Raw Data

After running the pipeline once (so `artifact/` is populated):

```bash
dvc add artifact/<timestamp>/data_ingestion/feature_store/sensor.csv
git add artifact/<timestamp>/data_ingestion/feature_store/sensor.csv.dvc .gitignore
git commit -m "data: track sensor.csv with dvc"
dvc push
git push
```

### Step 4 — Reproduce the Pipeline

```bash
# Rerun only the stages whose dependencies changed
dvc repro

# Force rerun all stages
dvc repro --force

# Check what has changed
dvc status

# Sync data to/from DagsHub
dvc push
dvc pull
```

### Step 5 — Compare Runs

```bash
# Show current metrics
dvc metrics show

# Diff metrics between last two commits
dvc metrics diff HEAD~1 HEAD
```

Or compare visually in the MLflow UI on DagsHub.

### DVC Pipeline Stages (`dvc.yaml`)

| Stage | Input | Output |
|---|---|---|
| `data_ingestion` | MongoDB collection | `artifact/data_ingestion/` |
| `data_validation` | Ingested CSVs | `artifact/data_validation/` |
| `data_transformation` | Validated CSVs | `artifact/data_transformation/` |
| `model_trainer` | Transformed numpy arrays | `artifact/model_trainer/` + MLflow run |

All tunable parameters live in `params.yaml`. Change a value there and `dvc repro` reruns only the affected downstream stages.

---

## AWS Deployment

Full step-by-step instructions are in [`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md). Summary below.

### Infrastructure Required

| Service | Purpose |
|---|---|
| **S3** | Store pipeline artifacts and trained models |
| **ECR** | Docker image registry |
| **EC2** | Host the FastAPI container (t2.medium minimum) |
| **IAM** | CI/CD user with S3 + ECR + EC2 permissions |

### Quick Setup

```bash
# 1. Create S3 bucket
aws s3 mb s3://aps-failure-classification --region us-east-1

# 2. Create ECR repository
aws ecr create-repository \
  --repository-name aps-failure-classification \
  --region us-east-1

# 3. Launch EC2 (Ubuntu 22.04, t2.medium, ports 22/80/8080 open)
# 4. Install Docker on EC2
# 5. Set up GitHub Actions self-hosted runner on EC2
# 6. Add all secrets to GitHub (see below)
```

### Required GitHub Secrets

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | `us-east-1` |
| `ECR_REPOSITORY_NAME` | `aps-failure-classification` |
| `AWS_ECR_LOGIN_URI` | `<account-id>.dkr.ecr.us-east-1.amazonaws.com` |
| `MONGO_DB_URL` | MongoDB Atlas connection string |
| `DAGSHUB_USER_TOKEN` | DagsHub personal access token |
| `MLFLOW_TRACKING_URI` | DagsHub MLflow tracking URI |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |

### Deploy

Push to `main` branch — GitHub Actions handles the rest:

```bash
git push origin main
```

**Actions tab shows 3 jobs running in sequence:**
1. `Continuous Integration` — flake8 lint + pytest
2. `Continuous Delivery` — Docker build + push to ECR
3. `Continuous Deployment` — EC2 pulls image + runs container

Your app goes live at `http://<EC2_PUBLIC_IP>:8080`.

---

## Folder Structure

```
APS-failure-classification/
├── .github/
│   └── workflows/
│       └── main.yml            # GitHub Actions CI/CD
├── config/
│   └── schema.yaml             # Data schema: 171 columns, drop list
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_data_validation.py
│   ├── test_model_trainer.py
│   ├── test_estimator.py
│   └── test_classification_metric.py
├── sensor/                     # Main application package
│   ├── cloud_storage/          # AWS S3 sync utilities
│   ├── components/             # Core pipeline stages (6 components)
│   ├── configuration/          # MongoDB connection setup
│   ├── constants/              # All constants and path definitions
│   ├── data_access/            # MongoDB → DataFrame export layer
│   ├── entity/                 # Config & artifact dataclasses
│   ├── ml/
│   │   ├── metric/             # F1, precision, recall computation
│   │   └── model/              # SensorModel wrapper, ModelResolver
│   ├── pipeline/
│   │   └── training_pipeline.py
│   ├── utils/                  # YAML, numpy, dill helpers
│   ├── exception.py
│   └── logger.py
├── dvc.yaml                    # DVC pipeline stage definitions
├── params.yaml                 # All tunable pipeline parameters
├── main.py                     # FastAPI entrypoint
├── streamlit_app.py            # Streamlit UI
├── Dockerfile
├── pyproject.toml              # Project metadata + pinned deps (uv)
├── requirements.txt            # Flat dependency list
├── .env.example                # Environment variable template
├── .flake8                     # Linting configuration
├── AWS_DEPLOYMENT.md           # Step-by-step AWS deployment guide
├── MLFLOW_DVC_SETUP.md         # MLflow + DVC one-time setup guide
└── runtime.txt                 # Python version (3.11.13)
```

---

## API Reference

### `GET /train`

Triggers the full 6-stage training pipeline.

**Response:**
```json
{ "status": "Training successful" }
```

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

**Label mapping:** `0` → `neg` (no APS failure) · `1` → `pos` (APS failure detected)

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
| Imputation strategy | Constant fill (0) via SimpleImputer |
| Resampling | SMOTETomek |

**Cost Matrix:**
- False Positive (unnecessary repair): **Cost₁**
- False Negative (missed APS fault): **Cost₂ = 50 × Cost₁**

---

## CI/CD Pipeline

```
Developer Push to main
        │
        ▼
GitHub Actions
        ├── 1. Continuous Integration (ubuntu-latest)
        │       ├── flake8 lint (blocking: syntax errors only)
        │       ├── flake8 style (non-blocking)
        │       └── pytest --cov (coverage report uploaded as artifact)
        │
        ├── 2. Continuous Delivery (ubuntu-latest)
        │       ├── Docker build
        │       └── Push image to AWS ECR
        │
        └── 3. Continuous Deployment (self-hosted EC2 runner)
                ├── Pull latest image from ECR
                ├── Stop + remove old container
                └── docker run -p 80:8080 (with all env vars injected)
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Author

**Tiyyagura Chandra Reddy**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github)](https://github.com/iamchandra21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/iamchandra21)
[![Email](https://img.shields.io/badge/Email-chandra.tiyyagura%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:chandra.tiyyagura@gmail.com)
