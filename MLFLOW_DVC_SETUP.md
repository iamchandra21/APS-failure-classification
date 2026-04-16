# MLflow + DVC Setup Guide (DagsHub)

Run these steps **once** on your local machine after pulling the latest code.

---

## Step 1 — Install New Dependencies

```bash
uv pip install -r requirements.txt
```

---

## Step 2 — Create a DagsHub Account & Connect Your Repo

1. Go to [dagshub.com](https://dagshub.com) → sign up / log in with GitHub
2. Click **Create Repo → Connect a repo** → select `APS-failure-classification`
3. DagsHub will mirror your GitHub repo and give you a tracking URI

---

## Step 3 — Get Your DagsHub Token

1. Go to **https://dagshub.com/user/settings/tokens**
2. Click **Generate new token** → give it a name → copy it

---

## Step 4 — Add Variables to Your .env

Copy `.env.example` → `.env` and fill in the DagsHub section:

```bash
cp .env.example .env
```

Edit `.env`:

```
DAGSHUB_USER_TOKEN=your_token_here
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/APS-failure-classification.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=your_token_here
```

---

## Step 5 — Initialise DVC

```bash
# Initialise DVC in the repo (creates .dvc/ folder)
dvc init

# Add DagsHub as the DVC remote storage
dvc remote add origin https://dagshub.com/<your-username>/APS-failure-classification.dvc

# Store credentials locally (never committed to git)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-username>
dvc remote modify origin --local password your_token_here

# Set as default remote
dvc remote default origin

# Commit the DVC config
git add .dvc/config .dvcignore
git commit -m "chore: init dvc with dagshub remote"
```

---

## Step 6 — Track Data with DVC

After running the pipeline once (so artifacts/ folder is populated):

```bash
# Track the feature store CSV
dvc add artifact/data_ingestion/feature_store/sensor.csv

# Push data to DagsHub remote
dvc push

# Commit the .dvc pointer file
git add artifact/data_ingestion/feature_store/sensor.csv.dvc .gitignore
git commit -m "data: track sensor.csv with dvc"
git push
```

---

## Step 7 — Add GitHub Secrets

Go to **GitHub → Settings → Secrets and variables → Actions** and add:

| Secret Name | Value |
|---|---|
| `DAGSHUB_USER_TOKEN` | your DagsHub token |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<username>/APS-failure-classification.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | your DagsHub username |

---

## Step 8 — Run the Pipeline & View Results

```bash
# Run full pipeline via main.py (MLflow will log automatically)
python main.py
```

Then visit your MLflow UI:

```
https://dagshub.com/<your-username>/APS-failure-classification/experiments
```

You will see:
- All hyperparameter values from GridSearchCV
- Train / test F1, precision, recall
- Feature importance chart (top 20 features by gain)
- Registered model in the Model Registry tab

---

## DVC Daily Workflow

```bash
# Run a single pipeline stage (reruns only if deps changed)
dvc repro model_trainer

# Run full pipeline from scratch
dvc repro

# Check what has changed
dvc status

# Push updated artifacts to DagsHub
dvc push

# Pull artifacts on a new machine
dvc pull
```

---

## Comparing Experiment Runs

```bash
# Show metrics across all runs
dvc metrics show

# Compare two runs side by side
dvc metrics diff HEAD~1 HEAD
```

Or use the MLflow UI at DagsHub for visual comparison with charts.

---

## Folder Structure After Setup

```
.dvc/
  config          ← DVC remote config (committed)
  cache/          ← local cache (git-ignored)
dvc.yaml          ← pipeline stage definitions
params.yaml       ← all tunable parameters
metrics.json      ← latest run metrics (tracked by git, not DVC)
artifact/
  data_ingestion/
    feature_store/
      sensor.csv      ← actual file git-ignored
      sensor.csv.dvc  ← pointer file committed to git
```
