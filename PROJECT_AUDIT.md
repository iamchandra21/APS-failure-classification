# APS Failure Classification — Project Audit Report

> Generated for interview preparation. Covers all files, code, configs, and documentation.

---

## 🔴 CRITICAL — Must Fix Before Showcasing

---

### 1. `sensor/components/model_trainer.py` — Stub method never implemented

```python
# Line 21
def perform_hyperparameter_tuning(self):...
```

The `perform_hyperparameter_tuning` method is defined as a stub and is **never called**. The XGBoost model trains with default hyperparameters, which is suboptimal for a cost-sensitive classification problem. This is the most glaring gap in the ML pipeline.

**Fix:** Implement using `GridSearchCV` or `Optuna` with parameters like `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` (critical for imbalanced data).

---

### 2. `sensor/components/data_validation.py` — `drop_zero_std_columns` not implemented

```python
# Line 113
#TODO: drop_zero_std_columns
def drop_zero_std_columns(self, dataframe):
    pass
```

A column with zero standard deviation (constant value) will cause `RobustScaler` to produce NaN or zero-division. This is a real risk given the dataset has 171 features. The function is defined but never filled in and never called.

**Fix:** Implement to detect and drop constant columns before scaling:
```python
def drop_zero_std_columns(self, dataframe):
    zero_std_cols = [col for col in dataframe.columns if dataframe[col].std() == 0]
    return dataframe.drop(columns=zero_std_cols)
```

---

### 3. `sensor/constants/prediction_pipeline/__init__.py` — Empty file

The prediction pipeline constants file is completely empty, meaning the prediction pathway has no dedicated configuration. This makes prediction behaviour opaque and tightly coupled to the training pipeline constants.

**Fix:** Populate with prediction-specific constants: batch size, input schema path, output path, model path.

---

### 4. Zero Unit Tests Across the Entire Project

There are no test files anywhere — no `tests/` folder, no `test_*.py` files, no `conftest.py`. The GitHub Actions CI/CD pipeline references a test step but it is a `echo "testing"` placeholder:

```yaml
# .github/workflows/main.yml — Integration stage
- name: Run Unit Tests
  run: |
    echo "testing"   # ← NOT a real test
```

**Fix:** Add `tests/` directory with at minimum:
- `test_data_ingestion.py` — validate output CSV shapes
- `test_data_validation.py` — test column count checks
- `test_data_transformation.py` — test imputer/scaler output
- `test_model_trainer.py` — test model loads and predicts

---

### 5. `sensor/components/model_evaluation.py` — Evaluates on full dataset, not test set

The model evaluation component loads the full dataset and evaluates both old and new models on it. This contaminates the evaluation with training data and inflates reported metrics. A fair comparison should be done on the held-out test set only.

**Fix:** Load only the test artifact path from `DataValidationArtifact` and evaluate exclusively on it.

---

### 6. `streamlit_app.py` — Spawns FastAPI as a subprocess (fragile)

```python
# streamlit_app.py
subprocess.Popen(["uvicorn", "main:app", ...])
```

This creates a tight coupling and will break in environments where the port is already in use, or where subprocess spawning is restricted (e.g., Docker, cloud platforms).

**Fix:** Decouple the Streamlit UI from the FastAPI backend. Use an environment variable for the API base URL and call it via HTTP requests.

---

## 🟡 SHOULD FIX — Important for Code Quality

---

### 7. Multiple Typos in File and Variable Names

| File / Symbol | Typo | Correct |
|---|---|---|
| `sensor/configuration/monogo_db_connection.py` | `monogo` | `mongo` |
| `sensor/ml/metric/classification_merric.py` | `merric` | `metric` |
| `sensor/constants/env_varibale.py` | `varibale` | `variable` |
| `SensorModel.trained_model_fie_path` (config_entity.py) | `fie` | `file` |
| `self.clinet` in `MongoDBClient` | `clinet` | `client` |

These are visible to anyone reading your code during a code review interview.

---

### 8. `sensor/cloud_storage/s3_syncer.py` — Uses `os.system()` with no error checking

```python
os.system(f"aws s3 sync {folder} s3://{aws_bucket_url}/")
```

`os.system()` returns an exit code but it is ignored. A failed S3 sync (network timeout, permission error, wrong bucket) would silently continue the pipeline without error.

**Fix:** Use `subprocess.run(..., check=True)` to raise on non-zero exit, or use `boto3` directly for proper error handling and retry.

---

### 9. No Logging for S3 Sync Operations

The `S3Sync` class in `s3_syncer.py` has no logger calls. Given this is a critical step (pushing model artifacts to the cloud), failures or successes should be visible in the log file.

---

### 10. `data_transformation.py` — `SimpleImputer(strategy="constant", fill_value=0)` may not be the best choice

Imputing all missing values with 0 can mislead the model when 0 is a valid sensor reading. A `strategy="median"` or `strategy="mean"` would be more statistically sound for sensor data.

---

### 11. `model_trainer.py` — No cross-validation

The model is evaluated on a single train/test split with no k-fold cross-validation. For a dataset of 60,000 rows, stratified k-fold would give a more reliable performance estimate — especially important given the class imbalance (98.3% negative class).

---

### 12. `main.py` — No request/response logging middleware

The FastAPI app has no access log or request tracking. For a production system, middleware to log request details (timestamp, endpoint, response time) should be included.

---

### 13. `requirements.txt` — Version pinning inconsistent

Some packages have pinned versions (`pandas==1.5.3`), others do not (`scikit-learn`, `xgboost`). This can cause reproducibility issues.

---

### 14. `config/schema.yaml` — No description or inline comments

The schema file lists 171 column names with no documentation on what each sensor attribute represents. Adding even brief comments on major groups of features would improve maintainability.

---

## 🟢 NICE TO HAVE — Improvements for Impressiveness

---

### 15. Add SHAP / Feature Importance Analysis

XGBoost has built-in feature importance. Adding a SHAP analysis in the notebook or as a pipeline step would make this significantly more impressive for ML interviews and demonstrate awareness of model explainability.

---

### 16. Add Custom Cost-Aware Evaluation Metric

The dataset explicitly states that false negatives cost 50x more than false positives. The model currently optimizes F1-score. A custom cost metric (`Cost_1 * FP + Cost_2 * FN`) would better align the model objective with the business problem.

---

### 17. Add Prediction Confidence Scores

The `/predict` endpoint returns binary labels. Including `predict_proba` scores in the output would allow downstream systems to apply custom thresholds based on cost tolerance.

---

### 18. Add a `Makefile` or `scripts/` for common commands

Common commands like `make train`, `make predict`, `make test`, `make docker-build` would make the project more developer-friendly and are appreciated in professional settings.

---

### 19. Add `.env.example` file

There is a `.gitignore` that excludes `.env`, but no `.env.example` to show new contributors what environment variables are required.

---

### 20. Add Precision-Recall Curve Visualization

Given the heavy class imbalance, ROC-AUC is misleading. A PR curve plot saved as an artifact would demonstrate ML awareness and provide better model insight.

---

### 21. `notebooks/` — The EDA notebook could be more comprehensive

The existing notebook (`Scania_APS_failure_prediction.ipynb`) is a good start. Adding sections for:
- Class imbalance visualization
- Missing value heatmap
- Feature correlation analysis
- SHAP summary plot post-training

would make it much more portfolio-worthy.

---

### 22. `Dockerfile` — Pin AWS CLI version

```dockerfile
RUN apt-get install awscli   # unpinned
```

Pinning the AWS CLI version ensures reproducible builds.

---

## Summary Count

| Priority | Count |
|---|---|
| 🔴 Critical | 6 |
| 🟡 Should Fix | 8 |
| 🟢 Nice to Have | 8 |
| **Total** | **22** |
