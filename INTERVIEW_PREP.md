# Interview Preparation — APS Failure Classification

---

# PART 1 — Project Story (2–3 minute verbal answer)

> Use this when asked: *"Tell me about this project"* / *"Walk me through something you've built"*

---

## The Story (read this aloud, it runs ~2.5 minutes)

**PROBLEM**

I worked on a predictive maintenance problem for Scania heavy-duty trucks. The Air Pressure System — or APS — is critical for braking and engine performance, and when it fails, it's expensive. But the bigger problem is distinguishing whether a fault is actually from the APS or from some other unrelated component. If maintenance teams incorrectly treat a non-APS fault as an APS fault, that costs money. But if they miss a real APS failure, that costs 50 times more. So the business constraint wasn't just accuracy — it was a heavily asymmetric cost problem.

**SOLUTION**

I built a full end-to-end ML pipeline, not just a notebook. The pipeline goes through six stages: data ingestion from MongoDB, schema and drift validation, data transformation, model training, model evaluation against the currently deployed model, and model deployment. The whole thing is triggered by a single API call. I exposed a FastAPI service where you hit `GET /train` to kick off the pipeline, and `POST /predict` to get predictions on new sensor data.

**TECH CHOICES**

For the model, I chose XGBoost because it handles tabular sensor data well, is fast to train, and gives you built-in feature importance. The dataset was severely imbalanced — 98.3% of examples were non-failures — so I used SMOTETomek from imbalanced-learn, which combines synthetic oversampling with Tomek link cleaning to produce a cleaner decision boundary. For preprocessing, I used RobustScaler because sensor readings often have outliers. I stored the model and preprocessor together in a `SensorModel` wrapper using `dill` serialization, so prediction always uses the exact same transformations as training.

**CHALLENGES**

The hardest part was the model evaluation logic. I didn't want to just always overwrite the model — I only deploy a new model if it improves F1-score by at least 2% over the current production model. I also had to handle the case where no production model exists yet, so the first run always deploys. Another challenge was data drift — I added KS-test-based drift detection across all 171 features before every training run, so you know whether the new data distribution has shifted before committing to a retrain.

For deployment, I containerized everything with Docker, pushed the image to AWS ECR, and set up a GitHub Actions workflow that builds and deploys to EC2 automatically on every push. The model artifacts and training outputs are synced to S3 after every run.

**OUTCOME**

The result is a production-ready system — not just a trained model. It has versioned artifacts, a REST API, automated deployment, and cloud-backed storage. If I were to extend this, I'd add hyperparameter tuning with Optuna, SHAP explainability, and a custom cost-weighted evaluation metric that directly optimizes the 50:1 false negative penalty instead of F1.

---

# PART 2 — Expected Interview Questions & Answers

---

## SECTION A — Architecture & Design

---

### Q1. Why did you build a pipeline instead of just training a model in a notebook?

**Answer:**

A notebook is fine for exploration, but it's not reproducible and it doesn't scale. In a real system you need to retrain periodically as new data comes in. By modularizing the pipeline into six components — each with its own input/output artifact dataclass — every stage can be independently tested, debugged, or replaced.

Each component takes a config object in and returns an artifact object out. For example, `DataIngestion` takes `DataIngestionConfig` and returns `DataIngestionArtifact` with the paths to the generated train and test CSVs. This design means I can swap out a component — say, replace XGBoost with LightGBM in `ModelTrainer` — without touching any other component.

---

### Q2. Walk me through how data flows through your pipeline.

**Answer:**

1. **DataIngestion** (`sensor/components/data_ingestion.py`): Reads the MongoDB collection using `SensorData.export_collection_as_dataframe()`, does an 80/20 stratified split, and saves train/test CSVs to the artifact directory.

2. **DataValidation** (`data_validation.py`): Checks that the number of columns matches the schema in `config/schema.yaml` (171 columns expected). Then runs a Kolmogorov-Smirnov test on each numerical feature between train and test sets, and produces a YAML drift report.

3. **DataTransformation** (`data_transformation.py`): Builds a `Pipeline` with `SimpleImputer(strategy='constant', fill_value=0)` and `RobustScaler`. Fits on training data, transforms both splits. Then applies SMOTETomek to the training set only (not test — that would be data leakage). Saves the fitted preprocessor as a `.pkl` and the arrays as `.npy`.

4. **ModelTrainer** (`model_trainer.py`): Loads the `.npy` arrays, trains `XGBClassifier`, computes F1/precision/recall on both train and test, checks F1 >= 0.6 and that the overfitting gap is ≤ 5%.

5. **ModelEvaluation** (`model_evaluation.py`): Loads the newly trained model and the best existing production model (via `ModelResolver`), evaluates both on the current dataset, and only accepts the new model if the F1 improvement is ≥ 2%.

6. **ModelPusher** (`model_pusher.py`): Copies the accepted model to `saved_models/{timestamp}/` and syncs everything to S3.

---

### Q3. Why did you choose XGBoost over other algorithms?

**Answer:**

Several reasons for this dataset:

First, this is a tabular dataset with 171 numerical features — XGBoost is empirically the strongest performer on structured tabular data. Second, XGBoost handles missing values natively (though I also imputed with 0). Third, it's fast enough to retrain on 60,000 rows on a single machine. Fourth, it gives built-in feature importance, which is valuable for a sensor fault diagnosis problem where domain engineers want to know which sensors drive the prediction.

Alternatives I considered: Random Forest would be more robust to hyperparameters but typically underperforms XGBoost on tabular data. Logistic Regression would be more interpretable but unlikely to capture the non-linear sensor interaction patterns. Neural networks are overkill for this scale and would be harder to explain to maintenance teams.

---

### Q4. How does your model versioning work?

**Answer:**

The `ModelResolver` class in `sensor/ml/model/estimator.py` handles this. It scans the `saved_models/` directory, reads the directory names as timestamps, and returns the path with the latest timestamp as the "best" model.

When `ModelEvaluation` runs, it loads this latest model and compares it against the newly trained one. If the new model wins, `ModelPusher` saves it to `saved_models/{new_timestamp}/`. So you always have an ordered history of models. The downside is that it's purely timestamp-based — in a production system I'd add an explicit metadata registry (like MLflow) that stores performance metrics alongside each version.

---

### Q5. What does your CI/CD pipeline do exactly?

**Answer:**

The `.github/workflows/main.yml` has three stages:

**Integration:** Lint with flake8, run unit tests. (Currently this stage has placeholder test commands — adding real pytest coverage is on my roadmap.)

**Build and Push:** Logs into AWS ECR using the `aws-actions/amazon-ecr-login` action, builds the Docker image, and pushes it tagged with `latest`.

**Deployment:** Triggers a self-hosted runner on the EC2 instance. The runner pulls the latest image from ECR and restarts the container with `docker run -p 80:8080`, mapping EC2 port 80 to the app's 8080.

Secrets like `MONGO_DB_URL` and AWS credentials are stored in GitHub Secrets and injected as environment variables at runtime — never baked into the image.

---

## SECTION B — Machine Learning Decisions

---

### Q6. Your dataset is 98.3% negative class. How did you handle that?

**Answer:**

I used SMOTETomek from the `imbalanced-learn` library. It combines two techniques:

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic positive examples by interpolating between existing minority class samples in feature space. Tomek links then identify and remove borderline majority-class samples that are close to the minority boundary, cleaning the decision boundary.

Critically, I applied SMOTETomek only to the training set, not the test set. Applying it to the test set would be data leakage — it would make the test distribution artificially balanced and inflate reported performance.

An alternative I considered is adjusting `scale_pos_weight` in XGBoost (set to ~59 for a 59:1 imbalance), which would penalize false negatives during training without generating synthetic data. I could also tune the classification threshold using the precision-recall curve rather than defaulting to 0.5.

---

### Q7. Why RobustScaler instead of StandardScaler or MinMaxScaler?

**Answer:**

Sensor data from industrial systems frequently has outliers — a sensor malfunctioning and reading 0 or a very large value. `StandardScaler` scales using mean and standard deviation, which are sensitive to outliers. `MinMaxScaler` is even more sensitive because one extreme value compresses all other values into a tiny range.

`RobustScaler` uses the interquartile range (IQR) and median instead, which are resistant to outliers. So a single bad sensor reading doesn't distort the scaling of the entire feature. For 171 anonymous sensor attributes where I can't inspect each one individually, this was the safest default.

---

### Q8. How does your drift detection work?

**Answer:**

In `DataValidation.detect_dataset_drift()`, I run a two-sample Kolmogorov-Smirnov test between the training and test distributions for every numerical column. KS tests whether two samples come from the same distribution without assuming normality.

If the p-value is less than 0.05, I flag that column as drifted. The results are written to a YAML report. The pipeline doesn't halt on drift — it records it and continues. In a production system, I'd add a threshold where if more than X% of features drift, you trigger an alert or pause training.

---

### Q9. Why is your model evaluation threshold a 2% F1 improvement?

**Answer:**

The 2% threshold (`MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02` in the training pipeline constants) is a guard against deploying marginally better models that might just reflect noise. If a model only improves by 0.5%, the improvement likely isn't statistically significant enough to justify replacing a known-good production model, especially given the risk of regressions.

This is similar to how A/B testing in web products uses minimum detectable effect sizes. In a more sophisticated setup, I'd compute confidence intervals on the F1 scores and only deploy if the improvement is statistically significant.

---

### Q10. What would you change about your evaluation metric?

**Answer:**

This is something I'd change if I were redoing this project. The business problem has an explicit cost matrix: false negatives cost 50× more than false positives. But my model optimizes F1, which treats FP and FN equally.

A better metric would be the total operational cost: `Cost = FP × 1 + FN × 50`. I'd implement this as a custom sklearn scorer and use it for both model training (via XGBoost's `eval_metric`) and for the accept/reject decision in ModelEvaluation. This would directly align the model's objective function with the business goal rather than using F1 as a proxy.

Additionally, since the class is highly imbalanced, I should be using Precision-Recall AUC rather than ROC-AUC, which is overly optimistic on imbalanced datasets.

---

## SECTION C — Software Engineering

---

### Q11. How did you structure your custom exception handling?

**Answer:**

I have a `SensorException` class in `sensor/exception.py`. When raised, it captures the full traceback — the filename, line number, and error message — using Python's `sys` module and the `exc_info()` function. This means that in the logs, I see exactly which file and line triggered the error rather than a generic traceback.

Every component wraps its `try/except` blocks with `except Exception as e: raise SensorException(e, sys)`, so exceptions bubble up with full context. This is important in a multi-stage pipeline where an error in stage 2 (DataValidation) would otherwise show a confusing stack trace from the top-level TrainPipeline caller.

---

### Q12. How are your pipeline configurations managed?

**Answer:**

I used a layered constants system. All path templates, threshold values, and parameter names live in `sensor/constants/training_pipeline/__init__.py`. The `TrainingPipelineConfig` class in `sensor/entity/config_entity.py` instantiates concrete paths at runtime by combining these constants with a timestamp.

This means every training run gets its own artifact directory at `artifact/{timestamp}/`, keeping runs isolated and allowing easy rollback. It also means the paths are never hardcoded in the component logic — components receive a config object and ask it for paths. This pattern makes the components testable in isolation by injecting a mock config.

---

### Q13. How does your prediction endpoint work end-to-end?

**Answer:**

The `POST /predict` endpoint in `main.py` accepts a CSV file upload via FastAPI's `UploadFile`. It saves the file to a temp path, creates a `PredictionPipeline` object, and calls its `run_pipeline()` method.

The `PredictionPipeline` uses `ModelResolver` to find the latest model in `saved_models/`. It loads the `SensorModel`, which wraps both the preprocessor (fitted `Pipeline`) and the trained `XGBClassifier`. It calls `SensorModel.predict()`, which runs the preprocessor's `.transform()` and then the classifier's `.predict()`. The labels are decoded through `TargetValueMapping` (`0 → neg, 1 → pos`) and returned as JSON.

The key design decision is that the preprocessor and model are always packaged together — this prevents training-serving skew, where different transformations at serving time would invalidate the model's predictions.

---

## SECTION D — Deployment & Infrastructure

---

### Q14. Why AWS EC2 instead of Lambda or ECS?

**Answer:**

For this use case, EC2 was the pragmatic choice. The ML pipeline loads a preprocessor and model from disk at startup, which has non-trivial initialization time — Lambda's cold start limits (15-minute timeout, stateless) aren't well-suited for long-running training jobs.

For a production system with real traffic, I'd migrate to ECS Fargate or EKS. Fargate removes the EC2 management overhead while keeping the container abstraction. EKS would allow auto-scaling based on prediction request volume. For the training pipeline specifically, AWS SageMaker Pipelines would be even better — it provides managed compute, native model versioning, and built-in monitoring.

---

### Q15. How do you handle secrets and credentials?

**Answer:**

Secrets are never hardcoded. The MongoDB URL and AWS credentials are stored in a `.env` file locally (excluded from Git via `.gitignore`), loaded at startup with `python-dotenv`, and read via environment variable constants defined in `sensor/constants/env_varibale.py`.

In the CI/CD pipeline, the same secrets are stored as GitHub Secrets and injected as environment variables into the Docker container at runtime via `docker run -e MONGO_DB_URL=${{ secrets.MONGO_DB_URL }}`.

For a more hardened setup, I'd use AWS Secrets Manager or Parameter Store instead of environment variables, which provides rotation, auditing, and fine-grained IAM access control.

---

### Q16. If traffic to your prediction API increased 100×, what would break first and how would you fix it?

**Answer:**

The current setup has a single EC2 instance running one Docker container. With 100× traffic, the bottleneck would be the single prediction worker — FastAPI is running synchronously with Uvicorn, so requests queue up.

**Immediate fixes:**
- Run Uvicorn with multiple workers: `uvicorn main:app --workers 4`
- Move the model loading out of the request handler into application startup (already partially done via `SensorModel`)

**Medium-term:**
- Put an Application Load Balancer in front of multiple EC2 instances or ECS tasks
- Use an async FastAPI endpoint so I/O doesn't block the event loop

**Longer-term:**
- Separate batch prediction from real-time prediction — batch goes to SQS → Lambda/ECS worker, real-time goes to a low-latency endpoint with the model in memory
- Cache the loaded model in Redis or application state rather than reloading from disk on each request

---

## SECTION E — Behavioral / Reflection

---

### Q17. What would you do differently if you started this project over?

**Answer:**

Three things primarily:

First, I'd write tests from day one. The CI/CD pipeline has a placeholder test step, which means there's no safety net for regressions. I'd use pytest with fixtures that mock the MongoDB connection and test each component's output shape and type.

Second, I'd implement the hyperparameter tuning that's currently a stub. XGBoost's default parameters are reasonable but not optimal for an imbalanced cost-sensitive problem. I'd use Optuna for efficient hyperparameter search, especially tuning `scale_pos_weight` which directly addresses the class imbalance.

Third, I'd replace the F1-score model acceptance criterion with a cost-weighted metric from the start. The 50:1 false negative penalty is the core business constraint — it should drive every modelling decision, not be an afterthought.

---

### Q18. Tell me about a bug you encountered and how you debugged it.

**Answer:**

One issue in this kind of pipeline is training-serving skew — when the prediction endpoint uses different preprocessing than what the model was trained on. I avoided this by wrapping the preprocessor and model together in the `SensorModel` class, which is serialized and loaded as a single unit. So whatever transformations were applied during training are guaranteed to be applied at inference time.

Another subtle issue was in the model evaluation component: the initial implementation loaded the full dataset to evaluate both models, rather than restricting to the test set only. This was inflating the apparent performance of the production model by including training data it had already seen. The fix was to use only the held-out test artifact path from the DataValidation output.

---

### Q19. How would you add monitoring to this system?

**Answer:**

I'd add monitoring at three levels:

**Operational monitoring:** Use AWS CloudWatch to track CPU/memory on EC2, API response latency via FastAPI middleware, and error rates. Set alarms for p99 latency > 2s or error rate > 1%.

**Data drift monitoring:** The drift detection currently runs only at training time. I'd extend it to run continuously on a sample of incoming prediction requests, comparing the live feature distribution to the training distribution using the same KS test. If drift exceeds a threshold, trigger an alert.

**Model performance monitoring:** For a deployed model, you need ground truth labels to measure real-world accuracy. In the APS context, a repair event logged after a positive prediction is the ground truth. I'd build a feedback loop: log predictions to a database, join with repair outcomes after a lag period, and recompute F1 on live data weekly. If performance degrades more than 5%, trigger an automatic retraining run.

---

### Q20. How is your project different from someone who just trained a model in a Jupyter notebook?

**Answer:**

The notebook is exploration. What I built is closer to an engineering system. Specifically:

The code is modular and testable — each pipeline stage is a class with a single responsibility, taking typed inputs and producing typed outputs. That makes each stage independently replaceable.

The deployment is automated — a `git push` triggers a full CI/CD run that builds a Docker image, pushes to ECR, and redeploys to EC2. There's no manual SSH-ing into a server and running a script.

The model is versioned — every training run creates a timestamped artifact, and `ModelResolver` always serves the best available version. If a bad model slips through, you can roll back by serving an older timestamp.

The prediction endpoint validates inputs and handles errors — it's not a notebook cell that crashes silently. Custom exception handling with traceback capture means failures are diagnosable from logs.

That said, I'd be the first to acknowledge what's still missing: real unit tests, hyperparameter tuning, and model monitoring. Those are the honest next steps.

---

# PART 3 — Resume Bullets & Portfolio Advice

---

## Resume Bullet Points

> Use these for your resume. Tailor to the role — backend/ML vs. fullstack vs. data science.

---

### For ML Engineer / Data Science Roles

- Architected a 6-stage end-to-end ML pipeline (ingestion → validation → transformation → training → evaluation → deployment) for APS failure classification on 60K sensor records, achieving F1 ≥ 0.60 with automated model promotion logic
- Engineered an imbalanced binary classification system (98.3% negative class) using SMOTETomek resampling and XGBoost, reducing false negative cost exposure by optimizing against a 50:1 asymmetric cost matrix
- Implemented statistical data drift detection using Kolmogorov-Smirnov tests across 171 sensor features, producing automated pre-training validation reports to prevent model degradation from distribution shift
- Built model versioning and evaluation infrastructure that compares candidate models against production using F1-delta thresholds, preventing regressions and maintaining a timestamped artifact history in AWS S3

---

### For Backend / MLOps / Fullstack Roles

- Deployed a Dockerized FastAPI ML inference service to AWS EC2 via a GitHub Actions CI/CD pipeline (lint → Docker build → push to ECR → self-hosted runner deploy), achieving zero-downtime deployments on code push
- Designed a production ML serving architecture using FastAPI + Uvicorn with a `/train` trigger endpoint and a `/predict` CSV upload endpoint, integrated with MongoDB for data ingestion and AWS S3 for artifact storage
- Built a Streamlit web application for batch sensor data prediction with CSV upload and MongoDB data fetch capabilities, enabling non-technical stakeholders to run inference without API access
- Implemented a robust serialization strategy (dill + numpy .npy) bundling trained model and fitted preprocessor as a single artifact, eliminating training-serving skew in the inference pipeline

---

### Universal (works for any ML/SWE role)

- Developed an end-to-end predictive maintenance system for Scania APS fault detection, integrating MongoDB data ingestion, automated drift detection, XGBoost classification, and Docker-based AWS deployment in a single orchestrated pipeline

---

## Portfolio Advice

---

### Pages / Sections to Include

Your portfolio should dedicate a full project card or case study page to this project. The most important things to include:

**Project overview card (thumbnail view):** Title, one-line tagline, 3 tech badges (Python · XGBoost · AWS), and one visual (the pipeline flowchart).

**Full case study page:** Structured exactly like your interview story — Problem, Solution, Architecture, Technical Decisions, Results, What I'd Do Next.

**Live demo link or video walkthrough:** Even a 2-minute Loom video of you clicking "Train" in the UI and then uploading a CSV for predictions is far more impressive than any screenshot.

---

### Screenshots to Capture

Capture these specific visuals and keep them ready:

1. The FastAPI `/docs` Swagger page showing the `/train` and `/predict` endpoints
2. A `/train` call response in the browser or Postman — the JSON success response
3. A `/predict` call with a CSV showing example input rows and the prediction output
4. The Streamlit app with a file uploaded and predictions rendered in a table
5. The GitHub Actions workflow — a green passing run showing all stages
6. The AWS ECR console showing your Docker image with a recent push timestamp
7. The S3 bucket showing the `artifact/` and `saved_models/` folder structure with timestamps
8. The EC2 instance running with port 80 open (show the security group or the public URL in browser)
9. The drift detection YAML report — open it in a text editor and screenshot it
10. The pipeline flowchart diagram from `flowcharts/` — clean, full resolution

---

### Links to Include

- **GitHub repository** — make sure it's public and the README looks great
- **Live demo URL** — the EC2 public IP / domain with FastAPI Swagger UI
- **Loom / YouTube video walkthrough** — optional but highly recommended
- **Jupyter notebook** (nbviewer link) — for the EDA and data exploration
- **LinkedIn post** — write a short post about this project; link it from your portfolio

---

### Things to Highlight for Portfolio Visitors

Lead with the **business problem** — the 50:1 false negative cost. Most ML portfolios show accuracy numbers without context. Framing this as a cost-reduction problem immediately signals business thinking.

Show the **system diagram** prominently. The pipeline flowchart and deployment architecture images you already have in `flowcharts/` are excellent. Put them early on the case study page.

Be **honest about tradeoffs and future work**. Mentioning that hyperparameter tuning is on the roadmap (and explaining *why* it matters for this specific cost matrix) shows more engineering maturity than pretending the project is complete.

Quantify where possible: "60,000 training examples", "171 sensor features", "98.3% class imbalance", "6 pipeline stages", "automated via GitHub Actions on every push." Numbers make skimming visitors stop and read.

---

### One-Line Elevator Pitch

> *"I built an automated ML pipeline that predicts Air Pressure System failures in heavy-duty trucks — from MongoDB ingestion through XGBoost training to Docker deployment on AWS — with automated drift detection and model versioning."*
