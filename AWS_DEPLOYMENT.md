# AWS Deployment Guide — APS Failure Classification

## Prerequisites

- AWS account with billing enabled
- AWS CLI installed locally (`aws --version`)
- Docker installed locally (`docker --version`)
- Git repository pushed to GitHub
- MongoDB Atlas cluster running with data loaded

---

## Step 1 — Create IAM User for CI/CD

1. Go to **AWS Console → IAM → Users → Create User**
2. Set username: `aps-cicd-user`
3. Select **Attach policies directly** and add:
   - `AmazonEC2FullAccess`
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonS3FullAccess`
4. After creation go to **Security credentials → Create access key**
5. Choose **CLI** as use case
6. Save the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` — you won't see them again

---

## Step 2 — Configure AWS CLI Locally

```bash
aws configure
# AWS Access Key ID: <your key>
# AWS Secret Access Key: <your secret>
# Default region name: us-east-1
# Default output format: json
```

---

## Step 3 — Create S3 Bucket

```bash
aws s3 mb s3://aps-failure-classification --region us-east-1
```

Verify it was created:

```bash
aws s3 ls | grep aps-failure-classification
```

---

## Step 4 — Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name aps-failure-classification \
  --region us-east-1
```

The response will include a `repositoryUri` that looks like:

```
123456789012.dkr.ecr.us-east-1.amazonaws.com/aps-failure-classification
```

Save the base URI (`123456789012.dkr.ecr.us-east-1.amazonaws.com`) — you'll need it for GitHub Secrets.

---

## Step 5 — Launch EC2 Instance

1. Go to **EC2 → Launch Instance**
2. Configure with these settings:

| Setting | Value |
|---|---|
| Name | `aps-failure-classification` |
| AMI | Ubuntu Server 22.04 LTS (64-bit x86) |
| Instance type | `t2.medium` (minimum — model training needs RAM) |
| Key pair | Create new → download `.pem` file |
| Storage | 20 GB gp3 |

3. Under **Network settings → Edit**, add these inbound rules to the Security Group:

| Type | Port | Source |
|---|---|---|
| SSH | 22 | My IP |
| HTTP | 80 | 0.0.0.0/0 |
| Custom TCP | 8080 | 0.0.0.0/0 |

4. Click **Launch instance** and wait for it to reach running state
5. Note the **Public IPv4 DNS** (e.g. `ec2-xx-xx-xx-xx.compute-1.amazonaws.com`)

---

## Step 6 — Install Docker on EC2

SSH into your instance:

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_DNS>
```

Then run:

```bash
# Update packages
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add ubuntu user to docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker ubuntu
newgrp docker

# Verify installation
docker --version
```

---

## Step 7 — Set Up GitHub Actions Self-Hosted Runner on EC2

This allows GitHub Actions to deploy directly to your EC2 instance.

1. Go to your GitHub repo → **Settings → Actions → Runners → New self-hosted runner**
2. Select **Linux** and **x64**
3. Run the commands GitHub provides on your EC2 instance:

```bash
# On EC2 — create a folder for the runner
mkdir actions-runner && cd actions-runner

# Download the runner (use the exact URL GitHub gives you)
curl -o actions-runner-linux-x64-2.x.x.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.x.x/actions-runner-linux-x64-2.x.x.tar.gz

tar xzf ./actions-runner-linux-x64-2.x.x.tar.gz

# Configure — use the token GitHub gives you
./config.sh \
  --url https://github.com/<your-username>/APS-failure-classification \
  --token <YOUR_REGISTRATION_TOKEN>

# Install as a systemd service so it survives reboots
sudo ./svc.sh install
sudo ./svc.sh start
```

4. Go back to **GitHub → Settings → Actions → Runners** — your runner should show as **Idle**

---

## Step 8 — Add GitHub Repository Secrets

Go to **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

Add each of the following:

| Secret Name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key from Step 1 |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key from Step 1 |
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `ECR_REPOSITORY_NAME` | `aps-failure-classification` |
| `AWS_ECR_LOGIN_URI` | base ECR URI, e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com` |
| `EC2_PUBLIC_DNS` | your EC2 public DNS from Step 5 |
| `MONGO_DB_URL` | your full MongoDB Atlas connection string |

---

## Step 9 — GitHub Actions Workflow

Ensure your `.github/workflows/main.yml` contains all three jobs. The `Continuous-Deployment` job should look like this:

```yaml
Continuous-Deployment:
  needs: build-and-push-ecr-image
  runs-on: self-hosted
  steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ secrets.AWS_DEFAULT_REGION }}

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Pull latest image from ECR
      run: |
        docker pull ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest

    - name: Stop and remove existing container
      run: |
        docker stop aps-cont || true
        docker rm aps-cont || true

    - name: Run Docker container
      run: |
        docker run -d \
          -p 8080:8080 \
          --name aps-cont \
          -e MONGO_DB_URL="${{ secrets.MONGO_DB_URL }}" \
          -e AWS_ACCESS_KEY_ID="${{ secrets.AWS_ACCESS_KEY_ID }}" \
          -e AWS_SECRET_ACCESS_KEY="${{ secrets.AWS_SECRET_ACCESS_KEY }}" \
          -e AWS_DEFAULT_REGION="${{ secrets.AWS_DEFAULT_REGION }}" \
          ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest

    - name: Clean up old Docker images
      run: docker image prune -f
```

---

## Step 10 — Trigger Deployment

Push any change to the `main` branch to kick off the full pipeline:

```bash
git add .
git commit -m "feat: trigger aws deployment"
git push origin main
```

Monitor progress in **GitHub → Actions tab**. You will see three jobs run in sequence:

1. `integration` — runs flake8 linting and pytest tests
2. `build-and-push-ecr-image` — builds Docker image and pushes to ECR
3. `Continuous-Deployment` — EC2 pulls the image and starts the container

---

## Step 11 — Verify Deployment

```bash
# SSH into EC2 and check container is running
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_DNS>
docker ps

# View container logs
docker logs aps-cont

# Test the API from your local machine
curl http://<EC2_PUBLIC_IP>:8080/

# Trigger training via API
curl -X POST http://<EC2_PUBLIC_IP>:8080/train

# Test prediction endpoint
curl -X POST http://<EC2_PUBLIC_IP>:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [...]}'
```

Your FastAPI app is live at:
```
http://<EC2_PUBLIC_IP>:8080
```

Interactive API docs available at:
```
http://<EC2_PUBLIC_IP>:8080/docs
```

---

## Artifacts Folder Structure

After a successful pipeline run, the following structure will be created in S3:

```
artifacts/
└── <timestamp>/
    ├── data_ingestion/
    │   ├── feature_store/
    │   │   └── sensor.csv
    │   └── ingested/
    │       ├── train.csv
    │       └── test.csv
    ├── data_validation/
    │   ├── validated/
    │   │   ├── train.csv
    │   │   └── test.csv
    │   ├── invalidated/
    │   └── drift_report/
    │       └── report.yaml
    ├── data_transformation/
    │   ├── transformed/
    │   │   ├── train.npy
    │   │   └── test.npy
    │   └── transformed_object/
    │       └── preprocessing.pkl
    ├── model_trainer/
    │   └── trained_model/
    │       └── model.pkl
    └── model_evaluation/
        └── report.yaml

saved_models/
└── <timestamp>/
    └── model.pkl
```

---

## Troubleshooting

**Container exits immediately**
```bash
docker logs aps-cont
```
Usually caused by a missing environment variable — check that `MONGO_DB_URL` secret is set correctly.

**ECR push fails in CI**
Confirm the IAM user has `AmazonEC2ContainerRegistryFullAccess` and the `AWS_ECR_LOGIN_URI` secret does not include the repository name (base URI only).

**Self-hosted runner shows offline**
```bash
# SSH into EC2
cd ~/actions-runner
sudo ./svc.sh status
sudo ./svc.sh start
```

**Port 8080 unreachable from browser**
Check that the EC2 Security Group has an inbound rule for **Custom TCP port 8080** from `0.0.0.0/0`.

**Out of disk space on EC2**
```bash
docker system prune -af
```

**S3 sync fails during pipeline**
Verify the bucket name in `sensor/constants/training_pipeline.py` matches the bucket created in Step 3 exactly.
