# ML-Flow-Pipeline-With-DVC

A comprehensive MLOps pipeline for the Boston Housing dataset using MLflow for experiment tracking, DVC for data versioning, and Kubernetes/Kubeflow for orchestration.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Minikube Setup](#minikube-setup)
  - [Kubeflow Pipelines Setup](#kubeflow-pipelines-setup)
  - [DVC Remote Storage Setup](#dvc-remote-storage-setup)
  - [Local Development Setup](#local-development-setup)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Compiling the Pipeline](#compiling-the-pipeline)
  - [Running the Pipeline](#running-the-pipeline)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [CI/CD](#cicd)

---

## 🎯 Project Overview

### ML Problem

This project implements an end-to-end MLOps pipeline for predicting house prices using the **Boston Housing dataset**. The problem is a **regression task** where we predict the median value of owner-occupied homes (MEDV) based on various features such as crime rate, number of rooms, accessibility to highways, etc.

### Pipeline Components

The pipeline consists of four main stages:

1. **Data Extraction**: Pulls data from DVC remote storage or uses local files
2. **Data Preprocessing**: Performs feature scaling and train/test split
3. **Model Training**: Trains a RandomForestRegressor model
4. **Model Evaluation**: Computes and logs regression metrics (MSE, RMSE, MAE, R²)

### Technology Stack

- **MLflow**: Experiment tracking, model versioning, and artifact storage
- **DVC**: Data version control and remote storage
- **Kubeflow Pipelines**: Containerized pipeline orchestration (optional)
- **Scikit-learn**: Machine learning models and preprocessing
- **Python 3.10**: Core programming language
- **Docker**: Containerization
- **Kubernetes/Minikube**: Container orchestration

---

## 🚀 Setup Instructions

### Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.10+**
- **Docker** (for containerization)
- **kubectl** (Kubernetes command-line tool)
- **Minikube** (for local Kubernetes cluster)
- **Git**

### Minikube Setup

Minikube allows you to run Kubernetes locally for testing and development.

#### 1. Install Minikube

**macOS:**
```bash
brew install minikube
```

**Linux:**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

**Windows:**
Download and install from [Minikube releases](https://github.com/kubernetes/minikube/releases)

#### 2. Start Minikube

```bash
# Start Minikube with sufficient resources for Kubeflow
minikube start --cpus=4 --memory=8192 --disk-size=20g

# Verify Minikube is running
minikube status

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server
```

#### 3. Configure kubectl

```bash
# Set kubectl to use Minikube context
kubectl config use-context minikube

# Verify connection
kubectl cluster-info
```

#### 4. Access Minikube Dashboard (Optional)

```bash
minikube dashboard
```

This will open the Kubernetes dashboard in your browser.

### Kubeflow Pipelines Setup

Kubeflow Pipelines provides a platform for building and deploying portable, scalable ML workflows.

#### Option 1: Using Kubeflow Pipelines Standalone (Recommended for Local Development)

1. **Install Kubeflow Pipelines SDK:**

```bash
pip install kfp
```

2. **Start Kubeflow Pipelines UI using Docker:**

```bash
# Pull the Kubeflow Pipelines UI image
docker pull gcr.io/ml-pipeline/frontend:latest

# Run the UI (optional, for visualization)
docker run -d -p 3000:3000 --name kfp-ui gcr.io/ml-pipeline/frontend:latest
```

3. **For Full Kubeflow Installation on Minikube:**

```bash
# Install kustomize
brew install kustomize  # macOS
# or download from https://kustomize.io/

# Clone Kubeflow manifests
git clone https://github.com/kubeflow/manifests.git
cd manifests

# Install Kubeflow (this may take 15-30 minutes)
kubectl apply -k example

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod --all -n kubeflow --timeout=300s
```

4. **Access Kubeflow Pipelines UI:**

```bash
# Port forward to access the UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Open http://localhost:8080 in your browser
```

#### Option 2: Using MLflow (Current Implementation)

The current pipeline uses MLflow for orchestration. To use MLflow:

```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### DVC Remote Storage Setup

DVC (Data Version Control) manages data files and tracks them in remote storage.

#### 1. Install DVC

```bash
pip install dvc

# Verify installation
dvc --version
```

#### 2. Initialize DVC

```bash
# Navigate to project root
cd /path/to/ML-Flow-Pipeline-With-DVC

# Initialize DVC repository
dvc init

# Commit DVC configuration
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

#### 3. Set Up Remote Storage

Choose one of the following remote storage options:

**Option A: Local Directory (for testing)**

```bash
# Create remote directory
mkdir -p ../dvc_remote

# Add remote
dvc remote add -d myremote ../dvc_remote

# Verify
dvc remote list
```

**Option B: Google Drive**

```bash
# Install DVC Google Drive plugin
pip install dvc[gdrive]

# Add Google Drive remote (replace FOLDER_ID with your Google Drive folder ID)
dvc remote add -d myremote gdrive://FOLDER_ID

# Authenticate (follow prompts)
dvc remote modify myremote gdrive_use_service_account true
```

**Option C: AWS S3**

```bash
# Install DVC S3 plugin
pip install dvc[s3]

# Configure AWS credentials
aws configure

# Add S3 remote (replace with your bucket name)
dvc remote add -d myremote s3://your-bucket-name/dvc-storage
```

**Option D: Use the Setup Script**

```bash
# Run the automated setup script
chmod +x setup_dvc.sh
./setup_dvc.sh
```

#### 4. Add Data to DVC

```bash
# Add data file to DVC tracking
dvc add data/raw/boston_housing.csv

# Commit DVC files to Git
git add data/raw/boston_housing.csv.dvc data/raw/.gitignore
git commit -m "Add data file to DVC"

# Push data to remote storage
dvc push
```

#### 5. Verify DVC Setup

```bash
# Check DVC status
dvc status

# List tracked files
dvc list .
```

### Local Development Setup

1. **Clone the Repository:**

```bash
git clone <your-repo-url>
cd ML-Flow-Pipeline-With-DVC
```

2. **Create Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Pull Data from DVC:**

```bash
dvc pull
```

5. **Verify Setup:**

```bash
# Check if data file exists
ls -la data/raw/boston_housing.csv

# Test MLflow
mlflow --version
```

---

## 🔄 Pipeline Walkthrough

### Compiling the Pipeline

The pipeline can be run in two modes:

#### Mode 1: MLflow Pipeline (Current Implementation)

The current implementation uses MLflow for orchestration. No compilation is needed - the pipeline runs directly as a Python script.

#### Mode 2: Kubeflow Pipelines (For Kubernetes Deployment)

If you want to compile the pipeline for Kubeflow Pipelines:

1. **Install Kubeflow Pipelines SDK:**

```bash
pip install kfp
```

2. **Compile the Pipeline:**

```bash
# The pipeline components are defined in components/*.yaml
# Compile using kfp compiler (if you have a Kubeflow pipeline definition)
kfp dsl-compile --py pipelines/kubeflow_pipeline.py --output pipeline.yaml
```

### Running the Pipeline

#### Method 1: Using MLflow (Recommended)

1. **Start MLflow UI (Optional):**

```bash
mlflow ui
# Access at http://localhost:5000
```

2. **Run the Pipeline:**

```bash
# Using default config
python -m pipelines.pipeline

# Using custom config
python -m pipelines.pipeline --config configs/config.yaml
```

3. **View Results:**

- **MLflow UI**: Open http://localhost:5000 to view experiments, metrics, and artifacts
- **Local Outputs**: Check `outputs/models/` and `outputs/metrics/` directories

#### Method 2: Using Docker

1. **Build Docker Image:**

```bash
docker build -t mlflow-pipeline:latest .
```

2. **Run Container:**

```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/outputs:/app/outputs \
           -v $(pwd)/mlruns:/app/mlruns \
           mlflow-pipeline:latest
```

#### Method 3: Using Kubeflow Pipelines (If Configured)

1. **Upload Pipeline to Kubeflow:**

```bash
# Using kubectl (if pipeline is compiled)
kubectl apply -f pipeline.yaml

# Or using KFP SDK
python upload_pipeline.py
```

2. **Create and Run Experiment:**

- Access Kubeflow Pipelines UI at http://localhost:8080
- Create a new experiment
- Upload the compiled pipeline
- Configure parameters and run

#### Method 4: Using Jenkins (CI/CD)

The project includes a `Jenkinsfile` for CI/CD automation:

```bash
# If Jenkins is configured, the pipeline will run automatically on push
git push origin main
```

---

## 📁 Project Structure

```
ML-Flow-Pipeline-With-DVC/
├── components/                 # Pipeline component definitions (YAML)
│   ├── extract_data_component.yaml
│   ├── preprocess_data_component.yaml
│   ├── train_model_component.yaml
│   ├── evaluate_model_component.yaml
│   └── README.md
├── configs/                    # Configuration files
│   └── config.yaml
├── data/                       # Data directory (tracked by DVC)
│   ├── raw/                    # Raw data files
│   │   └── boston_housing.csv
│   └── processed/              # Processed data files
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── scaler.pkl
├── mlruns/                     # MLflow experiment tracking data
├── outputs/                    # Pipeline outputs
│   ├── models/                 # Trained models
│   │   └── model.pkl
│   └── metrics/                # Evaluation metrics
│       └── metrics.json
├── pipelines/                  # Pipeline orchestration
│   ├── __init__.py
│   └── pipeline.py            # Main pipeline script
├── src/                        # Source code
│   ├── __init__.py
│   ├── pipeline_components.py # Pipeline component implementations
│   ├── model_training.py      # Model training utilities
│   └── utils.py               # Helper functions
├── .dvc/                       # DVC configuration (git-ignored)
├── Dockerfile                  # Docker image definition
├── Jenkinsfile                 # CI/CD pipeline definition
├── requirements.txt            # Python dependencies
├── setup_dvc.sh               # DVC setup automation script
└── README.md                   # This file
```

---

## ⚙️ Configuration

The pipeline configuration is defined in `configs/config.yaml`:

```yaml
run_name: "boston_housing_mlflow"
seed: 42

data:
  dvc_path: "data.dvc"
  local_raw_path: "data/raw/boston_housing.csv"
  processed_dir: "data/processed"
  test_size: 0.2

model:
  output_dir: "outputs/models"
  n_estimators: 200
  max_depth: 10

evaluation:
  output_path: "outputs/metrics/metrics.json"
```

### Customizing Configuration

Edit `configs/config.yaml` to modify:
- **Model hyperparameters**: `n_estimators`, `max_depth`
- **Data split**: `test_size`
- **Random seed**: `seed`
- **Output paths**: All output directories and file paths

---

## 💡 Usage Examples

### Example 1: Run Pipeline with Default Config

```bash
python -m pipelines.pipeline
```

### Example 2: Run Pipeline with Custom Config

```bash
python -m pipelines.pipeline --config configs/config.yaml
```

### Example 3: Pull Latest Data and Run Pipeline

```bash
# Pull latest data from DVC remote
dvc pull

# Run pipeline
python -m pipelines.pipeline --config configs/config.yaml
```

### Example 4: View MLflow Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
# Browse experiments, compare runs, view metrics and artifacts
```

### Example 5: Compare Multiple Runs

```bash
# Run pipeline multiple times with different configs
python -m pipelines.pipeline --config configs/config.yaml

# Edit config.yaml to change hyperparameters
# Run again
python -m pipelines.pipeline --config configs/config.yaml

# Compare in MLflow UI
mlflow ui
```

---

## 🔧 CI/CD

The project includes a `Jenkinsfile` for continuous integration and deployment:

### Jenkins Pipeline Stages

1. **Checkout**: Clone repository
2. **Install Dependencies**: Install Python packages
3. **DVC Setup**: Pull data from DVC remote
4. **Run Pipeline**: Execute ML pipeline
5. **Archive Results**: Store artifacts

### Running Jenkins Pipeline

1. **Set up Jenkins** (if not already configured)
2. **Create Pipeline Job** pointing to the `Jenkinsfile`
3. **Configure Credentials** for DVC remote (if needed)
4. **Trigger Pipeline** manually or on Git push

---

## 📊 Monitoring and Tracking

### MLflow Tracking

- **Experiments**: All runs are tracked in MLflow
- **Metrics**: MSE, RMSE, MAE, R² are logged automatically
- **Parameters**: Hyperparameters and configuration are logged
- **Artifacts**: Models, scalers, and metrics files are stored
- **Model Registry**: Models can be registered and versioned

### Accessing MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Then open http://localhost:5000 in your browser.

---

## 🐛 Troubleshooting

### Common Issues

1. **DVC Pull Fails**
   - Verify remote storage is configured: `dvc remote list`
   - Check credentials for cloud storage
   - Ensure data file exists in remote: `dvc fetch`

2. **MLflow Tracking Errors**
   - Check MLflow tracking URI: `echo $MLFLOW_TRACKING_URI`
   - Ensure `mlruns/` directory is writable
   - Verify MLflow is installed: `pip show mlflow`

3. **Minikube Issues**
   - Restart Minikube: `minikube stop && minikube start`
   - Check cluster status: `minikube status`
   - View logs: `minikube logs`

4. **Kubeflow Pipelines Not Accessible**
   - Check pod status: `kubectl get pods -n kubeflow`
   - Verify port forwarding: `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`
   - Check service status: `kubectl get svc -n kubeflow`

---

## 📝 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy MLOps! 🚀**

