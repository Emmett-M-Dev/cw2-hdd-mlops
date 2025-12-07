# Hard Drive Failure Prediction - MLOps Pipeline

> **CW2: MLOps Implementation using Azure Machine Learning**
> Predictive maintenance system for hard drive failures using scalable ML infrastructure

[![Azure ML](https://img.shields.io/badge/Azure-ML-blue)](https://ml.azure.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-green)](https://github.com/features/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [MLOps Architecture](#mlops-architecture)
- [Azure ML Deployment](#azure-ml-deployment)
- [Model Iterations](#model-iterations)
- [Setup Instructions](#setup-instructions)
- [Testing & Evaluation](#testing--evaluation)
- [CI/CD Pipeline](#cicd-pipeline)
- [Local Development](#local-development)
- [Performance & Scalability](#performance--scalability)
- [References](#references)

---

## 🎯 Problem Statement

Hard drive failures in data centers cause:
- **Data loss** and service disruptions
- **High replacement costs** (millions of drives deployed globally)
- **Operational overhead** in reactive maintenance

**Solution**: Predictive maintenance using machine learning to identify at-risk drives before failure, enabling proactive replacement and minimizing downtime.

### Why MLOps is Critical:
- **Scale**: Data centers have millions of drives generating continuous telemetry
- **Real-time**: Predictions must be served with low latency
- **Iteration**: Models need regular retraining as drive populations change
- **Reproducibility**: Experiments must be tracked and versions managed
- **Deployment**: Models must be deployed reliably across environments

---

## 🏗️ MLOps Architecture

### Infrastructure Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure ML Workspace                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Dataset    │───▶│   Compute    │───▶│ Experiments  │ │
│  │  (Tabular)   │    │   Cluster    │    │  (MLflow)    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                 │            │
│                                                 ▼            │
│                           ┌──────────────────────────────┐  │
│                           │    Model Registry            │  │
│                           │  (v1: Logistic, v2: Forest)  │  │
│                           └──────────────────────────────┘  │
│                                                 │            │
│                                                 ▼            │
│                           ┌──────────────────────────────┐  │
│                           │   Managed Endpoint           │  │
│                           │   (Real-time inference)      │  │
│                           └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   GitHub Actions CI/CD   │
                    │  (Automated testing &    │
                    │   deployment pipeline)   │
                    └──────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Cloud Platform** | Azure Machine Learning | Scalable ML infrastructure |
| **Experiment Tracking** | MLflow (integrated) | Version control for experiments |
| **Model Registry** | Azure ML Model Registry | Centralized model versioning |
| **Compute** | Azure ML Compute Clusters | Auto-scaling training infrastructure |
| **Deployment** | Azure ML Managed Endpoints | Production model serving |
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Testing** | pytest | Unit & integration tests |
| **Version Control** | Git + GitHub | Code & configuration management |

---

## ☁️ Azure ML Deployment

### Workspace Configuration

- **Workspace Name**: `cw2-hdd-workspace`
- **Region**: UK South
- **Resource Group**: `cw2-hdd-mlops`
- **Compute**: Standard_DS3_v2 (0-1 nodes, auto-scaling)

### Dataset

- **Name**: `hdd_balanced_dataset`
- **Format**: Tabular CSV
- **Size**: 8,961 samples
- **Class Distribution**: 66.7% non-failure, 33.3% failure (balanced for training)
- **Features**:
  - `capacity_bytes` (normalized 0-1)
  - `lifetime` (normalized 0-1)
  - `model_encoded` (integer 0-54)
- **Target**: `failure` (binary: 0=healthy, 1=will fail)

### Experiment: `hdd_failure_prediction`

Both iterations tracked with comprehensive metrics via MLflow integration.

---

## 🔬 Model Iterations

### Iteration 1: Logistic Regression (Baseline)

**Purpose**: Establish baseline performance with interpretable linear model

**Configuration**:
```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

**Results**:
- **Accuracy**: 74.1%
- **Precision**: 71.4%
- **Recall**: 37.1%
- **F1-Score**: 48.8%
- **ROC-AUC**: 69.7%

**Analysis**: Simple model struggles with complex patterns in drive failure data. Low recall indicates many failures missed (high false negatives).

### Iteration 2: Random Forest (Improved)

**Purpose**: Capture non-linear relationships and feature interactions

**Configuration**:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

**Results**:
- **Accuracy**: 98.1%
- **Precision**: 99.1%
- **Recall**: 95.2%
- **F1-Score**: 97.1%
- **ROC-AUC**: 99.9%

**Analysis**: Significant improvement across all metrics. High recall critical for catching failures before they occur.

### Performance Comparison

| Metric | Iteration 1 (Logistic) | Iteration 2 (Random Forest) | Improvement |
|--------|------------------------|----------------------------|-------------|
| Accuracy | 74.1% | 98.1% | **+24.0 pp** |
| Precision | 71.4% | 99.1% | **+27.7 pp** |
| Recall | 37.1% | 95.2% | **+58.1 pp** |
| F1-Score | 48.8% | 97.1% | **+48.3 pp** |
| ROC-AUC | 69.7% | 99.9% | **+30.2 pp** |

**Key Insight**: Random Forest's ability to model complex feature interactions dramatically improves failure detection, particularly recall (critical for preventing data loss).

### Feature Importance (Random Forest)

1. **lifetime** (0.52) - Most predictive feature
2. **capacity_bytes** (0.28) - Moderate importance
3. **model_encoded** (0.20) - Least important

---

## 🚀 Setup Instructions

### Prerequisites

1. **Azure Subscription**: Active subscription with Azure ML access
2. **Python 3.11**: Local Python environment
3. **Git**: Version control
4. **Azure CLI** (optional): For advanced configuration

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/cw2-hdd-mlops.git
cd cw2-hdd-mlops
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 3: Configure Azure ML Workspace

**Option A: Azure Portal** (Recommended for first-time setup)

1. Go to [portal.azure.com](https://portal.azure.com)
2. Create Resource Group: `cw2-hdd-mlops`
3. Create Azure ML Workspace: `cw2-hdd-workspace`
4. Download `config.json` from workspace Overview page
5. Place `config.json` in project root

**Option B: Azure CLI**

```bash
az login
az ml workspace create \
  --name cw2-hdd-workspace \
  --resource-group cw2-hdd-mlops \
  --location uksouth
```

### Step 4: Create Compute Resources

In [Azure ML Studio](https://ml.azure.com):

1. **Compute Cluster**:
   - Name: `training-cluster`
   - VM Size: Standard_DS3_v2
   - Min nodes: 0 (auto-scale to zero for cost saving)
   - Max nodes: 1
   - Idle time: 120 seconds

2. **Compute Instance** (optional, for notebook development):
   - Name: `dev-instance`
   - VM Size: Standard_DS3_v2
   - Auto-shutdown: 30 minutes

### Step 5: Upload Dataset

```bash
python scripts/upload_data_to_azure.py
```

Verify in Azure ML Studio → Data → Datasets → `hdd_balanced_dataset`

### Step 6: Run Training Experiments

#### Iteration 1: Logistic Regression

```bash
python scripts/submit_azure_job.py --iteration 1
```

Monitor progress:
- Command line output shows portal URL
- Open URL to watch real-time logs
- First run takes ~10-15 mins (environment build)
- Subsequent runs: ~3-5 mins

#### Iteration 2: Random Forest

```bash
python scripts/submit_azure_job.py --iteration 2
```

### Step 7: Register Models

```bash
python scripts/register_models.py
```

Verify in Azure ML Studio → Models → `hdd_failure_predictor` (v1 and v2)

### Step 8: Deploy Model (Optional)

**Via Azure ML Studio** (Easiest):

1. Go to Models → `hdd_failure_predictor` → v2
2. Click "Deploy" → "Real-time endpoint"
3. Configure:
   - Name: `hdd-predictor-endpoint`
   - Compute: Azure Container Instance
   - VM: Standard_DS3_v2
   - Upload scoring script: `src/models/score_azure.py`
4. Deploy (~10 mins)
5. Test in Studio → Endpoints → Test tab

---

## 🧪 Testing & Evaluation

### Automated Test Suite

```bash
# Run all tests
python src/tests/test_model.py

# With pytest (verbose)
pytest src/tests/test_model.py -v
```

### Test Coverage

| Test Category | Tests | Purpose |
|--------------|-------|---------|
| **Data Validation** | 5 | Schema, types, missing values, target distribution |
| **Reproducibility** | 1 | Deterministic predictions with fixed seed |
| **Performance Regression** | 4 | Minimum thresholds for accuracy, F1, precision, recall |
| **Model Artifacts** | 3 | Model file existence, methods, prediction capability |
| **Consistency** | 1 | Identical outputs across runs |

### Performance Thresholds

Minimum acceptable performance (Iteration 1 baseline):
- Accuracy: ≥ 70%
- F1-Score: ≥ 65%
- Precision: ≥ 60%
- Recall: ≥ 60%

Models failing these thresholds are flagged for investigation.

### Evaluation Methods

1. **Holdout Validation**: 80/20 train-test split with stratification
2. **Metrics**: Comprehensive classification metrics (accuracy, precision, recall, F1, ROC-AUC)
3. **Confusion Matrix**: Visual analysis of TP/TN/FP/FN
4. **Feature Importance**: Understanding model decisions (Random Forest)
5. **Comparison**: Side-by-side iteration performance

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Triggered on:
- Push to `main` branch
- Pull requests
- Manual dispatch

### Pipeline Stages

```yaml
1. Test Locally
   ├─ Checkout code
   ├─ Setup Python 3.11
   ├─ Install dependencies
   ├─ Run data validation tests
   └─ Lint code (flake8)

2. Deploy to Azure ML (main branch only)
   ├─ Azure login (service principal)
   ├─ Setup Azure ML SDK
   ├─ Create config from secrets
   ├─ Submit Iteration 1 training
   ├─ Submit Iteration 2 training
   ├─ Register models
   └─ Cleanup secrets
```

### Required GitHub Secrets

1. **AZURE_CREDENTIALS**: Service principal for Azure authentication
2. **AML_CONFIG**: Azure ML workspace configuration JSON

See [GitHub Actions Setup Guide](docs/github-actions-setup.md) for details.

---

## 💻 Local Development

### Initial Experimentation: Docker + MLflow

During development, we used Docker Compose for local MLflow tracking:

```bash
# Start MLflow tracking server
docker-compose up mlflow

# Access UI
http://localhost:5000
```

**Purpose**: Rapid prototyping and understanding MLflow workflow before cloud deployment.

**Files**:
- `Dockerfile`: Containerized model serving
- `docker-compose.yml`: Local MLflow server
- `notebooks/01_model_experimentation.ipynb`: Interactive experimentation

### Migration to Azure ML

**Benefits of Azure ML over local Docker**:
- ✅ Auto-scaling compute (0-N nodes)
- ✅ Managed MLflow tracking (no server maintenance)
- ✅ Enterprise model registry
- ✅ Production-grade endpoints
- ✅ Built-in monitoring and logging
- ✅ CI/CD integration

---

## 📊 Performance & Scalability

### Scalability Evaluation

| Aspect | Local Docker | Azure ML | Scalability Gain |
|--------|-------------|----------|------------------|
| **Compute** | Single machine | Auto-scaling 0-N nodes | ∞ |
| **Data** | Local files | Azure Blob Storage | PB-scale |
| **Training Time** | Fixed (local CPU) | Parallel (cloud GPU/CPU) | N×faster |
| **Model Serving** | Single endpoint | Load-balanced endpoints | High availability |
| **Cost** | Fixed (hardware) | Pay-per-use | Optimized |

### Current Limitations

1. **Free Tier Constraints**:
   - Compute hours limited
   - Storage capped
   - Endpoint quotas

2. **Training Time**:
   - First run: ~10-15 mins (environment build)
   - Subsequent: ~3-5 mins
   - Optimization: Pre-build environment images

3. **Dataset Size**:
   - Current: 8,961 samples
   - Production: Millions of drives
   - Solution: Batch processing, distributed training

4. **Model Complexity**:
   - Current: 3 features
   - Production: 100+ SMART metrics
   - Solution: Feature selection, dimensionality reduction

### Addressing Weaknesses

1. **Cost Management**:
   - Min nodes = 0 (scale to zero)
   - Auto-shutdown compute instances
   - Delete resources when not in use
   - Monitor spending with Azure Cost Management

2. **Performance Optimization**:
   - Hyperparameter tuning with Azure ML Hyperdrive
   - Model compression for faster inference
   - Caching frequently-used predictions

3. **Scalability Improvements**:
   - Implement batch inference pipelines
   - Use Azure ML Pipelines for orchestration
   - Enable distributed training for large datasets

---

## 🎓 Key Learnings

### MLOps Best Practices Demonstrated

1. **Experiment Tracking**: Every run logged with parameters, metrics, and artifacts
2. **Model Versioning**: Clear iteration progression with comparison
3. **Reproducibility**: Fixed random seeds, versioned dependencies
4. **Automated Testing**: CI/CD catches regressions before deployment
5. **Scalable Infrastructure**: Cloud-native design supports growth
6. **Documentation**: Comprehensive README and inline comments

### Real-World Applicability

This MLOps pipeline demonstrates:
- **Production-Ready**: Deployable endpoints with monitoring
- **Maintainable**: Clear code structure, automated tests
- **Scalable**: Cloud infrastructure grows with demand
- **Collaborative**: Git workflow, documented processes
- **Cost-Effective**: Auto-scaling reduces waste

---

## 📚 References

1. Backblaze. (2024). *Hard Drive Stats*. https://www.backblaze.com/b2/hard-drive-test-data.html
2. Microsoft. (2024). *Azure Machine Learning Documentation*. https://docs.microsoft.com/en-us/azure/machine-learning/
3. MLflow. (2024). *MLflow Documentation*. https://mlflow.org/docs/latest/index.html
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
5. Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NIPS.
6. GitHub. (2024). *GitHub Actions Documentation*. https://docs.github.com/en/actions

---

## 👤 Author

**Emmet Murray**
Student ID: B00827883
Ulster University
Course: CW2 - MLOps Implementation

---

## 📄 License

This project is for academic purposes (CW2 submission).

---

## 🙏 Acknowledgments

- Azure for Students subscription
- Ulster University MLOps module teaching materials
- Open-source ML/MLOps community (scikit-learn, MLflow, Azure ML)

