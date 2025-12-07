# HDD Failure Prediction MLOps Project - Presentation Narrative

**Student Name**: [Your Name]
**Student Number**: [Your Number]
**Project**: Hard Drive Failure Prediction with Azure ML and MLOps

---

## SLIDE 0: Title Slide

**Title**: HDD Failure Prediction: End-to-End MLOps Pipeline

**One-line Description**: Predicting hard drive failures using machine learning with Azure ML, MLflow tracking, and automated CI/CD deployment.

**Visual**: Project logo or architecture diagram thumbnail

---

## SLIDES 1-2: Problem Discussion & MLOps Considerations

### Slide 1: The Problem

**Business Context**:
- Data centers operate millions of hard drives storing critical data
- Unexpected drive failures cause:
  - Data loss and service disruptions
  - High replacement costs
  - Operational overhead in reactive maintenance

**Current Approach**: Reactive replacement after failure occurs

**Proposed Solution**: Predictive maintenance using machine learning to identify at-risk drives before failure

**Dataset**: Backblaze HDD SMART statistics
- 8,961 samples (balanced 50/50 failure/healthy)
- 3 key features: capacity utilization, drive age, model type
- Binary classification: 0 = healthy, 1 = will fail

**Video Script**:
> "Hard drive failures in data centers cost companies millions annually. Instead of waiting for drives to fail, we can predict failures in advance using SMART sensor data. This project builds a complete MLOps pipeline to train, deploy, and serve a machine learning model that predicts HDD failures with 98% accuracy."

---

### Slide 2: MLOps Considerations

**Why MLOps is Critical for This Problem**:

1. **Scale**: Data centers have millions of drives generating continuous telemetry
   - Cannot manually retrain models
   - Need automated pipelines

2. **Model Evolution**: Drive populations change over time
   - New drive models introduced
   - Degradation patterns evolve
   - Regular retraining required

3. **Reproducibility**: Must track which model version made which prediction
   - For auditing and compliance
   - To roll back if performance degrades

4. **Deployment**: Models must be deployed reliably across environments
   - Development → Testing → Production
   - Consistent behavior required

5. **Monitoring**: Model performance degrades over time (concept drift)
   - Need to detect when retraining is needed
   - Track prediction accuracy in production

**Key MLOps Challenges Addressed**:
- ✅ Experiment tracking (MLflow)
- ✅ Model versioning (Azure ML Model Registry)
- ✅ Automated training (GitHub Actions CI/CD)
- ✅ Environment reproducibility (Docker, conda)
- ✅ Scalable compute (Azure ML clusters)
- ✅ Deployment (Flask REST API)

**Video Script**:
> "This isn't just a machine learning problem—it's an MLOps problem. We need automated training pipelines, version-controlled models, and reliable deployment. Our solution uses Azure ML for scalable cloud training, MLflow for experiment tracking, and GitHub Actions for CI/CD automation."

---

## SLIDES 3-6: Solution Design & Development Process

### Slide 3: Architecture Overview

**System Architecture** (show diagram):

```
┌─────────────────┐
│ Local Dev       │
│ (VS Code)       │
└────────┬────────┘
         │ git push
         ↓
┌─────────────────┐
│ GitHub Repo     │
│                 │
└────────┬────────┘
         │ trigger
         ↓
┌─────────────────┐      ┌──────────────────┐
│ GitHub Actions  │──────→ Azure Login       │
│ CI/CD Pipeline  │      │ (Service Principal)│
└────────┬────────┘      └──────────────────┘
         │ submit jobs
         ↓
┌─────────────────────────────────────────┐
│ Azure ML Workspace                      │
│                                         │
│  ┌──────────────┐    ┌───────────────┐ │
│  │ Compute      │    │ MLflow        │ │
│  │ Cluster      │───→│ Tracking      │ │
│  │ (training)   │    │               │ │
│  └──────────────┘    └───────────────┘ │
│         │                               │
│         ↓                               │
│  ┌──────────────┐                      │
│  │ Model        │                      │
│  │ Registry     │                      │
│  └──────────────┘                      │
└─────────────────────────────────────────┘
         │ download
         ↓
┌─────────────────┐
│ Local Deployment│
│ (Flask API)     │
│ Port 5001       │
└─────────────────┘
```

**Key Components**:
1. **Local Development**: VS Code, Git
2. **CI/CD**: GitHub Actions (automated job submission)
3. **Cloud Training**: Azure ML (scalable compute)
4. **Experiment Tracking**: MLflow (metrics, parameters, artifacts)
5. **Model Registry**: Azure ML (versioned model storage)
6. **Deployment**: Flask REST API (local demonstration)

**Video Script**:
> "Our architecture follows MLOps best practices. Code lives in GitHub, triggering automated Azure ML training jobs via GitHub Actions. Models are tracked in MLflow, registered in Azure ML, and deployed via REST API."

---

### Slide 4: Development Process - Iterative Approach

**3 Model Iterations**:

| Iteration | Algorithm | Purpose | Environment | Accuracy |
|-----------|-----------|---------|-------------|----------|
| 1 | Logistic Regression | Baseline | Azure ML | 74.1% |
| 2 | Random Forest | Improved | Azure ML | 98.1% |
| 3 | Random Forest | Deployment Demo | Local | 98.1% |

**Why 3 Iterations?**
- **Iteration 1**: Establish baseline with simple model
- **Iteration 2**: Improve with ensemble method
- **Iteration 3**: Solve environment compatibility for deployment

**Hyperparameters** (Iteration 2 & 3):
- `n_estimators=100` (number of trees)
- `max_depth=10` (tree depth)
- `min_samples_split=5` (minimum samples to split)
- `random_state=42` (reproducibility)

**Video Script**:
> "We followed an iterative development process. Iteration 1 established a 74% accuracy baseline with Logistic Regression. Iteration 2 improved to 98% using Random Forest. When we encountered deployment issues with sklearn version mismatches, we trained Iteration 3 locally with identical hyperparameters to enable demonstration."

---

### Slide 5: Cloud Training Pipeline (Azure ML)

**Training Script**: `src/model.py`
- Single script handles both iterations via `--iteration` parameter
- Logs metrics automatically to MLflow
- Saves model artifacts to Azure ML

**Job Configuration**: `job.yaml`
```yaml
command: python src/model.py --iteration ${{inputs.iteration}}
compute: training-cluster
experiment_name: hdd_failure_prediction
environment:
  image: mcr.microsoft.com/azureml/curated/sklearn-1.0-ubuntu20.04-py38-cpu:latest
```

**Key Design Decisions**:
1. **Use curated Docker image**: Faster startup, no custom environment build
2. **Parameterized script**: One script for all iterations
3. **Auto-scaling compute**: 0-1 nodes (cost optimization)

**Results** (from Azure ML):
- ✅ 4 successful training jobs completed
- ✅ All metrics logged to MLflow
- ✅ Models saved to Azure ML storage

**Video Script**:
> "Training runs on Azure ML compute clusters using a curated sklearn Docker image. Our job.yaml file defines the training command, and GitHub Actions submits jobs automatically. The cluster auto-scales from 0 to 1 nodes, minimizing costs."

---

### Slide 6: CI/CD Automation (GitHub Actions)

**Workflow**: `.github/workflows/train.yml`

**Trigger**: Push to `main` branch

**Steps**:
1. Checkout code
2. Azure login (Service Principal)
3. Install Azure ML CLI
4. Submit Iteration 1 job
5. Submit Iteration 2 job

**Benefits**:
- ✅ Automated retraining on every code change
- ✅ No manual job submission required
- ✅ Reproducible training runs
- ✅ Secure authentication (GitHub Secrets)

**Evidence**: Show GitHub Actions workflow run

**Video Script**:
> "Every push to main triggers GitHub Actions, which automatically submits training jobs to Azure ML. This ensures reproducibility and eliminates manual errors. Authentication uses Azure Service Principal stored in GitHub Secrets."

---

## SLIDES 7-8: Key Features (Testing & Evaluation)

### Slide 7: Model Testing Framework

**Test Suite**: `src/tests/test_model.py`

**Test Categories**:

1. **Data Validation** (5 tests):
   - Schema correctness
   - No missing values
   - Correct data types
   - Target distribution (50/50 balance)
   - Feature ranges

2. **Reproducibility** (1 test):
   - Deterministic predictions with fixed random seed
   - Same input → Same output (always)

3. **Performance Regression** (4 tests):
   - Accuracy ≥ 70% (baseline threshold)
   - F1-Score ≥ 65%
   - Precision ≥ 60%
   - Recall ≥ 60%

4. **Model Artifacts** (3 tests):
   - Model file exists
   - Model has `predict()` method
   - Model produces valid predictions

**Test Execution**:
```bash
pytest src/tests/test_model.py -v
```

**Why Testing Matters**:
- Catch data quality issues early
- Prevent model performance regressions
- Ensure deployment readiness

**Video Script**:
> "Our testing framework validates data quality, ensures reproducibility, and prevents performance regressions. Before any model is deployed, it must pass all tests including minimum accuracy thresholds."

---

### Slide 8: Evaluation Methods & Results

**Metrics Tracked** (for all iterations):
- **Accuracy**: Overall correctness
- **Precision**: Of predicted failures, how many were correct?
- **Recall**: Of actual failures, how many did we catch? (CRITICAL)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Discrimination ability

**Why Recall is Critical**:
> In predictive maintenance, missing a failure (false negative) is MUCH worse than a false alarm (false positive). We prioritize recall.

**Results Comparison**:

| Metric | Iteration 1 | Iteration 2/3 | Improvement |
|--------|-------------|---------------|-------------|
| **Accuracy** | 74.1% | 98.1% | +24.0pp |
| **Precision** | 69.2% | 99.1% | +29.9pp |
| **Recall** | 35.6% | 95.2% | +59.6pp |
| **F1-Score** | 48.8% | 97.1% | +48.3pp |
| **ROC-AUC** | 69.7% | 99.9% | +30.2pp |

**Key Achievement**: **95.2% recall** means we catch 95% of failures before they happen!

**Evaluation Process**:
1. Train/test split: 80/20 (stratified)
2. Fixed random seed: 42 (reproducibility)
3. Metrics logged automatically to MLflow
4. Comparison charts in MLflow UI

**Video Script**:
> "Random Forest achieves 98% accuracy and critically, 95% recall—catching 95 out of 100 failures before they happen. All metrics are logged to MLflow, enabling easy comparison between iterations."

---

## SLIDES 9-10: Limitations & Scalability

### Slide 9: Limitations & Challenges

**Current Limitations**:

1. **Dataset Size**: Only 8,961 samples
   - Real data centers have millions of drives
   - More data would improve generalization

2. **Feature Count**: Only 3 features
   - SMART data has 50+ attributes
   - More features could improve accuracy further

3. **Azure ML Deployment Failed**: HTTP 502 errors
   - MLflow auto-generated container failed liveness probes
   - Known Azure ML limitation with sklearn pickle versions
   - Pivoted to local Flask deployment for demonstration

4. **Environment Compatibility**:
   - Azure ML: Python 3.8 + sklearn 1.0.2
   - Local: Python 3.11 + sklearn 1.3.0
   - Pickle format incompatible
   - **Solution**: Retrained Iteration 3 locally

5. **No Production Monitoring**: Current deployment lacks:
   - Model drift detection
   - Prediction logging
   - Performance alerting

6. **Single Region**: Deployed only in UK South
   - No multi-region redundancy
   - Higher latency for global users

**What We Learned**:
- ✅ Version pinning is critical in MLOps
- ✅ Environment reproducibility prevents deployment issues
- ✅ Cloud PaaS has limitations—flexibility matters
- ✅ Local MLflow serving provides reliable fallback

**Video Script**:
> "We encountered real-world MLOps challenges. Azure ML's managed endpoint failed due to sklearn version mismatches between training and serving environments. This taught us the importance of dependency versioning. We solved it by retraining locally, demonstrating adaptability."

---

### Slide 10: Scalability Evaluation

**Current Setup**:
- **Training**: Azure ML compute cluster
  - VM: Standard_DS3_v2 (4 vCPUs, 14GB RAM)
  - Nodes: 0-1 (auto-scaling)
  - Dataset: 8,961 rows (~1MB)
  - Training time: ~3 minutes

- **Serving**: Flask development server
  - Single-threaded
  - No load balancing
  - Localhost only

**Production Scaling Path**:

| Component | Current | Production Scale |
|-----------|---------|------------------|
| **Training Data** | 8,961 rows | Millions of drive records |
| **Compute Nodes** | 0-1 | 4-8 nodes (distributed training) |
| **Model Serving** | Flask dev server | Containerized (Docker + Kubernetes) |
| **Load Balancing** | None | Azure Load Balancer |
| **Auto-scaling** | Manual compute | Horizontal pod autoscaling |
| **Monitoring** | Manual MLflow check | Azure Application Insights |
| **Data Pipeline** | Manual CSV | Azure Data Factory (automated) |
| **Retraining** | Manual trigger | Scheduled (weekly/monthly) |

**Scaling Strategies**:

1. **Data Ingestion**:
   - Current: Static CSV file
   - Scaled: Azure Blob Storage + Data Factory
   - Stream SMART data from drives continuously

2. **Training**:
   - Current: Single-node training
   - Scaled: Distributed training across 4-8 nodes
   - Hyperparameter tuning with Azure HyperDrive

3. **Deployment**:
   - Current: Local Flask server
   - Scaled: Docker container → Azure Kubernetes Service (AKS)
   - Auto-scaling based on request volume

4. **Model Registry**:
   - Current: Azure ML (already scales)
   - Supports unlimited model versions

5. **Monitoring**:
   - Current: Manual checks
   - Scaled: Azure Application Insights
   - Alerts on accuracy drop or prediction drift

**Cost Considerations**:
- ✅ Auto-scaling compute: Only pay when training
- ✅ Serverless options: Azure Functions for inference
- ✅ Spot instances: 80% cost reduction for batch inference

**Video Script**:
> "Our current setup handles thousands of predictions. For production scale with millions of drives, we'd deploy to Azure Kubernetes Service with auto-scaling, implement distributed training across multiple nodes, and add continuous monitoring with Application Insights. The architecture is designed to scale."

---

## SLIDE 11: Recorded Demonstration

**Demo Script** (5 minutes total):

### Part 1: Azure ML Training (1.5 min)

1. **Show GitHub Repository** (15 sec)
   - Navigate to `.github/workflows/train.yml`
   - Highlight job submission code

2. **Show GitHub Actions** (30 sec)
   - Navigate to Actions tab
   - Show successful workflow run
   - Point out Iteration 1 and Iteration 2 steps

3. **Show Azure ML Studio** (45 sec)
   - Navigate to Experiments → `hdd_failure_prediction`
   - Show 4 completed runs
   - Click into Iteration 2 run
   - Show metrics: accuracy 98.1%, F1 97.1%

---

### Part 2: Model Registry (0.5 min)

4. **Show Model Registry** (30 sec)
   - Navigate to Models → `hdd_failure_predictor`
   - Show v1 (Logistic Regression) and v2 (Random Forest)
   - Show tags: iteration, algorithm, framework

---

### Part 3: MLflow Experiment Tracking (1 min)

5. **Show MLflow UI** (30 sec)
   - Navigate to http://127.0.0.1:5000
   - Show all runs (Iterations 1, 2, and 3)
   - Compare metrics across runs

6. **Show Iteration 3 Details** (30 sec)
   - Click into Iteration 3 run
   - Show parameters: n_estimators, max_depth
   - Show metrics: 98.1% accuracy
   - Show artifacts: model.pkl saved

---

### Part 4: Local Deployment & Testing (1.5 min)

7. **Start Model Server** (30 sec)
   - Open Terminal 1
   - Run: `C:\Users\Emmet\anaconda3\python.exe scripts\serve_model_local.py`
   - Show output: "Running on http://127.0.0.1:5001"

8. **Test API** (45 sec)
   - Open Terminal 2
   - Run: `C:\Users\Emmet\anaconda3\python.exe scripts\test_api.py`
   - Show 3 test cases:
     - Low Risk → HEALTHY (0)
     - Medium Risk → HEALTHY (0)
     - High Risk → FAILURE RISK (1)
   - Highlight: API returns correct predictions

9. **Show Code** (15 sec)
   - Open `scripts/serve_model_local.py`
   - Highlight Flask `/invocations` endpoint

---

### Part 5: Testing Framework (0.5 min)

10. **Run Tests** (30 sec)
    - Run: `pytest src/tests/test_model.py -v`
    - Show: All tests passing (green checkmarks)
    - Highlight: Performance regression tests

---

**Demonstration Talking Points**:
> "Let me walk you through the complete pipeline. [GitHub Actions] Code pushes trigger automated training jobs. [Azure ML] Jobs run on scalable compute clusters. [MLflow] All experiments are tracked with metrics and parameters. [Model Registry] Models are versioned for auditability. [Deployment] The final model serves predictions via REST API. [Testing] Automated tests ensure quality before deployment."

---

## SLIDE 12: Conclusions & References

### Conclusions

**What We Built**:
- ✅ End-to-end MLOps pipeline for HDD failure prediction
- ✅ 98.1% accuracy, 95.2% recall (catches 95% of failures)
- ✅ Automated CI/CD with GitHub Actions
- ✅ Cloud-based training on Azure ML
- ✅ Experiment tracking with MLflow
- ✅ Model versioning in Azure ML Registry
- ✅ Working REST API deployment

**Key Achievements**:
1. **Automation**: No manual job submission—CI/CD handles all training
2. **Reproducibility**: Fixed seeds, versioned code, tracked experiments
3. **Scalability**: Cloud compute clusters ready to scale to millions of records
4. **Testing**: Comprehensive test suite prevents regressions
5. **Problem-Solving**: Overcame environment compatibility issues through local retraining

**MLOps Principles Demonstrated**:
- ✅ Version control (Git)
- ✅ Experiment tracking (MLflow)
- ✅ Model registry (Azure ML)
- ✅ CI/CD automation (GitHub Actions)
- ✅ Containerization (Docker - curated images)
- ✅ Scalable compute (Azure ML clusters)
- ✅ Testing framework (pytest)
- ✅ Deployment (REST API)

**Lessons Learned**:
1. **Dependency versioning is critical**: sklearn 1.0.2 vs 1.3.0 broke deployment
2. **Cloud PaaS has limitations**: Azure ML endpoint failed, flexibility saved us
3. **Iterative development works**: Start simple (Logistic), improve (Random Forest)
4. **Testing catches issues early**: Data validation prevented bad model training

**Future Work**:
1. Fix Azure ML managed endpoint deployment
2. Implement model monitoring and drift detection
3. Add more SMART features (50+ attributes available)
4. Scale to larger dataset (millions of drives)
5. Implement A/B testing framework
6. Add explainability (SHAP values for predictions)
7. Multi-region deployment for global coverage

**Impact**:
> "This system could prevent data loss in production data centers by predicting 95% of drive failures before they occur, enabling proactive replacement and minimizing downtime."

---

### References

1. **Backblaze Hard Drive Stats**
   Backblaze. (2024). *Hard Drive Test Data*.
   https://www.backblaze.com/b2/hard-drive-test-data.html

2. **Azure Machine Learning Documentation**
   Microsoft. (2024). *Azure Machine Learning*.
   https://docs.microsoft.com/azure/machine-learning/

3. **MLflow Documentation**
   MLflow. (2024). *MLflow: A Platform for ML Development and Productionization*.
   https://mlflow.org/docs/latest/

4. **scikit-learn User Guide**
   scikit-learn developers. (2024). *scikit-learn: Machine Learning in Python*.
   https://scikit-learn.org/stable/user_guide.html

5. **GitHub Actions Documentation**
   GitHub. (2024). *Automating your workflow with GitHub Actions*.
   https://docs.github.com/actions

6. **SMART Attribute Reference**
   Wikipedia. (2024). *S.M.A.R.T. - Self-Monitoring, Analysis and Reporting Technology*.
   https://en.wikipedia.org/wiki/S.M.A.R.T.

7. **MLOps Best Practices**
   Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media.

8. **Random Forest Algorithm**
   Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.

---

## VIDEO RECORDING CHECKLIST

### Before Recording:
- [ ] Restart model server: `python scripts/serve_model_local.py`
- [ ] Verify MLflow UI accessible: http://127.0.0.1:5000
- [ ] Test API working: `python scripts/test_api.py`
- [ ] Open all browser tabs:
  - GitHub repo
  - GitHub Actions
  - Azure ML Studio (Experiments)
  - Azure ML Studio (Models)
  - MLflow UI
- [ ] Close unnecessary applications (clean desktop)
- [ ] Set screen resolution to 1920x1080
- [ ] Test microphone audio levels

### Recording Order:
1. Record screen with narration (5 minutes)
2. Follow demo script exactly
3. Speak clearly and pace yourself
4. Pause briefly between sections
5. If you make a mistake, pause, then re-do that section

### After Recording:
- [ ] Watch full video for errors
- [ ] Check audio quality
- [ ] Verify all demonstrations visible
- [ ] Export in high quality (1080p)
- [ ] Upload to required platform

---

## SLIDE DESIGN TIPS

### Visual Hierarchy:
- **Slide 0**: Large title, your name prominent
- **Slides 1-2**: Problem images (failed HDD, data center), bullet points
- **Slides 3-6**: Architecture diagrams, code snippets
- **Slides 7-8**: Tables (test results, metrics comparison)
- **Slides 9-10**: Charts (current vs scaled), limitation bullets
- **Slide 11**: "See Video Demonstration" with QR code or link
- **Slide 12**: Summary bullets, reference list (small font)

### Color Scheme:
- Azure blue (#0078D4) for Azure ML components
- Orange (#FF6F00) for MLflow
- Green (#28A745) for success metrics
- Red (#DC3545) for limitations
- Gray (#6C757D) for neutral content

### Fonts:
- Titles: Bold, 36-44pt
- Body: Regular, 18-24pt
- Code: Monospace, 14-16pt
- References: 10-12pt

---

**YOU NOW HAVE EVERYTHING YOU NEED FOR YOUR PRESENTATION!**

This narrative covers all 12 slides with:
- ✅ Clear problem statement
- ✅ MLOps justification
- ✅ Architecture explanation
- ✅ Development process
- ✅ Testing & evaluation
- ✅ Limitations & scalability
- ✅ Demo script (5 minutes)
- ✅ Conclusions & references

**Next Steps**:
1. Copy sections into PowerPoint slides
2. Add visuals (screenshots, diagrams, charts)
3. Practice narration
4. Record 5-minute video
5. Submit!
