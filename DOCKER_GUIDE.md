# Docker Deployment Guide

## Quick Start (2 Commands)

### Option 1: Run Everything with Docker Compose (Recommended)

```bash
# Start MLflow server (port 8080) and model server (port 5001)
docker-compose up -d

# View logs
docker-compose logs -f
```

**Access**:
- MLflow UI: http://localhost:8080
- Model API: http://localhost:5001/invocations
- Health check: http://localhost:5001/health

**Stop**:
```bash
docker-compose down
```

---

### Option 2: Run MLflow Only (Manual Model Serving)

```bash
# Build and run MLflow container
docker build -t mlflow-server .
docker run -d -p 8080:8080 -v ${PWD}/mlruns:/app/mlruns --name mlflow mlflow-server

# Then run model server locally
python scripts/serve_model_local.py
```

---

## Complete Setup (Step-by-Step)

### Prerequisites

- Docker Desktop installed and running
- Git Bash or PowerShell

### Step 1: Kill Existing Processes

```bash
# Kill the current MLflow server on port 5000 (if running)
# Find PID: netstat -ano | findstr :5000
# Kill: taskkill /PID <PID> /F

# Kill model server on port 5001
# Find PID: netstat -ano | findstr :5001
# Kill: taskkill /PID <PID> /F
```

### Step 2: Start Docker Services

```bash
# Navigate to project
cd c:\Users\Emmet\cw2-hdd-mlops

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

**Expected Output**:
```
NAME                IMAGE               COMMAND                  SERVICE         STATUS              PORTS
mlflow-server       mlflow-server       "mlflow server --hos…"   mlflow          running             0.0.0.0:8080->8080/tcp
model-server        python:3.11-slim    "sh -c 'pip install…"    model-server    running             0.0.0.0:5001->5001/tcp
```

### Step 3: Wait for Services to Start

```bash
# Follow logs (Ctrl+C to exit)
docker-compose logs -f

# Wait for:
# mlflow-server: "Listening at: http://0.0.0.0:8080"
# model-server: "Running on http://0.0.0.0:5001"
```

### Step 4: Verify MLflow UI

1. Open browser: http://localhost:8080
2. You should see MLflow UI
3. Experiment: `hdd_failure_prediction`
4. Check if Iteration 3 appears

### Step 5: Test Model API

```bash
# Run test script (from host, not in container)
python scripts/test_api.py
```

**Expected**:
- Server health check: 200
- 3 predictions returned successfully

---

## Training with Docker MLflow

### Train New Model (Iteration 4)

```bash
# Set environment variable to use Docker MLflow
export MLFLOW_TRACKING_URI=http://localhost:8080

# Or on Windows PowerShell:
$env:MLFLOW_TRACKING_URI = "http://localhost:8080"

# Train model
python scripts/train_local_iteration3.py
```

The model will automatically log to Docker MLflow on port 8080!

---

## What Changed?

### 1. MLflow Server
- **Before**: Python process on localhost:5000
- **After**: Docker container on localhost:8080
- **Benefit**: Containerized, matches lab setup

### 2. Model Server
- **Before**: Python process on localhost:5001
- **After**: Docker container on localhost:5001 (optional)
- **Benefit**: Can deploy both together

### 3. Training Script
- **Before**: Hardcoded `file:///c:/Users/.../mlruns`
- **After**: Uses `MLFLOW_TRACKING_URI` environment variable
- **Benefit**: Flexible - works with Docker or local

---

## Docker Compose Services

### mlflow Service
```yaml
ports: "8080:8080"           # MLflow UI
volumes: ./mlruns:/app/mlruns  # Persist experiments
```

### model-server Service
```yaml
ports: "5001:5001"           # Model API
volumes:
  - ./models/v3:/app/models/v3  # Model artifacts
  - ./scripts:/app/scripts      # Serving script
```

---

## Troubleshooting

### Port Already in Use

**Problem**: Port 8080 or 5001 already in use

**Solution**:
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8080 | xargs kill -9
```

### Container Won't Start

**Check logs**:
```bash
docker-compose logs mlflow
docker-compose logs model-server
```

**Restart**:
```bash
docker-compose down
docker-compose up -d
```

### MLflow Data Not Showing

**Problem**: Iteration 3 doesn't appear in Docker MLflow

**Reason**: Different tracking URIs (file:// vs http://)

**Solution**: Retrain with Docker MLflow:
```bash
export MLFLOW_TRACKING_URI=http://localhost:8080
python scripts/train_local_iteration3.py
```

---

## Architecture Diagrams

### Docker Setup

```
┌─────────────────────────────────────┐
│ Docker Host (Your Machine)         │
│                                     │
│  ┌────────────────────────────────┐│
│  │ mlflow-server Container        ││
│  │                                ││
│  │  MLflow Server                 ││
│  │  Port: 8080                    ││
│  │  Volume: ./mlruns              ││
│  └────────────┬───────────────────┘│
│               │                     │
│  ┌────────────┴───────────────────┐│
│  │ model-server Container         ││
│  │                                ││
│  │  Flask API                     ││
│  │  Port: 5001                    ││
│  │  Volume: ./models/v3           ││
│  └────────────────────────────────┘│
│                                     │
└─────────────────────────────────────┘
         │                │
         ↓                ↓
    localhost:8080   localhost:5001
    (MLflow UI)      (Model API)
```

---

## Benefits for Coursework

### Why Docker is Better

1. **Aligns with Labs**: Labs use Docker on port 8080
2. **Containerization**: Shows DevOps/MLOps best practices
3. **Reproducibility**: Same environment everywhere
4. **Easy Demo**: `docker-compose up` and done
5. **Professional**: Industry-standard deployment

### Presentation Talking Points

> "We containerized the entire stack using Docker. The MLflow tracking server runs in a Docker container on port 8080, matching the lab setup. This demonstrates infrastructure-as-code and ensures reproducibility across environments. A single `docker-compose up` command starts both the MLflow server and model API."

---

## Commands Reference

### Start Services
```bash
docker-compose up -d          # Start in background
docker-compose up             # Start with logs
```

### View Logs
```bash
docker-compose logs -f        # Follow all logs
docker-compose logs mlflow    # MLflow only
docker-compose logs model-server  # Model server only
```

### Stop Services
```bash
docker-compose stop           # Stop (keep data)
docker-compose down           # Stop and remove containers
docker-compose down -v        # Stop and remove volumes (DELETES DATA!)
```

### Rebuild
```bash
docker-compose build          # Rebuild images
docker-compose up -d --build  # Rebuild and start
```

### Status
```bash
docker-compose ps             # Show running services
docker stats                  # Show resource usage
```

---

## Environment Variables

### For Training Scripts

**Windows PowerShell**:
```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:8080"
python scripts/train_local_iteration3.py
```

**Linux/Mac/Git Bash**:
```bash
export MLFLOW_TRACKING_URI=http://localhost:8080
python scripts/train_local_iteration3.py
```

### Permanent (Optional)

Create `.env` file:
```bash
MLFLOW_TRACKING_URI=http://localhost:8080
```

Docker Compose will automatically load it.

---

## Next Steps

1. ✅ Stop current MLflow process (port 5000)
2. ✅ Run `docker-compose up -d`
3. ✅ Access http://localhost:8080
4. ✅ Test API: `python scripts/test_api.py`
5. ✅ Take screenshots for presentation
6. ✅ Update slides to show Docker deployment

**You now have a production-ready containerized MLOps stack!** 🐳
