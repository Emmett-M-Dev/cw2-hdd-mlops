# Docker Migration Summary

## What Changed

### Before (Local Python Processes)
- MLflow server: `python -m mlflow server --port 5000`
- Model server: `python scripts/serve_model_local.py`
- **Port 5000** for MLflow (non-standard)

### After (Docker Containers)
- MLflow server: Docker container on **port 8080**
- Model server: Docker container on **port 5001**
- One command: `docker-compose up -d`

---

## Why Docker on Port 8080?

### Aligns with Lab Setup
- Labs use MLflow on port 8080
- Professional standard
- Matches course materials

### Benefits
1. **Containerization**: Shows DevOps/MLOps skills
2. **Reproducibility**: Same environment everywhere
3. **Easy Deployment**: One command starts everything
4. **Professional**: Industry-standard practice

---

## What You Need to Do

### Step 1: Stop Current Processes

Kill the MLflow server currently running on port 5000:

**Windows**:
```bash
# Find process
netstat -ano | findstr :5000

# Kill it
taskkill /PID <PID> /F
```

**Optional**: Also kill model server on 5001 if using Docker for both

---

### Step 2: Start Docker Services

```bash
cd c:\Users\Emmet\cw2-hdd-mlops
docker-compose up -d
```

Wait ~30 seconds for containers to start.

---

### Step 3: Verify Services

**Check MLflow UI**: http://localhost:8080

**Check Model API**:
```bash
python scripts/test_api.py
```

---

## Files Created

| File | Purpose |
|------|---------|
| `Dockerfile` | MLflow server container |
| `docker-compose.yml` | Multi-service orchestration |
| `.dockerignore` | Efficient Docker builds |
| `DOCKER_GUIDE.md` | Complete Docker documentation |

---

## Training Script Changes

### Before
```python
mlflow.set_tracking_uri("file:///c:/Users/Emmet/cw2-hdd-mlops/mlruns")
```

### After
```python
import os
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
mlflow.set_tracking_uri(mlflow_uri)
```

**Benefit**: Flexible - works with Docker or local file storage

---

## For Your Presentation

### Updated Talking Points

**Deployment Section**:
> "We containerized the entire MLOps stack using Docker and Docker Compose. The MLflow tracking server runs on port 8080, matching the lab setup. Model serving runs on port 5001. A single `docker-compose up` command starts both services, demonstrating infrastructure-as-code and ensuring reproducibility."

**Architecture Slide**: Add Docker logos/icons to the diagram

**Demo**: Show `docker-compose up -d` command

---

## Troubleshooting

### Port 8080 Already in Use

**Check**:
```bash
netstat -ano | findstr :8080
```

**Kill process**:
```bash
taskkill /PID <PID> /F
```

### Docker Not Running

Start Docker Desktop first, then run `docker-compose up -d`

### Iteration 3 Not Showing in Docker MLflow

**Reason**: It was logged to local file storage, not Docker MLflow

**Solution**: Retrain with Docker:
```bash
$env:MLFLOW_TRACKING_URI = "http://localhost:8080"
python scripts/train_local_iteration3.py
```

This will create Iteration 4 in Docker MLflow (same model, logged to containerized server).

---

## Commands Quick Reference

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Check status
docker-compose ps
```

---

## What Stays the Same

- ✅ Azure ML training (Iterations 1 & 2)
- ✅ GitHub Actions CI/CD
- ✅ Model Registry in Azure
- ✅ Test API script
- ✅ All performance metrics

**Only change**: MLflow UI moves from localhost:5000 → localhost:8080 (Docker)

---

## Academic Justification

### Why This Improves Your Project

1. **Containerization Best Practice**: Docker is industry standard
2. **Lab Alignment**: Matches course port configuration
3. **Reproducibility**: Container ensures same environment
4. **Scalability**: Can deploy to Kubernetes easily
5. **Professional**: Shows real-world MLOps skills

### For Report

> "To ensure reproducibility and align with course lab configurations, we containerized the MLflow tracking server using Docker. This demonstrates infrastructure-as-code principles and facilitates deployment across different environments. The containerized setup runs on port 8080, matching the standard lab configuration."

---

## Next Steps

1. ✅ Files already committed to GitHub
2. Run `docker-compose up -d`
3. Access http://localhost:8080
4. Update presentation slides (port 5000 → 8080)
5. Take new screenshots showing Docker setup
6. Update video demo to show `docker-compose up`

**Your setup is now fully containerized and production-ready!** 🐳
