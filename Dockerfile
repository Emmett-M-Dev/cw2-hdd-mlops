# MLflow Server Dockerfile
# Runs MLflow tracking server on port 8080

FROM python:3.11-slim

WORKDIR /app

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.0 \
    scikit-learn==1.3.0 \
    pandas==2.0.0 \
    flask==3.0.0

# Expose MLflow UI port
EXPOSE 8080

# Volume for MLflow tracking data
VOLUME ["/app/mlruns"]

# Start MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--backend-store-uri", "file:///app/mlruns"]
