"""
Simple Model Serving Script
============================
Serves the HDD failure prediction model via Flask REST API.
This bypasses MLflow serving issues with sklearn version mismatches.

Usage:
    python scripts/serve_model_simple.py

Then test with:
    python scripts/test_api.py
"""

import pickle
import json
from flask import Flask, request, jsonify
import numpy as np

# Configuration
MODEL_PATH = "models/v2/model.pkl"
PORT = 5001

# Load model
print(f"Loading model from {MODEL_PATH}...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully: {type(model).__name__}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("\nTrying to load from local MLflow run...")
    # Fallback to local MLflow run
    import mlflow
    mlflow.set_tracking_uri("file:///c:/Users/Emmet/cw2-hdd-mlops/mlruns")

    # Get latest run
    experiment = mlflow.get_experiment_by_name("Default")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    latest_run_id = runs.iloc[0]['run_id']

    model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
    print(f"Model loaded from MLflow run: {latest_run_id}")

# Create Flask app
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route('/invocations', methods=['POST'])
def predict():
    """
    Make predictions on input data.

    Expected input format (MLflow style):
    {
        "dataframe_split": {
            "columns": ["capacity_bytes", "lifetime", "model_encoded"],
            "data": [[0.5, 0.3, 17]]
        }
    }

    OR simple format:
    {
        "data": [[0.5, 0.3, 17]]
    }
    """
    try:
        # Parse request
        req_data = request.get_json()

        # Extract data
        if 'dataframe_split' in req_data:
            # MLflow format
            data = req_data['dataframe_split']['data']
        elif 'data' in req_data:
            # Simple format
            data = req_data['data']
        else:
            return jsonify({"error": "Missing 'data' or 'dataframe_split' in request"}), 400

        # Convert to numpy array
        X = np.array(data)

        # Make predictions
        predictions = model.predict(X).tolist()

        # Return MLflow-compatible response
        response = {
            "predictions": predictions
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"STARTING HDD FAILURE PREDICTION MODEL SERVER")
    print(f"{'='*70}")
    print(f"\nModel: {type(model).__name__}")
    print(f"Endpoint: http://127.0.0.1:{PORT}/invocations")
    print(f"Health: http://127.0.0.1:{PORT}/health")
    print(f"\nPress CTRL+C to stop\n")

    app.run(host='127.0.0.1', port=PORT, debug=False)
