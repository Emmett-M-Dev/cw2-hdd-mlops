"""
Local Model Serving
===================
Serves Iteration 3 model via Flask REST API.

Usage:
    python scripts/serve_model_local.py
"""

import pickle
from flask import Flask, request, jsonify
import numpy as np

# Load model
print("Loading Iteration 3 model...")
with open("models/v3/model.pkl", 'rb') as f:
    model = pickle.load(f)
print(f"Model loaded: {type(model).__name__}")

# Create Flask app
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": type(model).__name__}), 200

@app.route('/invocations', methods=['POST'])
def predict():
    """Make predictions (MLflow compatible format)."""
    try:
        req_data = request.get_json()

        # Extract data (support both formats)
        if 'dataframe_split' in req_data:
            data = req_data['dataframe_split']['data']
        elif 'data' in req_data:
            data = req_data['data']
        else:
            return jsonify({"error": "Missing 'data' or 'dataframe_split'"}), 400

        # Predict
        X = np.array(data)
        predictions = model.predict(X).tolist()

        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("HDD FAILURE PREDICTION MODEL SERVER (ITERATION 3)")
    print("="*70)
    print(f"\nEndpoint: http://127.0.0.1:5001/invocations")
    print(f"Health: http://127.0.0.1:5001/health")
    print(f"\nTest with: python scripts/test_api.py")
    print(f"\nPress CTRL+C to stop\n")

    app.run(host='127.0.0.1', port=5001, debug=False)
