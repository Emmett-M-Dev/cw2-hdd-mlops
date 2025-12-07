"""
Azure ML Scoring Script for HDD Failure Prediction
===================================================
This script is used by Azure ML to serve the model as a web service.

Input format:
{
    "data": [
        [capacity_bytes, lifetime, model_encoded],
        [capacity_bytes, lifetime, model_encoded],
        ...
    ]
}

Output format:
[
    {
        "prediction": 0 or 1,
        "failure_probability": 0.0-1.0,
        "risk_level": "LOW" | "MEDIUM" | "HIGH"
    },
    ...
]
"""

import json
import joblib
import numpy as np
import os


def init():
    """
    Called when the service is loaded.
    Load the model from the Azure ML model directory.
    """
    global model

    print("Initializing scoring service...")

    # Azure ML sets AZUREML_MODEL_DIR environment variable
    # This points to the directory where the model is stored
    model_path = os.path.join(
        os.getenv('AZUREML_MODEL_DIR', '/var/azureml-app/azureml-models'),
        'hdd_failure_predictor',
        '1',  # version number
        'model'
    )

    # Try to find the model file
    if os.path.exists(model_path):
        # MLflow model structure
        model_file = os.path.join(model_path, 'model.pkl')
    else:
        # Fallback: try direct path
        model_file = os.path.join(os.getenv('AZUREML_MODEL_DIR', '.'), 'model.pkl')

    try:
        model = joblib.load(model_file)
        print(f"✅ Model loaded successfully from: {model_file}")
        print(f"   Model type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"   Tried path: {model_file}")
        print(f"   AZUREML_MODEL_DIR: {os.getenv('AZUREML_MODEL_DIR')}")
        raise


def run(raw_data):
    """
    Called for each prediction request.

    Args:
        raw_data: JSON string containing input data

    Returns:
        JSON string containing predictions with risk levels
    """
    try:
        # Parse input data
        data = json.loads(raw_data)

        # Handle different input formats
        if isinstance(data, dict) and 'data' in data:
            input_data = np.array(data['data'])
        elif isinstance(data, list):
            input_data = np.array(data)
        else:
            return json.dumps({
                'error': 'Invalid input format. Expected {"data": [[features...]]} or [[features...]]'
            })

        # Validate input shape
        if input_data.ndim != 2:
            return json.dumps({
                'error': f'Input must be 2D array. Got shape: {input_data.shape}'
            })

        if input_data.shape[1] != 3:
            return json.dumps({
                'error': f'Expected 3 features [capacity_bytes, lifetime, model_encoded]. Got {input_data.shape[1]}'
            })

        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]  # Probability of failure (class 1)

        # Format results with risk levels
        results = []
        for pred, prob in zip(predictions, probabilities):
            # Categorize risk level based on probability
            if prob > 0.7:
                risk = "HIGH"
            elif prob > 0.3:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            results.append({
                'prediction': int(pred),
                'failure_probability': float(prob),
                'risk_level': risk
            })

        return json.dumps(results)

    except json.JSONDecodeError:
        return json.dumps({'error': 'Invalid JSON format'})

    except Exception as e:
        return json.dumps({'error': f'Prediction error: {str(e)}'})


# For local testing
if __name__ == "__main__":
    # Test data
    test_input = {
        "data": [
            [0.5, 0.3, 17],    # Typical drive
            [0.99, 0.95, 23],  # High risk
            [0.1, 0.05, 5]     # Low risk
        ]
    }

    print("Testing scoring script locally...")
    print(f"\nInput:\n{json.dumps(test_input, indent=2)}")

    # Simulate init
    print("\nCalling init()...")
    try:
        init()
    except:
        print("Note: init() may fail locally without AZUREML_MODEL_DIR set")
        print("This is expected - the script will work in Azure ML deployment\n")

    # Simulate run
    print("\nNote: To test run(), deploy to Azure ML or mock the model")
