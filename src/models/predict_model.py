"""
Hard Drive Failure Prediction - Prediction Script
==================================================
This script loads a trained model and makes predictions on new data.
Can load models from local pickle files or MLflow Model Registry.

Usage:
    python src/models/predict_model.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient


# Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
MODEL_NAME = "hdd_failure_predictor"
LOCAL_MODEL_PATH = "models/model_random_forest.pkl"
FEATURE_COLUMNS = ['capacity_bytes', 'lifetime', 'model_encoded']


def load_model_from_pickle(model_path: str):
    """Load model from local pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded from: {model_path}")
    return model


def load_model_from_mlflow(model_name: str, stage: str = "Staging"):
    """Load model from MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"Model loaded from MLflow: {model_name} ({stage})")
    return model


def predict(model, data: pd.DataFrame) -> np.ndarray:
    """Make predictions using the loaded model."""
    # Ensure we only use the feature columns
    X = data[FEATURE_COLUMNS]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


def predict_single(model, capacity_bytes: float, lifetime: float, model_encoded: int) -> dict:
    """Make prediction for a single hard drive."""
    input_data = pd.DataFrame({
        'capacity_bytes': [capacity_bytes],
        'lifetime': [lifetime],
        'model_encoded': [model_encoded]
    })

    prediction, probability = predict(model, input_data)

    result = {
        'prediction': int(prediction[0]),
        'probability_failure': float(probability[0]),
        'risk_level': 'HIGH' if probability[0] > 0.7 else ('MEDIUM' if probability[0] > 0.3 else 'LOW'),
        'interpretation': 'Failure predicted' if prediction[0] == 1 else 'No failure predicted'
    }

    return result


def batch_predict(model, data: pd.DataFrame) -> pd.DataFrame:
    """Make predictions for a batch of hard drives."""
    predictions, probabilities = predict(model, data)

    results = data.copy()
    results['predicted_failure'] = predictions
    results['failure_probability'] = probabilities
    results['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )

    return results


def main():
    """Main prediction demonstration."""
    print("="*60)
    print("HDD FAILURE PREDICTION - INFERENCE")
    print("="*60)

    # Try to load model from MLflow first, fall back to local pickle
    try:
        print("\n--- Loading Model from MLflow ---")
        model = load_model_from_mlflow(MODEL_NAME, stage="Staging")
    except Exception as e:
        print(f"Could not load from MLflow: {e}")
        print("\n--- Loading Model from Local File ---")
        model = load_model_from_pickle(LOCAL_MODEL_PATH)

    # Example single prediction
    print("\n" + "="*60)
    print("SINGLE PREDICTION EXAMPLE")
    print("="*60)

    # Example input (using normalized values from the dataset)
    example_input = {
        'capacity_bytes': 0.5,  # Normalized capacity
        'lifetime': 0.5,        # Normalized lifetime
        'model_encoded': 17     # Model encoding from CW1
    }

    result = predict_single(
        model,
        capacity_bytes=example_input['capacity_bytes'],
        lifetime=example_input['lifetime'],
        model_encoded=example_input['model_encoded']
    )

    print(f"\nInput:")
    for key, value in example_input.items():
        print(f"  {key}: {value}")

    print(f"\nPrediction:")
    print(f"  Failure: {result['interpretation']}")
    print(f"  Probability: {result['probability_failure']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")

    # Batch prediction example
    print("\n" + "="*60)
    print("BATCH PREDICTION EXAMPLE")
    print("="*60)

    # Load some test data
    test_data_path = "data/processed/cleaned_hdd_from_faulty.csv"
    if os.path.exists(test_data_path):
        df = pd.read_csv(test_data_path)
        sample_data = df[FEATURE_COLUMNS].head(5)

        results = batch_predict(model, sample_data)

        print("\nBatch predictions (first 5 records):")
        print(results[['capacity_bytes', 'lifetime', 'model_encoded',
                       'predicted_failure', 'failure_probability', 'risk_level']].to_string())

        # Summary statistics
        print("\n--- Prediction Summary ---")
        print(f"Total records: {len(results)}")
        print(f"Predicted failures: {results['predicted_failure'].sum()}")
        print(f"High risk: {(results['risk_level'] == 'HIGH').sum()}")
        print(f"Medium risk: {(results['risk_level'] == 'MEDIUM').sum()}")
        print(f"Low risk: {(results['risk_level'] == 'LOW').sum()}")
    else:
        print(f"Test data not found at: {test_data_path}")

    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
