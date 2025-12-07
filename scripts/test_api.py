"""
Test MLflow Model API
======================
Test the locally served MLflow model with sample HDD data.

Usage:
    1. Start MLflow server:
       mlflow models serve -m models/v2 -p 5001 --no-conda

    2. Run this test:
       python scripts/test_api.py
"""

import requests
import json
import time

# MLflow serving endpoint
ENDPOINT = "http://127.0.0.1:5001/invocations"

# Test data: capacity_bytes, lifetime, model_encoded
# Based on HDD dataset features used in training
test_samples = [
    {
        "name": "Low Risk - New Drive",
        "data": [[0.1, 0.05, 5]]  # Low capacity usage, low lifetime, model 5
    },
    {
        "name": "Medium Risk - Mid-life Drive",
        "data": [[0.5, 0.3, 17]]  # Medium capacity usage, medium lifetime, model 17
    },
    {
        "name": "High Risk - Old Drive",
        "data": [[0.99, 0.95, 23]]  # High capacity usage, high lifetime, model 23
    }
]


def test_model_api():
    """Test the MLflow model serving endpoint."""
    print("=" * 70)
    print("TESTING MLFLOW MODEL API")
    print("=" * 70)
    print(f"\nEndpoint: {ENDPOINT}")
    print("\nTesting with 3 sample HDD records...\n")

    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:5001/health", timeout=2)
        print(f"Server health check: {response.status_code}")
    except requests.exceptions.RequestException:
        print("\nERROR: MLflow server not running!")
        print("\nPlease start the server first:")
        print("  mlflow models serve -m models/v2 -p 5001 --no-conda")
        return

    # Test each sample
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{'-' * 70}")
        print(f"Test {i}: {sample['name']}")
        print(f"{'-' * 70}")

        # Prepare request
        payload = {
            "dataframe_split": {
                "columns": ["capacity_bytes", "lifetime", "model_encoded"],
                "data": sample["data"]
            }
        }

        print(f"\nInput features:")
        print(f"  Capacity bytes (normalized): {sample['data'][0][0]:.2f}")
        print(f"  Lifetime (normalized):       {sample['data'][0][1]:.2f}")
        print(f"  Model encoded:               {sample['data'][0][2]}")

        # Send request
        try:
            response = requests.post(
                ENDPOINT,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10
            )

            if response.status_code == 200:
                prediction = response.json()
                print(f"\nPrediction result:")
                print(f"  Raw prediction: {prediction}")

                # Interpret prediction
                pred_value = prediction['predictions'][0]
                if pred_value == 0:
                    print(f"  Interpretation: HEALTHY - Drive is operating normally")
                else:
                    print(f"  Interpretation: FAILURE RISK - Drive may fail soon")

            else:
                print(f"\nERROR: HTTP {response.status_code}")
                print(f"Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Request failed")
            print(f"Details: {str(e)}")

    # Summary
    print(f"\n{'=' * 70}")
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Take screenshot of this output")
    print("  2. Take screenshot of MLflow server terminal")
    print("  3. Include in coursework report as evidence of deployment")
    print()


if __name__ == "__main__":
    test_model_api()
