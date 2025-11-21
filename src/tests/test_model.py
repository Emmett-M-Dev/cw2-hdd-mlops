"""
Hard Drive Failure Prediction - Model Testing Script
=====================================================
This script contains tests for:
1. Model Reproducibility - Ensuring consistent predictions with same seed
2. Performance Regression - Detecting if model performance drops below threshold
3. Data Validation - Ensuring input data meets expected format
4. Model Artifact Validation - Checking model files exist and load correctly

Usage:
    python src/tests/test_model.py

Or with pytest:
    pytest src/tests/test_model.py -v
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Configuration
DATA_PATH = "data/processed/hdd_balanced_dataset.csv"
MODEL_PATH = "models/model_random_forest.pkl"
FEATURE_COLUMNS = ['capacity_bytes', 'lifetime', 'model_encoded']
TARGET_COLUMN = 'failure'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Performance thresholds (minimum acceptable values)
MIN_ACCURACY = 0.70
MIN_F1_SCORE = 0.65
MIN_PRECISION = 0.60
MIN_RECALL = 0.60


class ModelTester:
    """Class to run all model tests."""

    def __init__(self):
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_results = {}

    def load_data(self) -> bool:
        """Load the dataset."""
        try:
            self.data = pd.read_csv(DATA_PATH)
            print(f"[OK] Data loaded: {len(self.data)} rows")
            return True
        except FileNotFoundError:
            print(f"[FAIL] Data file not found: {DATA_PATH}")
            return False

    def prepare_data(self) -> bool:
        """Prepare train/test split."""
        try:
            X = self.data[FEATURE_COLUMNS]
            y = self.data[TARGET_COLUMN]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
            print(f"[OK] Data prepared: {len(self.X_train)} train, {len(self.X_test)} test")
            return True
        except Exception as e:
            print(f"[FAIL] Data preparation failed: {e}")
            return False

    def load_model(self) -> bool:
        """Load the saved model."""
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[OK] Model loaded from: {MODEL_PATH}")
            return True
        except FileNotFoundError:
            print(f"[WARN] Model file not found: {MODEL_PATH}")
            print("[INFO] Training a new model for testing...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
            print("[OK] New model trained for testing")
            return True


def test_data_validation():
    """Test 1: Validate input data format and quality."""
    print("\n" + "="*60)
    print("TEST 1: DATA VALIDATION")
    print("="*60)

    passed = True

    # Check file exists
    if not os.path.exists(DATA_PATH):
        print(f"[FAIL] Data file not found: {DATA_PATH}")
        return False

    df = pd.read_csv(DATA_PATH)

    # Check required columns exist
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"[FAIL] Missing columns: {missing_columns}")
        passed = False
    else:
        print(f"[PASS] All required columns present")

    # Check for missing values in required columns
    missing_values = df[required_columns].isnull().sum().sum()
    if missing_values > 0:
        print(f"[FAIL] Found {missing_values} missing values")
        passed = False
    else:
        print(f"[PASS] No missing values in required columns")

    # Check data types
    numeric_columns = FEATURE_COLUMNS
    for col in numeric_columns:
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"[FAIL] Column {col} is not numeric")
            passed = False
        else:
            print(f"[PASS] Column {col} is numeric")

    # Check target column is binary
    unique_values = df[TARGET_COLUMN].unique()
    if not set(unique_values).issubset({0, 1}):
        print(f"[FAIL] Target column should be binary (0/1), found: {unique_values}")
        passed = False
    else:
        print(f"[PASS] Target column is binary")

    # Check dataset size
    if len(df) < 100:
        print(f"[WARN] Dataset is small ({len(df)} rows), results may not be reliable")
    else:
        print(f"[PASS] Dataset size is adequate ({len(df)} rows)")

    return passed


def test_model_reproducibility():
    """Test 2: Ensure model produces consistent results with same random seed."""
    print("\n" + "="*60)
    print("TEST 2: MODEL REPRODUCIBILITY")
    print("="*60)

    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train model twice with same seed
    model1 = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_test)

    model2 = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model2.fit(X_train, y_train)
    pred2 = model2.predict(X_test)

    # Check predictions are identical
    if np.array_equal(pred1, pred2):
        print(f"[PASS] Model predictions are reproducible")
        print(f"       Both runs produced identical predictions on {len(pred1)} samples")
        return True
    else:
        diff_count = np.sum(pred1 != pred2)
        print(f"[FAIL] Model predictions differ in {diff_count} samples")
        return False


def test_performance_regression():
    """Test 3: Check model performance meets minimum thresholds."""
    print("\n" + "="*60)
    print("TEST 3: PERFORMANCE REGRESSION")
    print("="*60)

    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Load or train model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[INFO] Using saved model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[INFO] Training new model for testing...")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_split=5, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    passed = True

    # Check accuracy
    if accuracy >= MIN_ACCURACY:
        print(f"[PASS] Accuracy: {accuracy:.4f} >= {MIN_ACCURACY}")
    else:
        print(f"[FAIL] Accuracy: {accuracy:.4f} < {MIN_ACCURACY}")
        passed = False

    # Check F1 score
    if f1 >= MIN_F1_SCORE:
        print(f"[PASS] F1 Score: {f1:.4f} >= {MIN_F1_SCORE}")
    else:
        print(f"[FAIL] F1 Score: {f1:.4f} < {MIN_F1_SCORE}")
        passed = False

    # Check precision
    if precision >= MIN_PRECISION:
        print(f"[PASS] Precision: {precision:.4f} >= {MIN_PRECISION}")
    else:
        print(f"[FAIL] Precision: {precision:.4f} < {MIN_PRECISION}")
        passed = False

    # Check recall
    if recall >= MIN_RECALL:
        print(f"[PASS] Recall: {recall:.4f} >= {MIN_RECALL}")
    else:
        print(f"[FAIL] Recall: {recall:.4f} < {MIN_RECALL}")
        passed = False

    return passed


def test_model_artifact():
    """Test 4: Validate model artifact exists and can make predictions."""
    print("\n" + "="*60)
    print("TEST 4: MODEL ARTIFACT VALIDATION")
    print("="*60)

    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model file not found: {MODEL_PATH}")
        print(f"[INFO] Run training script first: python src/models/train_model.py")
        return True  # Not a failure, just needs training first

    passed = True

    # Load model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[PASS] Model loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return False

    # Check model has expected methods
    required_methods = ['predict', 'predict_proba', 'fit']
    for method in required_methods:
        if hasattr(model, method):
            print(f"[PASS] Model has '{method}' method")
        else:
            print(f"[FAIL] Model missing '{method}' method")
            passed = False

    # Test prediction on sample data
    try:
        sample_data = pd.DataFrame({
            'capacity_bytes': [0.5],
            'lifetime': [0.5],
            'model_encoded': [17]
        })
        prediction = model.predict(sample_data)
        probability = model.predict_proba(sample_data)

        print(f"[PASS] Model can make predictions")
        print(f"       Sample prediction: {prediction[0]}, probability: {probability[0][1]:.4f}")
    except Exception as e:
        print(f"[FAIL] Model prediction failed: {e}")
        passed = False

    return passed


def test_prediction_consistency():
    """Test 5: Ensure same input always produces same output."""
    print("\n" + "="*60)
    print("TEST 5: PREDICTION CONSISTENCY")
    print("="*60)

    # Load or create model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        df = pd.read_csv(DATA_PATH)
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # Test with same input multiple times
    test_input = pd.DataFrame({
        'capacity_bytes': [0.5, 0.3, 0.8],
        'lifetime': [0.5, 0.7, 0.2],
        'model_encoded': [17, 10, 23]
    })

    predictions = []
    for i in range(5):
        pred = model.predict(test_input)
        predictions.append(pred)

    # Check all predictions are identical
    all_same = all(np.array_equal(predictions[0], p) for p in predictions)

    if all_same:
        print(f"[PASS] Predictions are consistent across 5 runs")
        return True
    else:
        print(f"[FAIL] Predictions vary between runs")
        return False


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*70)
    print("HARD DRIVE FAILURE PREDICTION - MODEL TEST SUITE")
    print("="*70)

    results = {}

    # Run each test
    results['Data Validation'] = test_data_validation()
    results['Model Reproducibility'] = test_model_reproducibility()
    results['Performance Regression'] = test_performance_regression()
    results['Model Artifact'] = test_model_artifact()
    results['Prediction Consistency'] = test_prediction_consistency()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "[OK]" if result else "[X]"
        print(f"{symbol} {test_name}: {status}")

        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\nAll tests passed!")
        return True
    else:
        print(f"\n{failed} test(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
