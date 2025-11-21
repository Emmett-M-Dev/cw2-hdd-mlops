"""
Hard Drive Failure Prediction - Production Training Script
===========================================================
This script trains the Random Forest model for HDD failure prediction
and logs all experiments to MLflow for tracking and versioning.

Usage:
    python src/models/train_model.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


# Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
EXPERIMENT_NAME = "hdd_failure_prediction"
DATA_PATH = "data/processed/hdd_balanced_dataset.csv"
MODEL_SAVE_PATH = "models/"
FEATURE_COLUMNS = ['capacity_bytes', 'lifetime', 'model_encoded']
TARGET_COLUMN = 'failure'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(data_path: str) -> pd.DataFrame:
    """Load and return the dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    return df


def prepare_data(df: pd.DataFrame):
    """Prepare features and target, split into train/test sets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    return metrics


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model (Iteration 1 - Baseline)."""
    print("\n" + "="*60)
    print("ITERATION 1: Training Logistic Regression (Baseline)")
    print("="*60)

    with mlflow.start_run(run_name="iteration_1_logistic_regression"):
        # Log parameters
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("iteration", 1)
        mlflow.log_param("max_iter", 1000)

        # Train model
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

        # Print results
        print("\nResults:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print(f"\nMLflow Run ID: {run_id}")

        return model, metrics, run_id


def train_random_forest(X_train, y_train, X_test, y_test,
                        n_estimators=100, max_depth=10, min_samples_split=5):
    """Train Random Forest model (Iteration 2 - Improved)."""
    print("\n" + "="*60)
    print("ITERATION 2: Training Random Forest (Improved)")
    print("="*60)

    with mlflow.start_run(run_name="iteration_2_random_forest"):
        # Log parameters
        mlflow.log_param("algorithm", "RandomForest")
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("iteration", 2)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log feature importance
        feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
        for feat, imp in feature_importance.items():
            mlflow.log_metric(f"importance_{feat}", imp)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

        # Print results
        print("\nResults:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print("\nFeature Importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        print(f"\nMLflow Run ID: {run_id}")

        return model, metrics, run_id


def save_model_locally(model, filename: str):
    """Save model to local pickle file."""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    filepath = os.path.join(MODEL_SAVE_PATH, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to: {filepath}")
    return filepath


def register_model(run_id: str, model_name: str = "hdd_failure_predictor"):
    """Register model in MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri, model_name)

    print(f"\nModel registered: {model_name}")
    print(f"Version: {registered_model.version}")

    # Transition to Staging
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage="Staging"
    )

    print(f"Model transitioned to Staging")

    return registered_model


def main():
    """Main training pipeline."""
    print("="*60)
    print("HDD FAILURE PREDICTION - MODEL TRAINING PIPELINE")
    print("="*60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nMLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    # Load data
    print("\n--- Loading Data ---")
    df = load_data(DATA_PATH)

    # Prepare data
    print("\n--- Preparing Data ---")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train Iteration 1: Logistic Regression
    model_lr, metrics_lr, run_id_lr = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )

    # Train Iteration 2: Random Forest
    model_rf, metrics_rf, run_id_rf = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # Compare iterations
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<15} {'Iteration 1 (LR)':<20} {'Iteration 2 (RF)':<20} {'Improvement':<15}")
    print("-"*70)
    for metric in metrics_lr.keys():
        diff = metrics_rf[metric] - metrics_lr[metric]
        print(f"{metric:<15} {metrics_lr[metric]:<20.4f} {metrics_rf[metric]:<20.4f} {diff:+.4f}")

    # Save best model locally
    print("\n--- Saving Best Model ---")
    save_model_locally(model_rf, "model_random_forest.pkl")
    save_model_locally(model_lr, "model_logistic_regression.pkl")

    # Register best model in MLflow
    print("\n--- Registering Model in MLflow Registry ---")
    register_model(run_id_rf)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nView experiments at: {MLFLOW_TRACKING_URI}")
    print(f"Iteration 1 Run ID: {run_id_lr}")
    print(f"Iteration 2 Run ID: {run_id_rf}")


if __name__ == "__main__":
    main()
