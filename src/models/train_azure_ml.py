"""
Hard Drive Failure Prediction - Azure ML Training Script
=========================================================
This script trains models on Azure ML with MLflow tracking integration.
Supports both Iteration 1 (Logistic Regression) and Iteration 2 (Random Forest).

Usage:
    python src/models/train_azure_ml.py --iteration 1
    python src/models/train_azure_ml.py --iteration 2
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from azureml.core import Run, Dataset, Workspace
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Get Azure ML run context
run = Run.get_context()

# Try to get workspace from run context, fallback to config file
try:
    ws = run.experiment.workspace
    print("✅ Running in Azure ML compute")
except AttributeError:
    # Running locally for testing
    ws = Workspace.from_config()
    print("✅ Running locally with Azure ML workspace connection")

# Configuration
FEATURE_COLUMNS = ['capacity_bytes', 'lifetime', 'model_encoded']
TARGET_COLUMN = 'failure'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data_from_azure(dataset_name='hdd_balanced_dataset'):
    """Load dataset from Azure ML Dataset."""
    print(f"Loading dataset '{dataset_name}' from Azure ML...")
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    df = dataset.to_pandas_dataframe()
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Features: {FEATURE_COLUMNS}")
    print(f"   Target: {TARGET_COLUMN}")
    return df


def prepare_data(df):
    """Prepare features and target, split into train/test sets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\n📊 Data Split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Class distribution: {y.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def train_model(iteration: int):
    """Train model based on iteration number."""
    print(f"\n{'='*70}")
    print(f"TRAINING ITERATION {iteration}")
    print(f"{'='*70}\n")

    # Load and prepare data
    df = load_data_from_azure()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Start MLflow run (Azure ML has built-in MLflow tracking)
    with mlflow.start_run():
        # Configure model based on iteration
        if iteration == 1:
            # Iteration 1: Logistic Regression (BASELINE)
            print("🔹 Model: Logistic Regression (Baseline)")
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            algorithm = "LogisticRegression"

            # Log parameters
            mlflow.log_param("max_iter", 1000)

        else:
            # Iteration 2: Random Forest (IMPROVED)
            print("🔹 Model: Random Forest (Improved)")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            algorithm = "RandomForest"

            # Log parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("min_samples_split", 5)

        # Log common parameters
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Train the model
        print("\n🔧 Training model...")
        model.fit(X_train, y_train)
        print("✅ Training complete!")

        # Make predictions
        print("\n📈 Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Log metrics to both MLflow and Azure ML
        print("\n📊 Metrics:")
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            run.log(name, value)  # Also log to Azure ML directly
            print(f"   {name}: {value:.4f}")

        # Log model to MLflow (Azure ML captures this automatically)
        print("\n💾 Logging model...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None  # We'll register manually later
        )

        # Log feature importance for Random Forest
        if iteration == 2 and hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
            print("\n🌟 Feature Importance:")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"   {feature}: {importance:.4f}")
                mlflow.log_metric(f"feature_importance_{feature}", importance)

        print(f"\n{'='*70}")
        print(f"✅ ITERATION {iteration} COMPLETE!")
        print(f"   Algorithm: {algorithm}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"{'='*70}\n")

        return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HDD failure prediction model on Azure ML')
    parser.add_argument(
        '--iteration',
        type=int,
        required=True,
        choices=[1, 2],
        help='Iteration number: 1 for Logistic Regression, 2 for Random Forest'
    )
    args = parser.parse_args()

    print(f"\n🚀 Starting Azure ML Training Job")
    print(f"   Iteration: {args.iteration}")
    print(f"   Workspace: {ws.name}")
    print(f"   Experiment: hdd_failure_prediction\n")

    train_model(args.iteration)

    print("🎉 Training job finished successfully!")
