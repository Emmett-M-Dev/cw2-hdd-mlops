"""
Train Iteration 3 Locally
==========================
Trains a Random Forest model with the current environment (sklearn 1.3.0)
for local deployment demonstration.

This model will be compatible with the local serving environment.

Usage:
    python scripts/train_local_iteration3.py
"""

import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuration
FEATURE_COLUMNS = ['capacity_bytes', 'lifetime', 'model_encoded']
TARGET_COLUMN = 'failure'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def main():
    print("="*70)
    print("TRAINING ITERATION 3: LOCAL RANDOM FOREST")
    print("="*70)
    print("\nThis model will be trained with current environment:")
    print("  - Python 3.11.4")
    print("  - scikit-learn 1.3.0")
    print("  - For local deployment demonstration\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv('data/processed/hdd_balanced_dataset.csv')
    print(f"Loaded {len(df)} rows")

    # Prepare data
    print("Preparing data...")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Start MLflow run
    mlflow.set_tracking_uri("file:///c:/Users/Emmet/cw2-hdd-mlops/mlruns")
    mlflow.set_experiment("Default")

    with mlflow.start_run(run_name="iteration_3_local_rf"):
        # Log parameters
        mlflow.log_param("iteration", 3)
        mlflow.log_param("algorithm", "RandomForest")
        mlflow.log_param("purpose", "Local deployment demo")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("sklearn_version", "1.3.0")

        # Fit model
        model.fit(X_train, y_train)
        print("Model trained!")

        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model directly to models/v3 for easy serving
        print("\nSaving model to models/v3...")
        import os
        os.makedirs("models/v3", exist_ok=True)

        with open("models/v3/model.pkl", 'wb') as f:
            pickle.dump(model, f)

        # Print results
        print(f"\n{'='*60}")
        print(f"ITERATION 3: Random Forest (Local)")
        print(f"{'='*60}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print(f"{'='*60}")

        print(f"\nModel saved to:")
        print(f"  - MLflow: {mlflow.get_artifact_uri()}")
        print(f"  - Direct: models/v3/model.pkl")

        print(f"\nNext step:")
        print(f"  python scripts/serve_model_local.py")

    return 0

if __name__ == "__main__":
    exit(main())
