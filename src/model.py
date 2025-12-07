"""
Azure ML Training Script with MLFlow
=====================================
Simple training script for Azure ML job submission.
No Azure SDK dependencies - pure MLFlow logging.

Usage:
    python src/model.py --iteration 1  # Logistic Regression
    python src/model.py --iteration 2  # Random Forest
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuration
FEATURE_COLUMNS = ['capacity_bytes', 'lifetime', 'model_encoded']
TARGET_COLUMN = 'failure'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Load dataset from CSV."""
    print("Loading data...")
    df = pd.read_csv('data/processed/hdd_balanced_dataset.csv')
    print(f"Loaded {len(df)} rows")
    return df


def prepare_data(df):
    """Split data into train/test sets."""
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
    return X_train, X_test, y_train, y_test


def train_model(iteration, X_train, X_test, y_train, y_test):
    """Train model based on iteration number."""

    with mlflow.start_run():
        print(f"\nTraining Iteration {iteration}...")

        # Select model based on iteration
        if iteration == 1:
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            algorithm = "LogisticRegression"
            print("Model: Logistic Regression (Baseline)")
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            algorithm = "RandomForest"
            print("Model: Random Forest (Improved)")

        # Log iteration and algorithm
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("algorithm", algorithm)

        # Train model
        print("Fitting model...")
        model.fit(X_train, y_train)

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

        # Log model (only once, no autolog to avoid conflicts)
        mlflow.sklearn.log_model(model, "model")

        # Print results
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}: {algorithm}")
        print(f"{'='*60}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print(f"{'='*60}\n")

        return model, metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train HDD failure prediction model')
    parser.add_argument(
        '--iteration',
        type=int,
        required=True,
        choices=[1, 2],
        help='Iteration number: 1 for Logistic Regression, 2 for Random Forest'
    )
    args = parser.parse_args()

    print("="*70)
    print(f"HDD FAILURE PREDICTION - ITERATION {args.iteration}")
    print("="*70)

    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train model
    model, metrics = train_model(args.iteration, X_train, X_test, y_train, y_test)

    print("Training complete!")
    return 0


if __name__ == "__main__":
    exit(main())
