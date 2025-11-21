"""
Hard Drive Failure Prediction - Data Loader Module
===================================================
This module provides utilities for loading and preprocessing data
for the HDD failure prediction pipeline.
"""

import os
import pandas as pd


# Configuration
RAW_DATA_PATH = "data/raw/hard_drive_failure_data.csv"
PROCESSED_DATA_PATH = "data/processed/hdd_balanced_dataset.csv"
FEATURE_COLUMNS = ['capacity_bytes', 'lifetime', 'model_encoded']
TARGET_COLUMN = 'failure'


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded raw data: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_processed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load processed/cleaned data from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded processed data: {len(df)} rows, {len(df.columns)} columns")
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple:
    """Extract features and target from dataframe."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def validate_data(df: pd.DataFrame) -> dict:
    """Validate data quality and return report."""
    report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().to_dict(),
        'has_missing': df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any(),
        'target_distribution': df[TARGET_COLUMN].value_counts().to_dict(),
        'feature_stats': df[FEATURE_COLUMNS].describe().to_dict()
    }
    return report


def get_data_summary(df: pd.DataFrame) -> str:
    """Generate a text summary of the dataset."""
    summary = []
    summary.append(f"Dataset Shape: {df.shape}")
    summary.append(f"Features: {FEATURE_COLUMNS}")
    summary.append(f"Target: {TARGET_COLUMN}")
    summary.append(f"\nTarget Distribution:")
    for val, count in df[TARGET_COLUMN].value_counts().items():
        pct = count / len(df) * 100
        summary.append(f"  {val}: {count} ({pct:.1f}%)")

    return "\n".join(summary)


if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader Module")
    print("="*50)

    df = load_processed_data()
    print(get_data_summary(df))

    report = validate_data(df)
    print(f"\nData Validation: {'PASS' if not report['has_missing'] else 'FAIL'}")
