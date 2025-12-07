"""
Submit Training Job to Azure ML
================================
This script submits a training job to Azure ML compute cluster.

Usage:
    python scripts/submit_azure_job.py --iteration 1  # Logistic Regression
    python scripts/submit_azure_job.py --iteration 2  # Random Forest
"""

import argparse
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment

def submit_training_job(iteration: int):
    """Submit training job to Azure ML."""
    print("="*70)
    print(f"SUBMITTING ITERATION {iteration} TO AZURE ML")
    print("="*70)

    # Connect to workspace - will use default authentication
    print("\n1 Connecting to Azure ML workspace...")

    # Set environment variable to force Azure CLI auth
    os.environ['AZURE_CLI_DISABLE_CONNECTION_VERIFICATION'] = '1'

    try:
        ws = Workspace.from_config()
    except Exception as e:
        print(f"   Failed with default auth, trying interactive...")
        from azureml.core.authentication import InteractiveLoginAuthentication
        interactive_auth = InteractiveLoginAuthentication(force=True)
        ws = Workspace.from_config(auth=interactive_auth)
    print(f"    Connected to: {ws.name}")

    # Create or get environment
    print("\n2 Setting up Python environment...")
    try:
        # Try to get existing environment
        env = Environment.get(workspace=ws, name="sklearn-mlflow-env")
        print(f"    Using existing environment: sklearn-mlflow-env")
    except:
        # Create new environment from conda file
        env = Environment.from_conda_specification(
            name="sklearn-mlflow-env",
            file_path="environment.yml"
        )
        print(f"    Created new environment from environment.yml")

    # Create experiment
    print("\n3 Setting up experiment...")
    experiment = Experiment(workspace=ws, name='hdd_failure_prediction')
    print(f"    Experiment: {experiment.name}")

    # Configure the training run
    print("\n4 Configuring training job...")
    run_name = f"iteration_{iteration}_{'logistic_regression' if iteration == 1 else 'random_forest'}"

    # Create a .amlignore file to exclude problematic files
    amlignore_path = os.path.join('.', '.amlignore')
    if not os.path.exists(amlignore_path):
        with open(amlignore_path, 'w') as f:
            f.write("# Ignore device files and unnecessary directories\n")
            f.write("nul\n")
            f.write(".git/\n")
            f.write(".venv/\n")
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
            f.write(".azureml/\n")
            f.write("mlruns/\n")
            f.write("notebooks/\n")
            f.write("models/\n")
            f.write("reports/\n")
            f.write(".claude/\n")

    config = ScriptRunConfig(
        source_directory='.',
        script='src/models/train_azure_ml.py',
        arguments=['--iteration', iteration],
        compute_target='training-cluster',
        environment=env
    )

    print(f"    Configuration:")
    print(f"      Script: src/models/train_azure_ml.py")
    print(f"      Iteration: {iteration}")
    print(f"      Compute: training-cluster")
    print(f"      Run name: {run_name}")

    # Submit the run
    print("\n5 Submitting job...")
    run = experiment.submit(config)

    print(f"\n JOB SUBMITTED SUCCESSFULLY!")
    print(f"\n Run Details:")
    print(f"    Run ID: {run.id}")
    print(f"    Run Name: {run_name}")
    print(f"    Status: {run.get_status()}")
    print(f"\n Monitor your run here:")
    print(f"   {run.get_portal_url()}")

    print(f"\n Waiting for job to complete...")
    print(f"   (This may take 10-15 minutes on first run due to environment build)")
    print(f"   (Subsequent runs will be faster: ~3-5 minutes)")
    print(f"\n Tip: Open the portal URL above to watch logs in real-time!\n")

    # Wait for completion with output
    run.wait_for_completion(show_output=True)

    # Display final results
    print(f"\n{'='*70}")
    print(f" JOB COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\n Final Metrics:")

    metrics = run.get_metrics()
    if metrics:
        for name, value in sorted(metrics.items()):
            if not name.startswith('feature_importance'):
                print(f"    {name}: {value:.4f}")

    print(f"\n Next Steps:")
    if iteration == 1:
        print(f"   1. Review metrics above")
        print(f"   2. Submit Iteration 2: python scripts/submit_azure_job.py --iteration 2")
        print(f"   3. Compare both iterations in Azure ML Studio")
    else:
        print(f"   1. Compare Iteration 1 vs Iteration 2 in Azure ML Studio")
        print(f"   2. Register the best model: python scripts/register_models.py")
        print(f"   3. Deploy the model via Azure ML Studio UI")

    print()

    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit training job to Azure ML')
    parser.add_argument(
        '--iteration',
        type=int,
        required=True,
        choices=[1, 2],
        help='Iteration number: 1 for Logistic Regression, 2 for Random Forest'
    )
    args = parser.parse_args()

    try:
        submit_training_job(args.iteration)
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("   1. Ensure config.json exists in project root")
        print("   2. Verify compute cluster 'training-cluster' exists in Azure ML")
        print("   3. Check dataset 'hdd_balanced_dataset' is uploaded")
        print("   4. Run: python scripts/upload_data_to_azure.py")
        exit(1)
