"""
Register Models in Azure ML Model Registry
===========================================
This script registers the trained models from experiments into Azure ML Model Registry.

Usage:
    python scripts/register_models.py
"""

from azureml.core import Workspace, Experiment

def register_models():
    """Register models from completed runs."""
    print("="*70)
    print("REGISTERING MODELS IN AZURE ML MODEL REGISTRY")
    print("="*70)

    # Connect to workspace
    print("\n1️⃣ Connecting to Azure ML workspace...")
    ws = Workspace.from_config()
    print(f"   ✅ Connected to: {ws.name}")

    # Get the experiment
    print("\n2️⃣ Retrieving experiment runs...")
    experiment = Experiment(ws, 'hdd_failure_prediction')
    print(f"   ✅ Experiment: {experiment.name}")

    # Get all runs
    runs = list(experiment.get_runs())
    print(f"   ✅ Found {len(runs)} total runs")

    # Filter for our iterations
    print("\n3️⃣ Finding iteration runs...")
    iteration_runs = {}

    for run in runs:
        try:
            # Get iteration parameter
            props = run.get_properties()
            tags = run.get_tags()
            params = {k.replace('azureml.', ''): v for k, v in props.items()}

            # Check if this run has iteration parameter
            if 'iteration' in run.get_metrics():
                iteration = int(run.get_metrics()['iteration'])
            elif 'iteration_1' in run.name or 'logistic' in run.name.lower():
                iteration = 1
            elif 'iteration_2' in run.name or 'forest' in run.name.lower():
                iteration = 2
            else:
                continue

            # Keep the most recent run for each iteration
            if iteration not in iteration_runs or run.created_time > iteration_runs[iteration].created_time:
                iteration_runs[iteration] = run
                print(f"   ✅ Iteration {iteration}: Run {run.id[:8]}...")

        except Exception as e:
            continue

    if len(iteration_runs) < 2:
        print(f"\n⚠️ Warning: Only found {len(iteration_runs)} iteration(s)")
        print("   Please ensure both iterations have been run successfully")

    # Register models
    print("\n4️⃣ Registering models...")
    registered_models = []

    for iteration, run in sorted(iteration_runs.items()):
        try:
            # Determine model details
            if iteration == 1:
                algorithm = "Logistic Regression"
                description = "Logistic Regression baseline model (Iteration 1)"
            else:
                algorithm = "Random Forest"
                description = "Random Forest improved model (Iteration 2)"

            print(f"\n   📦 Registering Iteration {iteration}...")
            print(f"      Run ID: {run.id}")
            print(f"      Algorithm: {algorithm}")

            # Register the model
            model = run.register_model(
                model_name='hdd_failure_predictor',
                model_path='outputs/model',
                description=description,
                tags={
                    'iteration': str(iteration),
                    'algorithm': algorithm,
                    'framework': 'scikit-learn',
                    'purpose': 'HDD failure prediction'
                }
            )

            # Get metrics for this run
            metrics = run.get_metrics()
            accuracy = metrics.get('accuracy', 0)
            f1_score = metrics.get('f1_score', 0)
            roc_auc = metrics.get('roc_auc', 0)

            print(f"      ✅ Registered as version {model.version}")
            print(f"      Metrics:")
            print(f"         • Accuracy: {accuracy:.4f}")
            print(f"         • F1-Score: {f1_score:.4f}")
            print(f"         • ROC-AUC: {roc_auc:.4f}")

            registered_models.append({
                'iteration': iteration,
                'version': model.version,
                'algorithm': algorithm,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'roc_auc': roc_auc
            })

        except Exception as e:
            print(f"      ❌ Error registering iteration {iteration}: {str(e)}")

    # Summary
    print("\n" + "="*70)
    print("✅ MODEL REGISTRATION COMPLETE!")
    print("="*70)

    if registered_models:
        print(f"\n📊 Registered Models Summary:")
        print(f"\n{'Iter':<6}{'Version':<10}{'Algorithm':<25}{'Accuracy':<12}{'F1-Score':<12}{'ROC-AUC'}")
        print("-"*70)
        for m in registered_models:
            print(f"{m['iteration']:<6}{m['version']:<10}{m['algorithm']:<25}{m['accuracy']:<12.4f}{m['f1_score']:<12.4f}{m['roc_auc']:.4f}")

        print(f"\n💡 Next Steps:")
        print(f"   1. Go to Azure ML Studio → Models → hdd_failure_predictor")
        print(f"   2. Review and compare model versions")
        print(f"   3. Deploy the best model (likely Iteration 2)")
        print(f"   4. Select model → Deploy → Real-time endpoint")

    else:
        print("\n⚠️ No models were registered. Please check:")
        print("   1. Both training jobs completed successfully")
        print("   2. Models were saved in the runs")
        print("   3. Run names contain 'iteration' or algorithm names")

    print()

    return registered_models


if __name__ == "__main__":
    try:
        register_models()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("   1. Ensure config.json exists in project root")
        print("   2. Verify training jobs completed successfully")
        print("   3. Check experiment 'hdd_failure_prediction' exists in Azure ML")
        exit(1)
