"""
Upload HDD Dataset to Azure ML
================================
This script uploads the balanced HDD failure dataset to Azure ML workspace.

Usage:
    python scripts/upload_data_to_azure.py
"""

from azureml.core import Workspace, Dataset
from azureml.core.authentication import AzureCliAuthentication
import os

def upload_dataset():
    """Upload dataset to Azure ML workspace."""
    print("="*70)
    print("UPLOADING DATASET TO AZURE ML")
    print("="*70)

    # Connect to workspace
    print("\n1️⃣ Connecting to Azure ML workspace...")
    cli_auth = AzureCliAuthentication()
    ws = Workspace.from_config()
    print(f"   ✅ Connected to: {ws.name}")
    print(f"   📍 Region: {ws.location}")
    print(f"   📦 Resource Group: {ws.resource_group}")

    # Get default datastore
    print("\n2️⃣ Getting default datastore...")
    datastore = ws.get_default_datastore()
    print(f"   ✅ Datastore: {datastore.name}")

    # Check if file exists
    data_path = 'data/processed/hdd_balanced_dataset.csv'
    print(f"\n3️⃣ Checking local data file...")
    if not os.path.exists(data_path):
        print(f"   ❌ ERROR: File not found: {data_path}")
        print("   Please ensure the data file exists before running this script.")
        return False

    print(f"   ✅ Found: {data_path}")

    # Upload file to datastore
    print(f"\n4️⃣ Uploading file to Azure Blob Storage...")
    datastore.upload_files(
        files=[data_path],
        target_path='hdd_data',
        overwrite=True,
        show_progress=True
    )
    print("   ✅ Upload complete!")

    # Register as dataset
    print(f"\n5️⃣ Registering dataset in Azure ML...")
    dataset = Dataset.Tabular.from_delimited_files(
        path=[(datastore, 'hdd_data/hdd_balanced_dataset.csv')]
    )

    dataset = dataset.register(
        workspace=ws,
        name='hdd_balanced_dataset',
        description='Balanced HDD failure prediction dataset (8,961 samples, 33% failure rate)',
        create_new_version=True,
        tags={
            'source': 'CW1',
            'format': 'CSV',
            'features': 'capacity_bytes, lifetime, model_encoded',
            'target': 'failure',
            'samples': '8961'
        }
    )

    print(f"   ✅ Dataset registered:")
    print(f"      Name: {dataset.name}")
    print(f"      Version: {dataset.version}")
    print(f"      ID: {dataset.id}")

    # Verify dataset
    print(f"\n6️⃣ Verifying dataset...")
    df = dataset.to_pandas_dataframe()
    print(f"   ✅ Dataset shape: {df.shape}")
    print(f"   ✅ Columns: {df.columns.tolist()}")
    print(f"\n   Preview:")
    print(df.head())

    print("\n" + "="*70)
    print("✅ DATASET UPLOAD COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Dataset Name: {dataset.name}")
    print(f"   • Version: {dataset.version}")
    print(f"   • Rows: {len(df)}")
    print(f"   • Columns: {len(df.columns)}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Go to Azure ML Studio → Data → Datasets")
    print(f"   2. Verify '{dataset.name}' appears in the list")
    print(f"   3. Run training script: python scripts/submit_azure_job.py --iteration 1")
    print()

    return True


if __name__ == "__main__":
    try:
        upload_dataset()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("   1. Ensure config.json exists in project root")
        print("   2. Verify you're logged into Azure CLI: az login")
        print("   3. Check data file exists: data/processed/hdd_balanced_dataset.csv")
        exit(1)
