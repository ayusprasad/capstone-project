# register model with versioning support
import dagshub, sys, pathlib
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
import json
import mlflow
import logging
from src.logger import logging
import os
from datetime import datetime
import shutil
import dagshub, warnings, pathlib, dotenv
from pathlib import Path
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


load_dotenv = dotenv.load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

dagshub_token = os.getenv("MLFLOW_TRACKING_URI")
dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")

# Initialize DagHub and MLflow when available. In CI or non-DagsHub
# environments this may fail; treat initialization as non-fatal so the
# registration step can still update local registry files used by DVC.
# --------------------------------------------------------------------------------------------


capstone_token = os.getenv("CAPSTONE_TEST")
if capstone_token:
    # Set MLflow/DagsHub credentials from the provided token
    os.environ["MLFLOW_TRACKING_USERNAME"] = capstone_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = capstone_token
    dagshub_url = "https://dagshub.com"
    repo_owner = dagshub_repo_owner or os.getenv("USER") or "ayusprasad"
    repo_name = dagshub_repo_name or "capstone-project"
    try:
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    except Exception as e:
        logging.warning('Failed to set MLflow tracking URI from CAPSTONE_TEST: %s', e)
else:
    # Fall back to an explicit MLflow URI if provided via env (local dev)
    if dagshub_token:
        try:
            mlflow.set_tracking_uri(dagshub_token)
        except Exception as e:
            logging.warning('Failed to set MLflow tracking URI: %s', e)

# Initialize DagsHub integration when available. In CI (or non-DagsHub
# environments) this may fail (repo not found); treat that as non-fatal so
# the evaluation step can still run and produce DVC-tracked outputs.
try:
    dagshub.init(
        repo_owner=dagshub_repo_owner,
        repo_name=dagshub_repo_name,
        mlflow=True,
    )
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
except Exception as e:
    logging.warning('DagsHub initialization skipped: %s', e)













#----------------------------------------------------------------------------------------
# if dagshub_token:
#     try:
#         mlflow.set_tracking_uri(dagshub_token)
#     except Exception as e:
#         logging.warning('Failed to set MLflow tracking URI: %s', e)

# try:
#     dagshub.init(
#         repo_owner=dagshub_repo_owner,
#         repo_name=dagshub_repo_name,
#         mlflow=True,
#     )
#     print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
# except Exception as e:
#     logging.warning('DagsHub initialization skipped: %s', e)

# ------------------------------------------------------------------------------------------

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def load_existing_registry(registry_path: str) -> dict:
    """Load existing model registry or create new one."""
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                data = json.load(f)
                # Check if it's the old format (without "models" key)
                if "models" not in data:
                    # Old format detected, migrate it
                    logging.info('Migrating old registry format to new format')
                    return {"models": {}}
                return data
        except Exception as e:
            logging.warning(f'Error loading registry: {e}. Creating new registry.')
            return {"models": {}}
    return {"models": {}}


def get_next_version(registry: dict, model_name: str) -> int:
    """Get the next version number for a model."""
    try:
        if "models" not in registry:
            return 1
        if model_name not in registry["models"]:
            return 1
        if "versions" not in registry["models"][model_name]:
            return 1
        return len(registry["models"][model_name]["versions"]) + 1
    except Exception as e:
        logging.error(f'Error getting next version: {e}')
        return 1


def register_model(model_name: str, model_info: dict):
    """Register a model with versioning support."""
    try:
        # Get experiment ID and run
        experiment_id = None
        if 'run_id' in model_info:
            try:
                # Get run info
                run = mlflow.get_run(model_info['run_id'])
                experiment_id = run.info.experiment_id
                logging.info(f'Got experiment_id: {experiment_id}')
                
                # Register in MLflow Model Registry
                try:
                    model_path = model_info.get('model_path', 'model')
                    model_uri = f"runs:/{model_info['run_id']}/{model_path}"
                    registered_model = mlflow.register_model(
                        model_uri, 
                        model_name
                    )
                    logging.info(
                        f"Registered model: {registered_model.name}"
                        f" v{registered_model.version}"
                    )
                    
                    # Move to staging
                    client = mlflow.tracking.MlflowClient()
                    client.transition_model_version_stage(
                        name=model_name,
                        version=registered_model.version,
                        stage="Staging"
                    )
                    logging.info(
                        f"Model {model_name} v{registered_model.version}"
                        " moved to Staging"
                    )
                except Exception as e:
                    logging.warning(f'MLflow registration failed: {e}')
            except Exception as e:
                logging.warning(f'Could not get experiment_id: {e}')
                experiment_id = 'unknown'
        
        # Load metrics from reports/metrics.json
        metrics = {}
        try:
            with open('reports/metrics.json', 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logging.warning(f'Could not load metrics.json: {e}')
        
        # Load or create registry
        model_registry_path = 'reports/model_registry.json'
        registry = load_existing_registry(model_registry_path)
        
        # Ensure registry has proper structure
        if "models" not in registry:
            registry = {"models": {}}
        
        # Get version number
        version = get_next_version(registry, model_name)
        
        # Create version data
        version_data = {
            'version': version,
            'run_id': model_info['run_id'],
            'experiment_id': experiment_id,
            'model_path': f'models/model_v{version}.pkl',
            'original_model_path': model_info.get('model_path', 'models/model.pkl'),
            'model_uri': f"runs:/{model_info['run_id']}/{model_info.get('model_path', 'model')}",
            'mlflow_run_url': f"https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}.mlflow/#/experiments/{experiment_id}/runs/{model_info['run_id']}",
            'mlflow_experiment_url': f"https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}.mlflow/#/experiments/{experiment_id}",
            'registered_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics,
            'status': 'production'  # Can be: development, staging, production, archived
        }
        
        # Initialize model entry if it doesn't exist
        if model_name not in registry["models"]:
            registry["models"][model_name] = {
                "name": model_name,
                "description": "Logistic Regression model for sentiment analysis",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latest_version": version,
                "versions": []
            }
        
        # Ensure versions list exists
        if "versions" not in registry["models"][model_name]:
            registry["models"][model_name]["versions"] = []
        
        # Add new version
        registry["models"][model_name]["versions"].append(version_data)
        registry["models"][model_name]["latest_version"] = version
        registry["models"][model_name]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Copy model file with version number
        original_model = 'models/model.pkl'
        versioned_model = f'models/model_v{version}.pkl'
        if os.path.exists(original_model):
            shutil.copy2(original_model, versioned_model)
            logging.info(f'Model copied to {versioned_model}')
        else:
            logging.warning(f'Original model file not found: {original_model}')
        
        # Save registry
        with open(model_registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
        
        logging.info(f'Model {model_name} v{version} registered successfully')
        
        # Print beautiful output
        print(f'\n{"="*80}')
        print(f'🎉 MODEL REGISTERED SUCCESSFULLY!')
        print(f'{"="*80}')
        print(f'📦 Model Name: {model_name}')
        print(f'🔢 Version: {version}')
        print(f'📊 Status: {version_data["status"]}')
        print(f'⏰ Registered at: {version_data["registered_at"]}')
        
        if metrics:
            print(f'\n📈 Metrics:')
            for metric_name, metric_value in metrics.items():
                print(f'   • {metric_name}: {metric_value:.4f}')
        
        print(f'\n🔗 MLflow Links:')
        print(f'   📊 View Run: {version_data["mlflow_run_url"]}')
        print(f'   🧪 View Experiment: {version_data["mlflow_experiment_url"]}')
        print(f'\n💾 Files:')
        print(f'   • Model: {versioned_model}')
        print(f'   • Registry: {model_registry_path}')
        print(f'\n💡 Note: DagHub free tier does not support Model Registry UI.')
        print(f'   Your model is tracked locally in {model_registry_path}')
        print(f'{"="*80}\n')
        
    except Exception as e:
        logging.error('Error saving model metadata: %s', e)
        import traceback
        traceback.print_exc()
        raise


def list_models():
    """List all registered models and their versions."""
    registry_path = 'reports/model_registry.json'
    if not os.path.exists(registry_path):
        print("No models registered yet.")
        return
    
    registry = load_existing_registry(registry_path)
    
    if "models" not in registry or not registry["models"]:
        print("No models registered yet.")
        return
    
    print(f'\n{"="*80}')
    print(f'📚 REGISTERED MODELS')
    print(f'{"="*80}\n')
    
    for model_name, model_data in registry["models"].items():
        print(f'Model: {model_name}')
        print(f'Description: {model_data.get("description", "N/A")}')
        print(f'Latest Version: v{model_data.get("latest_version", 1)}')
        print(f'Total Versions: {len(model_data.get("versions", []))}')
        print(f'\nVersions:')
        
        for version in reversed(model_data.get("versions", [])):  # Show latest first
            print(f'\n  v{version["version"]} - {version.get("status", "unknown")}')
            print(f'  Registered: {version.get("registered_at", "N/A")}')
            if version.get("metrics"):
                print(f'  Metrics: {", ".join([f"{k}: {v:.4f}" for k, v in version.get("metrics", {}).items()])}')
            print(f'  Run URL: {version.get("mlflow_run_url", "N/A")}')
        
        print(f'\n{"-"*80}\n')


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
        
        # Uncomment to list all models after registration
        print("\n")
        list_models()
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()