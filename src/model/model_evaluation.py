"""Model evaluation script that handles MLflow tracking with fallbacks."""
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import dotenv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# Set up logging early to catch any initialization issues
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv = dotenv.load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

dagshub_token = os.getenv("MLFLOW_TRACKING_URI")
dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")

# Production / CI: if a project token is provided via CAPSTONE_TEST, use it
# to authenticate with DagsHub and set the MLflow tracking URI. This is
# optional — when running locally without that token, we fall back to any
# `MLFLOW_TRACKING_URI` from the environment or continue offline.
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

# # Below code block is for local use
# # -------------------------------------------------------------------------------------
# # Configure MLflow tracking URI if provided via environment.
# if dagshub_token:
#     try:
#         mlflow.set_tracking_uri(dagshub_token)
#     except Exception as e:
#         logging.warning('Failed to set MLflow tracking URI: %s', e)

# # Initialize DagsHub integration when available. In CI (or non-DagsHub
# # environments) this may fail (repo not found); treat that as non-fatal so
# # the evaluation step can still run and produce DVC-tracked outputs.
# try:
#     dagshub.init(
#         repo_owner=dagshub_repo_owner,
#         repo_name=dagshub_repo_name,
#         mlflow=True,
#     )
#     print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
# except Exception as e:
#     # Common failure is DagsHubRepoNotFoundError when not running inside a
#     # DagsHub repository. Log a warning and continue; evaluation should still
#     # complete and produce the expected artifacts for DVC.
#     logging.warning('DagsHub initialization skipped: %s', e)
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

# Dummy context for when MLflow is unavailable
class DummyContext:
    """Context manager that mimics MLflow run context when MLflow fails."""
    
    def __init__(self):
        """Initialize with a dummy run info object."""
        self.info = type('RunInfo', (), {'run_id': 'no-mlflow'})()
    
    def __enter__(self):
        """Enter the context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        pass


def main():
    """Main function to evaluate the model and log metrics."""
    experiment_name = "my-dvc-pipeline"
    tracking_available = False
    
    try:
        # Import MLflow here to avoid scoping issues
        import mlflow  
        import mlflow.sklearn

        # Initialize MLflow tracking
        tracking_uri = mlflow.get_tracking_uri()
        if not tracking_uri:
            local_store = Path("mlruns").absolute()
            local_store.mkdir(parents=True, exist_ok=True)
            tracking_uri = f"file://{local_store}"
            mlflow.set_tracking_uri(tracking_uri)
        logging.info("Using MLflow tracking at %s", tracking_uri)
        
        # Get or create experiment safely
        experiment = None
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except Exception as e:
            logging.info("Couldn't find experiment: %s", e)
        
        # Create if it doesn't exist
        if experiment is None:
            try:
                mlflow.create_experiment(experiment_name)
                logging.info("Created experiment %s", experiment_name)
            except Exception as e:
                logging.info("Couldn't create experiment: %s", e)
        
        # Set experiment and start run
        mlflow.set_experiment(experiment_name)
        run_ctx = mlflow.start_run()
        tracking_available = True
        logging.info("Started MLflow run")
    except Exception as e:
        logging.warning("MLflow tracking disabled: %s", e)
        run_ctx = DummyContext()
    
    # Continue with model evaluation using either MLflow or dummy context
    with run_ctx as run:
        try:
            # Core model evaluation
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            import numpy as np
            X_test = np.array(test_data.iloc[:, :-1].values)
            y_test = np.array(test_data.iloc[:, -1].values)

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')

            # Only attempt MLflow operations if tracking is available
            if tracking_available and mlflow:
                # Log metrics
                for name, value in metrics.items():
                    try:
                        mlflow.log_metric(name, value)
                    except Exception as e:
                        logging.warning('Failed to log metric %s: %s', name, e)
                
                # Log parameters if model has get_params
                if hasattr(clf, 'get_params'):
                    for name, value in clf.get_params().items():
                        try:
                            mlflow.log_param(name, value)
                        except Exception as e:
                            logging.warning(
                                'Failed to log parameter %s: %s',
                                name,
                                e)
                
                # Log model artifact
                try:
                    model_log = __import__('mlflow.sklearn')
                    model_log.sklearn.log_model(clf, "model")
                    logging.info('Model logged to MLflow')
                except Exception as e:
                    logging.warning('Failed to log model: %s', e)

                # Try model registry
                try:
                    model_name = "my_model"
                    run_id = getattr(run.info, '_run_uuid',
                                   getattr(run.info, 'run_id', None))
                    if not run_id:
                        run_id = 'unknown'
                    uri = f"runs:/{run_id}/model"
                    mv = mlflow.register_model(uri, model_name)
                    msg = f'Registered as {model_name} v{mv.version}'
                    logging.info(msg)
                except Exception as e:
                    logging.warning('Model registry failed: %s', e)

                # Log metrics file as artifact
                try:
                    mlflow.log_artifact('reports/metrics.json')
                except Exception as e:
                    logging.warning('Failed to log metrics file: %s', e)
            info_path = 'reports/experiment_info.json'
            if tracking_available and run and run.info:
                run_id = getattr(
                    run.info,
                    '_run_uuid',
                    getattr(run.info, 'run_id', 'no-mlflow')
                )
            else:
                run_id = 'no-mlflow'
            save_model_info(run_id, "model", info_path)

        except Exception as e:
            err_msg = 'Failed to complete the model evaluation process: %s'
            logging.error(err_msg, e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
