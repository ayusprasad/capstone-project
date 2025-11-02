import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os,sys,pathlib
import dotenv
from dotenv import dotenv_values        
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from src.logger import logging

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

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log the model artifact to MLflow (as an artifact). We log the model
            # even if the Model Registry operation (register_model) fails on
            # certain tracking backends (e.g., Dagshub free tier).
            model_name = "my_model"
            try:
                mlflow.sklearn.log_model(clf, "model")
                logging.info('Model artifact logged to MLflow under "model"')
            except Exception as e:
                logging.warning('Failed to log model artifact to MLflow: %s', e)

            # Save model info to a local file that downstream steps (DVC) expect.
            info_path = 'reports/experiment_info.json'
            save_model_info(run.info.run_id, "model", info_path)

            # Attempt to register the model in the MLflow Model Registry.
            # Some backends (Dagshub free tier) may not support the registry API;
            # if registration fails, catch the error and continue so the run
            # completes and produces the expected outputs for DVC.
            try:
                registered_model_uri = f"runs:/{run.info.run_id}/model"
                mv = mlflow.register_model(registered_model_uri, model_name)
                logging.info('Model registered: %s v%s', model_name, mv.version)
                print(f"Model registered as: {model_name} (version {mv.version})")
            except Exception as e:
                logging.warning('Model registry operation failed: %s', e)
                print(f"Warning: model registry operation failed: {e}")

            # Log the metrics file to MLflow as an artifact so it's attached to the run
            try:
                mlflow.log_artifact('reports/metrics.json')
            except Exception as e:
                logging.warning('Failed to log metrics artifact to MLflow: %s', e)

        except Exception as e:
            err_msg = 'Failed to complete the model evaluation process: %s'
            logging.error(err_msg, e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
