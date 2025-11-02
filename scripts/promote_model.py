# promote model

import os
import logging
import mlflow
import dagshub

def promote_model():
    dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    dagshub_url = "https://dagshub.com"
    repo_owner = dagshub_repo_owner or "ayusprasad"
    repo_name = dagshub_repo_name or "capstone-project"

    # Production / CI: if a project token is provided via CAPSTONE_TEST, use it
    # to authenticate with DagsHub and set the MLflow tracking URI. This is
    # optional — when running locally without that token, we fall back to any
    # `MLFLOW_TRACKING_URI` from the environment or continue offline.
    capstone_token = os.getenv("CAPSTONE_TEST")
    if capstone_token:
        # Set MLflow/DagsHub credentials from the provided token
        os.environ["MLFLOW_TRACKING_USERNAME"] = capstone_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = capstone_token
        try:
            mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        except Exception as e:
            logging.warning('Failed to set MLflow tracking URI from CAPSTONE_TEST: %s', e)
    else:
        # Fall back to an explicit MLflow URI if provided via env (local dev)
        dagshub_token = os.getenv("MLFLOW_TRACKING_URI")
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
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True,
        )
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    except Exception as e:
        logging.warning('DagsHub initialization skipped: %s', e)

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()
