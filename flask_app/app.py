from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
try:
    from prometheus_client import (
        Counter,
        Histogram,
        generate_latest,
        CollectorRegistry,
        CONTENT_TYPE_LATEST,
    )
except Exception:
    # Provide lightweight no-op fallbacks so the app can import and run
    # even when prometheus_client isn't installed (useful for CI/test).
    class _NoopMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1):
            return None

        def observe(self, value):
            return None

    class CollectorRegistry:
        pass

    def generate_latest(registry):
        return b""

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    Counter = _NoopMetric
    Histogram = _NoopMetric
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import dotenv
from pathlib import Path
import sys
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Below code block is for local use
# -------------------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
load_dotenv = dotenv.load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

dagshub_token = os.getenv("MLFLOW_TRACKING_URI")
dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)

# If a registry version was found, try to load from MLflow Model Registry.
# Otherwise, fall back to a local model file `models/model.pkl`.
model = None
if model_version:
    model_uri = f'models:/{model_name}/{model_version}'
    print(f"Fetching model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model from registry: {model_name} v{model_version}")
    except Exception as e:
        # If registry fetch fails, log and fall back to local file below
        print(f"Warning: failed to load model from registry ({e}). Falling back to local model.")
else:
    print(f"No model version found in registry for '{model_name}'. Trying local model file.")

# Local fallback
local_model_path = 'models/model.pkl'
# If registry loading failed, prefer any versioned model created by the
# local registry (reports/model_registry.json) before falling back to the
# unversioned local model. This lets the project use the locally registered
# versions when the remote registry is not available on the tracking backend.
if model is None:
    # Try reading local model registry JSON
    registry_json = 'reports/model_registry.json'
    try:
        if os.path.exists(registry_json):
            with open(registry_json, 'r') as rf:
                import json as _json

                reg = _json.load(rf)
            if reg and 'models' in reg and model_name in reg['models']:
                latest = reg['models'][model_name].get('latest_version')
                if latest:
                    versioned_path = f'models/model_v{latest}.pkl'
                    if os.path.exists(versioned_path):
                        with open(versioned_path, 'rb') as f:
                            model = pickle.load(f)
                        print(f"Loaded model from local registry: {versioned_path}")
    except Exception as e:
        print(f"Warning: failed reading local registry '{registry_json}': {e}")

    # If still no model, try the unversioned local model file
    if model is None:
        if os.path.exists(local_model_path):
            try:
                with open(local_model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"Loaded local model from: {local_model_path}")
            except Exception as e:
                print(f"Error: failed to load local model file '{local_model_path}': {e}")
                raise
        else:
            raise RuntimeError(
                f"No model available: registry returned version={model_version} and "
                f"no local model found. Please register a model or create the local model file."
            )

# Load vectorizer (required for preprocessing)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    # Clean text
    text = normalize_text(text)
    # Convert to features
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    result = model.predict(features_df)
    prediction = result[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker
