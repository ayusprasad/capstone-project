from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError
try:
    from prometheus_client import (
        Counter,
        Histogram,
        generate_latest,
        CollectorRegistry,
        CONTENT_TYPE_LATEST,
    )
except Exception:
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

# Load environment variables
project_root = Path(__file__).resolve().parents[1]
load_dotenv = dotenv.load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

dagshub_token = os.getenv("MLFLOW_TRACKING_URI")
dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")

# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define custom metrics
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", 
    ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", 
    ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", 
    ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "my_model"

def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        print(f"Warning: could not fetch registered model versions: {e}")
        return None

model_version = get_latest_model_version(model_name)

# Try loading from MLflow Model Registry
model = None
if model_version:
    model_uri = f'models:/{model_name}/{model_version}'
    print(f"Fetching model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model from registry: {model_name} v{model_version}")
    except Exception as e:
        print(f"Warning: failed to load model from registry ({e}). Falling back to local model.")
else:
    print(f"No model version found in registry for '{model_name}'. Trying local model file.")

# Local fallback - try local registry first
if model is None:
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

# Try loading from models/model.pkl or models/model.joblib
if model is None:
    # Try .joblib first
    joblib_path = 'models/model.joblib'
    pkl_path = 'models/model.pkl'
    
    if os.path.exists(joblib_path):
        try:
            model = joblib.load(joblib_path)
            print(f"Loaded local model from: {joblib_path}")
        except Exception as e:
            print(f"Error loading {joblib_path}: {e}")
    
    if model is None and os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded local model from: {pkl_path}")
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            raise
    
    if model is None:
        raise RuntimeError(
            f"No model available: registry returned version={model_version} and "
            f"no local model found. Please register a model or create the local model file."
        )

# Load vectorizer (CRITICAL: Don't overwrite model here!)
vectorizer = None
vec_pkl_path = 'models/vectorizer.pkl'
vec_joblib_path = 'models/vectorizer.joblib'

# Try loading vectorizer from .pkl first
if os.path.exists(vec_pkl_path):
    try:
        vectorizer = joblib.load(vec_pkl_path)
        print(f"Loaded vectorizer with joblib from: {vec_pkl_path}")
    except Exception:
        try:
            with open(vec_pkl_path, 'rb') as vf:
                vectorizer = pickle.load(vf)
            print(f"Loaded vectorizer with pickle from: {vec_pkl_path}")
        except Exception as e:
            print(f"Error loading vectorizer from {vec_pkl_path}: {e}")

# Try .joblib if .pkl failed
if vectorizer is None and os.path.exists(vec_joblib_path):
    try:
        vectorizer = joblib.load(vec_joblib_path)
        print(f"Loaded vectorizer from: {vec_joblib_path}")
    except Exception as e:
        print(f"Error loading vectorizer from {vec_joblib_path}: {e}")

if vectorizer is None:
    print(f"Warning: No vectorizer file found at {vec_pkl_path} or {vec_joblib_path}")
else:
    # Verify vectorizer is fitted
    try:
        if hasattr(vectorizer, 'vocabulary_') and vectorizer.vocabulary_:
            print(f"✓ Vectorizer loaded successfully with {len(vectorizer.vocabulary_)} features")
        else:
            print("WARNING: Vectorizer has no vocabulary!")
    except Exception as e:
        print(f"Warning: Could not verify vectorizer vocabulary: {e}")

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

    try:
        text = request.form.get("text", "")
        if not text:
            return render_template("index.html", result="Error: No text provided"), 400
        
        # Clean text
        cleaned_text = normalize_text(text)
        
        # Check if vectorizer exists
        if vectorizer is None:
            print("Error: Vectorizer not loaded")
            return render_template("index.html", result="Error: Vectorizer not available"), 500
        
        # Convert to features
        try:
            features = vectorizer.transform([cleaned_text])
        except NotFittedError as nfe:
            print(f"Vectorizer not fitted: {nfe}")
            # Try fallback: pass raw text directly to model
            try:
                print("Attempting fallback: pass text directly to model.predict")
                result = model.predict([cleaned_text])
                prediction = result[0]
                PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
                REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
                return render_template("index.html", result=prediction)
            except Exception as e:
                print(f"Fallback prediction failed: {e}")
                return render_template("index.html", 
                                     result="Error: Vectorizer not fitted properly"), 500
        except Exception as e:
            print(f"Error transforming text: {e}")
            return render_template("index.html", 
                                 result=f"Error processing text: {str(e)}"), 500

        # Convert to DataFrame for prediction
        features_df = pd.DataFrame(
            features.toarray(), 
            columns=[str(i) for i in range(features.shape[1])]
        )

        # Predict
        try:
            result = model.predict(features_df)
            prediction = result[0]
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return render_template("index.html", 
                                 result=f"Error during prediction: {str(e)}"), 500

        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return render_template("index.html", result=prediction)
    
    except Exception as e:
        print(f"Unexpected error in /predict: {e}")
        return render_template("index.html", result=f"Unexpected error: {str(e)}"), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)