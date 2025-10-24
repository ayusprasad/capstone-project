import os
import re
import string
import traceback
import joblib
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG = {
    "data_path": os.getenv("DATA_PATH", "notebooks/data.csv"),
    "test_size": 0.2,
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
    "dagshub_repo_owner": os.getenv("DAGSHUB_REPO_OWNER"),
    "dagshub_repo_name": os.getenv("DAGSHUB_REPO_NAME"),
    "experiment_name": "Bow vs TfIdf Comparison",  # CHANGED: New experiment name
    "local_model_dir": os.getenv("LOCAL_MODEL_DIR", "notebooks/models")
}

os.makedirs(CONFIG["local_model_dir"], exist_ok=True)

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])

try:
    dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True) # type: ignore
    print("DagHub initialized successfully")
except Exception as e:
    print(f"Warning: dagshub.init() issue: {e}")

# Use a NEW experiment name to avoid the deleted experiment error
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        return text
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].astype(str).apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must contain 'review' and 'sentiment' columns.")
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).astype(int)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# ========================== HELPERS ==========================
def log_model_params(algo_name, model):
    """Log model hyperparameters to MLflow"""
    params_to_log = {}
    try:
        if algo_name == 'LogisticRegression':
            params_to_log["C"] = getattr(model, "C", "unknown")
            params_to_log["max_iter"] = getattr(model, "max_iter", "unknown")
            params_to_log["solver"] = getattr(model, "solver", "unknown")
        elif algo_name == 'MultinomialNB':
            params_to_log["alpha"] = getattr(model, "alpha", "unknown")
        elif algo_name == 'XGBoost':
            params_to_log["n_estimators"] = getattr(model, "n_estimators", "unknown")
            params_to_log["learning_rate"] = getattr(model, "learning_rate", "unknown")
            params_to_log["max_depth"] = getattr(model, "max_depth", "unknown")
        elif algo_name == 'RandomForest':
            params_to_log["n_estimators"] = getattr(model, "n_estimators", "unknown")
            params_to_log["max_depth"] = getattr(model, "max_depth", "unknown")
            params_to_log["min_samples_split"] = getattr(model, "min_samples_split", "unknown")
        elif algo_name == 'GradientBoosting':
            params_to_log["n_estimators"] = getattr(model, "n_estimators", "unknown")
            params_to_log["learning_rate"] = getattr(model, "learning_rate", "unknown")
            params_to_log["max_depth"] = getattr(model, "max_depth", "unknown")

        if params_to_log:
            mlflow.log_params(params_to_log)
    except Exception as e:
        print(f"Failed to log params for {algo_name}: {e}")

def save_and_log_model(model, vectorizer, model_name, algo_name):
    """
    Save model and vectorizer locally, then upload as artifacts to MLflow.
    This approach works with DagHub's MLflow server.
    """
    try:
        # Create a temporary directory for this model
        model_dir = os.path.join(CONFIG["local_model_dir"], model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Saved vectorizer to: {vectorizer_path}")
        
        # Create a model info file
        info_path = os.path.join(model_dir, "model_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Model: {algo_name}\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Files:\n")
            f.write(f"  - model.joblib: Trained model\n")
            f.write(f"  - vectorizer.joblib: Text vectorizer\n")
            f.write(f"\nUsage:\n")
            f.write(f"import joblib\n")
            f.write(f"model = joblib.load('model.joblib')\n")
            f.write(f"vectorizer = joblib.load('vectorizer.joblib')\n")
            f.write(f"\n# Make predictions\n")
            f.write(f"new_text = ['Your text here']\n")
            f.write(f"features = vectorizer.transform(new_text)\n")
            f.write(f"prediction = model.predict(features)\n")
        
        # Log all files as artifacts
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(vectorizer_path, artifact_path="model")
        mlflow.log_artifact(info_path, artifact_path="model")
        
        print(f"✓ Successfully uploaded model artifacts for {model_name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to save/log model {model_name}: {e}")
        traceback.print_exc()
        return False

def create_metrics_summary(all_results):
    """Create a summary CSV of all experiments"""
    try:
        summary_path = os.path.join(CONFIG["local_model_dir"], "experiments_summary.csv")
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(summary_path, index=False)
        print(f"\nExperiments summary saved to: {summary_path}")
        return summary_path
    except Exception as e:
        print(f"Failed to create summary: {e}")
        return None

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    all_results = []
    
    with mlflow.start_run(run_name="Bow_vs_TfIdf_Comparison") as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"\n{'='*80}")
        print(f"Starting Parent Run: {parent_run_id}")
        print(f"{'='*80}\n")
        
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                run_name = f"{algo_name}_with_{vec_name}"
                
                with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                    try:
                        print(f"\n{'-'*80}")
                        print(f"Training: {run_name}")
                        print(f"{'-'*80}")
                        
                        # Create fresh instances
                        vec = vectorizer.__class__(**vectorizer.get_params())
                        model = algorithm.__class__(**algorithm.get_params())
                        
                        # Prepare data
                        X = vec.fit_transform(df['review'])
                        y = df['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=CONFIG["test_size"], random_state=42
                        )
                        
                        # Log preprocessing params (fixed for sparse matrices)
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"],
                            "train_size": X_train.shape[0],  # Fixed: use shape[0] instead of len()
                            "test_size_samples": X_test.shape[0],  # Fixed: use shape[0] instead of len()
                            "total_features": X.shape[1]
                        })
                        
                        # Train
                        print(f"Training {algo_name}...")
                        model.fit(X_train, y_train)
                        
                        # Log model hyperparameters
                        log_model_params(algo_name, model)
                        
                        # Evaluate
                        print(f"Evaluating {algo_name}...")
                        y_pred = model.predict(X_test)
                        
                        metrics = {
                            "accuracy": float(accuracy_score(y_test, y_pred)),
                            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                            "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
                        }
                        
                        mlflow.log_metrics(metrics)
                        
                        # Save and log model artifacts
                        model_name = f"{algo_name}_{vec_name}"
                        save_and_log_model(model, vec, model_name, algo_name)
                        
                        # Store results for summary
                        result = {
                            "Algorithm": algo_name,
                            "Vectorizer": vec_name,
                            "Accuracy": metrics["accuracy"],
                            "Precision": metrics["precision"],
                            "Recall": metrics["recall"],
                            "F1-Score": metrics["f1_score"],
                            "Run_ID": child_run.info.run_id
                        }
                        all_results.append(result)
                        
                        # Print results
                        print(f"\n{'='*50}")
                        print(f"Results for {algo_name} with {vec_name}:")
                        print(f"{'='*50}")
                        print(f"Accuracy:  {metrics['accuracy']:.4f}")
                        print(f"Precision: {metrics['precision']:.4f}")
                        print(f"Recall:    {metrics['recall']:.4f}")
                        print(f"F1-Score:  {metrics['f1_score']:.4f}")
                        print(f"{'='*50}\n")
                        
                    except Exception as e:
                        print(f"\n✗ Error training {algo_name} with {vec_name}: {e}")
                        traceback.print_exc()
                        try:
                            mlflow.log_param("error", str(e)[:250])
                        except:
                            pass
        
        # Create and log summary
        if all_results:
            summary_path = create_metrics_summary(all_results)
            if summary_path:
                try:
                    mlflow.log_artifact(summary_path, artifact_path="summary")
                    
                    # Print final summary
                    print(f"\n{'='*80}")
                    print("FINAL SUMMARY - ALL EXPERIMENTS")
                    print(f"{'='*80}\n")
                    df_results = pd.DataFrame(all_results)
                    print(df_results.to_string(index=False))
                    print(f"\n{'='*80}")
                    
                    # Find best model
                    best_model = df_results.loc[df_results['F1-Score'].idxmax()]
                    print(f"\n🏆 BEST MODEL:")
                    print(f"   Algorithm: {best_model['Algorithm']}")
                    print(f"   Vectorizer: {best_model['Vectorizer']}")
                    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
                    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
                    print(f"{'='*80}\n")
                    
                except Exception as e:
                    print(f"Failed to log summary: {e}")

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS - BOW vs TF-IDF COMPARISON")
    print("="*80 + "\n")
    
    print(f"Loading data from: {CONFIG['data_path']}")
    df = load_data(CONFIG["data_path"])
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Positive samples: {(df['sentiment'] == 1).sum()}")
    print(f"✓ Negative samples: {(df['sentiment'] == 0).sum()}")
    
    train_and_evaluate(df)
    
    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\n📊 View your experiments at:")
    print(f"   {CONFIG['mlflow_tracking_uri']}")
    print("\n" + "="*80 + "\n")