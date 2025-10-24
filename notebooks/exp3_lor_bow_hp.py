import os
import re
import string
import traceback
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    "experiment_name": "LogisticRegression Hyperparameter Tuning",
    "local_model_dir": os.getenv("LOCAL_MODEL_DIR", "notebooks/models"),
    "cv_folds": 5,
    "scoring_metric": "f1"
}

os.makedirs(CONFIG["local_model_dir"], exist_ok=True)

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])

try:
    dagshub.init( # type: ignore
        repo_owner=CONFIG["dagshub_repo_owner"], 
        repo_name=CONFIG["dagshub_repo_name"], 
        mlflow=True
    )
    print("✓ DagHub initialized successfully")
except Exception as e:
    print(f"Warning: dagshub.init() issue: {e}")

mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def preprocess_text(text):
    """Applies comprehensive text preprocessing."""
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = " ".join([
            lemmatizer.lemmatize(word) 
            for word in text.split() 
            if word not in stop_words
        ])
        return text.strip()
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# ========================== LOAD & PREPARE DATA ==========================
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    try:
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must contain 'review' and 'sentiment' columns.")
        
        print("Preprocessing text...")
        df["review"] = df["review"].astype(str).apply(preprocess_text)
        
        df = df[df["sentiment"].isin(["positive", "negative"])]
        df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
        
        print(f"✓ Total samples: {len(df)}")
        print(f"✓ Positive samples: {(df['sentiment'] == 1).sum()}")
        print(f"✓ Negative samples: {(df['sentiment'] == 0).sum()}")
        
        print("Vectorizing text with TF-IDF...")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["review"])
        y = df["sentiment"]
        
        print(f"✓ Feature dimensions: {X.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=42
        )
        
        print(f"✓ Train samples: {X_train.shape[0]}")
        print(f"✓ Test samples: {X_test.shape[0]}")
        
        return (X_train, X_test, y_train, y_test), vectorizer
        
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        raise

# ========================== SAVE MODEL ARTIFACTS ==========================
def save_and_log_artifacts(model, vectorizer, params, run_name):
    """Save model and vectorizer as artifacts."""
    try:
        model_dir = os.path.join(CONFIG["local_model_dir"], "hyperparameter_tuning", run_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        joblib.dump(vectorizer, vectorizer_path)
        
        # Create info file
        info_path = os.path.join(model_dir, "model_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Logistic Regression - Hyperparameter Tuning\n")
            f.write(f"{'='*60}\n")
            f.write(f"Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nFiles:\n")
            f.write(f"  - model.joblib: Trained model\n")
            f.write(f"  - vectorizer.joblib: TF-IDF vectorizer\n")
        
        # Log artifacts
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(vectorizer_path, artifact_path="model")
        mlflow.log_artifact(info_path, artifact_path="model")
        
        print(f"✓ Saved artifacts for {run_name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to save artifacts: {e}")
        traceback.print_exc()
        return False

# ========================== TRAIN WITH GRIDSEARCH ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains Logistic Regression with GridSearch and logs all results to MLflow."""
    
    # Define hyperparameter grid
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [1000]
    }
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING - LOGISTIC REGRESSION")
    print(f"{'='*80}\n")
    
    with mlflow.start_run(run_name="GridSearch_LogisticRegression") as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"Parent Run ID: {parent_run_id}\n")
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=CONFIG["cv_folds"],
            scoring=CONFIG["scoring_metric"],
            n_jobs=-1,
            verbose=1
        )
        
        print("Running GridSearchCV...")
        grid_search.fit(X_train, y_train)
        print("✓ GridSearchCV completed\n")
        
        # Store all results
        all_results = []
        
        # Log each parameter combination
        print(f"{'='*80}")
        print("LOGGING ALL PARAMETER COMBINATIONS")
        print(f"{'='*80}\n")
        
        for idx, (params, mean_score, std_score) in enumerate(zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"]
        ), 1):
            
            run_name = f"LR_C{params['C']}_pen{params['penalty']}"
            
            with mlflow.start_run(run_name=run_name, nested=True):
                try:
                    # Train model with these params
                    model = LogisticRegression(**params)
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = {
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                        "mean_cv_f1": float(mean_score),
                        "std_cv_f1": float(std_score)
                    }
                    
                    # Log parameters
                    mlflow.log_params(params)
                    mlflow.log_params({
                        "cv_folds": CONFIG["cv_folds"],
                        "test_size": CONFIG["test_size"]
                    })
                    
                    # Log metrics
                    mlflow.log_metrics(metrics)
                    
                    # Save artifacts
                    save_and_log_artifacts(model, vectorizer, params, run_name)
                    
                    # Store for summary
                    result = {
                        "Run": idx,
                        "C": params["C"],
                        "Penalty": params["penalty"],
                        "Accuracy": metrics["accuracy"],
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1-Score": metrics["f1_score"],
                        "CV_F1_Mean": metrics["mean_cv_f1"],
                        "CV_F1_Std": metrics["std_cv_f1"]
                    }
                    all_results.append(result)
                    
                    print(f"[{idx}] C={params['C']:<6} Penalty={params['penalty']:<3} | "
                          f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1_score']:.4f} | "
                          f"CV_F1={metrics['mean_cv_f1']:.4f}±{metrics['std_cv_f1']:.4f}")
                    
                except Exception as e:
                    print(f"✗ Error with params {params}: {e}")
                    traceback.print_exc()
        
        # Log best model info in parent run
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_
        
        # Get test metrics for best model
        y_pred_best = best_model.predict(X_test)
        best_test_metrics = {
            "best_cv_f1_score": float(best_f1),
            "best_test_accuracy": float(accuracy_score(y_test, y_pred_best)),
            "best_test_precision": float(precision_score(y_test, y_pred_best, zero_division=0)),
            "best_test_recall": float(recall_score(y_test, y_pred_best, zero_division=0)),
            "best_test_f1_score": float(f1_score(y_test, y_pred_best, zero_division=0))
        }
        
        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metrics(best_test_metrics)
        
        # Save best model
        save_and_log_artifacts(best_model, vectorizer, best_params, "BEST_MODEL")
        
        # Create summary CSV
        summary_path = os.path.join(CONFIG["local_model_dir"], "hyperparameter_tuning_summary.csv")
        df_summary = pd.DataFrame(all_results)
        df_summary = df_summary.sort_values("F1-Score", ascending=False)
        df_summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="summary")
        
        # Print summary
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING RESULTS")
        print(f"{'='*80}\n")
        print(df_summary.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("🏆 BEST MODEL")
        print(f"{'='*80}")
        print(f"Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nCross-Validation F1 Score: {best_f1:.4f}")
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {best_test_metrics['best_test_accuracy']:.4f}")
        print(f"  Precision: {best_test_metrics['best_test_precision']:.4f}")
        print(f"  Recall:    {best_test_metrics['best_test_recall']:.4f}")
        print(f"  F1-Score:  {best_test_metrics['best_test_f1_score']:.4f}")
        print(f"{'='*80}\n")

# ========================== MAIN EXECUTION ==========================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION - HYPERPARAMETER TUNING")
    print("="*80 + "\n")
    
    # Load and prepare data
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data(CONFIG["data_path"])
    
    # Train and log models
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)
    
    print("\n" + "="*80)
    print("✓ HYPERPARAMETER TUNING COMPLETED")
    print("="*80)
    print(f"\n📊 View your experiments at:")
    print(f"   {CONFIG['mlflow_tracking_uri']}")
    print("\n" + "="*80 + "\n")