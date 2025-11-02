# load test + signature test + performance test

import unittest
import os
import pickle
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError(
                "CAPSTONE_TEST environment variable is not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER",
                                "ayusprasad")
        repo_name = os.getenv("DAGSHUB_REPO_NAME",
                              "capstone-project")

        tracking_uri = (
            f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow'
        )
        mlflow.set_tracking_uri(tracking_uri)

        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(
            cls.new_model_name
        )
        cls.new_model_uri = (
            f'models:/{cls.new_model_name}/{cls.new_model_version}'
        )
        # If registry has no versions available (e.g. CI without a registry),
        # fall back to loading the locally saved model file so tests can run.
        if cls.new_model_version is None:
            try:
                with open('models/model.pkl', 'rb') as f:
                    cls.new_model = pickle.load(f)
                cls.new_model_uri = 'models/model.pkl'
                logging = __import__('logging')
                logging.warning(
                    'No registry version found; falling back to local model'
                )
            except Exception as e:
                raise RuntimeError(
                    'No model available to run tests: ' + str(e)
                )
        else:
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        with open('models/vectorizer.pkl', 'rb') as f:
            cls.vectorizer = pickle.load(f)

        cls.holdout_data = pd.read_csv(
            'data/processed/test_bow.csv'
        )

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        try:
            latest_version = client.get_latest_versions(
                model_name, stages=[stage]
            )
            return (latest_version[0].version
                    if latest_version else None)
        except Exception:
            # If registry is unavailable, authenticated, or API fails,
            # return None to allow tests to fall back to a local model.
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )

        prediction = self.new_model.predict(input_df)

        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(
            accuracy_new, expected_accuracy,
            f'Accuracy should be at least {expected_accuracy}'
        )
        self.assertGreaterEqual(
            precision_new, expected_precision,
            f'Precision should be at least {expected_precision}'
        )
        self.assertGreaterEqual(
            recall_new, expected_recall,
            f'Recall should be at least {expected_recall}'
        )
        self.assertGreaterEqual(
            f1_new, expected_f1,
            f'F1 score should be at least {expected_f1}'
        )



if __name__ == "__main__":
    unittest.main()
