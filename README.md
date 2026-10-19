# Sentiment Analysis Lab

[![CI](https://github.com/ayusprasad/capstone-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/capstone-project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20app-000000?logo=flask&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Model%20tracking-0194E2?logo=mlflow&logoColor=white)
![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

A movie-review sentiment classifier with a Flask interface, reusable model artifacts, and an MLflow/DVC experimentation workflow.

## Features

- Submit a review and receive a positive or negative sentiment prediction.
- Train and compare model experiments with MLflow tracking.
- Keep datasets and model artifacts versioned through DVC and the model registry.
- Run the app locally with checked-in artifacts; remote registry integration is optional.

## Architecture

~~~text
Flask UI → text normalization → vectorizer → sentiment classifier
Training code and notebooks → MLflow registry / DVC artifacts
~~~

## Local setup

### Prerequisites

- Python 3.11 or newer
- Git
- Optional DVC remote and MLflow registry credentials for experiments

### Install

~~~bash
git clone https://github.com/ayusprasad/capstone-project.git
cd capstone-project
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Activate the environment (source .venv/bin/activate on macOS/Linux or .venv/Scripts/Activate.ps1 in PowerShell), then run the Flask app:

~~~bash
python flask_app/app.py
~~~

Open the local address printed by Flask. Text prediction uses the bundled model artifacts. Configure .env from .env.example only if connecting to an MLflow registry.

## Quality checks

~~~bash
python -m pip install ruff pytest
ruff check --select E4,E7,E9,F tests flask_app/app.py
python -m nltk.downloader stopwords wordnet
python -m pytest -q
python -m compileall -q flask_app src tests
docker build -t sentiment-analysis .
~~~

GitHub Actions runs lint, tests, compilation, and a Docker build on pushes and pull requests. The separate AWS deployment workflow is available only through manual dispatch.

## Security

Keep registry credentials and AWS deployment secrets in local environment files or GitHub Actions secrets. Never commit real tokens. CI uses bundled model artifacts and does not contact a hosted registry.
