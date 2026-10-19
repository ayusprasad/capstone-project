# 🎓 Capstone Project: End-to-End MLOps Pipeline & Serving

An enterprise-grade MLOps repository showcasing **DVC pipeline orchestration**, **MLflow experiment tracking**, **Continuous Integration**, **Docker containerization**, and **Kubernetes deployment**.

---

## 🌟 Key Components
- **Data & Model Versioning (DVC):** Tracks datasets and intermediate models across ingestion, preprocessing, training, and evaluation stages defined in `dvc.yaml`.
- **Flask Serving API (`flask_app/`):** Real-time inference service with built-in Prometheus observability metrics (`/metrics`).
- **Pretrained Artifacts (`models/`):** Pretrained joblib and pickle models for immediate serving without requiring a full pipeline retrain.
- **Production Container:** Python 3.11 slim Dockerfile.

---

## 🚀 Quickstart

### 1. Run Serving App with Docker
```bash
docker build -t capstone-mlops-app .
docker run -p 5000:5000 capstone-mlops-app
```
Access the application at [http://localhost:5000](http://localhost:5000).

---

### 2. Local Standalone Serving
```bash
cd flask_app
pip install -r requirements.txt
python app.py
```

---

### 3. Run Full DVC Pipeline
```bash
pip install -r requirements.txt
dvc repro
```