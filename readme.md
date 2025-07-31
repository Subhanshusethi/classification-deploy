# 🐶🐱 Scratch Detection FastAPI ML Service

This project is a full-stack machine learning deployment pipeline using FastAPI, PyTorch, DVC, GitHub Actions, Docker, and Gmail alerts.

---

## ✅ Features Implemented

### 🔹 Phase 1 – Local ML App
- ✅ FastAPI app for image classification
- ✅ CLIP + custom classifier (predicts 'scratches' or 'no_scratches')
- ✅ Dockerized application (`Dockerfile`)
- ✅ Inference via `/upload_image` endpoint

### 🔹 Phase 2 – Versioning & Reproducibility
- ✅ Git-tracked code and structure
- ✅ DVC-tracked dataset (`classification_dataset/`)
- ✅ DVC-tracked model checkpoints (`ckpts/*.pth`)
- ✅ `dvc.yaml` pipeline for training
- ✅ MLflow logging for metrics and parameters

### 🔹 Phase 3 – CI/CD
- ✅ GitHub Actions:
  - Runs tests on every push
  - Pulls model/data from Google Drive using DVC + service account
  - Builds Docker image and pushes to Docker Hub
- ✅ App deployed to Render

### 🔹 Phase 4 – Monitoring & Alerts
- ✅ Logged predictions to `inference_log.txt`
- ✅ Gmail email alerts for image loading or inference errors

### 🔹 Phase 5 – Retraining Pipeline
- ✅ Supports retraining via `dvc repro`
- 🕒 GitHub Action trigger for auto-retraining: planned
- 🕒 A/B model testing: planned

---

## 📦 API Endpoint

### `POST /upload_image`
Upload an image file (JPG/PNG) via `multipart/form-data`

**Returns:**
```json
{
  "pred": "scratches",
  "prob_no_scratches": 0,
  "prob_scratches": 1
}
```

---

## 🚀 Deployment & Usage

### Local:
```bash
uvicorn main:app --reload
```

### Docker:
```bash
docker build -t scratch-detector .
docker run -p 8000:8000 scratch-detector
```

### Render:
- Deploy via GitHub integration using Dockerfile

---

## 🔐 Environment Variables

| Name | Description |
|------|-------------|
| `ALERT_EMAIL` | Gmail address used to send alerts |
| `ALERT_APP_PASSWORD` | Gmail App Password |
| `ALERT_TO` | Your receiving email address |

---

## 🧪 Testing

Run unit tests using:

```bash
pytest tests/
```
