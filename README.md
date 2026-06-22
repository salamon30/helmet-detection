# 🪖 Helmet Detection — PPE Compliance System

[![CI/CD](https://github.com/salamon30/helmet-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/salamon30/helmet-detection/actions/workflows/ci.yml)

Real-time helmet detection for construction sites using YOLOv8s. Detects whether workers are wearing helmets and tracks compliance rate.

## Live Demo

| | URL |
|--|--|
| **Streamlit Dashboard** | [helmet-streamlit.onrender.com](https://helmet-streamlit.onrender.com) |
| **REST API** | [helmet-detection-6kf7.onrender.com/docs](https://helmet-detection-6kf7.onrender.com/docs) |

> Free tier — first request may take ~30 seconds to wake up.

## Model Performance

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Helmet | 93.7% | 92.1% | 96.1% |
| No_Helmet | 91.1% | 88.3% | 93.7% |
| **Overall** | **92.4%** | **90.2%** | **94.9%** |

- Architecture: YOLOv8s (pretrained on ImageNet)
- Dataset: 6176 images — Kaggle hard-hat-detection + Roboflow PPE
- Training: Google Colab, Tesla T4 GPU, 50 epochs
- Inference speed: ~103 FPS

## Features

- Image upload — helmet detection on photos
- Video upload — frame-by-frame processing + download annotated video
- Live webcam — real-time detection via browser
- Compliance alarm — visual alert when compliance drops below 50%
- Model Analysis tab — training curves, per-class metrics

## API Usage

```bash
# Health check
curl https://helmet-detection-6kf7.onrender.com/health

# Run detection on an image
curl -X POST https://helmet-detection-6kf7.onrender.com/predict \
  -F "file=@your_image.jpg"
```

**Example response:**
```json
{
  "detections": [
    {"label": "helmet", "confidence": 0.92, "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150}},
    {"label": "no_helmet", "confidence": 0.85, "bbox": {"x1": 300, "y1": 60, "x2": 380, "y2": 140}}
  ],
  "counts": {"helmet": 1, "no_helmet": 1},
  "total": 2,
  "compliance_rate": 0.5
}
```

## Project Structure

```
helmet-detection/
├── app.py                   # Streamlit dashboard
├── api/
│   ├── main.py              # FastAPI app (/health, /predict)
│   └── inference.py         # YOLOv8 inference module
├── notebooks/
│   ├── train_yolov8.ipynb       # YOLOv8s training (Colab)
│   └── train_faster_rcnn.ipynb  # Faster R-CNN training (Colab)
├── Dockerfile               # API container
├── Dockerfile.streamlit     # Streamlit container
├── requirements.txt         # Streamlit dependencies
├── requirements-api.txt     # API dependencies
└── helmet_v3_best.pt        # Trained model weights
```

## Local Setup

```bash
git clone https://github.com/salamon30/helmet-detection.git
cd helmet-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run.sh
```

## Docker

```bash
# Build and run API
docker build -t helmet-api .
docker run -p 8000:8000 helmet-api

# Build and run Streamlit
docker build -f Dockerfile.streamlit -t helmet-streamlit .
docker run -p 8501:8501 helmet-streamlit
```

## Training

Colab notebooks are in `notebooks/`. Update your Kaggle API key and Drive paths before running.

- [YOLOv8s notebook](notebooks/train_yolov8.ipynb)
- [Faster R-CNN notebook](notebooks/train_faster_rcnn.ipynb)
