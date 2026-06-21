# Helmet Detection Dashboard

## Live API
**Base URL:** `https://helmet-detection-6kf7.onrender.com`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Run helmet detection on an image |
| `/docs` | GET | Interactive Swagger UI |

**Quick test:**
```bash
curl -X POST https://helmet-detection-6kf7.onrender.com/predict \
  -F "file=@your_image.jpg"
```

> Note: Free tier spins down after inactivity — first request may take ~30 seconds.

Real-time PPE (Personal Protective Equipment) helmet detection using YOLOv8s. Detects whether workers are wearing helmets on construction sites.

## Features

- **Image upload** — detect helmets in photos
- **Video upload** — process video files frame by frame
- **Live webcam** — real-time detection via browser camera
- **Compliance tracking** — shows helmet/no-helmet counts and compliance rate

## Model

- Architecture: YOLOv8s
- Dataset: 6177 images (Kaggle hard-hat-detection + Roboflow PPE dataset)
- mAP50: **0.949**
- Classes: `Helmet`, `No_Helmet`

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/salamon30/helmet-detection.git
cd helmet-detection
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the model

Download `helmet_v3_best.pt` and place it in the project folder:

```
helmet-detection/
├── app.py
├── requirements.txt
├── helmet_v3_best.pt   ← place here
└── README.md
```

> The model is not included in the repository due to file size. Train your own using the Colab notebook or request the file from the project owner.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Training

The model was trained on Google Colab with Tesla T4 GPU using:

- Base model: `yolov8s.pt`
- Epochs: 50
- Image size: 640
- Batch size: 16
- Early stopping patience: 10

## Results

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Helmet | 0.937 | 0.921 | 0.961 |
| No_Helmet | 0.911 | 0.883 | 0.937 |
| **Overall** | **0.924** | **0.902** | **0.949** |
