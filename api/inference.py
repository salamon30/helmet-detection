from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / "helmet_v3_best.pt"

HELMET_CLASSES = {"helmet", "Helmet"}
NO_HELMET_CLASSES = {"head", "no_helmet", "No_Helmet"}

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def predict(image_bytes: bytes, conf_threshold: float = 0.55) -> Dict:
    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")

    model = get_model()
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    result = results[0]
    names = result.names

    detections: List[Dict] = []
    counts = {"helmet": 0, "no_helmet": 0}

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            raw_class = names[cls_id]
            if raw_class not in HELMET_CLASSES | NO_HELMET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf.item()), 4)
            label = "helmet" if raw_class in HELMET_CLASSES else "no_helmet"
            counts[label] += 1

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    total = counts["helmet"] + counts["no_helmet"]
    compliance = round(counts["helmet"] / total, 4) if total > 0 else None

    return {
        "detections": detections,
        "counts": counts,
        "total": total,
        "compliance_rate": compliance,
    }
