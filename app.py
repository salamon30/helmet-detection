from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "helmet_v3_best.pt"

# Supports both naming conventions: "head" (v1/v2) and "no_helmet" (Berke v3+)
HELMET_CLASSES = {"helmet", "Helmet"}
NO_HELMET_CLASSES = {"head", "no_helmet", "No_Helmet"}

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "helmet":    (34, 197, 94),
    "Helmet":    (34, 197, 94),
    "head":      (220, 38, 38),
    "no_helmet": (220, 38, 38),
    "No_Helmet": (220, 38, 38),
}

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Helmet Detection Dashboard",
    page_icon="🪖",
    layout="wide",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main-title { font-size: 2.0rem; font-weight: 700; margin-bottom: 0.25rem; }
            .subtitle { color: #6b7280; margin-bottom: 1rem; }
            .block-card {
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 1rem 1.1rem;
                background: #ffffff;
            }
            .compliance-safe   { color: #16a34a; font-weight: 700; font-size: 1.5rem; text-align: center; }
            .compliance-warn   { color: #d97706; font-weight: 700; font-size: 1.5rem; text-align: center; }
            .compliance-danger { color: #dc2626; font-weight: 700; font-size: 1.5rem; text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compliance_css(rate: float) -> str:
    if rate >= 0.9:
        return "compliance-safe"
    if rate >= 0.5:
        return "compliance-warn"
    return "compliance-danger"


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _normalise(class_name: str) -> str:
    return "head" if class_name.lower() in {"no_helmet", "head"} else "helmet"


def draw_detections(
    frame: np.ndarray,
    detections: List[Tuple[int, int, int, int, float, str]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Draw bounding boxes and return per-class counts."""
    counts: Dict[str, int] = {"helmet": 0, "head": 0}
    output = frame.copy()

    for x1, y1, x2, y2, conf, raw_class in detections:
        display = _normalise(raw_class)
        color = CLASS_COLORS.get(raw_class, (255, 255, 255))
        counts[display] = counts.get(display, 0) + 1

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{display} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            output, label, (x1 + 3, max(y1 - 5, th + 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    total = counts["helmet"] + counts["head"]
    rate = counts["helmet"] / total if total > 0 else 0.0
    hud = f"Helmet: {counts['helmet']}  Head: {counts['head']}  Compliance: {rate:.0%}"
    cv2.putText(output, hud, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, hud, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return output, counts


def _show_metrics(counts: Dict[str, int]) -> None:
    helmets = counts.get("helmet", 0)
    heads   = counts.get("head", 0)
    total   = helmets + heads
    rate    = helmets / total if total > 0 else 0.0
    css     = _compliance_css(rate)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Helmet ✅", helmets)
    c2.metric("No Helmet ⚠️", heads)
    c3.metric("Total", total)
    c4.markdown(
        f'<p style="font-size:.75rem;color:#6b7280;margin:0">COMPLIANCE</p>'
        f'<p class="{css}">{rate:.0%}</p>',
        unsafe_allow_html=True,
    )


def predict_frame(
    model: YOLO, frame: np.ndarray, conf_threshold: float
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Run inference on a single BGR frame and return annotated frame + counts."""
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    result = results[0]
    names = result.names

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            class_name = names[cls_id]
            if class_name not in HELMET_CLASSES | NO_HELMET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf.item())
            detections.append((x1, y1, x2, y2, conf, class_name))

    return draw_detections(frame, detections)


# ─── Detection Modes ──────────────────────────────────────────────────────────

class HelmetVideoProcessor(VideoProcessorBase):
    confidence: float = 0.55

    def __init__(self) -> None:
        self._model = load_model(str(MODEL_PATH))
        self.latest_counts: Dict[str, int] = {"helmet": 0, "head": 0}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated, counts = predict_frame(self._model, img, self.confidence)
        self.latest_counts = counts
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def image_mode(model: YOLO, conf_threshold: float) -> None:
    st.markdown("### Upload an image")
    uploaded = st.file_uploader(
        "JPG / PNG / BMP / WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated, counts = predict_frame(model, frame, conf_threshold)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    _show_metrics(counts)


def video_mode(model: YOLO, conf_threshold: float) -> None:
    st.markdown("### Upload a video")
    uploaded = st.file_uploader(
        "Supported formats: MP4, AVI, MOV, MKV",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    if not uploaded:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as in_tmp:
        in_tmp.write(uploaded.read())
        in_path = in_tmp.name

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("Unable to open video file.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_tmp:
        out_path = out_tmp.name

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    progress = st.progress(0)
    status   = st.empty()
    max_counts: Dict[str, int] = {"helmet": 0, "head": 0}

    for idx in range(1, total + 1 if total > 0 else int(1e9)):
        ok, frame = cap.read()
        if not ok:
            break
        annotated, counts = predict_frame(model, frame, conf_threshold)
        writer.write(annotated)
        max_counts["helmet"] = max(max_counts["helmet"], counts.get("helmet", 0))
        max_counts["head"]   = max(max_counts["head"],   counts.get("head",   0))
        if total > 0:
            progress.progress(min(idx / total, 1.0))
        status.text(f"Processing frame {idx}/{total if total > 0 else '?'}")

    cap.release()
    writer.release()
    progress.empty()
    status.empty()

    st.success("Processing complete.")
    st.video(out_path)
    st.markdown("**Peak counts (single frame)**")
    _show_metrics(max_counts)


def webcam_mode(conf_threshold: float) -> None:
    st.markdown("### Live webcam detection")
    st.caption("Grant camera access in your browser, then press **Start**.")

    webrtc_ctx = webrtc_streamer(
        key="helmet-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HelmetVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence = conf_threshold
        _show_metrics(webrtc_ctx.video_processor.latest_counts)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_styles()

    st.markdown('<div class="main-title">🪖 Helmet Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">YOLOv8 · Real-time PPE compliance for construction sites — images, videos & webcam</div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error(f"Model not found: `{MODEL_PATH}`. Place `best.pt` in the `models/` folder.")
        st.stop()

    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.55,
            step=0.05,
            help="Raise to reduce false positives. 0.55 is a good starting point.",
        )
        mode = st.radio(
            "Mode",
            options=["Image", "Video", "Webcam"],
            captions=["Upload a photo", "Upload a video clip", "Real-time camera"],
        )
        st.markdown("---")
        st.markdown("**Legend**")
        st.markdown("🟢 `helmet` — PPE compliant")
        st.markdown("🔴 `head` — No helmet")
        st.markdown("---")
        st.caption(f"Model: `{MODEL_PATH.name}`")

    model = load_model(str(MODEL_PATH))

    if mode == "Image":
        image_mode(model, conf_threshold)
    elif mode == "Video":
        video_mode(model, conf_threshold)
    else:
        webcam_mode(conf_threshold)


if __name__ == "__main__":
    main()
