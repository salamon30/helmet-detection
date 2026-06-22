from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import av
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "helmet_v3_best.pt"

HELMET_CLASSES = {"helmet", "Helmet"}
NO_HELMET_CLASSES = {"head", "no_helmet", "No_Helmet"}

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "helmet":    (34, 197, 94),
    "Helmet":    (34, 197, 94),
    "head":      (220, 38, 38),
    "no_helmet": (220, 38, 38),
    "No_Helmet": (220, 38, 38),
}

# Training curve data extracted from Colab log (50 epochs, mAP50)
TRAIN_MAP50 = [
    0.740, 0.753, 0.829, 0.840, 0.843, 0.871, 0.866, 0.880, 0.876, 0.897,
    0.908, 0.883, 0.908, 0.907, 0.915, 0.899, 0.918, 0.916, 0.914, 0.920,
    0.924, 0.929, 0.930, 0.933, 0.934, 0.929, 0.933, 0.938, 0.939, 0.936,
    0.934, 0.940, 0.938, 0.942, 0.940, 0.945, 0.942, 0.943, 0.945, 0.943,
    0.940, 0.944, 0.949, 0.946, 0.946, 0.942, 0.948, 0.949, 0.948, 0.949,
]

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Helmet Detection Dashboard",
    page_icon="🪖",
    layout="wide",
)

# ─── Styles ───────────────────────────────────────────────────────────────────

def inject_styles() -> None:
    pass


# ─── Helpers ──────────────────────────────────────────────────────────────────

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


def _show_alarm(rate: float) -> None:
    if rate < 0.5:
        st.error(f"⚠️ COMPLIANCE ALARM — Helmet compliance is critically low: {rate:.0%} — immediate action required!")


def _show_metrics(counts: Dict[str, int]) -> None:
    helmets = counts.get("helmet", 0)
    heads   = counts.get("head", 0)
    total   = helmets + heads
    rate    = helmets / total if total > 0 else 0.0

    _show_alarm(rate)

    css = _compliance_css(rate)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Helmet ✅", helmets)
    c2.metric("No Helmet ⚠️", heads)
    c3.metric("Total", total)
    c4.metric("COMPLIANCE", f"{rate:.0%}")


def predict_frame(
    model: YOLO, frame: np.ndarray, conf_threshold: float
) -> Tuple[np.ndarray, Dict[str, int]]:
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

    with open(out_path, "rb") as f:
        st.download_button(
            label="Download annotated video",
            data=f,
            file_name="helmet_detection_output.mp4",
            mime="video/mp4",
        )

    st.markdown("**Peak counts (single frame)**")
    _show_metrics(max_counts)


RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
}


def webcam_mode(conf_threshold: float) -> None:
    st.markdown("### Live webcam detection")
    st.info("**Cloud limitation:** WebRTC is not supported on the hosted version. To use webcam detection, run the app locally with `./run.sh` and open `localhost:8501`.")
    st.caption("Grant camera access in your browser, then press **Start**.")

    webrtc_ctx = webrtc_streamer(
        key="helmet-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HelmetVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence = conf_threshold
        _show_metrics(webrtc_ctx.video_processor.latest_counts)


# ─── Model Analysis Tab ───────────────────────────────────────────────────────

def model_analysis_tab() -> None:
    st.markdown("## Model Performance")

    # ── Metrics cards ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Overall")
        st.metric("mAP50",    "94.9%")
        st.metric("mAP50-95", "61.5%")
        st.metric("Precision", "92.4%")
        st.metric("Recall",    "90.2%")

    with col2:
        st.markdown("#### 🟢 Helmet")
        st.metric("mAP50",    "96.1%")
        st.metric("mAP50-95", "61.8%")
        st.metric("Precision", "93.7%")
        st.metric("Recall",    "92.1%")

    with col3:
        st.markdown("#### 🔴 No Helmet")
        st.metric("mAP50",    "93.7%")
        st.metric("mAP50-95", "61.2%")
        st.metric("Precision", "91.1%")
        st.metric("Recall",    "88.3%")

    st.divider()

    # ── Training curve ──
    st.markdown("#### Training Curve — mAP50 over 50 Epochs")

    epochs = list(range(1, 51))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=TRAIN_MAP50,
        mode="lines+markers",
        name="mAP50",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=4),
        hovertemplate="Epoch %{x}<br>mAP50: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=0.949,
        line_dash="dash",
        line_color="#16a34a",
        annotation_text="Best: 0.949",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="mAP50",
        yaxis=dict(range=[0.70, 0.98]),
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        plot_bgcolor="#f9fafb",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Training config ──
    st.markdown("#### Training Configuration")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
| Parameter | Value |
|-----------|-------|
| Model | YOLOv8s |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
""")
    c2.markdown("""
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR₀ | 0.001667 |
| Patience | 10 |
| GPU | Tesla T4 |
""")
    c3.markdown("""
| Dataset | Images |
|---------|--------|
| Kaggle hard-hat | ~5 000 |
| Roboflow PPE | ~1 200 |
| **Total** | **6 176** |
| Train / Val / Test | 70/20/10% |
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_styles()

    st.title("🪖 Helmet Detection Dashboard")
    st.caption("YOLOv8s · Real-time PPE compliance for construction sites")

    tab_detect, tab_analysis = st.tabs(["Detection", "Model Analysis"])

    with tab_detect:
        if not MODEL_PATH.exists():
            st.error(f"Model not found: `{MODEL_PATH}`.")
            st.stop()

        with st.sidebar:
            st.markdown("## ⚙️ Settings")
            conf_threshold = st.slider(
                "Confidence threshold",
                min_value=0.10,
                max_value=0.95,
                value=0.55,
                step=0.05,
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

    with tab_analysis:
        model_analysis_tab()


if __name__ == "__main__":
    main()
