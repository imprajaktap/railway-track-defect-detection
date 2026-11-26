
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Railway AI | Defect Monitoring",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "railway"
    / "yolov8n_baseline"
    / "weights"
    / "best.pt"
)

DATASET_CONFIG = BASE_DIR / "railway_data.yaml"

CLASS_NAMES = {
    0: "Spalling",
    1: "Wheel Burn",
    2: "Squat"
}

# Actual independent test-set results
TEST_METRICS = {
    "Precision": 66.09,
    "Recall": 71.65,
    "mAP@50": 70.03,
    "mAP@50-95": 34.06
}

CLASS_METRICS = pd.DataFrame({
    "Defect": ["Spalling", "Wheel Burn", "Squat"],
    "Precision": [60.5, 71.7, 66.0],
    "Recall": [72.8, 80.2, 61.9],
    "mAP@50": [71.6, 75.8, 62.7],
    "mAP@50-95": [36.8, 37.4, 27.9]
})


# ============================================================
# 2. CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1018;
    color: #edf2f7;
}

[data-testid="stSidebar"] {
    background: #111925;
    border-right: 1px solid #263445;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f2f6fc !important;
}

p, label {
    color: #aab7c8;
}

.metric-card {
    background: linear-gradient(145deg, #172333, #111a27);
    border: 1px solid #293b50;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #9aabc0;
    font-size: 13px;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 29px;
    font-weight: 700;
    color: #f4f7fb;
}

.metric-caption {
    color: #8295aa;
    font-size: 12px;
    margin-top: 5px;
}

.section-card {
    background: #111b28;
    border: 1px solid #293b50;
    border-radius: 14px;
    padding: 20px;
}

.status-dot {
    color: #42d392;
    font-weight: 600;
}

.small-muted {
    color: #8295aa;
    font-size: 12px;
}

div[data-testid="stMetric"] {
    background: #142131;
    border: 1px solid #293b50;
    padding: 16px;
    border-radius: 12px;
}

.stButton > button {
    background: #2878e5;
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1d66c5;
    color: white;
}

hr {
    border-color: #293b50;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# ============================================================
# 4. MODEL AND FILE VALIDATION
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "Trained model checkpoint was not found."
    )

    st.code(str(MODEL_PATH))

    st.info(
        "Check that best.pt exists in the model weights directory."
    )

    st.stop()

if not DATASET_CONFIG.exists():
    st.error("railway_data.yaml was not found.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# 5. DETECTION FUNCTIONS
# ============================================================

def detect_image(image, confidence):

    image_array = np.array(image.convert("RGB"))

    results = model.predict(
        source=image_array,
        conf=confidence,
        imgsz=416,
        device="cpu",
        verbose=False
    )

    result = results[0]

    annotated_image = result.plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            coordinates = box.xyxy[0].tolist()

            detections.append({
                "Defect": result.names[class_id],
                "Confidence": round(score * 100, 2),
                "X1": round(coordinates[0], 2),
                "Y1": round(coordinates[1], 2),
                "X2": round(coordinates[2], 2),
                "Y2": round(coordinates[3], 2)
            })

    return annotated_image, detections


def detect_video(input_path, output_path, confidence, progress):

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to create output video.")

    detections = []
    frame_number = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=416,
                device="cpu",
                verbose=False
            )

            result = results[0]

            annotated_frame = result.plot()

            writer.write(annotated_frame)

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())

                    detections.append({
                        "Frame": frame_number + 1,
                        "Defect": result.names[class_id],
                        "Confidence": round(score * 100, 2)
                    })

            frame_number += 1

            if total_frames > 0:

                progress.progress(
                    min(frame_number / total_frames, 1.0),
                    text=f"Processing frame {frame_number} of {total_frames}"
                )

    finally:

        cap.release()
        writer.release()

    progress.progress(1.0, text="Processing completed.")

    return detections, frame_number


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚆 RAILWAY AI")

    st.caption("INTELLIGENT TRACK MONITORING")

    st.divider()

    st.markdown("### Model status")

    st.markdown(
        '<p class="status-dot">● Model loaded successfully</p>',
        unsafe_allow_html=True
    )

    st.caption("YOLOv8 Nano | CPU Inference")

    st.divider()

    st.markdown("### Detection settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Detections below this confidence are filtered out."
    )

    st.caption(
        f"Current threshold: {confidence * 100:.0f}%"
    )

    st.divider()

    st.markdown("### Supported defects")

    st.write("01 · Spalling")
    st.write("02 · Wheel Burn")
    st.write("03 · Squat")

    st.divider()

    st.caption("Academic prototype | YOLOv8n")


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    '<p class="small-muted">RAILWAY INFRASTRUCTURE / AI MONITORING</p>',
    unsafe_allow_html=True
)

st.title("Track Defect Intelligence")

st.markdown(
    "Automated railway surface defect detection, "
    "localization and visual analysis."
)

st.divider()


# ============================================================
# 8. NAVIGATION
# ============================================================

dashboard_tab, image_tab, video_tab, model_tab = st.tabs([
    "Overview",
    "Image Inspection",
    "Video Inspection",
    "Model Performance"
])


# ============================================================
# 9. OVERVIEW DASHBOARD
# ============================================================

with dashboard_tab:

    st.subheader("Model performance overview")

    st.caption(
        "Independent test evaluation · 402 images · YOLOv8n"
    )

    columns = st.columns(4)

    for column, (label, value) in zip(
        columns,
        TEST_METRICS.items()
    ):

        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.2f}%</div>
                <div class="metric-caption">Test set result</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown("### Defect categories")

        st.dataframe(
            CLASS_METRICS,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown("### Detection classes")

        class_counts = pd.DataFrame({
            "Defect": [
                "Spalling",
                "Wheel Burn",
                "Squat"
            ],
            "Training annotations": [
                1104,
                962,
                1954
            ]
        })

        st.bar_chart(
            class_counts.set_index("Defect")
        )

    st.divider()

    st.markdown("### Evaluation information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Training images", "3,216")
    col2.metric("Validation images", "402")
    col3.metric("Test images", "402")

    st.info(
        "Performance figures shown here are the recorded "
        "independent test-set evaluation results. They are "
        "not live predictions."
    )


# ============================================================
# 10. IMAGE INSPECTION
# ============================================================

with image_tab:

    st.subheader("Image defect inspection")

    st.write(
        "Upload a railway track image to identify and localize defects."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        try:

            image = Image.open(uploaded_image).convert("RGB")

            st.markdown("### Input image")

            st.image(
                image,
                use_container_width=True
            )

            if st.button(
                "Analyze image",
                type="primary",
                key="analyze_image"
            ):

                with st.spinner(
                    "YOLO is analyzing the image..."
                ):

                    annotated, detections = detect_image(
                        image,
                        confidence
                    )

                st.divider()

                st.markdown("### Detection output")

                st.image(
                    annotated,
                    use_container_width=True
                )

                if detections:

                    df = pd.DataFrame(detections)

                    st.success(
                        f"{len(detections)} detection(s) found."
                    )

                    a, b, c = st.columns(3)

                    a.metric(
                        "Total detections",
                        len(detections)
                    )

                    b.metric(
                        "Defect categories",
                        df["Defect"].nunique()
                    )

                    c.metric(
                        "Mean confidence",
                        f"{df['Confidence'].mean():.1f}%"
                    )

                    st.markdown("### Detection details")

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download detection report",
                        data=df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="image_detection_report.csv",
                        mime="text/csv"
                    )

                else:

                    st.warning(
                        "No defects detected above the selected "
                        "confidence threshold."
                    )

                annotated_pil = Image.fromarray(annotated)

                from io import BytesIO

                buffer = BytesIO()

                annotated_pil.save(
                    buffer,
                    format="JPEG",
                    quality=95
                )

                st.download_button(
                    "Download annotated image",
                    data=buffer.getvalue(),
                    file_name="railway_annotated.jpg",
                    mime="image/jpeg"
                )

        except Exception as error:

            st.error(f"Image processing error: {error}")


# ============================================================
# 11. VIDEO INSPECTION
# ============================================================

with video_tab:

    st.subheader("Video defect inspection")

    st.write(
        "Upload railway footage to process frames using the trained model."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "Start video analysis",
            type="primary",
            key="analyze_video"
        ):

            input_path = None
            output_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    delete=False
                ) as input_file:

                    input_file.write(
                        uploaded_video.getvalue()
                    )

                    input_path = Path(input_file.name)

                with tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    delete=False
                ) as output_file:

                    output_path = Path(output_file.name)

                progress = st.progress(
                    0,
                    text="Preparing video..."
                )

                with st.spinner(
                    "Processing video frames. This may take time on CPU."
                ):

                    detections, frame_count = detect_video(
                        input_path,
                        output_path,
                        confidence,
                        progress
                    )

                st.success(
                    f"Completed processing {frame_count} frames."
                )

                if frame_count == 0:
                    st.warning("No readable video frames were found.")

                else:

                    video_bytes = output_path.read_bytes()

                    st.markdown("### Processed video")

                    st.video(video_bytes)

                    st.download_button(
                        "Download processed video",
                        data=video_bytes,
                        file_name="railway_processed_video.mp4",
                        mime="video/mp4"
                    )

                if detections:

                    df = pd.DataFrame(detections)

                    st.markdown("### Detection summary")

                    a, b, c = st.columns(3)

                    a.metric(
                        "Total detections",
                        len(df)
                    )

                    b.metric(
                        "Frames with detections",
                        df["Frame"].nunique()
                    )

                    c.metric(
                        "Defect categories",
                        df["Defect"].nunique()
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download video detection report",
                        data=df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="video_detection_report.csv",
                        mime="text/csv"
                    )

                    counts = (
                        df["Defect"]
                        .value_counts()
                        .rename_axis("Defect")
                        .to_frame("Detections")
                    )

                    st.markdown("### Detected defect distribution")

                    st.bar_chart(counts)

                else:

                    st.warning(
                        "No defects detected above the selected "
                        "confidence threshold."
                    )

            except Exception as error:

                st.error(
                    f"Video processing failed: {error}"
                )

            finally:

                if input_path is not None:
                    input_path.unlink(missing_ok=True)

                if output_path is not None:
                    output_path.unlink(missing_ok=True)


# ============================================================
# 12. MODEL PERFORMANCE
# ============================================================

with model_tab:

    st.subheader("Model evaluation")

    st.write(
        "The following results were measured using the "
        "independent 402-image test dataset."
    )

    st.markdown("### Overall performance")

    st.dataframe(
        pd.DataFrame({
            "Metric": list(TEST_METRICS.keys()),
            "Result (%)": list(TEST_METRICS.values())
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Class-wise performance")

    st.dataframe(
        CLASS_METRICS,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Model configuration")

    configuration = {
        "Model": "YOLOv8n",
        "Framework": "Ultralytics",
        "Input resolution": "416 × 416",
        "Training epochs": 50,
        "Batch size": 4,
        "Training device": "CPU",
        "Training images": 3216,
        "Validation images": 402,
        "Test images": 402,
        "Classes": "Spalling, Wheel Burn, Squat"
    }

    st.json(configuration)

    st.warning(
        "This model is an academic prototype. Its results "
        "do not establish railway safety certification or "
        "operational inspection readiness."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8295aa; padding:12px;">
        RAILWAY AI · DEEP LEARNING BASED DEFECT DETECTION
        <br>
        Academic Project · YOLOv8n · Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)
