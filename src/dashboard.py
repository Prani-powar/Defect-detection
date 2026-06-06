from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

try:
    import winsound
except ModuleNotFoundError:
    winsound = None

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except ModuleNotFoundError:
    av = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import LOG_CSV_PATH, LOG_IMAGE_DIR, PRODUCT_TASKS, RAW_DATA_DIR
from src.utils import (
    ensure_directories,
    load_labels,
    load_trained_model,
    predict_image_file,
    timestamp_string,
    write_prediction_log,
)


st.set_page_config(page_title="Apple Inspection Dashboard", page_icon="A", layout="wide")

FEEDBACK_CSV_PATH = LOG_CSV_PATH.parent / "feedback.csv"
FEEDBACK_FIELDS = [
    "timestamp",
    "original_prediction_time",
    "original_filename",
    "image_path",
    "predicted_class",
    "feedback_label",
    "ai_was_correct",
    "training_copy_path",
]

DISPLAY_LABELS = {
    "fresh": "Fresh",
    "rotten": "Rotten",
    "not_fruit": "Not a fruit",
    "uncertain": "Not sure",
}
FRUIT_WORDS = {
    "apple",
    "granny_smith",
    "banana",
    "orange",
    "lemon",
    "pineapple",
    "pomegranate",
    "fig",
    "strawberry",
    "custard_apple",
}
NON_FRUIT_IMAGENET_THRESHOLD = 0.55
NOT_FRUIT_MODEL_CONFIDENCE_THRESHOLD = 0.72
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

CSS = """
<style>
.stApp {
    background: #0f141b;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1240px;
}
.hero {
    padding: 1.45rem 1.55rem;
    border: 1px solid #334155;
    border-radius: 8px;
    background: #18202b;
    margin-bottom: 1.05rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .25);
}
.hero-kicker {
    color: #8be2ad;
    font-size: .8rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: .35rem;
}
.hero h1 {
    margin: 0;
    font-size: 2.15rem;
    letter-spacing: 0;
    color: #f8fafc;
}
.hero p {
    margin: .45rem 0 0 0;
    color: #cbd5e1;
    max-width: 780px;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 750;
    color: #f8fafc;
    margin: .15rem 0 .2rem 0;
}
.section-copy {
    color: #d7dee8;
    margin: 0 0 1rem 0;
}
.panel {
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem;
    background: #18202b;
    box-shadow: 0 8px 20px rgba(0, 0, 0, .18);
    margin-bottom: 1rem;
}
.metric-card {
    border: 1px solid #334155;
    border-radius: 8px;
    background: #18202b;
    padding: .9rem 1rem;
    min-height: 92px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, .16);
}
.metric-label {
    color: #cbd5e1;
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
}
.metric-value {
    color: #ffffff;
    font-size: 1.95rem;
    line-height: 1.1;
    font-weight: 780;
    margin-top: .35rem;
}
.metric-fresh { border-top: 4px solid #1f9d55; }
.metric-rotten { border-top: 4px solid #d92d20; }
.metric-neutral { border-top: 4px solid #64748b; }
.metric-warn { border-top: 4px solid #f59e0b; }
.metric-defect { border-top: 4px solid #8b5cf6; }
.result-card {
    border: 1px solid #334155;
    border-radius: 8px;
    background: #111827;
    margin-bottom: .85rem;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(0, 0, 0, .18);
}
.status-fresh, .status-rotten, .status-uncertain {
    padding: .85rem .95rem;
    border-radius: 0;
    margin-bottom: 0;
}
.status-fresh { border-left: 6px solid #2dd36f; background: #10271b; }
.status-rotten { border-left: 6px solid #ff5b5b; background: #2b1618; }
.status-uncertain { border-left: 6px solid #f59e0b; background: #2a210d; }
.result-title {
    color: #ffffff;
    font-size: 1rem;
    font-weight: 760;
    margin-bottom: .25rem;
}
.result-line {
    color: #f1f5f9;
    font-weight: 650;
}
.verdict-text {
    color: #ffffff;
    font-size: 1.45rem;
    line-height: 1.15;
    font-weight: 850;
    margin: .2rem 0 .35rem 0;
}
.small-muted { color: #cbd5e1; font-size: .9rem; }
div[data-testid="stTabs"] button {
    font-weight: 700;
    color: #f8fafc;
    opacity: 1;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff;
    border-bottom-color: #2dd36f;
}
div[data-testid="stTabs"] div[role="tablist"] {
    border-bottom: 1px solid #334155;
}
div.stButton > button, div[data-testid="stFormSubmitButton"] button {
    border-radius: 8px;
    min-height: 2.75rem;
    font-weight: 700;
    border: 1px solid #64748b;
    color: #ffffff;
    background: #1f2937;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    border-color: #2dd36f;
    color: #ffffff;
    background: #263445;
}
div[data-testid="stFileUploader"] section {
    border-radius: 8px;
    border-color: #475569;
    background: #151c26;
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {
    color: #f8fafc;
}
label, .stMarkdown, .stCaption, p, span {
    color: #f8fafc;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #f8fafc;
    font-weight: 700;
}
</style>
"""


@st.cache_resource
def cached_model():
    return load_trained_model()


@st.cache_resource
def cached_object_model():
    try:
        return MobileNetV2(weights="imagenet")
    except Exception:
        return None


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "source",
            "original_filename",
            "image_path",
            "predicted_class",
            "is_rotten",
            "confidence",
            "fresh_probability",
            "rotten_probability",
            "fresh_count",
            "rotten_count",
            "uncertain_count",
        ]
    )


def load_history() -> pd.DataFrame:
    if not LOG_CSV_PATH.exists():
        return empty_history()
    history = pd.read_csv(LOG_CSV_PATH)
    for column in empty_history().columns:
        if column not in history.columns:
            history[column] = ""
    return history


def load_feedback() -> pd.DataFrame:
    if not FEEDBACK_CSV_PATH.exists():
        return pd.DataFrame(columns=FEEDBACK_FIELDS)
    feedback = pd.read_csv(FEEDBACK_CSV_PATH)
    for column in FEEDBACK_FIELDS:
        if column not in feedback.columns:
            feedback[column] = ""
    return feedback


def max_count(history: pd.DataFrame, column: str) -> int:
    if history.empty or column not in history.columns:
        return 0
    values = pd.to_numeric(history[column], errors="coerce").dropna()
    return int(values.max()) if not values.empty else 0


def next_counts(history: pd.DataFrame, prediction: str) -> dict[str, int]:
    counts = {
        "fresh": max_count(history, "fresh_count"),
        "rotten": max_count(history, "rotten_count"),
        "uncertain": max_count(history, "uncertain_count"),
    }
    count_key = prediction if prediction in {"fresh", "rotten"} else "uncertain"
    counts[count_key] = counts.get(count_key, 0) + 1
    return counts


def opposite_label(label: str) -> str:
    return "rotten" if label == "fresh" else "fresh"


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def status_class(label: str) -> str:
    if label == "rotten":
        return "status-rotten"
    if label == "fresh":
        return "status-fresh"
    return "status-uncertain"


def use_local_opencv_camera() -> bool:
    return sys.platform.startswith("win")


def play_rotten_alert() -> None:
    if winsound is not None:
        winsound.Beep(1200, 500)
        return
    components.html(
        """
        <script>
        (() => {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;

            const context = new AudioContext();
            const oscillator = context.createOscillator();
            const gain = context.createGain();

            oscillator.type = "sine";
            oscillator.frequency.value = 1200;
            gain.gain.setValueAtTime(0.18, context.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.5);

            oscillator.connect(gain);
            gain.connect(context.destination);
            oscillator.start();
            oscillator.stop(context.currentTime + 0.5);
            setTimeout(() => context.close(), 650);
        })();
        </script>
        """,
        height=0,
    )


def detect_fruit_like(image_path: Path) -> tuple[bool, str, float]:
    object_model = cached_object_model()
    if object_model is None:
        return True, "object check unavailable", 0.0

    try:
        image = keras_image.load_img(image_path, target_size=(224, 224))
        array = keras_image.img_to_array(image)
        batch = np.expand_dims(array, axis=0)
        batch = preprocess_input(batch)
        predictions = object_model.predict(batch, verbose=0)
        decoded = decode_predictions(predictions, top=10)[0]
    except Exception:
        return True, "object check failed", 0.0

    top_label = decoded[0][1].lower()
    top_score = float(decoded[0][2])
    for _, label, score in decoded:
        normalized = label.lower()
        if any(word in normalized for word in FRUIT_WORDS):
            return True, normalized, float(score)

    if top_score >= NON_FRUIT_IMAGENET_THRESHOLD:
        return False, top_label, top_score
    return True, f"uncertain object: {top_label}", top_score


def predict_with_not_fruit_check(image_path: Path, labels: list[str]) -> tuple[str, float, dict[str, float], str, float]:
    if "not_fruit" in labels:
        model = cached_model()
        prediction, confidence, probabilities = predict_image_file(model, image_path, labels)
        return prediction, confidence, probabilities, "trained 3-class model", probabilities.get("not_fruit", 0.0)

    fruit_like, object_label, object_confidence = detect_fruit_like(image_path)
    if not fruit_like:
        return "not_fruit", object_confidence, {"not_fruit": 1.0}, object_label, object_confidence

    model = cached_model()
    prediction, confidence, probabilities = predict_image_file(model, image_path, labels)

    if prediction == "not_fruit":
        return prediction, confidence, probabilities, object_label, object_confidence

    if confidence < NOT_FRUIT_MODEL_CONFIDENCE_THRESHOLD:
        probabilities = {**probabilities, "not_fruit": 1.0 - confidence}
        return "not_fruit", 1.0 - confidence, probabilities, object_label, object_confidence

    return prediction, confidence, probabilities, object_label, object_confidence


def save_uploaded_image(uploaded_file) -> Path:
    ensure_directories([LOG_IMAGE_DIR])
    original_name = getattr(uploaded_file, "name", "browser_camera.jpg") or "browser_camera.jpg"
    suffix = Path(original_name).suffix.lower() or ".jpg"
    safe_name = Path(original_name).stem.replace(" ", "_")[:40]
    output_path = LOG_IMAGE_DIR / f"upload_{timestamp_string()}_{safe_name}{suffix}"
    with output_path.open("wb") as handle:
        if hasattr(uploaded_file, "getbuffer"):
            handle.write(uploaded_file.getbuffer())
        else:
            shutil.copyfileobj(uploaded_file, handle)
    return output_path


def is_blank_frame(frame) -> bool:
    if frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray)) < 8.0 or float(np.mean(gray)) < 8.0


def capture_webcam_image(camera_index: int = 0, countdown_seconds: int = 0) -> Path:
    ensure_directories([LOG_IMAGE_DIR])
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. Try another camera number or close other camera apps."
        )

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        last_frame = None
        for _ in range(30):
            ok, frame = camera.read()
            if ok:
                last_frame = frame
            time.sleep(0.05)

        if is_blank_frame(last_frame):
            raise RuntimeError(
                "The webcam opened, but it returned a blank image. Try camera index 1 or 2, "
                "or allow camera access in Windows Settings."
            )

        if countdown_seconds:
            countdown_box = st.empty()
            for number in range(countdown_seconds, 0, -1):
                countdown_box.markdown(f"### Capturing in {number}...")
                time.sleep(1)

        captured_frame = None
        for _ in range(10):
            camera.read()
            ok, frame = camera.read()
            if ok and not is_blank_frame(frame):
                captured_frame = frame
                break
            time.sleep(0.05)

        if captured_frame is None:
            raise RuntimeError("Could not capture a clear webcam image. Try again with better lighting.")

        output_path = LOG_IMAGE_DIR / f"webcam_{timestamp_string()}.jpg"
        cv2.imwrite(str(output_path), captured_frame)
        return output_path
    finally:
        camera.release()


def capture_single_frame(camera, timeout_seconds: float = 2.0):
    started = time.time()
    last_frame = None
    while time.time() - started < timeout_seconds:
        ok, frame = camera.read()
        if ok and not is_blank_frame(frame):
            return frame
        if ok:
            last_frame = frame
        time.sleep(0.05)
    return last_frame


def save_frame(frame, prefix: str = "live") -> Path:
    ensure_directories([LOG_IMAGE_DIR])
    output_path = LOG_IMAGE_DIR / f"{prefix}_{timestamp_string()}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


def inspect_saved_image(
    image_path: Path,
    original_filename: str,
    source: str,
    history: pd.DataFrame,
    labels: list[str],
) -> dict:
    prediction, confidence, probabilities, object_label, object_confidence = predict_with_not_fruit_check(image_path, labels)
    if prediction == "rotten":
        play_rotten_alert()
    counts = next_counts(history, prediction)
    write_prediction_log(
        str(image_path),
        prediction,
        confidence,
        counts,
        probabilities=probabilities,
        source=source,
        original_filename=original_filename,
    )
    return {
        "original_filename": original_filename,
        "image_path": str(image_path),
        "predicted_class": prediction,
        "is_rotten": prediction == "rotten",
        "confidence": confidence,
        "fresh_probability": probabilities.get("fresh", 0.0),
        "rotten_probability": probabilities.get("rotten", 0.0),
        "not_fruit_probability": probabilities.get("not_fruit", 0.0),
        "object_label": object_label,
        "object_confidence": object_confidence,
    }


def inspect_uploaded_files(uploaded_files: list, history: pd.DataFrame, labels: list[str]) -> list[dict]:
    results = []
    running_history = history.copy()
    for uploaded_file in uploaded_files:
        saved_path = save_uploaded_image(uploaded_file)
        result = inspect_saved_image(saved_path, uploaded_file.name, "dashboard_upload", running_history, labels)
        results.append(result)
        counts = next_counts(running_history, result["predicted_class"])
        running_history = pd.concat(
            [
                running_history,
                pd.DataFrame(
                    [
                        {
                            "fresh_count": counts["fresh"],
                            "rotten_count": counts["rotten"],
                            "uncertain_count": counts["uncertain"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return results


def inspect_browser_camera_photo(uploaded_photo, source: str, labels: list[str], history: pd.DataFrame) -> dict:
    saved_path = save_uploaded_image(uploaded_photo)
    return inspect_saved_image(
        saved_path,
        getattr(uploaded_photo, "name", "browser_camera.jpg"),
        source,
        history,
        labels,
    )


class WebRtcConveyorProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.labels: list[str] = []
        self.interval_seconds = 1.0
        self.max_photos = 60
        self.last_capture_at = 0.0
        self.captured_count = 0
        self.results: list[dict] = []
        self.latest_result: dict | None = None
        self.alert_token = 0
        self.ready = False

    def configure(self, labels: list[str], interval_seconds: float, max_photos: int) -> None:
        with self.lock:
            self.labels = labels
            self.interval_seconds = max(0.5, float(interval_seconds))
            self.max_photos = int(max_photos)
            self.ready = True

    def snapshot(self) -> tuple[dict | None, list[dict], int, int, int]:
        with self.lock:
            return (
                self.latest_result,
                list(self.results),
                self.alert_token,
                self.captured_count,
                self.max_photos,
            )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.time()
        overlay_label = "Waiting"
        overlay_color = (255, 190, 0)

        should_capture = False
        with self.lock:
            if (
                self.ready
                and self.captured_count < self.max_photos
                and now - self.last_capture_at >= self.interval_seconds
            ):
                self.last_capture_at = now
                self.captured_count += 1
                capture_number = self.captured_count
                labels = list(self.labels)
                should_capture = True
            else:
                capture_number = self.captured_count
                labels = list(self.labels)
                if self.latest_result:
                    overlay_label = display_label(str(self.latest_result["predicted_class"]))

        if should_capture and labels:
            try:
                image_path = save_frame(image, "webrtc_live")
                prediction, confidence, probabilities, object_label, object_confidence = predict_with_not_fruit_check(image_path, labels)
                counts = next_counts(load_history(), prediction)
                write_prediction_log(
                    str(image_path),
                    prediction,
                    confidence,
                    counts,
                    probabilities=probabilities,
                    source="webrtc_live_inspection",
                    original_filename=image_path.name,
                )
                result = {
                    "original_filename": image_path.name,
                    "image_path": str(image_path),
                    "predicted_class": prediction,
                    "is_rotten": prediction == "rotten",
                    "confidence": confidence,
                    "fresh_probability": probabilities.get("fresh", 0.0),
                    "rotten_probability": probabilities.get("rotten", 0.0),
                    "not_fruit_probability": probabilities.get("not_fruit", 0.0),
                    "object_label": object_label,
                    "object_confidence": object_confidence,
                }
                row = {
                    "photo": capture_number,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "result": display_label(prediction),
                    "confidence": f"{confidence * 100:.2f}%",
                    "image_path": str(image_path),
                }
                with self.lock:
                    self.latest_result = result
                    self.results.insert(0, row)
                    self.results = self.results[:50]
                    if prediction == "rotten":
                        self.alert_token += 1
                overlay_label = display_label(prediction)
            except Exception as error:
                overlay_label = f"Error: {error}"

        if overlay_label == "Rotten":
            overlay_color = (80, 80, 255)
        elif overlay_label == "Fresh":
            overlay_color = (80, 220, 80)
        elif overlay_label == "Not a fruit":
            overlay_color = (0, 180, 255)

        cv2.rectangle(image, (12, 12), (min(image.shape[1] - 12, 520), 78), (15, 20, 27), -1)
        cv2.putText(image, overlay_label, (24, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, overlay_color, 3)
        cv2.putText(image, f"Captures: {capture_number}/{self.max_photos}", (24, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


def result_card(result: dict) -> None:
    label = str(result["predicted_class"])
    confidence = float(result["confidence"]) * 100
    object_label = str(result.get("object_label", ""))
    object_line = f'<div class="small-muted">Object check: {object_label}</div>' if object_label else ""
    st.markdown(
        f"""
        <div class="result-card">
            <div class="{status_class(label)}">
                <div class="result-title">{Path(result['original_filename']).name}</div>
                <div class="verdict-text">{display_label(label)}</div>
                <div class="small-muted">Confidence: {confidence:.2f}%</div>
                {object_line}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    image_path = Path(result["image_path"])
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)


def save_feedback(row: pd.Series, feedback_label: str, ai_was_correct: bool) -> Path:
    source_path = Path(str(row.get("image_path", "")))
    if not source_path.exists():
        raise FileNotFoundError(f"Saved inspection image not found: {source_path}")

    target_dir = RAW_DATA_DIR / feedback_label
    ensure_directories([target_dir, FEEDBACK_CSV_PATH.parent])
    suffix = source_path.suffix.lower() or ".jpg"
    training_copy = target_dir / f"feedback_{feedback_label}_{timestamp_string()}{suffix}"
    shutil.copy2(source_path, training_copy)

    file_exists = FEEDBACK_CSV_PATH.exists()
    feedback_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "original_prediction_time": row.get("timestamp", ""),
        "original_filename": row.get("original_filename", ""),
        "image_path": str(source_path),
        "predicted_class": row.get("predicted_class", ""),
        "feedback_label": feedback_label,
        "ai_was_correct": str(ai_was_correct),
        "training_copy_path": str(training_copy),
    }
    with FEEDBACK_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = pd.DataFrame([feedback_row])
        writer.to_csv(handle, header=not file_exists, index=False)
    return training_copy


def retrain_model() -> tuple[bool, str]:
    split = subprocess.run(
        [sys.executable, "src/split_dataset.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    if split.returncode != 0:
        return False, split.stdout + "\n" + split.stderr
    train = subprocess.run(
        [sys.executable, "src/train_model.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    return train.returncode == 0, split.stdout + "\n" + train.stdout + "\n" + train.stderr


def render_metrics(history: pd.DataFrame) -> None:
    fresh_count = max_count(history, "fresh_count")
    rotten_count = max_count(history, "rotten_count")
    uncertain_count = max_count(history, "uncertain_count")
    total_count = fresh_count + rotten_count + uncertain_count
    inspected_count = fresh_count + rotten_count
    defect_percentage = (rotten_count / inspected_count * 100) if inspected_count else 0.0

    cards = [
        ("Total", total_count, "metric-neutral"),
        ("Fresh", fresh_count, "metric-fresh"),
        ("Rotten", rotten_count, "metric-rotten"),
        ("Uncertain", uncertain_count, "metric-warn"),
        ("Defect %", f"{defect_percentage:.1f}%", "metric-defect"),
    ]
    metric_cols = st.columns(5)
    for column, (label, value, class_name) in zip(metric_cols, cards):
        column.markdown(
            f"""
            <div class="metric-card {class_name}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_table(history: pd.DataFrame) -> None:
    display_columns = [
        "timestamp",
        "source",
        "original_filename",
        "predicted_class",
        "confidence",
        "fresh_probability",
        "rotten_probability",
        "image_path",
    ]
    st.dataframe(
        history[display_columns].tail(150).sort_index(ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def section_header(title: str, copy: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<p class="section-copy">{copy}</p>', unsafe_allow_html=True)


def open_panel() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)


def close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_browser_live_inspection(labels: list[str]) -> None:
    open_panel()
    section_header(
        "Browser Conveyor Live Inspection",
        "Use this on the deployed website. Your browser streams the selected webcam, and the app analyzes one frame every interval.",
    )
    st.info(
        "Choose your external webcam in the browser permission popup. This is the cloud workaround for conveyor live inspection."
    )

    interval_seconds = st.number_input("Seconds between analyzed frames", min_value=1, max_value=10, value=1, step=1)
    max_photos = st.number_input("Frames this run", min_value=1, max_value=500, value=60, step=1)

    if webrtc_streamer is None:
        st.warning("WebRTC is not installed yet. The snapshot camera below will still work after you allow camera permission.")
        browser_photo = st.camera_input("Take one inspection photo", key="browser_live_camera_fallback")
        if st.button("Analyze Camera Photo", use_container_width=True):
            if browser_photo is None:
                st.warning("Take a photo first.")
            else:
                result = inspect_browser_camera_photo(browser_photo, "browser_live_camera", labels, load_history())
                st.session_state["batch_results"] = [result]
                result_card(result)
        close_panel()
        return

    ctx = webrtc_streamer(
        key="apple-conveyor-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=WebRtcConveyorProcessor,
        async_processing=True,
    )

    latest_box = st.empty()
    progress_box = st.empty()
    table_box = st.empty()

    if ctx.video_processor:
        ctx.video_processor.configure(labels, float(interval_seconds), int(max_photos))

    if not ctx.state.playing:
        st.info("Click START above, allow camera permission, then select your external webcam if the browser asks.")
        close_panel()
        return

    while ctx.state.playing:
        if not ctx.video_processor:
            time.sleep(0.3)
            continue

        latest, rows, alert_token, captured_count, target_count = ctx.video_processor.snapshot()
        if alert_token != st.session_state.get("last_webrtc_alert_token", 0):
            st.session_state["last_webrtc_alert_token"] = alert_token
            play_rotten_alert()

        if latest:
            st.session_state["batch_results"] = [latest]
            with latest_box.container():
                result_card(latest)
        if target_count:
            progress_box.progress(min(captured_count / target_count, 1.0), text=f"Analyzed {captured_count} of {target_count}")
        if rows:
            table_box.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if captured_count >= target_count:
            st.success("Browser conveyor inspection run finished. Stop the stream or start a new run.")
            break
        time.sleep(0.5)
    close_panel()


def render_live_inspection(labels: list[str]) -> None:
    if not use_local_opencv_camera():
        render_browser_live_inspection(labels)
        return

    open_panel()
    section_header(
        "Live Conveyor Inspection",
        "Use this when apples move in front of a fixed webcam. It captures one photo every second and logs each result.",
    )

    control_col1, control_col2, control_col3 = st.columns(3)
    camera_index = control_col1.selectbox("Live camera source", [0, 1, 2, 3], index=0)
    interval_seconds = control_col2.number_input("Seconds between photos", min_value=1, max_value=10, value=1, step=1)
    max_photos = control_col3.number_input("Photos this run", min_value=1, max_value=500, value=60, step=1)

    start_col, stop_col = st.columns(2)
    if "live_running" not in st.session_state:
        st.session_state["live_running"] = False
    if "live_results" not in st.session_state:
        st.session_state["live_results"] = []

    if start_col.button("Start Live Inspection", use_container_width=True):
        st.session_state["live_running"] = True
        st.session_state["live_results"] = []
    if stop_col.button("Stop Live Inspection", use_container_width=True):
        st.session_state["live_running"] = False

    latest_box = st.empty()
    progress_box = st.empty()
    table_box = st.empty()

    if not st.session_state["live_running"]:
        if st.session_state["live_results"]:
            table_box.dataframe(pd.DataFrame(st.session_state["live_results"]), use_container_width=True, hide_index=True)
        else:
            st.info("Live inspection is stopped. Click Start Live Inspection when the conveyor is ready.")
        close_panel()
        return

    camera = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not camera.isOpened():
        st.session_state["live_running"] = False
        st.error(f"Could not open camera source {camera_index}. Try another source or close other camera apps.")
        return

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(20):
            camera.read()
            time.sleep(0.03)

        for photo_number in range(1, int(max_photos) + 1):
            if not st.session_state.get("live_running", False):
                break

            frame = capture_single_frame(camera)
            if frame is None or is_blank_frame(frame):
                latest_box.warning("Camera returned a blank frame. Live inspection stopped.")
                st.session_state["live_running"] = False
                break

            image_path = save_frame(frame, "live")
            history = load_history()
            prediction, confidence, probabilities, object_label, object_confidence = predict_with_not_fruit_check(image_path, labels)
            if prediction == "rotten":
                play_rotten_alert()
            counts = next_counts(history, prediction)
            write_prediction_log(
                str(image_path),
                prediction,
                confidence,
                counts,
                probabilities=probabilities,
                source="live_inspection",
                original_filename=image_path.name,
            )

            result = {
                "photo": photo_number,
                "prediction": prediction,
                "result": display_label(prediction),
                "confidence": f"{confidence * 100:.2f}%",
                "fresh_probability": f"{probabilities.get('fresh', 0.0) * 100:.2f}%",
                "rotten_probability": f"{probabilities.get('rotten', 0.0) * 100:.2f}%",
                "not_fruit_probability": f"{probabilities.get('not_fruit', 0.0) * 100:.2f}%",
                "object_check": object_label,
                "image_path": str(image_path),
            }
            st.session_state["live_results"].insert(0, result)
            st.session_state["live_results"] = st.session_state["live_results"][:50]

            with latest_box.container():
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="{status_class(prediction)}">
                            <div class="result-title">Live capture #{photo_number}</div>
                            <div class="verdict-text">{display_label(prediction)}</div>
                            <div class="small-muted">Confidence: {confidence * 100:.2f}%</div>
                            <div class="small-muted">Object check: {object_label}</div>
                            <div class="small-muted">Saved image: {image_path.name}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(str(image_path), use_container_width=True)
            progress_box.progress(photo_number / int(max_photos), text=f"Captured {photo_number} of {int(max_photos)}")
            table_box.dataframe(pd.DataFrame(st.session_state["live_results"]), use_container_width=True, hide_index=True)
            time.sleep(float(interval_seconds))

        st.session_state["live_running"] = False
        st.success("Live inspection run finished.")
    finally:
        camera.release()
    close_panel()


def main() -> None:
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Smart manufacturing inspection</div>
            <h1>Apple Defect Detection Dashboard</h1>
            <p>Inspect multiple apple images, review the AI result, and feed corrections back into the training data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Active task: apple quality | Registered tasks: {', '.join(PRODUCT_TASKS)}")

    labels = load_labels()
    history = load_history()
    render_metrics(history)

    inspect_tab, live_tab, learn_tab, history_tab = st.tabs(
        ["Batch Inspection", "Live Inspection", "Feedback & Learning", "History"]
    )

    with inspect_tab:
        left, right = st.columns([0.9, 1.1], gap="large")
        with left:
            open_panel()
            section_header("Batch Upload", "Upload one or many apple photos, then inspect them together.")
            with st.form(f"batch_inspection_form_{st.session_state['uploader_key']}", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Choose apple images",
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=True,
                    key=f"apple_uploader_{st.session_state['uploader_key']}",
                )
                submitted = st.form_submit_button("Submit Batch Inspection", use_container_width=True)

            if submitted:
                if not uploaded_files:
                    st.warning("Upload at least one apple image first.")
                else:
                    try:
                        st.session_state["batch_results"] = inspect_uploaded_files(uploaded_files, history, labels)
                        st.session_state["uploader_key"] += 1
                        st.rerun()
                    except FileNotFoundError as error:
                        st.error(str(error))

            st.divider()
            if use_local_opencv_camera():
                section_header("Camera Capture", "Capture one photo from the webcam and analyze it immediately.")
                camera_index = st.selectbox(
                    "Camera source",
                    options=[0, 1, 2, 3],
                    index=0,
                    help="If the saved photo is blank, try camera 1 or 2.",
                )
                camera_col1, camera_col2 = st.columns(2)
                instant_clicked = camera_col1.button("Instant Photo", use_container_width=True)
                countdown_clicked = camera_col2.button("3 Second Countdown", use_container_width=True)

                if instant_clicked or countdown_clicked:
                    try:
                        seconds = 3 if countdown_clicked else 0
                        with st.spinner("Opening webcam..."):
                            captured_path = capture_webcam_image(camera_index, seconds)
                        refreshed_history = load_history()
                        result = inspect_saved_image(
                            captured_path,
                            captured_path.name,
                            "webcam_countdown" if countdown_clicked else "webcam_instant",
                            refreshed_history,
                            labels,
                        )
                        st.session_state["batch_results"] = [result]
                        st.rerun()
                    except RuntimeError as error:
                        st.error(str(error))
            else:
                section_header("Browser Camera Capture", "Use your laptop or phone camera through the website browser.")
                browser_photo = st.camera_input("Take one apple inspection photo", key="batch_browser_camera")
                if st.button("Analyze Browser Camera Photo", use_container_width=True):
                    if browser_photo is None:
                        st.warning("Allow camera permission and take a photo first.")
                    else:
                        with st.spinner("Analyzing browser camera photo..."):
                            result = inspect_browser_camera_photo(
                                browser_photo,
                                "browser_camera",
                                labels,
                                load_history(),
                            )
                        st.session_state["batch_results"] = [result]
                        st.rerun()
            close_panel()

        with right:
            open_panel()
            section_header("Inspection Results", "Latest analyzed images appear here after upload, camera capture, or live inspection.")
            results = st.session_state.get("batch_results", [])
            if not results:
                refreshed = load_history()
                if refreshed.empty:
                    st.info("No inspection results yet.")
                else:
                    latest_rows = refreshed.tail(3).sort_index(ascending=False)
                    for _, row in latest_rows.iterrows():
                        result_card(
                            {
                                "original_filename": row.get("original_filename", ""),
                                "image_path": row.get("image_path", ""),
                                "predicted_class": row.get("predicted_class", ""),
                                "is_rotten": str(row.get("predicted_class", "")) == "rotten",
                                "confidence": float(row.get("confidence", 0) or 0),
                                "fresh_probability": float(row.get("fresh_probability", 0) or 0),
                                "rotten_probability": float(row.get("rotten_probability", 0) or 0),
                            }
                        )
            else:
                for result in results:
                    result_card(result)
            close_panel()

    with live_tab:
        render_live_inspection(labels)

    with learn_tab:
        open_panel()
        section_header("Review AI Answer", "Tell the system whether a prediction was correct, then retrain when you are ready.")
        refreshed_history = load_history()
        if refreshed_history.empty:
            st.info("No predictions available for feedback yet.")
        else:
            options = list(refreshed_history.tail(100).sort_index(ascending=False).index)

            def option_label(index: int) -> str:
                row = refreshed_history.loc[index]
                return f"{row.get('timestamp', '')} | {row.get('original_filename', '')} | AI: {row.get('predicted_class', '')}"

            selected_index = st.selectbox("Select a prediction to review", options, format_func=option_label)
            selected = refreshed_history.loc[selected_index]

            review_left, review_right = st.columns([0.9, 1.1], gap="large")
            with review_left:
                image_path = Path(str(selected.get("image_path", "")))
                if image_path.exists():
                    st.image(str(image_path), caption=str(selected.get("original_filename", image_path.name)), use_container_width=True)
                else:
                    st.warning("Image file for this record was not found.")
            with review_right:
                predicted = str(selected.get("predicted_class", ""))
                label_options = ["fresh", "rotten", "not_fruit"]
                default_index = label_options.index(predicted) if predicted in label_options else 2
                st.markdown(f"**AI prediction:** `{display_label(predicted)}`")
                feedback_label = st.radio(
                    "Correct label",
                    label_options,
                    index=default_index,
                    horizontal=True,
                    format_func=display_label,
                )

                if st.button("Save Feedback Label", use_container_width=True):
                    ai_was_correct = feedback_label == predicted
                    saved = save_feedback(selected, feedback_label, ai_was_correct)
                    st.success(f"Added to training data as {display_label(feedback_label)}: {saved}")

                st.divider()
                section_header(
                    "Retrain After Feedback",
                    "Feedback is saved immediately. Add several Not a fruit examples, then retrain to teach the model the new category.",
                )
                if st.button("Retrain Model With Feedback", use_container_width=True):
                    with st.spinner("Splitting data and retraining model. Keep this page open."):
                        ok, output = retrain_model()
                    if ok:
                        cached_model.clear()
                        st.success("Model retrained successfully.")
                    else:
                        st.error("Retraining failed. See details below.")
                    with st.expander("Training output"):
                        st.code(output[-6000:])

            feedback = load_feedback()
            section_header("Feedback Log")
            if feedback.empty:
                st.info("No feedback saved yet.")
            else:
                st.dataframe(feedback.tail(50).sort_index(ascending=False), use_container_width=True, hide_index=True)
        close_panel()

    with history_tab:
        open_panel()
        section_header("Prediction History", "Every inspection is saved here with image path, prediction, and probabilities.")
        refreshed_history = load_history()
        if refreshed_history.empty:
            st.info("No history yet.")
        else:
            render_history_table(refreshed_history)
            latest_images = refreshed_history.tail(9).sort_index(ascending=False)
            section_header("Recent Images")
            cols = st.columns(3)
            for index, (_, row) in enumerate(latest_images.iterrows()):
                image_path = Path(str(row.get("image_path", "")))
                with cols[index % 3]:
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    st.caption(
                        f"{str(row.get('predicted_class', '')).title()} | "
                        f"{float(row.get('confidence', 0) or 0) * 100:.2f}%"
                    )

            st.download_button(
                "Download CSV History",
                data=LOG_CSV_PATH.read_bytes(),
                file_name="apple_inspection_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if FEEDBACK_CSV_PATH.exists():
                st.download_button(
                    "Download Feedback CSV",
                    data=FEEDBACK_CSV_PATH.read_bytes(),
                    file_name="apple_feedback_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        close_panel()


if __name__ == "__main__":
    main()

