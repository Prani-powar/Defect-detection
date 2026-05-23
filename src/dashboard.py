from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time
try:
    import winsound
except ImportError:
    winsound = None

import cv2
import numpy as np
import pandas as pd
import streamlit as st

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

CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0

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
.mobile-note {
    border: 1px solid #375a7f;
    border-left: 6px solid #38bdf8;
    border-radius: 8px;
    background: #102033;
    padding: .9rem 1rem;
    color: #e0f2fe;
    margin: .75rem 0 1rem 0;
}
@media (max-width: 760px) {
    .block-container {
        padding: .75rem .7rem 1.5rem .7rem;
        max-width: 100%;
    }
    .hero {
        padding: 1rem;
        margin-bottom: .8rem;
    }
    .hero h1 {
        font-size: 1.45rem;
        line-height: 1.18;
    }
    .hero p {
        font-size: .92rem;
    }
    .section-title {
        font-size: 1.12rem;
    }
    .section-copy {
        font-size: .9rem;
    }
    .panel {
        padding: .85rem;
    }
    .metric-card {
        min-height: 76px;
        padding: .72rem .8rem;
        margin-bottom: .55rem;
    }
    .metric-value {
        font-size: 1.55rem;
    }
    div[data-testid="stTabs"] div[role="tablist"] {
        overflow-x: auto;
        white-space: nowrap;
        gap: .25rem;
    }
    div[data-testid="stTabs"] button {
        min-width: max-content;
        padding-left: .55rem;
        padding-right: .55rem;
        font-size: .85rem;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        width: 100%;
    }
    div[data-testid="stFileUploader"] section {
        padding: .7rem;
    }
    .result-title, .result-line, .small-muted {
        overflow-wrap: anywhere;
    }
    div[data-testid="stDataFrame"] {
        overflow-x: auto;
    }
}
</style>
"""


@st.cache_resource
def cached_model():
    return load_trained_model()


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
    counts[prediction] = counts.get(prediction, 0) + 1
    return counts


def opposite_label(label: str) -> str:
    return "rotten" if label == "fresh" else "fresh"


def save_uploaded_image(uploaded_file) -> Path:
    ensure_directories([LOG_IMAGE_DIR])
    original_name = getattr(uploaded_file, "name", "mobile_camera.jpg") or "mobile_camera.jpg"
    suffix = Path(original_name).suffix.lower() or ".jpg"
    safe_name = Path(original_name).stem.replace(" ", "_")[:40]
    output_path = LOG_IMAGE_DIR / f"upload_{timestamp_string()}_{safe_name}{suffix}"
    with output_path.open("wb") as handle:
        shutil.copyfileobj(uploaded_file, handle)
    return output_path


def beep_for_rotten() -> None:
    if winsound is not None:
        winsound.Beep(1200, 500)


def is_blank_frame(frame) -> bool:
    if frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray)) < 8.0 or float(np.mean(gray)) < 8.0


def capture_webcam_image(camera_index: int = 0, countdown_seconds: int = 0) -> Path:
    ensure_directories([LOG_IMAGE_DIR])
    camera = cv2.VideoCapture(camera_index, CAMERA_BACKEND)
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


def test_camera_source(camera_index: int) -> Path:
    camera = cv2.VideoCapture(camera_index, CAMERA_BACKEND)
    if not camera.isOpened():
        raise RuntimeError(f"Camera source {camera_index} could not be opened.")
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = capture_single_frame(camera, timeout_seconds=3.0)
        if frame is None or is_blank_frame(frame):
            raise RuntimeError(f"Camera source {camera_index} opened but returned a blank frame.")
        return save_frame(frame, f"camera_test_{camera_index}")
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
    model = cached_model()
    prediction, confidence, probabilities = predict_image_file(model, image_path, labels)
    if prediction == "rotten":
        beep_for_rotten()
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


def result_card(result: dict) -> None:
    label = str(result["predicted_class"])
    confidence = float(result["confidence"]) * 100
    status_class = "status-rotten" if label == "rotten" else "status-fresh"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="{status_class}">
                <div class="result-title">{Path(result['original_filename']).name}</div>
                <div class="result-line">{label.title()} | Confidence: {confidence:.2f}% | Rotten: {result['is_rotten']}</div>
                <div class="small-muted">Fresh: {result['fresh_probability'] * 100:.2f}% | Rotten: {result['rotten_probability'] * 100:.2f}%</div>
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
        "is_rotten",
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


def render_live_inspection(labels: list[str]) -> None:
    open_panel()
    section_header(
        "Live Conveyor Inspection",
        "Use this when apples move in front of a fixed webcam. It captures one photo every second and logs each result.",
    )
    st.markdown(
        """
        <div class="mobile-note">
            Select the webcam facing the conveyor. If you have two webcams, test Camera 0 and Camera 1 first.
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col1, control_col2, control_col3 = st.columns(3)
    camera_index = control_col1.selectbox(
        "Live camera source",
        [0, 1, 2, 3, 4, 5],
        index=0,
        format_func=lambda value: f"Camera {value}",
    )
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
    if st.button("Test Live Camera", use_container_width=True):
        try:
            preview_path = test_camera_source(int(camera_index))
            st.success(f"Camera {camera_index} works.")
            st.image(str(preview_path), caption=f"Camera {camera_index} preview", use_container_width=True)
        except RuntimeError as error:
            st.error(str(error))

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

    camera = cv2.VideoCapture(int(camera_index), CAMERA_BACKEND)
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

        model = cached_model()
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
            prediction, confidence, probabilities = predict_image_file(model, image_path, labels)
            if prediction == "rotten":
                beep_for_rotten()
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
                "is_rotten": prediction == "rotten",
                "confidence": f"{confidence * 100:.2f}%",
                "fresh_probability": f"{probabilities.get('fresh', 0.0) * 100:.2f}%",
                "rotten_probability": f"{probabilities.get('rotten', 0.0) * 100:.2f}%",
                "image_path": str(image_path),
            }
            st.session_state["live_results"].insert(0, result)
            st.session_state["live_results"] = st.session_state["live_results"][:50]

            status_class = "status-rotten" if prediction == "rotten" else "status-fresh"
            with latest_box.container():
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="{status_class}">
                            <div class="result-title">Live capture #{photo_number}</div>
                            <div class="result-line">{prediction.title()} | Confidence: {confidence * 100:.2f}% | Rotten: {prediction == 'rotten'}</div>
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


def render_website_camera(labels: list[str]) -> None:
    open_panel()
    section_header(
        "Website Camera",
        "Use this on the deployed website. The browser opens the camera on your phone or laptop, then the app analyzes that photo.",
    )
    st.markdown(
        """
        <div class="mobile-note">
            On the deployed website, the server cannot directly control your USB webcams. Use this browser camera option, or upload photos from your device.
        </div>
        """,
        unsafe_allow_html=True,
    )
    browser_photo = st.camera_input("Take apple photo with this device")
    if st.button("Analyze Website Camera Photo", use_container_width=True):
        if browser_photo is None:
            st.warning("Take a camera photo first.")
        else:
            try:
                saved_path = save_uploaded_image(browser_photo)
                refreshed_history = load_history()
                result = inspect_saved_image(
                    saved_path,
                    getattr(browser_photo, "name", "website_camera.jpg"),
                    "website_camera",
                    refreshed_history,
                    labels,
                )
                st.session_state["batch_results"] = [result]
                st.success("Website camera photo analyzed.")
                result_card(result)
            except FileNotFoundError as error:
                st.error(str(error))
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

    inspect_tab, website_camera_tab, live_tab, learn_tab, history_tab = st.tabs(
        ["Batch Inspection", "Website Camera", "Live Inspection", "Feedback & Learning", "History"]
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
            section_header("Mobile Camera", "On phone, use this to take a photo with the browser camera and analyze it.")
            st.markdown(
                """
                <div class="mobile-note">
                    For mobile/cloud use, this is the best camera option. The OpenCV webcam buttons below are mainly for the local conveyor PC.
                </div>
                """,
                unsafe_allow_html=True,
            )
            mobile_photo = st.camera_input("Take one apple photo")
            if st.button("Analyze Mobile Photo", use_container_width=True):
                if mobile_photo is None:
                    st.warning("Take a mobile camera photo first.")
                else:
                    try:
                        saved_path = save_uploaded_image(mobile_photo)
                        refreshed_history = load_history()
                        result = inspect_saved_image(
                            saved_path,
                            getattr(mobile_photo, "name", "mobile_camera.jpg"),
                            "mobile_camera",
                            refreshed_history,
                            labels,
                        )
                        st.session_state["batch_results"] = [result]
                        st.rerun()
                    except FileNotFoundError as error:
                        st.error(str(error))

            st.divider()
            section_header("Multi-Camera Capture", "Choose which webcam should capture the apple image.")
            camera_index = st.selectbox(
                "Camera source",
                options=[0, 1, 2, 3, 4, 5],
                index=0,
                format_func=lambda value: f"Camera {value}",
                help="Try Camera 0, then 1, then 2 to find your USB webcams.",
            )
            camera_col1, camera_col2, camera_col3 = st.columns(3)
            test_clicked = camera_col1.button("Test Camera", use_container_width=True)
            instant_clicked = camera_col2.button("Instant Photo", use_container_width=True)
            countdown_clicked = camera_col3.button("3 Second Countdown", use_container_width=True)

            if test_clicked:
                try:
                    preview_path = test_camera_source(camera_index)
                    st.success(f"Camera {camera_index} works.")
                    st.image(str(preview_path), caption=f"Camera {camera_index} preview", use_container_width=True)
                except RuntimeError as error:
                    st.error(str(error))

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

    with website_camera_tab:
        render_website_camera(labels)

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
                corrected = opposite_label(predicted)
                st.markdown(f"**AI prediction:** `{predicted}`")
                st.markdown(f"**If wrong, it will be saved as:** `{corrected}`")

                correct_col, wrong_col = st.columns(2)
                if correct_col.button("AI Is Correct", use_container_width=True):
                    saved = save_feedback(selected, predicted, True)
                    st.success(f"Added to training data as {predicted}: {saved}")
                if wrong_col.button("AI Is Wrong", use_container_width=True):
                    saved = save_feedback(selected, corrected, False)
                    st.success(f"Added corrected training data as {corrected}: {saved}")

                st.divider()
                section_header(
                    "Retrain After Feedback",
                    "Feedback is saved immediately. Retraining updates the actual model, but it can take several minutes.",
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

