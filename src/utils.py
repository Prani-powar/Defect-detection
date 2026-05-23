import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import (
    CLASSES,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    LABELS_PATH,
    LOG_CSV_PATH,
    LOG_IMAGE_DIR,
    MODEL_PATH,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_FIELDNAMES = [
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


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_labels(labels: List[str], path: Path = LABELS_PATH) -> None:
    ensure_directories([path.parent])
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_labels(path: Path = LABELS_PATH) -> List[str]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return CLASSES


def load_trained_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with: python src/train_model.py"
        )
    return load_model(model_path)


def preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE)
    array = preprocess_input(resized.astype(np.float32))
    return np.expand_dims(array, axis=0)


def predict_array(model, image_array: np.ndarray, labels: List[str]) -> Tuple[str, float, Dict[str, float]]:
    probabilities = model.predict(image_array, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    class_probabilities = {
        labels[index]: float(probabilities[index]) for index in range(len(labels))
    }
    return labels[best_index], float(probabilities[best_index]), class_probabilities


def predict_image_file(model, image_path: Path, labels: List[str]) -> Tuple[str, float, Dict[str, float]]:
    return predict_array(model, preprocess_image(image_path), labels)


def is_confident(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    return confidence >= threshold


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def ensure_log_schema(csv_path: Path = LOG_CSV_PATH) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        old_fieldnames = reader.fieldnames or []
        rows = list(reader)
    if old_fieldnames == LOG_FIELDNAMES:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, "") for field in LOG_FIELDNAMES}
            if not normalized["is_rotten"] and normalized["predicted_class"]:
                normalized["is_rotten"] = str(normalized["predicted_class"] == "rotten")
            writer.writerow(normalized)


def write_prediction_log(
    image_path: str,
    predicted_class: str,
    confidence: float,
    counts: Dict[str, int],
    csv_path: Path = LOG_CSV_PATH,
    probabilities: Dict[str, float] | None = None,
    source: str = "camera",
    original_filename: str = "",
) -> None:
    ensure_directories([csv_path.parent, LOG_IMAGE_DIR])
    ensure_log_schema(csv_path)
    file_exists = csv_path.exists()
    probabilities = probabilities or {}
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "original_filename": original_filename,
        "image_path": image_path,
        "predicted_class": predicted_class,
        "is_rotten": str(predicted_class == "rotten"),
        "confidence": f"{confidence:.4f}",
        "fresh_probability": f"{probabilities.get('fresh', 0.0):.4f}",
        "rotten_probability": f"{probabilities.get('rotten', 0.0):.4f}",
        "fresh_count": counts.get("fresh", 0),
        "rotten_count": counts.get("rotten", 0),
        "uncertain_count": counts.get("uncertain", 0),
    }
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def center_roi(frame: np.ndarray, width_ratio: float = 0.45, height_ratio: float = 0.55):
    height, width = frame.shape[:2]
    roi_width = int(width * width_ratio)
    roi_height = int(height * height_ratio)
    x1 = (width - roi_width) // 2
    y1 = (height - roi_height) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return x1, y1, x2, y2


def draw_status(frame: np.ndarray, label: str, confidence: float, counts: Dict[str, int]) -> None:
    x1, y1, x2, y2 = center_roi(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 190, 0), 2)
    color = (0, 180, 0) if label == "fresh" else (0, 0, 220)
    if label == "uncertain":
        color = (0, 165, 255)
    cv2.putText(frame, f"{label}: {confidence * 100:.1f}%", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Fresh: {counts.get('fresh', 0)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 0), 2)
    cv2.putText(frame, f"Rotten: {counts.get('rotten', 0)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 220), 2)
    cv2.putText(frame, f"Uncertain: {counts.get('uncertain', 0)}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
