import queue
import threading
import time

import cv2

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import CAPTURED_DIR, COOLDOWN_SECONDS, LOG_IMAGE_DIR, PREDICT_EVERY_N_FRAMES
from src.utils import (
    draw_status,
    ensure_directories,
    is_confident,
    load_labels,
    load_trained_model,
    predict_image_file,
    timestamp_string,
    write_prediction_log,
)


def camera_worker(image_queue: queue.Queue, stop_event: threading.Event) -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Check camera permission or connect a webcam.")
        stop_event.set()
        return

    frame_number = 0
    last_capture_time = 0.0
    while not stop_event.is_set():
        ok, frame = camera.read()
        if not ok:
            print("Could not read from webcam.")
            stop_event.set()
            break

        frame_number += 1
        if frame_number % PREDICT_EVERY_N_FRAMES == 0 and time.time() - last_capture_time >= COOLDOWN_SECONDS:
            image_path = CAPTURED_DIR / f"queue_{timestamp_string()}.jpg"
            cv2.imwrite(str(image_path), frame)
            image_queue.put(image_path)
            last_capture_time = time.time()

        draw_status(frame, "queueing", 0.0, {"fresh": 0, "rotten": 0, "uncertain": 0})
        cv2.imshow("Queue Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    camera.release()
    cv2.destroyAllWindows()


def prediction_worker(image_queue: queue.Queue, stop_event: threading.Event) -> None:
    model = load_trained_model()
    labels = load_labels()
    counts = {label: 0 for label in labels}
    counts["uncertain"] = 0

    while not stop_event.is_set() or not image_queue.empty():
        try:
            image_path = image_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        prediction, confidence, _ = predict_image_file(model, image_path, labels)
        label = prediction if is_confident(confidence) else "uncertain"
        counts[label] = counts.get(label, 0) + 1
        write_prediction_log(str(image_path), label, confidence, counts)
        print(f"{image_path.name}: {label} ({confidence * 100:.2f}%) | {counts}")
        image_queue.task_done()


def main() -> None:
    ensure_directories([CAPTURED_DIR, LOG_IMAGE_DIR])
    image_queue = queue.Queue()
    stop_event = threading.Event()

    capture_thread = threading.Thread(target=camera_worker, args=(image_queue, stop_event), daemon=True)
    predict_thread = threading.Thread(target=prediction_worker, args=(image_queue, stop_event), daemon=True)

    capture_thread.start()
    predict_thread.start()
    capture_thread.join()
    image_queue.join()
    stop_event.set()
    predict_thread.join(timeout=2)


if __name__ == "__main__":
    main()
