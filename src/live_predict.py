import shutil
import time

import cv2

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import (
    CAPTURED_DIR,
    CONFIDENCE_THRESHOLD,
    LOG_IMAGE_DIR,
    PREDICT_EVERY_N_FRAMES,
    COOLDOWN_SECONDS,
)
from src.utils import (
    draw_status,
    ensure_directories,
    is_confident,
    load_labels,
    load_trained_model,
    preprocess_frame,
    predict_array,
    timestamp_string,
    write_prediction_log,
)


def main() -> None:
    ensure_directories([CAPTURED_DIR, LOG_IMAGE_DIR])
    model = load_trained_model()
    labels = load_labels()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Check camera permission or connect a webcam.")
        return

    counts = {label: 0 for label in labels}
    counts["uncertain"] = 0
    frame_number = 0
    last_count_time = 0.0
    latest_label = "waiting"
    latest_confidence = 0.0

    print("Press q to quit live prediction.")
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Could not read from webcam.")
            break

        frame_number += 1
        now = time.time()
        can_count = now - last_count_time >= COOLDOWN_SECONDS

        if frame_number % PREDICT_EVERY_N_FRAMES == 0 and can_count:
            prediction, confidence, _ = predict_array(model, preprocess_frame(frame), labels)
            latest_label = prediction if is_confident(confidence) else "uncertain"
            latest_confidence = confidence

            image_name = f"{timestamp_string()}_{latest_label}.jpg"
            captured_path = CAPTURED_DIR / image_name
            log_image_path = LOG_IMAGE_DIR / image_name
            cv2.imwrite(str(captured_path), frame)
            shutil.copy2(captured_path, log_image_path)

            counts[latest_label] = counts.get(latest_label, 0) + 1
            write_prediction_log(str(log_image_path), latest_label, confidence, counts)
            if latest_label != "uncertain" and confidence >= CONFIDENCE_THRESHOLD:
                last_count_time = now

        draw_status(frame, latest_label, latest_confidence, counts)
        cv2.imshow("Live Apple Quality Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
