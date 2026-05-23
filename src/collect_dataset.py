from pathlib import Path
import time

import cv2

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import CLASSES, RAW_DATA_DIR
from src.utils import ensure_directories, timestamp_string


SAVE_DEBOUNCE_SECONDS = 0.6
KEY_TO_CLASS = {"f": "fresh", "r": "rotten"}


def save_frame(frame, class_name: str, counter: int) -> Path:
    output_dir = RAW_DATA_DIR / class_name
    ensure_directories([output_dir])
    filename = f"{class_name}_{timestamp_string()}_{counter:03d}.jpg"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), frame)
    return output_path


def main() -> None:
    for class_name in CLASSES:
        ensure_directories([RAW_DATA_DIR / class_name])

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Check camera permission or connect a webcam.")
        return

    counters = {class_name: len(list((RAW_DATA_DIR / class_name).glob("*.jpg"))) for class_name in CLASSES}
    last_save_time = 0.0

    print("Press f to save a fresh apple, r to save a rotten apple, q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Could not read from webcam.")
            break

        cv2.putText(frame, "f=fresh  r=rotten  q=quit", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2)
        y = 75
        for class_name in CLASSES:
            cv2.putText(frame, f"{class_name}: {counters[class_name]}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 32
        cv2.imshow("Apple Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        pressed = chr(key) if key != 255 else ""
        if pressed in KEY_TO_CLASS and time.time() - last_save_time >= SAVE_DEBOUNCE_SECONDS:
            class_name = KEY_TO_CLASS[pressed]
            counters[class_name] += 1
            output_path = save_frame(frame, class_name, counters[class_name])
            last_save_time = time.time()
            print(f"Saved {class_name}: {output_path}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
