from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Keep this list small for the first prototype. Add more labels later, for example:
# CLASSES = ["fresh", "rotten", "bruised"]
PRODUCT_NAME = "apple"
CLASSES = ["fresh", "rotten"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 5
CONFIDENCE_THRESHOLD = 0.75
PREDICT_EVERY_N_FRAMES = 10
COOLDOWN_SECONDS = 2

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"
CAPTURED_DIR = PROJECT_ROOT / "data" / "captured"

MODEL_PATH = PROJECT_ROOT / "models" / "product_classifier.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
LOG_CSV_PATH = PROJECT_ROOT / "logs" / "predictions.csv"
LOG_IMAGE_DIR = PROJECT_ROOT / "logs" / "images"

# Future product tasks can be registered here without changing the app scripts.
PRODUCT_TASKS = {
    "apple_quality": {
        "product": PRODUCT_NAME,
        "classes": CLASSES,
        "model_path": MODEL_PATH,
        "labels_path": LABELS_PATH,
    }
}
