import random
import shutil
from pathlib import Path

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import CLASSES, RAW_DATA_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.utils import ensure_directories, image_files


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def clear_split_dirs() -> None:
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            ensure_directories([class_dir])
            for image_path in image_files(class_dir):
                image_path.unlink()


def copy_files(files, output_dir: Path) -> None:
    ensure_directories([output_dir])
    for source_path in files:
        shutil.copy2(source_path, output_dir / source_path.name)


def split_class(class_name: str) -> None:
    source_dir = RAW_DATA_DIR / class_name
    files = image_files(source_dir)
    if not files:
        print(f"No images found for {class_name} in {source_dir}")
        return

    random.Random(RANDOM_SEED).shuffle(files)
    train_end = int(len(files) * TRAIN_RATIO)
    val_end = train_end + int(len(files) * VAL_RATIO)

    copy_files(files[:train_end], TRAIN_DIR / class_name)
    copy_files(files[train_end:val_end], VAL_DIR / class_name)
    copy_files(files[val_end:], TEST_DIR / class_name)
    print(
        f"{class_name}: {train_end} train, {val_end - train_end} val, {len(files) - val_end} test"
    )


def main() -> None:
    clear_split_dirs()
    for class_name in CLASSES:
        split_class(class_name)


if __name__ == "__main__":
    main()
