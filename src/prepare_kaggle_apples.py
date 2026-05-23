import argparse
import re
import shutil
from pathlib import Path

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import CLASSES, RAW_DATA_DIR
from src.utils import ensure_directories, image_files


APPLE_TOKENS = {"apple", "apples", "freshapple", "rottenapple"}
FRESH_TOKENS = {"fresh", "freshapple", "healthy", "good", "ripe"}
ROTTEN_TOKENS = {"rotten", "rottenapple", "bad", "defect", "defective", "damaged", "spoiled"}


def normalize_text(path: Path) -> list[str]:
    text = " ".join(path.parts).lower()
    return [token for token in re.split(r"[^a-z0-9]+", text) if token]


def has_apple_context(tokens: list[str]) -> bool:
    joined = " ".join(tokens)
    return any(token in tokens or token in joined for token in APPLE_TOKENS)


def target_class_for(path: Path) -> str | None:
    tokens = normalize_text(path)
    joined = " ".join(tokens)
    if not has_apple_context(tokens):
        return None
    if any(token in tokens or token in joined for token in ROTTEN_TOKENS):
        return "rotten"
    if any(token in tokens or token in joined for token in FRESH_TOKENS):
        return "fresh"
    return None


def import_dataset(source_dir: Path) -> None:
    if not source_dir.exists():
        print(f"Dataset folder not found: {source_dir}")
        return

    counts = {"fresh": 0, "rotten": 0, "ignored": 0}
    ensure_directories([RAW_DATA_DIR / class_name for class_name in CLASSES])

    for image_path in image_files(source_dir):
        class_name = target_class_for(image_path)
        if class_name is None:
            counts["ignored"] += 1
            continue
        counts[class_name] += 1
        output_name = f"kaggle_{class_name}_{counts[class_name]:06d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, RAW_DATA_DIR / class_name / output_name)

    print("Import complete.")
    print(f"Fresh: {counts['fresh']}")
    print(f"Rotten: {counts['rotten']}")
    print(f"Ignored: {counts['ignored']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize downloaded Kaggle apple datasets into data/raw/fresh and data/raw/rotten."
    )
    parser.add_argument("--source", required=True, help="Folder that contains the extracted Kaggle dataset")
    args = parser.parse_args()
    import_dataset(Path(args.source))


if __name__ == "__main__":
    main()
