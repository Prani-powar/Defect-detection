import argparse
from pathlib import Path

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from src.utils import load_labels, load_trained_model, predict_image_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an apple is fresh or rotten.")
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    labels = load_labels()
    model = load_trained_model()
    prediction, confidence, probabilities = predict_image_file(model, image_path, labels)

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Class probabilities:")
    for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {label}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
