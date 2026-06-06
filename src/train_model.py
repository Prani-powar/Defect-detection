import csv
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import bootstrap  # noqa: F401
except ModuleNotFoundError:
    import src.bootstrap  # noqa: F401
from config import (
    BATCH_SIZE,
    CLASSES,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


AUTOTUNE = tf.data.AUTOTUNE
HISTORY_CSV = MODEL_PATH.parent.parent / "logs" / "training_history.csv"
REPORT_PATH = MODEL_PATH.parent.parent / "logs" / "classification_report.txt"


def ensure_directories(paths) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_labels(labels, path=MODEL_PATH.parent / "labels.json") -> None:
    ensure_directories([path.parent])
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def build_dataset(folder, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def prepare_dataset(dataset, augment: bool = False):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.15),
        ]
    )

    def prepare(images, labels):
        if augment:
            images = data_augmentation(images, training=True)
        return preprocess_input(images), labels

    return dataset.map(prepare, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def class_weights_from_dataset(dataset) -> dict[int, float]:
    class_ids = []
    for _, labels in dataset:
        class_ids.extend(np.argmax(labels.numpy(), axis=1))
    if not class_ids:
        return {}
    counts = np.bincount(np.asarray(class_ids), minlength=len(CLASSES))
    total = int(np.sum(counts))
    weights = np.zeros(len(CLASSES), dtype=np.float32)
    for index, count in enumerate(counts):
        weights[index] = total / (len(CLASSES) * count) if count else 0.0
    return {index: float(weight) for index, weight in enumerate(weights)}


def build_model(num_classes: int):
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def plot_history(histories) -> None:
    ensure_directories([MODEL_PATH.parent, MODEL_PATH.parent.parent / "logs"])
    rows = []
    epoch_number = 1
    for history in histories:
        history_data = history.history
        for offset in range(len(history_data.get("loss", []))):
            rows.append(
                {
                    "epoch": epoch_number,
                    "accuracy": history_data.get("accuracy", [""])[offset],
                    "val_accuracy": history_data.get("val_accuracy", [""])[offset],
                    "loss": history_data.get("loss", [""])[offset],
                    "val_loss": history_data.get("val_loss", [""])[offset],
                }
            )
            epoch_number += 1

    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "accuracy", "val_accuracy", "loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model, test_dataset) -> None:
    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probabilities, axis=1))

    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

    lines = ["Classification report", ""]
    for index, class_name in enumerate(CLASSES):
        true_positive = matrix[index, index]
        predicted_total = matrix[:, index].sum()
        actual_total = matrix[index, :].sum()
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        lines.append(
            f"{class_name}: precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1:.4f}, support={actual_total}"
        )
    lines.extend(["", "Confusion matrix", str(matrix)])

    ensure_directories([REPORT_PATH.parent])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Confusion matrix:")
    print(matrix)


def main() -> None:
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("Dataset folders are missing. Run python src/split_dataset.py first.")
        return

    train_raw = build_dataset(TRAIN_DIR, shuffle=True)
    val_raw = build_dataset(VAL_DIR, shuffle=False)
    test_raw = build_dataset(TEST_DIR, shuffle=False)
    class_weight = class_weights_from_dataset(train_raw)

    train_dataset = prepare_dataset(train_raw, augment=True)
    val_dataset = prepare_dataset(val_raw)
    test_dataset = prepare_dataset(test_raw)

    model, base_model = build_model(len(CLASSES))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ]

    history_head = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_fine = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    ensure_directories([MODEL_PATH.parent])
    model.save(MODEL_PATH)
    save_labels(CLASSES)
    plot_history([history_head, history_fine])
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    evaluate_model(model, test_dataset)


if __name__ == "__main__":
    main()
