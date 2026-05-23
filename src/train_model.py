import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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
from src.utils import ensure_directories, save_labels


AUTOTUNE = tf.data.AUTOTUNE


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
    logs_dir = MODEL_PATH.parent.parent / "logs"
    accuracy = []
    val_accuracy = []
    loss = []
    val_loss = []
    for history in histories:
        accuracy.extend(history.history.get("accuracy", []))
        val_accuracy.extend(history.history.get("val_accuracy", []))
        loss.extend(history.history.get("loss", []))
        val_loss.extend(history.history.get("val_loss", []))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label="train accuracy")
    plt.plot(val_accuracy, label="val accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(logs_dir / "training_curves.png", dpi=160)
    plt.close()


def evaluate_model(model, test_dataset) -> None:
    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probabilities, axis=1))

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    matrix = confusion_matrix(y_true, y_pred)
    logs_dir = MODEL_PATH.parent.parent / "logs"
    ensure_directories([logs_dir])
    (logs_dir / "classification_report.txt").write_text(
        f"Classification report\n\n{report}\n\nConfusion matrix\n{matrix}\n",
        encoding="utf-8",
    )
    print(report)
    print("Confusion matrix:")
    print(matrix)


def main() -> None:
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("Dataset folders are missing. Run python src/split_dataset.py first.")
        return

    train_raw = build_dataset(TRAIN_DIR, shuffle=True)
    val_raw = build_dataset(VAL_DIR, shuffle=False)
    test_raw = build_dataset(TEST_DIR, shuffle=False)

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
