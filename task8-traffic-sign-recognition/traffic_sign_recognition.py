#!/usr/bin/env python3
"""Task 8: Traffic sign recognition (CNN, multi-class).

Expected dataset layout:
  data_dir/
    0/*.png|jpg
    1/*.png|jpg
    ...

Usage:
  python traffic_sign_recognition.py --data-dir /path/to/Train
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    HAS_TF = True
except Exception:
    HAS_TF = False


RANDOM_STATE = 42
IMG_SIZE = (64, 64)


@dataclass
class ModelResult:
    model: str
    accuracy: float
    f1_macro: float


def create_output_dir(base: str = "outputs") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def list_images(data_dir: str, max_per_class: int | None = None) -> pd.DataFrame:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    rows: list[dict[str, Any]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.ppm", "*.bmp"):
            imgs.extend(class_dir.glob(ext))
        imgs = sorted(imgs)
        if max_per_class is not None:
            imgs = imgs[:max_per_class]

        for img in imgs:
            rows.append({"path": str(img), "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No images found under class subfolders.")
    return df


def load_image(path: str, size: tuple[int, int] = IMG_SIZE) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def build_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[int, str]]:
    label_names = sorted(df["label"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    images: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        arr = load_image(row["path"])
        if arr is None:
            continue
        images.append(arr)
        labels.append(label_to_idx[row["label"]])

    if len(images) < 100:
        raise ValueError("Too few valid images loaded.")

    X = np.stack(images)
    y = np.array(labels)
    return X, y, label_to_idx, idx_to_label


def get_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_custom_cnn(num_classes: int) -> tf.keras.Model:
    aug = get_augmentation()

    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            aug,
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(num_classes: int) -> tf.keras.Model:
    aug = get_augmentation()

    base = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = aug(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_eval(
    model: tf.keras.Model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    epochs: int,
    batch_size: int,
    idx_to_label: dict[int, str],
) -> tuple[ModelResult, pd.DataFrame]:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    labels_sorted = np.arange(len(idx_to_label))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=labels_sorted),
        display_labels=[idx_to_label[i] for i in labels_sorted],
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=False, xticks_rotation=90)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{safe}.png"), dpi=150)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title(f"Training Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"training_curve_{safe}.png"), dpi=150)
    plt.close()

    pred_df = pd.DataFrame(
        {
            "y_true": [idx_to_label[i] for i in y_test],
            "y_pred": [idx_to_label[i] for i in y_pred],
        }
    )

    return ModelResult(model=model_name, accuracy=acc, f1_macro=f1m), pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 8 - Traffic Sign Recognition")
    parser.add_argument("--data-dir", required=True, help="Path to training image folders")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap per class for faster runs")
    parser.add_argument("--epochs", type=int, default=12, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    if not HAS_TF:
        raise RuntimeError(
            "TensorFlow is required for Task 8. Use Python 3.11 environment and install tensorflow."
        )

    out_dir = create_output_dir("outputs")

    df = list_images(args.data_dir, max_per_class=args.max_per_class)
    print(f"Total images found: {len(df)}")
    print("Class counts:")
    print(df["label"].value_counts().sort_index().to_string())

    X, y, _, idx_to_label = build_dataset(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    results: list[ModelResult] = []

    print("\nTraining custom CNN...")
    custom_model = build_custom_cnn(num_classes=len(idx_to_label))
    custom_res, custom_pred_df = train_and_eval(
        custom_model,
        "Custom CNN",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        idx_to_label=idx_to_label,
    )
    results.append(custom_res)
    custom_pred_df.to_csv(os.path.join(out_dir, "predictions_custom_cnn.csv"), index=False)

    print("\nTraining transfer model (MobileNetV2)...")
    transfer_model = build_transfer_model(num_classes=len(idx_to_label))
    transfer_res, transfer_pred_df = train_and_eval(
        transfer_model,
        "MobileNetV2 Transfer",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        out_dir,
        epochs=max(6, args.epochs // 2),
        batch_size=args.batch_size,
        idx_to_label=idx_to_label,
    )
    results.append(transfer_res)
    transfer_pred_df.to_csv(os.path.join(out_dir, "predictions_mobilenet_transfer.csv"), index=False)

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("f1_macro", ascending=False)
    results_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["accuracy"])
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated: model comparison, confusion matrices, training curves, and predictions.")


if __name__ == "__main__":
    main()
