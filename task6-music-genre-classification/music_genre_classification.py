#!/usr/bin/env python3
"""Task 6: Music genre classification (tabular + image/CNN transfer learning).

Expected dataset layout (GTZAN):
    data_dir/
      blues/*.wav
      classical/*.wav
      country/*.wav
      disco/*.wav
      hiphop/*.wav
      jazz/*.wav
      metal/*.wav
      pop/*.wav
      reggae/*.wav
      rock/*.wav

Usage:
    python music_genre_classification.py --data-dir /path/to/genres_original
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    HAS_TF = True
except Exception:
    HAS_TF = False


RANDOM_STATE = 42


@dataclass
class RunResult:
    model: str
    accuracy: float
    f1_macro: float


def create_output_dir(base: str = "outputs") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def list_audio_files(data_dir: str, max_per_genre: int | None = None) -> pd.DataFrame:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data directory not found or not a folder: {data_dir}")

    rows: list[dict[str, Any]] = []
    for genre_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = sorted([f for f in genre_dir.glob("*.wav")])
        if max_per_genre is not None:
            files = files[:max_per_genre]
        for f in files:
            rows.append({"path": str(f), "genre": genre_dir.name})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No .wav files found under genre subfolders.")
    return df


def extract_tabular_features(audio_path: str, sr: int = 22050, duration: float = 30.0) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    feats = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            chroma.mean(axis=1),
            chroma.std(axis=1),
            [float(spec_centroid.mean()), float(spec_centroid.std())],
            [float(spec_rolloff.mean()), float(spec_rolloff.std())],
            [float(zcr.mean()), float(zcr.std())],
        ]
    )
    return feats


def run_tabular_model(files_df: pd.DataFrame, out_dir: str) -> tuple[RunResult, pd.DataFrame]:
    print("Extracting tabular audio features (MFCC/chroma/spectral)...")
    features: list[np.ndarray] = []
    labels: list[str] = []

    for _, row in files_df.iterrows():
        try:
            vec = extract_tabular_features(row["path"])
            features.append(vec)
            labels.append(row["genre"])
        except Exception:
            continue

    if len(features) < 50:
        raise ValueError("Too few valid audio files for tabular training.")

    X = np.vstack(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    param_dist = {
        "rf__n_estimators": [200, 300, 500],
        "rf__max_depth": [None, 12, 20, 30],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    labels_sorted = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title("Tabular Model Confusion Matrix (RandomForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_tabular_rf.png"), dpi=150)
    plt.close(fig)

    feat_df = pd.DataFrame(X)
    feat_df["genre"] = y
    feat_df.to_csv(os.path.join(out_dir, "tabular_features_dataset.csv"), index=False)

    with open(os.path.join(out_dir, "tabular_best_params.txt"), "w", encoding="utf-8") as f:
        f.write("Best RandomForest parameters (tabular):\n")
        for k, v in search.best_params_.items():
            f.write(f"- {k}: {v}\n")

    result = RunResult(model="Tabular RF (MFCC+features)", accuracy=acc, f1_macro=f1m)
    eval_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    return result, eval_df


def audio_to_mel_image(
    audio_path: str,
    sr: int = 22050,
    duration: float = 30.0,
    n_mels: int = 128,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    mel_img = np.stack([mel_db, mel_db, mel_db], axis=-1)

    mel_img = tf.image.resize(mel_img, target_size).numpy().astype(np.float32)
    return mel_img


def build_transfer_model(num_classes: int, input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.25),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_image_transfer_learning(
    files_df: pd.DataFrame,
    out_dir: str,
    epochs: int,
    batch_size: int,
) -> tuple[RunResult | None, pd.DataFrame | None]:
    if not HAS_TF:
        print("TensorFlow not installed; skipping image-based transfer learning.")
        return None, None

    print("Building mel-spectrogram image dataset for transfer learning...")
    images: list[np.ndarray] = []
    labels: list[str] = []

    for _, row in files_df.iterrows():
        try:
            img = audio_to_mel_image(row["path"])
            images.append(img)
            labels.append(row["genre"])
        except Exception:
            continue

    if len(images) < 50:
        print("Too few valid images for CNN training; skipping image-based approach.")
        return None, None

    X = np.stack(images)
    y_raw = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_transfer_model(num_classes=len(le.classes_))
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    labels_sorted = np.arange(len(le.classes_))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=False, xticks_rotation=45)
    plt.title("Image Model Confusion Matrix (MobileNetV2 Transfer)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_image_transfer.png"), dpi=150)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Transfer Learning Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transfer_learning_training_curve.png"), dpi=150)
    plt.close()

    eval_df = pd.DataFrame(
        {
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred),
        }
    )

    result = RunResult(model="Image Transfer (MobileNetV2)", accuracy=acc, f1_macro=f1m)
    return result, eval_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6 - Music Genre Classification")
    parser.add_argument("--data-dir", required=True, help="Path to GTZAN genre folders")
    parser.add_argument(
        "--max-per-genre",
        type=int,
        default=None,
        help="Optional cap for files per genre (useful for faster experimentation)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for image transfer-learning model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for image model")
    args = parser.parse_args()

    out_dir = create_output_dir("outputs")
    files_df = list_audio_files(args.data_dir, max_per_genre=args.max_per_genre)

    print(f"Total audio files discovered: {len(files_df)}")
    print("Files per genre:")
    print(files_df["genre"].value_counts().sort_index().to_string())

    results: list[RunResult] = []

    tabular_result, tabular_eval = run_tabular_model(files_df, out_dir)
    results.append(tabular_result)
    tabular_eval.to_csv(os.path.join(out_dir, "predictions_tabular.csv"), index=False)

    image_result, image_eval = run_image_transfer_learning(
        files_df,
        out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    if image_result is not None and image_eval is not None:
        results.append(image_result)
        image_eval.to_csv(os.path.join(out_dir, "predictions_image_transfer.csv"), index=False)

    comparison_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("f1_macro", ascending=False)
    comparison_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plt.figure(figsize=(8, 5))
    plt.bar(comparison_df["model"], comparison_df["f1_macro"])
    plt.title("F1-Macro Comparison")
    plt.ylabel("F1-Macro")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_macro_comparison.png"), dpi=150)
    plt.close()

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated: model metrics, confusion matrices, predictions, and comparison chart.")


if __name__ == "__main__":
    main()
