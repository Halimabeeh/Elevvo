# Task 6: Music Genre Classification

This task classifies songs into music genres using both:
- Tabular audio features (MFCC/chroma/spectral) with Scikit-learn
- Image-based transfer learning on mel spectrograms (CNN)

Recommended dataset: **GTZAN (Kaggle)**

## Tools & Libraries

- Python
- Librosa
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

## Covered Topics

- Multi-class classification
- Audio data / CNNs

## Objectives

- Classify songs into genres based on extracted audio features
- Preprocess MFCC-like tabular features and/or spectrogram images
- Train and evaluate multi-class models
- Bonus:
- Compare tabular vs image-based approaches
- Use transfer learning on spectrograms

## Dataset Layout

Expected GTZAN folder layout:

```text
genres_original/
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
```

## Run

From this folder:

```bash
python3 music_genre_classification.py --data-dir "/path/to/genres_original"
```

Faster experiment run (fewer files + fewer epochs):

```bash
python3 music_genre_classification.py --data-dir "/path/to/genres_original" --max-per-genre 40 --epochs 3
```

## TensorFlow Note (Image-Based Bonus)

- The tabular approach runs with the main repo `requirements.txt`.
- Image-based transfer learning needs TensorFlow, which is typically unavailable on Python 3.14.
- For the image bonus, create a Python 3.11 virtual environment and install:

```bash
pip install -r requirements-image.txt
```

## What the script does

1. Loads all `.wav` files from genre folders.
2. Tabular pipeline:
- Extracts MFCC/chroma/spectral summary features
- Trains/tunes RandomForest classifier
- Saves confusion matrix and predictions
3. Image pipeline (bonus):
- Converts audio to mel-spectrogram images
- Uses MobileNetV2 transfer learning
- Saves confusion matrix, training curve, and predictions
4. Compares both approaches using accuracy and macro-F1.

## Outputs

All outputs are saved in `outputs/`:
- `model_comparison.csv`
- `f1_macro_comparison.png`
- `predictions_tabular.csv`
- `confusion_matrix_tabular_rf.png`
- `tabular_best_params.txt`
- `tabular_features_dataset.csv`
- `predictions_image_transfer.csv` (if TensorFlow available)
- `confusion_matrix_image_transfer.png` (if TensorFlow available)
- `transfer_learning_training_curve.png` (if TensorFlow available)
