# Task 8: Traffic Sign Recognition

This task performs multi-class traffic sign classification using deep learning.

Recommended dataset: **GTSRB (Kaggle)**

## Tools & Libraries

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Covered Topics

- Computer vision (CNN)
- Multi-class classification

## Objectives

- Classify traffic sign images into their classes
- Preprocess images (resize + normalization)
- Train a CNN model and evaluate with accuracy + confusion matrix
- Bonus:
- Apply data augmentation
- Compare custom CNN vs pre-trained MobileNet transfer learning

## Dataset Layout

Expected folder-style dataset:

```text
Train/
  0/*.png
  1/*.png
  ...
  42/*.png
```

## Run

From this folder:

```bash
python3 traffic_sign_recognition.py --data-dir "/path/to/Train"
```

Faster run:

```bash
python3 traffic_sign_recognition.py --data-dir "/path/to/Train" --max-per-class 400 --epochs 8
```

## What the script does

1. Loads images from class folders.
2. Preprocesses images:
- Resize to `64x64`
- Normalize pixel values to `[0,1]`
3. Splits into train/validation/test.
4. Trains two models:
- Custom CNN
- MobileNetV2 transfer model
5. Uses data augmentation in training (bonus).
6. Evaluates and compares models using accuracy and macro-F1.
7. Saves confusion matrices and training curves.

## Outputs

All outputs are saved in `outputs/`:
- `model_comparison.csv`
- `accuracy_comparison.png`
- `confusion_matrix_custom_cnn.png`
- `confusion_matrix_mobilenetv2_transfer.png`
- `training_curve_custom_cnn.png`
- `training_curve_mobilenetv2_transfer.png`
- `predictions_custom_cnn.csv`
- `predictions_mobilenet_transfer.csv`

## Environment Note

TensorFlow may not be available on Python 3.14. Use a Python 3.11 environment for this task.
