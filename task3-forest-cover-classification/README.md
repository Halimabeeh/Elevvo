# Task 3: Forest Cover Type Classification

This task builds classification models to predict forest cover type from cartographic and environmental features.

Recommended dataset: **Covertype (UCI)**

## Tools & Libraries

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

## Covered Topics

- Multi-class classification
- Tree-based modeling
- Hyperparameter tuning
- Binary classification on imbalanced data (derived experiment)

## Objectives

- Clean and preprocess data, including categorical handling
- Compare different models (Random Forest vs XGBoost)
- Perform hyperparameter tuning
- Evaluate with multi-class metrics
- Visualize confusion matrix and feature importance
- Bonus: include binary imbalanced-data analysis

## Run

From this folder:

```bash
python3 forest_cover_classification.py --csv "covtype.csv"
```

Or with absolute path:

```bash
python3 "/Users/Halima/Documents/New project/task3-forest-cover-classification/forest_cover_classification.py" --csv "/absolute/path/to/covtype.csv"
```

## What the script does

1. Loads and cleans the dataset (duplicate removal).
2. Detects target column (`Cover_Type`/`target`/`class` variants).
3. Preprocesses features:
- Numeric imputation
- Categorical imputation + one-hot encoding
4. Splits data using stratification.
5. Trains/tunes Random Forest using `RandomizedSearchCV`.
6. Trains/tunes XGBoost using `RandomizedSearchCV` (if `xgboost` is installed).
7. Compares model metrics:
- Accuracy
- Macro F1
- Weighted F1
8. Saves confusion matrices and classification reports.
9. Saves top feature importances for the best model.
10. Runs a binary imbalanced experiment (minority class vs rest) and reports precision/recall/F1/ROC-AUC.

## Outputs

All outputs are saved in `outputs/`:
- `model_comparison_multiclass.csv`
- `classification_report_randomforest.txt`
- `classification_report_xgboost.txt` (if XGBoost available)
- `confusion_matrix_randomforest.png`
- `confusion_matrix_xgboost.png` (if XGBoost available)
- `best_model_params.txt`
- `feature_importance_best_model.png`
- `feature_importance_best_model.csv`
- `binary_imbalanced_metrics.csv`
- `binary_confusion_matrix_randomforest.png`
- `binary_confusion_matrix_xgboost.png` (if XGBoost available)
