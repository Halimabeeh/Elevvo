# Task 4: Loan Approval Prediction (Binary Classification)

This task predicts whether a loan application will be approved using binary classification with imbalanced-data handling.

Recommended dataset: **Loan-Approval-Prediction-Dataset (Kaggle)**

## Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Matplotlib
- imbalanced-learn (SMOTE bonus)

## Covered Topics

- Binary classification
- Imbalanced data
- Model comparison (Logistic Regression vs Decision Tree)
- Hyperparameter tuning

## Objectives

- Handle missing values and categorical features
- Train/evaluate binary classification models for imbalanced data
- Focus on precision, recall, and F1-score
- Bonus: use SMOTE to address class imbalance

## Run

From this folder:

```bash
python3 loan_approval_classification.py --csv "loan_data.csv"
```

Or with absolute path:

```bash
python3 "/Users/Halima/Documents/New project/task4-loan-approval-classification/loan_approval_classification.py" --csv "/absolute/path/to/loan_data.csv"
```

## What the script does

1. Loads and cleans data (duplicate removal).
2. Detects target column (`Loan_Status` / `status` / `target` variants).
3. Encodes binary target labels (e.g., Y/N, Yes/No, Approved/Rejected).
4. Preprocesses data:
- Numeric imputation
- Categorical imputation + one-hot encoding
5. Trains and tunes baseline models:
- Logistic Regression (`class_weight='balanced'`)
- Decision Tree (`class_weight='balanced'`)
6. Bonus (if `imbalanced-learn` is installed):
- Logistic Regression + SMOTE
- Decision Tree + SMOTE
7. Evaluates and compares models using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
8. Saves confusion matrices, precision-recall curves, and feature importance.

## Outputs

All outputs are saved in `outputs/`:
- `model_comparison_binary.csv`
- `best_model_params.txt`
- `feature_importance_best_model.csv`
- `feature_importance_best_model.png`
- `confusion_matrix_*.png`
- `pr_curve_*.png`
