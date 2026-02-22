#!/usr/bin/env python3
"""Task 4: Loan approval prediction (binary, imbalanced).

Usage:
    python loan_approval_classification.py --csv /path/to/loan_data.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    best_estimator: Any
    best_params: dict[str, Any]
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: dict[str, float]


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    norm_map = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        if candidate in norm_map:
            return norm_map[candidate]
    return None


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV loaded but is empty.")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    print(f"Rows before cleaning: {before}")
    print(f"Duplicates removed: {removed}")
    print(f"Rows after cleaning: {len(df)}")

    return df


def detect_target_column(df: pd.DataFrame) -> str:
    target_col = find_column(
        df.columns,
        [
            "loanstatus",
            "loanapproved",
            "loanapprovalstatus",
            "status",
            "target",
            "class",
            "y",
        ],
    )
    if target_col is None:
        raise ValueError(
            "Could not detect target column. Expected one of: Loan_Status / loan_approved / status / target / class."
        )
    return target_col


def encode_binary_target(y: pd.Series) -> tuple[pd.Series, dict[Any, int]]:
    # Common dataset values: Y/N, Yes/No, Approved/Rejected, 1/0.
    unique_vals = y.dropna().astype(str).str.strip().str.lower().unique().tolist()

    positive_tokens = {
        "y",
        "yes",
        "approved",
        "approve",
        "1",
        "true",
        "loanapproved",
    }
    negative_tokens = {
        "n",
        "no",
        "rejected",
        "reject",
        "0",
        "false",
        "loandenied",
    }

    if y.dtype.kind in {"i", "u", "f"} and set(pd.Series(y).dropna().unique()).issubset({0, 1}):
        return y.astype(int), {0: 0, 1: 1}

    mapping: dict[Any, int] = {}
    for raw in y.dropna().unique():
        token = str(raw).strip().lower()
        if token in positive_tokens:
            mapping[raw] = 1
        elif token in negative_tokens:
            mapping[raw] = 0

    if len(mapping) < 2:
        # Fallback for binary columns with custom labels.
        classes = sorted(y.dropna().astype(str).unique().tolist())
        if len(classes) != 2:
            raise ValueError("Target must be binary for this task. Detected more than 2 classes.")
        fallback_map = {classes[0]: 0, classes[1]: 1}
        encoded = y.astype(str).map(fallback_map)
        return encoded.astype(int), fallback_map

    encoded = y.map(mapping)
    if encoded.isna().any():
        unknown = y[encoded.isna()].dropna().unique().tolist()
        raise ValueError(f"Unrecognized target labels found: {unknown}")

    return encoded.astype(int), mapping


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def eval_binary(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
    }


def run_logistic_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    with_smote: bool,
) -> ModelResult:
    pre = build_preprocessor(X_train)

    if with_smote:
        if not HAS_IMBLEARN:
            raise RuntimeError("SMOTE requested but imbalanced-learn is not installed.")
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", pre),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        param_dist = {
            "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
            "model__class_weight": [None, "balanced"],
        }
        name = "LogisticRegression + SMOTE"
    else:
        pipe = Pipeline(
            steps=[
                ("preprocessor", pre),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        param_dist = {
            "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
        }
        name = "LogisticRegression (class_weight)"

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=10,
        cv=4,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_prob = best.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return ModelResult(
        name=name,
        best_estimator=best,
        best_params=search.best_params_,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=eval_binary(y_test, y_pred, y_prob),
    )


def run_decision_tree_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    with_smote: bool,
) -> ModelResult:
    pre = build_preprocessor(X_train)

    if with_smote:
        if not HAS_IMBLEARN:
            raise RuntimeError("SMOTE requested but imbalanced-learn is not installed.")
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", pre),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        name = "DecisionTree + SMOTE"
        param_dist = {
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__class_weight": [None, "balanced"],
        }
    else:
        pipe = Pipeline(
            steps=[
                ("preprocessor", pre),
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        name = "DecisionTree (class_weight)"
        param_dist = {
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__criterion": ["gini", "entropy", "log_loss"],
        }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=12,
        cv=4,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_prob = best.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return ModelResult(
        name=name,
        best_estimator=best,
        best_params=search.best_params_,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=eval_binary(y_test, y_pred, y_prob),
    )


def save_confusion(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_precision_recall_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    out_path: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_feature_importance(result: ModelResult, out_dir: str, top_n: int = 20) -> None:
    model = result.best_estimator.named_steps["model"]
    pre = result.best_estimator.named_steps["preprocessor"]

    if hasattr(model, "coef_"):
        # Logistic regression coefficient magnitudes.
        names = pre.get_feature_names_out()
        importance = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        names = pre.get_feature_names_out()
        importance = model.feature_importances_
    else:
        return

    imp_df = (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(os.path.join(out_dir, "feature_importance_best_model.csv"), index=False)

    top = imp_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"Top {top_n} Feature Importance: {result.name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance_best_model.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 - Loan Approval Classification")
    parser.add_argument("--csv", required=True, help="Path to loan approval CSV")
    args = parser.parse_args()

    df = load_and_clean_data(args.csv)
    target_col = detect_target_column(df)

    y_raw = df[target_col]
    y, label_map = encode_binary_target(y_raw)
    X = df.drop(columns=[target_col])

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Detected target column: {target_col}")
    print(f"Target mapping: {label_map}")
    print("Train class distribution:")
    print(y_train.value_counts().sort_index().to_string())

    results: list[ModelResult] = []

    print("\nTraining Logistic Regression (weighted)...")
    results.append(run_logistic_tuned(X_train, y_train, X_test, y_test, with_smote=False))

    print("Training Decision Tree (weighted)...")
    results.append(run_decision_tree_tuned(X_train, y_train, X_test, y_test, with_smote=False))

    if HAS_IMBLEARN:
        print("Training Logistic Regression + SMOTE...")
        results.append(run_logistic_tuned(X_train, y_train, X_test, y_test, with_smote=True))

        print("Training Decision Tree + SMOTE...")
        results.append(run_decision_tree_tuned(X_train, y_train, X_test, y_test, with_smote=True))
    else:
        print("imblearn not installed. Skipping SMOTE bonus models. Install with: pip install imbalanced-learn")

    rows = []
    for r in results:
        row = {"model": r.name, **r.metrics}
        rows.append(row)

        safe_name = r.name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        save_confusion(
            y_test,
            r.y_pred,
            title=f"Confusion Matrix - {r.name}",
            out_path=os.path.join(out_dir, f"confusion_matrix_{safe_name}.png"),
        )
        save_precision_recall_curve(
            y_test,
            r.y_prob,
            model_name=r.name,
            out_path=os.path.join(out_dir, f"pr_curve_{safe_name}.png"),
        )

    metrics_df = pd.DataFrame(rows).sort_values(by=["f1", "recall", "precision"], ascending=False)
    metrics_df.to_csv(os.path.join(out_dir, "model_comparison_binary.csv"), index=False)

    print("\nModel comparison (binary / imbalanced focus):")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_name = metrics_df.iloc[0]["model"]
    best_result = next(r for r in results if r.name == best_name)

    with open(os.path.join(out_dir, "best_model_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_result.name}\n")
        f.write("Best parameters:\n")
        for k, v in best_result.best_params.items():
            f.write(f"- {k}: {v}\n")

    save_feature_importance(best_result, out_dir)

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated files include comparison metrics, confusion matrices, PR curves, and feature importance.")


if __name__ == "__main__":
    main()
