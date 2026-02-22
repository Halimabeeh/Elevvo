#!/usr/bin/env python3
"""Task 3: Forest cover type prediction (multi-class + imbalanced binary).

Usage:
    python forest_cover_classification.py --csv /path/to/covtype.csv
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RANDOM_STATE = 42


@dataclass
class ModelRun:
    name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    y_pred: np.ndarray
    metrics: dict[str, float]


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV loaded but is empty.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    print(f"Rows before cleaning: {before}")
    print(f"Duplicates removed: {removed}")
    print(f"Rows after cleaning: {len(cleaned)}")
    return cleaned


def create_output_dir(base: str = "outputs") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


def evaluate_multiclass(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def run_random_forest_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fast: bool = False,
) -> ModelRun:
    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )

    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 12, 18, 25],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced_subsample"],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=6 if fast else 16,
        cv=2 if fast else 3,
        scoring="f1_weighted",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    return ModelRun(
        name="RandomForest",
        best_estimator=best_model,
        best_params=search.best_params_,
        y_pred=y_pred,
        metrics=evaluate_multiclass(y_test, y_pred),
    )


def run_xgboost_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fast: bool = False,
) -> ModelRun | None:
    if not HAS_XGBOOST:
        return None

    # XGBoost expects multiclass targets encoded as 0..num_class-1.
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    preprocessor = build_preprocessor(X_train)
    n_classes = int(pd.Series(y_train_enc).nunique())

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )

    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1, 5, 10],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=6 if fast else 14,
        cv=2 if fast else 3,
        scoring="f1_weighted",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train_enc)
    best_model = search.best_estimator_
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc.astype(int))

    return ModelRun(
        name="XGBoost",
        best_estimator=best_model,
        best_params=search.best_params_,
        y_pred=y_pred,
        metrics=evaluate_multiclass(y_test, y_pred),
    )


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_feature_importance(model_run: ModelRun, out_dir: str, top_n: int = 20) -> None:
    preprocessor = model_run.best_estimator.named_steps["preprocessor"]
    model = model_run.best_estimator.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print(f"Skipping feature importance for {model_run.name}: model does not expose feature_importances_.")
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        print("Skipping feature importance plot: feature-name and importance length mismatch.")
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    top_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"Top {top_n} Feature Importances ({model_run.name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance_best_model.png"), dpi=150)
    plt.close()

    importance_df.to_csv(os.path.join(out_dir, "feature_importance_best_model.csv"), index=False)


def run_binary_imbalanced_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: str,
    fast: bool = False,
) -> pd.DataFrame:
    class_counts = y_train.value_counts()
    minority_class = class_counts.idxmin()

    y_train_bin = (y_train == minority_class).astype(int)
    y_test_bin = (y_test == minority_class).astype(int)

    print("\nBinary imbalanced experiment")
    print(f"Minority class selected as positive class: {minority_class}")
    print(
        f"Train class balance (0/1): {(y_train_bin == 0).sum()}/{(y_train_bin == 1).sum()}"
    )

    preprocessor = build_preprocessor(X_train)

    rf_bin = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120 if fast else 300,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_bin.fit(X_train, y_train_bin)
    rf_pred = rf_bin.predict(X_test)
    rf_prob = rf_bin.predict_proba(X_test)[:, 1]

    results = [
        {
            "model": "RandomForest (binary, balanced)",
            "precision": float(precision_score(y_test_bin, rf_pred, zero_division=0)),
            "recall": float(recall_score(y_test_bin, rf_pred, zero_division=0)),
            "f1": float(f1_score(y_test_bin, rf_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test_bin, rf_prob)),
        }
    ]

    save_confusion_matrix(
        y_test_bin,
        rf_pred,
        "Binary Confusion Matrix - RandomForest",
        os.path.join(out_dir, "binary_confusion_matrix_randomforest.png"),
    )

    if HAS_XGBOOST and not fast:
        neg = int((y_train_bin == 0).sum())
        pos = int((y_train_bin == 1).sum())
        scale_pos_weight = float(neg / max(pos, 1))

        xgb_bin = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        xgb_bin.fit(X_train, y_train_bin)
        xgb_pred = xgb_bin.predict(X_test)
        xgb_prob = xgb_bin.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": "XGBoost (binary, weighted)",
                "precision": float(precision_score(y_test_bin, xgb_pred, zero_division=0)),
                "recall": float(recall_score(y_test_bin, xgb_pred, zero_division=0)),
                "f1": float(f1_score(y_test_bin, xgb_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test_bin, xgb_prob)),
            }
        )

        save_confusion_matrix(
            y_test_bin,
            xgb_pred,
            "Binary Confusion Matrix - XGBoost",
            os.path.join(out_dir, "binary_confusion_matrix_xgboost.png"),
        )

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 - Forest Cover Classification")
    parser.add_argument("--csv", required=True, help="Path to Covertype CSV file")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings (fewer CV folds/iterations and lighter binary stage).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fraction of rows to use (0 < value <= 1.0), stratified by target.",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    df = clean_data(df)

    target_col = find_column(
        df.columns,
        [
            "covertype",
            "covertype",
            "target",
            "class",
            "label",
        ],
    )
    if target_col is None:
        raise ValueError(
            "Could not detect target column. Expected one of: Cover_Type / cover_type / target / class / label."
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if not (0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be > 0 and <= 1.0")
    if args.sample_frac < 1.0:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=args.sample_frac,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        print(f"Using stratified sample fraction: {args.sample_frac:.2f} ({len(X)} rows)")

    out_dir = create_output_dir("outputs")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    runs: list[ModelRun] = []

    print("\nTraining RandomForest with hyperparameter tuning...")
    rf_run = run_random_forest_tuning(X_train, y_train, X_test, y_test, fast=args.fast)
    runs.append(rf_run)

    print("Training XGBoost with hyperparameter tuning...")
    xgb_run = run_xgboost_tuning(X_train, y_train, X_test, y_test, fast=args.fast)
    if xgb_run is not None:
        runs.append(xgb_run)
    else:
        print("XGBoost not installed. Skipping XGBoost comparison. Install with: pip install xgboost")

    metrics_rows = []
    for run in runs:
        row = {
            "model": run.name,
            **run.metrics,
        }
        metrics_rows.append(row)

        report = classification_report(y_test, run.y_pred)
        with open(os.path.join(out_dir, f"classification_report_{run.name.lower()}.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        save_confusion_matrix(
            y_test,
            run.y_pred,
            title=f"Confusion Matrix - {run.name}",
            out_path=os.path.join(out_dir, f"confusion_matrix_{run.name.lower()}.png"),
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1_weighted", ascending=False)
    metrics_df.to_csv(os.path.join(out_dir, "model_comparison_multiclass.csv"), index=False)

    print("\nMulti-class model comparison:")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_model_name = metrics_df.iloc[0]["model"]
    best_run = next(run for run in runs if run.name == best_model_name)

    with open(os.path.join(out_dir, "best_model_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_run.name}\n")
        f.write("Best parameters:\n")
        for key, value in best_run.best_params.items():
            f.write(f"- {key}: {value}\n")

    save_feature_importance(best_run, out_dir)

    binary_metrics_df = run_binary_imbalanced_experiment(
        X_train,
        X_test,
        y_train,
        y_test,
        out_dir,
        fast=args.fast,
    )
    binary_metrics_df.to_csv(os.path.join(out_dir, "binary_imbalanced_metrics.csv"), index=False)

    print("\nBinary imbalanced metrics:")
    print(binary_metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated: model comparison, confusion matrices, classification reports, and feature importances.")


if __name__ == "__main__":
    main()
