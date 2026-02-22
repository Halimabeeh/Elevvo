#!/usr/bin/env python3
"""Student performance analysis: regression + clustering.

Usage:
    python student_performance_analysis.py --csv /path/to/student_performance.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


RANDOM_STATE = 42


@dataclass
class ExperimentResult:
    name: str
    r2: float
    mae: float
    rmse: float


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(c): c for c in columns}
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


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    print(f"Rows before cleaning: {before}")
    print(f"Duplicates removed: {removed}")
    print(f"Rows after cleaning: {len(cleaned)}")
    return cleaned


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray) -> ExperimentResult:
    return ExperimentResult(
        name=name,
        r2=r2_score(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def run_linear_experiment(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[ExperimentResult, np.ndarray]:
    preprocessor = build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    result = evaluate_model(name, y_test, preds)
    return result, preds


def run_polynomial_experiment(
    study_col: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    degree: int,
) -> tuple[ExperimentResult, np.ndarray]:
    if study_col not in X_train.columns:
        raise ValueError(
            f"Study-hours column '{study_col}' not found in X for polynomial regression."
        )

    # Polynomial experiment is intentionally kept on study hours only for interpretability.
    poly_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("regressor", LinearRegression()),
        ]
    )

    X_train_poly = X_train[[study_col]]
    X_test_poly = X_test[[study_col]]

    poly_pipeline.fit(X_train_poly, y_train)
    preds = poly_pipeline.predict(X_test_poly)
    result = evaluate_model(f"Polynomial (degree={degree}, {study_col} only)", y_test, preds)
    return result, preds


def get_feature_combo_groups(columns: list[str]) -> dict[str, list[str]]:
    """Build sensible feature-drop groups from available dataset columns."""
    keyword_groups: dict[str, list[str]] = {
        "Sleep": ["sleep"],
        "Participation": ["participation"],
        "Wellbeing": ["sleep", "physical"],
        "Engagement": ["participation", "attendance", "extracurricular"],
        "Home support": ["parental", "family", "internet"],
        "Academic support": ["tutoring", "teacher", "school"],
    }

    normalized_to_original = {normalize_name(col): col for col in columns}
    groups: dict[str, list[str]] = {}
    for group_name, keywords in keyword_groups.items():
        matched = [
            original
            for normalized, original in normalized_to_original.items()
            if any(keyword in normalized for keyword in keywords)
        ]
        if matched:
            groups[group_name] = sorted(set(matched))
    return groups


def run_feature_combo_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[ExperimentResult], dict[str, np.ndarray]]:
    combo_results: list[ExperimentResult] = []
    combo_predictions: dict[str, np.ndarray] = {}

    # 1) Single-feature ablation for a focused set of important columns.
    key_tokens = [
        "sleep",
        "participation",
        "attendance",
        "internet",
        "tutoring",
        "parental",
        "physical",
        "extracurricular",
    ]
    candidate_single_drops = [
        col for col in X_train.columns if any(token in normalize_name(col) for token in key_tokens)
    ]

    for col in sorted(candidate_single_drops):
        X_train_drop = X_train.drop(columns=[col])
        X_test_drop = X_test.drop(columns=[col])
        name = f"Linear Regression (without {col})"
        result, preds = run_linear_experiment(name, X_train_drop, X_test_drop, y_train, y_test)
        combo_results.append(result)
        combo_predictions[name] = preds

    # 2) Grouped-feature ablations for broader combinations.
    groups = get_feature_combo_groups(X_train.columns.tolist())
    for group_name, drop_cols in groups.items():
        if len(drop_cols) >= len(X_train.columns):
            continue
        X_train_drop = X_train.drop(columns=drop_cols)
        X_test_drop = X_test.drop(columns=drop_cols)
        name = f"Linear Regression (without {group_name}: {', '.join(drop_cols)})"
        result, preds = run_linear_experiment(name, X_train_drop, X_test_drop, y_train, y_test)
        combo_results.append(result)
        combo_predictions[name] = preds

    return combo_results, combo_predictions


def create_output_dir(base: str = "outputs") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def plot_eda(df: pd.DataFrame, target_col: str, study_col: str | None, out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    df[target_col].hist(bins=25)
    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "target_distribution.png"), dpi=150)
    plt.close()

    if study_col and study_col in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[study_col], df[target_col], alpha=0.65)
        plt.title(f"{study_col} vs {target_col}")
        plt.xlabel(study_col)
        plt.ylabel(target_col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "study_vs_score_scatter.png"), dpi=150)
        plt.close()


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_clustering(df: pd.DataFrame, target_col: str | None, out_dir: str) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col and target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    if numeric_df.shape[1] < 2:
        print("Skipping clustering: not enough numeric predictor features.")
        return

    numeric_df = numeric_df.dropna()
    if len(numeric_df) < 10:
        print("Skipping clustering: too few rows after dropping missing values.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    k_values = list(range(2, 7))
    inertias = []
    silhouettes = []

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_idx = int(np.argmax(silhouettes))
    best_k = k_values[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Plot (KMeans)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clustering_elbow.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouettes, marker="o")
    plt.title("Silhouette Scores by k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clustering_silhouette.png"), dpi=150)
    plt.close()

    final_model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    final_labels = final_model.fit_predict(X_scaled)
    clustered = numeric_df.copy()
    clustered["cluster"] = final_labels

    print(f"Best k from silhouette analysis: {best_k}")
    print("Cluster sizes:")
    print(clustered["cluster"].value_counts().sort_index())

    if numeric_df.shape[1] >= 2:
        cols = numeric_df.columns[:2]
        plt.figure(figsize=(8, 5))
        plt.scatter(
            clustered[cols[0]],
            clustered[cols[1]],
            c=clustered["cluster"],
            cmap="tab10",
            alpha=0.75,
        )
        plt.title(f"Clusters Visualized on {cols[0]} and {cols[1]}")
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cluster_scatter.png"), dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Student Performance Regression + Clustering")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    args = parser.parse_args()

    df = load_data(args.csv)
    df = basic_cleaning(df)

    target_col = find_column(
        df.columns,
        [
            "examscore",
            "finalscore",
            "score",
            "finalgrade",
            "mathscore",
        ],
    )
    if target_col is None:
        raise ValueError(
            "Could not detect target column. Expected one of: exam_score/final_score/score/final_grade."
        )

    study_col = find_column(
        df.columns,
        [
            "hoursstudied",
            "studyhours",
            "studytimeweekly",
            "studytime",
        ],
    )

    out_dir = create_output_dir("outputs")
    plot_eda(df, target_col, study_col, out_dir)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results: list[ExperimentResult] = []
    predictions: dict[str, np.ndarray] = {}

    baseline_result, baseline_pred = run_linear_experiment(
        "Linear Regression (all features)", X_train, X_test, y_train, y_test
    )
    results.append(baseline_result)
    predictions[baseline_result.name] = baseline_pred

    combo_results, combo_predictions = run_feature_combo_experiments(
        X_train, X_test, y_train, y_test
    )
    results.extend(combo_results)
    predictions.update(combo_predictions)

    if study_col and study_col in X.columns:
        study_only_train = X_train[[study_col]]
        study_only_test = X_test[[study_col]]
        study_only_result, study_only_pred = run_linear_experiment(
            f"Linear Regression ({study_col} only)",
            study_only_train,
            study_only_test,
            y_train,
            y_test,
        )
        results.append(study_only_result)
        predictions[study_only_result.name] = study_only_pred

        for degree in [2, 3]:
            poly_result, poly_pred = run_polynomial_experiment(
                study_col, X_train, X_test, y_train, y_test, degree=degree
            )
            results.append(poly_result)
            predictions[poly_result.name] = poly_pred
    else:
        print("Study-hours column not detected. Skipping study-hours-only and polynomial experiments.")

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("r2", ascending=False)
    print("\nModel performance comparison:")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    results_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)

    best_name = results_df.iloc[0]["name"]
    best_pred = predictions[best_name]
    plot_predictions(
        y_test,
        best_pred,
        title=f"Actual vs Predicted ({best_name})",
        out_path=os.path.join(out_dir, "best_model_actual_vs_predicted.png"),
    )

    plt.figure(figsize=(10, 5))
    plt.bar(results_df["name"], results_df["r2"])
    plt.title("R2 by Model")
    plt.ylabel("R2")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_r2_comparison.png"), dpi=150)
    plt.close()

    run_clustering(df, target_col, out_dir)

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated files include EDA plots, model comparisons, prediction plot, and clustering visuals.")


if __name__ == "__main__":
    main()
