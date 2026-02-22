#!/usr/bin/env python3
"""Task 2: Customer clustering (unsupervised learning).

Usage:
    python customer_clustering_analysis.py --csv /path/to/Mall_Customers.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass
class KMeansResult:
    best_k: int
    labels: np.ndarray
    inertias: list[float]
    silhouettes: list[float]
    tested_k: list[int]


@dataclass
class DBSCANResult:
    eps: float
    min_samples: int
    labels: np.ndarray
    silhouette: float | None


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty.")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    print(f"Rows before cleaning: {before}")
    print(f"Duplicates removed: {removed}")
    print(f"Rows after cleaning: {len(df)}")

    return df


def detect_feature_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    income_col = find_column(
        df.columns,
        [
            "annualincomek",
            "annualincome",
            "income",
            "yearlyincome",
        ],
    )
    spending_col = find_column(
        df.columns,
        [
            "spendingscore1100",
            "spendingscore",
            "score",
        ],
    )
    age_col = find_column(
        df.columns,
        [
            "age",
        ],
    )

    if income_col is None or spending_col is None:
        raise ValueError(
            "Could not detect required columns for clustering. "
            "Expected income and spending score columns."
        )

    return income_col, spending_col, age_col


def create_output_dir(base: str = "outputs") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def plot_data_overview(df: pd.DataFrame, income_col: str, spending_col: str, out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[income_col], df[spending_col], alpha=0.75)
    plt.title("Customers: Income vs Spending Score")
    plt.xlabel(income_col)
    plt.ylabel(spending_col)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "income_vs_spending_raw.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df[income_col].hist(bins=20)
    plt.title(f"Distribution of {income_col}")
    plt.xlabel(income_col)

    plt.subplot(1, 2, 2)
    df[spending_col].hist(bins=20)
    plt.title(f"Distribution of {spending_col}")
    plt.xlabel(spending_col)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_distributions.png"), dpi=150)
    plt.close()


def prepare_features(df: pd.DataFrame, income_col: str, spending_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    features = df[[income_col, spending_col]].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    missing_before = int(features.isna().sum().sum())
    if missing_before > 0:
        print(f"Missing values in clustering features before imputation: {missing_before}")

    features = features.fillna(features.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return features, X_scaled


def run_kmeans_optimal(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10) -> KMeansResult:
    tested_k = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in tested_k:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))

    best_idx = int(np.argmax(silhouettes))
    best_k = tested_k[best_idx]

    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    final_labels = final_model.fit_predict(X_scaled)

    return KMeansResult(
        best_k=best_k,
        labels=final_labels,
        inertias=inertias,
        silhouettes=silhouettes,
        tested_k=tested_k,
    )


def run_dbscan_search(X_scaled: np.ndarray) -> DBSCANResult:
    eps_candidates = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    min_samples_candidates = [4, 5, 6, 8]

    best: DBSCANResult | None = None

    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

            core_mask = labels != -1
            unique_clusters = set(labels[core_mask])
            if len(unique_clusters) < 2:
                continue

            score = float(silhouette_score(X_scaled[core_mask], labels[core_mask]))
            candidate = DBSCANResult(
                eps=eps,
                min_samples=min_samples,
                labels=labels,
                silhouette=score,
            )

            if best is None or (candidate.silhouette or -1.0) > (best.silhouette or -1.0):
                best = candidate

    if best is not None:
        return best

    fallback_model = DBSCAN(eps=0.5, min_samples=5)
    fallback_labels = fallback_model.fit_predict(X_scaled)
    return DBSCANResult(eps=0.5, min_samples=5, labels=fallback_labels, silhouette=None)


def plot_kmeans_diagnostics(result: KMeansResult, out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(result.tested_k, result.inertias, marker="o")
    plt.title("K-Means Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kmeans_elbow.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(result.tested_k, result.silhouettes, marker="o")
    plt.title("K-Means Silhouette Scores")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kmeans_silhouette.png"), dpi=150)
    plt.close()


def plot_clusters(
    features: pd.DataFrame,
    labels: np.ndarray,
    income_col: str,
    spending_col: str,
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(
        features[income_col],
        features[spending_col],
        c=labels,
        cmap="tab10",
        alpha=0.8,
        s=45,
    )
    plt.title(title)
    plt.xlabel(income_col)
    plt.ylabel(spending_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def average_spending_by_cluster(
    features: pd.DataFrame,
    labels: np.ndarray,
    spending_col: str,
    label_name: str,
) -> pd.DataFrame:
    cluster_df = features.copy()
    cluster_df[label_name] = labels

    avg = (
        cluster_df.groupby(label_name, dropna=False)[spending_col]
        .mean()
        .reset_index()
        .rename(columns={spending_col: "avg_spending_score"})
        .sort_values(by="avg_spending_score", ascending=False)
    )
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 - Customer Clustering")
    parser.add_argument("--csv", required=True, help="Path to Mall Customer CSV file")
    args = parser.parse_args()

    df = load_and_clean_data(args.csv)
    income_col, spending_col, age_col = detect_feature_columns(df)

    out_dir = create_output_dir("outputs")
    plot_data_overview(df, income_col, spending_col, out_dir)

    features, X_scaled = prepare_features(df, income_col, spending_col)

    kmeans_result = run_kmeans_optimal(X_scaled)
    plot_kmeans_diagnostics(kmeans_result, out_dir)
    plot_clusters(
        features,
        kmeans_result.labels,
        income_col,
        spending_col,
        title=f"K-Means Clusters (k={kmeans_result.best_k})",
        out_path=os.path.join(out_dir, "kmeans_clusters.png"),
    )

    features_out = df.copy()
    features_out["kmeans_cluster"] = kmeans_result.labels

    kmeans_spending = average_spending_by_cluster(
        features, kmeans_result.labels, spending_col, "kmeans_cluster"
    )

    print("\nK-Means results")
    print(f"Best k (by silhouette): {kmeans_result.best_k}")
    print("Average spending score by K-Means cluster:")
    print(kmeans_spending.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    dbscan_result = run_dbscan_search(X_scaled)
    features_out["dbscan_cluster"] = dbscan_result.labels

    plot_clusters(
        features,
        dbscan_result.labels,
        income_col,
        spending_col,
        title=f"DBSCAN Clusters (eps={dbscan_result.eps}, min_samples={dbscan_result.min_samples})",
        out_path=os.path.join(out_dir, "dbscan_clusters.png"),
    )

    dbscan_spending = average_spending_by_cluster(
        features, dbscan_result.labels, spending_col, "dbscan_cluster"
    )

    print("\nDBSCAN results")
    print(f"Best params searched: eps={dbscan_result.eps}, min_samples={dbscan_result.min_samples}")
    if dbscan_result.silhouette is not None:
        print(f"DBSCAN silhouette (excluding noise): {dbscan_result.silhouette:.4f}")
    else:
        print("DBSCAN silhouette: not available (insufficient non-noise clusters).")
    print("Average spending score by DBSCAN cluster (-1 means noise):")
    print(dbscan_spending.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    kmeans_spending.to_csv(os.path.join(out_dir, "kmeans_avg_spending_by_cluster.csv"), index=False)
    dbscan_spending.to_csv(os.path.join(out_dir, "dbscan_avg_spending_by_cluster.csv"), index=False)
    features_out.to_csv(os.path.join(out_dir, "customer_clusters_with_labels.csv"), index=False)

    if age_col is not None:
        print(f"Detected age column: {age_col}")

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated plots include raw exploration, K-Means diagnostics, and K-Means/DBSCAN cluster visuals.")


if __name__ == "__main__":
    main()
