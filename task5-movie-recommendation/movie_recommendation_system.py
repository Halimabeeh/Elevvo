#!/usr/bin/env python3
"""Task 5: Movie recommendation system (collaborative filtering + SVD).

Supports MovieLens 100K style files:
- ratings: u.data (tab-separated, no header) or CSV with user/movie/rating columns
- movies: optional u.item or CSV (for movie titles)

Usage:
    python movie_recommendation_system.py --ratings /path/to/u.data --movies /path/to/u.item --user-id 1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


RANDOM_STATE = 42


@dataclass
class EvalResult:
    model: str
    k: int
    precision_at_k: float
    evaluated_users: int


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_ratings(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file not found: {path}")

    # Try CSV first; if schema is not usable, fall back to MovieLens u.data format.
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()

    if df.empty or df.shape[1] < 3:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )

    user_col = find_column(df.columns, ["userid", "userid", "user", "uid", "user_id"])
    movie_col = find_column(df.columns, ["movieid", "itemid", "movie", "iid", "movie_id"])
    rating_col = find_column(df.columns, ["rating", "score", "stars"])

    if user_col is None or movie_col is None or rating_col is None:
        # Final fallback for no-header CSV with 3+ columns.
        if df.shape[1] >= 3:
            cols = list(df.columns)
            user_col, movie_col, rating_col = cols[0], cols[1], cols[2]
        else:
            raise ValueError("Could not detect user/movie/rating columns in ratings file.")

    out = df[[user_col, movie_col, rating_col]].copy()
    out.columns = ["user_id", "movie_id", "rating"]

    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce")
    out["movie_id"] = pd.to_numeric(out["movie_id"], errors="coerce")
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out.dropna().astype({"user_id": int, "movie_id": int})

    if out.empty:
        raise ValueError("Ratings data is empty after cleaning.")

    return out


def load_movies(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Movies file not found: {path}")

    # Try MovieLens u.item pipe-delimited format first.
    try:
        movies = pd.read_csv(
            path,
            sep="|",
            header=None,
            encoding="latin-1",
            usecols=[0, 1],
            names=["movie_id", "title"],
            engine="python",
        )
        if movies["movie_id"].notna().any():
            movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
            movies = movies.dropna().astype({"movie_id": int})
            return movies
    except Exception:
        pass

    # Fallback generic CSV.
    movies = pd.read_csv(path)
    id_col = find_column(movies.columns, ["movieid", "movie_id", "itemid", "id"])
    title_col = find_column(movies.columns, ["title", "movietitle", "name"])
    if id_col is None or title_col is None:
        return None

    out = movies[[id_col, title_col]].copy()
    out.columns = ["movie_id", "title"]
    out["movie_id"] = pd.to_numeric(out["movie_id"], errors="coerce")
    out = out.dropna().astype({"movie_id": int})
    return out


def create_output_dir(base: str = "outputs") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def train_test_split_per_user(
    ratings: pd.DataFrame,
    min_user_ratings: int = 5,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []

    for _, user_df in ratings.groupby("user_id"):
        if len(user_df) < min_user_ratings:
            train_parts.append(user_df)
            continue

        test_idx = rng.choice(user_df.index, size=1, replace=False)[0]
        test_rows.append(user_df.loc[[test_idx]])
        train_parts.append(user_df.drop(index=test_idx))

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=ratings.columns)
    return train_df, test_df


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)


def recommend_user_based(
    user_id: int,
    user_item: pd.DataFrame,
    top_k: int = 10,
    neighbors: int = 25,
) -> list[int]:
    if user_id not in user_item.index:
        return []

    sim_matrix = cosine_similarity(user_item.values)
    sim_df = pd.DataFrame(sim_matrix, index=user_item.index, columns=user_item.index)

    user_ratings = user_item.loc[user_id]
    seen = set(user_ratings[user_ratings > 0].index.tolist())

    similar_users = sim_df.loc[user_id].drop(index=user_id).sort_values(ascending=False).head(neighbors).index
    weighted_scores: dict[int, float] = {}
    weight_totals: dict[int, float] = {}

    for nbr in similar_users:
        sim = float(sim_df.loc[user_id, nbr])
        nbr_ratings = user_item.loc[nbr]
        for movie_id, rating in nbr_ratings.items():
            if movie_id in seen or rating <= 0:
                continue
            weighted_scores[movie_id] = weighted_scores.get(movie_id, 0.0) + sim * float(rating)
            weight_totals[movie_id] = weight_totals.get(movie_id, 0.0) + abs(sim)

    scored = []
    for movie_id, score in weighted_scores.items():
        denom = weight_totals.get(movie_id, 1e-9)
        scored.append((movie_id, score / denom))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in scored[:top_k]]


def recommend_item_based(
    user_id: int,
    user_item: pd.DataFrame,
    top_k: int = 10,
) -> list[int]:
    if user_id not in user_item.index:
        return []

    item_user = user_item.T
    item_sim = cosine_similarity(item_user.values)
    item_sim_df = pd.DataFrame(item_sim, index=item_user.index, columns=item_user.index)

    user_ratings = user_item.loc[user_id]
    seen = set(user_ratings[user_ratings > 0].index.tolist())

    scores: dict[int, float] = {}
    for seen_movie in seen:
        rating = float(user_ratings[seen_movie])
        sims = item_sim_df.loc[seen_movie].sort_values(ascending=False)
        for movie_id, sim in sims.items():
            if movie_id in seen or sim <= 0:
                continue
            scores[movie_id] = scores.get(movie_id, 0.0) + sim * rating

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in ranked[:top_k]]


def recommend_svd(
    user_id: int,
    user_item: pd.DataFrame,
    top_k: int = 10,
    n_components: int = 20,
) -> list[int]:
    if user_id not in user_item.index:
        return []

    mat = user_item.values
    max_components = max(2, min(n_components, min(mat.shape) - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(mat)
    reconstructed = np.dot(reduced, svd.components_)

    user_idx = user_item.index.get_loc(user_id)
    preds = reconstructed[user_idx]
    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index.tolist())

    movie_scores = []
    for i, movie_id in enumerate(user_item.columns):
        if movie_id in seen:
            continue
        movie_scores.append((int(movie_id), float(preds[i])))

    movie_scores.sort(key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in movie_scores[:top_k]]


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    if not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for movie in top_k if movie in relevant)
    return hits / k


def evaluate_precision_at_k(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_item: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    rows = []
    for user_id, grp in test_df.groupby("user_id"):
        relevant = set(grp[grp["rating"] >= 4]["movie_id"].tolist())
        if not relevant:
            continue

        rec_user = recommend_user_based(int(user_id), user_item, top_k=k)
        rec_item = recommend_item_based(int(user_id), user_item, top_k=k)
        rec_svd = recommend_svd(int(user_id), user_item, top_k=k)

        rows.append(
            {
                "user_id": int(user_id),
                "user_based_precision_at_k": precision_at_k(rec_user, relevant, k),
                "item_based_precision_at_k": precision_at_k(rec_item, relevant, k),
                "svd_precision_at_k": precision_at_k(rec_svd, relevant, k),
            }
        )

    return pd.DataFrame(rows)


def enrich_titles(movie_ids: list[int], movies_df: pd.DataFrame | None) -> list[str]:
    if movies_df is None or not movie_ids:
        return [str(mid) for mid in movie_ids]

    title_map = dict(zip(movies_df["movie_id"], movies_df["title"]))
    return [title_map.get(mid, str(mid)) for mid in movie_ids]


def pad_list(values: list[str], size: int, fill: str = "") -> list[str]:
    if len(values) >= size:
        return values[:size]
    return values + [fill] * (size - len(values))


def save_summary_plot(results: list[EvalResult], out_dir: str) -> None:
    labels = [r.model for r in results]
    vals = [r.precision_at_k for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.title("Precision@K Comparison")
    plt.ylabel("Precision@K")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_at_k_comparison.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5 - Movie Recommendation System")
    parser.add_argument("--ratings", required=True, help="Path to ratings file (u.data or CSV)")
    parser.add_argument("--movies", help="Optional path to movie metadata (u.item or CSV)")
    parser.add_argument("--user-id", type=int, help="User ID for example recommendations")
    parser.add_argument("--k", type=int, default=10, help="Top-K recommendation cutoff")
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    movies = load_movies(args.movies)

    print(f"Rows in ratings data: {len(ratings)}")
    print(f"Unique users: {ratings['user_id'].nunique()}")
    print(f"Unique movies: {ratings['movie_id'].nunique()}")

    out_dir = create_output_dir("outputs")

    train_df, test_df = train_test_split_per_user(ratings)
    print(f"Train interactions: {len(train_df)}")
    print(f"Held-out test interactions: {len(test_df)}")

    user_item = build_user_item_matrix(train_df)

    eval_df = evaluate_precision_at_k(train_df, test_df, user_item, k=args.k)
    if eval_df.empty:
        raise ValueError("Evaluation produced no rows. Check that test users have relevant (rating>=4) held-out items.")

    summary = [
        EvalResult(
            model="User-based CF",
            k=args.k,
            precision_at_k=float(eval_df["user_based_precision_at_k"].mean()),
            evaluated_users=int(len(eval_df)),
        ),
        EvalResult(
            model="Item-based CF",
            k=args.k,
            precision_at_k=float(eval_df["item_based_precision_at_k"].mean()),
            evaluated_users=int(len(eval_df)),
        ),
        EvalResult(
            model="SVD",
            k=args.k,
            precision_at_k=float(eval_df["svd_precision_at_k"].mean()),
            evaluated_users=int(len(eval_df)),
        ),
    ]

    summary_df = pd.DataFrame([r.__dict__ for r in summary]).sort_values("precision_at_k", ascending=False)
    summary_df.to_csv(os.path.join(out_dir, "precision_at_k_summary.csv"), index=False)
    eval_df.to_csv(os.path.join(out_dir, "precision_at_k_per_user.csv"), index=False)
    save_summary_plot(summary, out_dir)

    print("\nPrecision@K results:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    selected_user = args.user_id
    if selected_user is None:
        selected_user = int(user_item.index[0])

    rec_user = recommend_user_based(selected_user, user_item, top_k=args.k)
    rec_item = recommend_item_based(selected_user, user_item, top_k=args.k)
    rec_svd = recommend_svd(selected_user, user_item, top_k=args.k)

    user_titles = pad_list(enrich_titles(rec_user[: args.k], movies), args.k)
    item_titles = pad_list(enrich_titles(rec_item[: args.k], movies), args.k)
    svd_titles = pad_list(enrich_titles(rec_svd[: args.k], movies), args.k)

    rec_df = pd.DataFrame(
        {
            "rank": list(range(1, args.k + 1)),
            "user_based": user_titles,
            "item_based": item_titles,
            "svd": svd_titles,
        }
    )
    rec_df.to_csv(os.path.join(out_dir, f"recommendations_user_{selected_user}.csv"), index=False)

    print(f"\nTop-{args.k} recommendations for user {selected_user} (saved to CSV):")
    print(rec_df.to_string(index=False))

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated: precision@K summaries, per-user scores, recommendation lists, and comparison plot.")


if __name__ == "__main__":
    main()
