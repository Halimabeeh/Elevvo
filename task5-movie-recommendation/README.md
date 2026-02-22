# Task 5: Movie Recommendation System

This task builds a movie recommendation system using collaborative filtering.

Recommended dataset: **MovieLens 100K (Kaggle)**

## Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Covered Topics

- Recommendation systems
- Similarity-based modeling

## Objectives

- Recommend movies based on user similarity
- Use a user-item matrix to compute similarity scores
- Recommend top-rated unseen movies for a given user
- Evaluate recommendations using Precision@K
- Bonus:
- Item-based collaborative filtering
- Matrix factorization (SVD)

## Run

From this folder:

```bash
python3 movie_recommendation_system.py --ratings "u.data" --movies "u.item" --user-id 1 --k 10
```

If you only have ratings:

```bash
python3 movie_recommendation_system.py --ratings "u.data" --k 10
```

## What the script does

1. Loads ratings (`u.data` or CSV with user/movie/rating columns).
2. Optionally loads movie titles (`u.item` or CSV).
3. Creates a user-item matrix.
4. Builds recommenders:
- User-based collaborative filtering
- Item-based collaborative filtering (bonus)
- SVD matrix factorization (bonus)
5. Evaluates all recommenders using Precision@K on held-out interactions.
6. Generates top-K unseen-movie recommendations for a selected user.

## Outputs

All outputs are saved in `outputs/`:
- `precision_at_k_summary.csv`
- `precision_at_k_per_user.csv`
- `precision_at_k_comparison.png`
- `recommendations_user_<id>.csv`
