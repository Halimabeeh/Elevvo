# Student Performance Analysis (Regression + Clustering)

This project uses **Pandas**, **Matplotlib**, and **Scikit-learn** to:
- Clean and inspect student-performance data
- Train/test regression models for exam score prediction
- Compare linear vs polynomial regression
- Experiment with feature combinations
- Run KMeans clustering (unsupervised learning)

## Requirements

```bash
pip install pandas matplotlib scikit-learn numpy
```

## Run

```bash
python student_performance_analysis.py --csv /absolute/path/to/student_performance.csv
```

Recommended dataset: **Student Performance Factors (Kaggle)**.

## What it does

1. Loads and cleans data (duplicate removal + missing value handling in model pipelines).
2. Creates EDA visuals:
- Target distribution
- Study-hours vs score scatter (if study-hours column exists)
3. Splits data into train/test sets.
4. Trains and compares:
- Linear Regression (all features)
- Multiple feature-combination experiments:
  - Single-feature drops (for columns like sleep, participation, attendance, internet, tutoring, parental, etc. if present)
  - Grouped drops (wellbeing, engagement, home support, academic support)
- Linear Regression using study-hours only (if present)
- Polynomial Regression degree 2 and 3 using study-hours only (if present)
5. Evaluates using:
- R²
- MAE
- RMSE
6. Runs KMeans clustering on numeric predictors and saves elbow/silhouette plots.

## Outputs

All files are saved to `outputs/`:
- `target_distribution.png`
- `study_vs_score_scatter.png` (if available)
- `model_comparison.csv`
- `model_r2_comparison.png`
- `best_model_actual_vs_predicted.png`
- `clustering_elbow.png`
- `clustering_silhouette.png`
- `cluster_scatter.png` (if enough numeric features)
