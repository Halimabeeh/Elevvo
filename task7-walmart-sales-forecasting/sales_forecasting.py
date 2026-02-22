#!/usr/bin/env python3
"""Task 7: Walmart sales forecasting (time series regression).

Usage:
    python sales_forecasting.py --csv /path/to/train.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RANDOM_STATE = 42


@dataclass
class ForecastResult:
    model: str
    mae: float
    rmse: float
    mape: float
    r2: float


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    norm_map = {normalize_name(c): c for c in columns}
    for c in candidates:
        if c in norm_map:
            return norm_map[c]
    return None


def load_data(csv_path: str) -> pd.DataFrame:
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


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    date_col = find_column(df.columns, ["date", "datetime", "timestamp", "ds"])
    sales_col = find_column(
        df.columns,
        ["weeklysales", "sales", "revenue", "target", "y"],
    )
    if date_col is None or sales_col is None:
        raise ValueError("Could not detect date/sales columns. Expected columns like Date and Weekly_Sales.")
    return date_col, sales_col


def aggregate_daily(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.DataFrame:
    out = df[[date_col, sales_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[sales_col] = pd.to_numeric(out[sales_col], errors="coerce")
    out = out.dropna().sort_values(date_col)

    # Aggregate all stores/departments to one total-sales time series per date.
    out = out.groupby(date_col, as_index=False)[sales_col].sum()
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def add_time_features(ts: pd.DataFrame, date_col: str, sales_col: str) -> pd.DataFrame:
    f = ts.copy()
    f["year"] = f[date_col].dt.year
    f["month"] = f[date_col].dt.month
    f["day"] = f[date_col].dt.day
    f["dayofweek"] = f[date_col].dt.dayofweek
    f["weekofyear"] = f[date_col].dt.isocalendar().week.astype(int)
    f["quarter"] = f[date_col].dt.quarter
    f["is_month_start"] = f[date_col].dt.is_month_start.astype(int)
    f["is_month_end"] = f[date_col].dt.is_month_end.astype(int)

    # Lag and rolling features (bonus asked).
    f["lag_1"] = f[sales_col].shift(1)
    f["lag_2"] = f[sales_col].shift(2)
    f["lag_4"] = f[sales_col].shift(4)
    f["lag_8"] = f[sales_col].shift(8)
    f["rolling_mean_4"] = f[sales_col].shift(1).rolling(window=4).mean()
    f["rolling_mean_12"] = f[sales_col].shift(1).rolling(window=12).mean()
    f["rolling_std_4"] = f[sales_col].shift(1).rolling(window=4).std()

    f = f.dropna().reset_index(drop=True)
    return f


def seasonal_decompose_simple(ts: pd.DataFrame, date_col: str, sales_col: str, out_dir: str) -> None:
    s = ts.set_index(date_col)[sales_col].astype(float)

    # Trend via centered rolling mean.
    trend = s.rolling(window=12, center=True, min_periods=6).mean()

    # Seasonal component from detrended month-of-year means.
    detrended = s - trend
    month_avg = detrended.groupby(detrended.index.month).mean()
    seasonal = pd.Series([month_avg.get(m, 0.0) for m in s.index.month], index=s.index)

    resid = s - trend - seasonal

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(s.index, s.values)
    axes[0].set_title("Observed Sales")

    axes[1].plot(trend.index, trend.values)
    axes[1].set_title("Trend (Rolling Mean)")

    axes[2].plot(seasonal.index, seasonal.values)
    axes[2].set_title("Seasonal (Month-wise)")

    axes[3].plot(resid.index, resid.values)
    axes[3].set_title("Residual")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "seasonal_decomposition.png"), dpi=150)
    plt.close(fig)


def split_time_data(
    feat_df: pd.DataFrame,
    date_col: str,
    sales_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    n = len(feat_df)
    split_idx = int(n * (1 - test_size))

    train = feat_df.iloc[:split_idx].copy()
    test = feat_df.iloc[split_idx:].copy()

    feature_cols = [c for c in feat_df.columns if c not in [date_col, sales_col]]

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train[sales_col]
    y_test = test[sales_col]

    return X_train, X_test, y_train, y_test, test[[date_col, sales_col]].copy()


def evaluate(y_true: pd.Series, y_pred: np.ndarray, name: str) -> ForecastResult:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    denom = np.where(np.abs(y_true.values) < 1e-9, 1e-9, np.abs(y_true.values))
    mape = float(np.mean(np.abs((y_true.values - y_pred) / denom)) * 100)

    return ForecastResult(
        model=name,
        mae=float(mae),
        rmse=rmse,
        mape=mape,
        r2=float(r2_score(y_true, y_pred)),
    )


def run_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[ForecastResult], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    results: list[ForecastResult] = []
    preds: dict[str, np.ndarray] = {}
    best_params: dict[str, dict[str, Any]] = {}

    tscv = TimeSeriesSplit(n_splits=4)

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    name = "LinearRegression"
    results.append(evaluate(y_test, lin_pred, name))
    preds[name] = lin_pred
    best_params[name] = {"note": "default parameters"}

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 8, 12, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.8],
        },
        n_iter=12,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)
    name = "RandomForest"
    results.append(evaluate(y_test, rf_pred, name))
    preds[name] = rf_pred
    best_params[name] = rf_search.best_params_

    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gbr_search = RandomizedSearchCV(
        gbr,
        param_distributions={
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 0.85, 1.0],
        },
        n_iter=10,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    gbr_search.fit(X_train, y_train)
    gbr_best = gbr_search.best_estimator_
    gbr_pred = gbr_best.predict(X_test)
    name = "GradientBoosting"
    results.append(evaluate(y_test, gbr_pred, name))
    preds[name] = gbr_pred
    best_params[name] = gbr_search.best_params_

    if HAS_XGBOOST:
        try:
            xgb = XGBRegressor(
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                n_jobs=-1,
                tree_method="hist",
            )
            xgb_search = RandomizedSearchCV(
                xgb,
                param_distributions={
                    "n_estimators": [200, 300, 500],
                    "max_depth": [3, 4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.7, 0.85, 1.0],
                    "reg_lambda": [1, 5, 10],
                },
                n_iter=12,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            )
            xgb_search.fit(X_train, y_train)
            xgb_best = xgb_search.best_estimator_
            xgb_pred = xgb_best.predict(X_test)
            name = "XGBoost"
            results.append(evaluate(y_test, xgb_pred, name))
            preds[name] = xgb_pred
            best_params[name] = xgb_search.best_params_
        except Exception as e:
            print(f"XGBoost skipped due to runtime error: {e}")
    else:
        print("XGBoost not installed; skipping XGBoost bonus model.")

    return results, preds, best_params


def plot_actual_vs_pred(
    test_meta: pd.DataFrame,
    preds: dict[str, np.ndarray],
    date_col: str,
    sales_col: str,
    out_dir: str,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(test_meta[date_col], test_meta[sales_col], label="Actual", linewidth=2)
    for name, yhat in preds.items():
        plt.plot(test_meta[date_col], yhat, label=name, alpha=0.9)

    plt.title("Actual vs Predicted Sales (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actual_vs_predicted_all_models.png"), dpi=150)
    plt.close()


def plot_rolling_average(ts: pd.DataFrame, date_col: str, sales_col: str, out_dir: str) -> None:
    d = ts.copy()
    d["rolling_4"] = d[sales_col].rolling(4).mean()
    d["rolling_12"] = d[sales_col].rolling(12).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(d[date_col], d[sales_col], label="Actual sales", alpha=0.5)
    plt.plot(d[date_col], d["rolling_4"], label="4-period rolling mean")
    plt.plot(d[date_col], d["rolling_12"], label="12-period rolling mean")
    plt.title("Sales with Rolling Averages")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_averages.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 7 - Sales Forecasting")
    parser.add_argument("--csv", required=True, help="Path to Walmart sales CSV")
    args = parser.parse_args()

    df = load_data(args.csv)
    date_col, sales_col = detect_columns(df)
    ts = aggregate_daily(df, date_col, sales_col)

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    plot_rolling_average(ts, date_col, sales_col, out_dir)
    seasonal_decompose_simple(ts, date_col, sales_col, out_dir)

    feat_df = add_time_features(ts, date_col, sales_col)

    X_train, X_test, y_train, y_test, test_meta = split_time_data(
        feat_df,
        date_col,
        sales_col,
        test_size=0.2,
    )

    print(f"Detected date column: {date_col}")
    print(f"Detected sales column: {sales_col}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    results, preds, best_params = run_models(X_train, X_test, y_train, y_test)

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(by="rmse")
    print("\nModel comparison (sorted by RMSE):")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    results_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)

    with open(os.path.join(out_dir, "best_params.txt"), "w", encoding="utf-8") as f:
        for model_name, params in best_params.items():
            f.write(f"{model_name}:\n")
            for k, v in params.items():
                f.write(f"  - {k}: {v}\n")
            f.write("\n")

    pred_df = test_meta.copy()
    pred_df = pred_df.rename(columns={sales_col: "actual_sales"})
    for name, yhat in preds.items():
        pred_df[f"pred_{name}"] = yhat
    pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    plot_actual_vs_pred(test_meta, preds, date_col, sales_col, out_dir)

    print(f"\nSaved outputs in: {os.path.abspath(out_dir)}")
    print("Generated: rolling averages, seasonal decomposition, model comparison, and forecast plots.")


if __name__ == "__main__":
    main()
