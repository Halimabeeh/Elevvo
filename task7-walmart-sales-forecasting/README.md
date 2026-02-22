# Task 7: Walmart Sales Forecasting

This task forecasts future sales from historical Walmart sales data using time-series regression.

Recommended dataset: **Walmart Sales Forecast (Kaggle)**

## Tools & Libraries

- Python
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost (optional bonus)

## Covered Topics

- Time series forecasting
- Regression

## Objectives

- Predict future sales based on historical sales data
- Create time-based features (day, month, lag values)
- Apply regression models to forecast next-period sales
- Plot actual vs predicted values over time
- Bonus:
- Use rolling averages and seasonal decomposition
- Apply XGBoost with time-aware validation

## Run

From this folder:

```bash
python3 sales_forecasting.py --csv "train.csv"
```

Or with absolute path:

```bash
python3 "/Users/Halima/Documents/New project/task7-walmart-sales-forecasting/sales_forecasting.py" --csv "/absolute/path/to/train.csv"
```

## What the script does

1. Loads and cleans data.
2. Detects date and sales columns automatically.
3. Aggregates total sales per date.
4. Builds time-based features:
- Calendar features (day, month, week, quarter, etc.)
- Lag features (`lag_1`, `lag_2`, `lag_4`, `lag_8`)
- Rolling stats (`rolling_mean_4`, `rolling_mean_12`, `rolling_std_4`)
5. Uses chronological train/test split (no random shuffling).
6. Trains/evaluates regression models:
- Linear Regression
- Random Forest (time-aware CV tuning)
- Gradient Boosting (time-aware CV tuning)
- XGBoost optional (time-aware CV tuning)
7. Saves actual-vs-predicted plot and model metrics.
8. Bonus visuals:
- Rolling averages plot
- Seasonal decomposition plot

## Outputs

All outputs are saved in `outputs/`:
- `model_comparison.csv`
- `best_params.txt`
- `test_predictions.csv`
- `actual_vs_predicted_all_models.png`
- `rolling_averages.png`
- `seasonal_decomposition.png`
