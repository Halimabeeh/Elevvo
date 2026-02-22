# Task 2: Customer Segmentation with Clustering

This task applies **unsupervised learning** to segment mall customers based on:
- Annual income
- Spending score

Dataset recommendation: **Mall Customer (Kaggle)**

## Tools & Libraries

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Objective

- Cluster customers into segments based on income and spending behavior
- Scale features before clustering
- Apply K-Means and determine the optimal number of clusters
- Visualize clusters in 2D
- Bonus:
  - Try DBSCAN clustering
  - Analyze average spending per cluster

## Run

From this folder:

```bash
python3 customer_clustering_analysis.py --csv "Mall_Customers.csv"
```

Or with absolute path:

```bash
python3 "/Users/Halima/Documents/New project/task2-customer-clustering/customer_clustering_analysis.py" --csv "/absolute/path/to/Mall_Customers.csv"
```

## What the script does

1. Loads and cleans data (duplicate removal).
2. Detects income/spending columns automatically.
3. Performs feature scaling (`StandardScaler`).
4. Runs K-Means for `k=2..10`.
5. Selects best `k` using silhouette score.
6. Saves elbow and silhouette plots.
7. Visualizes final K-Means clusters.
8. Runs DBSCAN with parameter search (bonus).
9. Computes and saves average spending per cluster for K-Means and DBSCAN (bonus).

## Outputs

All outputs are saved in `outputs/`:
- `income_vs_spending_raw.png`
- `feature_distributions.png`
- `kmeans_elbow.png`
- `kmeans_silhouette.png`
- `kmeans_clusters.png`
- `dbscan_clusters.png`
- `kmeans_avg_spending_by_cluster.csv`
- `dbscan_avg_spending_by_cluster.csv`
- `customer_clusters_with_labels.csv`
