import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_config, load_transactions
from src.analytics.ml.anomaly_labeller import apply_anomaly_labelling
from src.processing.cleaner import clean_transactions


def run_local_outlier_factor(show_plot: bool = False, contamination: float = 0.03):
    config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")
    data = load_transactions(PROJECT_ROOT / "data" / "raw" / "finance.csv", config)
    

    data,expenses, _ = clean_transactions(data, config)

    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    expenses = expenses.copy()
    expenses[amount_col] = pd.to_numeric(expenses[amount_col], errors="coerce")
    expenses = expenses.dropna(subset=["Date", amount_col, category_col])
    expenses["Category_encoded"] = expenses[category_col].astype("category").cat.codes
    expenses["month_number"] = expenses["Date"].dt.month

    features = [amount_col, "Category_encoded", "month_number"]
    
    X = expenses[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    local_outlier_factor = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    local_outlier_factor.fit(X_scaled)
    
    scores = local_outlier_factor.negative_outlier_factor_
    predictions = local_outlier_factor.fit_predict(X_scaled)

    data["lof_score"] = np.nan
    data.loc[expenses.index, "lof_score"] = scores
    data["lof_prediction"] = np.nan
    data.loc[expenses.index, "lof_prediction"] = predictions
    
    data = apply_anomaly_labelling(data, thresholds={}, pred_col="lof_prediction")
    
    plot_data = data.dropna(subset=["lof_prediction"]).copy()
    plot_data[amount_col] = pd.to_numeric(plot_data[amount_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[amount_col, "Date"])

    plt.figure(figsize=(14, 6))
    sns.scatterplot(
        data=plot_data,
        x="Date",
        y=amount_col,
        hue="lof_prediction",
        palette={1.0: "steelblue", -1.0: "red"},
        alpha=0.7,
    )
    plt.xticks(rotation=45)
    plt.title("Transaction Amounts with Anomalies Highlighted")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "anomaly_detection_lof.png", dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    anomalies = plot_data[plot_data["lof_prediction"] == -1]
    print(f"Total anomalies detected: {len(anomalies)}")
    print(anomalies[["Date", category_col, amount_col]].sort_values(amount_col, ascending=False))
    print("\nAnomaly labels:")
    print(data["anomaly_label"].value_counts())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection on expense transactions.")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open plot window after saving anomaly_detection_lof.png",
    )
    args = parser.parse_args()
    run_local_outlier_factor(show_plot=args.show_plot) 









