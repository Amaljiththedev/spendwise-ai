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

# Add the project root to python path to allow importing from src
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_config, load_transactions
from src.analytics.ml.anomaly_labeller import apply_anomaly_labelling
from src.processing.cleaner import clean_transactions


def run_anomaly_detection(show_plot: bool = False) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")
    data = load_transactions(PROJECT_ROOT / "data" / "raw" / "finance.csv", config)
    data, expenses, _ = clean_transactions(data, config)

    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    expenses = expenses.copy()
    expenses[amount_col] = pd.to_numeric(expenses[amount_col], errors="coerce")
    expenses = expenses.dropna(subset=["Date", amount_col, category_col])
    expenses["Category_encoded"] = expenses[category_col].astype("category").cat.codes
    expenses["month_number"] = expenses["Date"].dt.month

    features = [amount_col, "Category_encoded", "month_number"]
    X = expenses[features]

    isolation_forest = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=0.03,
        random_state=42,
    )
    isolation_forest.fit(X)

    scores = isolation_forest.decision_function(X)
    predictions = isolation_forest.predict(X)

    data["anomaly_score"] = np.nan
    data["anomaly_prediction"] = np.nan
    data.loc[expenses.index, "anomaly_score"] = scores
    data.loc[expenses.index, "anomaly_prediction"] = predictions
    data = apply_anomaly_labelling(data, thresholds={})

    plot_data = data.dropna(subset=["anomaly_prediction"]).copy()
    plot_data[amount_col] = pd.to_numeric(plot_data[amount_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[amount_col, "Date"])

    plt.figure(figsize=(14, 6))
    sns.scatterplot(
        data=plot_data,
        x="Date",
        y=amount_col,
        hue="anomaly_prediction",
        palette={1.0: "steelblue", -1.0: "red"},
        alpha=0.7,
    )
    plt.xticks(rotation=45)
    plt.title("Transaction Amounts with Anomalies Highlighted")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "anomaly_detection.png", dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    anomalies = plot_data[plot_data["anomaly_prediction"] == -1]
    print(f"Total anomalies detected: {len(anomalies)}")
    print(anomalies[["Date", category_col, amount_col]].sort_values(amount_col, ascending=False))
    print("\nAnomaly labels:")
    print(data["anomaly_label"].value_counts())


def anomaly_categories(data, config):

    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection on expense transactions.")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open plot window after saving anomaly_detection.png",
    )
    args = parser.parse_args()
    run_anomaly_detection(show_plot=args.show_plot)


