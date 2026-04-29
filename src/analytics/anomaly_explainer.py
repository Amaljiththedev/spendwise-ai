"""
Explain why individual transactions were flagged as anomalies.

Produces human-readable reasons (amount deviation, frequency spike)
and a severity rating for each flagged transaction.
"""

import argparse
import pathlib
import sys

import pandas as pd
from typing import List, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_config, load_transactions
from src.analytics.ml.anomaly_labeller import apply_anomaly_labelling
from src.analytics.ml.feature_engineering import build_features
from src.processing.cleaner import clean_transactions
from src.analytics.ml.insight_generator import generate_insight


def explain_anomaly(transaction, expenses_df, config):
    """
    Generate human-readable explanation for a single anomalous transaction.

    Parameters
    ----------
    transaction : pd.Series
        A single row from the anomalies DataFrame.
    expenses_df : pd.DataFrame
        Full expenses DataFrame for computing category statistics.
    config : dict
        Project config.

    Returns
    -------
    dict with keys: date, category, amount, reasons, severity
    """
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]
    date_col = config["data"]["date_column"]

    month = transaction["month_number"]
    expenses = expenses_df.copy()
    category = transaction[category_col]
    amount = transaction[amount_col]
    cat_mean = expenses.groupby(category_col)[amount_col].mean()[category]
    cat_std = expenses.groupby(category_col)[amount_col].std()[category]

    catergory_freq = expenses[category_col].value_counts()

    # avg spending for this category for this month
    cat_transactions = expenses[expenses[category_col] == category]
    avg_monthly_freq = cat_transactions.groupby("month_number").size().mean()

    this_month_count = len(
        expenses[
            (expenses[category_col] == category) &
            (expenses["month_number"] == month)
        ]
    )

    reasons = []

    if amount > cat_mean + 2 * cat_std:
        multiplier = round(amount / cat_mean, 1)
        reasons.append(
            f"£{amount:.2f} is unusually high — "
            f"{multiplier}x above your usual average "
            f"of £{cat_mean:.2f} for {category}"
        )

    if amount < cat_mean - 2 * cat_std:
        multiplier = round(cat_mean / amount, 1)
        reasons.append(
            f"£{amount:.2f} is unusually low — "
            f"{multiplier}x below your usual average "
            f"of £{cat_mean:.2f} for {category}"
        )

    if this_month_count > avg_monthly_freq * 1.5:
        reasons.append(
            f"{this_month_count} {category} transactions "
            f"this month vs your usual {avg_monthly_freq:.0f}"
        )

    severity = "high" if len(reasons) > 1 else "medium" if reasons else "low"

    return {
        "date": str(transaction[date_col]),
        "category": category,
        "amount": amount,
        "reasons": reasons,
        "severity": severity,
    }


def explain_all_anomalies(anomalies_df, expenses_df, config):
    """Explain every anomaly in the DataFrame."""
    return [
        explain_anomaly(row, expenses_df, config)
        for _, row in anomalies_df.iterrows()
    ]


if __name__ == "__main__":
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")
    data = load_transactions(PROJECT_ROOT / "data" / "raw" / "finance.csv", config)
    data, expenses, _ = clean_transactions(data, config)

    amount_col = config["data"]["amount_column"]

    # Step 1 — build features using shared module
    expenses = expenses.copy()
    expenses[amount_col] = pd.to_numeric(expenses[amount_col], errors="coerce")
    expenses = expenses.dropna(subset=["Date", amount_col, config["data"]["category_column"]])

    feat_df, feat_names = build_features(expenses, config)
    X_scaled = StandardScaler().fit_transform(feat_df)

    # Step 2 — run Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=42)
    expenses["anomaly_label"] = clf.fit_predict(X_scaled)
    expenses["anomaly_score"] = clf.decision_function(X_scaled)

    # Step 3 — filter only flagged anomalies
    anomalies_df = expenses[expenses["anomaly_label"] == -1]
    print(f"Anomalies found: {len(anomalies_df)}")

    # Step 4 — explain each one
    results = explain_all_anomalies(anomalies_df, expenses, config)

    # Step 5 — print first 10
    for r in results[:10]:
        print(f"Anomaly: {r}")
        try:
            insight = generate_insight(r)
            print(f"💡 Insight: {insight}\n")
        except Exception as e:
            print(f"⚠️ Failed to generate insight via AI: {e}\n")