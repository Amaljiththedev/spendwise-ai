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







def explain_anomaly(transaction,expenses_df,config):
    

    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]
    date_col     = config["data"]["date_column"]
    

    month    = transaction["month"]
    expenses = expenses_df.copy()
    category = transaction[category_col]
    amount = transaction[amount_col]
    cat_mean = expenses.groupby(category_col)[amount_col].mean()[category]
    cat_std  = expenses.groupby(category_col)[amount_col].std()[category]

    catergory_freq = expenses[category_col].value_counts()

    #avg spending for this cat for this month

    cat_transactions = expenses[expenses[category_col] == category]
    avg_monthly_freq = cat_transactions.groupby("month").size().mean()

    this_month_count = len(
    expenses[
        (expenses[category_col] == category) &
        (expenses["month"] == month)
    ]
    )


    reasons = []

    if amount > cat_mean + 1.5 * cat_std:
        multiple = round(amount / cat_mean, 1)
        reasons.append(
            f"£{amount:.2f} is {multiple}x your usual "
            f"average of £{cat_mean:.2f} for {category}"
        )

    if this_month_count > avg_monthly_freq * 1.5:
        reasons.append(
            f"{this_month_count} {category} transactions "
            f"this month vs your usual {avg_monthly_freq:.0f}"
        )

    severity = "high" if len(reasons) > 1 else "medium" if reasons else "low"

    return {
        "date":     str(transaction[date_col]),
        "category": category,
        "amount":   amount,
        "reasons":  reasons,
        "severity": severity
    }






def explain_all_anomalies(anomalies_df, expenses_df, config):
    return [
        explain_anomaly(row, expenses_df, config)
        for _, row in anomalies_df.iterrows()
    ]



if __name__ == "__main__":
    config   = load_config(PROJECT_ROOT / "configs" / "settings.yaml")
    data     = load_transactions(PROJECT_ROOT / "data" / "raw" / "finance.csv", config)
    data, expenses, _ = clean_transactions(data, config)

    # Step 1 — run isolation forest on expenses
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    expenses = expenses.copy()
    expenses["category_encoded"] = le.fit_transform(expenses[config["data"]["category_column"]])

    features = expenses[["Amount", "month_number", "category_encoded"]]
    clf = IsolationForest(contamination=0.05, random_state=42)
    expenses["anomaly_label"] = clf.fit_predict(features)
    expenses["anomaly_score"]  = clf.decision_function(features)

    # Step 2 — filter only flagged anomalies
    anomalies_df = expenses[expenses["anomaly_label"] == -1]
    print(f"Anomalies found: {len(anomalies_df)}")

    # Step 3 — explain each one
    results = explain_all_anomalies(anomalies_df, expenses, config)

    # Step 4 — print first 5
    for r in results[:5]:
        print(r)