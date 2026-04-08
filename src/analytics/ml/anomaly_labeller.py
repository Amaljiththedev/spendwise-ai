import pandas as pd      
import numpy as np          
from typing import Tuple



def get_category_thresholds(df: pd.DataFrame, category: str) -> Tuple[float, float]:
    cat_data = df[df["Category"] == category]["Amount"]

    mean = cat_data.mean()
    std = cat_data.std()

    lower_threshold = mean - 1.5 * std
    upper_threshold = mean + 1.5 * std

    return lower_threshold, upper_threshold


def label_anomaly(row: pd.Series, thresholds: dict) -> str:
    category = row.get("Category", "")
    amount = row.get("Amount", 0)

    low_thresh, high_thresh = thresholds.get(category, (0, float("inf")))

    if amount >= high_thresh:
        return "high_spending"
    elif amount < low_thresh:
        return "low_spending"
    else:
        return "high_spending"




def apply_anomaly_labelling(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:

    df = df.copy()
    
    df["anomaly_label"] = "normal"


    flagged = df[df["anomaly_prediction"] == -1]

    if flagged.empty:
        return df




    categories = flagged["Category"].dropna().unique()

    thresholds = {category: get_category_thresholds(df, category) for category in categories}

    df.loc[flagged.index, "anomaly_label"] = flagged.apply(
        lambda row: label_anomaly(row, thresholds), axis=1
    )

    return df
def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """
    Returns a summary dict of anomaly label counts.

    Parameters
    ----------
    df : pd.DataFrame — dataframe with 'anomaly_label' column

    Returns
    -------
    dict : {
        "overspend"    : int,
        "unusual_low"  : int,
        "normal"       : int,
        "total_flagged": int
    }
    """
    counts = df["anomaly_label"].value_counts().to_dict()

    return {
        "overspend"    : counts.get("overspend", 0),
        "unusual_low"  : counts.get("unusual_low", 0),
        "normal"       : counts.get("normal", 0),
        "total_flagged": counts.get("overspend", 0) + counts.get("unusual_low", 0),
    }