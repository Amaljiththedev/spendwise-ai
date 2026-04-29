"""
Shared feature engineering for anomaly detection models.

Provides consistent feature construction for both global and per-category
anomaly detection across Isolation Forest and LOF pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# Day-of-week string → integer mapping (Monday=0 … Sunday=6)
_DOW_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


def build_features(
    expenses: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a feature matrix for global anomaly detection.

    Features produced:
        - Amount (numeric, from config)
        - month_number (1–12, already created by cleaner)
        - day_of_month (1–31, extracted from Date)
        - day_of_week_num (0=Mon … 6=Sun, derived from cleaner's day_of_week string)
        - One-hot encoded category columns (drop_first=True to avoid collinearity)

    Parameters
    ----------
    expenses : pd.DataFrame
        Cleaned expense transactions (output of clean_transactions).
    config : dict
        Project config loaded from settings.yaml.

    Returns
    -------
    feature_df : pd.DataFrame
        Feature matrix aligned to the expenses index.
    feature_names : list[str]
        Ordered list of feature column names.
    """
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    df = expenses.copy()

    # --- Numeric features ---
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df["day_of_month"] = df["Date"].dt.day
    df["day_of_week_num"] = df["day_of_week"].map(_DOW_MAP)

    # --- One-hot encode categories (drop first to avoid dummy-variable trap) ---
    cat_dummies = pd.get_dummies(
        df[category_col], prefix="cat", drop_first=True, dtype=int,
    )

    # --- Assemble feature matrix ---
    numeric_cols = [amount_col, "month_number", "day_of_month", "day_of_week_num"]
    feature_df = pd.concat([df[numeric_cols], cat_dummies], axis=1)
    feature_names = list(feature_df.columns)

    return feature_df, feature_names


def build_features_per_category(
    expenses: pd.DataFrame,
    config: dict,
    min_category_size: int = 30,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, List[str]]]:
    """
    Build per-category feature matrices for category-level anomaly detection.

    Categories with fewer than `min_category_size` transactions are skipped
    (they should fall back to the global model).

    Features produced (no category columns — irrelevant within a single category):
        - Amount
        - month_number
        - day_of_month
        - day_of_week_num

    Parameters
    ----------
    expenses : pd.DataFrame
        Cleaned expense transactions.
    config : dict
        Project config.
    min_category_size : int
        Minimum number of transactions required to fit a per-category model.

    Returns
    -------
    dict mapping category_name → (subset_df, feature_df, feature_names)
        subset_df : the original expense rows for this category
        feature_df : numeric feature matrix
        feature_names : list of feature column names
    """
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    df = expenses.copy()
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df["day_of_month"] = df["Date"].dt.day
    df["day_of_week_num"] = df["day_of_week"].map(_DOW_MAP)

    feature_cols = [amount_col, "month_number", "day_of_month", "day_of_week_num"]

    result: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, List[str]]] = {}

    for category, group in df.groupby(category_col):
        if len(group) < min_category_size:
            continue
        subset = group.copy()
        features = subset[feature_cols].copy()
        result[category] = (subset, features, list(feature_cols))

    return result
