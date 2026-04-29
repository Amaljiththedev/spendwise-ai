"""
Compare anomaly detection results: Isolation Forest vs LOF, global vs per-category.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_config, load_transactions
from src.analytics.ml.feature_engineering import build_features, build_features_per_category
from src.processing.cleaner import clean_transactions


def _run_global(expenses, config, contamination=0.03):
    """Run both models in global mode, return expenses with prediction columns."""
    amount_col = config["data"]["amount_column"]
    feat_df, _ = build_features(expenses, config)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df)

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=200, max_samples="auto",
        contamination=contamination, random_state=42,
    )
    iso.fit(X_scaled)
    expenses["iso_pred"] = iso.predict(X_scaled)
    expenses["iso_score"] = iso.decision_function(X_scaled)

    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    expenses["lof_pred"] = lof.fit_predict(X_scaled)
    expenses["lof_score"] = lof.negative_outlier_factor_

    return expenses


def _run_per_category(expenses, config, contamination=0.03):
    """Run both models per-category, return expenses with prediction columns."""
    amount_col = config["data"]["amount_column"]
    cat_data = build_features_per_category(expenses, config)

    expenses["iso_pred_pc"] = np.nan
    expenses["lof_pred_pc"] = np.nan

    scaler = StandardScaler()

    for category, (subset, feat_df, _) in cat_data.items():
        if len(subset) < 10:
            continue
        X_scaled = scaler.fit_transform(feat_df)

        # ISO
        iso = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42,
        )
        iso.fit(X_scaled)
        expenses.loc[subset.index, "iso_pred_pc"] = iso.predict(X_scaled)

        # LOF
        n_neighbors = min(20, len(subset) - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        expenses.loc[subset.index, "lof_pred_pc"] = lof.fit_predict(X_scaled)

    return expenses


def main():
    config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")
    data = load_transactions(PROJECT_ROOT / "data" / "raw" / "finance.csv", config)
    _, expenses, _ = clean_transactions(data, config)

    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    expenses = expenses.copy()
    expenses[amount_col] = pd.to_numeric(expenses[amount_col], errors="coerce")
    expenses = expenses.dropna(subset=["Date", amount_col, category_col])

    # --- Run all modes ---
    expenses = _run_global(expenses, config)
    expenses = _run_per_category(expenses, config)

    # --- Summary ---
    cols = ["Date", category_col, amount_col]

    iso_g = expenses[expenses["iso_pred"] == -1]
    lof_g = expenses[expenses["lof_pred"] == -1]
    iso_pc = expenses[expenses["iso_pred_pc"] == -1]
    lof_pc = expenses[expenses["lof_pred_pc"] == -1]

    print("=" * 70)
    print("ANOMALY DETECTION COMPARISON")
    print("=" * 70)
    print(f"\nTotal expense transactions: {len(expenses)}")
    print(f"Contamination rate:        3% (all models)")

    print(f"\n{'Method':<35} {'Anomalies':<12} {'% of total'}")
    print("-" * 55)
    for label, df in [
        ("Isolation Forest (global)", iso_g),
        ("Isolation Forest (per-category)", iso_pc),
        ("LOF (global)", lof_g),
        ("LOF (per-category)", lof_pc),
    ]:
        pct = len(df) / len(expenses) * 100
        print(f"{label:<35} {len(df):<12} {pct:.1f}%")

    # --- Overlap (global) ---
    both_g = expenses[(expenses["iso_pred"] == -1) & (expenses["lof_pred"] == -1)]
    print(f"\nGlobal: flagged by BOTH models: {len(both_g)}")

    # --- Category breakdown ---
    print("\n" + "=" * 70)
    print("ANOMALIES BY CATEGORY")
    print("=" * 70)

    print(f"\n{'Category':<20} {'ISO(g)':<10} {'LOF(g)':<10} {'ISO(pc)':<10} {'LOF(pc)'}")
    print("-" * 60)
    all_cats = sorted(expenses[category_col].unique())
    for cat in all_cats:
        ig = len(iso_g[iso_g[category_col] == cat])
        lg = len(lof_g[lof_g[category_col] == cat])
        ip = len(iso_pc[iso_pc[category_col] == cat])
        lp = len(lof_pc[lof_pc[category_col] == cat])
        print(f"{cat:<20} {ig:<10} {lg:<10} {ip:<10} {lp}")

    # --- Sample per-category anomalies ---
    print("\n" + "=" * 70)
    print("PER-CATEGORY ISO FOREST — TOP ANOMALIES BY CATEGORY")
    print("=" * 70)
    for cat in all_cats:
        cat_anom = iso_pc[iso_pc[category_col] == cat]
        if cat_anom.empty:
            continue
        print(f"\n--- {cat} ({len(cat_anom)} anomalies) ---")
        print(cat_anom[cols].sort_values(amount_col, ascending=False).head(5).to_string(index=False))

    # --- Dataset stats ---
    print("\n" + "=" * 70)
    print("DATASET CONTEXT — per-category spending stats")
    print("=" * 70)
    stats = expenses.groupby(category_col)[amount_col].agg(["mean", "std", "min", "max", "count"])
    print(stats.round(2).to_string())


if __name__ == "__main__":
    main()
