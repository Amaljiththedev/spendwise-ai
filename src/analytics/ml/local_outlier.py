
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_config, load_transactions
from src.analytics.ml.anomaly_labeller import apply_anomaly_labelling
from src.analytics.ml.feature_engineering import build_features, build_features_per_category
from src.processing.cleaner import clean_transactions


def _fit_lof(X_scaled, n_neighbors=20, contamination=0.03):
    """Fit LOF and return (scores, predictions)."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(X_scaled)
    scores = lof.negative_outlier_factor_
    return scores, predictions


def run_local_outlier_factor(
    show_plot: bool = False,
    per_category: bool = False,
    contamination: float = 0.03,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")
    data = load_transactions(PROJECT_ROOT / "data" / "raw" / "finance.csv", config)
    data, expenses, _ = clean_transactions(data, config)

    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    expenses = expenses.copy()
    expenses[amount_col] = pd.to_numeric(expenses[amount_col], errors="coerce")
    expenses = expenses.dropna(subset=["Date", amount_col, category_col])

    # Initialise result columns
    data["lof_score"] = np.nan
    data["lof_prediction"] = np.nan

    scaler = StandardScaler()

    if per_category:
        # --- Per-category mode ---
        cat_data = build_features_per_category(expenses, config)
        for category, (subset, feat_df, _) in cat_data.items():
            if len(subset) < 10:
                continue
            # LOF needs n_neighbors < n_samples
            n_neighbors = min(20, len(subset) - 1)
            X_scaled = scaler.fit_transform(feat_df)
            scores, preds = _fit_lof(
                X_scaled, n_neighbors=n_neighbors, contamination=contamination,
            )
            data.loc[subset.index, "lof_score"] = scores
            data.loc[subset.index, "lof_prediction"] = preds

        # Categories too small for per-category detection get global fallback
        covered_idx = data.index[data["lof_prediction"].notna()]
        uncovered = expenses[~expenses.index.isin(covered_idx)]
        if not uncovered.empty:
            feat_df, _ = build_features(uncovered, config)
            X_scaled = scaler.fit_transform(feat_df)
            scores, preds = _fit_lof(
                X_scaled, contamination=contamination,
            )
            data.loc[uncovered.index, "lof_score"] = scores
            data.loc[uncovered.index, "lof_prediction"] = preds
    else:
        # --- Global mode ---
        feat_df, _ = build_features(expenses, config)
        X_scaled = scaler.fit_transform(feat_df)
        scores, preds = _fit_lof(X_scaled, contamination=contamination)
        data.loc[expenses.index, "lof_score"] = scores
        data.loc[expenses.index, "lof_prediction"] = preds

    data = apply_anomaly_labelling(data, thresholds={}, pred_col="lof_prediction")

    # --- Plot ---
    plot_data = data.dropna(subset=["lof_prediction", amount_col, "Date"]).copy()
    plot_data[amount_col] = pd.to_numeric(plot_data[amount_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[amount_col])

    mode_label = "Per-Category" if per_category else "Global"
    plt.figure(figsize=(14, 6))
    sns.scatterplot(
        data=plot_data, x="Date", y=amount_col,
        hue="lof_prediction", palette={1.0: "steelblue", -1.0: "red"}, alpha=0.7,
    )
    plt.xticks(rotation=45)
    plt.title(f"Local Outlier Factor — {mode_label} Mode")
    plt.tight_layout()
    suffix = "_percat" if per_category else ""
    plt.savefig(PROJECT_ROOT / f"anomaly_detection_lof{suffix}.png", dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    # --- Summary ---
    anomalies = plot_data[plot_data["lof_prediction"] == -1]
    print(f"\nMode: {mode_label}")
    print(f"Total anomalies detected: {len(anomalies)}")
    print(anomalies[["Date", category_col, amount_col]].sort_values(amount_col, ascending=False))
    print("\nAnomaly labels:")
    print(data["anomaly_label"].value_counts())

    if per_category:
        print("\nAnomalies by category:")
        print(anomalies[category_col].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LOF anomaly detection on expense transactions.",
    )
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument(
        "--per-category", action="store_true",
        help="Fit a separate model per spending category instead of one global model.",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.03,
        help="Expected proportion of anomalies (default: 0.03).",
    )
    args = parser.parse_args()
    run_local_outlier_factor(
        show_plot=args.show_plot,
        per_category=args.per_category,
        contamination=args.contamination,
    )