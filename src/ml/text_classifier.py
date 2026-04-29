import pathlib
import sys
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Project setup ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_config, load_transactions
from src.processing.cleaner import clean_transactions


def main():

    # --- Load data ---
    config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")

    data = load_transactions(
        PROJECT_ROOT / "data" / "raw" / "finance.csv",
        config
    )

    data, expenses, income = clean_transactions(data, config)

    category_col = config["data"]["category_column"]
    description_col = config["data"]["description_column"]

    # --- Split data ---
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        expenses[description_col].fillna(""),
        expenses[category_col],
        test_size=0.2,
        random_state=42
    )

    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"\b\w{3,}\b",
        max_features=1000
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # --- Model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --- Predictions ---
    predictions = model.predict(X_test)

    # --- Evaluation ---
    accuracy = np.mean(predictions == y_test)
    print("\nAccuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # --- Save model ---
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("\nModel and vectorizer saved successfully")


if __name__ == "__main__":
    main()