import pathlib
import sys
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- Project setup ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_config, load_transactions
from src.processing.cleaner import clean_transactions
from src.rules.baseline import RuleBasedClassifier

def main():

    # --- Load data ---
    config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")

    data = load_transactions(
        PROJECT_ROOT / "data" / "raw" / "real_training_data.csv",
        config
    )

    data, expenses, income = clean_transactions(data, config)

    category_col = config["data"]["category_column"]
    description_col = config["data"]["description_column"]

    # --- Normalise Descriptions ---
    from src.processing.merchant_normaliser import normalise_merchant
    expenses[description_col] = expenses[description_col].apply(lambda x: normalise_merchant(str(x)))

    print("--- CATEGORY DISTRIBUTION ---")
    print(expenses[category_col].value_counts())
    print("\n")

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
    print("--- ML EVALUATION ---")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\n")

    # --- Cross-Validation ---
    print("--- CROSS VALIDATION ---")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", max_features=1000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    scores = cross_val_score(pipeline, 
                              expenses[description_col].fillna(""),
                              expenses[category_col],
                              cv=5, 
                              scoring="f1_weighted")

    print(f"CV F1 scores: {scores}")
    print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print("\n")

    # --- Baseline Comparison ---
    print("--- BASELINE COMPARISON ---")
    baseline = RuleBasedClassifier()
    baseline_preds = baseline.predict(X_test_text)
    baseline_accuracy = np.mean(baseline_preds == y_test)

    print(f"Rule-based accuracy: {baseline_accuracy:.2%}")
    print(f"ML model accuracy:   {accuracy:.2%}")
    print(f"Improvement:         {(accuracy - baseline_accuracy):.2%}")
    print("\n")

    # --- MLflow Tracking ---
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("spendwise-classifier")

    with mlflow.start_run(run_name="tfidf-logreg-realistic-data"):
        
        # log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", 1000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("training_data", "realistic_merchant_data")
        mlflow.log_param("n_categories", 7)
        mlflow.log_param("training_rows", len(expenses))

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("cv_f1_mean", scores.mean())
        mlflow.log_metric("cv_f1_std", scores.std())
        mlflow.log_metric("baseline_accuracy", baseline_accuracy)
        mlflow.log_metric("improvement_over_baseline", 
                           accuracy - baseline_accuracy)

        # log model
        mlflow.sklearn.log_model(model, "classifier")
        mlflow.sklearn.log_model(vectorizer, "vectorizer")

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    # --- Save model locally as well ---
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("Model and vectorizer saved successfully")


if __name__ == "__main__":
    main()