import pathlib
import sys

# --- Project setup ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Baseline Rule-Based Classifier ---
from src.rules.baseline import RuleBasedClassifier
from src.ml.text_classifier import build_tfidf
from src.ingestion.loader import load_config, load_transactions
from src.processing.cleaner import clean_transactions

# --- Load data ---
config = load_config(PROJECT_ROOT / "configs" / "settings.yaml")
data = load_transactions(PROJECT_ROOT / "data" / "raw" / "finance.csv", config)
cleaned_df, expenses_df, income_df = clean_transactions(df, config)
