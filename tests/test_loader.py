import sys
import pathlib
import pytest
import pandas as pd

# Add the project root to python path to allow importing from src
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.ingestion.loader import load_config, load_transactions, inspect_data

# Setup paths for tests
BASE_DIR = pathlib.Path(__file__).parent.parent
REAL_CONFIG_PATH = BASE_DIR / "configs" / "settings.yaml"
REAL_CSV_PATH = BASE_DIR / "data" / "raw" / "finance.csv"
def test_transactions_load_successfully():
    config = load_config(REAL_CONFIG_PATH)
    df = load_transactions(REAL_CSV_PATH, config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0                    # ← not 3000
    
    required_cols = config["data"]["required_columns"]
    for col in required_cols:
        assert col in list(df.columns)


def test_inspect_data_returns_correct_structure():
    config = load_config(REAL_CONFIG_PATH)
    df = load_transactions(REAL_CSV_PATH, config)
    report = inspect_data(df, config)
    
    expected_keys = [
        "total_rows", "total_columns", "null_values",
        "date_range", "columns", "amount_stats",
        "transactions_type", "categories_found",
        "description_length"
    ]
    
    for key in expected_keys:
        assert key in report
        
    assert report["total_rows"] == len(df)   # ← not 3000
    assert report["total_rows"] > 0
    assert "start_date" in report["date_range"]  # ← not a specific date
    assert "end_date" in report["date_range"]