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


def test_config_loads_successfully():
    result = load_config(REAL_CONFIG_PATH)
    assert isinstance(result, dict)
    assert "data" in result
    assert "required_columns" in result["data"]


def test_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("fake/path/config.yaml")


def test_transactions_load_successfully():
    config = load_config(REAL_CONFIG_PATH)
    df = load_transactions(REAL_CSV_PATH, config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1500
    
    required_cols = config["data"]["required_columns"]
    for col in required_cols:
        assert col in list(df.columns)


def test_file_not_found():
    config = load_config(REAL_CONFIG_PATH)
    with pytest.raises(FileNotFoundError):
        load_transactions("fake/transactions.csv", config)


def test_wrong_file_extension(tmp_path):
    config = load_config(REAL_CONFIG_PATH)
    
    # Create a real text file to ensure FileNotFoundError isn't raised first
    txt_file = tmp_path / "test.txt"
    txt_file.touch()
    
    with pytest.raises(ValueError):
        load_transactions(txt_file, config)


def test_missing_column(tmp_path):
    config = load_config(REAL_CONFIG_PATH)
    
    # Create an in-memory CSV missing the Amount column
    df_missing = pd.DataFrame({
        "Date": ["2020-01-01"], 
        "Transaction Description": ["Test"],
        "Category": ["Food"],
        "Type": ["Expense"]
        # "Amount" is intentionally absent here
    })
    
    fake_csv = tmp_path / "test.csv"
    df_missing.to_csv(fake_csv, index=False)
    
    with pytest.raises(ValueError) as excinfo:
        load_transactions(fake_csv, config)
        
    assert "Amount" in str(excinfo.value)


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
        
    assert report["total_rows"] == 1500
    assert report["date_range"]["start_date"] == "2020-01-02"
