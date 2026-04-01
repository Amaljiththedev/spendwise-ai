import pandas as pd
import pathlib
import yaml

# Robustly resolve the path relative to the project directory
BASE_DIR = pathlib.Path(__file__).parent.parent.parent
CSV_FILE_PATH = BASE_DIR / "data" / "raw" / "finance.csv"
CONFIG_FILE_PATH = BASE_DIR / "configs" / "settings.yaml"

def load_config(config_path=CONFIG_FILE_PATH):
    # Convert config_path to Path object
    config_path = pathlib.Path(config_path)
    
    # Check if it doesn't exist
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    # Open the file and return yaml.safe_load
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_transactions(file_path, config):
    file_path = pathlib.Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    if file_path.suffix != ".csv":
        raise ValueError(f"File is not a CSV file: {file_path}")
    
    encoding = config["data"]["encoding"]
    df = pd.read_csv(file_path, encoding=encoding)

    if len(df) == 0:
        raise ValueError(f"No transactions found in {file_path}")
    
    required = config["data"]["required_columns"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    return df


def inspect_data(df,config):
    

    date_col = config["data"]["date_column"]
    amount_col = config["data"]["amount_column"]
    description_col = config["data"]["description_column"]
    category_col = config["data"]["category_column"]
    type_col = config["data"]["type_column"]

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_values" : df.isnull().sum().to_dict(),
        "date_range" :{
            "start_date": str(df[date_col].min()),
            "end_date": 
            str(df[date_col].max()),
            
        },
        "columns": list(df.columns),
        "amount_stats" : df[amount_col].describe().to_dict(),
        "transactions_type" : df[type_col].value_counts().to_dict(),
        "categories_found":  df[category_col].unique().tolist(),
        "description_length": df[description_col].str.len().describe().to_dict(),
    }
    









if __name__ == "__main__":
    try:
        config = load_config(BASE_DIR / "configs" / "settings.yaml")
        df     = load_transactions(CSV_FILE_PATH, config)
        report = inspect_data(df, config)
        
        print(f"Loaded {report['total_rows']} transactions")
        print(f"Date range: {report['date_range']['start_date']} → {report['date_range']['end_date']}")
        print(f"Categories: {report['categories_found']}")
        print(f"Nulls: {report['null_values']}")
        print(f"Transactions type: {report['transactions_type']}")
        print(f"Amount: {report['amount_stats']}")
        print(f"Description length: {report['description_length']}")
        print(f"Columns: {report['columns']}")



    except FileNotFoundError as e:
        print(f"File error: {e}")
    except ValueError as e:
        print(f"Data error: {e}")   