import pandas as pd
import pathlib
import sys
import re


BASE_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ingestion.loader import load_config, load_transactions

def clean_dates(df, config):
    date_col = config["data"]["date_column"]
    date_format = config["data"]["cleaning"]["date_format"]
    
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    
    
    null_dates = df[date_col].isnull().sum()
    if null_dates > 0:
        raise ValueError(f"Found {null_dates} null dates")
    
    df['month'] = df[date_col].dt.to_period('M').astype(str)
    df['year'] = df[date_col].dt.year
    df['month_number']= df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.day_name()
    
    return df
        








def split_by_type(df, config):
    expense_label = config["data"]["expense_label"]
    income_label = config["data"]["income_label"]
    type_col = config["data"]["type_column"]
    
    expenses_df = df[df[type_col] == expense_label].copy()
    income_df = df[df[type_col] == income_label].copy()
    
    total = len(expenses_df) + len(income_df)

    if total != len(df):
        raise ValueError(
            f"Split lost transactions. "
            f"Expected {len(df)}, got {total}"
        )

    return expenses_df, income_df



def normalise_descriptions(df, config):
    desc_col = config["data"]["description_column"]
    df[desc_col] = df[desc_col].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    
    return df

def clean_transactions(df, config):
    df = clean_dates(df, config)
    df = normalise_descriptions(df, config)
    expenses_df, income_df = split_by_type(df, config)
    return df, expenses_df, income_df



if __name__ == "__main__":
    config = load_config(BASE_DIR / "configs" / "settings.yaml")
    df = load_transactions(BASE_DIR / "data" / "raw" / "finance.csv", config)
    
    cleaned_df, expenses_df, income_df = clean_transactions(df, config)
    
    print(f"Total: {len(cleaned_df)}")
    print(f"Expenses: {len(expenses_df)}")
    print(f"Income: {len(income_df)}")
    print(cleaned_df[['Date', 'month', 'year', 'day_of_week']].head())
    print(cleaned_df[config["data"]["description_column"]].head())
    



