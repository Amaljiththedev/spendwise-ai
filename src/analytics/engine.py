import pandas as pd
import pathlib
import sys

# Add the project root to python path to allow importing from src
BASE_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ingestion.loader import load_config, load_transactions
from src.processing.cleaner import clean_transactions

def monthly_summary(expenses_df, config):
    amount_col = config["data"]["amount_column"]

    summary = expenses_df.groupby("month")[amount_col].sum()

    return dict(sorted(summary.to_dict().items()))


def categrory_breakdown(expenses_df,config):
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]


    breakdown = expenses_df.groupby(category_col)[amount_col].sum()
    
    return dict(sorted(breakdown.to_dict().items()))

def monthly_category_breakdown(expenses_df,config):
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    breakdown = expenses_df.groupby(["month",category_col])[amount_col].sum()
    
    result = {}

    for (month,category),amount in breakdown.items():
        if month not in result:
            result[month] = {}
        result[month][category] = amount

    return dict(sorted(result.items()))


def top_categories(expenses_df,config,n=None):
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]
    
    if n is None:
        n = config["analytics"]["top_n_categories"]
    
    breakdown = expenses_df.groupby(category_col)[amount_col].sum()
    top = sorted(breakdown.nlargest(n).items())

    return dict(top)


def spending_trends(expenses_df,config):
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    monthly = expenses_df.groupby(['month', category_col])[amount_col].sum().unstack(fill_value=0)
    
    trend = monthly.pct_change() * 100
    latest = trend.iloc[-1].round(2).to_dict()


    result = {}

    for cat, val in latest.items():
        if val>0:
            direction = "increasing"
        else:
            direction = "decreasing"
        result[cat] = {"direction": direction, "percentage": abs(val)}

    return dict(sorted(result.items()))

def income_vs_expense(config,income_df,expenses_df):
    amount_col = config["data"]["amount_column"]

    monthly_income = income_df.groupby("month")[amount_col].sum()
    monthly_expenses = expenses_df.groupby("month")[amount_col].sum()

    summary = pd.DataFrame({
        "income": monthly_income,
        "expenses": monthly_expenses
    }).fillna(0)

    summary["net"] = summary["income"] - summary["expenses"]

    return summary.to_dict(orient="index")



def expense_distrubution(expenses_df,config):
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    category_expenses = expenses_df.groupby(category_col)[amount_col].sum()

    total_expense = category_expenses.sum()
    precentage_distribution = (category_expenses / total_expense * 100).round(2)

    return precentage_distribution.to_dict()



def average_transaction(expenses_df,config):
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    stats = expenses_df.groupby(category_col)[amount_col].agg(
        total_transactions="count",
        total_expenses="sum",
        
    )
    
    stats["average_transaction"] = (stats["total_expenses"] / stats["total_transactions"]).round(2)

    return stats.to_dict(orient="records")



def income_source_analysis(income_df,config):
    amount_col = config["data"]["amount_column"]
    category_col = config["data"]["category_column"]

    stats = income_df.groupby(category_col)[amount_col].agg(
        total_transactions="count",
        total_income="sum",
        average_income="mean",
        income_std="std"
    )
    stats["income_share_%"] = (stats["total_income"] / stats["total_income"].sum() * 100).round(2)
    return stats.reset_index()
    

def cash_flow_analysis(config,income_df,expenses_df):
    amount_col = config["data"]["amount_column"]
    date_col = config["data"]["date_column"]

    income_df["month"] = pd.to_datetime(income_df[date_col]).dt.to_period("M")
    expenses_df["month"] = pd.to_datetime(expenses_df[date_col]).dt.to_period("M")

    monthly_income = income_df.groupby(income_df[date_col].dt.to_period("M"))[amount_col].sum()

    monthly_expenses = expenses_df.groupby(expenses_df[date_col].dt.to_period("M"))[amount_col].sum()

    cash_flow = pd.concat([monthly_income,monthly_expenses],axis=1,keys=["income","expenses"])

    cash_flow.columns = ["income","expenses"]
    
    cash_flow["net"] = cash_flow["income"] - cash_flow["expenses"]

    cash_flow["cumulative_net"] = cash_flow["net"].cumsum()

    return cash_flow.fillna(0).reset_index()
    








if __name__ == "__main__":
    # Load configuration
    config = load_config(BASE_DIR / "configs" / "settings.yaml")
    
    # Load and clean data
    csv_path = BASE_DIR / "data" / "raw" / "finance.csv"
    df = load_transactions(csv_path, config)
    cleaned_df, expenses_df, income_df = clean_transactions(df, config)
    
    print("\n--- Monthly Summary ---")
    summary = monthly_summary(expenses_df, config)
    for month, total in list(summary.items())[:5]:
        print(f"  {month}: {total:,.2f}")

    print("\n--- Category Breakdown ---")
    breakdown = categrory_breakdown(expenses_df, config)
    for category, total in list(breakdown.items())[:5]:
        print(f"  {category}: {total:,.2f}")   
    
    print("\n--- Monthly Category Breakdown ---")
    monthly_breakdown = monthly_category_breakdown(expenses_df, config)
    for month, categories in list(monthly_breakdown.items())[:5]:
        print(f"  {month}: {categories}")       

    print("\n--- Top Categories ---")
    top = top_categories(expenses_df, config, n=3)
    for cat, total in top.items():
        print(f"  {cat}: £{total:,.2f}")

    
    print("\n--- Spending Trend (latest month) ---")
    trend = spending_trends(expenses_df, config)
    for cat, data in trend.items():
        print(f"  {cat}: {data['direction']} ({data['percentage']})")

    print("\n--- Income vs Expense ---")
    income_expense = income_vs_expense(config,income_df,expenses_df)
    for month, data in list(income_expense.items())[:10]:
        print(f"  {month}: {data['income']:,.2f} / {data['expenses']:,.2f} / {data['net']:,.2f}")

    print("\n--- Expense Distribution ---")
    distribution = expense_distrubution(expenses_df,config)
    for category, percentage in list(distribution.items())[:10]:
        print(f"  {category}: {percentage:,.2f}%")

    print("\n--- Average Transaction ---")
    average = average_transaction(expenses_df,config)
    print(average)

    print("\n--- Income Source Analysis ---")
    income_source = income_source_analysis(income_df,config)
    print(income_source)

    print("\n--- Cash Flow Analysis ---")
    cash_flow = cash_flow_analysis(config,income_df,expenses_df)
    print(cash_flow)