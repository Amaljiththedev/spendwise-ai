
import pandas as pd
import numpy as np
import pathlib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Add the project root to python path to allow importing from src
BASE_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ingestion.loader import load_config, load_transactions
from src.processing.cleaner import clean_transactions


class RuleBasedClassifier:
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base or self._default_knowledge_base()
        # Flatten all possible labels to get categories
        self.categories = list(self.knowledge_base.keys())

    def _default_knowledge_base(self):
        return {
            "Rent": ["rent", "house", "home", "apartment", "mortgage"],
            "Food & Drink": ["food", "drink", "restaurant", "cafe", "grocery", "supermarket", "tesco", "sainsbury", "coffee", "takeaway"],
            "Travel": ["uber", "ride", "taxi", "bus", "train", "petrol", "parking", "flight", "fare"],
            "Utilities": ["electricity", "water", "internet", "gas", "bill"],
            "Shopping": ["amazon", "purchase", "shopping", "clothes", "gift", "electronics"],
            "Entertainment": ["netflix", "cinema", "spotify", "gym", "book", "subscription"],
            "Salary": ["salary", "paycheck", "credited", "consulting"],
            "Other": ["general", "purchase", "service", "fee", "miscellaneous"],
            "Transfer": ["transfer"],
            "Payroll": ["payroll"],
            "Salary to savings": ["salary to savings"],
            "Refund": ["refund"],
            "HMRC": ["hmrc"],
            "Natwest": ["natwest"],
            "Barclays": ["barclays"],
            "HSBC": ["hsbc"],
            "LLOYDS": ["LLOYDS"],
            "Sainsbury's Bank": ["sainsbury's bank"],
            "Tesco Bank": ["tesco bank"],
            "First Direct": ["first direct"],
            "Starling": ["starling"],
            "Monzo": ["monzo"],
            "Revolut": ["revolut"],
            "N26": ["n26"],
            "Ally": ["ally"],
            "Capital One": ["capital one"],
            "Chase": ["chase"],
            "American Express": ["american express"]
        }

    def predict(self, descriptions):
        predictions = []
        for desc in descriptions:
            desc_lower = str(desc).lower()
            best_category = "Other"
            max_matches = 0
            
            for category, keywords in self.knowledge_base.items():
                # Count matches for each category
                matches = sum(1 for keyword in keywords if keyword.lower() in desc_lower)
                
                if matches > max_matches:
                    max_matches = matches
                    best_category = category
            
            predictions.append(best_category)
        return np.array(predictions)

            





if __name__ == "__main__":
    try:
        # 1. Load Configuration
        config = load_config()
        
        # 2. Load and Clean Data
        print("--- Loading and Cleaning Data ---")
        df = load_transactions(BASE_DIR / "data" / "raw" / "finance.csv", config)
        cleaned_df, expenses_df, income_df = clean_transactions(df, config)
        print(f"Loaded {len(df)} transactions.")
        print(f"Cleaned {len(expenses_df)} expenses and {len(income_df)} income entries.\n")

        # 3. Initialize and Run Classifier
        print("--- Running Rule-Based Prediction ---")
        classifier = RuleBasedClassifier()
        
        # Combine descriptions for prediction
        descriptions = expenses_df[config["data"]["description_column"]].tolist()
        predictions = classifier.predict(descriptions)
        
        # Add predictions to the dataframe for viewing
        expenses_df['Predicted_Category'] = predictions
        
        # 4. Display Results
        print("\nSample Predictions:")
        sample_cols = [config["data"]["description_column"], config["data"]["category_column"], 'Predicted_Category', config["data"]["amount_column"]]
        print(expenses_df[sample_cols].head(15).to_string(index=False))
        
        # 5. Simple Evaluation
        actuals = expenses_df[config["data"]["category_column"]]
        accuracy = (predictions == actuals).mean()
        print(f"\nRule-based Accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

            




