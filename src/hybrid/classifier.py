import pathlib
import sys
import joblib
import numpy as np
import re

# --- Project setup ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.baseline import RuleBasedClassifier
from src.processing.merchant_normaliser import normalise_merchant

class HybridClassifier:
    def __init__(self, model_path=None, vectorizer_path=None):
        self.rule_based = RuleBasedClassifier()
        
        models_dir = PROJECT_ROOT / "models"
        self.model_path = model_path or (models_dir / "ml_model.pkl")
        self.vectorizer_path = vectorizer_path or (models_dir / "vectorizer.pkl")
        
        # Load ML components
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find ML models at {self.model_path} or {self.vectorizer_path}. "
                "Please run text_classifier.py to train and save the models first."
            )

    def _get_top_ml_features(self, text, top_n=3):
        """Extracts the top TF-IDF features that contributed to the prediction."""
        X = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get non-zero elements in the sparse matrix
        non_zero_indices = X.nonzero()[1]
        
        if len(non_zero_indices) == 0:
            return ["No significant features found"]
            
        # Get the tf-idf scores for these indices
        scores = X[0, non_zero_indices].toarray()[0]
        
        # Sort by score descending
        sorted_indices = non_zero_indices[np.argsort(scores)[::-1]]
        
        top_words = [feature_names[i] for i in sorted_indices[:top_n]]
        return top_words

    def _get_rule_matches(self, desc_norm, category):
        """Finds the specific keywords that matched for a category."""
        keywords = self.rule_based.knowledge_base.get(category, [])
        matches = [kw for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', desc_norm)]
        return matches

    def predict(self, description: str):
        """
        Predicts the category of a transaction description using a hybrid approach.
        Returns:
            dict containing: prediction, confidence_type, explanation
        """
        desc_lower = str(description).lower()
        desc_norm = normalise_merchant(desc_lower)
        
        # 1. Try Rule-Based first
        rule_pred = self.rule_based.predict([description])[0]
        
        if rule_pred != "Other":
            matches = self._get_rule_matches(desc_norm, rule_pred)
            return {
                "prediction": rule_pred,
                "confidence_type": "Rule-based match",
                "explanation": f"Matched keywords: {matches}"
            }
            
        # 2. Fallback to ML Model
        X = self.vectorizer.transform([desc_norm])
        ml_pred = self.model.predict(X)[0]
        
        top_features = self._get_top_ml_features(desc_norm)
        
        return {
            "prediction": ml_pred,
            "confidence_type": "ML fallback",
            "explanation": f"Top TF-IDF features: {top_features}"
        }
