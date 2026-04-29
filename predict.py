import sys
import pathlib

# Add the project root to python path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid.classifier import HybridClassifier

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<transaction description>\"")
        print("Example: python predict.py \"UBR* LDN 8.40\"")
        sys.exit(1)

    description = sys.argv[1]
    
    try:
        classifier = HybridClassifier()
        result = classifier.predict(description)
        
        print(f"\nTransaction: {description}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence_type']}")
        print(f"Explanation: {result['explanation']}\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python src/ml/text_classifier.py")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
