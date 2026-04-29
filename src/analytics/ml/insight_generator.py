import time
from google import genai
from google.genai import errors

def generate_insight(anomaly: dict) -> str:
    # 1. Setup the Google AI Client
    client = genai.Client(api_key="AIzaSyAZ8PxdSz6K9USz9FmWxN2kf9uhjq2wkg8")
    
    reasons_text = "\n".join(anomaly["reasons"]) if anomaly["reasons"] else "flagged as statistically unusual"
    
    prompt = f"""You are a friendly financial advisor for students.
    
A transaction has been flagged as anomalous:
- Date: {anomaly['date']}
- Category: {anomaly['category']}
- Amount: £{anomaly['amount']:.2f}
- Severity: {anomaly['severity']}
- Reasons: {reasons_text}

Write exactly 2 sentences. Be specific with the numbers. 
Be helpful, not alarming. No generic advice."""

    # 2. Retry loop to handle potential service issues (built-in but made explicit here)
    for attempt in range(3):
        try:
            # Using 'gemini-flash-latest' as it is verified to be available for this API key
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text.replace("\n", " ").strip()
            
        except Exception as e:
            # Check for 503/429 or other retryable errors
            if "503" in str(e) or "429" in str(e):
                if attempt < 2:
                    time.sleep(2)
                    continue
            return f"Error: {str(e)}"