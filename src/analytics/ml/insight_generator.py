import os
from openai import OpenAI

def generate_insight(anomaly: dict) -> str:
    client = OpenAI(
        base_url="https://aiapiv2.pekpik.com/v1",
        api_key="sk-SuxLsZTEjwX70oe37FrpuZND2SS3q9CL9kkAJD3wO7NvqVsk"
    )
    
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

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content