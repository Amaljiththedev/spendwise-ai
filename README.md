# spendwise-ai

![CI](https://github.com/Amaljiththedev/spendwise-ai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> AI-powered personal finance analyser for students. Upload a bank transaction CSV — get intelligent spending insights, anomaly alerts, and plain-English financial advice back.

---

## What It Does

Students overspend because they don't clearly understand where their money goes. Most banking apps show transactions but generate no useful insights. spendwise-ai solves this with a production-grade ML pipeline that:

- Automatically categorises transactions using NLP
- Detects unusual spending with anomaly detection
- Explains why a transaction looks suspicious
- Generates plain-English insights using an LLM
- Tracks every ML experiment with MLflow

---

## Architecture

```
Bank CSV upload
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Ingestion  │────▶│   Cleaning   │────▶│    Analytics    │
│  loader.py  │     │  cleaner.py  │     │   engine.py     │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                  │
                    ┌─────────────────────────────┘
                    ▼
      ┌─────────────────────────┐
      │     ML Pipeline         │
      │                         │
      │  merchant_normaliser.py │  strips bank noise
      │  baseline.py            │  rule-based: 96.68%
      │  text_classifier.py     │  TF-IDF + LR: 98.1% F1
      │  anomaly_detection.py   │  Isolation Forest + LOF
      │  anomaly_explainer.py   │  reasons + severity
      │  insight_generator.py   │  LLM narrative layer
      └─────────────────────────┘
                    │
                    ▼
             FastAPI (coming)
             Dashboard (coming)
```

---

## ML Results

| Approach | Accuracy | Notes |
|---|---|---|
| Rule-based (no normaliser) | 61.00% | Baseline |
| Rule-based + merchant normaliser | 96.68% | +35.68% from noise stripping |
| TF-IDF + Logistic Regression | 98.1% CV F1 | 5-fold cross-validation |
| Improvement over raw baseline | +53.57% | Documented in MLflow |

**Cross-validation:** 98.1% ± 0.7% weighted F1 across 5 folds  
**Categories:** Food & Drink, Travel, Shopping, Entertainment, Rent, Utilities, Health & Fitness  
**Anomalies detected:** 62 across 5 years of data

---

## Features Built

### Data Pipeline
- CSV ingestion with schema validation and error handling
- Date parsing, merchant normalisation, type splitting
- Config-driven — no hardcoded column names

### ML Classification
- Rule-based baseline with keyword matching
- Merchant normaliser — strips reference codes, bank prefixes, special characters
- TF-IDF + Logistic Regression classifier
- Baseline vs ML comparison with MLflow tracking

### Anomaly Detection
- Isolation Forest flags statistically unusual transactions
- Local Outlier Factor confirms anomalies
- Anomaly explainer — generates human-readable reasons per flag
- Severity scoring — high, medium, low

### LLM Insight Layer
- Anthropic API integration
- Structured anomaly data passed as context
- LLM narrates pre-computed facts — never does the analysis itself
- 2-sentence plain English insight per anomaly

### Analytics Engine
- Monthly spending totals
- Category breakdown ranked by spend
- Monthly category breakdown
- Spending trend — increasing, stable, decreasing
- Top N categories from config

### MLOps
- MLflow experiment tracking — parameters, metrics, model artifacts
- GitHub Actions CI — runs pytest on every push
- Models saved to `models/` with joblib
- Config-driven architecture via `settings.yaml`

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| ML | Scikit-learn (TF-IDF, LogReg, Isolation Forest, LOF) |
| NLP | TF-IDF vectoriser, merchant normaliser |
| LLM | Anthropic API (Claude) |
| Experiment tracking | MLflow |
| Testing | pytest (17 passing tests) |
| CI/CD | GitHub Actions |
| Config | PyYAML |

---

## Quick Start

```bash
git clone https://github.com/Amaljiththedev/spendwise-ai.git
cd spendwise-ai

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Add your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Run the classifier:
```bash
python src/ml/text_classifier.py
```

View MLflow experiments:
```bash
mlflow ui
# Open http://localhost:5000
```

Run all tests:
```bash
pytest tests/ -v
```

---

## Project Structure

```
spendwise-ai/
├── .github/workflows/     CI pipeline
├── configs/
│   └── settings.yaml      all column names and config
├── data/
│   └── raw/               transaction CSVs
├── models/                saved model artifacts
├── notebooks/             exploration only
├── src/
│   ├── ingestion/
│   │   └── loader.py      CSV reading and validation
│   ├── processing/
│   │   ├── cleaner.py     date parsing, normalisation
│   │   └── merchant_normaliser.py  strips bank noise
│   ├── analytics/
│   │   ├── engine.py      monthly totals, trends
│   │   └── ml/
│   │       ├── anomaly_detection.py
│   │       ├── anomaly_explainer.py
│   │       └── insight_generator.py
│   ├── ml/
│   │   └── text_classifier.py  TF-IDF + LogReg
│   └── rules/
│       └── baseline.py    rule-based classifier
├── tests/
│   ├── test_loader.py     7 tests
│   └── test_cleaner.py    10 tests
└── requirements.txt
```

---

## Tests

```
pytest tests/ -v

tests/test_loader.py   7 passed
tests/test_cleaner.py  10 passed
─────────────────────
17 passed total
```

---

## MLflow Experiment

```
Experiment:    spendwise-classifier
Run:           tfidf-logreg-realistic-data
Model:         LogisticRegression
Vectoriser:    TF-IDF (max_features=1000)
Training rows: 2503
CV F1 mean:    0.993
Accuracy:      0.988
Baseline:      0.966
Improvement:   +0.022
```

Run `mlflow ui` to view the full experiment dashboard.

---

## Sample Output

```
Anomalies found: 62

{
  "date": "2020-11-12",
  "category": "Food & Drink",
  "amount": 1999.15,
  "reasons": [
    "£1999.15 is 1.9x your usual average of £1070.43 for Food & Drink",
    "5 Food & Drink transactions this month vs your usual 3"
  ],
  "severity": "high",
  "insight": "Your Food & Drink spending hit £1,999 this month —
              nearly twice your usual average, with 5 transactions
              instead of your typical 3. It is worth checking
              whether any of these were planned one-off expenses."
}
```

---

## What I'd Do Next

- FastAPI layer — expose all components as REST endpoints
- Financial health score — 0-100 composite metric
- Spending forecast — Ridge regression for next month prediction
- Subscription detector — recurring charge identification
- Docker container — single-command deployment
- React dashboard — visualise spending, anomalies, health score

---

## Why This Project

Built as a flagship AI/ML portfolio project demonstrating:

- **Data engineering** — ingestion, validation, cleaning pipeline
- **ML engineering** — classification, anomaly detection, evaluation
- **MLOps** — experiment tracking, CI/CD, config-driven architecture
- **LLM integration** — grounded narrative generation, not hallucination
- **Engineering discipline** — 17 tests, clean commits, professional structure

---

## Author

**Amaljith T A**  
MSc Advanced Data Science & AI — University of Liverpool  
[GitHub](https://github.com/Amaljiththedev) · [LinkedIn](https://www.linkedin.com/in/amaljith-t-a-ba85b334a/)