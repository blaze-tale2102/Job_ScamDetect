# 🛡️ Job Market Scam Detection Using Machine Learning

Classifies job postings as **Legit**, **Suspicious**, or **Likely Scam** using
an ensemble of NLP features and gradient-boosted trees, with explainable
red-flag highlighting.

---

## 📂 Project Structure

```
job-scam-detection/
├── data/
│   ├── raw/                    # Place fake_job_postings.csv here
│   └── processed/              # Train / val / test splits (auto-generated)
├── notebooks/
│   └── 01_eda.py               # Exploratory Data Analysis
├── src/
│   ├── preprocess.py           # Text cleaning & splitting
│   ├── features.py             # TF-IDF + engineered features
│   ├── train.py                # Full training pipeline
│   ├── evaluate.py             # Metrics & plots
│   └── predict.py              # Inference API
├── models/                     # Saved pipeline (auto-generated)
├── reports/
│   ├── figures/                # EDA & evaluation charts
│   └── metrics.json            # All model metrics
├── app/
│   └── app.py                  # Streamlit web app
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
├── requirements.txt
└── README.md
```

## 🚀 Setup

```bash
# 1. Clone / navigate to the project
cd job-scam-detection

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
#    → https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
#    Place fake_job_postings.csv in data/raw/
```

## 🔬 Usage

### Run EDA
```bash
python notebooks/01_eda.py
# Charts saved to reports/figures/
```

### Train models
```bash
python -m src.train
# Trains 7 model variants, picks the best by F1 on the fraud class,
# runs 5-fold CV, and saves the pipeline to models/
```

### Run tests
```bash
pytest tests/ -v
```

### Launch the Streamlit app
```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

## 📊 Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Multinomial NB | — | — | — | — | — |
| Linear SVM | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| **LightGBM** | — | — | — | — | — |
| LR + SMOTE | — | — | — | — | — |
| LightGBM + SMOTE | — | — | — | — | — |

> *Table populated after training. See `reports/metrics.json` for exact values.*

## 🧠 How It Works

1. **Text Preprocessing**: HTML stripping, URL/email/phone removal, stopword
   filtering, lowercasing.
2. **Feature Engineering**: TF-IDF (unigrams + bigrams) on cleaned text, plus
   11 hand-crafted features (logo presence, urgency words, ALL-CAPS ratio,
   contact-info in description, etc.).
3. **Modeling**: Seven classifiers compared — the best is selected by F1 on
   the fraud class (not accuracy) with threshold tuning on the validation set.
4. **Explainability**: Red-flag reasons are surfaced per prediction based on
   feature values and text pattern matching.

## ⚖️ Limitations & Ethical Notes

- **False positives** can harm legitimate employers whose postings are
  incorrectly flagged — this model should be used as a **screening aid**, not a
  final decision.
- **Class imbalance** (~5% fraud) makes high recall difficult without some
  precision trade-off. The threshold is tuned to balance both.
- The model is trained on a single static dataset (EMSCAD) and may not
  generalise well to new scam tactics, different languages, or different job
  platforms.
- **Bias**: The dataset may over-represent certain geographies or industries;
  predictions should be interpreted with caution.
- **Privacy**: No personal data is collected or stored by the app.

## 📜 License

This project is for educational and research purposes.
Dataset: [EMSCAD / Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (CC0).
