# 🛡️ Job Market Scam Detection Using Machine Learning

Classifies job postings as **Legit**, **Suspicious**, or **Likely Scam** using an ensemble of NLP features and gradient-boosted trees/linear models, with explainable red-flag highlighting and an interactive Next.js glassmorphic dashboard.

---

## 📂 Project Structure

```
job-scam-detection/
├── data/
│   ├── raw/                    # fake_job_postings.csv (EMSCAD dataset)
│   └── processed/              # Train / val / test splits (auto-generated)
├── notebooks/
│   └── 01_eda.py               # Exploratory Data Analysis & visualisations
├── src/
│   ├── preprocess.py           # Text cleaning, regex sanitization, stratified splitting
│   ├── features.py             # TF-IDF (unigrams+bigrams) + 11 engineered features
│   ├── train.py                # Model training, hyperparameter & threshold tuning
│   ├── evaluate.py             # Comprehensive test set evaluation & metric charts
│   └── predict.py              # Inference API with explainable red-flag detection
├── models/
│   ├── feature_builder.joblib  # Serialized feature extraction pipeline
│   ├── final_model.joblib      # Trained best classifier (Logistic Regression + SMOTE)
│   └── model_config.json       # Pipeline metadata & optimal classification threshold
├── reports/
│   ├── figures/                # EDA & evaluation charts (ROC, PR, Confusion Matrix)
│   └── metrics.json            # Final metrics for all 7 candidate models
├── app/
│   ├── server.py               # Flask REST API for prediction & stats
│   └── app.py                  # Streamlit web interface (legacy)
├── frontend/                   # Modern Next.js 16 Web Application
│   ├── app/
│   │   ├── page.js             # Scanner, Model Intelligence, and Anti-Scam Guide
│   │   ├── globals.css         # Glassmorphic dark UI design system
│   │   └── layout.js           # Next.js Root Layout
│   └── next.config.mjs         # API proxy rewrites to Flask backend
├── tests/
│   ├── test_preprocess.py      # Unit tests for text cleaning and transforms
│   └── test_predict.py         # Unit tests for prediction inference & thresholds
├── verify_predictions.py       # Quick end-to-end command-line verification script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **Python 3.9+**
- **Node.js 18+** & npm

### 2. Environment Setup

```bash
# Clone and navigate to the project directory
cd job-scam-detection

# Install Python dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

---

## 🔬 Running the Application

### Option A: Next.js + Flask Fullstack App (Recommended)

1. **Start the ML Backend (Port 5000):**
   ```bash
   python app/server.py
   ```
   *Loads the model pipeline at startup (`http://127.0.0.1:5000`).*

2. **Start the Next.js Frontend (Port 3000):**
   ```bash
   cd frontend
   npm run dev
   ```
   *(On Windows PowerShell if script execution is restricted, run: `cmd /c "npm run dev"` or `npm.cmd run dev`)*

3. **Open the Dashboard:**
   Visit **[http://localhost:3000](http://localhost:3000)** in your browser.

---

### Option B: Quick CLI Verification

Test the pipeline on sample postings directly from your terminal:
```bash
python verify_predictions.py
```

### Option C: Run Unit Tests

```bash
python -m pytest tests/ -v
```

### Option D: Legacy Streamlit UI

```bash
python -m streamlit run app/app.py
```

---

## 📊 Benchmark Results

All models were evaluated on an unseen held-out test set using threshold tuning on the validation set to optimize for **Fraud Class F1-Score** (addressing the ~5% positive class imbalance).

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Optimal Threshold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression + SMOTE (Selected)** | **0.8000** | **0.8923** | **0.8436** | **0.9914** | **0.9254** | **0.5263** |
| LightGBM | 0.9600 | 0.7385 | 0.8348 | 0.9891 | 0.9088 | 0.0976 |
| Logistic Regression | 0.8346 | 0.8154 | 0.8249 | 0.9910 | 0.9171 | 0.6638 |
| LightGBM + SMOTE | 0.8649 | 0.7385 | 0.7967 | 0.9899 | 0.8997 | 0.3784 |
| Random Forest | 0.9184 | 0.6923 | 0.7895 | 0.9880 | 0.8895 | 0.4500 |
| Multinomial Naive Bayes | 0.6061 | 0.3077 | 0.4082 | 0.9203 | 0.4476 | 0.6081 |
| Linear SVM | 0.3478 | 0.3077 | 0.3265 | 0.7053 | 0.2747 | 0.1388 |

### 5-Fold Stratified Cross-Validation (Selected Model)
- **Precision:** `0.8172 ± 0.0205`
- **Recall:** `0.7937 ± 0.0187`
- **F1 Score:** `0.8050 ± 0.0132`
- **ROC-AUC:** `0.9810 ± 0.0037`

*Detailed per-model evaluation plots and metrics are located in [reports/figures/](reports/figures/) and [reports/metrics.json](reports/metrics.json).*

---

## 🧠 How It Works

### 1. Text Preprocessing & Normalization
- Strip HTML markup, emails, phone numbers, and web URLs into canonical tokens.
- Lowercasing, punctuation normalization, and English stopword filtering.
- Stratified train/val/test splitting preserving class ratios.

### 2. Feature Engineering
Combines sublinear **TF-IDF n-grams (1-2)** across job fields with **11 engineered indicators**:
- **Metadata flags:** `telecommuting`, `has_company_logo`, `has_questions`.
- **Text heuristics:** Description word count, character count, ALL-CAPS word ratio.
- **Red-flag patterns:** Urgency triggers (*"act now"*, *"wire transfer"*, *"no experience"*), suspicious email domains (e.g. Gmail/Yahoo/freemail for enterprise roles), contact info in body text, and salary presence.

### 3. Explainability & Risk Highlighting
For every prediction, the engine generates an evidence-based breakdown:
- Classification verdict: **Legit** (<0.35), **Suspicious** (0.35 - threshold), or **Likely Scam** (≥ threshold).
- Transparent red-flag badges highlighting missing verification signals or high-risk language.

---

## 🔌 REST API Reference

The Flask backend exposes the following endpoints (proxied via `/api/*` in Next.js):

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/api/health` | `GET` | Health check, model status, and threshold parameters |
| `/api/examples` | `GET` | Curated sample job postings (`legit`, `suspicious`, `scam`) |
| `/api/metrics` | `GET` | Complete benchmark statistics from `metrics.json` |
| `/api/predict` | `POST` | Inference endpoint for evaluating single job postings |

#### Example Prediction Request:
```bash
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Data Entry Specialist",
    "description": "Earn $5000 weekly from home. Contact hr@freemail.biz immediately.",
    "has_company_logo": 0,
    "has_questions": 0
  }'
```

#### Example Prediction Response:
```json
{
  "status": "success",
  "verdict": "Likely Scam",
  "probability": 0.9421,
  "reasons": [
    "No company logo provided",
    "No screening questions",
    "Contains urgency or fast-cash keywords",
    "Direct contact info in description"
  ],
  "threshold": 0.5263
}
```

---

## ⚖️ Limitations & Ethical Considerations

- **Screening Aid:** This tool is designed to assist job seekers and platforms in triage; it should serve as an advisory screening tool rather than an automated decision maker.
- **False Positives:** Legitimate small businesses or startups that lack formal logos or screening questions may receive cautious ratings.
- **Dataset Context:** Trained on the EMSCAD (CC0) dataset. Evolving scams, newer phishing schemes, and regional language variations may require periodic retraining.
- **Privacy:** No user submissions or personal job applications are stored or logged.

---

## 📜 License & Acknowledgments

- **Dataset:** [EMSCAD / Real or Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (CC0).
- **License:** Educational and research purposes.
