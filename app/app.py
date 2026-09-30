"""
Streamlit app for Job Scam Detection.

Launch with::

    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so ``from src.…`` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.predict import load_pipeline, predict_single

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🛡️",
    layout="wide",
)

# ------------------------------------------------------------------
# Custom CSS for premium look
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* Header */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 50%, #ff6fd8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.03em;
}
.hero-sub {
    text-align: center;
    color: #a0a0c0;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Verdict badges */
.verdict-legit {
    background: linear-gradient(135deg, #00e676, #00c853);
    color: #1b5e20;
    font-size: 1.8rem;
    font-weight: 700;
    padding: 1rem 2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 230, 118, 0.35);
    animation: fadeIn 0.5s ease;
}
.verdict-suspicious {
    background: linear-gradient(135deg, #ffa726, #ff9800);
    color: #4e342e;
    font-size: 1.8rem;
    font-weight: 700;
    padding: 1rem 2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(255, 167, 38, 0.35);
    animation: fadeIn 0.5s ease;
}
.verdict-scam {
    background: linear-gradient(135deg, #ff1744, #d50000);
    color: #fff;
    font-size: 1.8rem;
    font-weight: 700;
    padding: 1rem 2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(255, 23, 68, 0.35);
    animation: fadeIn 0.5s ease;
}

/* Red-flag chips */
.flag-chip {
    display: inline-block;
    background: rgba(255, 23, 68, 0.12);
    border: 1px solid rgba(255, 23, 68, 0.3);
    color: #ff6e7f;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    margin: 0.25rem 0.3rem;
    font-size: 0.9rem;
    font-weight: 500;
    transition: transform 0.2s;
}
.flag-chip:hover {
    transform: scale(1.05);
}

/* Probability bar */
.prob-bar-bg {
    width: 100%;
    height: 28px;
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    overflow: hidden;
    margin: 0.5rem 0 1rem 0;
    border: 1px solid rgba(255,255,255,0.08);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 14px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #fff;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* Section headers */
.section-label {
    color: #9e9ec0;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}

/* Example buttons */
div.stButton > button {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
    color: #c0c0e0;
    font-weight: 500;
    transition: all 0.3s;
}
div.stButton > button:hover {
    background: rgba(123, 47, 247, 0.2);
    border-color: #7b2ff7;
    color: #fff;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load model (cached)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def get_pipeline():
    return load_pipeline()


try:
    model, fb, config = get_pipeline()
    model_loaded = True
except Exception as exc:
    model_loaded = False
    model_load_error = str(exc)


# ------------------------------------------------------------------
# Example postings
# ------------------------------------------------------------------

EXAMPLE_LEGIT = {
    "title": "Senior Data Engineer — Remote",
    "company_profile": (
        "TechCorp is a Series-C startup building the next-generation data "
        "platform. We have 500+ employees across 12 countries."
    ),
    "description": (
        "We are looking for a Senior Data Engineer to design, build, and "
        "maintain our data lake on AWS. You will work closely with the ML "
        "team to build robust ETL pipelines and ensure data quality across "
        "the organisation. Our stack includes Spark, Airflow, dbt, and "
        "Snowflake."
    ),
    "requirements": (
        "5+ years of experience with data engineering. "
        "Proficiency in Python and SQL. "
        "Experience with cloud data warehouses (Snowflake, BigQuery, or Redshift). "
        "BS in Computer Science or equivalent."
    ),
    "benefits": (
        "Competitive salary, equity, health/dental/vision insurance, "
        "401(k) match, unlimited PTO, annual learning stipend."
    ),
    "salary_range": "160000-210000",
    "location": "US — Remote",
    "employment_type": "Full-time",
    "telecommuting": 1,
    "has_company_logo": 1,
    "has_questions": 1,
}

EXAMPLE_SCAM = {
    "title": "EARN $5000/WEEK — NO EXPERIENCE NEEDED!",
    "company_profile": "",
    "description": (
        "MAKE MONEY FROM HOME IMMEDIATELY! We are hiring data entry clerks. "
        "NO SKILLS REQUIRED. Just send your resume and bank details to "
        "hr@quick-cash-jobs.xyz or contact us on WhatsApp +1-555-000-1234. "
        "URGENT — only 3 positions left! ACT NOW before it's too late. "
        "This is a once-in-a-lifetime opportunity!"
    ),
    "requirements": "",
    "benefits": "",
    "salary_range": "",
    "location": "",
    "employment_type": "",
    "telecommuting": 0,
    "has_company_logo": 0,
    "has_questions": 0,
}


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.markdown('<div class="hero-title">🛡️ Job Scam Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Paste a job posting and instantly find out if it\'s legitimate, '
    'suspicious, or a likely scam — powered by machine learning.</div>',
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(f"⚠️ Could not load model: {model_load_error}")
    st.info("Run `python -m src.train` first to train and save the model.")
    st.stop()

# --- Example buttons ---
col_ex1, col_ex2, _ = st.columns([1, 1, 3])
with col_ex1:
    use_legit = st.button("📋 Load legit example")
with col_ex2:
    use_scam = st.button("🚨 Load scam example")

if use_legit:
    st.session_state["example"] = EXAMPLE_LEGIT
if use_scam:
    st.session_state["example"] = EXAMPLE_SCAM

ex = st.session_state.get("example", {})

# --- Input form ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Job Posting Details</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Job title", value=ex.get("title", ""), key="inp_title")
    location = st.text_input("Location", value=ex.get("location", ""), key="inp_loc")
    employment_type = st.selectbox(
        "Employment type",
        ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"],
        index=(["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"]
               .index(ex.get("employment_type", "")) if ex.get("employment_type", "") in
               ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"] else 0),
        key="inp_etype",
    )
    salary_range = st.text_input("Salary range", value=ex.get("salary_range", ""), key="inp_sal")

with col2:
    has_logo = st.checkbox("Company has logo", value=bool(ex.get("has_company_logo", True)), key="inp_logo")
    has_qs = st.checkbox("Has screening questions", value=bool(ex.get("has_questions", False)), key="inp_qs")
    telecommute = st.checkbox("Telecommuting / remote", value=bool(ex.get("telecommuting", False)), key="inp_tc")

company_profile = st.text_area("Company profile", value=ex.get("company_profile", ""),
                               height=80, key="inp_cp")
description = st.text_area("Job description", value=ex.get("description", ""),
                           height=150, key="inp_desc")
requirements = st.text_area("Requirements", value=ex.get("requirements", ""),
                            height=80, key="inp_req")
benefits = st.text_area("Benefits", value=ex.get("benefits", ""),
                        height=80, key="inp_ben")

st.markdown('</div>', unsafe_allow_html=True)

# --- Analyse button ---
analyse = st.button("🔍  Analyse Posting", type="primary", use_container_width=True)

if analyse:
    if not description.strip() and not title.strip():
        st.warning("Please enter at least a **title** or **description**.")
        st.stop()

    with st.spinner("Analysing …"):
        result = predict_single(
            title=title,
            company_profile=company_profile,
            description=description,
            requirements=requirements,
            benefits=benefits,
            salary_range=salary_range,
            location=location,
            employment_type=employment_type,
            telecommuting=int(telecommute),
            has_company_logo=int(has_logo),
            has_questions=int(has_qs),
            model=model, fb=fb, config=config,
        )

    verdict = result["verdict"]
    proba = result["probability"]
    reasons = result["reasons"]

    st.markdown("---")

    # Verdict
    css_class = {
        "Legit": "verdict-legit",
        "Suspicious": "verdict-suspicious",
        "Likely Scam": "verdict-scam",
    }[verdict]
    icon = {"Legit": "✅", "Suspicious": "⚠️", "Likely Scam": "🚫"}[verdict]
    st.markdown(
        f'<div class="{css_class}">{icon} {verdict}</div>',
        unsafe_allow_html=True,
    )

    # Probability bar
    pct = int(proba * 100)
    if proba < 0.3:
        bar_color = "linear-gradient(90deg, #00e676, #69f0ae)"
    elif proba < 0.6:
        bar_color = "linear-gradient(90deg, #ffa726, #ff9800)"
    else:
        bar_color = "linear-gradient(90deg, #ff5252, #ff1744)"

    st.markdown(f"""
    <div style="margin-top:1.2rem;">
        <p class="section-label">Fraud Probability</p>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{max(pct,4)}%; background:{bar_color};">
                {pct}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Red flags
    if reasons:
        st.markdown('<p class="section-label" style="margin-top:1rem;">Red Flags Detected</p>',
                    unsafe_allow_html=True)
        chips = " ".join(f'<span class="flag-chip">⚑ {r}</span>' for r in reasons)
        st.markdown(f'<div style="margin-bottom:1rem;">{chips}</div>', unsafe_allow_html=True)
    elif verdict == "Legit":
        st.markdown(
            '<p style="color:#69f0ae; margin-top:1rem;">No red flags detected — '
            'this posting looks legitimate.</p>',
            unsafe_allow_html=True,
        )

# --- Footer ---
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#606080;font-size:0.8rem;">'
    '⚖️ <b>Disclaimer</b>: This tool is for informational purposes only. '
    'False positives may occur and can harm legitimate employers. '
    'Always verify job postings through official channels.</p>',
    unsafe_allow_html=True,
)
