"""
Prediction interface for job scam detection.

Loads the saved model + feature-builder once, then exposes
:func:`predict_single` for the Streamlit app and CLI use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features import (
    ENGINEERED_FEATURE_NAMES,
    URGENCY_PHRASES,
    FeatureBuilder,
    engineer_features,
)
from src.preprocess import clean_text, handle_missing_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_pipeline(
    models_dir: Optional[Path] = None,
) -> Tuple[Any, FeatureBuilder, dict]:
    """
    Load the saved model, feature builder, and config dict.

    Returns
    -------
    model, feature_builder, config
    """
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"
    model = joblib.load(models_dir / "final_model.joblib")
    fb: FeatureBuilder = joblib.load(models_dir / "feature_builder.joblib")
    with open(models_dir / "model_config.json") as f:
        config: dict = json.load(f)
    return model, fb, config


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict_single(
    *,
    title: str = "",
    company_profile: str = "",
    description: str = "",
    requirements: str = "",
    benefits: str = "",
    salary_range: str = "",
    location: str = "",
    employment_type: str = "",
    telecommuting: int = 0,
    has_company_logo: int = 1,
    has_questions: int = 0,
    model: Any = None,
    fb: Optional[FeatureBuilder] = None,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Predict whether a single job posting is a scam.

    Returns
    -------
    dict with keys ``verdict``, ``probability``, ``threshold``, ``reasons``.
    """
    if model is None or fb is None or config is None:
        model, fb, config = load_pipeline()

    # Build a one-row DataFrame
    row = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "salary_range": salary_range if salary_range else np.nan,
        "location": location,
        "employment_type": employment_type,
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
    }
    df = pd.DataFrame([row])
    df = handle_missing_values(df)
    df = engineer_features(df)

    combined = f"{title} {company_profile} {description} {requirements} {benefits}"
    clean = clean_text(combined)
    clean_series = pd.Series([clean])

    # Build features matching the model's expected input
    ftype = config.get("feature_type", "dense")
    if ftype == "dense":
        X = fb.transform_dense(clean_series, df)
    elif ftype == "sparse":
        X = fb.transform_sparse(clean_series, df)
    else:
        X = fb.transform_tfidf_only(clean_series)

    # Predict probability
    threshold = config.get("best_threshold", 0.5)
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    else:
        proba = float(model.decision_function(X)[0])

    # Three-tier verdict
    if proba >= 0.6:
        verdict = "Likely Scam"
    elif proba >= 0.3:
        verdict = "Suspicious"
    else:
        verdict = "Legit"

    reasons = _extract_red_flags(df.iloc[0], description, proba)

    return {
        "verdict": verdict,
        "probability": proba,
        "threshold": threshold,
        "reasons": reasons,
    }


# ------------------------------------------------------------------
# Red-flag extraction
# ------------------------------------------------------------------

def _extract_red_flags(
    row: pd.Series, raw_description: str, proba: float,
) -> List[str]:
    """Return a list of human-readable red-flag strings."""
    flags: List[str] = []

    if row.get("has_company_logo", 1) == 0:
        flags.append("No company logo provided")
    if row.get("has_salary", 1) == 0:
        flags.append("No salary range specified")
    if row.get("has_questions", 0) == 0:
        flags.append("No screening questions")
    if row.get("contains_email", 0) == 1:
        flags.append("Email address found in description")
    if row.get("contains_phone", 0) == 1:
        flags.append("Phone number found in description")
    if row.get("contains_whatsapp_telegram", 0) == 1:
        flags.append("WhatsApp / Telegram mentioned in description")
    if row.get("urgency_word_count", 0) > 0:
        desc_lower = raw_description.lower()
        matched = [p for p in URGENCY_PHRASES if p in desc_lower]
        if matched:
            flags.append(f"Urgency phrases: {', '.join(matched[:3])}")
    if row.get("allcaps_ratio", 0) > 0.15:
        flags.append(f"High ALL-CAPS ratio ({row['allcaps_ratio']:.0%})")
    if row.get("description_length", 0) < 100:
        flags.append("Very short job description")
    if row.get("telecommuting", 0) == 1:
        flags.append("Telecommuting position (slight risk factor)")

    if not flags and proba > 0.3:
        flags.append("Text patterns match known scam postings")

    return flags
