"""
Unit tests for ``src.predict``.

These tests require a trained model to exist in ``models/``.
If no model is found the tests are skipped.
"""

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Skip the entire module if no model has been trained yet.
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "final_model.joblib").exists(),
    reason="Trained model not found — run `python -m src.train` first.",
)

from src.predict import load_pipeline, predict_single  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    model, fb, config = load_pipeline(MODELS_DIR)
    return model, fb, config


class TestPredictSingle:
    def test_returns_expected_keys(self, pipeline):
        model, fb, config = pipeline
        result = predict_single(
            title="Software Engineer",
            description="We are looking for a talented engineer.",
            model=model, fb=fb, config=config,
        )
        assert "verdict" in result
        assert "probability" in result
        assert "threshold" in result
        assert "reasons" in result

    def test_probability_in_range(self, pipeline):
        model, fb, config = pipeline
        result = predict_single(
            title="Data Scientist",
            description="Join our analytics team.",
            model=model, fb=fb, config=config,
        )
        assert 0.0 <= result["probability"] <= 1.0

    def test_verdict_values(self, pipeline):
        model, fb, config = pipeline
        result = predict_single(
            title="Test", description="Test",
            model=model, fb=fb, config=config,
        )
        assert result["verdict"] in ("Legit", "Suspicious", "Likely Scam")

    def test_legit_posting_detected(self, pipeline):
        """A realistic, well-formed posting should not be flagged as a scam."""
        model, fb, config = pipeline
        result = predict_single(
            title="Senior Software Engineer",
            company_profile="Google LLC is a multinational technology company.",
            description=(
                "We are seeking a Senior Software Engineer to join our "
                "Cloud Platform team. You will design and implement scalable "
                "distributed systems. Requirements: BS in CS, 5+ years of "
                "experience with Python, Java, or Go."
            ),
            requirements="BS in Computer Science, 5+ years experience",
            benefits="Health insurance, 401k, stock options",
            salary_range="150000-200000",
            has_company_logo=1,
            has_questions=1,
            model=model, fb=fb, config=config,
        )
        # We don't assert == "Legit" (model may not be perfect) but
        # probability should be low.
        assert result["probability"] < 0.7

    def test_scam_posting_detected(self, pipeline):
        """A clearly scammy posting should score high."""
        model, fb, config = pipeline
        result = predict_single(
            title="EARN $5000 PER WEEK FROM HOME",
            description=(
                "NO EXPERIENCE NEEDED! APPLY NOW! Send your resume to "
                "hiring@scam-jobs.xyz or WhatsApp +1234567890. "
                "URGENT — limited positions! Act now!"
            ),
            has_company_logo=0,
            has_questions=0,
            model=model, fb=fb, config=config,
        )
        # Should be flagged — at least "Suspicious"
        assert result["verdict"] in ("Suspicious", "Likely Scam")
        assert len(result["reasons"]) >= 1

    def test_reasons_is_list(self, pipeline):
        model, fb, config = pipeline
        result = predict_single(
            title="Test", description="Test",
            model=model, fb=fb, config=config,
        )
        assert isinstance(result["reasons"], list)
