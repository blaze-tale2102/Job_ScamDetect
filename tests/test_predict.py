from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "final_model.joblib").exists(),
    reason="Trained model not found",
)

from src.predict import load_pipeline, predict_single


@pytest.fixture(scope="module")
def pipeline():
    model, fb, config = load_pipeline(MODELS_DIR)
    return model, fb, config


@pytest.fixture(scope="module")
def test_dataset():
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    if not test_csv.exists():
        pytest.skip("test data missing")
    return pd.read_csv(test_csv)


class TestModelTrainingAndInference:
    def test_pipeline_components_loaded(self, pipeline):
        model, fb, config = pipeline
        assert hasattr(model, "predict_proba")
        assert hasattr(fb, "tfidf")
        assert len(fb.tfidf.vocabulary_) > 1000
        assert "best_threshold" in config

    def test_returns_expected_schema(self, pipeline, test_dataset):
        model, fb, config = pipeline
        sample_row = test_dataset.iloc[0]

        result = predict_single(
            title=str(sample_row.get("title", "")),
            company_profile=str(sample_row.get("company_profile", "")),
            description=str(sample_row.get("description", "")),
            requirements=str(sample_row.get("requirements", "")),
            benefits=str(sample_row.get("benefits", "")),
            has_company_logo=int(sample_row.get("has_company_logo", 1)),
            has_questions=int(sample_row.get("has_questions", 0)),
            model=model,
            fb=fb,
            config=config,
        )

        assert "verdict" in result
        assert "probability" in result
        assert "threshold" in result
        assert "reasons" in result
        assert 0.0 <= result["probability"] <= 1.0
        assert result["verdict"] in ("Legit", "Suspicious", "Likely Scam")
        assert isinstance(result["reasons"], list)

    def test_real_legitimate_postings_from_dataset(self, pipeline, test_dataset):
        model, fb, config = pipeline
        legit_rows = test_dataset[test_dataset["fraudulent"] == 0].head(10)

        probabilities = []
        for _, row in legit_rows.iterrows():
            res = predict_single(
                title=str(row.get("title", "")),
                company_profile=str(row.get("company_profile", "")),
                description=str(row.get("description", "")),
                requirements=str(row.get("requirements", "")),
                benefits=str(row.get("benefits", "")),
                has_company_logo=int(row.get("has_company_logo", 1)),
                has_questions=int(row.get("has_questions", 0)),
                model=model,
                fb=fb,
                config=config,
            )
            probabilities.append(res["probability"])

        avg_legit_prob = np.mean(probabilities)
        assert avg_legit_prob < 0.30

    def test_real_fraudulent_postings_from_dataset(self, pipeline, test_dataset):
        model, fb, config = pipeline
        scam_rows = test_dataset[test_dataset["fraudulent"] == 1].head(10)

        probabilities = []
        for _, row in scam_rows.iterrows():
            res = predict_single(
                title=str(row.get("title", "")),
                company_profile=str(row.get("company_profile", "")),
                description=str(row.get("description", "")),
                requirements=str(row.get("requirements", "")),
                benefits=str(row.get("benefits", "")),
                has_company_logo=int(row.get("has_company_logo", 0)),
                has_questions=int(row.get("has_questions", 0)),
                model=model,
                fb=fb,
                config=config,
            )
            probabilities.append(res["probability"])

        avg_scam_prob = np.mean(probabilities)
        assert avg_scam_prob > 0.50

    def test_statistical_separation_between_classes(self, pipeline, test_dataset):
        model, fb, config = pipeline
        legits = test_dataset[test_dataset["fraudulent"] == 0].sample(25, random_state=42)
        scams = test_dataset[test_dataset["fraudulent"] == 1].sample(25, random_state=42)

        def get_prob(row):
            return predict_single(
                title=str(row.get("title", "")),
                company_profile=str(row.get("company_profile", "")),
                description=str(row.get("description", "")),
                requirements=str(row.get("requirements", "")),
                benefits=str(row.get("benefits", "")),
                has_company_logo=int(row.get("has_company_logo", 1)),
                has_questions=int(row.get("has_questions", 0)),
                model=model,
                fb=fb,
                config=config,
            )["probability"]

        legit_mean = np.mean([get_prob(r) for _, r in legits.iterrows()])
        scam_mean = np.mean([get_prob(r) for _, r in scams.iterrows()])
        margin = scam_mean - legit_mean
        assert margin >= 0.40
