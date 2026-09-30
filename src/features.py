"""
Feature engineering for job scam detection.

Two kinds of features are produced:

1. **Engineered features** — hand-crafted signals derived from raw columns
   (``has_company_logo``, ``urgency_word_count``, ``allcaps_ratio``, etc.).
2. **TF-IDF features** — unigram + bigram counts on the cleaned combined text.

The :class:`FeatureBuilder` wraps a ``TfidfVectorizer`` and a
``TruncatedSVD`` so that tree-based models receive a dense 300-d
representation of text plus the engineered columns.
"""

import re
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------------
# Urgency phrase list
# ------------------------------------------------------------------

URGENCY_PHRASES: List[str] = [
    "immediately", "no experience needed", "no experience required",
    "apply now", "urgent", "urgently", "hurry", "limited time",
    "act now", "asap", "right away", "start today",
    "no skills required", "no qualifications needed",
]

# ------------------------------------------------------------------
# Per-row feature extractors
# ------------------------------------------------------------------


def _has_salary(row: pd.Series) -> int:
    val = row.get("salary_range", None)
    if pd.isna(val) or str(val).strip() in ("", "unknown"):
        return 0
    return 1


def _contains_email(text: str) -> int:
    return int(bool(re.search(r"\S+@\S+\.\S+", str(text))))


def _contains_phone(text: str) -> int:
    return int(bool(re.search(r"[\+]?[\d\-\(\)\s]{7,15}", str(text))))


def _contains_whatsapp_telegram(text: str) -> int:
    t = str(text).lower()
    return int("whatsapp" in t or "telegram" in t)


def _urgency_word_count(text: str) -> int:
    t = str(text).lower()
    return sum(1 for phrase in URGENCY_PHRASES if phrase in t)


def _allcaps_ratio(text: str) -> float:
    words = str(text).split()
    if not words:
        return 0.0
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)


def _description_length(text: str) -> int:
    return len(str(text))


def _word_count(text: str) -> int:
    return len(str(text).split())


# ------------------------------------------------------------------
# Public constants
# ------------------------------------------------------------------

ENGINEERED_FEATURE_NAMES: List[str] = [
    "has_company_logo",
    "has_salary",
    "has_questions",
    "telecommuting",
    "contains_email",
    "contains_phone",
    "contains_whatsapp_telegram",
    "urgency_word_count",
    "allcaps_ratio",
    "description_length",
    "word_count",
]


# ------------------------------------------------------------------
# DataFrame-level feature engineering
# ------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered feature columns to *df* (returns a copy).

    Expects ``handle_missing_values`` to have been called first so that
    binary-flag columns already exist.
    """
    df = df.copy()

    # Binary flags already in dataset
    for col in ("has_company_logo", "has_questions", "telecommuting"):
        df[col] = df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Derived from raw description (before cleaning)
    raw_desc = df.get("description", pd.Series("", index=df.index)).fillna("")

    df["has_salary"]                  = df.apply(_has_salary, axis=1)
    df["contains_email"]              = raw_desc.apply(_contains_email)
    df["contains_phone"]              = raw_desc.apply(_contains_phone)
    df["contains_whatsapp_telegram"]  = raw_desc.apply(_contains_whatsapp_telegram)
    df["urgency_word_count"]          = raw_desc.apply(_urgency_word_count)
    df["allcaps_ratio"]               = raw_desc.apply(_allcaps_ratio)
    df["description_length"]          = raw_desc.apply(_description_length)
    df["word_count"]                  = raw_desc.apply(_word_count)

    return df


# ------------------------------------------------------------------
# FeatureBuilder — wraps TF-IDF + SVD
# ------------------------------------------------------------------


class FeatureBuilder:
    """
    Builds the combined feature matrix used by all models.

    * **Sparse output** (``transform_sparse``): raw TF-IDF + engineered
      columns — used by linear models (LR, SVM).
    * **Dense output** (``transform_dense``): TruncatedSVD on TF-IDF +
      engineered columns — used by tree models (RF, LightGBM).
    """

    def __init__(
        self,
        max_tfidf_features: int = 20_000,
        svd_components: int = 300,
        random_state: int = 42,
    ) -> None:
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.svd = TruncatedSVD(
            n_components=svd_components, random_state=random_state,
        )
        self.max_tfidf_features = max_tfidf_features
        self.svd_components = svd_components
        self.random_state = random_state
        self._is_fitted = False

    # ---------- fit / transform ----------

    def fit(
        self, text_series: pd.Series, eng_df: pd.DataFrame,
    ) -> "FeatureBuilder":
        """Fit TF-IDF vectorizer and SVD on **training data only**."""
        self.tfidf.fit(text_series)
        tfidf_matrix = self.tfidf.transform(text_series)
        self.svd.fit(tfidf_matrix)
        self._is_fitted = True
        return self

    def transform_sparse(
        self, text_series: pd.Series, eng_df: pd.DataFrame,
    ) -> sp.csr_matrix:
        """TF-IDF (sparse) ∥ engineered features → sparse CSR matrix."""
        tfidf_mat = self.tfidf.transform(text_series)
        eng_mat = sp.csr_matrix(
            eng_df[ENGINEERED_FEATURE_NAMES].values.astype(np.float32),
        )
        return sp.hstack([tfidf_mat, eng_mat], format="csr")

    def transform_dense(
        self, text_series: pd.Series, eng_df: pd.DataFrame,
    ) -> np.ndarray:
        """SVD(TF-IDF) ∥ engineered features → dense numpy array."""
        tfidf_mat = self.tfidf.transform(text_series)
        svd_mat = self.svd.transform(tfidf_mat)
        eng_mat = eng_df[ENGINEERED_FEATURE_NAMES].values.astype(np.float32)
        return np.hstack([svd_mat, eng_mat])

    def transform_tfidf_only(self, text_series: pd.Series) -> sp.csr_matrix:
        """Raw TF-IDF matrix (for baseline models that only use text)."""
        return self.tfidf.transform(text_series)

    # ---------- helpers ----------

    def get_feature_names_dense(self) -> List[str]:
        """Column names for the dense matrix."""
        svd_names = [f"svd_{i}" for i in range(self.svd_components)]
        return svd_names + list(ENGINEERED_FEATURE_NAMES)
