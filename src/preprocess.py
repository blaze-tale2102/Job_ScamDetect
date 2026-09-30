"""
Text preprocessing and data-splitting utilities for job scam detection.

Provides a deterministic pipeline:
    raw text → HTML unescape → strip tags → lowercase → remove URLs/emails/
    phone numbers → remove punctuation → remove stopwords → normalise whitespace.
"""

import html
import re
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# NLTK stopwords — loaded lazily so the import is fast
# ---------------------------------------------------------------------------
_STOPWORDS: Optional[Set[str]] = None


def _get_stopwords() -> Set[str]:
    """Return the NLTK English stopword set, downloading if necessary."""
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        try:
            from nltk.corpus import stopwords as _sw
            _STOPWORDS = set(_sw.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords as _sw
            _STOPWORDS = set(_sw.words("english"))
    return _STOPWORDS


# ---------------------------------------------------------------------------
# Individual cleaning steps (exported so tests can exercise them)
# ---------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Remove HTML tags and return plain text."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www. links."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_phone_numbers(text: str) -> str:
    """Remove sequences that look like phone numbers."""
    return re.sub(r"[\+]?[\d\s\-\(\)]{7,15}", " ", text)


def remove_punctuation(text: str) -> str:
    """Keep only alphanumeric characters and whitespace."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def remove_stopwords(text: str) -> str:
    """Drop English stopwords (space-separated tokens)."""
    stops = _get_stopwords()
    return " ".join(w for w in text.split() if w not in stops)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Full text-cleaning pipeline.

    Steps
    -----
    1. HTML-unescape entities (``&amp;`` → ``&``)
    2. Strip HTML tags
    3. Lower-case
    4. Remove URLs
    5. Remove email addresses
    6. Remove phone-number-like sequences
    7. Remove punctuation
    8. Remove stopwords
    9. Normalise whitespace

    Returns an empty string for null / whitespace-only input.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)
    text = strip_html(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# DataFrame-level helpers
# ---------------------------------------------------------------------------

TEXT_COLUMNS = [
    "title", "company_profile", "description", "requirements", "benefits",
]


def merge_text_fields(df: pd.DataFrame) -> pd.Series:
    """Concatenate the five main text columns into one ``combined_text`` field."""
    return df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in-place (on a copy):

    * Text columns          → ``""``
    * Categorical columns   → ``"unknown"``
    * Binary flag columns   → ``0``
    """
    df = df.copy()

    # Text
    text_cols = [c for c in TEXT_COLUMNS if c in df.columns]
    df[text_cols] = df[text_cols].fillna("")

    # Categorical
    cat_cols = [
        "location", "department", "employment_type",
        "required_experience", "required_education",
        "industry", "function",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")

    # Binary flags
    for c in ("telecommuting", "has_company_logo", "has_questions"):
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # Salary
    if "salary_range" in df.columns:
        df["salary_range"] = df["salary_range"].fillna("")

    return df


def create_splits(
    df: pd.DataFrame,
    target_col: str = "fraudulent",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split (70 / 15 / 15).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[target_col],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[target_col],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
