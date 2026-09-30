"""
Exploratory Data Analysis for the Fake Job Postings dataset.

Generates charts in ``reports/figures/`` and prints key statistics.

Usage::

    python notebooks/01_eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402
from wordcloud import WordCloud  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT_ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def main() -> None:
    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    csv_path = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    # ------------------------------------------------------------------
    # 1. Class distribution
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["fraudulent"].value_counts().sort_index()
    bars = ax.bar(["Legit (0)", "Fraud (1)"], counts.values,
                  color=["#4CAF50", "#E53935"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 100,
                f"{int(b.get_height()):,}", ha="center", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\n[OK] Class distribution saved")
    print(f"  Legit: {counts[0]:,}  |  Fraud: {counts[1]:,}  "
          f"({counts[1]/len(df):.2%})")

    # ------------------------------------------------------------------
    # 2. Missing values
    # ------------------------------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(missing.index, missing.values, color="#1976D2")
    ax.set_xlabel("Missing count")
    ax.set_title("Missing Values per Column")
    fig.tight_layout()
    fig.savefig(FIGURES / "missing_values.png", dpi=150)
    plt.close(fig)
    print("[OK] Missing-value chart saved")

    # ------------------------------------------------------------------
    # 3. Text-length distributions by class
    # ------------------------------------------------------------------
    for col in ("description", "requirements", "company_profile"):
        if col not in df.columns:
            continue
        df[f"{col}_len"] = df[col].fillna("").str.len()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ("description", "requirements", "company_profile")):
        len_col = f"{col}_len"
        if len_col not in df.columns:
            continue
        for label, color in [(0, "#4CAF50"), (1, "#E53935")]:
            subset = df[df["fraudulent"] == label][len_col]
            ax.hist(subset, bins=60, alpha=0.6, color=color,
                    label="Legit" if label == 0 else "Fraud")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Character length")
        ax.legend()
    fig.suptitle("Text Length Distributions by Class", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "text_length_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Text-length distributions saved")

    # ------------------------------------------------------------------
    # 4. Top words in fake vs. real (bar charts + word clouds)
    # ------------------------------------------------------------------
    from sklearn.feature_extraction.text import CountVectorizer

    for label, tag in [(0, "legit"), (1, "fraud")]:
        texts = df[df["fraudulent"] == label]["description"].fillna("").values
        vec = CountVectorizer(max_features=30, stop_words="english",
                              ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        freqs = dict(zip(vec.get_feature_names_out(),
                         X.toarray().sum(axis=0)))

        # Bar chart
        sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        words, counts_w = zip(*sorted_words)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(words[::-1], counts_w[::-1],
                color="#E53935" if tag == "fraud" else "#4CAF50")
        ax.set_title(f"Top 30 Terms — {tag.upper()} postings")
        ax.set_xlabel("Frequency")
        fig.tight_layout()
        fig.savefig(FIGURES / f"top_words_{tag}.png", dpi=150)
        plt.close(fig)

        # Word cloud
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap="Reds" if tag == "fraud" else "Greens")
        wc.generate_from_frequencies(freqs)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {tag.upper()}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"wordcloud_{tag}.png", dpi=150)
        plt.close(fig)

    print("[OK] Top-word charts and word clouds saved")

    # ------------------------------------------------------------------
    # 5. Correlation of binary flags with fraud label
    # ------------------------------------------------------------------
    flag_cols = ["telecommuting", "has_company_logo", "has_questions",
                 "fraudulent"]
    flag_cols = [c for c in flag_cols if c in df.columns]
    corr = df[flag_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Binary Feature Correlation with Fraud")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_correlation.png", dpi=150)
    plt.close(fig)
    print("[OK] Correlation heatmap saved")

    print(f"\nAll EDA figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
