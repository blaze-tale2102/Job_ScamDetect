"""
Evaluation utilities — metrics, threshold tuning, and diagnostic plots.

All plotting helpers use the ``Agg`` backend so they work headless.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import seaborn as sns                    # noqa: E402
from sklearn.metrics import (            # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute precision, recall, F1, ROC-AUC, and PR-AUC for the **fraud** class."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "pr_auc":    float(average_precision_score(y_true, y_proba)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """Return the threshold that maximises F1 on the positive (fraud) class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s))
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    """Save a Precision–Recall curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, "b-", linewidth=2,
            label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve — {title}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    """Save a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC-AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_path)
