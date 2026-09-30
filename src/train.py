"""
Model training pipeline for job scam detection.

Usage::

    python -m src.train                       # default path
    python -m src.train --data-path my.csv    # custom CSV
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import lightgbm as lgb

from src.evaluate import (
    evaluate_model,
    find_best_threshold,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.features import FeatureBuilder, engineer_features
from src.preprocess import clean_text, create_splits, handle_missing_values, merge_text_fields

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def load_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV."""
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
    df = pd.read_csv(data_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing + feature engineering pipeline."""
    df = handle_missing_values(df)
    df = engineer_features(df)
    df["combined_text"] = merge_text_fields(df)
    df["clean_text"] = df["combined_text"].apply(clean_text)
    return df


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------

def _make_models(y_train: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Return a dict of ``{name: config}`` where each *config* has:

    * ``model``  — an sklearn-compatible estimator
    * ``ftype``  — ``"tfidf"`` | ``"sparse"`` | ``"dense"``
    * ``smote``  — whether to apply SMOTE before fitting
    """
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    return {
        # ---- baselines (TF-IDF only) ----
        "Logistic Regression": {
            "model": LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE, C=1.0,
            ),
            "ftype": "tfidf",
            "smote": False,
        },
        "Multinomial NB": {
            "model": MultinomialNB(alpha=1.0),
            "ftype": "tfidf",
            "smote": False,
        },
        # ---- stronger (TF-IDF + engineered) ----
        "Linear SVM": {
            "model": CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced", max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
                cv=3,
            ),
            "ftype": "sparse",
            "smote": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "ftype": "dense",
            "smote": False,
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=63,
                scale_pos_weight=pos_weight,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            "ftype": "dense",
            "smote": False,
        },
        # ---- SMOTE variants ----
        "Logistic Regression + SMOTE": {
            "model": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, C=1.0,
            ),
            "ftype": "tfidf",
            "smote": True,
        },
        "LightGBM + SMOTE": {
            "model": lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=63,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            "ftype": "dense",
            "smote": True,
        },
    }


def _get_features(
    ftype: str, fb: FeatureBuilder,
    text: pd.Series, eng_df: pd.DataFrame,
    *,
    tfidf_cache=None, sparse_cache=None, dense_cache=None,
):
    """Select the right feature matrix for a model type."""
    if ftype == "tfidf":
        return tfidf_cache if tfidf_cache is not None else fb.transform_tfidf_only(text)
    if ftype == "sparse":
        return sparse_cache if sparse_cache is not None else fb.transform_sparse(text, eng_df)
    return dense_cache if dense_cache is not None else fb.transform_dense(text, eng_df)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def train_all_models(data_path: str | Path | None = None) -> Dict:
    """Train all models, evaluate, and save the best one."""
    print("=" * 60)
    print("JOB SCAM DETECTION — MODEL TRAINING")
    print("=" * 60)

    # --- 1. Load ---
    print("\n[1/7] Loading data ...")
    df = load_data(data_path)
    print(f"  Shape : {df.shape}")
    print(f"  Fraud : {df['fraudulent'].mean():.2%}")

    # --- 2. Preprocess ---
    print("\n[2/7] Preprocessing ...")
    df = prepare_data(df)

    # --- 3. Split ---
    print("\n[3/7] Splitting (70 / 15 / 15) ...")
    train_df, val_df, test_df = create_splits(df)
    print(f"  Train {len(train_df)}  |  Val {len(val_df)}  |  Test {len(test_df)}")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    # --- 4. Features ---
    print("\n[4/7] Building features ...")
    fb = FeatureBuilder(max_tfidf_features=20_000, svd_components=300,
                        random_state=RANDOM_STATE)
    fb.fit(train_df["clean_text"], train_df)

    # Pre-compute all feature variants
    X_train_tfidf  = fb.transform_tfidf_only(train_df["clean_text"])
    X_val_tfidf    = fb.transform_tfidf_only(val_df["clean_text"])
    X_test_tfidf   = fb.transform_tfidf_only(test_df["clean_text"])

    X_train_sparse = fb.transform_sparse(train_df["clean_text"], train_df)
    X_val_sparse   = fb.transform_sparse(val_df["clean_text"], val_df)
    X_test_sparse  = fb.transform_sparse(test_df["clean_text"], test_df)

    X_train_dense  = fb.transform_dense(train_df["clean_text"], train_df)
    X_val_dense    = fb.transform_dense(val_df["clean_text"], val_df)
    X_test_dense   = fb.transform_dense(test_df["clean_text"], test_df)

    y_train = train_df["fraudulent"].values
    y_val   = val_df["fraudulent"].values
    y_test  = test_df["fraudulent"].values

    cache = {
        "tfidf":  (X_train_tfidf, X_val_tfidf, X_test_tfidf),
        "sparse": (X_train_sparse, X_val_sparse, X_test_sparse),
        "dense":  (X_train_dense, X_val_dense, X_test_dense),
    }

    smote = SMOTE(random_state=RANDOM_STATE)

    # --- 5. Train & evaluate ---
    print("\n[5/7] Training models ...")
    models_cfg = _make_models(y_train)
    results: Dict[str, Dict] = {}
    trained_models: Dict[str, Any] = {}
    best_name: str = ""
    best_f1: float = 0.0

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in models_cfg.items():
        print(f"\n  > {name}")
        model = cfg["model"]
        ftype = cfg["ftype"]
        X_tr, X_va, X_te = cache[ftype]

        if cfg["smote"]:
            X_fit, y_fit = smote.fit_resample(X_tr, y_train)
        else:
            X_fit, y_fit = X_tr, y_train

        model.fit(X_fit, y_fit)
        trained_models[name] = model

        y_val_proba  = model.predict_proba(X_va)[:, 1]
        y_test_proba = model.predict_proba(X_te)[:, 1]

        thresh = find_best_threshold(y_val, y_val_proba)
        metrics = evaluate_model(y_test, y_test_proba, threshold=thresh)
        metrics["best_threshold"] = thresh
        results[name] = metrics

        print(f"    P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}  ROC={metrics['roc_auc']:.4f}  "
              f"PR={metrics['pr_auc']:.4f}  thr={thresh:.3f}")

        if metrics["f1"] > best_f1:
            best_f1, best_name = metrics["f1"], name

        safe = (name.replace(" ", "_").replace("+", "plus")
                    .replace("(", "").replace(")", ""))
        plot_confusion_matrix(
            y_test, (y_test_proba >= thresh).astype(int),
            title=name, save_path=figures_dir / f"cm_{safe}.png",
        )

    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: {best_name}  (F1 = {best_f1:.4f})")
    print(f"{'=' * 60}")

    # --- 6. 5-fold CV on best model ---
    print(f"\n[6/7] 5-fold CV on '{best_name}' ...")
    best_cfg = models_cfg[best_name]
    X_cv = cache[best_cfg["ftype"]][0]  # training features

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": []}

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_cv, y_train)):
        X_f_tr, y_f_tr = X_cv[tr_idx], y_train[tr_idx]
        X_f_va, y_f_va = X_cv[va_idx], y_train[va_idx]

        if best_cfg["smote"]:
            X_f_tr, y_f_tr = smote.fit_resample(X_f_tr, y_f_tr)

        # Clone model manually to avoid sklearn clone issues with wrappers
        fold_model = best_cfg["model"].__class__(
            **best_cfg["model"].get_params()
        )
        fold_model.fit(X_f_tr, y_f_tr)

        y_f_proba = fold_model.predict_proba(X_f_va)[:, 1]
        thr = results[best_name]["best_threshold"]
        y_f_pred = (y_f_proba >= thr).astype(int)

        cv_metrics["precision"].append(precision_score(y_f_va, y_f_pred, zero_division=0))
        cv_metrics["recall"].append(recall_score(y_f_va, y_f_pred, zero_division=0))
        cv_metrics["f1"].append(f1_score(y_f_va, y_f_pred, zero_division=0))
        cv_metrics["roc_auc"].append(roc_auc_score(y_f_va, y_f_proba))

    cv_summary = {k: f"{np.mean(v):.4f} +/- {np.std(v):.4f}" for k, v in cv_metrics.items()}
    print(f"  {cv_summary}")

    # --- 7. Save ---
    print("\n[7/7] Saving artifacts ...")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Retrain best model on full training set for deployment
    final_model = best_cfg["model"].__class__(**best_cfg["model"].get_params())
    X_final = cache[best_cfg["ftype"]][0]
    if best_cfg["smote"]:
        X_final, y_final = smote.fit_resample(X_final, y_train)
    else:
        y_final = y_train
    final_model.fit(X_final, y_final)

    joblib.dump(final_model, models_dir / "final_model.joblib")
    joblib.dump(fb, models_dir / "feature_builder.joblib")

    model_config = {
        "model_name": best_name,
        "model_type": best_cfg["ftype"],
        "best_threshold": results[best_name]["best_threshold"],
        "uses_smote": best_cfg["smote"],
        "feature_type": best_cfg["ftype"],
    }
    with open(models_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # PR + ROC curves for best model
    best_proba_test = trained_models[best_name].predict_proba(
        cache[best_cfg["ftype"]][2]
    )[:, 1]
    plot_pr_curve(y_test, best_proba_test, title=best_name,
                  save_path=figures_dir / "pr_curve_best.png")
    plot_roc_curve(y_test, best_proba_test, title=best_name,
                   save_path=figures_dir / "roc_curve_best.png")

    # Save metrics
    reports_dir = PROJECT_ROOT / "reports"
    all_results = {
        "best_model": best_name,
        "best_threshold": results[best_name]["best_threshold"],
        "test_metrics": {
            k: {mk: round(mv, 4) for mk, mv in v.items()}
            for k, v in results.items()
        },
        "cv_metrics": cv_summary,
    }
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  [OK] Model  -> {models_dir / 'final_model.joblib'}")
    print(f"  [OK] Config -> {models_dir / 'model_config.json'}")
    print(f"  [OK] Metrics-> {reports_dir / 'metrics.json'}")
    print(f"  [OK] Figures-> {figures_dir}")
    print("\nDone.")

    return all_results


# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train job scam detection models")
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()
    train_all_models(args.data_path)
