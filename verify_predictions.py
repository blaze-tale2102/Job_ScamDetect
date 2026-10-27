from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_pipeline, predict_single


def verify():
    print("=" * 70)
    print("VERIFYING MODEL PREDICTIONS ON REAL TEST POSTINGS FROM DATASET")
    print("=" * 70)

    model, fb, config = load_pipeline()
    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"

    if not test_csv.exists():
        print(f"Test dataset not found at {test_csv}.")
        return

    df = pd.read_csv(test_csv)

    legit_sample = df[df["fraudulent"] == 0].iloc[0]
    r_legit = predict_single(
        title=str(legit_sample.get("title", "")),
        company_profile=str(legit_sample.get("company_profile", "")),
        description=str(legit_sample.get("description", "")),
        requirements=str(legit_sample.get("requirements", "")),
        benefits=str(legit_sample.get("benefits", "")),
        has_company_logo=int(legit_sample.get("has_company_logo", 1)),
        has_questions=int(legit_sample.get("has_questions", 0)),
        model=model,
        fb=fb,
        config=config,
    )

    print("\n[SAMPLE 1] REAL LEGITIMATE JOB POSTING FROM DATASET")
    print("-" * 55)
    print(f"  Job ID in Dataset  : {legit_sample.get('job_id', 'N/A')}")
    print(f"  Job Title          : {legit_sample.get('title', 'N/A')}")
    print(f"  True Label (CSV)   : {legit_sample['fraudulent']} (Legitimate)")
    print(f"  ML Predicted Prob  : {r_legit['probability']:.2%}")
    print(f"  Model Verdict      : {r_legit['verdict']}")
    print(f"  Flagged Reasons    : {r_legit['reasons'] if r_legit['reasons'] else 'None (Clean)'}")

    fraud_sample = df[df["fraudulent"] == 1].iloc[0]
    r_fraud = predict_single(
        title=str(fraud_sample.get("title", "")),
        company_profile=str(fraud_sample.get("company_profile", "")),
        description=str(fraud_sample.get("description", "")),
        requirements=str(fraud_sample.get("requirements", "")),
        benefits=str(fraud_sample.get("benefits", "")),
        has_company_logo=int(fraud_sample.get("has_company_logo", 0)),
        has_questions=int(fraud_sample.get("has_questions", 0)),
        model=model,
        fb=fb,
        config=config,
    )

    print("\n[SAMPLE 2] REAL FRAUDULENT JOB POSTING FROM DATASET")
    print("-" * 55)
    print(f"  Job ID in Dataset  : {fraud_sample.get('job_id', 'N/A')}")
    print(f"  Job Title          : {fraud_sample.get('title', 'N/A')}")
    print(f"  True Label (CSV)   : {fraud_sample['fraudulent']} (Fraudulent)")
    print(f"  ML Predicted Prob  : {r_fraud['probability']:.2%}")
    print(f"  Model Verdict      : {r_fraud['verdict']}")
    print(f"  Flagged Reasons    : {r_fraud['reasons']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    verify()
