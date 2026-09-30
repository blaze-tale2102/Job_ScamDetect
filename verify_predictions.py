"""Quick end-to-end verification of the prediction pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import load_pipeline, predict_single

model, fb, config = load_pipeline()

# Test 1: Legit posting
r1 = predict_single(
    title="Senior Data Engineer - Remote",
    company_profile="TechCorp is a Series-C startup building the next-generation data platform.",
    description=(
        "We are looking for a Senior Data Engineer to design, build, and "
        "maintain our data lake on AWS. You will work closely with the ML team."
    ),
    requirements="5+ years of experience with data engineering. Proficiency in Python and SQL.",
    benefits="Competitive salary, equity, health insurance, 401k match.",
    salary_range="160000-210000",
    has_company_logo=1,
    has_questions=1,
    model=model, fb=fb, config=config,
)
print("=== LEGIT EXAMPLE ===")
print("Verdict:", r1["verdict"])
print("Probability: {:.2%}".format(r1["probability"]))
print("Reasons:", r1["reasons"])

# Test 2: Scam posting
r2 = predict_single(
    title="EARN 5000 PER WEEK NO EXPERIENCE NEEDED",
    description=(
        "MAKE MONEY FROM HOME IMMEDIATELY! NO SKILLS REQUIRED. "
        "Send your resume and bank details to hr@quick-cash-jobs.xyz "
        "or contact us on WhatsApp +1-555-000-1234. "
        "URGENT only 3 positions left! ACT NOW!"
    ),
    has_company_logo=0,
    has_questions=0,
    model=model, fb=fb, config=config,
)
print()
print("=== SCAM EXAMPLE ===")
print("Verdict:", r2["verdict"])
print("Probability: {:.2%}".format(r2["probability"]))
print("Reasons:", r2["reasons"])
