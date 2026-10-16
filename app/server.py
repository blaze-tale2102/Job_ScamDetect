"""
Flask REST API for Job Scam Detection.
Serves predictions, metrics, and preset examples to the Next.js frontend.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.predict import load_pipeline, predict_single

app = Flask(__name__)
# Enable CORS for all routes so Next.js on port 3000 can communicate smoothly
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load model pipeline at startup
print("[API] Loading ML model pipeline...")
try:
    model, fb, config = load_pipeline()
    model_loaded = True
    print(f"[API] Pipeline loaded successfully (Threshold: {config.get('threshold', 0.5):.4f})")
except Exception as e:
    model, fb, config = None, None, {}
    model_loaded = False
    print(f"[API] Error loading pipeline: {e}")

# Preset examples
PRESET_EXAMPLES = {
    "legit": {
        "title": "Senior Data Engineer — Remote",
        "company_profile": (
            "TechCorp is a Series-C startup building the next-generation data "
            "platform. We have 500+ employees across 12 countries."
        ),
        "description": (
            "We are looking for a Senior Data Engineer to design, build, and "
            "maintain our data lake on AWS. You will work closely with the ML "
            "team to build robust ETL pipelines and ensure data quality across "
            "the organisation. Our stack includes Spark, Airflow, dbt, and Snowflake."
        ),
        "requirements": (
            "5+ years of experience with data engineering. "
            "Proficiency in Python and SQL. "
            "Experience with cloud data warehouses (Snowflake, BigQuery, or Redshift). "
            "BS in Computer Science or equivalent."
        ),
        "benefits": (
            "Competitive salary, equity, health/dental/vision insurance, "
            "401(k) match, unlimited PTO, annual learning stipend."
        ),
        "salary_range": "160000-210000",
        "location": "US — Remote",
        "employment_type": "Full-time",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1,
    },
    "scam": {
        "title": "EARN $5000/WEEK — NO EXPERIENCE NEEDED!",
        "company_profile": "",
        "description": (
            "MAKE MONEY FROM HOME IMMEDIATELY! We are hiring data entry clerks. "
            "NO SKILLS REQUIRED. Just send your resume and bank details to "
            "hr@quick-cash-jobs.xyz or contact us on WhatsApp +1-555-000-1234. "
            "URGENT — only 3 positions left! ACT NOW before it's too late. "
            "This is a once-in-a-lifetime opportunity!"
        ),
        "requirements": "",
        "benefits": "",
        "salary_range": "",
        "location": "",
        "employment_type": "",
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
    },
    "suspicious": {
        "title": "Customer Service Representative - Immediate Start",
        "company_profile": "Rapidly expanding international eCommerce brand.",
        "description": (
            "Seeking motivated representatives for processing incoming customer orders. "
            "Must be able to start ASAP. Training provided online via wire transfer "
            "materials. Contact our recruitment desk at recruitment@freemail-support.biz."
        ),
        "requirements": "Basic computer skills, good communication.",
        "benefits": "Flexible schedule, weekly bonuses.",
        "salary_range": "",
        "location": "Remote",
        "employment_type": "Part-time",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
    }
}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "model_config": config,
    })


@app.route("/api/examples", methods=["GET"])
def get_examples():
    return jsonify(PRESET_EXAMPLES)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"status": "ok", "data": data})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Metrics file not found"}), 404


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model pipeline is not loaded on the server."}), 503

    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    description = data.get("description", "")

    if not title.strip() and not description.strip():
        return jsonify({
            "error": "Please provide at least a job title or description."
        }), 400

    try:
        result = predict_single(
            title=title,
            company_profile=data.get("company_profile", ""),
            description=description,
            requirements=data.get("requirements", ""),
            benefits=data.get("benefits", ""),
            salary_range=data.get("salary_range", ""),
            location=data.get("location", ""),
            employment_type=data.get("employment_type", ""),
            telecommuting=int(data.get("telecommuting", 0)),
            has_company_logo=int(data.get("has_company_logo", 1)),
            has_questions=int(data.get("has_questions", 0)),
            model=model,
            fb=fb,
            config=config,
        )
        return jsonify({
            "status": "success",
            "verdict": result["verdict"],
            "probability": float(result["probability"]),
            "reasons": result["reasons"],
            "threshold": float(config.get("threshold", 0.5)),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[API] Running Job Scam Detection API server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
