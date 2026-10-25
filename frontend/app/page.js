"use client";

import { useState, useEffect } from "react";

const PRESET_EXAMPLES = {
  legit: {
    title: "Senior Data Engineer — Remote",
    company_profile:
      "TechCorp is a Series-C startup building the next-generation data platform. We have 500+ employees across 12 countries.",
    description:
      "We are looking for a Senior Data Engineer to design, build, and maintain our data lake on AWS. You will work closely with the ML team to build robust ETL pipelines and ensure data quality across the organisation. Our stack includes Spark, Airflow, dbt, and Snowflake.",
    requirements:
      "5+ years of experience with data engineering. Proficiency in Python and SQL. Experience with cloud data warehouses (Snowflake, BigQuery, or Redshift). BS in Computer Science or equivalent.",
    benefits:
      "Competitive salary, equity, health/dental/vision insurance, 401(k) match, unlimited PTO, annual learning stipend.",
    salary_range: "160000-210000",
    location: "US — Remote",
    employment_type: "Full-time",
    telecommuting: true,
    has_company_logo: true,
    has_questions: true,
  },
  scam: {
    title: "EARN $5000/WEEK — NO EXPERIENCE NEEDED!",
    company_profile: "",
    description:
      "MAKE MONEY FROM HOME IMMEDIATELY! We are hiring data entry clerks. NO SKILLS REQUIRED. Just send your resume and bank details to hr@quick-cash-jobs.xyz or contact us on WhatsApp +1-555-000-1234. URGENT — only 3 positions left! ACT NOW before it's too late. This is a once-in-a-lifetime opportunity!",
    requirements: "",
    benefits: "",
    salary_range: "",
    location: "",
    employment_type: "",
    telecommuting: false,
    has_company_logo: false,
    has_questions: false,
  },
  suspicious: {
    title: "Customer Support Assistant - Immediate Start",
    company_profile: "Fast-growing digital distribution enterprise.",
    description:
      "Looking for energetic assistants to help process international customer checks and orders. Training materials delivered after initial deposit. Reach out to our HR coordinator via telegram @quick_hire_desk or send email to career@free-hiring-online.info immediately.",
    requirements: "Basic computer familiarity, prompt response time.",
    benefits: "Weekly compensation, flexible work from home.",
    salary_range: "",
    location: "Remote",
    employment_type: "Part-time",
    telecommuting: true,
    has_company_logo: false,
    has_questions: false,
  },
};

const INITIAL_FORM = {
  title: "",
  location: "",
  employment_type: "",
  salary_range: "",
  company_profile: "",
  description: "",
  requirements: "",
  benefits: "",
  has_company_logo: true,
  has_questions: false,
  telecommuting: false,
};

export default function HomePage() {
  const [activeTab, setActiveTab] = useState("detector");
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiConnected, setApiConnected] = useState(false);
  const [metricsData, setMetricsData] = useState(null);
  const [copied, setCopied] = useState(false);

  // Check backend API connection
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch("/api/health");
        if (res.ok) {
          const data = await res.json();
          if (data.status === "ok") {
            setApiConnected(true);
          }
        }
      } catch {
        setApiConnected(false);
      }
    }

    async function loadMetrics() {
      try {
        const res = await fetch("/api/metrics");
        if (res.ok) {
          const json = await res.json();
          if (json.status === "ok") {
            setMetricsData(json.data);
          }
        }
      } catch {
        // fallback metrics handled in render
      }
    }

    checkHealth();
    loadMetrics();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const loadPreset = (presetKey) => {
    if (presetKey === "clear") {
      setFormData(INITIAL_FORM);
      setResult(null);
      setError(null);
      return;
    }
    const preset = PRESET_EXAMPLES[presetKey];
    if (preset) {
      setFormData(preset);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.title.trim() && !formData.description.trim()) {
      setError("Please provide at least a job title or job description.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = {
        title: formData.title,
        company_profile: formData.company_profile,
        description: formData.description,
        requirements: formData.requirements,
        benefits: formData.benefits,
        salary_range: formData.salary_range,
        location: formData.location,
        employment_type: formData.employment_type,
        telecommuting: formData.telecommuting ? 1 : 0,
        has_company_logo: formData.has_company_logo ? 1 : 0,
        has_questions: formData.has_questions ? 1 : 0,
      };

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok || data.status === "error") {
        throw new Error(data.error || data.message || "Failed to analyze posting.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "An unexpected error occurred while communicating with the ML engine.");
    } finally {
      setLoading(false);
    }
  };

  const copyReport = () => {
    if (!result) return;
    const text = `Job Scam Analysis Report:
Verdict: ${result.verdict}
Fraud Probability: ${(result.probability * 100).toFixed(1)}%
Decision Threshold: ${(result.threshold * 100).toFixed(1)}%
Red Flags: ${result.reasons?.length ? result.reasons.join(", ") : "None detected"}
Job Title: ${formData.title || "N/A"}`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <main className="container">
      {/* Header */}
      <header className="app-header">
        <div className="header-badge">
          <span className={`badge-dot ${apiConnected ? "" : "error"}`}></span>
          {apiConnected ? "ML Engine Online" : "Connecting to ML Backend..."}
        </div>
        <h1 className="hero-title">
          🛡️ Safe<span className="gradient-text">Hire AI</span>
        </h1>
        <p className="hero-sub">
          Industrial-grade job fraud detection. Analyze postings with NLP feature embeddings,
          SMOTE-balanced gradient boosting, and explainable red-flag heuristics.
        </p>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`nav-tab-btn ${activeTab === "detector" ? "active" : ""}`}
            onClick={() => setActiveTab("detector")}
          >
            <span>🔍</span> Posting Scanner
          </button>
          <button
            className={`nav-tab-btn ${activeTab === "intelligence" ? "active" : ""}`}
            onClick={() => setActiveTab("intelligence")}
          >
            <span>📊</span> Model Intelligence
          </button>
          <button
            className={`nav-tab-btn ${activeTab === "guide" ? "active" : ""}`}
            onClick={() => setActiveTab("guide")}
          >
            <span>🚩</span> Red-Flag Guide
          </button>
        </nav>
      </header>

      {/* TAB 1: DETECTOR */}
      {activeTab === "detector" && (
        <section>
          {/* Preset Buttons */}
          <div className="glass-panel presets-card">
            <span className="presets-label">
              <span>⚡</span> Quick Sample Presets:
            </span>
            <div className="presets-buttons">
              <button
                type="button"
                className="preset-btn btn-legit"
                onClick={() => loadPreset("legit")}
              >
                <span>✅</span> Legit Data Engineer
              </button>
              <button
                type="button"
                className="preset-btn btn-scam"
                onClick={() => loadPreset("scam")}
              >
                <span>🚨</span> $5000/Wk Scam
              </button>
              <button
                type="button"
                className="preset-btn btn-suspicious"
                onClick={() => loadPreset("suspicious")}
              >
                <span>⚠️</span> Suspicious Check Scam
              </button>
              <button
                type="button"
                className="preset-btn btn-clear"
                onClick={() => loadPreset("clear")}
              >
                <span>🧹</span> Reset
              </button>
            </div>
          </div>

          {/* Main Grid: Form on Left, Results on Right */}
          <div className="main-grid">
            {/* Input Form */}
            <div className="glass-panel form-panel">
              <div className="panel-header">
                <h2 className="panel-title">
                  <span>📝</span> Job Posting Attributes
                </h2>
                <span style={{ fontSize: "0.8rem", color: "var(--text-subtle)" }}>
                  NLP & Metadata Fields
                </span>
              </div>

              <form onSubmit={handleSubmit}>
                <div className="form-group">
                  <label className="form-label" htmlFor="inp-title">
                    Job Title *
                  </label>
                  <input
                    id="inp-title"
                    type="text"
                    name="title"
                    className="form-input"
                    placeholder="e.g. Senior Machine Learning Engineer"
                    value={formData.title}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-grid-2">
                  <div className="form-group">
                    <label className="form-label" htmlFor="inp-location">
                      Location
                    </label>
                    <input
                      id="inp-location"
                      type="text"
                      name="location"
                      className="form-input"
                      placeholder="e.g. US — Remote or New York, NY"
                      value={formData.location}
                      onChange={handleChange}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label" htmlFor="inp-etype">
                      Employment Type
                    </label>
                    <select
                      id="inp-etype"
                      name="employment_type"
                      className="form-select"
                      value={formData.employment_type}
                      onChange={handleChange}
                    >
                      <option value="">Select type...</option>
                      <option value="Full-time">Full-time</option>
                      <option value="Part-time">Part-time</option>
                      <option value="Contract">Contract</option>
                      <option value="Temporary">Temporary</option>
                      <option value="Internship">Internship</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="inp-salary">
                    Salary Range
                  </label>
                  <input
                    id="inp-salary"
                    type="text"
                    name="salary_range"
                    className="form-input"
                    placeholder="e.g. 120000-150000 or $35/hr"
                    value={formData.salary_range}
                    onChange={handleChange}
                  />
                </div>

                {/* Verification checkboxes */}
                <div className="checkbox-group">
                  <label className="checkbox-item">
                    <input
                      type="checkbox"
                      name="has_company_logo"
                      checked={formData.has_company_logo}
                      onChange={handleChange}
                    />
                    <span>Company logo is present on posting</span>
                  </label>

                  <label className="checkbox-item">
                    <input
                      type="checkbox"
                      name="has_questions"
                      checked={formData.has_questions}
                      onChange={handleChange}
                    />
                    <span>Includes applicant screening questions</span>
                  </label>

                  <label className="checkbox-item">
                    <input
                      type="checkbox"
                      name="telecommuting"
                      checked={formData.telecommuting}
                      onChange={handleChange}
                    />
                    <span>Telecommuting / 100% Remote role</span>
                  </label>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="inp-desc">
                    Job Description *
                  </label>
                  <textarea
                    id="inp-desc"
                    name="description"
                    className="form-textarea"
                    rows={4}
                    placeholder="Paste the core responsibilities, day-to-day duties, and description..."
                    value={formData.description}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="inp-req">
                    Requirements
                  </label>
                  <textarea
                    id="inp-req"
                    name="requirements"
                    className="form-textarea"
                    rows={2}
                    placeholder="Qualifications, skills, education, or years of experience..."
                    value={formData.requirements}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="inp-cp">
                    Company Profile
                  </label>
                  <textarea
                    id="inp-cp"
                    name="company_profile"
                    className="form-textarea"
                    rows={2}
                    placeholder="Information about company mission, funding, or history..."
                    value={formData.company_profile}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="inp-ben">
                    Benefits
                  </label>
                  <textarea
                    id="inp-ben"
                    name="benefits"
                    className="form-textarea"
                    rows={2}
                    placeholder="Health insurance, 401(k), equity, PTO..."
                    value={formData.benefits}
                    onChange={handleChange}
                  />
                </div>

                {error && (
                  <div
                    style={{
                      background: "rgba(239, 68, 68, 0.12)",
                      border: "1px solid rgba(239, 68, 68, 0.3)",
                      color: "#fca5a5",
                      padding: "0.8rem 1rem",
                      borderRadius: "8px",
                      fontSize: "0.9rem",
                      marginBottom: "1rem",
                    }}
                  >
                    ⚠️ {error}
                  </div>
                )}

                <button type="submit" className="scan-btn" disabled={loading}>
                  {loading ? (
                    <>
                      <div className="spinner"></div>
                      <span>Analyzing Job Posting via ML Pipeline...</span>
                    </>
                  ) : (
                    <>
                      <span>🔍</span>
                      <span>Run Fraud Scan & Classification</span>
                    </>
                  )}
                </button>
              </form>
            </div>

            {/* Results Panel */}
            <div className="glass-panel results-panel">
              <div className="panel-header">
                <h2 className="panel-title">
                  <span>🎯</span> Diagnostic Output
                </h2>
                {result && (
                  <button
                    type="button"
                    onClick={copyReport}
                    style={{
                      background: "rgba(255,255,255,0.06)",
                      border: "1px solid rgba(255,255,255,0.12)",
                      color: "#c7d2fe",
                      padding: "0.3rem 0.7rem",
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontSize: "0.78rem",
                      fontFamily: "inherit",
                    }}
                  >
                    {copied ? "✓ Copied!" : "📋 Copy Summary"}
                  </button>
                )}
              </div>

              {!result && !loading && (
                <div className="empty-state">
                  <div className="empty-icon">🛡️</div>
                  <h3 className="empty-title">Awaiting Job Data</h3>
                  <p className="empty-desc">
                    Fill out the posting attributes or click one of the quick presets above to
                    trigger real-time classification.
                  </p>
                </div>
              )}

              {loading && (
                <div className="empty-state">
                  <div className="spinner" style={{ width: 44, height: 44, borderWidth: 3 }}></div>
                  <h3 className="empty-title" style={{ marginTop: "1.5rem" }}>
                    Evaluating Text & Signals
                  </h3>
                  <p className="empty-desc">
                    Tokenizing text, vectorizing TF-IDF n-grams, and running model inference...
                  </p>
                </div>
              )}

              {result && !loading && (
                <div>
                  {/* Verdict Banner */}
                  <div
                    className={`verdict-banner ${
                      result.verdict === "Legit"
                        ? "legit"
                        : result.verdict === "Suspicious"
                        ? "suspicious"
                        : "scam"
                    }`}
                  >
                    <div className="verdict-icon">
                      {result.verdict === "Legit"
                        ? "✅"
                        : result.verdict === "Suspicious"
                        ? "⚠️"
                        : "🚫"}
                    </div>
                    <div className="verdict-title">{result.verdict}</div>
                    <div className="verdict-desc">
                      {result.verdict === "Legit"
                        ? "This posting matches verified legitimate employment patterns."
                        : result.verdict === "Suspicious"
                        ? "Borderline characteristics detected. Further validation recommended."
                        : "High fraud probability. Severe scam indicators detected."}
                    </div>
                  </div>

                  {/* Fraud Probability Gauge */}
                  <div className="metric-box">
                    <div className="metric-header">
                      <span className="metric-title">Fraud Risk Probability</span>
                      <span
                        className="metric-value"
                        style={{
                          color:
                            result.probability < 0.3
                              ? "#34d399"
                              : result.probability < 0.6
                              ? "#fbbf24"
                              : "#f87171",
                        }}
                      >
                        {(result.probability * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="progress-track">
                      <div
                        className="progress-bar-fill"
                        style={{
                          width: `${Math.max(result.probability * 100, 3)}%`,
                          background:
                            result.probability < 0.3
                              ? "linear-gradient(90deg, #10b981, #34d399)"
                              : result.probability < 0.6
                              ? "linear-gradient(90deg, #f59e0b, #fbbf24)"
                              : "linear-gradient(90deg, #ef4444, #f87171)",
                          color:
                            result.probability < 0.3
                              ? "#10b981"
                              : result.probability < 0.6
                              ? "#f59e0b"
                              : "#ef4444",
                        }}
                      />
                    </div>

                    <div className="threshold-marker">
                      <span>0% (Clean)</span>
                      <span>
                        Threshold: {(result.threshold * 100).toFixed(1)}% (Optimized F1)
                      </span>
                      <span>100% (Confirmed Fraud)</span>
                    </div>
                  </div>

                  {/* Red-Flag Explanations */}
                  <div className="red-flags-section">
                    <div className="section-tag">
                      <span>🚩</span> Heuristic Red Flags & Explanations
                    </div>

                    {result.reasons && result.reasons.length > 0 ? (
                      <div className="flags-grid">
                        {result.reasons.map((reason, idx) => (
                          <div key={idx} className="flag-chip">
                            <span className="flag-chip-icon">⚑</span>
                            <span>{reason}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="no-flags-box">
                        <span>✨</span>
                        <span>
                          No heuristic red flags detected. Description structure, logo, contact
                          methods, and phrasing conform to legitimate standard postings.
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* TAB 2: MODEL INTELLIGENCE */}
      {activeTab === "intelligence" && (
        <section>
          {/* KPI Cards */}
          <div className="metrics-grid-3">
            <div className="glass-panel kpi-card">
              <div className="kpi-title">Best Model</div>
              <div className="kpi-value" style={{ fontSize: "1.35rem" }}>
                {metricsData?.best_model || "LR + SMOTE"}
              </div>
              <div className="kpi-sub">Optimized for imbalanced fraud recall</div>
            </div>

            <div className="glass-panel kpi-card">
              <div className="kpi-title">ROC-AUC Score</div>
              <div className="kpi-value">
                {metricsData?.test_metrics?.[metricsData?.best_model]?.roc_auc || "0.9914"}
              </div>
              <div className="kpi-sub">Exceptional class separation accuracy</div>
            </div>

            <div className="glass-panel kpi-card">
              <div className="kpi-title">Fraud Class F1</div>
              <div className="kpi-value">
                {metricsData?.test_metrics?.[metricsData?.best_model]?.f1
                  ? `${(metricsData.test_metrics[metricsData.best_model].f1 * 100).toFixed(1)}%`
                  : "84.4%"}
              </div>
              <div className="kpi-sub">Balancing Precision (80.0%) & Recall (89.2%)</div>
            </div>

            <div className="glass-panel kpi-card">
              <div className="kpi-title">Optimal Threshold</div>
              <div className="kpi-value">
                {metricsData?.best_threshold
                  ? `${(metricsData.best_threshold * 100).toFixed(1)}%`
                  : "52.6%"}
              </div>
              <div className="kpi-sub">Tuned on validation partition</div>
            </div>
          </div>

          {/* Model Comparison Table */}
          <div className="glass-panel" style={{ padding: "1.75rem", marginBottom: "2rem" }}>
            <h2 className="panel-title" style={{ marginBottom: "1rem" }}>
              <span>🏆</span> 7-Classifier Benchmark Comparison (EMSCAD Dataset)
            </h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginBottom: "1.25rem" }}>
              All models evaluated on the held-out test split under severe class imbalance (~5% fraud).
              The selected production model achieves an outstanding 89.2% scam recall while maintaining
              high precision.
            </p>

            <div className="metrics-table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Model Architecture</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>ROC-AUC</th>
                    <th>PR-AUC</th>
                    <th>Best Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {metricsData?.test_metrics ? (
                    Object.entries(metricsData.test_metrics).map(([name, m]) => {
                      const isBest = name === metricsData.best_model;
                      return (
                        <tr key={name} className={isBest ? "highlight" : ""}>
                          <td>
                            {name}
                            {isBest && <span className="badge-best">PROD SELECTED</span>}
                          </td>
                          <td>{(m.precision * 100).toFixed(1)}%</td>
                          <td>{(m.recall * 100).toFixed(1)}%</td>
                          <td>{(m.f1 * 100).toFixed(1)}%</td>
                          <td>{m.roc_auc.toFixed(4)}</td>
                          <td>{m.pr_auc.toFixed(4)}</td>
                          <td>{(m.best_threshold * 100).toFixed(1)}%</td>
                        </tr>
                      );
                    })
                  ) : (
                    <>
                      <tr className="highlight">
                        <td>
                          Logistic Regression + SMOTE
                          <span className="badge-best">PROD SELECTED</span>
                        </td>
                        <td>80.0%</td>
                        <td>89.2%</td>
                        <td>84.4%</td>
                        <td>0.9914</td>
                        <td>0.9254</td>
                        <td>52.6%</td>
                      </tr>
                      <tr>
                        <td>LightGBM</td>
                        <td>96.0%</td>
                        <td>73.8%</td>
                        <td>83.5%</td>
                        <td>0.9891</td>
                        <td>0.9088</td>
                        <td>9.8%</td>
                      </tr>
                      <tr>
                        <td>Random Forest</td>
                        <td>91.8%</td>
                        <td>69.2%</td>
                        <td>79.0%</td>
                        <td>0.9880</td>
                        <td>0.8895</td>
                        <td>45.0%</td>
                      </tr>
                      <tr>
                        <td>LightGBM + SMOTE</td>
                        <td>86.5%</td>
                        <td>73.8%</td>
                        <td>79.7%</td>
                        <td>0.9899</td>
                        <td>0.8997</td>
                        <td>37.8%</td>
                      </tr>
                      <tr>
                        <td>Logistic Regression</td>
                        <td>83.5%</td>
                        <td>81.5%</td>
                        <td>82.5%</td>
                        <td>0.9910</td>
                        <td>0.9171</td>
                        <td>66.4%</td>
                      </tr>
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {/* TAB 3: RED-FLAG GUIDE */}
      {activeTab === "guide" && (
        <section className="guide-grid">
          <div className="glass-panel guide-card">
            <div className="guide-icon">💸</div>
            <h3 className="guide-title">Unrealistic Compensation & No Experience</h3>
            <p className="guide-text">
              Scammers lure vulnerable job seekers with extravagant hourly or weekly salaries for
              generic roles (e.g. $5,000/week for basic data entry or package inspection).
            </p>
          </div>

          <div className="glass-panel guide-card">
            <div className="guide-icon">💬</div>
            <h3 className="guide-title">Off-Platform Instant Messaging</h3>
            <p className="guide-text">
              Directing applicants immediately to WhatsApp, Telegram, Skype, or free webmail domains
              (@gmail.com, @xyz) avoids enterprise security controls and audit trails.
            </p>
          </div>

          <div className="glass-panel guide-card">
            <div className="guide-icon">⚡</div>
            <h3 className="guide-title">Urgency & Pressure Tactics</h3>
            <p className="guide-text">
              Phrases like &ldquo;ACT NOW&rdquo;, &ldquo;ONLY 2 SPOTS LEFT&rdquo;, or &ldquo;START
              IMMEDIATELY WITHOUT INTERVIEW&rdquo; prevent candidates from conducting due diligence.
            </p>
          </div>

          <div className="glass-panel guide-card">
            <div className="guide-icon">🏢</div>
            <h3 className="guide-title">Missing Company Identity & Logos</h3>
            <p className="guide-text">
              Over 90% of fraudulent postings lack verified corporate logos, company descriptions,
              or direct links to registered corporate domain websites.
            </p>
          </div>

          <div className="glass-panel guide-card">
            <div className="guide-icon">💳</div>
            <h3 className="guide-title">Upfront Equipment Checks & Transfers</h3>
            <p className="guide-text">
              Sending a fraudulent check with instructions to wire surplus funds back for home office
              equipment is a prevalent counterfeit check scam.
            </p>
          </div>

          <div className="glass-panel guide-card">
            <div className="guide-icon">📝</div>
            <h3 className="guide-title">Absence of Screening Questions</h3>
            <p className="guide-text">
              Legitimate employers typically require tailored qualification answers or portfolio
              submissions, whereas scams minimize friction to capture more victims.
            </p>
          </div>
        </section>
      )}

      {/* Footer */}
      <footer className="app-footer">
        <p>
          ⚖️ <strong>Disclaimer</strong>: SafeHire AI is an analytical screening aid designed for
          threat mitigation. Predictions should be verified with official employer registries.
        </p>
        <p style={{ marginTop: "0.4rem" }}>
          Built with Next.js & Python ML Pipeline (TF-IDF, Gradient Boosting, SMOTE) | Trained on
          EMSCAD Dataset
        </p>
      </footer>
    </main>
  );
}
