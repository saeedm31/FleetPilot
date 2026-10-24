# Changelog

## 0.1.0 (2026-10-24)
- Simulator: 2,000 fleet customers, 36 months, orders won and lost.
- Purchase-likelihood model (XGBoost, calibrated) with SHAP reasons; logistic regression and rule-of-thumb baselines.
- Customer segments (k-means), named in plain words.
- Sales copilot: LangGraph + Claude (API or Amazon Bedrock), read-only tools, grounding check.
- Audit log (hash chain), MLflow tracking, registry and traces, cost meter with budget.
- Streamlit app, Docker Compose (app, pipeline, MLflow), CI, offline and live evaluation.
