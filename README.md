# 🚚 FleetPilot

**Which fleet customers will order new trucks in the next 6 months, why, and what to say to them.**
An ML model ranks customers and explains each score; an AI sales copilot answers reps' questions from the data,
with every number checked, logged and priced.

[![CI](https://github.com/saeedm31/FleetPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/saeedm31/FleetPilot/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20SHAP-orange)](https://xgboost.ai/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph%20%2B%20Claude-blueviolet)](https://www.langchain.com/langgraph)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?logo=mlflow)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](#-run-it)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

▶️ **[Live app](https://sh-fleetpilot.streamlit.app)** · 🏗️ **[Architecture](docs/architecture.md)** · 📇 **[Model card](docs/model_card.md)** · 🔒 **[Security](SECURITY.md)**

> **Simulated data** for a fictional truck maker: 2,000 fleet customers over 3 years, fake company names.
> The simulator knows the true buying rule, so the models can be scored honestly. The numbers show the method,
> not a real sales team's results.

---

## 💡 The problem

A truck maker's sales team can't call 2,000 fleet customers every month. Today many teams call the **oldest fleets
first**. But a fleet buys when several things line up: trucks getting old *and* service costs rising, financing or a
service contract ending, freight demand up, the last order long enough ago, and a good relationship with us.

FleetPilot does three things:

1. **Ranks every customer** by the chance they order from us in the next 6 months, with the **reasons** in plain words.
2. **Groups customers into segments** the team can plan around ("ageing trucks, high service costs").
3. **Answers reps' questions** in a chat: *"Which customers in Portugal should I call first this week, and why?"*,
   *"Draft a first email to C-3F9A21"*, *"How many deals did we lose to competitors?"*

## 🧠 How it works

```mermaid
flowchart LR
  D["CRM, fleet, service<br/>and contract data"] --> F["Features"]
  F --> M["XGBoost<br/>+ calibration"]
  M --> R["SHAP reasons"]
  F --> S["k-means segments"]
  R --> DB[("CRM copy<br/>pseudonymised, read-only")]
  S --> DB
  Q["Rep's question"] --> A["Copilot<br/>LangGraph + Claude"]
  A <--> T["Read-only tools<br/>+ guarded SQL"]
  T --> DB
  A --> C{"Check: every number<br/>from a tool?"}
  C --> ANS["Answer + cost"]
  A -.-> LOG[("Audit log<br/>+ MLflow traces")]
```

**ML**
- **Purchase likelihood:** XGBoost on fleet, service, contract and sales features, calibrated so that 30% means 30%.
  Trained on older months and tested on later months it never saw. Compared with a logistic regression and with the
  "oldest fleets first" rule.
- **Reasons:** SHAP values per customer, turned into phrases such as *"financing ends for 40% of the fleet within
  6 months"*.
- **Segments:** k-means on fleet size, age, mileage, our share, growth and service cost, named automatically.

**AI agent**
- **LangGraph** runs the loop: Claude picks tools → tools run → Claude answers → **check step**.
- **Tools:** `top_customers`, `customer_profile`, `similar_won_deals`, `segment_overview`, and `run_sql` (read-only,
  one SELECT, allow-listed tables, row and time limits).
- **Check step:** every number in the answer must appear in a tool result. If not, the answer goes back once to be
  fixed. The result is shown under each answer (✅ or ⚠️).
- **Claude** through the Anthropic API or **Amazon Bedrock** (EU region, no key), switched with `LLM_PROVIDER`.

**Production practice**
- **Audit log:** append-only, hash-chained record of every question, Claude call, tool call, check and answer, with
  tokens and cost. `python -m fleetpilot audit-verify` proves nothing was changed.
- **MLflow:** training runs, the model registry, and a trace of every copilot run.
- **Cost meter:** tokens and euros per answer; a monthly (company) or session (demo) budget stops new questions.
- **Privacy:** customer IDs are keyed pseudonyms; Claude and the logs never see company names.
- **Evaluation:** 24 questions with answers taken from the database (right tool? right facts? grounded? declines
  out-of-scope and personal-data requests?). Offline in CI with a scripted model; live with real Claude.
- **Engineering:** tests, ruff + pre-commit, CI with a full Docker Compose run, settings from environment variables,
  JSON logs, [architecture decision records](docs/adr), [model card](docs/model_card.md), [data card](docs/data_card.md).

## 📊 Results

Scored on months 25-29, which the model never saw. 20% of customers order within 6 months. "Top 10% finds" is the
share of all buyers a rep reaches by calling the 10% highest-scored customers.

| Model | AUC | PR-AUC | Top 10% finds | Top 20% finds |
|---|---|---|---|---|
| Rule of thumb: oldest fleets first | 0.724 | 0.428 | 26% | 42% |
| Logistic regression | 0.821 | 0.535 | 32% | 52% |
| **XGBoost, calibrated (what the app uses)** | **0.839** | **0.582** | **33%** (3.3x lift) | **55%** |

- **Calibration:** Brier score 0.119; the calibration chart is in the app's Model report tab.
- **Copilot, offline evaluation:** 24 of 24 questions pass with the scripted model (right tool, right facts,
  grounded, declines out-of-scope and personal-data requests).
- **Copilot, live evaluation:** add an `ANTHROPIC_API_KEY` repository secret and run the **Live evaluation**
  workflow. It scores real Claude on the same 24 questions and records demo runs for the app.

## 🖥️ The app

| Tab | What it shows |
|---|---|
| 📋 Pipeline | every customer ranked, with chance to order and main reason; filters by country, industry, segment |
| 🏢 Customer | score and rank, the SHAP reasons, score history with won and lost orders, similar won deals |
| 🤖 Copilot | the chat: answer, ✅/⚠️ check badge, tools used, tokens, € cost |
| 🧩 Segments | the five segments and their average chance to order |
| 📊 Model report | model vs baselines, calibration, copilot evaluation |
| 🧾 Audit log | chain status, every run, and every step of a run |

Demo mode works without a key (recorded real runs, or a scripted test model that shows the mechanics). Live mode
uses your Anthropic key for the session only.

## 🐳 Run it

```bash
git clone https://github.com/saeedm31/FleetPilot.git && cd FleetPilot
cp .env.example .env                 # add ANTHROPIC_API_KEY for live answers (optional)
docker compose up -d --build         # app: http://localhost:8501   MLflow: http://localhost:5000
docker compose run --rm pipeline     # retrain and register the model in MLflow
docker compose run --rm app python -m fleetpilot eval --live    # evaluate the copilot with real Claude
```

| Image | Purpose |
|---|---|
| `app` | Streamlit app + copilot, non-root, health check |
| `train` | batch job: simulate, train, score, register |
| `mlflow` | tracking server, model registry, trace UI |

**Without Docker:** `make install`, then `make build`, `streamlit run app.py`, `make test`.

**Command line:** `python -m fleetpilot ask "Which customers in Spain should I call first?"` · `eval [--live]` ·
`record` · `audit-verify`.

## 📁 Project structure

```
fleetpilot/
  sim.py               simulated customers, fleets, contracts, sales activity, orders
  features.py          features and their plain-English reasons
  model.py             XGBoost + calibration + SHAP; logistic and rule-of-thumb baselines
  segments.py          k-means segments with readable names
  pipeline.py          simulate -> train -> score -> segment -> CRM database (+ MLflow)
  store.py             SQLite CRM: read-only connection, SQL guard
  privacy.py           keyed pseudonyms
  agent/tools.py       the copilot's tools
  agent/graph.py       LangGraph: agent -> tools -> check, audit logging
  agent/check.py       grounding check
  agent/scripted.py    scripted stand-in for Claude (tests, offline evaluation)
  llm.py               Claude client (API or Bedrock) with cost meter
  audit.py             append-only hash-chained audit log
  evaluation.py        24-question evaluation set
app.py                 Streamlit app
docs/                  architecture, ADRs, model card, data card
```

## ⚠️ Limits

- Simulated data and a hand-made buying rule: real data is noisier and has gaps.
- The copilot answers from one month's snapshot; a real one would read the live CRM through read-only views.
- The check step catches numbers without a source, not wrong reasoning: reps stay in charge of the decision.

## License

[MIT](LICENSE)
