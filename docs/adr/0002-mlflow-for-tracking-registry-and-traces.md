# ADR 0002: MLflow for experiment tracking, model registry and agent traces

**Status:** accepted · **Date:** 2026-10-24

## Decision
Use **MLflow** (server in its own container) for: training runs (parameters, metrics, artifacts), the model registry
(`fleetpilot-purchase`), and traces of every copilot run (LangGraph + Claude autolog).

## Why
One tool covers both the ML model lifecycle and the GenAI agent, so a reviewer can follow a prediction from the training
run to the registered model, and an answer from the question to each Claude call and tool call.

## Alternatives
- Langfuse: excellent LLM tracing, but no ML model registry, and heavier to self-host (several databases).
- LangSmith: hosted, tied to the LangChain company.

## Consequences
MLflow is optional at runtime: without `MLFLOW_TRACKING_URI` the app still works, and the audit log still records
everything.
