"""Wiring used by the app and the CLI: settings -> data -> tools -> Claude client -> copilot."""

from __future__ import annotations

import logging
import sqlite3

from . import config as C
from .agent.graph import Copilot
from .agent.tools import Toolbox
from .audit import AuditLog
from .llm import Meter, MeteredClient, make_client
from .pipeline import ensure_built

log = logging.getLogger("fleetpilot")


def current_period(s: C.Settings) -> str:
    with sqlite3.connect(f"file:{s.crm_db}?mode=ro", uri=True) as con:
        return con.execute("SELECT max(period) FROM scores").fetchone()[0]


def enable_tracing(s: C.Settings) -> bool:
    """MLflow tracing of every graph run and Claude call, when MLFLOW_TRACKING_URI is set."""
    if not s.mlflow_uri:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(s.mlflow_uri)
        mlflow.set_experiment("fleetpilot-copilot")
        mlflow.langchain.autolog()  # LangGraph runs: one trace per question, a span per node
        mlflow.anthropic.autolog()  # each Claude call: prompt, tool calls, tokens
        return True
    except Exception as exc:  # tracing must never break the copilot
        log.warning("MLflow tracing not enabled: %s", exc)
        return False


def copilot(
    s: C.Settings | None = None,
    client=None,
    api_key: str | None = None,
    model: str | None = None,
    meter: Meter | None = None,
    actor: str = "R-DEMO",
) -> Copilot:
    s = ensure_built(s or C.settings())
    if client is None:
        client = make_client(s.llm_provider, api_key=api_key, region=s.aws_region, meter=meter)
    elif not isinstance(client, MeteredClient):
        client = MeteredClient(client, meter)
    enable_tracing(s)
    return Copilot(
        client,
        Toolbox(s.crm_db, s.max_sql_rows),
        model or s.agent_model,
        current_period(s),
        audit=AuditLog(s.audit_db),
        actor=actor,
        max_steps=s.max_agent_steps,
    )
