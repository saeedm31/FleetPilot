"""The copilot's tools: plain Python functions over the read-only CRM database.

Each tool returns JSON-ready data. Results are capped so prompts stay small. Only
pseudonymised customer IDs ever leave this module; company names are not in the database.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..store import SqlError, read_only, run_select, schema_text

ID_RE = re.compile(r"^C-[0-9A-F]{6}$")
MAX_TOOL_ROWS = 25


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return dict(name=self.name, description=self.description, input_schema=self.input_schema)


class Toolbox:
    def __init__(self, crm_db: Path, max_sql_rows: int = 200):
        self.db = Path(crm_db)
        self.con = read_only(self.db)
        self.max_sql_rows = max_sql_rows
        self.schema = schema_text(self.db)
        self.tools = {t.name: t for t in self._tools()}

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def run(self, name: str, args: dict) -> tuple[Any, bool]:
        """Returns (result, is_error). Errors are returned to the model, never raised."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool {name}", True
        try:
            return tool.fn(**(args or {})), False
        except (SqlError, ValueError, TypeError) as exc:
            return f"{type(exc).__name__}: {exc}", True

    # --- helpers ---
    def _q(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        cur = self.con.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

    @staticmethod
    def _records(df: pd.DataFrame, limit: int = MAX_TOOL_ROWS) -> dict:
        out = dict(rows=json.loads(df.head(limit).to_json(orient="records")), row_count=len(df))
        if len(df) > limit:
            out["note"] = f"showing {limit} of {len(df)} rows"
        return out

    @staticmethod
    def _check_id(customer_id: str) -> str:
        cid = str(customer_id).strip().upper()
        if not ID_RE.match(cid):
            raise ValueError("customer_id must look like C-1A2B3C")
        return cid

    # --- tools ---
    def top_customers(
        self,
        country: str | None = None,
        industry: str | None = None,
        segment: str | None = None,
        min_score: float = 0.0,
        limit: int = 10,
    ) -> dict:
        where, params = ["s.score >= ?"], [float(min_score)]
        for col, val in (("c.country", country), ("c.industry", industry), ("c.segment", segment)):
            if val:
                where.append(f"lower({col}) = lower(?)")
                params.append(val)
        limit = max(1, min(int(limit), MAX_TOOL_ROWS))
        df = self._q(
            f"""SELECT s.customer_id, s.score, s.rank, c.country, c.industry, c.segment, n.fleet_size,
                                n.avg_age_years, s.reason_1, s.reason_2, s.reason_3
                         FROM scores s JOIN customers c USING(customer_id) JOIN snapshot n USING(customer_id)
                         WHERE {" AND ".join(where)} ORDER BY s.score DESC LIMIT ?""",
            (*params, limit),
        )
        return self._records(df)

    def customer_profile(self, customer_id: str) -> dict:
        cid = self._check_id(customer_id)
        cust = self._q("SELECT * FROM customers WHERE customer_id=?", (cid,))
        if cust.empty:
            raise ValueError(f"no customer {cid}")
        snap = self._q("SELECT * FROM snapshot WHERE customer_id=?", (cid,))
        score = self._q(
            "SELECT score, rank, period, model_version, reason_1, reason_2, reason_3 FROM scores WHERE customer_id=?",
            (cid,),
        )
        deals = self._q(
            "SELECT deal_id, period, status, trucks, value_eur, trigger FROM deals WHERE customer_id=? ORDER BY period DESC LIMIT 5",
            (cid,),
        )
        return dict(
            customer=cust.iloc[0].to_dict(),
            snapshot=json.loads(snap.iloc[0].to_json()),
            score=score.iloc[0].to_dict(),
            recent_deals=json.loads(deals.to_json(orient="records")),
            customers_total=int(self._q("SELECT count(*) AS n FROM customers").iloc[0, 0]),
        )

    def similar_won_deals(self, customer_id: str, limit: int = 5) -> dict:
        cid = self._check_id(customer_id)
        me = self._q(
            """SELECT c.industry, n.fleet_size FROM customers c JOIN snapshot n USING(customer_id)
                        WHERE customer_id=?""",
            (cid,),
        )
        if me.empty:
            raise ValueError(f"no customer {cid}")
        ind, size = me.iloc[0]["industry"], float(me.iloc[0]["fleet_size"])
        df = self._q(
            """SELECT d.deal_id, d.customer_id, d.period, d.trucks, d.value_eur, d.trigger, n.fleet_size
                        FROM deals d JOIN customers c USING(customer_id) JOIN snapshot n USING(customer_id)
                        WHERE d.status='won' AND c.industry=? AND n.fleet_size BETWEEN ? AND ? AND d.customer_id<>?
                        ORDER BY d.period DESC LIMIT ?""",
            (ind, size * 0.5, size * 2.0, cid, max(1, min(int(limit), 10))),
        )
        return dict(industry=ind, fleet_size=size, **self._records(df))

    def segment_overview(self, segment: str | None = None) -> dict:
        if segment:
            return self._records(self._q("SELECT * FROM segments WHERE lower(segment)=lower(?)", (segment,)))
        return self._records(self._q("SELECT * FROM segments ORDER BY avg_score DESC"))

    def run_sql(self, query: str) -> dict:
        df = run_select(self.con, query, max_rows=self.max_sql_rows)
        out = self._records(df)
        if df.attrs.get("truncated"):
            out["note"] = f"result capped at {self.max_sql_rows} rows"
        return out

    def _tools(self) -> list[Tool]:
        s = {"type": "string"}
        return [
            Tool(
                "top_customers",
                "Customers most likely to order new trucks in the next 6 months, best first, "
                "with score (0-1), fleet facts and the model's top reasons. Optional filters: country (2-letter code "
                "like PT, DE), industry, segment, min_score.",
                {
                    "type": "object",
                    "properties": {
                        "country": s,
                        "industry": s,
                        "segment": s,
                        "min_score": {"type": "number"},
                        "limit": {"type": "integer"},
                    },
                },
                self.top_customers,
            ),
            Tool(
                "customer_profile",
                "Everything known about one customer: profile, latest fleet snapshot, score, "
                "reasons and the last 5 deals.",
                {"type": "object", "properties": {"customer_id": s}, "required": ["customer_id"]},
                self.customer_profile,
            ),
            Tool(
                "similar_won_deals",
                "Recent won deals of similar customers (same industry, fleet half to double "
                "the size), with what triggered each order. Use to suggest a pitch.",
                {
                    "type": "object",
                    "properties": {"customer_id": s, "limit": {"type": "integer"}},
                    "required": ["customer_id"],
                },
                self.similar_won_deals,
            ),
            Tool(
                "segment_overview",
                "Customer segments with size, fleet facts and average score. Optional: one segment.",
                {"type": "object", "properties": {"segment": s}},
                self.segment_overview,
            ),
            Tool(
                "run_sql",
                "Read-only SQL (SQLite dialect, one SELECT) for questions the other tools can't answer. "
                f"Tables:\n{self.schema}",
                {"type": "object", "properties": {"query": s}, "required": ["query"]},
                self.run_sql,
            ),
        ]
