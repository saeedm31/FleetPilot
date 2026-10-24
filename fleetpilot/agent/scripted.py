"""A scripted stand-in for Claude, for tests and offline checks of the agent's wiring.

It answers like the Messages API (content blocks, stop_reason, usage) with simple rules: pick
tools from keywords in the question, then write an answer that copies numbers from the tool
results. It is not a language model; it only proves the graph, tools, check and logs work.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from ..sim import COUNTRY_NAMES, INDUSTRY_NAMES

ID = re.compile(r"\bC-[0-9A-F]{6}\b")


class ScriptedMessages:
    def __init__(self, invent_number_once: bool = False):
        self.invent = invent_number_once
        self.n = 0

    def create(self, model, messages, tools=None, system=None, max_tokens=None, tool_choice=None, **_):
        self.n += 1
        last = messages[-1]["content"]
        question = next(m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str))
        usage = SimpleNamespace(
            input_tokens=900 + 40 * len(messages),
            output_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        if isinstance(last, str) and last == question and _out_of_scope(question):
            text = "I can only help with sales questions about our fleet customers, and I can't share personal data or change the database."
            return SimpleNamespace(content=[{"type": "text", "text": text}], stop_reason="end_turn", usage=usage)
        if isinstance(last, str) and last == question and (tool_choice or {}).get("type") != "none":
            blocks = [
                {"type": "tool_use", "id": f"t{self.n}{i}", "name": n, "input": a}
                for i, (n, a) in enumerate(self._plan(question))
            ]
            return SimpleNamespace(content=blocks, stop_reason="tool_use", usage=usage)
        results = [
            json.loads(b["content"])
            for m in messages
            if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == "tool_result" and not b.get("is_error")
        ]
        text = self._write(results, retry=isinstance(last, str) and last.startswith("Check failed"))
        return SimpleNamespace(content=[{"type": "text", "text": text}], stop_reason="end_turn", usage=usage)

    def _plan(self, q: str) -> list[tuple[str, dict]]:
        ql = q.lower()
        ids = ID.findall(q)
        if ids:
            return [("customer_profile", {"customer_id": ids[0]}), ("similar_won_deals", {"customer_id": ids[0]})]
        if "segment" in ql:
            return [("segment_overview", {})]
        if "how many" in ql and "deal" in ql:
            status = "lost" if "lost" in ql or "lose" in ql else "won"
            sql = f"SELECT status, count(*) AS deals FROM deals WHERE status = '{status}' GROUP BY status"
            return [("run_sql", {"query": sql})]
        if "how many" in ql:
            return [
                (
                    "run_sql",
                    {
                        "query": "SELECT country, count(*) AS customers FROM customers GROUP BY country ORDER BY customers DESC"
                    },
                )
            ]
        args: dict = {"limit": 5}
        for c in COUNTRY_NAMES:
            if (
                re.search(rf"\b{c}\b", q)
                or {
                    "PT": "portugal",
                    "DE": "germany",
                    "ES": "spain",
                    "FR": "france",
                    "IT": "italy",
                    "PL": "poland",
                    "NL": "netherlands",
                    "SE": "sweden",
                }[c]
                in ql
            ):
                args["country"] = c
        for ind in INDUSTRY_NAMES:
            if ind.split()[0] in ql:
                args["industry"] = ind
        return [("top_customers", args)]

    def _write(self, results: list[dict], retry: bool) -> str:
        lines = []
        if self.invent and not retry:
            lines.append("Pipeline value is about 1234567 euros.")
        for r in results:
            if "rows" in r:
                for row in r["rows"][:5]:
                    if "customer_id" in row and "score" in row:
                        lines.append(f"- {row['customer_id']}: score {row['score']:.2f}; {row.get('reason_1', '')}")
                    elif "segment" in row:
                        lines.append(
                            f"- {row['segment']}: {row['customers']} customers, average score {row['avg_score']}"
                        )
                    elif "deals" in row:
                        lines.append(f"- {row['status']} deals: {row['deals']}")
                    elif "customers" in row:
                        lines.append(f"- {row.get('country', '')}: {row['customers']} customers")
                    elif "deal_id" in row:
                        lines.append(f"- {row['deal_id']}: {row['trucks']} trucks, trigger: {row['trigger']}")
            elif "customer" in r:
                s = r["score"]
                lines.append(
                    f"{r['customer']['customer_id']} has a score of {s['score']:.2f} (rank {s['rank']}). "
                    f"Main reason: {s['reason_1']}."
                )
        return "\n".join(lines) or "The data has no answer to that."


def _out_of_scope(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ("weather", "phone", "address", "delete", "ignore your rules", "drop table"))


class ScriptedAnthropic:
    """Drop-in for anthropic.Anthropic in tests: `.messages.create(...)`."""

    def __init__(self, invent_number_once: bool = False):
        self.messages = ScriptedMessages(invent_number_once)
