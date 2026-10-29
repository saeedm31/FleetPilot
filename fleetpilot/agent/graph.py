"""The copilot as a LangGraph state machine:

    question -> agent (Claude picks tools or answers) -> tools -> agent -> ... -> check -> answer
                                                                      ^             |
                                                                      +-- retry ----+  (numbers not in tool results)

Nodes are plain methods, so the same steps also run without LangGraph installed
(a small built-in runner follows the same edges). Every step is written to the audit log.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from ..audit import AuditLog
from ..llm import MeteredClient, Usage, total
from .check import check
from .prompts import SYSTEM
from .tools import Toolbox

try:
    from langgraph.graph import END, StateGraph

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    END, HAS_LANGGRAPH = "__end__", False


class State(TypedDict, total=False):
    run_id: str
    question: str
    messages: list
    answer: str
    pending_tools: bool
    tool_calls: list
    tool_results: list
    steps: int
    retries: int
    check: dict
    recheck: bool


@dataclass
class Result:
    run_id: str
    question: str
    answer: str
    check: dict
    tool_calls: list
    usage: list[Usage]
    cost: dict
    seconds: float
    engine: str
    steps: int = 0
    extra: dict = field(default_factory=dict)


def _block(b: Any) -> dict:
    """SDK content block -> plain dict (also accepts dicts, used by the scripted test model)."""
    if isinstance(b, dict):
        return b
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return {"type": b.type}


class Copilot:
    def __init__(
        self,
        client: MeteredClient,
        toolbox: Toolbox,
        model: str,
        period: str,
        audit: AuditLog | None = None,
        actor: str = "R-DEMO",
        max_steps: int = 8,
        max_tokens: int = 1500,
    ):
        self.client, self.toolbox, self.model, self.audit = client, toolbox, model, audit
        self.actor, self.max_steps, self.max_tokens = actor, max_steps, max_tokens
        self.system = SYSTEM.format(period=period)
        self.graph = self._compile() if HAS_LANGGRAPH else None

    # --- nodes ---
    def agent(self, state: State) -> State:
        t0 = time.time()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.toolbox.specs(),
            messages=state["messages"],
        )
        if state.get("steps", 0) >= self.max_steps:
            kwargs["tool_choice"] = {"type": "none"}  # out of steps: must answer now
        mark = len(self.client.meter.calls)
        resp = self.client.create(step="agent", **kwargs)
        content = [_block(b) for b in resp.content]
        text = "".join(b.get("text", "") for b in content if b["type"] == "text").strip()
        wants_tools = getattr(resp, "stop_reason", None) == "tool_use" and any(b["type"] == "tool_use" for b in content)
        u = self.client.meter.since(mark)
        self._log(
            state,
            "llm_call",
            dict(
                model=self.model,
                step=state.get("steps", 0),
                stop_reason=getattr(resp, "stop_reason", None),
                tool_requests=[b["name"] for b in content if b["type"] == "tool_use"],
                usage=[x.to_dict() for x in u],
                ms=int(1000 * (time.time() - t0)),
            ),
        )
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": content}],
            "answer": text,
            "pending_tools": wants_tools,
        }

    def tools(self, state: State) -> State:
        last = state["messages"][-1]["content"]
        results, calls, raw = [], list(state.get("tool_calls", [])), list(state.get("tool_results", []))
        for b in (x for x in last if x["type"] == "tool_use"):
            t0 = time.time()
            res, is_error = self.toolbox.run(b["name"], b.get("input") or {})
            payload = res if isinstance(res, str) else json.dumps(res, default=str)
            if len(payload) > 20_000:
                payload = payload[:20_000] + " ...[truncated]"
            ms = int(1000 * (time.time() - t0))
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": payload, "is_error": is_error})
            calls.append(
                dict(
                    name=b["name"],
                    input=b.get("input") or {},
                    error=is_error,
                    ms=ms,
                    rows=(res.get("row_count") if isinstance(res, dict) else None),
                )
            )
            if not is_error:
                raw.append(payload)
            self._log(
                state,
                "tool_call",
                dict(name=b["name"], input=b.get("input") or {}, error=is_error, ms=ms, result_preview=payload[:400]),
            )
        steps = state.get("steps", 0) + 1
        return {
            "messages": state["messages"] + [{"role": "user", "content": results}],
            "tool_calls": calls,
            "tool_results": raw,
            "steps": steps,
        }

    def verify(self, state: State) -> State:
        res = check(state.get("answer", ""), state.get("tool_results", []), state["question"])
        self._log(state, "check", res)
        if not res["ok"] and state.get("retries", 0) < 1:
            note = (
                "Check failed: these numbers do not appear in any tool result: "
                + ", ".join(res["unverified"])
                + ". Rewrite the full answer using only numbers from tool results (call a tool if you need one)."
            )
            return {
                "messages": state["messages"] + [{"role": "user", "content": note}],
                "retries": 1,
                "check": res,
                "recheck": True,
            }
        return {"check": res, "recheck": False}

    # --- wiring ---
    @staticmethod
    def _after_agent(state: State) -> str:
        return "tools" if state.get("pending_tools") else "check"

    @staticmethod
    def _after_check(state: State) -> str:
        return "agent" if state.get("recheck") else END

    def _compile(self):
        g = StateGraph(State)
        g.add_node("agent", self.agent)
        g.add_node("tools", self.tools)
        g.add_node("check", self.verify)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", self._after_agent, {"tools": "tools", "check": "check"})
        g.add_edge("tools", "agent")
        g.add_conditional_edges("check", self._after_check, {"agent": "agent", END: END})
        return g.compile()

    def _run_plain(self, state: State) -> State:
        """Same nodes and edges, without LangGraph (used where it isn't installed)."""
        node = "agent"
        for _ in range(4 * self.max_steps + 4):
            step = {"agent": self.agent, "tools": self.tools, "check": self.verify}[node]
            state = {**state, **step(state)}
            node = {"agent": self._after_agent, "tools": lambda s: "agent", "check": self._after_check}[node](state)
            if node == END:
                return state
        raise RuntimeError("agent did not finish")

    # --- entry point ---
    def ask(self, question: str, history: list | None = None) -> Result:
        run_id = uuid.uuid4().hex[:12]
        t0 = time.time()
        mark = len(self.client.meter.calls)
        state: State = {
            "run_id": run_id,
            "question": question,
            "steps": 0,
            "retries": 0,
            "messages": list(history or []) + [{"role": "user", "content": question}],
            "tool_calls": [],
            "tool_results": [],
        }
        self._log(
            state, "run_start", dict(question=question, model=self.model, engine="langgraph" if self.graph else "plain")
        )
        if self.graph is not None:
            state = self.graph.invoke(
                state,
                config={"recursion_limit": 4 * self.max_steps + 6, "run_name": "fleetpilot-copilot", "tags": [run_id]},
            )
        else:
            state = self._run_plain(state)
        usage = self.client.meter.since(mark)
        cost = total(usage)
        res = Result(
            run_id=run_id,
            question=question,
            answer=state.get("answer", ""),
            check=state.get("check", {}),
            tool_calls=state.get("tool_calls", []),
            usage=usage,
            cost=cost,
            seconds=round(time.time() - t0, 2),
            engine="langgraph" if self.graph else "plain",
            steps=state.get("steps", 0),
        )
        self._log(
            state,
            "run_end",
            dict(
                answer=res.answer,
                check=res.check,
                tools=[c["name"] for c in res.tool_calls],
                cost_eur=cost["eur"],
                input_tokens=cost["input"],
                output_tokens=cost["output"],
                seconds=res.seconds,
            ),
        )
        return res

    def _log(self, state: State, kind: str, payload: dict) -> None:
        if self.audit is not None:
            self.audit.write(kind, payload, run_id=state.get("run_id"), actor=self.actor)


# --- drawing the graph (for the app) ---
_FALLBACK_EDGES = [
    ("__start__", "agent", False),
    ("agent", "tools", True),
    ("agent", "check", True),
    ("tools", "agent", False),
    ("check", "agent", True),
    ("check", END, True),
]


def structure() -> dict:
    """Nodes and edges of the copilot graph, read from the compiled LangGraph graph when it is installed."""
    fallback = {"source": "built-in runner", "nodes": ["__start__", "agent", "tools", "check", END]}
    fallback["edges"] = _FALLBACK_EDGES
    if not HAS_LANGGRAPH:
        return fallback
    try:
        g = Copilot.__new__(Copilot)._compile().get_graph()  # nodes are never run here
    except Exception:  # drawing must never break the app
        return fallback
    edges = [(e.source, e.target, bool(e.conditional)) for e in g.edges]
    return {"source": "LangGraph", "nodes": list(g.nodes), "edges": edges}


def run_path(events: list[dict]) -> list[str]:
    """The nodes one run went through, rebuilt from its audit log events (oldest first)."""
    path = ["__start__"]
    for e in events:
        node = {"llm_call": "agent", "tool_call": "tools", "check": "check"}.get(e["kind"])
        if node and node != path[-1]:
            path.append(node)
    return path + [END] if path[-1] == "check" else path


_LABELS = {
    "__start__": "question",
    "agent": "agent\nClaude decides",
    "tools": "tools\nread the data",
    "check": "check\nevery number\nhas a source?",
    END: "answer",
}
_EDGE_LABELS = {
    ("agent", "tools"): "needs data",
    ("agent", "check"): "has an answer",
    ("check", "agent"): "number without source:\nrewrite once",
    ("check", END): "all verified",
}


def to_dot(struct: dict, path: list[str] | None = None) -> str:
    """Graphviz DOT for the graph; the nodes and edges of `path` are highlighted, with how often each edge was used."""
    path = path or []
    used: dict[tuple[str, str], int] = {}
    for a, b in zip(path, path[1:], strict=False):
        used[(a, b)] = used.get((a, b), 0) + 1
    lines = [
        "digraph {",
        '  rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.5;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#2E4566"];',
        '  node [fillcolor="#16253D", fontcolor="#F3F1EA"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#8FA3BF", fontcolor="#B8C2D1"];',
    ]
    for n in struct["nodes"]:
        shape = "oval" if n in ("__start__", END) else "box"
        hot = ', fillcolor="#F2A65A", fontcolor="#0B1424", color="#F2A65A"' if n in path else ""
        label = _LABELS.get(n, n).replace("\n", "\\n")
        lines.append(f'  "{n}" [label="{label}", shape={shape}{hot}];')
    for a, b, conditional in struct["edges"]:
        count = used.get((a, b), 0)
        label = _EDGE_LABELS.get((a, b), "").replace("\n", "\\n")
        if count > 1:
            label = f"{label} (x{count})".strip()
        attrs = [f'label="{label}"'] if label else []
        if conditional:
            attrs.append("style=dashed")
        if count:
            attrs += ['color="#F2A65A"', "penwidth=2.5", 'fontcolor="#F2A65A"']
        lines.append(f'  "{a}" -> "{b}" [{", ".join(attrs)}];')
    lines.append("}")
    return "\n".join(lines)
