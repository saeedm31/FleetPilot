"""FleetPilot: who will renew their truck fleet, why, and what to say. ML + an AI sales copilot."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fleetpilot import config as C
from fleetpilot.agent.scripted import ScriptedAnthropic
from fleetpilot.agent.tools import Toolbox
from fleetpilot.audit import AuditLog
from fleetpilot.llm import AGENT_MODELS, PRICES_CHECKED, Meter, eur
from fleetpilot.pipeline import ensure_built
from fleetpilot.recordings import load_all
from fleetpilot.service import copilot

st.set_page_config(page_title="FleetPilot", page_icon="🚚", layout="wide")
BLUE, ORANGE, GREY, GREEN = "#2F5BB7", "#D9692A", "#8A94A6", "#2E8B57"
S = C.settings()
COMPANY_MODE = S.llm_provider == "bedrock"


@st.cache_resource(show_spinner="Building the demo data and training the model (first start only, ~20 s)...")
def load():
    ensure_built(S)
    tb = Toolbox(S.crm_db)
    q = lambda sql: pd.read_sql_query(sql, tb.con)  # noqa: E731
    pipeline = q("""SELECT s.rank, s.customer_id, s.score, c.country, c.industry, c.segment, n.fleet_size,
                           n.avg_age_years, s.reason_1, s.reason_2, s.reason_3, s.reasons_json, s.model_version
                    FROM scores s JOIN customers c USING(customer_id) JOIN snapshot n USING(customer_id) ORDER BY s.rank""")
    names = pd.read_csv(S.data_dir / "contacts.csv").set_index("customer_id")["company"]
    hist = pd.read_csv(S.data_dir / "history.csv.gz")
    metrics = json.loads((S.data_dir / "metrics.json").read_text())
    segments = q("SELECT * FROM segments ORDER BY avg_score DESC")
    return tb, pipeline, names, hist, metrics, segments


tb, pipeline, names, hist, metrics, segments = load()
audit = AuditLog(S.audit_db)
period = pipeline.attrs.get("period") or hist["period"].max()
ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("meter", Meter())
ss.setdefault("session_eur", 0.0)

# --- sidebar ---
with st.sidebar:
    st.title("🚚 FleetPilot")
    st.caption("Which fleet customers will order new trucks in the next 6 months, why, and what to say to them.")
    st.warning(
        "**Simulated data** for a fictional truck maker: 2,000 fleet customers, 3 years. Company names are fake.",
        icon="🧪",
    )
    st.divider()
    st.markdown("**Copilot settings**")
    if COMPANY_MODE:
        mode, api_key = "Live", None
        st.success(f"Claude via Amazon Bedrock · {S.aws_region}", icon="✅")
    else:
        mode = st.radio("Mode", ["Demo (no key)", "Live (your Anthropic key)"], index=0)
        api_key = None
        if mode.startswith("Live"):
            api_key = st.text_input(
                "Anthropic API key", type="password", help="Used only for this session. Never stored or logged."
            )
    model = st.selectbox(
        "Model",
        list(AGENT_MODELS),
        format_func=AGENT_MODELS.get,
        index=list(AGENT_MODELS).index(S.agent_model) if S.agent_model in AGENT_MODELS else 0,
    )
    budget = st.number_input(
        "Monthly budget (€)" if COMPANY_MODE else "Session budget (€)", 0.0, 1000.0, float(S.monthly_budget_eur), 1.0
    )
    month = datetime.now(UTC).strftime("%Y-%m")
    ends = [e for e in audit.read(limit=5000) if e["kind"] == "run_end"]
    month_eur = sum((e["payload"].get("cost_eur") or 0) for e in ends if e["ts"].startswith(month))
    spent = month_eur if COMPANY_MODE else ss.session_eur
    st.progress(min(spent / budget, 1.0) if budget else 0.0, text=f"Spent {eur(spent)} of {eur(budget)}")
    st.caption(f"Prices checked {PRICES_CHECKED}. Every answer shows its own cost.")

tab_pipe, tab_cust, tab_chat, tab_seg, tab_model, tab_audit = st.tabs(
    ["📋 Pipeline", "🏢 Customer", "🤖 Copilot", "🧩 Segments", "📊 Model report", "🧾 Audit log"]
)


def label(cid: str) -> str:
    return f"{cid} · {names.get(cid, '?')}"


# --- pipeline ---
with tab_pipe:
    gb = metrics["test"]["Gradient boosting (calibrated)"]
    rule = metrics["test"]["Rule of thumb: oldest fleets first"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(pipeline):,}")
    c2.metric("Expected orders, next 6 months", f"{pipeline['score'].sum():,.0f}")
    c3.metric("Score > 50%", f"{(pipeline['score'] > 0.5).sum():,}")
    c4.metric(
        "Top 10% call list finds",
        f"{gb['buyers_found_top10']:.0%} of buyers",
        f"oldest-fleet rule: {rule['buyers_found_top10']:.0%}",
        delta_color="off",
    )
    f1, f2, f3 = st.columns(3)
    country = f1.multiselect("Country", sorted(pipeline["country"].unique()))
    industry = f2.multiselect("Industry", sorted(pipeline["industry"].unique()))
    segment = f3.multiselect("Segment", sorted(pipeline["segment"].unique()))
    v = pipeline
    for col, sel in (("country", country), ("industry", industry), ("segment", segment)):
        if sel:
            v = v[v[col].isin(sel)]
    view = pd.DataFrame(
        {
            "Rank": v["rank"],
            "Customer": v["customer_id"].map(label),
            "Chance to order": (100 * v["score"]).round(0).astype(int).astype(str) + "%",
            "Country": v["country"],
            "Industry": v["industry"],
            "Segment": v["segment"],
            "Trucks": v["fleet_size"],
            "Main reason": v["reason_1"],
        }
    )
    st.dataframe(view.head(300), hide_index=True, width="stretch", height=480)
    st.caption(
        f"Data as of {period}. Company names are shown only here, in your browser: the copilot and the AI model "
        "see customer IDs only."
    )

# --- customer ---
with tab_cust:
    cid = st.selectbox("Customer", pipeline["customer_id"].head(300), format_func=label)
    row = pipeline.set_index("customer_id").loc[cid]
    prof = tb.customer_profile(cid)
    snap = prof["snapshot"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Chance to order in 6 months",
        f"{row['score']:.0%}",
        f"rank {int(row['rank'])} of {len(pipeline):,}",
        delta_color="off",
    )
    c2.metric(
        "Fleet", f"{snap['fleet_size']:.0f} trucks", f"avg age {snap['avg_age_years']:.1f} years", delta_color="off"
    )
    c3.metric("Our share of the fleet", f"{snap['our_share']:.0%}")
    c4.metric("Segment", row["segment"])
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Why this score** (SHAP: what pushed the chance up or down)")
        reasons = json.loads(row["reasons_json"])
        for r in reasons:
            st.markdown(f"{'🔼' if r['effect'] > 0 else '🔽'} {r['text']}")
        st.markdown("**Recent deals**")
        st.dataframe(pd.DataFrame(prof["recent_deals"]), hide_index=True, width="stretch")
    with right:
        h = hist[hist["customer_id"] == cid].sort_values("month")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=h["period"], y=h["score"], line=dict(color=BLUE, width=3), name="Chance to order (model)")
        )
        won = h[h["won"] == 1]
        lost = h[h["lost"] == 1]
        fig.add_trace(
            go.Scatter(
                x=won["period"],
                y=won["score"],
                mode="markers",
                marker=dict(size=13, color=GREEN, symbol="star"),
                name="Order won",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lost["period"],
                y=lost["score"],
                mode="markers",
                marker=dict(size=11, color=ORANGE, symbol="x"),
                name="Lost to competitor",
            )
        )
        fig.update_layout(
            height=320,
            yaxis=dict(title="chance (0-1)", range=[0, 1]),
            margin=dict(t=20),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, width="stretch")
        sim = tb.similar_won_deals(cid)
        st.markdown(f"**Similar won deals** ({sim['industry']}, similar fleet size)")
        st.dataframe(pd.DataFrame(sim["rows"]), hide_index=True, width="stretch")
    if st.button(f"Ask the copilot about {cid}"):
        ss.prefill = f"Why is {cid} likely to buy, and what should I pitch? Draft a short first email."
        st.info("Question added: open the 🤖 Copilot tab.")

# --- copilot ---
with tab_chat:
    st.markdown(
        "Ask about customers, segments or deals. The copilot (LangGraph + Claude) uses read-only tools on the CRM, "
        "and a **check step** makes sure every number in its answer comes from a tool result."
    )
    recs = load_all()
    use_scripted = False
    if not COMPANY_MODE and mode.startswith("Demo"):
        if recs:
            st.info(
                "Demo mode: replaying real recorded runs. Switch to Live in the sidebar to ask your own questions.",
                icon="▶️",
            )
        choice = st.radio(
            "Demo",
            ["Recorded real runs", "Scripted test model (not AI: shows the tools, check, log and cost)"]
            if recs
            else ["Scripted test model (not AI: shows the tools, check, log and cost)"],
            horizontal=True,
        )
        use_scripted = choice.startswith("Scripted")

    def show(res: dict):
        chk = res["check"]
        badge = (
            "✅ all numbers verified against tool results"
            if chk.get("ok")
            else f"⚠️ not verified: {', '.join(chk.get('unverified', []))}"
        )
        c = res["cost"]
        st.caption(
            f"{badge} · {c['calls']} model call(s), {c['input']:,} tokens in / {c['output']:,} out · "
            f"**{eur(c['eur'])}** · {res['seconds']} s" + (f" · run {res['run_id']}" if res.get("run_id") else "")
        )
        with st.expander(f"Tools used ({len(res['tool_calls'])})"):
            for t in res["tool_calls"]:
                st.code(
                    f"{t['name']}({json.dumps(t['input'])})  -> rows: {t.get('rows')}  {t['ms']} ms"
                    + ("  ERROR" if t.get("error") else ""),
                    language="text",
                )

    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("res"):
                show(m["res"])

    if not COMPANY_MODE and mode.startswith("Demo") and recs and not use_scripted:
        pick = st.selectbox("Recorded question", [r["question"] for r in recs])
        if st.button("Replay"):
            r = next(x for x in recs if x["question"] == pick)
            ss.chat += [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"], "res": r},
            ]
            st.rerun()
    else:
        q = st.chat_input("e.g. Which customers in Portugal should I call first this week, and why?")
        q = q or ss.pop("prefill", None)
        if q:
            if budget and (month_eur if COMPANY_MODE else ss.session_eur) >= budget:
                st.error(
                    f"The {'monthly' if COMPANY_MODE else 'session'} budget of {eur(budget)} is reached. Raise it in the sidebar."
                )
            elif not (COMPANY_MODE or use_scripted or api_key):
                st.warning("Enter your Anthropic API key in the sidebar, or use Demo mode.")
            else:
                ok = False
                with st.spinner("Thinking, calling tools, checking numbers..."):
                    try:
                        cp = copilot(
                            S,
                            client=ScriptedAnthropic() if use_scripted else None,
                            api_key=api_key,
                            model=model,
                            meter=ss.meter,
                            actor="R-DEMO",
                        )
                        r = cp.ask(q)
                        res = dict(
                            answer=r.answer,
                            check=r.check,
                            tool_calls=r.tool_calls,
                            cost=r.cost,
                            seconds=r.seconds,
                            run_id=r.run_id,
                        )
                        ss.session_eur += r.cost["eur"] or 0.0
                        ss.chat += [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": r.answer, "res": res},
                        ]
                        ok = True
                    except Exception as exc:
                        st.error(f"The copilot failed: {type(exc).__name__}: {exc}")
                if ok:
                    st.rerun()
    if ss.chat and st.button("Clear chat"):
        ss.chat = []
        st.rerun()

# --- segments ---
with tab_seg:
    st.markdown(
        "Five groups of customers with similar fleets (k-means on fleet size, age, mileage, our share, growth "
        "and service cost). Named from what sets each group apart."
    )
    fig = go.Figure(
        go.Bar(
            x=segments["avg_score"],
            y=segments["segment"],
            orientation="h",
            marker_color=BLUE,
            text=(100 * segments["avg_score"]).round(0).astype(int).astype(str) + "%",
        )
    )
    fig.update_layout(
        height=300,
        xaxis_title="average chance to order in 6 months",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=20),
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        segments.rename(columns={"avg_fleet": "avg trucks", "our_share": "our share", "avg_score": "avg chance"}),
        hide_index=True,
        width="stretch",
    )

# --- model report ---
with tab_model:
    m = metrics
    st.markdown(
        f"**Question:** will the customer order new trucks from us within {m['horizon_months']} months? "
        f"Trained on months {m['months']['train'][0]}-{m['months']['train'][1]}, calibrated on "
        f"{m['months']['calibration'][0]}-{m['months']['calibration'][1]}, tested on later months "
        f"{m['months']['test'][0]}-{m['months']['test'][1]} that the model never saw. Model: **{m['model_version']}**."
    )
    rows = [
        dict(
            Model=k,
            AUC=round(v["auc"], 3),
            **{"PR-AUC": round(v["pr_auc"], 3)},
            **{"Top 10% finds": f"{v['buyers_found_top10']:.0%}", "Top 20% finds": f"{v['buyers_found_top20']:.0%}"},
        )
        for k, v in m["test"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Base rate: {gb['base_rate']:.0%} of customers order within 6 months. 'Top 10% finds' = share of all buyers "
        "found by calling the 10% highest-scored customers."
    )
    cal = pd.DataFrame(m["calibration"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=GREY, dash="dash"), name="perfect"))
    fig.add_trace(
        go.Scatter(
            x=cal["predicted"], y=cal["actual"], mode="lines+markers", line=dict(color=BLUE, width=3), name="model"
        )
    )
    fig.update_layout(
        height=320, xaxis_title="predicted chance", yaxis_title="share that actually ordered", margin=dict(t=20)
    )
    left, right = st.columns(2)
    left.markdown("**Calibration:** does 30% mean 30%?")
    left.plotly_chart(fig, width="stretch")
    right.markdown("**Copilot evaluation** (`python -m fleetpilot eval --live`)")
    try:
        ev = json.loads((C.ROOT / "eval_report.json").read_text())
        right.metric("Questions passed", f"{ev['passed']} / {ev['cases']}", f"model {ev['model']}", delta_color="off")
        right.caption(
            f"Right tool: {ev['tools_ok']}/{ev['cases']} · facts found: {ev['facts_ok']}/{ev['cases']} · "
            f"grounded: {ev['grounded']}/{ev['cases']} · cost per question {eur(ev['cost_eur_per_question'])}"
        )
    except (OSError, ValueError, KeyError):
        right.info("No live evaluation report yet. Run it with an API key to fill this in.")
    st.info("Simulated data: these numbers show the method, not a real sales team's results.", icon="ℹ️")

# --- audit ---
with tab_audit:
    v = audit.verify()
    st.markdown(
        ("✅ **Audit chain intact**" if v["ok"] else f"❌ **Audit chain broken at row {v.get('broken_at')}**")
        + f" · {v['rows']} events. Each row stores a hash of the previous one, so edits or deletions show up."
    )
    ends = [e for e in audit.read(limit=500) if e["kind"] == "run_end"]
    if not ends:
        st.caption("No copilot runs logged yet in this deployment.")
    else:
        starts = {
            e["run_id"]: e["payload"].get("question", "") for e in audit.read(limit=2000) if e["kind"] == "run_start"
        }
        runs = pd.DataFrame(
            [
                dict(
                    time=e["ts"][:19],
                    run=e["run_id"],
                    rep=e["actor"],
                    question=starts.get(e["run_id"], ""),
                    verified=e["payload"].get("check", {}).get("ok"),
                    cost=eur(e["payload"].get("cost_eur")),
                    seconds=e["payload"].get("seconds"),
                )
                for e in ends[:50]
            ]
        )
        st.dataframe(runs, hide_index=True, width="stretch")
        pick = st.selectbox("Show every step of run", runs["run"])
        for e in audit.read(run_id=pick):
            with st.expander(f"{e['ts'][11:23]}  {e['kind']}"):
                st.json(e["payload"])
