"""FleetPilot: who will renew their truck fleet, why, and what to say. ML + an AI sales copilot."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fleetpilot import config as C
from fleetpilot.agent.graph import run_path, structure, to_dot
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
SHARED_SESSION_EUR = float(os.getenv("DEMO_SESSION_BUDGET_EUR", "0.5"))  # per visitor
SHARED_DAILY_EUR = float(os.getenv("DEMO_DAILY_BUDGET_EUR", "3"))  # all visitors together


def _shared_key() -> str | None:
    """The app owner's key for the public demo, from Streamlit secrets or the environment. Never shown."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets file
        key = None
    return key or os.getenv("ANTHROPIC_API_KEY") or None


SHARED_KEY = None if COMPANY_MODE else _shared_key()


@st.cache_resource
def shared_spend() -> dict:
    """Spend on the shared key, across all visitors of this app process."""
    return {"day": "", "eur": 0.0}


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
    e = q("""SELECT s.customer_id, s.score, c.country, c.industry, c.segment, n.fleet_size, n.avg_age_years,
                    n.share_over_6y, n.our_share, n.km_per_truck_year, n.service_cost_per_truck, n.service_cost_trend,
                    n.fleet_growth_12m, n.breakdowns_3m, n.visits_3m, n.quotes_3m, n.months_since_last_order,
                    n.orders_last_24m
             FROM scores s JOIN customers c USING(customer_id) JOIN snapshot n USING(customer_id)""")
    explore = pd.DataFrame(  # plain column names, so charts label themselves
        {
            "Customer": e["customer_id"],
            "Country": e["country"],
            "Industry": e["industry"],
            "Customer group": e["segment"],
            "Chance to order (%)": (100 * e["score"]).round(1),
            "Trucks in fleet": e["fleet_size"],
            "Average truck age (years)": e["avg_age_years"].round(2),
            "Trucks over 6 years old (%)": (100 * e["share_over_6y"]).round(1),
            "Our share of their trucks (%)": (100 * e["our_share"]).round(1),
            "Km per truck per year": e["km_per_truck_year"].round(0),
            "Service cost per truck (€/month)": e["service_cost_per_truck"].round(0),
            "Service cost vs a year ago (%)": (100 * e["service_cost_trend"]).round(1),
            "Fleet growth, 12 months (%)": (100 * e["fleet_growth_12m"]).round(1),
            "Breakdowns, last 3 months": e["breakdowns_3m"],
            "Sales visits, last 3 months": e["visits_3m"],
            "Quotes, last 3 months": e["quotes_3m"],
            "Months since last order": e["months_since_last_order"],
            "Orders, last 2 years": e["orders_last_24m"],
        }
    )
    return tb, pipeline, names, hist, metrics, segments, explore


tb, pipeline, names, hist, metrics, segments, explore = load()
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
    st.markdown("**AI assistant settings**")
    if COMPANY_MODE:
        mode, api_key = "Live", None
        st.success(f"Claude via Amazon Bedrock · {S.aws_region}", icon="✅")
    else:
        modes = ["AI (Claude)"] if SHARED_KEY else []
        modes += ["Demo (scripted, no AI)", "Live (your Anthropic key)"]
        mode = st.radio("Mode", modes, index=0)
        api_key = None
        if mode.startswith("AI"):
            api_key = SHARED_KEY
            st.caption(f"Free to try: up to €{SHARED_SESSION_EUR:.2f} per visitor and €{SHARED_DAILY_EUR:.2f} per day.")
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
    if mode.startswith("AI"):
        budget = SHARED_SESSION_EUR
    else:
        budget = st.number_input(
            "Monthly budget (€)" if COMPANY_MODE else "Session budget (€)",
            0.0,
            1000.0,
            float(S.monthly_budget_eur),
            1.0,
        )
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    shared = shared_spend()
    if shared["day"] != today:
        shared.update(day=today, eur=0.0)
    month = datetime.now(UTC).strftime("%Y-%m")
    ends = [e for e in audit.read(limit=5000) if e["kind"] == "run_end"]
    month_eur = sum((e["payload"].get("cost_eur") or 0) for e in ends if e["ts"].startswith(month))
    spent = month_eur if COMPANY_MODE else ss.session_eur
    st.progress(min(spent / budget, 1.0) if budget else 0.0, text=f"Spent {eur(spent)} of {eur(budget)}")
    st.caption(f"Prices checked {PRICES_CHECKED}. Every answer shows its own cost.")

st.markdown("### Which customers should the sales team call this month, and why?")
st.info(
    "**How to use it:** **1. 📞 Who to call**: every customer ranked by how likely they are to order new trucks "
    "soon. **2. 🔍 One customer**: pick a customer to see why, and what to talk about. **3. 💬 Ask the AI**: ask "
    "questions in plain English. The other tabs let you make your own charts, see customer groups, check how much "
    "to trust the model, and read a log of every AI answer.",
    icon="👋",
)

TABS = ["📞 Who to call", "🔍 One customer", "💬 Ask the AI", "📈 Explore the data", "👥 Customer groups"]
tabs = st.tabs(TABS + ["📊 Can we trust it?", "🧾 Activity log"])
tab_pipe, tab_cust, tab_chat, tab_explore, tab_seg, tab_model, tab_audit = tabs


def label(cid: str) -> str:
    return f"{cid} · {names.get(cid, '?')}"


# --- pipeline ---
with tab_pipe:
    gb = metrics["test"]["Gradient boosting (calibrated)"]
    rule = metrics["test"]["Rule of thumb: oldest fleets first"]
    st.markdown(
        "The call list for this month. Customers at the top are the most likely to order new trucks from us in the "
        "next 6 months. **Call from the top down.** The last column says why."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(pipeline):,}")
    c2.metric("Orders we expect in 6 months", f"{pipeline['score'].sum():,.0f}", help="Sum of all chances.")
    c3.metric("Likely buyers (chance over 50%)", f"{(pipeline['score'] > 0.5).sum():,}")
    c4.metric(
        "Calling the top 10% reaches",
        f"{gb['buyers_found_top10']:.0%} of buyers",
        f"old rule (oldest fleets first): {rule['buyers_found_top10']:.0%}",
        delta_color="off",
        help="Tested on past months the model never saw.",
    )
    f1, f2, f3 = st.columns(3)
    country = f1.multiselect("Country", sorted(pipeline["country"].unique()))
    industry = f2.multiselect("Industry", sorted(pipeline["industry"].unique()))
    segment = f3.multiselect("Customer group", sorted(pipeline["segment"].unique()))
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
            "Customer group": v["segment"],
            "Trucks": v["fleet_size"],
            "Main reason": v["reason_1"],
        }
    )
    st.dataframe(view.head(300), hide_index=True, width="stretch", height=480)
    st.caption(
        f"Top 300 shown. Data as of {period}. Company names are shown only here, in your browser: the AI sees "
        "customer IDs only."
    )

# --- customer ---
with tab_cust:
    st.markdown("Pick a customer to see **why** they are likely to buy, and what worked with similar customers.")
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
    c4.metric("Customer group", row["segment"])
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Why this chance** (🔼 raises it, 🔽 lowers it)")
        reasons = json.loads(row["reasons_json"])
        for r in reasons:
            st.markdown(f"{'🔼' if r['effect'] > 0 else '🔽'} {r['text']}")
        st.markdown("**Recent deals**")
        st.dataframe(pd.DataFrame(prof["recent_deals"]), hide_index=True, width="stretch")
    with right:
        st.markdown("**Chance to order, month by month**")
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
        st.markdown(f"**Deals we won with similar customers** ({sim['industry']}): ideas for the pitch")
        st.dataframe(pd.DataFrame(sim["rows"]), hide_index=True, width="stretch")
    if st.button(f"Ask the AI about {cid}"):
        ss.prefill = f"Why is {cid} likely to buy, and what should I pitch? Draft a short first email."
        st.info("Question added: open the 💬 Ask the AI tab.")

# --- copilot ---
with tab_chat:
    st.markdown(
        "Ask about customers, groups or deals in plain English, e.g. *who to call in Spain*, *draft an email*, "
        "*how many deals did we lose*. The AI only reads the sales data (it can't change anything), and a **check** "
        "makes sure every number in its answer comes from that data."
    )
    graph = structure()
    with st.expander("🧭 How the AI works: the LangGraph flow", expanded=not ss.chat):
        last = next((m["res"] for m in reversed(ss.chat) if (m.get("res") or {}).get("run_id")), None)
        path = run_path(audit.read(run_id=last["run_id"])) if last else []
        st.graphviz_chart(to_dot(graph, path))
        st.caption(
            "**agent**: Claude reads the question and decides: fetch data with a tool, or answer. "
            "**tools**: read-only tools run (call list, customer profile, similar deals, groups, safe SQL). "
            "**check**: every number in the answer must come from the data; if not, Claude rewrites once. "
            "Dashed lines are decisions. "
            + ("**Orange: the path of your last question.** " if len(path) > 1 else "Ask a question to see its path. ")
            + f"Drawn from the {graph['source']} graph in `fleetpilot/agent/graph.py`."
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
        steps = run_path(audit.read(run_id=res["run_id"])) if res.get("run_id") else []
        if len(steps) > 1:
            names = {"__start__": "question", "__end__": "answer"}
            st.caption("Path: " + " → ".join(names.get(n, n) for n in steps))
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
            if mode.startswith("AI") and shared["eur"] >= SHARED_DAILY_EUR:
                st.error("Today's free AI budget is used up. Try again tomorrow, or use your own key in Live mode.")
            elif mode.startswith("AI") and ss.session_eur >= budget:
                st.error(
                    f"You have used your free {eur(budget)} of AI answers. Use your own key in Live mode for more."
                )
            elif budget and (month_eur if COMPANY_MODE else ss.session_eur) >= budget:
                st.error(
                    f"The {'monthly' if COMPANY_MODE else 'session'} budget of {eur(budget)} is reached. Raise it in the sidebar."
                )
            elif not (COMPANY_MODE or use_scripted or api_key):
                st.warning("Enter your Anthropic API key in the sidebar, or pick another mode.")
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
                        if mode.startswith("AI"):
                            shared["eur"] += r.cost["eur"] or 0.0
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
        "All customers split into **5 groups with similar fleets**, so the team can plan one campaign per group "
        "instead of 2,000 single calls. Each name says what sets the group apart. The bars show how likely each "
        "group is to order in the next 6 months."
    )
    st.caption("Method: k-means on fleet size, truck age, mileage, our share, fleet growth and service cost.")
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
        segments.rename(
            columns={
                "segment": "group",
                "trucks": "trucks in total",
                "avg_fleet": "avg trucks per customer",
                "avg_age_years": "avg truck age (years)",
                "our_share": "our share of their trucks",
                "main_industry": "main industry",
                "avg_score": "avg chance to order",
            }
        ),
        hide_index=True,
        width="stretch",
    )

# --- explore ---
with tab_explore:
    st.markdown(
        "Make your own charts from the customer data: pick a chart and the columns. **Hover** a point for details, "
        "**drag** to zoom, **double-click** to reset. Try *Scatter* of truck age vs chance to order: old fleets are "
        "not always the likely buyers."
    )
    cats = ["Country", "Industry", "Customer group"]
    nums = [c for c in explore.columns if c not in ["Customer", *cats]]
    k1, k2, k3, k4 = st.columns(4)
    kind = k1.selectbox("Chart", ["Scatter", "Histogram", "Box plot", "Bar (average)"])
    y = None
    if kind == "Scatter":
        x = k2.selectbox("X axis", nums, index=nums.index("Average truck age (years)"))
        y = k3.selectbox("Y axis", nums, index=nums.index("Chance to order (%)"))
    elif kind == "Histogram":
        x = k2.selectbox("Value", nums, index=nums.index("Chance to order (%)"), key="ex_hist_value")
    else:
        x = k2.selectbox("Group by", cats, index=1)
        y = k3.selectbox("Value", nums, index=nums.index("Chance to order (%)"), key="ex_group_value")
    color = k4.selectbox("Colour by", ["None", *cats])
    color = None if color == "None" or color == x else color
    f1, f2 = st.columns(2)
    ex_country = f1.multiselect("Country", sorted(explore["Country"].unique()), key="ex_country")
    ex_industry = f2.multiselect("Industry", sorted(explore["Industry"].unique()), key="ex_industry")
    d = explore
    if ex_country:
        d = d[d["Country"].isin(ex_country)]
    if ex_industry:
        d = d[d["Industry"].isin(ex_industry)]
    if kind == "Scatter":
        fig = px.scatter(d, x=x, y=y, color=color, hover_name="Customer", opacity=0.6)
    elif kind == "Histogram":
        fig = px.histogram(d, x=x, color=color, nbins=40, barmode="overlay", opacity=0.75)
    elif kind == "Box plot":
        fig = px.box(d, x=x, y=y, color=color, points=False)
    else:
        keys = [x] + ([color] if color else [])
        g = d.groupby(keys, as_index=False)[y].mean().round(2)
        fig = px.bar(g, x=x, y=y, color=color, barmode="group", labels={y: f"Average: {y}"})
    fig.update_layout(height=480, margin=dict(t=30), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(d):,} customers · data as of {period} · simulated data.")

# --- model report ---
with tab_model:
    m = metrics
    st.markdown(
        "**Can we trust the scores?** We trained the model on older months, then checked it on later months it had "
        "never seen, and compared it with simpler methods. Higher is better in every column."
    )
    st.caption(
        f"Details: will the customer order new trucks from us within {m['horizon_months']} months? "
        f"Trained on months {m['months']['train'][0]}-{m['months']['train'][1]}, calibrated on "
        f"{m['months']['calibration'][0]}-{m['months']['calibration'][1]}, tested on later months "
        f"{m['months']['test'][0]}-{m['months']['test'][1]} that the model never saw. Model: **{m['model_version']}**."
    )
    plain = {
        "Gradient boosting (calibrated)": "FleetPilot model",
        "Logistic regression": "Simpler model (logistic regression)",
        "Rule of thumb: oldest fleets first": "Old rule: oldest fleets first",
    }
    rows = [
        dict(
            Model=plain.get(k, k),
            AUC=round(v["auc"], 3),
            **{"PR-AUC": round(v["pr_auc"], 3)},
            **{"Top 10% finds": f"{v['buyers_found_top10']:.0%}", "Top 20% finds": f"{v['buyers_found_top20']:.0%}"},
        )
        for k, v in m["test"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Base rate: {gb['base_rate']:.0%} of customers order within 6 months. **Top 10% finds** = share of all buyers "
        "reached by calling the 10% highest-ranked customers. **AUC** = how well the list puts buyers above "
        "non-buyers (1.0 perfect, 0.5 random). **PR-AUC** = the same, focused on the buyers."
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
    left.markdown("**Does 30% really mean 30%?** The closer the blue line is to the dashed line, the better.")
    left.plotly_chart(fig, width="stretch")
    right.markdown("**AI assistant test**: 24 questions with known answers (`python -m fleetpilot eval --live`)")
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
        "Every question asked to the AI, every step it took, and what it cost, saved so it can be checked later. "
        "Records can be added but not changed."
    )
    st.markdown(
        ("✅ **Audit chain intact**" if v["ok"] else f"❌ **Audit chain broken at row {v.get('broken_at')}**")
        + f" · {v['rows']} events. Each row stores a hash of the previous one, so edits or deletions show up."
    )
    ends = [e for e in audit.read(limit=500) if e["kind"] == "run_end"]
    if not ends:
        st.caption("No questions logged yet. Ask one in the 💬 Ask the AI tab and it shows up here.")
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
