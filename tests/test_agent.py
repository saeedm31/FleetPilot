import sqlite3

import pytest

from fleetpilot.agent.check import check
from fleetpilot.agent.graph import HAS_LANGGRAPH, run_path, structure, to_dot
from fleetpilot.agent.scripted import ScriptedAnthropic
from fleetpilot.agent.tools import Toolbox
from fleetpilot.audit import AuditLog
from fleetpilot.evaluation import run as run_eval
from fleetpilot.service import copilot
from fleetpilot.store import SqlError, read_only, run_select


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE scores SET score = 1",
        "DROP TABLE deals",
        "SELECT 1; DROP TABLE deals",
        "ATTACH DATABASE 'x.db' AS x",
        "PRAGMA table_info(customers)",
        "SELECT * FROM sqlite_master",
    ],
)
def test_agent_sql_is_read_only(built, sql):
    s, _ = built
    with pytest.raises(SqlError):
        run_select(read_only(s.crm_db), sql)


def test_select_works_and_is_capped(built):
    s, _ = built
    df = run_select(read_only(s.crm_db), "SELECT customer_id FROM customers", max_rows=50)
    assert len(df) == 50 and df.attrs["truncated"]


def test_grounding_check():
    tools = ['{"rows": [{"customer_id": "C-1A2B3C", "score": 0.873, "fleet_size": 42, "value_eur": 1234567}]}']
    assert check("C-1A2B3C: 87% chance, 42 trucks, deal worth €1,234,567", tools)["ok"]
    assert check("Score 0.87 and about 1.2 million euros", tools)["ok"]
    bad = check("C-1A2B3C will order 55 trucks", tools)
    assert not bad["ok"] and bad["unverified"] == ["55"]


def test_copilot_runs_tools_checks_and_logs(built):
    s, _ = built
    cp = copilot(s, client=ScriptedAnthropic())
    r = cp.ask("Which customers in Germany should I call first?")
    assert [c["name"] for c in r.tool_calls] == ["top_customers"]
    assert r.check["ok"] and "C-" in r.answer
    assert r.cost["calls"] == 2 and r.cost["eur"] > 0
    assert r.engine == ("langgraph" if HAS_LANGGRAPH else "plain")
    kinds = [e["kind"] for e in AuditLog(s.audit_db).read(run_id=r.run_id)]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and "tool_call" in kinds and "check" in kinds


def test_check_step_sends_invented_numbers_back(built):
    s, _ = built
    cp = copilot(s, client=ScriptedAnthropic(invent_number_once=True))
    r = cp.ask("Which customers in Spain should I call first?")
    assert r.check["ok"] and "1234567" not in r.answer  # rewritten after the check failed once
    checks = [e["payload"] for e in AuditLog(s.audit_db).read(run_id=r.run_id) if e["kind"] == "check"]
    assert [c["ok"] for c in checks] == [False, True]


def test_audit_log_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.db")
    for i in range(5):
        log.write("event", {"i": i}, run_id="r1", actor="R-TEST")
    assert log.verify() == {"ok": True, "rows": 5}
    with sqlite3.connect(tmp_path / "audit.db") as con:
        con.execute("""UPDATE events SET payload='{"i": 99}' WHERE seq=3""")
    assert log.verify()["broken_at"] == 3


def test_tools_reject_bad_ids(built):
    s, _ = built
    res, err = Toolbox(s.crm_db).run("customer_profile", {"customer_id": "'; DROP TABLE x; --"})
    assert err and "customer_id" in res


def test_offline_evaluation_wiring(built):
    s, _ = built
    rep = run_eval(copilot(s, client=ScriptedAnthropic()), s.crm_db)
    assert rep["cases"] >= 20
    assert rep["tools_ok"] == rep["cases"] and rep["grounded"] == rep["cases"]


def test_scripted_demo_answers_biggest_fleets_and_admits_what_it_cannot(built):
    s, _ = built
    cp = copilot(s, client=ScriptedAnthropic())
    r = cp.ask("Which customer has the most trucks in our db?")
    assert [c["name"] for c in r.tool_calls] == ["run_sql"] and "fleet_size DESC" in r.tool_calls[0]["input"]["query"]
    assert r.check["ok"] and "trucks" in r.answer
    r = cp.ask("What colour are the trucks of our customers?")
    assert r.tool_calls == [] and "doesn't understand" in r.answer


def test_graph_drawing_and_run_path(built):
    s, _ = built
    g = structure()
    assert {"agent", "tools", "check"} <= set(g["nodes"])
    assert ("tools", "agent") in [(a, b) for a, b, _ in g["edges"]]
    cp = copilot(s, client=ScriptedAnthropic(invent_number_once=True))
    r = cp.ask("Which customers in Spain should I call first?")
    path = run_path(AuditLog(s.audit_db).read(run_id=r.run_id))
    assert path == ["__start__", "agent", "tools", "agent", "check", "agent", "check", "__end__"]
    dot = to_dot(g, path)
    assert dot.startswith("digraph") and "(x2)" in dot and "#D9692A" in dot
