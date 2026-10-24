import sqlite3

import numpy as np

from fleetpilot import config as C
from fleetpilot.features import add_label
from fleetpilot.privacy import pseudonym
from fleetpilot.sim import simulate


def test_simulation_shapes_and_label_rate():
    customers, panel, deals = simulate(n=300, months=24)
    assert len(customers) == 300 and len(panel) == 300 * 24
    assert set(deals["status"]) <= {"won", "lost"}
    lab = add_label(panel, C.HORIZON)
    rate = lab["y"].mean()
    assert 0.05 < rate < 0.45
    assert lab.loc[lab["month"] > 24 - C.HORIZON, "y"].isna().all()  # unknown future stays unknown


def test_features_use_only_the_past():
    _, panel, _ = simulate(n=200, months=12)
    a = panel[panel["month"] == 5].set_index("cid")
    _, panel2, _ = simulate(n=200, months=6)  # same seed, data cut after month 5
    b = panel2[panel2["month"] == 5].set_index("cid")
    cols = ["fleet_size", "avg_age_years", "fin_end_6m_share", "quotes_3m", "service_cost_per_truck"]
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float))


def test_model_beats_rule_of_thumb_and_is_calibrated(built):
    _, m = built
    gb = m["test"]["Gradient boosting (calibrated)"]
    rule = m["test"]["Rule of thumb: oldest fleets first"]
    assert gb["auc"] > rule["auc"]
    assert gb["buyers_found_top10"] > rule["buyers_found_top10"]
    cal = m["calibration"]
    assert all(0 <= b["predicted"] <= 1 for b in cal)
    assert abs(np.mean([b["predicted"] for b in cal]) - np.mean([b["actual"] for b in cal])) < 0.1


def test_scores_have_reasons_and_no_names(built):
    s, _ = built
    con = sqlite3.connect(s.crm_db)
    rows = con.execute("SELECT customer_id, score, reason_1, reasons_json FROM scores").fetchall()
    assert len(rows) == 400
    assert all(r[0].startswith("C-") and 0 <= r[1] <= 1 and r[2] for r in rows)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        cols = [c[1] for c in con.execute(f"PRAGMA table_info({t})")]
        assert "company" not in cols and "cid" not in cols  # pseudonymised: no names, no internal IDs


def test_pseudonyms_are_stable_and_keyed():
    assert pseudonym(7, "k1") == pseudonym(7, "k1")
    assert pseudonym(7, "k1") != pseudonym(7, "k2")
    assert pseudonym(7, "k1").startswith("C-") and len(pseudonym(7, "k1")) == 8
