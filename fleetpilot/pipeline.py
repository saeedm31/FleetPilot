"""Build everything the app and the copilot read: simulate -> train -> score -> segment -> CRM database.

python -m fleetpilot build
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import NUMERIC, add_label
from .model import LogisticBaseline, PurchaseModel, calibration_table, evaluate, rule_of_thumb
from .privacy import pseudonym
from .segments import Segmenter
from .sim import simulate
from .store import write_db

log = logging.getLogger("fleetpilot.pipeline")


def build(settings: C.Settings | None = None, seed: int = C.SEED, n: int = C.N_CUSTOMERS) -> dict:
    s = settings or C.settings()
    t0 = time.time()
    out = Path(s.data_dir)
    out.mkdir(parents=True, exist_ok=True)

    # one extra month: "today" is the start of the month after the last full month of history,
    # so this month's orders are already in the snapshot (a customer who just ordered isn't top of the list)
    customers, panel, deals = simulate(seed=seed, n=n, months=C.N_MONTHS + 1)
    now_month = C.N_MONTHS
    deals = deals[deals["month"] < now_month].reset_index(drop=True)
    panel.loc[panel["month"] >= now_month, ["won", "lost"]] = 0  # the future is unknown
    panel = add_label(panel, C.HORIZON)
    log.info("simulated %d customers, %d snapshots, %d deals", len(customers), len(panel), len(deals))

    train = panel[panel["month"].between(*C.TRAIN_MONTHS)]
    calib = panel[panel["month"].between(*C.CALIB_MONTHS)]
    test = panel[panel["month"].between(*C.TEST_MONTHS)]
    model = PurchaseModel(seed=seed).fit(train, calib)
    version = f"purchase-{model.backend}-v1"

    scores = {
        "Gradient boosting (calibrated)": model.predict(test),
        "Logistic regression": LogisticBaseline().fit(train).predict(test),
        "Rule of thumb: oldest fleets first": rule_of_thumb(test),
    }
    metrics = dict(
        model_version=version,
        backend=model.backend,
        horizon_months=C.HORIZON,
        months=dict(train=C.TRAIN_MONTHS, calibration=C.CALIB_MONTHS, test=C.TEST_MONTHS),
        rows=dict(train=len(train), calibration=len(calib), test=len(test)),
        test={k: evaluate(test["y"], v, calibrated=k.startswith("Gradient")) for k, v in scores.items()},
        calibration=calibration_table(test["y"].to_numpy(), scores["Gradient boosting (calibrated)"]),
    )
    log.info(
        "test AUC %.3f, top-10%% finds %.0f%% of buyers",
        metrics["test"]["Gradient boosting (calibrated)"]["auc"],
        100 * metrics["test"]["Gradient boosting (calibrated)"]["buyers_found_top10"],
    )

    # --- today: score every customer and explain it ---
    now = panel[panel["month"] == now_month].set_index("cid").sort_index()
    now["score"] = model.predict(now)
    reasons = model.reasons(now, k=3)
    seg = Segmenter(seed=seed).fit(now)
    now["segment"] = seg.predict(now)
    pid = {i: pseudonym(i, s.pseudonym_key) for i in customers["cid"]}

    cust = customers.assign(
        customer_id=customers["cid"].map(pid), segment=now["segment"].reindex(customers["cid"]).to_numpy()
    )
    snapshot = now[NUMERIC].round(3).assign(customer_id=[pid[i] for i in now.index], period=now["period"])
    order = now["score"].rank(ascending=False, method="first").astype(int)
    score_tbl = pd.DataFrame(
        dict(
            customer_id=[pid[i] for i in now.index],
            period=now["period"],
            score=now["score"].round(4).to_numpy(),
            rank=order.to_numpy(),
            model_version=version,
            **{
                f"reason_{j + 1}": [
                    (f"{r[j]['text']} ({r[j]['direction']} the chance)" if j < len(r) else "") for r in reasons
                ]
                for j in range(3)
            },
            reasons_json=[json.dumps(r) for r in reasons],
        )
    )
    deal_tbl = deals.assign(deal_id=[f"D-{i + 1:05d}" for i in range(len(deals))], customer_id=deals["cid"].map(pid))[
        ["deal_id", "customer_id", "period", "status", "trucks", "value_eur", "trigger"]
    ]
    profile = seg.profile(now.reset_index(), list(now["segment"]))
    profile["avg_score"] = profile["segment"].map(now.groupby("segment")["score"].mean()).round(3)

    write_db(
        s.crm_db,
        dict(
            customers=cust[["customer_id", "country", "industry", "dealer", "customer_since", "segment"]],
            snapshot=snapshot[["customer_id", "period"] + NUMERIC],
            scores=score_tbl,
            deals=deal_tbl,
            segments=profile,
        ),
    )

    # rep-side only: real (here: fake) company names, never given to the agent or the LLM
    cust[["customer_id", "company"]].to_csv(out / "contacts.csv", index=False)
    # score history for charts: last 11 months, with what actually happened
    hist = panel[panel["month"] >= C.TEST_MONTHS[0]].copy()
    hist["score"] = model.predict(hist)
    hist.assign(customer_id=hist["cid"].map(pid))[
        ["customer_id", "period", "month", "score", "won", "lost", "avg_age_years", "fleet_size"]
    ].round(4).to_csv(out / "history.csv.gz", index=False)
    metrics["segments"] = profile.to_dict("records")
    metrics["built_seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    joblib.dump(dict(model=model, segmenter=seg, version=version), out / "model.joblib", compress=3)
    _mlflow_log(s, model, metrics, out)
    return metrics


def ensure_built(settings: C.Settings | None = None) -> C.Settings:
    s = settings or C.settings()
    if not (s.crm_db.exists() and (s.data_dir / "metrics.json").exists()):
        build(s)
    return s


def _mlflow_log(s: C.Settings, model: PurchaseModel, metrics: dict, out: Path) -> None:
    """Track the run and register the model, when an MLflow server is configured."""
    if not s.mlflow_uri:
        return
    try:
        import mlflow
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed; skipping tracking")
        return
    mlflow.set_tracking_uri(s.mlflow_uri)
    mlflow.set_experiment("fleetpilot-purchase-likelihood")
    gb = metrics["test"]["Gradient boosting (calibrated)"]
    with mlflow.start_run(run_name=metrics["model_version"]):
        mlflow.log_params(
            dict(
                backend=model.backend,
                horizon_months=metrics["horizon_months"],
                train_months=str(metrics["months"]["train"]),
                test_months=str(metrics["months"]["test"]),
            )
        )
        mlflow.log_metrics({k: v for k, v in gb.items() if isinstance(v, float)})
        mlflow.log_artifact(str(out / "metrics.json"))
        mlflow.log_artifact(str(out / "model.joblib"))
        if model.backend == "xgboost":
            try:
                mlflow.xgboost.log_model(model.model_, name="model", registered_model_name="fleetpilot-purchase")
            except Exception as exc:  # registry is optional for the demo
                log.warning("model registry step skipped: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build()["test"], indent=1))
    np.seterr(all="ignore")
