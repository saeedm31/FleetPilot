"""Model features and the plain-English reason behind each one."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sim import COUNTRY_NAMES, INDUSTRY_NAMES

NUMERIC = [
    "fleet_size",
    "our_share",
    "avg_age_years",
    "share_over_6y",
    "fin_end_6m_share",
    "contract_ends_6m",
    "km_per_truck_year",
    "service_cost_per_truck",
    "service_cost_trend",
    "breakdowns_3m",
    "visits_3m",
    "quotes_3m",
    "months_since_last_order",
    "orders_last_24m",
    "freight_index",
    "fleet_growth_12m",
    "telematics",
    "customer_years",
]
ONE_HOT = {"industry": INDUSTRY_NAMES, "country": COUNTRY_NAMES}


def add_label(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """y = 1 if the customer places an order with us in this month or the next horizon-1 months.
    Rows whose window runs past the end of the data get NaN (unknown yet)."""
    panel = panel.sort_values(["cid", "month"]).copy()
    fwd = panel.groupby("cid")["won"].transform(lambda s: s[::-1].rolling(horizon, min_periods=horizon).max()[::-1])
    panel["y"] = fwd
    return panel


def matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NUMERIC].astype(float).copy()
    for col, cats in ONE_HOT.items():
        for c in cats:
            x[f"{col}={c}"] = (df[col] == c).astype(float)
    return x


def feature_names() -> list[str]:
    return NUMERIC + [f"{col}={c}" for col, cats in ONE_HOT.items() for c in cats]


def reason_text(feature: str, value: float) -> str:
    """One short phrase a sales rep understands."""
    if feature.startswith(("industry=", "country=")):
        key, val = feature.split("=", 1)
        return f"{key}: {val}" if value else f"not {val}"
    v = value
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{feature.replace('_', ' ')}: unknown"
    texts = {
        "fleet_size": f"{v:.0f} trucks in the fleet",
        "our_share": f"our trucks are {v:.0%} of the fleet",
        "avg_age_years": f"fleet average age {v:.1f} years",
        "share_over_6y": f"{v:.0%} of trucks older than 6 years",
        "fin_end_6m_share": f"financing ends for {v:.0%} of the fleet within 6 months",
        "contract_ends_6m": "service contract ends within 6 months" if v else "service contract not due",
        "km_per_truck_year": f"{v:,.0f} km per truck per year",
        "service_cost_per_truck": f"service cost €{v:,.0f} per truck per month",
        "service_cost_trend": f"service cost {v:+.0%} vs a year ago",
        "breakdowns_3m": f"{v:.0f} breakdowns in the last 3 months",
        "visits_3m": f"{v:.0f} sales visits in the last 3 months",
        "quotes_3m": f"{v:.0f} quotes in the last 3 months",
        "months_since_last_order": f"last order {v:.0f} months ago",
        "orders_last_24m": f"{v:.0f} orders in the last 2 years",
        "freight_index": f"freight demand index {v:.2f}",
        "fleet_growth_12m": f"fleet {v:+.0%} in 12 months",
        "telematics": "uses our telematics" if v else "no telematics",
        "customer_years": f"customer for {v:.0f} years",
    }
    return texts.get(feature, f"{feature}: {v:.2f}")
