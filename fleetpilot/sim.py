"""Simulated sales data of a truck maker selling to fleet customers.

Every month each customer's fleet gets older, costs more to run and breaks down more often.
Financing contracts and service contracts end. Sales reps visit and send quotes. Freight
demand goes up and down. From all that, a hidden rule decides whether the customer orders
new trucks this month, and whether we win the order or a competitor does.

The models only see what a real sales team would see (fleet, service, contract and sales
data). The hidden rule and the per-customer random effect stay inside the simulator, so the
models can be scored honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

MAX_AGE = 200  # months tracked per truck age bucket
SHARP, SHIFT = 3.0, 4.5  # how clearly the drivers decide an order (vs luck)
BURN_IN = 36  # months simulated before the data starts, so fleets reach a steady state

INDUSTRY_NAMES = list(C.INDUSTRIES)
COUNTRY_NAMES = list(C.COUNTRIES)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(seed: int = C.SEED, n: int = C.N_CUSTOMERS, months: int = C.N_MONTHS):
    """Returns (customers, panel, deals).
    customers: one row per customer (static facts, internal id and fake company name)
    panel:     one row per customer per month: what the company knows at the start of the month,
               plus the outcome (won/lost order this month) used to build labels
    deals:     every order event (won or lost) with trucks and value
    """
    rng = np.random.default_rng(seed)
    total = months + BURN_IN
    month_index = pd.period_range(pd.Period(C.START_MONTH, "M") - BURN_IN, periods=total, freq="M")

    # --- customers ---
    ind = rng.choice(len(INDUSTRY_NAMES), n, p=[v[0] for v in C.INDUSTRIES.values()])
    ctry = rng.choice(len(COUNTRY_NAMES), n, p=list(C.COUNTRIES.values()))
    med = np.array([C.INDUSTRIES[INDUSTRY_NAMES[i]][1] for i in ind])
    fleet = np.clip(np.round(rng.lognormal(np.log(med), 0.7)), 3, 600).astype(int)
    km_year = np.array([C.INDUSTRIES[INDUSTRY_NAMES[i]][2] for i in ind]) * rng.lognormal(0, 0.15, n)
    trend = np.array([C.INDUSTRIES[INDUSTRY_NAMES[i]][3] for i in ind]) + rng.normal(0, 0.03, n)
    our_share = np.clip(rng.beta(4, 2.5, n), 0.1, 1.0)
    financed = rng.uniform(0.3, 0.95, n)  # share of trucks bought with financing
    customer_since = rng.integers(1995, 2023, n)
    telematics = rng.random(n) < 0.6
    dealer = np.array([f"{COUNTRY_NAMES[c]}-{rng.integers(1, 6)}" for c in ctry])
    hidden = rng.normal(0, 0.35, n)  # what the data can't explain
    contract_end = rng.integers(0, 36, n)  # month the service contract renews
    rep_attention = rng.lognormal(0, 0.3, n)

    # fleet by age in months: each customer bought trucks in a few past batches
    ages = np.zeros((n, MAX_AGE), dtype=np.int32)
    for i in range(n):
        left = fleet[i]
        while left > 0:
            k = min(left, max(1, int(rng.integers(1, max(2, fleet[i] // 2 + 1)))))
            ages[i, int(rng.integers(0, 130))] += k
            left -= k

    horizon = 240  # fixed length, so a shorter run draws the same random numbers for the months it has
    freight = 1.0 + 0.04 * np.sin(np.arange(horizon) / 6.0) + np.cumsum(rng.normal(0, 0.008, horizon))
    country_freight = freight[None, :] + rng.normal(0, 0.02, (len(COUNTRY_NAMES), 1))

    last_order = -rng.integers(6, 60, n).astype(float)
    order_log = [[] for _ in range(n)]
    fleet_hist = []
    cost_hist = []
    rows, deals = [], []
    age_years = (np.arange(MAX_AGE) + 0.5) / 12.0

    for m, period in enumerate(month_index):
        rec = m - BURN_IN  # month index in the published data
        size = ages.sum(axis=1).astype(float)
        avg_age = (ages * age_years).sum(axis=1) / np.maximum(size, 1)
        over6 = ages[:, 72:].sum(axis=1) / np.maximum(size, 1)
        fin_end = financed * ages[:, 54:60].sum(axis=1) / np.maximum(size, 1)
        contract_6m = ((contract_end - m) % 36) < 6
        cost = 350 * (1 + 0.10 * avg_age**1.3) * (km_year / 80_000) ** 0.5 * rng.lognormal(0, 0.08, n)
        cost_hist.append(cost)
        cost_12m = cost_hist[max(0, m - 12)]
        breakdowns = rng.poisson(size * 0.006 * (1 + 0.3 * avg_age) * 3)
        growth = size / np.maximum(fleet_hist[max(0, m - 12)] if fleet_hist else size, 1) - 1
        fleet_hist.append(size.copy())
        fr = country_freight[ctry, m]

        since = m - last_order
        old_fleet = _sigmoid(2.5 * (avg_age - 5.5))
        need = (
            -5.0
            + 1.6 * old_fleet
            + 1.2 * old_fleet * (cost / np.maximum(cost_12m, 1) > 1.08)
            + 3.5 * fin_end * (avg_age > 4)
            + 0.9 * contract_6m
            + 0.06 * breakdowns
            + 1.6 * np.clip(cost / np.maximum(cost_12m, 1) - 1, -0.3, 0.6)
            + 5.0 * (fr - 1)
            + 2.2 * np.clip(growth, -0.3, 0.5) * (fr > 1.0)
            + 0.3 * np.log(size)
            + 2.5 * trend
            + hidden
            + 0.3 * np.isin(period.month, [1, 2, 11, 12])
            - 1.6 * (since < 9)
            + 0.6 * ((since > 24) & (since < 48))
        )
        visits = rng.poisson(0.4 * rep_attention * (1 + 0.15 * np.log(size)) + 0.6 * _sigmoid(need + 2.0))
        quotes = rng.poisson(1.6 * _sigmoid(need + 1.6) + 0.05)
        p_order = _sigmoid(SHARP * (need + 5.0) - 5.0 - SHIFT + 0.35 * np.minimum(quotes, 3))
        orders = rng.random(n) < p_order
        p_win = _sigmoid(
            -1.2
            + 4.0 * (our_share > 0.45) * our_share
            + 0.8 * (quotes > 0) * (our_share > 0.3)
            + 0.25 * (visits > 1)
            + 0.02 * (2023 - customer_since)
            + 0.3 * telematics
            + rng.normal(0, 0.4, n)
        )
        won = orders & (rng.random(n) < p_win)
        lost = orders & ~won

        for i in range(n) if rec >= 0 else []:
            rows.append(
                dict(
                    cid=i,
                    month=rec,
                    period=str(period),
                    fleet_size=int(size[i]),
                    our_share=our_share[i],
                    avg_age_years=avg_age[i],
                    share_over_6y=over6[i],
                    fin_end_6m_share=fin_end[i],
                    contract_ends_6m=int(contract_6m[i]),
                    km_per_truck_year=km_year[i] if telematics[i] else np.nan,
                    service_cost_per_truck=cost[i],
                    service_cost_trend=cost[i] / max(cost_12m[i], 1) - 1,
                    breakdowns_3m=int(breakdowns[i]),
                    visits_3m=int(visits[i]),
                    quotes_3m=int(quotes[i]),
                    months_since_last_order=float(min(m - last_order[i], 60)),
                    orders_last_24m=sum(1 for o in order_log[i] if m - o <= 24),
                    freight_index=fr[i],
                    fleet_growth_12m=growth[i],
                    telematics=int(telematics[i]),
                    customer_years=2023 - customer_since[i] + rec / 12.0,
                    won=int(won[i]),
                    lost=int(lost[i]),
                    p_order_true=p_order[i],
                )
            )

        # orders change the fleet: old trucks out, new ones in
        for i in np.where(orders)[0]:
            old = ages[i, 60:].sum()
            k_replace = int(min(max(1, round(old * rng.uniform(0.3, 0.8))), size[i]))
            k_grow = (
                int(round(size[i] * min(max(trend[i], 0), 0.1) * rng.uniform(0.5, 2))) if size[i] < 3 * fleet[i] else 0
            )
            k = k_replace + k_grow
            removed, a = 0, MAX_AGE - 1
            while removed < k_replace and a >= 0:
                take = min(ages[i, a], k_replace - removed)
                ages[i, a] -= take
                removed += take
                a -= 1
            ages[i, 0] += k
            if rec >= 0:
                deals.append(
                    dict(
                        cid=i,
                        month=rec,
                        period=str(period),
                        status="won" if won[i] else "lost",
                        trucks=k,
                        value_eur=int(k * C.TRUCK_PRICE_EUR * rng.uniform(0.9, 1.1)),
                        trigger=_trigger(avg_age[i], fin_end[i], contract_6m[i], growth[i], breakdowns[i]),
                    )
                )
            last_order[i] = m
            order_log[i].append(m)
            if lost[i]:
                our_share[i] = max(0.05, our_share[i] * (1 - 0.5 * k / max(size[i], 1)))
            else:
                our_share[i] = min(1.0, our_share[i] + (1 - our_share[i]) * 0.5 * k / max(size[i], 1))
        # shrinking fleets sell their oldest trucks now and then
        shrink = (trend < 0) & (rng.random(n) < 0.02)
        for i in np.where(shrink & (ages.sum(axis=1) > 3))[0]:
            a = MAX_AGE - 1
            while ages[i, a] == 0 and a > 0:
                a -= 1
            ages[i, a] -= 1
        # one month older; trucks past MAX_AGE stay in the last bucket
        last = ages[:, -1].copy()
        ages[:, 1:] = ages[:, :-1]
        ages[:, 0] = 0
        ages[:, -1] += last

    customers = pd.DataFrame(
        dict(
            cid=np.arange(n),
            company=[_company(rng, INDUSTRY_NAMES[ind[i]], i) for i in range(n)],
            country=[COUNTRY_NAMES[c] for c in ctry],
            industry=[INDUSTRY_NAMES[i] for i in ind],
            dealer=dealer,
            customer_since=customer_since,
        )
    )
    panel = pd.DataFrame(rows).merge(customers[["cid", "country", "industry"]], on="cid")
    deals = pd.DataFrame(deals)
    return customers, panel, deals


def _trigger(age, fin, contract, growth, breakdowns) -> str:
    """The main reason behind an order, as a sales rep would write it in the CRM."""
    cands = [
        (age - 5.5, "ageing fleet"),
        (fin * 6, "financing ended"),
        (contract * 0.6, "service contract renewal"),
        (growth * 8, "fleet growth"),
        (breakdowns / 25, "rising breakdowns"),
    ]
    return max(cands)[1]


_PREFIX = [
    "Atlas",
    "Nordic",
    "Iberia",
    "Alpine",
    "Delta",
    "Orion",
    "Vector",
    "Summit",
    "Harbor",
    "Meridian",
    "Polar",
    "Crown",
    "Falcon",
    "Granite",
    "Horizon",
    "Lumen",
    "Rapid",
    "Sierra",
    "Tide",
    "Vista",
]
_SUFFIX = {
    "long-haul logistics": "Freight",
    "regional distribution": "Distribution",
    "construction": "Build",
    "urban delivery": "City Logistics",
    "waste & municipal": "Environmental Services",
}


def _company(rng, industry, i) -> str:
    return f"{_PREFIX[i % len(_PREFIX)]} {_SUFFIX[industry]} {i:04d}"
