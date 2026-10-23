"""The CRM database the copilot can read: SQLite, pseudonymised, opened read-only.

Safety for agent SQL, in layers:
1. the connection is opened read-only (SQLite `mode=ro`),
2. an authorizer allows only SELECT/READ of known tables (no ATTACH, PRAGMA, writes),
3. one statement only, a row cap, and a time limit.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

TABLES = {
    "customers": "one row per customer: customer_id, country, industry, dealer, customer_since, segment",
    "snapshot": "latest monthly snapshot per customer: fleet and service facts (fleet_size, avg_age_years, ...)",
    "scores": "latest purchase likelihood per customer: score (0-1), rank, reason_1..3, model_version, month",
    "deals": "past orders: deal_id, customer_id, period, status (won/lost), trucks, value_eur, trigger",
    "segments": "segment profiles: segment, customers, trucks, avg_fleet, avg_age_years, our_share, main_industry",
}


class SqlError(ValueError):
    pass


def write_db(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    with sqlite3.connect(tmp) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False)
        con.execute("CREATE INDEX idx_scores ON scores(customer_id)")
        con.execute("CREATE INDEX idx_deals ON deals(customer_id)")
    tmp.replace(path)


def _authorizer(action, arg1, arg2, dbname, source):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    if action in allowed:
        if action == sqlite3.SQLITE_READ and arg1 not in TABLES:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def read_only(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    con.set_authorizer(_authorizer)
    return con


def run_select(con: sqlite3.Connection, sql: str, max_rows: int = 200, timeout_s: float = 3.0) -> pd.DataFrame:
    q = sql.strip().rstrip(";").strip()
    if ";" in q:
        raise SqlError("Only one statement is allowed.")
    if not q.lower().startswith(("select", "with")):
        raise SqlError("Only SELECT queries are allowed.")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(q)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.DatabaseError as exc:
        raise SqlError(str(exc)) from exc
    finally:
        con.set_progress_handler(None, 0)
    df = pd.DataFrame(rows[:max_rows], columns=cols)
    df.attrs["truncated"] = len(rows) > max_rows
    return df


def columns(path: Path) -> dict[str, list[str]]:
    """Column names per table (uses a normal connection; PRAGMA is blocked on the agent's one)."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        return {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})")] for t in TABLES}


def schema_text(path: Path) -> str:
    """Tables, what they hold and their columns: given to the agent so it can write SQL."""
    cols = columns(path)
    return "\n".join(f"- {t}: {desc}\n  columns: {', '.join(cols[t])}" for t, desc in TABLES.items())
