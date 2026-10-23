"""Settings, read from environment variables (12-factor style), with safe defaults.

Nothing secret lives in code: API keys come from the environment or the app's key field.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- simulation ---
SEED = 11
N_CUSTOMERS = 2000
N_MONTHS = 36  # monthly snapshots
START_MONTH = "2023-11"
HORIZON = 6  # label: orders new trucks within the next 6 months

COUNTRIES = {"DE": 0.22, "FR": 0.15, "ES": 0.13, "IT": 0.13, "PL": 0.12, "NL": 0.09, "PT": 0.08, "SE": 0.08}
INDUSTRIES = {
    # name: (share of customers, median fleet size, km per truck per year, fleet growth trend)
    "long-haul logistics": (0.30, 40, 125_000, 0.03),
    "regional distribution": (0.25, 25, 80_000, 0.02),
    "construction": (0.18, 18, 45_000, 0.00),
    "urban delivery": (0.17, 15, 50_000, 0.05),
    "waste & municipal": (0.10, 12, 35_000, 0.01),
}
TRUCK_PRICE_EUR = 115_000

# --- time split (month index) ---
TRAIN_MONTHS = (3, 20)  # snapshots used for training
CALIB_MONTHS = (21, 24)  # probability calibration and early checks
TEST_MONTHS = (25, 29)  # scored like the future: never seen in training
# months 30-35 have no complete 6-month label window; month 35 is "today"


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("FLEETPILOT_DATA", ROOT / "data")))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))  # or "bedrock"
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "eu-north-1"))
    agent_model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "claude-sonnet-4-6"))
    pseudonym_key: str = field(default_factory=lambda: os.getenv("PSEUDONYM_KEY", "demo-only-key-change-me"))
    mlflow_uri: str | None = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI") or None)
    monthly_budget_eur: float = field(default_factory=lambda: float(os.getenv("MONTHLY_BUDGET_EUR", "10")))
    max_agent_steps: int = 8
    max_sql_rows: int = 200

    @property
    def crm_db(self) -> Path:
        return self.data_dir / "crm.db"

    @property
    def audit_db(self) -> Path:
        return self.data_dir / "audit.db"


def settings() -> Settings:
    return Settings()
