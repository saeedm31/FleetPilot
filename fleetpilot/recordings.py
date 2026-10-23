"""Recorded real copilot runs, replayed in the app's demo mode (no API key needed).

Record with a key:  python -m fleetpilot record
A recording keeps the question, answer, tool calls, check result and token counts. No key, no names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import ROOT

DIR = ROOT / "demo" / "recordings"

QUESTIONS = [
    "Which customers in Portugal should I call first this week, and why?",
    "Which customer segment should we focus on this quarter?",
    "How many deals did we lose to competitors, and what usually triggered them?",
]


def save(result, folder: Path = DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result.question.lower()).strip("-")[:60]
    rec = dict(
        question=result.question,
        answer=result.answer,
        check=result.check,
        tool_calls=result.tool_calls,
        usage=[u.to_dict() for u in result.usage],
        cost=result.cost,
        seconds=result.seconds,
        engine=result.engine,
        steps=result.steps,
    )
    path = folder / f"{slug}.json"
    path.write_text(json.dumps(rec, indent=1, default=str))
    return path


def load_all(folder: Path = DIR) -> list[dict]:
    out = []
    for p in sorted(folder.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


__all__ = ["QUESTIONS", "save", "load_all"]
