"""Grounding check: every number in the answer must appear in a tool result.

Numbers are compared after normalising (commas, €, %), allowing normal rounding, and
allowing scores (0-1) to be written as percentages. Customer and deal IDs, dates and
list numbering are not treated as facts.
"""

from __future__ import annotations

import json
import re

NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
IGNORE = [
    re.compile(p)
    for p in (r"\b[CDR]-[0-9A-F]{5,6}\b", r"\b\d{4}-\d{2}(?:-\d{2})?\b", r"^\s*\d+[.)]\s", r"\btop[- ]?\d+\b")
]


def numbers_in(text: str) -> list[tuple[str, float]]:
    clean = text
    for pat in IGNORE:
        clean = pat.sub(" ", clean) if not pat.pattern.startswith("^") else re.sub(pat.pattern, " ", clean, flags=re.M)
    out = []
    for m in NUM.finditer(clean):
        raw = m.group(0)
        try:
            out.append((raw, float(raw.replace(",", ""))))
        except ValueError:
            continue
    return out


def _source_numbers(obj, acc: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        acc.add(v)
        if abs(v) <= 1.0:
            acc.add(v * 100)  # 0.42 -> 42 %
        return
    if isinstance(obj, str):
        for _, v in numbers_in(obj):
            acc.add(v)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _source_numbers(v, acc)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _source_numbers(v, acc)


def source_numbers(tool_results: list, question: str = "") -> set[float]:
    acc: set[float] = set()
    for r in tool_results:
        _source_numbers(json.loads(r) if isinstance(r, str) and r[:1] in "[{" else r, acc)
    for _, v in numbers_in(question):
        acc.add(v)
    return acc


def _matches(raw: str, v: float, sources: set[float]) -> bool:
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for s in sources:
        if abs(abs(s) - abs(v)) <= tol:  # same number, rounded as written
            return True
        if decimals == 0 and abs(v) >= 1000 and abs(s - v) <= 0.005 * abs(s):  # "€1.2M"-style rounding of big values
            return True
        for unit in (1e3, 1e6):  # "345k", "1.2 million"
            if abs(s / unit - v) <= tol and abs(s) >= unit:
                return True
    return False


def check(answer: str, tool_results: list, question: str = "", small_ints_ok: int = 10) -> dict:
    sources = source_numbers(tool_results, question)
    unverified = []
    for raw, v in numbers_in(answer):
        if float(v).is_integer() and 0 <= v <= small_ints_ok:
            continue  # counts like "3 reasons" or "top 5"
        if not _matches(raw, v, sources):
            unverified.append(raw)
    return dict(ok=not unverified, unverified=sorted(set(unverified)), checked=len(numbers_in(answer)))
