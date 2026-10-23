"""Pseudonymisation: the model, the agent and the LLM only ever see customer IDs like C-3F9A21.

Company names live in a separate file that only the app reads, on the rep's side, to show
names next to IDs. They are never put in a prompt, a tool result or a log.
"""

from __future__ import annotations

import hashlib
import hmac


def pseudonym(internal_id: int, key: str) -> str:
    """Stable, keyed pseudonym. Without the key the mapping can't be rebuilt from the ID."""
    digest = hmac.new(key.encode(), str(internal_id).encode(), hashlib.sha256).hexdigest()
    return f"C-{digest[:6].upper()}"


def rep_id(name: str, key: str) -> str:
    digest = hmac.new(key.encode(), f"rep:{name}".encode(), hashlib.sha256).hexdigest()
    return f"R-{digest[:6].upper()}"
