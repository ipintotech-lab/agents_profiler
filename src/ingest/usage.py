"""Aggregate usage exports: records that carry token counters per workload
(or agent activations), but no steps, so they can't become ``TraceEvent``s.

Two shapes are recognized (field aliases in parentheses):

* **usage snapshots** -- ``record_id`` (``run_id``), ``workload_id``
  (``workload``, ``workflow``) and at least one token counter:
  ``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``,
  ``cache_creation_input_tokens``.
* **activations** -- ``agent_id`` (``agent``) and ``trigger_id``
  (``trigger``), optionally ``event_id``.

A file is aggregate only when every record matches one shape and none has
a ``node`` -- a record with a step name is a trace event and goes through
``normalize.py`` instead. CSV is accepted as well as JSON / JSON Lines.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
_RECORD_ALIASES = ("record_id", "run_id")
_WORKLOAD_ALIASES = ("workload_id", "workload", "workflow")
_AGENT_ALIASES = ("agent_id", "agent")
_TRIGGER_ALIASES = ("trigger_id", "trigger")


def _first(d: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if d.get(key) not in (None, ""):
            return d[key]
    return None


def _is_usage(r: dict) -> bool:
    return (
        _first(r, _RECORD_ALIASES) is not None
        and _first(r, _WORKLOAD_ALIASES) is not None
        and any(k in r for k in TOKEN_FIELDS)
    )


def _is_activation(r: dict) -> bool:
    return _first(r, _AGENT_ALIASES) is not None and _first(r, _TRIGGER_ALIASES) is not None


def parse_records(text: str) -> list[dict] | None:
    """Rows from JSON, JSON Lines or CSV text; ``None`` if none of them parse
    to a list of objects."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        rows = parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        try:
            rows = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        except json.JSONDecodeError:
            rows = list(csv.DictReader(io.StringIO(stripped)))
    if not rows or not all(isinstance(r, dict) for r in rows):
        return None
    return rows


def detect_kind(rows: list[dict]) -> str | None:
    """``"usage"``, ``"activations"`` or ``None`` (not an aggregate file)."""
    if not rows or any(r.get("node") for r in rows):
        return None
    if all(_is_usage(r) for r in rows):
        return "usage"
    if all(_is_activation(r) for r in rows):
        return "activations"
    return None


def _count(value: Any) -> int:
    if value in (None, ""):
        return 0
    n = int(float(value))
    if n < 0:
        raise ValueError(f"token counters can't be negative, got {value!r}")
    return n


def normalize_usage(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        counts = {k: _count(r.get(k)) for k in TOKEN_FIELDS}
        out.append({
            "record_id": str(_first(r, _RECORD_ALIASES)),
            "workload_id": str(_first(r, _WORKLOAD_ALIASES)),
            **counts,
        })
    return out


def normalize_activations(rows: list[dict]) -> list[dict]:
    return [
        {
            "event_id": str(r.get("event_id") or i + 1),
            "agent_id": str(_first(r, _AGENT_ALIASES)),
            "trigger_id": str(_first(r, _TRIGGER_ALIASES)),
        }
        for i, r in enumerate(rows)
    ]
