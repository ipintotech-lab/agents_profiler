"""History: events kept across uploads, so behavior statistics grow with
every analysis instead of starting over each time.

Opt-in per upload (the dashboard's "Keep in history" box, ``keep_history``
on ``/analyze``) because it stores the events' inputs and outputs. One
SQLite file at ``BEHAVIOR_DB`` (default ``data/behavior.sqlite``); on a
host with ephemeral disk (e.g. Railway without a volume) point it at a
mounted volume, or history disappears on every deploy.

Events are keyed by a hash of their content, so keeping the same upload
twice adds nothing -- duplicates would read as repeated identical inputs
and inflate consistency.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ingest.trace_schema import TraceEvent

DEFAULT_PATH = "data/behavior.sqlite"

MAX_EVENTS = 500_000
"""Upper bound on kept events; adding past it is refused so the file (and
the time to rebuild behavior from it) stays bounded."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_key TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    workflow TEXT NOT NULL,
    workflow_version TEXT,
    agent TEXT,
    node TEXT NOT NULL,
    model TEXT,
    added_at TEXT NOT NULL,
    event_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_workflow ON events (workflow, node);
"""


def db_path() -> Path:
    return Path(os.environ.get("BEHAVIOR_DB") or DEFAULT_PATH)


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def add(events: list[TraceEvent], dataset: str) -> dict:
    """Keep ``events`` (labelled ``dataset``, e.g. the file names). Returns
    how many were new and how many were already kept."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for e in events:
        payload = e.model_dump_json()
        rows.append(
            (
                hashlib.sha256(payload.encode()).hexdigest(),
                dataset, e.workflow, e.workflow_version, e.agent, e.node, e.model, now, payload,
            )
        )
    with closing(_connect()) as conn, conn:
        (kept,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        if kept + len(rows) > MAX_EVENTS:
            raise ValueError(f"history holds {kept} events; adding {len(rows)} would pass the {MAX_EVENTS} limit")
        before = conn.total_changes
        conn.executemany("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        added = conn.total_changes - before
    return {"added": added, "duplicates": len(rows) - added, "total": kept + added}


def load(workflow: str | None = None) -> list[TraceEvent]:
    query, params = "SELECT event_json FROM events", ()
    if workflow:
        query, params = query + " WHERE workflow = ?", (workflow,)
    with closing(_connect()) as conn:
        return [TraceEvent.model_validate_json(row[0]) for row in conn.execute(query + " ORDER BY rowid", params)]


def summary() -> dict:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT workflow, COUNT(*), COUNT(DISTINCT dataset), COUNT(DISTINCT workflow_version), "
            "MIN(added_at), MAX(added_at) FROM events GROUP BY workflow ORDER BY COUNT(*) DESC"
        ).fetchall()
    return {
        "path": str(db_path()),
        "total": sum(r[1] for r in rows),
        "workflows": [
            {"workflow": r[0], "events": r[1], "uploads": r[2], "versions": r[3], "first_added": r[4], "last_added": r[5]}
            for r in rows
        ],
    }


def clear(workflow: str | None = None, older_than_days: float | None = None) -> int:
    """Delete kept events -- all, one workflow's, and/or those added more
    than ``older_than_days`` ago. Returns how many were deleted."""
    clauses, params = [], []
    if workflow:
        clauses.append("workflow = ?")
        params.append(workflow)
    if older_than_days is not None:
        clauses.append("added_at < ?")
        params.append((datetime.now(UTC) - timedelta(days=older_than_days)).isoformat(timespec="seconds"))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with closing(_connect()) as conn, conn:
        return conn.execute("DELETE FROM events" + where, params).rowcount


def dataset_label(names: list[str]) -> str:
    return json.dumps(sorted(names))[:500]
