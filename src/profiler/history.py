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

In SaaS mode every call carries the caller's ``tenant`` (from the
session) and sees only that tenant's events; the public demo uses the
empty tenant.
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
    tenant TEXT NOT NULL DEFAULT '',
    dataset TEXT NOT NULL,
    workflow TEXT NOT NULL,
    workflow_version TEXT,
    agent TEXT,
    node TEXT NOT NULL,
    model TEXT,
    added_at TEXT NOT NULL,
    event_json TEXT NOT NULL
);
"""
_INDEX = "CREATE INDEX IF NOT EXISTS events_tenant_workflow ON events (tenant, workflow, node);"


def db_path() -> Path:
    return Path(os.environ.get("BEHAVIOR_DB") or DEFAULT_PATH)


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    if "tenant" not in {row[1] for row in conn.execute("PRAGMA table_info(events)")}:
        # a history file from before tenants: its events are the demo's
        conn.execute("ALTER TABLE events ADD COLUMN tenant TEXT NOT NULL DEFAULT ''")
    conn.execute(_INDEX)
    return conn


def _key(tenant: str, payload: str) -> str:
    # the tenant is part of the key, so the same event kept by two tenants
    # is two rows (and the demo's keys are unchanged)
    return hashlib.sha256(((tenant + "\0") if tenant else "").encode() + payload.encode()).hexdigest()


def add(events: list[TraceEvent], dataset: str, tenant: str = "") -> dict:
    """Keep ``events`` (labelled ``dataset``, e.g. the file names). Returns
    how many were new and how many were already kept."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for e in events:
        payload = e.model_dump_json()
        rows.append(
            (
                _key(tenant, payload),
                tenant,
                dataset, e.workflow, e.workflow_version, e.agent, e.node, e.model, now, payload,
            )
        )
    with closing(_connect()) as conn, conn:
        (kept,) = conn.execute("SELECT COUNT(*) FROM events WHERE tenant = ?", (tenant,)).fetchone()
        if kept + len(rows) > MAX_EVENTS:
            raise ValueError(f"history holds {kept} events; adding {len(rows)} would pass the {MAX_EVENTS} limit")
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO events (event_key, tenant, dataset, workflow, workflow_version, agent, node, "
            "model, added_at, event_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        added = conn.total_changes - before
    return {"added": added, "duplicates": len(rows) - added, "total": kept + added}


def load(workflow: str | None = None, tenant: str = "") -> list[TraceEvent]:
    query, params = "SELECT event_json FROM events WHERE tenant = ?", (tenant,)
    if workflow:
        query, params = query + " AND workflow = ?", (tenant, workflow)
    with closing(_connect()) as conn:
        return [TraceEvent.model_validate_json(row[0]) for row in conn.execute(query + " ORDER BY rowid", params)]


def summary(tenant: str = "") -> dict:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT workflow, COUNT(*), COUNT(DISTINCT dataset), COUNT(DISTINCT workflow_version), "
            "MIN(added_at), MAX(added_at) FROM events WHERE tenant = ? GROUP BY workflow ORDER BY COUNT(*) DESC",
            (tenant,),
        ).fetchall()
    return {
        "path": str(db_path()),
        "total": sum(r[1] for r in rows),
        "workflows": [
            {"workflow": r[0], "events": r[1], "uploads": r[2], "versions": r[3], "first_added": r[4], "last_added": r[5]}
            for r in rows
        ],
    }


def clear(workflow: str | None = None, older_than_days: float | None = None, tenant: str = "") -> int:
    """Delete the tenant's kept events -- all, one workflow's, and/or those
    added more than ``older_than_days`` ago. Returns how many were deleted."""
    clauses, params = ["tenant = ?"], [tenant]
    if workflow:
        clauses.append("workflow = ?")
        params.append(workflow)
    if older_than_days is not None:
        clauses.append("added_at < ?")
        params.append((datetime.now(UTC) - timedelta(days=older_than_days)).isoformat(timespec="seconds"))
    with closing(_connect()) as conn, conn:
        return conn.execute("DELETE FROM events WHERE " + " AND ".join(clauses), params).rowcount


def dataset_label(names: list[str]) -> str:
    return json.dumps(sorted(names))[:500]
