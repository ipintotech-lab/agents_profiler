"""Metadata database: users, tenants, memberships, sessions, files, jobs,
results and audit events.

SQLite at ``SAAS_DB`` (on Railway: the ``/data`` volume). Every
customer-owned table carries ``tenant_id``, and the accessors in
``saas.files`` / ``saas.jobs`` always filter on it -- there is no "get by
id" without a tenant. Postgres with row-level security is the P1
replacement once the web service runs more than one replica.
"""
from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from saas.config import get_settings

ROLES = ("owner", "admin", "member")

_SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    issuer TEXT NOT NULL,
    subject TEXT NOT NULL,
    email TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (issuer, subject)
);
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memberships (
    user_id TEXT NOT NULL REFERENCES users (id),
    tenant_id TEXT NOT NULL REFERENCES tenants (id),
    role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, tenant_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id),
    tenant_id TEXT NOT NULL REFERENCES tenants (id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_requests (
    state TEXT PRIMARY KEY,
    nonce TEXT NOT NULL,
    code_verifier TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants (id),
    uploaded_by TEXT NOT NULL REFERENCES users (id),
    original_filename TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    declared_type TEXT,
    detected_type TEXT,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    status_reason TEXT,
    created_at TEXT NOT NULL,
    scanned_at TEXT,
    deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS files_tenant ON files (tenant_id, status);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants (id),
    file_id TEXT NOT NULL REFERENCES files (id),
    created_by TEXT NOT NULL REFERENCES users (id),
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_tenant ON jobs (tenant_id, status);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, created_at);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants (id),
    job_id TEXT NOT NULL REFERENCES jobs (id),
    file_id TEXT NOT NULL REFERENCES files (id),
    storage_key TEXT NOT NULL,
    size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_tenant ON results (tenant_id, job_id);
CREATE TABLE IF NOT EXISTS download_tokens (
    token_hash TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants (id),
    file_id TEXT NOT NULL REFERENCES files (id),
    user_id TEXT NOT NULL REFERENCES users (id),
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    tenant_id TEXT,
    user_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    result TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS audit_tenant ON audit_events (tenant_id, at);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _connect() -> sqlite3.Connection:
    path = Path(get_settings().db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """A connection that commits on success and rolls back on error."""
    with closing(_connect()) as conn, conn:
        yield conn


def audit(
    conn: sqlite3.Connection,
    action: str,
    result: str = "ok",
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: str | None = None,
) -> None:
    """Record an audit event. ``detail`` is a short reason, never file
    content."""
    conn.execute(
        "INSERT INTO audit_events (at, tenant_id, user_id, action, resource_type, resource_id, result, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (now(), tenant_id, user_id, action, resource_type, resource_id, result, detail and detail[:200]),
    )


def upsert_user(conn: sqlite3.Connection, issuer: str, subject: str, email: str | None) -> tuple[str, bool]:
    """The user's id for this identity, creating it on first login.
    Returns ``(user_id, created)``."""
    row = conn.execute("SELECT id FROM users WHERE issuer = ? AND subject = ?", (issuer, subject)).fetchone()
    if row:
        conn.execute("UPDATE users SET email = ? WHERE id = ?", (email, row["id"]))
        return row["id"], False
    user_id = new_id()
    conn.execute(
        "INSERT INTO users (id, issuer, subject, email, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, issuer, subject, email, now()),
    )
    return user_id, True


def default_tenant(conn: sqlite3.Connection, user_id: str, name: str) -> str:
    """The tenant a login lands in: the user's oldest membership, or a new
    tenant they own (invitations to an existing tenant are P1)."""
    row = conn.execute(
        "SELECT tenant_id FROM memberships WHERE user_id = ? ORDER BY created_at, tenant_id LIMIT 1", (user_id,)
    ).fetchone()
    if row:
        return row["tenant_id"]
    tenant_id = new_id()
    conn.execute("INSERT INTO tenants (id, name, created_at) VALUES (?, ?, ?)", (tenant_id, name, now()))
    add_member(conn, user_id, tenant_id, "owner")
    return tenant_id


def add_member(conn: sqlite3.Connection, user_id: str, tenant_id: str, role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    conn.execute(
        "INSERT OR REPLACE INTO memberships (user_id, tenant_id, role, created_at) VALUES (?, ?, ?, ?)",
        (user_id, tenant_id, role, now()),
    )


def role_in(conn: sqlite3.Connection, user_id: str, tenant_id: str) -> str | None:
    row = conn.execute(
        "SELECT role FROM memberships WHERE user_id = ? AND tenant_id = ?", (user_id, tenant_id)
    ).fetchone()
    return row["role"] if row else None
