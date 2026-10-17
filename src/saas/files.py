"""Tenant-scoped files API: upload, list, read, download and delete.

Every query filters on the tenant from the session (``auth.principal``);
a file of another tenant is indistinguishable from one that doesn't
exist (404), so ids can't be probed. Status lifecycle:

    UPLOADING -> QUARANTINED -> VALIDATING -> READY | REJECTED
    READY -> PROCESSING -> COMPLETED | FAILED   (jobs, saas.jobs)
    any -> DELETED

Uploads land in the tenant's quarantine area under a random key, are
validated from their content (``saas.validation``) and scanned
(``saas.scanner``), and only then move to ``uploads/``. Nothing here
parses a file for analysis -- that happens in the sandboxed worker.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import urllib.parse
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from saas import db
from saas.auth import Principal, principal
from saas.config import get_settings
from saas.limits import RateLimiter
from saas.scanner import get_scanner
from saas.storage import get_storage, object_key
from saas.validation import Rejected, detect, display_filename

STATUSES = ("CREATED", "UPLOADING", "QUARANTINED", "VALIDATING", "READY", "PROCESSING",
            "COMPLETED", "FAILED", "REJECTED", "DELETED")
DOWNLOADABLE = ("READY", "PROCESSING", "COMPLETED", "FAILED")
DOWNLOAD_LINK_TTL_S = 60
MANAGERS = ("owner", "admin")

upload_limiter = RateLimiter(limit=30, window_s=60)

router = APIRouter(prefix="/api")

_PUBLIC_COLUMNS = "id, original_filename, detected_type, size, sha256, status, status_reason, created_at, scanned_at"


def _public(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "filename": row["original_filename"],
        "type": row["detected_type"],
        "size": row["size"],
        "sha256": row["sha256"],
        "status": row["status"],
        "reason": row["status_reason"],
        "created_at": row["created_at"],
        "scanned_at": row["scanned_at"],
    }


def get_file(conn: sqlite3.Connection, p: Principal, file_id: str, *, include_deleted: bool = False) -> sqlite3.Row:
    """The tenant's file, or 404 -- the only way to look one up."""
    row = conn.execute("SELECT * FROM files WHERE id = ? AND tenant_id = ?", (file_id, p.tenant_id)).fetchone()
    if row is None or (row["status"] == "DELETED" and not include_deleted):
        raise HTTPException(status_code=404, detail="file not found")
    return row


def _set_status(file_id: str, tenant_id: str, status: str, reason: str | None = None, **columns) -> None:
    assert status in STATUSES
    sets = ", ".join(["status = ?", "status_reason = ?", *(f"{c} = ?" for c in columns)])
    with db.connect() as conn:
        conn.execute(f"UPDATE files SET {sets} WHERE id = ? AND tenant_id = ?",  # column names are ours, never input
                     (status, reason, *columns.values(), file_id, tenant_id))


def _usage(conn: sqlite3.Connection, tenant_id: str) -> tuple[int, int]:
    """Bytes the tenant stores now, and bytes it uploaded in the last 24h
    (rejected uploads count: they cost the same to receive)."""
    since = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
    stored = conn.execute(
        "SELECT COALESCE(SUM(size), 0) FROM files WHERE tenant_id = ? AND status NOT IN ('REJECTED', 'DELETED')",
        (tenant_id,),
    ).fetchone()[0]
    daily = conn.execute(
        "SELECT COALESCE(SUM(size), 0) FROM files WHERE tenant_id = ? AND created_at >= ?", (tenant_id, since)
    ).fetchone()[0]
    return stored, daily


def _validate_and_store(file_id: str, tenant_id: str, quarantine_key: str, data: bytes, filename: str) -> dict:
    """QUARANTINED -> VALIDATING -> READY | REJECTED. Runs in a thread."""
    storage = get_storage()
    _set_status(file_id, tenant_id, "VALIDATING")
    try:
        detected = detect(data, filename)
        scan = get_scanner().scan(data)
        if not scan.clean:
            raise Rejected("file failed the malware scan")
    except Rejected as exc:
        storage.delete(quarantine_key)
        _set_status(file_id, tenant_id, "REJECTED", str(exc), scanned_at=db.now())
        return {"status": "REJECTED", "reason": str(exc)}
    except Exception:
        # scanner down, storage error: keep it out of the tenant's files, and
        # don't leak the internal error
        storage.delete(quarantine_key)
        _set_status(file_id, tenant_id, "REJECTED", "could not verify the file; try again later")
        raise
    final_key = object_key(tenant_id, "uploads", file_id)
    storage.move(quarantine_key, final_key)
    _set_status(file_id, tenant_id, "READY", None if scan.scanned else "not malware-scanned",
                storage_key=final_key, detected_type=detected.type,
                scanned_at=db.now() if scan.scanned else None)
    return {"status": "READY", "type": detected.type}


@router.post("/files", status_code=201)
async def upload_files(
    file: list[UploadFile] = File(...),  # noqa: B008
    p: Principal = Depends(principal),  # noqa: B008
) -> dict:
    """Upload one or more trace / usage files to the tenant. Each is
    checked against the size, count and quota limits, validated from its
    content and scanned; the response gives each file's id and status."""
    s = get_settings()
    upload_limiter.check(p.user_id)
    if len(file) > s.max_files_per_upload:
        raise HTTPException(status_code=413, detail=f"at most {s.max_files_per_upload} files per upload")
    storage = get_storage()
    received = []
    for upload in file:  # all size checks before anything is stored
        filename = display_filename(upload.filename)
        data = await upload.read(s.max_file_bytes + 1)
        if len(data) > s.max_file_bytes:
            with db.connect() as conn:
                db.audit(conn, "file.upload", "denied", tenant_id=p.tenant_id, user_id=p.user_id,
                         detail="file too large")
            raise HTTPException(status_code=413, detail=f"{filename} is larger than {s.max_file_bytes} bytes")
        received.append((upload, filename, data))
    out = []
    for upload, filename, data in received:
        file_id = db.new_id()
        quarantine_key = object_key(p.tenant_id, "quarantine", file_id)
        digest = hashlib.sha256(data).hexdigest()
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")  # quota check and insert as one step
            stored, daily = _usage(conn, p.tenant_id)
            if stored + len(data) > s.tenant_storage_bytes or daily + len(data) > s.daily_upload_bytes:
                which = "storage" if stored + len(data) > s.tenant_storage_bytes else "daily upload"
                db.audit(conn, "file.upload", "denied", tenant_id=p.tenant_id, user_id=p.user_id,
                         detail=f"{which} quota")
                conn.commit()
                raise HTTPException(status_code=413, detail=f"your organization's {which} quota is used up")
            duplicate = conn.execute(
                "SELECT id FROM files WHERE tenant_id = ? AND sha256 = ? AND status NOT IN ('REJECTED', 'DELETED')",
                (p.tenant_id, digest),
            ).fetchone()
            conn.execute(
                "INSERT INTO files (id, tenant_id, uploaded_by, original_filename, storage_key, declared_type, size, "
                "sha256, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'UPLOADING', ?)",
                (file_id, p.tenant_id, p.user_id, filename, quarantine_key,
                 (upload.content_type or "")[:100] or None, len(data), digest, db.now()),
            )
        await run_in_threadpool(storage.put, quarantine_key, data)
        _set_status(file_id, p.tenant_id, "QUARANTINED")
        outcome = await run_in_threadpool(_validate_and_store, file_id, p.tenant_id, quarantine_key, data, filename)
        with db.connect() as conn:
            db.audit(conn, "file.upload", "ok" if outcome["status"] == "READY" else "rejected",
                     tenant_id=p.tenant_id, user_id=p.user_id, resource_type="file", resource_id=file_id,
                     detail=outcome.get("reason"))
        out.append({"id": file_id, "filename": filename, "size": len(data), "sha256": digest,
                    "duplicate_of": duplicate["id"] if duplicate else None, **outcome})
    return {"files": out}


@router.get("/files")
def list_files(p: Principal = Depends(principal)) -> dict:  # noqa: B008
    s = get_settings()
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT {_PUBLIC_COLUMNS} FROM files WHERE tenant_id = ? AND status != 'DELETED' "
            "ORDER BY created_at DESC, id",
            (p.tenant_id,),
        ).fetchall()
        stored, daily = _usage(conn, p.tenant_id)
    return {
        "files": [_public(r) for r in rows],
        "usage": {"stored_bytes": stored, "storage_limit_bytes": s.tenant_storage_bytes,
                  "uploaded_today_bytes": daily, "daily_limit_bytes": s.daily_upload_bytes},
    }


@router.get("/files/{file_id}")
def read_file(file_id: str, p: Principal = Depends(principal)) -> dict:  # noqa: B008
    with db.connect() as conn:
        return _public(get_file(conn, p, file_id))


def _attachment(row: sqlite3.Row) -> Response:
    data = get_storage().get(row["storage_key"])
    name = row["original_filename"]
    fallback = "".join(c if c.isascii() and c.isprintable() and c not in '"\\;' else "_" for c in name)
    return Response(
        data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=\"{fallback}\"; "
                                   f"filename*=UTF-8''{urllib.parse.quote(name, safe='')}",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "X-Checksum-SHA256": row["sha256"],
        },
    )


def _downloadable(conn: sqlite3.Connection, p: Principal, file_id: str) -> sqlite3.Row:
    row = get_file(conn, p, file_id)
    if row["status"] not in DOWNLOADABLE:
        raise HTTPException(status_code=409, detail=f"file is {row['status'].lower()}, not downloadable")
    return row


@router.get("/files/{file_id}/download")
def download_file(file_id: str, p: Principal = Depends(principal)) -> Response:  # noqa: B008
    with db.connect() as conn:
        row = _downloadable(conn, p, file_id)
        db.audit(conn, "file.download", tenant_id=p.tenant_id, user_id=p.user_id, resource_type="file",
                 resource_id=file_id)
    return _attachment(row)


@router.post("/files/{file_id}/download-link")
def download_link(file_id: str, p: Principal = Depends(principal)) -> dict:  # noqa: B008
    """A one-use link to the file that works for ``DOWNLOAD_LINK_TTL_S``
    seconds -- the shape an S3 presigned URL takes in P1."""
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(UTC) + timedelta(seconds=DOWNLOAD_LINK_TTL_S)).isoformat(timespec="seconds")
    with db.connect() as conn:
        _downloadable(conn, p, file_id)
        conn.execute("DELETE FROM download_tokens WHERE expires_at < ?", (db.now(),))
        conn.execute(
            "INSERT INTO download_tokens (token_hash, tenant_id, file_id, user_id, expires_at) VALUES (?, ?, ?, ?, ?)",
            (hashlib.sha256(token.encode()).hexdigest(), p.tenant_id, file_id, p.user_id, expires),
        )
    return {"url": f"/api/downloads/{token}", "expires_at": expires}


@router.get("/downloads/{token}")
def download_by_link(token: str) -> Response:
    if not get_settings().saas:
        raise HTTPException(status_code=404, detail="not found")
    with db.connect() as conn:
        link = conn.execute(
            "DELETE FROM download_tokens WHERE token_hash = ? RETURNING tenant_id, file_id, user_id, expires_at",
            (hashlib.sha256(token.encode()).hexdigest(),),
        ).fetchone()
        if link is None or link["expires_at"] < db.now():
            raise HTTPException(status_code=404, detail="link expired or already used")
        owner = Principal(user_id=link["user_id"], tenant_id=link["tenant_id"], role="member", email=None,
                          platform_admin=False)
        row = _downloadable(conn, owner, link["file_id"])
        db.audit(conn, "file.download", tenant_id=owner.tenant_id, user_id=owner.user_id, resource_type="file",
                 resource_id=row["id"], detail="one-use link")
    return _attachment(row)


@router.delete("/files/{file_id}")
def delete_file(file_id: str, p: Principal = Depends(principal)) -> dict:  # noqa: B008
    """Delete the file, its results and pending jobs. Its row stays as a
    tombstone (status DELETED) for the audit trail."""
    storage = get_storage()
    with db.connect() as conn:
        row = get_file(conn, p, file_id)
        if p.role not in MANAGERS and row["uploaded_by"] != p.user_id:
            raise HTTPException(status_code=403, detail="only the uploader or an admin can delete this file")
        results = conn.execute(
            "SELECT id, storage_key FROM results WHERE tenant_id = ? AND file_id = ?", (p.tenant_id, file_id)
        ).fetchall()
        conn.execute("DELETE FROM results WHERE tenant_id = ? AND file_id = ?", (p.tenant_id, file_id))
        conn.execute(
            "UPDATE jobs SET status = 'CANCELLED', finished_at = ? WHERE tenant_id = ? AND file_id = ? "
            "AND status IN ('QUEUED', 'RUNNING')",
            (db.now(), p.tenant_id, file_id),
        )
        conn.execute("DELETE FROM download_tokens WHERE tenant_id = ? AND file_id = ?", (p.tenant_id, file_id))
        conn.execute(
            "UPDATE files SET status = 'DELETED', status_reason = NULL, deleted_at = ? WHERE id = ? AND tenant_id = ?",
            (db.now(), file_id, p.tenant_id),
        )
        db.audit(conn, "file.delete", tenant_id=p.tenant_id, user_id=p.user_id, resource_type="file",
                 resource_id=file_id, detail=f"{len(results)} results removed")
    for key in [row["storage_key"], *(r["storage_key"] for r in results)]:
        storage.delete(key)
    return {"deleted": file_id, "results_deleted": len(results)}
