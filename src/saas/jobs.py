"""Analysis jobs and their results, tenant-scoped like files.

``POST /api/files/{id}/jobs`` only queues a job with validated
parameters; ``saas.worker`` runs it in a sandboxed child process, never
the web request. Job status: QUEUED -> RUNNING -> COMPLETED | FAILED,
or CANCELLED when the file is deleted.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ingest import DEFAULT_DEMAND_MODEL
from profiler.behavior import OUTPUT_MATCHES
from profiler.workflow import ORDER_RESOLVERS
from saas import db
from saas.auth import Principal, principal
from saas.config import get_settings
from saas.files import get_file
from saas.storage import get_storage

ACTIVE = ("QUEUED", "RUNNING")
ANALYZABLE = ("READY", "COMPLETED", "FAILED")  # FAILED/COMPLETED: re-analysis

router = APIRouter(prefix="/api")


class JobRequest(BaseModel):
    """The only parameters a job accepts; anything else is a 422."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["analyze", "finops"] = "analyze"
    workflow: str | None = Field(None, max_length=200)
    unit_cost: float | None = Field(None, ge=0, le=1e9)
    unit_value: float | None = Field(None, ge=0, le=1e9)
    capacity_headroom: float = Field(2.0, gt=0, le=100)
    demand_model: str = Field(DEFAULT_DEMAND_MODEL, max_length=50)
    token_prices: tuple[float, float] | None = None
    output_match: str = "exact"
    order_resolver: str = "simple"
    runs_per_month: float | None = Field(None, gt=0, le=1e9)

    @model_validator(mode="after")
    def _choices(self):
        if self.output_match not in OUTPUT_MATCHES:
            raise ValueError(f"output_match must be one of {', '.join(OUTPUT_MATCHES)}")
        if self.order_resolver not in ORDER_RESOLVERS:
            raise ValueError(f"order_resolver must be one of {', '.join(ORDER_RESOLVERS)}")
        if self.token_prices and min(self.token_prices) < 0:
            raise ValueError("token prices can't be negative")
        return self


def _public(row: sqlite3.Row, result_id: str | None) -> dict:
    return {
        "id": row["id"],
        "file_id": row["file_id"],
        "kind": row["kind"],
        "params": json.loads(row["params"]),
        "status": row["status"],
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "result_id": result_id,
    }


def _result_id(conn: sqlite3.Connection, tenant_id: str, job_id: str) -> str | None:
    row = conn.execute("SELECT id FROM results WHERE tenant_id = ? AND job_id = ?", (tenant_id, job_id)).fetchone()
    return row["id"] if row else None


@router.post("/files/{file_id}/jobs", status_code=202)
def create_job(file_id: str, req: JobRequest, p: Principal = Depends(principal)) -> dict:  # noqa: B008
    """Queue an analysis (or re-analysis) of the file."""
    s = get_settings()
    params = req.model_dump(exclude={"kind"}, exclude_none=True)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")  # the active-job count and insert as one step
        row = get_file(conn, p, file_id)
        if row["status"] not in ANALYZABLE:
            raise HTTPException(status_code=409, detail=f"file is {row['status'].lower()}, not ready for analysis")
        active = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tenant_id = ? AND status IN ('QUEUED', 'RUNNING')", (p.tenant_id,)
        ).fetchone()[0]
        if active >= s.max_active_jobs:
            db.audit(conn, "job.create", "denied", tenant_id=p.tenant_id, user_id=p.user_id,
                     resource_type="file", resource_id=file_id, detail="active job limit")
            conn.commit()
            raise HTTPException(status_code=429, detail=f"at most {s.max_active_jobs} jobs can run at once",
                                headers={"Retry-After": "30"})
        job_id = db.new_id()
        conn.execute(
            "INSERT INTO jobs (id, tenant_id, file_id, created_by, kind, params, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'QUEUED', ?)",
            (job_id, p.tenant_id, file_id, p.user_id, req.kind, json.dumps(params), db.now()),
        )
        db.audit(conn, "job.create", tenant_id=p.tenant_id, user_id=p.user_id, resource_type="job",
                 resource_id=job_id, detail=req.kind)
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _public(job, None)


def _get_job(conn: sqlite3.Connection, p: Principal, job_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE id = ? AND tenant_id = ?", (job_id, p.tenant_id)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return row


@router.get("/jobs")
def list_jobs(file_id: str | None = None, p: Principal = Depends(principal)) -> dict:  # noqa: B008
    with db.connect() as conn:
        query = "SELECT * FROM jobs WHERE tenant_id = ?"
        args: list = [p.tenant_id]
        if file_id:
            query += " AND file_id = ?"
            args.append(file_id)
        rows = conn.execute(query + " ORDER BY created_at DESC, id LIMIT 200", args).fetchall()
        return {"jobs": [_public(r, _result_id(conn, p.tenant_id, r["id"])) for r in rows]}


@router.get("/jobs/{job_id}")
def read_job(job_id: str, p: Principal = Depends(principal)) -> dict:  # noqa: B008
    with db.connect() as conn:
        row = _get_job(conn, p, job_id)
        return _public(row, _result_id(conn, p.tenant_id, job_id))


@router.get("/results/{result_id}")
def read_result(result_id: str, p: Principal = Depends(principal)) -> Response:  # noqa: B008
    """The job's report (JSON, as the legacy /analyze or /finops returns)."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT storage_key FROM results WHERE id = ? AND tenant_id = ?", (result_id, p.tenant_id)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="result not found")
        db.audit(conn, "result.read", tenant_id=p.tenant_id, user_id=p.user_id, resource_type="result",
                 resource_id=result_id)
    return Response(get_storage().get(row["storage_key"]), media_type="application/json")
