"""The job worker: ``python -m saas.worker`` (a separate process, or on
Railway a separate service sharing the volume).

It claims one QUEUED job at a time, re-reads the job's tenant and file
from the database -- the queue carries ids only, so a job can't be
pointed at another tenant's file -- and runs ``saas.sandbox`` in a child
process with:

- an empty environment (no ``HF_TOKEN``, OIDC secret or database path),
- a throwaway working and temp directory,
- ``RLIMIT_AS`` (``JOB_MEMORY_BYTES``), ``RLIMIT_CPU``, ``RLIMIT_NOFILE``
  and ``RLIMIT_FSIZE`` limits, and a wall-clock timeout (``JOB_TIMEOUT_S``).

The child gets the file's bytes on stdin and has no access to storage
or the database. Network isolation (no egress) needs the worker's own
container -- P1 infrastructure.
"""
from __future__ import annotations

import argparse
import json
import logging
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta

from saas import db
from saas.config import get_settings
from saas.storage import get_storage, object_key

log = logging.getLogger("saas.worker")

MAX_RESULT_BYTES = 64 << 20
# libraries probe the temp dir at import (dill); writes are capped, and go
# to the job's own directory
MAX_CHILD_FILE_BYTES = 16 << 20
POLL_S = 1.0


def _limits(memory_bytes: int, cpu_s: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_CHILD_FILE_BYTES, MAX_CHILD_FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run_sandboxed(kind: str, params: dict, filename: str, data: bytes) -> dict:
    """``{"result": ...}`` or ``{"error": ...}`` from the sandbox child."""
    s = get_settings()
    header = json.dumps({"kind": kind, "params": params, "filename": filename}).encode() + b"\n"
    with tempfile.TemporaryDirectory(prefix="job-") as cwd:
        try:
            proc = subprocess.run(  # fixed argv, no shell
                [sys.executable, "-I", "-m", "saas.sandbox"],
                input=header + data,
                capture_output=True,
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "TMPDIR": cwd},
                timeout=s.job_timeout_s,
                preexec_fn=_limits(s.job_memory_bytes, s.job_timeout_s),  # safe: the worker is single-threaded
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"the analysis took longer than {s.job_timeout_s}s"}
    if proc.returncode != 0 or not proc.stdout:
        # the child's stderr can quote the file; log only its size
        log.warning("sandbox exited %s (stderr %d bytes)", proc.returncode, len(proc.stderr))
        log.debug("sandbox stderr tail: %s", proc.stderr[-2000:])  # may quote the file: debug only
        if proc.returncode < 0 or b"MemoryError" in proc.stderr:
            return {"error": "the analysis exceeded its CPU or memory limit"}
        return {"error": "the analysis failed"}
    if len(proc.stdout) > MAX_RESULT_BYTES:
        return {"error": "the analysis result is too large"}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": "the analysis failed"}


def _claim() -> dict | None:
    with db.connect() as conn:
        row = conn.execute(
            "UPDATE jobs SET status = 'RUNNING', started_at = ? WHERE id = ("
            "SELECT id FROM jobs WHERE status = 'QUEUED' ORDER BY created_at, id LIMIT 1"
            ") AND status = 'QUEUED' RETURNING id, tenant_id, file_id, kind, params",
            (db.now(),),
        ).fetchone()
        if row is None:
            return None
        job = dict(row)
        # the file as the database has it, in the job's own tenant
        file = conn.execute(
            "SELECT id, storage_key, original_filename, status FROM files WHERE id = ? AND tenant_id = ?",
            (job["file_id"], job["tenant_id"]),
        ).fetchone()
        if file is None or file["status"] == "DELETED":
            conn.execute("UPDATE jobs SET status = 'CANCELLED', finished_at = ? WHERE id = ?", (db.now(), job["id"]))
            return {}
        conn.execute("UPDATE files SET status = 'PROCESSING' WHERE id = ? AND tenant_id = ?",
                     (file["id"], job["tenant_id"]))
        job["file"] = dict(file)
    return job


def _finish(job: dict, outcome: dict) -> None:
    tenant_id, file_id = job["tenant_id"], job["file_id"]
    storage = get_storage()
    result_key = None
    if "result" in outcome:
        result_id = db.new_id()
        result_key = object_key(tenant_id, "results", result_id)
        payload = json.dumps(outcome["result"], default=str).encode()
        storage.put(result_key, payload)
    with db.connect() as conn:
        current = conn.execute("SELECT status FROM jobs WHERE id = ? AND tenant_id = ?",
                               (job["id"], tenant_id)).fetchone()
        file = conn.execute("SELECT status FROM files WHERE id = ? AND tenant_id = ?",
                            (file_id, tenant_id)).fetchone()
        if current["status"] != "RUNNING" or file["status"] == "DELETED":
            # the file was deleted while the job ran: keep nothing
            if result_key:
                storage.delete(result_key)
            return
        if result_key:
            conn.execute(
                "INSERT INTO results (id, tenant_id, job_id, file_id, storage_key, size, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (result_id, tenant_id, job["id"], file_id, result_key, len(payload), db.now()),
            )
        status = "COMPLETED" if result_key else "FAILED"
        conn.execute("UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                     (status, outcome.get("error"), db.now(), job["id"]))
        conn.execute("UPDATE files SET status = ? WHERE id = ? AND tenant_id = ?", (status, file_id, tenant_id))
        db.audit(conn, "job.finish", "ok" if result_key else "failed", tenant_id=tenant_id,
                 resource_type="job", resource_id=job["id"])


def requeue_stale() -> int:
    """Jobs left RUNNING by a worker that died go back to the queue."""
    cutoff = (datetime.now(UTC) - timedelta(seconds=2 * get_settings().job_timeout_s + 60)).isoformat(
        timespec="seconds")
    with db.connect() as conn:
        return conn.execute(
            "UPDATE jobs SET status = 'QUEUED', started_at = NULL WHERE status = 'RUNNING' AND started_at < ?",
            (cutoff,),
        ).rowcount


def run_once() -> bool:
    """Run the next queued job; ``False`` when the queue is empty."""
    job = _claim()
    if job is None:
        return False
    if not job:
        return True
    try:
        data = get_storage().get(job["file"]["storage_key"])
        outcome = run_sandboxed(job["kind"], json.loads(job["params"]), job["file"]["original_filename"], data)
    except Exception:
        log.exception("job %s failed before the sandbox", job["id"])
        outcome = {"error": "the analysis failed"}
    _finish(job, outcome)
    log.info("job %s %s", job["id"], "completed" if "result" in outcome else "failed")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="drain the queue and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not get_settings().saas:
        sys.exit("AUTH_MODE is demo: there are no customer jobs to run")
    requeue_stale()
    while True:
        if not run_once():
            if args.once:
                return
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
