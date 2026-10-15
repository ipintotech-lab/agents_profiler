"""Data-pipeline incident workflow: a nightly ETL job failed and the agent
works out why. Planted for the profiler to find:

* ``get_table_schema`` is called twice per run for one of a few tables and
  always returns the same schema (cache the result);
* ``check_job_status`` is also called with repeating inputs, but the job's
  status changes between calls -- NOT a cache candidate;
* ``decide_retry`` always makes the same choice for the same error type
  (replaceable by a rule);
* ``diagnose_failure`` varies for the same input (hidden state).
"""
import random

from ..ground_truth import apply_noise
from .common import Run, jittered_int, jittered_uniform

JOBS = ["orders_daily", "events_hourly", "users_snapshot", "revenue_rollup"]
TABLES = ["orders", "events", "users", "revenue", "sessions"]
ERRORS = ["timeout", "schema_mismatch", "null_constraint"]
_ROOT_CAUSES = ["upstream_late", "column_renamed", "bad_source_rows"]
_CAUSE_BY_ERROR = {"timeout": "upstream_late", "schema_mismatch": "column_renamed", "null_constraint": "bad_source_rows"}


def generate(rng: random.Random | None = None, variance: str = "mid") -> list[Run]:
    rng = rng or random.Random()
    job = rng.choice(JOBS)
    table = rng.choice(TABLES)
    error = rng.choice(ERRORS)
    job_run = f"jr-{rng.randint(0, 999999):06d}"
    run_id = f"etl-{rng.randint(0, 999999):06d}"
    run = Run("etl-incident-response", "data-pipeline-agent", run_id, "v1.0")
    parent = run.run_id

    def schema() -> None:
        run.action(
            "get_table_schema", event_type="tool_call", state={"table": table}, tool="warehouse.describe_table",
            latency_ms=jittered_int(rng, 300, 800, variance, floor=1), result=f"schema:{table}:v3", parent=parent,
        )

    status = apply_noise(rng, "check_job_status", "failed", ["failed", "retrying", "succeeded"])
    run.action(
        "check_job_status", event_type="tool_call", state={"job": job}, tool="scheduler.get_job_status",
        latency_ms=jittered_int(rng, 80, 160, variance, floor=1), result=f"status:{status}", parent=parent,
    )
    schema()
    run.action(
        "fetch_error_log", event_type="tool_call", state={"job": job, "job_run": job_run}, tool="scheduler.get_logs",
        latency_ms=jittered_int(rng, 400, 900, variance, floor=1), result=f"log:{job_run}", parent=parent,
    )
    cause = apply_noise(rng, "diagnose_failure", _CAUSE_BY_ERROR[error], _ROOT_CAUSES)
    run.action(
        "diagnose_failure", state={"error_type": error, "table": table}, decision=("root_cause", cause),
        model="demo-llm-v1",
        tokens=(jittered_int(rng, 3500, 5000, variance, floor=1), jittered_int(rng, 150, 300, variance, floor=1)),
        latency_ms=jittered_int(rng, 1800, 3200, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.02, 0.03, variance), 5), parent=parent,
    )
    schema()  # re-checks the schema before deciding -- same table, same answer
    run.action(
        "decide_retry", state={"error_type": error},
        decision=("retry_action", "retry" if error == "timeout" else "escalate"), model="demo-llm-v1",
        tokens=(jittered_int(rng, 900, 1300, variance, floor=1), jittered_int(rng, 10, 25, variance, floor=1)),
        latency_ms=jittered_int(rng, 350, 600, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.003, 0.005, variance), 5), parent=parent,
    )
    run.action(
        "trigger_rerun" if error == "timeout" else "page_data_owner", event_type="tool_call",
        state={"job": job, "job_run": job_run},
        tool="scheduler.rerun" if error == "timeout" else "pager.page_owner",
        latency_ms=jittered_int(rng, 150, 300, variance, floor=1), parent=parent,
    )
    run.action(
        "post_incident_note", event_type="tool_call", state={"job_run": job_run}, tool="slack.post_message",
        latency_ms=jittered_int(rng, 100, 200, variance, floor=1), parent=parent,
    )
    return [run]
