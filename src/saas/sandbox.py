"""The child process that parses and analyzes one customer file.

Started by ``saas.worker`` with ``python -I -m saas.sandbox``, an empty
environment (no secrets, no database path), a throwaway working
directory and rlimits on memory, CPU time, open files and file writes.
It reads a JSON header line and then the file's bytes on stdin, and writes
one JSON document on stdout: ``{"result": ...}`` or ``{"error": ...}``.
It never sees the database or storage, so a parser bug triggered by a
hostile file can't reach other tenants' data.

Error messages name what failed, never the content -- pydantic and json
errors quote input, so they're reduced to their type here.
"""
from __future__ import annotations

import json
import os
import sys

from pydantic import ValidationError

from ingest import normalize_raw_json, parse_raw_text
from ingest.trace_schema import TraceEvent
from ingest.usage import (
    detect_kind,
    normalize_activations,
    normalize_usage,
    parse_records,
)
from profiler import run_pipeline
from profiler.finops import build_finops
from profiler.usage import build_aggregate


class JobError(Exception):
    """A failure that is safe to show the customer."""


def _events(text: str) -> list[TraceEvent]:
    try:
        rows = normalize_raw_json(parse_raw_text(text))
        return [TraceEvent.model_validate(row) for row in rows]
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"][-1:]) for e in exc.errors()})[:5]
        raise JobError(f"the file's records don't match the trace format (fields: {', '.join(fields)})") from None
    except (json.JSONDecodeError, ValueError):
        rows = parse_records(text)
        if rows and detect_kind(rows):
            raise JobError(f"the file holds aggregate {detect_kind(rows)} records (no steps); "
                           "run a finops job for a usage report") from None
        raise JobError("the file could not be parsed as trace events") from None


def run(kind: str, params: dict, filename: str, data: bytes) -> dict:
    text = data.decode("utf-8-sig")
    if kind == "finops":
        rows = parse_records(text)
        usage_kind = detect_kind(rows) if rows else None
        if usage_kind is not None:
            try:
                normalize = normalize_usage if usage_kind == "usage" else normalize_activations
                normalized = normalize(rows)
            except ValueError:
                raise JobError(f"the {usage_kind} records could not be read") from None
            usage, activations = (normalized, []) if usage_kind == "usage" else ([], normalized)
            return build_aggregate(usage, activations, [{"name": filename, "kind": usage_kind,
                                                         "num_records": len(rows)}])
        events = _events(text)
        result = build_finops(events, params.get("output_match", "exact"), params.get("runs_per_month"))
    else:
        events = _events(text)
        prices = params.get("token_prices")
        result = run_pipeline(
            events,
            workflow=params.get("workflow"),
            unit_cost=params.get("unit_cost"),
            unit_value=params.get("unit_value"),
            capacity_headroom=params.get("capacity_headroom", 2.0),
            demand_model=params["demand_model"],
            token_prices=tuple(prices) if prices else None,
            output_match=params.get("output_match", "exact"),
            order_resolver=params.get("order_resolver", "simple"),
        )
        result["observe"]["files"] = [{"name": filename, "num_events": len(events)}]
        return result
    result["files"] = [{"name": filename, "num_events": len(events)}]
    return result


def main() -> None:
    # libraries print banners on stdout (the LP solver's license notice);
    # send fd 1 to stderr and keep the real stdout for the one JSON reply
    reply = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    header = json.loads(sys.stdin.buffer.readline())
    data = sys.stdin.buffer.read()
    try:
        out = {"result": run(header["kind"], header["params"], header["filename"], data)}
    except JobError as exc:
        out = {"error": str(exc)}
    except ValueError as exc:
        # the pipeline's own ValueErrors describe options (unknown
        # workflow, demand model), not the file's content
        out = {"error": str(exc)[:300]}
    except MemoryError:
        out = {"error": "the file needs more memory than a job is allowed"}
    reply.write(json.dumps(out, default=str))
    reply.flush()


if __name__ == "__main__":
    main()
