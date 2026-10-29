"""Normalize alternate raw trace-data shapes into the flat dict shape
``TraceEvent`` expects, so an uploaded file doesn't have to already match
this app's exact schema to be analyzed.

Handles these cases beyond a bare list of already-correct TraceEvent
dicts: events wrapped under a different top-level key (``{"events": [...]}``,
``{"data": {"traces": [...]}}``), an OpenTelemetry OTLP JSON span export
(``resourceSpans`` / ``scopeSpans`` / ``spans``), a Claude Code session
transcript (see ``claude_code.py``), OpenAI-style chat trajectories (see
``chat_trajectory.py``), FreeInference agentic-trace sessions (see
``freeinference.py``), and per-record field-name aliases (``spanId`` -> ``run_id``, ``duration_ms`` -> ``latency_ms``, etc).

A record that still can't be mapped to the required TraceEvent fields
after normalization raises ``ValueError`` naming the missing fields --
callers should reject the whole file rather than guess at bad data.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .chat_trajectory import convert_chat_trajectories, is_chat_trajectory
from .claude_code import convert_claude_code_transcript, is_claude_code_transcript
from .freeinference import convert_freeinference_sessions, is_freeinference_session

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "trace_id": ("trace_id", "traceId", "traceID", "trace"),
    "run_id": ("run_id", "runId", "spanId", "span_id", "call_id"),
    "workflow": ("workflow", "workflow_name", "workflowName"),
    "agent": ("agent", "agent_name", "agentName"),
    "event_type": ("event_type", "eventType", "type", "kind"),
    "node": ("node", "name", "span_name", "spanName"),
    "parent": ("parent", "parentSpanId", "parent_span_id", "parent_run_id"),
    "model": ("model", "model_name", "modelName"),
    "latency_ms": ("latency_ms", "duration_ms", "durationMs"),
    "cost": ("cost", "cost_usd", "costUsd"),
    "result": ("result", "output", "outputValue", "output_value"),
    "workflow_version": ("workflow_version", "workflowVersion"),
    "trigger": ("trigger", "triggered_by", "triggeredBy"),
    "user_input": ("user_input", "userInput", "user_prompt", "prompt"),
    "has_side_effects": ("has_side_effects", "hasSideEffects", "side_effects", "sideEffects"),
    "idempotent": ("idempotent", "isIdempotent"),
    "cache_eligible": ("cache_eligible", "cacheEligible", "cacheable"),
    "cache_ttl_seconds": ("cache_ttl_seconds", "cacheTtlSeconds", "ttl_seconds"),
    "invalidation_keys": ("invalidation_keys", "invalidationKeys"),
}

_CONTAINER_KEYS = ("events", "data", "traces", "runs", "items", "results", "rows")

_REQUIRED_FIELDS = ("trace_id", "run_id", "workflow", "event_type", "node")

_OTEL_VALUE_KEYS = ("stringValue", "intValue", "doubleValue", "boolValue")


def _first_present(d: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _flatten_otel_attributes(attributes: list[dict] | None) -> dict:
    """OTLP JSON attributes are ``[{"key": k, "value": {"stringValue": v, ...}}, ...]``."""
    flat: dict = {}
    for attr in attributes or []:
        key = attr.get("key")
        value = attr.get("value") or {}
        for value_key in _OTEL_VALUE_KEYS:
            if value_key in value:
                flat[key] = value[value_key]
                break
    return flat


def _unwrap_otel(parsed: dict) -> list[dict] | None:
    resource_spans = parsed.get("resourceSpans")
    if resource_spans is None:
        return None
    events: list[dict] = []
    for resource_span in resource_spans:
        resource_attrs = _flatten_otel_attributes((resource_span.get("resource") or {}).get("attributes"))
        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                span_attrs = _flatten_otel_attributes(span.get("attributes"))
                flat = {**resource_attrs, **span_attrs, **span}
                start_ns = int(span.get("startTimeUnixNano") or 0)
                end_ns = int(span.get("endTimeUnixNano") or 0)
                if end_ns > start_ns:
                    flat["latency_ms"] = (end_ns - start_ns) / 1e6
                events.append(flat)
    return events


def _unwrap_container(parsed: Any) -> list:
    if isinstance(parsed, list):
        if is_claude_code_transcript(parsed):
            return convert_claude_code_transcript(parsed)
        if parsed and all(is_freeinference_session(r) for r in parsed):
            return convert_freeinference_sessions(parsed)
        if parsed and all(is_chat_trajectory(r) for r in parsed):
            return convert_chat_trajectories(parsed)
        return parsed
    if isinstance(parsed, dict):
        if is_freeinference_session(parsed) or is_chat_trajectory(parsed):
            return _unwrap_container([parsed])
        otel_events = _unwrap_otel(parsed)
        if otel_events is not None:
            return otel_events
        for key in _CONTAINER_KEYS:
            if key in parsed:
                nested = _unwrap_container(parsed[key])
                if nested:
                    return nested
        return [parsed]
    raise ValueError(
        "unrecognized JSON shape: expected a list of events, a dict wrapping "
        "one (e.g. {\"events\": [...]}), or an OpenTelemetry OTLP span export"
    )


def normalize_event(raw: dict) -> dict:
    """Map one raw record's aliased/alternate field names onto the flat
    dict shape ``TraceEvent.model_validate`` expects."""
    normalized: dict = dict(raw)
    for canonical, aliases in _FIELD_ALIASES.items():
        if normalized.get(canonical) is not None:
            continue
        value = _first_present(raw, aliases)
        if value is not None:
            normalized[canonical] = value

    if not isinstance(normalized.get("tokens"), dict):
        input_tokens = _first_present(raw, ("input_tokens", "inputTokens", "prompt_tokens", "promptTokens"))
        output_tokens = _first_present(raw, ("output_tokens", "outputTokens", "completion_tokens", "completionTokens"))
        if input_tokens is not None or output_tokens is not None:
            normalized["tokens"] = {"input": input_tokens or 0, "output": output_tokens or 0}

    if not isinstance(normalized.get("tool"), dict):
        tool_name = _first_present(raw, ("tool_name", "toolName"))
        if tool_name is not None:
            normalized["tool"] = {"name": tool_name}

    if not isinstance(normalized.get("decision"), dict):
        decision_type = _first_present(raw, ("decision_type", "decisionType"))
        selected = _first_present(raw, ("selected", "decision_selected"))
        if decision_type is not None and selected is not None:
            normalized["decision"] = {"type": decision_type, "selected": selected}

    missing = [field for field in _REQUIRED_FIELDS if not normalized.get(field)]
    if missing:
        raise ValueError(f"could not normalize record (no value or known alias for: {', '.join(missing)}): {raw!r}")
    return normalized


def normalize_raw_json(parsed: Any) -> list[dict]:
    """Accept parsed JSON in any of several shapes -- a bare list of
    TraceEvent-shaped dicts, the same wrapped under a container key, or an
    OpenTelemetry OTLP JSON span export -- and return a flat list of dicts
    ready for ``TraceEvent.model_validate``.

    Raises ``ValueError`` (naming the offending record/field) on the first
    record that can't be normalized; the whole file should be rejected
    rather than silently dropping or guessing at unmappable data.
    """
    rows = _unwrap_container(parsed)
    return [normalize_event(row) for row in rows]


def parse_raw_text(text: str) -> Any:
    """Parse a trace file's text as one JSON document, falling back to
    JSON Lines (one JSON value per line) when it isn't one document --
    a pretty-printed array has multiple non-empty lines too, so the
    single-document parse has to win when it succeeds."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def hash_input_values(rows: list[dict]) -> list[dict]:
    """Replace every ``input_state`` value with ``"sha256:<digest>"``,
    keeping the keys, and the ``user_input`` (the user prompt). Equal values still hash equal, so determinism
    classification is unchanged, but commands, file paths, edit text, etc.
    no longer appear in the data."""
    def digest(value: object) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]

    hashed = []
    for row in rows:
        if isinstance(row.get("user_input"), str):
            row = {**row, "user_input": digest(row["user_input"])}
        state = row.get("input_state")
        if isinstance(state, dict):
            row = {
                **row,
                "input_state": {key: digest(value) for key, value in state.items()},
            }
        hashed.append(row)
    return hashed
