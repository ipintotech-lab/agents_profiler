"""Shared event builder for synthetic AI Agent Profiler traces."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

VARIANCES = ("low", "mid", "high")

# Scales the half-width of a field's jitter range around its center. "mid" is
# the original tuned range; "low"/"high" tighten or widen it, so the same
# center of mass is kept but the spread of generated values -- and therefore
# of downstream cost_component/score -- changes.
_VARIANCE_SCALE = {"low": 0.3, "mid": 1.0, "high": 2.5}


def jittered_uniform(rng: random.Random, low: float, high: float, variance: str, floor: float = 0.0) -> float:
    """A random float centered between ``low``/``high``, spread scaled by ``variance``."""
    scale = _VARIANCE_SCALE[variance]
    center = (low + high) / 2
    half_width = (high - low) / 2 * scale
    return max(floor, rng.uniform(center - half_width, center + half_width))


def jittered_int(rng: random.Random, low: int, high: int, variance: str, floor: int = 0) -> int:
    """A random int centered between ``low``/``high``, spread scaled by ``variance``."""
    scale = _VARIANCE_SCALE[variance]
    center = (low + high) / 2
    half_width = (high - low) / 2 * scale
    lo = max(floor, round(center - half_width))
    hi = max(lo, round(center + half_width))
    return rng.randint(lo, hi)


class Run:
    def __init__(self, workflow: str, agent: str, run_id: str, version: str, user_input: str | None = None):
        self.workflow = workflow
        # the request that started the run, as a real trace records it
        self.user_input = user_input
        self.agent = agent
        self.run_id = run_id
        self.trace_id = f"trace-{run_id}"
        self.version = version
        self.events: list[dict[str, Any]] = []

    def action(
        self,
        node: str,
        *,
        event_type: str = "decision",
        state: dict[str, Any] | None = None,
        decision: tuple[str, str] | None = None,
        tool: str | None = None,
        model: str | None = None,
        tokens: tuple[int, int] = (0, 0),
        latency_ms: float = 0,
        cost: float = 0,
        result: str = "success",
        parent: str | None = None,
    ) -> None:
        """Append one normalized profiler event; prompts/arguments are omitted."""
        event: dict[str, Any] = {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "workflow": self.workflow,
            "agent": self.agent,
            "event_type": event_type,
            "node": node,
            "input_state": state or {},
            "tokens": {"input": tokens[0], "output": tokens[1]},
            "latency_ms": latency_ms,
            "cost": cost,
            "result": result,
            "workflow_version": self.version,
        }
        if self.user_input:
            event["user_input"] = self.user_input
        if parent:
            event["parent"] = parent
        if decision:
            event["decision"] = {"type": decision[0], "selected": decision[1]}
        if tool:
            event["tool"] = {"name": tool}
        if model:
            event["model"] = model
        self.events.append(event)


def write_jsonl(runs: list[Run], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for run in runs:
            for event in run.events:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
    return sum(len(run.events) for run in runs)


def to_jsonl(runs: list[Run]) -> str:
    """Same line format as ``write_jsonl``, returned as a string instead of
    written to disk -- used by the API to hand generated data straight to
    the analysis pipeline."""
    lines = [json.dumps(event, separators=(",", ":")) for run in runs for event in run.events]
    return "\n".join(lines)
