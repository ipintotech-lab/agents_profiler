"""Shared event builder for synthetic AI Agent Profiler traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Run:
    def __init__(self, workflow: str, agent: str, run_id: str, version: str):
        self.workflow = workflow
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
