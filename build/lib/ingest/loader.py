"""Load normalized agent-workflow trace rows (JSON Lines, one event per
line, matching trace_schema.TraceEvent) from the agents workflow's
telemetry export."""
from __future__ import annotations

import json
from pathlib import Path

from .trace_schema import TraceEvent


def load_traces(path: str | Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(TraceEvent.model_validate(json.loads(line)))
    return events
