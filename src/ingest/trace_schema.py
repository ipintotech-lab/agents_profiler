"""Normalized agent-workflow trace event, matching the data model in
§7 of the AI Agent Profiler design doc (trace/run/workflow identity,
decision, tool, cost, tokens, latency, outcome)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0


class Decision(BaseModel):
    type: str
    selected: str


class Tool(BaseModel):
    name: str


class TraceEvent(BaseModel):
    trace_id: str
    run_id: str
    workflow: str
    agent: str | None = None
    event_type: str
    node: str
    parent: str | None = None
    input_state: dict = Field(default_factory=dict)
    decision: Decision | None = None
    tool: Tool | None = None
    model: str | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    cost: float = 0.0
    result: str | None = None
    workflow_version: str | None = None
