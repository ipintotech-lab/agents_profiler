"""Action identity: which observed calls count as "the same action" when a
recommendation reuses one call's result for another (cache) or replaces
the call with a rule (bypass).

Two calls share an identity only if they agree on agent, node, action
type, input, model and workflow version. A field the trace doesn't carry
becomes ``"unknown"`` -- never omitted -- so a call with no recorded model
is never merged with one that ran on a known model: missing is not equal.

``missing_fields`` names the identity fields that are unknown where they
matter. A tool call has no model by nature, so model is only required of
LLM actions. A recommendation built on a group with missing fields is
exploratory: it can't confirm the calls ran under the same configuration.

``determinism.classify`` keeps its own input-only key: its job is to judge
the action, and the fingerprint levels already show how much model and
version explain.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ingest.trace_schema import TraceEvent

IDENTITY_SCHEMA_VERSION = 1
"""Bump whenever the fields of ``ActionIdentity`` change, so identities
from different schema versions never compare equal."""

UNKNOWN = "unknown"


def action_type(event: TraceEvent) -> str:
    if event.decision is not None:
        return event.decision.type
    if event.tool is not None:
        return event.tool.name
    return event.event_type


@dataclass(frozen=True)
class ActionIdentity:
    agent: str
    node: str
    action_type: str
    input: str
    """Canonical JSON of ``input_state``."""
    model: str
    workflow_version: str
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "input"}


def action_identity(event: TraceEvent) -> ActionIdentity:
    return ActionIdentity(
        agent=event.agent or UNKNOWN,
        node=event.node,
        action_type=action_type(event),
        input=json.dumps(event.input_state, sort_keys=True, default=str),
        model=event.model or UNKNOWN,
        workflow_version=event.workflow_version or UNKNOWN,
    )


def missing_fields(identity: ActionIdentity, is_llm: bool) -> list[str]:
    """Identity fields that are unknown where they matter: model only for an
    LLM action (one where any call records a model)."""
    required = ["workflow_version"] + (["model"] if is_llm else [])
    return [f for f in required if getattr(identity, f) == UNKNOWN]


def describe(identity: ActionIdentity, is_llm: bool) -> dict:
    """The identity as shown on a recommendation."""
    return {**identity.to_dict(), "missing": missing_fields(identity, is_llm)}
