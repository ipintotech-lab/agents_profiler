"""What the person running the analysis tells us about their agents,
before the traces are read: how bad a wrong decision is, which steps change
something, which changes they'd consider. Traces show what happened; they
don't say what a mistake costs or what may be changed.

Every answer is a declaration, not a measurement: the context is marked
``source="user_declared"`` and ``status="unverified"``, and the report says
which conclusions rest on it. Answers can only make the analysis stricter
or fill a gap the traces leave:

- a step said to change something is treated as changing something, even if
  the trace declares otherwise;
- a step said to only read counts only where the trace declares nothing;
- the failure impact can raise a finding's risk, never lower it;
- allowed changes hide findings, and the report counts what was hidden.

Answers are per workflow; ``"*"`` applies to every workflow not named.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

CONTEXT_SCHEMA_VERSION = 1
ALL_WORKFLOWS = "*"

# The kinds of change a finding proposes. REVIEW_HIDDEN_STATE proposes no
# change, so it can't be switched off.
OPTIMIZATIONS = ("CACHE_RESULT", "CONDITIONAL_LLM_BYPASS", "EXTRACT_DETERMINISTIC_SUBSTEP", "DETERMINISTIC_REGION")


class WorkflowContext(BaseModel):
    domain: str | None = None
    goal: str | None = None
    failure_impact: Literal["low", "medium", "high"] | None = None
    side_effect_operations: list[str] = Field(default_factory=list)
    read_only_operations: list[str] = Field(default_factory=list)
    allowed_optimizations: list[Literal[OPTIMIZATIONS]] | None = None  # None: all
    requires_human_approval: bool = False
    # recorded for the report; the optimizer doesn't use them yet
    priority: Literal["cost", "speed", "quality"] | None = None
    model: str | None = None  # the LLM the agent runs on, e.g. "GLM-4.6"

    @model_validator(mode="after")
    def _no_operation_in_both(self) -> WorkflowContext:
        both = set(self.side_effect_operations) & set(self.read_only_operations)
        if both:
            raise ValueError(f"listed both as changing something and as read-only: {', '.join(sorted(both))}")
        return self

    def user_side_effect(self, names: set[str]) -> bool | None:
        """True if any of ``names`` (a step and its tools) was said to change
        something, False if one was said to only read, else None."""
        if names & set(self.side_effect_operations):
            return True
        if names & set(self.read_only_operations):
            return False
        return None

    def allows(self, kind: str) -> bool:
        return kind not in OPTIMIZATIONS or self.allowed_optimizations is None or kind in self.allowed_optimizations

    def needs_approval(self) -> bool:
        return self.requires_human_approval or self.failure_impact == "high"


class AnalysisContext(BaseModel):
    schema_version: Literal[CONTEXT_SCHEMA_VERSION] = CONTEXT_SCHEMA_VERSION
    source: Literal["user_declared"] = "user_declared"
    status: Literal["unverified"] = "unverified"
    workflows: dict[str, WorkflowContext] = Field(default_factory=dict)

    def for_workflow(self, workflow: str) -> WorkflowContext | None:
        return self.workflows.get(workflow) or self.workflows.get(ALL_WORKFLOWS)


def parse_context(text: str | None) -> AnalysisContext | None:
    """Parse the JSON a client sends; ``ValueError`` names what's wrong."""
    if not text or not text.strip():
        return None
    try:
        return AnalysisContext.model_validate(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse context: {exc}") from exc
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'context'}: {e['msg']}" for e in exc.errors())
        raise ValueError(f"invalid context: {problems}") from exc
