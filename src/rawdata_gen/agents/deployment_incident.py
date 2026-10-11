"""Deployment incident workflow with a rule-based rollback choice,
randomized so repeated generation produces different (but structurally
consistent) traces."""
import random

from ..ground_truth import apply_noise
from .common import Run, jittered_int, jittered_uniform

ERROR_RATES = ["normal", "elevated"]
_SEVERITIES = ["high", "low"]


def generate(rng: random.Random | None = None, variance: str = "mid") -> list[Run]:
    rng = rng or random.Random()
    deployment = f"deploy-{rng.randint(1000, 9999)}"
    error_rate = rng.choice(ERROR_RATES)
    rollback_allowed = rng.random() < 0.5
    run_id = f"deploy-{rng.randint(0, 999999):06d}"
    run = Run("deployment-incident-response", "release-agent", run_id, "v1.0")
    parent = run.run_id
    state = {"deployment_id": deployment, "error_rate": error_rate}
    run.action(
        "inspect_release_metrics", event_type="tool_call", state=state,
        tool="monitoring.get_release_metrics", latency_ms=jittered_int(rng, 150, 250, variance, floor=1),
        parent=parent,
    )
    severity_canonical = "high" if error_rate == "elevated" else "low"
    severity = apply_noise(rng, "assess_incident", severity_canonical, _SEVERITIES)
    run.action(
        # Canonicalized on error_rate/rollback_allowed only -- deployment_id
        # is unique per run and would otherwise prevent any repeated-input
        # evidence from ever accumulating for this node.
        "assess_incident", state={"error_rate": error_rate, "rollback_allowed": rollback_allowed},
        decision=("incident_severity", severity), model="demo-llm-v1",
        tokens=(jittered_int(rng, 600, 800, variance, floor=1), jittered_int(rng, 40, 75, variance, floor=1)),
        latency_ms=jittered_int(rng, 420, 650, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.004, 0.0065, variance), 5), parent=parent,
    )
    run.action(
        "choose_mitigation", state={"error_rate": error_rate, "rollback_allowed": rollback_allowed},
        decision=("mitigation", "rollback" if error_rate == "elevated" and rollback_allowed else "hold_and_alert"),
        parent=parent,
    )
    run.action(
        "apply_mitigation", event_type="tool_call",
        state={"deployment_id": deployment, "rollback_allowed": rollback_allowed},
        tool="release.apply_rollback" if rollback_allowed else "release.pause_rollout",
        latency_ms=jittered_int(rng, 300, 450, variance, floor=1), parent=parent,
    )
    run.action(
        "notify_oncall", event_type="tool_call", state={"deployment_id": deployment},
        tool="messaging.notify_oncall", latency_ms=jittered_int(rng, 100, 180, variance, floor=1), parent=parent,
    )
    return [run]
