"""Deployment incident workflow with a rule-based rollback choice,
randomized so repeated generation produces different (but structurally
consistent) traces."""
import random

from .common import Run

ERROR_RATES = ["normal", "elevated"]


def generate(rng: random.Random | None = None) -> list[Run]:
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
        tool="monitoring.get_release_metrics", latency_ms=rng.randint(150, 250), parent=parent,
    )
    run.action(
        "assess_incident", state={**state, "rollback_allowed": rollback_allowed},
        decision=("incident_severity", "high" if error_rate == "elevated" else "low"), model="demo-llm-v1",
        tokens=(rng.randint(600, 800), rng.randint(40, 75)), latency_ms=rng.randint(420, 650),
        cost=round(rng.uniform(0.004, 0.0065), 5), parent=parent,
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
        latency_ms=rng.randint(300, 450), parent=parent,
    )
    run.action(
        "notify_oncall", event_type="tool_call", state={"deployment_id": deployment},
        tool="messaging.notify_oncall", latency_ms=rng.randint(100, 180), parent=parent,
    )
    return [run]
