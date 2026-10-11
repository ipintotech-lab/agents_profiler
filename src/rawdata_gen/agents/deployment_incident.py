"""Deployment incident workflow with a rule-based rollback choice."""
from .common import Run


def generate() -> list[Run]:
    runs = []
    cases = [("deploy-5501", "elevated", True), ("deploy-5502", "normal", False), ("deploy-5503", "elevated", False)]
    for i, (deployment, error_rate, rollback_allowed) in enumerate(cases, 1):
        run = Run("deployment-incident-response", "release-agent", f"deploy-{i:02d}", "v1.0")
        parent = run.run_id
        state = {"deployment_id": deployment, "error_rate": error_rate}
        run.action("inspect_release_metrics", event_type="tool_call", state=state, tool="monitoring.get_release_metrics", latency_ms=180+i*15, parent=parent)
        run.action("assess_incident", state={**state, "rollback_allowed": rollback_allowed}, decision=("incident_severity", "high" if error_rate == "elevated" else "low"), model="demo-llm-v1", tokens=(650+i*35, 48+i*5), latency_ms=470+i*30, cost=.0045+i*.0003, parent=parent)
        run.action("choose_mitigation", state={"error_rate": error_rate, "rollback_allowed": rollback_allowed}, decision=("mitigation", "rollback" if error_rate == "elevated" and rollback_allowed else "hold_and_alert"), parent=parent)
        run.action("apply_mitigation", event_type="tool_call", state={"deployment_id": deployment, "rollback_allowed": rollback_allowed}, tool="release.apply_rollback" if rollback_allowed else "release.pause_rollout", latency_ms=320+i*25, parent=parent)
        run.action("notify_oncall", event_type="tool_call", state={"deployment_id": deployment}, tool="messaging.notify_oncall", latency_ms=110+i*10, parent=parent)
        runs.append(run)
    return runs
