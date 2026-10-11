"""CVE workflow: severity classification and routing, with randomized
case data so repeated generation produces different (but structurally
consistent) traces."""
import random

from .common import Run

SEVERITIES = ["low", "medium", "high", "critical"]


def generate(rng: random.Random | None = None) -> list[Run]:
    rng = rng or random.Random()
    cve = f"CVE-2026-{rng.randint(1000, 9999)}"
    severity = rng.choice(SEVERITIES)
    issue_exists = rng.random() < 0.5
    run_id = f"cve-{rng.randint(0, 999999):06d}"
    run = Run("cve-remediation-agent", "security-agent", run_id, "v1.0")
    parent = run.run_id
    state = {"severity": severity, "fix_available": not issue_exists}
    run.action(
        "classify_severity", state=state, decision=("severity_mapping", severity), model="demo-llm-v1",
        tokens=(rng.randint(700, 900), rng.randint(30, 60)), latency_ms=rng.randint(350, 500),
        cost=round(rng.uniform(0.003, 0.006), 5), parent=parent,
    )
    run.action(
        "lookup_component_owner", event_type="tool_call", state={"cve_id": cve},
        tool="catalog.get_component_owner", latency_ms=rng.randint(100, 200), parent=parent,
    )
    component = f"service-{rng.randint(1, 50)}"
    run.action(
        "analyze_remediation", state={**state, "component": component},
        decision=("remediation", "upgrade" if not issue_exists else "mitigate"), model="demo-llm-v1",
        tokens=(rng.randint(1700, 2200), rng.randint(200, 320)), latency_ms=rng.randint(1150, 1600),
        cost=round(rng.uniform(0.013, 0.02), 5), parent=parent,
    )
    run.action(
        "route_to_team", state={"severity": severity, "component_group": "platform"},
        decision=("team_route", "security-oncall" if severity in ("high", "critical") else "platform-team"),
        parent=parent,
    )
    run.action(
        "create_or_update_ticket", event_type="tool_call", state={"issue_exists": issue_exists},
        decision=("ticket_action", "update" if issue_exists else "create"), tool="jira.create_or_update",
        latency_ms=rng.randint(180, 260), parent=parent,
    )
    return [run]
