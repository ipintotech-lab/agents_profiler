"""CVE workflow: stable severity/routing with one variable analysis step."""
from .common import Run


def generate() -> list[Run]:
    runs = []
    cases = [("CVE-2026-4101", "critical", True), ("CVE-2026-4102", "high", False), ("CVE-2026-4103", "critical", False)]
    for i, (cve, severity, issue_exists) in enumerate(cases, 1):
        run = Run("cve-remediation-agent", "security-agent", f"cve-{i:02d}", "v1.0")
        parent = run.run_id
        state = {"severity": severity, "fix_available": not issue_exists}
        run.action("classify_severity", state=state, decision=("severity_mapping", severity), model="demo-llm-v1", tokens=(760+i*9, 36+i), latency_ms=390+i*12, cost=.0038+i*.0001, parent=parent)
        run.action("lookup_component_owner", event_type="tool_call", state={"cve_id": cve}, tool="catalog.get_component_owner", latency_ms=120+i*7, parent=parent)
        run.action("analyze_remediation", state={**state, "component": f"service-{i}"}, decision=("remediation", "upgrade" if not issue_exists else "mitigate"), model="demo-llm-v1", tokens=(1800+i*100, 220+i*12), latency_ms=1250+i*90, cost=.014+i*.001, parent=parent)
        run.action("route_to_team", state={"severity": severity, "component_group": "platform"}, decision=("team_route", "security-oncall" if severity == "critical" else "platform-team"), parent=parent)
        run.action("create_or_update_ticket", event_type="tool_call", state={"issue_exists": issue_exists}, decision=("ticket_action", "update" if issue_exists else "create"), tool="jira.create_or_update", latency_ms=210+i*10, parent=parent)
        runs.append(run)
    return runs
