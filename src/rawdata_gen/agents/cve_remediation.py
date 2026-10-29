"""CVE workflow: severity classification and routing, with randomized
case data so repeated generation produces different (but structurally
consistent) traces."""
import random

from ..ground_truth import apply_noise
from .common import Run, jittered_int, jittered_uniform

SEVERITIES = ["low", "medium", "high", "critical"]
_REMEDIATIONS = ["upgrade", "mitigate"]


_SURE_SEVERITIES = ("high", "critical")
_UNSURE_EXTRA_FLIP = 0.25


def _conditional_remediation(run_id: str, severity: str, canonical: str, noisy: str) -> str:
    """The planted region: for high/critical CVEs the remediation follows
    fix_available every time; for low/medium it flips 40% of the time --
    averaging the 80% in ``DETERMINISM_PROFILE``. The extra flips draw from
    a per-run RNG so the main sequence (and every other node) is unchanged."""
    if severity in _SURE_SEVERITIES:
        return canonical
    if noisy == canonical and random.Random(run_id).random() < _UNSURE_EXTRA_FLIP:
        return next(r for r in _REMEDIATIONS if r != canonical)
    return noisy


def generate(rng: random.Random | None = None, variance: str = "mid") -> list[Run]:
    rng = rng or random.Random()
    cve = f"CVE-2026-{rng.randint(1000, 9999)}"
    severity = rng.choice(SEVERITIES)
    issue_exists = rng.random() < 0.5
    run_id = f"cve-{rng.randint(0, 999999):06d}"
    run = Run("cve-remediation-agent", "security-agent", run_id, "v1.0", user_input=f"Remediate {cve}")
    parent = run.run_id
    state = {"severity": severity, "fix_available": not issue_exists}
    run.action(
        "classify_severity", state=state, decision=("severity_mapping", severity), model="demo-llm-v1",
        tokens=(jittered_int(rng, 700, 900, variance, floor=1), jittered_int(rng, 30, 60, variance, floor=1)),
        latency_ms=jittered_int(rng, 350, 500, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.003, 0.006, variance), 5), parent=parent,
    )
    run.action(
        "lookup_component_owner", event_type="tool_call", side_effects=False, state={"cve_id": cve},
        tool="catalog.get_component_owner", latency_ms=jittered_int(rng, 100, 200, variance, floor=1), parent=parent,
    )
    component = f"service-{rng.randint(1, 50)}"
    remediation_canonical = "upgrade" if not issue_exists else "mitigate"
    remediation = apply_noise(rng, "analyze_remediation", remediation_canonical, _REMEDIATIONS)
    remediation = _conditional_remediation(run_id, severity, remediation_canonical, remediation)
    run.action(
        "analyze_remediation", state={**state, "component": component},
        decision=("remediation", remediation), model="demo-llm-v1",
        tokens=(jittered_int(rng, 1700, 2200, variance, floor=1), jittered_int(rng, 200, 320, variance, floor=1)),
        latency_ms=jittered_int(rng, 1150, 1600, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.013, 0.02, variance), 5), parent=parent,
    )
    run.action(
        "route_to_team", state={"severity": severity, "component_group": "platform"},
        decision=("team_route", "security-oncall" if severity in ("high", "critical") else "platform-team"),
        parent=parent,
    )
    run.action(
        "create_or_update_ticket", event_type="tool_call", side_effects=True, state={"issue_exists": issue_exists},
        decision=("ticket_action", "update" if issue_exists else "create"), tool="jira.create_or_update",
        latency_ms=jittered_int(rng, 180, 260, variance, floor=1), parent=parent,
    )
    return [run]
