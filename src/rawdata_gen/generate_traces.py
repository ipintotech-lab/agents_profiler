"""Build configurable sets of mock-agent runs for the dashboard's
"generate sample data" path."""
import copy

from .agents.common import Run
from .agents.cve_remediation import generate as generate_cve
from .agents.deployment_incident import generate as generate_deployment
from .agents.support_triage import generate as generate_support

AGENT_GENERATORS = [generate_cve, generate_support, generate_deployment]


def _build_run(generator, action_count: int, run_id: str) -> Run:
    """Clone a generator's first run, resized to exactly ``action_count`` events."""
    template = generator()[0]
    template.run_id = run_id
    template.trace_id = f"trace-{run_id}"
    template.events = template.events[:action_count]
    while len(template.events) < action_count:
        event = copy.deepcopy(template.events[-1])
        event["node"] = f"{event['node']}_followup"
        template.events.append(event)
    for event in template.events:
        event["run_id"] = template.run_id
        event["trace_id"] = template.trace_id
        if "parent" in event:
            event["parent"] = template.run_id
    return template


def generate_configured(agent_count: int, actions_per_agent: list[int], total_lines: int | None = None) -> list[Run]:
    """Create runs for the selected agents with the requested action counts.

    Without ``total_lines`` one run per agent is created. With it, runs are
    repeated round-robin across the agents until exactly ``total_lines``
    events exist; the final run is truncated if needed.
    """
    if not 1 <= agent_count <= len(AGENT_GENERATORS):
        raise ValueError(f"agent_count must be between 1 and {len(AGENT_GENERATORS)}")
    if len(actions_per_agent) != agent_count:
        raise ValueError("provide exactly one action count per agent")
    if any(not 4 <= count <= 20 for count in actions_per_agent):
        raise ValueError("each action count must be between 4 and 20")
    if total_lines is not None and total_lines < 1:
        raise ValueError("total_lines must be at least 1")

    agents = list(zip(AGENT_GENERATORS, actions_per_agent))
    if total_lines is None:
        return [
            _build_run(generator, action_count, f"custom-{index:02d}")
            for index, (generator, action_count) in enumerate(agents, 1)
        ]

    runs = []
    remaining = total_lines
    cycle = 0
    while remaining > 0:
        cycle += 1
        for index, (generator, action_count) in enumerate(agents, 1):
            if remaining <= 0:
                break
            run = _build_run(generator, action_count, f"custom-{index:02d}-{cycle:04d}")
            run.events = run.events[:remaining]
            remaining -= len(run.events)
            runs.append(run)
    return runs
