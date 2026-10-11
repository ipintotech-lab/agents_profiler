"""Build configurable sets of mock-agent runs for the dashboard's
"generate sample data" path."""
import copy
import random
import uuid
from datetime import UTC, datetime

from .agents.common import VARIANCES, Run
from .agents.cve_remediation import generate as generate_cve
from .agents.deployment_incident import generate as generate_deployment
from .agents.support_triage import generate as generate_support
from .ground_truth import build_ground_truth_manifest

AGENT_GENERATORS = [generate_cve, generate_support, generate_deployment]


def _build_run(generator, action_count: int, run_id: str, rng: random.Random, variance: str) -> Run:
    """Clone a freshly-randomized run from ``generator``, resized to exactly
    ``action_count`` events."""
    template = generator(rng, variance)[0]
    template.run_id = run_id
    template.trace_id = f"trace-{run_id}"
    template.events = template.events[:action_count]
    base_event = template.events[-1]
    base_node = base_event["node"]
    followup_num = 1
    while len(template.events) < action_count:
        event = copy.deepcopy(base_event)
        event["node"] = f"{base_node}_followup_{followup_num}"
        template.events.append(event)
        followup_num += 1
    for event in template.events:
        event["run_id"] = template.run_id
        event["trace_id"] = template.trace_id
        if "parent" in event:
            event["parent"] = template.run_id
    return template


def generate_configured(
    agent_count: int,
    actions_per_agent: list[int],
    total_lines: int | None = None,
    rng: random.Random | None = None,
    variance: str = "mid",
) -> list[Run]:
    """Create runs for the selected agents with the requested action counts.

    Without ``total_lines`` one run per agent is created. With it, runs are
    repeated round-robin across the agents until exactly ``total_lines``
    events exist; the final run is truncated if needed. Field values
    (severities, categories, costs, latencies, etc.) are randomized on every
    call, so repeated generation with the same parameters produces different
    data. ``variance`` ("low"/"mid"/"high") scales how spread out the
    randomized cost/latency/token values are, to stress-test the scoring
    formula's sensitivity.
    """
    if not 1 <= agent_count <= len(AGENT_GENERATORS):
        raise ValueError(f"agent_count must be between 1 and {len(AGENT_GENERATORS)}")
    if len(actions_per_agent) != agent_count:
        raise ValueError("provide exactly one action count per agent")
    if any(not 4 <= count <= 20 for count in actions_per_agent):
        raise ValueError("each action count must be between 4 and 20")
    if total_lines is not None and (total_lines < 20 or total_lines % 20 != 0):
        raise ValueError("total_lines must be a positive multiple of 20")
    if variance not in VARIANCES:
        raise ValueError(f"variance must be one of {VARIANCES}")

    rng = rng or random.Random()
    agents = list(zip(AGENT_GENERATORS, actions_per_agent))
    if total_lines is None:
        return [
            _build_run(generator, action_count, f"custom-{index:02d}", rng, variance)
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
            run = _build_run(generator, action_count, f"custom-{index:02d}-{cycle:04d}", rng, variance)
            run.events = run.events[:remaining]
            remaining -= len(run.events)
            runs.append(run)
    return runs


def build_dataset_manifest(runs: list[Run], seed: int | None = None) -> dict:
    """Dataset-level metadata for a generated batch of runs: the
    ground-truth manifest plus identity fields (``dataset_id``,
    ``generated_at``, ``seed``) that let a specific generated dataset be
    named, reproduced, and told apart from another run with the same
    parameters.

    Kept separate from the trace rows themselves so ``TraceEvent`` stays
    schema-compatible -- this travels alongside the dataset (e.g. in the
    ``/generate`` response and the downloaded JSON).
    """
    agent_node_pairs = {
        (event["agent"], event["node"]) for run in runs for event in run.events if event.get("agent")
    }
    return {
        "ground_truth": build_ground_truth_manifest(agent_node_pairs),
        "dataset_id": uuid.uuid4().hex,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
    }
