"""Ground-truth determinism profile for the synthetic generator.

Each entry is the probability that a canonical input maps to its
"canonical" (expected) decision. A rate below 1.0 means the generator
deliberately injects noise on repeated identical inputs via
``apply_noise`` -- the mechanism that lets the same input produce both
the same output (most of the time) and a different output (sometimes),
so the profiler's determinism classifier has real non-deterministic
cases to detect, not just newly-randomized inputs on every call.

Nodes not listed here default to ``DEFAULT_DETERMINISM_RATE`` (1.0):
they are pure functions of their input and the generator never calls
``apply_noise`` for them, so a rate of 1.0 is actually true, not just
assumed.
"""
from __future__ import annotations

import random

DETERMINISM_PROFILE: dict[str, float] = {
    "analyze_remediation": 0.80,
    "classify_ticket": 0.80,
    "draft_response": 0.20,
    "assess_incident": 0.80,
}

DEFAULT_DETERMINISM_RATE = 1.00


def determinism_rate_for(node: str) -> float:
    return DETERMINISM_PROFILE.get(node, DEFAULT_DETERMINISM_RATE)


def apply_noise(rng: random.Random, node: str, canonical: str, alternatives: list[str]) -> str:
    """Return ``canonical`` with probability ``determinism_rate_for(node)``;
    otherwise a different value drawn uniformly from ``alternatives``.

    With ``rate >= 1.0`` (the default for any node not in
    ``DETERMINISM_PROFILE``) this always returns ``canonical`` -- no noise
    is ever injected for nodes the profile doesn't name.
    """
    rate = determinism_rate_for(node)
    pool = [a for a in alternatives if a != canonical]
    if rate >= 1.0 or not pool:
        return canonical
    return canonical if rng.random() < rate else rng.choice(pool)


def build_ground_truth_manifest(agent_node_pairs: set[tuple[str, str]]) -> dict[str, dict]:
    """Dataset-level ground-truth manifest: ``"agent/node"`` -> rate.

    Kept separate from ``TraceEvent`` on purpose -- trace rows stay
    schema-compatible, and the manifest travels alongside the dataset
    (e.g. in the ``/generate`` response and the downloaded JSON) instead
    of being embedded per-event.
    """
    return {
        f"{agent}/{node}": {"determinism_rate": determinism_rate_for(node)}
        for agent, node in sorted(agent_node_pairs)
    }
