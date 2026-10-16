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
    "write_review_comments": 0.30,
    "diagnose_failure": 0.80,
    # a tool reading a changing world: the job's status moves between calls
    "check_job_status": 0.50,
    "choose_sequence": 0.90,
    "write_outreach": 0.40,
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


# What each workflow's generator plants for the report's opportunity list
# to find (``expect: True``) -- and what it should NOT report
# (``expect: False``) -- named by the opportunity ``type`` and the ``node``
# (or, for a sequence, the ``steps`` it must contain in order). The POC
# compares this with what the profiler actually reports; the two won't
# match exactly, and the differences are the point.
EXPECTED_FINDINGS: dict[str, list[dict]] = {
    "cve-remediation-agent": [
        {"type": "CONDITIONAL_LLM_BYPASS", "node": "classify_severity", "expect": True,
         "why": "the severity label is a pure function of the input"},
        {"type": "REVIEW_HIDDEN_STATE", "node": "analyze_remediation", "expect": True,
         "why": "20% of identical inputs get the other remediation"},
        {"type": "DETERMINISTIC_REGION", "node": "analyze_remediation", "field": "severity", "expect": True,
         "why": "for high/critical CVEs the remediation always follows fix_available; low/medium flip 40%"},
        {"type": "DETERMINISTIC_REGION", "node": "classify_severity", "expect": False,
         "why": "a pure function everywhere -- no slice behaves differently from the rest"},
        {"type": "CACHE_RESULT", "node": "create_or_update_ticket", "expect": False,
         "why": "it writes a ticket -- a side effect, not a lookup to reuse"},
    ],
    "support-ticket-triage": [
        {"type": "REVIEW_HIDDEN_STATE", "node": "classify_ticket", "expect": True,
         "why": "20% of identical inputs get a different category"},
        {"type": "REVIEW_HIDDEN_STATE", "node": "draft_response", "expect": True,
         "why": "80% of identical inputs get a different response type"},
        {"type": "CACHE_RESULT", "node": "send_acknowledgement", "expect": False,
         "why": "it sends a message -- a side effect, not a lookup to reuse"},
    ],
    "deployment-incident-response": [
        {"type": "REVIEW_HIDDEN_STATE", "node": "assess_incident", "expect": True,
         "why": "20% of identical inputs get the other severity"},
        {"type": "CONDITIONAL_LLM_BYPASS", "node": "assess_incident", "expect": False,
         "why": "its choice varies for the same input, so no rule can replace it"},
        {"type": "DETERMINISTIC_REGION", "node": "assess_incident", "expect": False,
         "why": "the same 20% noise for every input -- no slice is safe to automate"},
    ],
    "pull-request-review": [
        {"type": "EXTRACT_DETERMINISTIC_SUBSTEP", "steps": ["fetch_diff", "run_linter", "run_tests"], "expect": True,
         "why": "a fixed checklist, with an LLM planner call before each step"},
        {"type": "CONDITIONAL_LLM_BYPASS", "node": "plan_next_step", "expect": True,
         "why": "the planner always picks the same next step after the same step"},
        {"type": "CONDITIONAL_LLM_BYPASS", "node": "decide_verdict", "expect": True,
         "why": "approve exactly when lint and tests pass"},
        {"type": "REVIEW_HIDDEN_STATE", "node": "write_review_comments", "expect": True,
         "why": "70% of identical inputs get a different comment style"},
    ],
    "etl-incident-response": [
        {"type": "CACHE_RESULT", "node": "get_table_schema", "expect": True,
         "why": "called twice per run for a few tables, same schema every time"},
        {"type": "CONDITIONAL_LLM_BYPASS", "node": "decide_retry", "expect": True,
         "why": "retry exactly when the error is a timeout"},
        {"type": "REVIEW_HIDDEN_STATE", "node": "diagnose_failure", "expect": True,
         "why": "20% of identical inputs get a different root cause"},
        {"type": "CACHE_RESULT", "node": "check_job_status", "expect": False,
         "why": "the job's status changes between calls"},
    ],
    "lead-qualification": [
        {"type": "CACHE_RESULT", "node": "enrich_company", "expect": True,
         "why": "10 companies, each always enriches to the same profile"},
        {"type": "CACHE_RESULT", "node": "assign_rep", "expect": True,
         "why": "the same region and tier always get the same rep"},
        {"type": "CONDITIONAL_LLM_BYPASS", "node": "score_lead", "expect": True,
         "why": "the tier is a pure function of company size"},
        {"type": "REVIEW_HIDDEN_STATE", "node": "write_outreach", "expect": True,
         "why": "60% of identical inputs get a different tone"},
        {"type": "REVIEW_HIDDEN_STATE", "node": "choose_sequence", "expect": True,
         "why": "only 10% of identical inputs vary -- a subtle case that needs more data"},
        {"type": "CACHE_RESULT", "node": "check_crm_duplicate", "expect": False,
         "why": "every lead is new, so no input repeats"},
    ],
}
