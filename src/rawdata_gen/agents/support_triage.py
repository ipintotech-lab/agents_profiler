"""Support workflow: classify, retrieve context, route and respond, with
randomized case data so repeated generation produces different (but
structurally consistent) traces."""
import random

from ..ground_truth import apply_noise
from .common import Run, jittered_int, jittered_uniform

CATEGORIES = ["billing", "access", "bug", "feature_request"]
URGENCIES = ["low", "medium", "high"]
_RESPONSE_TYPES = ["instructions", "account_help"]


def generate(rng: random.Random | None = None, variance: str = "mid") -> list[Run]:
    rng = rng or random.Random()
    ticket = f"S-{rng.randint(100, 999)}"
    category = rng.choice(CATEGORIES)
    urgency = rng.choice(URGENCIES)
    run_id = f"support-{rng.randint(0, 999999):06d}"
    run = Run("support-ticket-triage", "support-agent", run_id, "v1.0", user_input=f"Triage support ticket {ticket}")
    parent = run.run_id
    classification = apply_noise(rng, "classify_ticket", category, CATEGORIES)
    run.action(
        "classify_ticket", state={"category_hint": category, "urgency_hint": urgency},
        decision=("classification", classification), model="demo-llm-v1",
        tokens=(jittered_int(rng, 450, 650, variance, floor=1), jittered_int(rng, 35, 70, variance, floor=1)),
        latency_ms=jittered_int(rng, 300, 500, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.0025, 0.0045, variance), 5), parent=parent,
    )
    run.action(
        "fetch_customer_context", event_type="tool_call", state={"ticket_id": ticket},
        tool="crm.get_customer_context", latency_ms=jittered_int(rng, 120, 220, variance, floor=1), parent=parent,
    )
    run.action(
        # Canonicalized on category/urgency only -- ticket_id is unique per
        # run and would otherwise make every input_state unique, so no
        # repeated-input evidence could ever accumulate for this node.
        "select_queue", state={"category": category, "urgency": urgency},
        decision=("queue_selection", "priority-support" if urgency == "high" else f"{category}-support"),
        parent=parent,
    )
    response_canonical = "instructions" if category == "access" else "account_help"
    response_type = apply_noise(rng, "draft_response", response_canonical, _RESPONSE_TYPES)
    run.action(
        "draft_response", state={"category": category, "context_available": True},
        decision=("response_type", response_type),
        model="demo-llm-v1",
        tokens=(jittered_int(rng, 800, 1100, variance, floor=1), jittered_int(rng, 90, 160, variance, floor=1)),
        latency_ms=jittered_int(rng, 600, 950, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.005, 0.009, variance), 5), parent=parent,
    )
    run.action(
        "update_ticket", event_type="tool_call", state={"ticket_id": ticket, "urgency": urgency},
        tool="helpdesk.update_ticket", latency_ms=jittered_int(rng, 150, 250, variance, floor=1), parent=parent,
    )
    run.action(
        "send_acknowledgement", event_type="tool_call", state={"ticket_id": ticket},
        tool="messaging.send_reply", latency_ms=jittered_int(rng, 110, 190, variance, floor=1), parent=parent,
    )
    return [run]
