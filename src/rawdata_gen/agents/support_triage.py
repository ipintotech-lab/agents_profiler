"""Support workflow: classify, retrieve context, route and respond, with
randomized case data so repeated generation produces different (but
structurally consistent) traces."""
import random

from .common import Run

CATEGORIES = ["billing", "access", "bug", "feature_request"]
URGENCIES = ["low", "medium", "high"]


def generate(rng: random.Random | None = None) -> list[Run]:
    rng = rng or random.Random()
    ticket = f"S-{rng.randint(100, 999)}"
    category = rng.choice(CATEGORIES)
    urgency = rng.choice(URGENCIES)
    run_id = f"support-{rng.randint(0, 999999):06d}"
    run = Run("support-ticket-triage", "support-agent", run_id, "v1.0")
    parent = run.run_id
    state = {"category": category, "urgency": urgency, "ticket_id": ticket}
    run.action(
        "classify_ticket", state={"category_hint": category, "urgency_hint": urgency},
        decision=("classification", category), model="demo-llm-v1",
        tokens=(rng.randint(450, 650), rng.randint(35, 70)), latency_ms=rng.randint(300, 500),
        cost=round(rng.uniform(0.0025, 0.0045), 5), parent=parent,
    )
    run.action(
        "fetch_customer_context", event_type="tool_call", state={"ticket_id": ticket},
        tool="crm.get_customer_context", latency_ms=rng.randint(120, 220), parent=parent,
    )
    run.action(
        "select_queue", state=state,
        decision=("queue_selection", "priority-support" if urgency == "high" else f"{category}-support"),
        parent=parent,
    )
    run.action(
        "draft_response", state={"category": category, "context_available": True},
        decision=("response_type", "instructions" if category == "access" else "account_help"),
        model="demo-llm-v1", tokens=(rng.randint(800, 1100), rng.randint(90, 160)),
        latency_ms=rng.randint(600, 950), cost=round(rng.uniform(0.005, 0.009), 5), parent=parent,
    )
    run.action(
        "update_ticket", event_type="tool_call", state={"ticket_id": ticket, "urgency": urgency},
        tool="helpdesk.update_ticket", latency_ms=rng.randint(150, 250), parent=parent,
    )
    run.action(
        "send_acknowledgement", event_type="tool_call", state={"ticket_id": ticket},
        tool="messaging.send_reply", latency_ms=rng.randint(110, 190), parent=parent,
    )
    return [run]
