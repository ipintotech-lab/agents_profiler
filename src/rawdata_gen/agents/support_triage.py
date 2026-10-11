"""Support workflow: classify, retrieve context, route and respond."""
from .common import Run


def generate() -> list[Run]:
    runs = []
    cases = [("S-801", "billing", "medium"), ("S-802", "access", "high"), ("S-803", "billing", "low")]
    for i, (ticket, category, urgency) in enumerate(cases, 1):
        run = Run("support-ticket-triage", "support-agent", f"support-{i:02d}", "v1.0")
        parent = run.run_id
        state = {"category": category, "urgency": urgency, "ticket_id": ticket}
        run.action("classify_ticket", state={"category_hint": category, "urgency_hint": urgency}, decision=("classification", category), model="demo-llm-v1", tokens=(520+i*15, 42+i*3), latency_ms=340+i*25, cost=.0028+i*.0002, parent=parent)
        run.action("fetch_customer_context", event_type="tool_call", state={"ticket_id": ticket}, tool="crm.get_customer_context", latency_ms=145+i*12, parent=parent)
        run.action("select_queue", state=state, decision=("queue_selection", "priority-support" if urgency == "high" else f"{category}-support"), parent=parent)
        run.action("draft_response", state={"category": category, "context_available": True}, decision=("response_type", "instructions" if category == "access" else "account_help"), model="demo-llm-v1", tokens=(900+i*50, 110+i*10), latency_ms=680+i*55, cost=.006+i*.0005, parent=parent)
        run.action("update_ticket", event_type="tool_call", state={"ticket_id": ticket, "urgency": urgency}, tool="helpdesk.update_ticket", latency_ms=175+i*8, parent=parent)
        run.action("send_acknowledgement", event_type="tool_call", state={"ticket_id": ticket}, tool="messaging.send_reply", latency_ms=130+i*9, parent=parent)
        runs.append(run)
    return runs
