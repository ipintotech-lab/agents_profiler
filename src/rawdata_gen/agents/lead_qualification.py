"""Sales lead-qualification workflow. Planted for the profiler to find:

* ``enrich_company`` and ``assign_rep`` repeat with the same inputs and
  always return the same output (cache the result);
* ``score_lead`` always gives the same tier for the same company profile
  (replaceable by a rule);
* ``write_outreach`` varies a lot for the same input (hidden state), and
  ``choose_sequence`` varies only rarely (a subtle case: needs more data
  before it shows);
* ``check_crm_duplicate`` never repeats an input -- nothing to find.
"""
import random

from ..ground_truth import apply_noise
from .common import Run, jittered_int, jittered_uniform

DOMAINS = [f"{name}.com" for name in (
    "acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka", "tyrell", "cyberdyne",
)]
INDUSTRIES = ["fintech", "retail", "health"]
SIZES = ["smb", "mid_market", "enterprise"]
REGIONS = ["na", "emea", "apac"]
_SEQUENCES = ["nurture", "fast_track", "exec_outreach"]
_TONES = ["formal", "friendly", "technical"]


def generate(rng: random.Random | None = None, variance: str = "mid") -> list[Run]:
    rng = rng or random.Random()
    domain = rng.choice(DOMAINS)
    # a company's profile is fixed: the same domain always enriches the same way
    profile = random.Random(domain)
    industry, size, region = profile.choice(INDUSTRIES), profile.choice(SIZES), profile.choice(REGIONS)
    lead = f"lead-{rng.randint(0, 999999):06d}"
    run_id = f"leads-{rng.randint(0, 999999):06d}"
    run = Run("lead-qualification", "sales-agent", run_id, "v1.0", user_input=f"Qualify inbound lead {lead} from {domain}")
    parent = run.run_id
    tier = {"smb": "C", "mid_market": "B", "enterprise": "A"}[size]

    run.action(
        "enrich_company", event_type="tool_call", state={"domain": domain}, tool="clearbit.enrich",
        latency_ms=jittered_int(rng, 250, 600, variance, floor=1), result=f"profile:{industry}:{size}:{region}:billing=annual,plan=team,seats=25,health=green,tags=saas,b2b,active",
        parent=parent,
    )
    run.action(
        "check_crm_duplicate", event_type="tool_call", state={"lead_id": lead}, tool="crm.find_duplicates",
        latency_ms=jittered_int(rng, 100, 250, variance, floor=1), result="no_duplicate", parent=parent,
    )
    run.action(
        "score_lead", state={"industry": industry, "size": size}, decision=("lead_tier", tier), model="demo-llm-v1",
        tokens=(jittered_int(rng, 1500, 2200, variance, floor=1), jittered_int(rng, 10, 20, variance, floor=1)),
        latency_ms=jittered_int(rng, 500, 800, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.005, 0.008, variance), 5), parent=parent,
    )
    canonical_sequence = {"A": "exec_outreach", "B": "fast_track", "C": "nurture"}[tier]
    run.action(
        "choose_sequence", state={"tier": tier, "industry": industry},
        decision=("sequence", apply_noise(rng, "choose_sequence", canonical_sequence, _SEQUENCES)),
        model="demo-llm-v1",
        tokens=(jittered_int(rng, 1000, 1400, variance, floor=1), jittered_int(rng, 10, 20, variance, floor=1)),
        latency_ms=jittered_int(rng, 400, 650, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.003, 0.005, variance), 5), parent=parent,
    )
    run.action(
        "assign_rep", event_type="tool_call", state={"region": region, "tier": tier}, tool="crm.round_robin_owner",
        latency_ms=jittered_int(rng, 80, 150, variance, floor=1), result=f"rep:{region}-{tier}:owner={region}.{tier.lower()}@example.com,quota=120,team=inbound,territory={region},level=L2", parent=parent,
    )
    run.action(
        "write_outreach", state={"tier": tier, "industry": industry},
        decision=("tone", apply_noise(rng, "write_outreach", "friendly", _TONES)), model="demo-llm-v1",
        tokens=(jittered_int(rng, 2000, 3000, variance, floor=1), jittered_int(rng, 300, 600, variance, floor=1)),
        latency_ms=jittered_int(rng, 3000, 6000, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.018, 0.03, variance), 5), parent=parent,
    )
    run.action(
        "log_activity", event_type="tool_call", state={"lead_id": lead}, tool="crm.log_activity",
        latency_ms=jittered_int(rng, 60, 120, variance, floor=1), parent=parent,
    )
    run.action(
        "schedule_followup", event_type="tool_call", state={"lead_id": lead}, tool="calendar.schedule_task",
        latency_ms=jittered_int(rng, 90, 180, variance, floor=1), parent=parent,
    )
    return [run]
