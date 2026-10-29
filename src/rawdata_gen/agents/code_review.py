"""Pull-request review workflow. An LLM planner picks each next step
(``plan_next_step``, a ``next_action`` decision) of a fixed checklist --
fetch the diff, lint, test -- so the run is a scripted flow paid for with
LLM calls. Planted for the profiler to find:

* the fixed ``fetch_diff -> run_linter -> run_tests`` chain, with planner
  calls between the steps (combine into one scripted step);
* ``plan_next_step`` and ``decide_verdict`` always choose the same thing
  for the same input (replaceable by a rule);
* ``write_review_comments`` varies for the same input (hidden state).
"""
import random

from ..ground_truth import apply_noise
from .common import Run, jittered_int, jittered_uniform

REPOS = ["web-app", "billing-service", "mobile-api"]
_COMMENT_STYLES = ["brief", "detailed", "inline"]


def generate(rng: random.Random | None = None, variance: str = "mid") -> list[Run]:
    rng = rng or random.Random()
    repo = rng.choice(REPOS)
    pr = rng.randint(1000, 99999)
    lint_ok = rng.random() < 0.7
    tests_ok = rng.random() < 0.8
    run_id = f"review-{rng.randint(0, 999999):06d}"
    run = Run("pull-request-review", "code-review-agent", run_id, "v1.0", user_input=f"Review PR #{pr} in {repo}")
    parent = run.run_id

    def plan(after: str, step: str) -> None:
        run.action(
            "plan_next_step", state={"after": after}, decision=("next_action", step), model="demo-llm-v1",
            tokens=(jittered_int(rng, 1200, 1600, variance, floor=1), jittered_int(rng, 20, 40, variance, floor=1)),
            latency_ms=jittered_int(rng, 400, 700, variance, floor=1),
            cost=round(jittered_uniform(rng, 0.004, 0.006, variance), 5), parent=parent,
        )

    plan("start", "fetch_diff")
    run.action(
        "fetch_diff", event_type="tool_call", side_effects=False, state={"repo": repo, "pr": pr}, tool="github.get_pull_request_diff",
        latency_ms=jittered_int(rng, 200, 400, variance, floor=1), result=f"diff:{pr}", parent=parent,
    )
    plan("fetch_diff", "run_linter")
    run.action(
        "run_linter", event_type="tool_call", side_effects=False, state={"repo": repo, "pr": pr}, tool="ci.run_linter",
        latency_ms=jittered_int(rng, 3000, 6000, variance, floor=1), result="lint:pass" if lint_ok else "lint:fail",
        parent=parent,
    )
    plan("run_linter", "run_tests")
    run.action(
        "run_tests", event_type="tool_call", side_effects=False, state={"repo": repo, "pr": pr}, tool="ci.run_tests",
        latency_ms=jittered_int(rng, 20000, 60000, variance, floor=1), result="tests:pass" if tests_ok else "tests:fail",
        parent=parent,
    )
    run.action(
        "decide_verdict", state={"lint_ok": lint_ok, "tests_ok": tests_ok},
        decision=("verdict", "approve" if lint_ok and tests_ok else "request_changes"), model="demo-llm-v1",
        tokens=(jittered_int(rng, 2500, 3500, variance, floor=1), jittered_int(rng, 15, 30, variance, floor=1)),
        latency_ms=jittered_int(rng, 600, 900, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.008, 0.011, variance), 5), parent=parent,
    )
    style = apply_noise(rng, "write_review_comments", "brief" if lint_ok and tests_ok else "detailed", _COMMENT_STYLES)
    run.action(
        "write_review_comments", state={"lint_ok": lint_ok, "tests_ok": tests_ok},
        decision=("comment_style", style), model="demo-llm-v1",
        tokens=(jittered_int(rng, 3000, 4500, variance, floor=1), jittered_int(rng, 200, 450, variance, floor=1)),
        latency_ms=jittered_int(rng, 2500, 5000, variance, floor=1),
        cost=round(jittered_uniform(rng, 0.015, 0.025, variance), 5), parent=parent,
    )
    return [run]
