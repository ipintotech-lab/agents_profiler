"""Regression/stress-test sweep: generate synthetic data across a grid of
configurations, run each through the full pipeline, and report pass/fail
plus timing -- a QA tool for the pipeline itself, not something an end
user tunes by hand.

Rather than a full cross-product of every dimension (which would explode
combinatorially -- agent count x variance x demand model x total_lines x
formula overrides), each dimension is swept independently against a fixed
baseline config. That keeps the sweep's run count linear in the number of
dimensions while still exercising every option of every dimension at
least once.
"""
from __future__ import annotations

import time
from random import Random

from ingest.to_bpop_problem import DEMAND_MODELS
from ingest.trace_schema import TraceEvent
from rawdata_gen.agents.common import VARIANCES
from rawdata_gen.generate_traces import (
    AGENT_GENERATORS,
    build_dataset_manifest,
    generate_configured,
)

from .pipeline import run_pipeline

# total_lines must be a positive multiple of 20 (rawdata_gen constraint);
# these are the nearest valid multiples of 20 to the user-requested
# 1000/2500/5000/10000 range.
TOTAL_LINES_STEPS = (1000, 2500, 5000, 10000)

_BASELINE = {
    "agent_count": len(AGENT_GENERATORS),
    "actions_per_agent": [8] * len(AGENT_GENERATORS),
    "variance": "mid",
    "total_lines": 2000,
    "demand_model": "p95",
    "unit_cost": None,
    "unit_value": None,
    "capacity_headroom": 2.0,
}


def _case(label: str, group: str, **overrides) -> dict:
    cfg = {**_BASELINE, **overrides}
    return {"label": label, "group": group, "config": cfg}


def _build_cases() -> list[dict]:
    cases = [_case("baseline", "baseline")]

    for agent_count in range(1, len(AGENT_GENERATORS) + 1):
        cases.append(
            _case(
                f"agent_count={agent_count}",
                "agent_count",
                agent_count=agent_count,
                actions_per_agent=[8] * agent_count,
            )
        )

    for variance in VARIANCES:
        cases.append(_case(f"variance={variance}", "variance", variance=variance))

    for demand_model in DEMAND_MODELS:
        cases.append(_case(f"demand_model={demand_model}", "demand_model", demand_model=demand_model))

    for total_lines in TOTAL_LINES_STEPS:
        cases.append(_case(f"total_lines={total_lines}", "total_lines", total_lines=total_lines))

    for headroom in (1.0, 2.0, 5.0):
        cases.append(_case(f"capacity_headroom={headroom}", "formula_override", capacity_headroom=headroom))
    cases.append(
        _case(
            "unit_cost/unit_value override",
            "formula_override",
            unit_cost=0.01,
            unit_value=0.05,
        )
    )

    return cases


def generate_dataset_for_config(config: dict, seed: int) -> tuple[list[TraceEvent], dict]:
    rng = Random(seed)
    runs = generate_configured(
        config["agent_count"], config["actions_per_agent"], config["total_lines"], rng=rng, variance=config["variance"]
    )
    raw_events = [event for run in runs for event in run.events]
    events = [TraceEvent.model_validate(row) for row in raw_events]
    manifest = build_dataset_manifest(runs, seed=seed)
    return events, manifest


def run_sweep(seed: int = 1, cases: list[dict] | None = None) -> dict:
    """Run every swept configuration through the full pipeline, each with
    its own freshly-generated dataset (same ``seed`` across configs, for
    reproducibility), and report pass/fail plus timing and a few key
    output stats per case. ``cases`` overrides the default grid (see
    ``_build_cases``), e.g. to run a representative subset.
    """
    results = []
    for case in cases if cases is not None else _build_cases():
        config = case["config"]
        started = time.monotonic()
        row = {
            "label": case["label"],
            "group": case["group"],
            "config": dict(config),
        }
        try:
            events, manifest = generate_dataset_for_config(config, seed)
            report = run_pipeline(
                events,
                unit_cost=config["unit_cost"],
                unit_value=config["unit_value"],
                capacity_headroom=config["capacity_headroom"],
                ground_truth=manifest["ground_truth"],
                demand_model=config["demand_model"],
            )
            elapsed_ms = (time.monotonic() - started) * 1000
            validation = report.get("validation")
            expected = (report.get("expected_vs_found") or {}).get("summary")
            region_rows = [
                r for r in (report.get("expected_vs_found") or {}).get("rows", [])
                if r["type"] == "DETERMINISTIC_REGION" and r["expected"]
            ]
            row.update(
                {
                    "passed": True,
                    "error": None,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "num_events": report["observe"]["num_events"],
                    "selected_workflow": report["observe"]["selected_workflow"],
                    "x": report["profile"]["x"],
                    "y": report["profile"]["y"],
                    "objective": report["profile"]["objective"],
                    "false_negatives": validation["false_negatives"] if validation else None,
                    "false_positives": validation["false_positives"] if validation else None,
                    "accuracy": validation["accuracy"] if validation else None,
                    "insufficient_rate": validation["insufficient_evidence_rate"] if validation else None,
                    "planted_found": expected["planted_found"] if expected else None,
                    "planted_total": expected["planted_total"] if expected else None,
                    "false_alarms": expected["false_alarms"] if expected else None,
                    "regions": report["behavior"]["summary"]["regions"],
                    # the planted "deterministic only for some inputs" region
                    "region_found": all(r["found"] for r in region_rows) if region_rows else None,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- a sweep must survive and report every case's own failure
            elapsed_ms = (time.monotonic() - started) * 1000
            row.update(
                {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": round(elapsed_ms, 1),
                    "num_events": None,
                    "selected_workflow": None,
                    "x": None,
                    "y": None,
                    "objective": None,
                    "false_negatives": None,
                    "false_positives": None,
                    "accuracy": None,
                    "insufficient_rate": None,
                    "planted_found": None,
                    "planted_total": None,
                    "false_alarms": None,
                    "regions": None,
                    "region_found": None,
                }
            )
        results.append(row)

    passed = sum(1 for r in results if r["passed"])
    return {
        "seed": seed,
        "num_cases": len(results),
        "num_passed": passed,
        "num_failed": len(results) - passed,
        "all_passed": passed == len(results),
        "results": results,
    }
