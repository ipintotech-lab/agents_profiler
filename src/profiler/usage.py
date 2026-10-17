"""Report for aggregate usage exports (``ingest/usage.py``): token totals,
cache-read share, workload concentration, outlier records, and repeated
agent activations.

These exports have no steps, order, outcome or cost, so nothing here is a
saving: tokens are counters, not money (on a subscription they are quota),
and concentration only says where to look. Activations and usage snapshots
are reported separately -- nothing joins them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from ingest.usage import TOKEN_FIELDS

OUTLIER_FACTOR = 10.0
TOP_WORKLOADS = 10
TOP_PAIRS = 10


def _share(part: float, whole: float) -> float | None:
    return round(part / whole, 4) if whole else None


def usage_report(records: list[dict]) -> dict:
    totals = {k: sum(r[k] for r in records) for k in TOKEN_FIELDS}
    all_tokens = sum(totals.values())
    by_workload: dict[str, dict] = defaultdict(lambda: {"records": 0, "tokens": 0, "cache_read": 0})
    for r in records:
        w = by_workload[r["workload_id"]]
        w["records"] += 1
        w["tokens"] += sum(r[k] for k in TOKEN_FIELDS)
        w["cache_read"] += r["cache_read_input_tokens"]
    workloads = sorted(
        (
            {
                "workload_id": wid,
                **w,
                "share": _share(w["tokens"], all_tokens),
                "cache_read_share": _share(w["cache_read"], w["tokens"]),
            }
            for wid, w in by_workload.items()
        ),
        key=lambda w: -w["tokens"],
    )
    shares = [w["share"] or 0 for w in workloads]
    per_record = [sum(r[k] for k in TOKEN_FIELDS) for r in records]
    typical = median(per_record) if per_record else 0
    outliers = sorted(
        (
            {"record_id": r["record_id"], "workload_id": r["workload_id"], "tokens": t,
             "times_median": round(t / typical, 1) if typical else None}
            for r, t in zip(records, per_record, strict=True)
            if typical and t >= OUTLIER_FACTOR * typical
        ),
        key=lambda o: -o["tokens"],
    )
    signatures = Counter((r["workload_id"], r["input_tokens"], r["output_tokens"]) for r in records)
    return {
        "records": len(records),
        "workloads": len(workloads),
        "totals": {**totals, "all": all_tokens, "non_cache_read": all_tokens - totals["cache_read_input_tokens"]},
        "cache_read_share": _share(totals["cache_read_input_tokens"], all_tokens),
        "concentration": {
            "top1_share": round(sum(shares[:1]), 4),
            "top3_share": round(sum(shares[:3]), 4),
            # Herfindahl index: 1 = one workload has everything, 1/n = even
            "hhi": round(sum(s * s for s in shares), 4),
        },
        "median_record_tokens": typical,
        "outlier_factor": OUTLIER_FACTOR,
        "outliers": outliers,
        "by_workload": workloads[:TOP_WORKLOADS],
        "same_counts_records": sum(c for c in signatures.values() if c > 1),
    }


def activations_report(records: list[dict]) -> dict:
    pairs = Counter((r["agent_id"], r["trigger_id"]) for r in records)
    agents_per_trigger: dict[str, set] = defaultdict(set)
    for r in records:
        agents_per_trigger[r["trigger_id"]].add(r["agent_id"])
    repeated = [(p, c) for p, c in pairs.most_common() if c > 1]
    return {
        "records": len(records),
        "agents": len({r["agent_id"] for r in records}),
        "triggers": len(agents_per_trigger),
        "repeated_pairs": len(repeated),
        "activations_in_repeated_pairs": sum(c for _, c in repeated),
        "top_pairs": [{"agent_id": a, "trigger_id": t, "count": c} for (a, t), c in repeated[:TOP_PAIRS]],
        "fan_out": sorted(
            ({"trigger_id": t, "agents": len(s)} for t, s in agents_per_trigger.items()),
            key=lambda f: (-f["agents"], f["trigger_id"]),
        )[:TOP_PAIRS],
    }


LIMITS = [
    "Aggregate records only: no steps, order, timing, outcome or cost, so the profiler can't find repeated steps, check determinism or price a saving.",
    "Token counts are counters, not money. On a subscription, fewer tokens save quota or capacity, not dollars.",
    "A usage record is a snapshot that may cover several model calls; an outlier is a place to look, not proof of waste.",
    "Activations and usage are separate: a repeated agent/trigger pair can't be tied to its token cost.",
    "For step-level findings, export per-step events with run_id, node, sequence (or timestamp), tokens and a hashed input.",
]


def build_aggregate(usage: list[dict], activations: list[dict], files: list[dict]) -> dict:
    return {
        "kind": "aggregate",
        "files": files,
        "usage": usage_report(usage) if usage else None,
        "activations": activations_report(activations) if activations else None,
        "limits": LIMITS,
    }
