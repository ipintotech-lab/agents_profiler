"""Generate sample trace data from the command line.

    python -m rawdata_gen                       # asks, like the original run.sh
    python -m rawdata_gen --agents 2 --actions 4 6 --lines 40 -o data/agent_traces.jsonl
    python -m rawdata_gen --format claude-code --sessions 5 --prompts 3 -o data/sessions.jsonl
    python -m rawdata_gen --format claude-code --omit cost tokens   # an incomplete export

``events`` writes the mock agents' trace events; ``claude-code`` writes
Claude Code session transcripts (see ``claude_code_sessions.py``). Both are
files the dashboard and ``/analyze`` accept as uploads.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .agents.common import VARIANCES, write_jsonl
from .claude_code_sessions import OMITTABLE, generate_sessions
from .generate_traces import AGENT_GENERATORS, generate_configured


def ask_int(prompt: str, minimum: int, maximum: int | None = None, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if value >= minimum and (maximum is None or value <= maximum):
            return value
        print(f"Enter a number of at least {minimum}." if maximum is None else f"Enter a number from {minimum} to {maximum}.")


def _interactive(args: argparse.Namespace) -> None:
    """Fill in the arguments by asking, as the original repo's run.sh did."""
    kind = ask_int("Generate (1) agent trace events or (2) Claude Code sessions? [1]: ", 1, 2, default=1)
    if kind == 2:
        args.format = "claude-code"
        args.sessions = ask_int("Number of sessions (1-50) [3]: ", 1, 50, default=3)
        args.prompts = ask_int("User requests per session (1-10) [3]: ", 1, 10, default=3)
        return
    args.agents = ask_int(f"Number of agents to include (1-{len(AGENT_GENERATORS)}): ", 1, len(AGENT_GENERATORS))
    args.actions = [ask_int(f"Number of actions for agent {i} (4-20): ", 4, 20) for i in range(1, args.agents + 1)]
    lines = ask_int("Number of lines in raw data (multiple of 20, 0 = one run per agent) [0]: ", 0, default=0)
    args.lines = lines or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rawdata_gen", description=__doc__.split("\n\n")[0])
    parser.add_argument("--format", choices=["events", "claude-code"], default="events")
    parser.add_argument("--agents", type=int, help=f"events: number of mock agents, 1-{len(AGENT_GENERATORS)}")
    parser.add_argument("--actions", type=int, nargs="+", help="events: actions per agent run, one per agent (4-20)")
    parser.add_argument("--lines", type=int, help="events: total events, a multiple of 20 (default: one run per agent)")
    parser.add_argument("--variance", choices=VARIANCES, default="mid", help="events: spread of cost/latency/tokens")
    parser.add_argument("--sessions", type=int, default=3, help="claude-code: number of sessions (1-50)")
    parser.add_argument("--prompts", type=int, default=3, help="claude-code: user requests per session (1-10)")
    parser.add_argument("--omit", nargs="+", choices=OMITTABLE, default=[], help="claude-code: attributes to leave out")
    parser.add_argument("--seed", type=int, help="reproduce the same data")
    parser.add_argument("-o", "--output", type=Path, help="output file (default data/agent_traces.jsonl or data/sessions.jsonl)")
    args = parser.parse_args(argv)
    if argv is None and len(sys.argv) == 1:
        _interactive(args)

    rng = random.Random(args.seed)
    try:
        if args.format == "claude-code":
            output = args.output or Path("data/sessions.jsonl")
            sessions = generate_sessions(args.sessions, args.prompts, rng, tuple(args.omit))
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                for row in (r for s in sessions for r in s):
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
            omitted = f", without {', '.join(args.omit)}" if args.omit else ""
            print(f"Generated {len(sessions)} Claude Code sessions ({args.prompts} requests each{omitted}): {output}")
            return 0
        agents = args.agents or len(AGENT_GENERATORS)
        actions = args.actions or [5] * agents
        runs = generate_configured(agents, actions, args.lines, rng=rng, variance=args.variance)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or Path("data/agent_traces.jsonl")
    total = write_jsonl(runs, output)
    print(f"Generated {total} lines from {len(runs)} runs across {agents} agents: {output}")
    for run in runs[:10]:
        print(f"  {run.run_id} {run.agent} ({run.workflow}): {len(run.events)} actions")
    if len(runs) > 10:
        print(f"  ... and {len(runs) - 10} more runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
