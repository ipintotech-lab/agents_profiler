"""Convert trace files locally into normalized TraceEvent JSONL, for
uploading to ``/analyze`` without sending the raw source data.

Accepts any format ``/analyze`` does (TraceEvent JSON/JSONL, wrapped
JSON, OpenTelemetry OTLP JSON, Claude Code session transcripts). The
output keeps only TraceEvent fields -- for a Claude Code session that
already drops prompts, replies and tool outputs -- and ``--hash-inputs``
also replaces every ``input_state`` value with a hash, so tool arguments
(commands, file paths, edit text) never leave the machine either.

    python -m ingest.convert ~/.claude/projects/<project>/*.jsonl --hash-inputs -o traces.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .normalize import hash_input_values, normalize_raw_json, parse_raw_text
from .trace_schema import TraceEvent


def convert_files(paths: list[str | Path], hash_inputs: bool = False) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for path in paths:
        try:
            rows = normalize_raw_json(parse_raw_text(Path(path).read_text(encoding="utf-8")))
        except ValueError as exc:
            raise ValueError(f"could not parse {path}: {exc}") from exc
        if hash_inputs:
            rows = hash_input_values(rows)
        events.extend(TraceEvent.model_validate(row) for row in rows)
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest.convert", description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="trace files to convert")
    parser.add_argument("--hash-inputs", action="store_true", help="replace every input_state value with a hash")
    parser.add_argument("-o", "--output", help="output JSONL path (default: stdout)")
    args = parser.parse_args(argv)

    try:
        events = convert_files(args.files, hash_inputs=args.hash_inputs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    lines = "".join(event.model_dump_json(exclude_none=True) + "\n" for event in events)
    if args.output:
        Path(args.output).write_text(lines, encoding="utf-8")
        print(f"wrote {len(events)} events to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
