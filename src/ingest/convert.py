"""Convert trace files locally into normalized TraceEvent JSONL, for
uploading to ``/analyze`` without sending the raw source data.

Accepts any format ``/analyze`` does (TraceEvent JSON/JSONL, wrapped
JSON, OpenTelemetry OTLP JSON, Claude Code session transcripts, chat
trajectories, FreeInference sessions), and Hugging Face dataset links in
place of files (see ``huggingface.py``). The
output keeps only TraceEvent fields -- for a Claude Code session that
already drops prompts, replies and tool outputs -- and ``--hash-inputs``
also replaces every ``input_state`` value with a hash, so tool arguments
(commands, file paths, edit text) never leave the machine either.

    python -m ingest.convert ~/.claude/projects/<project>/*.jsonl --hash-inputs -o traces.jsonl
    python -m ingest.convert https://huggingface.co/datasets/cx-cmu/agent_trajectories \\
        --hf-file swebench.jsonl --limit 10 -o traces.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .huggingface import fetch_records, is_hf_url
from .normalize import hash_input_values, normalize_raw_json, parse_raw_text
from .trace_schema import TraceEvent


def convert_files(
    paths: list[str | Path],
    hash_inputs: bool = False,
    hf_file: str | None = None,
    hf_limit: int | None = 10,
    hf_token: str | None = None,
) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for path in paths:
        try:
            if is_hf_url(str(path)):
                records, _ = fetch_records(str(path), hf_file, hf_limit, hf_token)
                rows = normalize_raw_json(records)
            else:
                rows = normalize_raw_json(parse_raw_text(Path(path).read_text(encoding="utf-8")))
        except ValueError as exc:
            raise ValueError(f"could not parse {path}: {exc}") from exc
        if hash_inputs:
            rows = hash_input_values(rows)
        events.extend(TraceEvent.model_validate(row) for row in rows)
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest.convert", description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="trace files or Hugging Face dataset links to convert")
    parser.add_argument("--hash-inputs", action="store_true", help="replace every input_state value with a hash")
    parser.add_argument("-o", "--output", help="output JSONL path (default: stdout)")
    parser.add_argument("--hf-file", help="file to read from a Hugging Face dataset (e.g. swebench.jsonl)")
    parser.add_argument("--limit", type=int, default=10, help="records to read from a Hugging Face dataset (default 10)")
    parser.add_argument("--token", help="Hugging Face token (default: HF_TOKEN or `hf auth login`)")
    args = parser.parse_args(argv)

    try:
        events = convert_files(
            args.files, hash_inputs=args.hash_inputs, hf_file=args.hf_file, hf_limit=args.limit, hf_token=args.token
        )
    except (ValueError, RuntimeError, OSError) as exc:
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
