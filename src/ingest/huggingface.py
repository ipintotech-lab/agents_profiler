"""Fetch agent-trace records straight from a Hugging Face dataset.

Accepts a dataset link in any of the forms people paste --
``https://huggingface.co/datasets/<org>/<name>`` (tracking query strings
like ``?utm_source=...`` are ignored), the same with ``/blob/<rev>/<path>``,
``/resolve/<rev>/<path>`` or ``/tree/<rev>/<dir>``, ``hf://datasets/<org>/<name>/<path>``,
or a bare ``<org>/<name>`` -- and streams the first ``limit`` records of one
JSON / JSON Lines file (optionally ``.zst`` or ``.gz`` compressed), so a
multi-GB file never has to be downloaded whole.

Uses the Hub's plain HTTP API, so ``huggingface_hub`` / ``datasets`` aren't
needed. Gated datasets (e.g. ``cx-cmu/agent_trajectories``) need their
conditions accepted on the dataset page and a token: passed in, or from
``HF_TOKEN``, or the one ``hf auth login`` saved.

    python -m ingest.huggingface https://huggingface.co/datasets/cx-cmu/agent_trajectories \\
        --file swebench.jsonl --limit 10 --out-dir cmu_sessions
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# Files whose records can be read line by line (or as one JSON document).
SUPPORTED_SUFFIXES = (".jsonl", ".json", ".jsonl.zst", ".json.zst", ".jsonl.gz", ".json.gz")

# A whole-document ``.json`` file has to be read in full before it can be
# parsed; refuse anything bigger rather than exhaust memory.
MAX_JSON_DOCUMENT_BYTES = 200 * 1024 * 1024

_CHUNK = 1 << 16


@dataclass(frozen=True)
class HFRef:
    repo_id: str
    revision: str = "main"
    # A file (``swebench.jsonl``) or, from a ``/tree/`` link, a directory.
    path: str | None = None

    @property
    def page_url(self) -> str:
        return f"{HF_ENDPOINT}/datasets/{self.repo_id}"


def is_hf_url(value: str) -> bool:
    return value.startswith(("hf://datasets/", f"{HF_ENDPOINT}/datasets/", "https://huggingface.co/datasets/"))


def parse_hf_url(url: str) -> HFRef:
    """Parse a dataset link (see module docstring) into repo, revision and
    optional file/directory path."""
    url = url.strip()
    if url.startswith("hf://datasets/"):
        rest = url.removeprefix("hf://datasets/")
        parts = [p for p in urllib.parse.urlsplit(rest).path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"not a Hugging Face dataset link: {url}")
        repo, revision = parts[1], "main"
        if "@" in repo:
            repo, revision = repo.split("@", 1)
        return HFRef(f"{parts[0]}/{repo}", revision, "/".join(parts[2:]) or None)

    split = urllib.parse.urlsplit(url if "://" in url else f"{HF_ENDPOINT}/datasets/{url}")
    parts = [urllib.parse.unquote(p) for p in split.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"not a Hugging Face dataset link (expected .../datasets/<org>/<name>): {url}")
    repo_id = f"{parts[1]}/{parts[2]}"
    rest = parts[3:]
    if len(rest) >= 2 and rest[0] in ("blob", "resolve", "tree"):
        return HFRef(repo_id, rest[1], "/".join(rest[2:]) or None)
    if rest and rest[0] == "viewer":
        return HFRef(repo_id)
    return HFRef(repo_id, "main", "/".join(rest) or None)


def resolve_token(token: str | None = None) -> str | None:
    if token:
        return token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_file = hf_home / "token"
    try:
        return token_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _open(url: str, token: str | None, ref: HFRef):
    request = urllib.request.Request(url, headers={"User-Agent": "agents-profiler"})
    if token:
        # Unredirected: the Hub answers a file download with a redirect to a
        # signed CDN URL, which must not receive the token.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            hint = (
                "the token was rejected or hasn't been granted access"
                if token
                else "no Hugging Face token found -- set HF_TOKEN or run `hf auth login`"
            )
            raise PermissionError(
                f"Hugging Face refused access to {ref.repo_id} (HTTP {exc.code}): {hint}. "
                f"If the dataset is gated, accept its conditions at {ref.page_url} first."
            ) from exc
        if exc.code == 404:
            raise FileNotFoundError(f"not found on Hugging Face: {url}") from exc
        raise


def list_data_files(ref: HFRef, token: str | None = None) -> list[str]:
    """Paths of the files in the dataset this module can read, limited to
    ``ref.path`` when it names a directory."""
    api = f"{HF_ENDPOINT}/api/datasets/{ref.repo_id}"
    if ref.revision != "main":
        api += f"/revision/{urllib.parse.quote(ref.revision, safe='')}"
    with _open(api, resolve_token(token), ref) as resp:
        info = json.load(resp)
    files = [s["rfilename"] for s in info.get("siblings", [])]
    prefix = f"{ref.path.rstrip('/')}/" if ref.path else ""
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(SUPPORTED_SUFFIXES))


def _decompressor(path: str):
    if path.endswith(".zst"):
        try:
            from compression import zstd  # Python 3.14+
        except ImportError:
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError("reading .zst files needs Python 3.14+ or `pip install zstandard`") from exc
            return zstandard.ZstdDecompressor().decompressobj()
        return zstd.ZstdDecompressor()
    if path.endswith(".gz"):
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    return None


def _iter_lines(resp, path: str) -> Iterator[bytes]:
    decompressor = _decompressor(path)
    buffer = b""
    while True:
        chunk = resp.read(_CHUNK)
        if not chunk:
            break
        if decompressor is not None:
            chunk = decompressor.decompress(chunk)
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        yield from lines
    if buffer:
        yield buffer


def iter_records(ref: HFRef, path: str, token: str | None = None, limit: int | None = None) -> Iterator[Any]:
    """Stream records from one dataset file: one per line for JSON Lines,
    or the items of a top-level array for a ``.json`` document. Stops
    downloading once ``limit`` records have been read."""
    if not path.endswith(SUPPORTED_SUFFIXES):
        raise ValueError(f"unsupported file type: {path} (expected one of {', '.join(SUPPORTED_SUFFIXES)})")
    url = f"{HF_ENDPOINT}/datasets/{ref.repo_id}/resolve/{urllib.parse.quote(ref.revision, safe='')}/{urllib.parse.quote(path)}"
    count = 0
    with _open(url, resolve_token(token), ref) as resp:
        if ".jsonl" in path:
            for line in _iter_lines(resp, path):
                if limit is not None and count >= limit:
                    return
                if line.strip():
                    yield json.loads(line)
                    count += 1
            return
        raw = resp.read(MAX_JSON_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_JSON_DOCUMENT_BYTES:
            raise ValueError(f"{path} is a single JSON document over {MAX_JSON_DOCUMENT_BYTES // 2**20} MB; use a .jsonl file")
        if path.endswith(".gz"):
            raw = gzip.decompress(raw)
        elif path.endswith(".zst"):
            raw = _decompressor(path).decompress(raw)
        document = json.loads(raw)
        items = document if isinstance(document, list) else [document]
        yield from items[:limit] if limit is not None else items


def fetch_records(
    url: str, file: str | None = None, limit: int | None = 10, token: str | None = None
) -> tuple[list[Any], str]:
    """The first ``limit`` records of a dataset file, and that file's path.

    The file comes from ``file``, else from the link itself (a ``/blob/``
    link), else it's the dataset's only readable file; with several to
    choose from, raises ``ValueError`` listing them."""
    ref = parse_hf_url(url)
    path = file
    if path is None and ref.path and ref.path.endswith(SUPPORTED_SUFFIXES):
        path = ref.path
    if path is None:
        candidates = list_data_files(ref, token)
        if len(candidates) != 1:
            listed = ", ".join(candidates[:30]) + (" ..." if len(candidates) > 30 else "")
            raise ValueError(
                f"{ref.repo_id} has {len(candidates)} readable files; pick one with a file name"
                + (f": {listed}" if candidates else " (only JSON / JSON Lines files are supported, not Parquet)")
            )
        path = candidates[0]
    return list(iter_records(ref, path, token, limit)), path


def _record_name(record: Any, index: int) -> str:
    key = record.get("id") or record.get("session_id") if isinstance(record, dict) else None
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)) if key else f"record_{index:05d}"
    return f"{name}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest.huggingface", description=__doc__.split("\n\n")[0])
    parser.add_argument("url", help="Hugging Face dataset link")
    parser.add_argument("--file", help="file in the dataset to read (e.g. swebench.jsonl)")
    parser.add_argument("--limit", type=int, default=10, help="records to fetch (default 10)")
    parser.add_argument("--token", help="Hugging Face token (default: HF_TOKEN or `hf auth login`)")
    parser.add_argument("--list", action="store_true", help="list the dataset's readable files and exit")
    parser.add_argument("--out-dir", default="hf_records", help="directory to save one JSON file per record")
    args = parser.parse_args(argv)

    try:
        if args.list:
            for path in list_data_files(parse_hf_url(args.url), args.token):
                print(path)
            return 0
        records, path = fetch_records(args.url, args.file, args.limit, args.token)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        target = out / _record_name(record, index)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} records from {path} to {out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
