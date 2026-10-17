"""Upload validation: what a file is, decided from its bytes.

The filename's extension, the browser's Content-Type and anything the
page checked are hints at best. The allowlist is what the profiler reads:
UTF-8 JSON, JSON Lines and CSV. Archives and compressed files are
rejected outright (no zip bombs, nested archives or archive path
traversal to defend against), as are binaries, NUL bytes, and JSON
nested deep enough to exhaust the parser.
"""
from __future__ import annotations

import csv
import io
import json
import unicodedata
from dataclasses import dataclass

SUPPORTED_FILE_TYPES = {
    "json": {"extensions": (".json",), "mime": "application/json"},
    "jsonl": {"extensions": (".jsonl", ".ndjson"), "mime": "application/x-ndjson"},
    "csv": {"extensions": (".csv",), "mime": "text/csv"},
}
# JSON and JSON Lines can't always be told apart by content (a one-line
# JSON Lines file is a JSON document), so either extension fits either.
_COMPATIBLE = {"json": {"json", "jsonl"}, "jsonl": {"json", "jsonl"}, "csv": {"csv"}}

MAX_FILENAME_CHARS = 255
MAX_JSON_DEPTH = 64

_MAGIC = [
    (b"PK\x03\x04", "ZIP archive"),
    (b"PK\x05\x06", "ZIP archive"),
    (b"\x1f\x8b", "gzip archive"),
    (b"\x28\xb5\x2f\xfd", "zstd archive"),
    (b"BZh", "bzip2 archive"),
    (b"\xfd7zXZ\x00", "xz archive"),
    (b"7z\xbc\xaf\x27\x1c", "7z archive"),
    (b"Rar!", "RAR archive"),
    (b"%PDF", "PDF"),
    (b"\x7fELF", "executable"),
    (b"MZ", "executable"),
    (b"\xd0\xcf\x11\xe0", "Office document"),
]


class Rejected(ValueError):
    """The file isn't accepted; the message is safe to show (it never
    quotes the file's content)."""


@dataclass(frozen=True)
class Detected:
    type: str
    mime: str


def display_filename(name: str | None) -> str:
    """The customer's filename as metadata only: last path segment,
    NFC-normalized, control and bidi-override characters removed, capped
    in length. It is never used as a storage path."""
    name = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = unicodedata.normalize("NFC", name)
    name = "".join(c for c in name if unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    name = name.strip().strip(".") or "upload"
    if len(name) > MAX_FILENAME_CHARS:
        stem, dot, ext = name.rpartition(".")
        ext = ext[:16] if dot else ""
        name = (stem if dot else name)[: MAX_FILENAME_CHARS - len(ext) - 1] + (f".{ext}" if ext else "")
    return name


def extension_type(filename: str) -> str | None:
    lower = filename.lower()
    for kind, spec in SUPPORTED_FILE_TYPES.items():
        if lower.endswith(spec["extensions"]):
            return kind
    return None


def _depth(value, limit: int) -> int:
    stack, deepest = [(value, 1)], 0
    while stack:
        item, d = stack.pop()
        deepest = max(deepest, d)
        if deepest > limit:
            return deepest
        if isinstance(item, dict):
            stack.extend((v, d + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, d + 1) for v in item)
    return deepest


def _load_json(text: str):
    try:
        value = json.loads(text)
    except RecursionError as exc:
        raise Rejected(f"JSON is nested more than {MAX_JSON_DEPTH} levels deep") from exc
    if _depth(value, MAX_JSON_DEPTH) > MAX_JSON_DEPTH:
        raise Rejected(f"JSON is nested more than {MAX_JSON_DEPTH} levels deep")
    return value


def _is_json_lines(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    for ln in lines:
        try:
            if not isinstance(_load_json(ln), dict):
                return False
        except (ValueError, json.JSONDecodeError):
            return False
    return True


def _is_csv(text: str) -> bool:
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r]
    return len(rows) >= 2 and len(rows[0]) >= 2 and all(len(r) == len(rows[0]) for r in rows)


def detect(data: bytes, filename: str) -> Detected:
    """The file's type from its content, or ``Rejected``."""
    if not data:
        raise Rejected("file is empty")
    for magic, what in _MAGIC:
        if data.startswith(magic):
            raise Rejected(f"{what} files are not accepted; upload JSON, JSON Lines or CSV")
    if b"\0" in data:
        raise Rejected("binary content is not accepted; upload JSON, JSON Lines or CSV")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise Rejected("file is not UTF-8 text") from exc
    if not text.strip():
        raise Rejected("file has no content")
    claimed = extension_type(filename)
    if claimed is None:
        allowed = ", ".join(e for spec in SUPPORTED_FILE_TYPES.values() for e in spec["extensions"])
        raise Rejected(f"unsupported file extension; allowed: {allowed}")
    kind = None
    try:
        value = _load_json(text)
        kind = "json" if isinstance(value, (dict, list)) else None
    except Rejected:
        raise
    except (ValueError, json.JSONDecodeError):
        pass
    if kind is None and _is_json_lines(text):
        kind = "jsonl"
    if kind is None and claimed == "csv" and _is_csv(text):
        kind = "csv"
    if kind is None:
        raise Rejected("content is not valid JSON, JSON Lines or CSV")
    if kind not in _COMPATIBLE[claimed]:
        raise Rejected(f"content is {kind.upper()} but the file is named as {claimed.upper()}")
    return Detected(kind, SUPPORTED_FILE_TYPES[kind]["mime"])
