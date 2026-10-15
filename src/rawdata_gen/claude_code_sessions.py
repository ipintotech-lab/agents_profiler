"""Generate synthetic Claude Code session transcripts -- the JSONL that
Claude Code writes under ``~/.claude/projects/<project>/<sessionId>.jsonl``
-- so the profiler's Claude Code path (``ingest/claude_code.py``) can be
exercised without sharing a real session.

Each session is a coding agent working in one project: the user gives a few
requests (fix a failing test, add a field, update the README, look into a
slow module), and for each one the agent runs a short, realistic chain of
tool calls -- ``git status``, ``Grep``, ``Read``, ``Edit``, ``pytest`` -- with
the records a real transcript has: one ``user`` record per prompt and per
tool result, one ``assistant`` record per content block (sharing
``message.id``), token ``usage`` per response, timestamps, and a
session-level ``cost-state`` record with the cost per model.

The chains repeat on purpose, as real sessions do: the same file is read
again after it's edited, ``git status`` returns the same output until
something changes, and ``pytest`` fails before the fix and passes after.

``omit`` drops attributes to imitate an incomplete export: ``"tokens"``
(no ``usage``), ``"cost"`` (no ``cost-state``), ``"timing"`` (no
timestamps). The report's data notice should then name what is missing.
"""
from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

OMITTABLE = ("tokens", "cost", "timing")

VERSION = "2.1.0"
PROJECTS = ("billing-service", "inventory-api", "notes-app")
MODULES = ("orders", "invoices", "users", "payments", "reports")
FIELDS = ("discount_code", "currency", "archived_at", "region")

# Synthetic $ per 1M tokens (input, output) -- only to give the cost-state
# record a plausible value; cache reads cost 0.1x input, writes 1.25x.
MODEL_PRICES = {"claude-opus-5-5": (5.0, 25.0), "claude-sonnet-5-5": (3.0, 15.0)}

# Typical tool durations in ms (low, high).
_TOOL_MS = {
    "Read": (15, 90),
    "Edit": (30, 160),
    "Grep": (40, 300),
    "Glob": (20, 120),
    "Bash": (80, 900),
}
_PYTEST_MS = (2500, 14000)


def _tasks(module: str, field: str) -> dict[str, tuple[str, list]]:
    """Request text and tool steps for each kind of task. A step is a tool
    call ``(name, input)`` or a list of them made in one response. Outputs
    are filled in by ``_Workspace`` so repeats return the same output."""
    src, test = f"src/{module}.py", f"tests/test_{module}.py"
    status = ("Bash", {"command": "git status --short", "description": "Show working tree status"})
    pytest_one = ("Bash", {"command": f"python -m pytest -q {test}", "description": f"Run {module} tests"})
    return {
        "fix_test": (
            f"The {module} tests are failing, can you fix them?",
            [status, pytest_one, [("Read", {"file_path": src}), ("Read", {"file_path": test})],
             ("Edit", {"file_path": src, "old_string": "total = sum(items)", "new_string": "total = sum(i.amount for i in items)"}),
             pytest_one, status],
        ),
        "add_field": (
            f"Add a {field} field to the {module} API and cover it with a test.",
            [("Grep", {"pattern": f"class {module.title()}", "path": "src"}), ("Read", {"file_path": src}),
             ("Edit", {"file_path": src, "old_string": "    id: int\n", "new_string": f"    id: int\n    {field}: str | None = None\n"}),
             ("Read", {"file_path": test}),
             ("Edit", {"file_path": test, "old_string": "def test_create", "new_string": f"def test_{field}():\n    ...\n\n\ndef test_create"}),
             pytest_one, ("Bash", {"command": "git diff --stat", "description": "Summarize changes"})],
        ),
        "readme": (
            f"Update the README to describe the {module} endpoints.",
            [("Read", {"file_path": "README.md"}), ("Glob", {"pattern": "src/*.py"}), ("Read", {"file_path": src}),
             ("Edit", {"file_path": "README.md", "old_string": "## Endpoints\n", "new_string": f"## Endpoints\n\n### {module}\n"}),
             status],
        ),
        "slow": (
            f"Why is the {module} report slow?",
            [("Grep", {"pattern": "def build_report", "path": "src"}), ("Read", {"file_path": src}),
             ("Bash", {"command": f"python -m cProfile -s cumtime -m {module} | head -20", "description": "Profile the report"}),
             ("Read", {"file_path": src})],
        ),
    }


class _Workspace:
    """Tracks file versions so outputs repeat until something changes."""

    def __init__(self):
        self.versions: dict[str, int] = {}
        self.dirty: set[str] = set()
        self.fixed: set[str] = set()

    def run(self, name: str, tool_input: dict) -> tuple[str, bool]:
        path = tool_input.get("file_path", "")
        if name == "Read":
            v = self.versions.get(path, 0)
            return f"{path} (version {v})\n" + "".join(f"{n:>4}\tline {n} of {path}\n" for n in range(1, 6)), False
        if name == "Edit":
            self.versions[path] = self.versions.get(path, 0) + 1
            self.dirty.add(path)
            if path.startswith("src/"):
                self.fixed.add(path)
            return f"The file {path} has been updated successfully.", False
        if name == "Grep":
            return f"src/{tool_input['pattern'].split()[-1].lower().strip('(')}.py:12", False
        if name == "Glob":
            return "\n".join(f"src/{m}.py" for m in MODULES), False
        command = tool_input.get("command", "")
        if command.startswith("git status"):
            return "\n".join(f" M {p}" for p in sorted(self.dirty)) or "", False
        if command.startswith("git diff"):
            return f" {len(self.dirty)} files changed", False
        if "pytest" in command:
            src = "src/" + command.split("test_")[-1]
            if src in self.fixed:
                return "....                                     [100%]\n4 passed", False
            return "..F.                                     [100%]\nFAILED test_total - assert 0 == 42\n1 failed, 3 passed", True
        return "  ncalls  tottime  cumtime\n     500    1.900    2.400 build_report", False


def _usage(rng: random.Random, context: int, output: int) -> dict:
    created = rng.randint(200, 2500)
    return {
        "input_tokens": rng.randint(1, 8),
        "cache_creation_input_tokens": created,
        "cache_read_input_tokens": context,
        "output_tokens": output,
        "cache_creation": {"ephemeral_1h_input_tokens": created, "ephemeral_5m_input_tokens": 0},
    }


def _cost(model: str, usage: dict) -> float:
    price_in, price_out = MODEL_PRICES[model]
    return (
        usage["input_tokens"] * price_in
        + usage["cache_read_input_tokens"] * price_in * 0.1
        + usage["cache_creation_input_tokens"] * price_in * 1.25
        + usage["output_tokens"] * price_out
    ) / 1_000_000


def generate_session(
    rng: random.Random,
    prompts: int = 3,
    omit: tuple[str, ...] = (),
    project: str | None = None,
    start: datetime | None = None,
) -> list[dict]:
    """One session transcript: ``prompts`` user requests in ``project``,
    each worked through with tool calls. Returns the JSONL rows, in order."""
    bad = set(omit) - set(OMITTABLE)
    if bad:
        raise ValueError(f"omit must be among {OMITTABLE}, got {sorted(bad)}")
    session_id = str(uuid.UUID(int=rng.getrandbits(128)))
    project = project or rng.choice(PROJECTS)
    model = rng.choice(list(MODEL_PRICES))
    cwd = f"/home/dev/{project}"
    workspace = _Workspace()
    clock = start or datetime(2026, 10, 1, 9, 0, tzinfo=UTC) + timedelta(minutes=rng.randint(0, 600))
    rows: list[dict] = []
    parent: str | None = None
    context = rng.randint(8000, 15000)
    cost = 0.0

    def record(kind: str, message: dict, **extra) -> dict:
        nonlocal parent
        uid = str(uuid.UUID(int=rng.getrandbits(128)))
        row = {
            "type": kind, "sessionId": session_id, "uuid": uid, "parentUuid": parent, "cwd": cwd,
            "version": VERSION, "gitBranch": "main", "isSidechain": False, "userType": "external",
            "message": message, **extra,
        }
        if "timing" not in omit:
            row["timestamp"] = clock.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        rows.append(row)
        parent = uid
        return row

    def respond(content: list[dict], stop_reason: str) -> None:
        nonlocal clock, context, cost
        output = rng.randint(40, 120) + 60 * len(content)
        clock += timedelta(milliseconds=rng.randint(1200, 4000) + output * 12)
        usage = _usage(rng, context, output)
        context += usage["cache_creation_input_tokens"] + output
        cost += _cost(model, usage)
        message_id = f"msg_{rng.getrandbits(64):016x}"
        request_id = f"req_{rng.getrandbits(64):016x}"
        # Claude Code writes one assistant record per content block
        for block in content:
            message = {"id": message_id, "type": "message", "role": "assistant", "model": model,
                       "content": [block], "stop_reason": stop_reason, "stop_sequence": None}
            if "tokens" not in omit:
                message["usage"] = usage
            record("assistant", message, requestId=request_id)

    task_kinds = rng.sample(list(_tasks("x", "y")), k=min(prompts, 4)) + [
        rng.choice(["fix_test", "add_field", "readme", "slow"]) for _ in range(max(prompts - 4, 0))
    ]
    for kind in task_kinds:
        module, field = rng.choice(MODULES), rng.choice(FIELDS)
        text, steps = _tasks(module, field)[kind]
        prompt_id = str(uuid.UUID(int=rng.getrandbits(128)))
        clock += timedelta(seconds=rng.randint(20, 240))
        record("user", {"role": "user", "content": text}, promptId=prompt_id)
        for step in steps:
            calls = step if isinstance(step, list) else [step]
            blocks = [{"type": "tool_use", "id": f"toolu_{rng.getrandbits(64):016x}", "name": n, "input": i}
                      for n, i in calls]
            if rng.random() < 0.4:
                blocks.insert(0, {"type": "text", "text": "Let me look at that."})
            respond(blocks, "tool_use")
            for block in blocks:
                if block["type"] != "tool_use":
                    continue
                output, is_error = workspace.run(block["name"], block["input"])
                low, high = _PYTEST_MS if "pytest" in block["input"].get("command", "") else _TOOL_MS[block["name"]]
                clock += timedelta(milliseconds=rng.randint(low, high))
                record(
                    "user",
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": block["id"], "content": output, "is_error": is_error}
                    ]},
                    promptId=prompt_id,
                )
        respond([{"type": "text", "text": f"Done: {text[0].lower()}{text[1:]}"}], "end_turn")

    if "cost" not in omit:
        rows.append({
            "type": "cost-state", "sessionId": session_id, "totalCostUSD": cost,
            "modelUsage": {model: {"costUSD": cost}},
        })
    return rows


def generate_sessions(
    sessions: int = 3, prompts_per_session: int = 3, rng: random.Random | None = None, omit: tuple[str, ...] = ()
) -> list[list[dict]]:
    if not 1 <= sessions <= 50:
        raise ValueError("sessions must be between 1 and 50")
    if not 1 <= prompts_per_session <= 10:
        raise ValueError("prompts_per_session must be between 1 and 10")
    rng = rng or random.Random()
    # one project, so all sessions land in one workflow (claude-code/<project>)
    project = rng.choice(PROJECTS)
    return [generate_session(rng, prompts_per_session, omit, project) for _ in range(sessions)]
