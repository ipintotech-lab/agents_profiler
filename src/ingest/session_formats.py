"""Registry of the harness session-transcript formats the ingest layer can
convert into TraceEvent-shaped dicts (see ``normalize.py`` for how it's
used).

Adding a format
---------------
1. Write a converter module (use ``claude_code.py`` or ``pi_session.py`` as
   the model) exposing:
   * ``is_<format>(parsed) -> bool`` -- a detector that only fires on that
     format's records;
   * ``convert_<format>(rows) -> list[dict]`` -- returns flat dicts ready
     for ``TraceEvent.model_validate`` (see ``trace_schema.py``).
2. Register the pair below:
   * in ``TRANSCRIPT_FORMATS`` if the *whole file* is one session (e.g. a
     JSONL transcript where lines are records of one session);
   * in ``RECORD_FORMATS`` if *each JSON record* is one whole
     session/trajectory (e.g. a dataset of chat trajectories).

``TRANSCRIPT_FORMATS`` detectors are tried first, in order; then
``RECORD_FORMATS`` (which require *every* record to match one format).
Whatever survives goes through the generic field-alias normalization in
``normalize.py``.
"""
from __future__ import annotations

from typing import Any, Callable

from .chat_trajectory import convert_chat_trajectories, is_chat_trajectory
from .claude_code import convert_claude_code_transcript, is_claude_code_transcript
from .freeinference import convert_freeinference_sessions, is_freeinference_session
from .pi_session import convert_pi_session, is_pi_session

# (format name, detector, converter)
Format = tuple[str, Callable[[Any], bool], Callable[[list], list[dict]]]

# Whole-file session transcripts (JSONL logs where the file is one session).
TRANSCRIPT_FORMATS: tuple[Format, ...] = (
    ("claude-code", is_claude_code_transcript, convert_claude_code_transcript),
    ("pi", is_pi_session, convert_pi_session),
)

# Per-record formats: every JSON record in the file is one whole session.
RECORD_FORMATS: tuple[Format, ...] = (
    ("freeinference", is_freeinference_session, convert_freeinference_sessions),
    ("chat-trajectory", is_chat_trajectory, convert_chat_trajectories),
)
