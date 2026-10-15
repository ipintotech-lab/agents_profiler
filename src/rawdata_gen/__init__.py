"""Synthetic agent-telemetry generator. It started as
https://github.com/ipintotech-lab/agents_profile_rawdata_gen and now lives
here: the mock agents' trace events (``generate_traces``), Claude Code
session transcripts (``claude_code_sessions``), and a CLI
(``python -m rawdata_gen``).
"""
from .generate_traces import (
    generate_configured,
    verify_seed_range_reproducibility,
    verify_seed_reproducibility,
)

__all__ = [
    "generate_configured",
    "verify_seed_range_reproducibility",
    "verify_seed_reproducibility",
]
