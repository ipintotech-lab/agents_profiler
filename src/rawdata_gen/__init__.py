"""Synthetic agent-telemetry generator, vendored from
https://github.com/ipintotech-lab/agents_profile_rawdata_gen so the
dashboard can generate sample trace data without an external dependency.
"""
from .generate_traces import generate_configured, verify_seed_reproducibility

__all__ = ["generate_configured", "verify_seed_reproducibility"]
