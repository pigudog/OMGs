"""Runtime feature flags for privacy-sensitive observability."""

from __future__ import annotations

import os


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def api_trace_enabled() -> bool:
    return env_flag("OMGS_API_TRACE_ENABLED", default=True)


def api_trace_raw_enabled() -> bool:
    return env_flag("OMGS_API_TRACE_RAW_ENABLED", default=False)
