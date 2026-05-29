"""Local bridge from the OMGs release runtime to omgs_engine.

This module keeps the release runtime independent from the product-shell `omgs` package
while still using the shared `omgs_engine` dispatcher.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


def normalize_tool_name(tool_name: str) -> str:
    from omgs_engine.dispatcher import normalize_tool_name as _normalize_tool_name

    return _normalize_tool_name(tool_name)


def default_tool_call_id(tool_name: str) -> str:
    return f"omgs:{normalize_tool_name(tool_name)}"


def build_tool_input(query: str, **overrides: Any) -> dict[str, Any]:
    return {"query": str(query or "").strip(), **overrides}


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _extract_bridge_guidelines_env(tool_name: str, tool_input: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    projected = dict(tool_input)
    env: dict[str, str] = {}
    if tool_name != "guidelines":
        return projected, env

    dense_backend = str(projected.pop("__guideline_dense_backend", "") or "").strip().lower()
    if dense_backend:
        env["OMGS_ENGINE_GUIDELINES_DENSE_BACKEND"] = dense_backend
    return projected, env


def _strip_incompatible_guidelines_controls(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    if (
        tool_name == "guidelines"
        and str(tool_input.get("query_analysis_mode") or "").strip().lower() == "off"
        and "search_depth" in tool_input
    ):
        projected = dict(tool_input)
        projected.pop("search_depth", None)
        return projected
    return tool_input


class EngineIntegration:
    def __init__(self, service: Any | None = None) -> None:
        from omgs_engine.dispatcher import EngineDispatcher

        self._service = service or EngineDispatcher()

    def invoke_tool(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        tool_input: dict[str, Any],
        consumer: str = "cli",
        verbosity: str = "standard",
        include_artifacts: bool = True,
        include_debug: bool = False,
        include_snapshots: bool = True,
    ):
        canonical_tool_name = normalize_tool_name(tool_name)
        tool_input, bridge_env = _extract_bridge_guidelines_env(
            canonical_tool_name,
            tool_input,
        )
        normalized_tool_input = self._service.normalize_tool_input(
            canonical_tool_name,
            tool_input,
        )
        normalized_tool_input = _strip_incompatible_guidelines_controls(
            canonical_tool_name,
            normalized_tool_input,
        )
        with _temporary_env(bridge_env):
            return self._service.invoke_tool(
                tool_name=canonical_tool_name,
                tool_call_id=tool_call_id,
                tool_input=normalized_tool_input,
                consumer=consumer,
                verbosity=verbosity,
                include_artifacts=include_artifacts,
                include_debug=include_debug,
                include_snapshots=include_snapshots,
            )
