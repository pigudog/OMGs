"""Runtime configuration helpers for OMGs CLI runs."""

from __future__ import annotations

import json
import os
from typing import Any

from engine_bridge import normalize_tool_name


OMGS_QUERY_ANALYSIS_MODE_ENV = "OMGS_QUERY_ANALYSIS_MODE"
OMGS_QUERY_ANALYSIS_STYLE_ENV = "OMGS_QUERY_ANALYSIS_STYLE"
OMGS_QUERY_ANALYSIS_INPUT_ENV = "OMGS_QUERY_ANALYSIS_INPUT_JSON"
OMGS_EXTERNAL_SEARCH_DEPTH_ENV = "OMGS_EXTERNAL_SEARCH_DEPTH"
OMGS_EXTERNAL_FOLLOWUP_DEPTH_ENV = "OMGS_EXTERNAL_FOLLOWUP_DEPTH"
OMGS_EXTERNAL_FDA_MODE_ENV = "OMGS_EXTERNAL_FDA_MODE"
OMGS_EXTERNAL_CONFERENCE_MODE_ENV = "OMGS_EXTERNAL_CONFERENCE_MODE"
OMGS_ENGINE_PROVIDER_ENV = "OMGS_ENGINE_PROVIDER"
OMGS_ENGINE_MODEL_ENV = "OMGS_ENGINE_MODEL"
OMGS_GUIDELINE_SCOPE_ENV = "OMGS_GUIDELINE_SCOPE"
OMGS_GUIDELINES_FINAL_SELF_CHECK_ENV = "OMGS_GUIDELINES_FINAL_SELF_CHECK"
OMGS_GUIDELINES_RETRIEVAL_MODE_ENV = "OMGS_GUIDELINES_RETRIEVAL_MODE"
OMGS_GUIDELINES_DENSE_BACKEND_ENV = "OMGS_GUIDELINES_DENSE_BACKEND"


def _external_evidence_defaults(tool_input: dict[str, Any]) -> dict[str, Any]:
    tool_input.setdefault(
        "search_depth",
        str(os.environ.get(OMGS_EXTERNAL_SEARCH_DEPTH_ENV) or "").strip().lower() or "balanced",
    )
    tool_input.setdefault(
        "followup_depth",
        str(os.environ.get(OMGS_EXTERNAL_FOLLOWUP_DEPTH_ENV) or "").strip().lower() or "off",
    )
    tool_input.setdefault(
        "fda_mode",
        str(os.environ.get(OMGS_EXTERNAL_FDA_MODE_ENV) or "").strip().lower() or "auto",
    )
    tool_input.setdefault(
        "conference_mode",
        str(os.environ.get(OMGS_EXTERNAL_CONFERENCE_MODE_ENV) or "").strip().lower() or "auto",
    )
    tool_input.setdefault("query_analysis_mode", "auto")
    tool_input.setdefault("query_analysis_style", "single")
    return tool_input


def _guidelines_defaults(tool_input: dict[str, Any]) -> dict[str, Any]:
    tool_input.setdefault(
        "guideline_scope",
        str(os.environ.get(OMGS_GUIDELINE_SCOPE_ENV) or "").strip().lower() or "all",
    )
    tool_input.setdefault(
        "final_self_check",
        str(os.environ.get(OMGS_GUIDELINES_FINAL_SELF_CHECK_ENV) or "").strip().lower() or "off",
    )
    tool_input.setdefault(
        "guideline_retrieval_mode",
        str(os.environ.get(OMGS_GUIDELINES_RETRIEVAL_MODE_ENV) or "").strip().lower() or "dense_only",
    )
    tool_input.setdefault("guideline_dense_backend", guideline_dense_backend_from_env())
    return tool_input


def guideline_dense_backend_from_env() -> str:
    """Dense backend requested for engine guideline retrieval."""
    return (
        str(os.environ.get(OMGS_GUIDELINES_DENSE_BACKEND_ENV) or "")
        .strip()
        .lower()
        or "chroma"
    )


def tool_input_overrides_from_env(tool_name: str) -> dict[str, Any]:
    """Return engine tool input overrides from CLI/runtime environment variables."""
    canonical_tool_name = normalize_tool_name(tool_name)
    tool_input: dict[str, Any] = {}

    query_analysis_mode = str(os.environ.get(OMGS_QUERY_ANALYSIS_MODE_ENV) or "").strip().lower()
    query_analysis_style = str(os.environ.get(OMGS_QUERY_ANALYSIS_STYLE_ENV) or "").strip().lower()
    raw_query_analysis_input = str(os.environ.get(OMGS_QUERY_ANALYSIS_INPUT_ENV) or "").strip()
    if query_analysis_mode:
        tool_input["query_analysis_mode"] = query_analysis_mode
    if query_analysis_style:
        tool_input["query_analysis_style"] = query_analysis_style
    if raw_query_analysis_input:
        try:
            parsed = json.loads(raw_query_analysis_input)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            tool_input["query_analysis_input"] = parsed

    engine_provider = str(os.environ.get(OMGS_ENGINE_PROVIDER_ENV) or "").strip().lower()
    engine_model = str(os.environ.get(OMGS_ENGINE_MODEL_ENV) or "").strip()
    if engine_provider:
        tool_input["llm_provider"] = engine_provider
    if engine_model:
        tool_input["llm_model"] = engine_model

    if canonical_tool_name == "external_evidence":
        return _external_evidence_defaults(tool_input)
    if canonical_tool_name == "guidelines":
        return _guidelines_defaults(tool_input)
    return tool_input
