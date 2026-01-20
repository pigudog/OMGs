"""Utility package for OMGs/MDT pipeline.

This package contains small, reusable helpers that should NOT change any
clinical decision logic. Typical examples include color codes, JSON helpers,
date/time utilities, and formatters.

Note: Core functionality has been moved to dedicated modules:
- core/ - Agent, client, config
- host/ - Orchestration, experts, decision-making
- servers/ - RAG, report selection, trace logging
"""

from .console_utils import (
    Color,
    preview_text,
    print_prompt_budget,
    normalize_trial_compact,
    safe_parse_json_block,
    question_to_text,
)
from .time_utils import (
    parse_dt,
    parse_date,
    make_cutoff,
    filter_before,
    safe_date10,
    report_range,
    build_lab_timeline,
    build_imaging_timeline,
    build_pathology_timeline,
)
from .reference_cache import (
    ReferenceCache,
    get_reference_cache,
    extract_reference_tags,
    build_references_section,
)

__all__ = [
    # Console utilities
    "Color",
    "preview_text",
    "print_prompt_budget",
    "normalize_trial_compact",
    "safe_parse_json_block",
    "question_to_text",
    # Time utilities
    "parse_dt",
    "parse_date",
    "make_cutoff",
    "filter_before",
    "safe_date10",
    "report_range",
    "build_lab_timeline",
    "build_imaging_timeline",
    "build_pathology_timeline",
    # Reference cache
    "ReferenceCache",
    "get_reference_cache",
    "extract_reference_tags",
    "build_references_section",
]
