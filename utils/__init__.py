"""Utility package for OMGs/MDT pipeline.

This package contains small, reusable helpers that should NOT change any
clinical decision logic. Typical examples include color codes, JSON helpers,
date/time utilities, and formatters.

Note: Core functionality has been moved to dedicated modules:
- core/ - Agent, client, config
- orchestrator/ - Deliberation, experts, decision-making
- servers/ - data_extraction, context_assembly, evidence_retrieval, report_selection, provenance_tracking
"""

from .console_utils import (
    Color,
    preview_text,
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
    newest_date,
    format_duration,
)
from .mutation_interpretation import (
    NGS_INTERPRETATION_RULES,
    NGS_INTERPRETATION_RULES_SHORT,
    build_mutation_guidance,
)
from .patterns import (
    EVIDENCE_TAG_RE,
    EVIDENCE_CUES,
    extract_reference_tags,
)
from .reference_cache import (
    ReferenceCache,
    get_reference_cache,
    build_references_section,
)

__all__ = [
    # Console utilities
    "Color",
    "preview_text",
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
    "newest_date",
    "format_duration",
    # Mutation interpretation
    "NGS_INTERPRETATION_RULES",
    "NGS_INTERPRETATION_RULES_SHORT",
    "build_mutation_guidance",
    # Patterns
    "EVIDENCE_TAG_RE",
    "EVIDENCE_CUES",
    "extract_reference_tags",
    # Reference cache
    "ReferenceCache",
    "get_reference_cache",
    "build_references_section",
]
