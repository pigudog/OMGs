"""Agent Servers - Functional service layer for OMGs system.

This package contains:
- case_parser: EHR extraction and structuring
- info_delivery: Role-specific case view building
- evidence_search: RAG retrieval (guideline + PubMed)
- reports_selector: Clinical report selection per role
- trace: Observability and logging
"""

# case_parser is ehr_structurer.py - kept as separate file for backward compatibility
from .info_delivery import build_role_specific_case_view
from .evidence_search import (
    get_global_guideline_rag,
    pubmed_search_pack,
    merge_rag_packs,
    merge_rag_raw,
    build_rag_query_for_mdt,
    summarize_rag_evidence,
)
from .reports_selector import (
    load_patient_labs,
    load_patient_imaging,
    load_patient_pathology,
    load_patient_mutations,
    select_reports_for_roles,
    summarize_selected_reports,
    expert_select_reports,
)
from .trace import TraceLogger, VisualConfig, print_selected_reports_table, print_section, print_rag_hits_table, warn_missing_evidence_tags
from .trace import save_mdt_log, save_case_html_report

__all__ = [
    "build_role_specific_case_view",
    "get_global_guideline_rag",
    "pubmed_search_pack",
    "merge_rag_packs",
    "merge_rag_raw",
    "build_rag_query_for_mdt",
    "summarize_rag_evidence",
    "load_patient_labs",
    "load_patient_imaging",
    "load_patient_pathology",
    "load_patient_mutations",
    "select_reports_for_roles",
    "summarize_selected_reports",
    "expert_select_reports",
    "TraceLogger",
    "VisualConfig",
    "print_selected_reports_table",
    "print_section",
    "print_rag_hits_table",
    "warn_missing_evidence_tags",
    "save_mdt_log",
    "save_case_html_report",
]
