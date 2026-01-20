"""Agent Servers - Functional service layer for OMGs system.

This package contains:
- case_parser: EHR extraction and structuring (main implementation)
- info_delivery: Role-specific case view building
- evidence_search: RAG retrieval (guideline + PubMed)
- reports_selector: Clinical report selection per role
- trace: Observability utilities (VisualConfig, TraceLogger, console tables)
- reporters: Report generation (save_mdt_log, save_case_html_report)

Note: ehr_structurer.py in root is a thin entry-point that calls case_parser.main()
"""

from .case_parser import (
    process_file as process_ehr_file,
    apply_auto_fixes,
    try_parse_json,
    extract_json_object,
)
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
from .reporters import save_mdt_log, save_case_html_report

__all__ = [
    # case_parser (EHR extraction)
    "process_ehr_file",
    "apply_auto_fixes",
    "try_parse_json",
    "extract_json_object",
    # info_delivery
    "build_role_specific_case_view",
    # evidence_search
    "get_global_guideline_rag",
    "pubmed_search_pack",
    "merge_rag_packs",
    "merge_rag_raw",
    "build_rag_query_for_mdt",
    "summarize_rag_evidence",
    # reports_selector
    "load_patient_labs",
    "load_patient_imaging",
    "load_patient_pathology",
    "load_patient_mutations",
    "select_reports_for_roles",
    "summarize_selected_reports",
    "expert_select_reports",
    # trace
    "TraceLogger",
    "VisualConfig",
    "print_selected_reports_table",
    "print_section",
    "print_rag_hits_table",
    "warn_missing_evidence_tags",
    "save_mdt_log",
    "save_case_html_report",
]
