"""Agent Servers - Functional service layer for OMGs system.

This package contains:
- data_extraction: Clinical data extraction and structuring (EHR)
- context_assembly: Role-specific case view building
- evidence_retrieval: RAG retrieval (guideline + PubMed)
- report_selection: Clinical report selection per role
- provenance_tracking: Observability utilities (VisualConfig, TraceLogger, console tables)

Note: ehr_structurer.py in root is a thin entry-point that calls data_extraction.main()
"""

from .data_extraction import (
    process_file as process_ehr_file,
    apply_auto_fixes,
    try_parse_json,
)
from .context_assembly import build_role_specific_case_view
from .evidence_retrieval import (
    get_global_guideline_rag,
    get_nccn_rag,
    pubmed_search_pack,
    retrieve_mdt_evidence_sources,
    merge_rag_packs,
    merge_rag_raw,
    build_rag_query_for_mdt,
    summarize_rag_evidence,
)
from .report_selection import (
    load_patient_labs,
    load_patient_imaging,
    load_patient_pathology,
    load_patient_mutations,
    select_reports_for_roles,
    summarize_selected_reports,
    expert_select_reports,
)
from .provenance_tracking import TraceLogger, VisualConfig, print_selected_reports_table, print_section, print_rag_hits_table, warn_missing_evidence_tags

__all__ = [
    # data_extraction (EHR extraction)
    "process_ehr_file",
    "apply_auto_fixes",
    "try_parse_json",
    # context_assembly
    "build_role_specific_case_view",
    # evidence_retrieval
    "get_global_guideline_rag",
    "get_nccn_rag",
    "pubmed_search_pack",
    "retrieve_mdt_evidence_sources",
    "merge_rag_packs",
    "merge_rag_raw",
    "build_rag_query_for_mdt",
    "summarize_rag_evidence",
    # report_selection
    "load_patient_labs",
    "load_patient_imaging",
    "load_patient_pathology",
    "load_patient_mutations",
    "select_reports_for_roles",
    "summarize_selected_reports",
    "expert_select_reports",
    # provenance_tracking
    "TraceLogger",
    "VisualConfig",
    "print_selected_reports_table",
    "print_section",
    "print_rag_hits_table",
    "warn_missing_evidence_tags",
]
