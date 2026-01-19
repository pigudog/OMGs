"""Utility package for OMGs/MDT pipeline.

This package contains small, reusable helpers that should NOT change any
clinical decision logic. Typical examples include report/log writers,
formatters, and lightweight convenience functions.

Import style:
- Preferred (explicit):
    from utils.omgs_reports import save_mdt_log
- Convenience (re-exported here):
    from utils import save_mdt_log

`__all__` below defines what names are exported when users do:
    from utils import *
It is also a clear public API list for this package.
"""

# Re-export public helpers for convenient imports
from .reports_utils import *
from .console_utils import *
from .time_utils import *
from .trace_utils import *
from .select_utils import *
from .rag_utils import *
from .role_utils import *
from .core import *
# Public API of `utils`
__all__ = [
    "save_mdt_log",
    "save_case_html_report",
    "Color",
    "normalize_trial_compact",
    "parse_dt",
    "make_cutoff","report_range","safe_date10",
    "filter_before","VisualConfig", 
    "build_lab_timeline","build_imaging_timeline","build_pathology_timeline",
    "TraceLogger","preview_text","print_selected_reports_table","print_section","print_rag_hits_table","warn_missing_evidence_tags",
    "load_patient_labs","load_patient_imaging","load_patient_pathology","load_patient_mutations","read_jsonl","parse_ids","parse_date_any","summarize_selected_reports"
]