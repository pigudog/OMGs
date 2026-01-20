"""Observability & logging utilities for MDT pipeline.

This module contains visualization and tracing utilities:
- VisualConfig: Runtime visualization switches
- TraceLogger: Lightweight structured trace for auditability
- Console printing utilities for tables and sections

NOTE: Report generation (save_mdt_log, save_case_html_report) has been
moved to servers/reporters.py for better separation of concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
from datetime import datetime

from utils.console_utils import Color, preview_text
from utils.time_utils import parse_dt
from prettytable import PrettyTable


###############################################################################
# Visualization & Trace (non-functional, observability only)
###############################################################################

@dataclass
class VisualConfig:
    """Runtime visualization switches.

    NOTE: This class only changes CLI visibility / logs. It must NOT change
    any model behavior, selection logic, or outputs.
    """
    enable: bool = True
    show_tables: bool = True
    show_rag_table: bool = True
    show_token_budget: bool = True
    max_text_preview: int = 160


class TraceLogger:
    """Lightweight structured trace for auditability and later visualization."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.events: List[Dict[str, Any]] = []

    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        self.events.append({
            "ts": datetime.now().isoformat(),
            "event": str(event),
            "payload": payload or {},
        })

    def to_mermaid_flow(self) -> str:
        """A compact mermaid flowchart for the overall pipeline."""
        # Keep it stable: do not depend on event ordering.
        return (
            "flowchart TD\n"
            "  A[Load Case + Fingerprint] --> B[Load Reports]\n"
            "  B --> C[Report Selection per Role]\n"
            "  C --> D[Guideline+PubMed RAG]\n"
            "  D --> E[Init Specialist Agents]\n"
            "  E --> F[MDT Discussion Engine]\n"
            "  F --> G[Trial Matching]\n"
            "  G --> H[Final Chair Output]\n"
            "  H --> I[Save Logs]\n"
        )


###############################################################################
# Console Printing Utilities
###############################################################################

def print_section(title: str, subtitle: str = ""):
    """Print a formatted section header to console."""
    line = "=" * 78
    if subtitle:
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{title}{Color.RESET}  {Color.OKCYAN}{subtitle}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")
    else:
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{title}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")


###############################################################################
# Evidence Tag Validation
###############################################################################

_EVIDENCE_TAG_RE = re.compile(r"\[@(?:guideline|pubmed):", re.IGNORECASE)
_EVIDENCE_CUES = [
    "guideline",
    "evidence",
    "trial",
    "nccn",
    "esmo",
    "parp",
    "maintenance",
    "platinum-sensitive",
    "platinum resistant",
    "platinum-resistant",
    "bevacizumab",
    "immunotherapy",
    "study",
    "meta-analysis",
    "randomized",
]


def warn_missing_evidence_tags(
    text: str,
    role: str,
    trace: Optional["TraceLogger"] = None,
    max_preview: int = 160,
) -> bool:
    """Warn when literature-style claims lack guideline/pubmed tags."""
    if not text:
        return False
    if _EVIDENCE_TAG_RE.search(text):
        return False
    lower = text.lower()
    if not any(cue in lower for cue in _EVIDENCE_CUES):
        return False
    preview = preview_text(text, max_preview)
    print(f"{Color.WARNING}⚠ Evidence tags missing in {role}: {preview}{Color.RESET}")
    if trace is not None:
        trace.emit("evidence_tag_warning", {"role": role, "preview": preview})
    return True


###############################################################################
# Report Table Utilities
###############################################################################

def _newest_date(items: List[Dict[str, Any]]) -> str:
    """Get the newest date from a list of report dicts."""
    dts = []
    for r in items or []:
        dt = parse_dt(r.get("date")) or parse_dt(r.get("report_date")) or parse_dt(r.get("time"))
        if dt is not None:
            dts.append(dt)
    return (max(dts).strftime("%Y-%m-%d") if dts else "-")


def print_selected_reports_table(context: Dict[str, Dict[str, List[Dict[str, Any]]]], roles: List[str]):
    """PrettyTable summary: selected report counts and newest date per role/type."""
    tbl = PrettyTable([
        "Role",
        "Lab(n)", "Lab newest",
        "Img(n)", "Img newest",
        "Path(n)", "Path newest",
        "Mut(n)", "Mut newest",
    ])
    tbl.align = "l"

    for role in roles:
        labs = (context.get("lab", {}) or {}).get(role, [])
        imgs = (context.get("imaging", {}) or {}).get(role, [])
        paths = (context.get("pathology", {}) or {}).get(role, [])
        muts = (context.get("mutation", {}) or {}).get(role, [])
        tbl.add_row([
            role,
            len(labs), _newest_date(labs),
            len(imgs), _newest_date(imgs),
            len(paths), _newest_date(paths),
            len(muts), _newest_date(muts),
        ])

    print(f"\n{Color.BOLD}{Color.OKBLUE}📌 Selected Reports Overview{Color.RESET}")
    print(tbl)


def print_rag_hits_table(rag_raw: List[Dict[str, Any]], max_rows: int = 15):
    """PrettyTable for RAG hit inspection (source/doc/page/score + snippet preview)."""
    if not rag_raw:
        print(f"{Color.WARNING}⚠ RAG: no evidence found.{Color.RESET}")
        return

    tbl = PrettyTable(["Rank", "Score", "Source", "Doc/PMID", "Page", "Preview"])
    tbl.align = "l"

    # Show all results, not just top max_rows
    display_count = min(len(rag_raw), max_rows)
    for r in (rag_raw or [])[:display_count]:
        source = r.get("source") or "guideline"
        doc_label = r.get("doc_id", "")
        page = r.get("page", "")
        preview = r.get("text", "")
        if source == "pubmed":
            doc_label = r.get("pmid") or doc_label
            page = ""
            preview = r.get("abstract", "") or preview
        tbl.add_row([
            r.get("rank"),
            f"{float(r.get('score', 0.0)):.4f}",
            source,
            preview_text(doc_label, 28),
            page,
            preview_text(preview, 240),
        ])

    total_count = len(rag_raw)
    title = f"📚 RAG Hits (Showing {display_count}/{total_count})" if total_count > display_count else f"📚 RAG Hits (All {total_count})"
    print(f"\n{Color.BOLD}{Color.OKBLUE}{title}{Color.RESET}")
    print(tbl)
