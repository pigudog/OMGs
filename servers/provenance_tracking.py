"""Observability & logging utilities for MDT pipeline.

This module contains visualization and tracing utilities:
- VisualConfig: Runtime visualization switches
- TraceLogger: Lightweight structured trace for auditability
- Console printing utilities for tables and sections

NOTE: Default runtime output is handled by main.py's result bundle; detailed
artifact reporters are not part of the default pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
from datetime import datetime

from utils.console_utils import Color, preview_text
from utils.time_utils import parse_dt, newest_date
from utils.patterns import EVIDENCE_TAG_RE, EVIDENCE_CUES
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
            "  H --> I[Return Result]\n"
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors that occurred during the pipeline.
        
        Returns:
            Dictionary with error statistics:
            - total_errors: Total number of error events
            - errors_by_role: Count of errors per role
            - errors_by_stage: Count of errors per stage
            - error_details: List of all error events
        """
        error_events = [e for e in self.events if e.get("event") in ("agent_error", "pipeline_error")]
        
        errors_by_role: Dict[str, int] = {}
        errors_by_stage: Dict[str, int] = {}
        
        for event in error_events:
            payload = event.get("payload", {})
            role = payload.get("role", "unknown")
            stage = payload.get("stage", "unknown")
            
            errors_by_role[role] = errors_by_role.get(role, 0) + 1
            errors_by_stage[stage] = errors_by_stage.get(stage, 0) + 1
        
        return {
            "total_errors": len(error_events),
            "errors_by_role": errors_by_role,
            "errors_by_stage": errors_by_stage,
            "error_details": error_events
        }
    
    def has_errors(self) -> bool:
        """
        Check if any errors occurred during the pipeline.
        
        Returns:
            True if any error events were recorded
        """
        return any(e.get("event") in ("agent_error", "pipeline_error") for e in self.events)


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

def warn_missing_evidence_tags(
    text: str,
    role: str,
    trace: Optional["TraceLogger"] = None,
    max_preview: int = 160,
) -> bool:
    """Warn when literature-style claims lack guideline/pubmed tags."""
    if not text:
        return False
    if EVIDENCE_TAG_RE.search(text):
        return False
    lower = text.lower()
    if not any(cue in lower for cue in EVIDENCE_CUES):
        return False
    preview = preview_text(text, max_preview)
    print(f"{Color.WARNING}⚠ Evidence tags missing in {role}: {preview}{Color.RESET}")
    if trace is not None:
        trace.emit("evidence_tag_warning", {"role": role, "preview": preview})
    return True


###############################################################################
# Report Table Utilities
###############################################################################

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
            len(labs), newest_date(labs),
            len(imgs), newest_date(imgs),
            len(paths), newest_date(paths),
            len(muts), newest_date(muts),
        ])

    print(f"\n{Color.BOLD}{Color.OKBLUE}📌 Selected Reports Overview{Color.RESET}")
    print(tbl)


def print_rag_hits_table(rag_raw: List[Dict[str, Any]], max_rows: int = 15):
    """PrettyTable for RAG hit inspection organized by source type with beautiful formatting."""
    if not rag_raw:
        print(f"{Color.WARNING}RAG: no evidence found.{Color.RESET}")
        return

    guideline_results = [
        r for r in rag_raw
        if r.get("source") == "guideline" or str(r.get("source", "")).startswith("nccn")
    ]
    external_results = [
        r for r in rag_raw
        if r.get("source") in {"pubmed", "fda", "conference"}
    ]

    total_count = len(rag_raw)
    print(f"\n{Color.BOLD}{Color.OKBLUE}╔{'═' * 56}╗")
    print(f"║{' RAG Evidence Summary ':^{56}}║")
    print(f"╚{'═' * 56}╝{Color.RESET}")
    print(f"{Color.BOLD}Total Evidence: {total_count} | Guidelines: {len(guideline_results)} | External Evidence: {len(external_results)}{Color.RESET}\n")

    if guideline_results:
        print(f"{Color.BOLD}{Color.OKGREEN}┌{'─' * 56}┐")
        print(f"│{' Guidelines ':^{56}}│")
        print(f"└{'─' * 56}┘{Color.RESET}")
        guide_tbl = PrettyTable(["Rank", "Type", "ID", "Page", "Excerpt"])
        guide_tbl.align = "l"
        for r in guideline_results[:max_rows]:
            source = r.get("source", "")
            preview = r.get("text", "")
            if str(source).startswith("nccn"):
                if source == "nccn_safety_rule":
                    item_type = "NCCN Safety"
                elif source == "nccn_matcher_rule":
                    item_type = "NCCN Matcher"
                else:
                    item_type = "NCCN Pathway"
                item_id = r.get("rule_id") or r.get("node_id", "")
                page = "-"
                node_name = r.get("node_name", "")
                if node_name:
                    preview = f"{node_name}: {preview}"
            else:
                item_type = "Guideline"
                item_id = r.get("doc_id", "")
                page = r.get("page", "") or "-"
            guide_tbl.add_row([
                r.get("rank", "-"),
                item_type,
                preview_text(item_id, 24),
                page,
                preview_text(preview, 130),
            ])
        print(guide_tbl)
        print()

    if external_results:
        print(f"{Color.BOLD}{Color.WARNING}┌{'─' * 56}┐")
        print(f"│{' External Evidence ':^{56}}│")
        print(f"└{'─' * 56}┘{Color.RESET}")
        external_tbl = PrettyTable(["Rank", "Type", "ID", "Source", "Title / Excerpt"])
        external_tbl.align = "l"
        for r in external_results[:max_rows]:
            source = r.get("source", "")
            if source == "pubmed":
                item_type = "PubMed"
                item_id = r.get("pmid", "")
                source_name = r.get("journal", "") or r.get("metadata", {}).get("journal", "")
                title = r.get("title", "")
                excerpt = r.get("abstract", "")
            elif source == "fda":
                item_type = "FDA"
                item_id = r.get("source_id") or r.get("id", "")
                source_name = "FDA"
                title = r.get("title", "")
                excerpt = r.get("text", "") or r.get("summary", "")
            else:
                item_type = "Conference"
                item_id = r.get("source_id") or r.get("id", "")
                source_name = r.get("conference", "") or r.get("source_name", "")
                title = r.get("title", "")
                excerpt = r.get("text", "") or r.get("abstract", "") or r.get("summary", "")

            external_tbl.add_row([
                r.get("rank", "-"),
                item_type,
                preview_text(item_id, 18),
                preview_text(source_name, 20),
                preview_text(f"{title} | {excerpt}", 110),
            ])
        print(external_tbl)
        print()

    # Show warning if truncated
    if total_count > max_rows:
        print(f"{Color.WARNING}(Showing {max_rows}/{total_count} results){Color.RESET}")
