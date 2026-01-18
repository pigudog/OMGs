from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from .console_utils import Color
from .time_utils import parse_dt
from prettytable import PrettyTable, ALL
###############################################################################
# 📌 Visualization & Trace (non-functional, observability only)
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
            "  C --> D[Global Guideline RAG]\n"
            "  D --> E[Init Specialist Agents]\n"
            "  E --> F[MDT Discussion Engine]\n"
            "  F --> G[Trial Matching]\n"
            "  G --> H[Final Chair Output]\n"
            "  H --> I[Save Logs]\n"
        )


def print_section(title: str, subtitle: str = ""):
    line = "=" * 78
    if subtitle:
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{title}{Color.RESET}  {Color.OKCYAN}{subtitle}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")
    else:
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{title}{Color.RESET}")
        print(f"{Color.OKBLUE}{Color.BOLD}{line}{Color.RESET}")


def preview_text(x: Any, n: int = 160) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else (s[:n] + "…")


def print_selected_reports_table(context: Dict[str, Dict[str, List[Dict[str, Any]]]], roles: List[str]):
    """PrettyTable summary: selected report counts and newest date per role/type."""
    tbl = PrettyTable(["Role", "Lab(n)", "Lab newest", "Img(n)", "Img newest", "Path(n)", "Path newest", "Mut(n)"])
    tbl.align = "l"

    def _newest_date(lst: List[Dict[str, Any]]):
        dts = []
        for r in lst or []:
            dt = parse_dt(r.get("date")) or parse_dt(r.get("report_date")) or parse_dt(r.get("time"))
            if dt is not None:
                dts.append(dt)
        return (max(dts).strftime("%Y-%m-%d") if dts else "-")

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
            len(muts),
        ])

    print(f"\n{Color.BOLD}{Color.OKBLUE}📌 Selected Reports Overview{Color.RESET}")
    print(tbl)


def print_rag_hits_table(rag_raw: List[Dict[str, Any]], max_rows: int = 8):
    """PrettyTable for RAG hit inspection (doc_id/page/score + snippet preview)."""
    if not rag_raw:
        print(f"{Color.WARNING}⚠ RAG: no evidence found.{Color.RESET}")
        return

    tbl = PrettyTable(["Rank", "Score", "Doc", "Page", "Preview"])
    tbl.align = "l"

    for r in (rag_raw or [])[:max_rows]:
        tbl.add_row([
            r.get("rank"),
            f"{float(r.get('score', 0.0)):.4f}",
            preview_text(r.get("doc_id", ""), 28),
            r.get("page", ""),
            preview_text(r.get("text", ""), 110),
        ])

    print(f"\n{Color.BOLD}{Color.OKBLUE}📚 RAG Hits (Top){Color.RESET}")
    print(tbl)