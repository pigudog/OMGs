from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import os
import json
import html as _html
from datetime import datetime, timedelta
from utils.console_utils import Color
from utils.time_utils import parse_dt
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
            "  C --> D[Guideline+PubMed RAG]\n"
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


_EVIDENCE_TAG_RE = re.compile(r"\[@(?:guideline|pubmed):", re.IGNORECASE)
_EVIDENCE_CUES = [
    "guideline",
    "trial",
    "nccn",
    "esmo",
    "parp",
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


_EVIDENCE_TAG_RE = re.compile(r"\[@(?:guideline|pubmed):", re.IGNORECASE)
_EVIDENCE_CUES = [
    "guideline",
    "evidence",
    "trial",
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


###############################################################################
# MDT Reporting Utilities (observability-only)
###############################################################################

###############################################################################
# SAVE FULL MDT LOG TO DISK
###############################################################################
def save_mdt_log(
    question: str,
    final_output: str,
    initial_ops: Dict[str, Any],
    merged: str,
    final_round_ops: Dict[str, Any],
    interaction_log: Dict[str, Any],
    agent_logs: Dict[str, Any],
    log_dir: str = "mdt_logs",
    trace_events: Optional[List[Dict[str, Any]]] = None,
    trace_mermaid: Optional[str] = None,
) -> Dict[str, str]:
    """Save MDT log artifacts to disk.

    Outputs:
    - `mdt_history_<ts>.jsonl`: one JSON line containing full pipeline state
    - `mdt_history_<ts>.md`: human-readable discussion log

    Returns:
    - dict with keys: `ts`, `jsonl_path`, `md_path`

    NOTE: observability-only; must not change inference behavior.
    """
    print(f"{Color.OKBLUE}{Color.BOLD}📁 Saving MDT Logs...{Color.RESET}")

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fname_jsonl = os.path.join(log_dir, f"mdt_history_{ts}.jsonl")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "final_output": final_output,
        "initial_ops": initial_ops,
        "merged_summary": merged,
        "final_round_ops": final_round_ops,
        "interaction_log": interaction_log,
        "agent_logs": agent_logs,
        "trace_events": trace_events or [],
        "trace_mermaid": trace_mermaid or "",
    }
    with open(fname_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    fname_md = os.path.join(log_dir, f"mdt_history_{ts}.md")

    def fmt_block(title, content):
        return f"\n\n## {title}\n\n```\n{content}\n```\n"

    md = []
    md.append(f"# MDT Discussion Log — {ts}\n")
    md.append("Generated automatically by MDT Pipeline.\n\n")

    md.append(fmt_block("CASE (structured JSON)", question))
    md.append(fmt_block("FINAL MDT OUTPUT", final_output))

    if trace_mermaid:
        md.append("\n\n## PIPELINE FLOW (Mermaid)\n\n")
        md.append("```mermaid\n")
        md.append(trace_mermaid.strip() + "\n")
        md.append("```\n")

    if trace_events:
        md.append("\n\n## TRACE EVENTS (Structured)\n\n")
        md.append("```json\n")
        md.append(json.dumps(trace_events, ensure_ascii=False, indent=2) + "\n")
        md.append("```\n")

    md.append("\n\n## INITIAL EXPERT OPINIONS\n")
    for role, text in initial_ops.items():
        md.append(fmt_block(role, text))

    md.append(fmt_block("MERGED SUMMARY (Assistant)", merged))

    md.append("\n\n## INTERACTION LOG (Round × Turn)\n")
    for rnd, turns in interaction_log.items():
        md.append(f"\n### {rnd}\n")
        for turn, speakers in turns.items():
            md.append(f"\n#### {turn}\n")
            for speaker, tg in speakers.items():
                for target, msg in tg.items():
                    if msg:
                        md.append(f"- **{speaker} → {target}:** {msg}\n")

    md.append("\n\n## FINAL REFINED PLANS (per Round)\n")
    for rnd, ops in final_round_ops.items():
        md.append(f"\n### {rnd}\n")
        for role, text in ops.items():
            md.append(fmt_block(role, text))

    md.append("\n\n## RAW AGENT LOGS\n")
    for role, logs in agent_logs.items():
        md.append(f"\n### {role}\n")
        for entry in logs:
            md.append(
                f"- **User:** {entry['user_message']}\n"
                f"  \n  **Assistant:** {entry['assistant_reply']}\n"
                f"  \n  *({entry['timestamp']})*\n\n"
            )

    with open(fname_md, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"{Color.OKGREEN}✔ JSONL saved to: {fname_jsonl}{Color.RESET}")
    print(f"{Color.OKGREEN}✔ Markdown saved to: {fname_md}{Color.RESET}")

    return {
        "ts": ts,
        "jsonl_path": fname_jsonl,
        "md_path": fname_md,
    }


###############################################################################
# ✅ HTML Report (safe, no f-string braces)
###############################################################################

def _html_escape(x: Any) -> str:
    try:
        return _html.escape("" if x is None else str(x))
    except Exception:
        return _html.escape(repr(x))


def _html_pre(x: Any) -> str:
    return f"<pre>{_html_escape(x)}</pre>"


def _newest_date(items: List[Dict[str, Any]]) -> str:
    dts = []
    for r in items or []:
        dt = parse_dt(r.get("date")) or parse_dt(r.get("report_date")) or parse_dt(r.get("time"))
        if dt is not None:
            dts.append(dt)
    return (max(dts).strftime("%Y-%m-%d") if dts else "-")


def _render_selected_reports_table(context: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> str:
    """Render selected report counts + newest dates, including mutation."""
    roles = sorted(set((context.get("lab") or {}).keys())
                   | set((context.get("imaging") or {}).keys())
                   | set((context.get("pathology") or {}).keys())
                   | set((context.get("mutation") or {}).keys()))
    if not roles:
        return "<div class='warn'>No selected reports.</div>"

    header = (
        "<tr><th>Role</th>"
        "<th>Lab (n)</th><th>Lab newest</th>"
        "<th>Imaging (n)</th><th>Imaging newest</th>"
        "<th>Pathology (n)</th><th>Pathology newest</th>"
        "<th>Mutation (n)</th><th>Mutation newest</th></tr>"
    )

    rows: List[str] = []
    for role in roles:
        labs = (context.get("lab", {}) or {}).get(role, [])
        imgs = (context.get("imaging", {}) or {}).get(role, [])
        paths = (context.get("pathology", {}) or {}).get(role, [])
        muts = (context.get("mutation", {}) or {}).get(role, [])
        rows.append(
            "<tr>"
            f"<td class='mono'>{_html_escape(role)}</td>"
            f"<td>{len(labs)}</td><td>{_html_escape(_newest_date(labs))}</td>"
            f"<td>{len(imgs)}</td><td>{_html_escape(_newest_date(imgs))}</td>"
            f"<td>{len(paths)}</td><td>{_html_escape(_newest_date(paths))}</td>"
            f"<td>{len(muts)}</td><td>{_html_escape(_newest_date(muts))}</td>"
            "</tr>"
        )

    return "<table class='grid'>" + header + "".join(rows) + "</table>"


def _render_interaction_table(interaction_log: Dict[str, Any]) -> str:
    """Render interaction_log as a readable Round×Turn table."""
    if not interaction_log:
        return "<div class='warn'>No interaction log captured.</div>"

    rows: List[str] = []
    for rnd_name, rnd in (interaction_log or {}).items():
        if not isinstance(rnd, dict):
            continue
        for turn_name, turn in (rnd or {}).items():
            if not isinstance(turn, dict):
                continue
            for src, dst_map in (turn or {}).items():
                if not isinstance(dst_map, dict):
                    continue
                for dst, msg in (dst_map or {}).items():
                    if not msg:
                        continue
                    rows.append(
                        "<tr>"
                        f"<td class='mono'>{_html_escape(rnd_name)}</td>"
                        f"<td class='mono'>{_html_escape(turn_name)}</td>"
                        f"<td class='mono'>{_html_escape(src)}</td>"
                        f"<td class='mono'>{_html_escape(dst)}</td>"
                        f"<td>{_html_escape(msg)}</td>"
                        "</tr>"
                    )

    if not rows:
        return "<div class='warn'>No expert-to-expert messages were emitted.</div>"

    header = "<tr><th>Round</th><th>Turn</th><th>From</th><th>To</th><th>Message</th></tr>"
    return "<table class='grid'>" + header + "".join(rows) + "</table>"


def save_case_html_report(
    log_dir: str,
    ts: str,
    question_str: str,
    final_output: str,
    context: Dict[str, Dict[str, List[Dict[str, Any]]]],
    rag_query: str,
    rag_pack: str,
    rag_raw: List[Dict[str, Any]],
    global_guideline_digest: str,
    interaction_log: Dict[str, Any],
    question_raw: str,
    trial_note: str = "",
    initial_ops: Optional[Dict[str, Any]] = None,
    final_round_ops: Optional[Dict[str, Any]] = None,
    trace_events: Optional[List[Dict[str, Any]]] = None,
    trace_mermaid: Optional[str] = None,
    # Reserved for future: role ordering in the HTML layout (currently unused).
    roles_order: Optional[List[str]] = None,
) -> str:
    """Write a standalone per-case HTML report.

    Content includes:
    - final MDT output
    - raw question + structured case JSON
    - expert debate (Round×Turn)
    - selected clinical context + RAG pack
    - optional trace events + optional Mermaid flow

    Safety:
    - observability only
    - uses plain triple-quoted template (NO f-string) to avoid brace-related SyntaxError
    """

    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"mdt_report_{ts}.html")

    # Pretty JSON blocks
    try:
        context_pretty = json.dumps(context or {}, ensure_ascii=False, indent=2)
    except Exception:
        context_pretty = str(context)

    try:
        rag_raw_pretty = json.dumps(rag_raw or [], ensure_ascii=False, indent=2)
    except Exception:
        rag_raw_pretty = str(rag_raw)

    try:
        init_ops_pretty = json.dumps(initial_ops or {}, ensure_ascii=False, indent=2)
    except Exception:
        init_ops_pretty = str(initial_ops)

    try:
        final_round_ops_pretty = json.dumps(final_round_ops or {}, ensure_ascii=False, indent=2)
    except Exception:
        final_round_ops_pretty = str(final_round_ops)

    try:
        trace_events_pretty = json.dumps(trace_events or [], ensure_ascii=False, indent=2)
    except Exception:
        trace_events_pretty = str(trace_events)

    mermaid_block = ""
    if trace_mermaid:
        mermaid_block = (
            "<details class='mt' open><summary>Pipeline Flow (Mermaid)</summary>"
            "<pre class='mermaid'>" + _html_escape(trace_mermaid) + "</pre>"
            "</details>"
        )

    interaction_block = _render_interaction_table(interaction_log or {})
    selected_reports_table = _render_selected_reports_table(context or {})

    # NOTE: Do NOT use f-string here.
    template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>MDT Report</title>
  <style>
    :root {
      --border: #e5e7eb;
      --muted: #6b7280;
      --bg: #ffffff;
      --card: #ffffff;
      --soft: #f7f7f8;
      --accent: #2563eb;
    }
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; background: var(--bg); }
    h1 { font-size: 20px; margin: 0 0 12px 0; }
    h2 { font-size: 16px; margin: 0 0 10px 0; }
    h3 { font-size: 14px; margin: 10px 0 6px 0; }
    .meta { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
    .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .card { border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; background: var(--card); }
    details { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: var(--card); }
    summary { cursor: pointer; font-weight: 600; }
    pre { white-space: pre-wrap; word-wrap: break-word; background: var(--soft); padding: 12px; border-radius: 10px; overflow-x: auto; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .mt { margin-top: 12px; }
    .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }
    .warn { color: #b45309; font-weight: 600; }
    table.grid { width: 100%; border-collapse: collapse; font-size: 13px; }
    table.grid th, table.grid td { border: 1px solid var(--border); padding: 8px; vertical-align: top; }
    table.grid th { background: #f3f4f6; text-align: left; }
  </style>
</head>
<body>
  <h1>OMGs / MDT Report</h1>
  <div class=\"meta\">
    <div><b>Timestamp</b>: __TS__</div>
  </div>

  <div class=\"cols mt\">
    <div class=\"card\">
      <h2>Final MDT Output</h2>
      __FINAL_OUTPUT__
    </div>
    <div class=\"card\">
      <h2>Question (raw)</h2>
      <div class=\"hint\">Original user question for observability (passed from main via args.question_raw).</div>
      __QUESTION_RAW__

      <details class=\"mt\">
        <summary>Case (Structured JSON)</summary>
        __QUESTION_STRUCT__
      </details>
    </div>
  </div>

  <details class=\"mt\" open>
    <summary>Clinical Trial Recommendation</summary>
    __TRIAL_NOTE__
    <div class=\"hint\">Shown only when the trial matcher produced a recommendation.</div>
  </details>

  <details class=\"mt\" open>
    <summary>Expert Debate (Round × Turn)</summary>
    __INTERACTION_TABLE__
    <div class=\"hint\">Only includes messages that were actually emitted (non-empty).</div>
  </details>

  <details class=\"mt\">
    <summary>Initial Expert Opinions (raw)</summary>
    __INITIAL_OPS__
  </details>

  <details class=\"mt\">
    <summary>Final Refined Plans (per round, raw)</summary>
    __FINAL_ROUND_OPS__
  </details>

  <details class=\"mt\" open>
    <summary>Selected Clinical Context</summary>
    __SELECTED_REPORTS_TABLE__
    <div class=\"hint\">Counts and newest dates per role/report type (including mutation).</div>
    <details class=\"mt\">
      <summary>Raw Context (JSON)</summary>
      __CONTEXT__
    </details>
  </details>

  <details class=\"mt\">
    <summary>Guideline + PubMed RAG</summary>
    <h3>RAG Query</h3>
    __RAG_QUERY__
    <h3>Evidence Pack (top-k)</h3>
    __RAG_PACK__
    <h3>RAG Raw Hits (JSON)</h3>
    __RAG_RAW__
    <h3>Digest (<=8 bullets)</h3>
    __RAG_DIGEST__
  </details>

  __MERMAID_BLOCK__

  <details class=\"mt\">
    <summary>Trace Events (JSON)</summary>
    __TRACE_EVENTS__
  </details>

</body>
</html>
"""

    html_page = template
    html_page = html_page.replace("__TS__", _html_escape(ts))
    html_page = html_page.replace("__FINAL_OUTPUT__", _html_pre(final_output))
    html_page = html_page.replace("__QUESTION_RAW__", _html_pre(question_raw))
    html_page = html_page.replace("__QUESTION_STRUCT__", _html_pre(question_str))
    html_page = html_page.replace("__TRIAL_NOTE__", _html_pre(trial_note or "None"))
    html_page = html_page.replace("__INTERACTION_TABLE__", interaction_block)
    html_page = html_page.replace("__INITIAL_OPS__", _html_pre(init_ops_pretty))
    html_page = html_page.replace("__FINAL_ROUND_OPS__", _html_pre(final_round_ops_pretty))
    html_page = html_page.replace("__SELECTED_REPORTS_TABLE__", selected_reports_table)
    html_page = html_page.replace("__CONTEXT__", _html_pre(context_pretty))
    html_page = html_page.replace("__RAG_QUERY__", _html_pre(rag_query))
    html_page = html_page.replace("__RAG_PACK__", _html_pre(rag_pack))
    html_page = html_page.replace("__RAG_RAW__", _html_pre(rag_raw_pretty))
    html_page = html_page.replace("__RAG_DIGEST__", _html_pre(global_guideline_digest))
    html_page = html_page.replace("__TRACE_EVENTS__", _html_pre(trace_events_pretty))
    html_page = html_page.replace("__MERMAID_BLOCK__", mermaid_block)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_page)

    print(f"{Color.OKGREEN}✔ HTML report saved to: {out_path}{Color.RESET}")
    return out_path
