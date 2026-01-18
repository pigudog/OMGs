# utils/omgs_reports.py
"""OMGs MDT reporting utilities.

This module is **observability-only**:
- Saves the full MDT discussion state to disk (JSONL + Markdown)
- Renders a standalone per-case HTML report for review/replay

Safety:
- These functions MUST NOT affect model behavior or clinical decisions.
- They only serialize already-produced artifacts (inputs/outputs/logs/trace).
"""

from __future__ import annotations

import os
import json
import html as _html
from datetime import datetime
from typing import Any, Dict, List, Optional
from .console_utils import Color
from .time_utils import parse_dt

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
    <summary>Global Guideline RAG</summary>
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
