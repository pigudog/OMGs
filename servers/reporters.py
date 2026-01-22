"""Report generation utilities for MDT pipeline.

This module contains observability-only functions for generating
HTML and Markdown reports from MDT discussion logs.

Exported:
- save_mdt_log(): Save MDT artifacts to JSONL and Markdown files
- save_case_html_report(): Generate standalone HTML report for a case
"""
from __future__ import annotations

import os
import json
import html as _html
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.console_utils import Color
from utils.time_utils import parse_dt, format_duration


###############################################################################
# Helper Functions
###############################################################################

def _newest_date(items: List[Dict[str, Any]]) -> str:
    """Get the newest date from a list of report dicts."""
    dts = []
    for r in items or []:
        dt = parse_dt(r.get("date")) or parse_dt(r.get("report_date")) or parse_dt(r.get("time"))
        if dt is not None:
            dts.append(dt)
    return (max(dts).strftime("%Y-%m-%d") if dts else "-")


def _html_escape(x: Any) -> str:
    """Escape HTML special characters."""
    try:
        return _html.escape("" if x is None else str(x))
    except Exception:
        return _html.escape(repr(x))


def _html_pre(x: Any) -> str:
    """Wrap content in HTML pre tag with escaping."""
    return f"<pre>{_html_escape(x)}</pre>"


def _render_final_output_html(final_output: str) -> str:
    """
    Render final MDT output with styled References section.
    
    If the output contains a References section (marked by '---' and '## References'),
    it will be rendered with categorized subsections and better styling.
    """
    if not final_output:
        return "<pre>(No output)</pre>"
    
    # Look for References section marker
    refs_marker = "\n---\n## References"
    if refs_marker not in final_output:
        # No references section, render as plain pre
        return _html_pre(final_output)
    
    # Split into main content and references
    parts = final_output.split(refs_marker, 1)
    main_content = parts[0].strip()
    refs_content = parts[1].strip() if len(parts) > 1 else ""
    
    # Build HTML
    html_parts = []
    
    # Main content in pre block
    html_parts.append(f"<pre>{_html_escape(main_content)}</pre>")
    
    # References in styled section
    if refs_content:
        html_parts.append("<div class='refs-section mt'>")
        html_parts.append("<h3 class='refs-title'>References</h3>")
        
        # Parse by category sections
        categories = _parse_reference_categories(refs_content)
        
        # Define fixed order for categories (matching README format)
        category_order = ["Guidelines", "Literature", "Clinical Trials", "Clinical Reports"]
        
        # Category icon mapping
        icons = {
            "Guidelines": "📋",
            "Literature": "📚",
            "Clinical Trials": "🔬",
            "Clinical Reports": "📄",
        }
        
        # Render categories in fixed order, even if empty
        for cat_name in category_order:
            cat_entries = categories.get(cat_name, [])
            icon = icons.get(cat_name, "📎")
            css_class = cat_name.lower().replace(" ", "-")
            
            html_parts.append(f"<div class='ref-category ref-{css_class}'>")
            html_parts.append(f"<div class='ref-cat-header'>{icon} {_html_escape(cat_name)}</div>")
            
            if cat_entries:
                for entry in cat_entries:
                    tag = entry.get("tag", "")
                    details = entry.get("details", [])
                    
                    html_parts.append("<div class='ref-entry'>")
                    html_parts.append(f"<div class='ref-tag'>{_html_escape(tag)}</div>")
                    if details:
                        html_parts.append("<div class='ref-details'>")
                        for detail in details:
                            html_parts.append(f"<div class='ref-detail-line'>{_html_escape(detail)}</div>")
                        html_parts.append("</div>")
                    html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    return "\n".join(html_parts)


def _parse_reference_categories(refs_content: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse references content into categorized entries."""
    categories = {
        "Guidelines": [],
        "Literature": [],
        "Clinical Trials": [],
        "Clinical Reports": [],
    }
    
    current_category = None
    current_entry = None
    
    for line in refs_content.split("\n"):
        line_stripped = line.strip()
        
        # Check for category header
        if line_stripped.startswith("### "):
            # Save previous entry before switching category
            if current_entry and current_category:
                categories[current_category].append(current_entry)
                current_entry = None
            cat_name = line_stripped[4:].strip()
            if cat_name in categories:
                current_category = cat_name
            continue
        
        # Check for new reference entry (starts with [@)
        if line_stripped.startswith("[@"):
            # Save previous entry
            if current_entry and current_category:
                categories[current_category].append(current_entry)
            # Start new entry
            current_entry = {"tag": line_stripped, "details": []}
        elif line_stripped and current_entry:
            # Add detail line (remove leading spaces for cleaner display)
            detail = line_stripped.lstrip()
            if detail:
                current_entry["details"].append(detail)
    
    # Save last entry
    if current_entry and current_category:
        categories[current_category].append(current_entry)
    
    return categories


###############################################################################
# HTML Rendering Helpers
###############################################################################

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


def _render_pipeline_stats_html(stats: Optional[Dict[str, Any]]) -> str:
    """
    Render pipeline execution statistics as HTML.
    
    Args:
        stats: Dictionary containing pipeline statistics (from collect_pipeline_stats)
    
    Returns:
        HTML string for statistics card, or empty string if stats unavailable
    """
    if not stats:
        return ""
    
    html_parts = []
    html_parts.append("<div class='card stats-card' style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 2px solid #0ea5e9; margin-bottom: 16px;'>")
    html_parts.append("<h2 style='margin-top: 0; color: #0369a1;'>📊 Pipeline Execution Statistics</h2>")
    
    # Agent mode (new)
    agent_mode = stats.get("agent_mode", "")
    if agent_mode:
        html_parts.append("<div style='margin-bottom: 12px;'>")
        html_parts.append(f"<strong>🎯 Mode:</strong> <code style='background: #e0f2fe; padding: 2px 8px; border-radius: 4px; font-weight: 600;'>{_html_escape(agent_mode)}</code>")
        html_parts.append("</div>")
    
    # Total execution time
    total_seconds = stats.get("total_seconds", 0)
    if total_seconds > 0:
        human_readable = format_duration(total_seconds)
        html_parts.append("<div style='margin-bottom: 12px;'>")
        html_parts.append(f"<strong>⏱️ Total Execution Time:</strong> {total_seconds:.1f} seconds ({human_readable})")
        html_parts.append("</div>")
    
    # Total tokens
    total_input = stats.get("total_input_tokens", 0)
    total_output = stats.get("total_output_tokens", 0)
    total_tokens = stats.get("total_tokens", 0)
    
    if total_tokens > 0:
        html_parts.append("<div style='margin-bottom: 12px;'>")
        html_parts.append("<strong>🔢 Total Tokens:</strong>")
        html_parts.append("<ul style='margin: 6px 0 0 20px; padding: 0;'>")
        html_parts.append(f"<li>Input: {total_input:,}</li>")
        html_parts.append(f"<li>Output: {total_output:,}</li>")
        html_parts.append(f"<li>Total: {total_tokens:,}</li>")
        html_parts.append("</ul>")
        html_parts.append("</div>")
    
    # Models used
    model_stats = stats.get("model_stats", [])
    provider = stats.get("provider", "")
    if model_stats:
        html_parts.append("<div style='margin-bottom: 8px;'>")
        html_parts.append("<strong>🤖 Models Used:</strong>")
        html_parts.append("<ul style='margin: 6px 0 0 20px; padding: 0;'>")
        for model_stat in model_stats:
            model = model_stat.get("model", "unknown")
            call_count = model_stat.get("call_count", 0)
            input_tokens = model_stat.get("input_tokens", 0)
            output_tokens = model_stat.get("output_tokens", 0)
            model_total = model_stat.get("total_tokens", 0)
            
            # Format model name with provider if available
            model_display = _html_escape(model)
            if provider:
                provider_capitalized = provider.capitalize()
                model_display = f"{model_display} ({provider_capitalized})"
            
            html_parts.append("<li style='margin-bottom: 6px;'>")
            html_parts.append(f"<strong>{model_display}</strong>: ")
            html_parts.append(f"{call_count} call{'s' if call_count != 1 else ''}, ")
            html_parts.append(f"{model_total:,} tokens ")
            html_parts.append(f"({input_tokens:,} in / {output_tokens:,} out)")
            html_parts.append("</li>")
        html_parts.append("</ul>")
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    return "\n".join(html_parts)


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
# HTML Report Generation
###############################################################################

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
    pipeline_stats: Optional[Dict[str, Any]] = None,
    merged_summary: Optional[str] = None,
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
        # Use proper mermaid container for rendering (not escaped, mermaid needs raw syntax)
        mermaid_block = (
            "<details class='mt mermaid-container' open>"
            "<summary>Pipeline Flow</summary>"
            "<div class='mermaid-chart'>" + trace_mermaid.strip() + "</div>"
            "</details>"
        )

    interaction_block = _render_interaction_table(interaction_log or {})
    selected_reports_table = _render_selected_reports_table(context or {})
    pipeline_stats_html = _render_pipeline_stats_html(pipeline_stats)

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
    /* References section styling */
    .refs-section { border: 1px solid var(--border); border-radius: 12px; padding: 16px; background: var(--card); }
    .refs-title { font-size: 16px; font-weight: 700; color: #1f2937; margin: 0 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid var(--accent); }
    .ref-category { margin-bottom: 16px; }
    .ref-cat-header { font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 10px; padding: 6px 10px; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 6px; }
    .ref-entry { margin-bottom: 10px; padding: 10px 12px; background: #fafafa; border-radius: 8px; border-left: 3px solid var(--accent); }
    .ref-tag { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; font-weight: 600; color: var(--accent); margin-bottom: 6px; word-break: break-all; }
    .ref-details { font-size: 13px; color: #4b5563; line-height: 1.6; }
    .ref-detail-line { margin-bottom: 2px; }
    /* Category-specific colors */
    .ref-guidelines .ref-entry { border-left-color: #059669; }
    .ref-guidelines .ref-tag { color: #059669; }
    .ref-literature .ref-entry { border-left-color: #7c3aed; }
    .ref-literature .ref-tag { color: #7c3aed; }
    .ref-clinical-trials .ref-entry { border-left-color: #dc2626; }
    .ref-clinical-trials .ref-tag { color: #dc2626; }
    .ref-clinical-reports .ref-entry { border-left-color: #d97706; }
    .ref-clinical-reports .ref-tag { color: #d97706; }
    /* Mermaid chart styling */
    .mermaid-container { border-color: #10b981; }
    .mermaid-container > summary { color: #059669; }
    .mermaid-chart { padding: 16px; background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border-radius: 10px; margin-top: 10px; overflow-x: auto; }
    .mermaid-chart svg { max-width: 100%; height: auto; }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :root {
        --border: #374151;
        --muted: #9ca3af;
        --bg: #111827;
        --card: #1f2937;
        --soft: #374151;
        --accent: #60a5fa;
      }
      body { color: #f3f4f6; }
      h1, h2, h3 { color: #f9fafb; }
      table.grid th { background: #374151; color: #f3f4f6; }
      .refs-title { color: #f9fafb; }
      .ref-cat-header { background: linear-gradient(135deg, #374151 0%, #4b5563 100%); color: #f3f4f6; }
      .ref-entry { background: #374151; }
      .ref-details { color: #d1d5db; }
      .mermaid-chart { background: linear-gradient(135deg, #1f2937 0%, #374151 100%); }
      .stats-card { background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%) !important; border-color: #0ea5e9 !important; }
      .stats-card h2 { color: #7dd3fc !important; }
    }
    /* Print button */
    .print-btn {
      position: fixed; top: 20px; right: 20px; z-index: 1000;
      padding: 10px 18px; cursor: pointer;
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      color: white; border: none; border-radius: 8px;
      font-size: 14px; font-weight: 600;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
      transition: all 0.2s ease;
    }
    .print-btn:hover { background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); }
    .print-btn:active { transform: translateY(0); }
    /* Print-friendly styles */
    @media print {
      body { margin: 0; background: white; }
      .card, details { border: 1px solid #ccc; page-break-inside: avoid; }
      details[open] > summary ~ * { display: block; }
      .mermaid-chart { page-break-inside: avoid; }
      .stats-card { background: #f0f9ff !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .no-print { display: none !important; }
    }
    /* Responsive layout */
    @media (max-width: 768px) {
      body { margin: 12px; }
      .cols { grid-template-columns: 1fr; }
      table.grid { font-size: 11px; }
      table.grid th, table.grid td { padding: 4px; }
    }
  </style>
</head>
<body>
  <button onclick=\"window.print()\" class=\"print-btn no-print\">🖨️ Print Report</button>
  <h1>OMGs / MDT Report</h1>
  __PIPELINE_STATS__
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
    <summary>📋 MDT Discussion Summary (Assistant)</summary>
    __MERGED_SUMMARY__
    <div class=\"hint\">Structured summary of Key Knowledge, Controversies, Missing Info, and Working Plan.</div>
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
    <h3>Evidence Digest (1:1 per RAG result)</h3>
    __RAG_DIGEST__
  </details>

  __MERMAID_BLOCK__

  <details class=\"mt\">
    <summary>Trace Events (JSON)</summary>
    __TRACE_EVENTS__
  </details>

  <!-- Mermaid.js for flowchart rendering -->
  <script src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>
  <script>
    document.addEventListener('DOMContentLoaded', function() {
      mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        flowchart: {
          useMaxWidth: true,
          htmlLabels: true,
          curve: 'basis'
        },
        securityLevel: 'loose'
      });
      // Re-render mermaid charts in .mermaid-chart containers
      document.querySelectorAll('.mermaid-chart').forEach(function(el) {
        el.classList.add('mermaid');
      });
      mermaid.init(undefined, '.mermaid-chart');
    });
  </script>
</body>
</html>
"""

    html_page = template
    html_page = html_page.replace("__PIPELINE_STATS__", pipeline_stats_html)
    html_page = html_page.replace("__TS__", _html_escape(ts))
    html_page = html_page.replace("__FINAL_OUTPUT__", _render_final_output_html(final_output))
    html_page = html_page.replace("__QUESTION_RAW__", _html_pre(question_raw))
    html_page = html_page.replace("__QUESTION_STRUCT__", _html_pre(question_str))
    html_page = html_page.replace("__TRIAL_NOTE__", _html_pre(trial_note or "None"))
    html_page = html_page.replace("__MERGED_SUMMARY__", _html_pre(merged_summary or "(No summary available)"))
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
