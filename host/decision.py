"""Decision-making - Final MDT output generation and trial matching."""

import json
from typing import Dict, Any, Optional, List
from core.agent import Agent
from utils.console_utils import normalize_trial_compact


def _build_discussion_summary(interaction_log: Dict[str, Any], max_rounds: int = 2) -> str:
    """Build a compact summary of MDT discussions from interaction log."""
    if not interaction_log:
        return "No direct discussions occurred."
    
    summary_parts = []
    for round_key in sorted(interaction_log.keys())[-max_rounds:]:
        round_num = round_key.replace("Round ", "")
        round_msgs = []
        for turn_key in sorted(interaction_log[round_key].keys()):
            turn_num = turn_key.replace("Turn ", "")
            for src_role in interaction_log[round_key][turn_key]:
                for dst_role in interaction_log[round_key][turn_key][src_role]:
                    msg = interaction_log[round_key][turn_key][src_role].get(dst_role)
                    if msg:
                        round_msgs.append(f"R{round_num}T{turn_num}: {src_role}→{dst_role}: {msg[:100]}")
        if round_msgs:
            summary_parts.append(f"Round {round_num}: {' | '.join(round_msgs[:5])}")  # Limit to 5 messages per round
    
    return "\n".join(summary_parts) if summary_parts else "No direct discussions occurred."


def generate_final_output(
    chair_agent: Agent,
    all_round_ops: dict,
    clinic_time: str = None,
    merged: Optional[str] = None,
    initial_ops: Optional[Dict[str, str]] = None,
    interaction_log: Optional[Dict[str, Any]] = None
) -> str:
    """Generate final MDT decision output from Chair agent.
    
    Args:
        chair_agent: Chair agent instance
        all_round_ops: Final refined plans from all experts in all rounds
        clinic_time: Visit timestamp
        merged: MDT discussion summary (key knowledge, controversies, etc.)
        initial_ops: Initial opinions from all experts
        interaction_log: Full interaction log of MDT discussions
    """
    expert_final = json.dumps(all_round_ops, ensure_ascii=False, indent=2)
    
    # Build discussion context
    discussion_summary = ""
    if merged:
        discussion_summary += f"# MDT DISCUSSION SUMMARY\n{merged}\n\n"
    
    if initial_ops:
        initial_summary = "\n".join([f"- {role}: {op[:200]}" for role, op in initial_ops.items()])
        discussion_summary += f"# INITIAL EXPERT OPINIONS\n{initial_summary}\n\n"
    
    if interaction_log:
        interaction_summary = _build_discussion_summary(interaction_log)
        discussion_summary += f"# DISCUSSION INTERACTIONS\n{interaction_summary}\n\n"

    prompt = f"""
As the MDT chair for gynecologic oncology, you are seeing the patient at OUTPATIENT TIME: {clinic_time}.
Based on PATIENT FACTS + MDT discussion + FINAL refined plans from all experts, determine the CURRENT best management plan for this visit.

{discussion_summary}

# FINAL REFINED PLANS (All experts, all rounds)
{expert_final}

STRICT RULES:
- Any factual statement about past tests/treatments must include [@report_id|date] or say unknown.
- Any statement derived from guideline or PubMed literature must include [@guideline:doc_id|page] or [@pubmed:PMID].
- If you cite guideline/PubMed evidence in Core Treatment Strategy or Change Triggers, include at least one tag in that bullet.
- If experts disagree, pick the safest plan and state the key uncertainty.
- You MUST consider the MDT discussion summary and interactions above when making your decision.

# Response Format
Final Assessment:
<1–3 sentences: summarize histology/biology, current disease status, and key uncertainties>

Core Treatment Strategy:
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >

Change Triggers:
- < ≤20 words "if X, then adjust management from A to B" >
- < ≤20 words "if X, then adjust management from A to B" >
"""
    return chair_agent.chat(prompt)


###############################################################################
# Post-processing: Append References Section
###############################################################################

def parse_trial_from_note(trial_note: str) -> Optional[Dict[str, Any]]:
    """
    Parse trial recommendation from trial_note text.
    
    Expected format:
    Trial Recommendation:
    - id: <trial_id>
    - name: <trial_name>
    - Reason: <reason>
    
    Returns:
        Dict with trial_id, name, reason or None if not found/None
    """
    if not trial_note or "None" in trial_note.split("id:")[-1].split("\n")[0]:
        return None
    
    import re
    
    trial_id_match = re.search(r"-\s*id:\s*(\S+)", trial_note)
    trial_name_match = re.search(r"-\s*name:\s*(.+?)(?:\n|$)", trial_note)
    trial_reason_match = re.search(r"-\s*Reason:\s*(.+?)(?:\n|$)", trial_note)
    
    if not trial_id_match:
        return None
    
    trial_id = trial_id_match.group(1).strip()
    if trial_id.lower() == "none":
        return None
    
    return {
        "trial_id": trial_id,
        "name": trial_name_match.group(1).strip() if trial_name_match else "",
        "reason": trial_reason_match.group(1).strip() if trial_reason_match else "",
    }


def append_references_to_output(
    final_output: str,
    trial_note: str = "",
    report_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Post-process final MDT output to append a References section.
    
    Extracts all evidence tags from the output, looks up their details,
    and appends a formatted References block organized by type:
    - Guidelines
    - Literature (PubMed)
    - Clinical Trials
    - Clinical Reports
    
    Args:
        final_output: Raw final MDT output containing inline evidence tags
        trial_note: Trial recommendation text (to extract trial info)
        report_context: Dict with report data for lookup
    
    Returns:
        Enhanced output with References section appended
    """
    if not final_output:
        return final_output
    
    try:
        from utils.reference_cache import build_references_section, get_reference_cache
        
        # Get the reference cache (should already have RAG results stored)
        cache = get_reference_cache()
        
        # Parse trial info from trial_note and store in cache
        trial_info = {}
        if trial_note:
            parsed_trial = parse_trial_from_note(trial_note)
            if parsed_trial:
                trial_id = parsed_trial["trial_id"]
                trial_info[trial_id] = parsed_trial
                # Store in cache for References section lookup
                cache.store_trial(
                    trial_id=trial_id,
                    name=parsed_trial.get("name", ""),
                    reason=parsed_trial.get("reason", ""),
                )
                # Silently add trial tag for References (will appear in Clinical Trials section)
                # Don't add explicit "Recommended Trial:" line - let it be natural
                trial_tag = f"[@trial:{trial_id}]"
                if trial_tag.lower() not in final_output.lower():
                    # Just inject the tag at end so References picks it up
                    final_output = final_output.strip() + f" {trial_tag}"
        
        # Build the references section
        refs_section = build_references_section(
            final_output,
            cache=cache,
            trial_info=trial_info,
            report_context=report_context,
        )
        
        if refs_section:
            return final_output.strip() + "\n" + refs_section
        
        return final_output
    
    except Exception as e:
        # If anything fails, return original output without modification
        # This ensures the pipeline doesn't break due to reference formatting
        print(f"[WARNING] Failed to append references: {e}")
        return final_output


###############################################################################
# Build Enhanced Case Information for Trial Matching
###############################################################################
def build_enhanced_case_for_trial(
    case_json: Dict[str, Any],
    path_reports: List[Dict[str, Any]],
    mut_reports: List[Dict[str, Any]],
    question_str: str
) -> str:
    """
    Build enhanced case information string for trial matching by combining:
    - Original case JSON
    - Pathology reports (histology, diagnosis, IHC, molecular markers)
    - Mutation reports (raw_text with full mutation/genetic information)
    
    Args:
        case_json: Structured case JSON dictionary
        path_reports: List of pathology report dictionaries
        mut_reports: List of mutation report dictionaries
        question_str: Original case JSON string
    
    Returns:
        Enhanced case information string combining all sources
    """
    parts = []
    
    # Start with original case JSON
    parts.append("# ORIGINAL CASE JSON")
    parts.append(question_str)
    parts.append("")
    
    # Add pathology reports information
    if path_reports:
        parts.append("# PATHOLOGY REPORTS (Key Information for Trial Matching)")
        for i, path_rpt in enumerate(path_reports, 1):
            report_id = path_rpt.get("report_id", "Unknown")
            report_date = path_rpt.get("report_date", "")
            date_str = str(report_date)[:10] if report_date else "Unknown"
            
            path_info = []
            if path_rpt.get("histology"):
                path_info.append(f"Histology: {path_rpt.get('histology')}")
            if path_rpt.get("diagnosis"):
                path_info.append(f"Diagnosis: {path_rpt.get('diagnosis')}")
            if path_rpt.get("summary"):
                summary = path_rpt.get("summary", "")
                # Include full summary if available (may contain IHC, molecular info)
                path_info.append(f"Summary: {summary}")
            
            # Include other molecular/IHC fields if present
            for key in ["IHC", "molecular", "biomarkers", "grade", "stage"]:
                if path_rpt.get(key):
                    path_info.append(f"{key}: {path_rpt.get(key)}")
            
            if path_info:
                parts.append(f"## Pathology Report {i} (ID: {report_id}, Date: {date_str})")
                parts.append("\n".join(path_info))
                parts.append("")
    
    # Add mutation reports information (full raw_text is critical)
    if mut_reports:
        parts.append("# MUTATION / GENETIC REPORTS (Full Text for Trial Matching)")
        for i, mut_rpt in enumerate(mut_reports, 1):
            report_id = mut_rpt.get("report_id", "Unknown")
            report_date = mut_rpt.get("report_date", "")
            date_str = str(report_date)[:10] if report_date else "Unknown"
            raw_text = mut_rpt.get("raw_text", "")
            
            parts.append(f"## Mutation Report {i} (ID: {report_id}, Date: {date_str})")
            if raw_text:
                parts.append(raw_text)
            else:
                parts.append("(No raw_text available)")
            parts.append("")
    
    return "\n".join(parts)


###############################################################################
# Assistant Trial Suggestion
###############################################################################
def assistant_trial_suggestion(agent, case_json_str: str, trials_list: List[Dict[str, Any]]) -> str:
    """
    Generate clinical trial recommendation based on patient case.
    
    Args:
        agent: Agent instance for running the trial matching
        case_json_str: Enhanced case information string
        trials_list: List of available clinical trials
    
    Returns:
        Trial recommendation string
    """
    # Limit trials to prevent excessive token usage
    max_trials = 10
    trials_to_include = (trials_list or [])[:max_trials]
    
    prompt = f"""
You are an MDT assistant for gynecologic oncology clinical trial matching.

CRITICAL BEHAVIOR:
- You MUST NOT ask the user any questions.
- You MUST NOT request additional information.
- You MUST NOT output anything except the required template.
- Use ONLY the provided PATIENT CASE text and AVAILABLE TRIALS list.
- If eligibility is unclear due to missing key facts, you MUST output None.

PATIENT CASE (facts only; do not infer):
{case_json_str}

AVAILABLE TRIALS (compact; use id/name exactly as shown):
{json.dumps([
    normalize_trial_compact(t)
    for t in trials_to_include
], ensure_ascii=False, indent=2)}

DECISION RULE (be conservative):
Recommend ONE trial ONLY IF ALL are true:
1) Cancer type / primary site clearly matches.
2) Disease setting clearly matches (e.g., recurrent/advanced/metastatic and line is not fundamentally unclear).
3) Required biomarker/subtype is explicitly present in case text (if trial requires it).
4) No more than 2 critical eligibility confirmations remain.

If ANY of the above is not satisfied -> output None.

OUTPUT TEMPLATE (EXACT; no extra text):

Trial Recommendation:
- id: <trial id or None>
- name: <trial name or None>
- Reason: <1 short sentence>
- Missing eligibility confirmations (0-2 items):
  - item1 (or "None")
  - item2
""".strip()
    
    # Debug: log prompt size for troubleshooting
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(encoder.encode(prompt))
        case_tokens = len(encoder.encode(case_json_str))
        print(f"[Trial Matching] Prompt tokens: {prompt_tokens}, Patient case tokens: {case_tokens}, Trials: {len(trials_to_include)}")
    except Exception:
        pass
    
    answer = agent.chat(prompt).strip()
    return answer
