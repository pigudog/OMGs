"""Decision-making - Final MDT output generation and trial matching."""

import json
from typing import Dict, Any, Optional, List
from core.agent import Agent, AgentError
from utils.console_utils import normalize_trial_compact, Color


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
    interaction_log: Optional[Dict[str, Any]] = None,
    trial_note: Optional[str] = None,
    trace: Optional[Any] = None
) -> str:
    """Generate final MDT decision output from Chair agent.
    
    Args:
        chair_agent: Chair agent instance
        all_round_ops: Final refined plans from all experts in all rounds
        clinic_time: Visit timestamp
        merged: MDT discussion summary (key knowledge, controversies, etc.)
        initial_ops: Initial opinions from all experts
        interaction_log: Full interaction log of MDT discussions
        trial_note: Clinical trial recommendation from assistant (if any)
        trace: Optional TraceLogger for error tracking
    
    Returns:
        Final MDT output string, or fallback output if generation fails
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
    
    # Build trial recommendation section if available
    trial_section = ""
    if trial_note and trial_note.strip():
        trial_section = f"# CLINICAL TRIAL RECOMMENDATION (from assistant)\n{trial_note.strip()}\n\n"

    prompt = f"""
As the MDT chair for gynecologic oncology, you are seeing the patient at OUTPATIENT TIME: {clinic_time}.
Based on PATIENT FACTS + MDT discussion + FINAL refined plans from all experts, determine the CURRENT best management plan for this visit.

{discussion_summary}

# FINAL REFINED PLANS (All experts, all rounds)
{expert_final}

{trial_section}STRICT RULES:
- Any factual statement about past tests/treatments must include [@actual_report_id | LAB/Genomics/MR/CT] using actual report_id from report data (e.g., [@20220407|17300673 | LAB], [@OH2203828|2022-04-18 | Genomics], [@2022-12-29 | MR], [@2022-12-29 | CT]). Note: Always use spaces around | for consistency: [@xxx | yyy]. or say unknown.
- Any statement derived from guideline or PubMed literature must include [@guideline:doc_id | Page xx] or [@pubmed | PMID].
- If you cite guideline/PubMed evidence in Core Treatment Strategy or Change Triggers, include at least one tag in that bullet.
- If a clinical trial has been recommended by the assistant and you judge it appropriate for the patient, mention it naturally within Core Treatment Strategy or Change Triggers and cite it using [@trial | trial_id] format (e.g., [@trial | 350]).
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
    try:
        return chair_agent.chat(prompt)
    except AgentError as e:
        # Fallback: generate simplified output from final_round_ops
        print(f"{Color.WARNING}[WARNING] Chair final output generation failed: {e.original_error}{Color.RESET}")
        if trace:
            trace.emit("agent_error", {
                "role": "chair",
                "stage": "final_output",
                "error": str(e.original_error),
                "error_type": type(e.original_error).__name__,
                "fallback_used": True
            })
        
        # Build fallback output from expert plans
        fallback_lines = ["Final Assessment:", "Based on MDT discussion, treatment plan needs to be determined."]
        fallback_lines.append("\nCore Treatment Strategy:")
        
        # Extract plans from final_round_ops
        for round_key, round_ops in all_round_ops.items():
            if isinstance(round_ops, dict):
                for role, plan in round_ops.items():
                    if plan and not plan.startswith("[Error:"):
                        fallback_lines.append(f"- {role}: {plan[:100]}")
        
        if len(fallback_lines) == 3:  # Only header lines
            fallback_lines.append("- Treatment plan to be determined based on available expert opinions")
        
        fallback_lines.append("\nChange Triggers:")
        fallback_lines.append("- Monitor patient response and adjust as needed")
        
        return "\n".join(fallback_lines)
    except Exception as e:
        # Unexpected exception
        print(f"{Color.FAIL}[ERROR] Chair final output unexpected error: {e}{Color.RESET}")
        if trace:
            trace.emit("agent_error", {
                "role": "chair",
                "stage": "final_output",
                "error": str(e),
                "error_type": "UnexpectedException",
                "fallback_used": True
            })
        
        # Minimal fallback
        return f"""Final Assessment:
Unable to generate full assessment due to system error.

Core Treatment Strategy:
- Review expert opinions in MDT discussion
- Determine treatment plan based on available evidence

Change Triggers:
- Monitor patient condition and adjust management as needed"""


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
                # Check if chair naturally cited the trial; if not, add as fallback
                trial_tag = f"[@trial | {trial_id}]"
                if trial_tag.lower() not in final_output.lower():
                    # Fallback: chair should have cited this naturally per prompt instruction
                    print(f"[INFO] Trial {trial_id} was recommended but not cited by chair - adding tag as fallback")
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
        parts.append("⚠️ COMPREHENSIVE NGS PANEL (~20,000 genes) - INTERPRETATION RULES:")
        parts.append("• '未检出' (not detected) = NO pathogenic mutation found")
        parts.append("• '（视为阴性）' (considered negative) = NO pathogenic mutation found")
        parts.append("• '阴性' (negative) = negative result")
        parts.append("• Genes with specific variants (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation")
        parts.append("• If a gene is NOT mentioned, it means NO pathogenic mutation (comprehensive panel)")
        parts.append("")
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
    
    # Use safe_agent_call with retry for rate limit errors (429)
    from utils.error_handling import safe_agent_call
    
    answer = safe_agent_call(
        agent=agent,
        prompt=prompt,
        role="trial_selector",
        stage="trial_matching",
        fallback="Trial Recommendation:\n- id: None\n- name: None\n- Reason: Unable to process trial matching due to API error\n- Missing eligibility confirmations (0-2 items):\n  - None",
        trace=None,  # Trial matching doesn't use trace logger
        max_retries=2  # Retry up to 3 times for rate limits (429 errors) with exponential backoff
    ).strip()
    return answer
