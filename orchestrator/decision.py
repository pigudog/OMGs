"""Decision-making - Final MDT output generation."""

import re
from typing import Dict, Any, Optional
from core.agent import Agent
from core.config import get_mdt_prompts
from utils.error_handling import safe_agent_call
from utils.console_utils import Color


_FINAL_SECTION_LABELS = (
    "Final Assessment",
    "Core Treatment Strategy",
    "Change Triggers",
)


def _strip_duplicate_final_preamble(text: str) -> str:
    """Drop model preamble before the first structured final section."""
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    match = re.search(r"(?im)^\s*(?:#{1,6}\s*)?Final Assessment\s*:?\s*$", stripped)
    if match and match.start() > 0:
        return stripped[match.start():].strip()
    inline_match = re.search(r"(?i)\bFinal Assessment\s*:", stripped)
    if inline_match and inline_match.start() > 0:
        return stripped[inline_match.start():].strip()
    return stripped


def _normalize_final_section_headings(text: str) -> str:
    normalized = text
    for label in _FINAL_SECTION_LABELS:
        escaped = re.escape(label)
        normalized = re.sub(
            rf"(?im)^\s*(?:#{{1,6}}\s*)?{escaped}\s*:?\s*$",
            f"### {label}\n",
            normalized,
        )
        normalized = re.sub(
            rf"(?i)(?<!# )(?<![#\w]){escaped}\s*:\s*",
            f"\n\n### {label}\n\n",
            normalized,
        )
    normalized = re.sub(r"(?im)^\s*(?:#{1,6}\s*)?References\s*:?\s*$", "## References\n", normalized)
    return normalized


def _split_combined_reference_tags(text: str) -> str:
    """Split model-compressed tags like [@a | x; @b | y] into valid tags."""
    combined_tag_pattern = re.compile(r"\[((?:@[^\]]*?;\s*)+@[^\]]*?)\]")

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = [part.strip() for part in re.split(r";\s*(?=@)", inner) if part.strip()]
        if len(parts) <= 1 or not all(part.startswith("@") for part in parts):
            return match.group(0)
        return " ".join(f"[{part}]" for part in parts)

    return combined_tag_pattern.sub(replace, text)


def _bulletize_final_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"(### {re.escape(heading)}\n+)(.*?)(?=\n### |\n---|\n## References|\Z)",
        re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        body = match.group(2).strip()
        if not body:
            return prefix
        body = re.sub(r"(?<=\.)\s+-\s+", "\n- ", body)
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return prefix
        # Leave already-valid Markdown lists alone.
        if any(re.match(r"^(?:[-*+]|\d+\.)\s+", line) for line in lines):
            return prefix + "\n".join(lines) + "\n"
        bullets = []
        for line in lines:
            if line.startswith("#"):
                bullets.append(line)
            else:
                bullets.append(f"- {line}")
        return prefix + "\n".join(bullets) + "\n"

    return pattern.sub(replace, text)


def normalize_final_output_markdown(final_output: str) -> str:
    """Normalize chair final output into stable Markdown without changing citations."""
    text = _strip_duplicate_final_preamble(final_output)
    if not text:
        return text
    text = _split_combined_reference_tags(text)
    text = _normalize_final_section_headings(text)
    text = _bulletize_final_section(text, "Core Treatment Strategy")
    text = _bulletize_final_section(text, "Change Triggers")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


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


def _latest_assistant_synthesis(all_round_ops: dict) -> str:
    """Return the newest assistant round synthesis captured before final output."""
    if not isinstance(all_round_ops, dict):
        return ""
    for _round_key, round_ops in reversed(list(all_round_ops.items())):
        if not isinstance(round_ops, dict):
            continue
        synthesis = str(round_ops.get("assistant") or "").strip()
        if synthesis and not synthesis.startswith("[Error:"):
            return synthesis
    return ""


def _extract_named_section(text: str, label: str) -> str:
    labels = (
        "Key Knowledge",
        "Decision Stances",
        "Controversies",
        "Missing Info",
        "Working Plan",
        "Debate Impact",
    )
    next_labels = "|".join(re.escape(item) for item in labels if item != label)
    pattern = re.compile(
        rf"(?ims)^\s*(?:#{{1,6}}\s*)?{re.escape(label)}\s*:\s*(.*?)(?=^\s*(?:#{{1,6}}\s*)?(?:{next_labels})\s*:|\Z)"
    )
    match = pattern.search(str(text or ""))
    return match.group(1).strip() if match else ""


def _section_bullets(section: str, *, max_items: int, skip_prefixes: tuple[str, ...] = ()) -> list[str]:
    bullets: list[str] = []
    for raw_line in str(section or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line).strip()
        if not line or line.endswith(":"):
            continue
        if any(line.lower().startswith(prefix.lower()) for prefix in skip_prefixes):
            continue
        bullets.append(line)
        if len(bullets) >= max_items:
            break
    return bullets


def _build_final_output_fallback(
    all_round_ops: dict,
    *,
    merged: Optional[str] = None,
    trial_note: Optional[str] = None,
) -> str:
    """Build a usable final answer from completed MDT synthesis when the final LLM call fails."""
    synthesis = _latest_assistant_synthesis(all_round_ops) or str(merged or "")

    key_items = _section_bullets(_extract_named_section(synthesis, "Key Knowledge"), max_items=3)
    working_plan = _section_bullets(
        _extract_named_section(synthesis, "Working Plan"),
        max_items=4,
        skip_prefixes=("first", "after this information is available"),
    )
    missing_info = _section_bullets(_extract_named_section(synthesis, "Missing Info"), max_items=3)
    controversies = _section_bullets(_extract_named_section(synthesis, "Controversies"), max_items=2)

    if not key_items:
        key_items = ["MDT discussion completed, but final chair synthesis could not be regenerated."]
    if not working_plan:
        working_plan = [
            "Review completed MDT synthesis before finalizing treatment.",
            "Use available evidence and safety data to choose the next management step.",
        ]

    assessment = " ".join(key_items[:2])
    fallback_lines = [
        "Final Assessment:",
        assessment,
        "",
        "Core Treatment Strategy:",
    ]
    fallback_lines.extend(f"- {item}" for item in working_plan[:4])

    fallback_lines.extend(["", "Change Triggers:"])
    if missing_info:
        fallback_lines.append(f"- If missing data become available, update the regimen choice: {missing_info[0]}")
    if controversies:
        fallback_lines.append(f"- If MDT disagreement persists, resolve before treatment commitment: {controversies[0]}")
    if len(fallback_lines) <= 9:
        fallback_lines.append("- If toxicity or disease tempo changes, reassess treatment intensity and goals.")

    if trial_note and trial_note.strip():
        fallback_lines.extend(["", "Clinical Trial Note:", trial_note.strip()])

    return "\n".join(fallback_lines)


def _render_omgs_final_prompt(**values: Any) -> str:
    prompts = get_mdt_prompts().get("chair_modes", {})
    template = (prompts.get("omgs_final", {}) or {}).get("final_prompt")
    if not template:
        raise RuntimeError("Missing OMGs final chair prompt template")
    return str(template).format(**values).strip()


def generate_final_output(
    chair_agent: Agent,
    all_round_ops: dict,
    clinic_time: str = None,
    merged: Optional[str] = None,
    initial_ops: Optional[Dict[str, str]] = None,
    interaction_log: Optional[Dict[str, Any]] = None,
    trial_note: Optional[str] = None,
    trace: Optional[Any] = None,
    ref_tags_str: Optional[str] = None,
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
        ref_tags_str: Optional list of REFERENCE TAGS (guideline/pubmed/nccn) for chair to cite exactly (OMGs mode).
    
    Returns:
        Final MDT output string, or fallback output if generation fails
    """
    # Build explicit compact discussion context. The latest merged MDT memory
    # already incorporates initial views, effective discussion turns, and refined
    # plans, so avoid re-injecting older JSON blocks into the final chair prompt.
    discussion_summary = ""
    if merged:
        discussion_summary += f"# MDT DISCUSSION SUMMARY\n{merged}\n\n"
    
    # Build trial recommendation section if available
    trial_section = ""
    if trial_note and trial_note.strip():
        trial_section = f"# CLINICAL TRIAL RECOMMENDATION (from assistant)\n{trial_note.strip()}\n\n"

    # Inject REFERENCE TAGS when provided (e.g. from OMGs pipeline) so chair cites guideline/pubmed exactly
    ref_tags_block = ""
    if ref_tags_str and ref_tags_str.strip():
        ref_tags_block = (
            "## CITATION (guidelines, literature, reports, trial if recommended)\n\n"
            "### 1. REFERENCE TAGS — copy exactly (do not use [1] or [2])\n"
            f"{ref_tags_str.strip()}\n\n"
            "### 2. FORMAT BY TYPE\n"
            "- Reports: [@actual_report_id | LAB/Genomics/MR/CT/Pathology]  (e.g. [@LAB20251020TM | LAB], [@OH20251003 | Genomics], [@CT20250922 | CT], [@PX20251003 | Pathology])\n"
            "- Guideline: EXACT tag from list above, e.g. [@guideline:doc_id | Page xx] or [@guideline:doc_id | Pages xx-yy]\n"
            "- NCCN: [@guideline:nccn | rule_id]  EXACT from list above\n"
            "- PubMed: [@pubmed | PMID]  EXACT from list above\n"
            "- Trial (if recommended): [@trial | trial_id]  e.g. [@trial | 350]\n\n"
            "### 3. RULES\n"
            "- Use full tag only. [1][2] invalid; will not appear in References.\n"
            "- Report facts: [@report_id | LAB/Genomics/MR/CT/Pathology]. Guideline/NCCN/PubMed: EXACT tag from REFERENCE TAGS.\n"
            "- If trial recommended and you agree: cite [@trial | trial_id].\n"
            "- If experts disagree, pick safest plan and state key uncertainty.\n\n"
        )

    prompt = _render_omgs_final_prompt(
        clinic_time=clinic_time,
        discussion_summary=discussion_summary,
        trial_section=trial_section,
        ref_tags_block=ref_tags_block,
    )
    fallback = _build_final_output_fallback(
        all_round_ops,
        merged=merged,
        trial_note=trial_note,
    )
    return safe_agent_call(
        agent=chair_agent,
        prompt=prompt,
        role="chair",
        stage="final_output",
        fallback=fallback,
        trace=trace,
        max_retries=2,
        use_once=True,
        retry_delay_seconds=10.0,
    )


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
    
    trial_id_match = re.search(r"-\s*id:\s*(\S+)", trial_note)
    trial_name_match = re.search(r"-\s*name:\s*(.+?)(?:\n|$)", trial_note)
    trial_reason_match = re.search(r"-\s*Reason:\s*(.+?)(?:\n|$)", trial_note)
    if not trial_id_match:
        candidate_match = re.search(
            r"-\s*Candidate trial:\s*\[@trial\s*\|\s*([^\]]+)\]\s*(.+?)(?:\n|$)",
            trial_note,
            re.IGNORECASE,
        )
        if candidate_match:
            return {
                "trial_id": candidate_match.group(1).strip(),
                "name": candidate_match.group(2).strip(),
                "reason": "",
            }
    
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
    - External Evidence (PubMed, FDA, conferences)
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
    final_output = normalize_final_output_markdown(final_output)
    
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
                cached_trial = cache.get_trial(trial_id)
                trial_info[trial_id] = cached_trial or parsed_trial
                # Store sparse parsed notes only as a fallback. In the live
                # trial path, fit cards may already have richer metadata cached.
                if cached_trial is None:
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
        
        # Build references only from evidence the chair actually cited in the
        # final answer. Trial recommendation fallback above is the only
        # deliberate auto-citation path.
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


def split_references_from_output(final_output: str) -> tuple[str, str]:
    """
    Split a final MDT output into the decision body and generated References.

    The References section is deterministic post-processing, so native live-room
    rendering can stream the decision body while rendering references all at once.
    """
    text = str(final_output or "").strip()
    if not text:
        return "", ""
    match = re.search(r"(?im)^\s*(?:#{1,6}\s*)?References\s*:?\s*$", text)
    if not match:
        return text, ""
    body = text[: match.start()].rstrip()
    references = text[match.start() :].strip()
    divider = re.search(r"(?m)(?:^|\n)\s*---\s*$", body)
    if divider and not body[divider.end() :].strip():
        body = body[: divider.start()].rstrip()
    return body, references
