#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_ehr.py (EHR-only, minimal-change hardening)

Key changes:
- Removed scene classification/routing (no longer needed).
- Real retry: retries on API error OR empty output, with exponential backoff + jitter.
- Compact JSONL output (single line, no spaces) for large-scale processing.
- Stream input reading (supports large JSONL) while still showing total progress.
- Patient-id sanitization for safe filenames.
- Separate audit_ehr with token/finish/attempt/error tracking, plus optional JSON repair audit.
- Force English for EHR extraction by default.
- Removed `question` alias from outputs, and `ehr_extracted_text` is now optional (default off).
"""

import os
import json
import time
import argparse
import sys
import random
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from tqdm import tqdm

# ===========================
# Enhanced ANSI colors for CLI
# ===========================
_CLR_RED = "\033[91m"
_CLR_GREEN = "\033[92m"
_CLR_YELLOW = "\033[93m"
_CLR_BLUE = "\033[94m"
_CLR_MAGENTA = "\033[95m"
_CLR_CYAN = "\033[96m"
_CLR_WHITE = "\033[97m"
_CLR_GRAY = "\033[90m"
_CLR_RESET = "\033[0m"
_CLR_BOLD = "\033[1m"
_CLR_DIM = "\033[2m"

# Box drawing characters
_BOX_H = "─"
_BOX_V = "│"
_BOX_TL = "┌"
_BOX_TR = "┐"
_BOX_BL = "└"
_BOX_BR = "┘"
_BOX_T = "├"
_BOX_CHECK = "✓"
_BOX_CROSS = "✗"
_BOX_ARROW = "→"
_BOX_BULLET = "•"


def _sev_color(sev: str) -> str:
    s = (sev or "").lower()
    if s == "critical":
        return _CLR_RED + _CLR_BOLD
    if s == "major":
        return _CLR_YELLOW
    return _CLR_BLUE


def _sev_icon(sev: str) -> str:
    s = (sev or "").lower()
    if s == "critical":
        return "🔴"
    if s == "major":
        return "🟡"
    return "🔵"


def _step_icon(step: str) -> str:
    icons = {
        "extract": "📥",
        "review_self": "🔍",
        "review_validator": "✅",
        "refine": "🔧",
        "re_review": "🔄",
        "auto_fix": "⚙️",
        "write": "💾",
        "done": "✨",
    }
    for key, icon in icons.items():
        if key in step.lower():
            return icon
    return "▶️"


def _print_box(title: str, content: List[str], color: str = _CLR_CYAN):
    """Print a boxed section with title."""
    max_len = max(len(title), max((len(c) for c in content), default=20))
    width = min(max_len + 4, 100)
    print(f"{color}{_BOX_TL}{_BOX_H * (width - 2)}{_BOX_TR}{_CLR_RESET}")
    print(f"{color}{_BOX_V}{_CLR_RESET} {_CLR_BOLD}{title}{_CLR_RESET}{' ' * (width - len(title) - 3)}{color}{_BOX_V}{_CLR_RESET}")
    print(f"{color}{_BOX_T}{_BOX_H * (width - 2)}{_BOX_TR.replace(_BOX_TR, _BOX_T)}{_CLR_RESET}".replace(_BOX_T + _CLR_RESET, _CLR_RESET))
    for line in content:
        truncated = line[:width - 4] + "..." if len(line) > width - 4 else line
        padding = width - len(truncated) - 3
        print(f"{color}{_BOX_V}{_CLR_RESET} {truncated}{' ' * max(0, padding)}{color}{_BOX_V}{_CLR_RESET}")
    print(f"{color}{_BOX_BL}{_BOX_H * (width - 2)}{_BOX_BR}{_CLR_RESET}")


def _print_step(step: str, detail: str = "", status: str = "start"):
    """Print a step with icon and status."""
    icon = _step_icon(step)
    if status == "start":
        print(f"{_CLR_CYAN}{icon} [{step}]{_CLR_RESET} {_CLR_DIM}starting...{_CLR_RESET}")
    elif status == "done":
        print(f"{_CLR_GREEN}{icon} [{step}]{_CLR_RESET} {detail}")
    elif status == "error":
        print(f"{_CLR_RED}{icon} [{step}]{_CLR_RESET} {_CLR_RED}{detail}{_CLR_RESET}")
    else:
        print(f"{_CLR_CYAN}{icon} [{step}]{_CLR_RESET} {detail}")

PROJECT_ROOT = Path(__file__).resolve().parent


# Evaluation labels must not be copied into structured EHR outputs. The
# extractor should emit patient facts only; labels stay in evaluation files.
EVAL_LABEL_FIELDS = {
    "gold_plan",
    "gold_answer",
    "reference_answer",
    "expected_answer",
    "answer",
    "label",
}


def strip_ehr_output_passthrough_fields(row: Dict[str, Any], input_field: str) -> Dict[str, Any]:
    """Return source metadata safe to carry into structured EHR JSONL."""
    out = dict(row)
    for k in [
        "patient_id", "Time",
        "question", "question_raw",
        "ehr_extracted", "ehr_extracted_text",
        "audit_ehr",
        "source_text",
        *sorted(EVAL_LABEL_FIELDS),
    ]:
        out.pop(k, None)
    out.pop(input_field, None)
    return out
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from clients import OpenAIWrapper
from core.client import init_client  # Use unified client initialization
from utils.time_utils import parse_date  # Unified date parsing
from utils.console_utils import safe_parse_json_block


# ===========================
# Logging helpers
# ===========================

def _log(msg: str, *, verbose: bool = False):
    """Print logs only when verbose is enabled."""
    if verbose:
        print(msg)


# ===========================
# Message helpers
# ===========================

def build_messages(system_prompt: str, user_text: str):
    """Build Chat Completions message payload."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def prepend_document_time(src: str, time_val: Any) -> str:
    """Prefix source text with DOCUMENT_TIME if time_val is provided."""
    if not time_val:
        return src
    doc_time = str(time_val).strip()
    if not doc_time:
        return src
    return f"DOCUMENT_TIME: {doc_time}\n\n" + src


# ===========================
# Load prompt.json / scene.json
# ===========================
def load_prompt_config(prompt_path: str) -> Dict[str, Any]:
    data = json.loads(Path(prompt_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "prompts" in data:
        data = data["prompts"]
    if "MASTER_INSTRUCTIONS" not in data or "OUTPUT_SCHEMA" not in data:
        raise RuntimeError(f"{prompt_path} does not conform to schema-driven format")
    return data


def compact_schema_shape(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: compact_schema_shape(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [compact_schema_shape(schema[0])] if schema else []
    return schema


def build_system_prompt(cfg: Dict[str, Any], force_english: bool = False) -> str:
    master = cfg["MASTER_INSTRUCTIONS"].strip()
    schema_shape = compact_schema_shape(cfg["OUTPUT_SCHEMA"])
    final_req = cfg["FINAL_OUTPUT_REQUIREMENT"].strip()

    if force_english:
        master += (
            "\n\nLANGUAGE REQUIREMENT:\n"
            "- All free-text values MUST be in English.\n"
            "- Drugs/regimens/doses remain EXACTLY as source.\n"
        )

    schema_text = json.dumps(schema_shape, ensure_ascii=False)
    return master + "\n\nOUTPUT_SCHEMA_SHAPE:\n" + schema_text + "\n\n" + final_req


def build_review_prompt(cfg: Dict[str, Any]) -> Tuple[str, str]:
    review_instr = (cfg.get("REVIEW_INSTRUCTIONS") or "").strip()
    if not review_instr:
        review_instr = (
            "You are an EHR extraction quality reviewer for gynecologic oncology.\n"
            "Check the extracted JSON against the source text and identify concrete errors, omissions, or contradictions.\n"
            "Only cite issues that are clearly supported by the text. Do NOT invent facts.\n"
            "Return JSON only, matching REVIEW_SCHEMA. No extra text."
        )
    review_schema = cfg.get("REVIEW_SCHEMA") or {
        "issues": [
            {
                "severity": "critical|major|minor",
                "field_path": "string",
                "description": "string",
                "evidence_snippet": "string",
                "suggested_fix": "string",
            }
        ]
    }
    schema_text = json.dumps(review_schema, ensure_ascii=False)
    system_prompt = review_instr + "\n\nREVIEW_SCHEMA:\n" + schema_text
    return system_prompt, schema_text


# ===========================
# API call + retry mechanism
# ===========================
# key function
def call_extract_once(client, deployment, system_prompt, emr_text, *, max_completion_tokens):
    """One-shot call to Azure OpenAI / OpenAI Chat Completions.

    Returns:
        content, req_id, err, tokens_used, finish_reason, tokens_dict
        where tokens_dict = {"input_tokens": int, "output_tokens": int, "total_tokens": int}
    """
    messages = build_messages(system_prompt, emr_text)
    try:
        resp = client.chat_completion(
            model=deployment,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        req_id = getattr(resp, "id", "")
        usage = getattr(resp, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            tokens_used = getattr(usage, "total_tokens", input_tokens + output_tokens)
            tokens_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": tokens_used}
        else:
            tokens_used = 0
            tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        finish = getattr(resp.choices[0], "finish_reason", "")
        return content, req_id, None, tokens_used, finish, tokens_dict
    except Exception as e:
        return "", "", str(e), 0, "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}



def call_with_retry(func, *args, max_retries=4, delay=2, max_delay=30, verbose: bool = False, **kwargs):
    """
    Retry wrapper that works with call_extract_once-style return tuple.
    Retries when:
      - err is not None
      - content is empty (treated as failure)

    Returns:
        content, req_id, err, tokens, finish, errors, attempt, tokens_dict
        where tokens_dict = {"input_tokens": int, "output_tokens": int, "total_tokens": int}
    """
    errors: List[str] = []
    last = ("", "", "retry_failed", 0, "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    for attempt in range(max_retries):
        result = func(*args, **kwargs)
        # Handle both old 5-tuple and new 6-tuple formats
        if len(result) == 5:
            content, req_id, err, tokens, finish = result
            tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": tokens}
        else:
            content, req_id, err, tokens, finish, tokens_dict = result
        last = (content, req_id, err, tokens, finish, tokens_dict)

        if err is None and (content or "").strip():
            return content, req_id, None, tokens, finish, errors, attempt + 1, tokens_dict

        # record error
        if err:
            errors.append(err)
        else:
            errors.append("empty_output")

        # backoff
        sleep_s = min(max_delay, delay * (2 ** attempt)) + random.random()
        _log(f"[retry] attempt={attempt+1}/{max_retries} err={errors[-1]} sleep={sleep_s:.2f}s", verbose=verbose)
        time.sleep(sleep_s)

    # exhausted
    result = last
    if len(result) == 5:
        content, req_id, err, tokens, finish = result
        tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": tokens}
    else:
        content, req_id, err, tokens, finish, tokens_dict = result
    if err is None and not (content or "").strip():
        err = "empty_output"
    return content, req_id, err or "retry_failed", tokens, finish, errors, max_retries, tokens_dict


# ===========================
# JSON parsing / repairing
# ===========================
def try_parse_json(text: str):
    """Parse JSON and return (object, status) tuple.

    Uses safe_parse_json_block from console_utils for robust parsing.
    """
    if not text:
        return None, "empty_output"
    obj = safe_parse_json_block(text)
    if obj:
        return obj, "ok"
    return None, "json_decode_error"


def _parse_date_ymd(x: Any):
    """Parse date string to date object. Uses unified parse_date from time_utils."""
    return parse_date(x)


def _pfis_to_status(pfi_days: int) -> str:
    if pfi_days <= 28:
        return "Refractory"
    if pfi_days < 180:
        return "Resistant"
    return "Sensitive"


# ===========================
# Issue Classification & Refinement
# ===========================

def classify_issue(issue: Dict[str, Any]) -> str:
    """Classify review issue as fixable/ambiguous/truncation.
    
    Args:
        issue: Review issue dict with 'severity', 'field_path', 'description', etc.
    
    Returns:
        'fixable': LLM can fix with feedback (inference errors, wrong values)
        'truncation': Value was cut off (e.g., PLT:34 should be PLT:342)
        'ambiguous': Source text is unclear, needs human review
    """
    desc = (issue.get("description") or "").lower()
    severity = (issue.get("severity") or "").lower()
    
    # Truncation: value cutoff errors
    if "truncat" in desc or re.search(r"\d+\s+instead of\s+\d+", desc):
        return "truncation"
    
    # Fixable: LLM inference errors that can be corrected
    fixable_patterns = [
        "without support",
        "without any support",
        "not mentioned",
        "no .* mentioned",
        "incorrectly",
        "should be",
        "no explicit",
        "not a gene",
        "is not a",
        "no brca",
        "wildtype without",
    ]
    if any(re.search(p, desc) for p in fixable_patterns):
        return "fixable"
    
    # Critical/major issues are more likely fixable
    if severity in ["critical", "major"]:
        # Check if it's about missing evidence
        if any(x in desc for x in ["evidence", "stated", "documented", "mentioned"]):
            return "fixable"
    
    # Default: ambiguous - needs human review
    return "ambiguous"


def get_nested(obj: Dict[str, Any], path: str) -> Any:
    """Get value from nested dict using dot notation path.
    
    Supports array indexing: 'CASE_CORE.GENOMICS.alterations[0].gene'
    """
    parts = re.split(r'\.(?![^\[]*\])', path)  # Split on dots not inside brackets
    current = obj
    for part in parts:
        if current is None:
            return None
        # Handle array indexing
        match = re.match(r'([^\[]+)\[(\d+)\]', part)
        if match:
            key, idx = match.groups()
            current = current.get(key)
            if isinstance(current, list) and int(idx) < len(current):
                current = current[int(idx)]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
    return current


def set_nested(obj: Dict[str, Any], path: str, value: Any) -> None:
    """Set value in nested dict using dot notation path.
    
    Creates intermediate dicts/lists as needed.
    Supports array indexing: 'CASE_CORE.GENOMICS.alterations'
    """
    parts = re.split(r'\.(?![^\[]*\])', path)
    current = obj
    
    for i, part in enumerate(parts[:-1]):
        match = re.match(r'([^\[]+)\[(\d+)\]', part)
        if match:
            key, idx = match.groups()
            idx = int(idx)
            if key not in current:
                current[key] = []
            while len(current[key]) <= idx:
                current[key].append({})
            current = current[key][idx]
        else:
            if part not in current:
                current[part] = {}
            current = current[part]
    
    # Set the final value
    final_part = parts[-1]
    match = re.match(r'([^\[]+)\[(\d+)\]', final_part)
    if match:
        key, idx = match.groups()
        idx = int(idx)
        if key not in current:
            current[key] = []
        while len(current[key]) <= idx:
            current[key].append(None)
        current[key][idx] = value
    else:
        current[final_part] = value


def merge_refinements(original: Dict[str, Any], refinements: Dict[str, Any]) -> Dict[str, Any]:
    """Merge refined fields into original extraction.
    
    Args:
        original: Original extracted JSON
        refinements: Dict of {field_path: corrected_value}
    
    Returns:
        Merged dict with refinements applied
    """
    import copy
    result = copy.deepcopy(original)
    
    for path, value in refinements.items():
        try:
            set_nested(result, path, value)
        except Exception as e:
            print(f"[WARN] Failed to set {path}: {e}")
    
    return result


def should_refine(issues: List[Dict[str, Any]], iteration: int, config: Dict[str, Any]) -> bool:
    """Determine if refinement should be attempted.
    
    Args:
        issues: List of review issues
        iteration: Current iteration count (0-indexed)
        config: Refinement config with max_iterations, min_fixable_issues, issue_severity_threshold
    
    Returns:
        True if should attempt refinement
    """
    max_iterations = config.get("max_iterations", 2)
    min_fixable = config.get("min_fixable_issues", 1)
    severity_threshold = config.get("issue_severity_threshold", "major")
    
    if iteration >= max_iterations:
        return False
    
    # Filter by severity
    severity_order = {"critical": 0, "major": 1, "minor": 2}
    threshold_idx = severity_order.get(severity_threshold, 1)
    
    relevant_issues = [
        i for i in issues
        if severity_order.get((i.get("severity") or "").lower(), 2) <= threshold_idx
    ]
    
    # Count fixable issues
    fixable = [i for i in relevant_issues if classify_issue(i) in ["fixable", "truncation"]]
    return len(fixable) >= min_fixable


def build_refine_prompt(
    extracted_json: Dict[str, Any],
    issues: List[Dict[str, Any]],
    refine_instructions: str,
    source_text: str,
) -> str:
    """Build the refinement prompt for LLM.
    
    Args:
        extracted_json: Previously extracted JSON
        issues: List of issues to fix
        refine_instructions: Refinement instructions from config
        source_text: Original source text for reference
    
    Returns:
        Complete prompt string
    """
    # Filter to fixable/truncation issues only
    fixable_issues = [i for i in issues if classify_issue(i) in ["fixable", "truncation"]]
    
    # Format issues as readable text
    issues_text = "\n".join([
        f"- [{i.get('severity', 'unknown')}] {i.get('field_path', 'unknown')}: {i.get('description', '')}"
        for i in fixable_issues
    ])
    
    # Build the prompt
    prompt = f"""{refine_instructions}

=== PREVIOUSLY EXTRACTED JSON ===
{json.dumps(extracted_json, ensure_ascii=False, indent=2)}

=== ISSUES TO FIX ===
{issues_text}

=== ORIGINAL SOURCE TEXT (for reference) ===
{source_text[:8000]}  

Please provide the corrected fields as JSON."""
    
    return prompt


def call_refine_once(client, deployment: str, refine_prompt: str, *, max_completion_tokens: int):
    """One-shot call to LLM for refinement.

    Returns:
        content, req_id, err, tokens_used, finish_reason, tokens_dict
        where tokens_dict = {"input_tokens": int, "output_tokens": int, "total_tokens": int}
    """
    messages = [
        {"role": "system", "content": "You are an EHR extraction refinement engine. Output only valid JSON."},
        {"role": "user", "content": refine_prompt},
    ]
    try:
        resp = client.chat_completion(
            model=deployment,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        req_id = getattr(resp, "id", "")
        usage = getattr(resp, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            tokens_used = getattr(usage, "total_tokens", input_tokens + output_tokens)
            tokens_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": tokens_used}
        else:
            tokens_used = 0
            tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        finish = getattr(resp.choices[0], "finish_reason", "")
        return content, req_id, None, tokens_used, finish, tokens_dict
    except Exception as e:
        return "", "", str(e), 0, "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def refine_extraction(
    client,
    deployment: str,
    original_json: Dict[str, Any],
    issues: List[Dict[str, Any]],
    source_text: str,
    refine_instructions: str,
    *,
    max_completion_tokens: int = 8000,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Refine extraction based on review issues.
    
    Args:
        client: LLM client
        deployment: Model deployment name
        original_json: Previously extracted JSON
        issues: List of review issues
        source_text: Original source text
        refine_instructions: Refinement instructions from config
        max_completion_tokens: Max tokens for completion
        verbose: Enable verbose logging
    
    Returns:
        Tuple of (refined_json, refinement_audit)
    """
    # Build refinement prompt
    refine_prompt = build_refine_prompt(
        extracted_json=original_json,
        issues=issues,
        refine_instructions=refine_instructions,
        source_text=source_text,
    )
    
    _log(f"[refine] Calling LLM for refinement...", verbose=verbose)

    # Call LLM
    result = call_refine_once(
        client, deployment, refine_prompt, max_completion_tokens=max_completion_tokens
    )
    # Handle both old 5-tuple and new 6-tuple formats
    if len(result) == 5:
        content, req_id, err, tokens, finish = result
        tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": tokens}
    else:
        content, req_id, err, tokens, finish, tokens_dict = result

    audit = {
        "tokens": tokens,
        "tokens_detail": tokens_dict,
        "finish": finish,
        "req_id": req_id,
        "err": err,
    }
    
    if err:
        _log(f"[refine] LLM error: {err}", verbose=verbose)
        return original_json, {**audit, "status": "error", "refinements_applied": {}}
    
    # Parse refinements JSON
    parsed_refinements, parse_status = try_parse_json(content)
    
    if parsed_refinements is None:
        _log(f"[refine] Failed to parse refinements: {parse_status}", verbose=verbose)
        return original_json, {**audit, "status": parse_status, "refinements_applied": {}}
    
    if not isinstance(parsed_refinements, dict):
        _log(f"[refine] Refinements not a dict", verbose=verbose)
        return original_json, {**audit, "status": "invalid_format", "refinements_applied": {}}
    
    # Apply refinements
    refined_json = merge_refinements(original_json, parsed_refinements)
    
    _log(f"[refine] Applied {len(parsed_refinements)} refinements", verbose=verbose)
    
    return refined_json, {
        **audit,
        "status": "ok",
        "refinements_applied": parsed_refinements,
        "fields_refined": list(parsed_refinements.keys()),
    }


def compute_field_confidence(
    field_path: str,
    issues: List[Dict[str, Any]],
    refinement_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute confidence score for a field based on review and refinement history.
    
    Args:
        field_path: Dot-notation path to field
        issues: Final list of review issues
        refinement_history: List of refinement audits
    
    Returns:
        Dict with confidence info
    """
    # Check if field had issues
    field_issues = [i for i in issues if i.get("field_path", "").startswith(field_path)]
    
    # Check if field was refined
    times_refined = 0
    for r in refinement_history:
        if field_path in r.get("fields_refined", []):
            times_refined += 1
    
    # Compute confidence
    if field_issues:
        # Still has issues - low confidence
        max_severity = max([
            {"critical": 0, "major": 1, "minor": 2}.get(i.get("severity", "").lower(), 2)
            for i in field_issues
        ])
        confidence = "low" if max_severity <= 1 else "medium"
    elif times_refined > 0:
        # Was refined but no remaining issues - medium confidence
        confidence = "medium"
    else:
        # No issues, never refined - high confidence
        confidence = "high"
    
    return {
        "confidence": confidence,
        "had_issues": len(field_issues) > 0,
        "times_refined": times_refined,
        "remaining_issues": [i.get("description") for i in field_issues],
    }


def apply_auto_fixes(parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fixes: List[Dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return parsed, fixes

    def record_fix(path: str, before: Any, after: Any, reason: str):
        if before == after:
            return
        fixes.append({
            "path": path,
            "before": before,
            "after": after,
            "reason": reason,
        })

    case_core = parsed.get("CASE_CORE") or {}
    timeline = parsed.get("TIMELINE") or {}

    # PFI math + platinum status
    last_plat = _parse_date_ymd(case_core.get("last_platinum_end_date"))
    first_relapse = _parse_date_ymd(case_core.get("first_relapse_date"))
    if last_plat and first_relapse and first_relapse > last_plat:
        pfi_days = (first_relapse - last_plat).days
        record_fix("CASE_CORE.PFI_days", case_core.get("PFI_days"), str(pfi_days), "recompute_pfi_days")
        case_core["PFI_days"] = str(pfi_days)
        status = _pfis_to_status(pfi_days)
        record_fix("CASE_CORE.PLATINUM_STATUS", case_core.get("PLATINUM_STATUS"), status, "recompute_platinum_status")
        record_fix("CASE_CORE.PLATINUM_STATUS_CURRENT", case_core.get("PLATINUM_STATUS_CURRENT"), status, "recompute_platinum_status_current")
        record_fix("CASE_CORE.PLATINUM_PFI_CURRENT", case_core.get("PLATINUM_PFI_CURRENT"), str(pfi_days), "recompute_platinum_pfi_current")
        case_core["PLATINUM_STATUS"] = status
        case_core["PLATINUM_STATUS_CURRENT"] = status
        case_core["PLATINUM_PFI_CURRENT"] = str(pfi_days)

    # PLATINUM_HISTORY consistency (latest line)
    platinum_hist = case_core.get("PLATINUM_HISTORY") or []
    if isinstance(platinum_hist, list) and platinum_hist:
        dated = []
        for idx, row in enumerate(platinum_hist):
            if row is None or not isinstance(row, dict):
                continue
            end_dt = _parse_date_ymd(row.get("end_date"))
            dated.append((end_dt, idx, row))
        dated.sort(key=lambda x: (x[0] is None, x[0] or datetime.min.date(), x[1]))
        latest = dated[-1][2] if dated else None
        if latest and last_plat and first_relapse and first_relapse > last_plat:
            pfi_days = (first_relapse - last_plat).days
            status = _pfis_to_status(pfi_days)
            if latest is not None:
                record_fix("CASE_CORE.PLATINUM_HISTORY[-1].PFI_days", latest.get("PFI_days"), str(pfi_days), "align_platinum_history_pfi")
                record_fix("CASE_CORE.PLATINUM_HISTORY[-1].status", latest.get("status"), status, "align_platinum_history_status")
                latest["PFI_days"] = str(pfi_days)
                latest["status"] = status

    # Relapse evidence for biochemical relapse
    relapse = case_core.get("RELAPSE_DATE") or {}
    if relapse.get("type") == "Biochemical":
        evidence = relapse.get("evidence") or ""
        if not evidence:
            biomarkers = (case_core.get("BIOMARKERS") or {})
            ca125 = biomarkers.get("CA125")
            if ca125:
                new_evidence = f"CA125 {ca125}"
                record_fix("CASE_CORE.RELAPSE_DATE.evidence", evidence, new_evidence, "fill_relapse_evidence")
                relapse["evidence"] = new_evidence
                case_core["RELAPSE_DATE"] = relapse

    # LINE_OF_THERAPY: compute pfs_days if end_date and next line start_date available
    line_of_therapy = case_core.get("LINE_OF_THERAPY") or []
    if isinstance(line_of_therapy, list) and len(line_of_therapy) > 1:
        # Sort by start_date to ensure correct order
        for i, line in enumerate(line_of_therapy[:-1]):
            if line is None or not isinstance(line, dict):
                continue
            next_line = line_of_therapy[i + 1]
            if next_line is None or not isinstance(next_line, dict):
                continue
            end_date = _parse_date_ymd(line.get("end_date"))
            next_start = _parse_date_ymd(next_line.get("start_date"))
            if end_date and next_start and next_start > end_date:
                pfs_days = (next_start - end_date).days
                current_pfs = line.get("pfs_days")
                if current_pfs in [None, "", "Unknown"]:
                    record_fix(f"CASE_CORE.LINE_OF_THERAPY[{i}].pfs_days", current_pfs, str(pfs_days), "compute_pfs_days")
                    line["pfs_days"] = str(pfs_days)
    
    # Ensure MED_ONC has TOXICITIES and CLINICAL_TRIALS (schema: 7 top-level blocks only)
    med_onc = parsed.get("MED_ONC")
    if not isinstance(med_onc, dict):
        parsed["MED_ONC"] = med_onc = {}
    if "TOXICITIES" not in med_onc:
        med_onc["TOXICITIES"] = []
    if "CLINICAL_TRIALS" not in med_onc:
        med_onc["CLINICAL_TRIALS"] = []

    # Timeline ordering
    events = timeline.get("events")
    if isinstance(events, list) and events:
        indexed = []
        for idx, ev in enumerate(events):
            if ev is None or not isinstance(ev, dict):
                continue
            ev_date = _parse_date_ymd(ev.get("date"))
            indexed.append((ev_date, idx, ev))
        indexed.sort(key=lambda x: (x[0] is None, x[0] or datetime.min.date(), x[1]))
        new_events = [x[2] for x in indexed]
        if new_events != events:
            record_fix("TIMELINE.events", f"{len(events)} events", f"{len(new_events)} events (reordered)", "sort_timeline_by_date")
            timeline["events"] = new_events

    parsed["CASE_CORE"] = case_core
    parsed["TIMELINE"] = timeline
    return parsed, fixes


def run_ehr_review(
    client,
    deployment: str,
    review_system_prompt: str,
    source_text: str,
    extracted_json: Dict[str, Any],
    *,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    user = (
        "SOURCE_TEXT:\n" + (source_text or "") +
        "\n\nEXTRACTED_JSON:\n" + json.dumps(extracted_json, ensure_ascii=False, indent=2)
    )
    result = call_extract_once(
        client, deployment, review_system_prompt, user, max_completion_tokens=max_completion_tokens
    )
    # Handle both old 5-tuple and new 6-tuple formats
    if len(result) == 5:
        content, req_id, err, tokens, finish = result
        tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": tokens}
    else:
        content, req_id, err, tokens, finish, tokens_dict = result
    parsed, status = try_parse_json(content)
    return {
        "parsed": parsed,
        "parse_status": status,
        "tokens": tokens,
        "tokens_detail": tokens_dict,
        "finish": finish,
        "req_id": req_id,
        "err": err,
    }


def _redact_review_issues(issues: Any) -> Any:
    if not isinstance(issues, list):
        return issues

    redacted = []
    for issue in issues:
        if isinstance(issue, dict):
            cleaned = dict(issue)
            cleaned.pop("evidence_snippet", None)
            redacted.append(cleaned)
        else:
            redacted.append(issue)
    return redacted


def _redact_review_for_output(review: Any) -> Any:
    if not isinstance(review, dict):
        return review

    cleaned = dict(review)
    parsed = cleaned.get("parsed")
    if isinstance(parsed, dict):
        parsed_cleaned = dict(parsed)
        parsed_cleaned["issues"] = _redact_review_issues(parsed_cleaned.get("issues"))
        cleaned["parsed"] = parsed_cleaned
    return cleaned


def repair_json_once(client, deployment, bad_json_text, *, max_completion_tokens):
    sys_prompt = (
        "You are a JSON repair engine.\n"
        "Return ONLY valid JSON.\n"
        "No markdown. No explanation."
    )
    user = "Fix to valid JSON:\n\n" + bad_json_text
    return call_extract_once(
        client,
        deployment,
        sys_prompt,
        user,
        max_completion_tokens=max_completion_tokens,
    )


# ===========================
# Small utilities (P2)
# ===========================
def sanitize_record_id(x: Any, max_len: int = 60) -> str:
    """
    Make de-identified record labels safe for filenames and logs.
    """
    if x is None:
        s = ""
    elif isinstance(x, str):
        s = x
    else:
        try:
            s = json.dumps(x, ensure_ascii=False)
        except Exception:
            s = str(x)

    s = s.strip()
    if not s:
        return "UNKNOWN"

    # replace illegal filename chars and whitespace
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = s.strip("._-")
    return (s[:max_len] or "UNKNOWN")


def compact_json_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


# ===========================
# Helper: strip internal fields
# ===========================
def strip_internal_fields(ehr: Any) -> Any:
    """Remove non-factual / internal control fields to keep outputs compact."""
    if not isinstance(ehr, dict):
        return ehr

    # TIMELINE.constraints
    try:
        timeline = ehr.get("CASE_CORE", {}).get("TIMELINE", None)
        if isinstance(timeline, dict):
            timeline.pop("constraints", None)
    except Exception:
        pass

    # MED_ONC.notes.allowed_content
    try:
        med_onc = ehr.get("MED_ONC", None)
        if isinstance(med_onc, dict):
            notes = med_onc.get("notes", None)
            if isinstance(notes, dict):
                notes.pop("allowed_content", None)
                # If notes becomes empty, drop it
                if not notes:
                    med_onc.pop("notes", None)
    except Exception:
        pass

    # LAB_TRENDS.rules
    try:
        lab_trends = ehr.get("LAB_TRENDS", None)
        if isinstance(lab_trends, dict):
            lab_trends.pop("rules", None)
    except Exception:
        pass

    return ehr


# ===========================
# Main processing pipeline
# ===========================
def process_file(
    client,
    deployment,
    input_path,
    output_path,
    prompt_path,
    *,
    max_completion_tokens,
    retry,
    input_field,
    enable_json_repair,
    verbose: bool = False,
):
    print(f"🚀 begin: {input_path}")
    # Load EHR prompt config
    cfg_ehr = load_prompt_config(prompt_path)
    system_ehr = build_system_prompt(cfg_ehr, force_english=True)  # always English for EHR extraction
    review_system, _ = build_review_prompt(cfg_ehr)

    # Count total lines (keep tqdm total, but stream processing)
    with open(input_path, "r", encoding="utf-8") as fr:
        total = sum(1 for _ in fr)

    print(f"📌 Total records in input file: {total}")

    # Ensure output dirs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    review_output_path = str(output_path) + ".review.jsonl"
    with open(output_path, "w", encoding="utf-8") as fw, open(review_output_path, "w", encoding="utf-8") as fw_review, open(input_path, "r", encoding="utf-8") as fr:
        pbar = tqdm(total=total, ncols=120, desc="Processing")

        for idx, line in enumerate(fr, start=1):
            line = (line or "").strip()
            if not line:
                pbar.update(1)
                continue

            # Load per-line JSONL
            try:
                row = json.loads(line)
            except Exception:
                out_err = {
                    "meta_info": "PARSE_ERROR",
                    "Time": "",
                    "ehr_extracted": None,
                    "audit_ehr": {
                        "tokens": 0,
                        "finish": "",
                        "parse_status": "input_json_decode_error",
                        "attempts": 0,
                        "errors": ["input_json_decode_error"],
                        "req_id": "",
                        "err": "JSONDecodeError",
                        "repair_attempted": False,
                        "repair_used": False,
                        "repair_parse_status": None,
                    },
                    "error": "JSONDecodeError",
                }
                fw.write(compact_json_line(out_err) + "\n")
                pbar.update(1)
                continue

            # Retrieve input text (question / EMR content)
            src = row.get(input_field, "")
            if not isinstance(src, str):
                src = json.dumps(src, ensure_ascii=False)

            # Identify a de-identified record label for logs/sidecars. Do not
            # emit raw patient identifiers in structured EHR output.
            raw_record_id = row.get("meta_info") or row.get("patient_id") or row.get("name") or ""
            record_id = raw_record_id if isinstance(raw_record_id, str) else raw_record_id
            record_id_safe = sanitize_record_id(raw_record_id)
            Time_val = row.get("Time") or ""

            # Add DOCUMENT_TIME prefix (if available) to help the model resolve visit timing
            src_for_llm = prepend_document_time(src, Time_val)

            pbar.set_description(f"{idx}/{total} | {record_id_safe}")
            pbar.set_postfix_str("step=extract")
            record_t0 = time.time()

            # -------------------------------
            # 1) MAIN EHR EXTRACTION
            # -------------------------------
            print(f"\n{_CLR_BOLD}{_CLR_WHITE}{'─' * 60}{_CLR_RESET}")
            print(f"{_CLR_BOLD}📋 Record: {_CLR_CYAN}{record_id_safe}{_CLR_RESET}")
            print(f"{_CLR_BOLD}{_CLR_WHITE}{'─' * 60}{_CLR_RESET}")
            _print_step("extract_ehr", f"record={record_id_safe}", "start")
            t0 = time.time()
            result = call_with_retry(
                call_extract_once,
                client,
                deployment,
                system_ehr,
                src_for_llm,
                max_retries=retry,
                max_completion_tokens=max_completion_tokens,
                verbose=verbose,
            )
            # Handle both old 7-tuple and new 8-tuple formats (new includes tokens_dict)
            if len(result) == 7:
                ehr_content, req_id, err, tokens, finish, errs, attempts = result
                extract_tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": tokens}
            else:
                ehr_content, req_id, err, tokens, finish, errs, attempts, extract_tokens_dict = result
            finish_color = _CLR_GREEN if finish == "stop" else _CLR_RED
            _print_step("extract_ehr", f"finish={finish_color}{finish}{_CLR_RESET} tokens={_CLR_CYAN}{tokens}{_CLR_RESET} elapsed={time.time()-t0:.2f}s", "done")

            # Track total tokens and time for this patient
            total_tokens = 0
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens += tokens  # Accumulate tokens
            total_input_tokens += extract_tokens_dict.get("input_tokens", 0)
            total_output_tokens += extract_tokens_dict.get("output_tokens", 0)
            pbar.set_postfix_str("step=review_self")

            parsed, status = try_parse_json(ehr_content)

            repair_attempted = False
            repair_used = False
            repair_parse_status = None

            # Remove internal fields if parsed
            if parsed is not None:
                parsed = strip_internal_fields(parsed)

            # -------------------------------
            # 1b) Review + auto-fix (if parsed)
            # -------------------------------
            review_self = None
            review_validator = None
            review_fixes: List[Dict[str, Any]] = []
            if parsed is not None:
                _print_step("review_self", "", "start")
                t0 = time.time()
                review_self = run_ehr_review(
                    client=client,
                    deployment=deployment,
                    review_system_prompt=review_system,
                    source_text=src_for_llm,
                    extracted_json=parsed,
                    max_completion_tokens=min(4000, max_completion_tokens),
                )
                issues_self = (review_self.get("parsed") or {}).get("issues") or []
                _print_step("review_self", f"issues={_CLR_YELLOW}{len(issues_self)}{_CLR_RESET} elapsed={time.time()-t0:.2f}s", "done")
                # Accumulate tokens with input/output breakdown
                if review_self:
                    total_tokens += review_self.get("tokens", 0)
                    detail = review_self.get("tokens_detail", {})
                    total_input_tokens += detail.get("input_tokens", 0)
                    total_output_tokens += detail.get("output_tokens", 0)
                if issues_self:
                    print(f"  {_CLR_DIM}┌─ Self-review issues:{_CLR_RESET}")
                    for it in issues_self[:5]:
                        sev = it.get("severity", "unknown")
                        path = it.get("field_path", "")
                        desc = it.get("description", "")
                        color = _sev_color(sev)
                        icon = _sev_icon(sev)
                        print(f"  {_CLR_DIM}│{_CLR_RESET} {icon} {color}{path}{_CLR_RESET}: {desc[:80]}...")
                    print(f"  {_CLR_DIM}└{'─' * 50}{_CLR_RESET}")
                _print_step("review_validator", "", "start")
                pbar.set_postfix_str("step=review_validator")
                t0 = time.time()
                review_validator = run_ehr_review(
                    client=client,
                    deployment=deployment,
                    review_system_prompt=review_system,
                    source_text=src_for_llm,
                    extracted_json=parsed,
                    max_completion_tokens=min(4000, max_completion_tokens),
                )
                issues_validator = (review_validator.get("parsed") or {}).get("issues") or []
                _print_step("review_validator", f"issues={_CLR_YELLOW}{len(issues_validator)}{_CLR_RESET} elapsed={time.time()-t0:.2f}s", "done")
                # Accumulate tokens with input/output breakdown
                if review_validator:
                    total_tokens += review_validator.get("tokens", 0)
                    detail = review_validator.get("tokens_detail", {})
                    total_input_tokens += detail.get("input_tokens", 0)
                    total_output_tokens += detail.get("output_tokens", 0)
                if issues_validator:
                    print(f"  {_CLR_DIM}┌─ Validator issues:{_CLR_RESET}")
                    for it in issues_validator[:5]:
                        sev = it.get("severity", "unknown")
                        path = it.get("field_path", "")
                        desc = it.get("description", "")
                        color = _sev_color(sev)
                        icon = _sev_icon(sev)
                        print(f"  {_CLR_DIM}│{_CLR_RESET} {icon} {color}{path}{_CLR_RESET}: {desc[:80]}...")
                    print(f"  {_CLR_DIM}└{'─' * 50}{_CLR_RESET}")
                
                # -------------------------------
                # 1c) Iterative Refinement Loop (NEW)
                # -------------------------------
                refinement_config = cfg_ehr.get("REFINEMENT_CONFIG", {
                    "max_iterations": 2,
                    "min_fixable_issues": 1,
                    "issue_severity_threshold": "major"
                })
                refine_instructions = cfg_ehr.get("REFINE_INSTRUCTIONS", "")
                refinement_history: List[Dict[str, Any]] = []
                
                # Combine issues from both reviews
                all_issues = issues_self + issues_validator
                
                iteration = 0
                while refine_instructions and should_refine(all_issues, iteration, refinement_config):
                    iteration += 1
                    pbar.set_postfix_str(f"step=refine_{iteration}")
                    _print_step(f"refine #{iteration}", "", "start")
                    t0 = time.time()
                    
                    # Refine extraction based on issues
                    parsed, refine_audit = refine_extraction(
                        client=client,
                        deployment=deployment,
                        original_json=parsed,
                        issues=all_issues,
                        source_text=src_for_llm,
                        refine_instructions=refine_instructions,
                        max_completion_tokens=max_completion_tokens,
                        verbose=verbose,
                    )
                    refinement_history.append(refine_audit)
                    
                    fields_refined = refine_audit.get("fields_refined", [])
                    status = refine_audit.get("status", "unknown")
                    status_color = _CLR_GREEN if status == "ok" else _CLR_RED
                    _print_step(f"refine #{iteration}", f"status={status_color}{status}{_CLR_RESET} fields={_CLR_CYAN}{len(fields_refined)}{_CLR_RESET} elapsed={time.time()-t0:.2f}s", "done")
                    # Accumulate tokens with input/output breakdown
                    total_tokens += refine_audit.get("tokens", 0)
                    detail = refine_audit.get("tokens_detail", {})
                    total_input_tokens += detail.get("input_tokens", 0)
                    total_output_tokens += detail.get("output_tokens", 0)
                    if fields_refined:
                        print(f"  {_CLR_DIM}┌─ Refinements applied:{_CLR_RESET}")
                        for f in fields_refined[:5]:
                            new_val = refine_audit.get("refinements_applied", {}).get(f, "?")
                            print(f"  {_CLR_DIM}│{_CLR_RESET} {_CLR_GREEN}✓{_CLR_RESET} {_CLR_CYAN}{f}{_CLR_RESET} → {new_val}")
                        print(f"  {_CLR_DIM}└{'─' * 50}{_CLR_RESET}")
                    
                    # Re-review after refinement
                    if refine_audit.get("status") == "ok" and fields_refined:
                        pbar.set_postfix_str(f"step=re_review_{iteration}")
                        _print_step(f"re-review #{iteration}", "", "start")
                        t0 = time.time()
                        re_review = run_ehr_review(
                            client=client,
                            deployment=deployment,
                            review_system_prompt=review_system,
                            source_text=src_for_llm,
                            extracted_json=parsed,
                            max_completion_tokens=min(4000, max_completion_tokens),
                        )
                        all_issues = (re_review.get("parsed") or {}).get("issues") or []
                        issue_color = _CLR_GREEN if len(all_issues) == 0 else (_CLR_YELLOW if len(all_issues) < 5 else _CLR_RED)
                        _print_step(f"re-review #{iteration}", f"remaining={issue_color}{len(all_issues)}{_CLR_RESET} elapsed={time.time()-t0:.2f}s", "done")
                        # Accumulate tokens with input/output breakdown
                        total_tokens += re_review.get("tokens", 0)
                        detail = re_review.get("tokens_detail", {})
                        total_input_tokens += detail.get("input_tokens", 0)
                        total_output_tokens += detail.get("output_tokens", 0)
                    else:
                        # No refinements applied or error - stop loop
                        break
                
                # Add refinement summary to review record
                review_validator["refinement_history"] = refinement_history
                review_validator["refinement_iterations"] = iteration
                review_validator["final_issues"] = all_issues
                
                _print_step("auto_fix", "", "start")
                pbar.set_postfix_str("step=auto_fix")
                t0 = time.time()
                parsed, review_fixes = apply_auto_fixes(parsed)
                _print_step("auto_fix", f"fixes={_CLR_CYAN}{len(review_fixes)}{_CLR_RESET} elapsed={time.time()-t0:.2f}s", "done")
                if review_fixes:
                    print(f"  {_CLR_DIM}┌─ Auto fixes applied:{_CLR_RESET}")
                    for fx in review_fixes[:5]:
                        path = fx.get("path", "")
                        reason = fx.get("reason", "")
                        print(f"  {_CLR_DIM}│{_CLR_RESET} {_CLR_MAGENTA}⚙{_CLR_RESET} {path} → {reason}")
                    print(f"  {_CLR_DIM}└{'─' * 50}{_CLR_RESET}")
            else:
                _print_step("review", f"{_CLR_RED}skipped (parse failed){_CLR_RESET}", "error")
                pbar.set_postfix_str("step=write_output")

            # Attempt repair if JSON broken
            if parsed is None and enable_json_repair and (ehr_content or "").strip():
                repair_attempted = True
                result = repair_json_once(
                    client, deployment, ehr_content, max_completion_tokens=max_completion_tokens
                )
                # Handle both old 5-tuple and new 6-tuple formats
                if len(result) == 5:
                    rep_cont, rep_req, rep_err, rep_tokens, rep_finish = result
                    repair_tokens_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": rep_tokens}
                else:
                    rep_cont, rep_req, rep_err, rep_tokens, rep_finish, repair_tokens_dict = result
                parsed2, status2 = try_parse_json(rep_cont)
                repair_parse_status = status2

                if parsed2 is not None:
                    parsed2 = strip_internal_fields(parsed2)
                    parsed = parsed2
                    status = status2
                    ehr_content = rep_cont
                    repair_used = True
                    # Accumulate repair tokens with input/output breakdown
                    total_tokens += rep_tokens
                    total_input_tokens += repair_tokens_dict.get("input_tokens", 0)
                    total_output_tokens += repair_tokens_dict.get("output_tokens", 0)

            # -------------------------------
            # Write output JSONL (compact + ordered)
            #
            # Naming convention:
            # - question: final structured EHR JSON (dict) used by downstream agents
            # Raw source text is intentionally not copied into the structured output.
            # -------------------------------
            # Remove source text, duplicated payloads, and evaluation labels before
            # writing structured EHR output. `gold_plan` must remain external
            # to extraction so downstream model inputs cannot accidentally see it.
            out = strip_ehr_output_passthrough_fields(row, input_field)

            # Insert in desired order
            out["Time"] = Time_val

            # Primary outputs
            out["question"] = parsed

            # Extraction metrics (total tokens and time for this patient)
            # Unified naming with OMGs pipeline for consistent tracking
            extract_time_seconds = time.time() - record_t0
            out["extract_token"] = total_tokens
            out["extract_time"] = round(extract_time_seconds, 2)
            out["extract_input_token"] = total_input_tokens
            out["extract_output_token"] = total_output_tokens

            out["summary"] = {
                "extract_time_seconds": round(extract_time_seconds, 2),
                "extract_total_tokens": total_tokens,
                "extract_input_tokens": total_input_tokens,
                "extract_output_tokens": total_output_tokens,
            }

            # Optional duplicated keys remain off by default.

            # Get refinement info if available
            refinement_iterations = 0
            refinement_history_out = []
            final_issues = []
            if review_validator and isinstance(review_validator, dict):
                refinement_iterations = review_validator.get("refinement_iterations", 0)
                refinement_history_out = review_validator.get("refinement_history", [])
                final_issues = review_validator.get("final_issues", [])
            
            audit_ehr = {
                "tokens": tokens,
                "finish": finish,
                "parse_status": status,
                "attempts": attempts,
                "errors": errs,
                "req_id": req_id,
                "err": err,
                "repair_attempted": repair_attempted,
                "repair_used": repair_used,
                "repair_parse_status": repair_parse_status,
                "refinement_iterations": refinement_iterations,
                "refinement_status": "ok" if refinement_iterations > 0 else "not_needed",
            }
            
            # Compute field confidence for critical fields
            critical_fields = [
                "CASE_CORE.HRD", "CASE_CORE.BRCA1", "CASE_CORE.BRCA2",
                "CASE_CORE.PLATINUM_STATUS", "CASE_CORE.DIAGNOSIS.histology",
            ]
            field_confidence = {}
            for field in critical_fields:
                field_confidence[field] = compute_field_confidence(
                    field, final_issues, refinement_history_out
                )

            review_self_out = _redact_review_for_output(review_self)
            review_validator_out = _redact_review_for_output(review_validator)
            final_issues_out = _redact_review_issues(final_issues)
            
            review_record = {
                "meta_info": record_id,
                "Time": Time_val,
                "req_id": req_id,
                "parse_status": status,
                "audit_core": audit_ehr,
                "review": {
                    "self": review_self_out,
                    "validator": review_validator_out,
                    "auto_fixes": review_fixes,
                },
                "refinement": {
                    "iterations": refinement_iterations,
                    "history": refinement_history_out,
                    "final_issues": final_issues_out,
                },
                "field_confidence": field_confidence,
            }

            _print_step("write_output", "", "start")
            fw.write(compact_json_line(out) + "\n")
            if review_self or review_validator or review_fixes or audit_ehr:
                fw_review.write(compact_json_line(review_record) + "\n")
            _print_step("record_done", f"elapsed={time.time()-record_t0:.2f}s", "done")
            pbar.set_postfix_str("step=done")

            pbar.update(1)

        pbar.close()

    print(f"✅ Written to {output_path}")
    print(f"✅ Review sidecar: {review_output_path}")



# ===========================
# CLI
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--field", default="question")
    ap.add_argument("--max-completion-tokens", type=int, default=40000)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--db_path", default="data/logs/omgs_api_trace.db")
    ap.add_argument("--provider", type=str, default='auto',
                    choices=['azure', 'openai', 'openrouter', 'auto'],
                    help="LLM provider: 'azure', 'openai', 'openrouter', or 'auto' (auto-detect based on model name)")
    ap.add_argument("--disable-json-repair", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose retry logs")
    ap.add_argument("--quiet", action="store_true", help="Suppress retry logs (default)")

    args = ap.parse_args()

    # Initialize client with provider support
    if args.provider == 'auto':
        from core.client import init_client_from_config
        client = init_client_from_config(model=args.deployment, db_path=args.db_path)
    else:
        client = init_client(db_path=args.db_path, provider=args.provider)
    
    print(f"[INFO] Using provider: {client.provider}, deployment: {args.deployment}")

    process_file(
        client=client,
        deployment=args.deployment,
        input_path=args.input,
        output_path=args.output,
        prompt_path=args.prompts,
        max_completion_tokens=args.max_completion_tokens,
        retry=args.retries,
        input_field=args.field,
        enable_json_repair=not args.disable_json_repair,
        verbose=(args.verbose and not args.quiet),
    )


if __name__ == "__main__":
    main()
