# agent_omgs.py
# OMGs (Ovarian-cancer Multidisciplinary intelligent aGent System) - MDT Pipeline Orchestrator
# =============================================================================
#
# MODE HIERARCHY (increasing input availability):
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MODE               │ RAG (Guidelines/PubMed) │ REPORTS     │ MULTI-EXPERT │
# ├────────────────────┼─────────────────────────┼─────────────┼──────────────┤
# │ 1. CHAIR-R         │            ❌           │      ❌      │      ❌      │
# │ 2. CHAIR-E         │            ✅           │ Structured* │      ❌      │
# │ 3. CHAIR-D         │            ✅           │      ✅      │      ❌      │
# │ 4. OMGs            │            ✅           │      ✅      │      ✅      │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# * CHAIR-E uses structured case history from input JSON (LAB_TRENDS, TIMELINE)
#   NOT external report files like labs.json, imaging.json
#
# DATA SOURCE DEFINITIONS:
# - Structured Case History: LAB_TRENDS, TIMELINE embedded in input case_json
# - Evidence Pack: External files (labs.json, imaging.json, pathology.json, mutation.json)
# - RAG: Engine-backed Guidelines + local external evidence runtime + NCCN rules
#
# EVIDENCE TAG FORMATS:
# ┌────────────────────────────────────────────────────────────────────────────┐
# │ DATA TYPE              │ CITATION FORMAT                                │
# ├────────────────────────┼──────────────────────────────────────────────────┤
# │ Laboratory values      │ [@date | LAB] e.g., [@2022-12-29 | LAB]          │
# │ Guidelines (PDF)        │ exact guideline tag from RAG digest             │
# │ NCCN rules              │ [@guideline:nccn | rule_id]                     │
# │ PubMed literature      │ [@pubmed | PMID]                                │
# │ Imaging findings       │ [@date | MR/CT] e.g., [@2022-12-30 | CT]         │
# │ Pathology reports      │ [@date | Pathology] e.g., [@2022-03-28 | Path]   │
# │ Clinical trials        │ [@trial | trial_id]                              │
# └────────────────────────────────────────────────────────────────────────────┘
#
# NOTE: This file is part of the OMGs/MDT agent pipeline.
# Any changes here should preserve clinical logic and only adjust observability/debugging unless explicitly intended.
import os
import re
import sys
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils.console_utils import Color, normalize_trial_compact, safe_parse_json_block, question_to_text, preview_text
from servers.provenance_tracking import VisualConfig, TraceLogger, print_selected_reports_table, print_section, print_rag_hits_table, warn_missing_evidence_tags
from core.agent import AgentError
from utils.error_handling import safe_agent_call, get_fallback_response
from utils.time_utils import make_cutoff, parse_dt, safe_date10, filter_before, report_range
from utils.time_utils import build_lab_timeline, build_imaging_timeline, build_pathology_timeline
from servers.report_selection import (
    load_patient_labs, load_patient_imaging, load_patient_pathology, 
    load_patient_mutations, summarize_selected_reports, select_reports_for_roles,
    expert_select_reports
)
from servers.evidence_retrieval import (
    build_rag_query_for_mdt,
    emit_rag_digest_ready,
    sanitize_rag_query,
    summarize_rag_evidence,
    _get_rag_result_tag,
)
from orchestrator.experts import ROLES, ROLE_PERMISSIONS, init_expert_agent
from servers.context_assembly import safe_load_case_json, sanitize_case_for_decision, build_role_specific_case_view
from orchestrator.decision import generate_final_output, append_references_to_output, split_references_from_output
from core import Agent, init_client, get_paths_config, get_mdt_prompts
from utils.patterns import extract_reference_tags
from utils.mdt_runtime_protocol import build_runtime_protocol_digest
# Public API of `utils`
import random
import hashlib
import tiktoken
from pathlib import Path
from prettytable import PrettyTable, ALL
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_VALID_TRIAL_BACKENDS = {"local", "local_fuscc", "public", "both"}
_TRIAL_ENGINE = None
_DEFAULT_MDT_SYNTHESIS_TIMEOUT_SECONDS = 90.0
ABLATION_WO_SPECIALIST_INTERPRETATION = "OMGs w/o Specialist Interpretation"
ABLATION_WO_DELIBERATION = "OMGs w/o Deliberation"
ABLATION_FULL_OMGS = "OMGs"
_ABLATION_LABEL_ALIASES = {
    " ".join(("OMGs", "w/o", "Specialist", "Opinions")): ABLATION_WO_SPECIALIST_INTERPRETATION,
}


def normalize_ablation_output_labels(outputs: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Normalize ablation output labels while preserving canonical keys.

    Accept older local result-bundle aliases while preserving canonical labels.
    """
    if not isinstance(outputs, dict):
        return {}

    normalized: Dict[str, str] = {}
    alias_items: List[Tuple[str, str]] = []
    for key, value in outputs.items():
        key_str = str(key)
        canonical = _ABLATION_LABEL_ALIASES.get(key_str)
        if canonical:
            alias_items.append((canonical, value))
        else:
            normalized[key_str] = value

    for canonical, value in alias_items:
        normalized.setdefault(canonical, value)
    return normalized


def _normalize_reference_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", str(tag or "").strip()).lower()


def _mdt_synthesis_timeout_seconds() -> float:
    raw = str(os.environ.get("OMGS_MDT_SYNTHESIS_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return _DEFAULT_MDT_SYNTHESIS_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_MDT_SYNTHESIS_TIMEOUT_SECONDS
    return max(value, 1.0)


def _render_chair_mode_final_prompt(mode: str, **values: Any) -> str:
    prompts = get_mdt_prompts().get("chair_modes", {})
    template = (prompts.get(mode, {}) or {}).get("final_prompt")
    if not template:
        raise RuntimeError(f"Missing chair final prompt template for mode: {mode}")
    return str(template).format(**values).strip()


def _render_global_guideline_digester_instruction(agent_prompts: Dict[str, Any], rag_count: int) -> str:
    template = agent_prompts.get(
        "global_guideline_digester",
        "Digest RAG chunks into exactly {rag_count} evidence bullets (one per RAG result); no patient facts.",
    )
    return str(template).format(rag_count=rag_count)


def _is_rag_evidence_tag(tag: str) -> bool:
    tag_lower = str(tag or "").strip().lower()
    if re.match(r"\[@[^|\]]+\s+\|\s+(?:lab|genomics|mr|ct|imaging|pathology|case)\s*\]", tag_lower):
        return False
    return (
        tag_lower.startswith("[@guideline:")
        or tag_lower.startswith("[@pubmed")
        or tag_lower.startswith("[@fda")
        or tag_lower.startswith("[@conference")
        or tag_lower.startswith("[@trial")
        or tag_lower.startswith("[@nccn")
    )


def _build_reference_tags_for_rag(rag_raw: Optional[List[Dict[str, Any]]]) -> List[str]:
    tags: List[str] = []
    for i, r in enumerate(rag_raw or [], 1):
        tag = _get_rag_result_tag(r, i)
        if tag.startswith("[unknown source"):
            continue
        tags.append(tag)
    return tags


def _reference_tag_set(tags_text: str) -> set[str]:
    return {
        _normalize_reference_tag(tag)
        for tag in extract_reference_tags(tags_text or "")
        if _is_rag_evidence_tag(tag)
    }


def _format_reference_tags(tags: List[str]) -> str:
    return "\n".join(f"  - {tag}" for tag in tags) if tags else "  (No references available)"


def _dedupe_rag_raw_by_tag(rag_raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for i, item in enumerate(rag_raw or [], 1):
        tag = _get_rag_result_tag(item, i)
        key = _normalize_reference_tag(tag) if not tag.startswith("[unknown source") else ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(item)
    return deduped


def _rag_raw_by_normalized_tag(rag_raw: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    by_tag: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(rag_raw or [], 1):
        tag = _get_rag_result_tag(item, i)
        if tag.startswith("[unknown source"):
            continue
        by_tag.setdefault(_normalize_reference_tag(tag), item)
    return by_tag


def _compact_text(text: Any, max_chars: int = 360) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 1)].rstrip() + "…"


def _evidence_excerpt(text: Any, max_chars: int = 900) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    marker = " ... [middle omitted] ... "
    side_chars = max(120, (max_chars - len(marker)) // 2)
    head = cleaned[:side_chars].rstrip()
    tail = cleaned[-side_chars:].lstrip()
    return head + marker + tail


def _format_external_evidence_digest(
    raw: Optional[List[Dict[str, Any]]],
    heading: str,
    *,
    max_items: int = 8,
) -> str:
    lines = [f"# {heading}"]
    kept = 0
    for i, item in enumerate(raw or [], 1):
        source = str(item.get("source") or "").lower()
        if source not in {"pubmed", "fda", "conference"}:
            continue
        tag = _get_rag_result_tag(item, i)
        if tag.startswith("[unknown source"):
            continue
        title = _compact_text(item.get("title") or item.get("citation") or source.upper(), 120)
        body = _evidence_excerpt(
            item.get("summary") or item.get("abstract") or item.get("text") or item.get("selection_rationale"),
            900,
        )
        if body:
            lines.append(f"- {title}: {body} {tag}")
        else:
            lines.append(f"- {title} {tag}")
        kept += 1
        if kept >= max_items:
            break
    if kept == 0:
        lines.append("- No role-specific external evidence returned.")
    return "\n".join(lines)


def _format_role_rag_evidence_digest(
    raw: Optional[List[Dict[str, Any]]],
    heading: str,
    *,
    max_items: int = 8,
) -> str:
    lines = [f"# {heading}"]
    kept = 0
    for i, item in enumerate(raw or [], 1):
        tag = _get_rag_result_tag(item, i)
        if tag.startswith("[unknown source"):
            continue
        title = _compact_text(item.get("title") or item.get("rule_id") or item.get("doc_id") or item.get("source") or "evidence", 120)
        body = _evidence_excerpt(
            item.get("summary")
            or item.get("text")
            or item.get("abstract")
            or item.get("selection_rationale")
            or item.get("statement"),
            900,
        )
        if body:
            lines.append(f"- {title}: {body} {tag}")
        else:
            lines.append(f"- {title} {tag}")
        kept += 1
        if kept >= max_items:
            break
    if kept == 0:
        lines.append("- No role-specific guideline evidence returned.")
    return "\n".join(lines)


def _join_pack_sections(*sections: str) -> str:
    return "\n\n".join(section.strip() for section in sections if str(section or "").strip())


_ROLE_QUERY_ANCHOR_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("fralpha", "fr alpha", "folr1", "folate receptor alpha", "folate receptor-alpha"),
    ("her2", "erbb2"),
    ("brca", "brca1", "brca2"),
    ("hrd", "homologous recombination"),
    ("ccne1",),
    ("pd-l1", "pdl1", "pd l1"),
    ("msi", "microsatellite"),
    ("tmb", "tumor mutational burden"),
    ("tp53",),
    ("mirvetuximab", "elahere"),
    ("bevacizumab", "anti-vegf", "anti vegf", "vegf"),
    ("olaparib", "niraparib", "rucaparib", "parp"),
    ("paclitaxel", "taxane"),
    ("carboplatin", "cisplatin", "platinum"),
    ("doxorubicin", "topotecan", "gemcitabine"),
    ("pembrolizumab", "dostarlimab", "immunotherapy"),
    ("peritoneal",),
    ("pelvic",),
    ("bowel", "serosa", "obstruction", "perforation", "fistula"),
    ("liver", "hepatic"),
    ("thoracic", "lung", "pleural"),
    ("bone", "osseous"),
    ("ascites",),
    ("platinum-resistant", "platinum resistant"),
    ("platinum-sensitive", "platinum sensitive"),
    ("high-grade serous", "high grade serous", "hgsoc"),
    ("clear cell",),
    ("endometrioid",),
)


def _role_query_has_unsupported_anchor(query: str, source_text: str) -> bool:
    query_lower = str(query or "").lower()
    source_lower = str(source_text or "").lower()
    for aliases in _ROLE_QUERY_ANCHOR_GROUPS:
        if any(alias in query_lower for alias in aliases) and not any(alias in source_lower for alias in aliases):
            return True
    return False


def _build_role_query_validation_source(shared_query: str, rag_case_json: Any, rag_key_facts: str) -> str:
    return "\n".join(
        [
            str(rag_key_facts or ""),
            str(shared_query or ""),
            json.dumps(rag_case_json or {}, ensure_ascii=False),
        ]
    )


def _sanitize_role_external_query(query: str, fallback_query: str = "") -> Tuple[str, bool]:
    sanitized, changed = sanitize_rag_query(query or "")
    sanitized = _compact_text(sanitized, 260)
    if sanitized:
        return sanitized, changed
    fallback_sanitized, fallback_changed = sanitize_rag_query(fallback_query or "")
    return _compact_text(fallback_sanitized, 260), changed or fallback_changed


def _normalize_mdt_evidence_mode(value: Optional[str]) -> str:
    mode = str(value or "role_private").strip().lower().replace("-", "_")
    if mode == "shared":
        return "shared"
    return "role_private" if mode in {"role_private", "role_specific"} else "role_private"


def _ensure_ablation_supported_for_evidence_mode(ablation_enabled: bool, evidence_mode: str) -> None:
    """Ablation outputs are only defined for the role-private evidence protocol."""
    if ablation_enabled and evidence_mode != "role_private":
        raise ValueError(
            "OMGs ablation outputs require OMGS_MDT_EVIDENCE_MODE=role_private. "
            "Shared evidence mode is not a supported strict ablation protocol."
        )


def _build_role_external_queries(
    *,
    role_query_agent: "Agent",
    prompts: Dict[str, Any],
    shared_query: str,
    rag_case_json: Any,
    rag_key_facts: str,
    roles: List[str],
    trace: Optional["TraceLogger"] = None,
) -> Dict[str, str]:
    shared_query, _ = _sanitize_role_external_query(shared_query, "")
    role_query_template = prompts.get(
        "role_external_query_user_template",
        "# KEY FACTS\n{key_facts}\n\n# SHARED QUERY\n{shared_query}\n\n# STRUCTURED CASE JSON\n{case_json}\n\nGenerate role-specific external evidence search queries.",
    )
    user_prompt = role_query_template.format(
        key_facts=rag_key_facts or "(none)",
        shared_query=shared_query or "",
        case_json=json.dumps(rag_case_json or {}, ensure_ascii=False, separators=(",", ":")),
    )
    validation_source = _build_role_query_validation_source(shared_query, rag_case_json, rag_key_facts)
    queries = {role: shared_query for role in roles}
    if trace:
        trace.emit("role_external_query_start", {"roles": roles, "query_source": "llm"})
    try:
        raw = role_query_agent.chat_once(user_prompt)
        parsed = safe_parse_json_block(raw)
        if not isinstance(parsed, dict):
            raise ValueError("role external query builder did not return a JSON object")
    except Exception as exc:
        if trace:
            trace.emit(
                "role_external_query_result",
                {
                    "fallback_all": True,
                    "reason": str(exc),
                    "queries": queries,
                },
            )
        return queries

    accepted: Dict[str, str] = {}
    for role in roles:
        candidate = _compact_text(parsed.get(role), 260)
        reason = ""
        if not candidate:
            reason = "missing_or_empty"
            candidate = shared_query
        elif _role_query_has_unsupported_anchor(candidate, validation_source):
            reason = "unsupported_anchor"
            candidate = shared_query
        else:
            reason = "accepted"
        candidate, sanitized_changed = _sanitize_role_external_query(candidate, shared_query)
        if sanitized_changed:
            reason = f"{reason}+sanitized" if reason else "sanitized"
        accepted[role] = candidate
        if trace:
            trace.emit(
                "role_external_query_result",
                {
                    "role": role,
                    "query": candidate,
                    "fallback": candidate == shared_query and reason != "accepted",
                    "reason": reason,
                },
            )
    return accepted


def _format_initial_opinions_for_prompt(initial_ops: Dict[str, str]) -> str:
    lines = ["# INITIAL SPECIALIST INTERPRETATION"]
    for role in ROLES:
        if role not in (initial_ops or {}):
            continue
        lines.append(f"## {role}\n{str(initial_ops.get(role) or '').strip()}")
    return "\n\n".join(lines)


def _report_evidence_key(modality: str, report: Dict[str, Any]) -> str:
    report_id = str((report or {}).get("report_id") or "").strip()
    if report_id:
        return f"{modality}:{report_id}"
    payload = json.dumps(report or {}, ensure_ascii=False, sort_keys=True)
    return f"{modality}:hash:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def _dedupe_reports_by_modality(
    report_context: Dict[str, Dict[str, List[Dict[str, Any]]]],
    modalities: Tuple[str, ...] = ("lab", "imaging", "pathology", "mutation"),
) -> Dict[str, List[Dict[str, Any]]]:
    """Union selected reports across roles for matched comparator inputs."""
    deduped: Dict[str, List[Dict[str, Any]]] = {}
    for modality in modalities:
        by_key: Dict[str, Dict[str, Any]] = {}
        for reports in ((report_context or {}).get(modality, {}) or {}).values():
            for report in reports or []:
                if not isinstance(report, dict):
                    continue
                by_key.setdefault(_report_evidence_key(modality, report), report)
        deduped[modality] = list(by_key.values())
    return deduped


def _format_matched_report_dossier(
    report_context: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> str:
    reports_by_modality = _dedupe_reports_by_modality(report_context)
    headings = {
        "lab": "LAB REPORTS",
        "imaging": "IMAGING REPORTS",
        "pathology": "PATHOLOGY REPORTS",
        "mutation": "MUTATION / MOLECULAR REPORTS",
    }
    sections: List[str] = ["# Clinical Reports (PATIENT FACTS)"]
    for modality in ("lab", "imaging", "pathology", "mutation"):
        reports = reports_by_modality.get(modality) or []
        if not reports:
            continue
        sections.append(
            f"## {headings[modality]} (PATIENT FACTS)\n"
            + json.dumps(reports, ensure_ascii=False, indent=2)
        )
    if len(sections) == 1:
        sections.append("No selected clinical reports available.")
    return "\n\n".join(sections)


def _build_matched_direct_chair_agent(
    *,
    question_str: str,
    model: str,
    client: Any,
    report_context: Dict[str, Dict[str, List[Dict[str, Any]]]],
    case_fingerprint: str,
    visit_time: Optional[str],
    global_guideline_digest: str,
    role_recruited_evidence_pack: str,
) -> "Agent":
    """Create a direct chair comparator from matched case, report and evidence inputs."""
    case_json = sanitize_case_for_decision(safe_load_case_json(question_str))
    case_view = build_role_specific_case_view("chair", case_json)
    visit_time_str = visit_time or "Unknown visit date"
    role_prompt = get_mdt_prompts().get("role_prompts", {}).get("chair", "").strip()
    if not role_prompt:
        role_prompt = (
            "You are the MDT chair. Integrate evidence, maintain safety, "
            "and provide a structured management recommendation."
        )
    instruction = _join_pack_sections(
        build_runtime_protocol_digest("chair"),
        f"OUTPATIENT VISIT TIME (today's clinic decision point): {visit_time_str}",
        f"CASE_FINGERPRINT: {case_fingerprint}",
        role_prompt,
        (
            "# HARD RULES (critical)\n"
            "1) All decisions are for THIS visit date and future care, not for past timepoints.\n"
            "2) PATIENT FACTS come ONLY from:\n"
            "   - Role-Specific Case View, and\n"
            "   - Clinical Reports selected for this role (including mutation reports if provided).\n"
            "3) GLOBAL Guideline Digest is ONLY general reference:\n"
            "   - MUST NOT be treated as patient-specific facts.\n"
            "   - Never invent labs/imaging/mutations from guidelines.\n"
            "4) Any claim derived from guideline/PubMed evidence MUST include evidence tag:\n"
            "   - applies to treatment strategy categories, guideline/consensus statements, or trial/literature evidence\n"
            "   - use the exact guideline tag from the digest: [@guideline:doc_id | Page xx] or [@guideline:doc_id | Pages xx-yy]\n"
            "   - PubMed format: [@pubmed | PMID]\n"
            "4b) At least ONE bullet must be evidence-based and include a guideline tag from the digest or [@pubmed | PMID].\n"
            "5) Any claim about labs/imaging/pathology/molecular MUST include evidence tag:\n"
            "   - format: [@actual_report_id | LAB/Genomics/MR/CT/Pathology] using actual report_id from report data\n"
            "   - Examples: [@LAB20251020TM | LAB], [@OH20251003 | Genomics], [@CT20250922 | CT], [@PETCT20251021 | CT], [@PX20251003 | Pathology]\n"
            "   - Note: Always use spaces around | for consistency: [@xxx | yyy]\n"
            "   - Use the exact report_id value from the Clinical Reports section above\n"
            "   - If no report supports it, say \"unknown/needs update\".\n"
            "6) If Case View conflicts with Clinical Reports:\n"
            "   - Prefer Clinical Reports; note discrepancy briefly.\n"
            "7) Do NOT hallucinate. If missing, defer to correct specialty."
        ),
        "# Role-Specific Case View (PATIENT FACTS)\n" + case_view,
        _format_matched_report_dossier(report_context),
        "# GLOBAL Guideline + PubMed Digest (NOT PATIENT FACTS)\n" + str(global_guideline_digest or "").strip(),
        str(role_recruited_evidence_pack or "").strip(),
    )
    ag = Agent(
        instruction=instruction,
        role="chair",
        model_info=model,
        client=client,
        max_tokens=100000,
        max_prompt_tokens=100000,
    )
    ag.inject_assistant("System ready for MDT discussion.")
    return ag


def _merge_reference_tag_texts(*tag_texts: Optional[str]) -> str:
    seen: set[str] = set()
    lines: List[str] = []
    for text in tag_texts:
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("("):
                continue
            tag = stripped[2:].strip() if stripped.startswith("- ") else stripped
            norm = _normalize_reference_tag(tag)
            if norm in seen:
                continue
            seen.add(norm)
            lines.append(f"  - {tag}")
    return "\n".join(lines) if lines else "  (No references available)"


def _build_role_recruited_evidence_pack(
    *,
    report_context: Dict[str, Dict[str, List[Dict[str, Any]]]],
    role_guideline_digest_by_role: Dict[str, str],
    reference_tags_by_role: Optional[Dict[str, str]] = None,
) -> str:
    """Build an evidence pack from already recruited non-chair evidence."""
    sections: List[str] = ["# Additional recruited evidence"]

    modality_labels = {
        "lab": "Laboratory reports",
        "imaging": "Imaging reports",
        "pathology": "Pathology reports",
        "mutation": "Mutation / molecular reports",
    }
    for modality in ("lab", "imaging", "pathology", "mutation"):
        by_key: Dict[str, Dict[str, Any]] = {}
        role_reports = (report_context or {}).get(modality, {}) or {}
        chair_seen = {
            _report_evidence_key(modality, report)
            for report in role_reports.get("chair", []) or []
            if isinstance(report, dict)
        }
        for role in ROLES:
            if role == "chair":
                continue
            for report in role_reports.get(role, []) or []:
                if not isinstance(report, dict):
                    continue
                key = _report_evidence_key(modality, report)
                if key in chair_seen:
                    continue
                entry = by_key.setdefault(key, {"roles": [], "report": report})
                if role not in entry["roles"]:
                    entry["roles"].append(role)
        if not by_key:
            continue
        lines = [f"## {modality_labels[modality]}"]
        for entry in by_key.values():
            lines.append(json.dumps(entry["report"], ensure_ascii=False, indent=2))
        sections.append("\n".join(lines))

    role_sections: List[str] = []
    for role in ROLES:
        if role == "chair":
            continue
        digest = str((role_guideline_digest_by_role or {}).get(role) or "").strip()
        tags = str((reference_tags_by_role or {}).get(role) or "").strip()
        if not digest and not tags:
            continue
        role_parts = [f"## {role} guideline/external evidence"]
        if digest:
            role_parts.append(digest)
        if tags:
            role_parts.append("# Reference tags available to this role\n" + tags)
        role_sections.append("\n\n".join(role_parts))
    if role_sections:
        sections.append("# Additional guideline and external evidence\n" + "\n\n".join(role_sections))

    return _join_pack_sections(*sections)


def _build_evidence_only_ablation_context(role_recruited_evidence_pack: str) -> str:
    return _join_pack_sections(
        str(role_recruited_evidence_pack or "").strip(),
        "Key Knowledge:\n- Additional recruited evidence is available for chair synthesis.",
        "Decision Stances:\n- Chair synthesizes from available case facts, recruited evidence, citations, and trial note.",
        "Controversies:\n- None explicitly stated in specialist interpretation.",
        "Missing Info:\n- Use only decision-changing missing data apparent from the available inputs.",
        "Working Plan:\n- Chair to generate the final structured recommendation from available inputs.",
    )


def _build_matched_initial_opinions_context(
    initial_ops: Dict[str, str],
) -> str:
    return _join_pack_sections(
        "Key Knowledge:\n- Independent specialist initial interpretation is provided below.",
        _format_initial_opinions_for_prompt(initial_ops),
        "Decision Stances:\n- Use the specialist write-ups to identify current stance and rationale.",
        "Controversies:\n- Use disagreements or uncertainty explicitly stated in the independent write-ups.",
        "Missing Info:\n- Use decision-changing missing items explicitly stated in the independent write-ups.",
        "Working Plan:\n- Chair to generate the final structured recommendation from available inputs.",
    )


def _generate_matched_ablation_output(
    *,
    label: str,
    chair_agent: "Agent",
    clinic_time: Optional[str],
    merged: str,
    trial_note: str,
    ref_tags_str: str,
    report_context: Dict[str, Dict[str, List[Dict[str, Any]]]],
    trace: TraceLogger,
) -> str:
    """Run a final-chair-input-matched ablation output without changing upstream inputs."""
    print_section(f"Ablation: {label}", "generating matched output")
    trace.emit("ablation_output_start", {"label": label})
    final_output = generate_final_output(
        chair_agent=chair_agent,
        all_round_ops={},
        clinic_time=clinic_time,
        merged=merged,
        initial_ops={},
        interaction_log={},
        trial_note=trial_note,
        trace=trace,
        ref_tags_str=ref_tags_str,
    )
    final_output = append_references_to_output(
        final_output,
        trial_note=trial_note,
        report_context=report_context,
    )
    warn_missing_evidence_tags(final_output, role=f"{label}/final_output", trace=trace)
    trace.emit("ablation_output_end", {"label": label, "final_output_chars": len(final_output or "")})
    return final_output


def _build_matched_ablation_outputs(
    *,
    chair_agent: "Agent",
    question_str: str,
    model: str,
    client: Any,
    clinic_time: Optional[str],
    case_fingerprint: str,
    initial_ops: Dict[str, str],
    merged_for_final: str,
    full_final_output: str,
    public_evidence_block: str,
    trial_note: str,
    ref_tags_str: str,
    report_context: Dict[str, Dict[str, List[Dict[str, Any]]]],
    role_guideline_digest_by_role: Dict[str, str],
    reference_tags_by_role: Optional[Dict[str, str]],
    trace: TraceLogger,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    role_recruited_evidence_pack = _build_role_recruited_evidence_pack(
        report_context=report_context,
        role_guideline_digest_by_role=role_guideline_digest_by_role,
        reference_tags_by_role=reference_tags_by_role,
    )
    role_recruited_ref_tags = _merge_reference_tag_texts(
        ref_tags_str,
        *((reference_tags_by_role or {}).get(role) for role in ROLES),
    )
    frozen_chair_kwargs = {
        "question_str": question_str,
        "model": model,
        "client": client,
        "report_context": report_context,
        "case_fingerprint": case_fingerprint,
        "visit_time": str(clinic_time) if clinic_time else None,
        "global_guideline_digest": role_guideline_digest_by_role.get("chair", ""),
        "role_recruited_evidence_pack": role_recruited_evidence_pack,
    }
    matched_contexts = {
        ABLATION_WO_SPECIALIST_INTERPRETATION: (
            _build_evidence_only_ablation_context(role_recruited_evidence_pack),
            role_recruited_ref_tags,
            _build_matched_direct_chair_agent(**frozen_chair_kwargs),
        ),
        ABLATION_WO_DELIBERATION: (
            _build_matched_initial_opinions_context(initial_ops),
            role_recruited_ref_tags,
            chair_agent,
        ),
    }
    for label, (matched_merged, matched_ref_tags, matched_chair_agent) in matched_contexts.items():
        try:
            outputs[label] = _generate_matched_ablation_output(
                label=label,
                chair_agent=matched_chair_agent,
                clinic_time=clinic_time,
                merged=matched_merged,
                trial_note=trial_note,
                ref_tags_str=matched_ref_tags,
                report_context=report_context,
                trace=trace,
            )
        except Exception as exc:
            print(f"{Color.WARNING}[WARNING] Matched ablation '{label}' failed; continuing full OMGs: {exc}{Color.RESET}")
            trace.emit(
                "pipeline_error",
                {
                    "stage": f"matched_ablation_{label}",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
    outputs[ABLATION_FULL_OMGS] = full_final_output
    trace.emit(
        "ablation_matched_context_ready",
        {
            "labels": list(outputs.keys()),
            "public_evidence_chars": len(public_evidence_block or ""),
            "role_recruited_evidence_chars": len(role_recruited_evidence_pack or ""),
            "merged_for_final_chars": len(merged_for_final or ""),
        },
    )
    return outputs


def _print_ablation_outputs(ablation_outputs: Dict[str, str]) -> None:
    """Print the final outputs for controlled ablations together for comparison."""
    if not ablation_outputs:
        return

    ablation_outputs = normalize_ablation_output_labels(ablation_outputs)
    labels = [
        (ABLATION_WO_SPECIALIST_INTERPRETATION, ABLATION_WO_SPECIALIST_INTERPRETATION),
        (ABLATION_WO_DELIBERATION, ABLATION_WO_DELIBERATION),
        (ABLATION_FULL_OMGS, ABLATION_FULL_OMGS),
    ]
    print_section("Controlled Ablation Final Outputs", "side-by-side review")
    for key, label in labels:
        output = str(ablation_outputs.get(key) or "").strip()
        print(f"{Color.BOLD}{Color.OKBLUE}\n--- {label} ---{Color.RESET}")
        if output:
            print(output)
        else:
            print(f"{Color.WARNING}[missing: {key} output was not generated]{Color.RESET}")


def _print_role_private_evidence_queries(
    *,
    role_queries: Dict[str, str],
    role_guideline_raw_by_role: Dict[str, List[Dict[str, Any]]],
    role_external_raw_by_role: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Print role-private evidence query visibility without dumping full packs."""
    if not role_queries:
        return
    table = PrettyTable(["Role", "Query", "Guidelines", "External"])
    table.align = "l"
    for role, query in role_queries.items():
        table.add_row(
            [
                role,
                preview_text(query, 120),
                len(role_guideline_raw_by_role.get(role, []) or []),
                len(role_external_raw_by_role.get(role, []) or []),
            ]
        )
    print(f"\n{Color.BOLD}{Color.OKBLUE}Role-Specific Evidence Queries{Color.RESET}")
    print(table)


def _build_public_evidence_atom(tag: str, raw: Dict[str, Any], claim: str) -> Dict[str, str]:
    return {
        "tag": tag,
        "source": str(raw.get("source") or ""),
        "title": _compact_text(raw.get("title") or raw.get("citation") or raw.get("source_id") or tag, 140),
        "snippet": _evidence_excerpt(raw.get("summary") or raw.get("abstract") or raw.get("text"), 900),
        "claim": _compact_text(claim, 360),
    }


def _format_public_evidence_atoms(atoms: Optional[List[Dict[str, str]]]) -> str:
    lines: List[str] = []
    for atom in atoms or []:
        tag = atom.get("tag") or ""
        title = atom.get("title") or atom.get("source") or "evidence"
        snippet = atom.get("snippet") or ""
        claim = atom.get("claim") or ""
        detail = f"{title}"
        if snippet:
            detail += f": {snippet}"
        if claim:
            detail += f" | Public claim: {claim}"
        lines.append(f"- {tag} {detail}".strip())
    return "\n".join(lines)


def _split_combined_reference_brackets(text: str) -> str:
    """Split malformed combined tags like [@A | X; @B | Y] into separate brackets."""
    if not text or "; @" not in text:
        return text

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if inner.count("@") < 2:
            return match.group(0)
        parts = [part.strip() for part in re.split(r";\s*(?=@)", inner) if part.strip()]
        if len(parts) < 2 or not all(part.startswith("@") for part in parts):
            return match.group(0)
        return " ".join(f"[{part}]" for part in parts)

    return re.sub(r"\[([^\[\]]*;\s*@[\s\S]*?)\]", _replace, text)


def _sanitize_discussion_evidence_tags(
    msg: str,
    allowed_reference_tags: set[str],
    *,
    role: str,
    stage: str,
    trace: Optional["TraceLogger"] = None,
) -> str:
    """Remove model-invented RAG evidence tags while leaving report tags intact."""
    if not msg:
        return msg

    cleaned = _split_combined_reference_brackets(msg)
    for tag in extract_reference_tags(cleaned):
        if not _is_rag_evidence_tag(tag):
            continue
        if _normalize_reference_tag(tag) in allowed_reference_tags:
            continue
        cleaned = cleaned.replace(tag, "").strip()
        if trace:
            trace.emit(
                "mdt_invalid_evidence_tag_removed",
                {
                    "role": role,
                    "stage": stage,
                    "tag": tag,
                },
            )
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _extract_final_bottom_line_text(final_output: str) -> str:
    text = str(final_output or "").strip()
    marker_match = re.search(r"(?im)^\s*(?:#{1,6}\s*)?Final Assessment\s*:?\s*$", text)
    if marker_match:
        text = text[marker_match.end() :].strip()
    else:
        marker = "Final Assessment:"
        idx = text.find(marker)
        if idx >= 0:
            text = text[idx + len(marker) :].strip()
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if paragraphs:
        return paragraphs[0]
    return text[:300].rstrip() + "…" if len(text) > 300 else text


def _normalize_trial_backend(value: Optional[str]) -> str:
    backend = str(value or "local").strip().lower()
    return backend if backend in _VALID_TRIAL_BACKENDS else "local"


def _get_trial_engine():
    global _TRIAL_ENGINE
    if _TRIAL_ENGINE is None:
        from engine_bridge import EngineIntegration

        _TRIAL_ENGINE = EngineIntegration()
    return _TRIAL_ENGINE


def _clean_trial_query_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"unknown", "not applicable", "not_applicable", "none", "null", "n/a"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _trial_query_has_any(query: str, patterns: tuple[str, ...]) -> bool:
    lower = query.lower()
    return any(pattern in lower for pattern in patterns)


def _trial_short_value(value: Any, *, limit: int = 80) -> str:
    text = _clean_trial_query_value(value)
    if not text:
        return ""
    text = re.split(r";|\.\s+", text, maxsplit=1)[0].strip()
    return text[:limit].rstrip()


def _trial_biomarker_summary(key: str, value: Any) -> str:
    text = _trial_short_value(value, limit=100)
    if not text:
        return ""
    lower = text.lower()
    if key == "FRalpha":
        if (
            "not eligible" in lower
            or "ineligible" in lower
            or "negative" in lower
            or "below" in lower
            or "not detected" in lower
        ):
            return f"FRalpha {text}"
        if "eligible" in lower:
            return "FRalpha eligible"
        if "positive" in lower or "high" in lower:
            return "FRalpha positive"
    if key == "HER2":
        ihc = re.search(r"\bIHC\s*[0-3]\+|\b[0-3]\+", text, re.IGNORECASE)
        if "negative" in lower and not ihc:
            return f"HER2 {text}"
        amplified = "ISH non-amplified" if "non-amplified" in lower or "non amplified" in lower else ""
        parts = ["HER2"]
        if ihc:
            parts.append(ihc.group(0).replace(" ", ""))
        if amplified:
            parts.append(amplified)
        if len(parts) == 1:
            return f"HER2 {text}"
        return " ".join(parts)
    return f"{key} {text}"


def _build_trial_missing_core_facts(rag_query: str, case_json: Dict[str, Any]) -> str:
    case_core = case_json.get("CASE_CORE") if isinstance(case_json, dict) else {}
    case_core = case_core if isinstance(case_core, dict) else {}
    diagnosis = case_core.get("DIAGNOSIS") if isinstance(case_core.get("DIAGNOSIS"), dict) else {}
    biomarkers = case_core.get("BIOMARKERS") if isinstance(case_core.get("BIOMARKERS"), dict) else {}
    query = rag_query or ""
    facts: list[str] = []

    if not _trial_query_has_any(
        query,
        (
            "platinum resistant",
            "platinum-resistant",
            "platinum sensitive",
            "platinum-sensitive",
            "platinum refractory",
            "platinum-refractory",
        ),
    ):
        platinum = _clean_trial_query_value(
            case_core.get("PLATINUM_STATUS_CURRENT") or case_core.get("PLATINUM_STATUS")
        )
        if platinum:
            facts.append(f"platinum status {platinum}")

    histology = _clean_trial_query_value(diagnosis.get("histology") or diagnosis.get("primary"))
    if histology and histology.lower() not in query.lower():
        facts.append(f"histology {histology}")

    if not _trial_query_has_any(query, ("ecog", "performance status")):
        ecog = _clean_trial_query_value(case_core.get("ECOG"))
        if ecog:
            facts.append(f"ECOG {ecog}")

    marker_candidates = [
        ("FRalpha", ("frα", "fralpha", "folr1", "folate receptor")),
        ("HER2", ("her2",)),
        ("BRCA1", ("brca",)),
        ("BRCA2", ("brca",)),
        ("HRD", ("hrd",)),
        ("MSI", ("msi",)),
        ("TMB", ("tmb",)),
        ("PDL1_CPS", ("pd-l1", "pdl1", "pd l1")),
    ]
    marker_facts: list[str] = []
    for key, aliases in marker_candidates:
        if _trial_query_has_any(query, aliases):
            continue
        value = biomarkers.get(key, case_core.get(key))
        summary = _trial_biomarker_summary(key, value)
        if summary and summary not in marker_facts:
            marker_facts.append(summary)
    facts.extend(marker_facts)

    return "; ".join(facts)


def _build_trial_query(rag_query: str, case_json: Dict[str, Any]) -> str:
    base = _clean_trial_query_value(rag_query) or "ovarian cancer clinical trial matching"
    fallback_facts = _build_trial_missing_core_facts(base, case_json)
    if not fallback_facts:
        return base
    return f"{base}; {fallback_facts}"


def _trial_query_analysis_input(query: str) -> Dict[str, Any]:
    return {
        "source_group": "trial",
        "query_analysis_style": "single",
        "main_query": query,
        "main_dense_query": query,
        "main_sparse_query": query,
        "sub_queries": [],
    }


def _query_trial_bundle(question_str: str, *, backend: str):
    from engine_bridge import build_tool_input
    from engine_bridge import default_tool_call_id
    from utils.runtime_config import tool_input_overrides_from_env

    overrides = tool_input_overrides_from_env("trial")
    overrides["query_analysis_mode"] = "off"
    overrides["query_analysis_input"] = _trial_query_analysis_input(question_str)
    tool_result = _get_trial_engine().invoke_tool(
        tool_name="trial",
        tool_call_id=default_tool_call_id("trial"),
        tool_input=build_tool_input(
            query=question_str,
            backend=backend,
            **overrides,
        ),
        consumer="cli",
        verbosity="standard",
        include_artifacts=True,
        include_debug=False,
        include_snapshots=True,
    )
    trial_bundle = dict(tool_result.get("tool_output") or {}).get("bundle")
    if trial_bundle is None:
        raise RuntimeError("trial tool did not return an EvidenceBundle.")
    return trial_bundle


def _trial_record_source_id(record: Any) -> str:
    direct = getattr(record, "source_id", None)
    if direct:
        return str(direct)
    provenance = getattr(record, "provenance", None)
    source_id = getattr(provenance, "source_id", None)
    return str(source_id or "")


def _trial_record_is_fit(record: Any) -> bool:
    highlights = getattr(record, "highlights", None) or []
    return any(str(highlight).strip().lower() == "final_judge=fit" for highlight in highlights)


def _trial_record_final_judge(record: Any, raw_hit: Optional[Dict[str, Any]] = None) -> str:
    hit = raw_hit or {}
    decision = str(hit.get("final_judge_decision") or hit.get("decision") or "").strip().lower()
    if decision:
        return decision
    for highlight in getattr(record, "highlights", None) or []:
        value = str(highlight).strip().lower()
        if value.startswith("final_judge="):
            return value.split("=", 1)[1].strip()
    return ""


def _trial_record_score(raw_hit: Optional[Dict[str, Any]]) -> float:
    hit = raw_hit or {}
    for key in ("final_score", "criteria_score", "first_stage_score"):
        try:
            value = float(hit.get(key))
        except (TypeError, ValueError):
            continue
        return value
    return 0.0


def _trial_record_is_screening_candidate(record: Any, raw_hit: Optional[Dict[str, Any]] = None) -> bool:
    """Surface strong unclear matches for human screening without calling them fit."""
    decision = _trial_record_final_judge(record, raw_hit)
    if decision == "fit" or _trial_record_is_fit(record):
        return True
    if decision != "unclear":
        return False
    return _trial_record_score(raw_hit) >= 0.90


def _trial_bundle_source_payload(bundle: Any) -> Dict[str, Any]:
    debug = dict(getattr(bundle, "debug", None) or {})
    return dict(debug.get("source_payload") or {})


def _build_trial_hit_index(bundle: Any) -> Dict[Tuple[str, str], Dict[str, Any]]:
    sources = (_trial_bundle_source_payload(bundle).get("sources") or {})
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for source, payload in sources.items():
        if not isinstance(payload, dict):
            continue
        for hit in list(payload.get("trial_hits") or []):
            if not isinstance(hit, dict):
                continue
            keys = [
                str(hit.get("trial_id") or "").strip(),
                str(hit.get("source_trial_id") or "").strip(),
                str(hit.get("brief_title") or "").strip(),
                str(hit.get("title") or "").strip(),
                str(hit.get("official_title") or "").strip(),
            ]
            for key in keys:
                if key:
                    lookup[(str(source), key)] = hit
    return lookup


def _lookup_trial_hit(record: Any, hit_index: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [
        _trial_record_source_id(record),
        str(getattr(record, "citation", "") or "").strip(),
        str(getattr(record, "title", "") or "").strip(),
    ]
    backend = str(getattr(record, "backend_name", "") or "").strip()
    for key in candidates:
        hit = hit_index.get((backend, key))
        if hit:
            return hit
    return {}


def _serialize_fit_trial_record(record: Any, *, raw_hit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    provenance = getattr(record, "provenance", None)
    hit = raw_hit or {}
    source_id = _trial_record_source_id(record)
    citation = str(getattr(record, "citation", "") or "").strip()
    return {
        "source_id": source_id,
        "source_trial_id": str(hit.get("source_trial_id") or citation or source_id).strip(),
        "backend": str(getattr(record, "backend_name", "") or hit.get("source") or "").strip(),
        "title": str(hit.get("title") or hit.get("brief_title") or hit.get("official_title") or getattr(record, "title", "") or "").strip(),
        "official_title": str(hit.get("official_title") or "").strip(),
        "phase": str(hit.get("phase") or "").strip(),
        "status": str(hit.get("status") or "").strip(),
        "sponsor": str(hit.get("sponsor") or "").strip(),
        "decision": str(hit.get("final_judge_decision") or hit.get("decision") or ("fit" if _trial_record_is_fit(record) else "candidate")).strip(),
        "reason": str(hit.get("final_judge_reason") or hit.get("reason") or getattr(record, "selection_rationale", "") or getattr(record, "summary", "") or "").strip(),
        "summary": str(hit.get("summary") or hit.get("brief_summary") or getattr(record, "summary", "") or "").strip(),
        "description": str(hit.get("description") or hit.get("detailed_description") or "").strip(),
        "highlights": [str(highlight) for highlight in (getattr(record, "highlights", None) or [])],
        "citation": citation,
        "source_url": str(hit.get("source_url") or hit.get("trial_url") or getattr(provenance, "source_url", None) or "").strip(),
        "selection_rationale": str(getattr(record, "selection_rationale", "") or ""),
        "updated": str(hit.get("updated") or "").strip(),
        "match_signals": [str(item).strip() for item in list(hit.get("match_signals") or hit.get("final_judge_matches") or []) if str(item).strip()],
        "conflict_signals": [str(item).strip() for item in list(hit.get("conflict_signals") or hit.get("final_judge_conflicts") or []) if str(item).strip()],
        "matched_inclusion_criteria": list(hit.get("matched_inclusion_criteria") or []),
        "matched_exclusion_criteria": list(hit.get("matched_exclusion_criteria") or []),
        "interventions": list(hit.get("interventions") or []),
        "contact_items": list(hit.get("contact_items") or []),
        "design": dict(hit.get("design") or {}),
        "eligibility": dict(hit.get("eligibility") or {}),
        "locations": list(hit.get("locations") or []),
        "scores": {
            "first_stage": hit.get("first_stage_score"),
            "criteria": hit.get("criteria_score"),
            "final": hit.get("final_score"),
        },
    }


def _store_fit_trials_in_reference_cache(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    try:
        from utils.reference_cache import get_reference_cache

        cache = get_reference_cache()
        for record in records:
            trial_id = str(record.get("source_id") or "").strip()
            if not trial_id:
                continue
            cache.store_trial(
                trial_id=trial_id,
                name=str(record.get("title") or record.get("official_title") or trial_id),
                reason=str(record.get("reason") or record.get("selection_rationale") or record.get("summary") or ""),
                metadata=dict(record),
            )
    except Exception as exc:
        print(f"{Color.WARNING}[WARNING] Failed to cache trial references: {exc}{Color.RESET}")


def _emit_trial_cards_and_note(records: List[Dict[str, Any]]) -> str:
    if not records:
        return ""
    table = PrettyTable(["Trial ID", "Title", "Decision", "Reason"])
    table.align = "l"
    for record in records:
        table.add_row(
            [
                preview_text(record.get("source_id") or "", 18),
                preview_text(record.get("title") or record.get("official_title") or "", 60),
                preview_text(record.get("decision") or "", 12),
                preview_text(record.get("reason") or record.get("selection_rationale") or "", 80),
            ]
        )
    print(f"\n{Color.BOLD}{Color.OKBLUE}Clinical Trial Candidates ({len(records)}){Color.RESET}")
    print(table)
    trial_lines = [
        f"- Candidate trial: [@trial | {record['source_id']}]  {record['title']}"
        for record in records
        if record.get("source_id") and record.get("title")
    ]
    return "Candidate clinical trials:\n" + "\n".join(trial_lines) if trial_lines else ""


def _run_chair_gated_trial_query(
    *,
    chair_agent: "Agent",
    question_str: str,
    trace: Optional["TraceLogger"],
    trial_query: Optional[str] = None,
    mdt_context: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    trial_note = ""
    fit_trial_records: List[Dict[str, Any]] = []
    cleaned_trial_query = _clean_trial_query_value(trial_query)
    query_for_trial = cleaned_trial_query or question_str
    mdt_context_text = (mdt_context or "No explicit MDT discussion context provided.").strip()

    trial_backend = _normalize_trial_backend(os.environ.get("OMGS_TRIAL_BACKEND", "local"))
    if trace:
        trace.emit("trial_gate_start", {"backend": trial_backend})
    print(f"{Color.OKBLUE}[TRIAL] Clinical trial screening started: backend={trial_backend}{Color.RESET}")

    chair_gate_prompt = (
        f"MDT_CONTEXT:\n{mdt_context_text}\n\n"
        f"TRIAL_QUERY:\n{query_for_trial}\n\n"
        "Based only on the above, should this patient be screened for available clinical trials? "
        "Answer ONLY 'yes' or 'no' with one brief reason."
    )
    try:
        gate_response = chair_agent.chat_once(chair_gate_prompt).strip().lower()
        wants_trial = gate_response.startswith("y")
        if trace:
            trace.emit(
                "trial_gate_response",
                {"wants_trial": wants_trial, "response": gate_response[:200]},
            )
        print(
            f"{Color.OKBLUE}[TRIAL] Chair gate: {'yes' if wants_trial else 'no'} "
            f"({preview_text(gate_response, 120)}){Color.RESET}"
        )
    except Exception as exc:
        wants_trial = False
        print(f"{Color.WARNING}[TRIAL] Chair gate failed; skipping trial query: {exc}{Color.RESET}")
        if trace:
            trace.emit(
                "trial_gate_response",
                {"wants_trial": False, "error": str(exc), "error_type": type(exc).__name__},
            )

    if not wants_trial:
        if trace:
            trace.emit(
                "trial_query_end",
                {
                    "backend": trial_backend,
                    "skipped": True,
                    "reason": "chair_declined",
                    "fit_count": 0,
                    "fit_trial_ids": [],
                    "fit_trial_records": [],
                    "trial_recommended": False,
                },
            )
        return trial_note, fit_trial_records

    try:
        print(f"{Color.OKBLUE}[TRIAL] Querying trial engine...{Color.RESET}")
        if trace:
            trace.emit(
                "trial_query_start",
                {
                    "query": query_for_trial,
                    "backend": trial_backend,
                    "query_source": "rag_query" if cleaned_trial_query else "question_str",
                },
            )
        trial_bundle = _query_trial_bundle(query_for_trial, backend=trial_backend)
        hit_index = _build_trial_hit_index(trial_bundle)
        all_records = list(getattr(trial_bundle, "records", None) or [])
        screened_records: List[Tuple[Any, Dict[str, Any]]] = []
        for record in all_records:
            raw_hit = _lookup_trial_hit(record, hit_index)
            if _trial_record_is_screening_candidate(record, raw_hit):
                screened_records.append((record, raw_hit))
        fit_trial_records = [
            _serialize_fit_trial_record(record, raw_hit=raw_hit)
            for record, raw_hit in screened_records[:5]
        ]
        _store_fit_trials_in_reference_cache(fit_trial_records)
        print(
            f"{Color.OKBLUE}[TRIAL] Trial engine returned {len(all_records)} records; "
            f"{len(fit_trial_records)} candidate(s) surfaced.{Color.RESET}"
        )

        if trace:
            trace.emit(
                "trial_query_end",
                {
                    "backend": trial_backend,
                    "record_count": len(all_records),
                    "fit_count": len(fit_trial_records),
                    "fit_trial_ids": [record["source_id"] for record in fit_trial_records],
                    "fit_trial_records": fit_trial_records,
                    "trial_recommended": bool(fit_trial_records),
                },
            )

        trial_note = _emit_trial_cards_and_note(fit_trial_records)
        return trial_note, fit_trial_records
    except Exception as exc:
        print(f"{Color.WARNING}[WARNING] Trial engine query failed: {exc}{Color.RESET}")
        if trace:
            trace.emit(
                "trial_query_end",
                {
                    "backend": trial_backend,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "record_count": 0,
                    "fit_count": 0,
                    "fit_trial_ids": [],
                    "fit_trial_records": [],
                    "trial_recommended": False,
                },
            )
        return trial_note, fit_trial_records

###############################################################################
# 6. MULTI-ROUND MDT DISCUSSION ENGINE
###############################################################################
def run_mdt_discussion(
    agents: Dict[str, "Agent"],
    assistant: "Agent",
    num_rounds: int = 2,
    num_turns: int = 2,
    max_merged_chars: int = 10000,
    max_turn_delta_chars: int = 900,
    max_targets_per_speaker: int = 4,
    visit_time: Optional[str] = None,
    trace: Optional["TraceLogger"] = None,
    reference_tags_str: Optional[str] = None,
    reference_tags_by_role: Optional[Dict[str, str]] = None,
    private_reference_raw_by_tag: Optional[Dict[str, Dict[str, Any]]] = None,
    public_evidence_atoms: Optional[List[Dict[str, str]]] = None,
    after_initial_summary_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, str], str, Dict[str, Dict[str, str]], Dict[str, Dict[str, Dict[str, Dict[str, Optional[str]]]]]]:
    """
    Run multi-round MDT discussion engine with role-based agents.
    
    Note: Each agent's system prompt already contains case_view + clinical reports + 
    global_guideline_digest, so we don't need to repeat the full question in each round.
    
    Args:
        agents: Dictionary mapping role names to Agent instances
        assistant: Assistant agent for summarizing discussions
        num_rounds: Number of discussion rounds
        num_turns: Number of turns per round
        max_merged_chars: Maximum characters to keep in merged context
        max_turn_delta_chars: Deprecated compatibility parameter; cleaned transcript is bounded by max_merged_chars
        max_targets_per_speaker: Maximum targets a speaker can address per turn
        visit_time: Optional visit timestamp for temporal context
        reference_tags_str: Optional exact RAG evidence tags available for discussion citations
    
    Returns:
        Tuple of (initial_ops, merged_context, final_round_ops, interaction_log)
    """

    def _clip_tail(s: str, max_chars: int) -> str:
        if not s:
            return ""
        return s if len(s) <= max_chars else s[-max_chars:]

    def _append_bounded(base: str, addition: str, max_chars: int) -> str:
        if not addition:
            return _clip_tail(base, max_chars)
        if not base:
            return _clip_tail(addition, max_chars)
        return _clip_tail(base + "\n" + addition, max_chars)

    def _clip_head(s: str, max_chars: int) -> str:
        if not s:
            return ""
        return s if len(s) <= max_chars else s[:max_chars]
    
    # pack the context of the MDT:
    # (current compact memory)
    # [MDT_DISCUSSION_TRANSCRIPT]
    # (cleaned effective messages only)
    def _pack_context(memory: str, transcript: str, max_chars: int, memory_ratio: float = 0.75) -> str:
        """Keep structured memory in the FRONT and cleaned discussion transcript in the TAIL."""
        memory = memory or ""
        transcript = transcript or ""
        if not transcript:
            return _clip_head(memory, max_chars)

        sep = "\n\n[MDT_DISCUSSION_TRANSCRIPT]\n"
        # Allocate budget: preserve memory first
        mem_budget = max(200, int(max_chars * memory_ratio))
        # Remaining budget goes to transcript (+separator)
        transcript_budget = max(0, max_chars - min(len(memory), mem_budget) - len(sep))

        mem_part = _clip_head(memory, mem_budget)
        transcript_part = _clip_tail(transcript, transcript_budget)
        if not transcript_part:
            return _clip_head(mem_part, max_chars)
        return _clip_head(mem_part + sep + transcript_part, max_chars)

    # Load MDT prompts from config
    mdt_prompts = get_mdt_prompts().get("mdt_discussion", {})
    reference_tags_str = (reference_tags_str or "").strip()
    reference_tags_by_role = reference_tags_by_role or {}
    private_reference_raw_by_tag = private_reference_raw_by_tag or {}
    if public_evidence_atoms is None:
        public_evidence_atoms = []
    public_evidence_seen = {
        _normalize_reference_tag(atom.get("tag") or "")
        for atom in public_evidence_atoms
        if atom.get("tag")
    }

    def _reference_tags_for_role(role: str) -> str:
        base = (reference_tags_by_role.get(role) or reference_tags_str or "").strip()
        public_block = _format_public_evidence_atoms(public_evidence_atoms)
        if public_block:
            base = (base + "\n\nROLE_SCOPED_CITED_EVIDENCE:\n" + public_block).strip()
        return base

    def _publish_public_evidence_from_message(msg: str) -> None:
        for tag in extract_reference_tags(msg or ""):
            norm = _normalize_reference_tag(tag)
            if not norm or norm in public_evidence_seen:
                continue
            raw = private_reference_raw_by_tag.get(norm)
            if not raw:
                continue
            public_evidence_atoms.append(_build_public_evidence_atom(tag, raw, msg))
            public_evidence_seen.add(norm)
    
    print(f"{Color.OKCYAN}{Color.BOLD}🧠 Starting MDT Discussion Engine...{Color.RESET}")
    agent_list = list(agents.keys())

    emoji_pool = ["👨‍⚕️","👩‍⚕️","🧑‍⚕️","👨🏻‍⚕️","👩🏼‍⚕️","👨🏽‍⚕️","👩🏽‍⚕️","👨🏾‍⚕️","👩🏾‍⚕️","🧑🏾‍⚕️"]
    random.shuffle(emoji_pool)
    role_to_emoji = {r: emoji_pool[i % len(emoji_pool)] for i, r in enumerate(agent_list)}
    chair_role = "chair" if "chair" in agent_list else agent_list[0]

    last_msg_by_pair = {} # record the last message by pair to avoid duplicate messages from the same speaker-target pair, which helps reduce redundancy but may suppress necessary repeated clarifications in some scenarios; adjust based on actual usage.

    # INITIAL OPINIONS
    print(f"{Color.BOLD}{Color.OKBLUE}\n📌 Collecting Initial Opinions...{Color.RESET}")
    initial_ops = {} # we will use this to store the initial opinions of the experts
    initial_opinion_prompt = mdt_prompts.get("initial_opinion", 
        "Give INITIAL opinion (use ONLY your system-provided patient facts).\n"
        "Return up to 3 bullets, each ≤20 words.\n"
        "Use valid Markdown bullets: each bullet starts with '- ' on its own line; do not compress bullets into one paragraph.\n"
        "State your current stance or domain implication for today's treatment choice when relevant.\n"
        "If key data missing, name it only if it would change today's treatment choice or safety.\n"
        "Do NOT ask for administrative report_ids, source labels, or already-present data.\n"
        "At least ONE bullet must be evidence-based and include an exact guideline tag from the digest or [@pubmed | PMID].\n"
        "If you reference treatment strategy categories, guidelines, trials, or literature evidence, include exact guideline tags, [@pubmed | PMID], or [@trial | id]."
    )
    # initial_ops is a dictionary that stores the initial opinions of the experts
    for i, (role, ag) in enumerate(agents.items(), start=1):
        print(f"{Color.OKGREEN} [{i}/{len(agents)}] {role}:{Color.RESET}")
        if trace:
            trace.emit("mdt_initial_opinion_role_start", {"role": role, "order": i})
        
        # Use safe_agent_call for error handling
        try:
            op = ag.chat(initial_opinion_prompt) # get the initial opinion of the expert
            role_reference_tags = _reference_tags_for_role(role)
            op = _sanitize_discussion_evidence_tags(
                op,
                _reference_tag_set(role_reference_tags),
                role=role,
                stage="initial_opinion",
                trace=trace,
            )
            _publish_public_evidence_from_message(op)
            if trace:
                trace.emit("mdt_initial_opinion_role_end", {"role": role, "chars": len(op or '')})
            print(f"{Color.OKCYAN}   {role_to_emoji[role]} {op}{Color.RESET}")
            warn_missing_evidence_tags(op, role=f"{role}/initial", trace=trace) # warn the missing evidence tags in the initial opinion
            initial_ops[role] = op # store the initial opinion of the expert in the initial_ops dictionary
        except AgentError as e:
            error_msg = get_fallback_response(role, "initial_opinion")
            print(f"{Color.WARNING}[WARNING] {role} failed: {e.original_error}{Color.RESET}")
            print(f"{Color.OKCYAN}   {role_to_emoji[role]} {error_msg}{Color.RESET}")
            initial_ops[role] = error_msg
            if trace:
                trace.emit("agent_error", {
                    "role": role,
                    "stage": "initial_opinion",
                    "error": str(e.original_error),
                    "error_type": type(e.original_error).__name__
                })
        except Exception as e:
            # Catch any unexpected exceptions
            error_msg = get_fallback_response(role, "initial_opinion")
            print(f"{Color.FAIL}[ERROR] {role} unexpected error: {e}{Color.RESET}")
            print(f"{Color.OKCYAN}   {role_to_emoji[role]} {error_msg}{Color.RESET}")
            initial_ops[role] = error_msg
            if trace:
                trace.emit("agent_error", {
                    "role": role,
                    "stage": "initial_opinion",
                    "error": str(e),
                    "error_type": "UnexpectedException"
                })

    summarize_template = mdt_prompts.get("summarize_initial_template",
        "Summarize expert opinions concisely for MDT.\n{opinions}\n\n"
        "Output:\nKey Knowledge:\n- ...\nControversies:\n- ...\nMissing Info:\n- ...\nWorking Plan:\n- ..."
        "\nDebate Focus:\n- one clinical question the next turn should resolve"
    )
    # summarize the initial opinions of the experts
    # merged is the structured memory of the MDT
    fallback_merged = f"Key Knowledge:\n{json.dumps(initial_ops, ensure_ascii=False, indent=2)}\n\nControversies:\n- To be determined\n\nMissing Info:\n- To be determined\n\nWorking Plan:\n- To be determined"
    
    merged = safe_agent_call(
        agent=assistant,
        prompt=summarize_template.format(opinions=json.dumps(initial_ops, ensure_ascii=False, separators=(',', ':'))),
        role="assistant",
        stage="initial_summary",
        fallback=fallback_merged,
        trace=trace,
        max_retries=3,  # Retry up to 3 times for rate limits (429 errors) with exponential backoff
        use_once=True,
        timeout=_mdt_synthesis_timeout_seconds(),
    )
    # Structured MDT memory (always kept at the front)
    memory_state = _clip_head(merged, max_merged_chars)
    # Cleaned transcript is the explicit discussion memory that replaces
    # private agent chat history for debate turns.
    discussion_transcript = ""
    merged = _pack_context(memory_state, discussion_transcript, max_merged_chars)
    initial_summary_payload = {
        "initial_ops": dict(initial_ops),
        "merged": merged,
        "final_round_ops": {},
        "interaction_log": {},
    }
    if after_initial_summary_callback is not None:
        after_initial_summary_callback(initial_summary_payload)

    interaction_log = {
        f"Round {r}": {
            f"Turn {t}": {s: {d: None for d in agent_list} for s in agent_list}
            for t in range(1, num_turns + 1)
        }
        for r in range(1, num_rounds + 1)
    }
    final_round_ops = {}

    # Load prompt templates
    speak_prompt_template = mdt_prompts.get("speak_prompt_template",
        "ROLE: {role}. VISIT: {visit_time}\n"
        "Default is NOT to speak. Speak ONLY if you can resolve/challenge a clinical conflict, safety issue, decision-changing missing item, or new-critical evidence.\n"
        "Do NOT speak just to request administrative report_ids, source labels, or data already present.\n"
        "Do NOT merely repeat points already captured in MDT_DISCUSSION_TRANSCRIPT.\n\n"
        "CONTEXT (latest):\n{context}\n\n"
        "Allowed targets: [{allowed_targets}]\n"
        "REFERENCE TAGS (copy exactly if used):\n{reference_tags}\n"
        "Guideline/PubMed/FDA/conference/trial evidence must use exact tags from REFERENCE TAGS; never invent evidence tags.\n"
        'Return ONE-LINE JSON only:{{"speak":"yes/no","messages":[{{"target":"<role>","message":"<1-2 sentences>","why":"conflict|safety|missing|new"}}]}}'
    )
    round_synthesis_template = mdt_prompts.get(
        "round_synthesis_template",
        "Update MDT global knowledge concisely by integrating the explicit MDT discussion transcript.\n"
        "Do NOT decide treatment independently.\n"
        "Do NOT introduce facts not present in CURRENT_MDT_GLOBAL_KNOWLEDGE or MDT_DISCUSSION_TRANSCRIPT.\n"
        "Preserve useful citations.\n"
        "Preserve uncertainty: do not strengthen \"consider/possible/conditional\" into \"recommend/confirmed\" unless MDT_DISCUSSION_TRANSCRIPT explicitly did so.\n"
        "Consider ALL non-no-objection messages in MDT_DISCUSSION_TRANSCRIPT.\n\n"
        "CURRENT_MDT_GLOBAL_KNOWLEDGE:\n{memory_state}\n\n"
        "MDT_DISCUSSION_TRANSCRIPT:\n{discussion_context}\n\n"
        "Output:\n"
        "Key Knowledge:\n- ...\n"
        "Decision Stances:\n- role -> current stance and reason\n"
        "Controversies:\n- concrete option A vs option B conflict to resolve, or \"None\" if consensus is confirmed\n"
        "Missing Info:\n- only decision-changing missing data; omit administrative source/report_id gaps\n"
        "Working Plan:\n- ...\n"
        "Debate Impact:\n- what changed, narrowed, stayed conditional, or remained unresolved after this round"
    )

    # MAIN DISCUSSION
    # Rounds
    for r in range(1, num_rounds + 1):
        print(f"{Color.WARNING}{Color.BOLD}\n==================== ROUND {r} ===================={Color.RESET}")
        round_key = f"Round {r}"

        # Carry forward the explicit structured memory. Avoid re-summarizing
        # summary-only state before any new debate has occurred in this round.
        summary = memory_state
        memory_state = _clip_head(f"[MDT_GLOBAL_KNOWLEDGE]\n{summary}", max_merged_chars)
        merged = _pack_context(memory_state, discussion_transcript, max_merged_chars)

        MDT_should_stop = False
        round_had_speakers = False
        # Turns
        for t in range(1, num_turns + 1):
            print(f"{Color.BOLD}{Color.OKCYAN}\n--- Turn {t} ---{Color.RESET}")
            turn_key = f"Turn {t}"
            num_speakers = 0
            turn_transcript_lines = []
            merged = _pack_context(memory_state, discussion_transcript, max_merged_chars)
            ctx_for_turn = merged

            for role, ag in agents.items():
                allowed_targets = [x for x in agent_list if x != role]
                allowed_targets_str = ",".join(allowed_targets)
                role_reference_tags = _reference_tags_for_role(role)
                allowed_reference_tags = _reference_tag_set(role_reference_tags)

                speak_prompt = speak_prompt_template.format(
                    role=role,
                    visit_time=visit_time or 'Unknown',
                    context=ctx_for_turn,
                    allowed_targets=allowed_targets_str,
                    reference_tags=role_reference_tags or "(No guideline/literature reference tags available)"
                )

                try:
                    resp = ag.chat_once(speak_prompt)
                    data = safe_parse_json_block(resp)
                except AgentError as e:
                    # Skip this agent's turn if it fails
                    if trace:
                        trace.emit("agent_error", {
                            "role": role,
                            "stage": f"turn_{t}",
                            "error": str(e.original_error),
                            "error_type": type(e.original_error).__name__
                        })
                    continue
                except Exception as e:
                    # Skip this agent's turn for unexpected errors
                    print(f"{Color.WARNING}[WARNING] {role} failed to speak in turn {t}: {e}{Color.RESET}")
                    if trace:
                        trace.emit("agent_error", {
                            "role": role,
                            "stage": f"turn_{t}",
                            "error": str(e),
                            "error_type": "UnexpectedException"
                        })
                    continue

                if data and str(data.get("speak", "no")).lower() != "yes":
                    no_objection = "I have no additional objections from my role."
                    print(f"{Color.OKGREEN}  {role_to_emoji[role]} {role}:{Color.RESET} [no_objection] {no_objection}")
                    if trace:
                        trace.emit(
                            "mdt_no_objection",
                            {
                                "round": r,
                                "turn": t,
                                "role": role,
                            },
                        )
                    continue
                if not data:
                    continue

                msgs = data.get("messages", None)
                if not isinstance(msgs, list):
                    old_msg = (data.get("message") or "").strip()
                    old_targets = data.get("targets") or []
                    if old_msg and isinstance(old_targets, list) and old_targets:
                        msgs = [{"target": tr, "message": old_msg, "why": "unspecified"} for tr in old_targets]
                    else:
                        continue

                accepted_any = False
                used_targets = set()

                for item in msgs:
                    if not isinstance(item, dict):
                        continue
                    target = item.get("target", None)
                    msg = (item.get("message") or "").strip()
                    why = (item.get("why") or "").strip().lower()

                    if not msg:
                        continue
                    msg = _sanitize_discussion_evidence_tags(
                        msg,
                        allowed_reference_tags,
                        role=f"{role}->{target or chair_role}",
                        stage=f"R{r}T{t}",
                        trace=trace,
                    )
                    if not msg:
                        continue
                    if why not in {"conflict", "safety", "missing", "new"}:
                        why = "unspecified"

                    if target not in allowed_targets:
                        target = chair_role
                    if target == role:
                        continue

                    if target in used_targets:
                        continue
                    if len(used_targets) >= max_targets_per_speaker:
                        break

                    key = (role, target)
                    if last_msg_by_pair.get(key) == msg:
                        continue
                    last_msg_by_pair[key] = msg

                    used_targets.add(target)
                    accepted_any = True
                    _publish_public_evidence_from_message(msg)

                    interaction_log[round_key][turn_key][role][target] = msg
                    print(f"{Color.OKGREEN}  {role_to_emoji[role]} {role} → {role_to_emoji[target]} {target}:{Color.RESET} [{why}] {msg}")
                    
                    # Optional: Warn if message mentions evidence but lacks tags
                    # (Not enforced, just a helpful reminder)
                    if trace:
                        warn_missing_evidence_tags(msg, role=f"{role}->{target}/turn_{t}", trace=trace)

                    turn_transcript_lines.append(f"[R{r}T{t}] {role} -> {target} ({why}): {msg}")

                if accepted_any:
                    num_speakers += 1

            if num_speakers == 0:
                print(f"{Color.WARNING} ⚠ No experts spoke in this turn → Skip remaining turns and finalize this round.{Color.RESET}")
                if trace:
                    trace.emit(
                        "mdt_turn_no_speakers",
                        {"round": r, "turn": t, "reason": "all_speak_no_or_failed"},
                    )
                MDT_should_stop = True
                break

            round_had_speakers = True
            if turn_transcript_lines:
                turn_transcript = "\n".join(turn_transcript_lines)
                discussion_transcript = _append_bounded(discussion_transcript, turn_transcript, max_merged_chars)
                merged = _pack_context(memory_state, discussion_transcript, max_merged_chars)

        # Synthesize this round once. The assistant acts as a stateless recorder:
        # explicit memory + this round's discussion are the continuity contract.
        final_round_ops[round_key] = {}
        
        discussion_context = discussion_transcript or "No direct discussions in this round."
        
        if round_had_speakers:
            fallback_memory = _clip_head(
                memory_state + "\n\n"
                f"Debate Impact:\n- Round {r} added substantive discussion: {_clip_tail(discussion_context, 600)}",
                max_merged_chars,
            )
            round_synthesis = safe_agent_call(
                agent=assistant,
                prompt=round_synthesis_template.format(
                    memory_state=memory_state,
                    round=r,
                    discussion_context=discussion_context,
                ),
                role="assistant",
                stage=f"round_{r}_synthesis",
                fallback=fallback_memory,
                trace=trace,
                max_retries=3,
                use_once=True,
                timeout=_mdt_synthesis_timeout_seconds(),
            )
            warn_missing_evidence_tags(round_synthesis, role=f"assistant/round_{r}_synthesis", trace=trace)
            print(f"{Color.BOLD}{Color.OKBLUE}\n📘 Round {r} Synthesis:{Color.RESET}")
            print(f"{Color.OKGREEN}{round_synthesis}{Color.RESET}\n")
            final_round_ops[round_key]["assistant"] = round_synthesis
            memory_state = _clip_head(round_synthesis, max_merged_chars)
        else:
            stop_note = "No substantive new objections emerged; the existing MDT memory stands."
            final_round_ops[round_key]["assistant"] = stop_note

        merged = _pack_context(memory_state, discussion_transcript, max_merged_chars)

        if MDT_should_stop:
            print(f"{Color.WARNING}{Color.BOLD}🚫 MDT stopped early after Round {r}. No further rounds will be executed.{Color.RESET}")
            return initial_ops, merged, final_round_ops, interaction_log
    return initial_ops, merged, final_round_ops, interaction_log

# Helper to extract genomic info from GENOMICS and top-level fields.
def _extract_genomic_status(case_json: Dict[str, Any]) -> Dict[str, str]:
    """Extract HRD/BRCA status from structured genomic fields.
    
    Returns dict with keys: HRD, BRCA1, BRCA2, other_alterations
    Prioritizes explicit values over Unknown.
    """
    case_core = case_json.get("CASE_CORE") or {}
    genomics = case_core.get("GENOMICS") or {}
    result = {
        "HRD": "Unknown",
        "BRCA1": "Unknown", 
        "BRCA2": "Unknown",
        "other_alterations": []
    }
    
    # First try top-level fields.
    top_level_hrd = case_core.get("HRD", "Unknown")
    top_level_brca1 = case_core.get("BRCA1", "Unknown")
    top_level_brca2 = case_core.get("BRCA2", "Unknown")
    
    if top_level_hrd not in ["Unknown", "Not applicable", None, ""]:
        result["HRD"] = top_level_hrd
    if top_level_brca1 not in ["Unknown", "Not applicable", None, ""]:
        result["BRCA1"] = top_level_brca1
    if top_level_brca2 not in ["Unknown", "Not applicable", None, ""]:
        result["BRCA2"] = top_level_brca2
    
    # Then try GENOMICS section (can override Unknown)
    hrd_status = genomics.get("HRD_STATUS") or {}
    if isinstance(hrd_status, dict):
        hrd_result = hrd_status.get("result", "Unknown")
        if hrd_result not in ["Unknown", "Not_applicable", None, ""] and result["HRD"] == "Unknown":
            result["HRD"] = hrd_result
    
    # Check alterations array for BRCA1/BRCA2 and other genes
    alterations = genomics.get("alterations") or []
    for alt in alterations:
        if not isinstance(alt, dict):
            continue
        gene = alt.get("gene", "").upper()
        status = alt.get("status", "Unknown")
        significance = alt.get("clinical_significance", "Unknown")
        
        # For BRCA1/BRCA2, update if we have better info
        if gene == "BRCA1" and result["BRCA1"] == "Unknown":
            if status == "Mutated" or (status != "Wildtype" and significance in ["Pathogenic", "Likely_pathogenic"]):
                result["BRCA1"] = "Mutated"
            elif status == "Wildtype":
                result["BRCA1"] = "Wildtype"
        elif gene == "BRCA2" and result["BRCA2"] == "Unknown":
            if status == "Mutated" or (status != "Wildtype" and significance in ["Pathogenic", "Likely_pathogenic"]):
                result["BRCA2"] = "Mutated"
            elif status == "Wildtype":
                result["BRCA2"] = "Wildtype"
        # Collect other HRR/actionable genes
        elif gene not in ["BRCA1", "BRCA2", ""] and status == "Mutated":
            result["other_alterations"].append(f"{gene}:{status}")
    
    return result


# important for evidence search!!!!
def _build_rag_key_facts(case_json: Dict[str, Any], mut_reports: List[Dict[str, Any]]) -> str:
    """Build KEY FACTS string for RAG query from case data.
    
    Simply includes raw mutation report text directly - let LLM parse it.
    """
    parts: List[str] = []
    case_core = case_json.get("CASE_CORE") or {}
    diagnosis = case_core.get("DIAGNOSIS") or {}
    if diagnosis:
        primary = diagnosis.get("primary") or "Unknown"
        hist = diagnosis.get("histology") or "Unknown"
        comps = diagnosis.get("components") or []
        comp_txt = f" components={';'.join([str(x) for x in comps])}" if comps else ""
        parts.append(f"DIAGNOSIS: primary={primary}; histology={hist};{comp_txt}")
    pathology = case_json.get("PATHOLOGY") or {}
    specimens = pathology.get("specimens") if isinstance(pathology, dict) else None
    if isinstance(specimens, list) and specimens:
        diag = specimens[0].get("diagnosis") or ""
        if diag:
            parts.append(f"PATHOLOGY: {preview_text(diag, 160)}")
    plat = case_core.get("PLATINUM_STATUS_CURRENT") or case_core.get("PLATINUM_STATUS")
    pfi = case_core.get("PLATINUM_PFI_CURRENT") or case_core.get("PFI_days")
    if plat or pfi:
        parts.append(f"PLATINUM: status={plat or 'Unknown'}; pfi_days={pfi or 'Unknown'}")
    
    # Only use GENETICS from case_core if NO mutation reports are available
    # If mutation reports exist, they are the source of truth - don't use case_core values
    # This prevents using "not reported" or "Unknown" from case_core when actual reports exist
    # important for evidence search!!!! must be before mutation reports are included
    if not mut_reports:
        hrd = case_core.get("HRD")
        brca1 = case_core.get("BRCA1")
        brca2 = case_core.get("BRCA2")
        substantive = [
            value
            for value in (hrd, brca1, brca2)
            if str(value or "").strip().lower()
            not in {"", "unknown", "not applicable", "not_applicable", "none", "null", "n/a", "na"}
        ]
        if substantive:
            parts.append(f"GENETICS: HRD={hrd or 'Unknown'}; BRCA1={brca1 or 'Unknown'}; BRCA2={brca2 or 'Unknown'}")
    
    biomarkers = case_core.get("BIOMARKERS") or {}
    if biomarkers:
        keys = ["CA125", "HE4", "CA19-9", "CA15-3", "AFP", "CEA", "TMB", "MSI", "PDL1_CPS"]
        items = [
            f"{k}={biomarkers.get(k)}"
            for k in keys
            if str(biomarkers.get(k) or "").strip().lower()
            not in {"", "unknown", "not reported", "not applicable", "not_applicable", "none", "null", "n/a", "na"}
        ]
        if items:
            parts.append("BIOMARKERS: " + "; ".join(items[:6]))
    
    # Include full mutation report raw_text directly - let LLM parse it
    # This takes precedence over case_core GENETICS values
    # !!!!important; mut_reports is gold criteria for evidence search!!!!
    if mut_reports:
        latest = mut_reports[-1]
        rid = latest.get("report_id") or "Unknown"
        rdate = latest.get("report_date") or ""
        raw = latest.get("raw_text") or ""
        # Include full text (up to 3000 chars to avoid token bloat, but should cover most reports)
        raw_text = preview_text(raw, 3000) if raw else ""
        parts.append(f"MUTATION_REPORT: id={rid}; date={str(rdate)[:10]}; full_text={raw_text}")
    
    return "\n".join(parts)


###############################################################################
# INTERACTION DIRECTION MATRIX（PrettyTable）
###############################################################################
def _count_interactions(
    interaction_log: Dict[str, Dict[str, Dict[str, Dict[str, Optional[str]]]]],
    src: str,
    dst: str
) -> int:
    """Count total interactions from src to dst across all rounds and turns."""
    c = 0
    for rnd in interaction_log.values():
        for turn in rnd.values():
            msg = turn.get(src, {}).get(dst)
            if msg:
                c += 1
    return c


def print_interaction_matrix(
    interaction_log: Dict[str, Dict[str, Dict[str, Dict[str, Optional[str]]]]],
    roles_order: List[str] = ROLES
) -> None:
    print(f"\n{Color.BOLD}{Color.OKBLUE}📊 Interaction Direction Matrix (All Rounds × Turns){Color.RESET}")
    print("Legend: . none | ->N A→B count | <-N B→A count | <->a/b both directions\n")

    agent_list = list(roles_order)
    tbl = PrettyTable([""] + agent_list)
    tbl.align = "c"
    tbl.hrules = ALL
    tbl.vrules = ALL
    tbl.padding_width = 1

    for A in agent_list:
        row = [A]
        for B in agent_list:
            if A == B:
                row.append("")
                continue
            a2b = _count_interactions(interaction_log, A, B)
            b2a = _count_interactions(interaction_log, B, A)
            if a2b == 0 and b2a == 0:
                cell = "."
            elif a2b > 0 and b2a == 0:
                cell = f"->{a2b}"
            elif a2b == 0 and b2a > 0:
                cell = f"<-{b2a}"
            else:
                cell = f"<->{a2b}/{b2a}"
            row.append(cell)
        tbl.add_row(row)
    print(tbl)


###############################################################################
###############################################################################
#  MAIN ENTRY
###############################################################################
###############################################################################
def process_omgs_multi_expert_query(
    question: Any,
    question_raw: Optional[str],
    model: str,
    args: Any,
    time: Optional[str] = None,
    meta_info: Optional[str] = None,
    labs_json: Optional[str] = None,
    imaging_json: Optional[str] = None,
    pathology_json: Optional[str] = None,
    mutation_json: Optional[str] = None,
    device: str = "auto",
    topk: int = 5,
    case_filter_buffer_days: int = 120,
    strict_context_prune: bool = False,
) -> str:
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== MDT Multi-Expert Pipeline Start ==={Color.RESET}")
    
    # Load paths configuration
    paths_config = get_paths_config()
    # print(paths_config)
    # Use config paths if not explicitly provided (backward compatibility)
    if labs_json is None:
        labs_json = paths_config["data_files"]["lab_reports"]
    if imaging_json is None:
        imaging_json = paths_config["data_files"]["imaging_reports"]
    if pathology_json is None:
        pathology_json = paths_config["data_files"]["pathology_reports"]
    if mutation_json is None:
        mutation_json = paths_config["data_files"]["mutation_reports"]
    
    # --- Visualization switches (no functional impact) ---
    visual = VisualConfig(
        enable=True,
        show_tables=True,
        show_rag_table=True,
        show_token_budget=False,
    )

    # Trace collection is always ON
    trace = TraceLogger(enabled=True)
    trace.emit("pipeline_start", {
        "visit_time": str(time) if time else None,
        "meta_info": str(meta_info),
        "run_id": str(getattr(args, "run_id", "") or ""),
    })

    print_section("MDT PIPELINE", "Observability ON (always)")

    client = args.client
    print(f"{Color.OKBLUE}{Color.BOLD}🕒 Query Time: {time}{Color.RESET}")

    # Normalize question (supports dict/list/str), strip observability-only
    # fields, and compute stable CASE fingerprint from decision inputs only.
    raw_question_str = question_to_text(question)
    case_json = sanitize_case_for_decision(safe_load_case_json(raw_question_str))
    question_str = question_to_text(case_json) if case_json else raw_question_str
    case_fingerprint = hashlib.sha1(question_str.encode("utf-8")).hexdigest()[:12]

    print(f"{Color.OKBLUE}{Color.BOLD}🧾 CASE_FINGERPRINT: {case_fingerprint}{Color.RESET}")
    trace.emit("case_fingerprint", {"case_fingerprint": case_fingerprint})

    ###########################################################################
    # LOAD REPORTS (unchanged logic except already improved date handling)
    ###########################################################################
    print_section("1) Load Clinical Reports")

    cutoff_dt = make_cutoff(time, days_after=1)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S") if cutoff_dt else "None"
    print(f"{Color.OKBLUE}{Color.BOLD}⏱️  CUTOFF_DT (time + 1d): {cutoff_str}{Color.RESET}")
    
    # Load reports with error handling - each load is independent
    # If a file is missing, the function returns empty lists, so we can continue
    try:
        lab_timeline_raw, lab_reports = load_patient_labs(meta_info, labs_json)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to load lab reports: {e}. Continuing with empty lab data.{Color.RESET}")
        lab_timeline_raw, lab_reports = [], []
    
    try:
        im_timeline_raw, im_reports = load_patient_imaging(meta_info, imaging_json)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to load imaging reports: {e}. Continuing with empty imaging data.{Color.RESET}")
        im_timeline_raw, im_reports = [], []

    path_timeline_raw, path_reports = [], []
    if pathology_json:
        try:
            path_timeline_raw, path_reports = load_patient_pathology(meta_info, pathology_json)
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to load pathology reports: {e}. Continuing with empty pathology data.{Color.RESET}")
            path_timeline_raw, path_reports = [], []

    mut_reports: List[Dict[str, Any]] = []
    if meta_info and mutation_json:
        try:
            mut_reports = load_patient_mutations(meta_info, mutation_json)
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to load mutation reports: {e}. Continuing with empty mutation data.{Color.RESET}")
            mut_reports = []

    try:
        print(f"{Color.OKCYAN}{Color.BOLD}[LAB] before filter: {report_range(lab_reports, 'report_date')}{Color.RESET}")
        print(f"{Color.OKCYAN}{Color.BOLD}[IMG] before filter: {report_range(im_reports, 'report_date')}{Color.RESET}")
        if pathology_json:
            print(f"{Color.OKCYAN}{Color.BOLD}[PATH] before filter: {report_range(path_reports, 'report_date')}{Color.RESET}")
        if mut_reports:
            print(f"{Color.OKCYAN}{Color.BOLD}[MUT] before filter: {report_range(mut_reports, 'report_date')}{Color.RESET}")
    except:
        pass

    trace.emit("reports_loaded", {
        "lab_n": len(lab_reports),
        "img_n": len(im_reports),
        "path_n": len(path_reports) if pathology_json else 0,
        "mut_n": len(mut_reports) if mut_reports else 0,
        "cutoff_dt": cutoff_str,
    })

    if cutoff_dt is not None:
        lab_reports = filter_before(lab_reports, "report_date", cutoff_dt)
        im_reports = filter_before(im_reports, "report_date", cutoff_dt)
        path_reports = filter_before(path_reports, "report_date", cutoff_dt)
        mut_reports = filter_before(mut_reports, "report_date", cutoff_dt)

    # rebuild fresh timelines
    lab_timeline = build_lab_timeline(lab_reports)
    im_timeline = build_imaging_timeline(im_reports)
    path_timeline = build_pathology_timeline(path_reports) if pathology_json else []

    ###########################################################################
    # REPORT SELECTION
    ###########################################################################
    context: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "lab": {},
        "imaging": {},
        "pathology": {},
        "mutation": {},
    }

    print_section("2) Report Selection (per Role)")
    context = select_reports_for_roles(
        roles=ROLES,
        role_permissions=ROLE_PERMISSIONS,
        lab_timeline=lab_timeline,
        lab_reports=lab_reports,
        im_timeline=im_timeline,
        im_reports=im_reports,
        path_timeline=path_timeline,
        path_reports=path_reports,
        mut_reports=mut_reports,
        pathology_json=pathology_json,
        agent_class=Agent,
        expert_select_fn=expert_select_reports,
        model=model,
        client=client,
        color=Color,
        use_shared=True,
    )

    trace.emit("reports_selected", summarize_selected_reports(context))
    if visual.show_tables:
        print_selected_reports_table(context, roles=ROLES)

    ###########################################################################
    # GLOBAL GUIDELINE RAG
    ###########################################################################
    print_section("3) Guideline + PubMed RAG")
    # Load agent prompts from config
    agent_prompts = get_mdt_prompts().get("agents", {})
    
    rag_query_builder = Agent(
        instruction=agent_prompts.get("rag_query_builder", 
            "Construct concise English MDT guideline query."),
        role="rag_query_builder",
        model_info=model,
        client=client,
        max_tokens=5000,
        max_prompt_tokens=20000,
    )
    
    rag_key_facts = _build_rag_key_facts(case_json, mut_reports)
    trace.emit("rag_key_facts", {"facts": rag_key_facts})
    
    # Build RAG query with error handling
    # IMPORTANT: Inject HRD/BRCA values from GENOMICS section or mutation reports
    # to override "Unknown" values that confuse the LLM
    rag_question_str = question_str  # Default to original
    
    # First try to get genomic info from structured GENOMICS section
    genomic_status = _extract_genomic_status(case_json)
    
    import copy
    rag_case_json = copy.deepcopy(case_json)
    
    # Apply GENOMICS-derived values
    if genomic_status["HRD"] != "Unknown":
        rag_case_json.setdefault("CASE_CORE", {})["HRD"] = genomic_status["HRD"]
    if genomic_status["BRCA1"] != "Unknown":
        rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = genomic_status["BRCA1"]
    if genomic_status["BRCA2"] != "Unknown":
        rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = genomic_status["BRCA2"]
    
    # Fallback: If still Unknown, try to extract from raw mutation reports
    if mut_reports and (genomic_status["HRD"] == "Unknown" or genomic_status["BRCA1"] == "Unknown" or genomic_status["BRCA2"] == "Unknown"):
        latest_mut = mut_reports[-1]
        raw_text = latest_mut.get("raw_text", "")
        if raw_text:
            # Extract HRD status
            if "HRD" in raw_text and genomic_status["HRD"] == "Unknown":
                if "阴性" in raw_text or "negative" in raw_text.lower():
                    rag_case_json.setdefault("CASE_CORE", {})["HRD"] = "Negative"
                elif "阳性" in raw_text or "positive" in raw_text.lower():
                    rag_case_json.setdefault("CASE_CORE", {})["HRD"] = "Positive"
            # Extract BRCA1 status
            if "BRCA1" in raw_text and genomic_status["BRCA1"] == "Unknown":
                if any(x in raw_text for x in ["未检出", "阴性", "视为阴性"]):
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = "Negative"
                elif "突变" in raw_text and "致病" in raw_text:
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = "Positive"
            # Extract BRCA2 status
            if "BRCA2" in raw_text and genomic_status["BRCA2"] == "Unknown":
                if any(x in raw_text for x in ["未检出", "阴性", "视为阴性"]):
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = "Negative"
                elif "突变" in raw_text and "致病" in raw_text:
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = "Positive"
    
    rag_question_str = json.dumps(rag_case_json, ensure_ascii=False)
    
    try:
        rag_query = build_rag_query_for_mdt(rag_query_builder, rag_question_str, key_facts=rag_key_facts)
        print(f"{Color.OKCYAN}[RAG] Query preview: {preview_text(rag_query, 220)}{Color.RESET}")
    except Exception as e:
        # Fallback: use simplified query from case JSON
        print(f"{Color.WARNING}[WARNING] RAG query builder failed: {e}{Color.RESET}")
        case_core = case_json.get("CASE_CORE", {}) or {}
        diagnosis = case_core.get("DIAGNOSIS", {}) or {}
        primary = diagnosis.get("primary", "ovarian cancer")
        rag_query = f"{primary} treatment guidelines"
        if trace:
            trace.emit("pipeline_error", {
                "stage": "rag_query_build",
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback_used": True
            })
    
    # Use global guideline RAG (respects config: use_per_role_rag / default_role)
    # NCCN rules are loaded separately for safety/conflict
    from servers.evidence_retrieval import (
        mdt_external_pubmed_topk_from_engine_depth,
        retrieve_mdt_evidence_sources,
        retrieve_mdt_external_evidence_lanes,
        retrieve_mdt_guideline_evidence_lanes,
    )
    external_pubmed_topk = mdt_external_pubmed_topk_from_engine_depth()

    evidence_mode = _normalize_mdt_evidence_mode(os.environ.get("OMGS_MDT_EVIDENCE_MODE"))
    ablation_enabled = bool(getattr(args, "omgs_ablation_outputs", False))
    _ensure_ablation_supported_for_evidence_mode(ablation_enabled, evidence_mode)
    trace.emit("mdt_evidence_mode", {"mode": evidence_mode})
    # RAG retrieval: NCCN, guidelines, and external evidence run concurrently.
    rag_pack, rag_raw, rag_sources = retrieve_mdt_evidence_sources(
        case_question=rag_case_json,
        rag_query=rag_query,
        device=device,
        guideline_topk=topk,
        nccn_topk=3,
        external_topk=external_pubmed_topk,
        nccn_unavailable_pack="(NCCN: constraints unavailable)",
        guideline_unavailable_pack="(Guideline: evidence unavailable)",
        external_unavailable_pack="(PUBMED: no evidence found)",
    )
    nccn_pack, nccn_raw = rag_sources["nccn"]
    guideline_pack, guideline_raw = rag_sources["guideline"]
    pubmed_pack, pubmed_raw = rag_sources["external"]
    authority_pack = _join_pack_sections(
        f"# NCCN CONSTRAINTS\n{nccn_pack}",
        f"# GUIDELINE EVIDENCE\n{guideline_pack}",
    )
    authority_raw = _dedupe_rag_raw_by_tag(list(nccn_raw or []) + list(guideline_raw or []))
    role_external_queries: Dict[str, str] = {}
    role_guideline_raw_by_role: Dict[str, List[Dict[str, Any]]] = {}
    role_guideline_digest_by_role_private: Dict[str, str] = {}
    role_external_raw_by_role: Dict[str, List[Dict[str, Any]]] = {}
    role_external_digest_by_role: Dict[str, str] = {}
    private_reference_raw_by_tag: Dict[str, Dict[str, Any]] = {}
    public_evidence_atoms: List[Dict[str, str]] = []
    all_reference_raw_for_cache = list(rag_raw or [])
    
    # Store RAG results in reference cache for later retrieval
    try:
        from utils.reference_cache import get_reference_cache
        ref_cache = get_reference_cache()
        ref_cache.store_rag_results(rag_raw)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to cache RAG references: {e}{Color.RESET}")
    
    trace.emit("rag_query", {"query": rag_query})
    trace.emit("rag_hits", {"source": "nccn", "topk": 3, "n": len(nccn_raw or [])})
    trace.emit("rag_hits", {"source": "guideline", "topk": topk, "n": len(guideline_raw or [])})
    trace.emit("rag_hits", {"source": "pubmed", "topk": external_pubmed_topk, "n": len(pubmed_raw or [])})
    trace.emit("rag_hits_merged", {"n": len(rag_raw or [])})
    if visual.enable and visual.show_tables and visual.show_rag_table:
        print_rag_hits_table(rag_raw)

    # Count RAG results for dynamic instruction (1:1 mapping: each RAG result gets one bullet)
    digest_raw = authority_raw if evidence_mode == "role_private" else rag_raw
    digest_pack = authority_pack if evidence_mode == "role_private" else rag_pack
    rag_count = len(digest_raw) if digest_raw else 0
    # Keep the evidence digest instruction dynamic so each RAG result maps to one bullet.
    guideline_digester = Agent(
        instruction=_render_global_guideline_digester_instruction(agent_prompts, rag_count),
        role="global_guideline_digester",
        model_info=model,
        client=client,
        max_tokens=4000,
        max_prompt_tokens=3500,
    )
    
    # RAG evidence summarization with error handling
    try:
        global_guideline_digest = summarize_rag_evidence(guideline_digester, digest_pack, rag_raw=digest_raw)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] RAG evidence summarization failed: {e}{Color.RESET}")
        # Fallback: use first 3 RAG results as digest
        if digest_raw and len(digest_raw) > 0:
            digest_lines = []
            for i, r in enumerate(digest_raw[:3], 1):
                source = r.get("source", "")
                if source == "guideline":
                    tag = _get_rag_result_tag(r, i)
                elif str(source).startswith("nccn"):
                    tag = _get_rag_result_tag(r, i)
                elif source == "pubmed":
                    pmid = r.get("pmid", "")
                    tag = f"[@pubmed | {pmid}]"
                elif source == "fda":
                    source_id = r.get("source_id", f"FDA_{i}")
                    tag = f"[@fda | {source_id}]"
                elif source == "conference":
                    source_id = r.get("source_id", f"CONF_{i}")
                    tag = f"[@conference | {source_id}]"
                else:
                    tag = f"[unknown source {i}]"
                text = r.get("text", "") or r.get("abstract", "")
                preview = text[:200] + "..." if len(text) > 200 else text
                digest_lines.append(f"- {preview} {tag}")
            global_guideline_digest = "\n".join(digest_lines) if digest_lines else "# No RAG evidence available"
        else:
            global_guideline_digest = "# No RAG evidence available"
        if trace:
            trace.emit("pipeline_error", {
                "stage": "rag_summarization",
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback_used": True
            })
    role_guideline_digest_by_role: Dict[str, str] = {}
    reference_tags_by_role: Dict[str, str] = {}
    if evidence_mode == "role_private":
        role_external_roles = [role for role in ROLES if role != "chair"]
        rag_prompts = get_mdt_prompts().get("rag", {})
        role_query_agent = Agent(
            instruction=rag_prompts.get(
                "role_external_query_builder",
                "Rewrite the shared ovarian cancer MDT query into concise role-specific external evidence queries. Return JSON only.",
            ),
            role="role_external_query_builder",
            model_info=model,
            client=client,
            max_tokens=1200,
            max_prompt_tokens=12000,
        )
        role_external_queries = _build_role_external_queries(
            role_query_agent=role_query_agent,
            prompts=rag_prompts,
            shared_query=rag_query,
            rag_case_json=rag_case_json,
            rag_key_facts=rag_key_facts,
            roles=role_external_roles,
            trace=trace,
        )
        role_guideline_results = retrieve_mdt_guideline_evidence_lanes(
            lane_queries={role: role_external_queries.get(role) or rag_query for role in role_external_roles},
            topk=topk,
            unavailable_pack="(Guideline: no role-specific evidence found)",
        )
        for role in role_external_roles:
            role_guideline_pack, role_guideline_raw = role_guideline_results.get(
                role,
                ("(Guideline: no role-specific evidence found)", []),
            )
            del role_guideline_pack
            role_guideline_raw_by_role[role] = list(role_guideline_raw or [])
            role_guideline_digest_by_role_private[role] = _format_role_rag_evidence_digest(
                role_guideline_raw,
                f"{role} Guideline Evidence",
                max_items=topk,
            )
            trace.emit(
                "rag_hits",
                {"source": "guideline.role_private", "role": role, "topk": topk, "n": len(role_guideline_raw or [])},
            )
        role_external_results = retrieve_mdt_external_evidence_lanes(
            lane_queries={role: role_external_queries.get(role) or rag_query for role in role_external_roles},
            topk=external_pubmed_topk,
            unavailable_pack="(PUBMED: no role-specific evidence found)",
        )
        for role in role_external_roles:
            role_pack, role_raw = role_external_results.get(
                role,
                ("(PUBMED: no role-specific evidence found)", []),
            )
            del role_pack
            role_external_raw_by_role[role] = list(role_raw or [])
            role_external_digest_by_role[role] = _format_external_evidence_digest(
                role_raw,
                f"{role} External Evidence",
            )
            trace.emit(
                "rag_hits",
                {"source": "pubmed.role_private", "role": role, "topk": external_pubmed_topk, "n": len(role_raw or [])},
            )
        if visual.enable and visual.show_tables:
            _print_role_private_evidence_queries(
                role_queries=role_external_queries,
                role_guideline_raw_by_role=role_guideline_raw_by_role,
                role_external_raw_by_role=role_external_raw_by_role,
            )
        private_reference_raw_by_tag = _rag_raw_by_normalized_tag(
            [
                item
                for role in role_external_roles
                for item in role_guideline_raw_by_role.get(role, [])
            ]
            + [
                item
                for role in role_external_roles
                for item in role_external_raw_by_role.get(role, [])
            ]
        )
        all_reference_raw_for_cache = _dedupe_rag_raw_by_tag(
            list(authority_raw or [])
            + list(pubmed_raw or [])
            + [
                item
                for role in role_external_roles
                for item in role_guideline_raw_by_role.get(role, [])
            ]
            + [
                item
                for role in role_external_roles
                for item in role_external_raw_by_role.get(role, [])
            ]
        )
        trace.emit("rag_hits_merged", {"n": len(all_reference_raw_for_cache), "mode": "role_private"})
        try:
            from utils.reference_cache import get_reference_cache

            ref_cache = get_reference_cache()
            ref_cache.store_rag_results(all_reference_raw_for_cache)
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to cache role-private RAG references: {e}{Color.RESET}")

        scope_line = (
            "Evidence scope: shared NCCN authority plus guideline/external evidence assigned to your MDT role."
        )
        nccn_digest = _format_role_rag_evidence_digest(nccn_raw, "Shared NCCN Evidence", max_items=3)
        shared_external_digest = _format_external_evidence_digest(pubmed_raw, "Chair External Evidence")
        for role in ROLES:
            if role == "chair":
                role_guideline_digest_by_role[role] = _join_pack_sections(
                    scope_line,
                    global_guideline_digest,
                    shared_external_digest,
                )
                continue
            guideline_digest = role_guideline_digest_by_role_private.get(
                role,
                "# Role Guideline Evidence\n- No role-specific guideline evidence returned.",
            )
            role_guideline_digest_by_role[role] = _join_pack_sections(
                scope_line,
                nccn_digest,
                guideline_digest,
                role_external_digest_by_role.get(
                    role,
                    "# Role External Evidence\n- No role-specific external evidence returned.",
                ),
            )
        chair_authority_tags = _build_reference_tags_for_rag(authority_raw)
        nccn_tags = _build_reference_tags_for_rag(nccn_raw)
        for role in ROLES:
            if role == "chair":
                reference_tags_by_role[role] = _format_reference_tags(
                    chair_authority_tags + _build_reference_tags_for_rag(pubmed_raw)
                )
                continue
            role_private_raw = list(role_guideline_raw_by_role.get(role, [])) + list(role_external_raw_by_role.get(role, []))
            reference_tags_by_role[role] = _format_reference_tags(
                nccn_tags + _build_reference_tags_for_rag(role_private_raw)
            )
    emit_rag_digest_ready(role_guideline_digest_by_role.get("chair", global_guideline_digest))
    
    ###########################################################################
    # INIT SPECIALIST AGENTS
    ###########################################################################
    agents = {}
    failed_roles = []
    for role in ROLES:
        try:
            agents[role] = init_expert_agent(
                role=role,
                question=question_str,
                model=model,
                client=client,
                context=context,
                case_fingerprint=case_fingerprint,
                global_guideline_digest=role_guideline_digest_by_role.get(role, global_guideline_digest),
                device=device,
                topk=topk,
                visit_time=str(time) if time else None,
            )
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to initialize {role} agent: {e}{Color.RESET}")
            failed_roles.append(role)
            if trace:
                trace.emit("pipeline_error", {
                    "stage": "agent_init",
                    "role": role,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
    
    # If chair failed, use first successfully initialized agent as chair
    if "chair" in failed_roles and agents:
        first_role = list(agents.keys())[0]
        print(f"{Color.WARNING}[WARNING] Chair agent failed. Using {first_role} as fallback chair.{Color.RESET}")
        agents["chair"] = agents[first_role]
        failed_roles.remove("chair")
    
    # Ensure we have at least one agent
    if not agents:
        raise RuntimeError("Failed to initialize any expert agents. Cannot proceed with MDT discussion.")
    
    if failed_roles:
        print(f"{Color.WARNING}[WARNING] Some roles failed to initialize: {failed_roles}. Continuing with available agents.{Color.RESET}")

    assistant = Agent(
        instruction=agent_prompts.get("assistant", 
            "You are MDT assistant. Summarize only. Do not decide treatment."),
        role="assistant",
        model_info=model,
        client=client,
        max_tokens=10000,
        max_prompt_tokens=10000,
    )

    ablation_outputs: Dict[str, str] = {}
    trial_query = _build_trial_query(rag_query, rag_case_json)
    trace.emit("trial_query_compact", {"query": trial_query})

    ###########################################################################
    # MDT DISCUSSION
    ###########################################################################
    print_section("4) MDT Discussion Engine")
    trace.emit(
        "mdt_discussion_start",
        {
            "num_rounds": 2,
            "num_turns": 2,
            "roles": list(ROLES),
            "role_count": len(ROLES),
        },
    )

    # run the MDT discussion engine
    discussion_reference_tags = "\n".join(
        f"  - {tag}" for tag in _build_reference_tags_for_rag(rag_raw)
    ) or "  (No references available)"
    initial_ops, merged, final_round_ops, interaction_log = run_mdt_discussion(
        agents=agents,
        assistant=assistant,
        num_rounds=2, # 2 for formal； 1 for test
        num_turns=2,  # 2
        visit_time=str(time) if time else None,
        trace=trace,
        reference_tags_str=discussion_reference_tags,
        reference_tags_by_role=reference_tags_by_role if evidence_mode == "role_private" else None,
        private_reference_raw_by_tag=private_reference_raw_by_tag if evidence_mode == "role_private" else None,
        public_evidence_atoms=public_evidence_atoms if evidence_mode == "role_private" else None,
    )
    trace.emit(
        "mdt_discussion_end",
        {
            "merged_chars": len(merged or ""),
            "rounds": 2,
            "round_count": 2,
            "role_count": len(ROLES),
        },
    )

    print_interaction_matrix(interaction_log, roles_order=ROLES)

    ###########################################################################
    # Chair-Gated Clinical Trial Query (Phase 2C)
    ###########################################################################
    trial_note, _fit_trials_records = _run_chair_gated_trial_query(
        chair_agent=agents["chair"],
        question_str=question_str,
        trial_query=trial_query,
        trace=trace,
        mdt_context=merged,
    )

    ###########################################################################
    # FINAL OUTPUT
    ###########################################################################
    print_section("5) Final Chair Output")
    trace.emit("final_output_start", {})
    print(f"{Color.BOLD}{Color.OKBLUE}\n📘 Generating final MDT output...{Color.RESET}")
    # Build REFERENCE TAGS for chair (tag only, no index) so chair can cite guideline/pubmed
    public_private_raw_for_final = []
    if evidence_mode == "role_private":
        for atom in public_evidence_atoms:
            raw = private_reference_raw_by_tag.get(_normalize_reference_tag(atom.get("tag") or ""))
            if raw:
                public_private_raw_for_final.append(raw)
    final_visible_raw = (
        _dedupe_rag_raw_by_tag(list(authority_raw or []) + list(pubmed_raw or []) + public_private_raw_for_final)
        if evidence_mode == "role_private"
        else rag_raw
    )
    ref_tags_for_final = [f"  - {tag}" for tag in _build_reference_tags_for_rag(final_visible_raw)]
    ref_tags_str = "\n".join(ref_tags_for_final) if ref_tags_for_final else "  (No references available)"
    merged_for_final = merged
    public_evidence_block = _format_public_evidence_atoms(public_evidence_atoms)
    if evidence_mode == "role_private" and public_evidence_block:
        merged_for_final = _join_pack_sections(
            merged,
            "[ROLE_SCOPED_CITED_EVIDENCE]\n" + public_evidence_block,
        )
    # print(merged)
    # print(initial_ops)
    # print(interaction_log)
    final_output = generate_final_output(
        chair_agent=agents["chair"],
        all_round_ops=final_round_ops,
        clinic_time=time,
        merged=merged_for_final,
        initial_ops=initial_ops,
        interaction_log=interaction_log,
        trial_note=trial_note,
        trace=trace,
        ref_tags_str=ref_tags_str,
    )
    # Post-process: append References section with evidence details
    final_output = append_references_to_output(
        final_output,
        trial_note=trial_note,
        report_context=context,
    )
    if ablation_enabled:
        print_section("Matched Ablation Outputs")
        ablation_outputs = _build_matched_ablation_outputs(
            chair_agent=agents["chair"],
            question_str=question_str,
            model=model,
            client=client,
            clinic_time=time,
            case_fingerprint=case_fingerprint,
            initial_ops=initial_ops,
            merged_for_final=merged_for_final,
            full_final_output=final_output,
            public_evidence_block=public_evidence_block,
            trial_note=trial_note,
            ref_tags_str=ref_tags_str,
            report_context=context,
            role_guideline_digest_by_role=role_guideline_digest_by_role,
            reference_tags_by_role=reference_tags_by_role if evidence_mode == "role_private" else None,
            trace=trace,
        )
        setattr(args, "_last_ablation_outputs", ablation_outputs)
        trace.emit("ablation_outputs_ready", {"labels": list(ablation_outputs.keys())})
    final_decision_body, final_references = split_references_from_output(final_output)
    final_bottom_line = _extract_final_bottom_line_text(final_decision_body)
    if final_bottom_line:
        print(f"{Color.OKBLUE}[FINAL] Chair final synthesis ready.{Color.RESET}")
    if ablation_enabled:
        _print_ablation_outputs(ablation_outputs)
    else:
        print(final_output)
    warn_missing_evidence_tags(final_output, role="chair/final_output", trace=trace)
    trace.emit("final_output_end", {"final_output_chars": len(final_output or "")})

    print(f"{Color.BOLD}{Color.OKGREEN}\n=== MDT Multi-Expert Pipeline End ==={Color.RESET}")
    return final_output


###############################################################################
# CHAIR-SA(K) - Single Agent with Knowledge Only
###############################################################################
def process_chair_e_query(
    question: Any,
    question_raw: Optional[str],
    model: str,
    args: Any,
    time: Optional[str] = None,
    meta_info: Optional[str] = None,
    device: str = "auto",
    topk: int = 5,
) -> str:
    """
    ===========================================================================
    CHAIR-E - Single Agent with Knowledge + Structured Case History (evidence-augmented)
    ===========================================================================

    Mode Hierarchy (increasing input availability):
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1. CHAIR-R       → No RAG, No Reports                                    │
    │ 2. CHAIR-E       → RAG (Guidelines + PubMed) + Structured Case History    │
    │ 3. CHAIR-D       → RAG + Evidence Pack (full clinical reports)            │
    │ 4. OMGs          → RAG + Evidence Pack + Multi-expert MDT Discussion      │
    └─────────────────────────────────────────────────────────────────────────────┘

    CHAIR-E Design:
    - K = Knowledge: Guidelines + Literature (RAG-retrieved)
    - Structured Case History: LAB_TRENDS, TIMELINE from input case_json
    - NOT included: External clinical report files (labs.json, imaging.json, etc.)

    Key Distinction from CHAIR-D:
    - CHAIR-E: Uses structured case data embedded in input JSON
    - CHAIR-D: Loads external report files and LLM-filters them

    Args:
        question: Case data (dict/list/str) containing structured case history
        question_raw: Original raw question text
        model: Model/deployment name
        args: CLI arguments with client
        time: Visit timestamp
        meta_info: Patient identifier
        device: Compatibility parameter for evidence retrieval; local embeddings are not used
        topk: Top-k RAG results

    Returns:
        Final MDT-style output string
    """
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== CHAIR-E Pipeline Start ==={Color.RESET}")
    print(f"{Color.OKCYAN}[INFO] Knowledge + Structured Case History mode{Color.RESET}")
    
    # Load paths configuration
    paths_config = get_paths_config()
    
    # Trace collection
    trace = TraceLogger(enabled=True)
    trace.emit("pipeline_start", {
        "mode": "chair_e",
        "visit_time": str(time) if time else None,
        "meta_info": str(meta_info),
        "run_id": str(getattr(args, "run_id", "") or ""),
    })
    
    client = args.client
    print(f"{Color.OKBLUE}{Color.BOLD}🕒 Query Time: {time}{Color.RESET}")

    # Normalize question and compute case fingerprint
    raw_question_str = question_to_text(question)
    case_json = sanitize_case_for_decision(safe_load_case_json(raw_question_str))
    question_str = question_to_text(case_json) if case_json else raw_question_str
    case_fingerprint = hashlib.sha1(question_str.encode("utf-8")).hexdigest()[:12]

    print(f"{Color.OKBLUE}{Color.BOLD}🧾 CASE_FINGERPRINT: {case_fingerprint}{Color.RESET}")
    trace.emit("case_fingerprint", {"case_fingerprint": case_fingerprint})

    ###########################################################################
    # KNOWLEDGE RETRIEVAL (Guidelines + PubMed)
    ###########################################################################
    print_section("1) Knowledge Retrieval (Guidelines + PubMed)")

    # Load agent prompts
    agent_prompts = get_mdt_prompts().get("agents", {})
    
    rag_query_builder = Agent(
        instruction=agent_prompts.get("rag_query_builder", 
            "Construct concise English MDT guideline query."),
        role="rag_query_builder",
        model_info=model,
        client=client,
        max_tokens=5000,
        max_prompt_tokens=20000,
    )

    # Build key facts for RAG query (no mutation reports in K mode)
    rag_key_facts = _build_rag_key_facts(case_json, [])
    trace.emit("rag_key_facts", {"facts": rag_key_facts})
    
    try:
        rag_query = build_rag_query_for_mdt(rag_query_builder, question_str, key_facts=rag_key_facts)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] RAG query builder failed: {e}{Color.RESET}")
        case_core = case_json.get("CASE_CORE", {}) or {}
        diagnosis = case_core.get("DIAGNOSIS", {}) or {}
        primary = diagnosis.get("primary", "ovarian cancer")
        rag_query = f"{primary} treatment guidelines"

    # RAG retrieval - NCCN (constraints) + PDF guidelines (evidence) + PubMed
    from servers.evidence_retrieval import (
        mdt_external_pubmed_topk_from_engine_depth,
        retrieve_mdt_evidence_sources,
    )

    external_pubmed_topk = mdt_external_pubmed_topk_from_engine_depth()
    rag_pack, rag_raw, rag_sources = retrieve_mdt_evidence_sources(
        case_question=case_json,
        rag_query=rag_query,
        device=device,
        guideline_topk=topk,
        nccn_topk=3,
        external_topk=external_pubmed_topk,
    )
    nccn_pack, nccn_raw = rag_sources["nccn"]
    guideline_pack, guideline_raw = rag_sources["guideline"]
    pubmed_pack, pubmed_raw = rag_sources["external"]

    # Build REFERENCE TAGS list for final output (tag only, no index — avoid model citing [1][2])
    ref_tags_for_final = []
    for i, r in enumerate(rag_raw, 1):
        source = r.get("source", "")
        if source == "guideline":
            tag = _get_rag_result_tag(r, i)
        elif source == "nccn_safety_rule":
            rule_id = r.get("rule_id", f"SAFETY_{i}")
            tag = f"[@guideline:nccn | {rule_id}]"
        elif source == "nccn_matcher_rule":
            rule_id = r.get("rule_id", r.get("node_id", f"MATCHER_{i}"))
            tag = f"[@guideline:nccn | {rule_id}]"
        elif source == "nccn_decision_node":
            node_id = r.get("node_id", r.get("rule_id", f"NODE_{i}"))
            tag = f"[@guideline:nccn | {node_id}]"
        elif source == "pubmed":
            pmid = r.get("pmid", "")
            tag = f"[@pubmed | {pmid}]"
        elif source == "fda":
            source_id = r.get("source_id", f"FDA_{i}")
            tag = f"[@fda | {source_id}]"
        elif source == "conference":
            source_id = r.get("source_id", f"CONF_{i}")
            tag = f"[@conference | {source_id}]"
        else:
            tag = f"[unknown source {i}]"
        ref_tags_for_final.append(f"  - {tag}")
    ref_tags_str = "\n".join(ref_tags_for_final) if ref_tags_for_final else "  (No references available)"

    # Store RAG results in reference cache
    try:
        from utils.reference_cache import get_reference_cache
        ref_cache = get_reference_cache()
        ref_cache.store_rag_results(rag_raw)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to cache RAG references: {e}{Color.RESET}")

    trace.emit("rag_query", {"query": rag_query})
    trace.emit("rag_hits", {"source": "nccn", "topk": 3, "n": len(nccn_raw or [])})
    trace.emit("rag_hits", {"source": "guideline", "topk": topk, "n": len(guideline_raw or [])})
    trace.emit("rag_hits", {"source": "pubmed", "topk": external_pubmed_topk, "n": len(pubmed_raw or [])})
    trace.emit("rag_hits_merged", {"n": len(rag_raw or [])})
    print_rag_hits_table(rag_raw)
    
    # Generate knowledge digest
    rag_count = len(rag_raw) if rag_raw else 0
    guideline_digester = Agent(
        instruction=_render_global_guideline_digester_instruction(agent_prompts, rag_count),
        role="global_guideline_digester",
        model_info=model,
        client=client,
        max_tokens=4000,
        max_prompt_tokens=3500,
    )
    
    try:
        global_guideline_digest = summarize_rag_evidence(guideline_digester, rag_pack, rag_raw=rag_raw)
        print(f"{Color.OKCYAN}[RAG] Knowledge digest ready: {preview_text(global_guideline_digest, 220)}{Color.RESET}")
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] RAG summarization failed: {e}{Color.RESET}")
        global_guideline_digest = "# No knowledge evidence available"
    
    ###########################################################################
    # INITIALIZE CHAIR AGENT (Knowledge only)
    ###########################################################################
    print_section("2) Initialize Chair Agent")
    
    from orchestrator.experts import get_role_prompt
    from utils.mdt_runtime_protocol import build_runtime_protocol_digest

    visit_time_str = str(time) if time else "Unknown visit date"
    runtime_protocol_digest = build_runtime_protocol_digest("chair")
    role_prompt = get_role_prompt("chair")
    
    # Build case view (structured summary of case)
    from servers.context_assembly import build_role_specific_case_view
    case_view = build_role_specific_case_view("chair", case_json)
    
    instruction = f"""
{runtime_protocol_digest}

OUTPATIENT VISIT TIME (today's clinic decision point): {visit_time_str}

CASE_FINGERPRINT: {case_fingerprint}

{role_prompt}

# HARD RULES (critical)
1) All decisions are for THIS visit date and future care, not for past timepoints.
2) PATIENT FACTS come ONLY from Role-Specific Case View below.
3) Retrieved guideline/PubMed/NCCN evidence is general reference, NOT patient-specific facts.
4) Laboratory data may be available in Role-Specific Case View under LAB_TRENDS.
   - Reference lab values using [@date | LAB] format (e.g., [@2022-12-29 | LAB] for CA125)
   - Use actual dates from the lab history (see "latest.date" and "history[]")
5) Any claim derived from guideline/PubMed/NCCN evidence MUST include evidence tag:
   - Guidelines: use EXACT format from REFERENCE TAGS
   - PubMed: [@pubmed | PMID]
   - NCCN rules: [@guideline:nccn | rule_id]
6) If information is missing, clearly state what data would be needed.

# Role-Specific Case View (PATIENT FACTS + LAB_TRENDS + TIMELINE)
{case_view}

# GLOBAL Guideline + PubMed Digest (NOT PATIENT FACTS)
{global_guideline_digest}
""".strip()
    
    chair_agent = Agent(
        instruction=instruction,
        role="chair",
        model_info=model,
        client=client,
        max_tokens=2000,
        max_prompt_tokens=20000,
    )
    chair_agent.inject_assistant("System ready for MDT decision.")
    print(f"{Color.OKGREEN}✔ Initialized CHAIR-E agent{Color.RESET}")
    
    ###########################################################################
    # GENERATE FINAL OUTPUT
    ###########################################################################
    print_section("3) Generate Final Output")

    final_prompt = _render_chair_mode_final_prompt(
        "chair_e",
        visit_time_str=visit_time_str,
        ref_tags_str=ref_tags_str,
    )
    
    try:
        final_output = chair_agent.chat(final_prompt)
    except Exception as e:
        print(f"{Color.FAIL}[ERROR] Chair final output failed: {e}{Color.RESET}")
        final_output = f"""Final Assessment:
Final chair synthesis was unavailable for this run.

Core Treatment Strategy:
- Review case data and available guidelines
- Obtain necessary clinical reports

Change Triggers:
- Reassess after the missing synthesis step is rerun"""
    
    # Append references (use case-based report context for [@report_id | LAB] etc.)
    report_context = build_case_report_context(case_json)
    final_output = append_references_to_output(final_output, trial_note="", report_context=report_context)
    print(final_output)
    warn_missing_evidence_tags(final_output, role="chair_e/final_output", trace=trace)
    
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== CHAIR-E Pipeline End ==={Color.RESET}")
    return final_output


###############################################################################
# CHAIR-D - Single Agent with Knowledge + Evidence Pack
###############################################################################
def process_chair_d_query(
    question: Any,
    question_raw: Optional[str],
    model: str,
    args: Any,
    time: Optional[str] = None,
    meta_info: Optional[str] = None,
    labs_json: Optional[str] = None,
    imaging_json: Optional[str] = None,
    pathology_json: Optional[str] = None,
    mutation_json: Optional[str] = None,
    device: str = "auto",
    topk: int = 5,
) -> str:
    """
    ===========================================================================
    CHAIR-D - Single Agent with Knowledge + Evidence Pack (dossier-augmented)
    ===========================================================================

    Mode Hierarchy (increasing input availability):
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1. CHAIR-R       → No RAG, No Reports                                    │
    │ 2. CHAIR-E       → RAG (Guidelines + PubMed) + Structured Case History    │
    │ 3. CHAIR-D       → RAG + Evidence Pack (full clinical reports)            │
    │ 4. OMGs          → RAG + Evidence Pack + Multi-expert MDT Discussion      │
    └─────────────────────────────────────────────────────────────────────────────┘

    CHAIR-D Design:
    - K = Knowledge: Guidelines + Literature (RAG-retrieved)
    - EP = Evidence Pack: External report files (labs.json, imaging.json, etc.)
    - Loads and LLM-filters clinical reports per role permissions
    - Full patient evidence: Labs, Imaging, Pathology, Genomics

    Key Distinction from CHAIR-E:
    - CHAIR-E: Uses structured case data embedded in input JSON
    - CHAIR-D: Loads external report files and LLM-filters them

    Args:
        question: Case data (dict/list/str)
        question_raw: Original raw question text
        model: Model/deployment name
        args: CLI arguments with client
        time: Visit timestamp
        meta_info: Patient identifier
        labs_json: Path to lab reports (from config if not provided)
        imaging_json: Path to imaging reports (from config if not provided)
        pathology_json: Path to pathology reports (from config if not provided)
        mutation_json: Path to mutation reports (from config if not provided)
        device: Compatibility parameter for evidence retrieval; local embeddings are not used
        topk: Top-k RAG results

    Returns:
        Final MDT-style output string
    """
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== CHAIR-D Pipeline Start ==={Color.RESET}")
    print(f"{Color.OKCYAN}[INFO] Knowledge + Evidence Pack mode{Color.RESET}")
    
    # Load paths configuration
    paths_config = get_paths_config()
    
    # Use config paths if not explicitly provided
    if labs_json is None:
        labs_json = paths_config["data_files"]["lab_reports"]
    if imaging_json is None:
        imaging_json = paths_config["data_files"]["imaging_reports"]
    if pathology_json is None:
        pathology_json = paths_config["data_files"]["pathology_reports"]
    if mutation_json is None:
        mutation_json = paths_config["data_files"]["mutation_reports"]
    
    # Trace collection
    trace = TraceLogger(enabled=True)
    trace.emit("pipeline_start", {
        "mode": "chair_d",
        "visit_time": str(time) if time else None,
        "meta_info": str(meta_info),
        "run_id": str(getattr(args, "run_id", "") or ""),
    })
    
    client = args.client
    print(f"{Color.OKBLUE}{Color.BOLD}🕒 Query Time: {time}{Color.RESET}")
    
    # Normalize question and compute case fingerprint
    raw_question_str = question_to_text(question)
    case_json = sanitize_case_for_decision(safe_load_case_json(raw_question_str))
    question_str = question_to_text(case_json) if case_json else raw_question_str
    case_fingerprint = hashlib.sha1(question_str.encode("utf-8")).hexdigest()[:12]
    
    print(f"{Color.OKBLUE}{Color.BOLD}🧾 CASE_FINGERPRINT: {case_fingerprint}{Color.RESET}")
    trace.emit("case_fingerprint", {"case_fingerprint": case_fingerprint})
    
    ###########################################################################
    # LOAD EVIDENCE PACK (All Reports)
    ###########################################################################
    print_section("1) Load Evidence Pack (All Reports)")
    
    cutoff_dt = make_cutoff(time, days_after=1)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S") if cutoff_dt else "None"
    print(f"{Color.OKBLUE}{Color.BOLD}⏱️  CUTOFF_DT (time + 1d): {cutoff_str}{Color.RESET}")
    
    # Load all reports
    try:
        lab_timeline_raw, lab_reports = load_patient_labs(meta_info, labs_json)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to load lab reports: {e}{Color.RESET}")
        lab_timeline_raw, lab_reports = [], []
    
    try:
        im_timeline_raw, im_reports = load_patient_imaging(meta_info, imaging_json)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to load imaging reports: {e}{Color.RESET}")
        im_timeline_raw, im_reports = [], []
    
    path_timeline_raw, path_reports = [], []
    if pathology_json:
        try:
            path_timeline_raw, path_reports = load_patient_pathology(meta_info, pathology_json)
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to load pathology reports: {e}{Color.RESET}")
    
    mut_reports: List[Dict[str, Any]] = []
    if meta_info and mutation_json:
        try:
            mut_reports = load_patient_mutations(meta_info, mutation_json)
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to load mutation reports: {e}{Color.RESET}")
    
    trace.emit("reports_loaded", {
        "lab_n": len(lab_reports),
        "img_n": len(im_reports),
        "path_n": len(path_reports),
        "mut_n": len(mut_reports),
    })
    
    # Filter by cutoff
    if cutoff_dt is not None:
        lab_reports = filter_before(lab_reports, "report_date", cutoff_dt)
        im_reports = filter_before(im_reports, "report_date", cutoff_dt)
        path_reports = filter_before(path_reports, "report_date", cutoff_dt)
        mut_reports = filter_before(mut_reports, "report_date", cutoff_dt)
    
    # Build timelines
    lab_timeline = build_lab_timeline(lab_reports)
    im_timeline = build_imaging_timeline(im_reports)
    path_timeline = build_pathology_timeline(path_reports) if pathology_json else []
    
    print(f"{Color.OKCYAN}Evidence Pack loaded: Labs={len(lab_reports)}, Imaging={len(im_reports)}, Path={len(path_reports)}, Mut={len(mut_reports)}{Color.RESET}")
    
    ###########################################################################
    # REPORT SELECTION (LLM-based filtering for Chair)
    ###########################################################################
    print_section("2) Report Selection for Chair")
    
    # CHAIR-D receives the dossier-level clinical reports for the single-chair comparator.
    CHAIR_SA_PERMISSIONS = {
        "chair": {"lab": True, "imaging": True, "pathology": True, "mutation": True, "guideline": "chair"},
    }
    
    context = select_reports_for_roles(
        roles=["chair"],
        role_permissions=CHAIR_SA_PERMISSIONS,
        lab_timeline=lab_timeline,
        lab_reports=lab_reports,
        im_timeline=im_timeline,
        im_reports=im_reports,
        path_timeline=path_timeline,
        path_reports=path_reports,
        mut_reports=mut_reports,
        pathology_json=pathology_json,
        agent_class=Agent,
        expert_select_fn=expert_select_reports,
        model=model,
        client=client,
        color=Color,
        use_shared=True,
    )
    
    # Extract selected reports for chair
    selected_lab = context.get("lab", {}).get("chair", [])
    selected_imaging = context.get("imaging", {}).get("chair", [])
    selected_pathology = context.get("pathology", {}).get("chair", [])
    selected_mutation = context.get("mutation", {}).get("chair", [])
    
    print(f"{Color.OKCYAN}{Color.BOLD}🧩 Selected reports for Chair:{Color.RESET}")
    print(f"  Labs: {len(selected_lab)}, Imaging: {len(selected_imaging)}, Pathology: {len(selected_pathology)}, Mutation: {len(selected_mutation)}")
    
    trace.emit("reports_selected", {
        "lab": len(selected_lab),
        "imaging": len(selected_imaging),
        "pathology": len(selected_pathology),
        "mutation": len(selected_mutation),
    })
    
    ###########################################################################
    # KNOWLEDGE RETRIEVAL (Guidelines + PubMed)
    ###########################################################################
    print_section("3) Knowledge Retrieval (Guidelines + PubMed)")
    
    agent_prompts = get_mdt_prompts().get("agents", {})
    
    rag_query_builder = Agent(
        instruction=agent_prompts.get("rag_query_builder", 
            "Construct concise English MDT guideline query."),
        role="rag_query_builder",
        model_info=model,
        client=client,
        max_tokens=5000,
        max_prompt_tokens=20000,
    )
    
    # Build key facts for RAG query (include mutation reports)
    rag_key_facts = _build_rag_key_facts(case_json, mut_reports)
    trace.emit("rag_key_facts", {"facts": rag_key_facts})
    
    # Build RAG query with GENOMICS-enhanced case
    # Uses the GENOMICS section when available and falls back to top-level
    # molecular fields and raw mutation reports.
    genomic_status = _extract_genomic_status(case_json)
    
    import copy
    rag_case_json = copy.deepcopy(case_json)
    
    # Apply GENOMICS-derived values
    if genomic_status["HRD"] != "Unknown":
        rag_case_json.setdefault("CASE_CORE", {})["HRD"] = genomic_status["HRD"]
    if genomic_status["BRCA1"] != "Unknown":
        rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = genomic_status["BRCA1"]
    if genomic_status["BRCA2"] != "Unknown":
        rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = genomic_status["BRCA2"]
    
    # Fallback: extract from raw mutation reports if still Unknown
    if mut_reports and (genomic_status["HRD"] == "Unknown" or genomic_status["BRCA1"] == "Unknown" or genomic_status["BRCA2"] == "Unknown"):
        latest_mut = mut_reports[-1]
        raw_text = latest_mut.get("raw_text", "")
        if raw_text:
            if "HRD" in raw_text and genomic_status["HRD"] == "Unknown":
                if "阴性" in raw_text or "negative" in raw_text.lower():
                    rag_case_json.setdefault("CASE_CORE", {})["HRD"] = "Negative"
                elif "阳性" in raw_text or "positive" in raw_text.lower():
                    rag_case_json.setdefault("CASE_CORE", {})["HRD"] = "Positive"
            if "BRCA1" in raw_text and genomic_status["BRCA1"] == "Unknown":
                if any(x in raw_text for x in ["未检出", "阴性", "视为阴性"]):
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = "Negative"
                elif "突变" in raw_text and "致病" in raw_text:
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = "Positive"
            if "BRCA2" in raw_text and genomic_status["BRCA2"] == "Unknown":
                if any(x in raw_text for x in ["未检出", "阴性", "视为阴性"]):
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = "Negative"
                elif "突变" in raw_text and "致病" in raw_text:
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = "Positive"
    
    rag_question_str = json.dumps(rag_case_json, ensure_ascii=False)
    
    try:
        rag_query = build_rag_query_for_mdt(rag_query_builder, rag_question_str, key_facts=rag_key_facts)
        print(f"{Color.OKCYAN}[RAG] Query preview: {preview_text(rag_query, 220)}{Color.RESET}")
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] RAG query builder failed: {e}{Color.RESET}")
        case_core = case_json.get("CASE_CORE", {}) or {}
        diagnosis = case_core.get("DIAGNOSIS", {}) or {}
        primary = diagnosis.get("primary", "ovarian cancer")
        rag_query = f"{primary} treatment guidelines"
    
    # RAG retrieval - NCCN (constraints) + PDF guidelines (evidence) + PubMed
    from servers.evidence_retrieval import (
        mdt_external_pubmed_topk_from_engine_depth,
        retrieve_mdt_evidence_sources,
    )

    external_pubmed_topk = mdt_external_pubmed_topk_from_engine_depth()
    rag_pack, rag_raw, rag_sources = retrieve_mdt_evidence_sources(
        case_question=case_json,
        rag_query=rag_query,
        device=device,
        guideline_topk=topk,
        nccn_topk=3,
        external_topk=external_pubmed_topk,
    )
    nccn_pack, nccn_raw = rag_sources["nccn"]
    guideline_pack, guideline_raw = rag_sources["guideline"]
    pubmed_pack, pubmed_raw = rag_sources["external"]

    # Build REFERENCE TAGS list for final output (tag only, no index — avoid model citing [1][2])
    ref_tags_for_final = []
    for i, r in enumerate(rag_raw, 1):
        source = r.get("source", "")
        if source == "guideline":
            tag = _get_rag_result_tag(r, i)
        elif source == "nccn_safety_rule":
            rule_id = r.get("rule_id", f"SAFETY_{i}")
            tag = f"[@guideline:nccn | {rule_id}]"
        elif source == "nccn_matcher_rule":
            rule_id = r.get("rule_id", r.get("node_id", f"MATCHER_{i}"))
            tag = f"[@guideline:nccn | {rule_id}]"
        elif source == "nccn_decision_node":
            node_id = r.get("node_id", r.get("rule_id", f"NODE_{i}"))
            tag = f"[@guideline:nccn | {node_id}]"
        elif source == "pubmed":
            pmid = r.get("pmid", "")
            tag = f"[@pubmed | {pmid}]"
        elif source == "fda":
            source_id = r.get("source_id", f"FDA_{i}")
            tag = f"[@fda | {source_id}]"
        elif source == "conference":
            source_id = r.get("source_id", f"CONF_{i}")
            tag = f"[@conference | {source_id}]"
        else:
            tag = f"[unknown source {i}]"
        ref_tags_for_final.append(f"  - {tag}")
    ref_tags_str = "\n".join(ref_tags_for_final) if ref_tags_for_final else "  (No references available)"

    # Store RAG results in reference cache
    try:
        from utils.reference_cache import get_reference_cache
        ref_cache = get_reference_cache()
        ref_cache.store_rag_results(rag_raw)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to cache RAG references: {e}{Color.RESET}")

    trace.emit("rag_query", {"query": rag_query})
    trace.emit("rag_hits", {"source": "nccn", "topk": 3, "n": len(nccn_raw or [])})
    trace.emit("rag_hits", {"source": "guideline", "topk": topk, "n": len(guideline_raw or [])})
    trace.emit("rag_hits", {"source": "pubmed", "topk": external_pubmed_topk, "n": len(pubmed_raw or [])})
    trace.emit("rag_hits_merged", {"n": len(rag_raw or [])})
    print_rag_hits_table(rag_raw)
    
    # Generate knowledge digest
    rag_count = len(rag_raw) if rag_raw else 0
    guideline_digester = Agent(
        instruction=_render_global_guideline_digester_instruction(agent_prompts, rag_count),
        role="global_guideline_digester",
        model_info=model,
        client=client,
        max_tokens=4000,
        max_prompt_tokens=3500,
    )
    
    try:
        global_guideline_digest = summarize_rag_evidence(guideline_digester, rag_pack, rag_raw=rag_raw)
        print(f"{Color.OKCYAN}[RAG] Knowledge digest ready: {preview_text(global_guideline_digest, 220)}{Color.RESET}")
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] RAG summarization failed: {e}{Color.RESET}")
        global_guideline_digest = "# No knowledge evidence available"
    
    ###########################################################################
    # BUILD EVIDENCE PACK CONTEXT (using SELECTED reports)
    ###########################################################################
    print_section("4) Initialize Chair Agent with Evidence Pack")
    
    from orchestrator.experts import get_role_prompt
    from utils.mdt_runtime_protocol import build_runtime_protocol_digest
    from servers.context_assembly import build_role_specific_case_view
    
    visit_time_str = str(time) if time else "Unknown visit date"
    runtime_protocol_digest = build_runtime_protocol_digest("chair")
    role_prompt = get_role_prompt("chair")
    case_view = build_role_specific_case_view("chair", case_json)
    
    # Build Evidence Pack string (using SELECTED reports, not all reports)
    evidence_pack = ""
    
    # Lab reports (selected)
    if selected_lab:
        evidence_pack += "# LAB REPORTS (PATIENT FACTS) - SELECTED\n"
        evidence_pack += json.dumps(selected_lab, ensure_ascii=False, indent=2) + "\n\n"
    
    # Imaging reports (selected)
    if selected_imaging:
        evidence_pack += "# IMAGING REPORTS (PATIENT FACTS) - SELECTED\n"
        evidence_pack += json.dumps(selected_imaging, ensure_ascii=False, indent=2) + "\n\n"
    
    # Pathology reports (selected)
    if selected_pathology:
        evidence_pack += "# PATHOLOGY REPORTS (PATIENT FACTS) - SELECTED\n"
        evidence_pack += json.dumps(selected_pathology, ensure_ascii=False, indent=2) + "\n\n"
    
    # Mutation reports (selected)
    if selected_mutation:
        evidence_pack += "# MUTATION / MOLECULAR REPORTS (PATIENT FACTS) - SELECTED\n"
        evidence_pack += "⚠️ COMPREHENSIVE NGS PANEL (~20,000 genes) - INTERPRETATION RULES:\n"
        evidence_pack += "• '未检出' (not detected) = NO pathogenic mutation found\n"
        evidence_pack += "• '（视为阴性）' (considered negative) = NO pathogenic mutation found\n"
        evidence_pack += "• '阴性' (negative) = negative result\n"
        evidence_pack += "• Genes with specific variants (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation\n"
        evidence_pack += "• If a gene is NOT mentioned, it means NO pathogenic mutation (comprehensive panel)\n\n"
        evidence_pack += json.dumps(selected_mutation, ensure_ascii=False, indent=2) + "\n\n"
    
    if not evidence_pack.strip():
        evidence_pack = "# No clinical reports available.\n\n"
    
    instruction = f"""
{runtime_protocol_digest}

OUTPATIENT VISIT TIME (today's clinic decision point): {visit_time_str}

CASE_FINGERPRINT: {case_fingerprint}

{role_prompt}

# HARD RULES (critical)
1) All decisions are for THIS visit date and future care, not for past timepoints.
2) PATIENT FACTS come ONLY from:
   - Role-Specific Case View, and
   - Evidence Pack (selected clinical reports below).
3) Retrieved guideline/PubMed/NCCN evidence is general reference, NOT patient-specific facts.
4) Any claim derived from guideline/PubMed/NCCN evidence MUST include evidence tag:
   - Guidelines: use EXACT format from REFERENCE TAGS
   - PubMed: [@pubmed | PMID]
   - NCCN rules: [@guideline:nccn | rule_id]
5) Any claim about labs/imaging/pathology/molecular MUST include evidence tag:
   - format: [@actual_report_id | LAB/Genomics/MR/CT/Pathology] using actual report_id from report data
6) If Case View conflicts with Clinical Reports, prefer Clinical Reports.

# Role-Specific Case View (PATIENT FACTS)
{case_view}

# Evidence Pack (PATIENT FACTS - selected clinical reports)
{evidence_pack}

# GLOBAL Guideline + PubMed Digest (NOT PATIENT FACTS)
{global_guideline_digest}
""".strip()
    
    chair_agent = Agent(
        instruction=instruction,
        role="chair",
        model_info=model,
        client=client,
        max_tokens=2000,
        max_prompt_tokens=30000,
    )
    chair_agent.inject_assistant("System ready for MDT decision.")
    print(f"{Color.OKGREEN}✔ Initialized CHAIR-D agent{Color.RESET}")

    ###########################################################################
    # Chair-Gated Clinical Trial Query (Phase 2C)
    ###########################################################################
    print_section("5) Clinical Trial Matching")
    print(f"{Color.OKBLUE}[CHAIR-D] Checking clinical trials...{Color.RESET}")
    trial_query = _build_trial_query(rag_query, rag_case_json)
    trace.emit("trial_query_compact", {"query": trial_query})
    trial_note, _fit_trials_records = _run_chair_gated_trial_query(
        chair_agent=chair_agent,
        question_str=question_str,
        trial_query=trial_query,
        trace=trace,
    )
    
    ###########################################################################
    # GENERATE FINAL OUTPUT
    ###########################################################################
    print_section("6) Generate Final Output")
    
    # Build trial section for prompt
    trial_section = ""
    if trial_note and trial_note.strip():
        trial_section = f"# CLINICAL TRIAL RECOMMENDATION\n{trial_note.strip()}\n\n"

    final_prompt = _render_chair_mode_final_prompt(
        "chair_d",
        visit_time_str=visit_time_str,
        ref_tags_str=ref_tags_str,
        trial_section=trial_section,
    )
    
    try:
        final_output = chair_agent.chat(final_prompt)
    except Exception as e:
        print(f"{Color.FAIL}[ERROR] Chair final output failed: {e}{Color.RESET}")
        final_output = f"""Final Assessment:
Final chair synthesis was unavailable for this run.

Core Treatment Strategy:
- Review case data and available evidence
- Consult specialist team

Change Triggers:
- Reassess after the missing synthesis step is rerun"""
    
    # Build report context for references (using selected reports)
    report_context = {
        "lab": {"chair": selected_lab},
        "imaging": {"chair": selected_imaging},
        "pathology": {"chair": selected_pathology},
        "mutation": {"chair": selected_mutation},
    }
    
    # Append references
    final_output = append_references_to_output(final_output, trial_note=trial_note, report_context=report_context)
    print(final_output)
    warn_missing_evidence_tags(final_output, role="chair_d/final_output", trace=trace)
    
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== CHAIR-D Pipeline End ==={Color.RESET}")
    return final_output


###############################################################################
# Helper: Build report context from case_json for CHAIR-R/CHAIR-D evidence tags
###############################################################################
def build_case_report_context(case_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a report_context from case_json for CHAIR-R/CHAIR-D mode.
    
    Extracts date-based information from structured case data to enable
    evidence tag lookup in build_references_section.
    
    The case_json may contain:
    - CASE_CORE: diagnosis, biomarkers, treatment history
    - TIMELINE: treatment timeline with dates
    - MED_ONC: genetic testing results
    - RADIOLOGY: imaging studies
    - PATHOLOGY: pathology specimens
    - LAB_TRENDS: lab results over time
    
    Args:
        case_json: Structured case data dictionary
    
    Returns:
        report_context dict compatible with _find_report_in_context:
        {
            "lab": {"chair": [...]},
            "imaging": {"chair": [...]},
            "pathology": {"chair": [...]},
            "mutation": {"chair": [...]},
            "case": {"chair": [...]},  # For general case facts
        }
    """
    import re
    
    context = {
        "lab": {"chair": []},
        "imaging": {"chair": []},
        "pathology": {"chair": []},
        "mutation": {"chair": []},
        "case": {"chair": []},
    }
    
    # Helper to extract dates from text
    def extract_dates_from_text(text: str) -> List[str]:
        """Extract date patterns from text."""
        if not text:
            return []
        # Match various date formats: YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD, YYYY.MM, YYYY-MM
        patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2022-01-17
            r'\d{4}\.\d{2}\.\d{2}',  # 2022.01.17
            r'\d{4}/\d{2}/\d{2}',  # 2022/01/17
            r'\d{4}\.\d{1,2}\.\d{1,2}',  # 2022.1.17
            r'\d{4}-\d{1,2}-\d{1,2}',  # 2022-1-17
        ]
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, str(text)))
        return list(set(dates))
    
    # Convert case_json to string for date extraction
    case_str = json.dumps(case_json, ensure_ascii=False) if isinstance(case_json, dict) else str(case_json)
    all_dates = extract_dates_from_text(case_str)
    
    # Process structured case data if available
    if isinstance(case_json, dict):
        # Extract from CASE_CORE
        core = case_json.get("CASE_CORE", {})
        if core:
            # Diagnosis, biomarkers, treatment history -> case
            for key in ["DIAGNOSIS", "BIOMARKERS", "CURRENT_STATUS", "LINE_OF_THERAPY"]:
                if core.get(key):
                    context["case"]["chair"].append({
                        "report_id": key,
                        "type": "case",
                        "summary": str(core.get(key))[:200],
                    })
        
        # Extract from TIMELINE
        timeline = case_json.get("TIMELINE", {})
        if isinstance(timeline, dict):
            for date, events in timeline.items():
                context["case"]["chair"].append({
                    "report_id": date,
                    "date": date,
                    "type": "case",
                    "summary": str(events)[:200] if events else "",
                })
        
        # Extract from MED_ONC (genetic testing)
        med_onc = case_json.get("MED_ONC", {})
        if med_onc:
            genetic = med_onc.get("genetic_testing", {})
            if genetic:
                for key, value in genetic.items() if isinstance(genetic, dict) else []:
                    context["mutation"]["chair"].append({
                        "report_id": key,
                        "type": "mutation",
                        "summary": str(value)[:200],
                    })
        
        # Extract from RADIOLOGY
        radiology = case_json.get("RADIOLOGY", {})
        if radiology:
            studies = radiology.get("studies", [])
            for study in studies if isinstance(studies, list) else []:
                date = study.get("date", "")
                context["imaging"]["chair"].append({
                    "report_id": date,
                    "date": date,
                    "type": "imaging",
                    "impression": study.get("impression", ""),
                    "summary": study.get("impression", "")[:200],
                })
        
        # Extract from PATHOLOGY
        pathology = case_json.get("PATHOLOGY", {})
        if pathology:
            specimens = pathology.get("specimens", [])
            for spec in specimens if isinstance(specimens, list) else []:
                date = spec.get("date", "")
                context["pathology"]["chair"].append({
                    "report_id": date,
                    "date": date,
                    "type": "pathology",
                    "diagnosis": spec.get("diagnosis", ""),
                    "summary": spec.get("diagnosis", "")[:200],
                })
        
        # Extract from LAB_TRENDS
        labs = case_json.get("LAB_TRENDS", {})
        if isinstance(labs, dict):
            for marker, values in labs.items():
                if isinstance(values, list):
                    for entry in values:
                        if isinstance(entry, dict):
                            date = entry.get("date", "")
                            context["lab"]["chair"].append({
                                "report_id": date,
                                "date": date,
                                "type": "lab",
                                "result": f"{marker}: {entry.get('value', '')}",
                                "summary": f"{marker}: {entry.get('value', '')}",
                            })
    
    # Add all extracted dates as general case references (fallback)
    for date in all_dates:
        # Check if date already exists in any category
        existing = False
        for cat in context.values():
            for report in cat.get("chair", []):
                if report.get("report_id") == date or report.get("date") == date:
                    existing = True
                    break
            if existing:
                break
        
        if not existing:
            context["case"]["chair"].append({
                "report_id": date,
                "date": date,
                "type": "case",
                "summary": f"Case data from {date}",
            })
    
    return context


###############################################################################
# CHAIR-R - Records-only chair baseline
###############################################################################
def process_chair_r_query(
    question: Any,
    question_raw: Optional[str],
    model: str,
    args: Any,
    time: Optional[str] = None,
    meta_info: Optional[str] = None,
) -> str:
    """
    ===========================================================================
    CHAIR-R - Records-only single-chair baseline
    ===========================================================================

    Mode Hierarchy (increasing input availability):
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1. CHAIR-R       → No RAG, No Reports                                    │
    │ 2. CHAIR-E       → RAG (Guidelines + PubMed) + Structured Case History    │
    │ 3. CHAIR-D       → RAG + Evidence Pack (full clinical reports)            │
    │ 4. OMGs          → RAG + Evidence Pack + Multi-expert MDT Discussion      │
    └─────────────────────────────────────────────────────────────────────────────┘

    CHAIR-R Design:
    - No RAG (no Guidelines, no PubMed)
    - No external report files
    - Uses ONLY the case_json input for patient facts
    - Purpose: records-only single-chair comparator

    Args:
        question: Case data (dict/list/str)
        question_raw: Original raw question text
        model: Model/deployment name
        args: CLI arguments with client
        time: Visit timestamp
        meta_info: Patient identifier

    Returns:
        Final MDT-style output string
    """
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== CHAIR-R Pipeline Start ==={Color.RESET}")
    print(f"{Color.OKCYAN}[INFO] Records-only chair baseline - no RAG, no reports{Color.RESET}")
    
    # Load paths configuration
    paths_config = get_paths_config()
    
    # Trace collection
    trace = TraceLogger(enabled=True)
    trace.emit("pipeline_start", {
        "mode": "chair_r",
        "visit_time": str(time) if time else None,
        "meta_info": str(meta_info),
        "run_id": str(getattr(args, "run_id", "") or ""),
    })
    
    client = args.client
    print(f"{Color.OKBLUE}{Color.BOLD}🕒 Query Time: {time}{Color.RESET}")
    
    # Normalize question and compute case fingerprint
    raw_question_str = question_to_text(question)
    case_json = sanitize_case_for_decision(safe_load_case_json(raw_question_str))
    question_str = question_to_text(case_json) if case_json else raw_question_str
    case_fingerprint = hashlib.sha1(question_str.encode("utf-8")).hexdigest()[:12]
    
    print(f"{Color.OKBLUE}{Color.BOLD}🧾 CASE_FINGERPRINT: {case_fingerprint}{Color.RESET}")
    trace.emit("case_fingerprint", {"case_fingerprint": case_fingerprint})
    
    ###########################################################################
    # INITIALIZE CHAIR AGENT (No RAG, No Reports)
    ###########################################################################
    print_section("1) Initialize Chair Agent")
    
    from orchestrator.experts import get_role_prompt
    from utils.mdt_runtime_protocol import build_runtime_protocol_digest
    from servers.context_assembly import build_role_specific_case_view
    
    visit_time_str = str(time) if time else "Unknown visit date"
    runtime_protocol_digest = build_runtime_protocol_digest("chair")
    role_prompt = get_role_prompt("chair")
    case_view = build_role_specific_case_view("chair", case_json)
    
    instruction = f"""
{runtime_protocol_digest}

OUTPATIENT VISIT TIME (today's clinic decision point): {visit_time_str}

CASE_FINGERPRINT: {case_fingerprint}

{role_prompt}

# MODE: CHAIR-R (records-only chair baseline)
This mode uses the structured case view only; no retrieved guideline/PubMed evidence or clinical reports are provided.

# HARD RULES (critical)
1) All decisions are for THIS visit date and future care, not for past timepoints.
2) PATIENT FACTS come ONLY from Role-Specific Case View below.
3) Retrieved guideline/PubMed evidence and clinical reports are unavailable in this mode.
4) If critical information is missing, clearly state what data would be needed.
5) Do not invent evidence support; if no provided fact supports a claim, state uncertainty.
6) Citation (case data only - no trial, no guidelines):
   ### FORMAT BY TYPE
   - Lab: [@date | LAB]  e.g. [@2022-01-17 | LAB]
   - Imaging: [@date | MR] or [@date | CT]  e.g. [@2022-08-19 | CT]
   - Genetic: [@date | Genomics]  e.g. [@2021-09 | Genomics]
   - Pathology: [@date | Pathology]  e.g. [@2021-09-08 | Pathology]
   Use spaces around | . Do not use [@date | CASE].

# Role-Specific Case View (PATIENT FACTS)
{case_view}
""".strip()
    
    chair_agent = Agent(
        instruction=instruction,
        role="chair",
        model_info=model,
        client=client,
        max_tokens=2000,
        max_prompt_tokens=20000,
    )
    chair_agent.inject_assistant("System ready for MDT decision.")
    print(f"{Color.OKGREEN}✔ Initialized CHAIR-R agent (records-only baseline){Color.RESET}")
    
    ###########################################################################
    # GENERATE FINAL OUTPUT
    ###########################################################################
    print_section("2) Generate Final Output")
    
    final_prompt = _render_chair_mode_final_prompt(
        "chair_r",
        visit_time_str=visit_time_str,
    )
    
    try:
        final_output = chair_agent.chat(final_prompt)
    except Exception as e:
        print(f"{Color.FAIL}[ERROR] Chair final output failed: {e}{Color.RESET}")
        final_output = f"""Final Assessment:
Final chair synthesis was unavailable for this run.

Core Treatment Strategy:
- Review structured case data and available evidence
- Obtain required clinical reports or specialist review

Change Triggers:
- Reassess after the missing synthesis step is rerun"""
    
    # Build report context from case_json for evidence tag lookup
    report_context = build_case_report_context(case_json)
    
    # Append references section (auto-generated from evidence tags in output)
    final_output = append_references_to_output(final_output, trial_note="", report_context=report_context)
    
    print(final_output)
    trace.emit("final_output_end", {"final_output_chars": len(final_output or "")})
    
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== CHAIR-R Pipeline End ==={Color.RESET}")
    return final_output


###############################################################################
# AUTO - Intelligent Routing Mode
###############################################################################
def process_auto_query(
    question: Any,
    question_raw: Optional[str],
    model: str,
    args: Any,
    time: Optional[str] = None,
    meta_info: Optional[str] = None,
    **kwargs
) -> str:
    """
    Auto mode - Intelligent routing based on case complexity.
    Optional demo/operations helper; not used in the reported evaluation path.
    
    A routing agent analyzes the case and selects the appropriate mode:
    - chair_r: Records-only chair baseline
    - chair_e: Cases needing guideline/literature reference (evidence-augmented)
    - chair_d: Cases needing full evidence (reports + trials) (dossier-augmented)
    - omgs: Complex cases requiring multi-expert discussion
    
    Args:
        question: Case data (dict/list/str)
        question_raw: Original raw question text
        model: Model/deployment name
        args: CLI arguments with client
        time: Visit timestamp
        meta_info: Patient identifier
        **kwargs: Additional arguments passed to selected mode
    
    Returns:
        Final MDT-style output string from selected mode
    """
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== Auto Mode - Intelligent Routing ==={Color.RESET}")
    
    client = args.client
    
    # Normalize question
    raw_question_str = question_to_text(question)
    case_json = sanitize_case_for_decision(safe_load_case_json(raw_question_str))
    question_str = question_to_text(case_json) if case_json else raw_question_str
    
    ###########################################################################
    # ROUTING AGENT - Analyze Case Complexity
    ###########################################################################
    print_section("1) Routing Agent - Analyze Case Complexity")
    
    routing_prompt = f"""
# OMGs System Background (for routing decision)
OMGs (Ovarian-cancer Multidisciplinary intelligent aGent System) is specifically designed for:
- Complex ovarian cancer patients requiring multi-line therapy
- Full lifecycle treatment management (from diagnosis through recurrence)
- Multidisciplinary decision support integrating oncology, radiology, pathology, and nuclear medicine

# Your Task
Analyze the following case and determine which processing mode is most appropriate.

# Available Modes
1. chair_r (CHAIR-R): Records-only chair baseline
2. chair_e (CHAIR-E): Evidence-augmented (guidelines + literature) - for cases needing evidence reference
3. chair_d (CHAIR-D): Dossier-augmented (reports + trials) - for complex cases with available data
4. omgs: Full multi-agent MDT discussion - for highly complex cases requiring multi-specialty debate

# Complexity Factors to Consider
- Line of therapy: newly diagnosed/first-line (simple) -> 2-3 lines (medium) -> 4+ lines (complex)
- Genetic testing: None/simple (simple) → BRCA/HRD present (medium) → Multiple complex mutations (complex)
- Platinum status: Clear (simple) → Borderline (medium) → Complex/contradictory (complex)
- Comorbidities: None/few (simple) → Moderate (medium) → Multiple/severe (complex)
- Clinical questions: Single clear question (simple) → 2-3 questions (medium) → Multiple difficult decisions (complex)

# Case to Analyze
{question_str}

# Output Format (JSON only, no other text)
{{"mode": "chair_r|chair_e|chair_d|omgs", "reason": "brief explanation in English"}}
"""
    
    routing_agent = Agent(
        instruction="You are a clinical triage agent for OMGs. Analyze case complexity and select the appropriate processing mode.",
        role="router",
        model_info=model,
        client=client,
        max_tokens=500,
        max_prompt_tokens=10000,
    )
    
    # Default fallback
    selected_mode = "chair_r"
    routing_reason = "Default fallback because routing was unavailable."
    
    try:
        routing_response = routing_agent.chat(routing_prompt)
        print(f"{Color.OKCYAN}Routing response received.{Color.RESET}")
        
        # Parse JSON response
        routing_data = safe_parse_json_block(routing_response)
        if routing_data and isinstance(routing_data, dict):
            mode = routing_data.get("mode", "").lower().strip()
            if mode in ["chair_r", "chair_e", "chair_d", "omgs"]:
                selected_mode = mode
                routing_reason = routing_data.get("reason", "No reason provided.")
            else:
                print(f"{Color.WARNING}[WARNING] Invalid mode '{mode}', using default{Color.RESET}")
        else:
            print(f"{Color.WARNING}[WARNING] Failed to parse routing response, using default{Color.RESET}")
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Routing agent failed: {e}, using default mode{Color.RESET}")
    
    print(f"{Color.BOLD}{Color.OKBLUE}📊 Selected Mode: {selected_mode}{Color.RESET}")
    print(f"{Color.OKCYAN}   Reason: {routing_reason}{Color.RESET}")
    
    ###########################################################################
    # EXECUTE SELECTED MODE
    ###########################################################################
    print_section(f"2) Execute Selected Mode: {selected_mode}")
    
    # Set auto routing flag so child functions record agent_mode as "auto(xxx)"
    args._auto_routed_mode = f"auto({selected_mode})"
    
    if selected_mode == "chair_r":
        return process_chair_r_query(
            question=question,
            question_raw=question_raw,
            model=model,
            args=args,
            time=time,
            meta_info=meta_info,
        )
    elif selected_mode == "chair_e":
        return process_chair_e_query(
            question=question,
            question_raw=question_raw,
            model=model,
            args=args,
            time=time,
            meta_info=meta_info,
        )
    elif selected_mode == "chair_d":
        return process_chair_d_query(
            question=question,
            question_raw=question_raw,
            model=model,
            args=args,
            time=time,
            meta_info=meta_info,
            **kwargs
        )
    else:  # omgs
        return process_omgs_multi_expert_query(
            question=question,
            question_raw=question_raw,
            model=model,
            args=args,
            time=time,
            meta_info=meta_info,
            **kwargs
        )
