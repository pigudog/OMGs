# =============================================================================
# Evidence Retrieval Core
# -----------------------------------------------------------------------------
# MDT evidence retrieval module. The file is organized by source family; search
# the banners below to locate a section quickly.
#
#   SHARED
#     - "SHARED: TAG & TEXT UTILS"         citation-tag rendering and text cleaning
#     - "SHARED: RAG QUERY BUILDER (MDT)"  case JSON to English RAG query
#     - "SHARED: QUERY SANITIZATION"       de-identification and query cleaning
#     - "SHARED: EVIDENCE DIGEST"          pack/raw evidence to tagged bullets
#     - "SHARED: PACK / RAW MERGE"         guideline and external evidence merge
#
#   NCCN
#     - "NCCN PUBLIC ENTRY"                structural rules, then engine fallback
#     - "NCCN STRUCTURAL MATCHING"         case features to compact matcher rules
#     - "NCCN QUERY BUILDER"               case dict to engine fallback query string
#     - "NCCN ENGINE FALLBACK ADAPTER"     omgs_engine.nccn adapter
#
#   GUIDELINE
#     - "GUIDELINE RAG ENTRY POINTS"       get_guideline_rag / get_global_guideline_rag
#     - "GUIDELINE ENGINE ADAPTER"         omgs_engine.guidelines adapter
#
#   EXTERNAL EVIDENCE
#     - "EXTERNAL EVIDENCE SEARCH"         pubmed_search_pack
#
# The NCCN-specific query builder serves only NCCN engine fallback. The shared
# literature/guideline query builder is build_rag_query_for_mdt.
# =============================================================================

# =============================================================================
# MODULE-LEVEL CONFIG & DEBUG
# -----------------------------------------------------------------------------
# Module-level debug flag and lazy engine handles.
# =============================================================================
import os
import queue
import sys
import threading
from typing import Callable, Dict, Any, Optional, Tuple, List
import re
import time
import json
from utils.mutation_interpretation import build_mutation_guidance, NGS_INTERPRETATION_RULES

# Debug logs are written only when OMG_DEBUG_RAG is enabled.
_DEBUG_ENABLED = os.environ.get("OMG_DEBUG_RAG", "").lower() in ("1", "true", "yes")
_NCCN_ENGINE = None
_GUIDELINES_ENGINE = None
_EXTERNAL_EVIDENCE_ENGINE = None

_DEFAULT_MDT_EVIDENCE_SOURCE_TIMEOUT_SECONDS = 90.0
_DEFAULT_MDT_RAG_QUERY_TIMEOUT_SECONDS = 20.0
_DEFAULT_MDT_RAG_DIGEST_TIMEOUT_SECONDS = 45.0


def _debug_log(location: str, message: str, data: Dict[str, Any] = None):
    """Write debug log entry if debugging is enabled."""
    if not _DEBUG_ENABLED:
        return
    try:
        log_dir = os.path.join("data", "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "rag_debug.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "local-debug",
                "runId": "run1",
                "hypothesisId": "B",
                "location": location,
                "message": message,
                "data": data or {},
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass


# =============================================================================
# SHARED: TAG & TEXT UTILS
# -----------------------------------------------------------------------------
# 跨数据源共用的小工具。被 guideline / external evidence / NCCN 三条链路共同引用：
#   - _get_rag_result_tag:   把一条 rag_raw 转成带页码的 [@guideline:... | Page ...],
#                            [@pubmed | ...],
#                            [@guideline:nccn | ...], [@fda | ...],
#                            [@conference | ...] 等形式的引用标记。
#   - _humanize_engine_text / _flatten_text: engine 返回文本的 un-escape & 压行，
#                            目前主要被 NCCN engine fallback adapter 使用。
# =============================================================================
def _get_rag_result_tag(result: Dict[str, Any], index: int) -> str:
    """Build the exact evidence tag for one rag_raw item."""
    source = result.get("source", "")
    if source == "guideline":
        doc_id = result.get("doc_id", "")
        return _format_guideline_tag(doc_id, result.get("page"), result.get("page_label"))
    if source == "pubmed":
        pmid = result.get("pmid", "")
        return f"[@pubmed | {pmid}]"
    if source == "fda":
        source_id = result.get("source_id") or result.get("label_id") or f"FDA_{index}"
        return f"[@fda | {source_id}]"
    if source == "conference":
        source_id = result.get("source_id") or result.get("abstract_id") or f"CONF_{index}"
        return f"[@conference | {source_id}]"
    if source == "nccn_safety_rule":
        rule_id = result.get("rule_id", f"SAFETY_{index}")
        return f"[@guideline:nccn | {rule_id}]"
    if source == "nccn_matcher_rule":
        rule_id = result.get("rule_id", result.get("node_id", f"MATCHER_{index}"))
        return f"[@guideline:nccn | {rule_id}]"
    if source == "nccn_decision_node":
        node_id = result.get("node_id", result.get("rule_id", f"NODE_{index}"))
        return f"[@guideline:nccn | {node_id}]"
    return f"[unknown source {index}]"


def _humanize_engine_text(text: Any) -> str:
    """Render engine strings for pack output without losing NCCN markers."""
    value = str(text or "").strip()
    return value.replace("\\(", "(").replace("\\)", ")")


def _flatten_text(text: Any) -> str:
    return " ".join(_humanize_engine_text(text).split())


# =============================================================================
# SHARED: RAG QUERY BUILDER (MDT)
# -----------------------------------------------------------------------------
# Shared query generator for guideline and external evidence retrieval. The
# input is structured case JSON and the output is one English query of up to
# 40 words.
#
# Distinguish this from _build_nccn_query_from_case, which is a rule-based
# query string for NCCN engine fallback.
#
# When a MUTATION_REPORT is present, its raw text is prioritized as the source
# of molecular facts before query sanitization.
# =============================================================================
# Ovarian histology aliases live in a data file so that the WHO / NCCN
# taxonomy can evolve without code changes. See
#   src/omgs/products/omgs/mdt_core/files/ovarian_histology_aliases.json
# for schema, data sources and references. Loaded lazily and cached at the
# module level; the matching order is longest-alias-first so that specific
# subtypes (e.g. 高级别浆液性癌) win over generic backstops (e.g. 浆液性癌).
_HISTOLOGY_ALIAS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "files", "ovarian_histology_aliases.json"
)
# Minimal fallback whitelist used if the JSON file is missing / malformed.
_HISTOLOGY_ENGLISH_TERMS_FALLBACK: Tuple[str, ...] = (
    "carcinoma", "cancer", "adenocarcinoma", "serous", "clear cell",
    "endometrioid", "mucinous", "undifferentiated", "carcinosarcoma",
)
# Ordered alias entry: (alias, canonical_en, case_insensitive).
# aliases_zh are case-sensitive (Chinese has no case anyway); aliases_en
# are marked case-insensitive so that "HGSC" / "hgsc" / "HgSc" all match.
_HistAlias = Tuple[str, str, bool]
_HISTOLOGY_CACHE: Optional[Tuple[List[_HistAlias], Tuple[str, ...]]] = None


def _load_histology_aliases() -> Tuple[List[_HistAlias], Tuple[str, ...]]:
    """Return (ordered_aliases, english_term_whitelist), cached after first call.

    ordered_aliases is a list of (alias, canonical_en, case_insensitive) tuples
    sorted by alias length descending so longer / more specific aliases win.
    """
    global _HISTOLOGY_CACHE
    if _HISTOLOGY_CACHE is not None:
        return _HISTOLOGY_CACHE

    try:
        with open(_HISTOLOGY_ALIAS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[WARNING] Histology alias file unavailable ({e}); falling back to minimal whitelist.")
        _HISTOLOGY_CACHE = ([], _HISTOLOGY_ENGLISH_TERMS_FALLBACK)
        return _HISTOLOGY_CACHE

    pairs: List[_HistAlias] = []
    for entry in payload.get("entries", []) or []:
        canonical = str(entry.get("canonical_en") or "").strip()
        if not canonical:
            continue
        for alias in entry.get("aliases_zh", []) or []:
            alias_str = str(alias).strip()
            if alias_str:
                pairs.append((alias_str, canonical, False))
        for alias in entry.get("aliases_en", []) or []:
            alias_str = str(alias).strip()
            if alias_str:
                pairs.append((alias_str, canonical, True))
    pairs.sort(key=lambda t: -len(t[0]))

    whitelist_raw = payload.get("english_term_whitelist") or []
    whitelist = tuple(str(term).lower().strip() for term in whitelist_raw if str(term).strip())
    if not whitelist:
        whitelist = _HISTOLOGY_ENGLISH_TERMS_FALLBACK

    _HISTOLOGY_CACHE = (pairs, whitelist)
    return _HISTOLOGY_CACHE


def _clean_histology_for_query(histology: str) -> str:
    """
    Normalize a histology string to an NCCN-canonical English phrase for RAG queries.

    Taxonomy source: WHO Female Genital Tumours 5th ed (2020) + NCCN Ovarian /
    Less Common Ovarian Histopathologies / Germ Cell / Sex Cord-Stromal v2024,
    loaded from files/ovarian_histology_aliases.json.

    Args:
        histology: Histology string, possibly Chinese, English, or mixed.

    Returns:
        Canonical English phrase (e.g. "high-grade serous carcinoma"). Returns
        "" when the input is empty or fully unrecognized non-English text.
    """
    if not histology:
        return ""

    ordered_aliases, english_terms = _load_histology_aliases()
    hist_clean = histology.strip()
    lower = hist_clean.lower()

    # Step 1: longest-alias-first match. Runs before the English early-exit so
    # that mixed tokens like "恶性Brenner瘤" normalize to their NCCN-canonical
    # phrase instead of being salvaged to bare "Brenner", and so that English
    # abbreviations like "HGSC" / "SCCOHT" normalize to the full phrase instead
    # of being returned verbatim by the ASCII fallback.
    for alias, english, case_insensitive in ordered_aliases:
        if case_insensitive:
            if alias.lower() in lower:
                return english
        else:
            if alias in hist_clean:
                return english

    # Step 2: input already contains English histology vocabulary. Strip
    # non-ASCII so e.g. "clear cell carcinoma 透明细胞癌" -> "clear cell carcinoma".
    if any(term in lower for term in english_terms):
        ascii_only = re.sub(r"[^\x00-\x7F]+", " ", hist_clean)
        return re.sub(r"\s+", " ", ascii_only).strip()

    # Step 3: salvage any embedded latin letters (e.g. "HER2 阳性" -> "HER2").
    english_words = re.findall(r"[A-Za-z][A-Za-z\-]*", hist_clean)
    if english_words:
        return " ".join(english_words)

    return ""


def _as_mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _compact_text(value: Any, *, limit: int = 120) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in (
            "result",
            "interpretation",
            "status",
            "score",
            "value",
            "text",
            "summary",
            "description",
        ):
            text = _compact_text(value.get(key), limit=limit)
            if text:
                return text
        parts = []
        for key, item in value.items():
            text = _compact_text(item, limit=limit)
            if text:
                parts.append(f"{key} {text}")
            if len(parts) >= 2:
                break
        return "; ".join(parts)[:limit].rstrip()
    if isinstance(value, list):
        parts = [_compact_text(item, limit=limit) for item in value]
        text = "; ".join(part for part in parts if part)
        return text[:limit].rstrip()
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.lower() in {"unknown", "none", "null", "not reported"}:
        return ""
    return text[:limit].rstrip()


def _first_nonempty(*values: Any, limit: int = 120) -> str:
    for value in values:
        text = _compact_text(value, limit=limit)
        if text:
            return text
    return ""


def _normalize_platinum_status_for_query(value: str) -> str:
    text = _compact_text(value, limit=60).lower()
    if not text:
        return ""
    text = text.replace("_", "-").replace(" ", "-")
    text = re.sub(r"^-?platinum-?", "", text)
    if text in {"resistant", "refractory", "sensitive"}:
        return f"platinum-{text}"
    return f"platinum-{text}" if not text.startswith("platinum-") else text


def _pfi_text_for_query(value: str) -> str:
    text = _compact_text(value, limit=40)
    if not text:
        return ""
    if re.search(r"\b(day|days|month|months|week|weeks|year|years)\b", text, re.IGNORECASE):
        return f"PFI {text}"
    return f"PFI {text} days"


def _biomarker_text_for_query(label: str, value: Any) -> str:
    text = _compact_text(value, limit=90)
    if not text:
        return ""
    if label.lower() in {"her2", "fralpha"}:
        # Older extracted JSON sometimes stores interpretive prose after the
        # semicolon; keep the assay state and avoid half-sentence truncation.
        text = re.split(r";|\.\s+", text, maxsplit=1)[0].strip()
    return f"{label} {text}"


def _append_unique_text(items: List[str], text: str, *, limit: int = 180) -> None:
    cleaned = _compact_text(text, limit=limit)
    if not cleaned:
        return
    key = re.sub(r"\W+", "", cleaned).lower()
    if any(re.sub(r"\W+", "", item).lower() == key for item in items):
        return
    items.append(cleaned)


def _iter_text_fragments(value: Any, *, limit: int = 240) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        fragments: List[str] = []
        for item in value.values():
            fragments.extend(_iter_text_fragments(item, limit=limit))
        return fragments
    if isinstance(value, list):
        fragments = []
        for item in value:
            fragments.extend(_iter_text_fragments(item, limit=limit))
        return fragments
    text = _compact_text(value, limit=limit)
    return [text] if text else []


def _find_first_matching_text(value: Any, patterns: tuple[str, ...], *, limit: int = 120) -> str:
    for fragment in _iter_text_fragments(value, limit=240):
        lower = fragment.lower()
        if any(pattern in lower for pattern in patterns):
            return _compact_text(fragment, limit=limit)
    return ""


def _looks_like_unperformed_marker_text(text: str, marker_patterns: tuple[str, ...]) -> bool:
    lower = text.lower()
    if not any(pattern in lower for pattern in marker_patterns):
        return False
    aliases = "|".join(re.escape(pattern) for pattern in sorted(marker_patterns, key=len, reverse=True))
    no_marker_re = re.compile(rf"\b(?:no|without)\b[^.;\n]{{0,80}}\b(?:{aliases})\b")
    marker_not_done_re = re.compile(
        rf"\b(?:{aliases})\b[^.;\n]{{0,80}}\b(?:"
        r"not reported|not available|not performed|not tested|"
        r"was not reported|were not reported|was not performed|were not performed|"
        r"was not tested|were not tested"
        r")\b"
    )
    did_not_include_re = re.compile(rf"\bdid not include\b[^.;\n]{{0,80}}\b(?:{aliases})\b")
    if no_marker_re.search(lower) or marker_not_done_re.search(lower) or did_not_include_re.search(lower):
        return True
    return False


def _collect_pathology_marker(specimens: Any, marker_patterns: tuple[str, ...], *, limit: int = 120) -> str:
    if not isinstance(specimens, list):
        return ""
    # Prefer structured current IHC/molecular results across all specimens before
    # using raw text; older narrative notes may only say a marker was not tested.
    for specimen in specimens:
        if not isinstance(specimen, dict):
            continue
        for section_name in ("ihc", "molecular"):
            section = specimen.get(section_name)
            if not isinstance(section, list):
                continue
            for item in section:
                if not isinstance(item, dict):
                    continue
                marker_text = " ".join(_iter_text_fragments(item.get("marker"), limit=80))
                test_text = " ".join(_iter_text_fragments(item.get("test"), limit=80))
                value_text = " ".join(_iter_text_fragments(item.get("result"), limit=limit))
                combined = " ".join(part for part in (marker_text, test_text, value_text) if part)
                lower = combined.lower()
                if any(pattern in lower for pattern in marker_patterns):
                    return _compact_text(value_text or combined, limit=limit)
    for specimen in specimens:
        if not isinstance(specimen, dict):
            continue
        raw_match = _find_first_matching_text(specimen.get("raw_text"), marker_patterns, limit=limit)
        if raw_match and not _looks_like_unperformed_marker_text(raw_match, marker_patterns):
            return raw_match
    return ""


def _collect_genomic_alteration(root: Dict[str, Any], case_core: Dict[str, Any], gene: str, *, limit: int = 90) -> str:
    gene_lower = gene.lower()
    genomics = _as_mapping(case_core.get("GENOMICS"))
    alterations = genomics.get("alterations")
    if isinstance(alterations, list):
        for alteration in alterations:
            if not isinstance(alteration, dict):
                continue
            if _compact_text(alteration.get("gene"), limit=40).lower() != gene_lower:
                continue
            status = _compact_text(alteration.get("status"), limit=50)
            variant = _compact_text(alteration.get("variant"), limit=limit)
            return _compact_text(" ".join(part for part in (status, variant) if part), limit=limit)
    med_onc = _as_mapping(root.get("MED_ONC"))
    return _find_first_matching_text(med_onc.get("genetic_testing"), (gene_lower,), limit=limit)


def _collect_brca_text(root: Dict[str, Any], case_core: Dict[str, Any], *, limit: int = 140) -> str:
    genomics = _as_mapping(case_core.get("GENOMICS"))
    alterations = genomics.get("alterations")
    parts: List[str] = []
    if isinstance(alterations, list):
        for alteration in alterations:
            if not isinstance(alteration, dict):
                continue
            gene = _compact_text(alteration.get("gene"), limit=40)
            if not re.search(r"\bBRCA\s*1\b|\bBRCA\s*2\b|\bBRCA1\b|\bBRCA2\b|BRCA1/2", gene, re.IGNORECASE):
                continue
            details = " ".join(
                part
                for part in (
                    gene,
                    _compact_text(alteration.get("status"), limit=60),
                    _compact_text(alteration.get("variant"), limit=120),
                    _compact_text(alteration.get("clinical_significance"), limit=80),
                )
                if part
            )
            _append_unique_text(parts, details, limit=limit)
    if parts:
        return "; ".join(parts)[:limit].rstrip()
    med_onc = _as_mapping(root.get("MED_ONC"))
    return _find_first_matching_text(med_onc.get("genetic_testing"), ("brca",), limit=limit)


def _collect_prior_exposure(root: Dict[str, Any], case_core: Dict[str, Any]) -> str:
    sources: List[Any] = []
    med_onc = _as_mapping(root.get("MED_ONC"))
    sources.append(med_onc.get("prior_systemic_therapies"))
    maintenance = _as_mapping(case_core.get("MAINTENANCE_DETAIL"))
    sources.append(maintenance.get("regimens"))
    lines = case_core.get("LINE_OF_THERAPY")
    if isinstance(lines, list):
        sources.extend(line.get("regimen") for line in lines if isinstance(line, dict))
    text = " ".join(fragment.lower() for source in sources for fragment in _iter_text_fragments(source, limit=240))
    exposures: List[str] = []
    if "bevacizumab" in text:
        exposures.append("bevacizumab")
    if any(term in text for term in ("parp", "niraparib", "olaparib", "rucaparib")):
        exposures.append("PARP inhibitor")
    if exposures:
        return "prior " + " and ".join(exposures) + " exposure"
    return ""


def _collect_mdt_query_markers(root: Dict[str, Any], case_core: Dict[str, Any]) -> List[str]:
    biomarkers = _as_mapping(case_core.get("BIOMARKERS"))
    pathology = _as_mapping(root.get("PATHOLOGY"))
    specimens = pathology.get("specimens")
    marker_parts: List[str] = []

    fralpha = _first_nonempty(
        biomarkers.get("FRalpha"),
        _collect_pathology_marker(specimens, ("fralpha", "folr1", "folate receptor"), limit=120),
        _find_first_matching_text(case_core.get("CURRENT_STATUS"), ("fralpha", "folr1", "folate receptor"), limit=120),
    )
    if fralpha:
        text = _biomarker_text_for_query("FRalpha", fralpha)
        if "folate receptor alpha adc" not in text.lower():
            text = f"{text} folate receptor alpha ADC"
        _append_unique_text(marker_parts, text)

    her2 = _first_nonempty(
        biomarkers.get("HER2"),
        _collect_pathology_marker(specimens, ("her2",), limit=120),
        _find_first_matching_text(case_core.get("CURRENT_STATUS"), ("her2",), limit=120),
    )
    if her2:
        _append_unique_text(marker_parts, _biomarker_text_for_query("HER2", her2))

    genomics = _as_mapping(case_core.get("GENOMICS"))
    hrd_status = _as_mapping(genomics.get("HRD_STATUS"))
    hrd = _first_nonempty(case_core.get("HRD"), hrd_status.get("result"), limit=40)
    brca_text = _first_nonempty(
        biomarkers.get("BRCA"),
        _collect_brca_text(root, case_core),
    )
    brca1 = _first_nonempty(case_core.get("BRCA1"), limit=32)
    brca2 = _first_nonempty(case_core.get("BRCA2"), limit=32)
    if brca_text and "no pathogenic" in brca_text.lower():
        brca_phrase = "BRCA1/2 no pathogenic variant"
    elif brca_text:
        brca_phrase = _biomarker_text_for_query("BRCA1/2", brca_text)
    elif brca1 and brca2 and brca1.lower() == brca2.lower() == "wildtype":
        brca_phrase = "BRCA1/2 no pathogenic variant"
    else:
        brca_values = "; ".join(
            part for part in (_biomarker_text_for_query("BRCA1", brca1), _biomarker_text_for_query("BRCA2", brca2)) if part
        )
        brca_phrase = brca_values
    if hrd and brca_phrase:
        _append_unique_text(marker_parts, f"{brca_phrase} HRD {hrd}")
    elif hrd:
        _append_unique_text(marker_parts, _biomarker_text_for_query("HRD", hrd))
    elif brca_phrase:
        _append_unique_text(marker_parts, brca_phrase)

    ccne1 = _first_nonempty(
        biomarkers.get("CCNE1"),
        _collect_genomic_alteration(root, case_core, "CCNE1", limit=90),
        _find_first_matching_text(case_core.get("CURRENT_STATUS"), ("ccne1",), limit=90),
    )
    if ccne1:
        _append_unique_text(marker_parts, f"CCNE1 {ccne1} clinical trial")

    for label, value in (
        ("MSI", biomarkers.get("MSI")),
        ("TMB", biomarkers.get("TMB")),
        ("PD-L1", biomarkers.get("PDL1_CPS")),
    ):
        text = _biomarker_text_for_query(label, value)
        if text:
            _append_unique_text(marker_parts, text)

    return marker_parts


def _truthy_env(value: str | None, *, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return default


def _mdt_rag_query_timeout_seconds() -> float:
    raw = str(os.environ.get("OMGS_MDT_RAG_QUERY_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return _DEFAULT_MDT_RAG_QUERY_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_MDT_RAG_QUERY_TIMEOUT_SECONDS
    return max(value, 1.0)


def _mdt_rag_digest_timeout_seconds() -> float:
    raw = str(os.environ.get("OMGS_MDT_RAG_DIGEST_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return _DEFAULT_MDT_RAG_DIGEST_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_MDT_RAG_DIGEST_TIMEOUT_SECONDS
    return max(value, 1.0)


def _build_deterministic_mdt_rag_query(question: str, key_facts: str | None = None) -> str:
    """
    Build the default MDT evidence query without an LLM call.

    This keeps the live room from spending a minute or more in the query-builder
    model before any NCCN/guideline/external completion can stream to the UI.
    """
    try:
        parsed = json.loads(question) if isinstance(question, str) else question
    except Exception:
        parsed = {}
    root = _as_mapping(parsed)
    case_core = _as_mapping(root.get("CASE_CORE"))
    diagnosis = _as_mapping(case_core.get("DIAGNOSIS"))
    current_status = _compact_text(case_core.get("CURRENT_STATUS"), limit=180)

    histology = _clean_histology_for_query(
        _first_nonempty(diagnosis.get("histology"), diagnosis.get("primary"), limit=100)
    )
    platinum_status = _first_nonempty(
        case_core.get("PLATINUM_STATUS_CURRENT"),
        case_core.get("PLATINUM_STATUS"),
        limit=40,
    )
    pfi = _first_nonempty(case_core.get("PLATINUM_PFI_CURRENT"), case_core.get("PFI_days"), limit=24)
    ecog = _first_nonempty(case_core.get("ECOG"), limit=20)

    parts: List[str] = []
    if platinum_status:
        status = _normalize_platinum_status_for_query(platinum_status)
        if pfi:
            status = f"{status} {_pfi_text_for_query(pfi)}"
        parts.append(status)
    if histology:
        parts.append(histology)
    elif current_status:
        parts.append(current_status)

    marker_parts = _collect_mdt_query_markers(root, case_core)
    if marker_parts:
        parts.append("; ".join(marker_parts[:6]))
    prior_exposure = _collect_prior_exposure(root, case_core)
    if prior_exposure:
        parts.append(prior_exposure)
    if ecog:
        parts.append(f"ECOG {ecog}")

    query = "; ".join(part for part in parts if part)
    if not query and key_facts:
        query = _compact_text(key_facts.replace("\n", "; "), limit=260)
    if not query:
        query = "ovarian cancer treatment guidelines"
    return query


def build_rag_query_for_mdt(agent, question: str, key_facts: str | None = None) -> str:
    """
    Generate a concise English RAG query from structured CASE JSON.
    
    Args:
        agent: Agent instance for running the query builder
        question: Structured JSON string containing the full case information
    
    Returns:
        A concise English query string (<=40 words) for RAG retrieval
    """
    deterministic_query = _build_deterministic_mdt_rag_query(question, key_facts=key_facts)
    use_llm_query = _truthy_env(os.environ.get("OMGS_MDT_LLM_RAG_QUERY"), default=True)
    if not use_llm_query:
        sanitized, changed = sanitize_rag_query(deterministic_query)
        if changed:
            print("[WARNING] RAG query contained potential identifiers and was sanitized before logging/search.")
        return sanitized

    from core.config import get_mdt_prompts
    rag_prompts = get_mdt_prompts().get("rag", {})
    
    facts_block = f"# KEY FACTS (from structured case)\n{key_facts}\n\n" if key_facts else ""
    
    # IMPORTANT: If MUTATION_REPORT exists, ignore GENETICS section from case_core
    # Mutation reports are the source of truth - case_core may have "not reported" even when reports exist
    has_mutation_report = key_facts and "MUTATION_REPORT" in key_facts
    
    # Add mutation report interpretation guidance if mutation report is present
    mutation_guidance = ""
    if has_mutation_report:
        # Extract the full raw_text from mutation report
        mut_report_raw = ""
        if key_facts:
            mut_match = re.search(r'MUTATION_REPORT:.*?full_text=([^\n]+)', key_facts, re.DOTALL)
            if mut_match:
                mut_report_raw = mut_match.group(1).strip()
        
        # Build comprehensive mutation guidance with raw text and interpretation rules
        mutation_guidance = build_mutation_guidance(mut_report_raw)
    
    # Only add GENETICS guidance if NO mutation report exists (mutation report takes precedence)
    genetics_guidance = ""
    if not has_mutation_report and key_facts and "GENETICS:" in key_facts:
        genetics_match = re.search(r'GENETICS:\s*HRD=([^;]+);\s*BRCA1=([^;]+);\s*BRCA2=([^;]+)', key_facts)
        if genetics_match:
            hrd_val = genetics_match.group(1).strip()
            brca1_val = genetics_match.group(2).strip()
            brca2_val = genetics_match.group(3).strip()
            genetics_guidance = "\nCRITICAL: In KEY FACTS, you see 'GENETICS: HRD={}; BRCA1={}; BRCA2={}'. ".format(hrd_val, brca1_val, brca2_val)
            if hrd_val != "Unknown" and hrd_val != "unknown":
                genetics_guidance += f"HRD test WAS performed, result is {hrd_val.lower()}. You MUST say 'HRD-{hrd_val.lower()}' in your query, NOT 'not reported'. "
            if brca1_val != "Unknown" and brca1_val != "unknown":
                genetics_guidance += f"BRCA1 test WAS performed, result is {brca1_val.lower()}. "
            if brca2_val != "Unknown" and brca2_val != "unknown":
                genetics_guidance += f"BRCA2 test WAS performed, result is {brca2_val.lower()}. "
            if (brca1_val != "Unknown" and brca1_val != "unknown") or (brca2_val != "Unknown" and brca2_val != "unknown"):
                genetics_guidance += "You MUST say 'BRCA-negative' or 'BRCA1/BRCA2-negative' in your query, NOT 'not reported'. "
            genetics_guidance += "The word 'Negative' or 'Positive' means the test was done. Only 'Unknown' means not tested.\n\n"
    
    query_builder_template = rag_prompts.get("query_builder",
        "You are preparing a single concise English query to retrieve guideline/clinical evidence "
        "for this ovarian cancer MDT case.\n\n"
        "# STRUCTURED_CASE_TEXT\n{question}\n\n"
        "Write ONE line (<=40 words) focusing on:\n"
        "- tumor type/histology and platinum status;\n"
        "- key metastases / disease extent;\n"
        "- key molecular markers if mentioned (e.g., BRCA/HRD/MSI/PD-L1);\n"
        "- major clinical constraints (e.g., anemia, organ function, performance).\n"
        "Do NOT mention report_ids, dates, hospital names, or patient identifiers.\n"
        "If KEY FACTS include histology or platinum/genetic status, you MUST include them.\n"
        "Do NOT say 'unknown' if a KEY FACT is provided.\n"
        "Output ONLY the query text."
    )
    # Build prompt with mutation_guidance at the END (right before final output instruction)
    # This addresses LLM position bias - important instructions should be near the output
    base_prompt = query_builder_template.format(question=question)
    
    # Insert mutation_guidance right before "Output ONLY" for maximum impact
    if mutation_guidance and "Output ONLY" in base_prompt:
        parts = base_prompt.rsplit("Output ONLY", 1)
        base_prompt = parts[0] + mutation_guidance + "Output ONLY" + parts[1]
    elif mutation_guidance:
        base_prompt = base_prompt + "\n" + mutation_guidance
    
    # Add genetics_guidance if present (no mutation report)
    if genetics_guidance and "Output ONLY" in base_prompt:
        parts = base_prompt.rsplit("Output ONLY", 1)
        base_prompt = parts[0] + genetics_guidance + "Output ONLY" + parts[1]
    elif genetics_guidance:
        base_prompt = base_prompt + "\n" + genetics_guidance
    
    prompt = facts_block + base_prompt
    
    try:
        timeout_seconds = _mdt_rag_query_timeout_seconds()
        try:
            raw_query = agent.run_selection(prompt, timeout=timeout_seconds).strip()
        except TypeError:
            raw_query = agent.run_selection(prompt).strip()
    except Exception as e:
        print(
            f"[WARNING] RAG query builder failed or timed out; using deterministic fallback: {e}",
            flush=True,
        )
        raw_query = deterministic_query
    
    sanitized, changed = sanitize_rag_query(raw_query)
    if changed:
        print("[WARNING] RAG query contained potential identifiers and was sanitized before logging/search.")
    if key_facts:
        hist_match = re.search(r"histology=([^;\n]+)", key_facts, re.IGNORECASE)
        if hist_match:
            hist_raw = hist_match.group(1).strip()
            hist_clean = _clean_histology_for_query(hist_raw)
            
            # Only append if we have a cleaned English histology
            if hist_clean and hist_clean.lower() not in ["unknown", "not specified", ""]:
                if re.search(r"histology\s+(not\s+specified|unknown)", sanitized, re.IGNORECASE):
                    sanitized = re.sub(
                        r"histology\s+(not\s+specified|unknown)",
                        f"histology {hist_clean}",
                        sanitized,
                        flags=re.IGNORECASE,
                    )
                elif re.search(r"\bhistology\b", sanitized, re.IGNORECASE) is None:
                    sanitized = sanitized.rstrip(".;") + f"; histology: {hist_clean}"
    return sanitized


# =============================================================================
# SHARED: QUERY SANITIZATION
# -----------------------------------------------------------------------------
# 在写日志 / 发给外部检索之前，把 email / 长数字串 / 身份证 / 病历号等明显 PII
# 从 query 中剔除。guideline 和 pubmed 的 query 都会过这一层；NCCN fallback 的
# query 是从 case dict 结构化拼出来的，不经过自由文本，通常不需要。
# =============================================================================
def sanitize_rag_query(query: str) -> tuple[str, bool]:
    """Remove obvious identifiers from RAG query (defensive de-id before logging/search)."""
    if not query:
        return "", False
    original = query
    q = query
    # Emails
    q = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", q)
    # Labelled identifiers. Accept both "MRN: 123" and "MRN 123" forms.
    english_identifier_labels = (
        r"meta[_\s-]?info|patient[_\s-]?id|patient\s+identifier|"
        r"report[_\s-]?id|case[_\s-]?id|sample[_\s-]?id|mrn|"
        r"medical\s+record(?:\s+number)?|hospital\s+number"
    )
    q = re.sub(
        rf"(?i)\b(?:{english_identifier_labels})\b\s*(?::|=|#|-|\bis\b)?\s*[A-Za-z0-9][A-Za-z0-9._/-]{{2,}}",
        "[REDACTED_ID]",
        q,
    )
    chinese_identifier_labels = r"(?:住院号|病历号|病案号|身份证号|患者编号|报告编号)"
    q = re.sub(
        rf"{chinese_identifier_labels}\s*(?::|：|=|#|-)?\s*[A-Za-z0-9][A-Za-z0-9._/-]{{2,}}",
        "[REDACTED_ID]",
        q,
    )
    # Common report-code forms copied from structured report IDs.
    q = re.sub(
        r"\b(?:LAB|MR|CT|PX|PATH|IMG|OH|GEN|REPORT)[-_]?\d{6,}[A-Za-z0-9_-]*\b",
        "[REDACTED_ID]",
        q,
        flags=re.IGNORECASE,
    )
    # Phones / long digit sequences (avoid removing short clinical numbers like CA-125)
    q = re.sub(r"\b\d{8,}\b", "[REDACTED_ID]", q)
    # Collapse extra whitespace
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q, (q != original)


# =============================================================================
# SHARED: EVIDENCE DIGEST
# -----------------------------------------------------------------------------
# 把 (rag_pack, rag_raw) 交给 LLM 汇总成 1:1 的 evidence bullets。支持
# guideline / nccn_safety_rule / nccn_matcher_rule / nccn_decision_node / pubmed / fda / conference。每条
# bullet 必须带对应引用 tag（由 SHARED: TAG & TEXT UTILS 的 _get_rag_result_tag
# 生成），以便下游 chair 整合时能反查回原始 chunk。
# =============================================================================
def summarize_rag_evidence(agent, rag_pack: str, rag_raw: List[Dict[str, Any]] = None) -> str:
    """
    Summarize guideline RAG chunks into actionable evidence bullets.
    
    Args:
        agent: Agent instance for running the summarization
        rag_pack: String containing RAG evidence chunks
        rag_raw: Optional list of raw RAG results for counting and reference mapping
    
    Returns:
        Summarized evidence as plain text bullets (one per RAG result, 1:1 mapping)
    """
    from core.config import get_mdt_prompts
    rag_prompts = get_mdt_prompts().get("rag", {})
    
    # Build explicit reference mapping for each RAG result
    total_count = len(rag_raw) if rag_raw else 0
    guideline_count = len([r for r in (rag_raw or []) if r.get("source") == "guideline"])
    nccn_count = len([r for r in (rag_raw or []) if str(r.get("source", "")).startswith("nccn")])
    pubmed_count = len([r for r in (rag_raw or []) if r.get("source") == "pubmed"])
    fda_count = len([r for r in (rag_raw or []) if r.get("source") == "fda"])
    conference_count = len([r for r in (rag_raw or []) if r.get("source") == "conference"])
    
    # Build reference tags list for explicit mapping
    ref_tags_list = []
    for i, r in enumerate(rag_raw or [], 1):
        tag = _get_rag_result_tag(r, i)
        ref_tags_list.append(f"  [{i}] {tag}")
    
    ref_tags_str = "\n".join(ref_tags_list) if ref_tags_list else ""
    
    count_info = ""
    if total_count > 0:
        count_info = (
            f"\nCRITICAL: There are exactly {total_count} RAG results "
            f"({guideline_count} guidelines, {nccn_count} NCCN, {pubmed_count} PubMed, "
            f"{fda_count} FDA, {conference_count} conferences).\n"
            f"You MUST output exactly {total_count} bullets, one per result, in order.\n\n"
            f"REFERENCE TAGS (use these EXACTLY):\n{ref_tags_str}\n\n"
            "Each bullet MUST use the corresponding tag from the list above.\n"
        )
    
    evidence_summarizer_template = rag_prompts.get("evidence_summarizer",
        "# RAG CHUNKS\n{rag_pack}\n\n"
        "{count_info}"
        "Summarize into evidence bullets for MDT decision-making.\n"
        "Rules:\n"
        "- Output exactly {total_count} bullets, one per RAG result, in order.\n"
        "- Each bullet summarizes ONE RAG chunk with its corresponding tag.\n"
        "- Each bullet must be actionable evidence (guideline/trial-based).\n"
        "- Do NOT restate patient-specific facts.\n"
        "- Avoid long quotes; keep each bullet concise (1-2 sentences).\n"
        "- Each bullet MUST include the exact evidence tag from the REFERENCE TAGS list.\n"
        "- Output ONLY plain text bullets, no numbering."
    )
    prompt = evidence_summarizer_template.format(
        rag_pack=rag_pack, 
        count_info=count_info,
        total_count=total_count,
    )
    try:
        timeout_seconds = _mdt_rag_digest_timeout_seconds()
        try:
            return agent.run_selection(prompt, timeout=timeout_seconds)
        except TypeError:
            return agent.run_selection(prompt)
    except Exception as e:
        # Fallback: create simple digest from RAG raw results
        print(f"[WARNING] RAG evidence summarization failed: {e}")
        if not rag_raw:
            return "# No RAG evidence available"
        
        digest_lines = []
        for i, r in enumerate(rag_raw[:min(total_count, 8)], 1):  # Limit to 8 bullets
            source = r.get("source", "")
            if source == "guideline":
                tag = _get_rag_result_tag(r, i)
                text = r.get("text", "")
            elif source in {"nccn_safety_rule", "nccn_matcher_rule", "nccn_decision_node"}:
                tag = _get_rag_result_tag(r, i)
                text = r.get("text", "") or r.get("node_name", "")
            elif source == "pubmed":
                tag = _get_rag_result_tag(r, i)
                text = r.get("abstract", "") or r.get("title", "")
            elif source in {"fda", "conference"}:
                tag = _get_rag_result_tag(r, i)
                text = r.get("text", "") or r.get("abstract", "") or r.get("summary", "") or r.get("title", "")
            else:
                tag = f"[unknown source {i}]"
                text = r.get("text", "") or ""
            
            # Create a simple bullet from the text
            preview = text[:150].strip() if text else "Evidence available"
            if len(text) > 150:
                preview += "..."
            digest_lines.append(f"- {preview} {tag}")
        
        return "\n".join(digest_lines) if digest_lines else "# No RAG evidence available"


# =============================================================================
# SHARED: PACK / RAW MERGE
# -----------------------------------------------------------------------------
# 把 guideline 和 pubmed 两条独立检索的结果合并成上游消费的统一 pack / raw。
# NCCN 走自己的 (nccn_pack, nccn_raw)，不参与这里的合并。
# =============================================================================
def merge_rag_packs(guideline_pack: str, pubmed_pack: str) -> str:
    parts = []
    if guideline_pack:
        parts.append("# GUIDELINE RAG\n" + guideline_pack)
    if pubmed_pack:
        parts.append("# PUBMED RAG\n" + pubmed_pack)
    if not parts:
        return "(RAG: no evidence found)"
    return "\n\n".join(parts)


def merge_rag_raw(guideline_raw, pubmed_raw):
    return (guideline_raw or []) + (pubmed_raw or [])


def _retrieve_mdt_nccn_source(
    *,
    case_question: Any,
    rag_query: str,
    device: str,
    topk: int,
    unavailable_pack: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        nccn_pack, nccn_raw = get_nccn_rag(
            question=case_question,
            device=device,
            topk=topk,
        )
        if "(NCCN: no structural match found)" in nccn_pack:
            nccn_pack, nccn_raw = get_nccn_rag(
                question=rag_query,
                device=device,
                topk=topk,
            )
        if nccn_pack == "(RAG: initialization failed)":
            return unavailable_pack, []
        return nccn_pack, nccn_raw or []
    except Exception as exc:
        print(f"[WARNING] NCCN RAG failed: {exc}")
        return unavailable_pack, []


def _retrieve_mdt_guideline_source(
    *,
    rag_query: str,
    device: str,
    topk: int,
    unavailable_pack: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        guideline_pack, guideline_raw = get_global_guideline_rag(
            question=rag_query,
            device=device,
            topk=topk,
        )
        if guideline_pack == "(RAG: initialization failed)":
            return unavailable_pack, []
        return guideline_pack, guideline_raw or []
    except Exception as exc:
        print(f"[WARNING] Guideline RAG failed: {exc}")
        return unavailable_pack, []


def _retrieve_mdt_guideline_role_source(
    *,
    role: str,
    rag_query: str,
    topk: int,
    unavailable_pack: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        guideline_pack, guideline_raw = rag_search_pack(
            query=rag_query,
            topk=topk,
            guideline_scope=_guideline_scope_for_role(role),
        )
        if guideline_pack == "(RAG: initialization failed)":
            return unavailable_pack, []
        return guideline_pack, guideline_raw or []
    except Exception as exc:
        print(f"[WARNING] role-private guideline RAG failed for {role}: {exc}")
        return unavailable_pack, []


def _nccn_matcher_topk(topk: Any) -> int:
    try:
        value = int(topk)
    except (TypeError, ValueError):
        value = 3
    return max(1, value)


def _retrieve_mdt_external_source(
    *,
    rag_query: str,
    topk: int,
    unavailable_pack: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        pubmed_pack, pubmed_raw = pubmed_search_pack(
            query=rag_query,
            topk=topk,
        )
        return pubmed_pack, pubmed_raw or []
    except Exception as exc:
        print(f"[WARNING] PubMed RAG failed: {exc}")
        return unavailable_pack, []


def retrieve_mdt_external_evidence(
    *,
    rag_query: str,
    topk: int = 8,
    unavailable_pack: str = "(PUBMED: no evidence found)",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve one external-evidence lane for MDT evidence assembly."""
    return _retrieve_mdt_external_source(
        rag_query=rag_query,
        topk=topk,
        unavailable_pack=unavailable_pack,
    )


_ENGINE_EXTERNAL_PUBMED_TOPK_BY_DEPTH = {
    "fast": 6,
    "balanced": 8,
    "standard": 8,
    "default": 8,
    "deep": 10,
    "exhaustive": 10,
    "full": 10,
    "multi": 10,
}


def mdt_external_pubmed_topk_from_engine_depth() -> int:
    """Return the PubMed cap that matches the engine external-evidence depth."""
    try:
        from utils.runtime_config import tool_input_overrides_from_env

        overrides = tool_input_overrides_from_env("external_evidence")
        depth = str(overrides.get("search_depth") or "balanced").strip().lower()
    except Exception:
        depth = str(os.environ.get("OMGS_EXTERNAL_SEARCH_DEPTH") or "balanced").strip().lower()
    return _ENGINE_EXTERNAL_PUBMED_TOPK_BY_DEPTH.get(depth, 8)


def _retrieve_mdt_evidence_lanes(
    *,
    lane_queries: Dict[str, str],
    source_label: str,
    worker: Callable[..., Tuple[str, List[Dict[str, Any]]]],
    kwargs_for_lane: Callable[[str, str], Dict[str, Any]],
    unavailable_pack: str,
) -> Dict[str, Tuple[str, List[Dict[str, Any]]]]:
    """Retrieve multiple role-private evidence lanes with bounded parallelism and timeout."""
    timeout_seconds = _mdt_evidence_source_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds
    output_queue: queue.Queue = queue.Queue()
    results: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    pending = [name for name, query in (lane_queries or {}).items() if str(query or "").strip()]
    pending_set = set(pending)
    lane_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    for name in pending:
        lane_queue.put(name)

    def _run_lane_worker() -> None:
        while not stop_event.is_set():
            try:
                name = lane_queue.get_nowait()
            except queue.Empty:
                return
            if stop_event.is_set():
                return
            _run_mdt_evidence_source_worker(
                name=name,
                worker=worker,
                kwargs=kwargs_for_lane(name, lane_queries[name]),
                output_queue=output_queue,
            )

    worker_count = min(len(pending), _mdt_external_lane_parallelism())
    for i in range(worker_count):
        thread = threading.Thread(
            target=_run_lane_worker,
            name=f"mdt-{source_label}-evidence-lane-{i + 1}",
            daemon=True,
        )
        thread.start()

    while pending_set:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_event.set()
            break
        try:
            name, status, pack, raw, exc = output_queue.get(timeout=remaining)
        except queue.Empty:
            stop_event.set()
            break
        if name not in pending_set:
            continue
        pending_set.remove(name)
        if status == "failed":
            print(f"[WARNING] role-private {source_label} evidence retrieval failed for {name}: {exc}", flush=True)
            results[name] = (unavailable_pack, [])
            continue
        results[name] = (pack, raw or [])

    for name in pending_set:
        print(
            f"[WARNING] role-private {source_label} evidence retrieval for {name} timed out after "
            f"{timeout_seconds:.1f}s; continuing with fallback.",
            flush=True,
        )
        results[name] = (unavailable_pack, [])

    for name in lane_queries or {}:
        results.setdefault(name, (unavailable_pack, []))
    return results


def retrieve_mdt_guideline_evidence(
    *,
    role: str,
    rag_query: str,
    topk: int = 5,
    unavailable_pack: str = "(Guideline: no role-specific evidence found)",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve one role-scoped guideline lane for MDT evidence assembly."""
    return _retrieve_mdt_guideline_role_source(
        role=role,
        rag_query=rag_query,
        topk=topk,
        unavailable_pack=unavailable_pack,
    )


def retrieve_mdt_guideline_evidence_lanes(
    *,
    lane_queries: Dict[str, str],
    topk: int = 5,
    unavailable_pack: str = "(Guideline: no role-specific evidence found)",
) -> Dict[str, Tuple[str, List[Dict[str, Any]]]]:
    """Retrieve multiple role-scoped guideline lanes with the external-lane pattern."""
    return _retrieve_mdt_evidence_lanes(
        lane_queries=lane_queries,
        source_label="guideline",
        worker=_retrieve_mdt_guideline_role_source,
        kwargs_for_lane=lambda role, query: {
            "role": role,
            "rag_query": query,
            "topk": topk,
            "unavailable_pack": unavailable_pack,
        },
        unavailable_pack=unavailable_pack,
    )


def retrieve_mdt_external_evidence_lanes(
    *,
    lane_queries: Dict[str, str],
    topk: int = 8,
    unavailable_pack: str = "(PUBMED: no role-specific evidence found)",
) -> Dict[str, Tuple[str, List[Dict[str, Any]]]]:
    """Retrieve multiple external-evidence lanes with bounded parallelism and timeout."""
    return _retrieve_mdt_evidence_lanes(
        lane_queries=lane_queries,
        source_label="external",
        worker=_retrieve_mdt_external_source,
        kwargs_for_lane=lambda _role, query: {
            "rag_query": query,
            "topk": topk,
            "unavailable_pack": unavailable_pack,
        },
        unavailable_pack=unavailable_pack,
    )


def _mdt_external_lane_parallelism() -> int:
    raw = str(os.environ.get("OMGS_MDT_EXTERNAL_EVIDENCE_LANE_PARALLELISM") or "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        return 1
    return max(1, value)


def _mdt_evidence_source_timeout_seconds() -> float:
    raw = str(os.environ.get("OMGS_MDT_EVIDENCE_SOURCE_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return _DEFAULT_MDT_EVIDENCE_SOURCE_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_MDT_EVIDENCE_SOURCE_TIMEOUT_SECONDS
    return max(value, 1.0)


def _source_status_line(name: str, raw: List[Dict[str, Any]] | None, *, status: str) -> str:
    return f"[OMGS_EVIDENCE_SOURCE_READY] source={name} n={len(raw or [])} status={status}"


def _emit_source_status_line(name: str, raw: List[Dict[str, Any]] | None, *, status: str) -> None:
    """
    Emit live-room control markers on the real process stdout.

    Some engine/tool calls temporarily replace ``sys.stdout`` while worker
    threads are still running. The parent live stream only sees the subprocess'
    real stdout, so source-ready markers must bypass that temporary capture.
    """
    target = getattr(sys, "__stdout__", None) or sys.stdout
    target.write(_source_status_line(name, raw, status=status) + "\n")
    target.flush()


def emit_mdt_evidence_source_ready(
    name: str,
    raw: List[Dict[str, Any]] | None,
    *,
    status: str = "completed",
) -> None:
    """Emit the same source-ready control marker used by live MDT retrieval."""
    _emit_source_status_line(name, raw, status=status)


def emit_rag_digest_ready(digest: str | None = None) -> None:
    """
    Emit the live-room digest completion marker on the real process stdout.

    The UI only needs the leading ``rag_digest`` control line to complete the
    shared prep task; keep a short preview for operator logs without streaming
    the whole evidence digest as an stdout control payload.
    """
    preview = re.sub(r"\s+", " ", str(digest or "")).strip()
    if len(preview) > 240:
        preview = preview[:240].rstrip() + "..."
    target = getattr(sys, "__stdout__", None) or sys.stdout
    target.write(f"rag_digest {preview}\n" if preview else "rag_digest\n")
    target.flush()


def _run_mdt_evidence_source_worker(
    *,
    name: str,
    worker,
    kwargs: Dict[str, Any],
    output_queue,
) -> None:
    try:
        pack, raw = worker(**kwargs)
        output_queue.put((name, "completed", pack, raw or [], None))
    except Exception as exc:
        output_queue.put((name, "failed", "", [], exc))


def retrieve_mdt_evidence_sources(
    *,
    case_question: Any,
    rag_query: str,
    device: str = "auto",
    guideline_topk: int = 5,
    nccn_topk: int = 3,
    external_topk: int = 8,
    nccn_unavailable_pack: str = "(NCCN: unavailable)",
    guideline_unavailable_pack: str = "(Guideline: unavailable)",
    external_unavailable_pack: str = "(PUBMED: unavailable)",
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Tuple[str, List[Dict[str, Any]]]]]:
    """
    Retrieve MDT authority and external evidence concurrently.

    Returns:
        Tuple of (merged_pack, merged_raw, source_results), where source_results has
        nccn/guideline/external keys and each value is (pack, raw).
    """
    tasks = {
        "nccn": (
            _retrieve_mdt_nccn_source,
            {
                "case_question": case_question,
                "rag_query": rag_query,
                "device": device,
                "topk": nccn_topk,
                "unavailable_pack": nccn_unavailable_pack,
            },
            nccn_unavailable_pack,
        ),
        "guideline": (
            _retrieve_mdt_guideline_source,
            {
                "rag_query": rag_query,
                "device": device,
                "topk": guideline_topk,
                "unavailable_pack": guideline_unavailable_pack,
            },
            guideline_unavailable_pack,
        ),
        "external": (
            _retrieve_mdt_external_source,
            {
                "rag_query": rag_query,
                "topk": external_topk,
                "unavailable_pack": external_unavailable_pack,
            },
            external_unavailable_pack,
        ),
    }

    timeout_seconds = _mdt_evidence_source_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds
    output_queue: queue.Queue = queue.Queue()
    results: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    pending = set(tasks)
    for name, (worker, kwargs, _unavailable_pack) in tasks.items():
        thread = threading.Thread(
            target=_run_mdt_evidence_source_worker,
            kwargs={
                "name": name,
                "worker": worker,
                "kwargs": kwargs,
                "output_queue": output_queue,
            },
            name=f"mdt-evidence-{name}",
            daemon=True,
        )
        thread.start()

    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            name, status, pack, raw, exc = output_queue.get(timeout=remaining)
        except queue.Empty:
            break
        if name not in pending:
            continue
        pending.remove(name)
        if status == "failed":
            unavailable_pack = tasks[name][2]
            print(f"[WARNING] {name} evidence retrieval failed: {exc}", flush=True)
            results[name] = (unavailable_pack, [])
            _emit_source_status_line(name, [], status="failed")
            continue
        results[name] = (pack, raw or [])
        _emit_source_status_line(name, raw, status="completed")

    for name in tasks:
        if name not in pending:
            continue
        unavailable_pack = tasks[name][2]
        print(
            f"[WARNING] {name} evidence retrieval timed out after {timeout_seconds:.1f}s; continuing with fallback.",
            flush=True,
        )
        results[name] = (unavailable_pack, [])
        _emit_source_status_line(name, [], status="timeout")

    nccn_pack, nccn_raw = results["nccn"]
    guideline_pack, guideline_raw = results["guideline"]
    external_pack, external_raw = results["external"]
    rag_pack = (
        f"# NCCN CONSTRAINTS\n{nccn_pack}\n\n"
        f"# GUIDELINE EVIDENCE\n{guideline_pack}\n\n"
        f"# LITERATURE\n{external_pack}"
    )
    rag_raw = (nccn_raw or []) + (guideline_raw or []) + (external_raw or [])
    return rag_pack, rag_raw, results


# =============================================================================
# NCCN PUBLIC ENTRY
# -----------------------------------------------------------------------------
# MDT 侧 NCCN 检索的**唯一**对外入口。两级策略：
#   1. 先跑 get_nccn_structural_match（见 NCCN STRUCTURAL MATCHING），
#      基于本地 JSON 做 feature → rule 硬匹配。这是 algorithm-centric guideline
#      的首选路径（命中时引用 id 形如 PLATINUM_RESISTANT_RECURRENCE）。
#   2. structural 空匹配时，fallback 到 in-process omgs_engine.nccn tool（见
#      NCCN ENGINE FALLBACK ADAPTER），从 NCCN 图谱取主路径 + supporting
#      refs，命中时引用 id 形如 OV-7（带连字符的 page code）。
#
# 无论走哪条分支，返回值都是 (nccn_pack, nccn_raw)，raw 的 source 为
# "nccn_matcher_rule" 或 "nccn_decision_node"，下游的引用 tag 由
# _get_rag_result_tag 统一生成。
# =============================================================================
def get_nccn_rag(question, device="auto", topk=None):
    """
    Load NCCN JSON rules for safety gating and conflict resolution.

    NCCN is used as atomic rules (constraints), NOT as generative evidence.
    This function implements a hybrid approach:
    1. First attempts STRUCTURAL MATCHING based on case features (platinum_status, stage, etc.)
    2. Falls back to omgs_engine NCCN graph retrieval if structural match fails or yields no results

    Args:
        question: Either a RAG query string OR structured case dict with features
        device: Kept for backward compatibility; ignored in engine fallback path
        topk: Kept for backward compatibility; ignored in engine fallback path

    Returns:
        Tuple of (nccn_pack, nccn_raw) - formatted evidence string and raw results
    """
    matcher_topk = _nccn_matcher_topk(topk)

    # Try structural matching first - pass question as case data
    structural_pack, structural_raw = get_nccn_structural_match(question, topk=matcher_topk)

    # If structural matching succeeded with results, use it
    if structural_raw and len(structural_raw) > 0:
        return structural_pack, structural_raw

    # Fallback 分支：engine 的 nccn tool 接受一条英文 query，不读 device/topk。
    # 这两个参数仅为保住历史签名，engine 路径下显式丢弃以避免误导下一位读者。
    del device

    # 上游可能传 case dict（来自 case_parser）或已经成形的 query string。
    # 走 engine 前先把 dict 形态压成一条 NCCN 专用的 query。
    if isinstance(question, dict):
        query = _build_nccn_query_from_case(question)
    else:
        query = question

    try:
        from engine_bridge import build_tool_input
        from engine_bridge import default_tool_call_id
        from utils.runtime_config import tool_input_overrides_from_env

        engine = _nccn_engine()
        result = engine.invoke_tool(
            tool_name="nccn",
            tool_call_id=default_tool_call_id("nccn"),
            tool_input=build_tool_input(
                query=query,
                **tool_input_overrides_from_env("nccn"),
            ),
            consumer="mdt",
        )
        bundle = _bundle_from_tool_result(result)
        return _build_nccn_pack_from_bundle(bundle)
    except Exception as e:
        # 任何 engine 端异常（import 失败 / Neo4j 连接 / 超时 / bundle 字段缺失）
        # 都降级成 unavailable，不向上冒泡，避免 deliberation 崩。
        print(f"[WARNING] NCCN engine fallback unavailable: {e}")
        return "(NCCN: engine fallback unavailable)", []


# =============================================================================
# NCCN STRUCTURAL MATCHING
# -----------------------------------------------------------------------------
# NCCN 的**首选**命中路径：加载本地 nccn_ovarian_rules.json，把 case features
# （platinum / stage / histology / brca / hrd / msi 等）硬匹配到 decision_nodes
# 和 safety_rules，返回 (nccn_pack, nccn_raw)。
#
# 组成：
#   - get_nccn_structural_match: 对外入口，被 get_nccn_rag 首先调用
#   - _extract_case_features:    把各种命名的 case dict 字段归一化为标准 key
#   - _match_case_to_nccn:       依次跑 safety rules / 5 条 decision 策略
#
# 注意 structural match 完全不走 omgs_engine；两条路径的 node_id 格式也不同
# （structural = PLATINUM_RESISTANT_*，engine = OV-7 这类 page code），
# 下游 reference_cache 需要同时接受两种。
# =============================================================================
def get_nccn_structural_match(case_data, topk: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """
    STRUCTURAL MATCHING for NCCN rules - matches case features directly to decision nodes.

    This is the primary matching method for NCCN (per paper: "algorithm-centric guidelines
    are not used as free-text retrieval sources; instead, recommendations are decomposed
    into atomic, machine-readable rules").

    Args:
        case_data: Either a dict with case features OR a RAG query string (for backward compatibility)
                   Expected keys: platinum_status, stage, histology, line_of_therapy, brca_status, hrd_status

    Returns:
        Tuple of (nccn_pack, nccn_raw) - formatted NCCN constraints and raw matched rules
    """
    import os
    import json

    # Try to load the compact NCCN matcher rules. The rule file in files/ is a
    # reference artifact and is not the default structural-match source.
    nccn_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "files",
        "nccn_ovarian_matcher_rules.manual.json",
    )

    try:
        with open(nccn_path, 'r', encoding='utf-8') as f:
            nccn_data = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load NCCN JSON for structural matching: {e}")
        return "(NCCN: structural matching unavailable)", []

    # Extract case features if case_data is a dict, otherwise return empty
    if isinstance(case_data, dict):
        features = _extract_case_features(case_data)
    else:
        # If it's a string (RAG query), can't do structural matching
        return "(NCCN: requires structured case data for matching)", []

    # Perform structural matching
    matched_rules = _match_case_to_nccn_matcher_rules(case_data, features, nccn_data)
    matched_rules = matched_rules[:_nccn_matcher_topk(topk)]

    if not matched_rules:
        return "(NCCN: no structural match found)", []

    # Format output
    pack_lines = ["# NCCN CONSTRAINTS (Matcher Rules)"]
    raw_results = []

    for i, rule in enumerate(matched_rules, 1):
        rule_id = rule.get("id") or f"MATCHER_{i}"
        pages = rule.get("pages") or ["nccn"]
        text = rule.get("return", "")
        pack_lines.append(f"[M{i}] {text} [@guideline:nccn | {rule_id}]")
        raw_results.append({
            "rank": i,
            "score": 1.0,
            "source": "nccn_matcher_rule",
            "rule_id": rule_id,
            "unique_id": rule.get("unique_id"),
            "node_id": rule_id,
            "type": "matcher_rule",
            "pages": pages,
            "keywords": rule.get("keywords") or [],
            "text": text,
        })

    pack = "\n".join(pack_lines)
    return pack, raw_results


def _extract_case_features(case: dict) -> dict:
    """
    Extract case features for NCCN structural matching.

    Args:
        case: Structured case dict with various field names (case_parser output)

    Returns:
        Normalized features dict with standardized keys
    """
    # Map various field names to standardized keys
    feature_map = {
        'platinum_status': ['platinum_status', 'PLATINUM_STATUS', 'platinumStatus'],
        'platinum_status_current': ['platinum_status_current', 'PLATINUM_STATUS_CURRENT', 'platinumStatusCurrent'],
        'stage': ['stage', 'STAGE', 'Stage'],
        'stage_clinical': ['stage_clinical', 'STAGE_CLINICAL'],
        'histology': ['histology', 'HISTOLOGY', 'Histology'],
        'brca_status': ['brca_status', 'BRCA_STATUS', 'brca', 'BRCA'],
        'hrd_status': ['hrd_status', 'HRD_STATUS', 'hrd', 'HRD'],
        'line_of_therapy': ['line_of_therapy', 'LINE_OF_THERAPY', 'line', 'Line'],
        'response': ['response', 'RESPONSE', 'Response'],
        'msi_status': ['msi_status', 'MSI_STATUS', 'msi', 'MSI'],
        'tmb_status': ['tmb_status', 'TMB_STATUS', 'tmb', 'TMB'],
        'pd_l1': ['pd_l1', 'PD_L1', 'pdl1', 'PDL1'],
        'fra_status': ['fra_status', 'FRA_STATUS', 'fra', 'FRα', 'fra'],
        'bevacizumab_prior': ['bevacizumab_prior', 'BEVACIZUMAB_PRIOR', 'prior_bevacizumab'],
        'is_recurrent': ['is_recurrent', 'IS_RECURRENT', 'recurrent', 'Recurrent'],
        'pfi_days': ['pfi_days', 'PFI_DAYS', 'PFI'],
    }

    features = {}

    # Also look inside CASE_CORE for nested structure (common in EHR extraction)
    case_core = case.get('CASE_CORE') or {}

    for std_key, alt_keys in feature_map.items():
        found = False
        # First, try top-level keys
        for alt_key in alt_keys:
            if alt_key in case:
                val = case[alt_key]
                if val is not None and str(val).lower() not in ['unknown', 'not specified', 'none', '']:
                    features[std_key] = str(val).strip()
                    found = True
                    break
        # If not found, try nested inside CASE_CORE
        if not found:
            for alt_key in alt_keys:
                if alt_key in case_core:
                    val = case_core[alt_key]
                    if val is not None and str(val).lower() not in ['unknown', 'not specified', 'none', '']:
                        features[std_key] = str(val).strip()
                        break

    return features


def _iter_text_values(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_text_values(nested)
        return
    if isinstance(value, list):
        for nested in value:
            yield from _iter_text_values(nested)
        return
    yield str(value)


def _case_match_text(case: dict, features: dict) -> str:
    """Build a current-state matcher text without letting prior history dominate.

    The manual matcher rules use AND semantics over short keyword phrases. To
    avoid false positives from old treatment lines (for example a current
    platinum-resistant case whose PLATINUM_HISTORY includes earlier
    platinum-sensitive relapses), this text intentionally favors current status,
    diagnosis, biomarkers, comorbidities, preferences, and key prior exposure
    summaries rather than the full recursive EHR payload.
    """
    parts = list(_iter_text_values(features))
    case_core = case.get("CASE_CORE") if isinstance(case, dict) else {}
    case_core = case_core if isinstance(case_core, dict) else {}

    current_paths = [
        case_core.get("CURRENT_STATUS"),
        case_core.get("PLATINUM_STATUS_CURRENT"),
        case_core.get("PLATINUM_PFI_CURRENT"),
        case_core.get("PLATINUM_STATUS"),
        case_core.get("PFI_days"),
        case_core.get("ECOG"),
        case_core.get("DIAGNOSIS"),
        case_core.get("STAGE"),
        case_core.get("RELAPSE_DATE"),
        case_core.get("VISIT_CONTEXT"),
        case_core.get("BIOMARKERS"),
        case_core.get("GENOMICS"),
        case_core.get("MEDICAL_HISTORY"),
        case_core.get("MAINTENANCE"),
        case_core.get("MAINTENANCE_DETAIL"),
    ]
    for item in current_paths:
        parts.extend(_iter_text_values(item))

    biomarkers = case_core.get("BIOMARKERS")
    if isinstance(biomarkers, dict):
        for marker, value in biomarkers.items():
            parts.append(f"{marker} {value}")
    for marker in ("HRD", "BRCA1", "BRCA2"):
        value = case_core.get(marker)
        if value:
            parts.append(f"{marker} {value}")

    med_onc = case.get("MED_ONC") if isinstance(case, dict) else {}
    if isinstance(med_onc, dict):
        parts.extend(_iter_text_values(med_onc.get("current_regimen")))
        parts.extend(_iter_text_values(med_onc.get("planned_next_regimen")))
        parts.extend(_iter_text_values(med_onc.get("prior_systemic_therapies")))
        parts.extend(_iter_text_values(med_onc.get("genetic_testing")))
        parts.extend(_iter_text_values(med_onc.get("TOXICITIES")))
        parts.extend(_iter_text_values(med_onc.get("CLINICAL_TRIALS")))

    radiology = case.get("RADIOLOGY") if isinstance(case, dict) else {}
    if isinstance(radiology, dict):
        parts.extend(_iter_text_values(radiology.get("studies")))

    pathology = case.get("PATHOLOGY") if isinstance(case, dict) else {}
    if isinstance(pathology, dict):
        parts.extend(_iter_text_values(pathology.get("specimens")))

    lab_trends = case.get("LAB_TRENDS") if isinstance(case, dict) else {}
    if isinstance(lab_trends, dict):
        parts.extend(_iter_text_values(lab_trends.get("labs")))
        parts.extend(_iter_text_values(lab_trends.get("milestones")))

    text = " ".join(parts).lower()

    derived: list[str] = []
    platinum = str(features.get("platinum_status_current") or features.get("platinum_status") or "").lower()
    if "resistant" in platinum:
        derived.extend(["platinum-resistant", "platinum-resistant disease", "recurrence"])
    if "sensitive" in platinum:
        derived.extend(["platinum-sensitive disease", "recurrence"])
    if "refractory" in platinum:
        derived.extend(["platinum-refractory disease", "recurrence"])
    if "fralpha" in text and any(term in text for term in ["positive", "eligible", "high"]):
        derived.append("fralpha positive")
    if any(term in text for term in ["brca", "hrd", "fralpha", "folr1", "her2", "parp", "biomarker"]):
        derived.append("biomarker")
    if "brca" in text and any(term in text for term in ["testing", "germline", "somatic", "wildtype", "mutation"]):
        derived.append("brca testing")
    if any(term in text for term in ["heterogeneous", "heterogeneity"]) and any(
        marker in text for marker in ["fralpha", "folr1", "her2", "pd-l1", "pdl1", "biomarker"]
    ):
        derived.append("biomarker heterogeneity")
    if "recurrent" in text and any(term in text for term in ["biopsy", "implant", "specimen"]):
        derived.append("recurrent biopsy")
    if ("fralpha" in text or "folr1" in text) and ("adc" in text or "mirvetuximab" in text):
        derived.append("fralpha adc")
    if any(term in text for term in ["ophthalmology", "dry eye", "keratitis", "vision loss"]):
        derived.append("ophthalmology follow-up")
    if "bevacizumab" in text:
        derived.append("prior bevacizumab")
    if any(term in text for term in ["olaparib", "niraparib", "rucaparib", "parp"]):
        derived.append("prior parp inhibitor")
    if "clinical trial" in text or "trial" in text:
        derived.append("clinical trial")
    if "no visceral crisis" in text or "without visceral crisis" in text:
        derived.append("no visceral crisis")
    if ("ca125" in text or "ca-125" in text or "he4" in text) and any(
        term in text for term in ["rising", "rapid", "rise", "increased"]
    ):
        derived.append("rising ca-125")
    if any(term in text for term in ["low-volume", "low volume", "low-moderate", "low–moderate"]):
        derived.append("low-volume disease")

    return f"{text} {' '.join(derived)}".lower()


def _match_case_to_nccn_matcher_rules(case: dict, features: dict, nccn_data: dict) -> List[dict]:
    rules = nccn_data.get("rules") or []
    if not isinstance(rules, list):
        return []

    text = _case_match_text(case, features)
    matched: List[dict] = []
    seen: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = [
            str(keyword or "").strip().lower()
            for keyword in (rule.get("keywords") or [])
            if str(keyword or "").strip()
        ]
        if not keywords:
            continue
        if all(_matcher_keyword_present(text, keyword) for keyword in keywords):
            unique_id = str(rule.get("unique_id") or rule.get("id") or "")
            if unique_id in seen:
                continue
            seen.add(unique_id)
            matched.append(rule)
    return matched


def _matcher_keyword_present(text: str, keyword: str) -> bool:
    """Match a keyword phrase without allowing stage I to match stage II-IV."""
    pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _match_case_to_nccn(features: dict, nccn_data: dict) -> Tuple[List[dict], List[dict]]:
    """
    Match case features to NCCN decision nodes and safety rules.

    Args:
        features: Normalized case features
        nccn_data: Loaded NCCN JSON rules

    Returns:
        Tuple of (matched_nodes, safety_issues)
    """
    matched_nodes = []
    safety_issues = []

    # Get decision nodes and safety rules
    decision_nodes = nccn_data.get('decision_nodes', [])
    safety_rules = nccn_data.get('safety_rules', [])

    platinum = features.get('platinum_status', '').lower()
    stage = features.get('stage', '').upper()
    histology = features.get('histology', '').lower()
    brca = features.get('brca_status', '').lower()
    hrd = features.get('hrd_status', '').lower()
    is_recurrent = features.get('is_recurrent', '').lower() in ['true', 'yes', '1', 'recurrent']

    # ===== MATCH SAFETY RULES =====
    for rule in safety_rules:
        trigger = rule.get('trigger_expression', '').lower()

        # Check platinum-related safety rules
        if 'platinum_status' in trigger:
            if 'resistant' in platinum or 'refractory' in platinum:
                if 'resistant' in trigger or 'refractory' in trigger:
                    safety_issues.append(rule)
                    continue

        # Check PARP inhibitor safety rules
        if 'parp inhibitor' in trigger:
            if any(p in trigger for p in ['olaparib', 'niraparib', 'rucaparib']):
                # Block for clear cell histology
                if 'clear cell' in histology:
                    if rule.get('action') == 'BLOCK':
                        safety_issues.append(rule)
                        continue

        # Check FRα testing rule
        if 'mirvetuximab' in trigger:
            # This is a verification rule, add if relevant biomarker
            fra = features.get('fra_status', '').lower()
            if fra:
                safety_issues.append(rule)

        # Check immunotherapy rules
        if any(imm in trigger for imm in ['pembrolizumab', 'dostarlimab', 'nivolumab']):
            msi = features.get('msi_status', '').lower()
            if msi in ['msi-h', 'dmmr', 'msi']:
                safety_issues.append(rule)

    # ===== MATCH DECISION NODES =====
    for node in decision_nodes:
        node_id = node.get('id', '')
        node_keywords = [k.lower() for k in node.get('keywords', [])]
        eligibility = node.get('eligibility', '').lower()

        matched = False
        match_reason = ""

        # Strategy 1: Platinum status + recurrence pattern matching
        if 'PLATINUM_SENSITIVE' in node_id and platinum in ['sensitive', 'partial response']:
            if is_recurrent or 'PFI' in eligibility:
                matched = True
                match_reason = "platinum-sensitive recurrence"
        elif 'PLATINUM_RESISTANT' in node_id and platinum in ['resistant', 'progressive disease']:
            matched = True
            match_reason = "platinum-resistant recurrence"
        elif 'PLATINUM_REFRACTORY' in node_id and platinum == 'refractory':
            matched = True
            match_reason = "platinum-refractory disease"

        # Strategy 2: Stage-based matching
        if not matched:
            if 'INITIAL_TREATMENT_STAGE_I' in node_id and stage in ['I', 'IA', 'IB', '1']:
                matched = True
                match_reason = "stage I initial treatment"
            elif 'INITIAL_TREATMENT_STAGE_II' in node_id and stage in ['II', '2']:
                matched = True
                match_reason = "stage II initial treatment"
            elif 'INITIAL_TREATMENT_STAGE_III_IV' in node_id and stage in ['III', 'IV', '3', '4']:
                matched = True
                match_reason = "stage III/IV initial treatment"

        # Strategy 3: Molecular biomarker matching
        if not matched:
            if 'MOLECULAR_BIOMARKER' in node_id:
                if brca or hrd or features.get('msi_status'):
                    matched = True
                    match_reason = "biomarker-driven treatment"

        # Strategy 4: Maintenance therapy matching
        if not matched:
            if 'MAINTENANCE' in node_id:
                if brca or hrd:
                    matched = True
                    match_reason = "maintenance therapy eligibility"

        # Strategy 5: Immunotherapy matching
        if not matched:
            if 'IMMUNOTHERAPY' in node_id:
                msi = features.get('msi_status', '').lower()
                if msi in ['msi-h', 'dmmr']:
                    matched = True
                    match_reason = "immunotherapy indication"

        if matched:
            matched_nodes.append(node)

    return matched_nodes, safety_issues


# =============================================================================
# NCCN QUERY BUILDER
# -----------------------------------------------------------------------------
# 仅服务 NCCN engine fallback（get_nccn_rag 内部）。把 case dict 里的
# platinum_status / stage / histology / line / brca / hrd 等结构化字段拼成一条
# 紧凑的英文 query 给 engine。与 build_rag_query_for_mdt（LLM 生成）不同，
# 这里是纯规则拼接，无 LLM 调用，失败模式几乎为零。
# =============================================================================
def _build_nccn_query_from_case(case: dict) -> str:
    """
    Build a concise RAG query from structured case data for NCCN RAG fallback.

    Args:
        case: Structured case dict with features like platinum_status, stage, histology, etc.

    Returns:
        Query string for RAG embedding search
    """
    parts = ["ovarian cancer"]

    # Extract key features
    if isinstance(case, dict):
        platinum = case.get("platinum_status") or case.get("PLATINUM_STATUS") or ""
        stage = case.get("stage") or case.get("STAGE") or ""
        histology = case.get("histology") or case.get("HISTOLOGY") or ""
        line = case.get("line_of_therapy") or case.get("LINE_OF_THERAPY") or ""
        brca = case.get("brca_status") or case.get("BRCA_STATUS") or ""
        hrd = case.get("hrd_status") or case.get("HRD_STATUS") or ""

        if platinum:
            parts.append(f"platinum {platinum.lower()}")
        if stage:
            parts.append(f"stage {stage}")
        if histology:
            # Clean histology for query
            hist_clean = histology.lower()
            for term in ["carcinoma", "cancer"]:
                hist_clean = hist_clean.replace(term, "").strip()
            if hist_clean:
                parts.append(hist_clean)
        if line:
            parts.append(f"line {line.lower()}")
        if brca:
            parts.append(f"BRCA {brca.lower()}")
        if hrd:
            parts.append(f"HRD {hrd.lower()}")

    return " ".join(parts)


# =============================================================================
# NCCN ENGINE FALLBACK ADAPTER
# -----------------------------------------------------------------------------
# 当 NCCN structural match 未命中时，由 get_nccn_rag（见 NCCN PUBLIC ENTRY）
# 走 in-process omgs_engine 的 nccn tool；这里是把 engine 返回的 EvidenceBundle
# 适配回当前 (nccn_pack, nccn_raw) 契约的所有内部函数。
#
# 数据源: engine_bridge.EngineIntegration -> omgs_engine.dispatcher.EngineDispatcher
# 契约:   engine.invoke_tool(tool_name="nccn", tool_input={"query": str})
# 输出:   (pack_string, [ {rank, score, source="nccn_decision_node", node_id,
#                           node_name, type, text, metadata}, ])  长度恒为 1。
# 主锚点优先级: guideline_path_pointer > guideline_page_pointer > candidate_page_pointer
# =============================================================================
def _nccn_engine() -> Any:
    """Lazily construct the product-level engine seam."""
    global _NCCN_ENGINE
    if _NCCN_ENGINE is None:
        from engine_bridge import EngineIntegration

        _NCCN_ENGINE = EngineIntegration()
    return _NCCN_ENGINE


def _bundle_from_tool_result(result: Dict[str, Any]) -> Any:
    tool_output = dict(result.get("tool_output") or {})
    bundle = tool_output.get("bundle")
    if bundle is None:
        raise RuntimeError("Tool result did not contain an EvidenceBundle.")
    return bundle


def _pick_nccn_primary_record(records: List[Any]) -> Optional[Any]:
    priority = {
        "guideline_path_pointer": 0,
        "guideline_page_pointer": 1,
        "candidate_page_pointer": 2,
    }
    candidates = [
        record for record in records
        if getattr(record, "evidence_type", "") in priority
    ]
    if not candidates:
        return records[0] if records else None
    return min(candidates, key=lambda record: priority.get(getattr(record, "evidence_type", ""), 99))


def _supporting_nccn_records(records: List[Any], primary: Any) -> List[Any]:
    primary_id = getattr(getattr(primary, "provenance", None), "source_id", None)
    supporting = []
    for record in records:
        if record is primary:
            continue
        record_type = getattr(record, "evidence_type", "")
        if record_type in {
            "guideline_footnote",
            "guideline_reference_page",
            "guideline_reference_family",
            "guideline_footnote_reference_family",
        }:
            supporting.append(record)
            continue
        # Be conservative: if another primary-like record points somewhere else, treat as extra support.
        if primary_id and getattr(getattr(record, "provenance", None), "source_id", None) != primary_id:
            supporting.append(record)
    return supporting


def _nccn_path_preview(bundle: Any) -> Optional[Dict[str, Any]]:
    debug = getattr(bundle, "debug", None) or {}
    if not isinstance(debug, dict):
        return None
    normalized = debug.get("normalized_trace") or {}
    if not isinstance(normalized, dict):
        return None
    preview = normalized.get("path_preview")
    return preview if isinstance(preview, dict) else None


def _format_nccn_path_lines(path_preview: Optional[Dict[str, Any]]) -> List[str]:
    if not path_preview:
        return []

    nodes = path_preview.get("nodes")
    edges = path_preview.get("edges")
    if not isinstance(nodes, list) or not nodes:
        return []
    if not isinstance(edges, list):
        edges = []

    lines: List[str] = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "").strip()
        node_label = str(node.get("node_label") or "").strip()
        page_code = str(node.get("page_code") or "").strip()
        text = _humanize_engine_text(node.get("text"))
        text_lines = text.splitlines() if text else [""]
        first_line = text_lines[0].strip() if text_lines else ""

        prefix_parts = [part for part in [node_id, node_label] if part]
        prefix = " | ".join(prefix_parts)
        if page_code:
            prefix = f"{prefix} [{page_code}]" if prefix else f"[{page_code}]"

        if first_line:
            lines.append(f"{prefix} | {first_line}" if prefix else first_line)
        elif prefix:
            lines.append(prefix)

        for extra_line in text_lines[1:]:
            extra = extra_line.rstrip()
            if extra:
                lines.append(extra)

        if idx < len(edges):
            edge = edges[idx]
            if isinstance(edge, dict):
                edge_type = str(edge.get("type") or "").strip()
                if edge_type:
                    lines.append(f"-[{edge_type}]->")
    return lines


def _build_nccn_pack_from_bundle(bundle: Any) -> Tuple[str, List[Dict[str, Any]]]:
    records = list(getattr(bundle, "records", ()) or ())
    if not records:
        return "(NCCN: no engine fallback match)", []

    primary = _pick_nccn_primary_record(records)
    if primary is None:
        return "(NCCN: no engine fallback match)", []

    title = _humanize_engine_text(getattr(primary, "title", "") or "NCCN constraint")
    provenance = getattr(primary, "provenance", None)
    node_id = str(
        getattr(provenance, "source_id", "")
        or getattr(provenance, "label", "")
        or title
    ).strip()
    if not node_id:
        return "(NCCN: no engine fallback match)", []

    path_preview = _nccn_path_preview(bundle)
    path_lines = _format_nccn_path_lines(path_preview)
    supporting_records = _supporting_nccn_records(records, primary)

    pack_lines = [f"{title} [@guideline:nccn | {node_id}]"]
    if path_lines:
        pack_lines.append("    Path:")
        for line in path_lines:
            pack_lines.append(f"      {line}")

    summary = _humanize_engine_text(getattr(primary, "summary", ""))
    if summary:
        pack_lines.append("    Summary:")
        for line in summary.splitlines():
            pack_lines.append(f"      {line}")

    for record in supporting_records:
        record_type = getattr(record, "evidence_type", "")
        record_title = _humanize_engine_text(getattr(record, "title", ""))
        record_summary = _humanize_engine_text(getattr(record, "summary", ""))
        citation = _humanize_engine_text(getattr(record, "citation", ""))
        highlights = [_humanize_engine_text(item) for item in getattr(record, "highlights", ()) if _humanize_engine_text(item)]

        if record_type == "guideline_footnote":
            label = citation or record_title or "footnote"
            pack_lines.append("    Footnote:")
            pack_lines.append(f"      {label}: {record_summary}".rstrip())
            continue

        section_label = "Supporting"
        if record_type == "guideline_reference_page":
            section_label = "Supporting page"
        elif record_type in {"guideline_reference_family", "guideline_footnote_reference_family"}:
            section_label = "Supporting family"

        pack_lines.append(f"    {section_label}:")
        if record_title and record_summary:
            pack_lines.append(f"      {record_title}: {record_summary}")
        elif record_title:
            pack_lines.append(f"      {record_title}")
        elif record_summary:
            pack_lines.append(f"      {record_summary}")
        for highlight in highlights:
            pack_lines.append(f"      - {highlight}")

    text_parts: List[str] = []
    if path_lines:
        text_parts.append(f"Path: {' '.join(path_lines)}")
    if summary:
        text_parts.append(f"Summary: {_flatten_text(summary)}")
    for record in supporting_records:
        record_type = getattr(record, "evidence_type", "")
        record_title = _humanize_engine_text(getattr(record, "title", ""))
        record_summary = _humanize_engine_text(getattr(record, "summary", ""))
        citation = _humanize_engine_text(getattr(record, "citation", ""))
        highlights = [_flatten_text(item) for item in getattr(record, "highlights", ()) if _flatten_text(item)]

        if record_type == "guideline_footnote":
            label = citation or record_title or "footnote"
            text_parts.append(f"Footnote {label}: {_flatten_text(record_summary)}")
            continue

        if record_title and record_summary:
            text_parts.append(f"{record_title}: {_flatten_text(record_summary)}")
        elif record_summary:
            text_parts.append(_flatten_text(record_summary))
        elif record_title:
            text_parts.append(record_title)
        text_parts.extend(highlights)

    metadata = {
        "engine_evidence_type": getattr(primary, "evidence_type", ""),
        "path_node_ids": [
            str(node.get("node_id") or "").strip()
            for node in (path_preview or {}).get("nodes", [])
            if isinstance(node, dict) and str(node.get("node_id") or "").strip()
        ],
        "path_edge_types": [
            str(edge.get("type") or "").strip()
            for edge in (path_preview or {}).get("edges", [])
            if isinstance(edge, dict) and str(edge.get("type") or "").strip()
        ],
        "supporting_record_types": [
            getattr(record, "evidence_type", "")
            for record in supporting_records
        ],
        "supporting_source_ids": [
            str(getattr(getattr(record, "provenance", None), "source_id", "")).strip()
            for record in supporting_records
            if str(getattr(getattr(record, "provenance", None), "source_id", "")).strip()
        ],
    }

    raw = [{
        "rank": 1,
        "score": 1.0,
        "source": "nccn_decision_node",
        "node_id": node_id,
        "node_name": title,
        "type": "treatment_pathway",
        "text": " ".join(part for part in text_parts if part).strip(),
        "metadata": metadata,
    }]

    return "\n".join(pack_lines), raw


# =============================================================================
# GUIDELINE RAG ENTRY POINTS
# -----------------------------------------------------------------------------
# Guideline retrieval entry points. The MDT pipeline uses
# get_global_guideline_rag; get_guideline_rag keeps a role argument for
# compatibility with role-scoped calls. Both delegate to engine-backed
# rag_search_pack.
#
# The NCCN entry point is get_nccn_rag in the NCCN section.
# =============================================================================
def get_guideline_rag(role, question, device="auto", topk=5):
    """
    Load guideline evidence for a given role via the current engine guidelines tool.

    Args:
        role: MDT role name (chair, oncologist, radiologist, pathologist, nuclear)
        question: Query string for RAG search
        device: Deprecated compatibility parameter; ignored.
        topk: Number of top results to return
    
    Returns:
        Formatted RAG evidence string for the role
    """
    del device

    rag_pack, _ = rag_search_pack(
        query=question,
        topk=topk,
        guideline_scope=_guideline_scope_for_role(role),
    )
    return f"# GUIDELINE RAG for {role}\n{rag_pack}\n"


def get_global_guideline_rag(question, device="auto", topk=5):
    """
    Load global guideline RAG (always uses default_role from config, typically chair).

    This is the main entry point used by the MDT pipeline for global guideline retrieval.
    Uses the current Evidence mode guidelines extraction path through EngineIntegration.

    Args:
        question: Query string for RAG search
        device: Deprecated compatibility parameter; ignored.
        topk: Number of top results to return

    Returns:
        Tuple of (rag_pack, rag_raw) - formatted evidence string and raw results
    """
    del device

    return rag_search_pack(
        query=question,
        topk=topk,
        guideline_scope="chair",
    )


# =============================================================================
# GUIDELINE ENGINE ADAPTER
# -----------------------------------------------------------------------------
# The release runtime calls omgs_engine.guidelines through EngineIntegration. The
# standalone evaluation path submits caller-provided single queries, disables
# final guideline self-check and uses the Chroma dense backend through runtime
# configuration. Raw records keep source="guideline" for citation assembly.
# =============================================================================
def _guidelines_engine() -> Any:
    """Lazily construct the product-level engine seam for guidelines."""
    global _GUIDELINES_ENGINE
    if _GUIDELINES_ENGINE is None:
        from engine_bridge import EngineIntegration

        _GUIDELINES_ENGINE = EngineIntegration()
    return _GUIDELINES_ENGINE


def _guideline_scope_for_role(role: str | None) -> str:
    normalized = str(role or "").strip().lower()
    return {
        "chair": "chair",
        "oncologist": "medical_oncology",
        "medical_oncology": "medical_oncology",
        "surgeon": "surgery",
        "surgery": "surgery",
        "radiologist": "radiology",
        "radiology": "radiology",
        "pathologist": "pathology",
        "pathology": "pathology",
        "nuclear": "nuclear_medicine",
        "nuclear_medicine": "nuclear_medicine",
    }.get(normalized, "all")


def _guideline_query_analysis_input(sanitized: str, *, query_analysis_style: str) -> Dict[str, Any]:
    return {
        "source_group": "guidelines",
        "query_analysis_style": query_analysis_style,
        "main_query": sanitized,
        "main_dense_query": sanitized,
        "main_sparse_query": sanitized,
        "sub_queries": [],
    }


def _guideline_doc_id_from_record(record: Any, *, index: int) -> str:
    provenance = getattr(record, "provenance", None)
    for value in (
        getattr(provenance, "source_id", None),
        getattr(record, "citation", None),
        getattr(record, "title", None),
    ):
        normalized = str(value or "").strip()
        if normalized:
            return _short_guideline_doc_id(normalized, index=index)
    return f"guideline_{index}"


def _guideline_source_id_from_record(record: Any) -> str:
    provenance = getattr(record, "provenance", None)
    return str(getattr(provenance, "source_id", "") or "").strip()


def _short_guideline_doc_id(value: str, *, index: int) -> str:
    """Convert verbose engine chunk ids into compact, human-readable citation ids."""
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_")
    if "article_laparoscopy_score_advanced_ovarian_carcinoma_pilot_study" in safe.lower():
        return "FAGOTTI_SCORE_2006"
    if safe and len(safe) <= 40 and not re.search(r"(?:recommendation|graded|table|block|chunk)", safe, re.IGNORECASE):
        return safe

    lower = safe.lower()
    families: List[str] = []
    for token in ("nccn", "esgo", "esmo", "esp", "asco", "csco", "sgo"):
        if re.search(rf"(?:^|_){token}(?:_|$)", lower):
            families.append(token.upper())
    if not families:
        first_token = next((part for part in re.split(r"[_-]+", safe) if part and not part.isdigit()), "")
        families.append((first_token or "GUIDELINE").upper()[:12])

    if "ovarian" in lower and "OVARIAN" not in families:
        families.append("OVARIAN")

    year_match = re.search(r"(20\d{2})", safe)
    if year_match:
        families.append(year_match.group(1))

    compact = "_".join(families)
    compact = re.sub(r"_+", "_", compact).strip("_")
    return compact[:64] or f"guideline_{index}"


def _format_guideline_tag(doc_id: Any, page: Any = None, page_label: Any = None) -> str:
    """Build a page-bearing guideline citation tag without leaking missing-page sentinels."""
    safe_doc_id = str(doc_id or "").strip()
    label = str(page_label or "").strip()
    if label:
        return f"[@guideline:{safe_doc_id} | {label}]"
    if isinstance(page, int):
        return f"[@guideline:{safe_doc_id} | Page {page}]"
    page_text = str(page or "").strip()
    if page_text and page_text.lower() not in {"none", "null", "na", "n/a"}:
        return f"[@guideline:{safe_doc_id} | Page {page_text}]"
    return ""


def _guideline_page_from_record(record: Any) -> Any:
    page, _label = _guideline_page_info_from_record(record)
    return page


def _known_guideline_page_span_from_record(record: Any) -> Tuple[Any, str]:
    """Return curated bibliographic page spans for rare non-PDF guideline records."""
    provenance = getattr(record, "provenance", None)
    identity_text = " ".join(
        str(value or "").strip().lower()
        for value in (
            getattr(record, "source_key", None),
            getattr(record, "citation", None),
            getattr(record, "title", None),
            getattr(provenance, "source_id", None),
            getattr(provenance, "source_url", None),
        )
        if str(value or "").strip()
    )
    if (
        "article_laparoscopy_score_advanced_ovarian_carcinoma_pilot_study" in identity_text
        or "10.1245/aso.2006.08.021" in identity_text
    ):
        return 1156, "Pages 1156-1161"
    return None, ""


def _guideline_page_info_from_record(record: Any) -> Tuple[Any, str]:
    provenance = getattr(record, "provenance", None)
    location_text = " ".join(
        str(value or "").strip()
        for value in (
            getattr(provenance, "location_hint", None),
            getattr(record, "citation", None),
        )
        if str(value or "").strip()
    )
    span_match = re.search(r"\bpage[_\s-]*span\s*[:=#-]?\s*(\d+)\s*[-–]\s*(\d+)\b", location_text, re.IGNORECASE)
    if not span_match:
        span_match = re.search(r"\bpages?\s+(\d+)\s*[-–]\s*(\d+)\b", location_text, re.IGNORECASE)
    if span_match:
        start = int(span_match.group(1))
        end = int(span_match.group(2))
        return start, f"Pages {start}-{end}"
    number_match = re.search(r"\bpage[_\s-]*(?:number|from)?\s*[:=#-]?\s*(\d+)\b", location_text, re.IGNORECASE)
    if number_match:
        page = int(number_match.group(1))
        return page, f"Page {page}"
    generic_match = re.search(r"\b(?:page|p\.?)\s*[:#-]?\s*(\d+)\b", location_text, re.IGNORECASE)
    if generic_match:
        page = int(generic_match.group(1))
        return page, f"Page {page}"
    return _known_guideline_page_span_from_record(record)


def _is_guideline_record(record: Any) -> bool:
    source_group = str(getattr(record, "source_group", "") or "").lower()
    backend = str(getattr(record, "backend_name", "") or "").lower()
    evidence_type = str(getattr(record, "evidence_type", "") or "").lower()
    return (
        source_group == "guidelines"
        or backend in {"guidelines", "guidelines_hybrid_rag", "guidelines_dense_rag"}
        or evidence_type.startswith("guideline")
    )


def _build_guideline_pack_from_bundle(bundle: Any, *, topk: int) -> Tuple[str, List[Dict[str, Any]]]:
    records = [
        record for record in list(getattr(bundle, "records", ()) or ())
        if _is_guideline_record(record)
    ][:topk]
    if not records:
        return "(RAG: no evidence found)", []

    lines, raw = [], []
    for i, record in enumerate(records, 1):
        provenance = getattr(record, "provenance", None)
        doc_id = _guideline_doc_id_from_record(record, index=i)
        original_source_id = _guideline_source_id_from_record(record)
        page, page_label = _guideline_page_info_from_record(record)
        if not page_label:
            continue
        title = str(getattr(record, "title", "") or "").strip()
        summary = str(getattr(record, "summary", "") or "").strip()
        text = summary or title
        snippet = text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"

        page_tag = f"[{page_label.upper()}]" if page_label else ""
        citation_tag = _format_guideline_tag(doc_id, page, page_label)
        heading = title or doc_id
        lines.append(f"[{i}] {doc_id} {page_tag} {citation_tag}\n    {heading}\n    {snippet}")

        raw.append({
            "rank": i,
            "score": 0.0,
            "source": "guideline",
            "doc_id": doc_id,
            "original_doc_id": original_source_id,
            "page": page,
            "page_label": page_label,
            "text": text,
            "title": title,
            "summary": summary,
            "source_url": getattr(provenance, "source_url", None),
            "location_hint": getattr(provenance, "location_hint", None),
            "backend_name": getattr(record, "backend_name", ""),
            "evidence_type": getattr(record, "evidence_type", ""),
            "citation": getattr(record, "citation", None),
            "selection_rationale": getattr(record, "selection_rationale", None),
        })

    if not raw:
        return "(RAG: no evidence found)", []
    pack = "RAG Evidence Pack (top={}):\n".format(topk) + "\n".join(lines)
    return pack, raw


def _engine_guideline_search_pack(
    *,
    query: str,
    topk: int,
    guideline_scope: str | None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the current omgs_engine guidelines tool explicitly."""
    from engine_bridge import build_tool_input
    from engine_bridge import default_tool_call_id
    from utils.runtime_config import tool_input_overrides_from_env

    overrides = tool_input_overrides_from_env("guidelines")
    if guideline_scope is not None:
        overrides["guideline_scope"] = guideline_scope
    overrides["query_analysis_mode"] = "off"
    query_analysis_style = "single"
    overrides["query_analysis_style"] = query_analysis_style
    overrides["query_analysis_input"] = _guideline_query_analysis_input(
        query,
        query_analysis_style=query_analysis_style,
    )
    tool_input = build_tool_input(
        query=query,
        **overrides,
    )
    result = _guidelines_engine().invoke_tool(
        tool_name="guidelines",
        tool_call_id=default_tool_call_id("guidelines"),
        tool_input=tool_input,
        consumer="mdt",
        include_artifacts=False,
        include_debug=False,
        include_snapshots=True,
    )
    bundle = _bundle_from_tool_result(result)
    return _build_guideline_pack_from_bundle(bundle, topk=topk)


def rag_search_pack(
    query: str,
    index_dir: str = "",
    model_name="BAAI/bge-m3",
    device="auto",
    topk: int = 5,
    collection_name: str = "chair_chunks",
    guideline_scope: str | None = None,
):
    """
    Guideline search via the current guidelines engine tool.
    
    Returns:
        Tuple of (pack_string, raw_results_list)
        If retrieval fails, returns ("(RAG: retrieval failed: ...)", [])
    """
    del index_dir, model_name, device, collection_name
    sanitized, _ = sanitize_rag_query(query or "")
    if not sanitized:
        return "(RAG: empty query)", []
    try:
        return _engine_guideline_search_pack(
            query=sanitized,
            topk=topk,
            guideline_scope=guideline_scope,
        )
    except Exception as exc:
        return f"(RAG: retrieval failed: {exc})", []


# =============================================================================
# PUBMED EXTERNAL EVIDENCE ADAPTER
# -----------------------------------------------------------------------------
# External evidence is routed through EngineIntegration and the local
# omgs_engine frozen evidence runtime. When a resident service is enabled, it
# runs on local loopback for prebuilt local indexes/models and is not an
# external PubMed web endpoint. The returned raw records may include
# source == "pubmed", "fda" or "conference".
# =============================================================================
def _external_evidence_engine() -> Any:
    """Lazily construct the product-level engine seam for external evidence."""
    global _EXTERNAL_EVIDENCE_ENGINE
    if _EXTERNAL_EVIDENCE_ENGINE is None:
        from engine_bridge import EngineIntegration

        _EXTERNAL_EVIDENCE_ENGINE = EngineIntegration()
    return _EXTERNAL_EVIDENCE_ENGINE


def _external_record_source(record: Any) -> str:
    backend = str(getattr(record, "backend_name", "") or "").lower()
    evidence_type = str(getattr(record, "evidence_type", "") or "").lower()
    if backend == "pubmed" or evidence_type == "publication":
        return "pubmed"
    if backend in {"fda", "fda_postgres"} or evidence_type == "regulatory_label":
        return "fda"
    if backend == "conference_json" or evidence_type == "conference_abstract":
        return "conference"
    return ""


def _pubmed_pmid_from_record(record: Any) -> str:
    citation = str(getattr(record, "citation", "") or "").strip()
    if citation.upper().startswith("PMID "):
        return citation[5:].strip()
    provenance = getattr(record, "provenance", None)
    source_id = str(getattr(provenance, "source_id", "") or "").strip()
    if source_id.isdigit():
        return source_id
    return ""


def _external_record_source_id(record: Any, *, fallback_prefix: str, index: int) -> str:
    provenance = getattr(record, "provenance", None)
    source_id = str(getattr(provenance, "source_id", "") or "").strip()
    return source_id or f"{fallback_prefix}_{index}"


def _external_query_analysis_input(sanitized: str, *, query_analysis_style: str) -> Dict[str, Any]:
    return {
        "source_group": "external_evidence",
        "query_analysis_style": query_analysis_style,
        "main_query": sanitized,
        "main_dense_query": sanitized,
        "main_sparse_query": sanitized,
        "sub_queries": [],
        "fda": {
            "drug_candidates": [],
            "biomarker_candidates": [],
            "disease_context": sanitized,
            "topics": [sanitized],
        },
        "conferences": {
            "selection_focus": sanitized,
            "priority_topics": [sanitized],
            "must_match_terms": [],
        },
    }


def _build_pubmed_pack_from_bundle(bundle: Any, *, topk: int) -> Tuple[str, List[Dict[str, Any]]]:
    records_all = [
        record for record in list(getattr(bundle, "records", ()) or ())
        if _external_record_source(record) in {"pubmed", "fda", "conference"}
    ]
    pubmed_records = [
        record for record in records_all
        if _external_record_source(record) == "pubmed"
    ][:topk]
    supplement_records = [
        record for record in records_all
        if _external_record_source(record) in {"fda", "conference"}
    ]
    records = pubmed_records + supplement_records
    if not records:
        return "(External Evidence: no evidence found)", []

    pubmed_lines, fda_lines, conference_lines, raw = [], [], [], []
    for i, record in enumerate(records, 1):
        provenance = getattr(record, "provenance", None)
        source = _external_record_source(record)
        pmid = _pubmed_pmid_from_record(record)
        title = str(getattr(record, "title", "") or "").strip()
        summary = str(getattr(record, "summary", "") or "").strip()
        snippet = summary.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"

        source_id = _external_record_source_id(record, fallback_prefix=source.upper(), index=i)
        if source == "pubmed":
            citation_tag = f"[@pubmed | {pmid}]" if pmid else ""
            pmid_text = f"PMID {pmid}" if pmid else "PMID"
            pubmed_lines.append(f"[{i}] {pmid_text} {citation_tag}\n    {title}\n    {snippet}")
        elif source == "fda":
            citation_tag = f"[@fda | {source_id}]"
            fda_lines.append(f"[{i}] FDA {citation_tag}\n    {title}\n    {snippet}")
        elif source == "conference":
            citation_tag = f"[@conference | {source_id}]"
            conference_lines.append(f"[{i}] Conference {citation_tag}\n    {title}\n    {snippet}")

        raw.append({
            "rank": i,
            "score": None,
            "source": source,
            "pmid": pmid,
            "title": title,
            "abstract": summary,
            "summary": summary,
            "text": summary,
            "journal": getattr(provenance, "location_hint", None),
            "pub_date": None,
            "doi": None,
            "impact_factor": None,
            "similarity": None,
            "source_id": source_id,
            "source_url": getattr(provenance, "source_url", None),
            "backend_name": getattr(record, "backend_name", ""),
            "evidence_type": getattr(record, "evidence_type", ""),
            "citation": getattr(record, "citation", None),
            "selection_rationale": getattr(record, "selection_rationale", None),
        })

    sections = []
    if pubmed_lines:
        sections.append("PUBMED Publications:\n" + "\n".join(pubmed_lines))
    if fda_lines:
        sections.append("FDA Labels:\n" + "\n".join(fda_lines))
    if conference_lines:
        sections.append("Conference Abstracts:\n" + "\n".join(conference_lines))
    pack = "External Evidence Pack:\n" + "\n\n".join(sections)
    return pack, raw


def pubmed_search_pack(
    query: str,
    topk: int = 8,
):
    """
    PubMed literature search via the current external_evidence engine tool.
    Returns formatted pack + raw hits for logging.
    """
    sanitized, _ = sanitize_rag_query(query or "")
    if not sanitized:
        return "(PUBMED: empty query)", []
    try:
        from engine_bridge import build_tool_input
        from engine_bridge import default_tool_call_id
        from utils.runtime_config import tool_input_overrides_from_env

        overrides = tool_input_overrides_from_env("external_evidence")
        overrides["query_analysis_mode"] = "off"
        query_analysis_style = str(overrides.get("query_analysis_style") or "single")
        overrides["query_analysis_input"] = _external_query_analysis_input(
            sanitized,
            query_analysis_style=query_analysis_style,
        )
        if not str(os.environ.get("OMGS_EXTERNAL_SEARCH_DEPTH") or "").strip():
            overrides["search_depth"] = "balanced"
        overrides.setdefault("followup_depth", "off")
        if not str(os.environ.get("OMGS_EXTERNAL_FDA_MODE") or "").strip():
            overrides["fda_mode"] = "auto"
        if not str(os.environ.get("OMGS_EXTERNAL_CONFERENCE_MODE") or "").strip():
            overrides["conference_mode"] = "auto"
        tool_input = build_tool_input(
            query=sanitized,
            **overrides,
        )
        result = _external_evidence_engine().invoke_tool(
            tool_name="external_evidence",
            tool_call_id=default_tool_call_id("external_evidence"),
            tool_input=tool_input,
            consumer="mdt",
            include_artifacts=False,
            include_debug=False,
            include_snapshots=True,
        )
        bundle = _bundle_from_tool_result(result)
        return _build_pubmed_pack_from_bundle(bundle, topk=topk)
    except Exception as exc:
        return f"(PUBMED: retrieval failed: {exc})", []
