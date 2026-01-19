from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from .time_utils import safe_date10,parse_dt
import re
import json
# ============================================================
# JSONL IO + id parsing
# ============================================================

_JSONL_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_JSONL_INDEX_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


def read_jsonl(path: str, warn_bad_lines: bool = True, max_warn: int = 5) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    bad_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                bad_count += 1
                if warn_bad_lines and bad_count <= max_warn:
                    preview = (line[:160] + "...") if len(line) > 160 else line
                    print(f"[WARNING] Bad JSONL line {idx} in {path}: {preview}")
                continue
    if warn_bad_lines and bad_count > max_warn:
        print(f"[WARNING] Bad JSONL lines in {path}: {bad_count} total (showing first {max_warn})")
    return data


def _get_jsonl_index(path: str) -> Dict[str, List[Dict[str, Any]]]:
    if not path:
        return {}
    if path in _JSONL_INDEX_CACHE:
        return _JSONL_INDEX_CACHE[path]
    data = _JSONL_CACHE.get(path)
    if data is None:
        data = read_jsonl(path)
        _JSONL_CACHE[path] = data
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in data:
        key = str(row.get("meta_info") or "")
        if key:
            index[key].append(row)
    _JSONL_INDEX_CACHE[path] = index
    return index


def parse_ids(text: str) -> List[str]:
    """Parse a selection string into a list of report_ids (keeps |)."""
    if not text:
        return []
    t = text.strip()
    # 'none' => empty
    if re.fullmatch(r"(?i)\s*none\s*[\.\!。]?\s*", t):
        return []
    t = t.replace("，", ",").replace("、", ",").replace("\n", ",")
    tokens = re.findall(r"[A-Za-z0-9_\-|\:]+", t)
    seen, out = set(), []
    for x in tokens:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ============================================================
# Patient report loaders
# ============================================================

def load_patient_labs(meta_info: str, json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not meta_info or not json_path:
        return [], []
    index = _get_jsonl_index(json_path)
    patient = list(index.get(str(meta_info), []))
    patient.sort(key=lambda r: (parse_dt(r.get("report_date")) or datetime.min))
    timeline = [
        {
            "report_id": rpt.get("report_id"),
            "date": safe_date10(rpt.get("report_date")),
            "summary": rpt.get("summary"),
        }
        for rpt in patient
    ]
    return timeline, patient


def load_patient_imaging(meta_info: str, json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not meta_info or not json_path:
        return [], []
    index = _get_jsonl_index(json_path)
    patient = list(index.get(str(meta_info), []))
    patient.sort(key=lambda r: (parse_dt(r.get("report_date")) or datetime.min))
    timeline: List[Dict[str, Any]] = []
    for rpt in patient:
        imp = rpt.get("impression") or ""
        imp_short = (imp[:80] + "...") if len(imp) > 80 else imp
        timeline.append(
            {
                "report_id": rpt.get("report_id"),
                "date": safe_date10(rpt.get("report_date")),
                "modality": rpt.get("modality"),
                "summary": imp_short,
            }
        )
    return timeline, patient


def load_patient_pathology(meta_info: str, json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load pathology reports for a patient. Returns (timeline, full_reports)."""
    if not meta_info or not json_path:
        return [], []

    index = _get_jsonl_index(json_path)
    patient = list(index.get(str(meta_info), []))
    patient.sort(key=lambda r: (parse_dt(r.get("report_date")) or datetime.min))
    timeline: List[Dict[str, Any]] = []
    for rpt in patient:
        # Pathology reports typically have histology, IHC, molecular info
        summary = rpt.get("summary") or rpt.get("histology") or rpt.get("diagnosis") or ""
        summary_short = (summary[:80] + "...") if len(summary) > 80 else summary
        timeline.append(
            {
                "report_id": rpt.get("report_id"),
                "date": safe_date10(rpt.get("report_date")),
                "summary": summary_short,
                "histology": rpt.get("histology"),
            }
        )
    return timeline, patient


###############################################################################
# Mutation Report Loader
###############################################################################
def load_patient_mutations(meta_info: str, json_path: str = "mutation_reports.jsonl") -> List[Dict[str, Any]]:
    """
    Load all mutation reports for a patient.
    
    The `raw_text` field from mutation_reports.jsonl is treated as patient facts
    and provided directly to relevant roles (chair, oncologist, pathologist).
    
    Args:
        meta_info: Patient identifier for matching reports
        json_path: Path to mutation reports JSONL file
    
    Returns:
        List of mutation report dictionaries, sorted by date
    """
    if not meta_info or not json_path:
        return []

    index = _get_jsonl_index(json_path)
    patient = list(index.get(str(meta_info), []))
    patient.sort(key=lambda r: (parse_date_any(r.get("report_date")) or datetime.min.date()))
    return patient



###############################################################################
# Date Parsing and Report Helpers
###############################################################################
def parse_date_any(d: str):
    """
    Parse date-like strings:
    - "2022-12-09"
    - "2022-12-09T00:00:00"
    - "2023-06-05T15:12:06"
    Return datetime.date or None
    """
    if not d:
        return None
    s = str(d).strip()
    if len(s) >= 10:
        s = s[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def summarize_selected_reports(context: dict):
    """
    For debug/trace: show report_id|date per role.
    """
    out = {}
    for typ in ("lab", "imaging", "pathology", "mutation"):
        out[typ] = {}
        for role, lst in (context.get(typ, {}) or {}).items():
            out[typ][role] = [f"{x.get('report_id')}|{(x.get('date') or '')[:10]}" for x in (lst or [])]
    return out
