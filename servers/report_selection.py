from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from utils.time_utils import safe_date10, parse_dt, parse_date
from utils.console_utils import safe_parse_json_block
import re
import json
import os
# ============================================================
# JSONL IO + id parsing
# ============================================================

_JSONL_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_JSONL_INDEX_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


def read_jsonl(path: str, warn_bad_lines: bool = True, max_warn: int = 5) -> List[Dict[str, Any]]:
    """
    Read JSONL file and return list of dictionaries.
    
    Returns empty list if file doesn't exist or cannot be read.
    This ensures the pipeline continues even when data files are missing.
    """
    data: List[Dict[str, Any]] = []
    bad_count = 0
    
    # Check if file exists
    if not path:
        return []
    
    if not os.path.exists(path):
        if warn_bad_lines:
            print(f"[WARNING] JSONL file not found: {path}. Returning empty list.")
        return []
    
    try:
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
    except FileNotFoundError:
        if warn_bad_lines:
            print(f"[WARNING] JSONL file not found: {path}. Returning empty list.")
        return []
    except PermissionError:
        if warn_bad_lines:
            print(f"[WARNING] Permission denied reading JSONL file: {path}. Returning empty list.")
        return []
    except Exception as e:
        if warn_bad_lines:
            print(f"[WARNING] Error reading JSONL file {path}: {e}. Returning empty list.")
        return []
    
    return data


def _get_jsonl_index(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get indexed JSONL data by meta_info.
    
    Returns empty dict if file doesn't exist or cannot be read.
    """
    if not path:
        return {}
    if path in _JSONL_INDEX_CACHE:
        return _JSONL_INDEX_CACHE[path]
    
    try:
        data = _JSONL_CACHE.get(path)
        if data is None:
            data = read_jsonl(path)  # Returns [] if file doesn't exist
            _JSONL_CACHE[path] = data
        
        index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in data:
            key = str(row.get("meta_info") or "")
            if key:
                index[key].append(row)
        _JSONL_INDEX_CACHE[path] = index
        return index
    except Exception as e:
        print(f"[WARNING] Failed to index JSONL file {path}: {e}. Returning empty index.")
        _JSONL_INDEX_CACHE[path] = {}
        return {}


def _pack_reports_for_prompt(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the stable report fields passed into expert prompts."""
    return [
        {
            "report_id": str(r.get("report_id")),
            "date": (r.get("report_date") or r.get("date") or "")[:19],
            "raw_text": r.get("raw_text") or "",
        }
        for r in (reports or [])
    ]


def parse_ids(text: str) -> List[str]:
    """Parse a selection string into a list of report_ids."""
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


def _fallback_select_reports(
    reports_raw: List[Dict[str, Any]],
    report_type: str,
    max_keep: int = 3,
) -> List[Dict[str, Any]]:
    """Deterministic fallback when model-assisted selection is unavailable.

    For lab reports, prefer recent abnormal/tumor-marker summaries but keep
    latest reports as fill-ins because normal organ-function labs can be
    treatment-eligibility evidence.
    """
    if not reports_raw:
        return []

    sorted_reports = sorted(
        reports_raw,
        key=lambda r: parse_dt(r.get("report_date") or r.get("date")) or datetime.min,
    )

    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    def _add(report: Dict[str, Any]) -> None:
        if len(selected) >= max_keep:
            return
        rid = str(report.get("report_id"))
        if rid in selected_ids:
            return
        selected.append(report)
        selected_ids.add(rid)

    newest_first = list(reversed(sorted_reports))

    if report_type == "lab":
        # Always preserve the most recent lab first; normal current organ-function
        # labs can be safety evidence for systemic treatment eligibility.
        if newest_first:
            _add(newest_first[0])
        for rpt in newest_first:
            summary = str(rpt.get("summary") or "")
            if "Abn(" in summary or "Tumor" in summary:
                _add(rpt)
    else:
        for rpt in newest_first:
            _add(rpt)

    for rpt in newest_first:
        _add(rpt)

    selected = sorted(
        selected,
        key=lambda r: parse_dt(r.get("report_date") or r.get("date")) or datetime.min,
    )
    return _pack_reports_for_prompt(selected)


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
    patient.sort(key=lambda r: (parse_date(r.get("report_date")) or datetime.min.date()))
    return patient



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


def select_reports_for_roles(
    *,
    roles: List[str],
    role_permissions: Dict[str, Dict[str, Any]],
    lab_timeline: List[Dict[str, Any]],
    lab_reports: List[Dict[str, Any]],
    im_timeline: List[Dict[str, Any]],
    im_reports: List[Dict[str, Any]],
    path_timeline: List[Dict[str, Any]],
    path_reports: List[Dict[str, Any]],
    mut_reports: List[Dict[str, Any]],
    pathology_json: Optional[str],
    agent_class,
    expert_select_fn,
    model,
    client,
    color,
    use_shared: bool = True,
    max_tokens: int = 220,
    max_prompt_tokens: int = 2000,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Select reports per role. If use_shared is True, run selection once per report type and reuse.
    """
    context: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "lab": {},
        "imaging": {},
        "pathology": {},
        "mutation": {},
    }

    instruction = (
        "You are a report-filtering module in an MDT pipeline.\n"
        "Goal: select the MINIMAL set of reports NECESSARY for current decision-making.\n"
        "Prefer newest; max 3. JSON only."
    )

    def _first_role_with(perm_key: str) -> str:
        for r in roles:
            if role_permissions[r].get(perm_key):
                return r
        return roles[0] if roles else "chair"

    shared_selected = {"lab": [], "imaging": [], "pathology": [], "mutation": []}
    if use_shared:
        selector = agent_class(
            instruction=instruction,
            role="shared_selector",
            model_info=model,
            client=client,
            max_tokens=max_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
        lab_role = _first_role_with("lab")
        img_role = _first_role_with("imaging")
        path_role = _first_role_with("pathology")

        print(f"{color.OKBLUE}\n📄 Selecting reports once (shared mode){color.RESET}")
        print(f"{color.OKCYAN} - Using {lab_role} perspective for lab reports{color.RESET}")
        shared_selected["lab"] = expert_select_fn(selector, lab_role, lab_timeline, lab_reports, "lab")
        print(f"{color.OKCYAN} - Using {img_role} perspective for imaging reports{color.RESET}")
        shared_selected["imaging"] = expert_select_fn(selector, img_role, im_timeline, im_reports, "imaging")
        if pathology_json:
            print(f"{color.OKCYAN} - Passing pathology reports to pathology-enabled roles{color.RESET}")
            shared_selected["pathology"] = _pack_reports_for_prompt(path_reports)
        if mut_reports:
            print(f"{color.OKCYAN} - Passing molecular/genomics reports to mutation-enabled roles{color.RESET}")
            shared_selected["mutation"] = _pack_reports_for_prompt(mut_reports)

    for role in roles:
        if use_shared:
            print(f"{color.OKBLUE}\n📄 Assigning shared selected reports to {role}...{color.RESET}")
        else:
            print(f"{color.OKBLUE}\n📄 Selecting reports for {role}...{color.RESET}")

        if role_permissions[role]["lab"]:
            if use_shared:
                context["lab"][role] = shared_selected["lab"]
            else:
                selector = agent_class(
                    instruction=instruction,
                    role=f"{role}_selector",
                    model_info=model,
                    client=client,
                    max_tokens=max_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                )
                context["lab"][role] = expert_select_fn(selector, role, lab_timeline, lab_reports, "lab")
        else:
            context["lab"][role] = []

        if role_permissions[role]["imaging"]:
            if use_shared:
                context["imaging"][role] = shared_selected["imaging"]
            else:
                selector = agent_class(
                    instruction=instruction,
                    role=f"{role}_selector",
                    model_info=model,
                    client=client,
                    max_tokens=max_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                )
                context["imaging"][role] = expert_select_fn(selector, role, im_timeline, im_reports, "imaging")
        else:
            context["imaging"][role] = []

        if role_permissions[role]["pathology"] and pathology_json:
            if use_shared:
                context["pathology"][role] = shared_selected["pathology"]
            else:
                context["pathology"][role] = _pack_reports_for_prompt(path_reports)
        else:
            context["pathology"][role] = []

        if role_permissions[role].get("mutation") and mut_reports:
            context["mutation"][role] = shared_selected["mutation"] if use_shared else _pack_reports_for_prompt(mut_reports)
        else:
            context["mutation"][role] = []

    return context


###############################################################################
# EXPERT REPORT SELECTION (stateless, low-token)
###############################################################################
def expert_select_reports(agent, role, reports_timeline, reports_raw, report_type, max_keep=3):
    """
    Select clinically relevant reports for each MDT role.
    Ensures the latest CBC/LFT/renal/tumor markers are ALWAYS included,
    even if normal.
    """

    role_rules = {
        "oncologist": {
            "lab": (
                "Pick the MOST RECENT CBC, LFT, renal, and tumor markers "
                "(normal or abnormal). These ALWAYS determine systemic therapy safety."
            ),
            "imaging": "Return none.",
            "pathology": "Return none.",
        },

        "radiologist": {
            "lab": "Return none.",
            "imaging": (
                "Pick the latest CT/MRI that defines disease status "
                "(progression/new lesions/complications). If stable and irrelevant, skip."
            ),
            "pathology": "Return none.",
        },

        "nuclear": {
            "lab": "Return none.",
            "imaging": (
                "Pick PET/CT or imaging clarifying metabolic activity. "
                "If multiple, prefer the latest decisive one."
            ),
            "pathology": "Return none.",
        },

        "pathologist": {
            "lab": "Return none.",
            "imaging": "Return none.",
            "pathology": (
                "Pick pathology that changes diagnosis: histology, grade, STIC, implants, "
                "IHC, molecular, or biomarkers affecting systemic therapy."
            ),
        },

        "chair": {
            "lab": (
                "Pick the MOST RECENT CBC, LFT, renal function, and key tumor markers "
                "(normal or abnormal). These determine treatment eligibility."
            ),
            "imaging": (
                "Pick only imaging that confirms progression/response or urgent complications. "
                "Prefer newest CT/MRI relevant to management."
            ),
            "pathology": "If available, pick pathology altering diagnosis or biomarkers.",
        },
    }

    # role rule fallback
    rule = role_rules.get(role, {}).get(report_type, "Pick only the newest decisive reports.")

    # ------------ Case: timeline empty → fallback to latest local ------------
    if not reports_timeline:
        return _fallback_select_reports(reports_raw, report_type, max_keep=max_keep)

    # ------------ LLM Selection Prompt ------------
    prompt = f"""
ROLE: {role}
REPORT_TYPE: {report_type}

TIMELINE (choose ONLY from these report_ids):
{json.dumps(reports_timeline, ensure_ascii=False)}

IMPORTANT: Use the EXACT report_id values from the TIMELINE above. 
When referencing reports in evidence tags, use format: [@actual_report_id | LAB/Genomics/MR/CT/Pathology]
Always use spaces around | for consistency: [@xxx | yyy]
Examples: [@LAB20251020TM | LAB] for lab reports, [@OH20251003 | Genomics] for genomics reports, [@CT20250922 | CT], [@PETCT20251021 | CT], and [@PX20251003 | Pathology] for pathology/IHC reports.

SELECTION RULE (must follow):
- {rule}
- Keep at most {max_keep} report_ids.
- Prefer the NEWEST reports.
- Stability does NOT mean exclusion: normal labs MUST be selected if they are CBC/LFT/renal/markers.
- Exclude irrelevant follow-ups that do NOT influence treatment decisions.
- Avoid empty list unless truly nothing is relevant.

Return STRICT JSON only:
{{"report_ids": ["id1","id2"], "reason": "<=20 words"}}
""".strip()

    # ------------- Run the agent -------------
    try:
        resp = agent.run_selection(prompt)
        data = safe_parse_json_block(resp)
    except Exception as exc:
        print(f"[WARNING] Report selector failed for {role}/{report_type}: {exc}. Using deterministic fallback.")
        return _fallback_select_reports(reports_raw, report_type, max_keep=max_keep)

    # Robustly accept common LLM JSON shapes:
    # 1) {"report_ids": [...], "reason": "..."}
    # 2) ["id1","id2"]
    # 3) {"ids": [...]} or {"reports": [...]} (best-effort)
    if isinstance(data, list):
        ids = data
    elif isinstance(data, dict):
        ids = data.get("report_ids", data.get("ids", data.get("reports", [])))
    else:
        ids = []

    if not isinstance(ids, list):
        ids = []

    # Allow list items to be either strings or small dicts like {"report_id": "..."}
    cleaned_ids: List[str] = []
    for x in ids:
        if isinstance(x, dict):
            x = x.get("report_id") or x.get("id")
        if x is None:
            continue
        sx = str(x).strip()
        if sx:
            cleaned_ids.append(sx)

    # de-dup + cap
    seen = set()
    ids = []
    for rid in cleaned_ids:
        if rid in seen:
            continue
        ids.append(rid)
        seen.add(rid)
        if len(ids) >= max_keep:
            break

    picked = []
    idset = set(ids)

    for rpt in (reports_raw or []):
        rid = str(rpt.get("report_id"))
        if rid in idset:
            picked.append({
                "report_id": rid,
                "date": (rpt.get("report_date", "") or rpt.get("date", ""))[:19],
                "raw_text": rpt.get("raw_text", ""),
            })

    # ------------ Fallback: if LLM returns nothing but raw exists ------------
    if not picked and reports_raw:
        picked = _fallback_select_reports(reports_raw, report_type, max_keep=max_keep)

    return picked
