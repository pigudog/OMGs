"""Console utilities - Pure helper functions for OMGs pipeline.

This module contains:
- Color: ANSI color codes for terminal output
- preview_text: Truncate text for display
- print_prompt_budget: Debug helper for token budget
- normalize_trial_compact: Normalize trial records to compact schema
- safe_parse_json_block: Safe JSON parsing from model outputs
- question_to_text: Normalize question input to stable text
"""

from typing import Any, Dict
import json


class Color:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def preview_text(x: Any, n: int = 160) -> str:
    """Truncate text for display, replacing newlines with spaces.
    
    Args:
        x: Input value (will be converted to string)
        n: Maximum length before truncation (default 160)
    
    Returns:
        Truncated string with ellipsis if needed
    """
    s = "" if x is None else str(x)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else (s[:n] + "…")


def print_prompt_budget(label: str, prompt: str) -> None:
    """Debug helper to print prompt token budget (observability only).
    
    Args:
        label: Label for the prompt (e.g., "role/initial")
        prompt: The prompt text to count tokens for
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode(prompt))
        print(f"{Color.OKCYAN}[TOKEN_BUDGET] {label}: ~{tokens} tokens{Color.RESET}")
    except Exception:
        pass


def normalize_trial_compact(t: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize heterogeneous trial records (English/Chinese keys) into a compact schema.

    IMPORTANT:
    - Must be loss-minimizing for hospital IIT sheets.
    - Keep key operational fields (doctor/contact/phone/treatment text) so matching doesn't drop content.
    - Convert basic HTML breaks into readable text.
    """

    def _html_to_text(x: Any) -> str:
        s = "" if x is None else str(x)
        # normalize common HTML line breaks used in the source JSON
        s = s.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
        s = s.replace("\r\n", "\n")
        # strip a few noisy div wrappers without heavy HTML parsing
        s = s.replace("<div>", "").replace("</div>", "")
        return s.strip()

    t = t or {}

    # --- Common bilingual key mapping (English and Chinese keys for data compatibility) ---
    name = (
        t.get("name")
        or t.get("trial_name")
        or t.get("项目名称")  # Chinese: "project name"
        or t.get("研究名称")  # Chinese: "study name"
        or t.get("标题")  # Chinese: "title"
    )

    phase = (
        t.get("phase")
        or t.get("trial_phase")
        or t.get("期别")  # Chinese: "phase"
        or t.get("试验分期")  # Chinese: "trial phase"
    )

    # Some datasets store cancer type/conditions under Chinese keys
    conditions = (
        t.get("conditions")
        or t.get("cancer_type")
        or t.get("disease")
        or t.get("适应症")  # Chinese: "indication"
        or t.get("疾病")  # Chinese: "disease"
        or t.get("肿瘤类型")  # Chinese: "tumor type"
        or t.get("病种")  # Chinese: "disease type"
    )

    line_of_therapy = (
        t.get("line_of_therapy")
        or t.get("treatment_line")
        or t.get("目标受试者治疗线数")  # Chinese: "target subject treatment line number"
        or t.get("治疗线数")  # Chinese: "treatment line number"
    )

    biomarker = (
        t.get("biomarker")
        or t.get("required_biomarker")
        or t.get("marker")
        or t.get("生物标志物")  # Chinese: "biomarker"
        or t.get("分子标志物")  # Chinese: "molecular marker"
        or t.get("PD-L1")
    )

    # Treatment / regimen text (often the most informative field)
    treatment = (
        t.get("treatment")
        or t.get("regimen")
        or t.get("intervention")
        or t.get("arms")
        or t.get("试验治疗/用药")  # Chinese: "trial treatment/medication"
        or t.get("治疗方案")  # Chinese: "treatment plan"
        or t.get("用药")  # Chinese: "medication"
    )

    doctor = t.get("doctor") or t.get("PI") or t.get("医生") or t.get("研究者")  # Chinese: "doctor", "researcher"

    sponsor = (
        t.get("sponsor")
        or t.get("applicant")
        or t.get("申请单位")  # Chinese: "applicant unit"
        or t.get("申办方")  # Chinese: "sponsor"
        or t.get("申办单位")  # Chinese: "sponsor unit"
    )

    lead_or_participation = (
        t.get("lead_or_participation")
        or t.get("lead")
        or t.get("牵头/参与")  # Chinese: "lead/participate"
        or t.get("牵头")  # Chinese: "lead"
        or t.get("参与")  # Chinese: "participate"
    )

    contacts = t.get("contacts") or t.get("联系人") or t.get("contact")  # Chinese: "contact person"
    phones = t.get("phones") or t.get("联系电话") or t.get("phone")  # Chinese: "contact phone"

    if isinstance(contacts, str):
        contacts = [contacts]
    if isinstance(phones, str):
        phones = [phones]

    # NOTE: some hospital IIT sheets put inclusion/exclusion into one Chinese field
    key_inclusion = (
        t.get("key_inclusion")
        or t.get("inclusion_criteria")
        or t.get("入组标准")  # Chinese: "enrollment criteria"
        or t.get("入组排除标准")  # Chinese: "enrollment/exclusion criteria"
        or t.get("纳入标准")  # Chinese: "inclusion criteria"
    )

    key_exclusion = (
        t.get("key_exclusion")
        or t.get("exclusion_criteria")
        or t.get("排除标准")  # Chinese: "exclusion criteria"
    )

    # keep original raw blocks as well, to avoid any future content loss
    return {
        "id": t.get("id"),
        "name": name,
        "phase": phase,
        "conditions": conditions,
        "line_of_therapy": line_of_therapy,
        "biomarker": biomarker,
        "doctor": doctor,
        "sponsor": sponsor,
        "lead_or_participation": lead_or_participation,
        "treatment": _html_to_text(treatment),
        "key_inclusion": _html_to_text(key_inclusion),
        "key_exclusion": _html_to_text(key_exclusion),
        "contacts": contacts or [],
        "phones": phones or [],
        "_raw": t,
    }


def safe_parse_json_block(text: str) -> dict:
    """Safe JSON parsing for model outputs.
    
    Attempts to extract and parse JSON from potentially malformed model output.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}
    return {}


def _stable_json_dumps(x) -> str:
    """Deterministically serialize dict/list for hashing/logging."""
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def question_to_text(question) -> str:
    """Normalize question input (dict/list/str/other) into a stable JSON/text string."""
    if isinstance(question, (dict, list)):
        return _stable_json_dumps(question)
    return str(question)
