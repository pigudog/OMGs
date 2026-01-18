from typing import Any, Dict, List, Optional
# utils/tools.py
class Color:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# Helper for trial normalization (bilingual/heterogeneous fields)
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

    # --- Common bilingual key mapping ---
    name = (
        t.get("name")
        or t.get("trial_name")
        or t.get("项目名称")
        or t.get("研究名称")
        or t.get("标题")
    )

    phase = (
        t.get("phase")
        or t.get("trial_phase")
        or t.get("期别")
        or t.get("试验分期")
    )

    # Some datasets store cancer type/conditions under Chinese keys
    conditions = (
        t.get("conditions")
        or t.get("cancer_type")
        or t.get("disease")
        or t.get("适应症")
        or t.get("疾病")
        or t.get("肿瘤类型")
        or t.get("病种")
    )

    line_of_therapy = (
        t.get("line_of_therapy")
        or t.get("treatment_line")
        or t.get("目标受试者治疗线数")
        or t.get("治疗线数")
    )

    biomarker = (
        t.get("biomarker")
        or t.get("required_biomarker")
        or t.get("marker")
        or t.get("生物标志物")
        or t.get("分子标志物")
        or t.get("PD-L1")
    )

    # Treatment / regimen text (often the most informative field)
    treatment = (
        t.get("treatment")
        or t.get("regimen")
        or t.get("intervention")
        or t.get("arms")
        or t.get("试验治疗/用药")
        or t.get("治疗方案")
        or t.get("用药")
    )

    doctor = t.get("doctor") or t.get("PI") or t.get("医生") or t.get("研究者")

    sponsor = (
        t.get("sponsor")
        or t.get("applicant")
        or t.get("申请单位")
        or t.get("申办方")
        or t.get("申办单位")
    )

    lead_or_participation = (
        t.get("lead_or_participation")
        or t.get("lead")
        or t.get("牵头/参与")
        or t.get("牵头")
        or t.get("参与")
    )

    contacts = t.get("contacts") or t.get("联系人") or t.get("contact")
    phones = t.get("phones") or t.get("联系电话") or t.get("phone")

    if isinstance(contacts, str):
        contacts = [contacts]
    if isinstance(phones, str):
        phones = [phones]

    # NOTE: some hospital IIT sheets put inclusion/exclusion into one Chinese field
    key_inclusion = (
        t.get("key_inclusion")
        or t.get("inclusion_criteria")
        or t.get("入组标准")
        or t.get("入组排除标准")
        or t.get("纳入标准")
    )

    key_exclusion = (
        t.get("key_exclusion")
        or t.get("exclusion_criteria")
        or t.get("排除标准")
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

###############################################################################
# 🔧 Helper: safe JSON parsing for model outputs
###############################################################################
def safe_parse_json_block(text: str) -> dict:
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

# --- Stable JSON/text helpers for case fingerprinting and normalization ---
def _stable_json_dumps(x) -> str:
    """Deterministically serialize dict/list for hashing/logging."""
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def question_to_text(question) -> str:
    """Normalize question input (dict/list/str/other) into a stable JSON/text string."""
    if isinstance(question, (dict, list)):
        return _stable_json_dumps(question)
    return str(question)

###############################################################################
# 8. CHAIR FINAL OUTPUT
###############################################################################
def generate_final_output(chair_agent, all_round_ops, clinic_time):
    expert_final = json.dumps(all_round_ops, ensure_ascii=False, indent=2)

    prompt = f"""
As the MDT chair for gynecologic oncology, you are seeing the patient at OUTPATIENT TIME: {clinic_time}.
Based on PATIENT FACTS + MDT discussion + FINAL refined plans from all experts, determine the CURRENT best management plan for this visit.

STRICT RULES:
- Any factual statement about past tests/treatments must include [@report_id|date] or say unknown.
- If experts disagree, pick the safest plan and state the key uncertainty.

# FINAL REFINED PLANS (All experts, last round)
{expert_final}

# Response Format
Final Assessment:
<1–3 sentences: summarize histology/biology, current disease status, and key uncertainties>

Core Treatment Strategy:
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >

Change Triggers:
- < ≤20 words “if X, then adjust management from A to B” >
- < ≤20 words “if X, then adjust management from A to B” >
"""
    return chair_agent.chat(prompt)



###############################################################################
# ⭐ Assistant Trial Suggestion
###############################################################################
import json

def assistant_trial_suggestion(agent, case_json_str, trials_list):
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
    for t in (trials_list or [])[:40]
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
    answer = agent.chat(prompt).strip()
    return answer
