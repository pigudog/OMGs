###############################################################################
# MODEL4. OMGs – MDT MULTI EXPERT + CLINICAL TRIAL ASSISTANT VERSION
###############################################################################
from typing import Any, Dict, List, Optional
import json
from datetime import datetime
from .console_utils import safe_parse_json_block,Color
from .core import Agent
from .select_utils import parse_date_any

###############################################################################
# 0. FIXED ROLES + PERMISSIONS
###############################################################################
ROLES = ["chair", "oncologist", "radiologist", "pathologist", "nuclear"]

ROLE_PERMISSIONS = {
    "chair":        {"lab": True,  "imaging": True,  "pathology": False, "mutation": True,  "guideline": "chair"},
    "oncologist":   {"lab": True,  "imaging": False, "pathology": False, "mutation": True,  "guideline": "oncologist"},
    "radiologist":  {"lab": False, "imaging": True,  "pathology": False, "mutation": False, "guideline": "radiologist"},
    "pathologist":  {"lab": False, "imaging": False, "pathology": True,  "mutation": True,  "guideline": "pathologist"},
    "nuclear":      {"lab": False, "imaging": True,  "pathology": False, "mutation": False, "guideline": "nuclear"},
}

###############################################################################
# 2. ROLE-SPECIFIC CASE VIEW BUILDER
# Builds role-specific case views compatible with structured JSON input
###############################################################################
def safe_load_case_json(question) -> dict:
    if isinstance(question, dict):
        return question
    if isinstance(question, list):
        return {"_list": question}
    try:
        return json.loads(str(question))
    except Exception:
        return {}


def build_role_specific_case_view(role, case_json):
    core = case_json.get("CASE_CORE", {})
    timeline = case_json.get("TIMELINE", {})
    med_onc = case_json.get("MED_ONC", {})
    radiology = case_json.get("RADIOLOGY", {})
    pathology = case_json.get("PATHOLOGY", {})
    nuc = case_json.get("NUC_MED", {})
    labs = case_json.get("LAB_TRENDS", {})

    if role == "chair":
        return json.dumps(case_json, ensure_ascii=False, indent=2)

    if role == "oncologist":
        return json.dumps({
            "DIAGNOSIS": core.get("DIAGNOSIS"),
            "LINE_OF_THERAPY": core.get("LINE_OF_THERAPY"),
            "MAINTENANCE": core.get("MAINTENANCE"),
            "RELAPSE_DATE": core.get("RELAPSE_DATE"),
            "BIOMARKERS": core.get("BIOMARKERS"),
            "GENETICS": med_onc.get("genetic_testing"),
            "CURRENT_STATUS": core.get("CURRENT_STATUS"),
            "LAB_TRENDS": labs
        }, ensure_ascii=False, indent=2)

    if role == "radiologist":
        return json.dumps({
            "IMAGING_STUDIES": radiology.get("studies", []),
            "IMAGING_TRENDS": [
                {
                    "date": s.get("date", ""),
                    "impression": s.get("impression", ""),
                    "trend": s.get("trend_vs_prior", "Unknown")
                }
                for s in radiology.get("studies", [])
            ],
            "PET_IF_AVAILABLE": nuc.get("studies", [])
        }, ensure_ascii=False, indent=2)

    if role == "pathologist":
        return json.dumps({
            "HISTOLOGY_AND_IHC": pathology.get("specimens", []),
            "MOLECULAR": [
                core.get("BIOMARKERS", {}),
                med_onc.get("genetic_testing", {})
            ]
        }, ensure_ascii=False, indent=2)

    if role == "nuclear":
        return json.dumps({
            "PET_CT": nuc.get("studies", []),
            "IMAGING_CONTEXT": [
                {
                    "date": s.get("date", ""),
                    "impression": s.get("impression", "")
                }
                for s in radiology.get("studies", [])
            ]
        }, ensure_ascii=False, indent=2)

    return "{}"





###############################################################################
# 4. REPORT SELECTION (stateless, low-token)
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
        if not reports_raw:
            return []
        sorted_reports = sorted(
            reports_raw,
            key=lambda r: parse_date_any(r.get("report_date") or r.get("date")) or datetime.min.date()
        )
        latest_reports = sorted_reports[-max_keep:]
        return [
            {
                "report_id": str(r.get("report_id")),
                "date": (r.get("report_date", "") or r.get("date", ""))[:19],
                "raw_text": r.get("raw_text", ""),
            }
            for r in latest_reports
        ]

    # ------------ LLM Selection Prompt ------------
    prompt = f"""
ROLE: {role}
REPORT_TYPE: {report_type}

TIMELINE (choose ONLY from these report_ids):
{json.dumps(reports_timeline, ensure_ascii=False)}

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
    resp = agent.run_selection(prompt)
    data = safe_parse_json_block(resp)

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

    # ------------ Fallback: if LLM returns nothing but raw exists → choose newest ------------
    if not picked and reports_raw:
        sorted_reports = sorted(
            reports_raw,
            key=lambda r: parse_date_any(r.get("report_date") or r.get("date")) or datetime.min.date()
        )
        latest_reports = sorted_reports[-max_keep:]
        picked = [
            {
                "report_id": str(r.get("report_id")),
                "date": (r.get("report_date", "") or r.get("date", ""))[:19],
                "raw_text": r.get("raw_text", ""),
            }
            for r in latest_reports
        ]

    return picked



###############################################################################
# 5. INITIALIZE MDT SPECIALIST AGENT
###############################################################################
ROLE_PROMPTS = {
    "chair": """
# Context
You are the MDT chair. You integrate all specialties and maintain safety and coherence.

# Objective
Provide a high-level management direction (intent, safety, sequencing) without choosing specific drugs.
If information is missing, highlight what must be obtained before firm decisions.

# Style
Default: return up to 5 bullets. Each bullet ≤20 words. No fixed subheadings. No guideline quotes. Never recommend a specific agent.
Exception (FINAL OUTPUT): if the user prompt explicitly requests a structured format (e.g., "Final Assessment / Core Treatment Strategy / Change Triggers"), follow that format and IGNORE the 5-bullet constraint. Keep the Final Assessment to ONE short sentence.
""".strip(),

    "oncologist": """
# Context
You are the medical oncologist. You interpret systemic therapy history, toxicity, biomarkers, organ function, and intent.

# Objective
Identify systemic-treatment-relevant facts, constraints, and what further data are required to make a regimen decision.
You may describe treatment categories (e.g., maintenance, relapse therapy, surveillance), but must NOT name any specific drugs.

# Style
Return up to 5 bullets. Each ≤20 words. No fixed subheadings. No guideline quotes. No drug names.
""".strip(),

    "radiologist": """
# Context
You are the diagnostic radiologist. You interpret disease distribution, trend, and complications.

# Objective
Summarize actionable imaging findings (e.g., measurable disease, recurrence pattern, obstruction, complications).
Do NOT discuss systemic therapy choices.

# Style
Return up to 5 bullets. Each ≤20 words. Imaging only. No drug names. No treatment recommendations.
""".strip(),

    "pathologist": """
# Context
You are the pathologist. You interpret histology, IHC, and molecular pathology.

# Objective
Clarify diagnosis, grade, biomarker uncertainties, and which pathology details are missing.
Do NOT suggest treatment choices.

# Style
Return up to 5 bullets. Each ≤20 words. No drug names. No prognosis/treatment advice.
""".strip(),

    "nuclear": """
# Context
You are the nuclear medicine physician. You interpret PET-based metabolic patterns.

# Objective
Summarize metabolic findings and when PET meaningfully changes staging or suspicion of recurrence.
Do NOT comment on systemic therapy choices.

# Style
Return up to 5 bullets. Each ≤20 words. No drug names. No treatment recommendations.
""".strip(),
}

def init_expert_agent(
    role,
    question,
    model,
    client,
    context,
    case_fingerprint,
    global_guideline_digest: str,
    device="auto",
    topk=5,
    visit_time: Optional[str] = None,
):
    case_json = safe_load_case_json(question)
    case_view = build_role_specific_case_view(role, case_json)
    perm = ROLE_PERMISSIONS[role]

    # Clinical reports selected for this role
    clinical = ""
    if perm["lab"]:
        clinical += f"# LAB REPORTS (PATIENT FACTS) SELECTED BY {role}\n"
        clinical += json.dumps(context["lab"].get(role, []), ensure_ascii=False, indent=2) + "\n\n"

    if perm["imaging"]:
        clinical += f"# IMAGING REPORTS (PATIENT FACTS) SELECTED BY {role}\n"
        clinical += json.dumps(context["imaging"].get(role, []), ensure_ascii=False, indent=2) + "\n\n"

    if perm["pathology"]:
        clinical += f"# PATHOLOGY REPORTS (PATIENT FACTS) SELECTED BY {role}\n"
        clinical += json.dumps(context["pathology"].get(role, []), ensure_ascii=False, indent=2) + "\n\n"

    # MUTATION / MOLECULAR reports: provided directly as patient facts to chair / oncologist / pathologist
    mut_for_role = (context.get("mutation", {}) or {}).get(role, [])
    if mut_for_role:
        clinical += "# MUTATION / MOLECULAR REPORTS (PATIENT FACTS)\n"
        clinical += json.dumps(mut_for_role, ensure_ascii=False, indent=2) + "\n\n"

    if not clinical.strip():
        clinical = "# No clinical reports for this role.\n\n"

    role_prompt = ROLE_PROMPTS.get(role, "").strip() or (
        "Return up to 5 bullets. Each bullet ≤20 words. Use only provided information; do not hallucinate."
    )

    visit_time_str = visit_time or "Unknown visit date"

    instruction = f"""
OUTPATIENT VISIT TIME (today's clinic decision point): {visit_time_str}

CASE_FINGERPRINT: {case_fingerprint}

{role_prompt}

# HARD RULES (critical)
1) All decisions are for THIS visit date and future care, not for past timepoints.
2) PATIENT FACTS come ONLY from:
   - Role-Specific Case View, and
   - Clinical Reports selected for this role (including mutation reports if provided).
3) GLOBAL Guideline Digest is ONLY general reference:
   - MUST NOT be treated as patient-specific facts.
   - Never invent labs/imaging/mutations from guidelines.
4) Any claim about labs/imaging/pathology/molecular MUST include evidence tag:
   - format: [@report_id|YYYY-MM-DD]
   - If no report supports it, say "unknown/needs update".
5) If Case View conflicts with Clinical Reports:
   - Prefer Clinical Reports; note discrepancy briefly.
6) Do NOT hallucinate. If missing, defer to correct specialty.

# Role-Specific Case View (PATIENT FACTS)
{case_view}

# Clinical Reports (PATIENT FACTS)
{clinical}

# GLOBAL Guideline Digest (NOT PATIENT FACTS)
{global_guideline_digest}
""".strip()

    ag = Agent(
        instruction=instruction,
        role=role,
        model_info=model,
        client=client,
        max_tokens=900,
        max_prompt_tokens=6500,
    )
    ag.inject_assistant("System ready for MDT discussion.")
    print(f"{Color.OKGREEN}✔ Initialized agent for role: {role}{Color.RESET}")
    return ag

