"""Expert Agent Definitions - LLM-powered specialist agents for MDT."""

from typing import Any, Dict, List, Optional
import json
from core.agent import Agent
from servers.info_delivery import safe_load_case_json, build_role_specific_case_view
from servers.reports_selector import expert_select_reports
from utils.console_utils import Color
from utils.skill_loader import build_skill_digest

###############################################################################
# FIXED ROLES + PERMISSIONS
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
# ROLE PROMPTS
###############################################################################
ROLE_PROMPTS = {
    "chair": """
# Context
You are the MDT chair. You integrate all specialties and maintain safety and coherence.

# Objective
Provide a high-level management direction (intent, safety, sequencing) without choosing specific drugs.
If information is missing, highlight what must be obtained before firm decisions.

# Style
Default: return up to 3 bullets. Each bullet ≤20 words. No fixed subheadings. No guideline quotes. Never recommend a specific agent.
Exception (FINAL OUTPUT): if the user prompt explicitly requests a structured format (e.g., "Final Assessment / Core Treatment Strategy / Change Triggers"), follow that format and IGNORE the 5-bullet constraint. Keep the Final Assessment to ONE short sentence.
""".strip(),

    "oncologist": """
# Context
You are the medical oncologist. You interpret systemic therapy history, toxicity, biomarkers, organ function, and intent.

# Objective
Identify systemic-treatment-relevant facts, constraints, and what further data are required to make a regimen decision.
You may describe treatment categories (e.g., maintenance, relapse therapy, surveillance), but must NOT name any specific drugs.

# Style
Return up to 3 bullets. Each ≤20 words. No fixed subheadings. No guideline quotes. No drug names.
""".strip(),

    "radiologist": """
# Context
You are the diagnostic radiologist. You interpret disease distribution, trend, and complications.

# Objective
Summarize actionable imaging findings (e.g., measurable disease, recurrence pattern, obstruction, complications).
Do NOT discuss systemic therapy choices.

# Style
Return up to 3 bullets. Each ≤20 words. Imaging only. No drug names. No treatment recommendations.
""".strip(),

    "pathologist": """
# Context
You are the pathologist. You interpret histology, IHC, and molecular pathology.

# Objective
Clarify diagnosis, grade, biomarker uncertainties, and which pathology details are missing.
Do NOT suggest treatment choices.

# Style
Return up to 3 bullets. Each ≤20 words. No drug names. No prognosis/treatment advice.
""".strip(),

    "nuclear": """
# Context
You are the nuclear medicine physician. You interpret PET-based metabolic patterns.

# Objective
Summarize metabolic findings and when PET meaningfully changes staging or suspicion of recurrence.
Do NOT comment on systemic therapy choices.

# Style
Return up to 3 bullets. Each ≤20 words. No drug names. No treatment recommendations.
""".strip(),
}


def init_expert_agent(
    role: str,
    question: Any,
    model: str,
    client: Any,
    context: Dict[str, Dict[str, List[Dict[str, Any]]]],
    case_fingerprint: str,
    global_guideline_digest: str,
    device: str = "auto",
    topk: int = 5,
    visit_time: Optional[str] = None,
) -> Agent:
    """Initialize an expert agent for a specific MDT role."""
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
        clinical += "⚠️ COMPREHENSIVE NGS PANEL (~20,000 genes) - INTERPRETATION RULES:\n"
        clinical += "• '未检出' (not detected) = NO pathogenic mutation found\n"
        clinical += "• '（视为阴性）' (considered negative) = NO pathogenic mutation found\n"
        clinical += "• '阴性' (negative) = negative result\n"
        clinical += "• Genes with specific variants (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation\n"
        clinical += "• If a gene of interest is NOT mentioned in the report, it means NO pathogenic mutation (comprehensive panel)\n"
        clinical += "• NEVER say 'not tested' or 'not reported' - comprehensive NGS WAS done.\n"
        clinical += "• Only say 'unknown' if NO mutation report is provided at all.\n\n"
        clinical += json.dumps(mut_for_role, ensure_ascii=False, indent=2) + "\n\n"

    if not clinical.strip():
        clinical = "# No clinical reports for this role.\n\n"

    role_prompt = ROLE_PROMPTS.get(role, "").strip() or (
        "Return up to 5 bullets. Each bullet ≤20 words. Use only provided information; do not hallucinate."
    )

    visit_time_str = visit_time or "Unknown visit date"

    # Inject SKILL protocol for evidence format and role behavior enforcement
    skill_digest = build_skill_digest(role)

    instruction = f"""
{skill_digest}

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
4) Any claim derived from guideline/PubMed evidence MUST include evidence tag:
   - applies to treatment strategy categories, guideline/consensus statements, or trial/literature evidence
   - format: [@guideline:doc_id | Page xx] or [@pubmed | PMID]
4b) At least ONE bullet must be evidence-based and include [@guideline:doc_id | Page xx] or [@pubmed | PMID].
5) Any claim about labs/imaging/pathology/molecular MUST include evidence tag:
   - format: [@actual_report_id | LAB/Genomics/MR/CT] using actual report_id from report data
   - Examples: [@20220407|17300673 | LAB], [@OH2203828|2022-04-18 | Genomics], [@2022-12-29 | MR], [@2022-12-29 | CT]
   - Note: Always use spaces around | for consistency: [@xxx | yyy]
   - Use the exact report_id value from the Clinical Reports section above
   - If no report supports it, say "unknown/needs update".
6) If Case View conflicts with Clinical Reports:
   - Prefer Clinical Reports; note discrepancy briefly.
7) Do NOT hallucinate. If missing, defer to correct specialty.

# Role-Specific Case View (PATIENT FACTS)
{case_view}

# Clinical Reports (PATIENT FACTS)
{clinical}

# GLOBAL Guideline + PubMed Digest (NOT PATIENT FACTS)
{global_guideline_digest}
""".strip()

    # Chair agent needs more tokens due to comprehensive case view and reports
    # Other roles can use standard budget
    max_prompt_tokens = 20000 if role == "chair" else 30000
    
    ag = Agent(
        instruction=instruction,
        role=role,
        model_info=model,
        client=client,
        max_tokens=900,
        max_prompt_tokens=max_prompt_tokens,
    )
    ag.inject_assistant("System ready for MDT discussion.")
    print(f"{Color.OKGREEN}✔ Initialized agent for role: {role}{Color.RESET}")
    return ag
