"""Expert Agent Definitions - LLM-powered specialist agents for MDT."""

from typing import Any, Dict, List, Optional
import json
from core.agent import Agent
from core.config import get_mdt_prompts
from servers.context_assembly import safe_load_case_json, sanitize_case_for_decision, build_role_specific_case_view
from servers.report_selection import expert_select_reports
from utils.console_utils import Color
from utils.mutation_interpretation import NGS_INTERPRETATION_RULES_SHORT
from utils.mdt_runtime_protocol import build_runtime_protocol_digest

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
def get_role_prompts() -> Dict[str, str]:
    """Load editable role prompts from config/mdt_prompts.json."""
    prompts = get_mdt_prompts().get("role_prompts") or {}
    return {str(role): str(prompt).strip() for role, prompt in prompts.items()}


ROLE_PROMPTS: Dict[str, str] = get_role_prompts()


def get_role_prompt(role: str) -> str:
    """Return the configured prompt for one MDT role with a minimal fallback."""
    return get_role_prompts().get(role, "").strip() or (
        "Return up to 5 bullets. Each bullet <=20 words. "
        "Use only provided information; do not hallucinate."
    )


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
    case_json = sanitize_case_for_decision(safe_load_case_json(question))
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
        clinical += NGS_INTERPRETATION_RULES_SHORT + "\n\n"
        clinical += json.dumps(mut_for_role, ensure_ascii=False, indent=2) + "\n\n"

    if not clinical.strip():
        clinical = "# No clinical reports for this role.\n\n"

    role_prompt = get_role_prompt(role)

    visit_time_str = visit_time or "Unknown visit date"

    # Inject runtime protocol for evidence format and role behavior enforcement
    runtime_protocol_digest = build_runtime_protocol_digest(role)

    instruction = f"""
{runtime_protocol_digest}

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
   - use the exact guideline tag from the digest: [@guideline:doc_id | Page xx] or [@guideline:doc_id | Pages xx-yy]
   - PubMed format: [@pubmed | PMID]
4b) At least ONE bullet must be evidence-based and include a guideline tag from the digest or [@pubmed | PMID].
5) Any claim about labs/imaging/pathology/molecular MUST include evidence tag:
   - format: [@actual_report_id | LAB/Genomics/MR/CT/Pathology] using actual report_id from report data
   - Examples: [@LAB20251020TM | LAB], [@OH20251003 | Genomics], [@CT20250922 | CT], [@PETCT20251021 | CT], [@PX20251003 | Pathology]
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

    ag = Agent(
        instruction=instruction,
        role=role,
        model_info=model,
        client=client,
        max_tokens=100000,
        max_prompt_tokens=100000,
    )
    ag.inject_assistant("System ready for MDT discussion.")
    print(f"{Color.OKGREEN}✔ Initialized agent for role: {role}{Color.RESET}")
    return ag
