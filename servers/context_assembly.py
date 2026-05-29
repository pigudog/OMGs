"""Information Delivery Server - Builds role-specific case views."""

import json
from typing import Any, Dict

DECISION_CASE_KEYS = {
    "CASE_CORE",
    "TIMELINE",
    "MED_ONC",
    "RADIOLOGY",
    "PATHOLOGY",
    "NUC_MED",
    "LAB_TRENDS",
}


def safe_load_case_json(question) -> dict:
    """Safely load case JSON from various input formats."""
    if isinstance(question, dict):
        return question
    if isinstance(question, list):
        return {"_list": question}
    try:
        return json.loads(str(question))
    except Exception:
        return {}


def sanitize_case_for_decision(case_json: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only structured case sections used for clinical decision-making."""
    if not isinstance(case_json, dict):
        return {}
    nested_question = case_json.get("question")
    if isinstance(nested_question, dict) and not any(key in case_json for key in DECISION_CASE_KEYS):
        return sanitize_case_for_decision(nested_question)
    if not any(key in case_json for key in DECISION_CASE_KEYS):
        return dict(case_json)
    return {key: case_json[key] for key in DECISION_CASE_KEYS if key in case_json}


def _current_visit_context(core: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact current-visit context shared across MDT roles."""
    visit_context = core.get("VISIT_CONTEXT")
    result: Dict[str, Any] = {}
    if core.get("VISIT_DATE"):
        result["VISIT_DATE"] = core.get("VISIT_DATE")
    if visit_context:
        result["VISIT_CONTEXT"] = visit_context
    return result


def build_role_specific_case_view(role: str, case_json: Dict[str, Any]) -> str:
    """Build role-specific case view from full case JSON."""
    case_json = sanitize_case_for_decision(case_json)
    core = case_json.get("CASE_CORE", {})
    timeline = case_json.get("TIMELINE", {})
    med_onc = case_json.get("MED_ONC", {})
    radiology = case_json.get("RADIOLOGY", {})
    pathology = case_json.get("PATHOLOGY", {})
    nuc = case_json.get("NUC_MED", {})
    labs = case_json.get("LAB_TRENDS", {})
    current_visit_context = _current_visit_context(core)

    if role == "chair":
        return json.dumps(case_json, ensure_ascii=False, indent=2)

    if role == "oncologist":
        return json.dumps({
            "CURRENT_VISIT_CONTEXT": current_visit_context,
            "DIAGNOSIS": core.get("DIAGNOSIS"),
            "LINE_OF_THERAPY": core.get("LINE_OF_THERAPY"),
            "MAINTENANCE": core.get("MAINTENANCE"),
            "RELAPSE_DATE": core.get("RELAPSE_DATE"),
            "BIOMARKERS": core.get("BIOMARKERS"),
            "GENETICS": med_onc.get("genetic_testing"),
            "MEDICAL_HISTORY": core.get("MEDICAL_HISTORY"),
            "CURRENT_STATUS": core.get("CURRENT_STATUS"),
            "PATHOLOGY": pathology.get("specimens", []),
            "LAB_TRENDS": labs,
            "TOXICITIES": med_onc.get("TOXICITIES", []),
            "CLINICAL_TRIALS": med_onc.get("CLINICAL_TRIALS", []),
        }, ensure_ascii=False, indent=2)

    if role == "radiologist":
        return json.dumps({
            "CURRENT_VISIT_CONTEXT": current_visit_context,
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
            "CURRENT_VISIT_CONTEXT": current_visit_context,
            "HISTOLOGY_AND_IHC": pathology.get("specimens", []),
            "MOLECULAR": [
                core.get("BIOMARKERS", {}),
                med_onc.get("genetic_testing", {})
            ]
        }, ensure_ascii=False, indent=2)

    if role == "nuclear":
        return json.dumps({
            "CURRENT_VISIT_CONTEXT": current_visit_context,
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
