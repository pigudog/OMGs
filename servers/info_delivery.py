"""Information Delivery Server - Builds role-specific case views."""

import json
from typing import Any, Dict


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


def build_role_specific_case_view(role: str, case_json: Dict[str, Any]) -> str:
    """Build role-specific case view from full case JSON."""
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
