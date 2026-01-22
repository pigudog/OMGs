"""SKILL loader for OMGs runtime integration.

This module provides functions to load and inject SKILL content into Expert Agents
at runtime, enabling evidence format enforcement, role behavior constraints, and
OMGs system awareness with minimal token overhead (~75 tokens per agent).

Usage:
    from utils.skill_loader import build_skill_digest
    
    digest = build_skill_digest("chair")  # Returns ~75 token digest
"""
from pathlib import Path
from typing import Optional, Dict, Any
import re

_SKILL_CACHE: Optional[Dict[str, Any]] = None
SKILL_PATH = Path(__file__).parent.parent / "skills/omgs/SKILL.md"


def load_skill() -> Dict[str, Any]:
    """
    Load and parse SKILL.md, with caching.
    
    Returns:
        dict with keys:
            - name: skill name
            - loaded: whether SKILL.md was found and loaded
            - evidence_tags: list of evidence tag examples from SKILL
    """
    global _SKILL_CACHE
    if _SKILL_CACHE is not None:
        return _SKILL_CACHE
    
    if not SKILL_PATH.exists():
        _SKILL_CACHE = {"name": "omgs", "loaded": False}
        return _SKILL_CACHE
    
    content = SKILL_PATH.read_text(encoding="utf-8")
    
    # Extract evidence tags from content
    tags = re.findall(r'\[@\w+:[^\]]+\]', content)
    
    _SKILL_CACHE = {
        "name": "omgs",
        "loaded": True,
        "evidence_tags": list(set(tags))[:4],  # Keep unique tag examples
        "path": str(SKILL_PATH),
    }
    return _SKILL_CACHE


def build_skill_digest(role: str) -> str:
    """
    Build concise SKILL digest for injection into Expert Agent instruction.
    
    Args:
        role: Expert role name (chair, oncologist, radiologist, pathologist, nuclear)
    
    Returns:
        String digest (~75 tokens) to inject at start of agent instruction.
        Returns empty string if SKILL not loaded.
    """
    skill = load_skill()
    if not skill.get("loaded"):
        return ""
    
    # Role-specific constraints (aligned with ROLE_PROMPTS in host/experts.py)
    role_constraints = {
        "chair": "Integrate all specialties; no specific drug names; highlight missing info.",
        "oncologist": "Treatment categories only; no specific drug names.",
        "radiologist": "Imaging findings only; no drug names or treatment recommendations.",
        "pathologist": "Histology/molecular only; no drug names or prognosis/treatment advice.",
        "nuclear": "Metabolic findings only; no drug names or treatment recommendations.",
    }
    
    constraint = role_constraints.get(role, "Follow role-specific guidelines.")
    
    return f"""# OMGs SKILL PROTOCOL
System: OMGs (Ovarian-cancer Multidisciplinary aGent System)
Evidence tags REQUIRED: [@guideline:doc_id | Page xx], [@pubmed | PMID], [@trial | id], [@actual_report_id | LAB/Genomics/MR/CT]
Use actual report_id from report data (e.g., [@20220407|17300673 | LAB], [@OH2203828|2022-04-18 | Genomics])
Role constraint: {constraint}"""


def get_skill_info() -> Dict[str, Any]:
    """
    Get SKILL metadata for logging/tracing purposes.
    
    Returns:
        dict with skill name, loaded status, and path
    """
    skill = load_skill()
    return {
        "name": skill.get("name"),
        "loaded": skill.get("loaded", False),
        "path": skill.get("path", ""),
    }
