"""Shared regex patterns for evidence tags.

This module contains all regular expression patterns used for matching
evidence tags across the codebase. Centralizing these patterns ensures
consistency and makes updates easier.
"""

import re
from typing import List, Pattern

# =============================================================================
# Evidence tag patterns
# =============================================================================
# Evaluated outputs cite source-typed tags such as:
# - [@guideline:doc_id | Page xx]
# - [@guideline:nccn | rule_id]
# - [@pubmed | PMID]
# - [@fda | source_id]
# - [@conference | source_id]
# - [@trial | trial_id]
# - [@report_id | LAB/Genomics/MR/CT/Pathology]
# Additional parser branches below are retained only for backwards-compatible
# parsing of older local artifacts; current prompts request the formats above.
# =============================================================================

# Main evidence tag regex.
EVIDENCE_TAG_RE: Pattern[str] = re.compile(
    r"\[@guideline:[a-zA-Z0-9_\-]+\s*\|\s*Pages?\s+[^\]]+\]|"
    r"\[@guideline:[a-zA-Z0-9_\-]+\|[^\]]+\]|"
    r"\[@guideline:nccn\s*\|\s*[^\]]+\]|"
    r"\[@pubmed\s*\|\s*\d+\]|"
    r"\[@pubmed:\d+\]|"
    r"\[@fda\s*\|\s*[^\]]+\]|"
    r"\[@conference\s*\|\s*[^\]]+\]|"
    r"\[@trial\s*\|\s*[^\]]+\]|"
    r"\[@trial:[^\]]+\]|"
    r"\[@nccn\s*\|\s*[a-zA-Z0-9_\-]+\]|"
    r"\[@nccn:[a-zA-Z0-9_\-]+\]|"
    r"\[@[a-zA-Z0-9_\-|]+\s+\|\s+(?:LAB|Genomics|MR|CT|Imaging|Pathology|CASE)\s*\]|"
    r"\[@[a-zA-Z0-9_\-|]+\|[^\]]+\]",
    re.IGNORECASE
)

# Individual patterns for more granular matching
GUIDELINE_NEW_RE: Pattern[str] = re.compile(r"\[@guideline:[a-zA-Z0-9_\-]+\s*\|\s*Pages?\s+[^\]]+\]", re.IGNORECASE)
GUIDELINE_COMPACT_RE: Pattern[str] = re.compile(r"\[@guideline:[a-zA-Z0-9_\-]+\|[^\]]+\]", re.IGNORECASE)
GUIDELINE_NCCN_RE: Pattern[str] = re.compile(r"\[@guideline:nccn\s*\|\s*[^\]]+\]", re.IGNORECASE)
PUBMED_NEW_RE: Pattern[str] = re.compile(r"\[@pubmed\s*\|\s*\d+\]", re.IGNORECASE)
PUBMED_COMPACT_RE: Pattern[str] = re.compile(r"\[@pubmed:\d+\]", re.IGNORECASE)
FDA_NEW_RE: Pattern[str] = re.compile(r"\[@fda\s*\|\s*[^\]]+\]", re.IGNORECASE)
CONFERENCE_NEW_RE: Pattern[str] = re.compile(r"\[@conference\s*\|\s*[^\]]+\]", re.IGNORECASE)
TRIAL_NEW_RE: Pattern[str] = re.compile(r"\[@trial\s*\|\s*[^\]]+\]", re.IGNORECASE)
TRIAL_COMPACT_RE: Pattern[str] = re.compile(r"\[@trial:[^\]]+\]", re.IGNORECASE)
NCCN_NEW_RE: Pattern[str] = re.compile(r"\[@nccn\s*\|\s*[a-zA-Z0-9_\-]+\]", re.IGNORECASE)
NCCN_COMPACT_RE: Pattern[str] = re.compile(r"\[@nccn:[a-zA-Z0-9_\-]+\]", re.IGNORECASE)
REPORT_NEW_RE: Pattern[str] = re.compile(
    r"\[@[a-zA-Z0-9_\-|]+\s+\|\s+(?:LAB|Genomics|MR|CT|Imaging|Pathology|CASE)\s*\]", re.IGNORECASE
)
REPORT_COMPACT_RE: Pattern[str] = re.compile(r"\[@[a-zA-Z0-9_\-|]+\|[^\]]+\]")

# Keywords that indicate evidence-based claims
EVIDENCE_CUES: List[str] = [
    "guideline",
    "evidence",
    "trial",
    "nccn",
    "esmo",
    "parp",
    "maintenance",
    "platinum-sensitive",
    "platinum resistant",
    "platinum-resistant",
    "bevacizumab",
    "immunotherapy",
    "study",
    "meta-analysis",
    "randomized",
]


def extract_reference_tags(text: str) -> List[str]:
    """Extract all evidence reference tags from text.

    Args:
        text: Text containing evidence tags

    Returns:
        List of reference tags found (in order of appearance)
    """
    # Normalize text: split combined tags like [@tag1; @tag2] into [@tag1] [@tag2]
    normalized_text = re.sub(r';\s*@', '] [@', text)

    all_tags: List[str] = []

    # Guideline tags
    guideline_new = GUIDELINE_NEW_RE.findall(normalized_text)
    guideline_compact = GUIDELINE_COMPACT_RE.findall(normalized_text)
    guideline_tags = list(dict.fromkeys(guideline_new + guideline_compact))

    # PubMed tags
    pubmed_new = PUBMED_NEW_RE.findall(normalized_text)
    pubmed_compact = PUBMED_COMPACT_RE.findall(normalized_text)
    pubmed_tags = list(dict.fromkeys(pubmed_new + pubmed_compact))

    # External evidence supplement tags
    fda_tags = list(dict.fromkeys(FDA_NEW_RE.findall(normalized_text)))
    conference_tags = list(dict.fromkeys(CONFERENCE_NEW_RE.findall(normalized_text)))

    # Trial tags
    trial_new = TRIAL_NEW_RE.findall(normalized_text)
    trial_compact = TRIAL_COMPACT_RE.findall(normalized_text)
    trial_tags = list(dict.fromkeys(trial_new + trial_compact))

    # NCCN tags.
    guideline_nccn = GUIDELINE_NCCN_RE.findall(normalized_text)
    nccn_new = NCCN_NEW_RE.findall(normalized_text)
    nccn_compact = NCCN_COMPACT_RE.findall(normalized_text)
    nccn_tags = list(dict.fromkeys(guideline_nccn + nccn_new + nccn_compact))

    # Patient-report tags.
    report_new = REPORT_NEW_RE.findall(normalized_text)
    report_compact = REPORT_COMPACT_RE.findall(normalized_text)

    # Filter out source-typed tags from compact patient-report parsing.
    filtered_report_compact = []
    for tag in report_compact:
        tag_lower = tag.lower()
        if (
            not tag_lower.startswith("[@guideline:")
            and not tag_lower.startswith("[@pubmed")
            and not tag_lower.startswith("[@fda")
            and not tag_lower.startswith("[@conference")
            and not tag_lower.startswith("[@trial")
            and not tag_lower.startswith("[@nccn")
        ):
            filtered_report_compact.append(tag)

    report_tags = list(dict.fromkeys(report_new + filtered_report_compact))

    # Combine all tags preserving order
    all_tags = guideline_tags + pubmed_tags + fda_tags + conference_tags + trial_tags + nccn_tags + report_tags

    return all_tags
