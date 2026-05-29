"""Build compact MDT runtime protocol digests from mdt_prompts.json."""

from typing import Any, Dict

from core.config import get_mdt_prompts


def _runtime_protocol() -> Dict[str, Any]:
    return dict(get_mdt_prompts().get("runtime_protocol") or {})


def build_runtime_protocol_digest(role: str) -> str:
    """Return the runtime protocol snippet injected into an agent prompt."""

    protocol = _runtime_protocol()
    if not protocol:
        return ""

    role_config = (protocol.get("ROLE_CONSTRAINTS") or {}).get(role, {})
    constraint = role_config.get("description", "Follow role-specific guidelines.")
    evidence_tags = protocol.get("EVIDENCE_TAGS") or {}
    tags = [
        evidence_tags.get("report_format", "[@actual_report_id | LAB/Genomics/MR/CT/Pathology]"),
        evidence_tags.get("guideline", "[@guideline:doc_id | Page xx]"),
        evidence_tags.get("nccn_rule", "[@guideline:nccn | rule_id]"),
        evidence_tags.get("pubmed", "[@pubmed | PMID]"),
        evidence_tags.get("fda", "[@fda | source_id:section]"),
        evidence_tags.get("conference", "[@conference | abstract_id]"),
        evidence_tags.get("trial", "[@trial | id]"),
    ]

    template = protocol.get("RUNTIME_PROTOCOL_DIGEST_TEMPLATE", "")
    if not template:
        return ""

    return template.format(
        system_description=protocol.get("SYSTEM_DESCRIPTION", "OMGs system"),
        evidence_tags=", ".join(tags),
        role_constraint=constraint,
    )


def get_runtime_protocol_info() -> Dict[str, Any]:
    protocol = _runtime_protocol()
    return {
        "name": "omgs_mdt_runtime_protocol",
        "loaded": bool(protocol),
        "version": protocol.get("RUNTIME_PROTOCOL_VERSION", "unknown"),
    }
