"""Mutation interpretation rules for NGS panel reports.

This module contains standardized interpretation rules for comprehensive
NGS genetic testing results. These rules are shared across multiple modules
to ensure consistency.
"""

# =============================================================================
# NGS Panel Interpretation Rules
# =============================================================================
# These rules define how to interpret Chinese NGS report terminology.
# Key principles:
# - '未检出', '（视为阴性）', '阴性' = NO pathogenic mutation found
# - Specific variant notation (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation
# - If a gene is NOT mentioned, it means NO pathogenic mutation (comprehensive panel)
# =============================================================================

NGS_INTERPRETATION_RULES = """INTERPRETATION RULES (CRITICAL):
• '未检出' (not detected) = NO pathogenic mutation found
• '（视为阴性）' (considered negative) = NO pathogenic mutation found
• '阴性' (negative) = negative result
• If a gene of interest is NOT mentioned in the report, it means NO pathogenic mutation (comprehensive panel)
• Genes with specific variants listed (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation detected
• NEVER say 'not tested' or 'not reported' - comprehensive NGS WAS done.
• Only say 'unknown' if NO mutation report is provided at all."""

NGS_INTERPRETATION_RULES_SHORT = """⚠️ COMPREHENSIVE NGS PANEL (~20,000 genes) - INTERPRETATION RULES:
• '未检出' (not detected) = NO pathogenic mutation found
• '（视为阴性）' (considered negative) = NO pathogenic mutation found
• '阴性' (negative) = negative result
• Genes with specific variants (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation
• If a gene of interest is NOT mentioned in the report, it means NO pathogenic mutation (comprehensive panel)
• NEVER say 'not tested' or 'not reported' - comprehensive NGS WAS done.
• Only say 'unknown' if NO mutation report is provided at all."""


def build_mutation_guidance(raw_text: str, include_rules: bool = True) -> str:
    """Build mutation guidance string with raw text and interpretation rules.

    Args:
        raw_text: The raw mutation report text
        include_rules: Whether to include interpretation rules (default True)

    Returns:
        Formatted mutation guidance string
    """
    guidance = "\n⚠️ COMPREHENSIVE NGS GENETIC TEST RESULTS:\n"
    guidance += "This is a ~20,000 gene NGS panel report. The raw text is:\n"
    guidance += f'"""{raw_text}"""\n\n'

    if include_rules:
        guidance += NGS_INTERPRETATION_RULES

    return guidance
