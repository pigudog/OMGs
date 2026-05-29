"""Orchestrator - LLM-powered orchestration layer for OMGs system.

This package contains:
- deliberation: MDT discussion engine and main pipeline
- experts: Expert agent definitions and initialization
- decision: Final decision-making and output generation

Pipeline variants:
- process_omgs_multi_expert_query: Full multi-agent MDT with 5 experts
- process_chair_e_query: CHAIR-E - Single agent evidence-augmented (RAG)
- process_chair_d_query: CHAIR-D - Single agent dossier-augmented (RAG + evidence pack)
- process_chair_r_query: CHAIR-R - Simplest mode records-only (for testing)
- process_auto_query: optional exploratory routing for non-evaluation use
"""

from .deliberation import (
    process_omgs_multi_expert_query,
    run_mdt_discussion,
    process_chair_e_query,
    process_chair_d_query,
    process_chair_r_query,
    process_auto_query,
    normalize_ablation_output_labels,
)
from .experts import ROLES, ROLE_PERMISSIONS, ROLE_PROMPTS, get_role_prompt, get_role_prompts, init_expert_agent
from .decision import generate_final_output

__all__ = [
    # Pipeline variants
    "process_omgs_multi_expert_query",
    "process_chair_e_query",
    "process_chair_d_query",
    "process_chair_r_query",
    "process_auto_query",
    "normalize_ablation_output_labels",
    # MDT engine
    "run_mdt_discussion",
    # Expert definitions
    "ROLES",
    "ROLE_PERMISSIONS",
    "ROLE_PROMPTS",
    "get_role_prompt",
    "get_role_prompts",
    "init_expert_agent",
    # Decision helpers
    "generate_final_output",
]
