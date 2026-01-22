"""Central Host - LLM-powered orchestration layer for OMGs system.

This package contains:
- orchestrator: MDT discussion engine and main pipeline
- experts: Expert agent definitions and initialization
- decision: Final decision-making and output generation

Pipeline variants:
- process_omgs_multi_expert_query: Full multi-agent MDT with 5 experts
- process_chair_sa_k_query: Chair-SA(K) - Single agent with Knowledge only
- process_chair_sa_kep_query: Chair-SA(K+EP) - Single agent with Knowledge + Evidence Pack
- process_chair_sa_query: Chair-SA - Simplest mode for testing
- process_auto_query: Auto - Intelligent routing based on case complexity
"""

from .orchestrator import (
    process_omgs_multi_expert_query,
    run_mdt_discussion,
    process_chair_sa_k_query,
    process_chair_sa_kep_query,
    process_chair_sa_query,
    process_auto_query,
)
from .experts import ROLES, ROLE_PERMISSIONS, ROLE_PROMPTS, init_expert_agent
from .decision import generate_final_output, assistant_trial_suggestion, build_enhanced_case_for_trial

__all__ = [
    # Pipeline variants
    "process_omgs_multi_expert_query",
    "process_chair_sa_k_query",
    "process_chair_sa_kep_query",
    "process_chair_sa_query",
    "process_auto_query",
    # MDT engine
    "run_mdt_discussion",
    # Expert definitions
    "ROLES",
    "ROLE_PERMISSIONS",
    "ROLE_PROMPTS",
    "init_expert_agent",
    # Decision helpers
    "generate_final_output",
    "assistant_trial_suggestion",
    "build_enhanced_case_for_trial",
]
