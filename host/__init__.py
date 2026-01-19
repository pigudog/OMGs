"""Central Host - LLM-powered orchestration layer for OMGs system.

This package contains:
- orchestrator: MDT discussion engine and main pipeline
- experts: Expert agent definitions and initialization
- decision: Final decision-making and output generation
"""

from .orchestrator import process_omgs_multi_expert_query, run_mdt_discussion
from .experts import ROLES, ROLE_PERMISSIONS, ROLE_PROMPTS, init_expert_agent
from .decision import generate_final_output, assistant_trial_suggestion, build_enhanced_case_for_trial

__all__ = [
    "process_omgs_multi_expert_query",
    "run_mdt_discussion",
    "ROLES",
    "ROLE_PERMISSIONS",
    "ROLE_PROMPTS",
    "init_expert_agent",
    "generate_final_output",
    "assistant_trial_suggestion",
    "build_enhanced_case_for_trial",
]
