"""Core infrastructure for OMGs system.

This package contains:
- Agent: Stateful LLM agent wrapper
- Client: Azure OpenAI client initialization
- Config: Configuration loading utilities
"""

from .agent import Agent
from .client import init_client
from .config import (
    load_paths_config,
    get_paths_config,
    load_mdt_prompts,
    get_mdt_prompts,
    load_data,
    create_question,
    setup_model,
)

__all__ = [
    "Agent",
    "init_client",
    "load_paths_config",
    "get_paths_config",
    "load_mdt_prompts",
    "get_mdt_prompts",
    "load_data",
    "create_question",
    "setup_model",
]
