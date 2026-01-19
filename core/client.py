"""Azure OpenAI client initialization."""

import os
from typing import Optional
from aoai import OpenAIWrapper


def init_client(db_path: Optional[str] = None) -> OpenAIWrapper:
    """
    Initialize Azure OpenAI client using aoai.OpenAIWrapper.
    If db_path is not provided, loads from config file.
    """
    if db_path is None:
        # Import here to avoid circular dependency
        from core.config import get_paths_config
        config = get_paths_config()
        db_path = config["output_dirs"]["api_trace_db"]
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY.")

    return OpenAIWrapper(api_key=api_key, base_url=endpoint, db_path=db_path)
