"""Multi-provider LLM client initialization (Azure OpenAI, OpenAI, OpenRouter)."""

import os
from typing import Optional, Literal, Dict, Any
from clients import OpenAIWrapper


def _get_provider_config() -> Dict[str, Dict[str, Any]]:
    """
    Get provider configuration from config file or use defaults.
    
    Returns:
        Dictionary mapping provider names to their configuration
    """
    try:
        from core.config import get_paths_config
        config = get_paths_config()
        llm_config = config.get("llm", {})
        providers = llm_config.get("providers", {})
        
        # Merge with defaults to ensure all providers are available
        default_config = {
            "azure": {
                "api_key_env": "AZURE_OPENAI_API_KEY",
                "endpoint_env": "AZURE_OPENAI_ENDPOINT"
            },
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1"
            },
            "openrouter": {
                "api_key_env": "OPENROUTER_API_KEY",
                "base_url": "https://openrouter.ai/api/v1"
            }
        }
        
        # Update defaults with config file values
        for provider, config_dict in providers.items():
            if provider in default_config:
                default_config[provider].update(config_dict)
        
        return default_config
    except Exception:
        # Fallback to hardcoded defaults if config loading fails
        return {
            "azure": {
                "api_key_env": "AZURE_OPENAI_API_KEY",
                "endpoint_env": "AZURE_OPENAI_ENDPOINT"
            },
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1"
            },
            "openrouter": {
                "api_key_env": "OPENROUTER_API_KEY",
                "base_url": "https://openrouter.ai/api/v1"
            }
        }


def init_client(
    db_path: Optional[str] = None,
    provider: Literal["azure", "openai", "openrouter"] = "azure"
) -> OpenAIWrapper:
    """
    Initialize LLM client using aoai.OpenAIWrapper.
    Supports Azure OpenAI, OpenAI official API, and OpenRouter.
    If db_path is not provided, loads from config file.
    
    Parameters:
    - db_path: Optional SQLite database path for API logging
    - provider: Provider type: "azure" (default), "openai", or "openrouter"
    
    Returns:
    - OpenAIWrapper instance configured for the specified provider
    """
    if db_path is None:
        # Import here to avoid circular dependency
        from core.config import get_paths_config
        config = get_paths_config()
        db_path = config["output_dirs"]["api_trace_db"]
    
    provider_configs = _get_provider_config()
    
    if provider not in provider_configs:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(provider_configs.keys())}")
    
    config = provider_configs[provider]
    
    if provider == "azure":
        # Azure OpenAI requires both endpoint and API key
        endpoint = os.getenv(config.get("endpoint_env", "AZURE_OPENAI_ENDPOINT"))
        api_key = os.getenv(config.get("api_key_env", "AZURE_OPENAI_API_KEY"))
        
        if not endpoint or not api_key:
            raise RuntimeError(
                f"Missing {config.get('endpoint_env')} or {config.get('api_key_env')} "
                "environment variables for Azure OpenAI."
            )
        
        return OpenAIWrapper(
            api_key=api_key,
            base_url=endpoint,
            db_path=db_path,
            provider="azure"
        )
    
    elif provider == "openai":
        # OpenAI official API
        api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
        base_url = config.get("base_url", "https://api.openai.com/v1")
        
        if not api_key:
            raise RuntimeError(
                f"Missing {config.get('api_key_env')} environment variable for OpenAI."
            )
        
        return OpenAIWrapper(
            api_key=api_key,
            base_url=base_url,
            db_path=db_path,
            provider="openai"
        )
    
    elif provider == "openrouter":
        # OpenRouter
        api_key = os.getenv(config.get("api_key_env", "OPENROUTER_API_KEY"))
        base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        
        if not api_key:
            raise RuntimeError(
                f"Missing {config.get('api_key_env')} environment variable for OpenRouter."
            )
        
        return OpenAIWrapper(
            api_key=api_key,
            base_url=base_url,
            db_path=db_path,
            provider="openrouter"
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def init_client_from_config(
    model: str,
    db_path: Optional[str] = None
) -> OpenAIWrapper:
    """
    Initialize client based on model name, automatically detecting provider.
    
    Parameters:
    - model: Model name (e.g., "gpt-4" for Azure/OpenAI, "google/gemini-3-pro-preview" for OpenRouter)
    - db_path: Optional SQLite database path for API logging
    
    Returns:
    - OpenAIWrapper instance configured for the detected provider
    
    Detection Logic:
    - Models with "/" prefix (e.g., "google/", "anthropic/") → OpenRouter
    - Models starting with "gpt-" → Try Azure first (if env vars set), else OpenAI
    - Otherwise → Try providers in order: Azure, OpenAI, OpenRouter
    """
    # Check if model name suggests OpenRouter (has provider prefix like "google/", "anthropic/")
    if "/" in model and not model.startswith("gpt-"):
        return init_client(db_path=db_path, provider="openrouter")
    
    # For gpt-* models, try Azure first, then OpenAI
    if model.startswith("gpt-"):
        # Check if Azure is configured
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if azure_endpoint and azure_key:
            return init_client(db_path=db_path, provider="azure")
        else:
            # Fall back to OpenAI official API
            return init_client(db_path=db_path, provider="openai")
    
    # For other models, try providers in order
    # Try Azure first
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_endpoint and azure_key:
        return init_client(db_path=db_path, provider="azure")
    
    # Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return init_client(db_path=db_path, provider="openai")
    
    # Try OpenRouter as last resort
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return init_client(db_path=db_path, provider="openrouter")
    
    # If nothing is configured, default to Azure (will raise error if not configured)
    return init_client(db_path=db_path, provider="azure")
