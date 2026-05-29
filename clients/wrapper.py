# clients/wrapper.py
from openai import OpenAI
import os
import time
from datetime import datetime
from .logger import DBLogger
from utils.runtime_flags import api_trace_enabled, api_trace_raw_enabled


_OMGS_AZURE_GPT51_DEFAULTS = {
    "temperature": 1,
    "top_p": 1,
    "reasoning_effort": "none",
    "seed": 42,
}

_OMGS_OPENROUTER_DEFAULTS = {
    "temperature": 1,
    "top_p": 1,
    "seed": 42,
}


def _verbose_api_calls_enabled() -> bool:
    return os.getenv("OMGS_VERBOSE_API_CALLS", "").strip().lower() in {"1", "true", "yes", "on"}


def _format_trace_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 500:
        message = message[:500] + "..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _is_azure_gpt51_deployment(model: str) -> bool:
    raw_model = str(model or "").strip().lower()
    if not raw_model:
        return False
    normalized = "".join(ch for ch in raw_model if ch.isalnum())
    if "gpt51" in normalized:
        return True
    configured_aliases = [
        item.strip().lower()
        for item in str(os.getenv("OMGS_AZURE_GPT51_DEPLOYMENTS", "")).split(",")
        if item.strip()
    ]
    return raw_model in configured_aliases


def _apply_omgs_azure_gpt51_defaults(provider, model, kwargs):
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider == "azure" and _is_azure_gpt51_deployment(str(model or "")):
        merged = dict(kwargs)
        for key, value in _OMGS_AZURE_GPT51_DEFAULTS.items():
            merged.setdefault(key, value)
        return merged
    return kwargs


def _apply_omgs_openrouter_defaults(provider, model, kwargs):
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider != "openrouter":
        return kwargs

    merged = dict(kwargs)
    for key, value in _OMGS_OPENROUTER_DEFAULTS.items():
        merged.setdefault(key, value)

    return merged


class OpenAIWrapper:
    def __init__(self, api_key, base_url=None, db_path="data/logs/omgs_api_trace.db", provider="azure"):
        """
        Initialize wrapper around OpenAI / Azure OpenAI / OpenRouter client.
        
        Parameters:
        - api_key: API key for OpenAI, Azure OpenAI, or OpenRouter.
        - base_url: Optional. Required for Azure OpenAI such as:
                    https://<resource>.openai.azure.com/openai/v1/
                    For OpenRouter, defaults to https://openrouter.ai/api/v1
        - db_path: SQLite database path for request tracing and logging.
        - provider: Provider type, either "azure" (default), "openai", or "openrouter".
        """
        self.provider = provider
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,   # Can be None for OpenAI public endpoint
        )
        self.logger = DBLogger(db_path) if api_trace_enabled() else None
        self.log_raw_api_payloads = api_trace_raw_enabled()

    def chat_completion(self, *, model="gpt-5-mini", messages=None, extra_body=None, **kwargs):
        """
        A complete replacement of client.chat.completions.create()
        Logs request/response metadata to SQLite only when OMGS_API_TRACE_ENABLED=1.
        
        Parameters:
        - model: deployment name or model ID
        - messages: list of {"role": "...", "content": "..."}
        - extra_body: Optional. Extra body parameters (e.g., for OpenRouter reasoning)
        - kwargs: any other OpenAI ChatCompletion parameters
        
        Returns:
        - resp: raw API response
        """
        verbose_api_calls = _verbose_api_calls_enabled()
        if verbose_api_calls:
            print(f"Calling model: {model} (provider: {self.provider})")
        start = time.time()

        # Map deprecated max_tokens to max_completion_tokens for compatibility.
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        # Handle extra_body for OpenRouter (e.g., reasoning support)
        effective_extra_body = extra_body if self.provider == "openrouter" and extra_body else None
        if effective_extra_body:
            kwargs["extra_body"] = effective_extra_body

        kwargs = _apply_omgs_azure_gpt51_defaults(self.provider, model, kwargs)
        kwargs = _apply_omgs_openrouter_defaults(self.provider, model, kwargs)

        raw_request = None
        if self.logger and self.log_raw_api_payloads:
            raw_request = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            if effective_extra_body:
                raw_request["extra_body"] = effective_extra_body

        if verbose_api_calls:
            provider_name = "OpenRouter" if self.provider == "openrouter" else "OpenAI/Azure OpenAI"
            print(f"Sending request to {provider_name}...")

        # ---- Real API call ----
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            if self.logger:
                self.logger.log(
                    timestamp=str(datetime.now()),
                    provider=self.provider,
                    model=model,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                    seed=kwargs.get("seed"),
                    max_completion_tokens=kwargs.get("max_completion_tokens"),
                    reasoning_effort=kwargs.get("reasoning_effort"),
                    input_text=str(messages) if self.log_raw_api_payloads else None,
                    output_text=None,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    raw_request=raw_request,
                    raw_response=None,
                    latency_ms=latency_ms,
                    extra_body=effective_extra_body if self.log_raw_api_payloads else None,
                    reasoning_details=None,
                    raw_enabled=1 if self.log_raw_api_payloads else 0,
                    status="error",
                    error=_format_trace_error(exc),
                )
            raise

        # Compute latency
        latency_ms = (time.time() - start) * 1000

        # Extract output text
        output_text = resp.choices[0].message.content

        # Extract reasoning_details if present (OpenRouter reasoning support)
        reasoning_details = None
        if hasattr(resp.choices[0].message, "reasoning_details"):
            reasoning_details = getattr(resp.choices[0].message, "reasoning_details", None)

        # Extract token usage (OpenAI SDK v1 uses object attributes)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        else:
            input_tokens = output_tokens = total_tokens = None

        if self.logger:
            raw_response = resp.model_dump() if self.log_raw_api_payloads else None
            self.logger.log(
                timestamp=str(datetime.now()),
                provider=self.provider,
                model=model,
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                seed=kwargs.get("seed"),
                max_completion_tokens=kwargs.get("max_completion_tokens"),
                reasoning_effort=kwargs.get("reasoning_effort"),
                input_text=str(messages) if self.log_raw_api_payloads else None,
                output_text=output_text if self.log_raw_api_payloads else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                raw_request=raw_request,
                raw_response=raw_response,
                latency_ms=latency_ms,
                extra_body=effective_extra_body if self.log_raw_api_payloads else None,
                reasoning_details=reasoning_details if self.log_raw_api_payloads else None,
                raw_enabled=1 if self.log_raw_api_payloads else 0,
                status="ok",
            )

        # Return normal OpenAI response
        return resp
