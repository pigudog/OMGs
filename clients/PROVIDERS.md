# Multi-Provider LLM Support Guide

## Overview

OMGs supports multiple LLM providers, allowing you to use Azure OpenAI, OpenAI official API, or OpenRouter. You can specify the provider explicitly or let the system auto-detect based on the model name.

## Supported Providers

1. **azure**: Azure OpenAI (default, backward compatible)
2. **openai**: OpenAI official API
3. **openrouter**: OpenRouter (supports multiple models like Gemini, Claude, etc.)
4. **auto**: Auto-detect provider based on model name (default)

## Environment Variables

### Azure OpenAI
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-azure-key"
```

### OpenAI Official API
```bash
export OPENAI_API_KEY="your-openai-key"
```

### OpenRouter
```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Usage

### Method 1: Command Line (Recommended)

Use the `--provider` argument in `main.py`:

```bash
# Use Azure OpenAI (explicit)
python main.py --input_path data.jsonl --model gpt-4 --provider azure

# Use OpenAI official API
python main.py --input_path data.jsonl --model gpt-4 --provider openai

# Use OpenRouter
python main.py --input_path data.jsonl --model google/gemini-3-pro-preview --provider openrouter

# Auto-detect (default)
python main.py --input_path data.jsonl --model gpt-4 --provider auto
```

### Method 2: Programmatic API

#### Explicit Provider Selection

```python
from core.client import init_client

# Azure OpenAI
client = init_client(provider="azure")

# OpenAI official API
client = init_client(provider="openai")

# OpenRouter
client = init_client(provider="openrouter")
```

#### Auto-Detection Based on Model Name

```python
from core.client import init_client_from_config

# Auto-detect as Azure (if Azure env vars are set) or OpenAI
client = init_client_from_config(model="gpt-4")

# Auto-detect as OpenRouter (model has provider prefix)
client = init_client_from_config(model="google/gemini-3-pro-preview")
```

#### Using setup_model() with Provider

```python
from core.config import setup_model

# Auto-detect
model, client = setup_model("gpt-4")

# Explicit provider
model, client = setup_model("gpt-4", provider="openai")
```

### Method 3: Direct Use of OpenAIWrapper

```python
from clients import OpenAIWrapper
import os

# Azure
client = OpenAIWrapper(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    provider="azure"
)

# OpenAI
client = OpenAIWrapper(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
    provider="openai"
)

# OpenRouter
client = OpenAIWrapper(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    provider="openrouter"
)
```

## Auto-Detection Logic

The system automatically detects the provider based on model name:

1. **Models with provider prefix** (e.g., `google/`, `anthropic/`):
   - → Uses **OpenRouter**

2. **Models starting with `gpt-`**:
   - If Azure environment variables are set → Uses **Azure**
   - Otherwise → Uses **OpenAI**

3. **Other models**:
   - Tries providers in order: Azure → OpenAI → OpenRouter
   - Uses the first one with configured environment variables

## Reasoning Feature (OpenRouter)

OpenRouter supports reasoning mode for certain models (e.g., Gemini 3 Pro):

### In Agent

```python
from core import Agent, init_client

client = init_client(provider="openrouter")
agent = Agent(
    instruction="You are a helpful assistant.",
    role="assistant",
    model_info="google/gemini-3-pro-preview",
    client=client,
    enable_reasoning=True  # Enable reasoning
)

response = agent.chat("How many r's are in the word 'strawberry'?")
# Agent automatically handles reasoning_details and preserves it in subsequent conversations
```

### Direct API Call

```python
from core.client import init_client

client = init_client(provider="openrouter")

# First call with reasoning enabled
response = client.chat_completion(
    model="google/gemini-3-pro-preview",
    messages=[
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
    ],
    extra_body={"reasoning": {"enabled": True}}
)

# Extract assistant message and reasoning_details
assistant_msg = response.choices[0].message
reasoning_details = getattr(assistant_msg, "reasoning_details", None)

# Preserve reasoning_details in subsequent calls
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": assistant_msg.content,
        "reasoning_details": reasoning_details  # Preserve reasoning context
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second call, continue reasoning
response2 = client.chat_completion(
    model="google/gemini-3-pro-preview",
    messages=messages,
    extra_body={"reasoning": {"enabled": True}}
)
```

## Configuration

Provider configuration is stored in `config/paths.json`:

```json
{
  "llm": {
    "default_provider": "azure",
    "providers": {
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
  }
}
```

## Logging

All API calls (including `extra_body` and `reasoning_details`) are automatically logged to SQLite database (`api_trace.db`). The database table automatically upgrades to support new fields, no manual migration needed.

## Backward Compatibility

- `--provider` defaults to `'auto'`, maintaining existing behavior
- `init_client()` defaults to `provider="azure"`
- All existing code requires no changes
- Existing environment variable configuration continues to work

## Model Name Formats

| Provider | Model Format Examples |
|----------|----------------------|
| **Azure** | `gpt-4`, `gpt-35-turbo`, `gpt-4o`, `gpt-5.1` |
| **OpenAI** | `gpt-4`, `gpt-3.5-turbo`, `gpt-4o`, `gpt-4o-mini` |
| **OpenRouter** | `google/gemini-3-pro-preview`, `anthropic/claude-3-opus`, `meta-llama/llama-3.2-3b-instruct:free` |

## Quick Test

Test all configured providers or a specific provider:

```bash
# Test all providers (default)
python -m clients.test_connection

# Test only Azure
python -m clients.test_connection --provider azure

# Test only OpenAI
python -m clients.test_connection --provider openai

# Test only OpenRouter
python -m clients.test_connection --provider openrouter

# Test all (explicit)
python -m clients.test_connection --provider all
```

This will:
- Test Azure OpenAI connection (if `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` are set)
- Test OpenAI connection (if `OPENAI_API_KEY` is set)
- Test OpenRouter connection (if `OPENROUTER_API_KEY` is set)
- Test auto-detection functionality (when testing all)
- Try multiple common model names for each provider
- Display a summary of test results

**Example output:**
```
============================================================
OMGs LLM Provider Connection Test
============================================================

Testing Azure OpenAI...
[INFO] Loaded paths config from /Users/pigudogzyy/Documents/PythonProject/OMGs/config/paths.json
Calling model: gpt-4 (provider: azure)
Sending request to OpenAI/Azure OpenAI...
Calling model: gpt-35-turbo (provider: azure)
Sending request to OpenAI/Azure OpenAI...
Calling model: gpt-4o (provider: azure)
Sending request to OpenAI/Azure OpenAI...
Calling model: gpt-5.1 (provider: azure)
Sending request to OpenAI/Azure OpenAI...
✅ Azure OpenAI: Azure connection successful.

Testing OpenAI...
Calling model: gpt-4 (provider: openai)
Sending request to OpenAI/Azure OpenAI...
✅ OpenAI: The connection to OpenAI has been successfully established.

Testing OpenRouter...
Calling model: google/gemini-2.0-flash-exp:free (provider: openrouter)
Sending request to OpenRouter...
Calling model: google/gemini-flash-1.5-8b:free (provider: openrouter)
Sending request to OpenRouter...
Calling model: meta-llama/llama-3.2-3b-instruct:free (provider: openrouter)
Sending request to OpenRouter...
✅ OpenRouter: OpenRouter connection successful.

Testing auto-detection...
✅ Auto-detection: Working correctly (tested: Azure, OpenAI, OpenRouter)

============================================================
Summary:
============================================================
✅ PASS - Azure OpenAI
✅ PASS - OpenAI
✅ PASS - OpenRouter
✅ PASS - Auto-detection

🎉 All configured tests passed!
```

## Example: Compare Outputs from Multiple LLMs

```python
from core.client import init_client_from_config

models = [
    ("gpt-4", "azure"),           # Azure OpenAI
    ("gpt-4", "openai"),          # OpenAI official
    ("google/gemini-3-pro-preview", "openrouter"),  # OpenRouter
]

results = {}
for model, provider in models:
    if provider == "auto":
        client = init_client_from_config(model=model)
    else:
        from core.client import init_client
        client = init_client(provider=provider)
    
    response = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Your question"}]
    )
    results[f"{model} ({provider})"] = response.choices[0].message.content

# Compare results
for model_provider, content in results.items():
    print(f"{model_provider}: {content}")
```

## Notes

1. **Provider Priority**: When auto-detecting for `gpt-*` models, Azure is preferred if both Azure and OpenAI are configured.

2. **Reasoning Support**: Currently only some models support reasoning (e.g., Gemini 3 Pro via OpenRouter). Requires passing `extra_body={"reasoning": {"enabled": True}}` in calls.

3. **Error Handling**: If a provider fails to initialize (missing env vars), the system will raise a clear error message indicating which environment variables are needed.

## Future Extensions

The provider system is designed to be extensible. Future providers can be easily added by:
1. Adding configuration to `config/paths.json`
2. Extending `_get_provider_config()` in `core/client.py`
3. Adding provider-specific logic if needed

Potential future providers:
- Anthropic (Claude API)
- Google (Gemini API direct)
- Other custom base_url APIs
