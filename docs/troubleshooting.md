# Troubleshooting Guide

This guide covers error handling, common issues, and debugging techniques for OMGs.

## Error Handling System

OMGs implements comprehensive error handling to ensure the pipeline continues even when individual components fail:

### Error Handling Layers

1. **Agent Level** (`core/agent.py`): Catches API errors and raises `AgentError` with context
2. **MDT Discussion Level** (`host/orchestrator.py`): Handles expert failures with fallback responses
3. **Pipeline Level** (`host/orchestrator.py`): Handles RAG, report selection, and initialization failures
4. **RAG Level** (`servers/evidence_search.py`): Prevents infinite retries with timeout and request patching
5. **Retry Layer** (`utils/error_handling.py`): Automatic retry with exponential backoff for rate limit errors (429)

### Automatic Retry for Rate Limits (429 Errors)

The system automatically retries API calls that fail with 429 (rate limit) errors:
- **Retry Strategy**: Up to 2-3 retries (depending on operation) with exponential backoff
- **Backoff Timing**: 2^attempt seconds + random jitter (e.g., 2s, 4s, 8s)
- **Operations Covered**:
  - Assistant summary (initial and round summaries)
  - Memory updates
  - Clinical trial matching
- **Fallback**: If all retries fail, the system uses a sensible fallback response and continues the pipeline

### Fallback Strategies

| Failure Point | Fallback Behavior |
|---------------|-------------------|
| Expert initial opinion | Uses error placeholder, continues with other experts |
| Assistant summary | Uses JSON of initial opinions as merged context |
| Turn speaking | Skips that expert for the turn, continues discussion |
| Final plan | Uses error placeholder, included in final output |
| RAG retrieval | Returns empty results, continues without RAG evidence |
| RAG summarization | Uses first 3 RAG results as simple digest |
| Chair final output | Generates simplified output from expert plans |
| Expert initialization | Skips failed roles, uses first available as chair if needed |
| Missing `files/` directory | Returns empty lists for reports, continues with available data |
| Missing report JSONL files | Returns empty lists, prints warning, pipeline continues |
| Missing `rag_store/` directory | RAG initialization fails gracefully, continues without RAG |
| Missing `rag_store/reference_cache/` | Reference cache works in-memory only, no disk persistence |
| ChromaDB directory creation failure | Raises clear error message, RAG retrieval skipped |

### Network Issues with HuggingFace

If you see `Connection reset by peer` errors when downloading models:
- The system will timeout after 10 seconds and skip RAG retrieval
- Requests are patched to disable retries, preventing infinite background retries
- The pipeline continues with empty RAG results
- Check your network connection or pre-download the model locally

### Error Logging

All errors are logged to the trace system. Check error summary:
```python
from servers.trace import TraceLogger

trace = TraceLogger(enabled=True)
# ... run pipeline ...
error_summary = trace.get_error_summary()
print(f"Total errors: {error_summary['total_errors']}")
print(f"Errors by role: {error_summary['errors_by_role']}")
print(f"Errors by stage: {error_summary['errors_by_stage']}")
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: Missing AZURE_OPENAI_ENDPOINT` | Azure environment variables not set | Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`, or use `--provider openai`/`openrouter` |
| `RuntimeError: Missing OPENAI_API_KEY` | OpenAI environment variable not set | Set `OPENAI_API_KEY`, or use `--provider azure`/`openrouter` |
| `RuntimeError: Missing OPENROUTER_API_KEY` | OpenRouter environment variable not set | Set `OPENROUTER_API_KEY`, or use `--provider azure`/`openai` |
| `FileNotFoundError: files/lab_reports_summary.jsonl` | Data files missing | **System handles gracefully**: Returns empty lists, prints warning, continues. To fix: Create `files/` directory and add required JSONL files. |
| `[WARNING] Failed to load lab reports` | Report file missing or unreadable | **System handles gracefully**: Pipeline continues with empty report data. Check file permissions and paths in `config/paths.json`. |
| `Failed to load RAG index` | RAG index not built | Run `pdf_to_rag.py build` and `index` commands. System continues without RAG if index unavailable. |
| `RAG initialization failed` | Network issues or model not cached | Check network, or pre-download model. Pipeline continues without RAG. |
| `Failed to create RAG index directory` | Permission denied or disk full | Check directory permissions. System will skip RAG if directory cannot be created. |
| `[WARNING] Failed to create reference cache directory` | `rag_store/reference_cache/` cannot be created | **System handles gracefully**: Cache works in-memory only. Check parent directory permissions. |
| `AgentError: ... failed in chat` | API call failed | Check provider credentials and quota. Agent uses fallback response. |
| `Error code: 429` (Rate limit) | Model temporarily rate-limited upstream | **System automatically retries** with exponential backoff (2-3 attempts). If all retries fail, uses fallback. For persistent rate limits: wait and retry, use a different model, or add your own API key to OpenRouter for higher limits. |
| `Azure API error` | Invalid deployment name | Verify `--model` matches Azure deployment, or try `--provider openai`/`openrouter` |
| `OpenAI API error` | Invalid model name or API key | Verify `--model` is valid for OpenAI, check `OPENAI_API_KEY` |
| `OpenRouter API error` | Invalid model name or API key | Verify `--model` format (e.g., `google/gemini-*`), check `OPENROUTER_API_KEY` |
| `Provider initialization failed` | Missing environment variables for selected provider | Set required environment variables for the provider, or use `--provider auto` to auto-detect |
| `ModuleNotFoundError: No module named 'torch'` | Dependencies not installed | Run `pip install -r requirements.txt` |

## Debug Mode

Enable verbose logging:

```python
# In host/orchestrator.py
visual = VisualConfig(
    enable=True,
    show_tables=True,
    show_rag_table=True,
    show_token_budget=True,  # Enable token budget display
)
```

## Check API Trace

```python
import sqlite3

conn = sqlite3.connect('api_trace.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM api_calls ORDER BY timestamp DESC LIMIT 10")
for row in cursor.fetchall():
    print(row)
conn.close()
```

## Related Documentation

- [Installation Guide](installation.md) - Setup and configuration
- [Usage Guide](usage.md) - CLI arguments and usage
- [Configuration Guide](../config/README.md) - Configuration customization
