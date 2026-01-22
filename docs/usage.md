# Usage Guide

This guide covers CLI arguments, input/output formats, and detailed usage instructions for OMGs.

## Multi-Provider LLM Support

OMGs supports multiple LLM providers through the `--provider` argument:

- **`azure`**: Azure OpenAI (default, backward compatible)
- **`openai`**: OpenAI official API
- **`openrouter`**: OpenRouter (supports multiple models like Gemini, Claude, etc.)
- **`auto`**: Auto-detect provider based on model name (default)

You can use different providers for different steps of the pipeline. For example, use Azure for EHR extraction and OpenAI for MDT processing.

**Gemini Reasoning Models:** For `google/gemini-3-pro-preview` and similar reasoning models via OpenRouter:
- **Reasoning is mandatory**: The model requires reasoning to be enabled and cannot be disabled. The system automatically sends an empty `extra_body` (letting the API use default reasoning settings) instead of `reasoning: {enabled: false}`.
- **Automatic token scaling**: Since reasoning tokens are counted in `max_completion_tokens`, the system automatically scales up `max_tokens` by 8x (minimum 20,000) for Gemini 3 Pro models to ensure sufficient tokens for both reasoning (~1000-2000 tokens) and actual content output.
- **Example**: An expert agent with `max_tokens=900` will automatically get `max(900*8, 20000) = 20000` tokens when using Gemini 3 Pro.

See [clients/PROVIDERS.md](../clients/PROVIDERS.md) for detailed provider usage guide.

## CLI Arguments

### main.py

```bash
python main.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_path` | str | **Required** | Path to input JSONL file |
| `--model` | str | `gpt-5.1` | Model/deployment name |
| `--provider` | str | `auto` | LLM provider: `azure`, `openai`, `openrouter`, or `auto` (auto-detect based on model name) |
| `--agent` | str | `basic_baseline` | Agent type (use `omgs`) |
| `--num_samples` | int | 999999 | Number of samples to process |

### ehr_structurer.py

```bash
python ehr_structurer.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | **Required** | Path to input JSONL file |
| `--output` | str | **Required** | Path to output JSONL file |
| `--deployment` | str | **Required** | Model/deployment name |
| `--prompts` | str | **Required** | Path to prompts configuration file |
| `--provider` | str | `auto` | LLM provider: `azure`, `openai`, `openrouter`, or `auto` (auto-detect based on model name) |
| `--field` | str | `question` | Field name to extract |
| `--max-completion-tokens` | int | 40000 | Maximum completion tokens |
| `--retries` | int | 4 | Number of retry attempts |
| `--txt-dir` | str | `""` | Directory for text output files |
| `--db_path` | str | `api_trace.db` | Path to API trace database |
| `--disable-json-repair` | flag | False | Disable automatic JSON repair |
| `--verbose` | flag | False | Enable verbose retry logs |
| `--quiet` | flag | False | Suppress retry logs |

## Input Format

Each JSONL line should contain:

```json
{
  "meta_info": "patient_identifier",
  "Time": "2024-01-15T10:00:00",
  "question": {
    "CASE_CORE": {
      "DIAGNOSIS": {
        "primary": "High-grade serous ovarian carcinoma",
        "histology": "Serous",
        "stage": "IIIC"
      },
      "LINE_OF_THERAPY": "2nd line",
      "BIOMARKERS": {
        "CA125": "156 U/mL",
        "HRD": "Positive",
        "BRCA1": "Pathogenic mutation"
      },
      "CURRENT_STATUS": "Recurrent disease"
    },
    "TIMELINE": {
      "events": [
        {"date": "2023-01-15", "event": "Initial diagnosis"},
        {"date": "2023-06-01", "event": "Completed chemotherapy"}
      ]
    },
    "MED_ONC": {},
    "RADIOLOGY": {},
    "PATHOLOGY": {},
    "NUC_MED": {},
    "LAB_TRENDS": {}
  },
  "question_raw": "Original clinical question text",
  "scene": "recurrence",
  "gold_plan": "Reference treatment plan (optional)"
}
```

## Output Files

**Directory: `output_answer/{agent}_{timestamp}/`**

| File | Description |
|------|-------------|
| `results.json` | Structured results with all metadata |
| `results.txt` | Human-readable text output |

**Directory: `mdt_logs/`**

| File | Description |
|------|-------------|
| `mdt_history_{timestamp}.jsonl` | Complete MDT log in JSONL format |
| `mdt_history_{timestamp}.md` | Markdown discussion transcript |
| `mdt_report_{timestamp}.html` | Interactive HTML report |

## Output Schema

```json
{
  "scene": "recurrence",
  "question": { "normalized case object" },
  "response": "Final Assessment:\n...\n\nCore Treatment Strategy:\n...\n\nChange Triggers:\n...",
  "gold_plan": "reference answer if provided",
  "question_raw": "original question text",
  "Time": "2024-01-15T10:00:00",
  "meta_info": "patient_identifier"
}
```

## Data File Preparation

**Note**: The system can run even if some data files or directories are missing. Missing files will result in empty data for that component, but the pipeline will continue. However, for full functionality, ensure the following files exist:

```
OMGs/
├── input_ehr/
│   └── your_cases.jsonl          # Input case data
├── files/
│   ├── lab_reports_summary.jsonl # Lab reports
│   ├── imaging_reports.jsonl     # Imaging reports
│   └── mutation_reports.jsonl    # Mutation reports
├── rag_store/
│   └── chair/
│       └── index/
│           └── chroma/           # Pre-built RAG index
└── all_trials_filtered.json      # Clinical trials (optional)
```

## Two-Step Process

**Step 1: Extract and Structure EHR (if starting from raw notes)**

See [Examples](examples.md) for detailed workflow examples.

**Step 2: Run MDT Pipeline**

```bash
python main.py \
  --input_path ./output_ehr/structured.jsonl \
  --agent omgs \
  --model gpt-5.1 \
  --provider azure \
  --num_samples 10
```

## Check Outputs

```bash
# Results
ls -la output_answer/omgs_*/

# MDT Logs
ls -la mdt_logs/

# Open HTML report
open mdt_logs/mdt_report_*.html  # macOS
# or
xdg-open mdt_logs/mdt_report_*.html  # Linux
```

## Related Documentation

- [Examples](examples.md) - Complete workflow examples
- [Configuration Guide](../config/README.md) - Prompt and configuration customization
- [Multi-Provider Guide](../clients/PROVIDERS.md) - Detailed provider usage
