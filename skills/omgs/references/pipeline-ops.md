# Pipeline Operations Guide

Complete guide for running, configuring, and debugging the OMGs MDT pipeline.

## Table of Contents

- [CLI Usage](#cli-usage)
- [Environment Setup](#environment-setup)
- [Configuration Files](#configuration-files)
- [Output Structure](#output-structure)
- [Debugging](#debugging)
- [Common Operations](#common-operations)

---

## CLI Usage

### Basic Command

```bash
python main.py --input_path <path> --agent <mode> [--provider <provider>] [--model <model>] [--num_samples <n>]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_path` | str | **Required** | Input JSONL file path |
| `--model` | str | `gpt-5.1` | Model/deployment name |
| `--provider` | str | `auto` | LLM provider: `azure`, `openai`, `openrouter`, `auto` |
| `--agent` | str | `omgs` | Agent mode (see below) |
| `--num_samples` | int | 999999 | Max samples to process |

### Agent Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `omgs` | Full multi-agent MDT discussion | Complex cases requiring multi-specialty debate |
| `chair_sa` | Simplest single-agent mode | Environment/API testing |
| `chair_sa_k` | Single agent + Knowledge (RAG) | Cases needing guideline/literature reference |
| `chair_sa_kep` | Single agent + Knowledge + Evidence Pack | Complex cases with patient data |
| `auto` | **Intelligent routing** | Recommended - auto-selects mode based on complexity |

### Examples

```bash
# Intelligent routing (recommended)
python main.py --input_path ./data.jsonl --agent auto --provider azure --model gpt-5.1

# Full MDT discussion
python main.py --input_path ./data.jsonl --agent omgs --provider azure --model gpt-5.1

# Quick test with simplest mode
python main.py --input_path ./data.jsonl --agent chair_sa --provider azure --model gpt-5.1

# Single agent with knowledge + evidence
python main.py --input_path ./data.jsonl --agent chair_sa_kep --provider azure --model gpt-5.1
```

---

## Environment Setup

### Required Environment Variables

```bash
# Azure OpenAI (Required)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### Verification

```bash
# Check environment
python -c "import os; print('ENDPOINT:', os.environ.get('AZURE_OPENAI_ENDPOINT', 'NOT SET'))"
python -c "import os; print('API_KEY:', 'SET' if os.environ.get('AZURE_OPENAI_API_KEY') else 'NOT SET')"

# Test imports
python -c "from host import process_omgs_multi_expert_query; print('OK')"
```

---

## Configuration Files

### config/paths.json

Data file paths and output directory configuration.

```json
{
  "data_files": {
    "lab_reports": "files/lab_reports_summary.jsonl",
    "imaging_reports": "files/imaging_reports.jsonl",
    "pathology_reports": null,
    "mutation_reports": "files/mutation_reports.jsonl",
    "trials": "files/all_trials_filtered.json"
  },
  "rag_store": {
    "base_dir": "rag_store",
    "index_dir_template": "rag_store/{role}/index/chroma",
    "collection_name_template": "{role}_chunks",
    "embedding_model": "BAAI/bge-m3",
    "use_per_role_rag": false,
    "default_role": "chair",
    "available_roles": ["chair", "oncologist", "radiologist", "pathologist", "nuclear"]
  },
  "output_dirs": {
    "output_answer": "output_answer",
    "mdt_logs": "mdt_logs",
    "api_trace_db": "api_trace.db"
  }
}
```

**Key settings:**
- `use_per_role_rag`: `false` = shared RAG index, `true` = per-role RAG
- `default_role`: Which role's RAG index to use when `use_per_role_rag=false`
- `pathology_reports`: Set to `null` if not available

### config/mdt_prompts.json

MDT discussion prompts and Agent instruction configuration.

```json
{
  "mdt_discussion": {
    "initial_opinion": "Initial opinion prompt...",
    "summarize_initial_template": "Summarize template...",
    "round_summary_template": "Round summary template...",
    "speak_prompt_template": "Turn speak prompt...",
    "final_plan_template": "Final plan prompt..."
  },
  "rag": {
    "query_builder": "RAG query construction prompt...",
    "evidence_summarizer": "Evidence digest prompt..."
  },
  "agents": {
    "rag_query_builder": "...",
    "global_guideline_digester": "...",
    "assistant": "...",
    "trial_selector": "..."
  }
}
```

---

## Output Structure

### Directory Layout

```
output_answer/omgs_YYYY-MM-DD_HH-MM-SS/
├── results.json    # Structured results with metadata
└── results.txt     # Human-readable output

mdt_logs/
├── mdt_history_*.jsonl   # Full pipeline state (machine-readable)
├── mdt_history_*.md      # Discussion transcript (human-readable)
└── mdt_report_*.html     # Visual HTML report
```

### results.json Schema

```json
{
  "agent_mode": "auto(chair_sa_kep)",
  "scene": "recurrence",
  "question": { "CASE_CORE": {...}, "TIMELINE": {...}, ... },
  "response": "Final Assessment:\n...\nCore Treatment Strategy:\n...",
  "gold_plan": "reference answer (optional)",
  "question_raw": "original clinical question",
  "Time": "2024-01-15T10:00:00",
  "meta_info": "patient_identifier"
}
```

**Note:** `agent_mode` shows actual mode used. For `auto` mode, it displays `auto(selected_mode)`.

### HTML Report Features

- **Mode badge** in Pipeline Statistics card
- **MDT Discussion Summary** (Key Knowledge, Controversies, Missing Info, Working Plan)
- **Print button** (🖨️) for one-click printing
- Mermaid pipeline flowchart
- Color-coded References section (4 categories)
- Collapsible expert discussion logs
- Interaction direction matrix
- RAG hits table
- **Dark mode** (auto-detects system theme)
- **Responsive layout** for mobile viewing

---

## Debugging

### Enable Verbose Logging

Edit `host/orchestrator.py`:

```python
visual = VisualConfig(
    enable=True,
    show_tables=True,
    show_rag_table=True,
    show_token_budget=True,  # Enable token budget display
)
```

### Check API Trace

```python
import sqlite3
conn = sqlite3.connect('api_trace.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM api_calls ORDER BY timestamp DESC LIMIT 10")
for row in cursor.fetchall():
    print(row)
conn.close()
```

### MDT Discussion Parameters

In `orchestrator.py` → `run_mdt_discussion()`:

```python
initial_ops, merged, final_round_ops, interaction_log = run_mdt_discussion(
    agents=agents,
    assistant=assistant,
    num_rounds=2,        # Number of discussion rounds
    num_turns=2,         # Turns per round
    max_merged_chars=10000,
    max_turn_delta_chars=900,
    max_targets_per_speaker=4,
    visit_time=visit_time,
    trace=trace,
)
```

---

## Common Operations

### Build RAG Index

```bash
# Step 1: Build chunks from PDFs
python pdf_to_rag.py build \
  --pdf_dir rag_pdf/chair \
  --out_dir rag_store/chair/corpus \
  --chunk_size 1200 \
  --chunk_overlap 200

# Step 2: Index into ChromaDB
python pdf_to_rag.py index \
  --corpus_dir rag_store/chair/corpus \
  --index_dir rag_store/chair/index/chroma \
  --collection_name chair_chunks \
  --model BAAI/bge-m3 \
  --device cpu

# Step 3: Test search
python pdf_to_rag.py search \
  --index_dir rag_store/chair/index/chroma \
  --collection_name chair_chunks \
  --query "NACT in ovarian cancer"
```

### EHR Extraction

```bash
python ehr_structurer.py \
  --input input_ehr/raw_notes.jsonl \
  --output output_ehr/structured.jsonl \
  --deployment gpt-5.1 \
  --prompts config/prompts.json \
  --field question \
  --max-completion-tokens 40000 \
  --retries 4
```

### Batch Processing

```bash
#!/bin/bash
for file in output_ehr/*.jsonl; do
    echo "Processing: $file"
    python main.py --input_path "$file" --agent omgs
done
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: Missing AZURE_OPENAI_ENDPOINT` | Env vars not set | Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` |
| `FileNotFoundError: files/*.jsonl` | Data files missing | Check paths in `config/paths.json` |
| `Failed to load RAG index` | RAG not built | Run `pdf_to_rag.py build` then `index` |
| `Azure API error` | Invalid deployment | Verify `--model` matches Azure deployment name |
| `ModuleNotFoundError` | Dependencies missing | Run `pip install -r requirements.txt` |

### Function-Level Override

You can override default paths directly in code:

```python
from host import process_omgs_multi_expert_query

result = process_omgs_multi_expert_query(
    question=question,
    question_raw=question_raw,
    model=model,
    args=args,
    # Override paths
    labs_json="custom/labs.jsonl",
    imaging_json="custom/imaging.jsonl",
    pathology_json="custom/pathology.jsonl",
    mutation_json="custom/mutations.jsonl",
    trials_json_path="custom/trials.json",
    # RAG parameters
    device="cuda",
    topk=10
)
```
