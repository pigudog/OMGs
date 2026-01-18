# OMGs - Ovarian-cancer Multidisciplinary intelligent aGent System

[![Python 3.10](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**OMGs** (Ovarian-cancer Multidisciplinary intelligent aGent System) is a multi-agent clinical decision-support system for ovarian cancer MDT (multidisciplinary team) discussions. It simulates multiple specialist roles (Chair, Medical Oncology, Radiology, Pathology, Nuclear Medicine), runs multi-round deliberation, and produces structured MDT recommendations.

## 🏥 Clinical Significance

- **MDT-ready decision support**: aligns multi-specialty opinions to reduce fragmented reasoning in complex ovarian cancer care
- **Evidence with patient facts**: keeps patient facts and guideline evidence side by side for transparent reasoning
- **Traceable and reviewable**: full discussion logs and report selection enable audit and quality review
- **Safety boundaries by role**: role permissions and report evidence constrain output to reduce hallucination risk
- **Real-world clinical uplift**: supports regional hospitals and residents by improving decision quality in resource-limited settings

## ✨ Key Features

- **🤖 Multi-expert agents**: five specialist roles (Chair, Oncologist, Radiologist, Pathologist, Nuclear Medicine)
- **📊 Smart report selection**: role-specific filtering of labs, imaging, pathology, and mutation reports
- **🔍 RAG enhancement**: ChromaDB-backed guideline retrieval
- **💬 Multi-round MDT engine**: structured expert discussion to resolve conflicts and fill gaps
- **🧪 Clinical trial matching**: optional trial recommendation module
- **📝 Full observability**: JSONL logs, Markdown transcripts, HTML report, and interaction matrix
- **🔐 Role-based access control**: each expert only sees relevant report types

## 🏗️ System Architecture

```
Input case data
    ↓
[1] Load reports (lab / imaging / pathology / mutation)
    ↓
[2] Role-based report selection (permissions)
    ↓
[3] Global guideline RAG (ChromaDB + embeddings)
    ↓
[4] Initialize expert agents (5 roles)
    ↓
[5] MDT discussion engine (2 rounds × 2 turns)
    ↓
[6] Clinical trial matching (optional)
    ↓
[7] Final MDT decision output
    ↓
Save artifacts (JSON + TXT + HTML)
```

### Roles and Permissions

| Role | Lab Reports | Imaging Reports | Pathology Reports | Guideline Type |
|------|-------------|----------------|-------------------|----------------|
| Chair | ✅ | ✅ | ❌ | chair |
| Oncologist | ✅ | ❌ | ❌ | oncologist |
| Radiologist | ❌ | ✅ | ❌ | radiologist |
| Pathologist | ❌ | ❌ | ✅ | pathologist |
| Nuclear Medicine | ❌ | ✅ | ❌ | nuclear |

## 📋 Requirements

- **Python**: 3.8+
- **OS**: Linux, macOS, Windows
- **Azure OpenAI**: valid Azure OpenAI account
- **GPU** (optional): to accelerate local embeddings

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Clone (if needed)
git clone <repository-url>
cd OMGs

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key-here"
```

On Windows:

```cmd
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_API_KEY=your-api-key-here
```

### 3. Prepare data files

Ensure the following files exist:

- **Input cases** (`input_ehr/*.jsonl`): JSONL case data
- **Lab reports** (`files/lab_reports_summary.jsonl`)
- **Imaging reports** (`files/imaging_reports.jsonl`)
- **Mutation reports** (`files/mutation_reports.jsonl`)
- **Pathology reports** (optional): `files/pathology_reports.jsonl`
- **RAG index** (`rag_store/chair/index/chroma/`)
- **Clinical trials** (optional): `all_trials_filtered.json`

All data paths can be overridden via `config/paths.json` (see **Advanced Configuration**).

### 4. Run the system

```bash
python main.py \
    --input_path input_ehr/test_guo.jsonl \
    --model gpt-5.1 \
    --agent omgs \
    --num_samples 10
```

## 📖 Usage

### CLI arguments

```bash
python main.py [OPTIONS]
```

**Required:**
- `--input_path`: path to input JSONL file

**Optional:**
- `--model`: Azure deployment name (default: `gpt-5.1`)
- `--agent`: agent type (default: `basic_baseline`). Choices:
  `basic_baseline`, `basic_role`, `basic_rag`, `basic_rag_lab`, `basic_rag_lab_full`, `omgs`.
  Currently only `omgs` is implemented; other values will fallback to `omgs` with a warning.
- `--num_samples`: number of samples to process (default: 999999)

**Token parameters:**
- OpenAI Chat Completions deprecates `max_tokens`; use `max_completion_tokens` for output limits.

### Input format

Each JSONL line should include:

```json
{
  "meta_info": "patient identifier (for report matching)",
  "Time": "2024-01-15",
  "question": {
    "CASE_CORE": {
      "DIAGNOSIS": "diagnosis info",
      "LINE_OF_THERAPY": "line of therapy",
      "BIOMARKERS": {},
      "CURRENT_STATUS": "current status"
    },
    "TIMELINE": {},
    "MED_ONC": {},
    "RADIOLOGY": {},
    "PATHOLOGY": {},
    "LAB_TRENDS": {}
  },
  "question_raw": "original question",
  "scene": "scene tag",
  "gold_plan": "gold answer (optional)"
}
```

### Outputs

System outputs to `output_answer/{agent}_{timestamp}/`:

1. **results.json**: structured results (question, response, metadata)
2. **results.txt**: human-readable text output

And to `mdt_logs/`:

1. **mdt_history_{timestamp}.jsonl**: full MDT log (JSONL)
2. **mdt_history_{timestamp}.md**: Markdown discussion log
3. **mdt_report_{timestamp}.html**: interactive HTML report

## 📁 Project Structure

```
OMGs/
├── main.py                 # entry script (MDT pipeline)
├── agent_omgs.py           # MDT pipeline core
├── ehr_structurer.py       # EHR extraction / structuring
├── pdf_to_rag.py           # RAG corpus/index builder
├── requirements.txt        # dependencies
├── README.md               # this file
│
├── aoai/                   # Azure OpenAI wrapper
│   ├── wrapper.py
│   └── logger.py
│
├── utils/                  # utilities
│   ├── core.py             # Agent class + helpers
│   ├── role_utils.py       # roles and agent init
│   ├── rag_utils.py        # RAG retrieval
│   ├── select_utils.py     # report selection
│   ├── omgs_reports.py     # HTML/Markdown reporting
│   ├── console_utils.py    # console formatting
│   ├── time_utils.py       # timeline utilities
│   └── trace_utils.py      # observability
│
├── config/                 # configs
│   ├── prompts.json
│   └── paths.json
│
├── files/                  # data files
│   ├── lab_reports_summary.jsonl
│   ├── imaging_reports.jsonl
│   └── mutation_reports.jsonl
│
├── input_ehr/              # input cases
│   ├── test_guo.jsonl
│   └── ...
│
├── output_answer/          # outputs
│   └── omgs_YYYY-MM-DD_HH-MM-SS/
│       ├── results.json
│       └── results.txt
│
├── mdt_logs/               # MDT logs
│   ├── mdt_history_*.jsonl
│   ├── mdt_history_*.md
│   └── mdt_report_*.html
│
└── rag_store/              # RAG index
    └── chair/
        └── index/
            └── chroma/
```

## 🔧 Dependencies

Key packages (see `requirements.txt` for details):

- **openai** (≥1.0.0): Azure OpenAI client
- **chromadb** (≥0.4.0): vector database
- **langchain-huggingface** (≥0.0.1): embedding integration
- **torch** (≥2.0.0): deep learning framework
- **tiktoken** (≥0.5.0): token counting
- **tqdm** (≥4.65.0): progress bar
- **prettytable** (≥3.8.0): table rendering

## 💡 Examples

### Basic usage

```bash
python main.py --input_path input_ehr/test_guo.jsonl --num_samples 5
```

### Use a different model

```bash
python main.py \
    --input_path input_ehr/test_guo.jsonl \
    --model gpt-4 \
    --num_samples 10
```

### Batch processing

```bash
for file in input_ehr/*.jsonl; do
    python main.py --input_path "$file"
done

### EHR extraction / structuring (raw → JSONL)

Use `ehr_structurer.py` to convert raw notes into structured EHR JSON:

```bash
python ehr_structurer.py \
  --input input_ehr/raw_notes.jsonl \
  --output input_ehr/structured.jsonl \
  --deployment gpt-5.1 \
  --prompts config/prompts.json \
  --field question \
  --max-completion-tokens 40000 \
  --retries 4 \
  --txt-dir output_ehr/txt_out
```

Notes:
- `--field` is the input JSONL key that contains raw note text (default: `question`).
- `--txt-dir` is optional; when set, a per-patient TXT preview is saved.
- `--disable-json-repair` can be used to skip automatic JSON repair.

### Build / index / search RAG guidelines

```bash
# 1) Build TXT + chunks from PDFs
python pdf_to_rag.py build \
  --pdf_dir rag_pdf/chair \
  --out_dir rag_store/chair/corpus \
  --chunk_size 1200 \
  --chunk_overlap 200

# 2) Index chunks into Chroma
python pdf_to_rag.py index \
  --corpus_dir rag_store/chair/corpus \
  --index_dir rag_store/chair/index/chroma \
  --collection_name chair_chunks \
  --model BAAI/bge-m3 \
  --device cpu

# 3) Search the index
python pdf_to_rag.py search \
  --index_dir rag_store/chair/index/chroma \
  --collection_name chair_chunks \
  --model BAAI/bge-m3 \
  --device cpu \
  --query "NACT in ovarian cancer"
```
```

## 🔍 How It Works

### 1. Report loading and filtering

Reports are loaded by `meta_info`:
- Lab reports (CBC, LFT, renal, tumor markers)
- Imaging reports (CT, MRI, PET)
- Pathology reports (histology, IHC, molecular)
- Mutation reports

Reports are filtered by visit time and then per-role relevance.

### 2. Role-specific views

Each expert receives:
- **Role-specific case view** (only relevant fields)
- **Selected reports** (role permissions + clinical relevance)
- **Global guideline digest** from RAG

### 3. MDT discussion flow

1. **Initial opinions** per expert
2. **Multi-round discussion** (Round 1/2, Turn 1/2)
3. **Final refined plans** per expert

### 4. Final decision output

The Chair synthesizes discussion into the final MDT output:
- Final assessment
- Core treatment strategy
- Change triggers
- Trial suggestion (if applicable)

## 📊 Output Example

### JSON output schema

```json
{
  "scene": "scene tag",
  "question": "normalized question",
  "response": "final MDT decision",
  "gold_plan": "gold answer (if present)",
  "question_raw": "original question",
  "Time": "2024-01-15",
  "meta_info": "patient id"
}
```

### HTML report includes

- Interaction matrix across experts
- Report selection tables
- RAG hit table and digest
- Full discussion timeline
- Final output and trial suggestions

## ⚙️ Advanced Configuration

### Paths configuration (`config/paths.json`)

You can centralize all data and output paths here:

```json
{
  "data_files": {
    "lab_reports": "files/lab_reports_summary.jsonl",
    "imaging_reports": "files/imaging_reports.jsonl",
    "pathology_reports": null,
    "mutation_reports": "files/mutation_reports.jsonl",
    "trials": "all_trials_filtered.json"
  },
  "rag_store": {
    "index_dir_template": "rag_store/{role}/index/chroma",
    "collection_name_template": "{role}_chunks",
    "embedding_model": "BAAI/bge-m3",
    "use_per_role_rag": false,
    "default_role": "chair"
  },
  "output_dirs": {
    "output_answer": "output_answer",
    "mdt_logs": "mdt_logs",
    "api_trace_db": "api_trace.db"
  }
}
```

### Custom report paths (function-level override)

Configure in `agent_omgs.py`:

```python
process_omgs_multi_expert_query(
    question=question,
    question_raw=question_raw,
    model=model,
    args=args,
    labs_json="custom/labs.jsonl",
    imaging_json="custom/imaging.jsonl",
    pathology_json="custom/pathology.jsonl",
    mutation_json="custom/mutations.jsonl",
    trials_json_path="custom/trials.json"
)
```

### RAG configuration

- **Index path**: `rag_store/chair/index/chroma/` (or from `config/paths.json`)
- **Embedding model**: `BAAI/bge-m3` (see `rag_utils.py`)
- **Top-k**: default `topk=5`

## 🐛 Troubleshooting

1. **Missing environment variables**
   ```
   RuntimeError: Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY
   ```
   **Fix**: set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`

2. **File not found**
   ```
   FileNotFoundError: files/lab_reports_summary.jsonl
   ```
   **Fix**: check file paths and existence

3. **RAG index missing**
   ```
   Failed to load RAG index
   ```
   **Fix**: ensure `rag_store/chair/index/chroma/` exists

4. **Model deployment name error**
   ```
   Azure API error
   ```
   **Fix**: ensure `--model` matches Azure deployment name

## 📝 Development Notes

### Add a new specialist role

1. Add role in `utils/role_utils.py` to `ROLES`
2. Define permissions in `ROLE_PERMISSIONS`
3. Add role prompt in `ROLE_PROMPTS`
4. Update `build_role_specific_case_view`

### Add a new report type

1. Add loader in `utils/select_utils.py`
2. Extend `ROLE_PERMISSIONS`
3. Update `expert_select_reports`

## 📄 License

MIT License. See `LICENSE`.

## 🙏 Acknowledgements

Thanks to all contributors to the OMGs system.

## 📧 Contact

For questions or feedback:
- Open an Issue
- Email the maintainer

---

**⚠️ Medical Disclaimer**: This system is for research and education only. It does not replace professional medical diagnosis or treatment. All clinical decisions must be made by qualified healthcare professionals.
