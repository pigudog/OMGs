# OMGs - Ovarian-cancer Multidisciplinary intelligent aGent System

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**OMGs** (Ovarian-cancer Multidisciplinary intelligent aGent System) is a multi-agent clinical decision-support system for ovarian cancer MDT (multidisciplinary team) discussions. It simulates multiple specialist roles (Chair, Medical Oncology, Radiology, Pathology, Nuclear Medicine), runs multi-round deliberation, and produces structured MDT recommendations.

> [!NOTE]
> **Design Philosophy**: OMGs is specifically designed for ovarian cancer patients requiring **multi-line therapy** with **complex etiologies** and **multiple comorbidities**. Our goal is to provide comprehensive care throughout the **entire lifecycle** of ovarian cancer patients.
>
> **Tiered Care Recommendation**: Not all patients require full MDT discussion. For **simpler cases**, we recommend using the `--agent auto` mode, which intelligently routes cases to the appropriate processing level based on complexity assessment.
>
> **SKILL Protocol**: The `skills/omgs/` documentation is designed for future tool enhancements and IDE integration. Currently it only provides: (1) ~75-token evidence tagging prompt injection per agent, (2) Cursor IDE development context via `.cursorrules`. The SKILL docs do **not** increase runtime token consumption.

---

## Table of Contents

- [Clinical Significance](#-clinical-significance)
- [Key Features](#-key-features)
- [Recent Updates](#-recent-updates)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Agent Modes](#-agent-modes)
- [Documentation](#-documentation)
- [License](#-license)

---

## 🏥 Clinical Significance

### Why MDT Decision Support Matters

Multidisciplinary team (MDT) meetings are the gold standard for complex cancer care, but face challenges:

- **Information overload**: Specialists must synthesize vast amounts of patient data
- **Time constraints**: Limited meeting time for thorough case review
- **Regional disparities**: Resource-limited settings lack specialist expertise
- **Documentation gaps**: Discussion rationale often poorly captured

### How OMGs Addresses These Challenges

| Challenge | OMGs Solution |
|-----------|---------------|
| **Fragmented reasoning** | MDT-ready decision support aligns multi-specialty opinions |
| **Transparency** | Evidence with patient facts side by side for transparent reasoning |
| **Auditability** | Full discussion logs and report selection enable quality review |
| **Hallucination risk** | Role permissions and report evidence constrain output |
| **Resource limitations** | Supports regional hospitals and residents with AI-assisted decisions |

### Architecture Highlights

- **Modular Multi‑Agent Collaboration**: Five specialized expert agents (Chair, Oncology, Radiology, Pathology and Nuclear Medicine) coordinate through a central orchestrator and conduct two‑round deliberations, with each agent constrained to its role.
- **Open Evidence System**: Uses a Retrieval‑Augmented Generation (RAG) model combining clinical guidelines and PubMed literature; every conclusion is accompanied by standardized citations to ensure traceability.
- **Comprehensive Logging and Observability**: Automatically produces JSONL logs, Markdown transcripts and HTML reports; the HTML report includes flowcharts, discussion matrices and reference cards, facilitating audit and debugging.
- **SKILL Protocol**: Each agent receives a ~75‑token runtime skill digest enforcing citation format and role constraints, ensuring consistent behavior across agents.
- **Scalable Best‑Practice Design**: The architecture follows multi‑agent best practices and can evolve into a hierarchical model, allowing new roles or tasks to be added as needed.

### Clinical Workflow Integration

OMGs integrates into clinical workflow: **EHR System** → **Extract & Structure** → **MDT Discussion** → **Treatment Plan**, with evidence from Guidelines, PubMed, and Patient Reports.

---

## ✨ Key Features

### Multi-Agent Collaboration

- **🤖 Five Specialist Agents**: Chair, Medical Oncologist, Radiologist, Pathologist, Nuclear Medicine Physician
- **💬 Multi-Round Discussion**: Structured 2-round × 2-turn debate to resolve conflicts
- **🎯 Role-Based Permissions**: Each expert only accesses relevant report types

### Evidence Integration

- **🔍 RAG Enhancement**: ChromaDB-backed guideline and PubMed retrieval
- **📊 Smart Report Selection**: LLM-powered filtering of labs, imaging, pathology, mutations
- **🧬 Automatic Genetic Marker Extraction**: Automatically extracts HRD/BRCA status from mutation reports for accurate RAG queries
- **🧪 Clinical Trial Matching**: Optional trial recommendation module

### Observability & Traceability

- **📝 Full Logging**: JSONL logs, Markdown transcripts, HTML reports
- **📈 Interaction Matrix**: Visual representation of expert discussions
- **🔐 Evidence Tags**: All claims linked to source reports or guidelines
- **📊 Pipeline Statistics**: Automatic tracking and display of execution time, token usage, and model utilization in HTML reports

### Error Handling & Resilience

- **🛡️ Graceful Degradation**: Single agent failures don't crash the entire MDT pipeline
- **⏱️ Timeout Protection**: 10-second timeout for RAG model initialization prevents infinite retries
- **🔄 Automatic Retries**: Rate limit errors (429) are automatically retried with exponential backoff (2-3 attempts)
- **🔄 Automatic Fallbacks**: Failed operations use sensible defaults (e.g., empty RAG results, error placeholders)
- **📊 Error Tracking**: All errors logged to trace with detailed context for debugging
- **🌐 Network Resilience**: Handles HuggingFace model download failures gracefully, skips RAG if network unavailable
- **📁 Missing Directory Tolerance**: System continues operation even if `files/` or `rag_store/` directories are missing, using empty data or in-memory caches

### Open Evidence References

- **📋 Structured References**: Auto-generated reference section with 4 categories (Guidelines, Literature, Trials, Reports)
- **🎨 Visual HTML Report**: Color-coded reference cards with Mermaid flowchart
- **🔗 1:1 Evidence Mapping**: Each RAG result gets a dedicated digest bullet

### SKILL Protocol Integration

- **🧠 Modular Knowledge**: Self-contained skill package in `skills/omgs/` with progressive disclosure
- **⚡ Runtime Injection**: ~75 tokens per agent enforcing evidence format and role constraints
- **📖 Reference Guides**: Architecture, expert roles, extension guide, and pipeline ops documentation
- **🔧 Cursor Integration**: `.cursorrules` for automatic IDE context loading

---

## 🆕 Recent Updates

### v1.2 - EHR Iterative Refinement Architecture

**New Extract-Review-Refine Loop:**

```
Extract → Self-Review → Validator-Review → [Fixable Issues?]
                                              ↓ Yes
                                         Refine → Re-Review (max 2x)
                                              ↓ No
                                         Auto-Fix → Output with Confidence
```

- **Issue Classification**: `fixable` (LLM errors) / `truncation` / `ambiguous` (human review)
- **Iterative Correction**: Auto-fixes inference errors (e.g., BRCA wrongly set to Wildtype)
- **Field Confidence**: Per-field confidence scores (high/medium/low)
- **Enhanced Output**: txt_out includes review issues, refinements, and confidence for human review

**Concise Prompts**: `MASTER_INSTRUCTIONS` reduced from ~3200 to ~1200 chars (refinement catches errors)

### v1.1 - Multi-Mode Support & Intelligent Routing

| Mode | Description | Use Case |
|------|-------------|----------|
| `omgs` | Full multi-agent MDT discussion | Complex cases |
| `chair_sa` | Single-agent baseline | Testing |
| `chair_sa_k` | Single agent + Knowledge | Evidence reference |
| `chair_sa_kep` | Single agent + Knowledge + Evidence Pack | Complex with data |
| `auto` | **Intelligent routing** | Recommended |

**Auto Mode**: Routes cases to appropriate level based on complexity (line of therapy, genetic testing, platinum status).

**Usage:**
```bash
# Intelligent routing (recommended)
python main.py --input_path ./data.jsonl --agent auto --provider azure --model gpt-5.1
```

---

## 🏗️ System Architecture

OMGs follows a **three-layer architecture** (host/ servers/ core/) with multi-agent collaboration:

- **Input Layer**: Case data, clinical reports, guidelines
- **Core Infrastructure**: Agent class, multi-provider LLM client
- **Service Layer**: Case parser, evidence search, report selector
- **Orchestration Layer**: Multi-agent coordinator, expert agents, decision maker
- **Output Layer**: JSON results, HTML reports, MDT logs

**Pipeline**: Load → Filter → Select → RAG → Digest → Initialize → Discuss (2-round × 2-turn) → Trial Match → Synthesize → Output

📖 **[Complete Architecture Documentation](docs/system-architecture.md)** - High-level diagrams, detailed pipeline flow, roles & permissions, agent modes

---

## 🚀 Quick Start

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.10+ |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 5 GB | 20+ GB (with RAG index) |
| **GPU** | Not required | CUDA-compatible (for faster embeddings) |
| **OS** | Linux, macOS, Windows | Linux/macOS |

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-org/OMGs.git
   cd OMGs
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure LLM provider** (see [Installation Guide](docs/installation.md) for detailed setup):
   ```bash
   # Azure OpenAI
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_API_KEY="your-api-key-here"
   
   # Or OpenAI
   export OPENAI_API_KEY="your-openai-api-key"
   
   # Or OpenRouter
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   ```

### Run MDT Pipeline

```bash
# Basic usage (auto-detect provider)
python main.py --input_path ./input_ehr/test_cases.jsonl --agent omgs

# With specific provider and model
python main.py \
  --input_path ./input_ehr/test_cases.jsonl \
  --agent omgs \
  --model gpt-5.1 \
  --provider azure \
  --num_samples 10
```

For complete workflow examples, see [Examples](docs/examples.md).

---

## Agent Modes

OMGs supports multiple agent modes for different use cases:

| Mode | Description | Use Case |
|------|-------------|----------|
| `omgs` | Full multi-agent MDT discussion (default) | Complex cases requiring multi-specialty debate |
| `chair_sa` | Simplest single-agent mode | Environment/API testing |
| `chair_sa_k` | Single agent + Knowledge (Guidelines + PubMed) | Cases needing evidence reference |
| `chair_sa_kep` | Single agent + Knowledge + Evidence Pack | Complex cases with patient data |
| `auto` | **Intelligent routing** based on case complexity | Recommended for tiered care |

The `auto` mode automatically assesses case complexity and routes to the appropriate processing level, reducing resource usage for simpler cases while ensuring complex cases get full MDT support.

---

## 📚 Documentation

| Category | Document | Description |
|----------|----------|-------------|
| **Getting Started** | [Installation](docs/installation.md) | Setup, requirements, LLM config |
| | [Usage](docs/usage.md) | CLI, input/output formats |
| | [Examples](docs/examples.md) | Workflow examples |
| **Core** | [System Architecture](docs/system-architecture.md) | Architecture diagrams and pipeline |
| | [Project Structure](docs/project-structure.md) | Directory and modules |
| | [Evidence System](docs/evidence-system.md) | RAG and citations |
| | [Prompts Reference](docs/prompts-reference.md) | **Complete prompt documentation** |
| | [Configuration](config/README.md) | Config files overview |
| **Advanced** | [Contributing](docs/contributing.md) | Development guide |
| | [Troubleshooting](docs/troubleshooting.md) | Debugging guide |
| | [Evaluation](docs/evaluation.md) | Assessment system |
| | [Multi-Provider](clients/PROVIDERS.md) | LLM provider details |
| **Reference** | [Architecture](skills/omgs/references/architecture.md) | Three-layer design |
| | [Expert Roles](skills/omgs/references/expert-roles.md) | Role definitions |
| | [Extension Guide](skills/omgs/references/extension-guide.md) | Adding roles/reports |
| | [SKILL Protocol](skills/omgs/SKILL.md) | Runtime injection |

---

## 📄 License

MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgements

- Azure OpenAI, OpenAI, and OpenRouter for language model capabilities
- ChromaDB for vector storage
- HuggingFace for embedding models
- All contributors to the OMGs project

---

## 📧 Contact

For questions, issues, or contributions:
- Open a GitHub Issue
- Email the maintainers

---

## ⚠️ Medical Disclaimer

**This system is for research and educational purposes only.**

- It does NOT replace professional medical diagnosis or treatment
- All clinical decisions MUST be made by qualified healthcare professionals
- The system outputs are advisory and require expert validation
- Patient data must be handled according to applicable privacy regulations (HIPAA, GDPR, etc.)

---

*Last updated: January 2026*
