# Project Structure

This document describes the complete project structure of OMGs, including the purpose of each directory and key files.

## Directory Tree

```
OMGs/
├── main.py                     # Entry point - MDT pipeline
├── ehr_structurer.py           # EHR extraction entry script (calls servers/case_parser)
├── pdf_to_rag.py               # RAG corpus/index builder
├── requirements.txt            # Python dependencies
├── README.md                   # Main documentation
├── .cursorrules                # Cursor IDE integration rules
│
├── skills/                     # SKILL Protocol Package
│   └── omgs/
│       ├── SKILL.md            # Core skill definition (~100 lines)
│       └── references/         # Detailed reference guides
│           ├── architecture.md     # Three-layer architecture details
│           ├── expert-roles.md     # Role permissions and prompts
│           ├── extension-guide.md  # Adding roles/report types
│           └── pipeline-ops.md     # CLI, config, debugging
│
├── host/                       # Central Host (Orchestration Layer)
│   ├── __init__.py             # Package exports
│   ├── orchestrator.py         # MDT discussion engine + main pipeline
│   │                           #   - run_mdt_discussion()
│   │                           #   - process_omgs_multi_expert_query()
│   ├── experts.py              # Expert agent definitions
│   │                           #   - ROLES, ROLE_PERMISSIONS, ROLE_PROMPTS
│   │                           #   - init_expert_agent()
│   └── decision.py             # Final decision-making & post-processing
│                               #   - generate_final_output()
│                               #   - append_references_to_output()
│                               #   - parse_trial_from_note()
│                               #   - assistant_trial_suggestion()
│                               #   - build_enhanced_case_for_trial()
│
├── servers/                    # Agent Servers (Service Layer)
│   ├── __init__.py             # Package exports
│   ├── case_parser.py          # EHR extraction (full implementation)
│   │                           #   - process_file(), apply_auto_fixes()
│   │                           #   - try_parse_json(), main()
│   ├── info_delivery.py        # Role-specific case views
│   │                           #   - build_role_specific_case_view()
│   │                           #   - safe_load_case_json()
│   ├── evidence_search.py      # RAG retrieval & evidence summarization
│   │                           #   - get_global_guideline_rag()
│   │                           #   - pubmed_search_pack()
│   │                           #   - build_rag_query_for_mdt() → extracts HRD/BRCA from mutation reports
│   │                           #   - summarize_rag_evidence() → 1:1 digest
│   ├── reports_selector.py     # Clinical report selection
│   │                           #   - load_patient_labs/imaging/pathology/mutations()
│   │                           #   - select_reports_for_roles()
│   │                           #   - expert_select_reports()
│   ├── trace.py                # Observability utilities
│   │                           #   - TraceLogger, VisualConfig
│   │                           #   - print_selected_reports_table(), print_rag_hits_table()
│   │                           #   - warn_missing_evidence_tags()
│   └── reporters.py            # Report generation & HTML visualization
│                               #   - save_mdt_log() → JSONL + Markdown
│                               #   - save_case_html_report() → HTML report
│                               #   - _render_pipeline_stats_html() → Statistics card
│                               #   - _render_final_output_html() → References UI
│                               #   - Mermaid.js flowchart rendering
│
├── core/                       # Core Infrastructure
│   ├── __init__.py             # Package exports
│   ├── agent.py                # Stateful LLM Agent wrapper
│   │                           #   - Agent class with chat(), run_selection()
│   ├── client.py               # Multi-provider LLM client initialization
│   │                           #   - init_client(), init_client_from_config()
│   └── config.py               # Configuration loading
│                               #   - load_paths_config(), get_paths_config()
│                               #   - load_mdt_prompts(), get_mdt_prompts()
│                               #   - load_data(), create_question(), setup_model()
│
├── clients/                    # Multi-Provider LLM Client Wrapper
│   ├── __init__.py
│   ├── wrapper.py              # OpenAIWrapper class
│   ├── logger.py               # API logging
│   ├── test_connection.py      # Provider connection tests
│   └── PROVIDERS.md            # Multi-provider usage guide
│
├── utils/                      # Pure Utility Functions
│   ├── __init__.py             # Package exports
│   ├── console_utils.py        # Console formatting
│   │                           #   - Color class
│   ├── stats_collector.py      # Pipeline statistics collection
│   │                           #   - collect_pipeline_stats() → query API trace DB
│   │                           #   - Aggregates tokens, models, execution time
│   ├── time_utils.py           # Time-related utilities
│   │                           #   - format_duration() → human-readable time
│   │                           #   - preview_text(), print_prompt_budget()
│   │                           #   - normalize_trial_compact()
│   │                           #   - safe_parse_json_block()
│   │                           #   - question_to_text()
│   │                           #   - parse_dt(), parse_date()
│   │                           #   - make_cutoff(), filter_before()
│   │                           #   - build_lab/imaging/pathology_timeline()
│   ├── reference_cache.py      # Reference caching & Open Evidence system
│   │                           #   - ReferenceCache, get_reference_cache()
│   │                           #   - extract_reference_tags() - 4 types supported
│   │                           #   - build_references_section() - formatted refs
│   │                           #   - store_trial(), get_trial() - trial caching
│   └── skill_loader.py         # SKILL Protocol runtime loader
│                               #   - load_skill() - parse SKILL.md with caching
│                               #   - build_skill_digest() - ~75 token digest per role
│                               #   - get_skill_info() - metadata for tracing
│
├── config/                     # Configuration Files
│   ├── paths.json              # Data and output paths
│   ├── mdt_prompts.json        # MDT discussion prompts
│   ├── prompts.json            # EHR extraction prompts
│   └── README.md               # Configuration documentation
│
├── files/                      # Data Files
│   ├── lab_reports_summary.jsonl
│   ├── imaging_reports.jsonl
│   └── mutation_reports.jsonl
│
├── input_ehr/                  # Input Case Files
│   └── *.jsonl
│
├── output_answer/              # Pipeline Outputs
│   └── omgs_YYYY-MM-DD_HH-MM-SS/
│       ├── results.json
│       └── results.txt
│
├── output_ehr/                 # EHR Extraction Outputs
│   └── *.jsonl
│
├── mdt_logs/                   # MDT Discussion Logs
│   ├── mdt_history_*.jsonl     # Full pipeline state (machine-readable)
│   ├── mdt_history_*.md        # Human-readable discussion transcript
│   └── mdt_report_*.html       # Visual HTML report with:
│                               #   - Mermaid pipeline flowchart
│                               #   - Color-coded References section
│                               #   - Expert debate matrix
│
└── rag_store/                  # RAG Index Storage
    ├── chair/
    │   ├── corpus/
    │   │   ├── chunks/         # Chunked guideline text
    │   │   ├── meta/           # Document metadata
    │   │   └── staging_txt/    # Raw text files
    │   └── index/
    │       └── chroma/         # ChromaDB index
    ├── oncologist/
    ├── radiologist/
    ├── pathologist/
    └── nuclear/
```

## Three-Layer Architecture

OMGs follows a three-layer architecture:

1. **Core Layer** (`core/`): Infrastructure components (Agent, Client, Config)
2. **Service Layer** (`servers/`): Service components (Case Parser, Evidence Search, Reports Selector)
3. **Orchestration Layer** (`host/`): Central orchestration (Orchestrator, Experts, Decision)

For detailed architecture documentation, see [skills/omgs/references/architecture.md](../skills/omgs/references/architecture.md).

## Key Entry Points

- **`main.py`**: Main MDT pipeline entry point
- **`ehr_structurer.py`**: EHR extraction entry point
- **`pdf_to_rag.py`**: RAG index builder

## Related Documentation

- [Architecture Details](../skills/omgs/references/architecture.md) - Three-layer architecture
- [Extension Guide](../skills/omgs/references/extension-guide.md) - Adding roles and report types
- [Pipeline Operations](../skills/omgs/references/pipeline-ops.md) - CLI, config, debugging
