# OMGs Configuration

## Files Overview

| File | Purpose |
|------|---------|
| `paths.json` | Data paths, RAG stores, output directories |
| `mdt_prompts.json` | MDT discussion prompts, RAG prompts, agent instructions |
| `prompts.json` | EHR extraction schema and prompts |

---

## Quick Reference

### prompts.json - EHR Extraction

Defines structured medical history extraction with iterative refinement:

```
Extract → Review → Refine (max 2x) → Auto-Fix → Output
```

Key sections:
- `MASTER_INSTRUCTIONS`: Core extraction rules
- `OUTPUT_SCHEMA`: JSON schema for structured EHR
- `REFINE_INSTRUCTIONS`: Iterative correction prompts
- `REFINEMENT_CONFIG`: Loop control (max_iterations, severity threshold)
- `SCHEMA_METADATA`: Version and references for publication

### mdt_prompts.json - MDT Discussion

Controls multi-agent discussion flow:
- `mdt_discussion`: Expert opinion templates, round summaries
- `rag`: Query builder and evidence summarizer
- `agents`: Utility agent instructions

### paths.json - System Paths

```json
{
  "rag_stores": { "chair": "...", "oncologist": "..." },
  "output_dirs": { "answer": "...", "ehr": "..." },
  "trial_file": "files/all_trials_filtered.json"
}
```

---

## Evidence Tags

| Type | Format |
|------|--------|
| Guideline | `[@guideline:doc_id \| Page xx]` |
| PubMed | `[@pubmed \| PMID]` |
| Trial | `[@trial \| NCT_ID]` |
| Report | `[@report_id \| LAB/Genomics/MR/CT]` |

---

## Detailed Documentation

For comprehensive prompt documentation, see:

- **[Prompts Reference](../docs/prompts-reference.md)** - Complete prompt system documentation
- **[Extension Guide](../skills/omgs/references/extension-guide.md)** - Adding roles and customization
- **[Expert Roles](../skills/omgs/references/expert-roles.md)** - Role definitions and permissions
