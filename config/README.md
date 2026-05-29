# Configuration

This directory contains the public configuration files used by the standalone
OMGs release.

## Files

| File | Purpose |
|------|---------|
| `paths.json` | Default de-identified example data paths, output directory and provider environment-variable mapping. |
| `mdt_prompts.json` | Runtime protocol, role prompts, deliberation prompts, evidence-query prompts and chair-output templates. |
| `ehr_prompts.json` | Structured EHR extraction, review and refinement instructions, including the JSON output schema. |

## Data Paths

The committed defaults point to de-identified example fixtures:

```json
{
  "data_files": {
    "lab_reports": "files/lab_reports.example.jsonl",
    "imaging_reports": "files/imaging_reports.example.jsonl",
    "pathology_reports": "files/pathology_reports.example.jsonl",
    "mutation_reports": "files/mutation_reports.example.jsonl"
  },
  "output_dirs": {
    "output_answer": "output_answer",
    "api_trace_db": "data/logs/omgs_api_trace.db"
  }
}
```

Do not place real clinical report exports in this release tree. Use local
overrides or per-sample `report_paths` for institution-specific data.

## Prompt Boundary

`mdt_prompts.json` records the prompts and runtime rules used by the released
standalone MDT pipeline. Evidence-runtime prompts for PubMed, FDA, conference,
guideline, NCCN and trial construction are maintained in the companion
repositories listed in the top-level `README.md`.

## Evidence Tags

| Source type | Format |
|-------------|--------|
| Guideline | `[@guideline:doc_id \| Page xx]` |
| NCCN rule | `[@guideline:nccn \| rule_id]` |
| PubMed | `[@pubmed \| PMID]` |
| FDA | `[@fda \| source_id:section]` |
| Conference | `[@conference \| abstract_id]` |
| Trial | `[@trial \| id]` |
| Clinical report | `[@actual_report_id \| LAB/Genomics/MR/CT/Pathology]` |
