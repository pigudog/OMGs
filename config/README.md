# OMGs Configuration Guide

This document provides comprehensive documentation for the prompt system and configuration files in OMGs.

## Table of Contents

- [File Structure](#file-structure)
- [Expert Role Prompts](#expert-role-prompts)
- [Role Permission Matrix](#role-permission-matrix)
- [MDT Discussion Prompts](#mdt-discussion-prompts)
- [RAG Prompts](#rag-prompts)
- [Agent Instructions](#agent-instructions)
- [Evidence Tag Rules](#evidence-tag-rules)
- [Customization Guide](#customization-guide)

---

## File Structure

| File | Purpose |
|------|---------|
| `paths.json` | Data file paths, RAG store configuration, output directories |
| `mdt_prompts.json` | MDT discussion prompts, RAG prompts, agent instructions |

---

## Expert Role Prompts

**Location:** `host/experts.py` → `ROLE_PROMPTS`

These prompts define each expert's behavior during MDT discussions. Each role has:
- **Context**: What the expert represents
- **Objective**: What they should focus on
- **Constraints**: What they should NOT do

### chair (MDT Chair)

```
Context: MDT chair, integrates all specialties, maintains safety and coherence
Objective: Provide high-level management direction (intent, safety, sequencing)
Constraints: No specific drug names; if info missing, highlight what must be obtained
Style: Up to 3 bullets, each ≤20 words
```

The chair is unique in that during FINAL OUTPUT, they follow a structured format (Final Assessment / Core Treatment Strategy / Change Triggers) instead of the bullet constraint.

### oncologist (Medical Oncologist)

```
Context: Interprets systemic therapy history, toxicity, biomarkers, organ function
Objective: Identify systemic-treatment-relevant facts and constraints
Constraints: May describe treatment categories but must NOT name specific drugs
Style: Up to 3 bullets, each ≤20 words
```

### radiologist (Diagnostic Radiologist)

```
Context: Interprets disease distribution, trend, and complications
Objective: Summarize actionable imaging findings (measurable disease, recurrence, obstruction)
Constraints: Do NOT discuss systemic therapy choices
Style: Up to 3 bullets, each ≤20 words, imaging only
```

### pathologist (Pathologist)

```
Context: Interprets histology, IHC, and molecular pathology
Objective: Clarify diagnosis, grade, biomarker uncertainties, missing pathology details
Constraints: Do NOT suggest treatment choices
Style: Up to 3 bullets, each ≤20 words, no prognosis/treatment advice
```

### nuclear (Nuclear Medicine Physician)

```
Context: Interprets PET-based metabolic patterns
Objective: Summarize metabolic findings and when PET changes staging/recurrence suspicion
Constraints: Do NOT comment on systemic therapy choices
Style: Up to 3 bullets, each ≤20 words
```

---

## Role Permission Matrix

Each expert can only access specific types of clinical reports:

| Role | Lab | Imaging | Pathology | Mutation |
|------|:---:|:-------:|:---------:|:--------:|
| chair | ✓ | ✓ | - | ✓ |
| oncologist | ✓ | - | - | ✓ |
| radiologist | - | ✓ | - | - |
| pathologist | - | - | ✓ | ✓ |
| nuclear | - | ✓ | - | - |

**Location:** `host/experts.py` → `ROLE_PERMISSIONS`

This matrix ensures:
- Each expert only sees data relevant to their specialty
- Reduces hallucination by limiting context
- Enforces separation of concerns in MDT discussion

---

## MDT Discussion Prompts

**Location:** `config/mdt_prompts.json` → `mdt_discussion`

### initial_opinion

Expert's first assessment at the start of MDT discussion.

**Rules:**
- Up to 3 bullets, each ≤20 words
- At least ONE bullet must include evidence tag
- If key data missing, state exactly what needs updating

### summarize_initial_template

Assistant merges all expert opinions into structured summary.

**Output format:**
```
Key Knowledge:
- ...
Controversies:
- ...
Missing Info:
- ...
Working Plan:
- ...
```

### round_summary_template

Re-summarize global knowledge after each discussion round.

**Same output format as above.**

### speak_prompt_template

Controls when experts speak during turns. **Default is NOT to speak.**

**Speak only if:**
- `conflict`: Disagreement with another expert
- `safety`: Safety concern identified
- `missing`: Missing critical information
- `new`: New critical insight

**Output:** Single-line JSON with speak decision and messages

### final_plan_template

Expert's refined plan after discussion rounds.

**Rules:**
- Up to 3 bullets, each ≤20 words
- At least ONE bullet must include evidence tag
- Reference discussion evidence where applicable

---

## RAG Prompts

**Location:** `config/mdt_prompts.json` → `rag`

### query_builder

Constructs a concise English query for guideline/evidence retrieval.

**Focus areas:**
- Tumor type/histology and platinum status
- Key metastases / disease extent
- Key molecular markers (BRCA/HRD/MSI/PD-L1/ATM)
- Major clinical constraints (anemia, organ function, performance)

**Constraint:** ≤40 words, no patient identifiers

### evidence_summarizer

Digests RAG results into actionable evidence bullets.

**Rules:**
- Each bullet must be actionable evidence
- Do NOT restate patient-specific facts
- Each bullet MUST include evidence tag
- One bullet per RAG result, in order

---

## Agent Instructions

**Location:** `config/mdt_prompts.json` → `agents`

Short role descriptions for specialized utility agents:

| Agent | Purpose |
|-------|---------|
| `rag_query_builder` | Construct concise English MDT guideline query |
| `global_guideline_digester` | Digest RAG chunks into exactly N evidence bullets (1:1 mapping) |
| `assistant` | MDT assistant, summarize only, do not decide treatment |
| `trial_selector` | Clinical trial matching, recommend at most ONE trial |

---

## Evidence Tag Rules

All expert outputs must include evidence tags for traceability.

### Guidelines

```
[@guideline:doc_id | Page xx]
```

Example: `[@guideline:NCCN_OC_2024 | Page 45]`

### PubMed Literature

```
[@pubmed | PMID]
```

Example: `[@pubmed | 33758607]`

### Clinical Trials

```
[@trial | id]
```

Example: `[@trial | NCT04729387]`

### Patient Reports

```
[@actual_report_id | TYPE]
```

Types: `LAB`, `Genomics`, `MR`, `CT`

Examples:
- `[@20220407|17300673 | LAB]` - Lab report
- `[@OH2203828|2022-04-18 | Genomics]` - Genomics report
- `[@2022-12-29 | MR]` - MR imaging report
- `[@2022-12-29 | CT]` - CT imaging report

**Important:** Always use spaces around `|` for consistency: `[@xxx | yyy]`

---

## Customization Guide

### Modifying Expert Behavior

1. Edit `host/experts.py` → `ROLE_PROMPTS` dictionary
2. Each role has: Context, Objective, Style sections
3. Keep constraints clear to prevent hallucination

### Changing Report Access

1. Edit `host/experts.py` → `ROLE_PERMISSIONS` dictionary
2. Set `True`/`False` for each report type per role

### Adjusting Discussion Flow

1. Edit `config/mdt_prompts.json` → `mdt_discussion`
2. Modify templates while preserving placeholders (`{opinions}`, `{merged}`, etc.)

### Adding New Expert Roles

1. Add role to `ROLES` list in `host/experts.py`
2. Add entry to `ROLE_PERMISSIONS` with report access
3. Add entry to `ROLE_PROMPTS` with Context/Objective/Style
4. Update any role-specific logic in `orchestrator.py`

See [extension-guide.md](../skills/omgs/references/extension-guide.md) for detailed instructions.

---

## Quick Reference

### Prompt Flow in MDT

```
1. initial_opinion (each expert)
     ↓
2. summarize_initial_template (assistant)
     ↓
3. [Round N]
   ├── round_summary_template (assistant)
   ├── speak_prompt_template (experts, if triggered)
   └── final_plan_template (each expert)
     ↓
4. Final Output (chair synthesis)
```

### Key Files

| File | Contains |
|------|----------|
| `config/paths.json` | Data paths, RAG config, output dirs |
| `config/mdt_prompts.json` | All prompts and agent instructions |
| `host/experts.py` | Role prompts, permissions, agent initialization |
| `host/orchestrator.py` | MDT discussion flow and round/turn logic |
