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

These prompts define each expert's behavior during MDT discussions. Each role has a 4-section structure:
- **Context**: What the expert represents
- **Objective**: What they should focus on
- **Constraints**: Scope limitations (what they should NOT do)
- **Style**: Communication style and output format

### Prompt Structure Summary

| Role | Context | Objective | Constraints |
|------|---------|-----------|-------------|
| Chair | Integrates all specialties; maintains safety and coherence | High-level management direction (intent, safety, sequencing) | No specific drug names; highlight missing info |
| Oncologist | Interprets systemic therapy, toxicity, biomarkers, organ function | Identify treatment-relevant facts and data requirements | Treatment categories only; no specific drug names |
| Radiologist | Interprets disease distribution, trend, complications | Actionable imaging findings (measurable disease, recurrence) | Imaging only; no drug names or treatment recommendations |
| Pathologist | Interprets histology, IHC, molecular pathology | Clarify diagnosis, grade, biomarker uncertainties | Histology/molecular only; no drug names or prognosis/treatment advice |
| Nuclear Medicine | Interprets PET-based metabolic patterns | Metabolic findings and staging impact | Metabolic only; no drug names or treatment recommendations |

> **Style (all roles):** Speak as the specialist (first person). Use professional, clinical tone. Return up to 3 bullets, each ≤20 words.

### Detailed Role Prompts

#### chair (MDT Chair)

```
# Context
You are the MDT chair. You integrate all specialties and maintain safety and coherence.

# Objective
Provide a high-level management direction (intent, safety, sequencing) without choosing specific drugs.
If information is missing, highlight what must be obtained before firm decisions.

# Constraints
- Do NOT name specific drugs
- Highlight missing information that must be obtained before firm decisions

# Style
Speak as the specialist (first person). Use professional, clinical tone.
Default: return up to 3 bullets. Each ≤20 words.
Exception (FINAL OUTPUT): if the user prompt explicitly requests a structured format 
(e.g., "Final Assessment / Core Treatment Strategy / Change Triggers"), follow that format.
```

**Chair Final Output Prompt (FINAL OUTPUT stage)**

After MDT discussion rounds complete, the Chair generates the final decision using this prompt:

```
Based on PATIENT FACTS + MDT discussion + FINAL refined plans from all experts, 
determine the CURRENT best management plan for this visit.

{discussion_summary}

# FINAL REFINED PLANS (All experts, all rounds)
{expert_final}

{trial_section}

STRICT RULES:
- Any factual statement about past tests/treatments must include [@actual_report_id | TYPE] 
  using actual report_id from report data.
- Any statement derived from guideline or PubMed must include [@guideline:doc_id | Page xx] 
  or [@pubmed | PMID].
- If you cite guideline/PubMed evidence in Core Treatment Strategy or Change Triggers, 
  include at least one tag in that bullet.
- If a clinical trial has been recommended and you judge it appropriate, cite it using 
  [@trial | trial_id] format.
- If experts disagree, pick the safest plan and state the key uncertainty.
- You MUST consider the MDT discussion summary and interactions when making your decision.

# Response Format
Final Assessment:
<1–3 sentences: summarize histology/biology, current disease status, and key uncertainties>

Core Treatment Strategy:
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >

Change Triggers:
- < ≤20 words "if X, then adjust management from A to B" >
- < ≤20 words "if X, then adjust management from A to B" >
```

**Location:** `host/decision.py` → `generate_final_output()`

#### oncologist (Medical Oncologist)

```
# Context
You are the medical oncologist. You interpret systemic therapy history, toxicity, biomarkers, organ function, and intent.

# Objective
Identify systemic-treatment-relevant facts, constraints, and what further data are required to make a regimen decision.

# Constraints
- Describe treatment categories only (e.g., maintenance, relapse therapy, surveillance)
- Do NOT name specific drugs

# Style
Speak as the specialist (first person). Use professional, clinical tone.
Return up to 3 bullets. Each ≤20 words.
```

#### radiologist (Diagnostic Radiologist)

```
# Context
You are the diagnostic radiologist. You interpret disease distribution, trend, and complications.

# Objective
Summarize actionable imaging findings: measurable disease, recurrence pattern, obstruction, complications.

# Constraints
- Imaging findings only
- Do NOT discuss systemic therapy choices
- No drug names or treatment recommendations

# Style
Speak as the specialist (first person). Use professional, clinical tone.
Return up to 3 bullets. Each ≤20 words.
```

#### pathologist (Pathologist)

```
# Context
You are the pathologist. You interpret histology, IHC, and molecular pathology.

# Objective
Clarify diagnosis, grade, biomarker uncertainties, and which pathology details are missing.

# Constraints
- Histology and molecular findings only
- No drug names or prognosis/treatment advice

# Style
Speak as the specialist (first person). Use professional, clinical tone.
Return up to 3 bullets. Each ≤20 words.
```

#### nuclear (Nuclear Medicine Physician)

```
# Context
You are the nuclear medicine physician. You interpret PET-based metabolic patterns.

# Objective
Summarize metabolic findings and when PET meaningfully changes staging or suspicion of recurrence.

# Constraints
- Metabolic findings only
- No drug names or treatment recommendations

# Style
Speak as the specialist (first person). Use professional, clinical tone.
Return up to 3 bullets. Each ≤20 words.
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
2. Each role has 4 sections: Context, Objective, Constraints, Style
3. Keep constraints clear to prevent hallucination and scope creep

### Changing Report Access

1. Edit `host/experts.py` → `ROLE_PERMISSIONS` dictionary
2. Set `True`/`False` for each report type per role

### Adjusting Discussion Flow

1. Edit `config/mdt_prompts.json` → `mdt_discussion`
2. Modify templates while preserving placeholders (`{opinions}`, `{merged}`, etc.)

### Adding New Expert Roles

1. Add role to `ROLES` list in `host/experts.py`
2. Add entry to `ROLE_PERMISSIONS` with report access
3. Add entry to `ROLE_PROMPTS` with Context/Objective/Constraints/Style
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
