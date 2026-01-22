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
- [Related Documentation](#related-documentation)

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

## Mutation/Molecular Report Interpretation Rules

**Location:** `host/experts.py` → `init_expert_agent()`

When mutation/molecular reports are provided to experts (chair, oncologist, pathologist), a special interpretation prompt is injected to handle Chinese NGS reports correctly:

```
# MUTATION / MOLECULAR REPORTS (PATIENT FACTS)
⚠️ COMPREHENSIVE NGS PANEL (~20,000 genes) - INTERPRETATION RULES:
• '未检出' (not detected) = NO pathogenic mutation found
• '（视为阴性）' (considered negative) = NO pathogenic mutation found
• '阴性' (negative) = negative result
• Genes with specific variants (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation
• If a gene of interest is NOT mentioned in the report, it means NO pathogenic mutation (comprehensive panel)
• NEVER say 'not tested' or 'not reported' - comprehensive NGS WAS done.
• Only say 'unknown' if NO mutation report is provided at all.
```

**Why This Matters:**

| Chinese Term | Meaning | LLM Misinterpretation Risk |
|--------------|---------|---------------------------|
| 未检出 | Not detected (negative) | LLM may say "not tested" |
| 视为阴性 | Considered negative | LLM may ignore or misread |
| 阴性 | Negative | Generally understood |
| NM_xxx:exon:c.xxx:p.xxx | Positive variant | LLM may miss significance |

**Key Design Decision:**
- Comprehensive NGS panels test ~20,000 genes
- If a gene is NOT mentioned → it was tested and found NEGATIVE
- This prevents LLM from incorrectly stating "BRCA status unknown" when the report simply didn't list BRCA (because it was negative)

---

## MDT Discussion Prompts

**Location:** `config/mdt_prompts.json` → `mdt_discussion`

### initial_opinion

Expert's first assessment at the start of MDT discussion.

```
Give INITIAL opinion (use ONLY your system-provided patient facts).
Return up to 3 bullets, each ≤20 words.
If key data missing, say exactly what needs updating.
At least ONE bullet must be evidence-based and include [@guideline:doc_id | Page xx] or [@pubmed | PMID].
If you reference treatment strategy categories, guidelines, trials, or literature evidence, 
include tags [@guideline:doc_id | Page xx], [@pubmed | PMID], or [@trial | id].
For clinical reports, use actual report_id from report data with type: 
[@actual_report_id | LAB], [@actual_report_id | Genomics], [@actual_report_id | MR], [@actual_report_id | CT].
Always use spaces around | for consistency: [@xxx | yyy].
```

### summarize_initial_template

Assistant merges all expert opinions into structured summary.

```
Summarize expert opinions concisely for MDT.
{opinions}

Output:
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

```
MDT global knowledge:
{merged}

Re-summarize concisely. Must include:
Key Knowledge:
- ...
Controversies:
- ...
Missing Info:
- ...
Working Plan:
- ...
```

### speak_prompt_template (Expert Discussion with "why" Concept)

Controls when experts speak during turns. **Default is NOT to speak.**

This prompt implements the **"why" concept** - experts must justify why they choose to speak:

```
ROLE: {role}. VISIT: {visit_time}
Default is NOT to speak. Speak ONLY if: conflict | safety | missing-critical | new-critical.

CONTEXT (latest):
{context}

Allowed targets: [{allowed_targets}]

EVIDENCE TAGS (if your message references evidence):
- Any factual statement about past tests/treatments must include 
  [@actual_report_id | LAB/Genomics/MR/CT] using actual report_id from report data.
- Any statement derived from guideline or PubMed literature must include 
  [@guideline:doc_id | Page xx] or [@pubmed | PMID].
- If you cite guideline/PubMed evidence or reference clinical trials, 
  include appropriate tags.

Return ONE-LINE JSON only:
{
  "speak": "yes/no",
  "messages": [
    {
      "target": "<role>",
      "message": "<1-2 sentences with evidence tags if applicable>",
      "why": "conflict|safety|missing|new"
    }
  ]
}
```

**The "why" Field Values:**

| Value | Meaning | Example |
|-------|---------|---------|
| `conflict` | Disagreement with another expert's assessment | "I disagree with radiologist's interpretation of tumor response" |
| `safety` | Safety concern identified that must be addressed | "Renal function may contraindicate proposed regimen" |
| `missing` | Missing critical information needed for decision | "BRCA status required before maintenance therapy decision" |
| `new` | New critical insight not previously discussed | "Recent PET shows metabolic progression not reflected in CT" |

### final_plan_template

Expert's refined plan after discussion rounds.

```
Given MDT context:
{merged}

DISCUSSION HISTORY (this round):
{discussion_history}

Provide FINAL refined plan based on the above context and discussions.
Up to 3 bullets, each ≤20 words.
Any factual claim must include [@actual_report_id | LAB/Genomics/MR/CT] 
using actual report_id from report data. Always use spaces around | for consistency.
At least ONE bullet must be evidence-based and include [@guideline:doc_id | Page xx] or [@pubmed | PMID].
If you reference treatment strategy categories, guidelines, trials, or literature evidence, 
include tags [@guideline:doc_id | Page xx], [@pubmed | PMID], or [@trial | id].
If discussions mentioned specific evidence, you may reference it with appropriate tags.
```

---

## RAG Prompts

**Location:** `config/mdt_prompts.json` → `rag`

### query_builder

Constructs a concise English query for guideline/evidence retrieval.

```
You are preparing a single concise English query to retrieve guideline/clinical evidence 
for this ovarian cancer MDT case.

# STRUCTURED_CASE_TEXT
{question}

Write ONE line (<=40 words) focusing on:
- tumor type/histology and platinum status;
- key metastases / disease extent;
- key molecular markers if mentioned (e.g., BRCA/HRD/MSI/PD-L1/ATM);
- major clinical constraints (e.g., anemia, organ function, performance).

Do NOT mention report_ids, dates, hospital names, or patient identifiers.

IMPORTANT PRIORITY: If KEY FACTS section exists above, it takes PRIORITY over STRUCTURED_CASE_TEXT 
for genetic markers. If KEY FACTS shows 'MUTATION_REPORT' with 'HRD 阴性' (negative) or 
'BRCA 未检出/阴性' (negative), you MUST say 'HRD-negative' or 'BRCA-negative' in your query. 
NEVER say 'not reported' or 'not tested' when KEY FACTS contains MUTATION_REPORT data.

Output ONLY the query text.
```

### evidence_summarizer

Digests RAG results into actionable evidence bullets.

```
# RAG CHUNKS
{rag_pack}

{count_info}
Summarize into evidence bullets for MDT decision-making.

Rules:
- Each bullet must be actionable evidence (guideline/trial-based).
- Do NOT restate patient-specific facts.
- Avoid long quotes.
- Each bullet MUST include at least one evidence tag: [@guideline:doc_id | Page xx] or [@pubmed | PMID].
- Output ONLY plain text bullets, one per RAG result, in order.
```

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

### Detailed Agent Prompts

**rag_query_builder:**
```
Construct concise English MDT guideline query.
```

**global_guideline_digester:**
```
(Dynamic: 1:1 mapping) Digest RAG chunks into exactly N evidence bullets (one per RAG result); 
no patient facts.
```

**assistant:**
```
You are MDT assistant. Summarize only. Do not decide treatment.
```

**trial_selector:**
```
You are an MDT assistant for clinical trial matching in gynecologic oncology. 
Follow the trial recommendation gate strictly and recommend at most ONE trial.
```

### Trial Selector Detailed Prompt

**Location:** `host/decision.py` → `assistant_trial_suggestion()`

When performing clinical trial matching, the trial_selector uses this comprehensive prompt:

```
You are an MDT assistant for gynecologic oncology clinical trial matching.

CRITICAL BEHAVIOR:
- You MUST NOT ask the user any questions.
- You MUST NOT request additional information.
- You MUST NOT output anything except the required template.
- Use ONLY the provided PATIENT CASE text and AVAILABLE TRIALS list.
- If eligibility is unclear due to missing key facts, you MUST output None.

PATIENT CASE (facts only; do not infer):
{case_json_str}

AVAILABLE TRIALS (compact; use id/name exactly as shown):
{trials_json}

DECISION RULE (be conservative):
Recommend ONE trial ONLY IF ALL are true:
1) Cancer type / primary site clearly matches.
2) Disease setting clearly matches (e.g., recurrent/advanced/metastatic and line is not fundamentally unclear).
3) Required biomarker/subtype is explicitly present in case text (if trial requires it).
4) No more than 2 critical eligibility confirmations remain.

If ANY of the above is not satisfied -> output None.

OUTPUT TEMPLATE (EXACT; no extra text):

Trial Recommendation:
- id: <trial id or None>
- name: <trial name or None>
- Reason: <1 short sentence>
- Missing eligibility confirmations (0-2 items):
  - item1 (or "None")
  - item2
```

### Auto Mode Routing Agent Prompt

**Location:** `host/orchestrator.py` → `process_auto_query()`

When using `--agent auto`, the routing agent analyzes case complexity:

```
# OMGs System Background (for routing decision)
OMGs (Ovarian-cancer Multidisciplinary intelligent aGent System) is specifically designed for:
- Complex ovarian cancer patients requiring multi-line therapy
- Full lifecycle treatment management (from diagnosis through recurrence)
- Multidisciplinary decision support integrating oncology, radiology, pathology, and nuclear medicine

# Your Task
Analyze the following case and determine which processing mode is most appropriate.

# Available Modes
1. chair_sa: Simplest mode for environment testing or trivial queries
2. chair_sa_k: Single agent with Knowledge (guidelines + literature) - for cases needing evidence reference
3. chair_sa_kep: Single agent with Knowledge + Evidence Pack (reports + trials) - for complex cases with available data
4. omgs: Full multi-agent MDT discussion - for highly complex cases requiring multi-specialty debate

# Complexity Factors to Consider
- Line of therapy: 初诊/1线 (simple) → 2-3线 (medium) → 4线+ (complex)
- Genetic testing: None/simple (simple) → BRCA/HRD present (medium) → Multiple complex mutations (complex)
- Platinum status: Clear (simple) → Borderline (medium) → Complex/contradictory (complex)
- Comorbidities: None/few (simple) → Moderate (medium) → Multiple/severe (complex)
- Clinical questions: Single clear question (simple) → 2-3 questions (medium) → Multiple difficult decisions (complex)

# Case to Analyze
{question_str}

# Output Format (JSON only, no other text)
{"mode": "chair_sa|chair_sa_k|chair_sa_kep|omgs", "reason": "brief explanation in Chinese"}
```

**Routing Agent Instruction:**
```
You are a clinical triage agent for OMGs. 
Analyze case complexity and select the appropriate processing mode.
```

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

## Output Format

### results.json Schema

The `main.py` script generates `results.json` files in `output_answer/{agent}_{timestamp}/` directories. Each result entry includes:

```json
{
  "agent_mode": "omgs",
  "mode": "omgs",
  "model": "gpt-5.1",
  "scene": "recurrence",
  "question": { "CASE_CORE": {...}, "TIMELINE": {...}, ... },
  "response": "Final Assessment:\n...\nCore Treatment Strategy:\n...",
  "gold_plan": "reference answer (optional)",
  "question_raw": "original clinical question",
  "Time": "2024-01-15T10:00:00",
  "meta_info": "patient_identifier"
}
```

**Key Fields:**
- `agent_mode`: The actual agent mode used (e.g., `"omgs"`, `"chair_sa"`, `"auto(chair_sa_kep)"`)
- `mode`: Same as `agent_mode` (for backward compatibility and batch processing)
- `model`: The LLM model name used (e.g., `"gpt-5.1"`, `"anthropic/claude-opus-4.5"`)
- `meta_info`: Patient identifier used for grouping results in batch processing
- `response`: The final MDT decision output

**Note:** For `auto` mode, `agent_mode` displays as `auto(selected_mode)` to show the actual mode that was selected by the routing agent.

### Batch Processing

**Location:** `run_batch.sh`

The batch processing script automates multi-model evaluation workflows:

1. **EHR Structuring**: Runs `ehr_structurer.py` to process input JSONL files
2. **Multi-Model Execution**: Sequentially runs `main.py` with different agent and model configurations
3. **Result Extraction**: Automatically extracts and organizes results by patient (`meta_info`)

**Usage:**
```bash
./run_batch.sh
```

**Output Structure:**
```
output_batch/
└── {meta_info}/
    └── results.txt
```

Each `results.txt` file contains all model results for that patient, organized by `{mode}_{model}` sections:

```
==================================================
chair_sa_gpt-5.1
==================================================
{response content}

==================================================
omgs_anthropic/claude-opus-4.5
==================================================
{response content}
...
```

**Supported Configurations:**
- Chair-SA baseline group (3 configurations)
- OMGs baseline group (1 configuration)
- OpenRouter comparison group (6 configurations)

See the script comments for the complete list of model configurations.

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
| `run_batch.sh` | Batch processing script for multi-model evaluation |

---

## Related Documentation

- **[Multi-Provider Guide](../clients/PROVIDERS.md)** - Detailed LLM provider usage, configuration, and model-specific settings (Azure OpenAI, OpenAI, OpenRouter)
- **[Installation Guide](../docs/installation.md)** - System requirements and installation steps
- **[Usage Guide](../docs/usage.md)** - CLI arguments and usage instructions
- **[Evidence System](../docs/evidence-system.md)** - Open Evidence References system
- **[Extension Guide](../skills/omgs/references/extension-guide.md)** - Adding new roles and report types
