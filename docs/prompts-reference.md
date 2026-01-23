# OMGs Prompts Reference

Complete documentation for the prompt system in OMGs.

## Table of Contents

- [EHR Extraction Prompts](#ehr-extraction-prompts)
- [Expert Role Prompts](#expert-role-prompts)
- [MDT Discussion Prompts](#mdt-discussion-prompts)
- [RAG Prompts](#rag-prompts)
- [Agent Instructions](#agent-instructions)
- [Evidence Tag Rules](#evidence-tag-rules)

---

## EHR Extraction Prompts

**Location:** `config/prompts.json`

### Architecture: Extract-Review-Refine Loop

```
Extract → Self-Review → Validator-Review → [Fixable Issues?]
                                              ↓ Yes
                                         Refine → Re-Review → [Still Issues?]
                                              ↑_______________|
                                              ↓ No (max 2 iterations)
                                         Auto-Fix Rules → Output
```

### MASTER_INSTRUCTIONS

Core extraction rules for gynecologic oncology EHR:

| Section | Key Rules |
|---------|-----------|
| OUTPUT | Valid JSON only. Missing = "Unknown". Empty arrays = [] |
| VERBATIM | Drug names, regimens, histology exactly as source |
| DATES | YYYY-MM-DD \| YYYY-MM \| YYYY \| Unknown |
| LINE_OF_THERAPY | Verbatim regimen, cycles, response, discontinuation |
| GENOMICS | HRD/BRCA inference: "Mutated" only if explicit; "Wildtype" only if explicit negative; else "Unknown" |
| PLATINUM | PFI: ≤28d Refractory; 28-180d Resistant; ≥180d Sensitive |
| TOXICITIES | Only if explicitly mentioned with severity |
| CLINICAL_TRIALS | Trial name, NCT ID, phase if mentioned |

### REFINE_INSTRUCTIONS

Used during iterative refinement when review identifies fixable issues:

```
- Output ONLY corrected field values as JSON
- Use dot notation for paths (e.g., "CASE_CORE.BRCA1")
- HRD is a phenotype, NOT a gene mutation
- BRCA1/BRCA2 should only be Wildtype if explicitly tested negative
```

### REFINEMENT_CONFIG

```json
{
  "max_iterations": 2,
  "min_fixable_issues": 1,
  "issue_severity_threshold": "major"
}
```

### Issue Classification

| Type | Description | Action |
|------|-------------|--------|
| `fixable` | LLM inference errors (e.g., BRCA wrongly inferred) | Auto-refine |
| `truncation` | Value cutoff (e.g., PLT:34 vs PLT:342) | Auto-refine |
| `ambiguous` | Source text unclear | Human review |

### OUTPUT_SCHEMA (Key Fields)

```
CASE_CORE
├── VISIT_DATE, ECOG, PLATINUM_STATUS, SCENE
├── DIAGNOSIS (primary, histology, site, laterality)
├── SURGERY_DONE, NEOADJUVANT, ADJUVANT_TREATMENT
├── LINE_OF_THERAPY[] (line, regimen, cycles, response_assessment, discontinuation)
├── MAINTENANCE, RELAPSE
├── HRD, BRCA1, BRCA2 (legacy fields)
├── GENOMICS (testing_performed, HRD_STATUS, alterations[])
└── BIOMARKERS

TIMELINE.events[]
MED_ONC (current_regimen, genetic_testing)
RADIOLOGY.studies[]
PATHOLOGY.specimens[]
NUC_MED.studies[]
LAB_TRENDS.labs[]
TOXICITIES[]
CLINICAL_TRIALS[]
```

---

## Expert Role Prompts

**Location:** `host/experts.py` → `ROLE_PROMPTS`

### Structure

Each role has 4 sections: Context, Objective, Constraints, Style.

### Role Summary

| Role | Focus | Constraints |
|------|-------|-------------|
| Chair | Integrate specialties; high-level direction | No specific drugs; highlight missing info |
| Oncologist | Systemic therapy, toxicity, biomarkers | Treatment categories only |
| Radiologist | Disease distribution, trend, complications | Imaging only |
| Pathologist | Histology, IHC, molecular pathology | No treatment advice |
| Nuclear Medicine | PET metabolic patterns | Metabolic only |

### Role Permissions

| Role | Lab | Imaging | Pathology | Mutation |
|------|:---:|:-------:|:---------:|:--------:|
| chair | ✓ | ✓ | - | ✓ |
| oncologist | ✓ | - | - | ✓ |
| radiologist | - | ✓ | - | - |
| pathologist | - | - | ✓ | ✓ |
| nuclear | - | ✓ | - | - |

### Chair Final Output Format

```
Final Assessment:
<1–3 sentences: histology/biology, disease status, key uncertainties>

Core Treatment Strategy:
- < ≤20 words concrete decision >
- < ≤20 words concrete decision >

Change Triggers:
- < ≤20 words "if X, then adjust management" >
```

---

## MDT Discussion Prompts

**Location:** `config/mdt_prompts.json` → `mdt_discussion`

### Prompt Flow

```
1. initial_opinion (each expert)
     ↓
2. summarize_initial_template (assistant)
     ↓
3. [Round N]
   ├── round_summary_template
   ├── speak_prompt_template (experts, if triggered)
   └── final_plan_template (each expert)
     ↓
4. Final Output (chair synthesis)
```

### speak_prompt_template ("why" Concept)

Experts must justify speaking with one of:

| Value | Meaning |
|-------|---------|
| `conflict` | Disagreement with another expert |
| `safety` | Safety concern identified |
| `missing` | Missing critical information |
| `new` | New critical insight |

---

## RAG Prompts

**Location:** `config/mdt_prompts.json` → `rag`

### query_builder

Constructs concise English query for guideline retrieval:
- Focus: tumor type, platinum status, metastases, molecular markers, clinical constraints
- Exclude: report IDs, dates, patient identifiers
- Priority: KEY FACTS section over STRUCTURED_CASE_TEXT for genetic markers

### evidence_summarizer

Digests RAG chunks into actionable evidence bullets:
- Each bullet must include evidence tag
- No patient-specific facts
- One bullet per RAG result

---

## Agent Instructions

**Location:** `config/mdt_prompts.json` → `agents`

| Agent | Purpose |
|-------|---------|
| `rag_query_builder` | Construct MDT guideline query |
| `global_guideline_digester` | Digest RAG chunks (1:1 mapping) |
| `assistant` | MDT summarizer (no treatment decisions) |
| `trial_selector` | Clinical trial matching (at most ONE trial) |

### Trial Selector Decision Rule

Recommend trial ONLY IF ALL true:
1. Cancer type matches
2. Disease setting matches
3. Required biomarker present
4. ≤2 critical eligibility confirmations remain

---

## Evidence Tag Rules

All expert outputs must include evidence tags for traceability.

| Type | Format | Example |
|------|--------|---------|
| Guideline | `[@guideline:doc_id \| Page xx]` | `[@guideline:NCCN_OC_2024 \| Page 45]` |
| PubMed | `[@pubmed \| PMID]` | `[@pubmed \| 33758607]` |
| Trial | `[@trial \| id]` | `[@trial \| NCT04729387]` |
| Report | `[@report_id \| TYPE]` | `[@20220407 \| LAB]` |

Report types: `LAB`, `Genomics`, `MR`, `CT`

**Important:** Always use spaces around `|` for consistency.

---

## Mutation Report Interpretation

**Location:** `host/experts.py` → `init_expert_agent()`

Chinese NGS report terms:

| Term | Meaning | Correct Interpretation |
|------|---------|----------------------|
| 未检出 | Not detected | NEGATIVE (not "not tested") |
| 视为阴性 | Considered negative | NEGATIVE |
| 阴性 | Negative | NEGATIVE |
| NM_xxx:c.xxx:p.xxx | Variant notation | POSITIVE mutation |

**Key Rule:** Comprehensive NGS panels test ~20,000 genes. If a gene is NOT mentioned → tested and NEGATIVE.

---

## Auto Mode Routing

**Location:** `host/orchestrator.py` → `process_auto_query()`

| Mode | Use Case |
|------|----------|
| `chair_sa` | Environment testing, trivial queries |
| `chair_sa_k` | Cases needing evidence reference |
| `chair_sa_kep` | Complex cases with available data |
| `omgs` | Highly complex, multi-specialty debate |

Complexity factors: Line of therapy, genetic testing, platinum status, comorbidities, clinical questions.

---

## Customization

### Modifying Expert Behavior

1. Edit `host/experts.py` → `ROLE_PROMPTS`
2. Keep 4-section structure: Context, Objective, Constraints, Style
3. Keep constraints clear to prevent hallucination

### Changing Report Access

Edit `host/experts.py` → `ROLE_PERMISSIONS` dictionary.

### Adding New Expert Roles

1. Add to `ROLES` list
2. Add `ROLE_PERMISSIONS` entry
3. Add `ROLE_PROMPTS` entry
4. Update role logic in `orchestrator.py`

See [extension-guide.md](../skills/omgs/references/extension-guide.md) for details.
