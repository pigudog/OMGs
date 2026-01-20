# Expert Roles Reference

## Table of Contents

1. [Role Definitions](#role-definitions)
2. [Permission Matrix](#permission-matrix)
3. [Role Prompts](#role-prompts)
4. [Discussion Protocol](#discussion-protocol)

---

## Role Definitions

Source: `host/experts.py`

```python
ROLES = ["chair", "oncologist", "radiologist", "pathologist", "nuclear"]
```

| Role | Clinical Focus |
|------|----------------|
| **chair** | Overall synthesis, safety, sequencing |
| **oncologist** | Systemic therapy, biomarkers, toxicity |
| **radiologist** | Disease distribution, imaging trends |
| **pathologist** | Histology, IHC, molecular pathology |
| **nuclear** | PET/metabolic findings |

---

## Permission Matrix

Source: `host/experts.py` - `ROLE_PERMISSIONS`

```python
ROLE_PERMISSIONS = {
    "chair":        {"lab": True,  "imaging": True,  "pathology": False, "mutation": True,  "guideline": "chair"},
    "oncologist":   {"lab": True,  "imaging": False, "pathology": False, "mutation": True,  "guideline": "oncologist"},
    "radiologist":  {"lab": False, "imaging": True,  "pathology": False, "mutation": False, "guideline": "radiologist"},
    "pathologist":  {"lab": False, "imaging": False, "pathology": True,  "mutation": True,  "guideline": "pathologist"},
    "nuclear":      {"lab": False, "imaging": True,  "pathology": False, "mutation": False, "guideline": "nuclear"},
}
```

Visual matrix:

| Role | Lab | Imaging | Pathology | Mutation | Guideline RAG |
|------|:---:|:-------:|:---------:|:--------:|:-------------:|
| chair | Y | Y | - | Y | chair |
| oncologist | Y | - | - | Y | oncologist |
| radiologist | - | Y | - | - | radiologist |
| pathologist | - | - | Y | Y | pathologist |
| nuclear | - | Y | - | - | nuclear |

---

## Role Prompts

Source: `host/experts.py` - `ROLE_PROMPTS`

### Chair
```
Context: MDT chair. Integrate all specialties and maintain safety/coherence.
Objective: High-level management direction (intent, safety, sequencing) without specific drugs.
Style: Up to 3 bullets, ≤20 words each. Exception: FINAL OUTPUT may use structured format.
```

### Oncologist
```
Context: Medical oncologist. Systemic therapy history, toxicity, biomarkers, organ function.
Objective: Identify systemic-treatment-relevant facts/constraints. Categories only, no drug names.
Style: Up to 3 bullets, ≤20 words each. No drug names.
```

### Radiologist
```
Context: Diagnostic radiologist. Disease distribution, trend, complications.
Objective: Actionable imaging findings (measurable disease, recurrence, obstruction).
Style: Up to 3 bullets, ≤20 words each. Imaging only.
```

### Pathologist
```
Context: Pathologist. Histology, IHC, molecular pathology.
Objective: Diagnosis, grade, biomarker uncertainties, missing details.
Style: Up to 3 bullets, ≤20 words each. No treatment advice.
```

### Nuclear Medicine
```
Context: Nuclear medicine physician. PET-based metabolic patterns.
Objective: Metabolic findings, when PET changes staging/recurrence suspicion.
Style: Up to 3 bullets, ≤20 words each. No treatment recommendations.
```

---

## Discussion Protocol

### Evidence Tag Requirements

All experts must use evidence tags in their responses:

| Tag Type | Format | When to Use |
|----------|--------|-------------|
| Guideline | `[@guideline:doc_id\|page]` | Treatment strategy, consensus statements |
| PubMed | `[@pubmed:PMID]` | Literature evidence |
| Report | `[@report_id\|date]` | Patient-specific lab/imaging/pathology facts |
| Trial | `[@trial:id]` | Clinical trial recommendations |

### Discussion Turns

Speaking protocol (configured in `config/mdt_prompts.json`):

1. Default: DO NOT speak
2. Speak ONLY if: `conflict` | `safety` | `missing-critical` | `new-critical`
3. Response format:
```json
{"speak":"yes/no","messages":[{"target":"<role>","message":"<1-2 sentences>","why":"conflict|safety|missing|new"}]}
```

### Agent Initialization

`init_expert_agent()` builds each expert's system prompt with:

1. `OUTPATIENT VISIT TIME` - temporal context
2. `CASE_FINGERPRINT` - stable case hash
3. `Role prompt` - from ROLE_PROMPTS
4. `HARD RULES` - evidence tag requirements, patient facts rules
5. `Role-Specific Case View` - from `build_role_specific_case_view()`
6. `Clinical Reports` - filtered by role permissions
7. `GLOBAL Guideline + PubMed Digest` - shared RAG evidence
