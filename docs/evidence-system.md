# Open Evidence References System

OMGs implements an **Open Evidence** system that automatically generates a structured References section at the end of each MDT output. This ensures full traceability of clinical recommendations back to their source evidence.

## Overview

The Open Evidence system ensures that every clinical recommendation is backed by traceable evidence, including:
- Clinical guidelines
- PubMed literature
- Clinical trials
- Patient-specific reports (lab, imaging, pathology, genomics)

## Evidence Tag Formats

| Type | Format | Example |
|------|--------|---------|
| **Guidelines** | `[@guideline:doc_id \| Page xx]` | `[@guideline:nccn_ovarian_v3 \| Page 14]` |
| **Literature** | `[@pubmed \| PMID]` | `[@pubmed \| 33758607]` |
| **Clinical Trials** | `[@trial \| id]` | `[@trial \| 350]` |
| **Clinical Reports** | `[@actual_report_id \| LAB/Genomics/MR/CT]` | `[@20220407\|17300673 \| LAB]`, `[@OH2203828\|2022-04-18 \| Genomics]`, `[@2022-12-29 \| MR]`, `[@2022-12-29 \| CT]` |

**Important:** Always use spaces around `|` for consistency: `[@xxx | yyy]`

## Output Example

**Natural Trial Citation:** When a clinical trial is recommended by the assistant, the Chair agent naturally integrates it into the Core Treatment Strategy or Change Triggers with proper `[@trial | id]` citation, rather than appending it at the end. The trial recommendation is passed directly in the final output prompt, allowing the Chair to judge appropriateness and cite naturally.

```
Final Assessment:
Patient with platinum-resistant recurrent ovarian clear cell carcinoma...

Core Treatment Strategy:
- Correct anemia before systemic therapy [@20220407|17300673 | LAB]
- Consider non-platinum palliative chemotherapy [@guideline:nccn_ov | Page 14]
- Pursue clinical trial enrollment if eligible [@trial | 350]

---
## References

### Guidelines
[@guideline:nccn_ov | Page 14]
  Document: nccn_ov, Page 14
  Content: For platinum-resistant disease, consider...

### Literature
[@pubmed | 33758607]
  PMID: 33758607 | J Cancer | 2021
  Title: Updates of Pathogenesis for Ovarian Clear Cell Carcinoma

### Clinical Trials
[@trial | 350]
  Trial ID: 350
  Name: Phase Ib/II study of BL-B01D1 in gynecologic malignancies
  Rationale: Patient meets eligibility for recurrent disease

### Clinical Reports
[@20220407|17300673 | LAB]
  LAB ID: 20220407|17300673 | Date: 2022-04-07
  Content: Hemoglobin 8.2 g/dL (severe anemia)

[@OH2203828|2022-04-18 | Genomics]
  Genomics ID: OH2203828|2022-04-18 | Date: 2022-04-18
  Content: ATM mutation detected...
```

## HTML Report Features

The generated HTML report (`mdt_report_*.html`) includes:

- **📊 Pipeline Execution Statistics** (displayed at the top):
  - **Total Execution Time**: Shown in both seconds and human-readable format (e.g., "45.3 seconds (45.3秒)", "1分25秒")
  - **Total Token Usage**: Breakdown of input tokens, output tokens, and total tokens with thousands separators
  - **Models Used**: Per-model statistics including:
    - Model name with provider information (e.g., "gpt-5.1 (Azure)")
    - Number of API calls
    - Total tokens consumed
    - Input/output token breakdown
  - Statistics are automatically collected from the `api_trace.db` database during pipeline execution
- **Mermaid Pipeline Flowchart**: Visual representation of the MDT workflow
- **Color-coded References**: Each category has distinct styling:
  - 📋 Guidelines (green)
  - 📚 Literature (purple)
  - 🔬 Clinical Trials (red)
  - 📄 Clinical Reports (orange)
- **Interactive Details**: Collapsible sections for evidence, RAG hits, and trace events

## Evidence Digest (Dynamic 1:1 Mapping)

The RAG evidence digest maintains a strict 1:1 correspondence with retrieved results. The system dynamically instructs the digester to produce exactly N bullets for N RAG results:

```python
# Dynamic instruction: "Digest RAG chunks into exactly {rag_count} evidence bullets"
# For 10 RAG results → exactly 10 digest bullets
# Each bullet uses the EXACT citation tag from its source

# Example: 3 RAG results → 3 digest bullets
- Platinum-based chemotherapy is standard first-line... [@guideline:nccn_ov | Page 12]
- PARP inhibitors improve PFS in BRCA-mutated... [@pubmed | 33758607]
- Anti-VEGF therapy option for platinum-sensitive... [@guideline:esmo_ov | Page 10]
```

## Mutation Report Handling (Comprehensive NGS Panel)

OMGs treats mutation reports as **comprehensive NGS panel results** (~20,000 genes) and provides consistent interpretation rules across all components (RAG queries, expert agents, trial matching):

### Interpretation Rules

The system injects the following interpretation rules whenever mutation reports are present:

| Chinese Term | English Meaning | Interpretation |
|--------------|-----------------|----------------|
| `未检出` | Not detected | NO pathogenic mutation found |
| `（视为阴性）` | Considered negative | NO pathogenic mutation found |
| `阴性` | Negative | Negative result |
| `NM_xxx:exon:c.xxx:p.xxx` | Variant notation | POSITIVE mutation detected |
| Gene not mentioned | - | NO pathogenic mutation (comprehensive panel) |

### Key Features

1. **Comprehensive Panel Assumption**: Since mutation reports come from ~20,000 gene NGS panels:
   - Any gene NOT mentioned in the report = NO pathogenic mutation found
   - The system never says "not tested" or "not reported" when a mutation report exists

2. **Universal Application**: Interpretation rules are injected into:
   - RAG query building (`servers/evidence_search.py`)
   - Expert agent context (`host/experts.py`)
   - Clinical trial matching (`host/decision.py`)

3. **Tumor-Agnostic Design**: Unlike previous HRD/BRCA-specific logic, the new approach:
   - Passes full `raw_text` to agents
   - Lets agents focus on relevant genes based on tumor type
   - Works for all cancer types, not just epithelial ovarian cancer

### Example Mutation Report

```
MSH6 NM_001281492:exon3:c.G2971A:p.E991K（胚系）；
BRCA1 NM_007297:exon18:c.C5110T:p.R1704X（胚系）；
BRCA2 胚系和体系未检出致病突变（视为阴性）；
TP53 胚系和体系未检出致病突变（视为阴性）；
HRD 阴性（未检出阳性结果）
```

**Interpretation:**
- MSH6: **POSITIVE** (germline mutation detected)
- BRCA1: **POSITIVE** (germline mutation detected)
- BRCA2: **NEGATIVE** (视为阴性)
- TP53: **NEGATIVE** (视为阴性)
- HRD: **NEGATIVE** (阴性)
- Other genes (e.g., PTEN, PIK3CA): **NEGATIVE** (not mentioned = no mutation)

### Why This Matters

- **Clinical Accuracy**: Experts correctly interpret comprehensive NGS results
- **No False Negatives**: System never incorrectly states "not tested" when reports exist
- **Tumor-Type Flexibility**: Works for ovarian, endometrial, and other cancer types with different gene focuses

## Related Documentation

- [Configuration Guide](../config/README.md) - Evidence tag rules and customization
- [SKILL Protocol](../skills/omgs/SKILL.md) - Evidence tagging requirements
- [Expert Roles](../skills/omgs/references/expert-roles.md) - Role-specific evidence usage
