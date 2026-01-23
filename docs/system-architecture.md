# OMGs System Architecture

## Overview

OMGs follows a three-layer architecture (host/ servers/ core/) with multi-agent collaboration for MDT decision support.

![OMGs Multi-Agent Architecture](../draw.png)

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Input [Input Layer]
        CaseData[Case Data JSONL]
        Reports[Clinical Reports]
        Guidelines[Guideline PDFs]
    end
    
    subgraph Core [Core Infrastructure]
        Agent[Agent Class]
        Client[Multi-Provider LLM Client]
        Config[Configuration]
    end
    
    subgraph Servers [Agent Servers - Service Layer]
        CaseParser[Case Parser]
        InfoDelivery[Info Delivery]
        EvidenceSearch[Evidence Search]
        ReportsSelector[Reports Selector]
        Trace[Trace Logger]
    end
    
    subgraph Host [Central Host - Orchestration Layer]
        Orchestrator[Orchestrator]
        Experts[Expert Agents]
        Decision[Decision Maker]
    end
    
    subgraph Skill [SKILL Protocol]
        SkillMD[SKILL.md]
        SkillLoader[skill_loader.py]
    end
    
    subgraph Output [Output Layer]
        JSON[JSON Results]
        HTML[HTML Report]
        Logs[MDT Logs]
    end
    
    Input --> Core
    Core --> Servers
    Servers --> Host
    Skill --> Experts
    Host --> Output
```

### Layer Descriptions

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| **Input** | Case data, reports, guidelines | Data ingestion |
| **Core** | Agent class, LLM client, config | Foundation infrastructure |
| **Servers** | Case parser, evidence search, report selector | Service layer operations |
| **Host** | Orchestrator, experts, decision maker | Multi-agent coordination |
| **Skill** | SKILL.md, skill loader | Runtime skill injection |
| **Output** | JSON, HTML, logs | Result generation |

---

## Detailed Pipeline Flow

```mermaid
flowchart TD
    Start([Input Case]) --> Load[1. Load Case & Reports]
    
    Load --> Filter[2. Filter Reports by Visit Time]
    Filter --> Select[3. Role-Based Report Selection]
    
    Select --> RAG[4. Guideline & PubMed RAG]
    RAG --> Digest[5. Generate Evidence Digest]
    
    Digest --> Init[6. Initialize Expert Agents]
    Init --> Views[7. Build Role-Specific Views]
    
    Views --> Discussion{8. MDT Discussion Engine}
    
    Discussion --> |Round 1| R1T1[Turn 1: Initial Opinions]
    R1T1 --> R1T2[Turn 2: Cross-Expert Debate]
    R1T2 --> R1Final[Round 1 Final Plans]
    
    R1Final --> |Round 2| R2T1[Turn 1: Refined Discussion]
    R2T1 --> R2T2[Turn 2: Consensus Building]
    R2T2 --> R2Final[Round 2 Final Plans]
    
    R2Final --> Trial[9. Clinical Trial Matching]
    Trial --> Final[10. Chair Final Synthesis]
    
    Final --> Save[11. Save Artifacts]
    Save --> End([Output Results])
```

### Pipeline Stages

1. **Load Case & Reports**: Parse input JSONL, load clinical reports
2. **Filter Reports**: Time-based filtering for visit context
3. **Role-Based Selection**: LLM selects relevant reports per expert role
4. **RAG Retrieval**: Query guidelines and PubMed literature
5. **Evidence Digest**: Summarize RAG results into actionable bullets
6. **Initialize Agents**: Create expert agents with role-specific prompts
7. **Build Views**: Construct role-specific patient fact views
8. **MDT Discussion**: 2-round × 2-turn deliberation
9. **Trial Matching**: Optional clinical trial recommendation
10. **Final Synthesis**: Chair generates structured decision
11. **Save Artifacts**: Output JSON, HTML, logs

---

## Clinical Workflow Integration

```mermaid
flowchart LR
    subgraph Clinical_Workflow [Clinical Workflow]
        EHR[EHR System] --> Extract[OMGs: Extract & Structure]
        Extract --> MDT[OMGs: MDT Discussion]
        MDT --> Decision[MDT Decision]
        Decision --> Treatment[Treatment Plan]
    end
    
    subgraph Evidence [Evidence Sources]
        Guidelines[Clinical Guidelines]
        PubMed[PubMed Literature]
        Reports[Patient Reports]
    end
    
    Evidence --> MDT
```

### EHR Extraction Pipeline

```
Raw EHR Text → Extract → Self-Review → Validator-Review
                                      ↓
                              [Fixable Issues?]
                                      ↓ Yes
                              Refine → Re-Review (max 2x)
                                      ↓ No
                              Auto-Fix → Structured JSON
```

See [Prompts Reference](prompts-reference.md#ehr-extraction-prompts) for details.

---

## Roles and Permissions Matrix

| Role | Lab Reports | Imaging Reports | Pathology Reports | Mutation Reports | Primary Focus |
|------|:-----------:|:---------------:|:-----------------:|:----------------:|---------------|
| **Chair** | ✅ | ✅ | ❌ | ✅ | Overall synthesis & safety |
| **Oncologist** | ✅ | ❌ | ❌ | ✅ | Systemic therapy planning |
| **Radiologist** | ❌ | ✅ | ❌ | ❌ | Disease distribution & imaging |
| **Pathologist** | ❌ | ❌ | ✅ | ✅ | Histology & molecular markers |
| **Nuclear Medicine** | ❌ | ✅ | ❌ | ❌ | PET/metabolic findings |

### Permission Design Rationale

- **Separation of Concerns**: Each expert only sees relevant data
- **Reduced Hallucination**: Limited context prevents over-interpretation
- **MDT Simulation**: Mimics real-world MDT where specialists focus on their domain

---

## Agent Modes Architecture

```mermaid
flowchart TD
    Input[Case Input] --> Auto{auto mode?}
    Auto -->|Yes| Router[Complexity Router]
    Auto -->|No| Direct[Direct Mode]
    
    Router -->|Simple| SA[chair_sa]
    Router -->|Medium| SAK[chair_sa_k]
    Router -->|Complex| SAKEP[chair_sa_kep]
    Router -->|Very Complex| OMGS[omgs]
    
    Direct --> SA
    Direct --> SAK
    Direct --> SAKEP
    Direct --> OMGS
    
    SA --> Output[Results]
    SAK --> Output
    SAKEP --> Output
    OMGS --> Output
```

### Mode Comparison

| Mode | Agents | Knowledge | Evidence Pack | Use Case |
|------|--------|-----------|---------------|----------|
| `chair_sa` | 1 | ❌ | ❌ | Testing |
| `chair_sa_k` | 1 | ✅ | ❌ | Evidence reference |
| `chair_sa_kep` | 1 | ✅ | ✅ | Complex with data |
| `omgs` | 5 | ✅ | ✅ | Full MDT |

---

## Related Documentation

- **[Architecture Details](../skills/omgs/references/architecture.md)** - Three-layer architecture deep dive
- **[Expert Roles](../skills/omgs/references/expert-roles.md)** - Role definitions and permissions
- **[Prompts Reference](prompts-reference.md)** - Prompt system documentation
- **[Extension Guide](../skills/omgs/references/extension-guide.md)** - Adding new roles and components
