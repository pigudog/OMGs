# Extension Development Guide

## Table of Contents

1. [Adding a New Expert Role](#adding-a-new-expert-role)
2. [Adding a New Report Type](#adding-a-new-report-type)
3. [Building Role-Specific RAG Index](#building-role-specific-rag-index)
4. [Customizing Evidence Tags](#customizing-evidence-tags)

---

## Adding a New Expert Role

### Step 1: Define Role in `host/experts.py`

```python
# 1. Add to ROLES list
ROLES = ["chair", "oncologist", "radiologist", "pathologist", "nuclear", "surgeon"]

# 2. Add permissions
ROLE_PERMISSIONS["surgeon"] = {
    "lab": True,
    "imaging": True,
    "pathology": True,
    "mutation": False,
    "guideline": "surgeon"  # RAG index name
}

# 3. Add role prompt
ROLE_PROMPTS["surgeon"] = """
# Context
You are the surgical oncologist. You evaluate surgical candidacy and operative findings.

# Objective
Assess resectability, surgical history, and operative recommendations.

# Constraints
- Surgical assessment only
- No drug names or systemic therapy recommendations

# Style
Speak as the specialist (first person). Use professional, clinical tone.
Return up to 3 bullets. Each ≤20 words.
""".strip()
```

### Step 2: Update `servers/info_delivery.py`

Add case view builder for new role:

```python
def build_role_specific_case_view(role: str, case_json: Dict[str, Any]) -> str:
    # ... existing code ...
    
    if role == "surgeon":
        return json.dumps({
            "SURGICAL_HISTORY": case_json.get("SURGICAL_HISTORY", {}),
            "IMAGING_FOR_RESECTABILITY": case_json.get("RADIOLOGY", {}).get("studies", []),
            "PATHOLOGY": case_json.get("PATHOLOGY", {})
        }, ensure_ascii=False, indent=2)
```

### Step 3: Build RAG Index (Optional)

```bash
# Build corpus from PDFs
python pdf_to_rag.py build \
  --pdf_dir rag_pdf/surgeon \
  --out_dir rag_store/surgeon/corpus

# Index into ChromaDB
python pdf_to_rag.py index \
  --corpus_dir rag_store/surgeon/corpus \
  --index_dir rag_store/surgeon/index/chroma \
  --collection_name surgeon_chunks
```

---

## Adding a New Report Type

### Step 1: Add Loader in `servers/reports_selector.py`

```python
def load_patient_genetics(
    meta_info: str, 
    json_path: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load genetics reports for a patient.
    Returns: (timeline_entries, full_reports)
    """
    if not json_path or not os.path.exists(json_path):
        return [], []
    
    with open(json_path, "r", encoding="utf-8") as f:
        all_reports = [json.loads(line) for line in f]
    
    patient_reports = [r for r in all_reports if r.get("patient_id") == meta_info]
    
    timeline = []
    for r in patient_reports:
        timeline.append({
            "date": r.get("report_date"),
            "type": "genetics",
            "summary": r.get("summary", "")
        })
    
    return timeline, patient_reports
```

### Step 2: Extend Permissions in `host/experts.py`

```python
ROLE_PERMISSIONS["oncologist"]["genetics"] = True
ROLE_PERMISSIONS["pathologist"]["genetics"] = True
```

### Step 3: Update `select_reports_for_roles()`

In `servers/reports_selector.py`, add handling for new report type:

```python
def select_reports_for_roles(
    # ... existing params ...
    genetics_timeline: List[Dict] = None,
    genetics_reports: List[Dict] = None,
):
    # ... existing code ...
    
    # Add genetics handling
    if genetics_reports:
        for role in roles:
            perm = role_permissions[role]
            if perm.get("genetics", False):
                context["genetics"][role] = genetics_reports
```

### Step 4: Update Pipeline in `host/orchestrator.py`

```python
# In process_omgs_multi_expert_query():

# Load new report type
gen_timeline, gen_reports = load_patient_genetics(meta_info, genetics_json)
gen_reports = filter_before(gen_reports, "report_date", cutoff_dt)

# Pass to selection
context = select_reports_for_roles(
    # ... existing params ...
    genetics_timeline=gen_timeline,
    genetics_reports=gen_reports,
)
```

---

## Building Role-Specific RAG Index

### Directory Structure

```
rag_store/
├── chair/
│   ├── corpus/
│   │   ├── chunks/         # Chunked text
│   │   ├── meta/           # Document metadata
│   │   └── staging_txt/    # Raw extracted text
│   └── index/
│       └── chroma/         # ChromaDB index
├── oncologist/
├── radiologist/
├── pathologist/
└── nuclear/
```

### Build Commands

```bash
# Step 1: Build chunks from PDFs
python pdf_to_rag.py build \
  --pdf_dir rag_pdf/<role> \
  --out_dir rag_store/<role>/corpus \
  --chunk_size 1200 \
  --chunk_overlap 200

# Step 2: Index into ChromaDB
python pdf_to_rag.py index \
  --corpus_dir rag_store/<role>/corpus \
  --index_dir rag_store/<role>/index/chroma \
  --collection_name <role>_chunks \
  --model BAAI/bge-m3 \
  --device cpu

# Step 3: Test search
python pdf_to_rag.py search \
  --index_dir rag_store/<role>/index/chroma \
  --collection_name <role>_chunks \
  --model BAAI/bge-m3 \
  --query "ovarian cancer treatment"
```

### Enable Per-Role RAG

In `config/paths.json`:

```json
{
  "rag_store": {
    "use_per_role_rag": true,
    "default_role": "chair",
    "available_roles": ["chair", "oncologist", "radiologist", "pathologist", "nuclear"]
  }
}
```

---

## Customizing Evidence Tags

### Tag Extraction Logic

Source: `utils/reference_cache.py` - `extract_reference_tags()`

Supported patterns:
- `[@guideline:doc_id|page]` - Guideline
- `[@pubmed:PMID]` - PubMed
- `[@trial:id]` - Clinical trial
- `[@report_id|date]` - Clinical report

### Adding New Tag Type

1. Update regex patterns in `extract_reference_tags()`:

```python
# In utils/reference_cache.py
def extract_reference_tags(text: str) -> Dict[str, List[str]]:
    patterns = {
        "guideline": r'\[@guideline:([^\]]+)\]',
        "pubmed": r'\[@pubmed:(\d+)\]',
        "trial": r'\[@trial:([^\]]+)\]',
        "report": r'\[@([A-Za-z0-9_-]+)\|(\d{4}-\d{2}-\d{2})\]',
        # Add new type
        "custom": r'\[@custom:([^\]]+)\]',
    }
```

2. Update `build_references_section()` to format new type:

```python
def build_references_section(...):
    # ... existing code ...
    
    if refs.get("custom"):
        lines.append("\n### Custom References")
        for ref in refs["custom"]:
            lines.append(f"[@custom:{ref}]")
            lines.append(f"  Custom ID: {ref}")
```

3. Update HTML renderer in `servers/reporters.py` - `_render_final_output_html()`
