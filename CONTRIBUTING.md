# Development Guide

This guide covers how to extend OMGs by adding new roles, report types, and other development tasks.

## Adding a New Specialist Role

1. **Define role in `host/experts.py`:**

```python
# Add to ROLES list
ROLES = ["chair", "oncologist", "radiologist", "pathologist", "nuclear", "surgeon"]

# Add permissions
ROLE_PERMISSIONS["surgeon"] = {
    "lab": True,
    "imaging": True,
    "pathology": True,
    "mutation": False,
    "guideline": "surgeon"
}

# Add role prompt
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

2. **Update `servers/info_delivery.py`:**

```python
def build_role_specific_case_view(role: str, case_json: Dict[str, Any]) -> str:
    # ... existing code ...
    
    if role == "surgeon":
        return json.dumps({
            "SURGICAL_HISTORY": case_json.get("SURGICAL_HISTORY", {}),
            "IMAGING_FOR_RESECTABILITY": radiology.get("studies", []),
            "PATHOLOGY": pathology
        }, ensure_ascii=False, indent=2)
```

3. **Build RAG index for new role:**

```bash
python pdf_to_rag.py build --pdf_dir rag_pdf/surgeon --out_dir rag_store/surgeon/corpus
python pdf_to_rag.py index --corpus_dir rag_store/surgeon/corpus --index_dir rag_store/surgeon/index/chroma
```

## Adding a New Report Type

1. **Add loader in `servers/reports_selector.py`:**

```python
def load_patient_genetics(meta_info: str, json_path: str) -> Tuple[List, List]:
    # Similar to load_patient_labs
    ...
```

2. **Extend permissions in `host/experts.py`:**

```python
ROLE_PERMISSIONS["oncologist"]["genetics"] = True
```

3. **Update `select_reports_for_roles()` to include new type.**

## Running Tests

```bash
# Syntax check all modules
python -m py_compile main.py
python -m py_compile host/orchestrator.py
python -m py_compile host/decision.py

# Import test
python -c "from host import process_omgs_multi_expert_query; print('OK')"
```

## Related Documentation

- [Extension Guide](skills/omgs/references/extension-guide.md) - Detailed extension instructions
- [Expert Roles](skills/omgs/references/expert-roles.md) - Role definitions and permissions
- [Architecture](skills/omgs/references/architecture.md) - System architecture
- [Pipeline Operations](skills/omgs/references/pipeline-ops.md) - CLI and debugging
