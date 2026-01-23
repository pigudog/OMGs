# Contributing to OMGs

Thank you for your interest in contributing to OMGs! This guide covers how to extend the system.

## Quick Links

| Task | Documentation |
|------|--------------|
| **Add New Role** | [Extension Guide](../skills/omgs/references/extension-guide.md#adding-a-new-expert-role) |
| **Add Report Type** | [Extension Guide](../skills/omgs/references/extension-guide.md#adding-a-new-report-type) |
| **Build RAG Index** | [Extension Guide](../skills/omgs/references/extension-guide.md#building-role-specific-rag-index) |
| **Architecture** | [Architecture Details](../skills/omgs/references/architecture.md) |
| **Expert Roles** | [Role Definitions](../skills/omgs/references/expert-roles.md) |

---

## Common Development Tasks

### Adding a New Specialist Role

1. Add role to `host/experts.py`:
   - Add to `ROLES` list
   - Define `ROLE_PERMISSIONS`
   - Define `ROLE_PROMPTS`

2. Update `servers/info_delivery.py`:
   - Add role-specific case view builder

3. Build RAG index:
   ```bash
   python pdf_to_rag.py build --pdf_dir rag_pdf/{role} --out_dir rag_store/{role}/corpus
   python pdf_to_rag.py index --corpus_dir rag_store/{role}/corpus --index_dir rag_store/{role}/index/chroma
   ```

📖 **Detailed instructions**: [Extension Guide](../skills/omgs/references/extension-guide.md#adding-a-new-expert-role)

### Adding a New Report Type

1. Add loader in `servers/reports_selector.py`
2. Extend permissions in `host/experts.py`
3. Update `select_reports_for_roles()` function

📖 **Detailed instructions**: [Extension Guide](../skills/omgs/references/extension-guide.md#adding-a-new-report-type)

---

## Testing

```bash
# Syntax check
python -m py_compile main.py
python -m py_compile host/orchestrator.py

# Import test
python -c "from host import process_omgs_multi_expert_query; print('OK')"
```

---

## Development Resources

- **[Extension Guide](../skills/omgs/references/extension-guide.md)** - Complete step-by-step instructions
- **[Architecture](../skills/omgs/references/architecture.md)** - System design and layers
- **[Expert Roles](../skills/omgs/references/expert-roles.md)** - Role definitions and permissions
- **[Pipeline Operations](../skills/omgs/references/pipeline-ops.md)** - CLI and debugging
- **[Prompts Reference](prompts-reference.md)** - Prompt system documentation

---

## Questions?

- Open a GitHub Issue
- Check existing documentation in `docs/` and `skills/omgs/references/`
