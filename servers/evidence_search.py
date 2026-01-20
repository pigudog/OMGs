# =========================================================
# RAG Core (PersistentClient)
# =========================================================
import os
import requests
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, Optional, Tuple, List
import torch
import re
def _init_rag(
    index_dir: str,
    model_name: str = "BAAI/bge-m3",
    device: str = "auto",
    collection_name: str = "chair_chunks",
):
    """
    Initialize RAG system using ChromaDB PersistentClient API.
    
    Note: This implementation is consistent with pdf_to_rag.py and no longer uses
    the deprecated langchain_chroma.Chroma API.
    
    Args:
        index_dir: Directory path to the ChromaDB index
        model_name: Embedding model name (default: "BAAI/bge-m3")
        device: Device for embedding model ("auto", "cuda", or "cpu")
        collection_name: ChromaDB collection name
    
    Returns:
        Tuple of (collection, embedder) for RAG operations
    """

    # ---- device Auto ----
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- embedding ----
    embedder_base = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ---- BGE prefix ----
    class InstructionEmbedder:
        def __init__(self, base):
            self.base = base
            self.q_pref = "Represent this sentence for searching relevant passages: "
            self.p_pref = "Represent this sentence for retrieval: "

        def embed_query(self, text):
            return self.base.embed_query(self.q_pref + text)

        def embed_documents(self, texts):
            return self.base.embed_documents([self.p_pref + t for t in texts])

    embedder = InstructionEmbedder(embedder_base) if re.search("BAAI/bge", model_name) else embedder_base

    # ---- Chroma new API: PersistentClient ----
    os.makedirs(index_dir, exist_ok=True)
    client = PersistentClient(path=index_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    return collection, embedder


def rag_search_pack(
    query: str,
    index_dir: str,
    model_name="BAAI/bge-m3",
    device="auto",
    topk: int = 5,
    collection_name: str = "chair_chunks",
):
    """
    NEW RAG search using PersistentClient
    """
    collection, embedder = _init_rag(
        index_dir=index_dir,
        model_name=model_name,
        device=device,
        collection_name=collection_name
    )

    # ---- embed query ----
    qvec = embedder.embed_query(query)

    # ---- perform search ----
    results = collection.query(
        query_embeddings=[qvec],
        n_results=topk,
        include=["metadatas", "documents", "distances"]
    )

    if not results["documents"] or len(results["documents"][0]) == 0:
        return "(RAG: no evidence found)", []

    lines, raw = [], []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, (text, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        doc_id = meta.get("doc_id", "")
        page = meta.get("page_from")
        page_tag = f"[PAGE {page}]" if isinstance(page, int) else ""
        citation_tag = ""
        if doc_id:
            page_str = str(page) if isinstance(page, int) else "NA"
            citation_tag = f"[@guideline:{doc_id}|{page_str}]"

        snippet = text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"

        score = 1 - float(dist)  # cosine similarity reverse

        lines.append(f"[{i}] score={score:.4f} {doc_id} {page_tag} {citation_tag}\n    {snippet}")

        raw.append({
            "rank": i,
            "score": score,
            "source": "guideline",
            "doc_id": doc_id,
            "page": page,
            "text": text,
        })

    pack = "RAG Evidence Pack (top={}):\n".format(topk) + "\n".join(lines)
    return pack, raw


def pubmed_search_pack(
    query: str,
    topk: int = 5,
    endpoint: str = "http://495ga8uy7084.vicp.fun:17088/api/search-paper",
    timeout: int = 20,
):
    """
    PubMed RAG search via external API.
    Returns formatted pack + raw hits for logging.
    """
    sanitized, _ = sanitize_rag_query(query or "")
    if not sanitized:
        return "(PUBMED: empty query)", []
    try:
        resp = requests.post(endpoint, json={"query": sanitized, "topk": topk}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json() or {}
    except Exception as exc:
        return f"(PUBMED: retrieval failed: {exc})", []

    if payload.get("status") != "success":
        return "(PUBMED: no evidence found)", []

    hits = payload.get("data") or []
    if not hits:
        return "(PUBMED: no evidence found)", []
    if len(hits) < topk:
        print(f"[WARNING] PubMed returned {len(hits)} results (< topk={topk}).")

    lines, raw = [], []
    for i, hit in enumerate(hits[:topk], 1):
        pmid = str(hit.get("pmid") or "").strip()
        title = (hit.get("title") or "").strip()
        abstract = (hit.get("abstract") or "").strip()
        similarity = hit.get("similarity")
        score = float(similarity) if similarity is not None else None

        snippet = abstract.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"

        score_text = f"score={score:.4f} " if isinstance(score, (int, float)) else ""
        citation_tag = f"[@pubmed:{pmid}]" if pmid else ""
        lines.append(f"[{i}] {score_text}PMID {pmid} {citation_tag}\n    {title}\n    {snippet}")

        raw.append({
            "rank": i,
            "score": score,
            "source": "pubmed",
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": hit.get("journal_iso"),
            "pub_date": hit.get("pub_date"),
            "doi": hit.get("doi"),
            "impact_factor": hit.get("impact_factor"),
            "similarity": similarity,
        })

    pack = "PUBMED Evidence Pack (top={}):\n".format(topk) + "\n".join(lines)
    return pack, raw


def merge_rag_packs(guideline_pack: str, pubmed_pack: str) -> str:
    parts = []
    if guideline_pack:
        parts.append("# GUIDELINE RAG\n" + guideline_pack)
    if pubmed_pack:
        parts.append("# PUBMED RAG\n" + pubmed_pack)
    if not parts:
        return "(RAG: no evidence found)"
    return "\n\n".join(parts)


def merge_rag_raw(guideline_raw, pubmed_raw):
    return (guideline_raw or []) + (pubmed_raw or [])

###############################################################################
# RAG Query Builder for MDT
# Generates a concise English RAG query from structured CASE JSON
###############################################################################
def _clean_histology_for_query(histology: str) -> str:
    """
    Clean histology string to extract English terms only, removing Chinese descriptions.
    
    Args:
        histology: Histology string that may contain Chinese text
    
    Returns:
        Cleaned English-only histology string
    """
    if not histology:
        return ""
    
    # Common histology mappings (Chinese to English)
    histology_map = {
        "透明细胞癌": "clear cell carcinoma",
        "浆液性癌": "serous carcinoma",
        "高级别浆液性癌": "high-grade serous carcinoma",
        "低级别浆液性癌": "low-grade serous carcinoma",
        "子宫内膜样癌": "endometrioid carcinoma",
        "黏液性癌": "mucinous carcinoma",
        "未分化癌": "undifferentiated carcinoma",
        "癌肉瘤": "carcinosarcoma",
    }
    
    # Remove common Chinese prefixes/suffixes
    hist_clean = histology.strip()
    
    # Check if it's already mostly English (contains common English histology terms)
    english_terms = ["carcinoma", "cancer", "adenocarcinoma", "serous", "clear cell", 
                     "endometrioid", "mucinous", "undifferentiated", "carcinosarcoma"]
    has_english = any(term.lower() in hist_clean.lower() for term in english_terms)
    
    if has_english:
        # Extract English parts only (remove Chinese characters)
        # Keep alphanumeric, spaces, and common punctuation
        hist_clean = re.sub(r'[^\x00-\x7F]+', ' ', hist_clean)  # Remove non-ASCII
        hist_clean = re.sub(r'\s+', ' ', hist_clean).strip()
        return hist_clean
    
    # Try to map Chinese terms
    for chinese, english in histology_map.items():
        if chinese in hist_clean:
            return english
    
    # If no mapping found and contains Chinese, extract any English words present
    # Otherwise return empty (will be handled by query builder)
    english_words = re.findall(r'[a-zA-Z]+', hist_clean)
    if english_words:
        return ' '.join(english_words)
    
    return ""


def build_rag_query_for_mdt(agent, question: str, key_facts: str | None = None) -> str:
    """
    Generate a concise English RAG query from structured CASE JSON.
    
    Args:
        agent: Agent instance for running the query builder
        question: Structured JSON string containing the full case information
    
    Returns:
        A concise English query string (<=40 words) for RAG retrieval
    """
    from core.config import get_mdt_prompts
    rag_prompts = get_mdt_prompts().get("rag", {})
    
    facts_block = f"# KEY FACTS (from structured case)\n{key_facts}\n\n" if key_facts else ""
    query_builder_template = rag_prompts.get("query_builder",
        "You are preparing a single concise English query to retrieve guideline/clinical evidence "
        "for this ovarian cancer MDT case.\n\n"
        "# STRUCTURED_CASE_TEXT\n{question}\n\n"
        "Write ONE line (<=40 words) focusing on:\n"
        "- tumor type/histology and platinum status;\n"
        "- key metastases / disease extent;\n"
        "- key molecular markers if mentioned (e.g., BRCA/HRD/MSI/PD-L1/ATM);\n"
        "- major clinical constraints (e.g., anemia, organ function, performance).\n"
        "Do NOT mention report_ids, dates, hospital names, or patient identifiers.\n"
        "If KEY FACTS include histology or platinum/genetic status, you MUST include them.\n"
        "Do NOT say 'unknown' if a KEY FACT is provided.\n"
        "Output ONLY the query text."
    )
    prompt = facts_block + query_builder_template.format(question=question)
    raw_query = agent.run_selection(prompt).strip()
    sanitized, changed = sanitize_rag_query(raw_query)
    if changed:
        print("[WARNING] RAG query contained potential identifiers and was sanitized before logging/search.")
    if key_facts:
        hist_match = re.search(r"histology=([^;\n]+)", key_facts, re.IGNORECASE)
        if hist_match:
            hist_raw = hist_match.group(1).strip()
            hist_clean = _clean_histology_for_query(hist_raw)
            
            # Only append if we have a cleaned English histology
            if hist_clean and hist_clean.lower() not in ["unknown", "not specified", ""]:
                if re.search(r"histology\s+(not\s+specified|unknown)", sanitized, re.IGNORECASE):
                    sanitized = re.sub(
                        r"histology\s+(not\s+specified|unknown)",
                        f"histology {hist_clean}",
                        sanitized,
                        flags=re.IGNORECASE,
                    )
                elif re.search(r"\bhistology\b", sanitized, re.IGNORECASE) is None:
                    sanitized = sanitized.rstrip(".;") + f"; histology: {hist_clean}"
    return sanitized


def sanitize_rag_query(query: str) -> tuple[str, bool]:
    """Remove obvious identifiers from RAG query (defensive de-id before logging/search)."""
    if not query:
        return "", False
    original = query
    q = query
    # Emails
    q = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", q)
    # Phones / long digit sequences (avoid removing short clinical numbers like CA-125)
    q = re.sub(r"\b\d{8,}\b", "[REDACTED_ID]", q)
    # Common identifier labels
    q = re.sub(r"(?i)\b(meta[_\s-]?info|patient[_\s-]?id|report[_\s-]?id|mrn|住院号|病历号|身份证号)\s*[:=]\s*\S+", "", q)
    # Collapse extra whitespace
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q, (q != original)


def summarize_rag_evidence(agent, rag_pack: str, rag_raw: List[Dict[str, Any]] = None) -> str:
    """
    Summarize guideline RAG chunks into actionable evidence bullets.
    
    Args:
        agent: Agent instance for running the summarization
        rag_pack: String containing RAG evidence chunks
        rag_raw: Optional list of raw RAG results for counting and reference mapping
    
    Returns:
        Summarized evidence as plain text bullets (one per RAG result, 1:1 mapping)
    """
    from core.config import get_mdt_prompts
    rag_prompts = get_mdt_prompts().get("rag", {})
    
    # Build explicit reference mapping for each RAG result
    total_count = len(rag_raw) if rag_raw else 0
    guideline_count = len([r for r in (rag_raw or []) if r.get("source") == "guideline"])
    pubmed_count = len([r for r in (rag_raw or []) if r.get("source") == "pubmed"])
    
    # Build reference tags list for explicit mapping
    ref_tags_list = []
    for i, r in enumerate(rag_raw or [], 1):
        source = r.get("source", "")
        if source == "guideline":
            doc_id = r.get("doc_id", "")
            page = r.get("page", "NA")
            tag = f"[@guideline:{doc_id}|{page}]"
        elif source == "pubmed":
            pmid = r.get("pmid", "")
            tag = f"[@pubmed:{pmid}]"
        else:
            tag = f"[unknown source {i}]"
        ref_tags_list.append(f"  [{i}] {tag}")
    
    ref_tags_str = "\n".join(ref_tags_list) if ref_tags_list else ""
    
    count_info = ""
    if total_count > 0:
        count_info = f"""
CRITICAL: There are exactly {total_count} RAG results ({guideline_count} guidelines, {pubmed_count} PubMed).
You MUST output exactly {total_count} bullets, one per result, in order.

REFERENCE TAGS (use these EXACTLY):
{ref_tags_str}

Each bullet MUST use the corresponding tag from the list above.
"""
    
    evidence_summarizer_template = rag_prompts.get("evidence_summarizer",
        "# RAG CHUNKS\n{rag_pack}\n\n"
        "{count_info}"
        "Summarize into evidence bullets for MDT decision-making.\n"
        "Rules:\n"
        "- Output exactly {total_count} bullets, one per RAG result, in order.\n"
        "- Each bullet summarizes ONE RAG chunk with its corresponding tag.\n"
        "- Each bullet must be actionable evidence (guideline/trial-based).\n"
        "- Do NOT restate patient-specific facts.\n"
        "- Avoid long quotes; keep each bullet concise (1-2 sentences).\n"
        "- Each bullet MUST include the exact evidence tag from the REFERENCE TAGS list.\n"
        "- Output ONLY plain text bullets, no numbering."
    )
    prompt = evidence_summarizer_template.format(
        rag_pack=rag_pack, 
        count_info=count_info,
        total_count=total_count,
    )
    return agent.run_selection(prompt)

###############################################################################
# 3. RAG CONFIG HELPERS
###############################################################################
def get_rag_config():
    """Get RAG configuration from paths config."""
    from core.config import get_paths_config
    return get_paths_config()["rag_store"]


def get_rag_index_for_role(role: str):
    """
    Get RAG index_dir and collection_name for a specific role.
    
    If use_per_role_rag is False, returns the default role's RAG (chair).
    If use_per_role_rag is True, returns the role-specific RAG.
    
    Args:
        role: MDT role name (chair, oncologist, radiologist, pathologist, nuclear)
    
    Returns:
        Tuple of (index_dir, collection_name, embedding_model)
    """
    rag_config = get_rag_config()
    
    use_per_role = rag_config.get("use_per_role_rag", False)
    available_roles = rag_config.get("available_roles", ["chair"])
    default_role = rag_config.get("default_role", "chair")
    
    # Determine which role's RAG to use
    if use_per_role and role in available_roles:
        target_role = role
    else:
        target_role = default_role
    
    index_dir = rag_config["index_dir_template"].format(role=target_role)
    collection_name = rag_config.get("collection_name_template", "{role}_chunks").format(role=target_role)
    embedding_model = rag_config.get("embedding_model", "BAAI/bge-m3")
    
    return index_dir, collection_name, embedding_model


###############################################################################
# 4. LOAD SPECIALTY-SPECIFIC GUIDELINE RAG
# Supports both global (chair-only) and per-role RAG modes via config
###############################################################################
def get_guideline_rag(role, question, device="auto", topk=5):
    """
    Load guideline RAG for a given role.
    
    Behavior controlled by config/paths.json:
    - use_per_role_rag: false -> All roles use chair's RAG (current default)
    - use_per_role_rag: true  -> Each role uses its own specialty RAG
    
    Args:
        role: MDT role name (chair, oncologist, radiologist, pathologist, nuclear)
        question: Query string for RAG search
        device: Device for embedding model ("auto", "cuda", "cpu")
        topk: Number of top results to return
    
    Returns:
        Formatted RAG evidence string for the role
    """
    index_dir, collection_name, embedding_model = get_rag_index_for_role(role)

    rag_pack, _ = rag_search_pack(
        query=question,
        index_dir=index_dir,
        model_name=embedding_model,
        device=device,
        topk=topk,
        collection_name=collection_name,
    )
    return f"# GUIDELINE RAG for {role}\n{rag_pack}\n"


def get_global_guideline_rag(question, device="auto", topk=5):
    """
    Load global guideline RAG (always uses default_role from config, typically chair).
    
    This is the main entry point used by the MDT pipeline for global guideline retrieval.
    
    Args:
        question: Query string for RAG search
        device: Device for embedding model
        topk: Number of top results to return
    
    Returns:
        Tuple of (rag_pack, rag_raw) - formatted evidence string and raw results
    """
    rag_config = get_rag_config()
    default_role = rag_config.get("default_role", "chair")
    
    index_dir, collection_name, embedding_model = get_rag_index_for_role(default_role)
    
    return rag_search_pack(
        query=question,
        index_dir=index_dir,
        model_name=embedding_model,
        device=device,
        topk=topk,
        collection_name=collection_name,
    )