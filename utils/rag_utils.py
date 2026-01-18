# =========================================================
# RAG Core (PersistentClient)
# =========================================================
import os
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
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

        snippet = text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"

        score = 1 - float(dist)  # cosine similarity reverse

        lines.append(f"[{i}] score={score:.4f} {doc_id} {page_tag}\n    {snippet}")

        raw.append({
            "rank": i,
            "score": score,
            "doc_id": doc_id,
            "page": page,
            "text": text,
        })

    pack = "RAG Evidence Pack (top={}):\n".format(topk) + "\n".join(lines)
    return pack, raw

###############################################################################
# RAG Query Builder for MDT
# Generates a concise English RAG query from structured CASE JSON
###############################################################################
def build_rag_query_for_mdt(agent, question: str) -> str:
    """
    Generate a concise English RAG query from structured CASE JSON.
    
    Args:
        agent: Agent instance for running the query builder
        question: Structured JSON string containing the full case information
    
    Returns:
        A concise English query string (<=40 words) for RAG retrieval
    """
    prompt = (
        "You are preparing a single concise English query to retrieve guideline/clinical evidence "
        "for this ovarian cancer MDT case.\n\n"
        "# STRUCTURED_CASE_TEXT\n"
        f"{question}\n\n"
        "Write ONE line (<=40 words) focusing on:\n"
        "- tumor type/histology and platinum status;\n"
        "- key metastases / disease extent;\n"
        "- key molecular markers if mentioned (e.g., BRCA/HRD/MSI/PD-L1/ATM);\n"
        "- major clinical constraints (e.g., anemia, organ function, performance).\n"
        "Do NOT mention report_ids, dates, hospital names, or patient identifiers.\n"
        "Output ONLY the query text.\n"
    )
    return agent.run_selection(prompt)


def summarize_rag_evidence(agent, rag_pack: str) -> str:
    """
    Summarize guideline RAG chunks into actionable evidence bullets.
    
    Args:
        agent: Agent instance for running the summarization
        rag_pack: String containing RAG evidence chunks
    
    Returns:
        Summarized evidence as plain text bullets (<=8 bullets)
    """
    prompt = (
        "# RAG CHUNKS\n"
        f"{rag_pack}\n\n"
        "Summarize into <=8 bullets for MDT decision-making.\n"
        "Rules:\n"
        "- Each bullet must be actionable evidence (guideline/trial-based).\n"
        "- Do NOT restate patient-specific facts.\n"
        "- Avoid long quotes.\n"
        "- If sources have ids/metadata, keep them inline.\n"
        "- Output ONLY plain text bullets.\n"
    )
    return agent.run_selection(prompt)

###############################################################################
# 3. RAG CONFIG HELPERS
###############################################################################
def get_rag_config():
    """Get RAG configuration from paths config."""
    from .core import get_paths_config
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