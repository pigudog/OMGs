# ============ RAG Core (PersistentClient)
# =======================================
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
# 3. LOAD SPECIALTY-SPECIFIC GUIDELINE RAG
# Note: This interface is preserved but currently not called per-role
###############################################################################
def get_guideline_rag(role, question, device="auto", topk=5):
    gl_role = ROLE_PERMISSIONS[role]["guideline"]
    index_dir = f"rag_store/{gl_role}/index/chroma"
    collection_name = f"{gl_role}_chunks"

    rag_pack, _ = rag_search_pack(
        query=question,
        index_dir=index_dir,
        model_name="BAAI/bge-m3",
        device=device,
        topk=topk,
        collection_name=collection_name,
    )
    return f"# GUIDELINE RAG for {role}\n{rag_pack}\n"