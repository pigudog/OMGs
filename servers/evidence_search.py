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
import threading
import time
import json
def _init_rag(
    index_dir: str,
    model_name: str = "BAAI/bge-m3",
    device: str = "auto",
    collection_name: str = "chair_chunks",
    timeout_seconds: int = 30,
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
        timeout_seconds: Maximum time to wait for model initialization (default: 30)
    
    Returns:
        Tuple of (collection, embedder) for RAG operations
    
    Raises:
        RuntimeError: If model download fails or initialization times out
    """

    # ---- device Auto ----
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- embedding ----
    # #region debug log
    with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
        import json
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"evidence_search.py:43","message":"_init_rag entry","data":{"model_name":model_name,"timeout_seconds":timeout_seconds},"timestamp":int(time.time()*1000)})+"\n")
    # #endregion
    
    # Check if model is cached locally first to avoid unnecessary network calls
    try:
        from huggingface_hub import snapshot_download, HfFolder
        cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        model_cache_path = os.path.join(cache_dir, "hub", f"models--{model_name.replace('/', '--')}")
        model_is_cached = os.path.exists(model_cache_path) and os.path.isdir(model_cache_path)
        # #region debug log
        with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"evidence_search.py:50","message":"model cache check","data":{"model_is_cached":model_is_cached,"cache_path":model_cache_path},"timestamp":int(time.time()*1000)})+"\n")
        # #endregion
    except ImportError:
        # If huggingface_hub is not available, assume model is not cached
        model_is_cached = False
        # #region debug log
        with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"evidence_search.py:52","message":"huggingface_hub import failed","data":{},"timestamp":int(time.time()*1000)})+"\n")
        # #endregion
    
    # Set environment variables to reduce HuggingFace retries and timeout
    # This helps prevent infinite retries when network is unavailable
    original_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
    original_offline = os.environ.get("HF_HUB_OFFLINE")
    
    try:
        # Set aggressive timeout and FORCE offline mode if network check fails
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout_seconds)
        os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
        # #region debug log
        with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"evidence_search.py:61","message":"env vars set","data":{"HF_HUB_DOWNLOAD_TIMEOUT":str(timeout_seconds)},"timestamp":int(time.time()*1000)})+"\n")
        # #endregion
        
        # If model is not cached, try a quick network check before attempting download
        network_available = True
        if not model_is_cached:
            # Quick network connectivity check
            try:
                test_response = requests.get("https://huggingface.co", timeout=2)
                network_available = test_response.status_code == 200
                # #region debug log
                with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"evidence_search.py:68","message":"network check result","data":{"network_available":network_available,"status_code":test_response.status_code},"timestamp":int(time.time()*1000)})+"\n")
                # #endregion
                if not network_available:
                    raise RuntimeError(
                        f"Model '{model_name}' is not cached locally and HuggingFace is not accessible. "
                        f"RAG retrieval will be skipped. Please check your network connection or pre-download the model."
                    )
            except (requests.exceptions.RequestException, ConnectionError, OSError) as e:
                network_available = False
                # #region debug log
                with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"evidence_search.py:74","message":"network check failed","data":{"error":str(e)},"timestamp":int(time.time()*1000)})+"\n")
                # #endregion
                # FORCE offline mode to prevent any download attempts
                os.environ["HF_HUB_OFFLINE"] = "1"
                raise RuntimeError(
                    f"Model '{model_name}' is not cached locally and network is unavailable. "
                    f"RAG retrieval will be skipped. Original error: {e}"
                ) from e
        
        # CRITICAL: Monkey patch requests to disable retries BEFORE starting thread
        # This prevents HuggingFace from retrying in background threads
        # Save original implementations
        original_HTTPAdapter_init = requests.adapters.HTTPAdapter.__init__
        original_Session_init = requests.Session.__init__
        
        def patched_HTTPAdapter_init(self, *args, **kwargs):
            # Force max_retries=0 to disable all retries
            kwargs['max_retries'] = 0
            return original_HTTPAdapter_init(self, *args, **kwargs)
        
        def patched_Session_init(self, *args, **kwargs):
            original_Session_init(self, *args, **kwargs)
            # Mount no-retry adapters
            no_retry_adapter = requests.adapters.HTTPAdapter(max_retries=0)
            self.mount('http://', no_retry_adapter)
            self.mount('https://', no_retry_adapter)
        
        # Apply patches globally (affects all requests in this process)
        requests.adapters.HTTPAdapter.__init__ = patched_HTTPAdapter_init
        requests.Session.__init__ = patched_Session_init
        
        # #region debug log
        with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"evidence_search.py:120","message":"requests patched globally","data":{},"timestamp":int(time.time()*1000)})+"\n")
        # #endregion
        
        # Use threading with timeout to prevent hanging
        # Store patch state for cleanup
        result_container = {"embedder": None, "error": None, "completed": False, "patch_applied": True}
        
        def init_embedder():
            try:
                # #region debug log
                with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"evidence_search.py:130","message":"init_embedder thread started","data":{},"timestamp":int(time.time()*1000)})+"\n")
                # #endregion
                
                embedder_base = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": device, "trust_remote_code": True},
                    encode_kwargs={"normalize_embeddings": True},
                )
                
                result_container["embedder"] = embedder_base
                result_container["completed"] = True
                # #region debug log
                with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"evidence_search.py:142","message":"init_embedder completed","data":{},"timestamp":int(time.time()*1000)})+"\n")
                # #endregion
            except Exception as e:
                result_container["error"] = e
                result_container["completed"] = True
                # #region debug log
                with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"evidence_search.py:148","message":"init_embedder error","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)})+"\n")
                # #endregion
        
        # Start initialization in a separate thread with timeout
        init_thread = threading.Thread(target=init_embedder, daemon=True)
        # #region debug log
        with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"evidence_search.py:152","message":"thread start","data":{"timeout_seconds":timeout_seconds},"timestamp":int(time.time()*1000)})+"\n")
        # #endregion
        init_thread.start()
        init_thread.join(timeout=timeout_seconds)
        
        # Check if initialization completed or timed out
        if not result_container["completed"]:
            # Timeout occurred - FORCE offline mode to stop any ongoing downloads
            os.environ["HF_HUB_OFFLINE"] = "1"
            # #region debug log
            with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"evidence_search.py:180","message":"timeout occurred","data":{"thread_alive":init_thread.is_alive()},"timestamp":int(time.time()*1000)})+"\n")
            # #endregion
            # DO NOT restore requests behavior here - keep patch active so daemon thread
            # continues with no-retry behavior. It will fail quickly instead of retrying 5 times.
            raise RuntimeError(
                f"Embedding model '{model_name}' initialization timed out after {timeout_seconds} seconds. "
                f"This is likely due to network issues accessing HuggingFace. "
                f"RAG retrieval will be skipped."
            )
        
        if result_container["error"] is not None:
            # Initialization failed - FORCE offline mode
            os.environ["HF_HUB_OFFLINE"] = "1"
            # #region debug log
            with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"evidence_search.py:193","message":"init error handling","data":{"error":str(result_container["error"])},"timestamp":int(time.time()*1000)})+"\n")
            # #endregion
            # DO NOT restore requests behavior here - keep patch active
            error = result_container["error"]
            if isinstance(error, (requests.exceptions.RequestException, ConnectionError, OSError)):
                raise RuntimeError(
                    f"Failed to download/initialize embedding model '{model_name}' due to network issues. "
                    f"RAG retrieval will be skipped. Original error: {error}"
                ) from error
            else:
                raise RuntimeError(
                    f"Failed to initialize embedding model '{model_name}'. "
                    f"Original error: {error}"
                ) from error
        
        # Success case: restore original requests behavior
        # Only restore on success since daemon thread is done
        try:
            requests.adapters.HTTPAdapter.__init__ = original_HTTPAdapter_init
            requests.Session.__init__ = original_Session_init
        except:
            pass  # Ignore errors during restoration
        
        embedder_base = result_container["embedder"]
        # #region debug log
        with open("/Users/pigudogzyy/Documents/PythonProject/OMGs/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"evidence_search.py:191","message":"_init_rag success","data":{},"timestamp":int(time.time()*1000)})+"\n")
        # #endregion
    finally:
        # Restore original environment variables
        # But keep HF_HUB_OFFLINE=1 if we had an error to prevent background retries
        if original_timeout is not None:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = original_timeout
        elif "HF_HUB_DOWNLOAD_TIMEOUT" in os.environ:
            del os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]
        
        # Only restore HF_HUB_OFFLINE if we didn't set it due to an error
        # This prevents HuggingFace from continuing retries in background
        if "HF_HUB_OFFLINE" not in os.environ or os.environ.get("HF_HUB_OFFLINE") != "1":
            if original_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_offline
            elif "HF_HUB_OFFLINE" in os.environ:
                del os.environ["HF_HUB_OFFLINE"]
        
        # Note: We intentionally do NOT restore requests behavior in error/timeout cases
        # This ensures the daemon thread (if still running) continues with no-retry behavior
        # The patch will remain active until the process exits or the daemon thread completes
        # This prevents HuggingFace from retrying 5 times in the background

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
    # Create directory if it doesn't exist, handle errors gracefully
    try:
        os.makedirs(index_dir, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise RuntimeError(
            f"Failed to create RAG index directory '{index_dir}': {e}. "
            f"Please check permissions or create the directory manually."
        ) from e
    
    try:
        client = PersistentClient(path=index_dir)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChromaDB at '{index_dir}': {e}. "
            f"RAG retrieval will be skipped."
        ) from e

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
    
    Returns:
        Tuple of (pack_string, raw_results_list)
        If initialization fails, returns ("(RAG: initialization failed)", [])
    """
    try:
        collection, embedder = _init_rag(
            index_dir=index_dir,
            model_name=model_name,
            device=device,
            collection_name=collection_name
        )
    except RuntimeError as e:
        # Network or initialization failure: return empty results immediately
        error_msg = str(e)
        if "network" in error_msg.lower() or "connection" in error_msg.lower():
            print(f"[WARNING] RAG initialization failed due to network issues. Skipping RAG retrieval.")
        else:
            print(f"[WARNING] RAG initialization failed: {e}")
        return "(RAG: initialization failed)", []
    except Exception as e:
        # Any other unexpected error: also fail fast
        print(f"[WARNING] RAG initialization failed: {e}")
        return "(RAG: initialization failed)", []

    # ---- embed query ----
    try:
        qvec = embedder.embed_query(query)
    except Exception as e:
        print(f"[WARNING] RAG query embedding failed: {e}")
        return "(RAG: embedding failed)", []

    # ---- perform search ----
    try:
        results = collection.query(
            query_embeddings=[qvec],
            n_results=topk,
            include=["metadatas", "documents", "distances"]
        )
    except Exception as e:
        print(f"[WARNING] RAG search failed: {e}")
        return "(RAG: search failed)", []

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
            citation_tag = f"[@guideline:{doc_id} | Page {page_str}]"

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
        citation_tag = f"[@pubmed | {pmid}]" if pmid else ""
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
    
    # Common histology mappings (Chinese to English for data compatibility)
    # Chinese terms are kept for processing bilingual data sources
    histology_map = {
        "透明细胞癌": "clear cell carcinoma",  # Chinese: "clear cell carcinoma"
        "浆液性癌": "serous carcinoma",  # Chinese: "serous carcinoma"
        "高级别浆液性癌": "high-grade serous carcinoma",  # Chinese: "high-grade serous carcinoma"
        "低级别浆液性癌": "low-grade serous carcinoma",  # Chinese: "low-grade serous carcinoma"
        "子宫内膜样癌": "endometrioid carcinoma",  # Chinese: "endometrioid carcinoma"
        "黏液性癌": "mucinous carcinoma",  # Chinese: "mucinous carcinoma"
        "未分化癌": "undifferentiated carcinoma",  # Chinese: "undifferentiated carcinoma"
        "癌肉瘤": "carcinosarcoma",  # Chinese: "carcinosarcoma"
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

# ！！！ key for evidence search!!!!
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
    
    # IMPORTANT: If MUTATION_REPORT exists, ignore GENETICS section from case_core
    # Mutation reports are the source of truth - case_core may have "not reported" even when reports exist
    has_mutation_report = key_facts and "MUTATION_REPORT" in key_facts
    
    # Add mutation report interpretation guidance if mutation report is present
    mutation_guidance = ""
    if has_mutation_report:
        # Extract the full raw_text from mutation report
        mut_report_raw = ""
        if key_facts:
            mut_match = re.search(r'MUTATION_REPORT:.*?full_text=([^\n]+)', key_facts, re.DOTALL)
            if mut_match:
                mut_report_raw = mut_match.group(1).strip()
                print(f"mut_report_raw: {mut_report_raw}")
        
        # Build comprehensive mutation guidance with raw text and interpretation rules
        mutation_guidance = "\n⚠️ COMPREHENSIVE NGS GENETIC TEST RESULTS:\n"
        mutation_guidance += "This is a ~20,000 gene NGS panel report. The raw text is:\n"
        mutation_guidance += f'"""{mut_report_raw}"""\n\n'
        mutation_guidance += "INTERPRETATION RULES (CRITICAL):\n"
        mutation_guidance += "• '未检出' (not detected) = NO pathogenic mutation found\n"
        mutation_guidance += "• '（视为阴性）' (considered negative) = NO pathogenic mutation found\n"
        mutation_guidance += "• '阴性' (negative) = negative result\n"
        mutation_guidance += "• If a gene of interest is NOT mentioned in the report, it means NO pathogenic mutation (comprehensive panel)\n"
        mutation_guidance += "• Genes with specific variants listed (e.g., 'NM_xxx:exon:c.xxx:p.xxx') = POSITIVE mutation detected\n\n"
        mutation_guidance += "🚨 Include relevant genetic findings in your query based on the tumor type.\n"
        mutation_guidance += "🚨 NEVER say 'not tested' or 'not reported' - comprehensive NGS WAS done.\n\n"
    
    # Only add GENETICS guidance if NO mutation report exists (mutation report takes precedence)
    genetics_guidance = ""
    if not has_mutation_report and key_facts and "GENETICS:" in key_facts:
        genetics_match = re.search(r'GENETICS:\s*HRD=([^;]+);\s*BRCA1=([^;]+);\s*BRCA2=([^;]+)', key_facts)
        if genetics_match:
            hrd_val = genetics_match.group(1).strip()
            brca1_val = genetics_match.group(2).strip()
            brca2_val = genetics_match.group(3).strip()
            genetics_guidance = "\nCRITICAL: In KEY FACTS, you see 'GENETICS: HRD={}; BRCA1={}; BRCA2={}'. ".format(hrd_val, brca1_val, brca2_val)
            if hrd_val != "Unknown" and hrd_val != "unknown":
                genetics_guidance += f"HRD test WAS performed, result is {hrd_val.lower()}. You MUST say 'HRD-{hrd_val.lower()}' in your query, NOT 'not reported'. "
            if brca1_val != "Unknown" and brca1_val != "unknown":
                genetics_guidance += f"BRCA1 test WAS performed, result is {brca1_val.lower()}. "
            if brca2_val != "Unknown" and brca2_val != "unknown":
                genetics_guidance += f"BRCA2 test WAS performed, result is {brca2_val.lower()}. "
            if (brca1_val != "Unknown" and brca1_val != "unknown") or (brca2_val != "Unknown" and brca2_val != "unknown"):
                genetics_guidance += "You MUST say 'BRCA-negative' or 'BRCA1/BRCA2-negative' in your query, NOT 'not reported'. "
            genetics_guidance += "The word 'Negative' or 'Positive' means the test was done. Only 'Unknown' means not tested.\n\n"
    
    query_builder_template = rag_prompts.get("query_builder",
        "You are preparing a single concise English query to retrieve guideline/clinical evidence "
        "for this ovarian cancer MDT case.\n\n"
        "# STRUCTURED_CASE_TEXT\n{question}\n\n"
        "Write ONE line (<=40 words) focusing on:\n"
        "- tumor type/histology and platinum status;\n"
        "- key metastases / disease extent;\n"
        "- key molecular markers if mentioned (e.g., BRCA/HRD/MSI/PD-L1);\n"
        "- major clinical constraints (e.g., anemia, organ function, performance).\n"
        "Do NOT mention report_ids, dates, hospital names, or patient identifiers.\n"
        "If KEY FACTS include histology or platinum/genetic status, you MUST include them.\n"
        "Do NOT say 'unknown' if a KEY FACT is provided.\n"
        "Output ONLY the query text."
    )
    # Build prompt with mutation_guidance at the END (right before final output instruction)
    # This addresses LLM position bias - important instructions should be near the output
    base_prompt = query_builder_template.format(question=question)
    
    # Insert mutation_guidance right before "Output ONLY" for maximum impact
    if mutation_guidance and "Output ONLY" in base_prompt:
        parts = base_prompt.rsplit("Output ONLY", 1)
        base_prompt = parts[0] + mutation_guidance + "Output ONLY" + parts[1]
    elif mutation_guidance:
        base_prompt = base_prompt + "\n" + mutation_guidance
    
    # Add genetics_guidance if present (no mutation report)
    if genetics_guidance and "Output ONLY" in base_prompt:
        parts = base_prompt.rsplit("Output ONLY", 1)
        base_prompt = parts[0] + genetics_guidance + "Output ONLY" + parts[1]
    elif genetics_guidance:
        base_prompt = base_prompt + "\n" + genetics_guidance
    
    prompt = facts_block + base_prompt
    
    try:
        raw_query = agent.run_selection(prompt).strip()
    except Exception as e:
        # Fallback: construct simple query from key facts
        print(f"[WARNING] RAG query builder failed: {e}")
        if key_facts:
            # Extract basic info from key_facts
            hist_match = re.search(r"histology=([^;\n]+)", key_facts, re.IGNORECASE)
            plat_match = re.search(r"PLATINUM:\s*status=([^;]+)", key_facts, re.IGNORECASE)
            parts = []
            if hist_match:
                hist = _clean_histology_for_query(hist_match.group(1).strip())
                if hist:
                    parts.append(hist)
            if plat_match:
                parts.append(f"platinum {plat_match.group(1).strip().lower()}")
            raw_query = "ovarian cancer " + " ".join(parts) if parts else "ovarian cancer treatment"
        else:
            raw_query = "ovarian cancer treatment guidelines"
    
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
    # Common identifier labels (English and Chinese: medical record number, patient ID, ID card number)
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
            tag = f"[@guideline:{doc_id} | Page {page}]"
        elif source == "pubmed":
            pmid = r.get("pmid", "")
            tag = f"[@pubmed | {pmid}]"
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
    try:
        return agent.run_selection(prompt)
    except Exception as e:
        # Fallback: create simple digest from RAG raw results
        print(f"[WARNING] RAG evidence summarization failed: {e}")
        if not rag_raw:
            return "# No RAG evidence available"
        
        digest_lines = []
        for i, r in enumerate(rag_raw[:min(total_count, 8)], 1):  # Limit to 8 bullets
            source = r.get("source", "")
            if source == "guideline":
                doc_id = r.get("doc_id", "")
                page = r.get("page", "NA")
                tag = f"[@guideline:{doc_id} | Page {page}]"
                text = r.get("text", "")
            elif source == "pubmed":
                pmid = r.get("pmid", "")
                tag = f"[@pubmed | {pmid}]"
                text = r.get("abstract", "") or r.get("title", "")
            else:
                tag = f"[unknown source {i}]"
                text = r.get("text", "") or ""
            
            # Create a simple bullet from the text
            preview = text[:150].strip() if text else "Evidence available"
            if len(text) > 150:
                preview += "..."
            digest_lines.append(f"- {preview} {tag}")
        
        return "\n".join(digest_lines) if digest_lines else "# No RAG evidence available"

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