"""Reference Cache - Local storage and retrieval of RAG references (guidelines and PubMed).

This module provides a local cache for RAG references to avoid repeated model calls
when experts or final output need to look up full reference details.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib


class ReferenceCache:
    """Local cache for RAG references (guidelines and PubMed articles)."""
    
    def __init__(self, cache_dir: str = "rag_store/reference_cache"):
        """
        Initialize reference cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.guideline_cache_file = self.cache_dir / "guidelines.json"
        self.pubmed_cache_file = self.cache_dir / "pubmed.json"
        
        # In-memory cache
        self._guideline_cache: Dict[str, Dict[str, Any]] = {}
        self._pubmed_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.guideline_cache_file.exists():
            try:
                with open(self.guideline_cache_file, "r", encoding="utf-8") as f:
                    self._guideline_cache = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load guideline cache: {e}")
                self._guideline_cache = {}
        
        if self.pubmed_cache_file.exists():
            try:
                with open(self.pubmed_cache_file, "r", encoding="utf-8") as f:
                    self._pubmed_cache = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load PubMed cache: {e}")
                self._pubmed_cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.guideline_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._guideline_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save guideline cache: {e}")
        
        try:
            with open(self.pubmed_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._pubmed_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save PubMed cache: {e}")
    
    def _get_guideline_key(self, doc_id: str, page: Optional[int] = None) -> str:
        """Generate cache key for guideline reference."""
        if page is not None:
            return f"{doc_id}|{page}"
        return doc_id
    
    def _get_pubmed_key(self, pmid: str) -> str:
        """Generate cache key for PubMed reference."""
        return pmid
    
    def store_guideline(self, doc_id: str, page: Optional[int], text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a guideline reference in cache.
        
        Args:
            doc_id: Document ID
            page: Page number (optional)
            text: Full text content
            metadata: Additional metadata (score, rank, etc.)
        """
        key = self._get_guideline_key(doc_id, page)
        self._guideline_cache[key] = {
            "doc_id": doc_id,
            "page": page,
            "text": text,
            "metadata": metadata or {},
            "cached_at": datetime.now().isoformat(),
        }
        self._save_cache()
    
    def store_pubmed(self, pmid: str, title: str, abstract: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a PubMed reference in cache.
        
        Args:
            pmid: PubMed ID
            title: Article title
            abstract: Article abstract
            metadata: Additional metadata (score, journal, etc.)
        """
        key = self._get_pubmed_key(pmid)
        self._pubmed_cache[key] = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "metadata": metadata or {},
            "cached_at": datetime.now().isoformat(),
        }
        self._save_cache()
    
    def store_rag_results(self, rag_raw: List[Dict[str, Any]]):
        """
        Store all RAG results in cache.
        
        Args:
            rag_raw: List of raw RAG results from rag_search_pack or pubmed_search_pack
        """
        for result in rag_raw:
            source = result.get("source", "")
            if source == "guideline":
                doc_id = result.get("doc_id", "")
                page = result.get("page")
                text = result.get("text", "")
                metadata = {
                    "rank": result.get("rank"),
                    "score": result.get("score"),
                }
                if doc_id:
                    self.store_guideline(doc_id, page, text, metadata)
            elif source == "pubmed":
                pmid = result.get("pmid", "")
                title = result.get("title", "")
                abstract = result.get("abstract", "")
                metadata = {
                    "rank": result.get("rank"),
                    "score": result.get("score"),
                    "journal": result.get("journal"),
                    "pub_date": result.get("pub_date"),
                    "doi": result.get("doi"),
                    "impact_factor": result.get("impact_factor"),
                }
                if pmid:
                    self.store_pubmed(pmid, title, abstract, metadata)
    
    def get_guideline(self, doc_id: str, page: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a guideline reference from cache.
        
        Args:
            doc_id: Document ID
            page: Page number (optional)
        
        Returns:
            Cached reference dict or None if not found
        """
        key = self._get_guideline_key(doc_id, page)
        return self._guideline_cache.get(key)
    
    def get_pubmed(self, pmid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a PubMed reference from cache.
        
        Args:
            pmid: PubMed ID
        
        Returns:
            Cached reference dict or None if not found
        """
        key = self._get_pubmed_key(pmid)
        return self._pubmed_cache.get(key)
    
    def get_reference_by_tag(self, tag: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a reference by its evidence tag.
        
        Args:
            tag: Evidence tag in format [@guideline:doc_id|page] or [@pubmed:PMID]
        
        Returns:
            Cached reference dict or None if not found
        """
        import re
        
        # Parse guideline tag: [@guideline:doc_id|page]
        guideline_match = re.match(r"\[@guideline:([^|]+)\|?(\d+)?\]", tag, re.IGNORECASE)
        if guideline_match:
            doc_id = guideline_match.group(1)
            page_str = guideline_match.group(2)
            page = int(page_str) if page_str else None
            return self.get_guideline(doc_id, page)
        
        # Parse PubMed tag: [@pubmed:PMID]
        pubmed_match = re.match(r"\[@pubmed:(\d+)\]", tag, re.IGNORECASE)
        if pubmed_match:
            pmid = pubmed_match.group(1)
            return self.get_pubmed(pmid)
        
        return None
    
    def format_reference(self, tag: str) -> str:
        """
        Format a cached reference for display.
        
        Args:
            tag: Evidence tag
        
        Returns:
            Formatted reference string
        """
        ref = self.get_reference_by_tag(tag)
        if not ref:
            return f"[Reference not found: {tag}]"
        
        if "pmid" in ref:
            # PubMed reference
            title = ref.get("title", "N/A")
            abstract = ref.get("abstract", "")[:200] + "..." if len(ref.get("abstract", "")) > 200 else ref.get("abstract", "")
            journal = ref.get("metadata", {}).get("journal", "")
            pub_date = ref.get("metadata", {}).get("pub_date", "")
            return f"PubMed {ref['pmid']}: {title}\n  Journal: {journal}, {pub_date}\n  Abstract: {abstract}"
        else:
            # Guideline reference
            doc_id = ref.get("doc_id", "N/A")
            page = ref.get("page", "")
            text = ref.get("text", "")[:200] + "..." if len(ref.get("text", "")) > 200 else ref.get("text", "")
            page_str = f", Page {page}" if page else ""
            return f"Guideline {doc_id}{page_str}:\n  {text}"


# Global cache instance
_global_cache: Optional[ReferenceCache] = None


def get_reference_cache(cache_dir: str = "rag_store/reference_cache") -> ReferenceCache:
    """
    Get or create the global reference cache instance.
    
    Args:
        cache_dir: Directory to store cache files
    
    Returns:
        ReferenceCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ReferenceCache(cache_dir=cache_dir)
    return _global_cache


def extract_reference_tags(text: str) -> List[str]:
    """
    Extract all reference tags from text.
    
    Args:
        text: Text containing evidence tags
    
    Returns:
        List of reference tags found
    """
    import re
    guideline_tags = re.findall(r"\[@guideline:[^\]]+\]", text, re.IGNORECASE)
    pubmed_tags = re.findall(r"\[@pubmed:\d+\]", text, re.IGNORECASE)
    return guideline_tags + pubmed_tags
