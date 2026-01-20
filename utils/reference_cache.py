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
    """Local cache for RAG references (guidelines, PubMed, trials)."""
    
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
        self._trial_cache: Dict[str, Dict[str, Any]] = {}  # In-memory only
        
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
    
    def store_trial(self, trial_id: str, name: str, reason: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        Store a clinical trial reference in cache (in-memory only).
        
        Args:
            trial_id: Trial ID
            name: Trial name
            reason: Recommendation reason
            metadata: Additional metadata
        """
        self._trial_cache[str(trial_id)] = {
            "trial_id": trial_id,
            "name": name,
            "reason": reason,
            "metadata": metadata or {},
            "cached_at": datetime.now().isoformat(),
        }
    
    def get_trial(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a clinical trial reference from cache.
        
        Args:
            trial_id: Trial ID
        
        Returns:
            Cached trial dict or None if not found
        """
        return self._trial_cache.get(str(trial_id))
    
    def get_reference_by_tag(self, tag: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a reference by its evidence tag.
        
        Args:
            tag: Evidence tag in format:
                - [@guideline:doc_id|page]
                - [@pubmed:PMID]
                - [@trial:id]
        
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
        
        # Parse trial tag: [@trial:id]
        trial_match = re.match(r"\[@trial:([^\]]+)\]", tag, re.IGNORECASE)
        if trial_match:
            trial_id = trial_match.group(1)
            return self.get_trial(trial_id)
        
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
    
    Supported formats:
    - [@guideline:doc_id|page] - Guideline references
    - [@pubmed:PMID] - PubMed literature
    - [@trial:id] - Clinical trial references
    - [@report_id|date] - Clinical report references (e.g., [@20230103|2023-01-03])
    
    Also handles combined tags like [@tag1; @tag2] by splitting them.
    
    Args:
        text: Text containing evidence tags
    
    Returns:
        List of reference tags found (in order of appearance)
    """
    import re
    
    # First, normalize text: split combined tags like [@tag1; @tag2] into [@tag1] [@tag2]
    # This handles LLM output that combines multiple refs in one bracket
    normalized_text = re.sub(r';\s*@', '] [@', text)
    
    # Guideline tags: [@guideline:doc_id|page]
    # doc_id can contain underscores, letters, numbers, dashes
    guideline_tags = re.findall(r"\[@guideline:[a-zA-Z0-9_\-]+\|[^\]]+\]", normalized_text, re.IGNORECASE)
    
    # PubMed tags: [@pubmed:PMID]
    pubmed_tags = re.findall(r"\[@pubmed:\d+\]", normalized_text, re.IGNORECASE)
    
    # Trial tags: [@trial:id]
    trial_tags = re.findall(r"\[@trial:[^\]]+\]", normalized_text, re.IGNORECASE)
    
    # Report tags: [@report_id|date] - simple format without colons
    # Matches patterns like [@20230103|2023-01-03], [@OH2203828|2022-04-18], [@2022-12-29|CT]
    # More flexible: report_id can be alphanumeric with dashes/underscores
    report_tags = re.findall(r"\[@[a-zA-Z0-9_\-]+\|[^\]]+\]", normalized_text)
    
    # Filter out guideline/pubmed/trial from report_tags (they might have | too)
    filtered_report_tags = []
    for tag in report_tags:
        tag_lower = tag.lower()
        if (not tag_lower.startswith("[@guideline:") and 
            not tag_lower.startswith("[@pubmed:") and 
            not tag_lower.startswith("[@trial:")):
            filtered_report_tags.append(tag)
    
    return guideline_tags + pubmed_tags + trial_tags + filtered_report_tags


def build_references_section(
    text: str,
    cache: Optional[ReferenceCache] = None,
    max_content_length: int = 60,
    trial_info: Optional[Dict[str, Any]] = None,
    report_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a References section from evidence tags found in text.
    
    Extracts all evidence tags and generates a formatted References block:
    - [@guideline:doc_id|page] - Guideline references
    - [@pubmed:PMID] - PubMed literature references
    - [@trial:id] - Clinical trial references
    - [@report_id|date] - Clinical report references
    
    Args:
        text: Text containing evidence tags
        cache: ReferenceCache instance (uses global cache if None)
        max_content_length: Maximum length for content preview
        trial_info: Dict with trial details {id: {name, reason, ...}}
        report_context: Dict with report data for lookup
    
    Returns:
        Formatted References section string, or empty string if no tags found
    """
    if not text:
        return ""
    
    # Get cache instance
    if cache is None:
        cache = get_reference_cache()
    
    # Extract all tags
    all_tags = extract_reference_tags(text)
    if not all_tags:
        return ""
    
    # Categorize and deduplicate tags
    guideline_tags = []
    pubmed_tags = []
    trial_tags = []
    report_tags = []
    seen = set()
    
    for tag in all_tags:
        tag_lower = tag.lower()
        if tag_lower in seen:
            continue
        seen.add(tag_lower)
        
        if tag_lower.startswith("[@guideline:"):
            guideline_tags.append(tag)
        elif tag_lower.startswith("[@pubmed:"):
            pubmed_tags.append(tag)
        elif tag_lower.startswith("[@trial:"):
            trial_tags.append(tag)
        else:
            # Report reference [@report_id|date]
            report_tags.append(tag)
    
    ref_lines = []
    ref_lines.append("\n---")
    ref_lines.append("## References\n")
    
    # Always show all four categories in fixed order, even if empty
    # Order: Guidelines → Literature → Clinical Trials → Clinical Reports
    
    # === Guideline References ===
    ref_lines.append("### Guidelines\n")
    if guideline_tags:
        for tag in guideline_tags:
            ref = cache.get_reference_by_tag(tag)
            if ref is None:
                ref_lines.append(f"{tag}")
                ref_lines.append("  [Not cached]\n")
                continue
            
            doc_id = ref.get("doc_id", "N/A")
            page = ref.get("page")
            content = ref.get("text", "")
            
            if len(content) > max_content_length:
                content = content[:max_content_length].rsplit(" ", 1)[0] + "..."
            
            page_str = f", Page {page}" if page else ""
            ref_lines.append(f"{tag}")
            ref_lines.append(f"  Document: {doc_id}{page_str}")
            if content:
                ref_lines.append(f"  Content: {content}")
            ref_lines.append("")
    
    # === PubMed References ===
    ref_lines.append("### Literature\n")
    if pubmed_tags:
        for tag in pubmed_tags:
            ref = cache.get_reference_by_tag(tag)
            if ref is None:
                ref_lines.append(f"{tag}")
                ref_lines.append("  [Not cached]\n")
                continue
            
            pmid = ref.get("pmid", "")
            title = ref.get("title", "")
            metadata = ref.get("metadata", {})
            journal = metadata.get("journal", "")
            pub_date = metadata.get("pub_date", "")
            doi = metadata.get("doi", "")
            
            ref_lines.append(f"{tag}")
            # Format: PMID | Journal | Date
            info_parts = [f"PMID: {pmid}"]
            if journal:
                info_parts.append(journal)
            if pub_date:
                info_parts.append(pub_date)
            ref_lines.append(f"  {' | '.join(info_parts)}")
            if title:
                ref_lines.append(f"  Title: {title}")
            if doi:
                ref_lines.append(f"  DOI: {doi}")
            ref_lines.append("")
    
    # === Clinical Trial References ===
    ref_lines.append("### Clinical Trials\n")
    if trial_tags:
        for tag in trial_tags:
            # Extract trial ID from tag [@trial:id]
            import re
            match = re.match(r"\[@trial:([^\]]+)\]", tag, re.IGNORECASE)
            if not match:
                continue
            trial_id = match.group(1)
            
            # Try cache first, then fall back to trial_info parameter
            trial = cache.get_trial(trial_id)
            if trial is None and trial_info:
                trial = trial_info.get(trial_id) or trial_info.get(str(trial_id))
            
            ref_lines.append(f"{tag}")
            if trial:
                name = trial.get("name", "")
                reason = trial.get("reason", "")
                ref_lines.append(f"  Trial ID: {trial_id}")
                if name:
                    ref_lines.append(f"  Name: {name}")
                if reason:
                    ref_lines.append(f"  Rationale: {reason}")
            else:
                ref_lines.append(f"  Trial ID: {trial_id}")
                ref_lines.append("  [Details not available]")
            ref_lines.append("")
    
    # === Report References ===
    ref_lines.append("### Clinical Reports\n")
    if report_tags:
        for tag in report_tags:
            # Parse [@report_id|date] format
            import re
            match = re.match(r"\[@([^|\]]+)\|([^\]]+)\]", tag)
            if not match:
                continue
            report_id = match.group(1)
            report_date = match.group(2)
            
            # Look up report in context if available
            report_info = _find_report_in_context(report_id, report_context) if report_context else None
            
            ref_lines.append(f"{tag}")
            if report_info:
                rtype = report_info.get("type", "")
                # Try multiple fields for summary content
                summary = _extract_report_summary(report_info, max_content_length)
                
                type_label = rtype.capitalize() if rtype else "Report"
                ref_lines.append(f"  {type_label} ID: {report_id} | Date: {report_date}")
                if summary:
                    ref_lines.append(f"  Content: {summary}")
            else:
                ref_lines.append(f"  Report ID: {report_id} | Date: {report_date}")
            ref_lines.append("")
    
    return "\n".join(ref_lines)


def _extract_report_summary(report_info: Dict[str, Any], max_length: int = 60) -> str:
    """Extract a summary snippet from a report dict."""
    if not report_info:
        return ""
    
    # Priority order for different report types
    summary_fields = [
        # Common fields
        "summary", "impression", "conclusion",
        # Lab specific
        "result", "finding", "value",
        # Imaging specific  
        "findings",
        # Pathology specific
        "diagnosis", "histology",
        # Mutation specific
        "mutations", "gene_alterations",
        # Fallback
        "raw_text", "text", "content",
    ]
    
    for field in summary_fields:
        value = report_info.get(field)
        if value and isinstance(value, str) and value.strip():
            text = value.strip()
            # Clean up text (remove excessive whitespace)
            text = " ".join(text.split())
            if len(text) > max_length:
                text = text[:max_length].rsplit(" ", 1)[0] + "..."
            return text
    
    return ""


def _find_report_in_context(report_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a report by ID in the context data structure."""
    if not context or not report_id:
        return None
    
    # Normalize report_id for comparison
    report_id_str = str(report_id).strip()
    
    # Search through all report types and roles
    for report_type in ["lab", "imaging", "pathology", "mutation"]:
        type_data = context.get(report_type, {})
        if not isinstance(type_data, dict):
            continue
        for role, reports in type_data.items():
            if not isinstance(reports, list):
                continue
            for report in reports:
                if not isinstance(report, dict):
                    continue
                # Check multiple ID fields
                rid = (report.get("report_id", "") or 
                       report.get("id", "") or 
                       report.get("report_no", ""))
                if str(rid).strip() == report_id_str:
                    return {"type": report_type, **report}
    
    return None
