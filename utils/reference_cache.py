"""Reference Cache - Local storage and retrieval of RAG references.

This module provides a local cache for RAG references to avoid repeated model calls
when experts or final output need to look up full reference details.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from utils.patterns import extract_reference_tags


def _default_cache_dir() -> str:
    configured = os.getenv("OMGS_REFERENCE_CACHE_DIR", "").strip()
    if configured:
        return configured
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "tmp" / "reference_cache")


class ReferenceCache:
    """Local cache for RAG references (guidelines, NCCN rules, PubMed, external evidence, trials)."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize reference cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to
                OMGS_REFERENCE_CACHE_DIR or tmp/reference_cache.

        Note: If directory creation fails, cache will work in-memory only.
        """
        resolved_cache_dir = cache_dir or _default_cache_dir()
        self.cache_dir = Path(resolved_cache_dir)
        self._cache_dir_available = False

        # Try to create cache directory, but don't fail if it doesn't work
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_dir_available = True
        except (OSError, PermissionError) as e:
            # Cache directory unavailable - work in-memory only
            print(f"[WARNING] Failed to create reference cache directory '{cache_dir}': {e}. "
                  f"Cache will work in-memory only.")
            self._cache_dir_available = False

        # Cache files
        self.guideline_cache_file = self.cache_dir / "guidelines.json"
        self.pubmed_cache_file = self.cache_dir / "pubmed.json"
        self.nccn_cache_file = self.cache_dir / "nccn.json"
        self.external_cache_file = self.cache_dir / "external.json"
        self.trial_cache_file = self.cache_dir / "trials.json"

        # In-memory caches
        self._guideline_cache: Dict[str, Dict[str, Any]] = {}
        self._pubmed_cache: Dict[str, Dict[str, Any]] = {}
        self._nccn_cache: Dict[str, Dict[str, Any]] = {}  # NCCN rules cache
        self._external_cache: Dict[str, Dict[str, Any]] = {}  # FDA / conference cache
        self._trial_cache: Dict[str, Dict[str, Any]] = {}  # In-memory only

        # Load existing cache (only if directory is available)
        if self._cache_dir_available:
            self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        # Only load if cache directory is available
        if not self._cache_dir_available:
            return

        if self.guideline_cache_file.exists():
            try:
                with open(self.guideline_cache_file, "r", encoding="utf-8") as f:
                    self._guideline_cache = json.load(f)
            except (OSError, PermissionError, FileNotFoundError) as e:
                print(f"[WARNING] Failed to load guideline cache: {e}")
                self._guideline_cache = {}
            except Exception as e:
                print(f"[WARNING] Failed to load guideline cache: {e}")
                self._guideline_cache = {}

        if self.pubmed_cache_file.exists():
            try:
                with open(self.pubmed_cache_file, "r", encoding="utf-8") as f:
                    self._pubmed_cache = json.load(f)
            except (OSError, PermissionError, FileNotFoundError) as e:
                print(f"[WARNING] Failed to load PubMed cache: {e}")
                self._pubmed_cache = {}
            except Exception as e:
                print(f"[WARNING] Failed to load PubMed cache: {e}")
                self._pubmed_cache = {}

        if self.nccn_cache_file.exists():
            try:
                with open(self.nccn_cache_file, "r", encoding="utf-8") as f:
                    self._nccn_cache = json.load(f)
            except (OSError, PermissionError, FileNotFoundError) as e:
                print(f"[WARNING] Failed to load NCCN cache: {e}")
                self._nccn_cache = {}
            except Exception as e:
                print(f"[WARNING] Failed to load NCCN cache: {e}")
                self._nccn_cache = {}

        if self.external_cache_file.exists():
            try:
                with open(self.external_cache_file, "r", encoding="utf-8") as f:
                    self._external_cache = json.load(f)
            except (OSError, PermissionError, FileNotFoundError) as e:
                print(f"[WARNING] Failed to load external evidence cache: {e}")
                self._external_cache = {}
            except Exception as e:
                print(f"[WARNING] Failed to load external evidence cache: {e}")
                self._external_cache = {}

        if self.trial_cache_file.exists():
            try:
                with open(self.trial_cache_file, "r", encoding="utf-8") as f:
                    self._trial_cache = json.load(f)
            except (OSError, PermissionError, FileNotFoundError) as e:
                print(f"[WARNING] Failed to load trial cache: {e}")
                self._trial_cache = {}
            except Exception as e:
                print(f"[WARNING] Failed to load trial cache: {e}")
                self._trial_cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        # Only save if cache directory is available
        if not self._cache_dir_available:
            return  # Work-in-memory only

        try:
            with open(self.guideline_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._guideline_cache, f, ensure_ascii=False, indent=2)
        except (OSError, PermissionError, FileNotFoundError) as e:
            print(f"[WARNING] Failed to save guideline cache: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to save guideline cache: {e}")

        try:
            with open(self.pubmed_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._pubmed_cache, f, ensure_ascii=False, indent=2)
        except (OSError, PermissionError, FileNotFoundError) as e:
            print(f"[WARNING] Failed to save PubMed cache: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to save PubMed cache: {e}")

        try:
            with open(self.nccn_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._nccn_cache, f, ensure_ascii=False, indent=2)
        except (OSError, PermissionError, FileNotFoundError) as e:
            print(f"[WARNING] Failed to save NCCN cache: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to save NCCN cache: {e}")

        try:
            with open(self.external_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._external_cache, f, ensure_ascii=False, indent=2)
        except (OSError, PermissionError, FileNotFoundError) as e:
            print(f"[WARNING] Failed to save external evidence cache: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to save external evidence cache: {e}")

        try:
            with open(self.trial_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._trial_cache, f, ensure_ascii=False, indent=2)
        except (OSError, PermissionError, FileNotFoundError) as e:
            print(f"[WARNING] Failed to save trial cache: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to save trial cache: {e}")

    def _get_guideline_key(self, doc_id: str, page: Optional[int] = None) -> str:
        """Generate cache key for guideline reference."""
        if page is not None:
            return f"{doc_id}|{page}"
        return doc_id

    def _get_pubmed_key(self, pmid: str) -> str:
        """Generate cache key for PubMed reference."""
        return pmid

    def _get_nccn_key(self, rule_id: str, rule_type: str) -> str:
        """Generate cache key for NCCN rule reference."""
        return f"{rule_type}|{rule_id}"

    def _get_external_key(self, source: str, source_id: str) -> str:
        """Generate cache key for external evidence supplement references."""
        return f"{source}|{source_id}"

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

    def store_nccn_rule(self, rule_id: str, rule_type: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store an NCCN rule reference in cache.

        Args:
            rule_id: Rule ID from NCCN JSON
            rule_type: Type of rule (safety_rule, decision_node)
            text: Rule description/text
            metadata: Additional metadata (score, rank, node_name, etc.)
        """
        key = self._get_nccn_key(rule_id, rule_type)
        self._nccn_cache[key] = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "text": text,
            "metadata": metadata or {},
            "cached_at": datetime.now().isoformat(),
        }
        self._save_cache()

    def store_external(self, source: str, source_id: str, title: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Store an FDA label or conference abstract reference in memory."""
        key = self._get_external_key(source, source_id)
        self._external_cache[key] = {
            "source": source,
            "source_id": source_id,
            "title": title,
            "text": text,
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
                    "original_doc_id": result.get("original_doc_id"),
                    "title": result.get("title"),
                    "citation": result.get("citation"),
                    "page_label": result.get("page_label"),
                    "source_url": result.get("source_url"),
                    "location_hint": result.get("location_hint"),
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
            elif source in ("nccn_safety_rule", "nccn_matcher_rule", "nccn_decision_node"):
                # Store NCCN rules
                rule_id = result.get("rule_id") or result.get("node_id", "")
                if source == "nccn_safety_rule":
                    rule_type = "safety_rule"
                elif source == "nccn_matcher_rule":
                    rule_type = "matcher_rule"
                else:
                    rule_type = "decision_node"
                text = result.get("text", "")
                metadata = {
                    "rank": result.get("rank"),
                    "score": result.get("score"),
                    "node_name": result.get("node_name", ""),
                    "pages": result.get("pages", []),
                    "keywords": result.get("keywords", []),
                }
                if rule_id:
                    self.store_nccn_rule(rule_id, rule_type, text, metadata)
            elif source in ("fda", "conference"):
                source_id = result.get("source_id", "")
                title = result.get("title", "")
                text = result.get("text", "") or result.get("summary", "") or result.get("abstract", "")
                metadata = {
                    "rank": result.get("rank"),
                    "source_url": result.get("source_url"),
                    "journal": result.get("journal"),
                    "backend_name": result.get("backend_name"),
                    "evidence_type": result.get("evidence_type"),
                    "citation": result.get("citation"),
                }
                if source_id:
                    self.store_external(source, source_id, title, text, metadata)

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

    def get_nccn_rule(self, rule_id: str, rule_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve an NCCN rule reference from cache.

        Args:
            rule_id: Rule ID
            rule_type: Optional rule type (safety_rule, decision_node) for disambiguation

        Returns:
            Cached reference dict or None if not found
        """
        if rule_type:
            key = self._get_nccn_key(rule_id, rule_type)
            return self._nccn_cache.get(key)

        # If no rule_type specified, search both types
        for rt in ("safety_rule", "matcher_rule", "decision_node"):
            key = self._get_nccn_key(rule_id, rt)
            ref = self._nccn_cache.get(key)
            if ref:
                return ref
        return None

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
        self._save_cache()

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
            tag: Source-typed evidence tag from the generated recommendation.

        Returns:
            Cached reference dict or None if not found
        """

        # IMPORTANT: Check NCCN tags FIRST — [@guideline:nccn | rule_id] must be
        # handled before the generic guideline regex, which would incorrectly
        # match doc_id="nccn" and route to get_guideline() instead of get_nccn_rule().

        # Parse NCCN matcher tag.
        guideline_nccn_match = re.match(r"\[@guideline:nccn\s*\|\s*([^\]]+)\]", tag, re.IGNORECASE)
        if guideline_nccn_match:
            rule_id = guideline_nccn_match.group(1).strip()
            # Search both rule types — don't guess from the ID
            return self.get_nccn_rule(rule_id, rule_type=None)

        # Backwards-compatible NCCN parsing for older local artifacts.
        nccn_match = re.match(r"\[@nccn\s*\|\s*([^\]]+)\]", tag, re.IGNORECASE)
        if not nccn_match:
            nccn_match = re.match(r"\[@nccn:([^\]]+)\]", tag, re.IGNORECASE)
        if nccn_match:
            rule_id = nccn_match.group(1).strip()
            return self.get_nccn_rule(rule_id, rule_type=None)

        # Parse guideline tag.
        guideline_match = re.match(
            r"\[@guideline:([^|\]]+)\s*\|\s*(?:Pages?\s+)?(\d+)(?:\s*[-–]\s*\d+)?\]",
            tag,
            re.IGNORECASE,
        )
        if not guideline_match:
            # Backwards-compatible compact page suffix.
            guideline_match = re.match(r"\[@guideline:([^|]+)\|(\d+|\w+)\]", tag, re.IGNORECASE)
        if guideline_match:
            doc_id = guideline_match.group(1).strip()
            page_str = guideline_match.group(2).strip()
            try:
                page = int(page_str) if page_str.isdigit() else None
            except (ValueError, AttributeError):
                page = None
            return self.get_guideline(doc_id, page)

        # Parse PubMed tag.
        pubmed_match = re.match(r"\[@pubmed\s*\|\s*(\d+)\]", tag, re.IGNORECASE)
        if not pubmed_match:
            # Backwards-compatible compact PMID suffix.
            pubmed_match = re.match(r"\[@pubmed:(\d+)\]", tag, re.IGNORECASE)
        if pubmed_match:
            pmid = pubmed_match.group(1)
            return self.get_pubmed(pmid)

        # Parse FDA / conference supplement tags.
        external_match = re.match(r"\[@(fda|conference)\s*\|\s*([^\]]+)\]", tag, re.IGNORECASE)
        if external_match:
            source = external_match.group(1).strip().lower()
            source_id = external_match.group(2).strip()
            return self._external_cache.get(self._get_external_key(source, source_id))

        # Parse trial tag.
        trial_match = re.match(r"\[@trial\s*\|\s*([^\]]+)\]", tag, re.IGNORECASE)
        if not trial_match:
            # Backwards-compatible compact trial suffix.
            trial_match = re.match(r"\[@trial:([^\]]+)\]", tag, re.IGNORECASE)
        if trial_match:
            trial_id = trial_match.group(1).strip()
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
        elif ref.get("source") in {"fda", "conference"}:
            source = str(ref.get("source") or "").upper()
            title = ref.get("title", "N/A")
            text = ref.get("text", "")[:200] + "..." if len(ref.get("text", "")) > 200 else ref.get("text", "")
            location = ref.get("metadata", {}).get("journal", "")
            location_text = f"\n  Source: {location}" if location else ""
            return f"{source} {ref.get('source_id', 'N/A')}: {title}{location_text}\n  Text: {text}"
        elif "rule_id" in ref:
            # NCCN rule reference
            rule_id = ref.get("rule_id", "N/A")
            rule_type = ref.get("rule_type", "")
            text = ref.get("text", "")[:200] + "..." if len(ref.get("text", "")) > 200 else ref.get("text", "")
            return f"NCCN {rule_type} ({rule_id}): {text}"
        else:
            # Guideline reference
            doc_id = ref.get("doc_id", "N/A")
            page = ref.get("page", "")
            text = ref.get("text", "")[:200] + "..." if len(ref.get("text", "")) > 200 else ref.get("text", "")
            metadata = ref.get("metadata") or {}
            page_label = str(metadata.get("page_label") or "").strip()
            doc_label = (
                str(metadata.get("citation") or "").strip()
                or str(metadata.get("title") or "").strip()
                or str(doc_id)
            )
            page_str = ""
            if page_label and page_label.lower() not in doc_label.lower():
                page_str = f", {page_label}"
            elif not page_label and page:
                page_str = f", Page {page}"
            return f"Guideline {doc_label}{page_str}:\n  {text}"


# Global cache instance
_global_cache: Optional[ReferenceCache] = None


def get_reference_cache(cache_dir: Optional[str] = None) -> ReferenceCache:
    """
    Get or create the global reference cache instance.

    If cache directory cannot be created, returns a cache that works in-memory only.
    This ensures the pipeline continues even when rag_store directory is missing.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ReferenceCache(cache_dir=cache_dir)
    return _global_cache


def _format_tag_for_display(tag: str) -> str:
    """Format a tag for better display, extracting key info."""
    # Extract the core identifier from tags
    match = re.match(r"\[@([^:]+):([^\]|]+)(?:\s*\|\s*([^\]]+))?\]", tag, re.IGNORECASE)
    if match:
        source = match.group(1).upper()
        identifier = match.group(2).strip()
        extra = match.group(3).strip() if match.group(3) else ""
        if extra:
            return f"[{source} | {identifier} | {extra}]"
        return f"[{source} | {identifier}]"
    return tag


def _truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text with ellipsis if needed."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def build_references_section(
    text: str,
    cache: Optional[ReferenceCache] = None,
    max_content_length: int = 80,
    trial_info: Optional[Dict[str, Any]] = None,
    report_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a beautiful References section from evidence tags found in text.

    Extracts source-typed evidence tags and generates a formatted References block.

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
    nccn_tags = []
    pubmed_tags = []
    fda_tags = []
    conference_tags = []
    trial_tags = []
    report_tags = []
    seen = set()

    for tag in all_tags:
        tag_lower = tag.lower()
        if tag_lower in seen:
            continue
        seen.add(tag_lower)

        # Check for NCCN tags first because they are more specific.
        if tag_lower.startswith("[@nccn") or tag_lower.startswith("[@guideline:nccn"):
            nccn_tags.append(tag)
        # Check for guideline tags.
        elif tag_lower.startswith("[@guideline:"):
            guideline_tags.append(tag)
        # Check for PubMed tags.
        elif tag_lower.startswith("[@pubmed |") or tag_lower.startswith("[@pubmed:"):
            pubmed_tags.append(tag)
        # Check for FDA supplement tags: [@fda | ...]
        elif re.match(r"\[@fda\s*\|", tag, re.IGNORECASE):
            fda_tags.append(tag)
        # Check for conference supplement tags: [@conference | ...]
        elif re.match(r"\[@conference\s*\|", tag, re.IGNORECASE):
            conference_tags.append(tag)
        # Check for trial tags.
        elif re.match(r"\[@trial\s*\|", tag, re.IGNORECASE) or tag_lower.startswith("[@trial:"):
            trial_tags.append(tag)
        else:
            # Patient-report reference.
            report_tags.append(tag)

    # Markdown references appended to the answer (matches \n---\n## References and ### categories)
    ref_lines = []
    ref_lines.append("\n---")
    ref_lines.append("## References\n")

    # Guidelines: NCCN first, then other guidelines for reference rendering
    ref_lines.append("### Guidelines\n")
    for tag in nccn_tags + guideline_tags:
        ref = cache.get_reference_by_tag(tag)
        ref_lines.append(f"{tag}")
        if ref is None:
            ref_lines.append("  [Not cached]")
            ref_lines.append("")
            continue
        if ref.get("rule_id") is not None:
            rule_id = ref.get("rule_id", "N/A")
            rule_type = ref.get("rule_type", "")
            text = ref.get("text", "")
            node_name = ref.get("metadata", {}).get("node_name", "")
            ref_lines.append(f"  Rule ID: {rule_id}")
            ref_lines.append(f"  Type: {rule_type.replace('_', ' ').title()}")
            if node_name:
                ref_lines.append(f"  Node: {node_name}")
            content = _truncate_text(text, max_content_length)
            if content:
                ref_lines.append(f"  Summary: {content}")
        else:
            doc_id = ref.get("doc_id", "N/A")
            page = ref.get("page")
            content = ref.get("text", "")
            metadata = ref.get("metadata") or {}
            page_label = str(metadata.get("page_label") or "").strip()
            doc_label = (
                str(metadata.get("citation") or "").strip()
                or str(metadata.get("title") or "").strip()
                or str(doc_id)
            )
            page_str = ""
            if page_label and page_label.lower() not in doc_label.lower():
                page_str = f", {page_label}"
            elif not page_label and page:
                page_str = f", Page {page}"
            ref_lines.append(f"  Document: {doc_label}{page_str}")
            if doc_label != str(doc_id):
                ref_lines.append(f"  Source ID: {doc_id}")
            original_doc_id = str(metadata.get("original_doc_id") or "").strip()
            if original_doc_id and original_doc_id != str(doc_id):
                ref_lines.append(f"  Original Source ID: {original_doc_id}")
            if content:
                ref_lines.append(f"  Content: {_truncate_text(content, max_content_length)}")
        ref_lines.append("")

    # External Evidence: PubMed literature, FDA labels, and conference abstracts.
    ref_lines.append("### External Evidence\n")
    for tag in pubmed_tags:
        ref = cache.get_reference_by_tag(tag)
        ref_lines.append(f"{tag}")
        if ref is None:
            ref_lines.append("  [Not cached]")
            ref_lines.append("")
            continue
        pmid = ref.get("pmid", "")
        title = ref.get("title", "")
        metadata = ref.get("metadata", {})
        journal = metadata.get("journal", "")
        pub_date = metadata.get("pub_date", "")
        doi = metadata.get("doi", "") or ref.get("doi", "")
        info_parts = ["Type: PubMed", f"PMID: {pmid}"]
        if journal:
            info_parts.append(journal)
        if pub_date:
            info_parts.append(pub_date)
        ref_lines.append(f"  {' | '.join(info_parts)}")
        if title:
            ref_lines.append(f"  Title: {_truncate_text(title, 60)}")
        if doi:
            ref_lines.append(f"  DOI: {doi}")
        ref_lines.append("")

    for tag in fda_tags:
        ref = cache.get_reference_by_tag(tag)
        ref_lines.append(f"{tag}")
        if ref is None:
            ref_lines.append("  [Not cached]")
            ref_lines.append("")
            continue
        source_id = ref.get("source_id", "")
        title = ref.get("title", "")
        text = ref.get("text", "")
        ref_lines.append(f"  Type: FDA Label | Source ID: {source_id}")
        if title:
            ref_lines.append(f"  Title: {_truncate_text(title, 60)}")
        if text:
            ref_lines.append(f"  Content: {_truncate_text(text, max_content_length)}")
        ref_lines.append("")

    for tag in conference_tags:
        ref = cache.get_reference_by_tag(tag)
        ref_lines.append(f"{tag}")
        if ref is None:
            ref_lines.append("  [Not cached]")
            ref_lines.append("")
            continue
        source_id = ref.get("source_id", "")
        title = ref.get("title", "")
        text = ref.get("text", "")
        metadata = ref.get("metadata", {})
        venue = metadata.get("journal") or metadata.get("backend_name") or metadata.get("citation") or ""
        info_parts = ["Type: Conference", f"Source ID: {source_id}"]
        if venue:
            info_parts.append(f"Source: {venue}")
        ref_lines.append(f"  {' | '.join(info_parts)}")
        if title:
            ref_lines.append(f"  Title: {_truncate_text(title, 60)}")
        if text:
            ref_lines.append(f"  Content: {_truncate_text(text, max_content_length)}")
        ref_lines.append("")

    # Clinical Trials
    ref_lines.append("### Clinical Trials\n")
    for tag in trial_tags:
        match_new = re.match(r"\[@trial\s*\|\s*([^\]]+)\]", tag, re.IGNORECASE)
        if match_new:
            trial_id = match_new.group(1).strip()
        else:
            match_compact = re.match(r"\[@trial:([^\]]+)\]", tag, re.IGNORECASE)
            trial_id = match_compact.group(1) if match_compact else None
        if trial_id is None:
            continue
        trial = cache.get_trial(trial_id)
        if trial is None and trial_info:
            trial = trial_info.get(trial_id) or trial_info.get(str(trial_id))
        ref_lines.append(f"{tag}")
        ref_lines.append(f"  Trial ID: {trial_id}")
        if trial:
            if trial.get("name"):
                ref_lines.append(f"  Name: {trial.get('name')}")
            metadata = trial.get("metadata") if isinstance(trial.get("metadata"), dict) else {}
            source_trial_id = str(metadata.get("source_trial_id") or metadata.get("citation") or "").strip()
            phase = str(metadata.get("phase") or "").strip()
            status = str(metadata.get("status") or "").strip()
            sponsor = str(metadata.get("sponsor") or "").strip()
            source_url = str(metadata.get("source_url") or "").strip()
            info_parts = []
            if source_trial_id and source_trial_id != trial_id:
                info_parts.append(f"Registry ID: {source_trial_id}")
            if phase:
                info_parts.append(f"Phase: {phase}")
            if status:
                info_parts.append(f"Status: {status}")
            if sponsor:
                info_parts.append(f"Sponsor: {_truncate_text(sponsor, 60)}")
            if info_parts:
                ref_lines.append(f"  {' | '.join(info_parts)}")
            if trial.get("reason"):
                ref_lines.append(f"  Rationale: {_truncate_text(trial.get('reason', ''), 60)}")
            if source_url:
                ref_lines.append(f"  URL: {source_url}")
        else:
            ref_lines.append("  [Details not available]")
        ref_lines.append("")

    # Clinical Reports
    ref_lines.append("### Clinical Reports\n")
    for tag in report_tags:
        match_new = re.match(r"\[@([^|\]]+(?:\|[^|\]]+)*)\s+\|\s+([^\]]+)\]", tag)
        if match_new:
            report_id = match_new.group(1).strip()
            report_type_or_date = match_new.group(2).strip()
            report_info = _find_report_in_context(report_id, report_context) if report_context else None
            ref_lines.append(f"{tag}")
            if report_info:
                rtype = report_info.get("type", "")
                summary = _extract_report_summary(report_info, max_content_length)
                if report_type_or_date.upper() in ["LAB", "GENOMICS", "MR", "CT", "IMAGING", "PATHOLOGY", "CASE"]:
                    type_label = report_type_or_date
                else:
                    type_label = rtype.capitalize() if rtype else "Report"
                report_date = report_info.get("date", "") or report_info.get("report_date", "")
                if report_date:
                    date_str = str(report_date)[:10] if len(str(report_date)) > 10 else str(report_date)
                    ref_lines.append(f"  {type_label}: {report_id} ({date_str})")
                else:
                    ref_lines.append(f"  {type_label}: {report_id}")
                if summary:
                    ref_lines.append(f"  Content: {summary}")
            else:
                if report_type_or_date.upper() in ["LAB", "GENOMICS", "MR", "CT", "IMAGING", "PATHOLOGY", "CASE"]:
                    ref_lines.append(f"  {report_type_or_date}: {report_id}")
                else:
                    ref_lines.append(f"  Report: {report_id} | {report_type_or_date}")
            ref_lines.append("")
        else:
            match_compact = re.match(r"\[@([^|\]]+)\|([^\]]+)\]", tag)
            if match_compact:
                report_id = match_compact.group(1)
                report_date = match_compact.group(2)
                report_info = _find_report_in_context(report_id, report_context) if report_context else None
                ref_lines.append(f"{tag}")
                if report_info:
                    rtype = report_info.get("type", "")
                    summary = _extract_report_summary(report_info, max_content_length)
                    type_label = rtype.capitalize() if rtype else "Report"
                    ref_lines.append(f"  {type_label}: {report_id} | {report_date}")
                    if summary:
                        ref_lines.append(f"  Content: {summary}")
                else:
                    ref_lines.append(f"  Report: {report_id} | {report_date}")
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

    # Search through all report types and roles (including "case" for Chair-SA mode)
    for report_type in ["lab", "imaging", "pathology", "mutation", "case"]:
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
                       report.get("report_no", "") or
                       report.get("date", ""))
                if str(rid).strip() == report_id_str:
                    return {"type": report_type, **report}

    return None
