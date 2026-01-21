"""Configuration loading utilities."""

from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path


# =========================================================
# Paths Configuration
# =========================================================
_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def load_paths_config(config_path: str = "config/paths.json") -> Dict[str, Any]:
    """
    Load paths configuration from JSON file.
    Returns default config if file not found (backward compatibility).
    
    Args:
        config_path: Path to the configuration file (relative to project root)
    
    Returns:
        Dictionary containing path configurations
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    # Try to find config file relative to project root
    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / config_path
    
    default_config = {
        "data_files": {
            "lab_reports": "files/lab_reports_summary.jsonl",
            "imaging_reports": "files/imaging_reports.jsonl",
            "pathology_reports": None,
            "mutation_reports": "files/mutation_reports.jsonl",
            "trials": "all_trials_filtered.json"
        },
        "rag_store": {
            "base_dir": "rag_store",
            "index_dir_template": "rag_store/{role}/index/chroma",
            "collection_name_template": "{role}_chunks",
            "embedding_model": "BAAI/bge-m3",
            "use_per_role_rag": False,
            "default_role": "chair",
            "available_roles": ["chair", "oncologist", "radiologist", "pathologist", "nuclear"]
        },
        "output_dirs": {
            "output_answer": "output_answer",
            "mdt_logs": "mdt_logs",
            "api_trace_db": "api_trace.db"
        }
    }
    
    if not config_file.exists():
        print(f"[WARNING] Config file not found at {config_file}, using default paths")
        _CONFIG_CACHE = default_config
        return _CONFIG_CACHE
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        
        # Merge with defaults to ensure all keys exist
        config = {
            "data_files": {**default_config["data_files"], **loaded_config.get("data_files", {})},
            "rag_store": {**default_config["rag_store"], **loaded_config.get("rag_store", {})},
            "output_dirs": {**default_config["output_dirs"], **loaded_config.get("output_dirs", {})}
        }
        
        _CONFIG_CACHE = config
        print(f"[INFO] Loaded paths config from {config_file}")
        return config
    except Exception as e:
        print(f"[WARNING] Failed to load config from {config_file}: {e}, using default paths")
        _CONFIG_CACHE = default_config
        return _CONFIG_CACHE

def get_paths_config() -> Dict[str, Any]:
    """Get cached paths configuration."""
    if _CONFIG_CACHE is None:
        return load_paths_config()
    return _CONFIG_CACHE


# =========================================================
# MDT Prompts Configuration
# =========================================================
_MDT_PROMPTS_CACHE: Optional[Dict[str, Any]] = None

def load_mdt_prompts(config_path: str = "config/mdt_prompts.json") -> Dict[str, Any]:
    """
    Load MDT prompts configuration from JSON file.
    
    Args:
        config_path: Path to the MDT prompts configuration file
    
    Returns:
        Dictionary containing MDT prompts
    """
    global _MDT_PROMPTS_CACHE
    if _MDT_PROMPTS_CACHE is not None:
        return _MDT_PROMPTS_CACHE
    
    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        print(f"[WARNING] MDT prompts config not found at {config_file}")
        _MDT_PROMPTS_CACHE = {}
        return _MDT_PROMPTS_CACHE
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            _MDT_PROMPTS_CACHE = json.load(f)
        return _MDT_PROMPTS_CACHE
    except Exception as e:
        print(f"[WARNING] Failed to load MDT prompts from {config_file}: {e}")
        _MDT_PROMPTS_CACHE = {}
        return _MDT_PROMPTS_CACHE


def get_mdt_prompts() -> Dict[str, Any]:
    """Get cached MDT prompts configuration."""
    if _MDT_PROMPTS_CACHE is None:
        return load_mdt_prompts()
    return _MDT_PROMPTS_CACHE


# =========================================================
# Data Loading Utilities
# =========================================================
def load_data(test_path: str, train_path: str = None):
    """
    Load jsonl files:
    - test_path: required
    - train_path: optional (for exemplars)
    """
    test_qa = []
    examplers = []

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test_path does not exist: {test_path}")

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_qa.append(json.loads(line))

    if train_path and os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                examplers.append(json.loads(line))

    return test_qa, examplers


def create_question(sample: dict):
    """
    For clinical/MDT:
    - question: normalized/structured case input used by the pipeline
    - question_raw: original raw user query text (for observability/HTML)
    """
    q = sample.get("question", "")
    normalized, warnings = normalize_case_schema(q)
    if warnings:
        meta = sample.get("meta_info") or sample.get("patient_id") or "unknown"
        for w in warnings:
            print(f"[WARNING] Case schema ({meta}): {w}")
    return normalized


def normalize_case_schema(case_json: Any) -> tuple[Dict[str, Any], List[str]]:
    """
    Normalize case schema for MDT pipeline and return warnings.
    Keeps patient facts intact while aligning key section locations/types.
    """
    warnings: List[str] = []
    if not isinstance(case_json, dict):
        return {}, ["question is not a dict; replaced with empty object"]

    normalized = dict(case_json)

    def _ensure_dict(key: str):
        val = normalized.get(key)
        if val is None:
            normalized[key] = {}
            warnings.append(f"missing section '{key}'; inserted empty object")
        elif not isinstance(val, dict):
            normalized[key] = {}
            warnings.append(f"section '{key}' is not an object; replaced with empty object")

    # Core sections expected by the MDT pipeline
    for section in ["CASE_CORE", "TIMELINE", "MED_ONC", "RADIOLOGY", "PATHOLOGY", "NUC_MED", "LAB_TRENDS"]:
        _ensure_dict(section)

    case_core = normalized.get("CASE_CORE", {})
    timeline_top = normalized.get("TIMELINE", {})
    timeline_core = case_core.get("TIMELINE") if isinstance(case_core, dict) else None

    # Align TIMELINE between CASE_CORE and top-level
    if timeline_top == {} and isinstance(timeline_core, dict) and timeline_core:
        normalized["TIMELINE"] = dict(timeline_core)
        warnings.append("TIMELINE found only under CASE_CORE; copied to top-level")
    elif timeline_top and (timeline_core is None or timeline_core == {}):
        if isinstance(case_core, dict):
            case_core["TIMELINE"] = dict(timeline_top)
            normalized["CASE_CORE"] = case_core
            warnings.append("TIMELINE found only at top-level; copied into CASE_CORE")
    elif isinstance(timeline_core, dict) and isinstance(timeline_top, dict):
        if timeline_top and timeline_core and timeline_top != timeline_core:
            warnings.append("TIMELINE differs between CASE_CORE and top-level; kept top-level for pipeline")

    # Minimal validation for CASE_CORE subkeys
    if isinstance(case_core, dict):
        for key in ["DIAGNOSIS", "LINE_OF_THERAPY", "BIOMARKERS", "CURRENT_STATUS"]:
            if key not in case_core:
                warnings.append(f"CASE_CORE missing '{key}'")

    return normalized, warnings


def setup_model(model_name: str, provider: Optional[str] = None) -> tuple:
    """
    Initialize model and client.
    
    Parameters:
    - model_name: Model/deployment name
    - provider: Optional provider type ("azure", "openai", "openrouter", "auto").
                If None or "auto", auto-detects based on model name.
                Otherwise, uses the specified provider.
    
    Returns:
    - Tuple of (model_name, client)
    """
    # Import here to avoid circular dependency
    from core.client import init_client, init_client_from_config
    
    if provider is None or provider == "auto":
        # Auto-detect provider based on model name
        client = init_client_from_config(model=model_name)
    else:
        # Use specified provider
        client = init_client(provider=provider)
    
    return model_name, client
