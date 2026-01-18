# =========================================================
# 1. init_client
# =========================================================
from typing import Any, Dict, List, Optional
from aoai import OpenAIWrapper
from datetime import datetime, timedelta
import tiktoken
import os
import json
from pathlib import Path

# =========================================================
# 0. Configuration loader
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

###############################################################################
# Azure OpenAI Client Initialization
###############################################################################
def init_client(db_path: Optional[str] = None) -> OpenAIWrapper:
    """
    Initialize Azure OpenAI client using aoai.OpenAIWrapper.
    If db_path is not provided, loads from config file.
    """
    if db_path is None:
        config = get_paths_config()
        db_path = config["output_dirs"]["api_trace_db"]
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY.")

    return OpenAIWrapper(api_key=api_key, base_url=endpoint, db_path=db_path)


# =========================================================
# 2. Agent (stateful)
# =========================================================
class Agent:
    """
    A stateful agent wrapper around Azure OpenAI (via OpenAIWrapper).
    Maintains its own message history for multi-turn conversation.
    Also maintains a local interaction log for recording all model calls.
    """

    def __init__(
        self,
        instruction: str,
        role: str,
        model_info: str,
        client: OpenAIWrapper,
        examplers=None,
        max_tokens=5000,
        max_prompt_tokens: int = 6500,
        enable_local_log=True
    ):
        self.instruction = instruction
        self.role = role
        self.examplers = examplers or []
        self.client = client
        self.max_tokens = max_tokens
        self.deployment = model_info

        # Local interaction log for recording all model calls
        self.enable_local_log = enable_local_log
        self.local_log = []

        self.messages = []
        self.max_prompt_tokens = int(max_prompt_tokens)
        self._encoder = self._get_encoder()
        self._build_initial_messages()

    def _record_local(self, user_msg, reply):
        """Record current turn to local log."""
        if not self.enable_local_log:
            return
        self.local_log.append({
            "role": self.role,
            "user_message": user_msg,
            "assistant_reply": reply,
            "timestamp": datetime.now().isoformat()
        })

    def _build_initial_messages(self):
        system_text = self.instruction.strip()
        self.messages = [{"role": "system", "content": system_text}]

        for ex in self.examplers:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            r = ex.get("reason", "")
            self.messages.append({"role": "user", "content": q})
            self.messages.append({"role": "assistant", "content": f"{a}\n\n{r}"})

    # -------------------------
    # Token budget helpers
    # -------------------------
    def _get_encoder(self):
        """Best-effort tokenizer; Azure deployment names may not map to tiktoken model names."""
        try:
            return tiktoken.encoding_for_model(self.deployment)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        # Rough but stable: count tokens of role/content strings.
        n = 0
        for m in messages or []:
            n += len(self._encoder.encode((m.get("role") or "") + ":" + (m.get("content") or "")))
        return n

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Hard truncate a single message content to fit token budget (best-effort)."""
        if not text or max_tokens <= 0:
            return ""
        ids = self._encoder.encode(text)
        if len(ids) <= max_tokens:
            return text
        # Keep tail by default (often preserves output template and latest constraints)
        ids = ids[-max_tokens:]
        return self._encoder.decode(ids)

    def _trim_messages_to_budget(self, messages: List[Dict[str, str]], max_prompt_tokens: int) -> List[Dict[str, str]]:
        """Keep system + most recent turns within max_prompt_tokens. ALWAYS keep the latest message.

        Important: If system + latest message alone exceeds budget, truncate the latest message content
        instead of dropping it (prevents system-only calls).
        """
        if not messages:
            return []
        if max_prompt_tokens <= 0:
            return messages[-4:]  # ultra fallback

        system = [messages[0]] if messages and messages[0].get("role") == "system" else []
        rest = messages[1:] if system else list(messages)

        if not rest:
            return system

        # Always keep the latest message (usually the current user turn)
        last = rest[-1]
        cand = system + [last]

        # If even (system + last) is too large, truncate last content instead of dropping it.
        if self._estimate_tokens(cand) > max_prompt_tokens:
            sys_tokens = self._estimate_tokens(system)
            # leave a minimal budget for the last message
            budget_for_last = max(32, max_prompt_tokens - sys_tokens)
            truncated_last = dict(last)
            truncated_last["content"] = self._truncate_text_to_tokens(truncated_last.get("content") or "", budget_for_last)
            return system + [truncated_last]

        kept: List[Dict[str, str]] = []

        # Fill from the end (excluding last which is already kept)
        for m in reversed(rest[:-1]):
            cand2 = system + list(reversed(kept)) + [m, last]
            if self._estimate_tokens(cand2) > max_prompt_tokens:
                break
            kept.append(m)

        kept = list(reversed(kept))
        return system + kept + [last]

    def chat(self, message: str):
        """
        Sends a new user message and returns assistant response.
        Also records the interaction into local_log.
        """
        self.messages.append({"role": "user", "content": message})

        # Enforce prompt token budget (keeps system + most recent turns)
        self.messages = self._trim_messages_to_budget(self.messages, self.max_prompt_tokens)

        resp = self.client.chat_completion(
            model=self.deployment,
            messages=self.messages,
            max_completion_tokens=self.max_tokens
        )

        reply = resp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})

        # Automatically record this interaction in local log
        self._record_local(message, reply)

        return reply

    def temp_responses(self, message: str):
        reply = self.chat(message)
        return reply

    def inject_assistant(self, message: str):
        self.messages.append({"role": "assistant", "content": message})

    def run_selection(self, message: str):
        """
        Stateless selection (used for lab / imaging filtering).
        Does not affect the main memory or local log.
        """
        msgs = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": message},
        ]
        msgs = self._trim_messages_to_budget(msgs, min(self.max_prompt_tokens, 2500))

        resp = self.client.chat_completion(
            model=self.deployment,
            messages=msgs,
            max_completion_tokens=self.max_tokens
        )
        return resp.choices[0].message.content

# =========================================================
# 3. load_data (JSONL paths from outside)
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


# =========================================================
# 4. create_question
# =========================================================

def create_question(sample: dict):
    """
    For clinical/MDT:
    - question: normalized/structured case input used by the pipeline
    - question_raw: original raw user query text (for observability/HTML)
    """
    q = sample.get("question", "")
    return q

# =========================================================
# 5. setup_model (simple adapter)
# =========================================================

def setup_model(model_name: str):
    """
    Adapter for old code:
    Return (deployment_name, client)
    """
    client = init_client()
    return model_name, client

