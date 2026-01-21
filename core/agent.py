"""Stateful LLM Agent wrapper for OMGs system."""

from typing import Any, Dict, List, Optional
from datetime import datetime
import tiktoken
from clients import OpenAIWrapper


class AgentError(Exception):
    """Agent操作失败的自定义异常"""
    def __init__(self, role: str, operation: str, original_error: Exception):
        self.role = role
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"Agent {role} failed in {operation}: {original_error}")


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
        enable_local_log=True,
        enable_reasoning: bool = False
    ):
        self.instruction = instruction
        self.role = role
        self.examplers = examplers or []
        self.client = client
        self.max_tokens = max_tokens
        self.deployment = model_info
        self.enable_reasoning = enable_reasoning

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
        # Guard against oversized system prompts (can trigger API errors or truncation)
        sys_tokens = self._estimate_tokens([{"role": "system", "content": system_text}])
        if sys_tokens > self.max_prompt_tokens:
            budget = max(64, int(self.max_prompt_tokens))
            system_text = self._truncate_text_keep_head_tail(system_text, budget)
            print(
                f"[WARNING] System prompt for '{self.role}' exceeded token budget "
                f"({sys_tokens} > {self.max_prompt_tokens}). Truncated to fit."
            )
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

    def _truncate_text_keep_head_tail(self, text: str, max_tokens: int, head_ratio: float = 0.6) -> str:
        """Truncate by keeping head + tail to preserve rules and latest facts."""
        if not text or max_tokens <= 0:
            return ""
        ids = self._encoder.encode(text)
        if len(ids) <= max_tokens:
            return text
        head_tokens = max(1, int(max_tokens * head_ratio))
        tail_tokens = max_tokens - head_tokens
        head_ids = ids[:head_tokens]
        tail_ids = ids[-tail_tokens:] if tail_tokens > 0 else []
        return self._encoder.decode(head_ids) + "\n...\n" + self._encoder.decode(tail_ids)

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
        
        If enable_reasoning is True, supports OpenRouter reasoning mode and preserves
        reasoning_details in message history for continuation.
        
        Raises:
            AgentError: If the API call fails or response parsing fails
        """
        self.messages.append({"role": "user", "content": message})

        # Enforce prompt token budget (keeps system + most recent turns)
        self.messages = self._trim_messages_to_budget(self.messages, self.max_prompt_tokens)

        try:
            # Prepare extra_body for reasoning if enabled
            extra_body = None
            if self.enable_reasoning:
                extra_body = {"reasoning": {"enabled": True}}
            
            resp = self.client.chat_completion(
                model=self.deployment,
                messages=self.messages,
                max_completion_tokens=self.max_tokens,
                extra_body=extra_body
            )
            
            # Check if response is valid
            if not resp or not hasattr(resp, 'choices') or not resp.choices:
                raise ValueError("Empty or invalid API response")
            
            if not resp.choices[0] or not hasattr(resp.choices[0], 'message'):
                raise ValueError("Response missing message content")
            
            reply = resp.choices[0].message.content
            if reply is None:
                raise ValueError("Response content is None")
            
            # Build assistant message, preserving reasoning_details if present
            assistant_msg = {"role": "assistant", "content": reply}
            if self.enable_reasoning and hasattr(resp.choices[0].message, "reasoning_details"):
                reasoning_details = getattr(resp.choices[0].message, "reasoning_details", None)
                if reasoning_details is not None:
                    assistant_msg["reasoning_details"] = reasoning_details
            
            self.messages.append(assistant_msg)

            # Automatically record this interaction in local log
            self._record_local(message, reply)

            return reply
        except Exception as e:
            # Remove the user message we just added since the call failed
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
            raise AgentError(self.role, "chat", e)

    def temp_responses(self, message: str):
        reply = self.chat(message)
        return reply

    def inject_assistant(self, message: str):
        self.messages.append({"role": "assistant", "content": message})

    def run_selection(self, message: str):
        """
        Stateless selection (used for lab / imaging filtering).
        Does not affect the main memory or local log.
        
        Raises:
            AgentError: If the API call fails or response parsing fails
        """
        msgs = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": message},
        ]
        msgs = self._trim_messages_to_budget(msgs, min(self.max_prompt_tokens, 2500))

        try:
            # Prepare extra_body for reasoning if enabled
            extra_body = None
            if self.enable_reasoning:
                extra_body = {"reasoning": {"enabled": True}}
            
            resp = self.client.chat_completion(
                model=self.deployment,
                messages=msgs,
                max_completion_tokens=self.max_tokens,
                extra_body=extra_body
            )
            
            # Check if response is valid
            if not resp or not hasattr(resp, 'choices') or not resp.choices:
                raise ValueError("Empty or invalid API response")
            
            if not resp.choices[0] or not hasattr(resp.choices[0], 'message'):
                raise ValueError("Response missing message content")
            
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("Response content is None")
            
            return content
        except Exception as e:
            raise AgentError(self.role, "run_selection", e)
