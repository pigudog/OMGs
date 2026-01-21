"""Error handling utilities for OMGs system.

This module provides:
- safe_agent_call(): Unified agent call wrapper with error handling and fallback
- get_fallback_response(): Generate fallback responses based on context
- is_retryable_error(): Determine if an error is retryable
"""

import time
import random
from typing import Optional, Callable, Any
from core.agent import Agent, AgentError
from servers.trace import TraceLogger
from utils.console_utils import Color


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: The exception to check
    
    Returns:
        True if the error is retryable (e.g., network timeout, rate limit)
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Network/connection errors are retryable
    if "connection" in error_str or "timeout" in error_str or "network" in error_str:
        return True
    
    # Rate limit errors are retryable
    if "rate limit" in error_str or "ratelimit" in error_str or "429" in error_str:
        return True
    
    # API errors that might be transient
    if error_type in ["APIConnectionError", "RateLimitError", "Timeout"]:
        return True
    
    # Value/parsing errors are usually not retryable
    if error_type in ["ValueError", "KeyError", "AttributeError"]:
        return False
    
    # Default: assume not retryable
    return False


def get_fallback_response(role: str, stage: str) -> str:
    """
    Generate a fallback response when an agent call fails.
    
    Args:
        role: The role of the agent that failed
        stage: The stage where the failure occurred (e.g., "initial_opinion", "final_plan")
    
    Returns:
        A fallback response string
    """
    stage_fallbacks = {
        "initial_opinion": f"[Error: Unable to generate initial opinion. Role: {role}]",
        "final_plan": f"[Error: Unable to generate final plan. Role: {role}]",
        "turn_speak": f"[Error: Unable to speak in this turn. Role: {role}]",
        "summary": f"[Error: Unable to generate summary. Role: {role}]",
        "memory_update": f"[Error: Unable to update memory. Role: {role}]",
    }
    
    return stage_fallbacks.get(stage, f"[Error: Operation failed. Role: {role}, Stage: {stage}]")


def safe_agent_call(
    agent: Agent,
    prompt: str,
    role: str,
    stage: str,
    fallback: Optional[str] = None,
    trace: Optional[TraceLogger] = None,
    max_retries: int = 0,
    use_selection: bool = False
) -> str:
    """
    Safe agent call wrapper with error handling and fallback.
    
    Args:
        agent: The Agent instance to call
        prompt: The prompt to send to the agent
        role: The role name (for error reporting)
        stage: The stage name (e.g., "initial_opinion", "final_plan")
        fallback: Optional custom fallback response. If None, uses get_fallback_response()
        trace: Optional TraceLogger for error tracking
        max_retries: Maximum number of retries for retryable errors (default: 0, no retries)
        use_selection: If True, use run_selection() instead of chat()
    
    Returns:
        The agent response, or fallback response if all attempts fail
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            if use_selection:
                response = agent.run_selection(prompt)
            else:
                response = agent.chat(prompt)
            
            # Success - return response
            if trace and attempt > 0:
                trace.emit("agent_retry_success", {
                    "role": role,
                    "stage": stage,
                    "attempt": attempt + 1
                })
            return response
            
        except AgentError as e:
            last_error = e.original_error
            
            # Check if retryable and we have retries left
            if is_retryable_error(e.original_error) and attempt < max_retries:
                # Exponential backoff with jitter for rate limits
                # For 429 errors, wait longer: 2^attempt seconds (2s, 4s, 8s, ...)
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"{Color.WARNING}[RETRY] {role}/{stage} attempt {attempt + 1}/{max_retries + 1}, waiting {delay:.1f}s before retry...{Color.RESET}")
                time.sleep(delay)
                
                if trace:
                    trace.emit("agent_retry", {
                        "role": role,
                        "stage": stage,
                        "attempt": attempt + 1,
                        "error": str(e.original_error),
                        "error_type": type(e.original_error).__name__,
                        "delay_seconds": delay
                    })
                continue
            
            # Not retryable or out of retries - use fallback
            break
            
        except Exception as e:
            # Unexpected exception (not wrapped in AgentError)
            last_error = e
            break
    
    # All attempts failed - use fallback
    fallback_response = fallback if fallback is not None else get_fallback_response(role, stage)
    
    # Log error
    error_type = type(last_error).__name__ if last_error else "UnknownError"
    error_msg = str(last_error) if last_error else "Unknown error"
    
    print(f"{Color.WARNING}[WARNING] {role} failed at {stage}: {error_msg}{Color.RESET}")
    
    if trace:
        trace.emit("agent_error", {
            "role": role,
            "stage": stage,
            "error": error_msg,
            "error_type": error_type,
            "fallback_used": True,
            "attempts": max_retries + 1
        })
    
    return fallback_response
