#!/usr/bin/env python3
"""
Quick connection test for all LLM providers (Azure OpenAI, OpenAI, OpenRouter).

Usage:
    python clients/test_connection.py [--provider azure|openai|openrouter|all]
    # or
    python -m clients.test_connection [--provider azure|openai|openrouter|all]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


def test_azure() -> Tuple[bool, str]:
    """Test Azure OpenAI connection."""
    try:
        from core.client import init_client
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if not endpoint or not api_key:
            return False, "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY"
        
        client = init_client(provider="azure")
        
        # Simple test call - try common Azure model names
        test_models = ["gpt-4", "gpt-35-turbo", "gpt-4o", "gpt-5.1"]
        response = None
        last_error = None
        
        for model in test_models:
            try:
                response = client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'Azure connection successful' in one sentence."}],
                    max_completion_tokens=50
                )
                break
            except Exception as e:
                last_error = str(e)
                continue
        
        if response is None:
            return False, f"❌ Azure OpenAI failed: Could not connect with any model. Last error: {last_error}"
        
        result = response.choices[0].message.content
        return True, f"✅ Azure OpenAI: {result}"
        
    except Exception as e:
        return False, f"❌ Azure OpenAI failed: {str(e)}"


def test_openai() -> Tuple[bool, str]:
    """Test OpenAI official API connection."""
    try:
        from core.client import init_client
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return False, "Missing OPENAI_API_KEY"
        
        client = init_client(provider="openai")
        
        # Simple test call with common OpenAI models
        test_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
        response = None
        last_error = None
        
        for model in test_models:
            try:
                response = client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'OpenAI connection successful' in one sentence."}],
                    max_completion_tokens=50
                )
                break
            except Exception as e:
                last_error = str(e)
                continue
        
        if response is None:
            return False, f"❌ OpenAI failed: Could not connect with any model. Last error: {last_error}"
        
        result = response.choices[0].message.content
        return True, f"✅ OpenAI: {result}"
        
    except Exception as e:
        return False, f"❌ OpenAI failed: {str(e)}"


def test_openrouter() -> Tuple[bool, str]:
    """Test OpenRouter connection."""
    try:
        from core.client import init_client
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            return False, "Missing OPENROUTER_API_KEY"
        
        client = init_client(provider="openrouter")
        
        # Simple test call with common free models for testing
        test_models = [
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-flash-1.5-8b:free",
            "meta-llama/llama-3.2-3b-instruct:free"
        ]
        response = None
        last_error = None
        
        for model in test_models:
            try:
                response = client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'OpenRouter connection successful' in one sentence."}],
                    max_completion_tokens=50
                )
                break
            except Exception as e:
                last_error = str(e)
                continue
        
        if response is None:
            return False, f"❌ OpenRouter failed: Could not connect with any model. Last error: {last_error}"
        
        result = response.choices[0].message.content
        return True, f"✅ OpenRouter: {result}"
        
    except Exception as e:
        return False, f"❌ OpenRouter failed: {str(e)}"


def test_auto_detect() -> Tuple[bool, str]:
    """Test auto-detection based on model name."""
    try:
        from core.client import init_client_from_config
        
        detection_tests = []
        
        # Test Azure detection (only check if env vars are set)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if endpoint and azure_key:
            azure_client = init_client_from_config(model="gpt-4")
            if azure_client.provider != "azure":
                return False, f"Auto-detect failed: expected 'azure' for gpt-4, got '{azure_client.provider}'"
            detection_tests.append("Azure")
        
        # Test OpenAI detection (only check if env var is set)
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key:
            # If Azure is not configured, gpt-4 should use OpenAI
            if not (endpoint and azure_key):
                openai_client = init_client_from_config(model="gpt-4")
                if openai_client.provider != "openai":
                    return False, f"Auto-detect failed: expected 'openai' for gpt-4, got '{openai_client.provider}'"
            detection_tests.append("OpenAI")
        
        # Test OpenRouter detection (only check if env var is set)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if openrouter_key:
            openrouter_client = init_client_from_config(model="google/gemini-2.0-flash-exp:free")
            if openrouter_client.provider != "openrouter":
                return False, f"Auto-detect failed: expected 'openrouter' for google/*, got '{openrouter_client.provider}'"
            detection_tests.append("OpenRouter")
        
        if not detection_tests:
            return False, "Auto-detect skipped: No provider environment variables set"
        
        return True, f"✅ Auto-detection: Working correctly (tested: {', '.join(detection_tests)})"
        
    except Exception as e:
        return False, f"❌ Auto-detection failed: {str(e)}"


def main():
    """Run tests based on provider argument."""
    parser = argparse.ArgumentParser(description="Test LLM provider connections")
    parser.add_argument(
        '--provider',
        type=str,
        choices=['azure', 'openai', 'openrouter', 'all'],
        default='all',
        help="Provider to test: 'azure', 'openai', 'openrouter', or 'all' (default: all)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("OMGs LLM Provider Connection Test")
    print("=" * 60)
    if args.provider != 'all':
        print(f"Testing provider: {args.provider}")
    print()
    
    results = []
    
    # Run tests based on provider argument
    if args.provider in ['azure', 'all']:
        print("Testing Azure OpenAI...")
        azure_ok, azure_msg = test_azure()
        results.append(("Azure OpenAI", azure_ok, azure_msg))
        print(azure_msg)
        print()
    
    if args.provider in ['openai', 'all']:
        print("Testing OpenAI...")
        openai_ok, openai_msg = test_openai()
        results.append(("OpenAI", openai_ok, openai_msg))
        print(openai_msg)
        print()
    
    if args.provider in ['openrouter', 'all']:
        print("Testing OpenRouter...")
        openrouter_ok, openrouter_msg = test_openrouter()
        results.append(("OpenRouter", openrouter_ok, openrouter_msg))
        print(openrouter_msg)
        print()
    
    # Auto-detection test only runs when testing all providers
    if args.provider == 'all':
        print("Testing auto-detection...")
        auto_ok, auto_msg = test_auto_detect()
        results.append(("Auto-detection", auto_ok, auto_msg))
        print(auto_msg)
        print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, ok, msg in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status} - {name}")
    
    # Exit code - only fail if critical tests failed
    critical_tests = [r for r in results if r[0] != "Auto-detection"]
    all_critical_passed = all(ok for _, ok, _ in critical_tests if "Missing" not in str(_))
    
    # Check if at least one provider is configured (only when testing all)
    if args.provider == 'all':
        has_azure = os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")
        has_openai = os.getenv("OPENAI_API_KEY")
        has_openrouter = os.getenv("OPENROUTER_API_KEY")
        
        if not has_azure and not has_openai and not has_openrouter:
            print()
            print("⚠️  No provider environment variables found.")
            print("   Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for Azure, or")
            print("   Set OPENAI_API_KEY for OpenAI, or")
            print("   Set OPENROUTER_API_KEY for OpenRouter")
            sys.exit(1)
    
    if not all_critical_passed:
        print()
        print("⚠️  Some tests failed. Please check your environment variables and API keys.")
        sys.exit(1)
    else:
        print()
        print("🎉 All configured tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
