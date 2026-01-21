"""
Module entry point for running connection tests.

Usage:
    python -m clients.test_connection [--provider azure|openai|openrouter|all]
    # or
    python clients/test_connection.py [--provider azure|openai|openrouter|all]
"""

# Import path setup
import sys
from pathlib import Path

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from clients.test_connection import main

if __name__ == "__main__":
    main()
