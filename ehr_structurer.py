#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EHR Structurer - Entry point script for EHR extraction.

This is a thin wrapper that calls the actual implementation in servers/case_parser.py.
All logic is maintained in that module to avoid code duplication.

Usage:
    python ehr_structurer.py --input input.jsonl --output output.jsonl \
        --deployment gpt-4 --prompts config/prompts.json
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the actual implementation
from servers.case_parser import main

if __name__ == "__main__":
    main()
