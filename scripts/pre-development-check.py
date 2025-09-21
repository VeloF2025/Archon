#!/usr/bin/env python3
"""
Pre-development check script
Ensures enhanced spec process is followed before any development
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python pre-development-check.py <feature-name>")
        sys.exit(1)

    feature_name = sys.argv[1]
    workflow = EnhancedDevelopmentWorkflow(Path.cwd())

    if not workflow.validate_before_development(feature_name):
        print("[BLOCKED] Development cannot proceed without proper specification")
        sys.exit(1)

    print("[OK] Development approved")

if __name__ == "__main__":
    main()
