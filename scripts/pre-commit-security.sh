#!/bin/bash
# Pre-commit Security Hook for Archon
# ==================================
# 
# This hook prevents commits containing sensitive information from being made.
# It runs automatically before each commit to scan for credentials, API keys,
# and other sensitive data.
#
# Installation:
#   cp scripts/pre-commit-security.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or use git hooks:
#   git config core.hooksPath scripts

set -e

echo "🔒 Running pre-commit security validation..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required for security validation"
    exit 1
fi

# Run security validation
python3 scripts/security-validation.py --fail-on-violations

if [ $? -eq 0 ]; then
    echo "✅ Security validation passed - commit allowed"
    exit 0
else
    echo "❌ Security validation failed - commit blocked"
    echo ""
    echo "To fix these issues:"
    echo "1. Remove any exposed credentials from your files"
    echo "2. Use .env.example with placeholder values"  
    echo "3. Ensure .gitignore is properly configured"
    echo "4. Run: python3 scripts/security-validation.py"
    echo ""
    echo "To bypass this check (NOT RECOMMENDED):"
    echo "git commit --no-verify"
    exit 1
fi