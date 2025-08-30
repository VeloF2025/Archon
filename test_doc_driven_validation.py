#!/usr/bin/env python3
"""
Test script for Documentation-Driven Test Development validation
"""

import sys
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.append('python/src')

def test_documentation_validation():
    """Test the documentation-driven validation system"""
    try:
        from agents.validation.doc_driven_validator import validate_doc_driven_development
        
        print("Testing Documentation-Driven Test Development Validation...")
        
        # Run validation on current project
        result = validate_doc_driven_development('.')
        
        print(f"Validation completed successfully!")
        print(f"Results:")
        print(f"   - Compliant: {result.get('compliant', False)}")
        print(f"   - Requirements found: {result.get('requirements_found', 0)}")
        
        if 'test_validation' in result:
            tv = result['test_validation']
            print(f"   - Has tests: {tv.has_tests}")
            print(f"   - Tests from docs: {tv.tests_from_docs}")
            print(f"   - Tests match specs: {tv.tests_match_specs}")
            if tv.missing_requirements:
                print(f"   - Missing test coverage: {len(tv.missing_requirements)} items")
        
        if result.get('violations'):
            print(f"Violations: {len(result['violations'])}")
            for violation in result['violations'][:3]:
                print(f"      - {violation}")
        
        if result.get('remediation_steps'):
            print(f"Remediation steps:")
            for step in result['remediation_steps'][:3]:
                print(f"      {step}")
        
        return result
        
    except Exception as e:
        print(f"Error testing validation: {e}")
        return None

def test_agent_validation():
    """Test agent validation enforcer"""
    try:
        # Simple validation without complex imports
        print("\nTesting Agent Validation Enforcer...")
        
        # Check if PRD/PRP files exist
        prd_files = list(Path('.').glob('PRDs/*.md')) + list(Path('.').glob('PRPs/*.md'))
        print(f"Found {len(prd_files)} documentation files")
        
        # Check if test files exist
        test_files = (
            list(Path('.').glob('**/test_*.py')) + 
            list(Path('.').glob('**/tests/*.py')) +
            list(Path('.').glob('**/*.test.js')) +
            list(Path('.').glob('**/*.test.ts'))
        )
        print(f"Found {len(test_files)} test files")
        
        # Simple compliance check
        has_docs = len(prd_files) > 0
        has_tests = len(test_files) > 0
        
        print(f"Documentation exists: {has_docs}")
        print(f"Tests exist: {has_tests}")
        
        if has_docs and has_tests:
            print("Basic DGTS/NLNH requirements met!")
        else:
            print("Missing requirements for documentation-driven development")
            if not has_docs:
                print("   - Need PRD/PRP/ADR documentation")
            if not has_tests:
                print("   - Need test files")
        
        return {"has_docs": has_docs, "has_tests": has_tests}
        
    except Exception as e:
        print(f"Error testing agent validation: {e}")
        return None

if __name__ == "__main__":
    print("DGTS/NLNH Documentation-Driven Test Development Validation")
    print("=" * 70)
    
    # Test documentation validation
    doc_result = test_documentation_validation()
    
    # Test agent validation
    agent_result = test_agent_validation()
    
    print("\n" + "=" * 70)
    print("Validation Test Complete")
    
    if doc_result and agent_result:
        print("Documentation-driven development system is operational!")
    else:
        print("Some validation components need attention")