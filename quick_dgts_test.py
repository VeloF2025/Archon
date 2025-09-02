#!/usr/bin/env python3
"""
Quick test for DGTS/NLNH Documentation-Driven Test Development
"""

from pathlib import Path

def main():
    print("DGTS/NLNH Documentation-Driven Test Development Check")
    print("=" * 60)
    
    # Check documentation
    prd_files = list(Path('.').glob('PRDs/*.md'))
    prp_files = list(Path('.').glob('PRPs/*.md')) 
    adr_files = list(Path('.').glob('**/ADR*.md'))
    
    total_docs = len(prd_files) + len(prp_files) + len(adr_files)
    print(f"Documentation files: {total_docs}")
    print(f"  - PRDs: {len(prd_files)}")
    print(f"  - PRPs: {len(prp_files)}")
    print(f"  - ADRs: {len(adr_files)}")
    
    # Check tests (limit to project directories)
    test_patterns = [
        'python/tests/**/*.py',
        'python/src/**/test_*.py',
        'archon-ui-main/test/**/*.ts',
        'archon-ui-main/test/**/*.js'
    ]
    
    test_files = []
    for pattern in test_patterns:
        test_files.extend(Path('.').glob(pattern))
    
    print(f"Test files: {len(test_files)}")
    
    # Check implementation files (project only)
    impl_patterns = [
        'python/src/**/*.py',
        'archon-ui-main/src/**/*.ts',
        'archon-ui-main/src/**/*.tsx'
    ]
    
    impl_files = []
    for pattern in impl_patterns:
        impl_files.extend(Path('.').glob(pattern))
    
    print(f"Implementation files: {len(impl_files)}")
    
    # Calculate ratios
    if impl_files:
        test_ratio = len(test_files) / len(impl_files)
        print(f"Test coverage ratio: {test_ratio:.2f} (tests/impl)")
    
    # DGTS/NLNH Compliance Check
    print("\nDGTS/NLNH Compliance Check:")
    
    has_docs = total_docs > 0
    has_tests = len(test_files) > 0
    has_impl = len(impl_files) > 0
    
    print(f"  [{'PASS' if has_docs else 'FAIL'}] Documentation exists")
    print(f"  [{'PASS' if has_tests else 'FAIL'}] Tests exist")
    print(f"  [{'PASS' if has_impl else 'FAIL'}] Implementation exists")
    
    if has_docs and has_tests:
        print("\nStatus: DOCUMENTATION-DRIVEN DEVELOPMENT READY")
        print("Next steps:")
        print("  1. Agents must parse PRD/PRP/ADR for requirements")
        print("  2. Create tests from documented acceptance criteria")
        print("  3. Write implementation to pass doc-derived tests")
        print("  4. Validate tests cover all documented requirements")
    else:
        print("\nStatus: MISSING DGTS/NLNH REQUIREMENTS")
        if not has_docs:
            print("  - Create PRD/PRP/ADR documentation")
        if not has_tests:
            print("  - Create test files based on documentation")
    
    print("\nValidation System Files:")
    validation_files = [
        'python/src/agents/validation/doc_driven_validator.py',
        'python/src/agents/validation/agent_validation_enforcer.py',
        'python/src/agents/validation/dgts_validator.py'
    ]
    
    for vf in validation_files:
        exists = Path(vf).exists()
        print(f"  [{'EXISTS' if exists else 'MISSING'}] {vf}")
    
    print("=" * 60)
    print("DGTS/NLNH Test Complete")

if __name__ == "__main__":
    main()