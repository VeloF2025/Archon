#!/usr/bin/env python3
"""
DGTS/NLNH Workflow Demonstration
Shows how agents must follow documentation-driven development
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append('python/src')

def demonstrate_workflow():
    """Demonstrate the DGTS/NLNH workflow"""
    print("DGTS/NLNH Documentation-Driven Test Development Workflow")
    print("=" * 70)
    
    print("\n1. AGENT ACTIVATION - Must validate before development")
    print("   Command: enforce_agent_validation(agent_name, task)")
    print("   Example: enforce_agent_validation('code-implementer', 'Add user authentication')")
    
    print("\n2. DOCUMENTATION SCANNING")
    print("   System scans for:")
    print("   - PRD files: Product Requirements Documents")
    print("   - PRP files: Product Requirements Prompts") 
    print("   - ADR files: Architectural Decision Records")
    
    print("\n3. REQUIREMENTS EXTRACTION")
    print("   From documentation, extract:")
    print("   - Functional requirements")
    print("   - Acceptance criteria")
    print("   - User stories")
    print("   - Specifications")
    
    print("\n4. TEST CREATION VALIDATION")
    print("   Before ANY implementation:")
    print("   - Tests must exist that validate documented requirements")
    print("   - Tests must reference the documentation")
    print("   - Tests must cover acceptance criteria")
    print("   - DEVELOPMENT BLOCKED if tests missing")
    
    print("\n5. IMPLEMENTATION PHASE")
    print("   Only after tests exist:")
    print("   - Write minimal code to pass doc-derived tests")
    print("   - No features beyond documented requirements")
    print("   - No scope creep allowed")
    
    print("\n6. DGTS GAMING DETECTION")
    print("   System blocks:")
    print("   - Fake tests that always pass")
    print("   - Mock data instead of real implementations")
    print("   - Commented out validation rules")
    print("   - Stub functions that don't work")
    
    print("\n7. POST-DEVELOPMENT VALIDATION")
    print("   After implementation:")
    print("   - Scan for gaming patterns")
    print("   - Calculate gaming score")
    print("   - Block agents with high gaming scores")
    print("   - Monitor behavior patterns")
    
    print("\nARCHON PROJECT STATUS:")
    
    # Check current status
    prd_files = list(Path('.').glob('PRDs/*.md'))
    prp_files = list(Path('.').glob('PRPs/*.md'))
    adr_files = list(Path('.').glob('**/ADR*.md'))
    
    print(f"  Documentation: {len(prd_files + prp_files + adr_files)} files")
    print(f"    - PRDs: {len(prd_files)}")
    print(f"    - PRPs: {len(prp_files)}")  
    print(f"    - ADRs: {len(adr_files)}")
    
    if prp_files:
        print(f"  Latest PRP: {prp_files[0].name}")
    
    test_files = list(Path('python/tests').glob('**/*.py')) + list(Path('archon-ui-main/test').glob('**/*.ts'))
    print(f"  Test files: {len(test_files)}")
    
    validation_files = [
        'python/src/agents/validation/doc_driven_validator.py',
        'python/src/agents/validation/agent_validation_enforcer.py', 
        'python/src/agents/validation/dgts_validator.py'
    ]
    
    existing_validation = [vf for vf in validation_files if Path(vf).exists()]
    print(f"  Validation system: {len(existing_validation)}/{len(validation_files)} files")
    
    print("\nENFORCEMENT STATUS:")
    if len(existing_validation) == len(validation_files):
        print("  [ACTIVE] All validation components operational")
        print("  [READY] Agents can be validated before development")
        print("  [PROTECTED] DGTS gaming detection enabled")
    else:
        print("  [INCOMPLETE] Some validation components missing")
    
    print("\nNEXT DEVELOPMENT MUST FOLLOW:")
    print("  1. Agent calls enforce_agent_validation() first")
    print("  2. System validates documentation exists") 
    print("  3. System checks test coverage for requirements")
    print("  4. Agent blocked if tests missing or inadequate")
    print("  5. Agent creates tests from documented acceptance criteria")
    print("  6. Agent implements minimal code to pass tests")
    print("  7. System validates no gaming patterns introduced")
    print("  8. Agent marked complete or blocked based on behavior")
    
    print("\n" + "=" * 70)
    print("DGTS/NLNH WORKFLOW DEMONSTRATION COMPLETE")
    print("Documentation-driven development is now ENFORCED")

def show_example_usage():
    """Show example of how agents should use the system"""
    print("\nEXAMPLE AGENT WORKFLOW:")
    print("-" * 40)
    
    print("# Before starting any development:")
    print("from agents.validation.agent_validation_enforcer import enforce_agent_validation")
    print("")
    print("result = enforce_agent_validation(")
    print("    agent_name='code-implementer',")
    print("    task_description='Implement JWT authentication system'")
    print(")")
    print("")
    print("if not result.validation_passed:")
    print("    print('BLOCKED:', result.errors)")
    print("    for step in result.remediation_steps:")
    print("        print('FIX:', step)")
    print("    return  # Cannot proceed")
    print("")
    print("# Only if validation passes:")
    print("# 1. Parse PRD/PRP requirements")
    print("# 2. Create tests from acceptance criteria") 
    print("# 3. Implement code to pass tests")
    print("# 4. Run post-development validation")

if __name__ == "__main__":
    demonstrate_workflow()
    show_example_usage()