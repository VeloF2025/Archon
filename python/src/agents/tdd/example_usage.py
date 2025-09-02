#!/usr/bin/env python3
"""
Phase 9 TDD Enforcement - Example Usage
Demonstrates how to use the TDD enforcement system with browserbase-stagehand integration
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from .stagehand_test_engine import StagehandTestEngine, generate_tests_from_natural_language
from .browserbase_executor import BrowserbaseExecutor, execute_tests_in_cloud, validate_tdd_compliance
from .tdd_enforcement_gate import TDDEnforcementGate, enforce_tdd_compliance, is_feature_implementation_allowed
from .enhanced_dgts_validator import EnhancedDGTSValidator, validate_stagehand_tests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_phase9_tdd_enforcement():
    """
    Complete demonstration of Phase 9 TDD Enforcement system
    """
    
    print("🚀 Phase 9 TDD Enforcement with Browserbase-Stagehand Integration")
    print("=" * 80)
    
    # Example feature to implement
    feature_name = "user_authentication"
    project_path = "."
    
    # Feature requirements (normally from PRD/PRP)
    requirements = [
        "Users can log in with email and password",
        "Users can register new accounts", 
        "Users can reset forgotten passwords",
        "Authentication state persists across sessions",
        "Failed login attempts are rate limited"
    ]
    
    acceptance_criteria = [
        "Login form validates email format",
        "Password must be at least 8 characters",
        "Invalid credentials show error message", 
        "Successful login redirects to dashboard",
        "Registration creates new user account",
        "Password reset sends email with reset link",
        "Users remain logged in after page refresh",
        "5 failed attempts locks account for 15 minutes"
    ]
    
    user_stories = [
        "As a user, I want to log in so I can access my account",
        "As a user, I want to register so I can create a new account",
        "As a user, I want to reset my password so I can regain access"
    ]
    
    try:
        # Step 1: Generate Tests First (TDD Principle)
        print("\n📝 Step 1: Generating tests BEFORE implementation...")
        
        test_engine = StagehandTestEngine(project_path=project_path)
        test_generation_result = await test_engine.generate_tests_from_requirements(
            requirements=requirements,
            feature_name=feature_name,
            acceptance_criteria=acceptance_criteria,
            user_stories=user_stories
        )
        
        if not test_generation_result.success:
            print(f"❌ Test generation failed: {test_generation_result.message}")
            return
        
        print(f"✅ Generated {test_generation_result.total_tests} tests:")
        for test in test_generation_result.tests_generated:
            print(f"   - {test.name} ({test.test_type.value})")
        
        # Step 2: Validate TDD Compliance (Tests First Check)
        print("\n🛡️ Step 2: Validating TDD compliance...")
        
        enforcement_gate = TDDEnforcementGate(
            project_path=project_path,
            min_coverage_percentage=95.0,
            enable_gaming_detection=True
        )
        
        # Check if implementation is allowed (should fail - no tests passing yet)
        initial_validation = await enforcement_gate.validate_feature_development(
            feature_name=feature_name
        )
        
        print(f"Initial validation: {'✅ ALLOWED' if initial_validation.allowed else '❌ BLOCKED'}")
        print(f"Message: {initial_validation.message}")
        
        if initial_validation.violations:
            print("Violations detected:")
            for violation in initial_validation.violations[:3]:  # Show first 3
                print(f"   - {violation.severity.upper()}: {violation.description}")
        
        # Step 3: Execute Tests in Cloud (Should fail - no implementation yet)
        print("\n☁️ Step 3: Executing tests in Browserbase cloud...")
        
        test_files = [test.file_path for test in test_generation_result.tests_generated]
        
        executor = BrowserbaseExecutor(
            # api_key will come from environment
            max_concurrent_sessions=3
        )
        
        try:
            execution_result = await executor.execute_test_suite(test_files)
            
            print(f"Test execution: {'✅ PASSED' if execution_result.success else '❌ FAILED'}")
            print(f"Results: {execution_result.passed_tests}/{execution_result.total_tests} tests passed")
            print(f"Execution time: {execution_result.execution_time_ms}ms")
            
            if execution_result.errors:
                print("Errors:")
                for error in execution_result.errors[:2]:  # Show first 2 errors
                    print(f"   - {error}")
                    
        except Exception as e:
            print(f"⚠️ Test execution failed (expected without Browserbase API): {str(e)}")
        
        # Step 4: Enhanced Gaming Detection
        print("\n🕵️ Step 4: Enhanced gaming detection...")
        
        dgts_validator = EnhancedDGTSValidator(project_path)
        gaming_result = await dgts_validator.validate_stagehand_compliance()
        
        print(f"Gaming detection: {'✅ CLEAN' if gaming_result['compliant'] else '❌ GAMING DETECTED'}")
        print(f"Gaming score: {gaming_result['gaming_score']:.2f}/1.0")
        print(f"Sophistication level: {gaming_result['sophistication_level']}")
        
        if gaming_result['violations']:
            print(f"Gaming violations detected: {len(gaming_result['violations'])}")
            for violation in gaming_result['violations'][:2]:
                print(f"   - {violation['type']}: {violation['explanation']}")
        
        # Step 5: Demonstrate Implementation Block
        print("\n🚫 Step 5: Demonstrating implementation blocking...")
        
        # Try to check if implementation is allowed (should be blocked)
        implementation_allowed = is_feature_implementation_allowed(
            feature_name=feature_name,
            project_path=project_path
        )
        
        print(f"Implementation allowed: {'✅ YES' if implementation_allowed else '❌ NO'}")
        
        # Step 6: Final Enforcement Decision
        print("\n⚖️ Step 6: Final enforcement decision...")
        
        final_enforcement = await enforce_tdd_compliance(
            feature_name=feature_name,
            project_path=project_path
        )
        
        print(f"Final decision: {'✅ PROCEED' if final_enforcement.allowed else '❌ BLOCKED'}")
        print(f"Total violations: {final_enforcement.total_violations}")
        print(f"Critical violations: {final_enforcement.critical_violations}")
        
        if final_enforcement.blocked_features:
            print(f"Blocked features: {', '.join(final_enforcement.blocked_features)}")
        
        # Step 7: Generate Enforcement Report
        print("\n📊 Step 7: Generating enforcement report...")
        
        enforcement_report = await enforcement_gate.create_enforcement_report()
        
        print("Enforcement Statistics:")
        print(f"   - Success rate: {enforcement_report['enforcement_summary']['success_rate']:.1f}%")
        print(f"   - Total validations: {enforcement_report['enforcement_summary']['total_validations']}")
        print(f"   - Gaming attempts: {enforcement_report['gaming_detection']['attempts_detected']}")
        
        print("\n🎯 Phase 9 TDD Enforcement Demonstration Complete!")
        print("=" * 80)
        
        # Summary
        print("\n📋 SUMMARY:")
        print("✅ Tests generated from natural language requirements")
        print("✅ TDD compliance validation implemented")  
        print("✅ Cloud test execution integrated")
        print("✅ Enhanced gaming detection operational")
        print("✅ Implementation blocking enforced")
        print("✅ Comprehensive reporting available")
        
        return {
            "test_generation": test_generation_result,
            "enforcement_validation": final_enforcement,
            "gaming_detection": gaming_result,
            "execution_result": execution_result if 'execution_result' in locals() else None
        }
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"❌ Error during demonstration: {str(e)}")
        return None

async def demonstrate_gaming_detection():
    """
    Demonstrate advanced gaming detection capabilities
    """
    
    print("\n🕵️ Gaming Detection Demonstration")
    print("-" * 50)
    
    # Create example test content with gaming patterns
    gaming_test_content = '''
    import { test, expect } from '@playwright/test';
    import { Stagehand } from '@browserbasehq/stagehand';
    
    test('fake login test', async ({ page }) => {
        const stagehand = new Stagehand({ page, browserbaseAPIKey: 'fake' });
        
        // Gaming patterns:
        await stagehand.act('do nothing');  // Fake action
        await stagehand.observe('always true');  // Mock observation
        expect(true).toBeTruthy();  // Meaningless assertion
        
        // More sophisticated gaming:
        if (process.env.NODE_ENV === 'test') {
            return true;  // Conditional bypass
        }
    });
    '''
    
    print("Analyzing gaming patterns in test content...")
    
    from .enhanced_dgts_validator import detect_natural_language_gaming
    
    # Detect natural language gaming
    nl_violations = detect_natural_language_gaming(gaming_test_content)
    
    print(f"Natural language gaming violations: {len(nl_violations)}")
    for violation in nl_violations:
        print(f"   - Line {violation['line']}: {violation['gaming_phrase']}")
    
    # Full enhanced validation
    validator = EnhancedDGTSValidator(".")
    # Note: This would normally analyze actual project files
    
    print("✅ Gaming detection capabilities demonstrated")

async def simulate_tdd_workflow():
    """
    Simulate a complete TDD workflow from requirements to implementation
    """
    
    print("\n🔄 Complete TDD Workflow Simulation")
    print("-" * 50)
    
    workflow_steps = [
        "1. Requirements gathering from PRD/PRP",
        "2. Test generation from natural language",
        "3. Test execution (failing - no implementation)",
        "4. Gaming pattern detection",
        "5. Implementation blocking enforcement",
        "6. [After implementation] Test execution (passing)",
        "7. Final approval and deployment clearance"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print("\n🎯 This workflow ensures:")
    print("   ✅ Tests are always created first")
    print("   ✅ Implementation is blocked until tests pass")
    print("   ✅ Gaming attempts are detected and prevented")
    print("   ✅ Quality gates are enforced at every step")
    print("   ✅ Full compliance with TDD principles")

if __name__ == "__main__":
    """
    Run the complete Phase 9 TDD Enforcement demonstration
    """
    
    print("🚀 Starting Phase 9 TDD Enforcement Demonstration...")
    
    async def run_demo():
        # Main demonstration
        result = await demonstrate_phase9_tdd_enforcement()
        
        # Gaming detection demo
        await demonstrate_gaming_detection()
        
        # TDD workflow simulation
        await simulate_tdd_workflow()
        
        print("\n🎉 All demonstrations completed successfully!")
        return result
    
    # Run the demonstration
    try:
        result = asyncio.run(run_demo())
        
        if result:
            print("\n📊 Final Results Available:")
            print(f"   - Tests generated: {result['test_generation'].total_tests if result['test_generation'] else 'N/A'}")
            print(f"   - Gaming score: {result['gaming_detection']['gaming_score']:.2f}")
            print(f"   - Implementation allowed: {result['enforcement_validation'].allowed}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        logger.exception("Demonstration error")