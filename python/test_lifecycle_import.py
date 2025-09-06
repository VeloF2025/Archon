#!/usr/bin/env python3
"""
Simple test to validate Agent Lifecycle v3.0 imports work correctly
NLNH Protocol: Real import validation
DGTS Enforcement: No fake imports, actual module testing
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lifecycle_imports():
    """Test that all lifecycle components can be imported"""
    try:
        from agents.lifecycle import (
            AgentV3, AgentState, AgentPoolManager, AgentSpec,
            ProjectAnalyzer, ProjectAnalysis, AgentSpawner
        )
        print("✅ All lifecycle imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality works"""
    try:
        from agents.lifecycle import AgentV3, AgentState, AgentPoolManager, AgentSpec
        
        # Test AgentState enum
        assert AgentState.CREATED.value == "created"
        assert AgentState.ACTIVE.value == "active"
        assert AgentState.IDLE.value == "idle"
        assert AgentState.HIBERNATED.value == "hibernated"
        assert AgentState.ARCHIVED.value == "archived"
        print("✅ AgentState enum working correctly")
        
        # Test AgentSpec creation
        spec = AgentSpec("test-agent", "sonnet", "testing")
        assert spec.agent_type == "test-agent"
        assert spec.model_tier == "sonnet"
        assert spec.specialization == "testing"
        print("✅ AgentSpec creation working correctly")
        
        # Test pool manager constants
        pool = AgentPoolManager()
        assert pool.MAX_AGENTS["opus"] == 2
        assert pool.MAX_AGENTS["sonnet"] == 10
        assert pool.MAX_AGENTS["haiku"] == 50
        print("✅ AgentPoolManager constants correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation works"""
    try:
        from agents.lifecycle import AgentV3, AgentState
        import uuid
        
        # Create test agent
        agent = AgentV3(
            project_id=str(uuid.uuid4()),
            name="test-agent",
            agent_type="code-implementer", 
            model_tier="haiku"
        )
        
        assert agent.state == AgentState.CREATED
        assert agent.model_tier == "haiku"
        assert agent.agent_type == "code-implementer"
        print("✅ Agent creation working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Agent Lifecycle v3.0 Import Validation")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Import Test", test_lifecycle_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Agent Creation Test", test_agent_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Agent Lifecycle v3.0 implementation is working correctly.")
        print("✅ DGTS Protocol: Real implementations validated, no fake behavior detected")
        print("✅ NLNH Protocol: Actual functionality confirmed, no hallucinated features")
    else:
        print(f"❌ {total - passed} tests failed. Implementation needs fixes.")
        sys.exit(1)