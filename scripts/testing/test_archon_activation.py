#!/usr/bin/env python3
"""
Quick test of @Archon system activation
"""

import sys
import subprocess
import requests
import time

def test_archon_activation():
    """Test the complete @Archon activation system"""
    print("🚀 Testing @Archon System Activation")
    print("=" * 50)
    
    results = {
        "universal_rules_check": False,
        "services_healthy": False,
        "manifest_available": False,
        "agents_available": False,
        "mcp_tools_working": False
    }
    
    # Test 1: Universal Rules Checker
    print("1. Testing Universal Rules Checker...")
    try:
        result = subprocess.run(
            ["python3", "UNIVERSAL_RULES_CHECKER.py", "--path", "."],
            cwd="/mnt/c/Jarvis/AI Workspace/Archon",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "VALIDATION PASSED" in result.stdout:
            results["universal_rules_check"] = True
            print("✅ Universal Rules Checker: PASSED")
        else:
            print(f"❌ Universal Rules Checker: FAILED - {result.stderr}")
    except Exception as e:
        print(f"❌ Universal Rules Checker: ERROR - {e}")
    
    # Test 2: Service Health Check
    print("\n2. Testing Service Health...")
    services = {
        "server": "http://localhost:8181/health",
        "agents": "http://localhost:8052/health", 
        "ui": "http://localhost:3737"
    }
    
    healthy_services = 0
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service.capitalize()} Service: HEALTHY")
                healthy_services += 1
            else:
                print(f"❌ {service.capitalize()} Service: UNHEALTHY ({response.status_code})")
        except Exception as e:
            print(f"❌ {service.capitalize()} Service: ERROR - {e}")
    
    results["services_healthy"] = healthy_services >= 2  # At least server and agents
    
    # Test 3: MANIFEST.md Availability
    print("\n3. Testing MANIFEST.md Availability in Containers...")
    try:
        result = subprocess.run(
            ["docker", "exec", "archon-agents", "test", "-f", "/app/MANIFEST.md"],
            capture_output=True,
            timeout=10
        )
        
        if result.returncode == 0:
            results["manifest_available"] = True
            print("✅ MANIFEST.md: Available in containers")
        else:
            print("❌ MANIFEST.md: Missing from containers")
    except Exception as e:
        print(f"❌ MANIFEST.md Check: ERROR - {e}")
    
    # Test 4: Agents Availability
    print("\n4. Testing Specialized Agents...")
    try:
        response = requests.get("http://localhost:8052/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            agent_count = len(data.get("agents_available", []))
            if agent_count >= 15:  # Should have many specialized agents
                results["agents_available"] = True
                print(f"✅ Specialized Agents: {agent_count} agents available")
                
                # Show some key agents
                key_agents = ["system_architect", "security_auditor", "code_reviewer", "test_generator"]
                available_agents = data.get("agents_available", [])
                for agent in key_agents:
                    status = "✅" if agent in available_agents else "❌"
                    print(f"   {status} {agent}")
            else:
                print(f"❌ Specialized Agents: Only {agent_count} agents (expected ≥15)")
    except Exception as e:
        print(f"❌ Agents Check: ERROR - {e}")
    
    # Test 5: MCP Tools (Basic test)
    print("\n5. Testing MCP Integration...")
    try:
        # Basic connectivity test
        response = requests.get("http://localhost:8051/health", timeout=5)
        if response.status_code == 200:
            results["mcp_tools_working"] = True
            print("✅ MCP Tools: Service available")
        else:
            print("❌ MCP Tools: Service unavailable")
    except Exception as e:
        print(f"❌ MCP Tools: ERROR - {e}")
    
    # Final Assessment
    print("\n" + "=" * 50)
    print("📊 @ARCHON ACTIVATION TEST RESULTS")
    print("=" * 50)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nScore: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 @ARCHON SYSTEM FULLY OPERATIONAL!")
        print("Ready for specialized agent orchestration and coding assistance.")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠️ @ARCHON SYSTEM MOSTLY OPERATIONAL")
        print("Core functionality available with some limitations.")
        return True
    else:
        print("\n❌ @ARCHON SYSTEM NEEDS ATTENTION")
        print("Please address the failing tests before using @Archon commands.")
        return False

if __name__ == "__main__":
    success = test_archon_activation()
    sys.exit(0 if success else 1)