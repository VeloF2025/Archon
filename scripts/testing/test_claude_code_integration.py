#!/usr/bin/env python3
"""
Claude Code Integration Validation Test
Tests the comprehensive Claude Code Task tool integration
"""

import asyncio
import json
import time
from datetime import datetime
import httpx

async def test_claude_code_integration():
    """Test all aspects of Claude Code integration"""
    
    print("Testing Claude Code Integration")
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "overall_status": "PASS",
        "integration_rate": 0.0,
        "claude_code_bridge_working": False,
        "file_monitoring_available": False,
        "agents_tested": 0,
        "agents_passed": 0
    }
    
    base_url = "http://localhost:8181/api/claude-code"
    
    async with httpx.AsyncClient(timeout=30) as client:
        
        # Test 1: Health check
        print("Test 1: Health check")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                results["tests"]["health_check"] = {
                    "status": "PASS",
                    "bridge_working": health_data.get("integration_working", False),
                    "stats": health_data.get("stats", {})
                }
                results["claude_code_bridge_working"] = health_data.get("integration_working", False)
                results["integration_rate"] = health_data.get("stats", {}).get("integration_rate", 0.0)
                results["file_monitoring_available"] = health_data.get("stats", {}).get("file_monitoring_available", False)
                print(f"   PASS - Health check passed - Integration rate: {results['integration_rate']:.1%}")
            else:
                results["tests"]["health_check"] = {"status": "FAIL", "error": f"HTTP {response.status_code}"}
                results["overall_status"] = "FAIL"
                print(f"   FAIL - Health check failed - HTTP {response.status_code}")
        except Exception as e:
            results["tests"]["health_check"] = {"status": "FAIL", "error": str(e)}
            results["overall_status"] = "FAIL"
            print(f"   FAIL - Health check failed - {e}")
        
        # Test 2: Agents list
        print("Test 2: Available agents")
        try:
            response = await client.get(f"{base_url}/agents")
            if response.status_code == 200:
                agents_data = response.json()
                total_agents = len(agents_data.get("agents", []))
                available_agents = sum(1 for agent in agents_data.get("agents", []) if agent.get("available", False))
                
                results["tests"]["agents_list"] = {
                    "status": "PASS",
                    "total_agents": total_agents,
                    "available_agents": available_agents,
                    "integration_stats": agents_data.get("integration_stats", {})
                }
                print(f"   PASS - Agents list - {available_agents}/{total_agents} agents available")
            else:
                results["tests"]["agents_list"] = {"status": "FAIL", "error": f"HTTP {response.status_code}"}
                results["overall_status"] = "FAIL"
                print(f"   FAIL - Agents list failed - HTTP {response.status_code}")
        except Exception as e:
            results["tests"]["agents_list"] = {"status": "FAIL", "error": str(e)}
            results["overall_status"] = "FAIL"
            print(f"   FAIL - Agents list failed - {e}")
        
        # Test 3: Task execution (Python agent)
        print("Test 3: Python backend coder task")
        try:
            task_payload = {
                "subagent_type": "python_backend_coder",
                "description": "Integration test - create simple function",
                "prompt": "Create a Python function that returns 'Hello, Claude Code!'",
                "timeout": 20
            }
            
            start_time = time.time()
            response = await client.post(f"{base_url}/task", json=task_payload)
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                task_result = response.json()
                results["tests"]["python_task"] = {
                    "status": "PASS" if task_result.get("success", False) else "FAIL",
                    "execution_time": execution_time,
                    "agent_used": task_result.get("agent_used"),
                    "task_id": task_result.get("task_id"),
                    "result_length": len(str(task_result.get("result", "")))
                }
                
                if task_result.get("success", False):
                    results["agents_passed"] += 1
                    print(f"   PASS - Python task passed ({execution_time:.2f}s)")
                else:
                    results["overall_status"] = "FAIL"
                    print(f"   FAIL - Python task failed - {task_result.get('error', 'Unknown error')}")
                    
                results["agents_tested"] += 1
            else:
                results["tests"]["python_task"] = {"status": "FAIL", "error": f"HTTP {response.status_code}"}
                results["overall_status"] = "FAIL"
                print(f"   FAIL - Python task failed - HTTP {response.status_code}")
        except Exception as e:
            results["tests"]["python_task"] = {"status": "FAIL", "error": str(e)}
            results["overall_status"] = "FAIL"
            print(f"   FAIL - Python task failed - {e}")
        
        # Test 4: Task execution (TypeScript agent)
        print("Test 4: TypeScript frontend agent task")
        try:
            task_payload = {
                "subagent_type": "typescript_frontend_agent",
                "description": "Integration test - create React component",
                "prompt": "Create a simple React component that displays 'Integration Test'",
                "timeout": 20
            }
            
            start_time = time.time()
            response = await client.post(f"{base_url}/task", json=task_payload)
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                task_result = response.json()
                results["tests"]["typescript_task"] = {
                    "status": "PASS" if task_result.get("success", False) else "FAIL",
                    "execution_time": execution_time,
                    "agent_used": task_result.get("agent_used"),
                    "task_id": task_result.get("task_id"),
                    "result_length": len(str(task_result.get("result", "")))
                }
                
                if task_result.get("success", False):
                    results["agents_passed"] += 1
                    print(f"   PASS - TypeScript task passed ({execution_time:.2f}s)")
                else:
                    results["overall_status"] = "FAIL"
                    print(f"   FAIL - TypeScript task failed - {task_result.get('error', 'Unknown error')}")
                    
                results["agents_tested"] += 1
            else:
                results["tests"]["typescript_task"] = {"status": "FAIL", "error": f"HTTP {response.status_code}"}
                results["overall_status"] = "FAIL"
                print(f"   FAIL - TypeScript task failed - HTTP {response.status_code}")
        except Exception as e:
            results["tests"]["typescript_task"] = {"status": "FAIL", "error": str(e)}
            results["overall_status"] = "FAIL"
            print(f"   FAIL - TypeScript task failed - {e}")
        
        # Test 5: File trigger functionality
        print("Test 5: File trigger functionality")
        try:
            response = await client.post(f"{base_url}/file-trigger?file_path=integration_test.py&event_type=created")
            if response.status_code == 200:
                trigger_result = response.json()
                results["tests"]["file_trigger"] = {
                    "status": "PASS" if trigger_result.get("triggered", False) else "FAIL",
                    "message": trigger_result.get("message", ""),
                    "timestamp": trigger_result.get("timestamp")
                }
                print(f"   PASS - File trigger test passed")
            else:
                results["tests"]["file_trigger"] = {"status": "FAIL", "error": f"HTTP {response.status_code}"}
                results["overall_status"] = "FAIL"
                print(f"   FAIL - File trigger failed - HTTP {response.status_code}")
        except Exception as e:
            results["tests"]["file_trigger"] = {"status": "FAIL", "error": str(e)}
            results["overall_status"] = "FAIL"
            print(f"   FAIL - File trigger failed - {e}")
        
        # Test 6: Status endpoint
        print("Test 6: Bridge status endpoint")
        try:
            response = await client.get(f"{base_url}/status")
            if response.status_code == 200:
                status_data = response.json()
                results["tests"]["status_endpoint"] = {
                    "status": "PASS",
                    "bridge_working": status_data.get("claude_code_bridge_working", False),
                    "integration_rate": status_data.get("integration_rate", 0.0),
                    "available_agents": status_data.get("available_agents", 0),
                    "total_mapped": status_data.get("total_agents_mapped", 0)
                }
                print(f"   PASS - Status endpoint passed")
            else:
                results["tests"]["status_endpoint"] = {"status": "FAIL", "error": f"HTTP {response.status_code}"}
                results["overall_status"] = "FAIL"
                print(f"   FAIL - Status endpoint failed - HTTP {response.status_code}")
        except Exception as e:
            results["tests"]["status_endpoint"] = {"status": "FAIL", "error": str(e)}
            results["overall_status"] = "FAIL"
            print(f"   FAIL - Status endpoint failed - {e}")
    
    # Calculate final metrics
    passed_tests = sum(1 for test in results["tests"].values() if test.get("status") == "PASS")
    total_tests = len(results["tests"])
    test_success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    
    results["test_success_rate"] = test_success_rate
    results["passed_tests"] = passed_tests
    results["total_tests"] = total_tests
    
    # Print summary
    print("\nIntegration Test Summary")
    print(f"   Overall Status: {'PASS' if results['overall_status'] == 'PASS' else 'FAIL'}")
    print(f"   Test Success Rate: {test_success_rate:.1%} ({passed_tests}/{total_tests})")
    print(f"   Agent Integration Rate: {results['integration_rate']:.1%}")
    print(f"   Bridge Working: {'Yes' if results['claude_code_bridge_working'] else 'No'}")
    print(f"   Agents Tested: {results['agents_tested']}")
    print(f"   Agents Passed: {results['agents_passed']}")
    print(f"   File Monitoring Available: {'Yes' if results['file_monitoring_available'] else 'No'}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"claude_code_integration_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_claude_code_integration())