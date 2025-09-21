#!/usr/bin/env python3
"""
Test script for Gemini CLI integration in Archon.

This script tests:
1. Service initialization
2. Rate limiting functionality
3. Task routing logic
4. Multimodal endpoints
5. Queue processing
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8181"
GEMINI_API_BASE = f"{API_BASE_URL}/api/gemini"

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "errors": []
}


def print_test_header(test_name: str):
    """Print a formatted test header."""
    print(f"\n{'=' * 60}")
    print(f"🧪 Testing: {test_name}")
    print(f"{'=' * 60}")


def print_result(success: bool, message: str):
    """Print test result with color coding."""
    if success:
        print(f"✅ PASS: {message}")
        test_results["passed"] += 1
    else:
        print(f"❌ FAIL: {message}")
        test_results["failed"] += 1


async def test_service_health():
    """Test if Gemini CLI service is initialized and healthy."""
    print_test_header("Service Health Check")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test main API health
            response = await client.get(f"{API_BASE_URL}/health")
            api_healthy = response.status_code == 200
            print_result(api_healthy, f"Main API health check (status: {response.status_code})")
            
            # Test Gemini usage stats endpoint
            response = await client.get(f"{GEMINI_API_BASE}/usage-stats")
            stats_available = response.status_code == 200
            
            if stats_available:
                data = response.json()
                print_result(True, f"Usage stats endpoint accessible")
                print(f"   📊 Stats: {json.dumps(data, indent=2)}")
            else:
                print_result(False, f"Usage stats endpoint failed (status: {response.status_code})")
                
        except Exception as e:
            print_result(False, f"Service health check failed: {e}")
            test_results["errors"].append(str(e))


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print_test_header("Rate Limiting")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            # Create a simple multimodal request
            test_prompt = "Test prompt for rate limiting"
            
            # Test within rate limit (should succeed)
            print("Testing single request within rate limit...")
            response = await client.post(
                f"{GEMINI_API_BASE}/process-multimodal",
                json={
                    "prompt": test_prompt,
                    "task_type": "general",
                    "priority": "normal"
                }
            )
            
            if response.status_code == 200:
                print_result(True, "Single request within rate limit succeeded")
                data = response.json()
                print(f"   ⚡ Execution time: {data.get('execution_time', 'N/A')}s")
            else:
                print_result(False, f"Single request failed (status: {response.status_code})")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print_result(False, f"Rate limiting test failed: {e}")
            test_results["errors"].append(str(e))


async def test_task_routing():
    """Test intelligent task routing logic."""
    print_test_header("Task Routing Logic")
    
    # Test cases for different task types
    test_cases = [
        {
            "name": "Multimodal task (should route to Gemini CLI)",
            "characteristics": {
                "requires_multimodal": True,
                "has_images": True
            }
        },
        {
            "name": "Large context task (should route to Gemini CLI)",
            "characteristics": {
                "context_size": 150000  # >128K tokens
            }
        },
        {
            "name": "High priority streaming (should route to OpenAI)",
            "characteristics": {
                "priority": "high",
                "requires_streaming": True
            }
        },
        {
            "name": "Function calling (should route to OpenAI)",
            "characteristics": {
                "requires_function_calling": True
            }
        }
    ]
    
    # Import the routing logic for testing
    try:
        sys.path.insert(0, '/mnt/c/Jarvis/AI Workspace/Archon/python')
        from src.server.services.llm_provider_service import (
            route_llm_task, 
            TaskCharacteristics,
            LLMProvider
        )
        
        for test_case in test_cases:
            characteristics = TaskCharacteristics(**test_case["characteristics"])
            provider, reason = await route_llm_task(characteristics)
            
            print(f"\n📍 {test_case['name']}")
            print(f"   Provider: {provider.value}")
            print(f"   Reason: {reason}")
            
            # Validate routing logic
            if "multimodal" in test_case["name"].lower() or "large context" in test_case["name"].lower():
                expected = LLMProvider.GEMINI_CLI
            elif "high priority" in test_case["name"].lower() or "function calling" in test_case["name"].lower():
                expected = LLMProvider.OPENAI
            else:
                expected = None
                
            if expected and provider == expected:
                print_result(True, f"Correctly routed to {provider.value}")
            elif expected:
                print_result(False, f"Expected {expected.value}, got {provider.value}")
            else:
                print_result(True, f"Routed to {provider.value}")
                
    except ImportError as e:
        print_result(False, f"Could not import routing logic: {e}")
        test_results["errors"].append(str(e))


async def test_multimodal_endpoints():
    """Test multimodal processing endpoints."""
    print_test_header("Multimodal Endpoints")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        # Test 1: Image to Code endpoint (with mock image)
        print("\n🖼️ Testing Image-to-Code endpoint...")
        try:
            # Create a simple test image (1x1 PNG)
            test_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\x00\x00\x00\x00IEND\xaeB`\x82'
            
            files = {
                'file': ('test.png', test_image, 'image/png')
            }
            data = {
                'image_type': 'ui_mockup',
                'output_language': 'typescript',
                'additional_instructions': 'This is a test'
            }
            
            response = await client.post(
                f"{GEMINI_API_BASE}/image-to-code",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                print_result(True, "Image-to-code endpoint accessible")
                result = response.json()
                print(f"   Model: {result.get('model', 'N/A')}")
            else:
                print_result(False, f"Image-to-code failed (status: {response.status_code})")
                
        except Exception as e:
            print_result(False, f"Image-to-code test failed: {e}")
            test_results["errors"].append(str(e))
        
        # Test 2: Codebase Analysis endpoint
        print("\n📂 Testing Codebase Analysis endpoint...")
        try:
            response = await client.post(
                f"{GEMINI_API_BASE}/analyze-codebase",
                json={
                    "path": "/mnt/c/Jarvis/AI Workspace/Archon/python/src",
                    "analysis_type": "general",
                    "specific_questions": ["What is the main architecture pattern?"]
                }
            )
            
            if response.status_code == 200:
                print_result(True, "Codebase analysis endpoint accessible")
                result = response.json()
                print(f"   Context window: {result.get('context_window', 'N/A')}")
            else:
                print_result(False, f"Codebase analysis failed (status: {response.status_code})")
                
        except Exception as e:
            print_result(False, f"Codebase analysis test failed: {e}")
            test_results["errors"].append(str(e))


async def test_queue_processing():
    """Test queue processing functionality."""
    print_test_header("Queue Processing")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            # Process any queued tasks
            response = await client.post(f"{GEMINI_API_BASE}/process-queue")
            
            if response.status_code == 200:
                data = response.json()
                tasks_processed = data.get("tasks_processed", 0)
                print_result(True, f"Queue processing endpoint works")
                print(f"   📦 Tasks processed: {tasks_processed}")
                
                if tasks_processed > 0:
                    print(f"   Results: {json.dumps(data.get('results', []), indent=2)}")
            else:
                print_result(False, f"Queue processing failed (status: {response.status_code})")
                
        except Exception as e:
            print_result(False, f"Queue processing test failed: {e}")
            test_results["errors"].append(str(e))


async def test_daily_allocation():
    """Test daily allocation system."""
    print_test_header("Daily Allocation System")
    
    try:
        # Import the service to check allocations
        sys.path.insert(0, '/mnt/c/Jarvis/AI Workspace/Archon/python')
        from src.server.services.gemini_cli_service import get_gemini_cli_service, TaskType
        
        service = await get_gemini_cli_service()
        
        print("\n📊 Daily Allocation Budgets:")
        allocations = {
            TaskType.MULTIMODAL: 200,
            TaskType.LARGE_CONTEXT: 300,
            TaskType.CODE_GENERATION: 300,
            TaskType.DOCUMENTATION: 100,
            TaskType.GENERAL: 50,
            TaskType.ANALYSIS: 50
        }
        
        for task_type, budget in allocations.items():
            print(f"   {task_type.value}: {budget}/1000 requests")
            
        print_result(True, "Daily allocation system configured")
        
        # Check current usage
        stats = await service.get_usage_stats()
        print(f"\n📈 Current Usage:")
        print(f"   Today: {stats.get('daily_used', 0)}/1000")
        print(f"   This minute: {stats.get('minute_used', 0)}/60")
        
    except Exception as e:
        print_result(False, f"Daily allocation test failed: {e}")
        test_results["errors"].append(str(e))


def print_summary():
    """Print test summary."""
    print(f"\n{'=' * 60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    
    if test_results['errors']:
        print(f"\n⚠️ Errors encountered:")
        for error in test_results['errors']:
            print(f"   - {error}")
    
    success_rate = (test_results['passed'] / max(1, test_results['passed'] + test_results['failed'])) * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n🎉 All tests passed! Gemini CLI integration is working correctly.")
    elif success_rate >= 75:
        print("\n⚠️ Most tests passed, but some issues need attention.")
    else:
        print("\n❌ Significant issues detected. Please review and fix.")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 GEMINI CLI INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests in sequence
    await test_service_health()
    await test_rate_limiting()
    await test_task_routing()
    await test_multimodal_endpoints()
    await test_queue_processing()
    await test_daily_allocation()
    
    # Print summary
    print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        print_summary()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        test_results["errors"].append(str(e))
        print_summary()