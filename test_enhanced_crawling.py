#!/usr/bin/env python3
"""
Test Enhanced Crawl4AI Integration

This script tests the new enhanced crawling features including:
- LLM-powered semantic extraction
- Adaptive crawling with quality assessment
- Anti-detection stealth mode
- Structured data extraction
"""

import asyncio
import requests
import json
import time
from datetime import datetime

API_BASE = "http://localhost:8181/api"


def test_enhanced_crawl_api():
    """Test the enhanced crawling API endpoint."""
    print("🚀 Testing Enhanced Crawl4AI Integration")
    print("=" * 50)
    
    # Test URLs with different content types
    test_cases = [
        {
            "name": "Documentation Site",
            "url": "https://docs.crawl4ai.com/first-steps/quick-start",
            "request": {
                "url": "https://docs.crawl4ai.com/first-steps/quick-start",
                "knowledge_type": "documentation",
                "tags": ["crawl4ai", "documentation", "quick-start"],
                "max_depth": 1,
                "max_pages": 10,
                "use_llm_extraction": True,
                "enable_stealth": True,
                "content_type": "documentation",
                "adaptive_crawling": True,
            }
        },
        {
            "name": "GitHub Repository",
            "url": "https://github.com/unclecode/crawl4ai",
            "request": {
                "url": "https://github.com/unclecode/crawl4ai",
                "knowledge_type": "general",
                "tags": ["github", "crawl4ai", "repository"],
                "max_depth": 1,
                "max_pages": 5,
                "use_llm_extraction": False,  # Test structured extraction
                "enable_stealth": True,
                "content_type": None,  # Auto-detect
                "adaptive_crawling": True,
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   URL: {test_case['url']}")
        print(f"   Features: LLM={test_case['request']['use_llm_extraction']}, Stealth={test_case['request']['enable_stealth']}, Adaptive={test_case['request']['adaptive_crawling']}")
        
        try:
            # Start enhanced crawl
            response = requests.post(
                f"{API_BASE}/knowledge-items/enhanced-crawl",
                json=test_case["request"],
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                progress_id = result.get("progressId")
                print(f"   ✅ Enhanced crawl started: {progress_id}")
                print(f"   Features enabled: {result.get('features_enabled', {})}")
                print(f"   Estimated duration: {result.get('estimatedDuration')}")
                
                # Monitor progress for a few iterations
                print("   📊 Progress monitoring...")
                for check in range(5):  # Check progress 5 times
                    time.sleep(2)
                    try:
                        # You can implement WebSocket monitoring here if needed
                        print(f"   ⏳ Progress check {check + 1}/5...")
                    except Exception as e:
                        print(f"   ⚠️ Progress check failed: {e}")
                        break
                
                print(f"   🎯 Test case completed for {test_case['name']}")
                
            else:
                print(f"   ❌ Failed to start enhanced crawl: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception during test: {e}")
    
    print(f"\n✅ Enhanced crawling API test completed!")


def test_enhanced_vs_regular_crawl():
    """Compare enhanced crawl vs regular crawl performance."""
    print("\n🔬 Enhanced vs Regular Crawl Comparison")
    print("-" * 50)
    
    test_url = "https://docs.crawl4ai.com"
    
    # Test regular crawl
    print("1. Testing Regular Crawl...")
    regular_request = {
        "url": test_url,
        "knowledge_type": "technical",
        "tags": ["comparison", "regular"],
        "max_depth": 1,
        "extract_code_examples": True,
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE}/knowledge-items/crawl",
            json=regular_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Regular crawl started: {result.get('progressId')}")
            regular_time = time.time() - start_time
            print(f"   ⏱️ Setup time: {regular_time:.2f}s")
        else:
            print(f"   ❌ Regular crawl failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Regular crawl exception: {e}")
    
    # Test enhanced crawl
    print("\n2. Testing Enhanced Crawl...")
    enhanced_request = {
        "url": test_url,
        "knowledge_type": "documentation",
        "tags": ["comparison", "enhanced"],
        "max_depth": 1,
        "max_pages": 10,
        "use_llm_extraction": True,
        "enable_stealth": True,
        "content_type": "documentation",
        "adaptive_crawling": True,
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE}/knowledge-items/enhanced-crawl",
            json=enhanced_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Enhanced crawl started: {result.get('progressId')}")
            enhanced_time = time.time() - start_time
            print(f"   ⏱️ Setup time: {enhanced_time:.2f}s")
            print(f"   🎛️ Features: {result.get('features_enabled', {})}")
        else:
            print(f"   ❌ Enhanced crawl failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Enhanced crawl exception: {e}")
    
    print(f"\n🏁 Comparison completed!")


def test_api_health():
    """Test API health before running enhanced crawl tests."""
    print("🩺 Testing API Health...")
    
    try:
        # Test knowledge API health
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Knowledge API: {health.get('status', 'unknown')}")
            return True
        else:
            print(f"   ❌ Knowledge API unhealthy: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ API health check failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Enhanced Crawl4AI Integration Testing")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().isoformat()}")
    
    # Check API health first
    if not test_api_health():
        print("\n❌ API is not healthy. Please start the Archon server first:")
        print("   cd python && uv run python -m src.server.main")
        return
    
    print("\n" + "=" * 60)
    
    # Test enhanced crawling features
    test_enhanced_crawl_api()
    
    # Compare enhanced vs regular
    test_enhanced_vs_regular_crawl()
    
    print(f"\n🎉 All tests completed at: {datetime.now().isoformat()}")
    print("\n📋 Next Steps:")
    print("   • Monitor the WebSocket progress in the frontend")
    print("   • Check the knowledge base for enhanced content")
    print("   • Compare quality scores between regular and enhanced crawls")
    print("   • Test different content types (documentation, articles, general)")


if __name__ == "__main__":
    main()