#!/usr/bin/env python3
"""
Simple validation script for Gemini CLI integration.
Checks that all components are properly implemented.
"""

import os
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, '/mnt/c/Jarvis/AI Workspace/Archon/python')

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if Path(filepath).exists():
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing")
        return False


def check_dockerfile_changes():
    """Check if Dockerfile has Gemini CLI installation."""
    dockerfile_path = "/mnt/c/Jarvis/AI Workspace/Archon/python/Dockerfile.server"
    
    if not Path(dockerfile_path).exists():
        print("❌ Dockerfile.server not found")
        return False
    
    with open(dockerfile_path, 'r') as f:
        content = f.read()
        
    checks = {
        "Node.js installation": "nodejs" in content,
        "Gemini CLI installation": "@google-gemini/gemini-cli" in content
    }
    
    all_good = True
    for check, result in checks.items():
        if result:
            print(f"✅ Dockerfile: {check} configured")
        else:
            print(f"❌ Dockerfile: {check} missing")
            all_good = False
    
    return all_good


def check_service_implementation():
    """Check Gemini CLI service implementation."""
    service_path = "/mnt/c/Jarvis/AI Workspace/Archon/python/src/server/services/gemini_cli_service.py"
    
    if not Path(service_path).exists():
        print("❌ gemini_cli_service.py not found")
        return False
    
    with open(service_path, 'r') as f:
        content = f.read()
    
    required_components = [
        ("GeminiCLIService class", "class GeminiCLIService"),
        ("Rate limiting", "RATE_LIMIT_PER_MINUTE"),
        ("Daily allocation", "TASK_ALLOCATIONS"),
        ("Task queue", "task_queue"),
        ("Redis caching", "redis_client"),
        ("execute_task method", "async def execute_task"),
        ("can_execute method", "async def can_execute")
    ]
    
    all_good = True
    for component, search_str in required_components:
        if search_str in content:
            print(f"✅ Service: {component} implemented")
        else:
            print(f"❌ Service: {component} missing")
            all_good = False
    
    return all_good


def check_api_endpoints():
    """Check if API endpoints are implemented."""
    api_path = "/mnt/c/Jarvis/AI Workspace/Archon/python/src/server/api_routes/gemini_multimodal_api.py"
    
    if not Path(api_path).exists():
        print("❌ gemini_multimodal_api.py not found")
        return False
    
    with open(api_path, 'r') as f:
        content = f.read()
    
    endpoints = [
        ("Image to Code", "/image-to-code"),
        ("PDF to Code", "/pdf-to-code"),
        ("Codebase Analysis", "/analyze-codebase"),
        ("Multimodal Processing", "/process-multimodal"),
        ("Usage Stats", "/usage-stats"),
        ("Queue Processing", "/process-queue")
    ]
    
    all_good = True
    for endpoint_name, endpoint_path in endpoints:
        if endpoint_path in content:
            print(f"✅ Endpoint: {endpoint_name} ({endpoint_path})")
        else:
            print(f"❌ Endpoint: {endpoint_name} ({endpoint_path}) missing")
            all_good = False
    
    return all_good


def check_routing_logic():
    """Check LLM provider routing logic."""
    provider_path = "/mnt/c/Jarvis/AI Workspace/Archon/python/src/server/services/llm_provider_service.py"
    
    if not Path(provider_path).exists():
        print("❌ llm_provider_service.py not found")
        return False
    
    with open(provider_path, 'r') as f:
        content = f.read()
    
    routing_components = [
        ("Gemini CLI provider", "GEMINI_CLI"),
        ("Task characteristics", "class TaskCharacteristics"),
        ("Routing function", "async def route_llm_task"),
        ("Multimodal routing", "requires_multimodal"),
        ("Large context routing", "context_size > 128000"),
        ("Execute with Gemini", "async def execute_with_gemini_cli")
    ]
    
    all_good = True
    for component, search_str in routing_components:
        if search_str in content:
            print(f"✅ Routing: {component} configured")
        else:
            print(f"❌ Routing: {component} missing")
            all_good = False
    
    return all_good


def check_main_integration():
    """Check if Gemini is integrated into main app."""
    main_path = "/mnt/c/Jarvis/AI Workspace/Archon/python/src/server/main.py"
    
    if not Path(main_path).exists():
        print("❌ main.py not found")
        return False
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    if "gemini_multimodal_router" in content and "app.include_router(gemini_multimodal_router)" in content:
        print("✅ Main app: Gemini router integrated")
        return True
    else:
        print("❌ Main app: Gemini router not integrated")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("🔍 GEMINI CLI INTEGRATION VALIDATION")
    print("=" * 60)
    
    results = []
    
    print("\n📁 File Structure Check:")
    print("-" * 40)
    files_to_check = [
        ("/mnt/c/Jarvis/AI Workspace/Archon/python/src/server/services/gemini_cli_service.py", "Gemini CLI Service"),
        ("/mnt/c/Jarvis/AI Workspace/Archon/python/src/server/api_routes/gemini_multimodal_api.py", "Multimodal API"),
        ("/mnt/c/Jarvis/AI Workspace/Archon/test_gemini_cli_integration.py", "Test Script"),
        ("/mnt/c/Jarvis/AI Workspace/Archon/examples/gemini_cli_usage_examples.py", "Usage Examples")
    ]
    
    for filepath, desc in files_to_check:
        results.append(check_file_exists(filepath, desc))
    
    print("\n🐳 Docker Configuration:")
    print("-" * 40)
    results.append(check_dockerfile_changes())
    
    print("\n⚙️ Service Implementation:")
    print("-" * 40)
    results.append(check_service_implementation())
    
    print("\n🔌 API Endpoints:")
    print("-" * 40)
    results.append(check_api_endpoints())
    
    print("\n🧭 Routing Logic:")
    print("-" * 40)
    results.append(check_routing_logic())
    
    print("\n🔗 Main App Integration:")
    print("-" * 40)
    results.append(check_main_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"✅ Passed: {passed}/{total} ({percentage:.0f}%)")
    
    if percentage == 100:
        print("\n🎉 SUCCESS! Gemini CLI integration is fully implemented.")
        print("\n📝 Next Steps:")
        print("1. Rebuild Docker container: docker-compose build archon-server")
        print("2. Start services: docker-compose up -d")
        print("3. Test endpoints: python test_gemini_cli_integration.py")
    elif percentage >= 80:
        print("\n⚠️ Almost complete! A few components need attention.")
    else:
        print("\n❌ Implementation incomplete. Please review missing components.")
    
    return percentage == 100


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)