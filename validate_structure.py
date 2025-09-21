#!/usr/bin/env python3
"""
Structural validation for Archon Phase 1 implementation
Validates file structure, API endpoints, and configuration without running code
"""

import os
import sys
import re
from datetime import datetime

def validate_file_structure():
    """Validate that all required files exist"""
    print("📁 Validating file structure...")
    
    required_files = [
        # Pattern Recognition
        "python/src/agents/pattern_recognition/__init__.py",
        "python/src/agents/pattern_recognition/pattern_detector.py",
        "python/src/agents/pattern_recognition/pattern_storage.py", 
        "python/src/agents/pattern_recognition/pattern_analyzer.py",
        "python/src/agents/pattern_recognition/pattern_recommender.py",
        
        # Knowledge Graph
        "python/src/agents/knowledge_graph/__init__.py",
        "python/src/agents/knowledge_graph/graph_client.py",
        "python/src/agents/knowledge_graph/knowledge_ingestion.py",
        "python/src/agents/knowledge_graph/query_engine.py",
        "python/src/agents/knowledge_graph/relationship_mapper.py",
        "python/src/agents/knowledge_graph/graph_analyzer.py",
        
        # Predictive Assistant
        "python/src/agents/predictive_assistant/__init__.py",
        "python/src/agents/predictive_assistant/context_analyzer.py",
        "python/src/agents/predictive_assistant/suggestion_engine.py",
        "python/src/agents/predictive_assistant/completion_provider.py",
        "python/src/agents/predictive_assistant/predictor.py",
        
        # APIs
        "python/src/server/api_routes/pattern_recognition_api.py",
        "python/src/server/api_routes/knowledge_graph_api.py",
        "python/src/server/services/knowledge_graph_service.py",
        
        # Database schemas
        "python/pattern_recognition_schema.sql",
        "docker-compose.neo4j.yml",
        
        # Tests
        "python/tests/test_pattern_recognition.py",
        "python/tests/test_knowledge_graph.py",
        "python/tests/test_predictive_assistant.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print("\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def validate_api_endpoints():
    """Validate API endpoint definitions"""
    print("\n🌐 Validating API endpoints...")
    
    # Check Pattern Recognition API
    pr_api_file = "python/src/server/api_routes/pattern_recognition_api.py"
    if os.path.exists(pr_api_file):
        with open(pr_api_file, 'r') as f:
            content = f.read()
            
        pr_endpoints = [
            '"/detect"',
            '"/search"', 
            '"/recommend"',
            '"/refactor"',
            '"/insights"'
        ]
        
        missing_pr = []
        for endpoint in pr_endpoints:
            if endpoint not in content:
                missing_pr.append(endpoint)
        
        if missing_pr:
            print(f"❌ Pattern Recognition API missing endpoints: {missing_pr}")
            return False
        else:
            print("✅ Pattern Recognition API endpoints complete")
    
    # Check Knowledge Graph API
    kg_api_file = "python/src/server/api_routes/knowledge_graph_api.py"
    if os.path.exists(kg_api_file):
        with open(kg_api_file, 'r') as f:
            content = f.read()
            
        kg_endpoints = [
            '"/nodes"',
            '"/relationships"',
            '"/query"',
            '"/search"',
            '"/statistics"'
        ]
        
        missing_kg = []
        for endpoint in kg_endpoints:
            if endpoint not in content:
                missing_kg.append(endpoint)
        
        if missing_kg:
            print(f"❌ Knowledge Graph API missing endpoints: {missing_kg}")
            return False
        else:
            print("✅ Knowledge Graph API endpoints complete")
    
    return True

def validate_class_definitions():
    """Validate that key classes are properly defined"""
    print("\n🏗️ Validating class definitions...")
    
    class_checks = [
        ("python/src/agents/pattern_recognition/pattern_detector.py", ["PatternDetector", "CodePattern"]),
        ("python/src/agents/knowledge_graph/graph_client.py", ["Neo4jClient", "GraphNode", "GraphRelationship"]),
        ("python/src/agents/predictive_assistant/context_analyzer.py", ["ContextAnalyzer", "CodeContext"]),
        ("python/src/agents/predictive_assistant/suggestion_engine.py", ["SuggestionEngine", "CodeSuggestion"])
    ]
    
    for file_path, classes in class_checks:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            missing_classes = []
            for class_name in classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)
            
            if missing_classes:
                print(f"❌ {file_path} missing classes: {missing_classes}")
                return False
            else:
                print(f"✅ {os.path.basename(file_path)} classes defined")
        else:
            print(f"❌ File not found: {file_path}")
            return False
    
    return True

def validate_async_methods():
    """Validate that key async methods are defined"""
    print("\n⚡ Validating async methods...")
    
    async_checks = [
        ("python/src/agents/pattern_recognition/pattern_detector.py", ["async def detect_patterns"]),
        ("python/src/agents/knowledge_graph/graph_client.py", ["async def connect", "async def create_node"]),
        ("python/src/agents/predictive_assistant/context_analyzer.py", ["async def analyze_context"]),
        ("python/src/agents/predictive_assistant/suggestion_engine.py", ["async def generate_suggestions"])
    ]
    
    for file_path, methods in async_checks:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            missing_methods = []
            for method in methods:
                if method not in content:
                    missing_methods.append(method)
            
            if missing_methods:
                print(f"❌ {file_path} missing methods: {missing_methods}")
                return False
            else:
                print(f"✅ {os.path.basename(file_path)} async methods defined")
    
    return True

def validate_docker_setup():
    """Validate Docker configuration"""
    print("\n🐳 Validating Docker setup...")
    
    docker_file = "docker-compose.neo4j.yml"
    if os.path.exists(docker_file):
        with open(docker_file, 'r') as f:
            content = f.read()
        
        required_services = ["neo4j", "kafka", "redis"]
        missing_services = []
        
        for service in required_services:
            if service not in content:
                missing_services.append(service)
        
        if missing_services:
            print(f"❌ Docker compose missing services: {missing_services}")
            return False
        else:
            print("✅ Docker services configured")
    
    return True

def count_lines_of_code():
    """Count lines of code in Phase 1 implementation"""
    print("\n📊 Code metrics...")
    
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk("python/src/agents"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    for root, dirs, files in os.walk("python/src/server/api_routes"):
        for file in files:
            if file.endswith(".py") and ("pattern_recognition" in file or "knowledge_graph" in file):
                python_files.append(os.path.join(root, file))
    
    total_lines = 0
    total_files = 0
    
    for file_path in python_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                total_files += 1
    
    print(f"✅ Phase 1 Implementation:")
    print(f"   - Files: {total_files}")
    print(f"   - Lines of code: {total_lines}")
    
    return True

def validate_test_coverage():
    """Validate test file structure"""
    print("\n🧪 Validating test coverage...")
    
    test_files = [
        "python/tests/test_pattern_recognition.py",
        "python/tests/test_knowledge_graph.py", 
        "python/tests/test_predictive_assistant.py"
    ]
    
    total_test_methods = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Count test methods
            test_methods = len(re.findall(r'def test_\w+', content))
            total_test_methods += test_methods
            
            print(f"✅ {os.path.basename(test_file)}: {test_methods} test methods")
        else:
            print(f"❌ Missing test file: {test_file}")
            return False
    
    print(f"✅ Total test methods: {total_test_methods}")
    return True

def main():
    """Run all structural validations"""
    print("🏗️ Archon Phase 1 Structural Validation")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    validations = [
        validate_file_structure,
        validate_api_endpoints,
        validate_class_definitions,
        validate_async_methods,
        validate_docker_setup,
        count_lines_of_code,
        validate_test_coverage
    ]
    
    results = []
    for validation in validations:
        try:
            result = validation()
            results.append(result)
        except Exception as e:
            print(f"❌ Validation {validation.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📋 STRUCTURAL VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print("✅ ALL STRUCTURAL VALIDATIONS PASSED!")
        print("🎯 Phase 1 implementation is structurally complete")
        print("🚀 Ready for dependency installation and runtime testing")
    else:
        print(f"⚠️  {passed}/{total} validations passed")
        print("🔧 Some structural issues need attention")
    
    print(f"\nValidation completed at: {datetime.now().isoformat()}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)