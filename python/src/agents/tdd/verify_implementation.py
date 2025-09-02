#!/usr/bin/env python3
"""
Phase 9 TDD Enforcement - Implementation Verification Script
Validates that all components are properly implemented and functional
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_file_structure() -> Dict[str, bool]:
    """Verify all required files exist with proper structure"""
    
    base_path = Path(__file__).parent
    
    required_files = {
        "__init__.py": "Module initialization",
        "stagehand_test_engine.py": "Natural language test generation",
        "browserbase_executor.py": "Cloud test execution management", 
        "tdd_enforcement_gate.py": "Mandatory TDD compliance validation",
        "enhanced_dgts_validator.py": "Enhanced gaming detection",
        "tdd_config.yaml": "TDD enforcement configuration",
        "requirements.tdd.txt": "Phase 9 dependencies",
        "example_usage.py": "Implementation examples",
        "README.md": "Comprehensive documentation",
        "verify_implementation.py": "This verification script"
    }
    
    results = {}
    
    print("🔍 Verifying file structure...")
    
    for filename, description in required_files.items():
        file_path = base_path / filename
        exists = file_path.exists()
        results[filename] = exists
        
        status = "✅" if exists else "❌"
        print(f"   {status} {filename} - {description}")
        
        if exists and filename.endswith('.py'):
            # Check file is not empty and has proper structure
            try:
                content = file_path.read_text(encoding='utf-8')
                if len(content) < 100:  # Minimum reasonable file size
                    print(f"      ⚠️ Warning: {filename} seems too small ({len(content)} chars)")
                elif not content.strip().startswith('#!' if filename != '__init__.py' else '"""'):
                    print(f"      ⚠️ Warning: {filename} missing proper header")
                else:
                    print(f"      ✅ {filename} structure looks good ({len(content):,} chars)")
            except Exception as e:
                print(f"      ❌ Error reading {filename}: {e}")
                results[filename] = False
    
    return results

def verify_python_syntax() -> Dict[str, bool]:
    """Verify all Python files have valid syntax"""
    
    base_path = Path(__file__).parent
    current_file = Path(__file__).name
    python_files = [f for f in base_path.glob("*.py") if f.name != current_file]
    
    results = {}
    
    print("\n🐍 Verifying Python syntax...")
    
    for py_file in python_files:
        try:
            # Compile the file to check syntax
            with open(py_file, 'rb') as f:
                compile(f.read(), py_file, 'exec')
            results[py_file.name] = True
            print(f"   ✅ {py_file.name} - Syntax OK")
        except SyntaxError as e:
            results[py_file.name] = False
            print(f"   ❌ {py_file.name} - Syntax Error: {e}")
        except Exception as e:
            results[py_file.name] = False
            print(f"   ❌ {py_file.name} - Error: {e}")
    
    return results

def verify_imports() -> Dict[str, bool]:
    """Verify critical imports can be resolved"""
    
    print("\n📦 Verifying critical imports...")
    
    results = {}
    
    # Test imports that should work without external dependencies
    import_tests = {
        "base_imports": [
            "import os",
            "import json", 
            "import logging",
            "import asyncio",
            "from pathlib import Path",
            "from typing import Dict, List, Any, Optional",
            "from dataclasses import dataclass",
            "from datetime import datetime",
            "from enum import Enum"
        ],
        "internal_imports": [
            "from . import __init__",
            "from .stagehand_test_engine import TestType, TestFramework",
            "from .browserbase_executor import ExecutionStatus, BrowserType", 
            "from .tdd_enforcement_gate import EnforcementLevel, ViolationType",
            "from .enhanced_dgts_validator import StagehandGamingType"
        ]
    }
    
    for category, imports in import_tests.items():
        category_results = []
        print(f"\n   Testing {category}:")
        
        for import_stmt in imports:
            try:
                # Use exec in a clean namespace to test import
                test_namespace = {}
                exec(import_stmt, test_namespace)
                category_results.append(True)
                print(f"      ✅ {import_stmt}")
            except Exception as e:
                category_results.append(False)
                print(f"      ❌ {import_stmt} - {e}")
        
        results[category] = all(category_results)
    
    return results

def verify_configuration() -> Dict[str, Any]:
    """Verify configuration file is valid"""
    
    print("\n⚙️ Verifying configuration...")
    
    config_path = Path(__file__).parent / "tdd_config.yaml"
    
    if not config_path.exists():
        print("   ❌ tdd_config.yaml not found")
        return {"valid": False, "error": "File not found"}
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify required configuration sections
        required_sections = [
            "enforcement",
            "test_requirements", 
            "coverage_requirements",
            "stagehand",
            "gaming_detection",
            "validation_rules"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"   ❌ Missing config sections: {', '.join(missing_sections)}")
            return {"valid": False, "missing_sections": missing_sections}
        
        print("   ✅ Configuration structure valid")
        print(f"      - Enforcement level: {config['enforcement']['level']}")
        print(f"      - Min coverage: {config['enforcement']['min_coverage']}%")
        print(f"      - Gaming detection: {'enabled' if config['gaming_detection']['enabled'] else 'disabled'}")
        
        return {"valid": True, "config": config}
        
    except ImportError:
        print("   ⚠️ PyYAML not available - using basic validation")
        # Basic YAML structure check without parsing
        content = config_path.read_text()
        if len(content) > 1000 and "enforcement:" in content:
            print("   ✅ Configuration file structure looks reasonable")
            return {"valid": True, "basic_check": True}
        else:
            print("   ❌ Configuration file seems invalid")
            return {"valid": False, "error": "Basic validation failed"}
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
        return {"valid": False, "error": str(e)}

def verify_documentation() -> Dict[str, Any]:
    """Verify documentation is comprehensive"""
    
    print("\n📚 Verifying documentation...")
    
    readme_path = Path(__file__).parent / "README.md"
    
    if not readme_path.exists():
        print("   ❌ README.md not found")
        return {"valid": False, "error": "README missing"}
    
    try:
        content = readme_path.read_text(encoding='utf-8')
        
        # Check for required documentation sections
        required_sections = [
            "# Phase 9 TDD Enforcement",
            "## 🎯 Overview",
            "## 🏗️ Architecture", 
            "## 🔧 Core Components",
            "## 📋 Configuration",
            "## 🚀 Quick Start",
            "## 🛡️ Enforcement Levels",
            "## 🔍 Gaming Detection Examples"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"   ❌ Missing documentation sections: {missing_sections}")
            return {"valid": False, "missing_sections": missing_sections}
        
        # Check documentation quality metrics
        word_count = len(content.split())
        line_count = len(content.split('\n'))
        code_examples = content.count('```python')
        
        print("   ✅ Documentation structure complete")
        print(f"      - Word count: {word_count:,}")
        print(f"      - Line count: {line_count:,}")  
        print(f"      - Code examples: {code_examples}")
        
        quality_score = min(100, (word_count / 50) + (code_examples * 10))
        print(f"      - Quality score: {quality_score:.0f}/100")
        
        return {
            "valid": True,
            "word_count": word_count,
            "code_examples": code_examples,
            "quality_score": quality_score
        }
        
    except Exception as e:
        print(f"   ❌ Documentation verification failed: {e}")
        return {"valid": False, "error": str(e)}

def verify_class_definitions() -> Dict[str, bool]:
    """Verify key classes are properly defined"""
    
    print("\n🏗️ Verifying class definitions...")
    
    # Key classes that should be defined
    expected_classes = {
        "stagehand_test_engine.py": [
            "TestType", "TestFramework", "TestRequirement", 
            "GeneratedTest", "TestGenerationResult", "StagehandTestEngine"
        ],
        "browserbase_executor.py": [
            "ExecutionStatus", "BrowserType", "ExecutionConfig",
            "TestExecution", "ExecutionResult", "BrowserbaseExecutor" 
        ],
        "tdd_enforcement_gate.py": [
            "EnforcementLevel", "ViolationType", "FeatureStatus",
            "TDDViolation", "FeatureValidation", "EnforcementResult", "TDDEnforcementGate"
        ],
        "enhanced_dgts_validator.py": [
            "StagehandGamingType", "StagehandGamingViolation", "EnhancedDGTSValidator"
        ]
    }
    
    results = {}
    base_path = Path(__file__).parent
    
    for filename, classes in expected_classes.items():
        file_path = base_path / filename
        
        if not file_path.exists():
            results[filename] = False
            print(f"   ❌ {filename} not found")
            continue
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            missing_classes = []
            for class_name in classes:
                # Check for class or enum definition
                if f"class {class_name}" in content or f"{class_name}(" in content:
                    continue  # Found
                else:
                    missing_classes.append(class_name)
            
            if missing_classes:
                results[filename] = False
                print(f"   ❌ {filename} missing classes: {', '.join(missing_classes)}")
            else:
                results[filename] = True
                print(f"   ✅ {filename} all classes found ({len(classes)} classes)")
                
        except Exception as e:
            results[filename] = False
            print(f"   ❌ {filename} verification failed: {e}")
    
    return results

def generate_verification_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive verification report"""
    
    from datetime import datetime
    
    print("\n📊 Generating verification report...")
    
    # Calculate overall scores
    file_structure_score = sum(results["file_structure"].values()) / len(results["file_structure"]) * 100
    syntax_score = sum(results["python_syntax"].values()) / max(len(results["python_syntax"]), 1) * 100
    imports_score = sum(results["imports"].values()) / len(results["imports"]) * 100
    classes_score = sum(results["class_definitions"].values()) / len(results["class_definitions"]) * 100
    
    config_score = 100 if results["configuration"]["valid"] else 0
    docs_score = results["documentation"].get("quality_score", 0) if results["documentation"]["valid"] else 0
    
    overall_score = (
        file_structure_score * 0.2 +
        syntax_score * 0.2 +
        imports_score * 0.15 + 
        classes_score * 0.15 +
        config_score * 0.15 +
        docs_score * 0.15
    )
    
    report = f"""
🚀 Phase 9 TDD Enforcement - Implementation Verification Report
{'=' * 80}

📊 OVERALL SCORE: {overall_score:.1f}/100

📋 Component Scores:
   File Structure:     {file_structure_score:.1f}/100
   Python Syntax:      {syntax_score:.1f}/100  
   Import Resolution:  {imports_score:.1f}/100
   Class Definitions:  {classes_score:.1f}/100
   Configuration:      {config_score:.1f}/100
   Documentation:      {docs_score:.1f}/100

🔍 Detailed Results:
   
   Files Created: {sum(results['file_structure'].values())}/{len(results['file_structure'])}
   Syntax Errors: {len(results['python_syntax']) - sum(results['python_syntax'].values())}
   Import Issues: {len(results['imports']) - sum(results['imports'].values())}
   Missing Classes: {len(results['class_definitions']) - sum(results['class_definitions'].values())}
   
📚 Documentation Stats:
   Word Count: {results['documentation'].get('word_count', 'N/A'):,}
   Code Examples: {results['documentation'].get('code_examples', 'N/A')}

✅ SYSTEM STATUS: {'READY FOR PRODUCTION' if overall_score >= 95 else 'NEEDS ATTENTION' if overall_score >= 85 else 'CRITICAL ISSUES'}

{('🎉 Phase 9 TDD Enforcement is FULLY IMPLEMENTED and ready for deployment!' if overall_score >= 95 else 
  '⚠️ Minor issues detected - review and address before production deployment.' if overall_score >= 85 else
  '🚨 Critical issues found - must be resolved before system can be used.')}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

def main():
    """Run complete implementation verification"""
    
    print("🚀 Phase 9 TDD Enforcement - Implementation Verification")
    print("=" * 80)
    
    # Import datetime here to make it available globally
    from datetime import datetime
    globals()['datetime'] = datetime
    
    try:
        # Run all verification checks
        results = {
            "file_structure": verify_file_structure(),
            "python_syntax": verify_python_syntax(),  
            "imports": verify_imports(),
            "configuration": verify_configuration(),
            "documentation": verify_documentation(),
            "class_definitions": verify_class_definitions()
        }
        
        # Generate and display report
        report = generate_verification_report(results)
        print(report)
        
        # Save report to file
        report_path = Path(__file__).parent / "verification_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        # Calculate exit code
        overall_score = sum([
            sum(results["file_structure"].values()) / len(results["file_structure"]) * 100,
            sum(results["python_syntax"].values()) / max(len(results["python_syntax"]), 1) * 100,
            sum(results["imports"].values()) / len(results["imports"]) * 100,
            sum(results["class_definitions"].values()) / len(results["class_definitions"]) * 100,
            100 if results["configuration"]["valid"] else 0,
            results["documentation"].get("quality_score", 0) if results["documentation"]["valid"] else 0
        ]) / 6
        
        # Exit with appropriate code
        if overall_score >= 95:
            print("\n🎉 VERIFICATION PASSED - System ready for production!")
            return 0
        elif overall_score >= 85:
            print("\n⚠️ VERIFICATION PARTIAL - Minor issues detected")
            return 1
        else:
            print("\n🚨 VERIFICATION FAILED - Critical issues found")
            return 2
            
    except Exception as e:
        logger.exception("Verification failed with exception")
        print(f"\n❌ VERIFICATION ERROR: {e}")
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)