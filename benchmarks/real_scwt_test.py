#!/usr/bin/env python3
"""
Real SCWT (Standard Coding Workflow Test) Implementation
Tests actual Archon+ functionality and reports honest findings
"""

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib.util
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealSCWTTest:
    """Real SCWT test that measures actual Archon+ functionality"""
    
    def __init__(self, test_repo_path: str = "scwt-test-repo", results_path: str = "scwt-results"):
        self.test_repo_path = Path(test_repo_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Test metrics tracking
        self.test_start_time = 0
        self.baseline_results = {}
        self.phase_results = {}
        
    async def run_real_phase1_test(self) -> Dict[str, Any]:
        """Run actual Phase 1 test with real component testing"""
        logger.info("=== RUNNING REAL PHASE 1 SCWT TEST ===")
        
        test_start_time = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "phase": 1,
            "task": "Test actual Archon+ Phase 1 components",
            "test_duration_seconds": 0,
            "component_tests": {},
            "metrics": {
                "hallucination_rate": 0.0,
                "knowledge_reuse": 0.0, 
                "task_efficiency_time": 0.0,
                "communication_efficiency": 0.0,
                "precision": 0.0,
                "ui_usability": 0.0
            },
            "errors": [],
            "warnings": [],
            "overall_status": "UNKNOWN"
        }
        
        try:
            # Test 1: Agent Configuration System
            logger.info("Testing agent configuration system...")
            config_test = await self._test_agent_configurations()
            results["component_tests"]["agent_configurations"] = config_test
            
            # Test 2: PRP Template System
            logger.info("Testing PRP template system...")
            prp_test = await self._test_prp_system()
            results["component_tests"]["prp_templates"] = prp_test
            
            # Test 3: Trigger System
            logger.info("Testing proactive trigger system...")
            trigger_test = await self._test_trigger_system()
            results["component_tests"]["trigger_system"] = trigger_test
            
            # Test 4: Orchestration System
            logger.info("Testing orchestration system...")
            orchestration_test = await self._test_orchestration()
            results["component_tests"]["orchestration"] = orchestration_test
            
            # Test 5: UI Components
            logger.info("Testing UI components...")
            ui_test = await self._test_ui_components()
            results["component_tests"]["ui_components"] = ui_test
            
            # Calculate real metrics based on test results
            results["metrics"] = self._calculate_real_metrics(results["component_tests"])
            
            # Determine overall status
            results["overall_status"] = self._determine_gate_status(results)
            
        except Exception as e:
            logger.error(f"Critical error during Phase 1 test: {e}")
            results["errors"].append(f"Critical test failure: {str(e)}")
            results["overall_status"] = "FAILED"
        
        finally:
            results["test_duration_seconds"] = time.time() - test_start_time
        
        # Save results
        results_file = self.results_path / "real_phase1_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Real Phase 1 test completed in {results['test_duration_seconds']:.2f}s")
        return results
    
    async def _test_agent_configurations(self) -> Dict[str, Any]:
        """Test agent configuration system"""
        test_result = {
            "test_name": "Agent Configuration System",
            "status": "UNKNOWN",
            "details": {},
            "errors": []
        }
        
        try:
            config_path = Path("python/src/agents/configs")
            
            # Test 1: Check if config directory exists
            if not config_path.exists():
                test_result["errors"].append("Agent config directory does not exist")
                test_result["status"] = "FAILED"
                return test_result
            
            # Test 2: Count and validate config files
            config_files = list(config_path.glob("*.json"))
            non_registry_files = [f for f in config_files if f.name not in ["agent_registry.json", "template_registry.json"]]
            
            test_result["details"]["config_files_found"] = len(non_registry_files)
            test_result["details"]["config_files"] = [f.name for f in non_registry_files]
            
            # Test 3: Validate JSON structure of configs
            valid_configs = 0
            invalid_configs = []
            
            for config_file in non_registry_files:
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Check required fields
                    required_fields = ["role", "name", "description", "skills", "proactive_triggers"]
                    missing_fields = [field for field in required_fields if field not in config]
                    
                    if missing_fields:
                        invalid_configs.append(f"{config_file.name}: missing {missing_fields}")
                    else:
                        valid_configs += 1
                        
                except Exception as e:
                    invalid_configs.append(f"{config_file.name}: JSON parse error - {str(e)}")
            
            test_result["details"]["valid_configs"] = valid_configs
            test_result["details"]["invalid_configs"] = invalid_configs
            
            # Test 4: Check agent registry
            registry_file = config_path / "agent_registry.json"
            if registry_file.exists():
                try:
                    with open(registry_file, 'r') as f:
                        registry = json.load(f)
                    test_result["details"]["registry_loaded"] = True
                    test_result["details"]["registry_agent_count"] = registry.get("total_agents", 0)
                except Exception as e:
                    test_result["errors"].append(f"Registry parse error: {str(e)}")
            else:
                test_result["errors"].append("Agent registry file not found")
            
            # Determine status
            if len(invalid_configs) == 0 and valid_configs >= 20:
                test_result["status"] = "PASSED"
            elif valid_configs >= 15:
                test_result["status"] = "PARTIAL"
                test_result["errors"].append(f"Only {valid_configs}/22+ agents configured properly")
            else:
                test_result["status"] = "FAILED"
                test_result["errors"].append(f"Too few valid configs: {valid_configs}")
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"Test execution error: {str(e)}")
        
        return test_result
    
    async def _test_prp_system(self) -> Dict[str, Any]:
        """Test PRP template system"""
        test_result = {
            "test_name": "PRP Template System",
            "status": "UNKNOWN", 
            "details": {},
            "errors": []
        }
        
        try:
            prp_path = Path("python/src/agents/prompts/prp")
            
            # Test 1: Check PRP directory
            if not prp_path.exists():
                test_result["errors"].append("PRP templates directory does not exist")
                test_result["status"] = "FAILED"
                return test_result
            
            # Test 2: Count PRP template files
            prp_files = list(prp_path.glob("*.md"))
            test_result["details"]["prp_templates_found"] = len(prp_files)
            test_result["details"]["prp_files"] = [f.name for f in prp_files]
            
            # Test 3: Validate template content
            valid_templates = 0
            template_quality = []
            
            for prp_file in prp_files:
                try:
                    with open(prp_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for required sections
                    required_sections = ["## Context", "## Task Requirements", "## Examples", "## Output Format"]
                    missing_sections = [section for section in required_sections if section not in content]
                    
                    if not missing_sections:
                        valid_templates += 1
                        
                        # Quality metrics
                        has_variables = "{" in content and "}" in content
                        has_examples = "```" in content
                        template_length = len(content)
                        
                        template_quality.append({
                            "file": prp_file.name,
                            "has_variables": has_variables,
                            "has_code_examples": has_examples,
                            "content_length": template_length,
                            "quality_score": (int(has_variables) + int(has_examples) + min(template_length/1000, 10)) / 12
                        })
                    else:
                        test_result["errors"].append(f"{prp_file.name}: missing sections {missing_sections}")
                        
                except Exception as e:
                    test_result["errors"].append(f"{prp_file.name}: read error - {str(e)}")
            
            test_result["details"]["valid_templates"] = valid_templates
            test_result["details"]["template_quality"] = template_quality
            
            # Test 4: Test PRP Manager
            try:
                # Try to import and initialize PRP manager
                prp_manager_path = Path("python/src/agents/prompts/prp_manager.py")
                if prp_manager_path.exists():
                    spec = importlib.util.spec_from_file_location("prp_manager", prp_manager_path)
                    prp_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(prp_module)
                    
                    # Try to initialize PRP manager
                    prp_manager = prp_module.PRPManager()
                    test_result["details"]["prp_manager_loaded"] = True
                    test_result["details"]["loaded_templates_count"] = len(prp_manager.templates)
                else:
                    test_result["errors"].append("PRP manager file not found")
                    
            except Exception as e:
                test_result["errors"].append(f"PRP manager load error: {str(e)}")
            
            # Determine status
            if valid_templates >= 5 and len(test_result["errors"]) == 0:
                test_result["status"] = "PASSED"
            elif valid_templates >= 3:
                test_result["status"] = "PARTIAL"
            else:
                test_result["status"] = "FAILED"
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"Test execution error: {str(e)}")
        
        return test_result
    
    async def _test_trigger_system(self) -> Dict[str, Any]:
        """Test proactive trigger system"""
        test_result = {
            "test_name": "Proactive Trigger System",
            "status": "UNKNOWN",
            "details": {},
            "errors": []
        }
        
        try:
            trigger_path = Path("python/src/agents/triggers")
            
            # Test 1: Check trigger directory
            if not trigger_path.exists():
                test_result["errors"].append("Trigger system directory does not exist")
                test_result["status"] = "FAILED"
                return test_result
            
            # Test 2: Check for trigger files
            file_watcher_path = trigger_path / "file_watcher.py"
            trigger_engine_path = trigger_path / "trigger_engine.py"
            
            files_exist = {
                "file_watcher.py": file_watcher_path.exists(),
                "trigger_engine.py": trigger_engine_path.exists()
            }
            
            test_result["details"]["trigger_files"] = files_exist
            
            # Test 3: Try to import trigger modules
            importable_modules = {}
            
            if file_watcher_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("file_watcher", file_watcher_path)
                    watcher_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(watcher_module)
                    importable_modules["file_watcher"] = True
                    
                    # Test creating ProactiveTriggerManager
                    trigger_manager = watcher_module.ProactiveTriggerManager(
                        watch_paths=["."],
                        agent_callback=None
                    )
                    test_result["details"]["trigger_manager_created"] = True
                    test_result["details"]["trigger_rules_loaded"] = len(trigger_manager.trigger_rules)
                    
                except Exception as e:
                    importable_modules["file_watcher"] = False
                    test_result["errors"].append(f"File watcher import error: {str(e)}")
            
            if trigger_engine_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("trigger_engine", trigger_engine_path)
                    engine_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(engine_module)
                    importable_modules["trigger_engine"] = True
                    
                except Exception as e:
                    importable_modules["trigger_engine"] = False
                    test_result["errors"].append(f"Trigger engine import error: {str(e)}")
            
            test_result["details"]["importable_modules"] = importable_modules
            
            # Test 4: Pattern matching test
            if importable_modules.get("file_watcher", False):
                try:
                    test_files = ["app.py", "requirements.txt", "package.json", "LoginForm.tsx"]
                    pattern_matches = {}
                    
                    for test_file in test_files:
                        matches = trigger_manager.test_pattern_matching(test_file)
                        pattern_matches[test_file] = matches
                    
                    test_result["details"]["pattern_matching_test"] = pattern_matches
                    test_result["details"]["pattern_matching_working"] = any(matches for matches in pattern_matches.values())
                    
                except Exception as e:
                    test_result["errors"].append(f"Pattern matching test error: {str(e)}")
            
            # Determine status
            working_modules = sum(importable_modules.values())
            if working_modules == 2 and len(test_result["errors"]) <= 1:
                test_result["status"] = "PASSED"
            elif working_modules >= 1:
                test_result["status"] = "PARTIAL"
            else:
                test_result["status"] = "FAILED"
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"Test execution error: {str(e)}")
        
        return test_result
    
    async def _test_orchestration(self) -> Dict[str, Any]:
        """Test orchestration system"""
        test_result = {
            "test_name": "Orchestration System",
            "status": "UNKNOWN",
            "details": {},
            "errors": []
        }
        
        try:
            orchestration_path = Path("python/src/agents/orchestration")
            
            # Test 1: Check orchestration directory
            if not orchestration_path.exists():
                test_result["errors"].append("Orchestration directory does not exist")
                test_result["status"] = "FAILED"
                return test_result
            
            # Test 2: Check for orchestration files
            required_files = [
                "parallel_executor.py",
                "agent_pool.py", 
                "orchestrator.py"
            ]
            
            files_exist = {}
            for file_name in required_files:
                file_path = orchestration_path / file_name
                files_exist[file_name] = file_path.exists()
            
            test_result["details"]["orchestration_files"] = files_exist
            
            # Test 3: Try to import orchestration modules
            importable_modules = {}
            
            for file_name in required_files:
                if files_exist[file_name]:
                    try:
                        file_path = orchestration_path / file_name
                        module_name = file_name.replace(".py", "")
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        importable_modules[file_name] = True
                        
                    except Exception as e:
                        importable_modules[file_name] = False
                        test_result["errors"].append(f"{file_name} import error: {str(e)}")
                else:
                    importable_modules[file_name] = False
            
            test_result["details"]["importable_modules"] = importable_modules
            
            # Test 4: Test basic functionality
            if importable_modules.get("agent_pool.py", False):
                try:
                    # This would require major refactoring to work without dependencies
                    test_result["details"]["agent_pool_functionality"] = "Not tested - requires dependency injection"
                except Exception as e:
                    test_result["errors"].append(f"Agent pool test error: {str(e)}")
            
            # Determine status
            working_modules = sum(importable_modules.values())
            total_modules = len(required_files)
            
            if working_modules == total_modules:
                test_result["status"] = "PASSED"
            elif working_modules >= total_modules // 2:
                test_result["status"] = "PARTIAL"
            else:
                test_result["status"] = "FAILED"
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"Test execution error: {str(e)}")
        
        return test_result
    
    async def _test_ui_components(self) -> Dict[str, Any]:
        """Test UI components"""
        test_result = {
            "test_name": "UI Components",
            "status": "UNKNOWN",
            "details": {},
            "errors": []
        }
        
        try:
            ui_path = Path("archon-ui-main/src/components/agents")
            
            # Test 1: Check UI directory
            if not ui_path.exists():
                test_result["errors"].append("UI components directory does not exist")
                test_result["status"] = "FAILED"
                return test_result
            
            # Test 2: Check for UI component files
            required_components = [
                "AgentDashboard.tsx",
                "AgentControlPanel.tsx"
            ]
            
            hooks_path = Path("archon-ui-main/src/hooks")
            required_hooks = [
                "useAgentSystem.ts"
            ]
            
            components_exist = {}
            for component in required_components:
                component_path = ui_path / component
                components_exist[component] = component_path.exists()
            
            hooks_exist = {}
            if hooks_path.exists():
                for hook in required_hooks:
                    hook_path = hooks_path / hook
                    hooks_exist[hook] = hook_path.exists()
            
            test_result["details"]["components_exist"] = components_exist
            test_result["details"]["hooks_exist"] = hooks_exist
            
            # Test 3: Check component content quality
            component_quality = {}
            
            for component in required_components:
                if components_exist[component]:
                    try:
                        component_path = ui_path / component
                        with open(component_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        quality_metrics = {
                            "has_typescript": content.count("interface ") > 0,
                            "has_react_hooks": "useState" in content or "useEffect" in content,
                            "has_proper_imports": "import React" in content,
                            "content_length": len(content),
                            "has_error_handling": "try" in content or "catch" in content
                        }
                        
                        component_quality[component] = quality_metrics
                        
                    except Exception as e:
                        test_result["errors"].append(f"{component} quality check error: {str(e)}")
            
            test_result["details"]["component_quality"] = component_quality
            
            # Determine status
            components_found = sum(components_exist.values())
            hooks_found = sum(hooks_exist.values())
            
            total_required = len(required_components) + len(required_hooks)
            total_found = components_found + hooks_found
            
            if total_found == total_required:
                test_result["status"] = "PASSED"
            elif total_found >= total_required // 2:
                test_result["status"] = "PARTIAL"
            else:
                test_result["status"] = "FAILED"
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["errors"].append(f"Test execution error: {str(e)}")
        
        return test_result
    
    def _calculate_real_metrics(self, component_tests: Dict[str, Any]) -> Dict[str, float]:
        """Calculate realistic metrics based on actual test results"""
        
        # Count component statuses
        total_components = len(component_tests)
        passed_components = len([test for test in component_tests.values() if test["status"] == "PASSED"])
        partial_components = len([test for test in component_tests.values() if test["status"] == "PARTIAL"])
        
        # Calculate base precision (how much actually works)
        precision = (passed_components + (partial_components * 0.5)) / total_components if total_components > 0 else 0
        
        # Calculate specific metrics based on what actually exists
        metrics = {
            "precision": round(precision, 2),
            "hallucination_rate": round(1.0 - precision, 2),  # Inverse of precision
            "knowledge_reuse": 0.0,  # Not measurable yet - no real execution
            "task_efficiency_time": 0.0,  # Not measurable yet - no real task execution
            "communication_efficiency": 0.0,  # Not measurable yet - no agent communication
            "ui_usability": 0.0  # UI exists but not functionally tested
        }
        
        # Adjust based on specific component results
        if "agent_configurations" in component_tests:
            config_test = component_tests["agent_configurations"]
            if config_test["status"] == "PASSED":
                metrics["knowledge_reuse"] = 0.1  # Some structure for reuse exists
        
        if "prp_templates" in component_tests:
            prp_test = component_tests["prp_templates"]
            if prp_test["status"] == "PASSED":
                metrics["task_efficiency_time"] = 0.05  # Templates could improve efficiency
        
        if "ui_components" in component_tests:
            ui_test = component_tests["ui_components"]
            if ui_test["status"] == "PASSED":
                metrics["ui_usability"] = 0.02  # UI exists but not validated for usability
        
        return metrics
    
    def _determine_gate_status(self, results: Dict[str, Any]) -> str:
        """Determine if Phase 1 gate criteria are met - HONEST ASSESSMENT"""
        
        component_tests = results["component_tests"]
        metrics = results["metrics"]
        
        # Phase 1 gate requirements:
        # - Task efficiency ≥15% - NOT MET (0% - no real execution)
        # - Communication efficiency ≥10% - NOT MET (0% - no real communication)  
        # - Precision ≥85% - CHECK ACTUAL PRECISION
        # - UI usability ≥5% - NOT MET (0% - no real usability testing)
        
        gate_failures = []
        
        if metrics["task_efficiency_time"] < 0.15:
            gate_failures.append(f"Task efficiency {metrics['task_efficiency_time']*100:.0f}% < 15% required")
        
        if metrics["communication_efficiency"] < 0.10:
            gate_failures.append(f"Communication efficiency {metrics['communication_efficiency']*100:.0f}% < 10% required")
        
        if metrics["precision"] < 0.85:
            gate_failures.append(f"Precision {metrics['precision']*100:.0f}% < 85% required")
        
        if metrics["ui_usability"] < 0.05:
            gate_failures.append(f"UI usability {metrics['ui_usability']*100:.0f}% < 5% required")
        
        # Also check if critical components are working
        critical_failures = []
        for component_name, test_result in component_tests.items():
            if test_result["status"] == "FAILED":
                critical_failures.append(f"{component_name} failed")
        
        # HONEST GATE DECISION
        if gate_failures or critical_failures:
            decision = "FAILED"
            results["gate_failures"] = gate_failures + critical_failures
        else:
            decision = "PASSED"
        
        return decision

async def main():
    """Run real SCWT test"""
    print("=== REAL ARCHON+ PHASE 1 SCWT TEST ===")
    print("Testing actual functionality with honest reporting")
    print()
    
    tester = RealSCWTTest()
    
    # Run Phase 1 test
    results = await tester.run_real_phase1_test()
    
    # Print results
    print("=== TEST RESULTS ===")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Test Duration: {results['test_duration_seconds']:.2f} seconds")
    print()
    
    print("Component Test Results:")
    for component, test_result in results["component_tests"].items():
        status_emoji = "✅" if test_result["status"] == "PASSED" else "⚠️" if test_result["status"] == "PARTIAL" else "❌"
        print(f"  {status_emoji} {component}: {test_result['status']}")
        
        if test_result["errors"]:
            for error in test_result["errors"]:
                print(f"    - {error}")
    
    print()
    print("Metrics (Honest Assessment):")
    for metric, value in results["metrics"].items():
        print(f"  {metric}: {value*100:.1f}%" if value != 0 else f"  {metric}: Not measurable - no real execution")
    
    if "gate_failures" in results:
        print()
        print("Gate Failures:")
        for failure in results["gate_failures"]:
            print(f"  ❌ {failure}")
    
    print()
    print("=== CONCLUSION ===")
    if results["overall_status"] == "FAILED":
        print("❌ PHASE 1 GATE: FAILED")
        print("Reason: Framework components exist but lack real functionality")
        print("Recommendation: Implement actual agent execution before proceeding to Phase 2")
    else:
        print("✅ PHASE 1 GATE: PASSED")
    
    print()
    print(f"Detailed results saved to: scwt-results/real_phase1_results.json")

if __name__ == "__main__":
    asyncio.run(main())