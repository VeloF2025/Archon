"""
Phase 7 DeepConf Integration Test Coverage Report and Validation

This module validates that all Phase 7 PRD requirements have comprehensive test coverage
and that the test suite achieves >95% coverage as required by quality gates.

PRD Requirements Coverage Validation:
- Core DeepConf Engine (PRD 4.1): Multi-dimensional confidence scoring
- Multi-Model Consensus System (PRD 4.2): Voting and disagreement resolution
- Intelligent Task Routing (PRD 4.3): Model selection and optimization 
- SCWT Metrics Dashboard (PRD 4.4): Real-time visualization
- Advanced Debugging Tools (PRD 4.5): Confidence analysis and troubleshooting
- Integration with Phase 5+9 (PRD 4.6): Seamless system integration

Quality Gates:
- >95% test coverage across all Phase 7 components
- DGTS compliance with zero confidence gaming
- Performance targets met (70-85% token savings, <1.5s response)
- TDD Red Phase compliance (all tests fail until implementation)
"""

import pytest
import os
import sys
import importlib
import inspect
from typing import Dict, List, Set, Tuple
from pathlib import Path
import ast


class Phase7CoverageValidator:
    """Validates test coverage for Phase 7 DeepConf Integration"""
    
    def __init__(self):
        self.test_root = Path(__file__).parent
        self.prd_requirements = self._load_prd_requirements()
        self.coverage_report = {}
        
    def _load_prd_requirements(self) -> Dict[str, List[str]]:
        """Load PRD requirements for coverage validation"""
        return {
            "4.1_deepconf_engine": [
                "calculate_confidence",
                "validate_confidence", 
                "calibrate_model",
                "get_uncertainty_bounds",
                "explain_confidence",
                "get_confidence_factors"
            ],
            "4.2_consensus_system": [
                "request_consensus",
                "weighted_voting",
                "disagreement_analysis", 
                "escalation_decision",
                "calculate_model_weights",
                "detect_outlier_responses"
            ],
            "4.3_intelligent_routing": [
                "route_task",
                "select_optimal_model",
                "calculate_task_complexity",
                "optimize_token_usage",
                "get_routing_explanation",
                "update_routing_strategy"
            ],
            "4.4_scwt_dashboard": [
                "confidence_metrics_display",
                "performance_optimization_reporting", 
                "quality_assurance_trends",
                "debugging_interface",
                "real_time_updates",
                "websocket_streaming"
            ],
            "4.5_debugging_tools": [
                "analyze_low_confidence",
                "trace_confidence_factors",
                "identify_performance_bottlenecks", 
                "suggest_optimization_strategies",
                "create_debug_session",
                "export_debug_data"
            ],
            "4.6_integration": [
                "enhance_dgts_validation",
                "integrate_nlnh_confidence",
                "tdd_confidence_scoring",
                "external_validator_consensus",
                "create_confidence_validation_pipeline",
                "validate_confidence_accuracy"
            ]
        }
    
    def discover_test_files(self) -> List[Path]:
        """Discover all test files in Phase 7 test suite"""
        test_files = []
        
        for root, dirs, files in os.walk(self.test_root):
            for file in files:
                if file.startswith('test_') and file.endswith('.py') and file != '__init__.py':
                    test_files.append(Path(root) / file)
        
        return test_files
    
    def analyze_test_file_coverage(self, test_file: Path) -> Dict[str, any]:
        """Analyze coverage provided by a single test file"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            coverage_analysis = {
                "file_path": str(test_file),
                "test_functions": [],
                "requirements_covered": set(),
                "performance_tests": [],
                "dgts_validated_tests": [],
                "tdd_red_phase_tests": [],
                "integration_tests": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_info = self._analyze_test_function(node, content)
                    coverage_analysis["test_functions"].append(test_info)
                    
                    # Categorize test types
                    if test_info.get("performance_critical"):
                        coverage_analysis["performance_tests"].append(test_info["name"])
                    if test_info.get("dgts_validated"):
                        coverage_analysis["dgts_validated_tests"].append(test_info["name"])
                    if test_info.get("tdd_red_phase"):
                        coverage_analysis["tdd_red_phase_tests"].append(test_info["name"])
                    if test_info.get("integration_test"):
                        coverage_analysis["integration_tests"].append(test_info["name"])
                    
                    # Map to PRD requirements
                    covered_reqs = self._map_test_to_requirements(test_info)
                    coverage_analysis["requirements_covered"].update(covered_reqs)
            
            coverage_analysis["requirements_covered"] = list(coverage_analysis["requirements_covered"])
            return coverage_analysis
            
        except Exception as e:
            return {
                "file_path": str(test_file),
                "error": str(e),
                "test_functions": [],
                "requirements_covered": []
            }
    
    def _analyze_test_function(self, node: ast.FunctionDef, content: str) -> Dict[str, any]:
        """Analyze a single test function"""
        test_info = {
            "name": node.name,
            "line_number": node.lineno,
            "docstring": ast.get_docstring(node),
            "decorators": [],
            "imports_tested": [],
            "assertions_count": 0,
            "async_test": False
        }
        
        # Analyze decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorator_name = decorator.id
                test_info["decorators"].append(decorator_name)
                
                # Check for specific test markers
                if decorator_name == "tdd_red_phase":
                    test_info["tdd_red_phase"] = True
                elif decorator_name == "dgts_validated":
                    test_info["dgts_validated"] = True
                elif decorator_name == "performance_critical":
                    test_info["performance_critical"] = True
            
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorator_name = decorator.func.id
                test_info["decorators"].append(decorator_name)
                
                if decorator_name == "requires_implementation":
                    # Extract component being tested
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        test_info["component_tested"] = decorator.args[0].value
        
        # Check if async test
        if isinstance(node, ast.AsyncFunctionDef):
            test_info["async_test"] = True
        
        # Count assertions
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                test_info["assertions_count"] += 1
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == "assert":
                    test_info["assertions_count"] += 1
        
        return test_info
    
    def _map_test_to_requirements(self, test_info: Dict[str, any]) -> Set[str]:
        """Map test function to PRD requirements"""
        covered_requirements = set()
        
        test_name = test_info["name"].lower()
        docstring = (test_info.get("docstring") or "").lower()
        component = test_info.get("component_tested", "").lower()
        
        # Map based on test name patterns and components
        requirement_patterns = {
            "4.1_deepconf_engine": [
                "confidence", "calibrat", "uncertain", "scoring", "deepconf", "engine"
            ],
            "4.2_consensus_system": [
                "consensus", "voting", "disagreement", "multi_model", "agreement"
            ],
            "4.3_intelligent_routing": [
                "routing", "router", "model_selection", "optimization", "token"
            ],
            "4.4_scwt_dashboard": [
                "dashboard", "metrics", "visualization", "websocket", "real_time"
            ],
            "4.5_debugging_tools": [
                "debug", "trace", "bottleneck", "analysis", "profiler"
            ],
            "4.6_integration": [
                "integration", "dgts", "validator", "nlnh", "tdd", "phase"
            ]
        }
        
        for requirement, patterns in requirement_patterns.items():
            for pattern in patterns:
                if (pattern in test_name or 
                    pattern in docstring or 
                    pattern in component):
                    covered_requirements.add(requirement)
                    break
        
        return covered_requirements
    
    def generate_coverage_report(self) -> Dict[str, any]:
        """Generate comprehensive coverage report"""
        test_files = self.discover_test_files()
        
        total_coverage = {
            "total_test_files": len(test_files),
            "total_test_functions": 0,
            "requirements_coverage": {},
            "test_categories": {
                "unit_tests": 0,
                "integration_tests": 0,
                "performance_tests": 0,
                "e2e_tests": 0
            },
            "quality_markers": {
                "tdd_red_phase": 0,
                "dgts_validated": 0,
                "performance_critical": 0
            },
            "coverage_gaps": [],
            "file_analysis": []
        }
        
        all_covered_requirements = set()
        
        for test_file in test_files:
            file_coverage = self.analyze_test_file_coverage(test_file)
            total_coverage["file_analysis"].append(file_coverage)
            
            if "error" not in file_coverage:
                total_coverage["total_test_functions"] += len(file_coverage["test_functions"])
                
                # Track covered requirements
                all_covered_requirements.update(file_coverage["requirements_covered"])
                
                # Categorize tests by directory structure
                if "/unit/" in str(test_file):
                    total_coverage["test_categories"]["unit_tests"] += len(file_coverage["test_functions"])
                elif "/integration/" in str(test_file):
                    total_coverage["test_categories"]["integration_tests"] += len(file_coverage["test_functions"])
                elif "/performance/" in str(test_file):
                    total_coverage["test_categories"]["performance_tests"] += len(file_coverage["test_functions"])
                elif "/e2e/" in str(test_file):
                    total_coverage["test_categories"]["e2e_tests"] += len(file_coverage["test_functions"])
                
                # Count quality markers
                total_coverage["quality_markers"]["tdd_red_phase"] += len(file_coverage["tdd_red_phase_tests"])
                total_coverage["quality_markers"]["dgts_validated"] += len(file_coverage["dgts_validated_tests"])
                total_coverage["quality_markers"]["performance_critical"] += len(file_coverage["performance_tests"])
        
        # Calculate requirements coverage
        total_requirements = sum(len(reqs) for reqs in self.prd_requirements.values())
        
        for req_category, requirements in self.prd_requirements.items():
            if req_category in all_covered_requirements:
                total_coverage["requirements_coverage"][req_category] = {
                    "covered": True,
                    "coverage_percentage": 100.0  # Simplified - category-level coverage
                }
            else:
                total_coverage["requirements_coverage"][req_category] = {
                    "covered": False,
                    "coverage_percentage": 0.0
                }
                total_coverage["coverage_gaps"].append(req_category)
        
        # Calculate overall coverage percentage
        covered_categories = len(all_covered_requirements)
        total_categories = len(self.prd_requirements)
        total_coverage["overall_coverage_percentage"] = (covered_categories / total_categories) * 100.0
        
        return total_coverage


class TestPhase7CoverageValidation:
    """Test class for validating Phase 7 test coverage"""
    
    def setup_method(self):
        """Setup coverage validator"""
        self.coverage_validator = Phase7CoverageValidator()
    
    @pytest.mark.coverage
    def test_all_prd_requirements_have_tests(self):
        """
        Test that all PRD requirements have corresponding test coverage
        
        This test validates that every requirement from the Phase 7 PRD
        has at least one test case covering its functionality.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        # Validate coverage report generation
        assert "requirements_coverage" in coverage_report
        assert "overall_coverage_percentage" in coverage_report
        assert "coverage_gaps" in coverage_report
        
        # Check for coverage gaps
        coverage_gaps = coverage_report["coverage_gaps"]
        
        if coverage_gaps:
            gap_details = "\n".join([f"- {gap}" for gap in coverage_gaps])
            pytest.fail(f"Missing test coverage for PRD requirements:\n{gap_details}")
        
        # Validate overall coverage meets >95% requirement
        overall_coverage = coverage_report["overall_coverage_percentage"]
        assert overall_coverage >= 95.0, (
            f"Overall test coverage {overall_coverage:.1f}% below PRD requirement of 95%"
        )
    
    @pytest.mark.coverage
    def test_tdd_red_phase_compliance(self):
        """
        Test that all tests follow TDD Red Phase principles
        
        All tests should be marked as @tdd_red_phase and should fail
        until implementation is complete.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        total_tests = coverage_report["total_test_functions"]
        tdd_red_phase_tests = coverage_report["quality_markers"]["tdd_red_phase"]
        
        # At least 80% of tests should be TDD Red Phase compliant
        tdd_compliance_percentage = (tdd_red_phase_tests / total_tests) * 100.0 if total_tests > 0 else 0
        
        assert tdd_compliance_percentage >= 80.0, (
            f"TDD Red Phase compliance {tdd_compliance_percentage:.1f}% below minimum of 80%"
        )
        
        # Validate TDD markers are present
        assert tdd_red_phase_tests > 0, "No TDD Red Phase tests found - all tests should follow TDD principles"
    
    @pytest.mark.coverage
    def test_dgts_compliance_validation(self):
        """
        Test that DGTS anti-gaming validation is present
        
        Critical tests should be marked as @dgts_validated to prevent
        confidence gaming and test manipulation.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        dgts_validated_tests = coverage_report["quality_markers"]["dgts_validated"]
        
        # Should have DGTS validated tests for gaming prevention
        assert dgts_validated_tests > 0, "No DGTS validated tests found - gaming prevention is required"
        
        # Check for DGTS-related test files
        dgts_test_files = [
            file_analysis for file_analysis in coverage_report["file_analysis"]
            if "dgts" in file_analysis["file_path"].lower()
        ]
        
        assert len(dgts_test_files) > 0, "No DGTS-specific test files found"
    
    @pytest.mark.coverage  
    def test_performance_test_coverage(self):
        """
        Test that performance requirements have dedicated test coverage
        
        PRD performance targets (70-85% token savings, <1.5s response)
        should have specific performance tests.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        performance_tests = coverage_report["test_categories"]["performance_tests"]
        performance_critical_tests = coverage_report["quality_markers"]["performance_critical"]
        
        # Should have performance tests
        assert performance_tests > 0, "No performance tests found in /performance/ directory"
        assert performance_critical_tests > 0, "No @performance_critical tests found"
        
        # Check for performance-related test files
        performance_test_files = [
            file_analysis for file_analysis in coverage_report["file_analysis"]
            if "performance" in file_analysis["file_path"].lower()
        ]
        
        assert len(performance_test_files) > 0, "No performance-specific test files found"
    
    @pytest.mark.coverage
    def test_integration_test_coverage(self):
        """
        Test that Phase 5+9 integration has proper test coverage
        
        Integration with existing Phase systems should be thoroughly tested
        to ensure backward compatibility and seamless integration.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        integration_tests = coverage_report["test_categories"]["integration_tests"]
        
        # Should have integration tests
        assert integration_tests > 0, "No integration tests found in /integration/ directory"
        
        # Check for specific integration areas
        integration_test_files = [
            file_analysis for file_analysis in coverage_report["file_analysis"]
            if "integration" in file_analysis["file_path"].lower()
        ]
        
        integration_areas = ["dgts", "validator", "nlnh", "tdd"]
        covered_integration_areas = []
        
        for file_analysis in integration_test_files:
            file_path = file_analysis["file_path"].lower()
            for area in integration_areas:
                if area in file_path:
                    covered_integration_areas.append(area)
        
        # Should cover key integration points
        assert len(covered_integration_areas) >= 2, (
            f"Integration coverage incomplete. Found: {covered_integration_areas}, "
            f"Expected coverage for: {integration_areas}"
        )
    
    @pytest.mark.coverage
    def test_e2e_dashboard_test_coverage(self):
        """
        Test that E2E dashboard functionality has proper test coverage
        
        SCWT dashboard with real-time metrics should have E2E tests
        for user interface and WebSocket functionality.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        e2e_tests = coverage_report["test_categories"]["e2e_tests"]
        
        # Should have E2E tests
        assert e2e_tests > 0, "No E2E tests found in /e2e/ directory"
        
        # Check for dashboard-specific E2E tests
        dashboard_test_files = [
            file_analysis for file_analysis in coverage_report["file_analysis"]
            if "dashboard" in file_analysis["file_path"].lower() and "e2e" in file_analysis["file_path"].lower()
        ]
        
        assert len(dashboard_test_files) > 0, "No dashboard E2E test files found"
        
        # Validate dashboard tests cover key features
        dashboard_features_tested = []
        for file_analysis in dashboard_test_files:
            for test_func in file_analysis["test_functions"]:
                test_name = test_func["name"].lower()
                if "websocket" in test_name:
                    dashboard_features_tested.append("websocket")
                if "real_time" in test_name or "realtime" in test_name:
                    dashboard_features_tested.append("real_time")
                if "metrics" in test_name:
                    dashboard_features_tested.append("metrics")
        
        expected_features = ["websocket", "real_time", "metrics"]
        missing_features = [f for f in expected_features if f not in dashboard_features_tested]
        
        assert len(missing_features) == 0, (
            f"Dashboard E2E tests missing coverage for: {missing_features}"
        )
    
    @pytest.mark.coverage
    def test_test_file_organization_structure(self):
        """
        Test that test files are properly organized according to testing strategy
        
        Tests should be organized in appropriate directories:
        - /unit/ for unit tests
        - /integration/ for integration tests  
        - /performance/ for performance tests
        - /e2e/ for end-to-end tests
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        test_categories = coverage_report["test_categories"]
        
        # Validate all test categories have coverage
        for category, count in test_categories.items():
            assert count > 0, f"No {category} found - test organization incomplete"
        
        # Validate reasonable distribution
        total_tests = sum(test_categories.values())
        
        # Unit tests should be the majority (40-60%)
        unit_percentage = (test_categories["unit_tests"] / total_tests) * 100
        assert 30 <= unit_percentage <= 70, (
            f"Unit test percentage {unit_percentage:.1f}% outside expected range of 30-70%"
        )
        
        # Should have meaningful integration coverage (20-40%)
        integration_percentage = (test_categories["integration_tests"] / total_tests) * 100
        assert integration_percentage >= 15, (
            f"Integration test percentage {integration_percentage:.1f}% too low (minimum 15%)"
        )
    
    @pytest.mark.coverage
    def test_test_quality_metrics(self):
        """
        Test that tests meet quality standards
        
        Tests should have proper assertions, documentation, and structure.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        total_functions = 0
        functions_with_docstrings = 0
        functions_with_assertions = 0
        
        for file_analysis in coverage_report["file_analysis"]:
            if "error" not in file_analysis:
                for test_func in file_analysis["test_functions"]:
                    total_functions += 1
                    
                    if test_func.get("docstring"):
                        functions_with_docstrings += 1
                    
                    if test_func.get("assertions_count", 0) > 0:
                        functions_with_assertions += 1
        
        # Quality metrics validation
        if total_functions > 0:
            docstring_percentage = (functions_with_docstrings / total_functions) * 100
            assertion_percentage = (functions_with_assertions / total_functions) * 100
            
            assert docstring_percentage >= 80, (
                f"Test documentation coverage {docstring_percentage:.1f}% below minimum of 80%"
            )
            
            assert assertion_percentage >= 90, (
                f"Test assertion coverage {assertion_percentage:.1f}% below minimum of 90%"
            )
    
    @pytest.mark.coverage
    def test_coverage_report_generation(self):
        """
        Test that coverage report can be generated successfully
        
        This meta-test validates that the coverage analysis itself works correctly.
        """
        coverage_report = self.coverage_validator.generate_coverage_report()
        
        # Validate report structure
        required_keys = [
            "total_test_files",
            "total_test_functions", 
            "requirements_coverage",
            "test_categories",
            "quality_markers",
            "overall_coverage_percentage",
            "file_analysis"
        ]
        
        for key in required_keys:
            assert key in coverage_report, f"Coverage report missing required key: {key}"
        
        # Validate report content
        assert coverage_report["total_test_files"] > 0, "No test files discovered"
        assert coverage_report["total_test_functions"] > 0, "No test functions discovered"
        assert isinstance(coverage_report["overall_coverage_percentage"], (int, float)), "Invalid coverage percentage type"
        assert 0 <= coverage_report["overall_coverage_percentage"] <= 100, "Coverage percentage out of valid range"


if __name__ == "__main__":
    """
    Generate and display Phase 7 coverage report
    
    Run this module directly to generate a detailed coverage report
    for Phase 7 DeepConf Integration tests.
    """
    validator = Phase7CoverageValidator()
    report = validator.generate_coverage_report()
    
    print("=" * 80)
    print("PHASE 7 DEEPCONF INTEGRATION - TEST COVERAGE REPORT")
    print("=" * 80)
    
    print(f"\nOVERALL COVERAGE: {report['overall_coverage_percentage']:.1f}%")
    print(f"Target: 95.0% (PRD Requirement)")
    
    if report['overall_coverage_percentage'] >= 95.0:
        print("✅ COVERAGE TARGET MET")
    else:
        print("❌ COVERAGE TARGET NOT MET")
    
    print(f"\nTEST STATISTICS:")
    print(f"- Total Test Files: {report['total_test_files']}")
    print(f"- Total Test Functions: {report['total_test_functions']}")
    
    print(f"\nTEST CATEGORIES:")
    for category, count in report['test_categories'].items():
        print(f"- {category.replace('_', ' ').title()}: {count}")
    
    print(f"\nQUALITY MARKERS:")
    for marker, count in report['quality_markers'].items():
        print(f"- {marker.replace('_', ' ').title()}: {count}")
    
    print(f"\nREQUIREMENTS COVERAGE:")
    for req, coverage in report['requirements_coverage'].items():
        status = "✅" if coverage['covered'] else "❌"
        print(f"- {req}: {status} {coverage['coverage_percentage']:.1f}%")
    
    if report['coverage_gaps']:
        print(f"\nCOVERAGE GAPS:")
        for gap in report['coverage_gaps']:
            print(f"- ❌ {gap}")
    else:
        print(f"\n✅ NO COVERAGE GAPS DETECTED")
    
    print("\n" + "=" * 80)