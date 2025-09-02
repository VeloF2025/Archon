"""
SCWT Runner - Executes Standard Coding Workflow Tests
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from ..validation_engine import ValidationEngine
from ..config import ValidatorConfig
from ..models import ValidationRequest
from .test_cases import SCWTTestSuite, SCWTTestCase, TestType
from .metrics import SCWTMetrics, SCWTMetricsAggregator

logger = logging.getLogger(__name__)


class SCWTRunner:
    """Runs SCWT benchmark tests"""
    
    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
        self.engine = ValidationEngine(self.config)
        self.test_suite = SCWTTestSuite()
        self.aggregator = SCWTMetricsAggregator()
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize the runner and validation engine"""
        await self.engine.initialize()
        logger.info("SCWT Runner initialized")
    
    async def run_test(self, test_case: SCWTTestCase) -> SCWTMetrics:
        """Run a single SCWT test case"""
        
        logger.info(f"Running test: {test_case.id} - {test_case.name}")
        
        # Create metrics object
        metrics = SCWTMetrics(
            test_id=test_case.id,
            test_name=test_case.name,
            phase=self._get_phase_for_test(test_case)
        )
        
        # Set baseline values for comparison
        metrics.baseline_time_ms = 5000  # 5 second baseline
        metrics.baseline_tokens = 2000  # Baseline token usage
        metrics.baseline_iterations = 3  # Baseline iterations
        metrics.baseline_hallucination_rate = 0.3  # 30% baseline
        
        try:
            # Create validation request
            request = ValidationRequest(
                prompt=test_case.prompt,
                output=test_case.output,
                context=test_case.context,
                validation_type=test_case.validation_type,
                request_id=test_case.id
            )
            
            # Run validation
            start_time = time.time()
            response = await self.engine.validate(request)
            end_time = time.time()
            
            # Calculate metrics from response
            metrics.calculate_from_response(response)
            
            # Override with actual time if not set
            if metrics.validation_time_ms == 0:
                metrics.validation_time_ms = int((end_time - start_time) * 1000)
            
            # Calculate improvements against baseline
            metrics.calculate_improvements()
            
            # Check success criteria
            success = self._check_success_criteria(
                test_case,
                response,
                metrics
            )
            
            # Log results
            logger.info(
                f"Test {test_case.id} completed - "
                f"Status: {response.status}, "
                f"Issues: {len(response.issues)}, "
                f"Success: {success}"
            )
            
        except Exception as e:
            logger.error(f"Test {test_case.id} failed: {e}", exc_info=True)
            metrics.verdict_accuracy = 0  # Mark as failed
        
        return metrics
    
    async def run_phase(self, phase: int) -> List[SCWTMetrics]:
        """Run all tests for a specific phase"""
        
        logger.info(f"Running SCWT Phase {phase}")
        
        phase_tests = self.test_suite.get_phase_tests(phase)
        results = []
        
        for test in phase_tests:
            metrics = await self.run_test(test)
            results.append(metrics)
            self.aggregator.add_metrics(metrics)
            
            # Save individual result
            self._save_metrics(metrics)
        
        # Generate phase report
        phase_summary = self.aggregator.get_phase_summary(phase)
        self._save_phase_summary(phase, phase_summary)
        
        logger.info(
            f"Phase {phase} completed - "
            f"Tests: {len(results)}, "
            f"Pass rate: {phase_summary.get('pass_rate', 0):.1%}"
        )
        
        return results
    
    async def run_all_phases(self) -> Dict[str, Any]:
        """Run all SCWT phases"""
        
        logger.info("Starting complete SCWT benchmark")
        
        all_results = {}
        
        for phase in [1, 2, 3]:
            # Check if phase gate passes before continuing
            if phase > 1:
                gate_passed = self._check_phase_gate(phase - 1)
                if not gate_passed:
                    logger.warning(f"Phase {phase-1} gate failed, stopping benchmark")
                    break
            
            results = await self.run_phase(phase)
            all_results[f"phase_{phase}"] = results
        
        # Generate overall summary
        overall_summary = self.aggregator.get_overall_summary()
        self._save_overall_summary(overall_summary)
        
        logger.info(
            f"SCWT benchmark completed - "
            f"Overall success: {overall_summary['overall_success_rate']:.1%}"
        )
        
        return overall_summary
    
    async def run_specific_test(self, test_id: str) -> Optional[SCWTMetrics]:
        """Run a specific test by ID"""
        
        test_case = self.test_suite.get_test_case(test_id)
        if not test_case:
            logger.error(f"Test {test_id} not found")
            return None
        
        metrics = await self.run_test(test_case)
        self._save_metrics(metrics)
        
        return metrics
    
    async def run_test_type(self, test_type: TestType) -> List[SCWTMetrics]:
        """Run all tests of a specific type"""
        
        logger.info(f"Running all {test_type.value} tests")
        
        tests = self.test_suite.get_tests_by_type(test_type)
        results = []
        
        for test in tests:
            metrics = await self.run_test(test)
            results.append(metrics)
            self.aggregator.add_metrics(metrics)
            self._save_metrics(metrics)
        
        return results
    
    def _get_phase_for_test(self, test_case: SCWTTestCase) -> int:
        """Determine which phase a test belongs to"""
        
        phase_mapping = {
            TestType.HALLUCINATION: 1,
            TestType.GAMING: 1,
            TestType.CROSS_CHECK: 2,
            TestType.KNOWLEDGE_REUSE: 2,
            TestType.EFFICIENCY: 3,
            TestType.PRECISION: 3
        }
        
        return phase_mapping.get(test_case.test_type, 1)
    
    def _check_success_criteria(
        self,
        test_case: SCWTTestCase,
        response,
        metrics: SCWTMetrics
    ) -> bool:
        """Check if test meets success criteria"""
        
        criteria = test_case.success_criteria
        
        # Check expected status
        if response.status.value != test_case.expected_status:
            return False
        
        # Check specific criteria
        for key, expected_value in criteria.items():
            if key == "must_detect":
                # Check if specific patterns were detected
                detected = [issue.message for issue in response.issues]
                for pattern in expected_value:
                    if not any(pattern in msg for msg in detected):
                        return False
            
            elif key == "issue_count":
                if len(response.issues) < expected_value:
                    return False
            
            elif key == "gaming_patterns_detected":
                gaming_count = sum(
                    1 for issue in response.issues
                    if issue.category == "gaming"
                )
                if gaming_count < len(expected_value):
                    return False
            
            elif key == "knowledge_reuse_rate":
                if metrics.knowledge_reuse_rate < expected_value:
                    return False
            
            elif key == "precision":
                if metrics.precision < expected_value:
                    return False
        
        return True
    
    def _check_phase_gate(self, phase: int) -> bool:
        """Check if phase gate criteria are met"""
        
        phase_summary = self.aggregator.get_phase_summary(phase)
        
        if "error" in phase_summary:
            return False
        
        # Phase gate criteria from PRD
        gate_criteria = {
            1: 0.1,  # Phase 1: ≥10% improvement
            2: 0.4,  # Phase 2: ≥40% error reduction
            3: 0.7   # Phase 3: ≥70% targets met
        }
        
        required_pass_rate = gate_criteria.get(phase, 0.5)
        actual_pass_rate = phase_summary.get("pass_rate", 0)
        
        return actual_pass_rate >= required_pass_rate
    
    def _save_metrics(self, metrics: SCWTMetrics):
        """Save metrics to file"""
        
        filename = self.results_dir / f"{metrics.test_id}_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(metrics.get_summary(), f, indent=2)
    
    def _save_phase_summary(self, phase: int, summary: Dict[str, Any]):
        """Save phase summary to file"""
        
        filename = self.results_dir / f"phase_{phase}_summary.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_overall_summary(self, summary: Dict[str, Any]):
        """Save overall summary to file"""
        
        filename = self.results_dir / "scwt_overall_summary.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as markdown report
        self._generate_markdown_report(summary)
    
    def _generate_markdown_report(self, summary: Dict[str, Any]):
        """Generate markdown report from summary"""
        
        report = """# SCWT Benchmark Report

## Overall Results

"""
        
        report += f"- **Total Tests Run**: {summary['total_tests_run']}\n"
        report += f"- **Overall Success Rate**: {summary['overall_success_rate']:.1%}\n"
        report += f"- **Recommendation**: {summary['recommendation']}\n\n"
        
        # Add phase summaries
        for phase_key, phase_data in summary.get('phases', {}).items():
            phase_num = phase_key.split('_')[1]
            report += f"\n## Phase {phase_num} Results\n\n"
            report += f"- **Tests Run**: {phase_data['total_tests']}\n"
            report += f"- **Tests Passing**: {phase_data['tests_passing']}\n"
            report += f"- **Pass Rate**: {phase_data['pass_rate']:.1%}\n\n"
            
            report += "### Average Metrics\n\n"
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"
            
            for metric, value in phase_data['averages'].items():
                report += f"| {metric.replace('_', ' ').title()} | {value:.2%} |\n"
        
        # Save report
        filename = self.results_dir / "scwt_benchmark_report.md"
        with open(filename, 'w') as f:
            f.write(report)


async def main():
    """Main entry point for running SCWT benchmark"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and initialize runner
    runner = SCWTRunner()
    await runner.initialize()
    
    # Run all phases
    summary = await runner.run_all_phases()
    
    # Print summary
    print("\n" + "="*50)
    print("SCWT BENCHMARK COMPLETE")
    print("="*50)
    print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"Recommendation: {summary['recommendation']}")
    print(f"\nDetailed results saved to: {runner.results_dir}")


if __name__ == "__main__":
    asyncio.run(main())