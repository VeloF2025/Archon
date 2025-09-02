#!/usr/bin/env python3
"""
DeepConf Implementation Test Runner

Tests the core DeepConf Phase 7 implementation to ensure all components work correctly.
This validates the TDD Green phase - making the failing tests pass with real implementations.
"""

import sys
import os
import asyncio
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List

# Add the python src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python', 'src'))

try:
    from agents.deepconf.engine import DeepConfEngine, ConfidenceScore
    from agents.deepconf.consensus import MultiModelConsensus, ModelResponse
    from agents.deepconf.router import IntelligentRouter, RoutingStrategy
    from agents.deepconf.uncertainty import UncertaintyQuantifier, UncertaintyMethod
    from agents.deepconf.validation import ExternalValidatorDeepConfIntegration
    print("✅ All DeepConf modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DeepConf modules: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)

@dataclass
class MockAITask:
    """Mock AI task for testing"""
    task_id: str
    content: str
    complexity: str
    domain: str
    priority: str
    context_size: int = 1000
    expected_tokens: int = 500
    model_source: str = 'test'

@dataclass
class MockTaskContext:
    """Mock task context"""
    user_id: str
    session_id: str
    timestamp: float
    environment: str = "test"
    model_history: List[str] = None
    performance_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_history is None:
            self.model_history = []
        if self.performance_data is None:
            self.performance_data = {}

class DeepConfTestRunner:
    """Test runner for DeepConf implementation"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all DeepConf tests"""
        print("🚀 Starting DeepConf Implementation Tests")
        print("=" * 60)
        
        # Test DeepConf Engine
        await self.test_deepconf_engine()
        
        # Test Multi-Model Consensus
        await self.test_consensus_system()
        
        # Test Intelligent Router
        await self.test_intelligent_router()
        
        # Test Uncertainty Quantifier
        await self.test_uncertainty_quantifier()
        
        # Test Integration Layer
        await self.test_integration_layer()
        
        # Performance tests
        await self.test_performance_requirements()
        
        # Summary
        self.print_summary()
        
        return self.tests_failed == 0
    
    async def test_deepconf_engine(self):
        """Test DeepConf Engine functionality"""
        print("🔧 Testing DeepConf Engine...")
        
        try:
            engine = DeepConfEngine()
            
            # Create test task
            task = MockAITask(
                task_id="test-001",
                content="Create a React component for user authentication", 
                complexity="moderate",
                domain="frontend_development",
                priority="high"
            )
            
            context = MockTaskContext(
                user_id="test-user",
                session_id="test-session",
                timestamp=time.time()
            )
            
            # Test confidence calculation
            start_time = time.time()
            confidence_score = await engine.calculate_confidence(task, context)
            calculation_time = time.time() - start_time
            
            # Validate confidence score structure
            assert hasattr(confidence_score, 'overall_confidence')
            assert hasattr(confidence_score, 'factual_confidence')
            assert hasattr(confidence_score, 'reasoning_confidence')
            assert hasattr(confidence_score, 'contextual_confidence')
            assert hasattr(confidence_score, 'epistemic_uncertainty')
            assert hasattr(confidence_score, 'aleatoric_uncertainty')
            
            # Validate confidence bounds
            assert 0.0 <= confidence_score.overall_confidence <= 1.0
            assert 0.0 <= confidence_score.factual_confidence <= 1.0
            assert 0.0 <= confidence_score.reasoning_confidence <= 1.0
            assert 0.0 <= confidence_score.contextual_confidence <= 1.0
            
            # Validate uncertainty bounds
            bounds = await engine.get_uncertainty_bounds(confidence_score.overall_confidence)
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] <= confidence_score.overall_confidence <= bounds[1]
            
            # Test performance requirement (<1.5s)
            assert calculation_time < 1.5, f"Confidence calculation took {calculation_time:.3f}s, exceeds 1.5s limit"
            
            # Test calibration
            historical_data = [
                {"predicted_confidence": 0.8, "actual_success": True, "task_type": "frontend"},
                {"predicted_confidence": 0.6, "actual_success": False, "task_type": "frontend"},
                {"predicted_confidence": 0.9, "actual_success": True, "task_type": "frontend"}
            ]
            
            calibration_result = await engine.calibrate_model(historical_data)
            assert "calibration_improved" in calibration_result
            
            self.record_test_result("DeepConf Engine Core", True, calculation_time)
            
        except Exception as e:
            self.record_test_result("DeepConf Engine Core", False, 0, str(e))
    
    async def test_consensus_system(self):
        """Test Multi-Model Consensus system"""
        print("🤝 Testing Multi-Model Consensus...")
        
        try:
            consensus = MultiModelConsensus()
            
            # Create mock model responses
            task = MockAITask(
                task_id="consensus-001",
                content="Implement OAuth authentication",
                complexity="moderate", 
                domain="backend_development",
                priority="high"
            )
            
            models = ['gpt-4o', 'claude-3.5-sonnet', 'deepseek-v3']
            
            # Test consensus request
            start_time = time.time()
            consensus_result = await consensus.request_consensus(task, models)
            consensus_time = time.time() - start_time
            
            # Validate consensus result structure
            assert hasattr(consensus_result, 'agreed_response')
            assert hasattr(consensus_result, 'consensus_confidence')
            assert hasattr(consensus_result, 'agreement_level')
            assert hasattr(consensus_result, 'participating_models')
            
            # Validate consensus values
            assert 0.0 <= consensus_result.consensus_confidence <= 1.0
            assert 0.0 <= consensus_result.agreement_level <= 1.0
            assert len(consensus_result.participating_models) == len(models)
            
            # Test performance requirement (<5s for 3 models)
            assert consensus_time < 5.0, f"Consensus took {consensus_time:.3f}s, exceeds 5s limit"
            
            self.record_test_result("Multi-Model Consensus", True, consensus_time)
            
        except Exception as e:
            self.record_test_result("Multi-Model Consensus", False, 0, str(e))
    
    async def test_intelligent_router(self):
        """Test Intelligent Router"""
        print("🧠 Testing Intelligent Router...")
        
        try:
            router = IntelligentRouter()
            
            # Create test task
            task = MockAITask(
                task_id="routing-001",
                content="Optimize database queries for better performance",
                complexity="complex",
                domain="backend_development", 
                priority="high",
                expected_tokens=1500
            )
            
            # Test task routing
            start_time = time.time()
            routing_decision = await router.route_task(task, strategy=RoutingStrategy.BALANCED)
            routing_time = time.time() - start_time
            
            # Validate routing decision
            assert hasattr(routing_decision, 'selected_model')
            assert hasattr(routing_decision, 'confidence')
            assert hasattr(routing_decision, 'estimated_tokens')
            assert hasattr(routing_decision, 'estimated_cost')
            assert hasattr(routing_decision, 'estimated_time')
            
            # Validate routing values
            assert routing_decision.selected_model in ['gpt-4o', 'claude-3.5-sonnet', 'deepseek-v3', 'gpt-3.5-turbo', 'gemini-pro']
            assert 0.0 <= routing_decision.confidence <= 1.0
            assert routing_decision.estimated_tokens > 0
            assert routing_decision.estimated_cost > 0
            
            # Test performance requirement (<500ms)
            assert routing_time < 0.5, f"Routing took {routing_time:.3f}s, exceeds 500ms limit"
            
            # Test token optimization
            optimization_result = await router.optimize_token_usage(1000)
            assert hasattr(optimization_result, 'optimized_tokens')
            assert hasattr(optimization_result, 'savings_percentage')
            assert optimization_result.optimized_tokens <= 1000
            
            self.record_test_result("Intelligent Router", True, routing_time)
            
        except Exception as e:
            self.record_test_result("Intelligent Router", False, 0, str(e))
    
    async def test_uncertainty_quantifier(self):
        """Test Uncertainty Quantification"""
        print("📊 Testing Uncertainty Quantifier...")
        
        try:
            quantifier = UncertaintyQuantifier()
            
            # Test uncertainty quantification
            prediction = 0.8
            context = {
                'task_id': 'uncertainty-001',
                'confidence': 0.85,
                'model_source': 'test_model',
                'data_quality': 0.9,
                'task_complexity': 0.6
            }
            
            start_time = time.time()
            uncertainty_estimate = await quantifier.quantify_uncertainty(
                prediction, context, UncertaintyMethod.BAYESIAN
            )
            uncertainty_time = time.time() - start_time
            
            # Validate uncertainty estimate
            assert hasattr(uncertainty_estimate, 'epistemic_uncertainty')
            assert hasattr(uncertainty_estimate, 'aleatoric_uncertainty')
            assert hasattr(uncertainty_estimate, 'total_uncertainty')
            assert hasattr(uncertainty_estimate, 'confidence_intervals')
            
            # Validate uncertainty values
            assert 0.0 <= uncertainty_estimate.epistemic_uncertainty <= 1.0
            assert 0.0 <= uncertainty_estimate.aleatoric_uncertainty <= 1.0
            assert 0.0 <= uncertainty_estimate.total_uncertainty <= 1.0
            
            # Test confidence intervals
            intervals = uncertainty_estimate.confidence_intervals
            assert isinstance(intervals, dict)
            assert '95%' in intervals
            
            # Test performance requirement (<1s)
            assert uncertainty_time < 1.0, f"Uncertainty quantification took {uncertainty_time:.3f}s, exceeds 1s limit"
            
            # Test real-time updates
            update_result = await quantifier.update_uncertainty_realtime('test-task', {
                'prediction': 0.75,
                'confidence': 0.8,
                'model_source': 'updated_model'
            })
            assert isinstance(update_result, type(uncertainty_estimate))
            
            self.record_test_result("Uncertainty Quantifier", True, uncertainty_time)
            
        except Exception as e:
            self.record_test_result("Uncertainty Quantifier", False, 0, str(e))
    
    async def test_integration_layer(self):
        """Test Integration Layer"""
        print("🔗 Testing Integration Layer...")
        
        try:
            integration = ExternalValidatorDeepConfIntegration()
            
            # Create mock external validator
            class MockExternalValidator:
                async def validate(self, task):
                    return {
                        'passed': True,
                        'issues_found': [],
                        'recommendations': ['Add input validation'],
                        'validation_type': 'security_review'
                    }
            
            mock_validator = MockExternalValidator()
            
            # Test enhanced validation
            validation_task = {
                'task_id': 'validation-001',
                'code_content': 'def authenticate_user(username, password): return True',
                'validation_type': 'security_review',
                'requirements': ['input_validation', 'secure_authentication']
            }
            
            start_time = time.time()
            enhanced_result = await integration.validate_with_confidence(validation_task, mock_validator)
            validation_time = time.time() - start_time
            
            # Validate enhanced result
            assert hasattr(enhanced_result, 'validation_result')
            assert hasattr(enhanced_result, 'confidence_score')
            assert hasattr(enhanced_result, 'validation_confidence')
            assert hasattr(enhanced_result, 'uncertainty_analysis')
            assert hasattr(enhanced_result, 'quality_assessment')
            
            # Validate integration maintains backward compatibility
            assert 'passed' in enhanced_result.validation_result
            assert isinstance(enhanced_result.confidence_score, ConfidenceScore)
            assert 0.0 <= enhanced_result.validation_confidence <= 1.0
            
            # Test performance overhead is acceptable (<20% per PRD)
            baseline_time = 0.1  # Simulated baseline
            overhead_ratio = enhanced_result.enhancement_overhead / baseline_time
            assert overhead_ratio <= 0.2, f"Enhancement overhead {overhead_ratio:.2%} exceeds 20% limit"
            
            self.record_test_result("Integration Layer", True, validation_time)
            
        except Exception as e:
            self.record_test_result("Integration Layer", False, 0, str(e))
    
    async def test_performance_requirements(self):
        """Test overall performance requirements"""
        print("⚡ Testing Performance Requirements...")
        
        try:
            # Test concurrent operations
            engine = DeepConfEngine()
            
            # Create multiple tasks for concurrent processing
            tasks = []
            for i in range(10):
                task = MockAITask(
                    task_id=f"perf-{i}",
                    content=f"Performance test task {i}",
                    complexity="moderate",
                    domain="testing",
                    priority="medium"
                )
                context = MockTaskContext(
                    user_id=f"user-{i}",
                    session_id=f"session-{i}",
                    timestamp=time.time()
                )
                tasks.append((task, context))
            
            # Execute concurrent confidence calculations
            start_time = time.time()
            confidence_calculations = [
                engine.calculate_confidence(task, context) 
                for task, context in tasks
            ]
            
            results = await asyncio.gather(*confidence_calculations)
            concurrent_time = time.time() - start_time
            
            # Validate all calculations completed
            assert len(results) == 10
            for result in results:
                assert isinstance(result, ConfidenceScore)
                assert 0.0 <= result.overall_confidence <= 1.0
            
            # Performance should handle concurrency efficiently
            avg_time_per_task = concurrent_time / 10
            assert avg_time_per_task <= 2.0, f"Average time per concurrent task {avg_time_per_task:.3f}s too high"
            
            self.record_test_result("Performance Requirements", True, concurrent_time)
            
        except Exception as e:
            self.record_test_result("Performance Requirements", False, 0, str(e))
    
    def record_test_result(self, test_name: str, passed: bool, execution_time: float, error: str = None):
        """Record test result"""
        if passed:
            self.tests_passed += 1
            print(f"  ✅ {test_name} - {execution_time:.3f}s")
        else:
            self.tests_failed += 1
            print(f"  ❌ {test_name} - Failed: {error}")
            if error:
                print(f"     Error details: {error}")
        
        self.test_results.append({
            'test_name': test_name,
            'passed': passed,
            'execution_time': execution_time,
            'error': error
        })
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🏁 DeepConf Implementation Test Results")
        print("=" * 60)
        
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.tests_failed == 0:
            print("\n🎉 All tests passed! DeepConf implementation is working correctly.")
            print("✅ TDD Green Phase achieved - failing tests now pass with real implementation.")
        else:
            print(f"\n⚠️  {self.tests_failed} test(s) failed. See details above.")
        
        # Performance summary
        print("\n📊 Performance Summary:")
        for result in self.test_results:
            if result['passed'] and result['execution_time'] > 0:
                print(f"  {result['test_name']}: {result['execution_time']:.3f}s")

async def main():
    """Main test execution"""
    print("DeepConf Phase 7 Implementation Test Suite")
    print("Testing TDD Green Phase - Making failing tests pass")
    print()
    
    test_runner = DeepConfTestRunner()
    success = await test_runner.run_all_tests()
    
    if success:
        print("\n🟢 IMPLEMENTATION STATUS: COMPLETE")
        print("All core DeepConf components are functional and meet PRD requirements.")
        return 0
    else:
        print("\n🔴 IMPLEMENTATION STATUS: INCOMPLETE")  
        print("Some components need additional work to meet requirements.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)