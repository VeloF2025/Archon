"""
Unit Tests for Intelligent Task Routing & Model Selection

Tests the intelligent routing system as specified in Phase 7 PRD.
All tests follow TDD Red Phase - they will fail until implementation is complete.

PRD Requirements Tested:
- Task complexity analysis for appropriate model selection
- Cost-performance optimization with token efficiency
- Dynamic load balancing across multiple AI providers
- Performance prediction and resource constraint management
- Token optimization achieving 70-85% savings target

Performance Targets:
- Routing decisions: <500ms for simple tasks
- Token optimization: 70-85% reduction
- Cost optimization without quality degradation
- Performance predictions within ±10% variance
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Import test fixtures and helpers
from ...conftest import (
    tdd_red_phase, requires_implementation, performance_critical, dgts_validated,
    assert_token_efficiency_target, assert_response_time_target,
    MockAITask, MockTaskContext, TEST_CONFIG
)


class TestIntelligentRouterCore:
    """Test core intelligent routing functionality"""
    
    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    @performance_critical(0.5)
    async def test_route_task_basic_functionality(self, simple_ai_task, mock_task_context, performance_tracker):
        """
        Test basic task routing with model selection (PRD 4.3)
        
        WILL FAIL until IntelligentRouter.route_task is implemented with:
        - Task complexity assessment
        - Model capability matching
        - Cost-benefit analysis
        - Performance prediction
        """
        performance_tracker.start_tracking("basic_routing")
        
        # This will fail - IntelligentRouter doesn't exist yet (TDD Red Phase)
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Test routing decision for simple task
        routing_decision = await router.route_task(simple_ai_task)
        
        performance_tracker.end_tracking("basic_routing")
        
        # Validate routing decision structure
        assert "selected_model" in routing_decision
        assert "reasoning" in routing_decision
        assert "estimated_tokens" in routing_decision
        assert "estimated_cost" in routing_decision
        assert "estimated_time" in routing_decision
        assert "confidence_score" in routing_decision
        
        # Validate routing logic for simple task
        assert routing_decision["selected_model"] in ["gpt-4o-mini", "claude-3-haiku", "gpt-3.5-turbo"]
        assert routing_decision["estimated_tokens"] < 100  # Simple tasks should use fewer tokens
        assert routing_decision["estimated_cost"] < 0.01   # Should be cost-effective
        assert routing_decision["confidence_score"] >= 0.7
        
        # Performance validation (PRD requirement: <500ms for simple tasks)
        performance_tracker.assert_performance_target("basic_routing", 0.5)

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_task_complexity_analysis(self, mock_ai_task, complex_ai_task, simple_ai_task):
        """
        Test task complexity analysis algorithm (PRD 4.3)
        
        WILL FAIL until complexity analysis is implemented:
        - Content length analysis
        - Domain specificity assessment
        - Technical depth evaluation
        - Context requirements analysis
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Test complexity analysis for different task types
        simple_complexity = await router.calculate_task_complexity(simple_ai_task)
        moderate_complexity = await router.calculate_task_complexity(mock_ai_task)
        complex_complexity = await router.calculate_task_complexity(complex_ai_task)
        
        # Validate complexity analysis structure
        for complexity_result in [simple_complexity, moderate_complexity, complex_complexity]:
            assert "complexity_score" in complexity_result
            assert "complexity_category" in complexity_result
            assert "factors" in complexity_result
            assert "reasoning" in complexity_result
            
            # Validate complexity score bounds
            assert 0.0 <= complexity_result["complexity_score"] <= 1.0
            assert complexity_result["complexity_category"] in ["simple", "moderate", "complex", "expert"]
            assert len(complexity_result["factors"]) > 0
        
        # Validate complexity ordering
        assert simple_complexity["complexity_score"] < moderate_complexity["complexity_score"]
        assert moderate_complexity["complexity_score"] < complex_complexity["complexity_score"]
        
        # Validate complexity factors
        expected_factors = ["content_length", "domain_specificity", "technical_depth", "context_requirements"]
        for factor in expected_factors:
            assert factor in complex_complexity["factors"]

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_optimal_model_selection(self, mock_ai_task, mock_task_context):
        """
        Test optimal model selection based on constraints (PRD 4.3)
        
        WILL FAIL until model selection algorithm is implemented:
        - Capability-task matching
        - Cost-performance optimization
        - Resource availability consideration
        - Quality requirements satisfaction
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Define resource constraints
        resource_constraints = {
            "max_cost_per_request": 0.05,
            "max_response_time": 10.0,
            "min_quality_score": 0.8,
            "max_tokens": 2000,
            "preferred_providers": ["openai", "anthropic"],
            "avoid_providers": ["experimental_models"]
        }
        
        # Test model selection
        model_selection = await router.select_optimal_model(mock_ai_task, resource_constraints)
        
        # Validate model selection structure
        assert "model" in model_selection
        assert "provider" in model_selection
        assert "confidence" in model_selection
        assert "cost_efficiency" in model_selection
        assert "quality_prediction" in model_selection
        assert "reasoning" in model_selection
        
        # Validate constraint satisfaction
        assert model_selection["confidence"] >= 0.7
        assert model_selection["cost_efficiency"] >= 0.6
        assert model_selection["quality_prediction"] >= resource_constraints["min_quality_score"]
        
        # Selected provider should be in preferred list
        assert model_selection["provider"] in resource_constraints["preferred_providers"]

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_token_usage_optimization(self, mock_ai_task, mock_task_context):
        """
        Test token usage optimization achieving 70-85% savings (PRD 4.3)
        
        WILL FAIL until token optimization is implemented:
        - Baseline token estimation
        - Optimization strategy selection
        - Content compression techniques
        - Quality preservation validation
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Set confidence target
        confidence_target = 0.8
        
        # Test token optimization
        token_optimization = await router.optimize_token_usage(mock_ai_task, confidence_target)
        
        # Validate optimization structure
        assert "original_tokens" in token_optimization
        assert "optimized_tokens" in token_optimization
        assert "savings_percentage" in token_optimization
        assert "optimization_strategies" in token_optimization
        assert "quality_maintained" in token_optimization
        assert "confidence_impact" in token_optimization
        
        # Validate optimization targets (PRD requirement: 70-85% savings)
        original_tokens = token_optimization["original_tokens"]
        optimized_tokens = token_optimization["optimized_tokens"]
        savings_percentage = token_optimization["savings_percentage"]
        
        assert original_tokens > optimized_tokens, "Optimization should reduce tokens"
        assert savings_percentage >= 0.70, f"Token savings {savings_percentage:.2%} below PRD minimum of 70%"
        assert savings_percentage <= 0.85, f"Token savings {savings_percentage:.2%} above realistic maximum of 85%"
        assert token_optimization["quality_maintained"] is True, "Quality must be maintained during optimization"
        
        # Confidence impact should be minimal
        assert abs(token_optimization["confidence_impact"]) <= 0.05, "Confidence impact should be minimal"

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_dynamic_load_balancing(self):
        """
        Test dynamic load balancing across providers (PRD 4.3)
        
        WILL FAIL until load balancing is implemented:
        - Provider availability monitoring
        - Load distribution algorithms
        - Failover handling
        - Performance-based routing
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Mock provider status
        provider_status = {
            "openai": {"availability": 0.95, "avg_response_time": 1.2, "current_load": 0.7},
            "anthropic": {"availability": 0.98, "avg_response_time": 0.9, "current_load": 0.3},
            "deepseek": {"availability": 0.92, "avg_response_time": 1.5, "current_load": 0.8},
            "together": {"availability": 0.88, "avg_response_time": 2.1, "current_load": 0.9}
        }
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(10):
            task = Mock()
            task.task_id = f"load-test-{i}"
            task.content = f"Load balancing test task {i}"
            task.complexity = "moderate"
            task.priority = "normal"
            tasks.append(task)
        
        # Test load balancing decisions
        routing_decisions = []
        for task in tasks:
            decision = await router.route_with_load_balancing(task, provider_status)
            routing_decisions.append(decision)
        
        # Validate load distribution
        provider_assignments = {}
        for decision in routing_decisions:
            provider = decision["selected_provider"]
            provider_assignments[provider] = provider_assignments.get(provider, 0) + 1
        
        # Load should be distributed (no single provider gets all tasks)
        max_assignments = max(provider_assignments.values())
        total_assignments = sum(provider_assignments.values())
        max_percentage = max_assignments / total_assignments
        
        assert max_percentage <= 0.6, f"Load balancing failed: one provider got {max_percentage:.2%} of tasks"
        
        # High availability providers should get more assignments
        anthropic_assignments = provider_assignments.get("anthropic", 0)
        together_assignments = provider_assignments.get("together", 0)
        assert anthropic_assignments >= together_assignments, "Higher availability provider should get more tasks"

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_performance_prediction_accuracy(self, mock_ai_task, mock_task_context):
        """
        Test performance prediction accuracy (PRD 4.3)
        
        WILL FAIL until performance prediction is implemented:
        - Response time prediction
        - Quality score prediction
        - Cost estimation accuracy
        - Confidence prediction calibration
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Test performance prediction
        performance_prediction = await router.predict_performance(mock_ai_task, "gpt-4o")
        
        # Validate prediction structure
        assert "estimated_response_time" in performance_prediction
        assert "estimated_quality_score" in performance_prediction
        assert "estimated_cost" in performance_prediction
        assert "confidence_in_prediction" in performance_prediction
        assert "prediction_factors" in performance_prediction
        
        # Validate prediction bounds
        assert performance_prediction["estimated_response_time"] > 0
        assert 0.0 <= performance_prediction["estimated_quality_score"] <= 1.0
        assert performance_prediction["estimated_cost"] > 0
        assert 0.0 <= performance_prediction["confidence_in_prediction"] <= 1.0
        
        # Test prediction accuracy tracking
        historical_predictions = [
            {"predicted": 1.2, "actual": 1.1, "task_type": "moderate"},
            {"predicted": 0.8, "actual": 0.9, "task_type": "simple"},
            {"predicted": 2.1, "actual": 2.3, "task_type": "complex"}
        ]
        
        accuracy_metrics = await router.evaluate_prediction_accuracy(historical_predictions)
        
        # Validate accuracy metrics
        assert "mean_absolute_error" in accuracy_metrics
        assert "prediction_variance" in accuracy_metrics
        assert "calibration_score" in accuracy_metrics
        
        # PRD requirement: predictions within ±10% variance
        assert accuracy_metrics["prediction_variance"] <= 0.10, "Prediction variance exceeds PRD requirement of ±10%"


class TestIntelligentRouterResourceManagement:
    """Test resource constraint management and optimization"""
    
    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_gpu_availability_handling(self):
        """
        Test GPU availability and resource allocation (PRD 4.3)
        
        WILL FAIL until GPU resource management is implemented
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Mock GPU resource constraints
        gpu_constraints = {
            "total_gpu_memory": 24000,  # 24GB
            "available_gpu_memory": 8000,  # 8GB available
            "gpu_utilization": 0.6,
            "cuda_available": True,
            "gpu_queue_length": 3
        }
        
        # Create GPU-intensive task
        gpu_task = Mock()
        gpu_task.task_id = "gpu-intensive"
        gpu_task.content = "Fine-tune large language model"
        gpu_task.complexity = "expert"
        gpu_task.requires_gpu = True
        gpu_task.estimated_gpu_memory = 12000  # 12GB required
        
        # Test GPU-aware routing
        gpu_routing = await router.route_with_gpu_constraints(gpu_task, gpu_constraints)
        
        # Validate GPU routing
        assert "gpu_allocation" in gpu_routing
        assert "fallback_strategy" in gpu_routing
        
        # Should use fallback strategy due to insufficient GPU memory
        if gpu_constraints["available_gpu_memory"] < gpu_task.estimated_gpu_memory:
            assert gpu_routing["fallback_strategy"] is not None
            assert gpu_routing["fallback_strategy"] in ["cloud_gpu", "cpu_fallback", "queue_task"]

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_rate_limit_management(self):
        """
        Test rate limit handling and request queuing (PRD 4.3)
        
        WILL FAIL until rate limit management is implemented
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Mock rate limit status
        rate_limits = {
            "openai": {"requests_per_minute": 60, "current_usage": 55, "reset_time": 30},
            "anthropic": {"requests_per_minute": 100, "current_usage": 20, "reset_time": 45},
            "deepseek": {"requests_per_minute": 50, "current_usage": 48, "reset_time": 15}
        }
        
        # Create urgent task
        urgent_task = Mock()
        urgent_task.task_id = "urgent-request"
        urgent_task.priority = "critical"
        urgent_task.max_wait_time = 10  # 10 seconds max wait
        
        # Test rate limit aware routing
        rate_limited_routing = await router.route_with_rate_limits(urgent_task, rate_limits)
        
        # Validate rate limit handling
        assert "selected_provider" in rate_limited_routing
        assert "estimated_wait_time" in rate_limited_routing
        assert "rate_limit_strategy" in rate_limited_routing
        
        # Should select provider with available capacity
        selected_provider = rate_limited_routing["selected_provider"]
        provider_usage = rate_limits[selected_provider]["current_usage"]
        provider_limit = rate_limits[selected_provider]["requests_per_minute"]
        
        assert provider_usage < provider_limit, "Should select provider with available capacity"

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_cost_budget_optimization(self, mock_ai_task):
        """
        Test cost budget optimization and allocation (PRD 4.3)
        
        WILL FAIL until cost budget management is implemented
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Define cost budget constraints
        budget_constraints = {
            "daily_budget": 100.0,  # $100/day
            "current_spend": 75.0,  # $75 spent
            "remaining_budget": 25.0,  # $25 remaining
            "cost_per_request_limit": 0.10,  # $0.10 max per request
            "budget_alert_threshold": 0.9  # Alert at 90% budget used
        }
        
        # Test cost-optimized routing
        cost_optimized_routing = await router.route_with_budget_constraints(mock_ai_task, budget_constraints)
        
        # Validate cost optimization
        assert "selected_model" in cost_optimized_routing
        assert "estimated_cost" in cost_optimized_routing
        assert "budget_impact" in cost_optimized_routing
        assert "cost_optimization_applied" in cost_optimized_routing
        
        # Cost should be within budget limits
        estimated_cost = cost_optimized_routing["estimated_cost"]
        assert estimated_cost <= budget_constraints["cost_per_request_limit"]
        assert estimated_cost <= budget_constraints["remaining_budget"]
        
        # Budget alert should be triggered if near limit
        budget_after = budget_constraints["current_spend"] + estimated_cost
        budget_percentage = budget_after / budget_constraints["daily_budget"]
        
        if budget_percentage >= budget_constraints["budget_alert_threshold"]:
            assert "budget_alert" in cost_optimized_routing
            assert cost_optimized_routing["budget_alert"] is True


class TestIntelligentRouterEdgeCases:
    """Test edge cases and error conditions"""
    
    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_all_providers_unavailable_fallback(self):
        """
        Test fallback when all providers are unavailable
        
        WILL FAIL until provider fallback is implemented
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Mock all providers unavailable
        provider_status = {
            "openai": {"availability": 0.0, "status": "maintenance"},
            "anthropic": {"availability": 0.0, "status": "rate_limited"},
            "deepseek": {"availability": 0.0, "status": "timeout"}
        }
        
        emergency_task = Mock()
        emergency_task.task_id = "emergency-routing"
        emergency_task.priority = "critical"
        
        # Test emergency fallback routing
        fallback_routing = await router.emergency_fallback_routing(emergency_task, provider_status)
        
        # Validate fallback strategy
        assert "fallback_activated" in fallback_routing
        assert "fallback_strategy" in fallback_routing
        assert fallback_routing["fallback_activated"] is True
        
        expected_fallback_strategies = ["local_model", "cached_response", "queue_for_retry", "human_escalation"]
        assert fallback_routing["fallback_strategy"] in expected_fallback_strategies

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_invalid_task_routing_error_handling(self):
        """
        Test error handling for invalid or malformed tasks
        
        WILL FAIL until error handling is implemented
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Test with None task
        with pytest.raises(ValueError, match="Task cannot be None"):
            await router.route_task(None)
        
        # Test with invalid task structure
        invalid_task = Mock()
        invalid_task.content = None  # Invalid content
        invalid_task.task_id = None  # Invalid ID
        
        with pytest.raises(ValueError, match="Invalid task structure"):
            await router.route_task(invalid_task)
        
        # Test with unsupported complexity level
        unsupported_task = Mock()
        unsupported_task.content = "Valid content"
        unsupported_task.task_id = "valid-id"
        unsupported_task.complexity = "impossible"  # Unsupported complexity
        
        with pytest.raises(ValueError, match="Unsupported complexity level"):
            await router.calculate_task_complexity(unsupported_task)

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    @dgts_validated
    async def test_routing_gaming_prevention(self):
        """
        Test prevention of routing gaming and manipulation (DGTS compliance)
        
        WILL FAIL until anti-gaming measures are implemented
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Create task that attempts to game routing system
        gaming_task = Mock()
        gaming_task.content = "route_to_expensive_model_always"
        gaming_task.task_id = "gaming-attempt"
        gaming_task.complexity = "simple"  # Claims simple but tries to force expensive routing
        gaming_task.force_model = "gpt-4o"  # Attempts to bypass routing logic
        gaming_task.gaming_marker = "always_route_premium"
        
        # Test anti-gaming routing
        anti_gaming_routing = await router.route_task_with_gaming_detection(gaming_task)
        
        # Validate gaming prevention
        assert "gaming_detection" in anti_gaming_routing
        assert "routing_legitimacy_score" in anti_gaming_routing
        
        # Gaming should be detected and prevented
        assert anti_gaming_routing["gaming_detection"]["gaming_detected"] is True
        assert anti_gaming_routing["routing_legitimacy_score"] < 0.5
        
        # Should not route to expensive model for simple task
        selected_model = anti_gaming_routing["selected_model"]
        expensive_models = ["gpt-4o", "claude-3-opus", "gpt-4-32k"]
        assert selected_model not in expensive_models, "Should not route gaming task to expensive model"


class TestIntelligentRouterPerformance:
    """Test performance requirements from PRD"""
    
    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    @performance_critical(0.5)
    async def test_routing_decision_speed(self, simple_ai_task, performance_tracker):
        """
        Test routing decision speed meets PRD requirement of <500ms for simple tasks
        
        WILL FAIL until routing optimization achieves target performance
        """
        performance_tracker.start_tracking("routing_speed")
        
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Perform multiple routing decisions to test average speed
        routing_times = []
        for i in range(10):
            start_time = time.time()
            await router.route_task(simple_ai_task)
            end_time = time.time()
            routing_times.append(end_time - start_time)
        
        performance_tracker.end_tracking("routing_speed")
        
        # Calculate average routing time
        avg_routing_time = sum(routing_times) / len(routing_times)
        
        # PRD requirement: <500ms for simple tasks
        assert avg_routing_time <= 0.5, f"Average routing time {avg_routing_time:.3f}s exceeds PRD requirement of 500ms"

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_token_optimization_efficiency_target(self, mock_ai_task):
        """
        Test token optimization meets PRD efficiency target of 70-85% savings
        
        WILL FAIL until token optimization algorithms achieve target efficiency
        """
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Test multiple optimization scenarios
        optimization_results = []
        confidence_targets = [0.7, 0.8, 0.9]
        
        for confidence_target in confidence_targets:
            optimization = await router.optimize_token_usage(mock_ai_task, confidence_target)
            optimization_results.append(optimization)
        
        # Validate efficiency targets
        for result in optimization_results:
            savings_percentage = result["savings_percentage"]
            
            # PRD requirement: 70-85% token savings
            assert savings_percentage >= 0.70, f"Token savings {savings_percentage:.2%} below PRD minimum of 70%"
            assert result["quality_maintained"] is True, "Quality must be maintained during optimization"
        
        # Higher confidence targets should still achieve significant savings
        high_confidence_savings = optimization_results[-1]["savings_percentage"]  # 90% confidence target
        assert high_confidence_savings >= 0.60, "Should achieve >60% savings even at high confidence targets"

    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    @performance_critical(2.0)
    async def test_concurrent_routing_throughput(self, performance_tracker):
        """
        Test concurrent routing throughput and scalability
        
        WILL FAIL until concurrent routing optimization is implemented
        """
        performance_tracker.start_tracking("concurrent_routing")
        
        from archon.deepconf.routing import IntelligentRouter
        
        router = IntelligentRouter()
        
        # Create multiple concurrent routing tasks
        concurrent_tasks = []
        for i in range(50):  # Test with 50 concurrent tasks
            task = Mock()
            task.task_id = f"concurrent-{i}"
            task.content = f"Concurrent routing test {i}"
            task.complexity = ["simple", "moderate", "complex"][i % 3]
            concurrent_tasks.append(task)
        
        # Execute all routing decisions concurrently
        routing_operations = [router.route_task(task) for task in concurrent_tasks]
        results = await asyncio.gather(*routing_operations)
        
        performance_tracker.end_tracking("concurrent_routing")
        
        # Validate all routing completed successfully
        assert len(results) == 50
        for result in results:
            assert "selected_model" in result
            assert "estimated_cost" in result
        
        # Performance should handle concurrency efficiently (target: <2s for 50 concurrent)
        performance_tracker.assert_performance_target("concurrent_routing", 2.0)


# Integration hooks for future Phase integration testing

class TestIntelligentRouterIntegration:
    """Test integration with other Phase 7 systems"""
    
    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_deepconf_engine_integration(self):
        """Test integration with DeepConf confidence engine for routing decisions"""
        pytest.skip("Requires DeepConf Engine integration")
    
    @tdd_red_phase
    @requires_implementation("IntelligentRouter") 
    async def test_consensus_system_integration(self):
        """Test integration with Multi-Model Consensus for routing optimization"""
        pytest.skip("Requires Multi-Model Consensus integration")
    
    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_phase5_external_validator_routing_integration(self):
        """Test routing integration with Phase 5 External Validator"""
        pytest.skip("Requires Phase 5 External Validator integration")
        
    @tdd_red_phase
    @requires_implementation("IntelligentRouter")
    async def test_phase9_tdd_routing_integration(self):
        """Test routing integration with Phase 9 TDD Enforcement"""
        pytest.skip("Requires Phase 9 TDD Enforcement integration")