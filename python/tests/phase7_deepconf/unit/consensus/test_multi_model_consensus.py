"""
Unit Tests for Multi-Model Consensus System

Tests the multi-model consensus mechanisms as specified in Phase 7 PRD.
All tests follow TDD Red Phase - they will fail until implementation is complete.

PRD Requirements Tested:
- Voting systems (simple majority, weighted voting, confidence-weighted)  
- Disagreement resolution with automatic escalation
- Model performance tracking with dynamic weighting
- Critical decision protocols for high-stakes development
- Fallback strategies when consensus cannot be reached

Performance Targets:
- Consensus decisions: <3s average response time
- Agreement rate: >90% for critical decisions  
- Escalation threshold: 30% disagreement triggers escalation
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Import test fixtures and helpers
from ...conftest import (
    tdd_red_phase, requires_implementation, performance_critical, dgts_validated,
    assert_response_time_target, MockModelResponse, MockConsensusResult
)


class TestMultiModelConsensusCore:
    """Test core consensus system functionality"""
    
    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    @performance_critical(3.0)
    async def test_request_consensus_basic_voting(self, mock_model_responses, performance_tracker):
        """
        Test basic consensus request with multiple models (PRD 4.2)
        
        WILL FAIL until MultiModelConsensus.request_consensus is implemented with:
        - Multi-provider model orchestration
        - Basic voting mechanism implementation
        - Response aggregation and decision making
        """
        performance_tracker.start_tracking("consensus_request")
        
        # This will fail - MultiModelConsensus doesn't exist yet (TDD Red Phase)
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create mock critical task
        critical_task = Mock()
        critical_task.task_id = "critical-001"
        critical_task.content = "Implement secure payment processing system"
        critical_task.complexity = "complex"
        critical_task.domain = "security_critical"
        critical_task.priority = "critical"
        critical_task.requires_consensus = True
        
        # Define participating models
        ai_models = ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"]
        
        # Test consensus request
        consensus_result = await consensus_system.request_consensus(critical_task, ai_models)
        
        performance_tracker.end_tracking("consensus_request")
        
        # Validate consensus result structure
        assert hasattr(consensus_result, 'agreed_response')
        assert hasattr(consensus_result, 'consensus_confidence') 
        assert hasattr(consensus_result, 'agreement_level')
        assert hasattr(consensus_result, 'disagreement_points')
        assert hasattr(consensus_result, 'escalation_required')
        assert hasattr(consensus_result, 'participating_models')
        assert hasattr(consensus_result, 'processing_time')
        
        # Validate consensus metrics
        assert 0.0 <= consensus_result.consensus_confidence <= 1.0
        assert 0.0 <= consensus_result.agreement_level <= 1.0
        assert len(consensus_result.participating_models) == len(ai_models)
        assert consensus_result.processing_time > 0
        
        # Performance validation (PRD requirement: <3s average)
        performance_tracker.assert_performance_target("consensus_request", 3.0)

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_weighted_voting_mechanism(self, mock_model_responses):
        """
        Test weighted voting based on model performance (PRD 4.2)
        
        WILL FAIL until weighted voting algorithm is implemented:
        - Historical performance-based weighting
        - Confidence-weighted vote aggregation
        - Dynamic weight adjustment
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create model responses with different confidence levels
        responses = [
            MockModelResponse(
                model_name="gpt-4o",
                response_content="Use OAuth 2.0 with PKCE for secure authentication",
                confidence_score=0.92,
                processing_time=1.1,
                token_usage=180,
                metadata={"expertise_domain": "security", "historical_accuracy": 0.88}
            ),
            MockModelResponse(
                model_name="claude-3.5-sonnet", 
                response_content="Implement OAuth 2.0 with additional rate limiting",
                confidence_score=0.89,
                processing_time=0.9,
                token_usage=165,
                metadata={"expertise_domain": "security", "historical_accuracy": 0.91}
            ),
            MockModelResponse(
                model_name="deepseek-v3",
                response_content="Consider JWT tokens with refresh mechanism", 
                confidence_score=0.73,
                processing_time=1.3,
                token_usage=200,
                metadata={"expertise_domain": "general", "historical_accuracy": 0.82}
            )
        ]
        
        # Test weighted voting
        weighted_result = await consensus_system.weighted_voting(responses)
        
        # Validate weighted voting structure  
        assert "winner" in weighted_result
        assert "confidence" in weighted_result
        assert "vote_distribution" in weighted_result
        assert "weighting_factors" in weighted_result
        
        # Winner should be highest weighted model (not just highest confidence)
        assert weighted_result["winner"] in ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"]
        
        # Vote distribution should sum to 1.0
        vote_sum = sum(weighted_result["vote_distribution"].values())
        assert abs(vote_sum - 1.0) < 0.01, f"Vote distribution should sum to 1.0, got {vote_sum}"
        
        # Weighting factors should consider multiple dimensions
        expected_factors = ["confidence_score", "historical_accuracy", "domain_expertise", "processing_efficiency"]
        for factor in expected_factors:
            assert factor in weighted_result["weighting_factors"]

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_disagreement_analysis_and_escalation(self, mock_model_responses):
        """
        Test disagreement analysis with escalation logic (PRD 4.2)
        
        WILL FAIL until disagreement analysis is implemented:
        - Disagreement level calculation
        - Conflict point identification
        - Automatic escalation triggers
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create highly disagreeing responses
        disagreeing_responses = [
            MockModelResponse(
                model_name="gpt-4o",
                response_content="Use microservices architecture with Docker",
                confidence_score=0.85,
                processing_time=1.2,
                token_usage=150
            ),
            MockModelResponse(
                model_name="claude-3.5-sonnet",
                response_content="Use monolithic architecture for simplicity", 
                confidence_score=0.83,
                processing_time=1.1,
                token_usage=145
            ),
            MockModelResponse(
                model_name="deepseek-v3",
                response_content="Use serverless functions for scalability",
                confidence_score=0.79,
                processing_time=1.4,
                token_usage=160
            )
        ]
        
        # Test disagreement analysis
        disagreement_report = await consensus_system.disagreement_analysis(disagreeing_responses)
        
        # Validate disagreement analysis structure
        assert "disagreement_level" in disagreement_report
        assert "conflict_points" in disagreement_report
        assert "resolution_needed" in disagreement_report
        assert "semantic_similarity" in disagreement_report
        assert "approach_divergence" in disagreement_report
        
        # Disagreement should be detected (high divergence in responses)
        assert disagreement_report["disagreement_level"] > 0.3, "High disagreement should be detected"
        assert len(disagreement_report["conflict_points"]) > 0
        assert disagreement_report["resolution_needed"] is True
        
        # Test escalation decision
        escalation_action = await consensus_system.escalation_decision(disagreement_report)
        
        # Validate escalation structure
        assert "escalation_required" in escalation_action
        assert "escalation_type" in escalation_action
        assert "recommended_action" in escalation_action
        
        # High disagreement should trigger escalation (PRD: >30% disagreement)
        if disagreement_report["disagreement_level"] > 0.3:
            assert escalation_action["escalation_required"] is True
            assert escalation_action["escalation_type"] in ["human_review", "extended_consensus", "expert_consultation"]

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus") 
    async def test_model_performance_tracking(self):
        """
        Test model performance tracking and dynamic weighting (PRD 4.2)
        
        WILL FAIL until performance tracking is implemented:
        - Historical accuracy tracking per model
        - Domain-specific performance metrics
        - Dynamic weight adjustment over time
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create mock historical performance data
        mock_performance = {
            "gpt-4o": {
                "overall_accuracy": 0.88,
                "domain_performance": {
                    "security": 0.92,
                    "frontend": 0.85,
                    "backend": 0.87,
                    "database": 0.81
                },
                "confidence_calibration": 0.86,
                "response_consistency": 0.89,
                "task_completion_rate": 0.91
            },
            "claude-3.5-sonnet": {
                "overall_accuracy": 0.91,
                "domain_performance": {
                    "security": 0.89,
                    "frontend": 0.93,
                    "backend": 0.90,
                    "database": 0.88
                },
                "confidence_calibration": 0.92,
                "response_consistency": 0.94,
                "task_completion_rate": 0.89
            },
            "deepseek-v3": {
                "overall_accuracy": 0.82,
                "domain_performance": {
                    "security": 0.79,
                    "frontend": 0.84,
                    "backend": 0.83,
                    "database": 0.85
                },
                "confidence_calibration": 0.80,
                "response_consistency": 0.83,
                "task_completion_rate": 0.85
            }
        }
        
        # Test model weight calculation
        model_weights = consensus_system.calculate_model_weights(mock_performance)
        
        # Validate model weights structure
        assert isinstance(model_weights, dict)
        for model_name in ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"]:
            assert model_name in model_weights
            assert 0.0 <= model_weights[model_name] <= 1.0
        
        # Weights should sum to approximately 1.0
        weight_sum = sum(model_weights.values())
        assert abs(weight_sum - 1.0) < 0.01, f"Model weights should sum to 1.0, got {weight_sum}"
        
        # Higher performing models should get higher weights
        claude_weight = model_weights["claude-3.5-sonnet"]  # Highest overall accuracy
        gpt_weight = model_weights["gpt-4o"]
        deepseek_weight = model_weights["deepseek-v3"]  # Lowest overall accuracy
        
        assert claude_weight >= gpt_weight >= deepseek_weight, "Weights should reflect performance ranking"

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_outlier_response_detection(self, mock_model_responses):
        """
        Test outlier response detection and handling (PRD 4.2)
        
        WILL FAIL until outlier detection is implemented:
        - Statistical outlier identification
        - Response quality assessment
        - Outlier impact on consensus
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create responses with one clear outlier
        responses_with_outlier = [
            MockModelResponse(
                model_name="gpt-4o",
                response_content="Implement REST API with proper authentication",
                confidence_score=0.87,
                processing_time=1.1,
                token_usage=150
            ),
            MockModelResponse(
                model_name="claude-3.5-sonnet",
                response_content="Create REST API endpoints with JWT authentication",
                confidence_score=0.89,
                processing_time=0.9,
                token_usage=145
            ),
            MockModelResponse(
                model_name="deepseek-v3", 
                response_content="DELETE FROM users WHERE 1=1; -- Malicious SQL",
                confidence_score=0.45,  # Low confidence
                processing_time=2.5,    # Slow response
                token_usage=300         # Excessive tokens
            )
        ]
        
        # Test outlier detection
        outlier_analysis = consensus_system.detect_outlier_responses(responses_with_outlier)
        
        # Validate outlier analysis structure
        assert "outliers_detected" in outlier_analysis
        assert "outlier_models" in outlier_analysis
        assert "outlier_reasons" in outlier_analysis
        assert "consensus_impact" in outlier_analysis
        
        # Malicious/poor quality response should be detected as outlier
        assert outlier_analysis["outliers_detected"] is True
        assert "deepseek-v3" in outlier_analysis["outlier_models"]
        assert len(outlier_analysis["outlier_reasons"]) > 0
        
        # Expected outlier detection reasons
        expected_reasons = ["low_confidence", "excessive_processing_time", "high_token_usage", "content_quality"]
        detected_reasons = outlier_analysis["outlier_reasons"]
        assert any(reason in detected_reasons for reason in expected_reasons)


class TestMultiModelConsensusVotingSystems:
    """Test different voting mechanisms and consensus algorithms"""
    
    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_simple_majority_voting(self, mock_model_responses):
        """
        Test simple majority voting mechanism (PRD 4.2)
        
        WILL FAIL until simple majority voting is implemented
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create responses with clear majority
        majority_responses = [
            MockModelResponse("gpt-4o", "Use React for frontend", 0.85, 1.1, 150),
            MockModelResponse("claude-3.5-sonnet", "Use React for frontend development", 0.87, 0.9, 145),
            MockModelResponse("deepseek-v3", "Use Vue.js for frontend", 0.82, 1.2, 160),
            MockModelResponse("gpt-3.5-turbo", "React is best choice for frontend", 0.83, 1.0, 140),
            MockModelResponse("gemini-pro", "Consider Angular for frontend", 0.79, 1.3, 170)
        ]
        
        # Test simple majority voting
        voting_result = await consensus_system.simple_majority_vote(majority_responses)
        
        # Validate voting result
        assert "winner" in voting_result
        assert "vote_count" in voting_result
        assert "majority_achieved" in voting_result
        
        # React should win with 3/5 votes
        assert "react" in voting_result["winner"].lower()
        assert voting_result["majority_achieved"] is True
        assert voting_result["vote_count"]["react"] >= 3

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_confidence_weighted_consensus(self, mock_model_responses):
        """
        Test confidence-weighted consensus algorithm (PRD 4.2)
        
        WILL FAIL until confidence weighting is implemented
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create responses with varying confidence levels
        confidence_weighted_responses = [
            MockModelResponse("gpt-4o", "Use PostgreSQL database", 0.95, 1.1, 150),
            MockModelResponse("claude-3.5-sonnet", "Use MySQL database", 0.70, 0.9, 145),
            MockModelResponse("deepseek-v3", "Use MongoDB database", 0.60, 1.2, 160)
        ]
        
        # Test confidence-weighted consensus
        weighted_consensus = await consensus_system.confidence_weighted_consensus(confidence_weighted_responses)
        
        # Validate weighted consensus result
        assert "consensus_response" in weighted_consensus
        assert "weighted_confidence" in weighted_consensus
        assert "confidence_distribution" in weighted_consensus
        
        # High confidence response should dominate despite being minority
        assert "postgresql" in weighted_consensus["consensus_response"].lower()
        assert weighted_consensus["weighted_confidence"] > 0.85

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_expert_knowledge_weighting(self):
        """
        Test expert knowledge-based weighting system (PRD 4.2)
        
        WILL FAIL until expert knowledge weighting is implemented
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Define expert knowledge domains for models
        expert_domains = {
            "gpt-4o": ["general_programming", "web_development", "api_design"],
            "claude-3.5-sonnet": ["code_analysis", "architecture", "security"],
            "deepseek-v3": ["algorithms", "optimization", "machine_learning"],
            "codellama": ["code_generation", "debugging", "refactoring"]
        }
        
        # Test expert knowledge weighting for security task
        security_task = Mock()
        security_task.domain = "security"
        security_task.complexity = "high"
        security_task.expertise_required = ["security", "architecture"]
        
        expert_weights = await consensus_system.calculate_expert_weights(
            list(expert_domains.keys()),
            security_task
        )
        
        # Validate expert weights
        assert isinstance(expert_weights, dict)
        for model in expert_domains.keys():
            assert model in expert_weights
            assert 0.0 <= expert_weights[model] <= 1.0
        
        # Claude should get highest weight for security task (has security expertise)
        claude_weight = expert_weights["claude-3.5-sonnet"]
        assert claude_weight == max(expert_weights.values()), "Claude should have highest security expertise weight"


class TestMultiModelConsensusEdgeCases:
    """Test edge cases and error conditions"""
    
    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_no_consensus_possible_fallback(self):
        """
        Test fallback strategies when no consensus is possible (PRD 4.2)
        
        WILL FAIL until fallback strategies are implemented
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create completely disagreeing responses
        no_consensus_responses = [
            MockModelResponse("model1", "Solution A", 0.85, 1.0, 100),
            MockModelResponse("model2", "Solution B", 0.85, 1.0, 100),
            MockModelResponse("model3", "Solution C", 0.85, 1.0, 100)
        ]
        
        # Test fallback strategy activation
        fallback_result = await consensus_system.apply_fallback_strategy(no_consensus_responses)
        
        # Validate fallback result structure
        assert "fallback_activated" in fallback_result
        assert "fallback_strategy" in fallback_result
        assert "fallback_response" in fallback_result
        
        # Fallback should be activated
        assert fallback_result["fallback_activated"] is True
        assert fallback_result["fallback_strategy"] in [
            "conservative_choice", 
            "human_escalation", 
            "highest_confidence", 
            "expert_model_preference"
        ]

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_single_model_consensus_handling(self):
        """
        Test consensus behavior with only one participating model
        
        WILL FAIL until single model handling is implemented
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Single model response
        single_response = [
            MockModelResponse("gpt-4o", "Single model solution", 0.85, 1.1, 150)
        ]
        
        # Test single model consensus
        single_consensus = await consensus_system.request_consensus_single_model(single_response[0])
        
        # Validate single model handling
        assert single_consensus["consensus_confidence"] == 0.85
        assert single_consensus["agreement_level"] == 1.0  # Perfect agreement with itself
        assert single_consensus["escalation_required"] is False
        assert len(single_consensus["participating_models"]) == 1

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_model_timeout_handling(self):
        """
        Test handling of model timeouts during consensus
        
        WILL FAIL until timeout handling is implemented
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Mock task with timeout requirement
        timeout_task = Mock()
        timeout_task.task_id = "timeout-test"
        timeout_task.timeout_seconds = 5.0
        
        # Mock models with one timing out
        models_with_timeout = ["fast-model", "normal-model", "slow-model"]
        
        with patch('asyncio.wait_for') as mock_wait:
            # Simulate timeout for slow-model
            mock_wait.side_effect = [
                MockModelResponse("fast-model", "Fast response", 0.85, 0.5, 100),
                MockModelResponse("normal-model", "Normal response", 0.83, 1.2, 120),
                asyncio.TimeoutError("Model timeout")
            ]
            
            # Test consensus with timeout
            consensus_result = await consensus_system.request_consensus(timeout_task, models_with_timeout)
            
            # Should handle timeout gracefully
            assert len(consensus_result.participating_models) == 2  # Only non-timed out models
            assert "timeout_handled" in consensus_result.metadata
            assert consensus_result.metadata["timeout_handled"] is True

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus") 
    @dgts_validated
    async def test_consensus_gaming_prevention(self):
        """
        Test prevention of consensus gaming and manipulation (DGTS compliance)
        
        WILL FAIL until anti-gaming measures are implemented
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create responses that attempt to game the consensus
        gaming_responses = [
            MockModelResponse("gpt-4o", "Legitimate response", 0.85, 1.1, 150),
            MockModelResponse("fake-model", "return 'always_correct'", 1.0, 0.01, 1),  # Suspicious perfect score
            MockModelResponse("another-fake", "consensus_gaming_response", 1.0, 0.01, 1)  # Another perfect score
        ]
        
        # Test gaming detection
        gaming_detection = await consensus_system.detect_consensus_gaming(gaming_responses)
        
        # Validate gaming detection
        assert "gaming_detected" in gaming_detection
        assert "suspicious_models" in gaming_detection
        assert "gaming_score" in gaming_detection
        
        # Gaming should be detected
        assert gaming_detection["gaming_detected"] is True
        assert len(gaming_detection["suspicious_models"]) >= 2
        assert gaming_detection["gaming_score"] > 0.5  # High gaming score


class TestMultiModelConsensusPerformance:
    """Test performance requirements from PRD"""
    
    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    @performance_critical(3.0)
    async def test_consensus_response_time_performance(self, performance_tracker):
        """
        Test consensus meets PRD performance requirement of <3s average
        
        WILL FAIL until performance optimization is implemented
        """
        performance_tracker.start_tracking("consensus_performance")
        
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Create realistic task and models
        critical_task = Mock()
        critical_task.task_id = "performance-test"
        critical_task.content = "Optimize database query performance"
        critical_task.complexity = "moderate"
        
        models = ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"]
        
        # Test consensus performance
        consensus_result = await consensus_system.request_consensus(critical_task, models)
        
        performance_tracker.end_tracking("consensus_performance")
        
        # Validate result exists
        assert consensus_result is not None
        
        # Performance validation (PRD requirement: <3s average)
        performance_tracker.assert_performance_target("consensus_performance", 3.0)

    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_consensus_agreement_rate_target(self):
        """
        Test consensus achieves PRD target of >90% agreement for critical decisions
        
        WILL FAIL until consensus algorithms achieve target agreement rates
        """
        from archon.deepconf.consensus import MultiModelConsensus
        
        consensus_system = MultiModelConsensus()
        
        # Test multiple critical decisions to measure agreement rate
        critical_tasks = []
        for i in range(10):
            task = Mock()
            task.task_id = f"critical-{i}"
            task.content = f"Critical security decision {i}"
            task.complexity = "critical"
            task.domain = "security"
            critical_tasks.append(task)
        
        agreement_rates = []
        models = ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"]
        
        for task in critical_tasks:
            consensus_result = await consensus_system.request_consensus(task, models)
            agreement_rates.append(consensus_result.agreement_level)
        
        # Calculate average agreement rate
        avg_agreement_rate = sum(agreement_rates) / len(agreement_rates)
        
        # PRD requirement: >90% agreement for critical decisions
        assert avg_agreement_rate >= 0.9, (
            f"Average agreement rate {avg_agreement_rate:.2%} below PRD requirement of 90%"
        )

# Integration hooks for future Phase integration testing

class TestMultiModelConsensusIntegration:
    """Test integration with other Phase systems"""
    
    @tdd_red_phase
    @requires_implementation("MultiModelConsensus")
    async def test_deepconf_engine_integration(self):
        """Test integration with DeepConf confidence engine"""
        pytest.skip("Requires DeepConf Engine integration")
    
    @tdd_red_phase 
    @requires_implementation("MultiModelConsensus")
    async def test_intelligent_router_integration(self):
        """Test integration with Intelligent Router for model selection"""
        pytest.skip("Requires Intelligent Router integration")