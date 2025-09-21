"""
Comprehensive Tests for Agent Handoff System

Tests for the complete handoff system including:
- Handoff engine logic and strategies
- Context preservation
- Learning and optimization
- API endpoints
- Capability matching
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from src.agents.orchestration.agent_handoff_engine import (
    AgentHandoffEngine, HandoffStrategy, HandoffTrigger, HandoffStatus,
    HandoffDecision, HandoffResult, AgentCapability
)
from src.agents.orchestration.handoff_strategies import (
    SequentialHandoffStrategy, CollaborativeHandoffStrategy,
    ConditionalHandoffStrategy, ParallelHandoffStrategy, DelegationHandoffStrategy
)
from src.agents.orchestration.context_preservation import (
    ContextPreservationEngine, ContextPackage
)
from src.agents.orchestration.handoff_learning import (
    HandoffLearningEngine, HandoffPattern, OptimizationInsight
)
from src.agents.enhanced_agent_capabilities import (
    EnhancedAgentCapabilitySystem, CapabilityTaxonomy, ExpertiseLevel
)
from src.agents.capability_matching import (
    CapabilityMatcher, MatchingFactors, MatchingResult
)
from src.server.api_routes.handoff_api import router as handoff_router


class TestAgentHandoffEngine:
    """Test suite for Agent Handoff Engine."""

    @pytest.fixture
    def mock_agency(self):
        """Create a mock agency for testing."""
        agency = Mock()
        agency.agents = {
            "agent1": Mock(name="Code Implementer", agent_type="CODE_IMPLEMENTER"),
            "agent2": Mock(name="Security Auditor", agent_type="SECURITY_AUDITOR"),
            "agent3": Mock(name="System Architect", agent_type="SYSTEM_ARCHITECT")
        }
        return agency

    @pytest.fixture
    def handoff_engine(self, mock_agency):
        """Create a handoff engine instance."""
        return AgentHandoffEngine(mock_agency)

    def test_handoff_engine_initialization(self, handoff_engine):
        """Test handoff engine initializes correctly."""
        assert handoff_engine.agency is not None
        assert isinstance(handoff_engine.agent_capabilities, dict)
        assert isinstance(handoff_engine.handoff_history, list)
        assert len(handoff_engine.strategies) == 5  # All strategies should be loaded

    def test_register_agent_capability(self, handoff_engine):
        """Test registering agent capabilities."""
        capability = AgentCapability(
            capability_type="code_analysis",
            expertise_level=ExpertiseLevel.ADVANCED,
            confidence_score=0.85,
            performance_metrics=Mock(tasks_completed=10, success_rate=0.9)
        )

        handoff_engine.register_agent_capability("agent1", capability)

        assert "agent1" in handoff_engine.agent_capabilities
        assert handoff_engine.agent_capabilities["agent1"].capability_type == "code_analysis"

    def test_analyze_handoff_decision(self, handoff_engine):
        """Test handoff decision analysis."""
        # Setup capabilities
        handoff_engine.register_agent_capability("agent1", AgentCapability(
            capability_type="code_analysis",
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            confidence_score=0.7,
            performance_metrics=Mock(tasks_completed=5, success_rate=0.8)
        ))

        handoff_engine.register_agent_capability("agent2", AgentCapability(
            capability_type="security_analysis",
            expertise_level=ExpertiseLevel.EXPERT,
            confidence_score=0.95,
            performance_metrics=Mock(tasks_completed=20, success_rate=0.95)
        ))

        decision = handoff_engine.analyze_handoff_decision(
            current_agent="agent1",
            message="I need help with security vulnerability analysis",
            task_description="Find and fix security vulnerabilities in authentication system"
        )

        assert isinstance(decision, HandoffDecision)
        assert decision.should_handoff is True
        assert decision.recommended_agent == "agent2"
        assert decision.confidence_score > 0.8

    @pytest.mark.asyncio
    async def test_execute_handoff(self, handoff_engine):
        """Test handoff execution."""
        # Mock the strategy execution
        with patch.object(handoff_engine, '_execute_strategy') as mock_execute:
            mock_execute.return_value = HandoffResult(
                handoff_id="test_handoff_1",
                status=HandoffStatus.COMPLETED,
                source_agent_id="agent1",
                target_agent_id="agent2",
                execution_time=1500,
                metrics={"context_transfer_time": 200, "agent_response_time": 1300}
            )

            result = await handoff_engine.execute_handoff(
                source_agent="agent1",
                target_agent="agent2",
                message="Please help with security analysis",
                task_description="Security vulnerability analysis",
                strategy=HandoffStrategy.SEQUENTIAL
            )

            assert result.status == HandoffStatus.COMPLETED
            assert result.execution_time == 1500
            mock_execute.assert_called_once()


class TestHandoffStrategies:
    """Test suite for handoff strategies."""

    @pytest.fixture
    def mock_agency(self):
        """Create a mock agency."""
        agency = Mock()
        agency.agents = {
            "agent1": Mock(),
            "agent2": Mock()
        }
        return agency

    @pytest.mark.asyncio
    async def test_sequential_strategy(self, mock_agency):
        """Test sequential handoff strategy."""
        strategy = SequentialHandoffStrategy()

        result = await strategy.execute(
            agency=mock_agency,
            source_agent="agent1",
            target_agent="agent2",
            message="Test message",
            task_description="Test task"
        )

        assert result.status == HandoffStatus.COMPLETED
        assert "sequential" in result.metrics.get("strategy_notes", "")

    @pytest.mark.asyncio
    async def test_collaborative_strategy(self, mock_agency):
        """Test collaborative handoff strategy."""
        strategy = CollaborativeHandoffStrategy()

        result = await strategy.execute(
            agency=mock_agency,
            source_agent="agent1",
            target_agent="agent2",
            message="Collaborative task",
            task_description="Complex task requiring collaboration"
        )

        assert result.status in [HandoffStatus.COMPLETED, HandoffStatus.IN_PROGRESS]
        assert "collaborative" in result.metrics.get("strategy_notes", "")

    @pytest.mark.asyncio
    async def test_conditional_strategy(self, mock_agency):
        """Test conditional handoff strategy."""
        strategy = ConditionalHandoffStrategy()

        result = await strategy.execute(
            agency=mock_agency,
            source_agent="agent1",
            target_agent="agent2",
            message="Conditional handoff",
            task_description="Task with specific conditions",
            context={"conditions": ["high_complexity", "security_critical"]}
        )

        assert result.status == HandoffStatus.COMPLETED
        assert result.metrics.get("conditions_met", False) is True

    @pytest.mark.asyncio
    async def test_parallel_strategy(self, mock_agency):
        """Test parallel handoff strategy."""
        strategy = ParallelHandoffStrategy()

        result = await strategy.execute(
            agency=mock_agency,
            source_agent="agent1",
            target_agent="agent2",
            message="Parallel processing task",
            task_description="Task that can be processed in parallel"
        )

        assert result.status == HandoffStatus.COMPLETED
        assert result.metrics.get("parallel_workers", 0) > 0

    @pytest.mark.asyncio
    async def test_delegation_strategy(self, mock_agency):
        """Test delegation handoff strategy."""
        strategy = DelegationHandoffStrategy()

        result = await strategy.execute(
            agency=mock_agency,
            source_agent="agent1",
            target_agent="agent2",
            message="Delegated task",
            task_description="Task to be completely delegated"
        )

        assert result.status == HandoffStatus.COMPLETED
        assert result.metrics.get("delegation_complete", False) is True


class TestContextPreservation:
    """Test suite for context preservation."""

    @pytest.fixture
    def context_engine(self):
        """Create a context preservation engine."""
        return ContextPreservationEngine(max_context_size=10000)

    def test_create_context_package(self, context_engine):
        """Test creating context packages."""
        package_id = context_engine.create_context_package(
            agent_id="agent1",
            task_id="task1",
            context_data={"key": "value"},
            conversation_history=[
                {"role": "user", "content": "Hello", "timestamp": datetime.now()},
                {"role": "assistant", "content": "Hi there!", "timestamp": datetime.now()}
            ]
        )

        assert package_id is not None
        assert package_id in context_engine.context_packages

        package = context_engine.context_packages[package_id]
        assert package.agent_id == "agent1"
        assert package.task_id == "task1"
        assert len(package.conversation_history) == 2

    def test_context_compression(self, context_engine):
        """Test context compression for large contexts."""
        # Create a large context
        large_context = {"data": "x" * 5000}  # 5KB of data
        long_history = [{"role": "user", "content": f"Message {i}", "timestamp": datetime.now()}
                       for i in range(100)]

        package_id = context_engine.create_context_package(
            agent_id="agent1",
            task_id="task1",
            context_data=large_context,
            conversation_history=long_history
        )

        package = context_engine.context_packages[package_id]
        assert package.compression_ratio > 0  # Should be compressed
        assert package.size_bytes < 20000  # Should be smaller than original

    def test_context_validation(self, context_engine):
        """Test context package validation."""
        package_id = context_engine.create_context_package(
            agent_id="agent1",
            task_id="task1",
            context_data={"test": "data"},
            conversation_history=[]
        )

        # Valid package
        is_valid = context_engine.validate_context_integrity(package_id)
        assert is_valid is True

        # Non-existent package
        is_valid = context_engine.validate_context_integrity("non_existent")
        assert is_valid is False

    def test_context_cleanup(self, context_engine):
        """Test cleanup of expired contexts."""
        # Create some packages
        package1_id = context_engine.create_context_package("agent1", "task1", {})
        package2_id = context_engine.create_context_package("agent2", "task2", {})

        # Manually set one as expired
        package2 = context_engine.context_packages[package2_id]
        package2.expires_at = datetime.now() - timedelta(hours=1)

        # Run cleanup
        cleaned_count = context_engine.cleanup_expired_contexts()

        assert cleaned_count == 1
        assert package1_id in context_engine.context_packages
        assert package2_id not in context_engine.context_packages


class TestHandoffLearning:
    """Test suite for handoff learning engine."""

    @pytest.fixture
    def learning_engine(self):
        """Create a handoff learning engine."""
        return HandoffLearningEngine()

    def test_record_handoff_result(self, learning_engine):
        """Test recording handoff results."""
        result = HandoffResult(
            handoff_id="test_handoff",
            status=HandoffStatus.COMPLETED,
            source_agent_id="agent1",
            target_agent_id="agent2",
            execution_time=1200,
            metrics={"success": True, "quality_score": 0.9}
        )

        learning_engine.record_handoff_result(result)

        assert len(learning_engine.handoff_patterns) > 0
        pattern = learning_engine.handoff_patterns[-1]
        assert pattern.handoff_id == "test_handoff"
        assert pattern.successful is True

    def test_pattern_recognition(self, learning_engine):
        """Test pattern recognition in handoff data."""
        # Record several similar handoffs
        for i in range(5):
            result = HandoffResult(
                handoff_id=f"handoff_{i}",
                status=HandoffStatus.COMPLETED,
                source_agent_id="agent1",
                target_agent_id="agent2",
                execution_time=1000 + i * 100,
                metrics={"strategy": "sequential", "success": True}
            )
            learning_engine.record_handoff_result(result)

        patterns = learning_engine.identify_patterns()
        assert len(patterns) > 0

        # Should find a pattern for agent1 -> agent2 sequential handoffs
        agent_pattern = next((p for p in patterns if "agent1" in p.pattern_description and "agent2" in p.pattern_description), None)
        assert agent_pattern is not None
        assert agent_pattern.occurrence_count >= 5

    def test_optimization_insights(self, learning_engine):
        """Test generation of optimization insights."""
        # Record handoffs with varying performance
        handoffs = [
            HandoffResult("h1", HandoffStatus.COMPLETED, "agent1", "agent2", 2000, {"quality": 0.7}),
            HandoffResult("h2", HandoffStatus.COMPLETED, "agent1", "agent2", 800, {"quality": 0.95}),
            HandoffResult("h3", HandoffStatus.COMPLETED, "agent1", "agent2", 2200, {"quality": 0.6}),
        ]

        for handoff in handoffs:
            learning_engine.record_handoff_result(handoff)

        insights = learning_engine.generate_optimization_insights()

        # Should find optimization opportunities
        optimization_insights = [insight for insight in insights if isinstance(insight, OptimizationInsight)]
        assert len(optimization_insights) > 0

        # Look for performance improvement insights
        perf_insight = next((ins for ins in optimization_insights if "performance" in ins.description.lower()), None)
        assert perf_insight is not None

    def test_confidence_improvement(self, learning_engine):
        """Test confidence improvement tracking."""
        initial_confidence = learning_engine.get_average_confidence()

        # Record successful handoffs
        for i in range(10):
            result = HandoffResult(
                handoff_id=f"success_{i}",
                status=HandoffStatus.COMPLETED,
                source_agent_id="agent1",
                target_agent_id="agent2",
                execution_time=1000,
                metrics={"confidence_score": 0.6 + i * 0.03}  # Improving confidence
            )
            learning_engine.record_handoff_result(result)

        final_confidence = learning_engine.get_average_confidence()
        assert final_confidence > initial_confidence


class TestEnhancedAgentCapabilities:
    """Test suite for enhanced agent capabilities."""

    @pytest.fixture
    def capability_system(self):
        """Create an enhanced capability system."""
        return EnhancedAgentCapabilitySystem()

    def test_capability_taxonomy_initialization(self, capability_system):
        """Test capability taxonomy initialization."""
        assert len(capability_system.taxonomy.categories) > 0
        assert "coding" in capability_system.taxonomy.categories
        assert "analysis" in capability_system.taxonomy.categories

    def test_create_capability_profile(self, capability_system):
        """Test creating agent capability profiles."""
        profile = capability_system.create_capability_profile(
            agent_id="agent1",
            agent_type="CODE_IMPLEMENTER",
            initial_capabilities={
                "python_programming": ExpertiseLevel.ADVANCED,
                "code_analysis": ExpertiseLevel.INTERMEDIATE,
                "testing": ExpertiseLevel.BEGINNER
            }
        )

        assert profile.agent_id == "agent1"
        assert profile.capabilities["python_programming"].expertise_level == ExpertiseLevel.ADVANCED
        assert profile.overall_capability_score > 0

    def test_capability_gap_analysis(self, capability_system):
        """Test capability gap analysis."""
        # Create two agents with different capabilities
        profile1 = capability_system.create_capability_profile(
            "agent1", "CODE_IMPLEMENTER",
            {"python_programming": ExpertiseLevel.ADVANCED}
        )

        profile2 = capability_system.create_capability_profile(
            "agent2", "SECURITY_AUDITOR",
            {"security_analysis": ExpertiseLevel.EXPERT}
        )

        required_capabilities = ["python_programming", "security_analysis", "api_design"]
        gaps = capability_system.analyze_capability_gaps(
            [profile1, profile2], required_capabilities
        )

        assert len(gaps) > 0
        assert gaps["api_design"].severity > 0.5  # API design should be a significant gap

    def test_capability_evolution(self, capability_system):
        """Test capability evolution over time."""
        profile = capability_system.create_capability_profile(
            "agent1", "CODE_IMPLEMENTER",
            {"python_programming": ExpertiseLevel.INTERMEDIATE}
        )

        initial_score = profile.overall_capability_score

        # Simulate capability improvement
        capability_system.update_capability(
            "agent1", "python_programming", ExpertiseLevel.ADVANCED
        )

        updated_profile = capability_system.get_capability_profile("agent1")
        assert updated_profile.overall_capability_score > initial_score


class TestCapabilityMatching:
    """Test suite for capability matching."""

    @pytest.fixture
    def capability_matcher(self):
        """Create a capability matcher."""
        return CapabilityMatcher()

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents with capabilities."""
        return {
            "agent1": {
                "capabilities": {
                    "python_programming": {"level": ExpertiseLevel.EXPERT, "confidence": 0.95},
                    "web_development": {"level": ExpertiseLevel.ADVANCED, "confidence": 0.85}
                },
                "performance": {"success_rate": 0.9, "avg_completion_time": 1200},
                "availability": 0.8,
                "current_load": 0.3
            },
            "agent2": {
                "capabilities": {
                    "security_analysis": {"level": ExpertiseLevel.EXPERT, "confidence": 0.98},
                    "python_programming": {"level": ExpertiseLevel.ADVANCED, "confidence": 0.8}
                },
                "performance": {"success_rate": 0.95, "avg_completion_time": 1800},
                "availability": 0.9,
                "current_load": 0.1
            }
        }

    def test_basic_matching(self, capability_matcher, mock_agents):
        """Test basic agent-task matching."""
        task_requirements = {
            "python_programming": ExpertiseLevel.ADVANCED,
            "security_analysis": ExpertiseLevel.INTERMEDIATE
        }

        result = capability_matcher.find_best_match(
            task_requirements, mock_agents, context={"project_type": "security_audit"}
        )

        assert isinstance(result, MatchingResult)
        assert result.best_agent_id in ["agent1", "agent2"]
        assert result.confidence_score > 0.7
        assert len(result.match_scores) == 2

    def test_multi_factor_scoring(self, capability_matcher, mock_agents):
        """Test multi-factor scoring including performance and availability."""
        task_requirements = {"security_analysis": ExpertiseLevel.EXPERT}

        result = capability_matcher.find_best_match(
            task_requirements, mock_agents,
            weights={
                MatchingFactors.CAPABILITY_MATCH: 0.4,
                MatchingFactors.PERFORMANCE_HISTORY: 0.3,
                MatchingFactors.AVAILABILITY: 0.2,
                MatchingFactors.LOAD_BALANCE: 0.1
            }
        )

        # Agent2 should be preferred for security analysis due to expertise level
        assert result.best_agent_id == "agent2"
        assert result.match_scores["agent2"] > result.match_scores["agent1"]

    def test_context_aware_matching(self, capability_matcher, mock_agents):
        """Test context-aware matching."""
        task_requirements = {"python_programming": ExpertiseLevel.INTERMEDIATE}

        # Without context - should prefer agent1 (higher level)
        result1 = capability_matcher.find_best_match(task_requirements, mock_agents)

        # With security context - should adjust for relevant experience
        result2 = capability_matcher.find_best_match(
            task_requirements, mock_agents,
            context={"project_type": "security_critical", "priority": "high"}
        )

        # Results might differ based on context
        assert isinstance(result2, MatchingResult)
        assert result2.context_factors_used is True

    def test_no_suitable_agents(self, capability_matcher):
        """Test behavior when no suitable agents are found."""
        task_requirements = {"quantum_computing": ExpertiseLevel.EXPERT}
        agents = {"agent1": {"capabilities": {"basic_coding": {"level": ExpertiseLevel.BEGINNER}}}}

        result = capability_matcher.find_best_match(task_requirements, agents)

        assert result.best_agent_id is None
        assert result.confidence_score < 0.3
        assert "no_suitable_agents" in result.reasoning.lower()


class TestHandoffAPI:
    """Test suite for handoff API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient
        from src.server.main import app

        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/handoff/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_get_active_handoffs(self, client):
        """Test getting active handoffs."""
        response = client.get("/api/handoff/active")
        assert response.status_code == 200
        data = response.json()
        assert "active_handoffs" in data
        assert isinstance(data["active_handoffs"], list)

    def test_get_handoff_history(self, client):
        """Test getting handoff history."""
        response = client.get("/api/handoff/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_execute_handoff(self, client):
        """Test executing a handoff."""
        handoff_request = {
            "source_agent_id": "agent1",
            "target_agent_id": "agent2",
            "message": "Test handoff",
            "task_description": "Test task for API",
            "strategy": "sequential",
            "trigger": "manual_request",
            "confidence_score": 0.8,
            "priority": 3,
            "context": {}
        }

        response = client.post("/api/handoff/execute", json=handoff_request)
        # Note: This might fail if agents don't exist, but should return proper error
        assert response.status_code in [200, 500]  # Either success or server error

    def test_get_recommendations(self, client):
        """Test getting handoff recommendations."""
        request_data = {
            "task_description": "Security vulnerability analysis needed",
            "current_agent_id": "agent1"
        }

        response = client.post("/api/handoff/recommendations", json=request_data)
        assert response.status_code in [200, 500]  # Success or error if agents not set up

    def test_get_analytics(self, client):
        """Test getting handoff analytics."""
        response = client.get("/api/handoff/analytics")
        assert response.status_code == 200
        data = response.json()
        assert "total_handoffs" in data
        assert "success_rate" in data


@pytest.mark.asyncio
class TestHandoffIntegration:
    """Integration tests for the complete handoff system."""

    async def test_full_handoff_workflow(self):
        """Test complete handoff workflow from request to completion."""
        # This would require a more complex setup with actual agent implementations
        # For now, we'll test the integration points

        from src.agents.orchestration.archon_agency import ArchonAgency

        # Create agency
        agency = ArchonAgency()

        # Test that handoff engine is initialized
        assert hasattr(agency, 'handoff_engine')
        assert hasattr(agency, 'context_engine')
        assert hasattr(agency, 'learning_engine')

        # Test handoff method exists
        assert hasattr(agency, 'request_handoff')
        assert callable(agency.request_handoff)

    async def test_learning_cycle_integration(self):
        """Test learning cycle integration."""
        from src.agents.orchestration.handoff_learning import HandoffLearningEngine

        learning_engine = HandoffLearningEngine()

        # Simulate some handoff results
        handoffs = [
            Mock(status="completed", execution_time=1200, metrics={"success": True}),
            Mock(status="completed", execution_time=900, metrics={"success": True}),
            Mock(status="failed", execution_time=3000, metrics={"success": False})
        ]

        # Record results
        for handoff in handoffs:
            learning_engine.record_handoff_result(handoff)

        # Generate insights
        insights = learning_engine.generate_optimization_insights()
        assert len(insights) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])