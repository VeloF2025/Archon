"""
Comprehensive tests for DeepConf Advanced Debugging System - Phase 7 PRD Implementation

Tests all PRD-required functionality:
- DeepConfDebugger class with all methods
- Low confidence analysis with actionable insights  
- Confidence factor tracing with detailed breakdowns
- Performance bottleneck detection with root cause analysis
- Optimization suggestions with implementation guidance
- Debug session management with state persistence
- Data export functionality for external analysis

Author: Archon AI System
Version: 1.0.0
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
import tempfile
import os

# Import the debugging system
from src.agents.deepconf.debugger import (
    DeepConfDebugger,
    AITask,
    DebugSession,
    DebugReport,
    ConfidenceTrace,
    BottleneckAnalysis,
    OptimizationSuggestions,
    PerformanceData,
    TaskHistory,
    DebugExport,
    DebugSeverity,
    PerformanceCategory,
    DebugIssue
)
from src.agents.deepconf.engine import DeepConfEngine
from src.agents.deepconf.types import ConfidenceScore

class TestDeepConfDebugger:
    """Comprehensive test suite for DeepConf debugger"""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock DeepConf engine"""
        engine = Mock(spec=DeepConfEngine)
        engine.config = {
            'confidence_threshold': 0.7,
            'confidence_factors_weights': {
                'technical_complexity': 0.25,
                'domain_expertise': 0.20,
                'data_availability': 0.20,
                'model_capability': 0.20,
                'historical_performance': 0.10,
                'context_richness': 0.05
            }
        }
        return engine
    
    @pytest.fixture
    def debugger(self, mock_engine):
        """Create debugger instance with mock engine"""
        config = {
            'max_active_sessions': 5,
            'session_timeout': 3600,
            'confidence_threshold_critical': 0.3,
            'confidence_threshold_warning': 0.6,
            'bottleneck_detection_threshold': 2.0,
            'trace_depth': 5,
            'export_formats': ['json', 'csv'],
            'max_optimization_strategies': 3
        }
        return DeepConfDebugger(deepconf_engine=mock_engine, config=config)
    
    @pytest.fixture
    def sample_task(self):
        """Create sample AI task for testing"""
        return AITask(
            task_id="test_task_001",
            content="Test task for debugging analysis",
            domain="testing",
            complexity="moderate",
            priority="normal",
            model_source="test_model",
            context_size=1000
        )
    
    @pytest.fixture
    def low_confidence_score(self):
        """Create low confidence score for testing"""
        return ConfidenceScore(
            overall_confidence=0.25,
            factual_confidence=0.3,
            reasoning_confidence=0.2,
            contextual_confidence=0.25,
            epistemic_uncertainty=0.4,
            aleatoric_uncertainty=0.3,
            uncertainty_bounds=(0.15, 0.35),
            confidence_factors={
                'technical_complexity': 0.3,
                'domain_expertise': 0.2,
                'data_availability': 0.4,
                'model_capability': 0.25
            },
            primary_factors=['data_availability', 'domain_expertise'],
            confidence_reasoning="Low confidence due to multiple issues",
            model_source="test_model",
            timestamp=time.time(),
            task_id="test_task_001"
        )
    
    @pytest.fixture
    def sample_task_history(self):
        """Create sample task history for bottleneck testing"""
        return TaskHistory(
            task_id="test_task_001",
            execution_records=[
                {"timestamp": time.time(), "duration": 2.5, "success": True},
                {"timestamp": time.time(), "duration": 3.2, "success": True},
                {"timestamp": time.time(), "duration": 2.8, "success": False}
            ],
            performance_metrics=[
                {"computation_time": 2.5, "memory_usage": 85, "cache_hits": 0.6},
                {"computation_time": 3.2, "memory_usage": 92, "cache_hits": 0.5},
                {"computation_time": 2.8, "memory_usage": 88, "cache_hits": 0.65}
            ],
            confidence_history=[],
            total_executions=3,
            average_confidence=0.65,
            performance_trends={
                "computation_time": [2.5, 3.2, 2.8],
                "memory_usage": [85, 92, 88]
            }
        )
    
    @pytest.fixture
    def sample_performance_data(self):
        """Create sample performance data"""
        return PerformanceData(
            operation_times={
                "confidence_calculation": [0.15, 0.18, 0.16, 0.20, 0.17],
                "factor_analysis": [0.08, 0.09, 0.07, 0.10, 0.08]
            },
            memory_usage={
                "peak_usage": [45, 48, 52, 47, 50],
                "average_usage": [35, 37, 40, 36, 38]
            },
            cache_hit_rates={
                "confidence_cache": 0.65,
                "factor_cache": 0.72
            },
            error_rates={
                "calculation_errors": 0.02,
                "timeout_errors": 0.01
            },
            throughput_metrics={
                "requests_per_second": 25,
                "concurrent_calculations": 5
            },
            bottleneck_indicators={
                "computation_intensive": True,
                "memory_pressure": False
            }
        )

class TestDebuggerInitialization(TestDeepConfDebugger):
    """Test debugger initialization and configuration"""
    
    def test_debugger_initialization(self, debugger, mock_engine):
        """Test debugger initializes correctly with engine"""
        assert debugger.engine == mock_engine
        assert debugger.config['max_active_sessions'] == 5
        assert len(debugger.active_sessions) == 0
        assert len(debugger.session_history) == 0
        assert debugger._factor_analyzer is not None
        assert debugger._performance_analyzer is not None
        assert debugger._optimization_engine is not None
    
    def test_debugger_initialization_without_engine(self):
        """Test debugger initializes without engine"""
        debugger = DeepConfDebugger()
        assert debugger.engine is None
        assert debugger.config is not None
        assert 'max_active_sessions' in debugger.config

class TestLowConfidenceAnalysis(TestDeepConfDebugger):
    """Test low confidence analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_low_confidence_success(self, debugger, sample_task, low_confidence_score):
        """Test successful low confidence analysis"""
        # Act
        report = await debugger.analyze_low_confidence(sample_task, low_confidence_score)
        
        # Assert
        assert isinstance(report, DebugReport)
        assert report.task_id == sample_task.task_id
        assert report.confidence_score == low_confidence_score.overall_confidence
        assert len(report.issues) > 0  # Should find issues with low confidence
        assert len(report.recommendations) > 0
        assert report.severity_summary is not None
        assert report.factor_analysis is not None
        assert report.confidence_projection is not None
        
        # Check that issues are properly categorized
        severities = [issue.severity for issue in report.issues]
        assert any(severity in [DebugSeverity.CRITICAL, DebugSeverity.HIGH] for severity in severities)
    
    @pytest.mark.asyncio
    async def test_analyze_low_confidence_invalid_score(self, debugger, sample_task):
        """Test analysis with non-low confidence score raises error"""
        high_confidence_score = ConfidenceScore(
            overall_confidence=0.85,  # High confidence
            factual_confidence=0.9,
            reasoning_confidence=0.8,
            contextual_confidence=0.85,
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.1,
            uncertainty_bounds=(0.8, 0.9),
            confidence_factors={
                'technical_complexity': 0.8,
                'domain_expertise': 0.9,
                'data_availability': 0.85,
                'model_capability': 0.8
            },
            primary_factors=['domain_expertise', 'data_availability'],
            confidence_reasoning="High confidence score",
            model_source="test_model",
            timestamp=time.time(),
            task_id="test_task_001"
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="not considered low"):
            await debugger.analyze_low_confidence(sample_task, high_confidence_score)
    
    @pytest.mark.asyncio
    async def test_analyze_low_confidence_issue_detection(self, debugger, sample_task, low_confidence_score):
        """Test that analysis detects specific types of issues"""
        # Act
        report = await debugger.analyze_low_confidence(sample_task, low_confidence_score)
        
        # Assert - check for different issue categories
        issue_categories = [issue.category for issue in report.issues]
        assert 'confidence_factor' in issue_categories  # Should detect factor issues
        
        # Check for critical issues with very low factors
        critical_issues = [issue for issue in report.issues if issue.severity == DebugSeverity.CRITICAL]
        assert len(critical_issues) > 0
        
        # Check that recommendations are actionable
        for rec in report.recommendations:
            assert len(rec) > 10  # Recommendations should be substantial
            assert any(word in rec.lower() for word in ['improve', 'increase', 'consider', 'implement', 'review'])

class TestConfidenceTracing(TestDeepConfDebugger):
    """Test confidence factor tracing functionality"""
    
    @pytest.mark.asyncio
    async def test_trace_confidence_factors_success(self, debugger):
        """Test successful confidence factor tracing"""
        # Act
        trace = await debugger.trace_confidence_factors("test_confidence_001")
        
        # Assert
        assert isinstance(trace, ConfidenceTrace)
        assert trace.factor_name is not None
        assert 0.0 <= trace.raw_score <= 1.0
        assert 0.0 <= trace.weighted_score <= 1.0
        assert 0.0 <= trace.weight <= 1.0
        assert len(trace.calculation_steps) > 0
        assert len(trace.dependencies) >= 0
        assert trace.computation_time > 0
        assert 0.0 <= trace.confidence_contribution <= 1.0
        assert trace.trace_id is not None
        assert trace.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_trace_confidence_calculation_steps(self, debugger):
        """Test that tracing includes detailed calculation steps"""
        # Act
        trace = await debugger.trace_confidence_factors("test_confidence_002")
        
        # Assert calculation steps
        assert len(trace.calculation_steps) > 0
        for step in trace.calculation_steps:
            assert step.get('step') is not None
            assert step.get('operation') is not None
            assert step.get('calculation') is not None
            assert step.get('result') is not None
            assert step.get('computation_time') is not None
            assert isinstance(step['result'], (int, float))
            assert step['computation_time'] >= 0
    
    @pytest.mark.asyncio
    async def test_trace_confidence_dependencies(self, debugger):
        """Test that tracing identifies factor dependencies"""
        # Act
        trace = await debugger.trace_confidence_factors("test_confidence_003")
        
        # Assert dependencies
        assert isinstance(trace.dependencies, list)
        for dependency in trace.dependencies:
            assert isinstance(dependency, str)
            assert len(dependency) > 0

class TestPerformanceBottleneckAnalysis(TestDeepConfDebugger):
    """Test performance bottleneck detection and analysis"""
    
    @pytest.mark.asyncio
    async def test_identify_performance_bottlenecks_success(self, debugger, sample_task_history):
        """Test successful bottleneck identification"""
        # Act
        analysis = await debugger.identify_performance_bottlenecks(sample_task_history)
        
        # Assert
        assert isinstance(analysis, BottleneckAnalysis)
        assert analysis.bottleneck_id is not None
        assert isinstance(analysis.category, PerformanceCategory)
        assert isinstance(analysis.severity, DebugSeverity)
        assert len(analysis.description) > 0
        assert len(analysis.affected_operations) > 0
        assert len(analysis.root_causes) > 0
        assert len(analysis.optimization_suggestions) > 0
        assert analysis.performance_impact is not None
        assert analysis.estimated_improvement is not None
        assert analysis.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_bottleneck_analysis_categories(self, debugger, sample_task_history):
        """Test that bottleneck analysis categorizes issues correctly"""
        # Act
        analysis = await debugger.identify_performance_bottlenecks(sample_task_history)
        
        # Assert category is valid
        valid_categories = [cat.value for cat in PerformanceCategory]
        assert analysis.category.value in valid_categories
        
        # Assert optimization suggestions are relevant to category
        suggestions = analysis.optimization_suggestions
        assert len(suggestions) > 0
        
        if analysis.category == PerformanceCategory.COMPUTATION:
            assert any('cache' in suggestion.lower() or 'algorithm' in suggestion.lower() or 'parallel' in suggestion.lower() 
                     for suggestion in suggestions)
        elif analysis.category == PerformanceCategory.MEMORY:
            assert any('memory' in suggestion.lower() or 'pool' in suggestion.lower() or 'gc' in suggestion.lower()
                     for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_bottleneck_analysis_empty_history(self, debugger):
        """Test bottleneck analysis with minimal history"""
        minimal_history = TaskHistory(
            task_id="minimal_task",
            execution_records=[],
            performance_metrics=[],
            confidence_history=[],
            total_executions=0,
            average_confidence=0.5,
            performance_trends={}
        )
        
        # Act
        analysis = await debugger.identify_performance_bottlenecks(minimal_history)
        
        # Assert - should still provide analysis even with minimal data
        assert isinstance(analysis, BottleneckAnalysis)
        assert len(analysis.optimization_suggestions) > 0

class TestOptimizationSuggestions(TestDeepConfDebugger):
    """Test AI-powered optimization suggestions"""
    
    @pytest.mark.asyncio
    async def test_suggest_optimization_strategies_success(self, debugger, sample_performance_data):
        """Test successful optimization strategy generation"""
        # Act
        suggestions = await debugger.suggest_optimization_strategies(sample_performance_data)
        
        # Assert
        assert isinstance(suggestions, OptimizationSuggestions)
        assert len(suggestions.strategies) > 0
        assert len(suggestions.priority_ranking) > 0
        assert len(suggestions.implementation_roadmap) > 0
        assert suggestions.resource_requirements is not None
        assert suggestions.risk_assessment is not None
        assert len(suggestions.success_metrics) > 0
        
        # Check that strategies are well-formed
        for strategy in suggestions.strategies:
            assert strategy.strategy_id is not None
            assert len(strategy.title) > 0
            assert len(strategy.description) > 0
            assert strategy.implementation_complexity in ['low', 'medium', 'high']
            assert len(strategy.expected_improvement) > 0
            assert len(strategy.implementation_steps) > 0
            assert 0.0 <= strategy.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_optimization_strategies_prioritization(self, debugger, sample_performance_data):
        """Test that optimization strategies are properly prioritized"""
        # Act
        suggestions = await debugger.suggest_optimization_strategies(sample_performance_data)
        
        # Assert prioritization
        assert len(suggestions.priority_ranking) == len(suggestions.strategies)
        
        # Check that priority ranking contains all strategy IDs
        strategy_ids = set(s.strategy_id for s in suggestions.strategies)
        priority_ids = set(suggestions.priority_ranking)
        assert strategy_ids == priority_ids
    
    @pytest.mark.asyncio
    async def test_optimization_strategies_implementation_roadmap(self, debugger, sample_performance_data):
        """Test that implementation roadmap is realistic"""
        # Act
        suggestions = await debugger.suggest_optimization_strategies(sample_performance_data)
        
        # Assert roadmap structure
        roadmap = suggestions.implementation_roadmap
        assert len(roadmap) > 0
        
        for phase in roadmap:
            assert 'phase' in phase
            assert 'strategy_id' in phase
            assert 'title' in phase
            assert 'estimated_duration' in phase
            assert phase['estimated_duration'] in ['hours', 'days', 'weeks']

class TestDebugSessionManagement(TestDeepConfDebugger):
    """Test debug session creation and management"""
    
    def test_create_debug_session_success(self, debugger, sample_task):
        """Test successful debug session creation"""
        # Act
        session = debugger.create_debug_session(sample_task)
        
        # Assert
        assert isinstance(session, DebugSession)
        assert session.session_id is not None
        assert session.task == sample_task
        assert session.start_time is not None
        assert session.is_active is True
        assert session.end_time is None
        assert len(session.debug_reports) == 0
        assert len(session.performance_snapshots) == 0
        assert len(session.debug_actions) == 0
        
        # Check that session is tracked
        assert session.session_id in debugger.active_sessions
        assert debugger.active_sessions[session.session_id] == session
    
    def test_create_debug_session_max_sessions_limit(self, debugger, sample_task):
        """Test that session creation respects maximum active sessions limit"""
        # Create maximum number of sessions
        max_sessions = debugger.config['max_active_sessions']
        sessions = []
        
        for i in range(max_sessions):
            task = AITask(
                task_id=f"test_task_{i:03d}",
                content=f"Test task {i}",
                domain="testing"
            )
            session = debugger.create_debug_session(task)
            sessions.append(session)
        
        # Assert all sessions created
        assert len(debugger.active_sessions) == max_sessions
        
        # Try to create one more session - should raise error
        extra_task = AITask(
            task_id="extra_task",
            content="Extra task",
            domain="testing"
        )
        
        with pytest.raises(RuntimeError, match="Maximum active sessions"):
            debugger.create_debug_session(extra_task)
    
    def test_debug_session_cleanup(self, debugger, sample_task):
        """Test automatic cleanup of expired sessions"""
        # Create session
        session = debugger.create_debug_session(sample_task)
        
        # Manually set session as expired
        session.start_time = datetime.now() - timedelta(seconds=debugger.config['session_timeout'] + 100)
        session.is_active = True
        
        # Trigger cleanup by trying to create another session when at limit
        debugger.config['max_active_sessions'] = 1
        
        extra_task = AITask(task_id="extra", content="Extra", domain="testing")
        new_session = debugger.create_debug_session(extra_task)
        
        # Assert old session was cleaned up
        assert session.session_id not in debugger.active_sessions
        assert new_session.session_id in debugger.active_sessions
        assert len(debugger.session_history) >= 1

class TestDataExport(TestDeepConfDebugger):
    """Test comprehensive data export functionality"""
    
    def test_export_debug_data_json_success(self, debugger, sample_task):
        """Test successful JSON data export"""
        # Setup - create session with some data
        session = debugger.create_debug_session(sample_task)
        
        # Add some mock data to session
        session.debug_actions.append({
            'type': 'analyze',
            'timestamp': datetime.now(),
            'parameters': {'test': True}
        })
        session.performance_snapshots.append({
            'timestamp': time.time(),
            'memory_usage': 45.5,
            'computation_time': 0.15
        })
        
        # Act
        export = debugger.export_debug_data(session, export_format="json")
        
        # Assert
        assert isinstance(export, DebugExport)
        assert export.export_id is not None
        assert export.session == session
        assert export.export_format == "json"
        assert export.export_timestamp is not None
        assert export.analysis_summary is not None
        assert export.raw_data is not None
        assert len(export.visualizations) > 0
        assert export.metadata is not None
        
        # Check analysis summary structure
        summary = export.analysis_summary
        assert 'session_overview' in summary
        assert 'confidence_analysis' in summary
        assert 'performance_summary' in summary
        
        # Check that raw data is properly structured
        raw_data = export.raw_data
        assert 'session' in raw_data
        assert 'performance_data' in raw_data
        assert 'debug_issues' in raw_data
    
    def test_export_debug_data_invalid_format(self, debugger, sample_task):
        """Test export with invalid format raises error"""
        session = debugger.create_debug_session(sample_task)
        
        with pytest.raises(ValueError, match="not supported"):
            debugger.export_debug_data(session, export_format="invalid_format")
    
    def test_export_debug_data_csv_format(self, debugger, sample_task):
        """Test CSV export format (basic validation)"""
        session = debugger.create_debug_session(sample_task)
        
        # Act
        export = debugger.export_debug_data(session, export_format="csv")
        
        # Assert
        assert export.export_format == "csv"
        assert export.raw_data is not None

class TestDebuggerIntegration(TestDeepConfDebugger):
    """Test integration between different debugger components"""
    
    @pytest.mark.asyncio
    async def test_full_debugging_workflow(self, debugger, sample_task, low_confidence_score, sample_task_history, sample_performance_data):
        """Test complete debugging workflow from session creation to data export"""
        # Step 1: Create debug session
        session = debugger.create_debug_session(sample_task)
        assert session is not None
        
        # Step 2: Perform low confidence analysis
        debug_report = await debugger.analyze_low_confidence(sample_task, low_confidence_score)
        session.debug_reports.append(debug_report)
        
        # Step 3: Trace confidence factors
        trace = await debugger.trace_confidence_factors("test_trace")
        
        # Step 4: Analyze performance bottlenecks
        bottleneck_analysis = await debugger.identify_performance_bottlenecks(sample_task_history)
        
        # Step 5: Generate optimization suggestions
        optimization_suggestions = await debugger.suggest_optimization_strategies(sample_performance_data)
        
        # Step 6: Export debug data
        export = debugger.export_debug_data(session, export_format="json")
        
        # Assert complete workflow
        assert len(session.debug_reports) == 1
        assert isinstance(trace, ConfidenceTrace)
        assert isinstance(bottleneck_analysis, BottleneckAnalysis)
        assert isinstance(optimization_suggestions, OptimizationSuggestions)
        assert isinstance(export, DebugExport)
        
        # Check that all components worked together
        assert debug_report.task_id == sample_task.task_id
        assert trace.factor_name is not None
        assert len(bottleneck_analysis.optimization_suggestions) > 0
        assert len(optimization_suggestions.strategies) > 0
        assert export.session == session
    
    @pytest.mark.asyncio
    async def test_concurrent_debug_sessions(self, debugger):
        """Test handling of multiple concurrent debug sessions"""
        # Create multiple concurrent sessions
        tasks = [
            AITask(task_id=f"concurrent_{i}", content=f"Task {i}", domain="testing")
            for i in range(3)
        ]
        
        sessions = [debugger.create_debug_session(task) for task in tasks]
        
        # Perform analysis on all sessions concurrently
        async def analyze_session(session):
            low_confidence = ConfidenceScore(
                overall_confidence=0.2,
                factual_confidence=0.3,
                reasoning_confidence=0.1,
                contextual_confidence=0.2,
                epistemic_uncertainty=0.4,
                aleatoric_uncertainty=0.3,
                uncertainty_bounds=(0.1, 0.3),
                confidence_factors={'technical_complexity': 0.2},
                primary_factors=['technical_complexity'],
                confidence_reasoning="Test low confidence",
                model_source="test",
                timestamp=time.time(),
                task_id=session.task.task_id
            )
            return await debugger.analyze_low_confidence(session.task, low_confidence)
        
        # Run concurrent analysis
        reports = await asyncio.gather(*[analyze_session(session) for session in sessions])
        
        # Assert all analyses completed
        assert len(reports) == 3
        for i, report in enumerate(reports):
            assert report.task_id == f"concurrent_{i}"
            assert len(report.issues) > 0

class TestDebuggerErrorHandling(TestDeepConfDebugger):
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_analyze_low_confidence_with_invalid_task(self, debugger, low_confidence_score):
        """Test analysis with invalid task raises appropriate error"""
        invalid_task = AITask(
            task_id="",  # Invalid empty task_id
            content="",  # Invalid empty content
            domain="testing"
        )
        
        # Should handle invalid task gracefully or raise appropriate error
        # Implementation may validate task or handle gracefully
        try:
            report = await debugger.analyze_low_confidence(invalid_task, low_confidence_score)
            # If it doesn't raise an error, it should still work
            assert isinstance(report, DebugReport)
        except (ValueError, AttributeError):
            # It's acceptable to raise validation errors
            pass
    
    @pytest.mark.asyncio
    async def test_trace_confidence_factors_empty_id(self, debugger):
        """Test tracing with empty confidence ID"""
        # Should handle empty ID gracefully
        trace = await debugger.trace_confidence_factors("")
        assert isinstance(trace, ConfidenceTrace)
    
    def test_export_nonexistent_session(self, debugger, sample_task):
        """Test export with session that doesn't exist in active sessions"""
        # Create session but remove it from active sessions
        session = debugger.create_debug_session(sample_task)
        session_id = session.session_id
        del debugger.active_sessions[session_id]
        
        # Add to history to simulate completed session
        debugger.session_history.append(session)
        
        # Should be able to export historical sessions
        export = debugger.export_debug_data(session, export_format="json")
        assert isinstance(export, DebugExport)

class TestDebuggerPerformance(TestDeepConfDebugger):
    """Test debugger performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_low_confidence_analysis_performance(self, debugger, sample_task, low_confidence_score):
        """Test that low confidence analysis completes within reasonable time"""
        start_time = time.time()
        
        report = await debugger.analyze_low_confidence(sample_task, low_confidence_score)
        
        analysis_time = time.time() - start_time
        
        # Should complete within 5 seconds for test scenario
        assert analysis_time < 5.0
        assert isinstance(report, DebugReport)
    
    @pytest.mark.asyncio
    async def test_confidence_tracing_performance(self, debugger):
        """Test that confidence tracing completes quickly"""
        start_time = time.time()
        
        trace = await debugger.trace_confidence_factors("performance_test")
        
        trace_time = time.time() - start_time
        
        # Should complete within 1 second for test scenario  
        assert trace_time < 1.0
        assert isinstance(trace, ConfidenceTrace)
    
    def test_session_creation_performance(self, debugger):
        """Test that session creation is fast"""
        tasks = [
            AITask(task_id=f"perf_test_{i}", content=f"Task {i}", domain="testing")
            for i in range(10)
        ]
        
        start_time = time.time()
        sessions = [debugger.create_debug_session(task) for task in tasks]
        creation_time = time.time() - start_time
        
        # Should create 10 sessions in under 1 second
        assert creation_time < 1.0
        assert len(sessions) == 10
        assert all(isinstance(session, DebugSession) for session in sessions)

# Integration test fixtures for API testing (if needed)
@pytest.fixture
def client():
    """Create test client for API testing"""
    # This would be implemented if we want to test the API endpoints directly
    pass

@pytest.mark.asyncio
async def test_debugger_memory_usage():
    """Test that debugger doesn't leak memory during extended use"""
    debugger = DeepConfDebugger()
    
    # Create and destroy many sessions to test memory behavior
    for i in range(100):
        task = AITask(
            task_id=f"memory_test_{i}",
            content=f"Memory test task {i}",
            domain="testing"
        )
        session = debugger.create_debug_session(task)
        
        # Manually expire session to trigger cleanup
        if i > 20:  # Keep some sessions to trigger cleanup
            old_session_id = f"memory_test_{i-20}"
            if old_session_id in debugger.active_sessions:
                old_session = debugger.active_sessions[old_session_id]
                old_session.start_time = datetime.now() - timedelta(hours=2)
                debugger._cleanup_expired_sessions()
    
    # Should maintain reasonable number of active sessions due to cleanup
    assert len(debugger.active_sessions) <= debugger.config['max_active_sessions']

if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "--tb=short"])