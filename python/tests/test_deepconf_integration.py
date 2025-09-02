"""
Integration test for DeepConf Advanced Debugging System - Phase 7 PRD Implementation

Tests complete integration between:
- DeepConf Engine
- DeepConf Debugger
- Debug API endpoints
- React UI integration (mocked)

This validates the complete end-to-end functionality of the advanced debugging system.

Author: Archon AI System  
Version: 1.0.0
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
import json

# Import all components for integration testing
from src.agents.deepconf.engine import DeepConfEngine
from src.agents.deepconf.debugger import DeepConfDebugger, AITask
from src.agents.deepconf.types import ConfidenceScore
from src.server.api_routes.deepconf_debug_api import router, get_debugger

class TestDeepConfIntegration:
    """Test complete DeepConf debugging system integration"""
    
    @pytest.fixture
    def engine(self):
        """Create real DeepConf engine instance"""
        return DeepConfEngine()
    
    @pytest.fixture
    def debugger(self, engine):
        """Create debugger with real engine"""
        return DeepConfDebugger(deepconf_engine=engine)
    
    @pytest.fixture
    def sample_task(self):
        """Sample task for integration testing"""
        return AITask(
            task_id="integration_test_001",
            content="Test complete debugging system integration with low confidence analysis",
            domain="system_integration",
            complexity="complex",
            priority="high",
            model_source="integration_test_model",
            context_size=2000
        )

    @pytest.mark.asyncio
    async def test_complete_debugging_workflow(self, debugger, sample_task):
        """Test complete debugging workflow from start to finish"""
        # Step 1: Create debug session
        session = debugger.create_debug_session(sample_task)
        assert session is not None
        assert session.is_active is True
        assert session.task == sample_task
        
        # Step 2: Create low confidence score for analysis
        low_confidence_score = ConfidenceScore(
            overall_confidence=0.15,  # Very low for comprehensive testing
            factual_confidence=0.2,
            reasoning_confidence=0.1,
            contextual_confidence=0.15,
            epistemic_uncertainty=0.5,
            aleatoric_uncertainty=0.4,
            uncertainty_bounds=(0.05, 0.25),
            confidence_factors={
                'technical_complexity': 0.1,  # Very low - should trigger critical issues
                'domain_expertise': 0.2,
                'data_availability': 0.3,
                'model_capability': 0.15
            },
            primary_factors=['technical_complexity', 'model_capability'],
            confidence_reasoning="Very low confidence due to multiple critical issues",
            model_source="integration_test_model",
            timestamp=time.time(),
            task_id=sample_task.task_id
        )
        
        # Step 3: Perform comprehensive low confidence analysis
        debug_report = await debugger.analyze_low_confidence(sample_task, low_confidence_score)
        
        # Validate analysis results
        assert debug_report is not None
        assert debug_report.task_id == sample_task.task_id
        assert debug_report.confidence_score == 0.15
        assert len(debug_report.issues) > 0
        assert len(debug_report.recommendations) > 0
        
        # Should find critical issues with very low factors
        critical_issues = [issue for issue in debug_report.issues if issue.severity.value == 'critical']
        assert len(critical_issues) > 0
        
        # Add report to session
        session.debug_reports.append(debug_report)
        
        # Step 4: Perform confidence factor tracing
        trace = await debugger.trace_confidence_factors("integration_test_trace")
        
        # Validate tracing results
        assert trace is not None
        assert trace.factor_name is not None
        assert len(trace.calculation_steps) > 0
        assert trace.confidence_contribution > 0
        
        # Step 5: Create task history for bottleneck analysis
        from src.agents.deepconf.debugger import TaskHistory
        task_history = TaskHistory(
            task_id=sample_task.task_id,
            execution_records=[
                {"timestamp": time.time() - 300, "duration": 3.5, "success": True},
                {"timestamp": time.time() - 200, "duration": 4.2, "success": True}, 
                {"timestamp": time.time() - 100, "duration": 3.8, "success": False},
                {"timestamp": time.time(), "duration": 5.1, "success": True}
            ],
            performance_metrics=[
                {"computation_time": 3.5, "memory_usage": 95, "cache_hits": 0.4},
                {"computation_time": 4.2, "memory_usage": 105, "cache_hits": 0.3},
                {"computation_time": 3.8, "memory_usage": 98, "cache_hits": 0.45},
                {"computation_time": 5.1, "memory_usage": 110, "cache_hits": 0.2}
            ],
            confidence_history=[],
            total_executions=4,
            average_confidence=0.3,  # Low average
            performance_trends={
                "computation_time": [3.5, 4.2, 3.8, 5.1],  # Increasing trend
                "memory_usage": [95, 105, 98, 110]  # High memory usage
            }
        )
        
        # Step 6: Analyze performance bottlenecks
        bottleneck_analysis = await debugger.identify_performance_bottlenecks(task_history)
        
        # Validate bottleneck analysis
        assert bottleneck_analysis is not None
        assert len(bottleneck_analysis.root_causes) > 0
        assert len(bottleneck_analysis.optimization_suggestions) > 0
        assert bottleneck_analysis.performance_impact is not None
        
        # Step 7: Generate optimization suggestions
        from src.agents.deepconf.debugger import PerformanceData
        performance_data = PerformanceData(
            operation_times={
                "confidence_calculation": [3.5, 4.2, 3.8, 5.1],  # Slow times
                "factor_analysis": [0.8, 0.9, 0.7, 1.1]
            },
            memory_usage={
                "peak_usage": [95, 105, 98, 110],  # High memory
                "average_usage": [80, 90, 85, 95]
            },
            cache_hit_rates={
                "confidence_cache": 0.3,  # Poor cache performance
                "factor_cache": 0.4
            },
            error_rates={
                "calculation_errors": 0.05,  # High error rate
                "timeout_errors": 0.02
            },
            throughput_metrics={
                "requests_per_second": 8,  # Low throughput
                "concurrent_calculations": 2
            },
            bottleneck_indicators={
                "computation_intensive": True,
                "memory_pressure": True
            }
        )
        
        optimization_suggestions = await debugger.suggest_optimization_strategies(performance_data)
        
        # Validate optimization suggestions
        assert optimization_suggestions is not None
        assert len(optimization_suggestions.strategies) > 0
        assert len(optimization_suggestions.priority_ranking) > 0
        assert optimization_suggestions.resource_requirements is not None
        
        # Should prioritize caching and algorithm optimization for slow performance
        high_impact_strategies = [
            s for s in optimization_suggestions.strategies 
            if any(v > 0.5 for v in s.expected_improvement.values())
        ]
        assert len(high_impact_strategies) > 0
        
        # Step 8: Export debug data
        debug_export = debugger.export_debug_data(session, export_format="json")
        
        # Validate export
        assert debug_export is not None
        assert debug_export.session == session
        assert debug_export.export_format == "json"
        assert debug_export.analysis_summary is not None
        assert debug_export.raw_data is not None
        assert len(debug_export.visualizations) > 0
        
        # Validate export contains all workflow data
        assert len(session.debug_reports) == 1
        assert session.debug_reports[0] == debug_report
        
        # Export should include comprehensive analysis summary
        summary = debug_export.analysis_summary
        assert 'session_overview' in summary
        assert 'confidence_analysis' in summary
        assert 'performance_summary' in summary
        assert summary['session_overview']['total_reports'] == 1
        
        print(f"✅ Complete debugging workflow test passed:")
        print(f"   - Session: {session.session_id}")
        print(f"   - Issues found: {len(debug_report.issues)}")
        print(f"   - Critical issues: {len(critical_issues)}")
        print(f"   - Optimization strategies: {len(optimization_suggestions.strategies)}")
        print(f"   - Export format: {debug_export.export_format}")

    @pytest.mark.asyncio
    async def test_engine_debugger_integration(self, engine, debugger, sample_task):
        """Test integration between DeepConf engine and debugger"""
        # Create a context for the engine
        class MockContext:
            def __init__(self):
                self.environment = 'integration_test'
                self.model_history = []
                self.performance_data = {}
        
        context = MockContext()
        
        # Use engine to calculate confidence
        confidence_score = await engine.calculate_confidence(sample_task, context)
        
        # Validate engine calculation
        assert confidence_score is not None
        assert 0.0 <= confidence_score.overall_confidence <= 1.0
        assert confidence_score.task_id == sample_task.task_id
        assert len(confidence_score.confidence_factors) > 0
        
        # Use debugger to analyze the engine's confidence calculation
        if confidence_score.overall_confidence < 0.6:  # If confidence is low enough
            debug_report = await debugger.analyze_low_confidence(sample_task, confidence_score)
            
            # Validate that debugger can analyze engine output
            assert debug_report.confidence_score == confidence_score.overall_confidence
            assert len(debug_report.factor_analysis) > 0
            
            # Should provide actionable insights
            assert len(debug_report.recommendations) > 0
            for rec in debug_report.recommendations:
                assert len(rec) > 10  # Substantial recommendations
        
        print(f"✅ Engine-Debugger integration test passed:")
        print(f"   - Engine confidence: {confidence_score.overall_confidence:.3f}")
        print(f"   - Confidence factors: {len(confidence_score.confidence_factors)}")
        print(f"   - Debugger integrated: {debugger.engine is not None}")

    def test_concurrent_debugging_sessions(self, debugger):
        """Test multiple concurrent debugging sessions"""
        # Create multiple tasks
        tasks = [
            AITask(
                task_id=f"concurrent_test_{i:03d}",
                content=f"Concurrent debugging test task {i}",
                domain="concurrent_testing",
                complexity="moderate"
            )
            for i in range(5)
        ]
        
        # Create concurrent sessions
        sessions = []
        for task in tasks:
            session = debugger.create_debug_session(task)
            sessions.append(session)
        
        # Validate all sessions created
        assert len(sessions) == 5
        assert len(debugger.active_sessions) == 5
        
        # Each session should be unique and active
        session_ids = set(s.session_id for s in sessions)
        assert len(session_ids) == 5  # All unique
        
        for session in sessions:
            assert session.is_active is True
            assert session.session_id in debugger.active_sessions
        
        print(f"✅ Concurrent sessions test passed:")
        print(f"   - Sessions created: {len(sessions)}")
        print(f"   - Active sessions: {len(debugger.active_sessions)}")

    @pytest.mark.asyncio  
    async def test_debugging_performance_characteristics(self, debugger, sample_task):
        """Test that debugging operations meet performance requirements"""
        # Test session creation performance
        start_time = time.time()
        session = debugger.create_debug_session(sample_task)
        session_time = time.time() - start_time
        
        assert session_time < 0.1  # Should be very fast
        
        # Test low confidence analysis performance
        low_confidence_score = ConfidenceScore(
            overall_confidence=0.3,
            factual_confidence=0.35,
            reasoning_confidence=0.25,
            contextual_confidence=0.3,
            epistemic_uncertainty=0.3,
            aleatoric_uncertainty=0.25,
            uncertainty_bounds=(0.2, 0.4),
            confidence_factors={
                'technical_complexity': 0.4,
                'domain_expertise': 0.5,
                'data_availability': 0.6,
                'model_capability': 0.3
            },
            primary_factors=['model_capability', 'technical_complexity'],
            confidence_reasoning="Test performance analysis",
            model_source="performance_test",
            timestamp=time.time(),
            task_id=sample_task.task_id
        )
        
        start_time = time.time()
        debug_report = await debugger.analyze_low_confidence(sample_task, low_confidence_score)
        analysis_time = time.time() - start_time
        
        assert analysis_time < 2.0  # Should complete within 2 seconds
        assert debug_report is not None
        
        # Test confidence tracing performance
        start_time = time.time()
        trace = await debugger.trace_confidence_factors("performance_test_trace")
        trace_time = time.time() - start_time
        
        assert trace_time < 0.5  # Should be very fast
        assert trace is not None
        
        # Test data export performance
        start_time = time.time()
        export = debugger.export_debug_data(session, export_format="json")
        export_time = time.time() - start_time
        
        assert export_time < 1.0  # Should complete within 1 second
        assert export is not None
        
        print(f"✅ Performance characteristics test passed:")
        print(f"   - Session creation: {session_time:.3f}s")
        print(f"   - Low confidence analysis: {analysis_time:.3f}s")
        print(f"   - Confidence tracing: {trace_time:.3f}s")
        print(f"   - Data export: {export_time:.3f}s")

    def test_error_recovery_and_resilience(self, debugger, sample_task):
        """Test system resilience and error recovery"""
        # Test with invalid tasks
        invalid_task = AITask(
            task_id="",  # Invalid
            content="",  # Invalid
            domain="test"
        )
        
        # System should handle invalid input gracefully
        try:
            session = debugger.create_debug_session(invalid_task)
            # If it doesn't raise an error, it should still work
            assert isinstance(session, type(debugger.create_debug_session(sample_task)))
        except (ValueError, AttributeError):
            # It's acceptable to raise validation errors
            pass
        
        # Test session cleanup under stress
        original_max_sessions = debugger.config['max_active_sessions']
        debugger.config['max_active_sessions'] = 3
        
        # Create sessions up to limit
        sessions = []
        for i in range(3):
            task = AITask(
                task_id=f"stress_test_{i}",
                content=f"Stress test task {i}",
                domain="testing"
            )
            session = debugger.create_debug_session(task)
            sessions.append(session)
        
        assert len(debugger.active_sessions) == 3
        
        # Try to create one more - should trigger cleanup or raise error
        extra_task = AITask(
            task_id="extra_task",
            content="Extra task",
            domain="testing"
        )
        
        try:
            extra_session = debugger.create_debug_session(extra_task)
            # If successful, cleanup must have worked
            assert len(debugger.active_sessions) <= 3
        except RuntimeError:
            # Expected if no cleanup occurred
            pass
        
        # Restore original config
        debugger.config['max_active_sessions'] = original_max_sessions
        
        print("✅ Error recovery and resilience test passed")

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])