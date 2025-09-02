"""
E2E Tests for SCWT Metrics Dashboard with Confidence Visualization

Tests the real-time metrics dashboard as specified in Phase 7 PRD.
All tests follow TDD Red Phase - they will fail until implementation is complete.

PRD Requirements Tested:
- Real-time confidence metrics visualization with <2s latency
- Interactive debugging tools with drill-down capabilities
- WebSocket connections for live metrics streaming
- Performance optimization reporting and trend analysis
- Quality assurance metrics with confidence calibration charts

Dashboard Features Tested:
- Confidence overview with system-wide summary
- Performance optimization metrics (token savings, cost reduction)
- Quality assurance trends (hallucination reduction, precision)
- Debugging interface with confidence factor analysis
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import websockets
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Import test fixtures and helpers
from ...conftest import (
    tdd_red_phase, requires_implementation, performance_critical, dgts_validated,
    assert_response_time_target, TEST_CONFIG
)


class TestSCWTDashboardCore:
    """Test core SCWT dashboard functionality"""
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    @performance_critical(2.0)
    async def test_dashboard_initialization_and_loading(self, performance_tracker):
        """
        Test dashboard initialization with all components (PRD 4.4)
        
        WILL FAIL until SCWTMetricsDashboard is implemented with:
        - Dashboard component initialization
        - Metrics data fetching and aggregation
        - Real-time WebSocket connection setup
        - UI component rendering
        """
        performance_tracker.start_tracking("dashboard_init")
        
        # This will fail - SCWTMetricsDashboard doesn't exist yet (TDD Red Phase)
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test dashboard initialization
        initialization_result = await dashboard.initialize()
        
        performance_tracker.end_tracking("dashboard_init")
        
        # Validate initialization structure
        assert "components_loaded" in initialization_result
        assert "websocket_connected" in initialization_result
        assert "initial_data_fetched" in initialization_result
        assert "rendering_complete" in initialization_result
        
        # Validate component loading
        expected_components = [
            "confidence_overview",
            "performance_optimization", 
            "quality_assurance",
            "debugging_interface"
        ]
        
        for component in expected_components:
            assert component in initialization_result["components_loaded"]
        
        # WebSocket should be connected
        assert initialization_result["websocket_connected"] is True
        
        # Performance validation (PRD requirement: <2s dashboard load)
        performance_tracker.assert_performance_target("dashboard_init", 2.0)

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_confidence_overview_component(self):
        """
        Test confidence overview dashboard component (PRD 4.4)
        
        WILL FAIL until confidence overview is implemented:
        - System-wide confidence summary display
        - Phase breakdown with confidence metrics per phase
        - Trend analysis with historical confidence data
        - Alert system for low confidence scenarios
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test confidence overview data fetching
        confidence_overview = await dashboard.get_confidence_overview()
        
        # Validate confidence overview structure
        assert "system_wide_summary" in confidence_overview
        assert "phase_breakdown" in confidence_overview
        assert "trend_analysis" in confidence_overview
        assert "alerts_panel" in confidence_overview
        
        # Validate system-wide summary
        summary = confidence_overview["system_wide_summary"]
        assert "overall_confidence" in summary
        assert "active_tasks" in summary
        assert "confidence_distribution" in summary
        assert "last_updated" in summary
        
        # Overall confidence should be valid
        assert 0.0 <= summary["overall_confidence"] <= 1.0
        assert summary["active_tasks"] >= 0
        
        # Validate phase breakdown
        phase_breakdown = confidence_overview["phase_breakdown"]
        expected_phases = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7"]
        
        for phase in expected_phases:
            if phase in phase_breakdown:
                assert "confidence" in phase_breakdown[phase]
                assert "tasks_count" in phase_breakdown[phase]
                assert "status" in phase_breakdown[phase]
                assert 0.0 <= phase_breakdown[phase]["confidence"] <= 1.0
        
        # Validate trend analysis
        trend_analysis = confidence_overview["trend_analysis"]
        assert "historical_data" in trend_analysis
        assert "trend_direction" in trend_analysis
        assert "confidence_variance" in trend_analysis
        assert len(trend_analysis["historical_data"]) > 0

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_performance_optimization_metrics(self):
        """
        Test performance optimization dashboard metrics (PRD 4.4)
        
        WILL FAIL until performance metrics are implemented:
        - Token efficiency gains visualization
        - Cost reduction reporting with savings analysis
        - Model utilization statistics
        - Response time optimization charts
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test performance metrics fetching
        performance_metrics = await dashboard.get_performance_optimization_metrics()
        
        # Validate performance metrics structure
        assert "token_efficiency_gains" in performance_metrics
        assert "cost_reduction_analysis" in performance_metrics
        assert "model_utilization_stats" in performance_metrics
        assert "response_time_optimization" in performance_metrics
        
        # Validate token efficiency gains
        token_metrics = performance_metrics["token_efficiency_gains"]
        assert "baseline_usage" in token_metrics
        assert "optimized_usage" in token_metrics
        assert "savings_percentage" in token_metrics
        assert "efficiency_trend" in token_metrics
        
        # PRD requirement: 70-85% token savings should be reflected
        if token_metrics["savings_percentage"] is not None:
            assert token_metrics["savings_percentage"] >= 0.70, "Token savings should meet PRD target of 70%+"
        
        # Validate cost reduction analysis
        cost_analysis = performance_metrics["cost_reduction_analysis"]
        assert "cost_before_optimization" in cost_analysis
        assert "cost_after_optimization" in cost_analysis
        assert "total_savings" in cost_analysis
        assert "savings_trend" in cost_analysis
        
        # Validate model utilization
        utilization_stats = performance_metrics["model_utilization_stats"]
        assert "provider_distribution" in utilization_stats
        assert "model_performance" in utilization_stats
        assert "load_balancing_efficiency" in utilization_stats
        
        # Provider distribution should sum to approximately 100%
        if "provider_distribution" in utilization_stats:
            total_utilization = sum(utilization_stats["provider_distribution"].values())
            assert 0.95 <= total_utilization <= 1.05, "Provider distribution should sum to ~100%"

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_quality_assurance_dashboard(self):
        """
        Test quality assurance metrics dashboard (PRD 4.4)
        
        WILL FAIL until quality metrics are implemented:
        - Hallucination reduction tracking
        - Precision enhancement monitoring  
        - Confidence accuracy trend analysis
        - Validation success rate visualization
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test quality assurance metrics
        quality_metrics = await dashboard.get_quality_assurance_metrics()
        
        # Validate quality metrics structure
        assert "hallucination_reduction" in quality_metrics
        assert "precision_enhancement" in quality_metrics
        assert "confidence_accuracy_trend" in quality_metrics
        assert "validation_success_rate" in quality_metrics
        
        # Validate hallucination reduction metrics
        hallucination_metrics = quality_metrics["hallucination_reduction"]
        assert "baseline_rate" in hallucination_metrics
        assert "current_rate" in hallucination_metrics
        assert "reduction_percentage" in hallucination_metrics
        assert "trend_data" in hallucination_metrics
        
        # PRD requirement: 50% reduction in hallucinations
        if hallucination_metrics["reduction_percentage"] is not None:
            assert hallucination_metrics["reduction_percentage"] >= 0.50, "Hallucination reduction should meet PRD target of 50%+"
        
        # Validate precision enhancement
        precision_metrics = quality_metrics["precision_enhancement"]
        assert "overall_precision" in precision_metrics
        assert "precision_by_task_type" in precision_metrics
        assert "precision_trend" in precision_metrics
        
        # PRD requirement: ≥85% overall system precision
        if precision_metrics["overall_precision"] is not None:
            assert precision_metrics["overall_precision"] >= 0.85, "Overall precision should meet PRD target of ≥85%"
        
        # Validate confidence accuracy trend
        confidence_trend = quality_metrics["confidence_accuracy_trend"]
        assert "calibration_scores" in confidence_trend
        assert "accuracy_correlation" in confidence_trend
        assert "trend_direction" in confidence_trend
        
        # PRD requirement: ≥85% confidence accuracy correlation
        if confidence_trend["accuracy_correlation"] is not None:
            assert confidence_trend["accuracy_correlation"] >= 0.85, "Confidence accuracy should meet PRD target of ≥85%"

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_debugging_interface_component(self):
        """
        Test interactive debugging interface (PRD 4.4)
        
        WILL FAIL until debugging interface is implemented:
        - Confidence trace viewer with task execution timeline
        - Factor analysis with drill-down capabilities
        - Performance profiler for bottleneck identification
        - Optimization suggestions panel
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test debugging interface
        debugging_interface = await dashboard.get_debugging_interface()
        
        # Validate debugging interface structure
        assert "confidence_trace_viewer" in debugging_interface
        assert "factor_analysis_panel" in debugging_interface
        assert "performance_profiler" in debugging_interface
        assert "optimization_suggestions" in debugging_interface
        
        # Validate confidence trace viewer
        trace_viewer = debugging_interface["confidence_trace_viewer"]
        assert "active_traces" in trace_viewer
        assert "trace_history" in trace_viewer
        assert "trace_search" in trace_viewer
        
        # Validate factor analysis panel
        factor_analysis = debugging_interface["factor_analysis_panel"]
        assert "confidence_factors" in factor_analysis
        assert "factor_importance" in factor_analysis
        assert "factor_correlation" in factor_analysis
        
        # Validate performance profiler
        profiler = debugging_interface["performance_profiler"]
        assert "performance_bottlenecks" in profiler
        assert "execution_timeline" in profiler
        assert "resource_usage" in profiler
        
        # Validate optimization suggestions
        suggestions = debugging_interface["optimization_suggestions"]
        assert "active_suggestions" in suggestions
        assert "suggestion_categories" in suggestions
        assert "impact_predictions" in suggestions


class TestSCWTDashboardRealTimeFeatures:
    """Test real-time features and WebSocket connectivity"""
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    @performance_critical(2.0)
    async def test_websocket_connection_and_streaming(self, performance_tracker):
        """
        Test WebSocket connection for real-time metrics streaming (PRD 4.4)
        
        WILL FAIL until WebSocket streaming is implemented:
        - WebSocket connection establishment
        - Real-time metrics streaming
        - Connection resilience and reconnection
        - Message handling and parsing
        """
        performance_tracker.start_tracking("websocket_streaming")
        
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test WebSocket connection
        websocket_connection = await dashboard.establish_websocket_connection()
        
        # Validate connection establishment
        assert websocket_connection["connected"] is True
        assert "connection_id" in websocket_connection
        assert "endpoint_url" in websocket_connection
        
        # Test real-time metrics streaming
        metrics_stream = []
        stream_duration = 2.0  # Stream for 2 seconds
        
        async def collect_metrics():
            async for metric_update in dashboard.stream_metrics():
                metrics_stream.append(metric_update)
                if len(metrics_stream) >= 5:  # Collect at least 5 updates
                    break
        
        # Collect streaming metrics
        await asyncio.wait_for(collect_metrics(), timeout=stream_duration)
        
        performance_tracker.end_tracking("websocket_streaming")
        
        # Validate streaming data
        assert len(metrics_stream) > 0, "Should receive real-time metric updates"
        
        for metric_update in metrics_stream:
            assert "timestamp" in metric_update
            assert "metric_type" in metric_update
            assert "metric_data" in metric_update
            assert "confidence_snapshot" in metric_update
        
        # Performance validation (PRD requirement: <2s latency)
        performance_tracker.assert_performance_target("websocket_streaming", 2.0)

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_live_confidence_updates(self):
        """
        Test live confidence updates during task execution (PRD 4.4)
        
        WILL FAIL until live updates are implemented:
        - Real-time confidence score updates
        - Task progress monitoring
        - Confidence evolution visualization
        - Alert system for confidence drops
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Mock active task for live updates
        mock_task = {
            "task_id": "live-update-test",
            "status": "in_progress", 
            "initial_confidence": 0.75
        }
        
        # Test live confidence tracking
        confidence_updates = []
        
        async def track_confidence_updates():
            async for update in dashboard.track_task_confidence(mock_task["task_id"]):
                confidence_updates.append(update)
                if len(confidence_updates) >= 3:  # Track 3 updates
                    break
        
        # Start confidence tracking
        await asyncio.wait_for(track_confidence_updates(), timeout=3.0)
        
        # Validate live updates
        assert len(confidence_updates) >= 3, "Should receive multiple confidence updates"
        
        for i, update in enumerate(confidence_updates):
            assert "timestamp" in update
            assert "confidence_score" in update
            assert "task_progress" in update
            assert "confidence_factors" in update
            
            # Confidence should be valid
            assert 0.0 <= update["confidence_score"] <= 1.0
            assert 0.0 <= update["task_progress"] <= 1.0
            
            # Progress should increase over time
            if i > 0:
                prev_progress = confidence_updates[i-1]["task_progress"]
                current_progress = update["task_progress"]
                assert current_progress >= prev_progress, "Task progress should increase over time"

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_websocket_connection_resilience(self):
        """
        Test WebSocket connection resilience and auto-reconnection
        
        WILL FAIL until connection resilience is implemented
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Establish initial connection
        initial_connection = await dashboard.establish_websocket_connection()
        assert initial_connection["connected"] is True
        
        # Simulate connection drop
        await dashboard.simulate_connection_drop()
        
        # Test auto-reconnection
        reconnection_result = await dashboard.handle_connection_loss()
        
        # Validate reconnection handling
        assert "reconnection_attempted" in reconnection_result
        assert "reconnection_successful" in reconnection_result
        assert "fallback_activated" in reconnection_result
        
        # Should attempt reconnection
        assert reconnection_result["reconnection_attempted"] is True
        
        # If reconnection fails, fallback should be activated
        if not reconnection_result["reconnection_successful"]:
            assert reconnection_result["fallback_activated"] is True


class TestSCWTDashboardInteractivity:
    """Test interactive features and user interface"""
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_interactive_confidence_drill_down(self):
        """
        Test interactive confidence factor drill-down (PRD 4.4)
        
        WILL FAIL until drill-down functionality is implemented:
        - Clickable confidence scores
        - Factor breakdown expansion
        - Historical factor analysis
        - Factor correlation visualization
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test confidence score drill-down
        confidence_score_id = "test-confidence-001"
        drill_down_data = await dashboard.drill_down_confidence_factors(confidence_score_id)
        
        # Validate drill-down structure
        assert "confidence_breakdown" in drill_down_data
        assert "factor_analysis" in drill_down_data
        assert "historical_comparison" in drill_down_data
        assert "recommendations" in drill_down_data
        
        # Validate confidence breakdown
        breakdown = drill_down_data["confidence_breakdown"]
        assert "overall_confidence" in breakdown
        assert "factor_contributions" in breakdown
        assert "uncertainty_analysis" in breakdown
        
        # Factor contributions should sum to overall confidence (approximately)
        if "factor_contributions" in breakdown and breakdown["factor_contributions"]:
            total_contribution = sum(breakdown["factor_contributions"].values())
            overall_confidence = breakdown["overall_confidence"]
            assert abs(total_contribution - overall_confidence) <= 0.1, "Factor contributions should approximately sum to overall confidence"
        
        # Validate factor analysis
        factor_analysis = drill_down_data["factor_analysis"]
        assert "top_factors" in factor_analysis
        assert "factor_correlations" in factor_analysis
        assert "factor_trends" in factor_analysis
        
        # Should have meaningful recommendations
        recommendations = drill_down_data["recommendations"]
        assert len(recommendations) > 0
        for rec in recommendations:
            assert "recommendation_type" in rec
            assert "description" in rec
            assert "impact_estimate" in rec

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_performance_bottleneck_identification(self):
        """
        Test performance bottleneck identification and highlighting (PRD 4.4)
        
        WILL FAIL until bottleneck detection is implemented:
        - Automatic bottleneck detection
        - Performance impact analysis
        - Visual highlighting of issues
        - Optimization suggestion generation
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test bottleneck detection
        bottleneck_analysis = await dashboard.detect_performance_bottlenecks()
        
        # Validate bottleneck analysis structure
        assert "bottlenecks_detected" in bottleneck_analysis
        assert "bottleneck_details" in bottleneck_analysis
        assert "impact_assessment" in bottleneck_analysis
        assert "optimization_suggestions" in bottleneck_analysis
        
        # If bottlenecks are detected, validate details
        if bottleneck_analysis["bottlenecks_detected"]:
            details = bottleneck_analysis["bottleneck_details"]
            assert len(details) > 0
            
            for bottleneck in details:
                assert "bottleneck_type" in bottleneck
                assert "location" in bottleneck
                assert "severity" in bottleneck
                assert "performance_impact" in bottleneck
                
                # Severity should be valid
                assert bottleneck["severity"] in ["low", "medium", "high", "critical"]
                assert 0.0 <= bottleneck["performance_impact"] <= 1.0
        
        # Should provide optimization suggestions
        suggestions = bottleneck_analysis["optimization_suggestions"]
        if bottleneck_analysis["bottlenecks_detected"]:
            assert len(suggestions) > 0
            for suggestion in suggestions:
                assert "suggestion_type" in suggestion
                assert "description" in suggestion
                assert "expected_improvement" in suggestion

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_dashboard_customization_and_filters(self):
        """
        Test dashboard customization and filtering capabilities
        
        WILL FAIL until customization features are implemented
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test dashboard customization options
        customization_options = await dashboard.get_customization_options()
        
        # Validate customization structure
        assert "available_widgets" in customization_options
        assert "layout_options" in customization_options
        assert "filter_options" in customization_options
        assert "theme_options" in customization_options
        
        # Test applying custom configuration
        custom_config = {
            "widgets": ["confidence_overview", "performance_metrics"],
            "layout": "two_column",
            "filters": {"time_range": "last_24h", "confidence_threshold": 0.7},
            "theme": "dark"
        }
        
        customization_result = await dashboard.apply_customization(custom_config)
        
        # Validate customization application
        assert customization_result["applied"] is True
        assert "active_configuration" in customization_result
        
        # Active configuration should match requested config
        active_config = customization_result["active_configuration"]
        assert active_config["widgets"] == custom_config["widgets"]
        assert active_config["layout"] == custom_config["layout"]
        assert active_config["theme"] == custom_config["theme"]


class TestSCWTDashboardIntegration:
    """Test dashboard integration with other Phase 7 systems"""
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_deepconf_engine_integration(self):
        """
        Test dashboard integration with DeepConf Engine (PRD 4.4)
        
        WILL FAIL until DeepConf Engine integration is implemented:
        - Real-time confidence score display
        - Confidence calculation monitoring
        - Engine performance metrics
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        from archon.deepconf.engine import DeepConfEngine  # Will fail - doesn't exist
        
        dashboard = SCWTMetricsDashboard()
        engine = DeepConfEngine()
        
        # Test engine integration
        integration_status = await dashboard.integrate_with_deepconf_engine(engine)
        
        # Validate integration
        assert integration_status["integrated"] is True
        assert "engine_metrics_connected" in integration_status
        assert "real_time_scoring_enabled" in integration_status

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_consensus_system_integration(self):
        """
        Test dashboard integration with Multi-Model Consensus
        
        WILL FAIL until Consensus System integration is implemented
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        from archon.deepconf.consensus import MultiModelConsensus  # Will fail - doesn't exist
        
        dashboard = SCWTMetricsDashboard()
        consensus = MultiModelConsensus()
        
        # Test consensus integration
        integration_status = await dashboard.integrate_with_consensus_system(consensus)
        
        # Validate consensus integration
        assert integration_status["integrated"] is True
        assert "consensus_metrics_enabled" in integration_status
        assert "agreement_tracking_active" in integration_status

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_intelligent_router_integration(self):
        """
        Test dashboard integration with Intelligent Router
        
        WILL FAIL until Router integration is implemented
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        from archon.deepconf.routing import IntelligentRouter  # Will fail - doesn't exist
        
        dashboard = SCWTMetricsDashboard()
        router = IntelligentRouter()
        
        # Test router integration
        integration_status = await dashboard.integrate_with_intelligent_router(router)
        
        # Validate router integration
        assert integration_status["integrated"] is True
        assert "routing_metrics_enabled" in integration_status
        assert "performance_tracking_active" in integration_status


class TestSCWTDashboardEdgeCases:
    """Test edge cases and error handling"""
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_dashboard_with_no_data(self):
        """
        Test dashboard behavior with no historical data
        
        WILL FAIL until no-data handling is implemented
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Clear all data
        await dashboard.clear_all_data()
        
        # Test dashboard with no data
        empty_dashboard_state = await dashboard.get_dashboard_state()
        
        # Validate no-data handling
        assert "has_data" in empty_dashboard_state
        assert empty_dashboard_state["has_data"] is False
        assert "empty_state_message" in empty_dashboard_state
        assert "suggested_actions" in empty_dashboard_state
        
        # Should provide helpful suggestions for getting started
        suggestions = empty_dashboard_state["suggested_actions"]
        assert len(suggestions) > 0
        expected_suggestions = ["run_confidence_tasks", "enable_metrics_collection", "generate_sample_data"]
        for suggestion in suggestions:
            assert suggestion in expected_suggestions

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_dashboard_error_handling(self):
        """
        Test dashboard error handling and graceful degradation
        
        WILL FAIL until error handling is implemented
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Test with invalid configuration
        invalid_config = {"invalid_option": "invalid_value"}
        
        try:
            await dashboard.apply_customization(invalid_config)
            assert False, "Should raise error for invalid configuration"
        except ValueError as e:
            assert "invalid configuration" in str(e).lower()
        
        # Test with network/connection errors
        with patch('websockets.connect') as mock_connect:
            mock_connect.side_effect = ConnectionError("Network error")
            
            error_handling = await dashboard.handle_connection_error()
            
            assert "error_handled" in error_handling
            assert "fallback_mode" in error_handling
            assert error_handling["error_handled"] is True
            assert error_handling["fallback_mode"] is True

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    @dgts_validated
    async def test_dashboard_metrics_gaming_prevention(self):
        """
        Test prevention of dashboard metrics gaming (DGTS compliance)
        
        WILL FAIL until anti-gaming measures are implemented
        """
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Attempt to inject fake metrics
        fake_metrics = {
            "confidence_score": 1.0,  # Perfect score - suspicious
            "token_savings": 0.99,    # Unrealistic savings
            "hallucination_rate": 0.0  # Perfect - suspicious
        }
        
        # Dashboard should detect and reject gaming attempts
        gaming_detection = await dashboard.validate_metrics_authenticity(fake_metrics)
        
        # Validate gaming prevention
        assert "metrics_valid" in gaming_detection
        assert "gaming_detected" in gaming_detection
        assert "suspicious_patterns" in gaming_detection
        
        # Fake metrics should be flagged
        assert gaming_detection["gaming_detected"] is True
        assert gaming_detection["metrics_valid"] is False
        assert len(gaming_detection["suspicious_patterns"]) > 0


class TestSCWTDashboardPerformance:
    """Test dashboard performance requirements"""
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    @performance_critical(2.0)
    async def test_dashboard_load_performance(self, performance_tracker):
        """
        Test dashboard meets PRD performance requirement of <2s load time
        
        WILL FAIL until performance optimization achieves target
        """
        performance_tracker.start_tracking("dashboard_load")
        
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Load complete dashboard with all components
        full_dashboard = await dashboard.load_full_dashboard()
        
        performance_tracker.end_tracking("dashboard_load")
        
        # Validate dashboard loaded completely
        assert "loaded" in full_dashboard
        assert full_dashboard["loaded"] is True
        assert "components" in full_dashboard
        
        # Performance validation (PRD requirement: <2s load time)
        performance_tracker.assert_performance_target("dashboard_load", 2.0)

    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    @performance_critical(0.5)
    async def test_real_time_update_latency(self, performance_tracker):
        """
        Test real-time update latency meets <2s requirement
        
        WILL FAIL until real-time optimization achieves target latency
        """
        performance_tracker.start_tracking("update_latency")
        
        from archon.deepconf.dashboard import SCWTMetricsDashboard
        
        dashboard = SCWTMetricsDashboard()
        
        # Measure update latency
        start_time = time.time()
        
        # Trigger metric update
        await dashboard.trigger_metrics_update()
        
        # Wait for update to propagate to dashboard
        await dashboard.wait_for_update_propagation()
        
        end_time = time.time()
        update_latency = end_time - start_time
        
        performance_tracker.end_tracking("update_latency")
        
        # PRD requirement: <2s latency for real-time updates
        assert update_latency <= 2.0, f"Update latency {update_latency:.3f}s exceeds PRD requirement of 2s"

# Integration hooks for Phase 5+9 compatibility testing

class TestSCWTDashboardPhaseIntegration:
    """Test integration with other Phase systems"""
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_phase5_external_validator_dashboard_integration(self):
        """Test dashboard integration with Phase 5 External Validator"""
        pytest.skip("Requires Phase 5 External Validator integration")
    
    @tdd_red_phase
    @requires_implementation("SCWTMetricsDashboard")
    async def test_phase9_tdd_enforcement_dashboard_integration(self):
        """Test dashboard integration with Phase 9 TDD Enforcement"""
        pytest.skip("Requires Phase 9 TDD Enforcement integration")