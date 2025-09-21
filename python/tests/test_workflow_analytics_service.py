"""
Tests for workflow analytics service

Comprehensive test suite covering:
- Performance metrics calculation
- Cost analysis and optimization
- Bottleneck detection
- Trend analysis and forecasting
- Real-time dashboard metrics
- Usage pattern analysis
"""

import pytest
import json
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution, WorkflowAnalytics,
    ReactFlowNode, ReactFlowEdge, ReactFlowData, ExecutionStatus, StepType
)
from src.database import Base
from src.server.services.workflow_analytics_service import WorkflowAnalyticsService


@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def analytics_service():
    """Create analytics service instance"""
    return WorkflowAnalyticsService()


@pytest.fixture
def sample_workflow(test_db):
    """Create sample workflow for testing"""
    nodes = [
        ReactFlowNode(id="start", type="input", position={"x": 0, "y": 0}, data={"label": "Start"}),
        ReactFlowNode(id="process", type="agentTask", position={"x": 200, "y": 0}, data={"label": "Process"}),
        ReactFlowNode(id="end", type="output", position={"x": 400, "y": 0}, data={"label": "End"})
    ]
    edges = [
        ReactFlowEdge(id="e1", source="start", target="process"),
        ReactFlowEdge(id="e2", source="process", target="end")
    ]

    workflow = WorkflowDefinition(
        name="Test Workflow",
        description="For analytics testing",
        nodes=nodes,
        edges=edges,
        variables={"input_data": "default"}
    )
    test_db.add(workflow)
    test_db.commit()
    return workflow


@pytest.fixture
def sample_executions(test_db, sample_workflow):
    """Create sample executions for testing"""
    executions = []

    # Successful execution
    execution1 = WorkflowExecution(
        workflow_id=sample_workflow.id,
        status=ExecutionStatus.COMPLETED,
        inputs={"input_data": "test1"},
        results={"output": "success1"},
        started_at=datetime.now() - timedelta(hours=2),
        completed_at=datetime.now() - timedelta(hours=1, minutes=50),
        metrics={"duration": 10.5, "steps_executed": 3, "tokens_used": 1500}
    )
    executions.append(execution1)

    # Failed execution
    execution2 = WorkflowExecution(
        workflow_id=sample_workflow.id,
        status=ExecutionStatus.FAILED,
        inputs={"input_data": "test2"},
        errors=["Step process failed: timeout"],
        started_at=datetime.now() - timedelta(hours=1, minutes=30),
        completed_at=datetime.now() - timedelta(hours=1, minutes=20),
        metrics={"duration": 10.0, "steps_executed": 2, "tokens_used": 1000}
    )
    executions.append(execution2)

    # Recent successful execution
    execution3 = WorkflowExecution(
        workflow_id=sample_workflow.id,
        status=ExecutionStatus.COMPLETED,
        inputs={"input_data": "test3"},
        results={"output": "success3"},
        started_at=datetime.now() - timedelta(minutes=30),
        completed_at=datetime.now() - timedelta(minutes=20),
        metrics={"duration": 10.2, "steps_executed": 3, "tokens_used": 1400}
    )
    executions.append(execution3)

    for execution in executions:
        test_db.add(execution)
    test_db.commit()

    return executions


@pytest.fixture
def sample_step_executions(test_db, sample_executions):
    """Create sample step executions for testing"""
    step_executions = []

    # Steps for first execution
    step1 = StepExecution(
        execution_id=sample_executions[0].id,
        step_id="start",
        step_name="Start Step",
        step_type=StepType.INPUT,
        status=ExecutionStatus.COMPLETED,
        started_at=sample_executions[0].started_at,
        completed_at=sample_executions[0].started_at + timedelta(seconds=1),
        result={"processed": True},
        metrics={"duration": 1.0, "tokens_used": 100}
    )

    step2 = StepExecution(
        execution_id=sample_executions[0].id,
        step_id="process",
        step_name="Process Step",
        step_type=StepType.AGENT_TASK,
        status=ExecutionStatus.COMPLETED,
        started_at=sample_executions[0].started_at + timedelta(seconds=1),
        completed_at=sample_executions[0].started_at + timedelta(seconds=8),
        result={"analysis": "completed"},
        metrics={"duration": 7.0, "tokens_used": 1300}
    )

    step3 = StepExecution(
        execution_id=sample_executions[0].id,
        step_id="end",
        step_name="End Step",
        step_type=StepType.OUTPUT,
        status=ExecutionStatus.COMPLETED,
        started_at=sample_executions[0].started_at + timedelta(seconds=8),
        completed_at=sample_executions[0].completed_at,
        result={"output": "success1"},
        metrics={"duration": 2.5, "tokens_used": 100}
    )

    step_executions.extend([step1, step2, step3])

    # Steps for second execution (failed)
    step4 = StepExecution(
        execution_id=sample_executions[1].id,
        step_id="start",
        step_name="Start Step",
        step_type=StepType.INPUT,
        status=ExecutionStatus.COMPLETED,
        started_at=sample_executions[1].started_at,
        completed_at=sample_executions[1].started_at + timedelta(seconds=1),
        result={"processed": True},
        metrics={"duration": 1.0, "tokens_used": 100}
    )

    step5 = StepExecution(
        execution_id=sample_executions[1].id,
        step_id="process",
        step_name="Process Step",
        step_type=StepType.AGENT_TASK,
        status=ExecutionStatus.FAILED,
        started_at=sample_executions[1].started_at + timedelta(seconds=1),
        completed_at=sample_executions[1].completed_at,
        error="Timeout after 9 seconds",
        metrics={"duration": 9.0, "tokens_used": 900}
    )

    step_executions.extend([step4, step5])

    for step in step_executions:
        test_db.add(step)
    test_db.commit()

    return step_executions


@pytest.fixture
def sample_analytics(test_db, sample_workflow, sample_executions):
    """Create sample analytics records"""
    analytics_data = []

    # Analytics for today
    analytics1 = WorkflowAnalytics(
        workflow_id=sample_workflow.id,
        date=date.today(),
        executions_count=2,
        successful_executions=1,
        failed_executions=1,
        average_duration=10.35,
        total_tokens=2500,
        total_cost=0.125,
        performance_metrics={
            "throughput": 2.0,
            "error_rate": 0.5,
            "avg_response_time": 10.35
        }
    )
    analytics_data.append(analytics1)

    # Analytics for yesterday
    analytics2 = WorkflowAnalytics(
        workflow_id=sample_workflow.id,
        date=date.today() - timedelta(days=1),
        executions_count=5,
        successful_executions=4,
        failed_executions=1,
        average_duration=9.8,
        total_tokens=6000,
        total_cost=0.30,
        performance_metrics={
            "throughput": 5.0,
            "error_rate": 0.2,
            "avg_response_time": 9.8
        }
    )
    analytics_data.append(analytics2)

    for analytics in analytics_data:
        test_db.add(analytics)
    test_db.commit()

    return analytics_data


class TestWorkflowAnalyticsBasic:
    """Test basic workflow analytics functionality"""

    @pytest.mark.asyncio
    async def test_get_workflow_analytics(self, analytics_service, test_db, sample_workflow, sample_executions, sample_analytics):
        """Test getting workflow analytics"""
        analytics = await analytics_service.get_workflow_analytics(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert analytics is not None
        assert analytics["workflow_id"] == str(sample_workflow.id)
        assert analytics["workflow_name"] == sample_workflow.name
        assert "executions" in analytics
        assert "performance" in analytics
        assert "cost_analysis" in analytics
        assert "bottlenecks" in analytics
        assert "recommendations" in analytics

    @pytest.mark.asyncio
    async def test_get_workflow_analytics_not_exists(self, analytics_service, test_db):
        """Test getting analytics for non-existent workflow"""
        non_existent_id = str(uuid4())
        analytics = await analytics_service.get_workflow_analytics(
            workflow_id=non_existent_id,
            days=7,
            db=test_db
        )

        assert analytics is None

    @pytest.mark.asyncio
    async def test_get_workflow_analytics_no_data(self, analytics_service, test_db, sample_workflow):
        """Test getting analytics for workflow with no execution data"""
        analytics = await analytics_service.get_workflow_analytics(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert analytics is not None
        assert analytics["workflow_id"] == str(sample_workflow.id)
        assert analytics["executions"]["total"] == 0
        assert analytics["executions"]["successful"] == 0
        assert analytics["executions"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_get_workflow_analytics_date_range(self, analytics_service, test_db, sample_workflow, sample_analytics):
        """Test getting analytics with specific date range"""
        # Get analytics for today only
        analytics = await analytics_service.get_workflow_analytics(
            workflow_id=sample_workflow.id,
            days=1,
            db=test_db
        )

        assert analytics is not None
        assert len(analytics["trends"]) == 1  # Only today's data


class TestPerformanceMetrics:
    """Test performance metrics calculation"""

    @pytest.mark.asyncio
    async def test_calculate_performance_metrics(self, analytics_service, test_db, sample_workflow, sample_executions, sample_step_executions):
        """Test performance metrics calculation"""
        metrics = await analytics_service._calculate_performance_metrics(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert metrics is not None
        assert "executions_count" in metrics
        assert "success_rate" in metrics
        assert "average_duration" in metrics
        assert "total_duration" in metrics
        assert "min_duration" in metrics
        assert "max_duration" in metrics

        # Verify calculated values
        assert metrics["executions_count"] == 3
        assert metrics["successful_executions"] == 2
        assert metrics["failed_executions"] == 1
        assert metrics["success_rate"] == 2/3  # 66.67%
        assert metrics["average_duration"] == pytest.approx(10.23, rel=0.1)  # Average of 10.5, 10.0, 10.2

    @pytest.mark.asyncio
    async def test_calculate_step_performance(self, analytics_service, test_db, sample_workflow, sample_step_executions):
        """Test step performance calculation"""
        step_metrics = await analytics_service._calculate_step_performance(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert step_metrics is not None
        assert isinstance(step_metrics, dict)
        assert len(step_metrics) > 0

        # Check process step metrics
        if "process" in step_metrics:
            process_metrics = step_metrics["process"]
            assert "executions_count" in process_metrics
            assert "success_rate" in process_metrics
            assert "average_duration" in process_metrics
            assert "total_duration" in process_metrics

    @pytest.mark.asyncio
    async def test_identify_bottlenecks(self, analytics_service, test_db, sample_workflow, sample_step_executions):
        """Test bottleneck identification"""
        bottlenecks = await analytics_service._identify_bottlenecks(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert bottlenecks is not None
        assert isinstance(bottlenecks, list)

        # Should identify process step as potential bottleneck
        process_bottlenecks = [b for b in bottlenecks if b["step_id"] == "process"]
        if process_bottlenecks:
            bottleneck = process_bottlenecks[0]
            assert bottleneck["step_id"] == "process"
            assert "type" in bottleneck
            assert "severity" in bottleneck
            assert "description" in bottleneck


class TestCostAnalysis:
    """Test cost analysis functionality"""

    @pytest.mark.asyncio
    async def test_calculate_cost_analysis(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test cost analysis calculation"""
        cost_analysis = await analytics_service._calculate_cost_analysis(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert cost_analysis is not None
        assert "total_cost" in cost_analysis
        assert "average_cost_per_execution" in cost_analysis
        assert "cost_trends" in cost_analysis
        assert "cost_breakdown" in cost_analysis

        # Verify token-based cost calculation
        expected_total_cost = (1500 + 1000 + 1400) * 0.00005  # Assuming $0.00005 per token
        assert cost_analysis["total_cost"] == pytest.approx(expected_total_cost, rel=0.1)

    @pytest.mark.asyncio
    async def test_get_cost_optimization_recommendations(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test cost optimization recommendations"""
        recommendations = await analytics_service._get_cost_optimization_recommendations(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert recommendations is not None
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check recommendation structure
        for recommendation in recommendations:
            assert "type" in recommendation
            assert "description" in recommendation
            assert "potential_savings" in recommendation
            assert "priority" in recommendation


class TestTrendAnalysis:
    """Test trend analysis functionality"""

    @pytest.mark.asyncio
    async def test_analyze_trends(self, analytics_service, test_db, sample_workflow, sample_analytics):
        """Test trend analysis"""
        trends = await analytics_service._analyze_trends(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert trends is not None
        assert "execution_trend" in trends
        assert "performance_trend" in trends
        assert "cost_trend" in trends
        assert "success_rate_trend" in trends

    @pytest.mark.asyncio
    async def test_calculate_trend_direction(self, analytics_service):
        """Test trend direction calculation"""
        # Increasing trend
        data_points = [10, 15, 20, 25, 30]
        trend = analytics_service._calculate_trend_direction(data_points)
        assert trend["direction"] == "increasing"
        assert trend["strength"] > 0

        # Decreasing trend
        data_points = [30, 25, 20, 15, 10]
        trend = analytics_service._calculate_trend_direction(data_points)
        assert trend["direction"] == "decreasing"
        assert trend["strength"] < 0

        # Stable trend
        data_points = [20, 20, 20, 20, 20]
        trend = analytics_service._calculate_trend_direction(data_points)
        assert trend["direction"] == "stable"
        assert abs(trend["strength"]) < 0.1

    @pytest.mark.asyncio
    async def test_forecast_future_performance(self, analytics_service, test_db, sample_workflow, sample_analytics):
        """Test performance forecasting"""
        forecast = await analytics_service._forecast_future_performance(
            workflow_id=sample_workflow.id,
            days=7,
            forecast_days=3,
            db=test_db
        )

        assert forecast is not None
        assert "projected_executions" in forecast
        assert "projected_success_rate" in forecast
        assert "projected_costs" in forecast
        assert "confidence_interval" in forecast


class TestDashboardAnalytics:
    """Test dashboard analytics functionality"""

    @pytest.mark.asyncio
    async def test_get_dashboard_analytics(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test dashboard analytics"""
        dashboard = await analytics_service.get_dashboard_analytics(
            days=7,
            db=test_db
        )

        assert dashboard is not None
        assert "overview" in dashboard
        assert "active_workflows" in dashboard
        assert "recent_executions" in dashboard
        assert "system_metrics" in dashboard
        assert "top_performing_workflows" in dashboard
        assert "alerts" in dashboard

        # Check overview metrics
        overview = dashboard["overview"]
        assert "total_workflows" in overview
        assert "total_executions" in overview
        assert "average_success_rate" in overview
        assert "total_cost" in overview

    @pytest.mark.asyncio
    async def test_get_system_performance_analytics(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test system performance analytics"""
        performance = await analytics_service.get_system_performance_analytics(
            days=7,
            db=test_db
        )

        assert performance is not None
        assert "system_metrics" in performance
        assert "workflow_performance" in performance
        assert "bottlenecks" in performance
        assert "recommendations" in performance

    @pytest.mark.asyncio
    async def test_get_system_cost_analytics(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test system cost analytics"""
        cost_analytics = await analytics_service.get_system_cost_analytics(
            days=7,
            db=test_db
        )

        assert cost_analytics is not None
        assert "total_cost" in cost_analytics
        assert "cost_by_workflow" in cost_analytics
        assert "cost_trends" in cost_analytics
        assert "optimization_opportunities" in cost_analytics


class TestRealTimeMetrics:
    """Test real-time metrics functionality"""

    @pytest.mark.asyncio
    async def test_get_real_time_metrics(self, analytics_service, test_db, sample_workflow):
        """Test real-time metrics"""
        metrics = await analytics_service.get_real_time_metrics(db=test_db)

        assert metrics is not None
        assert "timestamp" in metrics
        assert "active_executions" in metrics
        assert "system_load" in metrics
        assert "recent_events" in metrics
        assert "performance_indicators" in metrics

    @pytest.mark.asyncio
    async def test_calculate_system_load(self, analytics_service, test_db):
        """Test system load calculation"""
        system_load = await analytics_service._calculate_system_load(db=test_db)

        assert system_load is not None
        assert "cpu_usage" in system_load
        assert "memory_usage" in system_load
        assert "active_workers" in system_load
        assert "queue_size" in system_load

    @pytest.mark.asyncio
    async def test_track_recent_events(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test recent events tracking"""
        events = await analytics_service._track_recent_events(db=test_db)

        assert events is not None
        assert isinstance(events, list)
        assert len(events) <= 10  # Should return at most 10 recent events


class TestUsagePatternAnalysis:
    """Test usage pattern analysis"""

    @pytest.mark.asyncio
    async def test_analyze_usage_patterns(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test usage pattern analysis"""
        patterns = await analytics_service._analyze_usage_patterns(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert patterns is not None
        assert "peak_usage_times" in patterns
        assert "execution_frequency" in patterns
        assert "input_patterns" in patterns
        assert "user_behavior" in patterns

    @pytest.mark.asyncio
    async def test_identify_peak_usage_times(self, analytics_service, test_db, sample_executions):
        """Test peak usage time identification"""
        peak_times = await analytics_service._identify_peak_usage_times(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert peak_times is not None
        assert isinstance(peak_times, list)
        assert len(peak_times) > 0

        # Check peak time structure
        for peak_time in peak_times:
            assert "hour" in peak_time
            assert "execution_count" in peak_time
            assert "percentage" in peak_time

    @pytest.mark.asyncio
    async def test_analyze_execution_frequency(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test execution frequency analysis"""
        frequency = await analytics_service._analyze_execution_frequency(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert frequency is not None
        assert "daily_average" in frequency
        assert "weekly_pattern" in frequency
        assert "trend" in frequency


class TestAlertingAndNotifications:
    """Test alerting and notification functionality"""

    @pytest.mark.asyncio
    async def test_generate_alerts(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test alert generation"""
        alerts = await analytics_service._generate_alerts(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert alerts is not None
        assert isinstance(alerts, list)

        # Should generate alert for high error rate (33%)
        error_alerts = [a for a in alerts if a["type"] == "high_error_rate"]
        if error_alerts:
            alert = error_alerts[0]
            assert alert["type"] == "high_error_rate"
            assert "severity" in alert
            assert "message" in alert

    @pytest.mark.asyncio
    async def test_check_performance_thresholds(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test performance threshold checking"""
        threshold_checks = await analytics_service._check_performance_thresholds(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert threshold_checks is not None
        assert isinstance(threshold_checks, dict)
        assert "duration_threshold" in threshold_checks
        assert "error_rate_threshold" in threshold_checks
        assert "cost_threshold" in threshold_checks

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test recommendation generation"""
        recommendations = await analytics_service._generate_recommendations(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert recommendations is not None
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check recommendation structure
        for recommendation in recommendations:
            assert "category" in recommendation
            assert "title" in recommendation
            assert "description" in recommendation
            assert "priority" in recommendation
            assert "estimated_impact" in recommendation


class TestAnalyticsDataManagement:
    """Test analytics data management"""

    @pytest.mark.asyncio
    async def test_aggregate_daily_analytics(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test daily analytics aggregation"""
        result = await analytics_service._aggregate_daily_analytics(
            target_date=date.today(),
            db=test_db
        )

        assert result is not None
        assert "workflow_id" in result
        assert "date" in result
        assert "executions_count" in result
        assert "successful_executions" in result
        assert "failed_executions" in result

    @pytest.mark.asyncio
    async def test_cleanup_old_analytics(self, analytics_service, test_db, sample_workflow, sample_analytics):
        """Test cleanup of old analytics data"""
        # Create old analytics record (365 days ago)
        old_analytics = WorkflowAnalytics(
            workflow_id=sample_workflow.id,
            date=date.today() - timedelta(days=365),
            executions_count=1,
            successful_executions=1,
            failed_executions=0,
            average_duration=5.0,
            total_tokens=500,
            total_cost=0.025
        )
        test_db.add(old_analytics)
        test_db.commit()

        # Cleanup analytics older than 180 days
        await analytics_service._cleanup_old_analytics(
            retention_days=180,
            db=test_db
        )

        # Verify old analytics was removed
        remaining = test_db.query(WorkflowAnalytics).filter(
            WorkflowAnalytics.workflow_id == sample_workflow.id
        ).all()

        assert len(remaining) == 2  # Only the two recent records should remain

    @pytest.mark.asyncio
    async def test_export_analytics_data(self, analytics_service, test_db, sample_workflow, sample_analytics):
        """Test analytics data export"""
        export_data = await analytics_service._export_analytics_data(
            workflow_id=sample_workflow.id,
            days=7,
            format="json",
            db=test_db
        )

        assert export_data is not None
        assert "workflow_id" in export_data
        assert "export_date" in export_data
        assert "data" in export_data
        assert "summary" in export_data

        # Check data format
        assert isinstance(export_data["data"], list)
        assert len(export_data["data"]) > 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_handle_database_errors(self, analytics_service, test_db):
        """Test handling of database errors"""
        # Mock database to raise exception
        with patch.object(test_db, 'query', side_effect=Exception("Database error")):
            analytics = await analytics_service.get_workflow_analytics(
                workflow_id=str(uuid4()),
                days=7,
                db=test_db
            )

            # Should handle error gracefully
            assert analytics is None

    @pytest.mark.asyncio
    async def test_handle_empty_data_sets(self, analytics_service, test_db, sample_workflow):
        """Test handling of empty data sets"""
        # Test with workflow but no executions
        analytics = await analytics_service.get_workflow_analytics(
            workflow_id=sample_workflow.id,
            days=7,
            db=test_db
        )

        assert analytics is not None
        assert analytics["executions"]["total"] == 0

    @pytest.mark.asyncio
    async def test_handle_invalid_date_ranges(self, analytics_service, test_db, sample_workflow):
        """Test handling of invalid date ranges"""
        # Test with negative days
        analytics = await analytics_service.get_workflow_analytics(
            workflow_id=sample_workflow.id,
            days=-1,
            db=test_db
        )

        # Should handle gracefully or default to reasonable range
        assert analytics is not None

    @pytest.mark.asyncio
    async def test_concurrent_analytics_requests(self, analytics_service, test_db, sample_workflow, sample_executions):
        """Test handling of concurrent analytics requests"""
        import asyncio

        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = analytics_service.get_workflow_analytics(
                workflow_id=sample_workflow.id,
                days=7,
                db=test_db
            )
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed successfully
        for result in results:
            assert isinstance(result, dict)
            assert result["workflow_id"] == str(sample_workflow.id)