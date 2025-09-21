"""
MCP Tools for Performance Knowledge Management

Provides MCP tools for capturing, analyzing, and utilizing performance
insights from workflow executions to optimize future runs and
improve overall system efficiency.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from mcp.server.fastmcp import Context

from ...database import get_db
from ...database.workflow_models import WorkflowExecution, ExecutionStatus
from ...server.services.workflow_knowledge_capture import WorkflowKnowledgeCapture
from ...server.services.knowledge_agent_bridge import KnowledgeAgentBridge
from ...server.services.workflow_analytics_service import get_workflow_analytics_service

logger = logging.getLogger(__name__)

def register_performance_knowledge_tools(mcp):
    """Register all performance knowledge management MCP tools"""

    @mcp.tool()
    async def archon_capture_performance_metrics(
        ctx: Context,
        execution_id: str,
        step_metrics: Dict[str, Any],
        overall_metrics: Dict[str, Any],
        resource_usage: Optional[Dict[str, Any]] = None,
        bottlenecks_identified: Optional[List[str]] = None,
        optimization_opportunities: Optional[List[str]] = None
    ) -> str:
        """
        Capture detailed performance metrics from workflow execution

        Records comprehensive performance data including step-level metrics,
        resource usage, bottlenecks, and optimization opportunities.

        Args:
            execution_id: Workflow execution ID
            step_metrics: Dictionary of step-specific performance metrics
            overall_metrics: Overall workflow performance metrics
            resource_usage: Resource consumption data (CPU, memory, network)
            bottlenecks_identified: List of identified bottlenecks
            optimization_opportunities: List of potential optimizations

        Returns:
            JSON with performance metrics capture status
        """
        try:
            knowledge_capture = WorkflowKnowledgeCapture()

            # Capture performance metrics
            metrics_id = await knowledge_capture.capture_performance_metrics(
                execution_id=execution_id,
                step_metrics=step_metrics,
                overall_metrics=overall_metrics,
                resource_usage=resource_usage or {},
                bottlenecks_identified=bottlenecks_identified or [],
                optimization_opportunities=optimization_opportunities or []
            )

            # Analyze for patterns
            patterns = await knowledge_capture.analyze_performance_patterns(
                execution_id=execution_id,
                metrics={"step_metrics": step_metrics, "overall_metrics": overall_metrics}
            )

            return json.dumps({
                "success": True,
                "metrics_id": metrics_id,
                "execution_id": execution_id,
                "patterns_identified": len(patterns),
                "bottlenecks_count": len(bottlenecks_identified or []),
                "optimization_opportunities_count": len(optimization_opportunities or []),
                "captured_at": datetime.now().isoformat(),
                "message": "Performance metrics captured successfully"
            })

        except Exception as e:
            logger.error(f"Failed to capture performance metrics: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to capture metrics: {str(e)}"
            })

    @mcp.tool()
    async def archon_analyze_workflow_efficiency(
        ctx: Context,
        workflow_id: str,
        period_days: Optional[int] = 30,
        include_step_analysis: Optional[bool] = True,
        benchmark_comparison: Optional[bool] = True
    ) -> str:
        """
        Analyze workflow efficiency over time

        Examines historical execution data to identify efficiency trends,
        patterns, and areas for improvement.

        Args:
            workflow_id: UUID of workflow to analyze
            period_days: Analysis period in days
            include_step_analysis: Whether to include step-level analysis
            benchmark_comparison: Whether to compare against benchmarks

        Returns:
            JSON with comprehensive efficiency analysis
        """
        try:
            knowledge_capture = WorkflowKnowledgeCapture()

            # Analyze efficiency
            efficiency_analysis = await knowledge_capture.analyze_workflow_efficiency(
                workflow_id=workflow_id,
                period_days=period_days or 30,
                include_step_analysis=include_step_analysis,
                benchmark_comparison=benchmark_comparison
            )

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "period_days": period_days or 30,
                "efficiency_analysis": efficiency_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to analyze workflow efficiency: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to analyze efficiency: {str(e)}"
            })

    @mcp.tool()
    async def archon_identify_performance_bottlenecks(
        ctx: Context,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        analysis_depth: Optional[str] = "detailed",
        include_recommendations: Optional[bool] = True
    ) -> str:
        """
        Identify performance bottlenecks in workflow execution

        Analyzes workflow execution data to identify bottlenecks,
        their causes, and potential solutions.

        Args:
            execution_id: Specific execution to analyze (optional)
            workflow_id: Workflow to analyze for historical bottlenecks (optional)
            analysis_depth: Analysis depth (quick, standard, detailed)
            include_recommendations: Whether to include optimization recommendations

        Returns:
            JSON with bottleneck analysis and recommendations
        """
        try:
            knowledge_capture = WorkflowKnowledgeCapture()

            # Identify bottlenecks
            bottlenecks = await knowledge_capture.identify_performance_bottlenecks(
                execution_id=execution_id,
                workflow_id=workflow_id,
                analysis_depth=analysis_depth or "detailed",
                include_recommendations=include_recommendations
            )

            return json.dumps({
                "success": True,
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "analysis_depth": analysis_depth or "detailed",
                "bottlenecks": bottlenecks,
                "recommendations_included": include_recommendations,
                "analyzed_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to identify performance bottlenecks: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to identify bottlenecks: {str(e)}"
            })

    @mcp.tool()
    async def archon_generate_performance_insights(
        ctx: Context,
        workflow_id: str,
        insight_types: Optional[List[str]] = None,
        time_period: Optional[Dict[str, str]] = None,
        min_confidence: Optional[float] = 0.7
    ) -> str:
        """
        Generate performance insights from workflow execution data

        Uses machine learning and pattern analysis to generate
        actionable insights about workflow performance.

        Args:
            workflow_id: UUID of workflow to analyze
            insight_types: Types of insights to generate (efficiency, cost, reliability, scalability)
            time_period: Time period for analysis {"start": "date", "end": "date"}
            min_confidence: Minimum confidence threshold for insights

        Returns:
            JSON with generated performance insights
        """
        try:
            analytics_service = await get_workflow_analytics_service()

            # Set default insight types
            default_types = ["efficiency", "cost", "reliability", "scalability"]
            insight_types = insight_types or default_types

            # Set default time period (last 30 days)
            if not time_period:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                time_period = {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }

            # Generate insights
            insights = await analytics_service.generate_performance_insights(
                workflow_id=workflow_id,
                insight_types=insight_types,
                time_period=time_period,
                min_confidence=min_confidence or 0.7
            )

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "insight_types": insight_types,
                "time_period": time_period,
                "min_confidence": min_confidence or 0.7,
                "insights": insights,
                "total_insights": len(insights),
                "generated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to generate performance insights: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to generate insights: {str(e)}"
            })

    @mcp.tool()
    async def archon_optimize_workflow_performance(
        ctx: Context,
        workflow_id: str,
        optimization_goals: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        optimization_level: Optional[str] = "moderate"
    ) -> str:
        """
        Generate performance optimization recommendations for workflow

        Analyzes workflow structure and execution history to provide
        specific optimization recommendations based on goals.

        Args:
            workflow_id: UUID of workflow to optimize
            optimization_goals: List of optimization goals (speed, cost, reliability, resource_usage)
            constraints: Optimization constraints (budget, time, resources)
            optimization_level: Optimization aggressiveness (conservative, moderate, aggressive)

        Returns:
            JSON with optimization recommendations and expected improvements
        """
        try:
            knowledge_capture = WorkflowKnowledgeCapture()

            # Generate optimization recommendations
            optimizations = await knowledge_capture.generate_workflow_optimizations(
                workflow_id=workflow_id,
                optimization_goals=optimization_goals,
                constraints=constraints or {},
                optimization_level=optimization_level or "moderate"
            )

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "optimization_goals": optimization_goals,
                "constraints": constraints or {},
                "optimization_level": optimization_level or "moderate",
                "optimizations": optimizations,
                "total_recommendations": len(optimizations),
                "generated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to optimize workflow performance: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to generate optimizations: {str(e)}"
            })

    @mcp.tool()
    async def archon_compare_performance_benchmarks(
        ctx: Context,
        execution_id: str,
        benchmark_type: Optional[str] = "historical",
        include_percentiles: Optional[bool] = True,
        comparison_window: Optional[int] = 30
    ) -> str:
        """
        Compare workflow execution performance against benchmarks

        Compares a specific execution against historical performance,
        industry benchmarks, or custom baselines.

        Args:
            execution_id: Execution ID to benchmark
            benchmark_type: Type of benchmark (historical, industry, custom)
            include_percentiles: Whether to include percentile comparisons
            comparison_window: Number of days for historical comparison

        Returns:
            JSON with benchmark comparison results
        """
        try:
            analytics_service = await get_workflow_analytics_service()

            # Get benchmark comparison
            comparison = await analytics_service.compare_with_benchmarks(
                execution_id=execution_id,
                benchmark_type=benchmark_type or "historical",
                include_percentiles=include_percentiles,
                comparison_window=comparison_window or 30
            )

            return json.dumps({
                "success": True,
                "execution_id": execution_id,
                "benchmark_type": benchmark_type or "historical",
                "comparison": comparison,
                "compared_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to compare performance benchmarks: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to compare benchmarks: {str(e)}"
            })

    @mcp.tool()
    async def archon_predict_performance_trends(
        ctx: Context,
        workflow_id: str,
        prediction_horizon: Optional[int] = 30,
        prediction_type: Optional[str] = "duration",
        confidence_interval: Optional[float] = 0.95
    ) -> str:
        """
        Predict future performance trends for workflow

        Uses historical data and machine learning to predict
        future performance metrics and trends.

        Args:
            workflow_id: UUID of workflow to predict
            prediction_horizon: Number of days to predict ahead
            prediction_type: Type of prediction (duration, cost, success_rate, resource_usage)
            confidence_interval: Confidence interval for predictions

        Returns:
            JSON with performance trend predictions
        """
        try:
            analytics_service = await get_workflow_analytics_service()

            # Generate predictions
            predictions = await analytics_service.predict_performance_trends(
                workflow_id=workflow_id,
                prediction_horizon=prediction_horizon or 30,
                prediction_type=prediction_type or "duration",
                confidence_interval=confidence_interval or 0.95
            )

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "prediction_horizon": prediction_horizon or 30,
                "prediction_type": prediction_type or "duration",
                "confidence_interval": confidence_interval or 0.95,
                "predictions": predictions,
                "generated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to predict performance trends: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to predict trends: {str(e)}"
            })

    @mcp.tool()
    async def archon_track_performance_improvements(
        ctx: Context,
        workflow_id: str,
        baseline_period: Dict[str, str],
        improvement_period: Dict[str, str],
        metrics_to_track: Optional[List[str]] = None
    ) -> str:
        """
        Track performance improvements over time

        Compares performance between two time periods to measure
        the impact of optimizations and improvements.

        Args:
            workflow_id: UUID of workflow to track
            baseline_period: {"start": "date", "end": "date"} baseline period
            improvement_period: {"start": "date", "end": "date"} improvement period
            metrics_to_track: List of metrics to track (duration, cost, success_rate, etc.)

        Returns:
            JSON with performance improvement metrics
        """
        try:
            analytics_service = await get_workflow_analytics_service()

            # Default metrics to track
            default_metrics = ["duration", "cost", "success_rate", "resource_usage"]
            metrics_to_track = metrics_to_track or default_metrics

            # Track improvements
            improvements = await analytics_service.track_performance_improvements(
                workflow_id=workflow_id,
                baseline_period=baseline_period,
                improvement_period=improvement_period,
                metrics_to_track=metrics_to_track
            )

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "baseline_period": baseline_period,
                "improvement_period": improvement_period,
                "metrics_tracked": metrics_to_track,
                "improvements": improvements,
                "analyzed_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to track performance improvements: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to track improvements: {str(e)}"
            })

    @mcp.tool()
    async def archon_store_performance_best_practices(
        ctx: Context,
        workflow_id: str,
        best_practices: List[Dict[str, Any]],
        practice_category: str,
        evidence_level: Optional[str] = "high",
        applicable_scenarios: Optional[List[str]] = None
    ) -> str:
        """
        Store performance best practices from workflow analysis

        Captures and stores best practices identified through
        performance analysis for future reference.

        Args:
            workflow_id: UUID of workflow where practices were identified
            best_practices: List of best practice dictionaries
            practice_category: Category of practices (efficiency, cost, reliability)
            evidence_level: Level of evidence supporting practices (high, medium, low)
            applicable_scenarios: Scenarios where practices apply

        Returns:
            JSON with best practices storage status
        """
        try:
            bridge = KnowledgeAgentBridge()

            # Store best practices
            practice_ids = await bridge.store_performance_best_practices(
                workflow_id=workflow_id,
                best_practices=best_practices,
                practice_category=practice_category,
                evidence_level=evidence_level or "high",
                applicable_scenarios=applicable_scenarios or []
            )

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "practice_category": practice_category,
                "practice_ids": practice_ids,
                "total_practices": len(best_practices),
                "evidence_level": evidence_level or "high",
                "stored_at": datetime.now().isoformat(),
                "message": "Performance best practices stored successfully"
            })

        except Exception as e:
            logger.error(f"Failed to store performance best practices: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to store best practices: {str(e)}"
            })

    @mcp.tool()
    async def archon_retrieve_performance_patterns(
        ctx: Context,
        query: str,
        pattern_types: Optional[List[str]] = None,
        workflow_categories: Optional[List[str]] = None,
        min_success_rate: Optional[float] = 0.8,
        limit: Optional[int] = 20
    ) -> str:
        """
        Retrieve performance patterns from knowledge base

        Search for and retrieve performance patterns, optimizations,
        and best practices from the knowledge base.

        Args:
            query: Search query for pattern matching
            pattern_types: Types of patterns to retrieve (bottleneck, optimization, efficiency)
            workflow_categories: Categories of workflows to focus on
            min_success_rate: Minimum success rate for patterns
            limit: Maximum number of patterns to return

        Returns:
            JSON with matching performance patterns
        """
        try:
            bridge = KnowledgeAgentBridge()

            # Retrieve patterns
            patterns = await bridge.retrieve_performance_patterns(
                query=query,
                pattern_types=pattern_types or ["bottleneck", "optimization", "efficiency"],
                workflow_categories=workflow_categories or [],
                min_success_rate=min_success_rate or 0.8,
                limit=limit or 20
            )

            return json.dumps({
                "success": True,
                "query": query,
                "pattern_types": pattern_types or ["bottleneck", "optimization", "efficiency"],
                "workflow_categories": workflow_categories or [],
                "min_success_rate": min_success_rate or 0.8,
                "patterns": patterns,
                "total_patterns": len(patterns),
                "limit": limit or 20,
                "retrieved_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to retrieve performance patterns: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to retrieve patterns: {str(e)}"
            })

    logger.info("✅ Performance Knowledge MCP tools registered successfully")