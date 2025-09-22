"""
Workflow Analytics Service

Provides comprehensive analytics and insights for workflow performance:
- Performance metrics and scoring
- Bottleneck identification and optimization
- Cost analysis and recommendations
- Usage pattern analysis
- Real-time analytics dashboard data
- Predictive analytics for workflow optimization

Following Archon server patterns and error handling standards
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func, extract
from sqlalchemy.sql import text

from ...database import get_db
from ...database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution,
    WorkflowMetrics, WorkflowAnalytics, ExecutionStatus, AgentType, ModelTier,
    calculate_workflow_performance_score, identify_workflow_bottlenecks
)
from ...database.agent_models import CostTracking

logger = logging.getLogger(__name__)

class AnalyticsPeriod(str, Enum):
    """Analytics time periods"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class MetricType(str, Enum):
    """Types of analytics metrics"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    COST = "cost"
    EFFICIENCY = "efficiency"
    USAGE = "usage"

@dataclass
class AnalyticsQuery:
    """Analytics query parameters"""
    workflow_id: Optional[str] = None
    project_id: Optional[str] = None
    period: AnalyticsPeriod = AnalyticsPeriod.DAY
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metrics: List[MetricType] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = list(MetricType)

class WorkflowAnalyticsService:
    """Main workflow analytics service"""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL

    async def get_workflow_performance(
        self,
        workflow_id: str,
        period_days: int = 30,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get comprehensive performance analytics for a workflow"""
        try:
            if not db:
                db = next(get_db())

            period_end = datetime.now()
            period_start = period_end - timedelta(days=period_days)

            # Get executions in period
            executions = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    WorkflowExecution.created_at >= period_start,
                    WorkflowExecution.created_at <= period_end
                )
            ).all()

            if not executions:
                return {
                    "workflow_id": workflow_id,
                    "period_days": period_days,
                    "total_executions": 0,
                    "message": "No executions found in specified period"
                }

            # Calculate basic metrics
            total_executions = len(executions)
            successful_executions = sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED)
            failed_executions = sum(1 for e in executions if e.status == ExecutionStatus.FAILED)
            cancelled_executions = sum(1 for e in executions if e.status == ExecutionStatus.CANCELLED)

            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
            failure_rate = failed_executions / total_executions if total_executions > 0 else 0.0

            # Time-based metrics
            completed_executions = [e for e in executions if e.execution_time_seconds and e.status == ExecutionStatus.COMPLETED]
            execution_times = [e.execution_time_seconds for e in completed_executions]

            avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
            median_execution_time = statistics.median(execution_times) if execution_times else 0.0
            min_execution_time = min(execution_times) if execution_times else 0.0
            max_execution_time = max(execution_times) if execution_times else 0.0

            # Cost analysis
            step_executions = db.query(StepExecution).join(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    WorkflowExecution.created_at >= period_start,
                    WorkflowExecution.created_at <= period_end
                )
            ).all()

            total_cost = sum(se.cost_usd or 0 for se in step_executions)
            avg_cost_per_execution = total_cost / total_executions if total_executions > 0 else 0.0

            # Token usage analysis
            total_tokens = sum(se.tokens_used or 0 for se in step_executions)
            avg_tokens_per_execution = total_tokens / total_executions if total_executions > 0 else 0

            # Bottleneck analysis
            bottlenecks = identify_workflow_bottlenecks(step_executions)

            # Calculate performance score
            performance_score = calculate_workflow_performance_score(
                success_rate,
                avg_execution_time,
                avg_cost_per_execution,
                1.0 - failure_rate  # Reliability score
            )

            return {
                "workflow_id": workflow_id,
                "period_days": period_days,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),

                # Execution metrics
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "cancelled_executions": cancelled_executions,
                "success_rate": round(success_rate, 4),
                "failure_rate": round(failure_rate, 4),

                # Time metrics
                "avg_execution_time_seconds": round(avg_execution_time, 2),
                "median_execution_time_seconds": round(median_execution_time, 2),
                "min_execution_time_seconds": min_execution_time,
                "max_execution_time_seconds": max_execution_time,

                # Cost metrics
                "total_cost_usd": round(total_cost, 4),
                "avg_cost_per_execution": round(avg_cost_per_execution, 4),
                "cost_efficiency": round(1.0 / avg_cost_per_execution if avg_cost_per_execution > 0 else 0, 4),

                # Token metrics
                "total_tokens": total_tokens,
                "avg_tokens_per_execution": round(avg_tokens_per_execution, 2),

                # Performance analysis
                "performance_score": round(performance_score, 4),
                "bottleneck_steps": bottlenecks,

                # Trends (comparisons)
                "performance_trend": await self._calculate_performance_trend(workflow_id, period_days, db),
                "cost_trend": await self._calculate_cost_trend(workflow_id, period_days, db)
            }

        except Exception as e:
            logger.error(f"Failed to get workflow performance for {workflow_id}: {e}")
            return {"error": str(e)}

    async def get_project_workflow_analytics(
        self,
        project_id: str,
        period_days: int = 30,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get workflow analytics for an entire project"""
        try:
            if not db:
                db = next(get_db())

            period_end = datetime.now()
            period_start = period_end - timedelta(days=period_days)

            # Get all workflow executions for project
            executions = db.query(WorkflowExecution).join(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    WorkflowExecution.created_at >= period_start,
                    WorkflowExecution.created_at <= period_end
                )
            ).all()

            # Get workflow definitions
            workflows = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.project_id == project_id
            ).all()

            # Project-level metrics
            total_workflows = len(workflows)
            active_workflows = len([w for w in workflows if w.status.value in ["PUBLISHED", "RUNNING"]])
            total_executions = len(executions)

            if total_executions == 0:
                return {
                    "project_id": project_id,
                    "period_days": period_days,
                    "total_workflows": total_workflows,
                    "active_workflows": active_workflows,
                    "total_executions": 0,
                    "message": "No executions found in specified period"
                }

            # Execution metrics
            successful_executions = sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED)
            failed_executions = sum(1 for e in executions if e.status == ExecutionStatus.FAILED)
            project_success_rate = successful_executions / total_executions if total_executions > 0 else 0.0

            # Time metrics
            completed_executions = [e for e in executions if e.execution_time_seconds]
            execution_times = [e.execution_time_seconds for e in completed_executions]
            avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0

            # Cost metrics
            step_executions = db.query(StepExecution).join(WorkflowExecution).join(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    WorkflowExecution.created_at >= period_start,
                    WorkflowExecution.created_at <= period_end
                )
            ).all()

            total_cost = sum(se.cost_usd or 0 for se in step_executions)
            avg_cost_per_execution = total_cost / total_executions if total_executions > 0 else 0.0

            # Workflow-level breakdown
            workflow_stats = {}
            for workflow in workflows:
                workflow_executions = [e for e in executions if e.workflow_id == workflow.id]
                if workflow_executions:
                    workflow_stats[str(workflow.id)] = {
                        "name": workflow.name,
                        "executions": len(workflow_executions),
                        "success_rate": sum(1 for e in workflow_executions if e.status == ExecutionStatus.COMPLETED) / len(workflow_executions),
                        "avg_execution_time": statistics.mean([e.execution_time_seconds for e in workflow_executions if e.execution_time_seconds] or [0])
                    }

            # Usage patterns
            usage_by_hour = await self._get_usage_by_hour(project_id, period_start, period_end, db)
            usage_by_day = await self._get_usage_by_day(project_id, period_start, period_end, db)

            # Agent usage analysis
            agent_usage = await self._get_agent_usage_stats(project_id, period_start, period_end, db)
            model_tier_usage = await self._get_model_tier_usage(project_id, period_start, period_end, db)

            return {
                "project_id": project_id,
                "period_days": period_days,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),

                # Project overview
                "total_workflows": total_workflows,
                "active_workflows": active_workflows,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "project_success_rate": round(project_success_rate, 4),

                # Performance metrics
                "avg_execution_time_seconds": round(avg_execution_time, 2),
                "total_cost_usd": round(total_cost, 4),
                "avg_cost_per_execution": round(avg_cost_per_execution, 4),

                # Workflow breakdown
                "workflow_statistics": workflow_stats,

                # Usage patterns
                "usage_by_hour": usage_by_hour,
                "usage_by_day": usage_by_day,

                # Agent analysis
                "agent_usage": agent_usage,
                "model_tier_usage": model_tier_usage,

                # Top performers
                "top_performing_workflows": await self._get_top_performing_workflows(project_id, period_start, period_end, db),
                "most_expensive_workflows": await self._get_most_expensive_workflows(project_id, period_start, period_end, db)
            }

        except Exception as e:
            logger.error(f"Failed to get project workflow analytics for {project_id}: {e}")
            return {"error": str(e)}

    async def get_real_time_metrics(
        self,
        project_id: Optional[str] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get real-time workflow metrics for dashboard"""
        try:
            if not db:
                db = next(get_db())

            # Get executions from last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)

            base_query = db.query(WorkflowExecution).filter(
                WorkflowExecution.created_at >= one_hour_ago
            )

            if project_id:
                base_query = base_query.join(WorkflowDefinition).filter(
                    WorkflowDefinition.project_id == project_id
                )

            recent_executions = base_query.all()

            # Active executions (running or paused)
            active_executions = base_query.filter(
                WorkflowExecution.status.in_([ExecutionStatus.RUNNING, ExecutionStatus.PAUSED])
            ).count()

            # Last hour metrics
            total_last_hour = len(recent_executions)
            successful_last_hour = sum(1 for e in recent_executions if e.status == ExecutionStatus.COMPLETED)
            failed_last_hour = sum(1 for e in recent_executions if e.status == ExecutionStatus.FAILED)

            # Calculate rates
            success_rate_last_hour = successful_last_hour / total_last_hour if total_last_hour > 0 else 0.0

            # Average execution time for completed executions
            completed_recent = [e for e in recent_executions if e.execution_time_seconds and e.status == ExecutionStatus.COMPLETED]
            avg_time_last_hour = statistics.mean([e.execution_time_seconds for e in completed_recent]) if completed_recent else 0.0

            # Cost for last hour
            if project_id:
                step_executions = db.query(StepExecution).join(WorkflowExecution).join(WorkflowDefinition).filter(
                    and_(
                        WorkflowDefinition.project_id == project_id,
                        StepExecution.created_at >= one_hour_ago
                    )
                ).all()
            else:
                step_executions = db.query(StepExecution).join(WorkflowExecution).filter(
                    StepExecution.created_at >= one_hour_ago
                ).all()

            cost_last_hour = sum(se.cost_usd or 0 for se in step_executions)

            # Current activity
            current_activity = []
            for execution in recent_executions[-10:]:  # Last 10 executions
                current_activity.append({
                    "execution_id": execution.execution_id,
                    "workflow_id": str(execution.workflow_id),
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "progress": execution.progress
                })

            return {
                "timestamp": datetime.now().isoformat(),
                "project_id": project_id,

                # Real-time counts
                "active_executions": active_executions,
                "total_executions_last_hour": total_last_hour,
                "successful_executions_last_hour": successful_last_hour,
                "failed_executions_last_hour": failed_last_hour,

                # Performance metrics
                "success_rate_last_hour": round(success_rate_last_hour, 4),
                "avg_execution_time_seconds": round(avg_time_last_hour, 2),
                "cost_last_hour_usd": round(cost_last_hour, 4),

                # Current activity
                "recent_activity": current_activity,

                # System health
                "system_status": "healthy",
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {"error": str(e)}

    async def generate_workflow_recommendations(
        self,
        workflow_id: str,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations for a workflow"""
        try:
            if not db:
                db = next(get_db())

            # Get last 30 days of data
            period_end = datetime.now()
            period_start = period_end - timedelta(days=30)

            executions = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    WorkflowExecution.created_at >= period_start,
                    WorkflowExecution.created_at <= period_end
                )
            ).all()

            if not executions:
                return [{"type": "info", "message": "No execution data available for recommendations"}]

            recommendations = []

            # Success rate analysis
            successful = sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED)
            failed = sum(1 for e in executions if e.status == ExecutionStatus.FAILED)
            success_rate = successful / len(executions) if executions else 0

            if success_rate < 0.8:
                recommendations.append({
                    "type": "critical",
                    "category": "reliability",
                    "message": f"Low success rate ({success_rate:.1%}). Review error handling and input validation.",
                    "priority": 1,
                    "potential_impact": "high"
                })
            elif success_rate < 0.95:
                recommendations.append({
                    "type": "warning",
                    "category": "reliability",
                    "message": f"Success rate could be improved ({success_rate:.1%}). Consider adding retry logic.",
                    "priority": 2,
                    "potential_impact": "medium"
                })

            # Performance analysis
            completed_executions = [e for e in executions if e.execution_time_seconds and e.status == ExecutionStatus.COMPLETED]
            if completed_executions:
                avg_time = statistics.mean([e.execution_time_seconds for e in completed_executions])
                if avg_time > 300:  # 5 minutes
                    recommendations.append({
                        "type": "warning",
                        "category": "performance",
                        "message": f"High average execution time ({avg_time:.0f}s). Consider optimization or parallel execution.",
                        "priority": 2,
                        "potential_impact": "medium"
                    })

            # Cost analysis
            step_executions = db.query(StepExecution).join(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    StepExecution.created_at >= period_start
                )
            ).all()

            total_cost = sum(se.cost_usd or 0 for se in step_executions)
            avg_cost = total_cost / len(executions) if executions else 0

            if avg_cost > 2.0:  # $2 per execution
                recommendations.append({
                    "type": "info",
                    "category": "cost",
                    "message": f"High average cost per execution (${avg_cost:.2f}). Consider using lower-tier models.",
                    "priority": 3,
                    "potential_impact": "low"
                })

            # Bottleneck analysis
            bottlenecks = identify_workflow_bottlenecks(step_executions)
            if bottlenecks:
                bottleneck = bottlenecks[0]  # Worst bottleneck
                recommendations.append({
                    "type": "warning",
                    "category": "performance",
                    "message": f"Step '{bottleneck['step_id']}' is {bottleneck['slowness_factor']:.1f}x slower than average.",
                    "priority": 2,
                    "potential_impact": "medium",
                    "details": {
                        "step_id": bottleneck["step_id"],
                        "avg_time": bottleneck["avg_time"],
                        "slowness_factor": bottleneck["slowness_factor"]
                    }
                })

            # Usage pattern analysis
            if len(executions) < 5:
                recommendations.append({
                    "type": "info",
                    "category": "usage",
                    "message": "Low usage in last 30 days. Consider archiving if not frequently used.",
                    "priority": 4,
                    "potential_impact": "low"
                })

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations for {workflow_id}: {e}")
            return [{"type": "error", "message": f"Failed to generate recommendations: {str(e)}"}]

    async def get_workflow_trends(
        self,
        workflow_id: str,
        period_days: int = 90,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get trend analysis for workflow metrics over time"""
        try:
            if not db:
                db = next(get_db())

            period_end = datetime.now()
            period_start = period_end - timedelta(days=period_days)

            # Group executions by day
            daily_stats = db.query(
                extract('date', WorkflowExecution.created_at).label('date'),
                func.count(WorkflowExecution.id).label('total_executions'),
                func.sum(func.case(
                    [(WorkflowExecution.status == ExecutionStatus.COMPLETED, 1)],
                    else_=0
                )).label('successful_executions'),
                func.avg(WorkflowExecution.execution_time_seconds).label('avg_execution_time')
            ).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    WorkflowExecution.created_at >= period_start,
                    WorkflowExecution.created_at <= period_end
                )
            ).group_by(
                extract('date', WorkflowExecution.created_at)
            ).order_by('date').all()

            # Calculate daily costs
            daily_costs = db.query(
                extract('date', StepExecution.created_at).label('date'),
                func.sum(StepExecution.cost_usd).label('total_cost')
            ).join(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    StepExecution.created_at >= period_start,
                    StepExecution.created_at <= period_end
                )
            ).group_by(
                extract('date', StepExecution.created_at)
            ).all()

            # Build trend data
            trend_data = []
            for i in range(period_days):
                current_date = period_start + timedelta(days=i)
                date_str = current_date.date()

                # Find stats for this date
                day_stats = next((s for s in daily_stats if s.date == date_str), None)
                day_cost = next((c for c in daily_costs if c.date == date_str), None)

                if day_stats:
                    success_rate = day_stats.successful_executions / day_stats.total_executions if day_stats.total_executions > 0 else 0.0
                    trend_data.append({
                        "date": date_str.isoformat(),
                        "executions": day_stats.total_executions,
                        "success_rate": round(success_rate, 4),
                        "avg_execution_time": round(day_stats.avg_execution_time or 0, 2),
                        "total_cost": round(day_cost.total_cost or 0, 4) if day_cost else 0.0
                    })
                else:
                    trend_data.append({
                        "date": date_str.isoformat(),
                        "executions": 0,
                        "success_rate": 0.0,
                        "avg_execution_time": 0.0,
                        "total_cost": 0.0
                    })

            # Calculate trends
            recent_data = trend_data[-7:]  # Last 7 days
            older_data = trend_data[-30:-7]  # Previous 23 days

            trends = {
                "executions_trend": self._calculate_trend(
                    sum(d["executions"] for d in recent_data),
                    sum(d["executions"] for d in older_data) if older_data else 0
                ),
                "success_rate_trend": self._calculate_trend(
                    statistics.mean([d["success_rate"] for d in recent_data]) if recent_data else 0,
                    statistics.mean([d["success_rate"] for d in older_data]) if older_data else 0
                ),
                "execution_time_trend": self._calculate_trend(
                    statistics.mean([d["avg_execution_time"] for d in recent_data]) if recent_data else 0,
                    statistics.mean([d["avg_execution_time"] for d in older_data]) if older_data else 0
                ),
                "cost_trend": self._calculate_trend(
                    sum(d["total_cost"] for d in recent_data),
                    sum(d["total_cost"] for d in older_data) if older_data else 0
                )
            }

            return {
                "workflow_id": workflow_id,
                "period_days": period_days,
                "trend_data": trend_data,
                "trends": {
                    "executions": trends["executions_trend"],
                    "success_rate": trends["success_rate_trend"],
                    "execution_time": trends["execution_time_trend"],
                    "cost": trends["cost_trend"]
                }
            }

        except Exception as e:
            logger.error(f"Failed to get workflow trends for {workflow_id}: {e}")
            return {"error": str(e)}

    # Private helper methods

    async def _calculate_performance_trend(
        self,
        workflow_id: str,
        period_days: int,
        db: Session
    ) -> str:
        """Calculate performance trend over time"""
        try:
            current_period_end = datetime.now()
            current_period_start = current_period_end - timedelta(days=period_days // 2)

            previous_period_end = current_period_start
            previous_period_start = previous_period_end - timedelta(days=period_days // 2)

            # Get current period success rate
            current_executions = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    WorkflowExecution.created_at >= current_period_start,
                    WorkflowExecution.created_at <= current_period_end
                )
            ).all()

            current_success_rate = (
                sum(1 for e in current_executions if e.status == ExecutionStatus.COMPLETED) /
                len(current_executions)
            ) if current_executions else 0.0

            # Get previous period success rate
            previous_executions = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    WorkflowExecution.created_at >= previous_period_start,
                    WorkflowExecution.created_at <= previous_period_end
                )
            ).all()

            previous_success_rate = (
                sum(1 for e in previous_executions if e.status == ExecutionStatus.COMPLETED) /
                len(previous_executions)
            ) if previous_executions else 0.0

            # Calculate trend
            if current_success_rate > previous_success_rate * 1.05:  # 5% improvement
                return "improving"
            elif current_success_rate < previous_success_rate * 0.95:  # 5% decline
                return "declining"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Failed to calculate performance trend: {e}")
            return "unknown"

    async def _calculate_cost_trend(
        self,
        workflow_id: str,
        period_days: int,
        db: Session
    ) -> str:
        """Calculate cost trend over time"""
        try:
            current_period_end = datetime.now()
            current_period_start = current_period_end - timedelta(days=period_days // 2)

            previous_period_end = current_period_start
            previous_period_start = previous_period_end - timedelta(days=period_days // 2)

            # Get current period average cost
            current_step_executions = db.query(StepExecution).join(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    StepExecution.created_at >= current_period_start,
                    StepExecution.created_at <= current_period_end
                )
            ).all()

            current_avg_cost = (
                sum(se.cost_usd or 0 for se in current_step_executions) /
                len(current_step_executions)
            ) if current_step_executions else 0.0

            # Get previous period average cost
            previous_step_executions = db.query(StepExecution).join(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    StepExecution.created_at >= previous_period_start,
                    StepExecution.created_at <= previous_period_end
                )
            ).all()

            previous_avg_cost = (
                sum(se.cost_usd or 0 for se in previous_step_executions) /
                len(previous_step_executions)
            ) if previous_step_executions else 0.0

            # Calculate trend
            if current_avg_cost > previous_avg_cost * 1.1:  # 10% increase
                return "increasing"
            elif current_avg_cost < previous_avg_cost * 0.9:  # 10% decrease
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Failed to calculate cost trend: {e}")
            return "unknown"

    async def _get_usage_by_hour(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session
    ) -> Dict[int, int]:
        """Get usage distribution by hour of day"""
        try:
            hourly_usage = db.query(
                extract('hour', WorkflowExecution.created_at).label('hour'),
                func.count(WorkflowExecution.id).label('count')
            ).join(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    WorkflowExecution.created_at >= start_date,
                    WorkflowExecution.created_at <= end_date
                )
            ).group_by(
                extract('hour', WorkflowExecution.created_at)
            ).all()

            # Fill in missing hours with 0
            result = {}
            for hour in range(24):
                usage = next((h.count for h in hourly_usage if h.hour == hour), 0)
                result[hour] = usage

            return result

        except Exception as e:
            logger.error(f"Failed to get usage by hour: {e}")
            return {}

    async def _get_usage_by_day(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session
    ) -> Dict[str, int]:
        """Get usage distribution by day of week"""
        try:
            daily_usage = db.query(
                extract('dow', WorkflowExecution.created_at).label('dow'),
                func.count(WorkflowExecution.id).label('count')
            ).join(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    WorkflowExecution.created_at >= start_date,
                    WorkflowExecution.created_at <= end_date
                )
            ).group_by(
                extract('dow', WorkflowExecution.created_at)
            ).all()

            days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            result = {}

            for i, day in enumerate(days):
                usage = next((d.count for d in daily_usage if d.dow == i), 0)
                result[day] = usage

            return result

        except Exception as e:
            logger.error(f"Failed to get usage by day: {e}")
            return {}

    async def _get_agent_usage_stats(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session
    ) -> Dict[str, Dict[str, Any]]:
        """Get agent usage statistics"""
        try:
            agent_stats = db.query(
                StepExecution.agent_type,
                func.count(StepExecution.id).label('execution_count'),
                func.sum(StepExecution.tokens_used).label('total_tokens'),
                func.sum(StepExecution.cost_usd).label('total_cost'),
                func.avg(StepExecution.execution_time_seconds).label('avg_time')
            ).join(WorkflowExecution).join(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    StepExecution.created_at >= start_date,
                    StepExecution.created_at <= end_date,
                    StepExecution.agent_type.isnot(None)
                )
            ).group_by(StepExecution.agent_type).all()

            result = {}
            for stat in agent_stats:
                result[stat.agent_type] = {
                    "execution_count": stat.execution_count,
                    "total_tokens": stat.total_tokens or 0,
                    "total_cost": round(stat.total_cost or 0, 4),
                    "avg_execution_time": round(stat.avg_time or 0, 2),
                    "avg_cost_per_execution": round((stat.total_cost or 0) / stat.execution_count, 4) if stat.execution_count > 0 else 0.0
                }

            return result

        except Exception as e:
            logger.error(f"Failed to get agent usage stats: {e}")
            return {}

    async def _get_model_tier_usage(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session
    ) -> Dict[str, Dict[str, Any]]:
        """Get model tier usage statistics"""
        try:
            tier_stats = db.query(
                StepExecution.model_tier,
                func.count(StepExecution.id).label('execution_count'),
                func.sum(StepExecution.tokens_used).label('total_tokens'),
                func.sum(StepExecution.cost_usd).label('total_cost')
            ).join(WorkflowExecution).join(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    StepExecution.created_at >= start_date,
                    StepExecution.created_at <= end_date,
                    StepExecution.model_tier.isnot(None)
                )
            ).group_by(StepExecution.model_tier).all()

            result = {}
            for stat in tier_stats:
                result[stat.model_tier] = {
                    "execution_count": stat.execution_count,
                    "total_tokens": stat.total_tokens or 0,
                    "total_cost": round(stat.total_cost or 0, 4)
                }

            return result

        except Exception as e:
            logger.error(f"Failed to get model tier usage: {e}")
            return {}

    async def _get_top_performing_workflows(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top performing workflows by success rate"""
        try:
            workflow_stats = db.query(
                WorkflowExecution.workflow_id,
                func.count(WorkflowExecution.id).label('total_executions'),
                func.sum(func.case(
                    [(WorkflowExecution.status == ExecutionStatus.COMPLETED, 1)],
                    else_=0
                )).label('successful_executions'),
                WorkflowDefinition.name
            ).join(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    WorkflowExecution.created_at >= start_date,
                    WorkflowExecution.created_at <= end_date
                )
            ).group_by(
                WorkflowExecution.workflow_id,
                WorkflowDefinition.name
            ).having(
                func.count(WorkflowExecution.id) >= 5  # At least 5 executions
            ).all()

            # Calculate success rates and sort
            results = []
            for stat in workflow_stats:
                success_rate = stat.successful_executions / stat.total_executions
                results.append({
                    "workflow_id": str(stat.workflow_id),
                    "name": stat.name,
                    "success_rate": round(success_rate, 4),
                    "total_executions": stat.total_executions,
                    "successful_executions": stat.successful_executions
                })

            return sorted(results, key=lambda x: x["success_rate"], reverse=True)[:limit]

        except Exception as e:
            logger.error(f"Failed to get top performing workflows: {e}")
            return []

    async def _get_most_expensive_workflows(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        db: Session,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get most expensive workflows by total cost"""
        try:
            workflow_costs = db.query(
                WorkflowExecution.workflow_id,
                WorkflowDefinition.name,
                func.sum(StepExecution.cost_usd).label('total_cost'),
                func.count(WorkflowExecution.id).label('execution_count')
            ).join(WorkflowDefinition).join(StepExecution).filter(
                and_(
                    WorkflowDefinition.project_id == project_id,
                    WorkflowExecution.created_at >= start_date,
                    WorkflowExecution.created_at <= end_date
                )
            ).group_by(
                WorkflowExecution.workflow_id,
                WorkflowDefinition.name
            ).all()

            results = []
            for stat in workflow_costs:
                if stat.total_cost:
                    results.append({
                        "workflow_id": str(stat.workflow_id),
                        "name": stat.name,
                        "total_cost": round(stat.total_cost or 0, 4),
                        "execution_count": stat.execution_count,
                        "avg_cost_per_execution": round((stat.total_cost or 0) / stat.execution_count, 4) if stat.execution_count > 0 else 0.0
                    })

            return sorted(results, key=lambda x: x["total_cost"], reverse=True)[:limit]

        except Exception as e:
            logger.error(f"Failed to get most expensive workflows: {e}")
            return []

    def _calculate_trend(self, current_value: float, previous_value: float) -> str:
        """Calculate trend direction between two values"""
        if previous_value == 0:
            return "unknown" if current_value == 0 else "increasing"

        change_ratio = (current_value - previous_value) / previous_value

        if change_ratio > 0.1:  # 10% increase
            return "increasing"
        elif change_ratio < -0.1:  # 10% decrease
            return "decreasing"
        else:
            return "stable"

# Global service instance
analytics_service: Optional[WorkflowAnalyticsService] = None

async def get_workflow_analytics_service() -> WorkflowAnalyticsService:
    """Get or create workflow analytics service instance"""
    global analytics_service
    if analytics_service is None:
        analytics_service = WorkflowAnalyticsService()
    return analytics_service