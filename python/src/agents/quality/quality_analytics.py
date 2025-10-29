"""
Quality Metrics and Analytics System
Collects, analyzes, and reports on quality assurance metrics
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from datetime import datetime, timedelta
import statistics
from collections import defaultdict

from .qa_framework import ValidationStage, ValidationSeverity
from .qa_orchestrator import WorkflowResult


class MetricType(Enum):
    """Types of quality metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TREND = "trend"


@dataclass
class QualityMetric:
    """Represents a quality metric"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityTrend:
    """Represents a quality trend over time"""
    metric_name: str
    time_period: str  # 'daily', 'weekly', 'monthly'
    start_date: datetime
    end_date: datetime
    data_points: List[Tuple[datetime, float]] = field(default_factory=list)
    trend_direction: str = "stable"  # 'improving', 'declining', 'stable'
    trend_strength: float = 0.0  # 0-1, how strong the trend is


@dataclass
class QualityInsight:
    """Represents a quality insight or recommendation"""
    insight_type: str  # 'improvement', 'warning', 'achievement'
    title: str
    description: str
    impact_level: str  # 'low', 'medium', 'high'
    recommendations: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class QualityAnalytics:
    """Quality analytics and metrics collection system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.db_path = Path(self.config.get('db_path', 'quality_metrics.db'))
        self.retention_days = self.config.get('retention_days', 90)

        # Initialize database
        self._init_database()

        # Metric definitions
        self.metric_definitions = {
            'total_workflows': MetricType.COUNTER,
            'successful_workflows': MetricType.COUNTER,
            'failed_workflows': MetricType.COUNTER,
            'workflow_success_rate': MetricType.GAUGE,
            'average_execution_time': MetricType.GAUGE,
            'violations_per_workflow': MetricType.HISTOGRAM,
            'critical_violations': MetricType.COUNTER,
            'test_coverage': MetricType.GAUGE,
            'code_quality_score': MetricType.GAUGE,
            'documentation_coverage': MetricType.GAUGE,
            'security_score': MetricType.GAUGE,
            'performance_score': MetricType.GAUGE
        }

    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                tags TEXT,
                metadata TEXT
            )
        ''')

        # Create workflow_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflow_results (
                id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                trigger TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                total_violations INTEGER NOT NULL,
                critical_violations INTEGER NOT NULL,
                error_violations INTEGER NOT NULL,
                execution_summary TEXT
            )
        ''')

        # Create violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                rule TEXT NOT NULL,
                severity TEXT NOT NULL,
                file_path TEXT,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflow_results (id)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics (name, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_results_timestamp ON workflow_results (start_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_rule_timestamp ON violations (rule, timestamp)')

        conn.commit()
        conn.close()

    async def record_workflow_result(self, result: WorkflowResult):
        """Record a workflow result and extract metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert workflow result
            cursor.execute('''
                INSERT OR REPLACE INTO workflow_results
                (id, stage, trigger, passed, start_time, end_time, total_violations,
                 critical_violations, error_violations, execution_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.request_id,
                result.stage.value,
                result.trigger.value,
                result.passed,
                result.start_time,
                result.end_time,
                len(result.all_violations),
                len([v for v in result.all_violations if v.severity == ValidationSeverity.CRITICAL]),
                len([v for v in result.all_violations if v.severity == ValidationSeverity.ERROR]),
                json.dumps(result.execution_summary)
            ))

            # Insert violations
            for violation in result.all_violations:
                cursor.execute('''
                    INSERT INTO violations (workflow_id, rule, severity, file_path, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    result.request_id,
                    violation.rule,
                    violation.severity.value,
                    violation.file_path,
                    result.start_time
                ))

            # Extract and record metrics
            metrics = self._extract_metrics_from_result(result)
            for metric in metrics:
                cursor.execute('''
                    INSERT INTO metrics (name, metric_type, value, timestamp, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    metric.metric_type.value,
                    metric.value,
                    metric.timestamp,
                    json.dumps(metric.tags),
                    json.dumps(metric.metadata)
                ))

            conn.commit()

        finally:
            conn.close()

    def _extract_metrics_from_result(self, result: WorkflowResult) -> List[QualityMetric]:
        """Extract quality metrics from a workflow result"""
        metrics = []
        timestamp = result.end_time or datetime.now()

        # Basic workflow metrics
        metrics.append(QualityMetric(
            name='total_workflows',
            metric_type=MetricType.COUNTER,
            value=1,
            timestamp=timestamp,
            tags={'stage': result.stage.value, 'trigger': result.trigger.value}
        ))

        metrics.append(QualityMetric(
            name='successful_workflows' if result.passed else 'failed_workflows',
            metric_type=MetricType.COUNTER,
            value=1,
            timestamp=timestamp,
            tags={'stage': result.stage.value, 'trigger': result.trigger.value}
        ))

        # Execution time metrics
        if result.end_time:
            execution_time_ms = (result.end_time - result.start_time).total_seconds() * 1000
            metrics.append(QualityMetric(
                name='average_execution_time',
                metric_type=MetricType.GAUGE,
                value=execution_time_ms,
                timestamp=timestamp,
                tags={'stage': result.stage.value}
            ))

        # Violation metrics
        total_violations = len(result.all_violations)
        if total_violations > 0:
            metrics.append(QualityMetric(
                name='violations_per_workflow',
                metric_type=MetricType.HISTOGRAM,
                value=float(total_violations),
                timestamp=timestamp,
                tags={'stage': result.stage.value}
            ))

        critical_violations = len([v for v in result.all_violations if v.severity == ValidationSeverity.CRITICAL])
        if critical_violations > 0:
            metrics.append(QualityMetric(
                name='critical_violations',
                metric_type=MetricType.COUNTER,
                value=float(critical_violations),
                timestamp=timestamp,
                tags={'stage': result.stage.value}
            ))

        # Stage-specific metrics
        if result.stage == ValidationStage.SPRINT_COMPLETION:
            self._extract_sprint_metrics(result, metrics, timestamp)
        elif result.stage == ValidationStage.GIT_COMMIT:
            self._extract_commit_metrics(result, metrics, timestamp)
        elif result.stage == ValidationStage.FILE_SUBMISSION:
            self._extract_file_metrics(result, metrics, timestamp)

        return metrics

    def _extract_sprint_metrics(self, result: WorkflowResult, metrics: List[QualityMetric], timestamp: datetime):
        """Extract sprint-specific metrics"""
        summary = result.execution_summary

        if 'goal_completion_rate' in summary:
            metrics.append(QualityMetric(
                name='goal_completion_rate',
                metric_type=MetricType.GAUGE,
                value=summary['goal_completion_rate'],
                timestamp=timestamp
            ))

        if 'quality_score' in summary:
            metrics.append(QualityMetric(
                name='code_quality_score',
                metric_type=MetricType.GAUGE,
                value=summary['quality_score'],
                timestamp=timestamp
            ))

        if 'test_pass_rate' in summary:
            metrics.append(QualityMetric(
                name='test_coverage',
                metric_type=MetricType.GAUGE,
                value=summary['test_pass_rate'],
                timestamp=timestamp
            ))

        if 'documentation_coverage' in summary:
            metrics.append(QualityMetric(
                name='documentation_coverage',
                metric_type=MetricType.GAUGE,
                value=summary['documentation_coverage'],
                timestamp=timestamp
            ))

    def _extract_commit_metrics(self, result: WorkflowResult, metrics: List[QualityMetric], timestamp: datetime):
        """Extract commit-specific metrics"""
        summary = result.execution_summary

        if 'commits_validated' in summary:
            metrics.append(QualityMetric(
                name='commits_per_validation',
                metric_type=MetricType.HISTOGRAM,
                value=float(summary['commits_validated']),
                timestamp=timestamp
            ))

    def _extract_file_metrics(self, result: WorkflowResult, metrics: List[QualityMetric], timestamp: datetime):
        """Extract file submission-specific metrics"""
        summary = result.execution_summary

        if 'files_validated' in summary:
            metrics.append(QualityMetric(
                name='files_per_submission',
                metric_type=MetricType.HISTOGRAM,
                value=float(summary['files_validated']),
                timestamp=timestamp
            ))

    def get_metric_history(
        self,
        metric_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[QualityMetric]:
        """Get historical data for a specific metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        query = "SELECT name, metric_type, value, timestamp, tags, metadata FROM metrics WHERE name = ?"
        params = [metric_name]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if tags:
            for key, value in tags.items():
                query += f" AND json_extract(tags, '$.{key}') = ?"
                params.append(value)

        query += " ORDER BY timestamp"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            QualityMetric(
                name=row[0],
                metric_type=MetricType(row[1]),
                value=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                tags=json.loads(row[4]) if row[4] else {},
                metadata=json.loads(row[5]) if row[5] else {}
            )
            for row in rows
        ]

    def calculate_trend(
        self,
        metric_name: str,
        time_period: str = 'daily',
        days_back: int = 30
    ) -> QualityTrend:
        """Calculate trend for a metric over time"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get historical data
        history = self.get_metric_history(metric_name, start_date, end_date)

        if not history:
            return QualityTrend(
                metric_name=metric_name,
                time_period=time_period,
                start_date=start_date,
                end_date=end_date,
                data_points=[],
                trend_direction="stable",
                trend_strength=0.0
            )

        # Group by time period
        if time_period == 'daily':
            grouped = self._group_by_day(history)
        elif time_period == 'weekly':
            grouped = self._group_by_week(history)
        else:  # monthly
            grouped = self._group_by_month(history)

        # Calculate trend
        data_points = [(timestamp, value) for timestamp, value in grouped.items()]
        data_points.sort(key=lambda x: x[0])

        if len(data_points) < 2:
            return QualityTrend(
                metric_name=metric_name,
                time_period=time_period,
                start_date=start_date,
                end_date=end_date,
                data_points=data_points,
                trend_direction="stable",
                trend_strength=0.0
            )

        # Simple linear regression for trend
        x_values = list(range(len(data_points)))
        y_values = [value for _, value in data_points]

        # Calculate slope
        n = len(data_points)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0

        # Determine trend direction and strength
        avg_value = sum(y_values) / len(y_values)
        normalized_slope = slope / avg_value if avg_value != 0 else 0

        if abs(normalized_slope) < 0.01:
            trend_direction = "stable"
            trend_strength = 0.0
        elif normalized_slope > 0:
            trend_direction = "improving"
            trend_strength = min(abs(normalized_slope), 1.0)
        else:
            trend_direction = "declining"
            trend_strength = min(abs(normalized_slope), 1.0)

        return QualityTrend(
            metric_name=metric_name,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            data_points=data_points,
            trend_direction=trend_direction,
            trend_strength=trend_strength
        )

    def _group_by_day(self, metrics: List[QualityMetric]) -> Dict[datetime, float]:
        """Group metrics by day"""
        grouped = defaultdict(list)
        for metric in metrics:
            day = metric.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            grouped[day].append(metric.value)

        # Calculate average for each day
        return {day: sum(values) / len(values) for day, values in grouped.items()}

    def _group_by_week(self, metrics: List[QualityMetric]) -> Dict[datetime, float]:
        """Group metrics by week"""
        grouped = defaultdict(list)
        for metric in metrics:
            # Get start of week (Monday)
            day = metric.timestamp
            week_start = day - timedelta(days=day.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            grouped[week_start].append(metric.value)

        return {week: sum(values) / len(values) for week, values in grouped.items()}

    def _group_by_month(self, metrics: List[QualityMetric]) -> Dict[datetime, float]:
        """Group metrics by month"""
        grouped = defaultdict(list)
        for metric in metrics:
            month_start = metric.timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            grouped[month_start].append(metric.value)

        return {month: sum(values) / len(values) for month, values in grouped.items()}

    def get_top_violations(self, days_back: int = 30, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common violations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = datetime.now() - timedelta(days=days_back)

        cursor.execute('''
            SELECT rule, severity, COUNT(*) as count
            FROM violations
            WHERE timestamp >= ?
            GROUP BY rule, severity
            ORDER BY count DESC
            LIMIT ?
        ''', (start_date.isoformat(), limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'rule': row[0],
                'severity': row[1],
                'count': row[2]
            }
            for row in rows
        ]

    def get_quality_score_overview(self, days_back: int = 7) -> Dict[str, Any]:
        """Get overview of quality scores"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get recent metrics
        metrics_to_analyze = [
            'workflow_success_rate',
            'test_coverage',
            'code_quality_score',
            'documentation_coverage'
        ]

        overview = {
            'period_days': days_back,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'metrics': {}
        }

        for metric_name in metrics_to_analyze:
            history = self.get_metric_history(metric_name, start_date, end_date)
            if history:
                latest_value = history[-1].value
                trend = self.calculate_trend(metric_name, 'daily', days_back)

                overview['metrics'][metric_name] = {
                    'current_value': latest_value,
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength,
                    'data_points': len(history)
                }
            else:
                overview['metrics'][metric_name] = {
                    'current_value': None,
                    'trend_direction': 'stable',
                    'trend_strength': 0.0,
                    'data_points': 0
                }

        return overview

    def generate_insights(self, days_back: int = 30) -> List[QualityInsight]:
        """Generate quality insights and recommendations"""
        insights = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get top violations
        top_violations = self.get_top_violations(days_back, 5)
        if top_violations:
            critical_violations = [v for v in top_violations if v['severity'] == 'critical']
            if critical_violations:
                insights.append(QualityInsight(
                    insight_type='warning',
                    title='Critical Violations Detected',
                    description=f"Found {len(critical_violations)} types of critical violations in the last {days_back} days",
                    impact_level='high',
                    recommendations=[
                        'Review and address critical violations immediately',
                        'Update coding standards and training if needed',
                        'Consider automated pre-commit hooks to prevent recurrence'
                    ],
                    metrics=['critical_violations']
                ))

        # Check quality trends
        quality_score_trend = self.calculate_trend('code_quality_score', 'weekly', days_back)
        if quality_score_trend.trend_direction == 'declining' and quality_score_trend.trend_strength > 0.1:
            insights.append(QualityInsight(
                insight_type='warning',
                title='Quality Score Declining',
                description=f"Code quality score has been declining over the past {days_back} days",
                impact_level='medium',
                recommendations=[
                    'Investigate root causes of quality decline',
                    'Increase code review coverage',
                    'Provide additional team training',
                    'Review recent changes to processes or tools'
                ],
                metrics=['code_quality_score']
            ))

        # Check test coverage
        test_coverage_history = self.get_metric_history('test_coverage', start_date, end_date)
        if test_coverage_history:
            avg_coverage = sum(m.value for m in test_coverage_history) / len(test_coverage_history)
            if avg_coverage < 0.90:
                insights.append(QualityInsight(
                    insight_type='improvement',
                    title='Low Test Coverage',
                    description=f"Average test coverage is {avg_coverage:.1%}, below recommended 90%",
                    impact_level='medium',
                    recommendations=[
                        'Set test coverage requirements in CI/CD pipeline',
                        'Allocate time for writing tests for existing code',
                        'Provide test-driven development training',
                        'Use code coverage tools to identify untested areas'
                    ],
                    metrics=['test_coverage']
                ))

        # Check workflow success rate
        success_rate_history = self.get_metric_history('workflow_success_rate', start_date, end_date)
        if success_rate_history:
            avg_success_rate = sum(m.value for m in success_rate_history) / len(success_rate_history)
            if avg_success_rate > 0.95:
                insights.append(QualityInsight(
                    insight_type='achievement',
                    title='High Workflow Success Rate',
                    description=f"Workflow success rate is {avg_success_rate:.1%}, excellent performance",
                    impact_level='low',
                    recommendations=[
                        'Continue current quality practices',
                        'Share success factors with other teams',
                        'Document best practices for future reference'
                    ],
                    metrics=['workflow_success_rate']
                ))

        return insights

    def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Delete old metrics
            cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_date.isoformat(),))
            metrics_deleted = cursor.rowcount

            # Delete old workflow results
            cursor.execute("DELETE FROM workflow_results WHERE start_time < ?", (cutoff_date.isoformat(),))
            workflows_deleted = cursor.rowcount

            # Delete old violations
            cursor.execute("DELETE FROM violations WHERE timestamp < ?", (cutoff_date.isoformat(),))
            violations_deleted = cursor.rowcount

            conn.commit()

            return {
                'metrics_deleted': metrics_deleted,
                'workflows_deleted': workflows_deleted,
                'violations_deleted': violations_deleted
            }

        finally:
            conn.close()

    def export_metrics(self, output_path: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """Export metrics to JSON file"""
        all_metrics = []

        for metric_name in self.metric_definitions.keys():
            history = self.get_metric_history(metric_name, start_date, end_date)
            all_metrics.extend(history)

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'total_metrics': len(all_metrics),
            'metrics': [
                {
                    'name': m.name,
                    'metric_type': m.metric_type.value,
                    'value': m.value,
                    'timestamp': m.timestamp.isoformat(),
                    'tags': m.tags,
                    'metadata': m.metadata
                }
                for m in all_metrics
            ]
        }

        Path(output_path).write_text(json.dumps(export_data, indent=2), encoding='utf-8')
        return len(all_metrics)