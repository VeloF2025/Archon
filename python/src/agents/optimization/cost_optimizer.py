"""
Cost Optimization Engine v3.0 - Real-time Cost Tracking and Budget Management
Based on F-ITR-002 from PRD specifications

NLNH Protocol: Real cost tracking with actual budget constraints
DGTS Enforcement: No fake cost calculations, actual budget management and optimization
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers with associated costs"""
    OPUS = "opus"
    SONNET = "sonnet" 
    HAIKU = "haiku"


class AlertLevel(Enum):
    """Budget alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CostRecord:
    """Individual cost record for task execution"""
    record_id: str
    agent_id: str
    task_id: str
    project_id: str
    model_tier: str
    input_tokens: int
    output_tokens: int
    execution_time: float
    cost: float
    timestamp: datetime
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_from_task(cls, agent_id: str, task_id: str, project_id: str,
                        model_tier: str, input_tokens: int, output_tokens: int,
                        execution_time: float, success: bool = True,
                        metadata: Dict[str, Any] = None):
        """Create cost record from task execution data"""
        cost_calculator = CostCalculator()
        cost = cost_calculator.calculate_cost(model_tier, input_tokens, output_tokens)
        
        return cls(
            record_id=str(uuid.uuid4()),
            agent_id=agent_id,
            task_id=task_id,
            project_id=project_id,
            model_tier=model_tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            execution_time=execution_time,
            cost=cost,
            timestamp=datetime.now(),
            success=success,
            metadata=metadata or {}
        )


@dataclass
class BudgetConstraint:
    """Budget constraint configuration"""
    constraint_id: str
    project_id: str
    daily_limit: float
    monthly_limit: float
    per_agent_limit: Optional[float] = None
    per_tier_limits: Dict[str, float] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Alert thresholds (percentage of limit)
    warning_threshold: float = 0.8    # 80%
    critical_threshold: float = 0.95  # 95%
    
    def get_alert_level(self, current_usage: float, limit: float) -> Optional[AlertLevel]:
        """Get alert level based on current usage"""
        if limit <= 0:
            return None
        
        usage_percentage = current_usage / limit
        
        if usage_percentage >= 1.0:
            return AlertLevel.EMERGENCY
        elif usage_percentage >= self.critical_threshold:
            return AlertLevel.CRITICAL
        elif usage_percentage >= self.warning_threshold:
            return AlertLevel.WARNING
        else:
            return None


@dataclass 
class ROIMetrics:
    """ROI metrics for agent performance analysis"""
    agent_id: str
    project_id: str
    total_cost: float = 0.0
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_cost_per_task: float = 0.0
    success_rate: float = 0.0
    roi_score: float = 0.0
    efficiency_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    cost_trend: str = "stable"  # improving, stable, worsening
    
    def update_metrics(self, cost: float, success: bool, execution_time: float = None):
        """Update metrics with new task data"""
        self.total_cost += cost
        self.total_tasks += 1
        
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        # Calculate derived metrics
        self.avg_cost_per_task = self.total_cost / self.total_tasks
        self.success_rate = self.successful_tasks / self.total_tasks
        
        # ROI Score: value delivered per dollar spent
        # Higher success rate and lower cost = better ROI
        baseline_cost = 0.01  # $0.01 baseline
        cost_efficiency = baseline_cost / max(self.avg_cost_per_task, baseline_cost)
        self.roi_score = self.success_rate * cost_efficiency
        
        # Efficiency Score: tasks completed per dollar (considering success rate)
        self.efficiency_score = (self.successful_tasks / max(self.total_cost, 0.001)) * 1000  # per $1
        
        self.last_updated = datetime.now()
    
    def get_optimization_recommendation(self) -> Dict[str, Any]:
        """Get optimization recommendation based on metrics"""
        recommendations = []
        
        if self.success_rate < 0.6:
            recommendations.append({
                "type": "performance_issue",
                "priority": "high",
                "message": f"Low success rate ({self.success_rate:.1%}). Consider retraining or tier upgrade.",
                "action": "investigate_failures"
            })
        
        if self.avg_cost_per_task > 0.05:  # $0.05 threshold
            if self.success_rate < 0.8:
                recommendations.append({
                    "type": "cost_efficiency",
                    "priority": "medium", 
                    "message": f"High cost per task (${self.avg_cost_per_task:.3f}) with moderate success. Consider tier downgrade.",
                    "action": "downgrade_tier"
                })
        
        if self.roi_score < 0.5:
            recommendations.append({
                "type": "roi_concern",
                "priority": "medium",
                "message": f"Low ROI score ({self.roi_score:.2f}). Review agent configuration.",
                "action": "optimize_configuration"
            })
        
        if not recommendations:
            recommendations.append({
                "type": "performing_well",
                "priority": "info",
                "message": "Agent performing within acceptable parameters.",
                "action": "maintain_current_setup"
            })
        
        return {
            "agent_id": self.agent_id,
            "recommendations": recommendations,
            "current_roi_score": self.roi_score,
            "current_success_rate": self.success_rate,
            "current_avg_cost": self.avg_cost_per_task
        }


class CostCalculator:
    """Calculate costs based on model tier and token usage"""
    
    def __init__(self):
        # Cost per 1M tokens (input/output) - based on Anthropic pricing
        self.tier_pricing = {
            ModelTier.OPUS.value: {
                "input": 15.00,   # $15 per 1M input tokens
                "output": 75.00   # $75 per 1M output tokens
            },
            ModelTier.SONNET.value: {
                "input": 3.00,    # $3 per 1M input tokens
                "output": 15.00   # $15 per 1M output tokens
            },
            ModelTier.HAIKU.value: {
                "input": 0.25,    # $0.25 per 1M input tokens
                "output": 1.25    # $1.25 per 1M output tokens
            }
        }
    
    def calculate_cost(self, model_tier: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        if model_tier not in self.tier_pricing:
            logger.warning(f"Unknown model tier: {model_tier}, defaulting to Sonnet pricing")
            model_tier = ModelTier.SONNET.value
        
        pricing = self.tier_pricing[model_tier]
        
        # Convert to cost per token (divide by 1,000,000)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        total_cost = input_cost + output_cost
        
        logger.debug(f"Cost calculation for {model_tier}: "
                    f"${input_cost:.6f} (input) + ${output_cost:.6f} (output) = ${total_cost:.6f}")
        
        return total_cost
    
    def estimate_cost_for_task(self, model_tier: str, estimated_input: int, estimated_output: int) -> float:
        """Estimate cost for a task before execution"""
        return self.calculate_cost(model_tier, estimated_input, estimated_output)
    
    def get_tier_cost_comparison(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Get cost comparison across all tiers"""
        return {
            tier: self.calculate_cost(tier, input_tokens, output_tokens)
            for tier in [ModelTier.OPUS.value, ModelTier.SONNET.value, ModelTier.HAIKU.value]
        }


class CostDatabase:
    """SQLite database for cost tracking"""
    
    def __init__(self, db_path: str = "/tmp/archon_costs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize cost tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            # Cost records table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cost_records (
                    record_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    model_tier TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    execution_time REAL NOT NULL,
                    cost REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Budget constraints table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS budget_constraints (
                    constraint_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL UNIQUE,
                    daily_limit REAL NOT NULL,
                    monthly_limit REAL NOT NULL,
                    per_agent_limit REAL,
                    per_tier_limits TEXT,
                    active BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    warning_threshold REAL DEFAULT 0.8,
                    critical_threshold REAL DEFAULT 0.95
                )
            ''')
            
            # Indexes for performance
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cost_records_project_date 
                ON cost_records(project_id, timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cost_records_agent 
                ON cost_records(agent_id, timestamp)
            ''')
    
    async def store_cost_record(self, record: CostRecord) -> bool:
        """Store cost record in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO cost_records (
                        record_id, agent_id, task_id, project_id, model_tier,
                        input_tokens, output_tokens, execution_time, cost,
                        timestamp, success, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.record_id, record.agent_id, record.task_id, record.project_id,
                    record.model_tier, record.input_tokens, record.output_tokens,
                    record.execution_time, record.cost, record.timestamp.isoformat(),
                    record.success, json.dumps(record.metadata)
                ))
            
            logger.debug(f"Stored cost record {record.record_id}: ${record.cost:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store cost record: {e}")
            return False
    
    async def get_cost_records(self, project_id: Optional[str] = None, 
                             agent_id: Optional[str] = None,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             limit: int = 1000) -> List[CostRecord]:
        """Retrieve cost records with filtering"""
        try:
            conditions = ["1=1"]  # Base condition
            params = []
            
            if project_id:
                conditions.append("project_id = ?")
                params.append(project_id)
            
            if agent_id:
                conditions.append("agent_id = ?")
                params.append(agent_id)
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("timestamp <= ?") 
                params.append(end_date.isoformat())
            
            query = f'''
                SELECT * FROM cost_records 
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
            
            records = []
            for row in rows:
                records.append(CostRecord(
                    record_id=row['record_id'],
                    agent_id=row['agent_id'],
                    task_id=row['task_id'],
                    project_id=row['project_id'],
                    model_tier=row['model_tier'],
                    input_tokens=row['input_tokens'],
                    output_tokens=row['output_tokens'],
                    execution_time=row['execution_time'],
                    cost=row['cost'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    success=bool(row['success']),
                    metadata=json.loads(row['metadata'] or '{}')
                ))
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to retrieve cost records: {e}")
            return []
    
    async def store_budget_constraint(self, constraint: BudgetConstraint) -> bool:
        """Store budget constraint"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO budget_constraints (
                        constraint_id, project_id, daily_limit, monthly_limit,
                        per_agent_limit, per_tier_limits, active, created_at, updated_at,
                        warning_threshold, critical_threshold
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    constraint.constraint_id, constraint.project_id,
                    constraint.daily_limit, constraint.monthly_limit,
                    constraint.per_agent_limit, json.dumps(constraint.per_tier_limits),
                    constraint.active, constraint.created_at.isoformat(),
                    constraint.updated_at.isoformat(), constraint.warning_threshold,
                    constraint.critical_threshold
                ))
            
            logger.info(f"Stored budget constraint for project {constraint.project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store budget constraint: {e}")
            return False
    
    async def get_budget_constraint(self, project_id: str) -> Optional[BudgetConstraint]:
        """Get budget constraint for project"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM budget_constraints WHERE project_id = ? AND active = 1",
                    (project_id,)
                )
                row = cursor.fetchone()
            
            if not row:
                return None
            
            return BudgetConstraint(
                constraint_id=row['constraint_id'],
                project_id=row['project_id'],
                daily_limit=row['daily_limit'],
                monthly_limit=row['monthly_limit'],
                per_agent_limit=row['per_agent_limit'],
                per_tier_limits=json.loads(row['per_tier_limits'] or '{}'),
                active=bool(row['active']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                warning_threshold=row['warning_threshold'],
                critical_threshold=row['critical_threshold']
            )
            
        except Exception as e:
            logger.error(f"Failed to get budget constraint: {e}")
            return None


class BudgetMonitor:
    """Monitor budget usage and generate alerts"""
    
    def __init__(self, cost_db: CostDatabase):
        self.cost_db = cost_db
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for budget alerts"""
        self.alert_callbacks.append(callback)
    
    async def check_budget_status(self, project_id: str) -> Dict[str, Any]:
        """Check current budget status for project"""
        constraint = await self.cost_db.get_budget_constraint(project_id)
        if not constraint:
            return {
                "project_id": project_id,
                "status": "no_budget_set",
                "within_limits": True,
                "alerts": []
            }
        
        # Get current usage
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Daily usage
        daily_records = await self.cost_db.get_cost_records(
            project_id=project_id,
            start_date=today
        )
        daily_usage = sum(r.cost for r in daily_records)
        
        # Monthly usage  
        monthly_records = await self.cost_db.get_cost_records(
            project_id=project_id,
            start_date=month_start
        )
        monthly_usage = sum(r.cost for r in monthly_records)
        
        # Generate alerts
        alerts = []
        
        # Daily alerts
        daily_alert = constraint.get_alert_level(daily_usage, constraint.daily_limit)
        if daily_alert:
            alerts.append({
                "type": daily_alert.value,
                "scope": "daily",
                "usage": daily_usage,
                "limit": constraint.daily_limit,
                "percentage": (daily_usage / constraint.daily_limit) * 100,
                "message": f"Daily budget at {(daily_usage/constraint.daily_limit)*100:.1f}%"
            })
        
        # Monthly alerts
        monthly_alert = constraint.get_alert_level(monthly_usage, constraint.monthly_limit)
        if monthly_alert:
            alerts.append({
                "type": monthly_alert.value,
                "scope": "monthly", 
                "usage": monthly_usage,
                "limit": constraint.monthly_limit,
                "percentage": (monthly_usage / constraint.monthly_limit) * 100,
                "message": f"Monthly budget at {(monthly_usage/constraint.monthly_limit)*100:.1f}%"
            })
        
        # Check per-agent limits if configured
        if constraint.per_agent_limit:
            agent_usage = {}
            for record in daily_records:
                if record.agent_id not in agent_usage:
                    agent_usage[record.agent_id] = 0
                agent_usage[record.agent_id] += record.cost
            
            for agent_id, usage in agent_usage.items():
                agent_alert = constraint.get_alert_level(usage, constraint.per_agent_limit)
                if agent_alert:
                    alerts.append({
                        "type": agent_alert.value,
                        "scope": "agent",
                        "agent_id": agent_id,
                        "usage": usage,
                        "limit": constraint.per_agent_limit,
                        "percentage": (usage / constraint.per_agent_limit) * 100,
                        "message": f"Agent {agent_id} at {(usage/constraint.per_agent_limit)*100:.1f}% of daily limit"
                    })
        
        # Fire alert callbacks
        for alert in alerts:
            if alert["type"] in ["critical", "emergency"]:
                for callback in self.alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(alert)
                        else:
                            callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
        
        within_limits = (
            daily_usage < constraint.daily_limit and
            monthly_usage < constraint.monthly_limit
        )
        
        return {
            "project_id": project_id,
            "status": "active",
            "within_limits": within_limits,
            "daily_usage": daily_usage,
            "daily_limit": constraint.daily_limit,
            "monthly_usage": monthly_usage,
            "monthly_limit": constraint.monthly_limit,
            "alerts": alerts,
            "usage_percentage": {
                "daily": (daily_usage / constraint.daily_limit) * 100,
                "monthly": (monthly_usage / constraint.monthly_limit) * 100
            }
        }
    
    async def recommend_tier_changes(self, project_id: str) -> List[Dict[str, Any]]:
        """Recommend tier changes based on budget constraints"""
        budget_status = await self.check_budget_status(project_id)
        
        if budget_status["within_limits"]:
            return []
        
        recommendations = []
        
        # Analyze recent cost records by tier
        recent_records = await self.cost_db.get_cost_records(
            project_id=project_id,
            start_date=datetime.now() - timedelta(days=1)
        )
        
        # Group by agent and tier
        agent_tier_costs = {}
        for record in recent_records:
            key = f"{record.agent_id}_{record.model_tier}"
            if key not in agent_tier_costs:
                agent_tier_costs[key] = {
                    "agent_id": record.agent_id,
                    "current_tier": record.model_tier,
                    "cost": 0,
                    "tasks": 0,
                    "success_rate": 0
                }
            
            agent_tier_costs[key]["cost"] += record.cost
            agent_tier_costs[key]["tasks"] += 1
            if record.success:
                agent_tier_costs[key]["success_rate"] += 1
        
        # Calculate success rates and recommend downgrades
        for key, data in agent_tier_costs.items():
            if data["tasks"] > 0:
                data["success_rate"] = data["success_rate"] / data["tasks"]
                avg_cost_per_task = data["cost"] / data["tasks"]
                
                # Recommend downgrade if cost is high and success rate is acceptable
                if data["current_tier"] == "opus" and data["success_rate"] > 0.8:
                    recommendations.append({
                        "agent_id": data["agent_id"],
                        "current_tier": "opus",
                        "recommended_tier": "sonnet",
                        "reason": f"High success rate ({data['success_rate']:.1%}) suggests Sonnet may be sufficient",
                        "potential_savings": avg_cost_per_task * 0.8  # ~80% savings
                    })
                elif data["current_tier"] == "sonnet" and data["success_rate"] > 0.9:
                    recommendations.append({
                        "agent_id": data["agent_id"],
                        "current_tier": "sonnet", 
                        "recommended_tier": "haiku",
                        "reason": f"Very high success rate ({data['success_rate']:.1%}) suggests Haiku may work",
                        "potential_savings": avg_cost_per_task * 0.9  # ~90% savings
                    })
        
        return recommendations


class ROIAnalyzer:
    """Analyze ROI and provide optimization recommendations"""
    
    def __init__(self, cost_db: CostDatabase):
        self.cost_db = cost_db
        self.roi_cache: Dict[str, ROIMetrics] = {}
    
    async def calculate_agent_roi(self, agent_id: str, project_id: str, 
                                days_back: int = 7) -> ROIMetrics:
        """Calculate ROI metrics for an agent"""
        cache_key = f"{agent_id}_{project_id}"
        
        # Get recent cost records
        start_date = datetime.now() - timedelta(days=days_back)
        records = await self.cost_db.get_cost_records(
            project_id=project_id,
            agent_id=agent_id,
            start_date=start_date
        )
        
        if not records:
            # Return empty metrics if no records
            return ROIMetrics(agent_id=agent_id, project_id=project_id)
        
        # Initialize or update ROI metrics
        if cache_key not in self.roi_cache:
            self.roi_cache[cache_key] = ROIMetrics(agent_id=agent_id, project_id=project_id)
        
        metrics = self.roi_cache[cache_key]
        
        # Reset for recalculation
        metrics.total_cost = 0.0
        metrics.total_tasks = 0
        metrics.successful_tasks = 0
        metrics.failed_tasks = 0
        
        # Process all records
        for record in records:
            metrics.update_metrics(record.cost, record.success, record.execution_time)
        
        return metrics
    
    async def get_project_roi_summary(self, project_id: str) -> Dict[str, Any]:
        """Get ROI summary for entire project"""
        # Get all agents in project
        records = await self.cost_db.get_cost_records(
            project_id=project_id,
            start_date=datetime.now() - timedelta(days=7)
        )
        
        if not records:
            return {"project_id": project_id, "agents": [], "total_cost": 0, "total_tasks": 0}
        
        # Group by agent
        agent_ids = list(set(r.agent_id for r in records))
        agent_rois = []
        
        for agent_id in agent_ids:
            roi_metrics = await self.calculate_agent_roi(agent_id, project_id)
            if roi_metrics.total_tasks > 0:
                agent_rois.append({
                    "agent_id": agent_id,
                    "roi_score": roi_metrics.roi_score,
                    "success_rate": roi_metrics.success_rate,
                    "avg_cost_per_task": roi_metrics.avg_cost_per_task,
                    "total_cost": roi_metrics.total_cost,
                    "total_tasks": roi_metrics.total_tasks,
                    "efficiency_score": roi_metrics.efficiency_score
                })
        
        # Calculate project totals
        total_cost = sum(a["total_cost"] for a in agent_rois)
        total_tasks = sum(a["total_tasks"] for a in agent_rois)
        avg_success_rate = sum(a["success_rate"] for a in agent_rois) / len(agent_rois) if agent_rois else 0
        
        return {
            "project_id": project_id,
            "agents": sorted(agent_rois, key=lambda x: x["roi_score"], reverse=True),
            "total_cost": total_cost,
            "total_tasks": total_tasks,
            "avg_success_rate": avg_success_rate,
            "cost_per_task": total_cost / total_tasks if total_tasks > 0 else 0,
            "top_performer": agent_rois[0] if agent_rois else None,
            "needs_attention": [a for a in agent_rois if a["roi_score"] < 0.5]
        }
    
    async def get_optimization_recommendations(self, project_id: str) -> List[Dict[str, Any]]:
        """Get optimization recommendations for project"""
        roi_summary = await self.get_project_roi_summary(project_id)
        recommendations = []
        
        for agent_data in roi_summary["agents"]:
            agent_id = agent_data["agent_id"]
            roi_metrics = await self.calculate_agent_roi(agent_id, project_id)
            
            agent_recommendations = roi_metrics.get_optimization_recommendation()
            recommendations.extend(agent_recommendations["recommendations"])
        
        return recommendations


class CostOptimizer:
    """Main cost optimization engine implementing F-ITR-002"""
    
    def __init__(self, db_path: str = "/tmp/archon_costs.db"):
        self.cost_calculator = CostCalculator()
        self.cost_db = CostDatabase(db_path)
        self.budget_monitor = BudgetMonitor(self.cost_db)
        self.roi_analyzer = ROIAnalyzer(self.cost_db)
        self.optimization_enabled = True
        
    async def track_task_cost(self, agent_id: str, task_id: str, project_id: str,
                            model_tier: str, input_tokens: int, output_tokens: int,
                            execution_time: float, success: bool = True,
                            metadata: Dict[str, Any] = None) -> CostRecord:
        """Track cost for executed task"""
        # Create cost record
        cost_record = CostRecord.create_from_task(
            agent_id=agent_id,
            task_id=task_id,
            project_id=project_id,
            model_tier=model_tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            execution_time=execution_time,
            success=success,
            metadata=metadata
        )
        
        # Store in database
        await self.cost_db.store_cost_record(cost_record)
        
        # Check budget constraints
        budget_status = await self.budget_monitor.check_budget_status(project_id)
        
        if not budget_status["within_limits"] and self.optimization_enabled:
            # Trigger optimization if over budget
            await self._trigger_automatic_optimization(project_id, budget_status)
        
        logger.info(f"Tracked cost for {agent_id}: ${cost_record.cost:.6f} ({model_tier})")
        
        return cost_record
    
    async def set_budget_constraints(self, project_id: str, daily_limit: float,
                                   monthly_limit: float, per_agent_limit: Optional[float] = None,
                                   per_tier_limits: Dict[str, float] = None) -> bool:
        """Set budget constraints for project"""
        constraint = BudgetConstraint(
            constraint_id=str(uuid.uuid4()),
            project_id=project_id,
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
            per_agent_limit=per_agent_limit,
            per_tier_limits=per_tier_limits or {}
        )
        
        success = await self.cost_db.store_budget_constraint(constraint)
        
        if success:
            logger.info(f"Set budget constraints for {project_id}: "
                       f"${daily_limit}/day, ${monthly_limit}/month")
        
        return success
    
    async def get_cost_dashboard(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive cost dashboard"""
        # Budget status
        budget_info = {}
        if project_id:
            budget_info = await self.budget_monitor.check_budget_status(project_id)
        
        # ROI analysis
        roi_info = {}
        if project_id:
            roi_info = await self.roi_analyzer.get_project_roi_summary(project_id)
        
        # Recent cost trends
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        recent_records = await self.cost_db.get_cost_records(
            project_id=project_id,
            start_date=start_date
        )
        
        # Aggregate by day and tier
        daily_costs = {}
        tier_costs = {}
        
        for record in recent_records:
            day = record.timestamp.date().isoformat()
            if day not in daily_costs:
                daily_costs[day] = 0
            daily_costs[day] += record.cost
            
            if record.model_tier not in tier_costs:
                tier_costs[record.model_tier] = 0
            tier_costs[record.model_tier] += record.cost
        
        return {
            "project_id": project_id,
            "budget_status": budget_info,
            "roi_analysis": roi_info,
            "cost_trends": {
                "daily_costs": daily_costs,
                "costs_by_tier": tier_costs,
                "total_cost_7_days": sum(r.cost for r in recent_records),
                "total_tasks_7_days": len(recent_records)
            },
            "optimization_recommendations": await self.roi_analyzer.get_optimization_recommendations(project_id) if project_id else [],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _trigger_automatic_optimization(self, project_id: str, budget_status: Dict[str, Any]):
        """Trigger automatic optimization when budget constraints are exceeded"""
        if not self.optimization_enabled:
            return
        
        logger.warning(f"Budget constraints exceeded for {project_id}, triggering optimization")
        
        # Get tier downgrade recommendations
        tier_recommendations = await self.budget_monitor.recommend_tier_changes(project_id)
        
        optimization_actions = []
        
        for recommendation in tier_recommendations:
            optimization_actions.append({
                "type": "tier_downgrade",
                "agent_id": recommendation["agent_id"],
                "from_tier": recommendation["current_tier"],
                "to_tier": recommendation["recommended_tier"],
                "reason": recommendation["reason"],
                "potential_savings": recommendation["potential_savings"]
            })
        
        if optimization_actions:
            logger.info(f"Generated {len(optimization_actions)} optimization actions for {project_id}")
            
            # In a real implementation, these actions would be applied automatically
            # For now, we log them as recommendations
            for action in optimization_actions:
                logger.info(f"Optimization recommendation: {action}")
    
    def enable_optimization(self, enabled: bool = True):
        """Enable or disable automatic optimization"""
        self.optimization_enabled = enabled
        logger.info(f"Automatic optimization {'enabled' if enabled else 'disabled'}")
    
    def add_budget_alert_callback(self, callback: Callable):
        """Add callback for budget alerts"""
        self.budget_monitor.add_alert_callback(callback)


# Example usage and testing
async def main():
    """Example usage of Cost Optimization Engine"""
    print("💰 Archon v3.0 Cost Optimization Engine")
    print("=" * 45)
    
    # Initialize cost optimizer
    optimizer = CostOptimizer("/tmp/archon_cost_demo.db")
    
    # Set budget constraints
    project_id = "cost-optimization-demo"
    print(f"\n💳 Setting budget constraints for {project_id}")
    await optimizer.set_budget_constraints(
        project_id=project_id,
        daily_limit=2.00,      # $2.00 per day
        monthly_limit=50.00,   # $50.00 per month
        per_agent_limit=0.50   # $0.50 per agent per day
    )
    
    # Add budget alert callback
    async def budget_alert_handler(alert):
        print(f"🚨 BUDGET ALERT: {alert['message']}")
    
    optimizer.add_budget_alert_callback(budget_alert_handler)
    
    # Simulate task executions with costs
    print(f"\n📊 Simulating task executions...")
    
    test_tasks = [
        ("architect-001", "opus", 5000, 2000, True, "Complex system design"),
        ("developer-002", "sonnet", 3000, 1500, True, "API implementation"),
        ("formatter-003", "haiku", 1000, 500, True, "Code formatting"),
        ("developer-002", "sonnet", 2800, 1400, False, "Failed database migration"),
        ("architect-001", "opus", 4500, 2200, True, "Performance optimization"),
    ]
    
    cost_records = []
    for i, (agent_id, tier, input_tokens, output_tokens, success, description) in enumerate(test_tasks):
        cost_record = await optimizer.track_task_cost(
            agent_id=agent_id,
            task_id=f"{project_id}-task-{i+1}",
            project_id=project_id,
            model_tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            execution_time=60.0 + (i * 15),
            success=success,
            metadata={"description": description}
        )
        cost_records.append(cost_record)
    
    # Get cost dashboard
    print(f"\n📋 Cost Dashboard:")
    dashboard = await optimizer.get_cost_dashboard(project_id)
    
    print(f"Budget Status:")
    budget = dashboard["budget_status"]
    print(f"  Daily Usage: ${budget['daily_usage']:.6f} / ${budget['daily_limit']:.2f}")
    print(f"  Monthly Usage: ${budget['monthly_usage']:.6f} / ${budget['monthly_limit']:.2f}")
    print(f"  Within Limits: {budget['within_limits']}")
    
    if budget["alerts"]:
        print(f"  Alerts: {len(budget['alerts'])} active")
        for alert in budget["alerts"]:
            print(f"    - {alert['type'].upper()}: {alert['message']}")
    
    print(f"\nROI Analysis:")
    roi = dashboard["roi_analysis"]
    if roi.get("agents"):
        print(f"  Total Cost (7 days): ${roi['total_cost']:.6f}")
        print(f"  Total Tasks: {roi['total_tasks']}")
        print(f"  Average Success Rate: {roi['avg_success_rate']:.1%}")
        
        if roi.get("top_performer"):
            top = roi["top_performer"]
            print(f"  Top Performer: {top['agent_id']} (ROI: {top['roi_score']:.2f})")
        
        if roi.get("needs_attention"):
            print(f"  Agents Needing Attention: {len(roi['needs_attention'])}")
    
    print(f"\nCost Trends:")
    trends = dashboard["cost_trends"]
    print(f"  Total Cost (7 days): ${trends['total_cost_7_days']:.6f}")
    print(f"  Costs by Tier:")
    for tier, cost in trends["costs_by_tier"].items():
        print(f"    {tier.capitalize()}: ${cost:.6f}")
    
    if dashboard["optimization_recommendations"]:
        print(f"\nOptimization Recommendations: {len(dashboard['optimization_recommendations'])}")
        for rec in dashboard["optimization_recommendations"][:3]:  # Show first 3
            print(f"  - {rec['type']}: {rec['message']}")
    
    print(f"\n✅ Cost Optimization Engine demo completed!")


if __name__ == "__main__":
    asyncio.run(main())