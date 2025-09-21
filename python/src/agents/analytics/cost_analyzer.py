"""
Cost Analyzer Module
Resource usage analysis and cost optimization recommendations
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

from .analytics_engine import AnalyticsEngine


class ResourceType(Enum):
    """Types of resources to track costs"""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
    DATABASE = "database"
    ML_INFERENCE = "ml_inference"
    GPU = "gpu"
    BANDWIDTH = "bandwidth"
    LICENSES = "licenses"


class CostModel(Enum):
    """Cost calculation models"""
    PAY_AS_YOU_GO = "pay_as_you_go"
    RESERVED = "reserved"
    SPOT = "spot"
    SUBSCRIPTION = "subscription"
    TIERED = "tiered"
    CUSTOM = "custom"


class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    RIGHT_SIZING = "right_sizing"
    RESERVED_INSTANCES = "reserved_instances"
    SPOT_INSTANCES = "spot_instances"
    AUTO_SCALING = "auto_scaling"
    CACHING = "caching"
    COMPRESSION = "compression"
    ARCHIVING = "archiving"
    DEDUPLICATION = "deduplication"
    BATCH_PROCESSING = "batch_processing"
    SERVERLESS = "serverless"


@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    resource_id: str
    resource_type: ResourceType
    usage_amount: float
    unit: str
    timestamp: datetime
    component: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostItem:
    """Individual cost item"""
    item_id: str
    resource_type: ResourceType
    amount: float
    currency: str = "USD"
    period_start: datetime = field(default_factory=datetime.now)
    period_end: Optional[datetime] = None
    description: str = ""
    component: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBreakdown:
    """Cost breakdown analysis"""
    total_cost: float
    currency: str
    period_start: datetime
    period_end: datetime
    by_resource: Dict[ResourceType, float]
    by_component: Dict[str, float]
    by_tag: Dict[str, Dict[str, float]]
    trends: Dict[str, List[float]]
    projections: Dict[str, float]


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    recommendation_id: str
    strategy: OptimizationStrategy
    resource_type: ResourceType
    estimated_savings: float
    estimated_savings_percent: float
    implementation_effort: str  # low, medium, high
    priority: int  # 1-5, 1 being highest
    description: str
    action_items: List[str]
    risks: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Budget configuration"""
    budget_id: str
    name: str
    amount: float
    currency: str
    period: str  # monthly, quarterly, yearly
    threshold_alerts: List[float]  # percentages
    components: List[str]
    resource_types: List[ResourceType]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostAnalyzer:
    """
    Comprehensive cost analysis and optimization system
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics_engine = analytics_engine
        self.resource_usage: Dict[str, List[ResourceUsage]] = defaultdict(list)
        self.cost_items: Dict[str, CostItem] = {}
        self.budgets: Dict[str, Budget] = {}
        self.recommendations: List[OptimizationRecommendation] = []
        self.cost_history: List[CostBreakdown] = []
        
        # Pricing configuration (example rates)
        self.pricing = {
            ResourceType.COMPUTE: {
                "unit": "vCPU-hour",
                "rate": 0.0416,  # $0.0416 per vCPU-hour
                "model": CostModel.PAY_AS_YOU_GO
            },
            ResourceType.MEMORY: {
                "unit": "GB-hour",
                "rate": 0.0052,  # $0.0052 per GB-hour
                "model": CostModel.PAY_AS_YOU_GO
            },
            ResourceType.STORAGE: {
                "unit": "GB-month",
                "rate": 0.023,  # $0.023 per GB-month
                "model": CostModel.TIERED,
                "tiers": [
                    {"limit": 50, "rate": 0.023},
                    {"limit": 500, "rate": 0.022},
                    {"limit": float('inf'), "rate": 0.021}
                ]
            },
            ResourceType.NETWORK: {
                "unit": "GB",
                "rate": 0.09,  # $0.09 per GB
                "model": CostModel.PAY_AS_YOU_GO
            },
            ResourceType.API_CALLS: {
                "unit": "1M requests",
                "rate": 0.40,  # $0.40 per million requests
                "model": CostModel.TIERED
            },
            ResourceType.DATABASE: {
                "unit": "RU-hour",
                "rate": 0.008,  # $0.008 per RU-hour
                "model": CostModel.PAY_AS_YOU_GO
            },
            ResourceType.ML_INFERENCE: {
                "unit": "1K predictions",
                "rate": 0.07,  # $0.07 per 1000 predictions
                "model": CostModel.PAY_AS_YOU_GO
            },
            ResourceType.GPU: {
                "unit": "GPU-hour",
                "rate": 0.90,  # $0.90 per GPU-hour
                "model": CostModel.SPOT,
                "spot_discount": 0.70  # 70% discount for spot instances
            }
        }
        
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start cost monitoring background tasks"""
        asyncio.create_task(self._monitor_usage())
        asyncio.create_task(self._check_budgets())
        asyncio.create_task(self._generate_recommendations())
    
    async def _monitor_usage(self):
        """Monitor resource usage continuously"""
        while True:
            try:
                # Collect usage metrics
                await self._collect_compute_usage()
                await self._collect_memory_usage()
                await self._collect_storage_usage()
                await self._collect_api_usage()
                
                # Calculate current costs
                await self._calculate_current_costs()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"Usage monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _check_budgets(self):
        """Check budget thresholds"""
        while True:
            try:
                for budget in self.budgets.values():
                    current_spend = await self._get_current_spend(budget)
                    budget_utilization = (current_spend / budget.amount) * 100
                    
                    # Check threshold alerts
                    for threshold in budget.threshold_alerts:
                        if budget_utilization >= threshold:
                            await self._trigger_budget_alert(budget, budget_utilization)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                print(f"Budget check error: {e}")
                await asyncio.sleep(3600)
    
    async def _generate_recommendations(self):
        """Generate cost optimization recommendations"""
        while True:
            try:
                self.recommendations.clear()
                
                # Analyze usage patterns
                await self._analyze_compute_optimization()
                await self._analyze_storage_optimization()
                await self._analyze_network_optimization()
                await self._analyze_database_optimization()
                
                # Sort recommendations by priority
                self.recommendations.sort(key=lambda x: (x.priority, -x.estimated_savings))
                
                await asyncio.sleep(86400)  # Generate daily
                
            except Exception as e:
                print(f"Recommendation generation error: {e}")
                await asyncio.sleep(86400)
    
    async def track_usage(self, resource_type: ResourceType, amount: float,
                         component: str, unit: Optional[str] = None,
                         tags: Optional[Dict[str, str]] = None) -> ResourceUsage:
        """Track resource usage"""
        import uuid
        
        usage = ResourceUsage(
            resource_id=str(uuid.uuid4()),
            resource_type=resource_type,
            usage_amount=amount,
            unit=unit or self.pricing[resource_type]["unit"],
            timestamp=datetime.now(),
            component=component,
            tags=tags or {}
        )
        
        self.resource_usage[component].append(usage)
        
        # Calculate cost
        cost = await self._calculate_cost(usage)
        if cost > 0:
            await self._record_cost(resource_type, cost, component, tags)
        
        return usage
    
    async def _calculate_cost(self, usage: ResourceUsage) -> float:
        """Calculate cost for resource usage"""
        if usage.resource_type not in self.pricing:
            return 0.0
        
        pricing_config = self.pricing[usage.resource_type]
        base_rate = pricing_config["rate"]
        model = pricing_config["model"]
        
        if model == CostModel.PAY_AS_YOU_GO:
            return usage.usage_amount * base_rate
        
        elif model == CostModel.TIERED:
            if "tiers" in pricing_config:
                total_cost = 0.0
                remaining = usage.usage_amount
                
                for tier in pricing_config["tiers"]:
                    tier_usage = min(remaining, tier["limit"])
                    total_cost += tier_usage * tier["rate"]
                    remaining -= tier_usage
                    if remaining <= 0:
                        break
                
                return total_cost
            else:
                return usage.usage_amount * base_rate
        
        elif model == CostModel.SPOT:
            discount = pricing_config.get("spot_discount", 1.0)
            return usage.usage_amount * base_rate * discount
        
        else:
            return usage.usage_amount * base_rate
    
    async def _record_cost(self, resource_type: ResourceType, amount: float,
                         component: str, tags: Optional[Dict[str, str]] = None):
        """Record a cost item"""
        import uuid
        
        cost_item = CostItem(
            item_id=str(uuid.uuid4()),
            resource_type=resource_type,
            amount=amount,
            component=component,
            tags=tags or {}
        )
        
        self.cost_items[cost_item.item_id] = cost_item
        
        # Record metric
        await self.analytics_engine.record_metric(
            f"cost.{resource_type.value}",
            amount,
            tags={"component": component, **(tags or {})}
        )
    
    async def get_cost_breakdown(self, start_date: datetime,
                                end_date: datetime) -> CostBreakdown:
        """Get cost breakdown for period"""
        # Filter cost items for period
        period_costs = [
            item for item in self.cost_items.values()
            if start_date <= item.period_start <= end_date
        ]
        
        # Calculate breakdowns
        by_resource = defaultdict(float)
        by_component = defaultdict(float)
        by_tag = defaultdict(lambda: defaultdict(float))
        
        total_cost = 0.0
        
        for item in period_costs:
            total_cost += item.amount
            by_resource[item.resource_type] += item.amount
            by_component[item.component] += item.amount
            
            for tag_key, tag_value in item.tags.items():
                by_tag[tag_key][tag_value] += item.amount
        
        # Calculate trends
        trends = await self._calculate_cost_trends(start_date, end_date)
        
        # Generate projections
        projections = await self._project_costs(total_cost, end_date)
        
        breakdown = CostBreakdown(
            total_cost=total_cost,
            currency="USD",
            period_start=start_date,
            period_end=end_date,
            by_resource=dict(by_resource),
            by_component=dict(by_component),
            by_tag=dict(by_tag),
            trends=trends,
            projections=projections
        )
        
        self.cost_history.append(breakdown)
        return breakdown
    
    async def create_budget(self, name: str, amount: float,
                          period: str = "monthly",
                          components: Optional[List[str]] = None,
                          resource_types: Optional[List[ResourceType]] = None,
                          threshold_alerts: Optional[List[float]] = None) -> Budget:
        """Create a budget"""
        import uuid
        
        budget = Budget(
            budget_id=str(uuid.uuid4()),
            name=name,
            amount=amount,
            currency="USD",
            period=period,
            threshold_alerts=threshold_alerts or [50, 75, 90, 100],
            components=components or [],
            resource_types=resource_types or list(ResourceType),
            created_at=datetime.now()
        )
        
        self.budgets[budget.budget_id] = budget
        return budget
    
    async def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Get current optimization recommendations"""
        if not self.recommendations:
            await self._generate_recommendations()
        
        return self.recommendations
    
    async def _analyze_compute_optimization(self):
        """Analyze compute resource optimization opportunities"""
        import uuid
        
        # Analyze CPU utilization
        cpu_metrics = await self.analytics_engine.get_metrics(
            metric_names=["system.cpu.utilization"],
            time_range=timedelta(days=7)
        )
        
        if cpu_metrics:
            avg_utilization = np.mean([m["value"] for m in cpu_metrics])
            
            if avg_utilization < 30:
                # Recommend downsizing
                current_cost = await self._estimate_current_compute_cost()
                potential_savings = current_cost * 0.40  # 40% savings potential
                
                self.recommendations.append(OptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=OptimizationStrategy.RIGHT_SIZING,
                    resource_type=ResourceType.COMPUTE,
                    estimated_savings=potential_savings,
                    estimated_savings_percent=40,
                    implementation_effort="medium",
                    priority=1,
                    description="Low CPU utilization detected. Consider downsizing compute instances.",
                    action_items=[
                        "Review instance utilization metrics",
                        "Identify over-provisioned instances",
                        "Test with smaller instance types",
                        "Implement auto-scaling policies"
                    ],
                    risks=[
                        "Potential performance impact during peak loads",
                        "May require application adjustments"
                    ]
                ))
            
            elif avg_utilization < 50:
                # Recommend spot instances
                current_cost = await self._estimate_current_compute_cost()
                potential_savings = current_cost * 0.30  # 30% savings with spot
                
                self.recommendations.append(OptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=OptimizationStrategy.SPOT_INSTANCES,
                    resource_type=ResourceType.COMPUTE,
                    estimated_savings=potential_savings,
                    estimated_savings_percent=30,
                    implementation_effort="low",
                    priority=2,
                    description="Consider using spot instances for non-critical workloads.",
                    action_items=[
                        "Identify fault-tolerant workloads",
                        "Implement spot instance handling",
                        "Set up fallback to on-demand instances"
                    ],
                    risks=[
                        "Instance interruptions possible",
                        "Requires application resilience"
                    ]
                ))
    
    async def _analyze_storage_optimization(self):
        """Analyze storage optimization opportunities"""
        import uuid
        
        # Analyze storage access patterns
        storage_metrics = await self.analytics_engine.get_metrics(
            metric_names=["storage.access_frequency"],
            time_range=timedelta(days=30)
        )
        
        if storage_metrics:
            # Identify cold data
            cold_data_percentage = 0.0  # Would calculate from metrics
            
            if cold_data_percentage > 30:
                current_storage_cost = await self._estimate_current_storage_cost()
                potential_savings = current_storage_cost * 0.60 * (cold_data_percentage / 100)
                
                self.recommendations.append(OptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=OptimizationStrategy.ARCHIVING,
                    resource_type=ResourceType.STORAGE,
                    estimated_savings=potential_savings,
                    estimated_savings_percent=int(60 * (cold_data_percentage / 100)),
                    implementation_effort="low",
                    priority=2,
                    description=f"{cold_data_percentage:.1f}% of data is cold. Archive to cheaper storage.",
                    action_items=[
                        "Identify infrequently accessed data",
                        "Set up lifecycle policies",
                        "Move cold data to archive storage",
                        "Implement data retrieval process"
                    ],
                    risks=[
                        "Increased retrieval time for archived data",
                        "Potential retrieval costs"
                    ]
                ))
    
    async def _analyze_network_optimization(self):
        """Analyze network optimization opportunities"""
        import uuid
        
        # Check for data transfer patterns
        network_metrics = await self.analytics_engine.get_metrics(
            metric_names=["network.egress"],
            time_range=timedelta(days=7)
        )
        
        if network_metrics:
            total_egress = sum(m["value"] for m in network_metrics)
            
            if total_egress > 1000:  # GB
                current_cost = total_egress * self.pricing[ResourceType.NETWORK]["rate"]
                potential_savings = current_cost * 0.40  # CDN can save 40%
                
                self.recommendations.append(OptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=OptimizationStrategy.CACHING,
                    resource_type=ResourceType.NETWORK,
                    estimated_savings=potential_savings,
                    estimated_savings_percent=40,
                    implementation_effort="medium",
                    priority=2,
                    description="High network egress detected. Implement CDN/caching.",
                    action_items=[
                        "Identify frequently accessed content",
                        "Implement CDN for static assets",
                        "Set up edge caching",
                        "Optimize API response caching"
                    ],
                    risks=[
                        "Cache invalidation complexity",
                        "Potential stale data issues"
                    ]
                ))
    
    async def _analyze_database_optimization(self):
        """Analyze database optimization opportunities"""
        import uuid
        
        # Check database utilization
        db_metrics = await self.analytics_engine.get_metrics(
            metric_names=["database.connections", "database.queries_per_second"],
            time_range=timedelta(days=7)
        )
        
        if db_metrics:
            # Analyze connection pooling opportunities
            connection_metrics = [m for m in db_metrics if "connections" in m.get("name", "")]
            
            if connection_metrics:
                avg_connections = np.mean([m["value"] for m in connection_metrics])
                
                if avg_connections < 10:
                    current_db_cost = await self._estimate_current_database_cost()
                    potential_savings = current_db_cost * 0.25
                    
                    self.recommendations.append(OptimizationRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        strategy=OptimizationStrategy.SERVERLESS,
                        resource_type=ResourceType.DATABASE,
                        estimated_savings=potential_savings,
                        estimated_savings_percent=25,
                        implementation_effort="high",
                        priority=3,
                        description="Low database utilization. Consider serverless database.",
                        action_items=[
                            "Evaluate serverless database options",
                            "Test application compatibility",
                            "Implement connection pooling",
                            "Migrate to serverless architecture"
                        ],
                        risks=[
                            "Cold start latency",
                            "Connection limit constraints",
                            "Migration complexity"
                        ]
                    ))
    
    async def _collect_compute_usage(self):
        """Collect compute usage metrics"""
        # Would integrate with actual monitoring systems
        pass
    
    async def _collect_memory_usage(self):
        """Collect memory usage metrics"""
        # Would integrate with actual monitoring systems
        pass
    
    async def _collect_storage_usage(self):
        """Collect storage usage metrics"""
        # Would integrate with actual monitoring systems
        pass
    
    async def _collect_api_usage(self):
        """Collect API usage metrics"""
        # Would integrate with actual monitoring systems
        pass
    
    async def _calculate_current_costs(self):
        """Calculate current period costs"""
        # Would calculate based on current usage
        pass
    
    async def _get_current_spend(self, budget: Budget) -> float:
        """Get current spend for budget"""
        total_spend = 0.0
        
        for item in self.cost_items.values():
            # Check if item matches budget criteria
            if budget.components and item.component not in budget.components:
                continue
            if budget.resource_types and item.resource_type not in budget.resource_types:
                continue
            
            # Check if item is in current budget period
            if self._is_in_budget_period(item.period_start, budget):
                total_spend += item.amount
        
        return total_spend
    
    def _is_in_budget_period(self, date: datetime, budget: Budget) -> bool:
        """Check if date is in current budget period"""
        now = datetime.now()
        
        if budget.period == "monthly":
            return date.year == now.year and date.month == now.month
        elif budget.period == "quarterly":
            return date.year == now.year and (date.month - 1) // 3 == (now.month - 1) // 3
        elif budget.period == "yearly":
            return date.year == now.year
        
        return False
    
    async def _trigger_budget_alert(self, budget: Budget, utilization: float):
        """Trigger budget alert"""
        alert_message = f"Budget '{budget.name}' is at {utilization:.1f}% utilization"
        
        await self.analytics_engine.create_alert(
            name=f"budget_alert_{budget.budget_id}",
            condition=f"utilization >= {utilization}",
            message=alert_message,
            severity="warning" if utilization < 90 else "critical"
        )
    
    async def _calculate_cost_trends(self, start_date: datetime,
                                   end_date: datetime) -> Dict[str, List[float]]:
        """Calculate cost trends"""
        trends = {
            "daily": [],
            "weekly": [],
            "monthly": []
        }
        
        # Would calculate actual trends from historical data
        # Placeholder implementation
        days = (end_date - start_date).days
        trends["daily"] = [np.random.uniform(100, 200) for _ in range(min(days, 30))]
        trends["weekly"] = [np.random.uniform(700, 1400) for _ in range(min(days // 7, 12))]
        trends["monthly"] = [np.random.uniform(3000, 6000) for _ in range(min(days // 30, 12))]
        
        return trends
    
    async def _project_costs(self, current_cost: float,
                          end_date: datetime) -> Dict[str, float]:
        """Project future costs"""
        # Simple linear projection
        days_in_month = 30
        current_day = datetime.now().day
        remaining_days = days_in_month - current_day
        
        daily_rate = current_cost / current_day if current_day > 0 else 0
        
        return {
            "end_of_month": current_cost + (daily_rate * remaining_days),
            "next_month": daily_rate * 30,
            "next_quarter": daily_rate * 90,
            "next_year": daily_rate * 365
        }
    
    async def _estimate_current_compute_cost(self) -> float:
        """Estimate current compute cost"""
        # Would calculate from actual usage
        return 1000.0  # Placeholder
    
    async def _estimate_current_storage_cost(self) -> float:
        """Estimate current storage cost"""
        # Would calculate from actual usage
        return 500.0  # Placeholder
    
    async def _estimate_current_database_cost(self) -> float:
        """Estimate current database cost"""
        # Would calculate from actual usage
        return 750.0  # Placeholder
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        total_cost = sum(item.amount for item in self.cost_items.values())
        
        return {
            "total_cost": total_cost,
            "currency": "USD",
            "num_cost_items": len(self.cost_items),
            "num_budgets": len(self.budgets),
            "num_recommendations": len(self.recommendations),
            "top_recommendations": [
                {
                    "strategy": rec.strategy.value,
                    "savings": rec.estimated_savings,
                    "description": rec.description
                }
                for rec in self.recommendations[:5]
            ]
        }