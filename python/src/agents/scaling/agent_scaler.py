"""
Dynamic Agent Scaling System for Enterprise Deployments

Implements intelligent agent scaling based on load and performance:
- Horizontal scaling for agent instances
- Vertical scaling for agent capabilities
- Predictive scaling based on historical data
- Cost-optimized scaling strategies
- Multi-dimensional scaling metrics

Target Performance:
- 0-1000 agent scaling in <30 seconds
- <5% resource over-provisioning
- Predictive scaling accuracy >90%
- Automatic cost optimization
- Zero-downtime scaling operations
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
import heapq
import weakref

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling operation directions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_CHANGE = "no_change"

class ScalingTrigger(Enum):
    """Reasons for scaling decisions"""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    SCHEDULED = "scheduled"
    PREDICTIVE = "predictive"
    MANUAL = "manual"
    COST_OPTIMIZATION = "cost_optimization"

class AgentState(Enum):
    """Agent lifecycle states"""
    CREATING = "creating"
    STARTING = "starting"
    ACTIVE = "active"
    IDLE = "idle"
    SCALING_DOWN = "scaling_down"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"

@dataclass
class AgentInstance:
    """Individual agent instance"""
    instance_id: str
    agent_type: str
    model_tier: str
    state: AgentState
    created_at: float
    last_activity: float
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    tasks_completed: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uptime_hours(self) -> float:
        """Get instance uptime in hours"""
        return (time.time() - self.created_at) / 3600

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (0-100)"""
        if self.tasks_completed == 0:
            return 0.0

        # Factors: task completion, response time, error rate, resource usage
        task_score = min(100, self.tasks_completed * 10)
        response_score = max(0, 100 - self.avg_response_time_ms / 10)
        error_score = max(0, 100 - self.error_rate * 100)
        resource_score = max(0, 100 - (self.cpu_usage + self.memory_usage_mb / 100) / 2)

        return (task_score + response_score + error_score + resource_score) / 4

@dataclass
class ScalingPolicy:
    """Scaling policy configuration"""
    agent_type: str
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization_mb: float = 512.0
    max_response_time_ms: float = 1000.0
    max_error_rate: float = 5.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period_seconds: int = 300
    scale_up_increment: int = 1
    scale_down_decrement: int = 1
    predictive_scaling_enabled: bool = True
    cost_optimization_enabled: bool = True

@dataclass
class ScalingDecision:
    """Scaling decision result"""
    direction: ScalingDirection
    trigger: ScalingTrigger
    current_instances: int
    target_instances: int
    reason: str
    confidence: float
    estimated_impact: Dict[str, Any]
    cost_impact: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingMetrics:
    """Comprehensive scaling metrics"""
    agent_type: str
    current_instances: int
    target_instances: int
    cpu_utilization: float
    memory_utilization_mb: float
    queue_length: int
    avg_response_time_ms: float
    error_rate: float
    throughput_per_sec: float
    cost_per_hour: float
    scaling_events_last_hour: int
    last_scale_time: float
    efficiency_score: float

class PredictiveScaler:
    """Predictive scaling based on historical patterns"""

    def __init__(self, window_size: int = 168):  # 7 days of hourly data
        self.window_size = window_size
        self.historical_data: Dict[str, List[Dict]] = defaultdict(list)
        self.patterns: Dict[str, Dict] = {}
        self.prediction_accuracy = 0.0

    def add_metrics(self, agent_type: str, metrics: Dict[str, Any]):
        """Add metrics to historical data"""
        timestamp = time.time()
        entry = {
            "timestamp": timestamp,
            "metrics": metrics
        }

        self.historical_data[agent_type].append(entry)

        # Keep only recent data
        if len(self.historical_data[agent_type]) > self.window_size:
            self.historical_data[agent_type] = self.historical_data[agent_type][-self.window_size:]

        # Update patterns
        self._update_patterns(agent_type)

    def _update_patterns(self, agent_type: str):
        """Update usage patterns for agent type"""
        data = self.historical_data[agent_type]
        if len(data) < 24:  # Need at least 24 hours of data
            return

        # Extract hourly patterns
        hourly_patterns = defaultdict(list)
        for entry in data:
            hour = datetime.fromtimestamp(entry["timestamp"]).hour
            hourly_patterns[hour].append(entry["metrics"])

        # Calculate patterns
        patterns = {}
        for hour, metrics_list in hourly_patterns.items():
            if metrics_list:
                avg_cpu = statistics.mean(m.get("cpu_utilization", 0) for m in metrics_list)
                avg_memory = statistics.mean(m.get("memory_utilization_mb", 0) for m in metrics_list)
                avg_queue = statistics.mean(m.get("queue_length", 0) for m in metrics_list)

                patterns[hour] = {
                    "avg_cpu_utilization": avg_cpu,
                    "avg_memory_utilization_mb": avg_memory,
                    "avg_queue_length": avg_queue,
                    "sample_count": len(metrics_list)
                }

        self.patterns[agent_type] = patterns

    def predict_load(self, agent_type: str, hours_ahead: int = 1) -> Dict[str, float]:
        """Predict future load based on historical patterns"""
        if agent_type not in self.patterns:
            return {"cpu_utilization": 0.0, "memory_utilization_mb": 0.0, "queue_length": 0}

        current_hour = datetime.now().hour
        target_hour = (current_hour + hours_ahead) % 24
        pattern = self.patterns[agent_type].get(target_hour, {})

        return {
            "cpu_utilization": pattern.get("avg_cpu_utilization", 0.0),
            "memory_utilization_mb": pattern.get("avg_memory_utilization_mb", 0.0),
            "queue_length": pattern.get("avg_queue_length", 0),
            "confidence": min(1.0, pattern.get("sample_count", 0) / 24)
        }

    def get_required_instances(self, agent_type: str, policy: ScalingPolicy, hours_ahead: int = 1) -> Tuple[int, float]:
        """Predict required number of instances"""
        predicted_load = self.predict_load(agent_type, hours_ahead)
        confidence = predicted_load.pop("confidence", 0.0)

        # Calculate required instances based on predicted load
        cpu_instances = max(1, int(predicted_load["cpu_utilization"] / policy.target_cpu_utilization))
        memory_instances = max(1, int(predicted_load["memory_utilization_mb"] / policy.target_memory_utilization_mb))
        queue_instances = max(1, int(predicted_load["queue_length"] / 10))  # 10 tasks per instance

        required_instances = max(cpu_instances, memory_instances, queue_instances)
        required_instances = min(required_instances, policy.max_instances)
        required_instances = max(required_instances, policy.min_instances)

        return required_instances, confidence

class CostOptimizer:
    """Cost optimization for scaling decisions"""

    def __init__(self):
        self.cost_history: List[Dict] = []
        self.cost_thresholds = {
            "high_cost_threshold": 0.8,  # 80% of budget
            "optimize_threshold": 0.6     # 60% of budget
        }

    def calculate_instance_cost(self, instance: AgentInstance) -> float:
        """Calculate hourly cost for an instance"""
        # Base costs by model tier
        tier_costs = {
            "opus": 15.0,    # $15/hour
            "sonnet": 3.0,   # $3/hour
            "haiku": 0.25    # $0.25/hour
        }

        base_cost = tier_costs.get(instance.model_tier.lower(), 3.0)

        # Adjust based on efficiency
        efficiency_multiplier = 1.0 + (1.0 - instance.efficiency_score / 100) * 0.5

        return base_cost * efficiency_multiplier

    def calculate_total_cost(self, instances: List[AgentInstance]) -> float:
        """Calculate total hourly cost for all instances"""
        return sum(self.calculate_instance_cost(instance) for instance in instances)

    def optimize_scaling(self,
                        current_instances: List[AgentInstance],
                        policy: ScalingPolicy,
                        target_count: int,
                        budget_limit: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:

        """Optimize scaling decision based on cost"""
        current_cost = self.calculate_total_cost(current_instances)

        # Estimate cost for target instances
        if target_count > len(current_instances):
            # Scaling up - estimate cost of new instances
            additional_instances = target_count - len(current_instances)
            estimated_new_cost = current_cost + (additional_instances * 3.0)  # Default $3/hour
        else:
            # Scaling down
            removed_instances = len(current_instances) - target_count
            estimated_new_cost = current_cost - (removed_instances * 3.0)

        # Check budget constraints
        if budget_limit and estimated_new_cost > budget_limit:
            # Reduce target to fit budget
            max_affordable = int(budget_limit / 3.0)  # Rough estimate
            target_count = min(target_count, max_affordable)
            estimated_new_cost = target_count * 3.0

        # Calculate cost impact
        cost_change = estimated_new_cost - current_cost
        cost_change_percent = (cost_change / current_cost * 100) if current_cost > 0 else 0

        cost_impact = {
            "current_cost_per_hour": current_cost,
            "estimated_new_cost_per_hour": estimated_new_cost,
            "cost_change_per_hour": cost_change,
            "cost_change_percent": cost_change_percent,
            "budget_utilization": (estimated_new_cost / budget_limit * 100) if budget_limit else 0
        }

        return target_count, cost_impact

class AgentScaler:
    """Main agent scaling system"""

    def __init__(self):
        self.instances: Dict[str, AgentInstance] = {}
        self.policies: Dict[str, ScalingPolicy] = {}
        self.scaling_history: List[ScalingDecision] = []
        self.last_scale_times: Dict[str, float] = {}

        # Components
        self.predictive_scaler = PredictiveScaler()
        self.cost_optimizer = CostOptimizer()

        # Scaling management
        self.running = False
        self.scaling_thread = None
        self.cooldown_periods: Dict[str, float] = {}

        # Metrics collection
        self.metrics_history: Dict[str, List[Dict]] = defaultdict(list)

    def start(self):
        """Start agent scaling system"""
        logger.info("Starting Agent Scaling System")
        self.running = True

        # Start scaling thread
        self.scaling_thread = threading.Thread(target=self._scaling_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()

        logger.info("Agent Scaling System started")

    def stop(self):
        """Stop agent scaling system"""
        logger.info("Stopping Agent Scaling System")
        self.running = False

        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=10)

        logger.info("Agent Scaling System stopped")

    def register_policy(self, agent_type: str, policy: ScalingPolicy):
        """Register scaling policy for agent type"""
        self.policies[agent_type] = policy
        self.last_scale_times[agent_type] = 0
        self.cooldown_periods[agent_type] = policy.cooldown_period_seconds

        logger.info(f"Registered scaling policy for {agent_type}")

    def add_instance(self, instance: AgentInstance):
        """Add new agent instance"""
        self.instances[instance.instance_id] = instance
        logger.info(f"Added agent instance {instance.instance_id}")

    def remove_instance(self, instance_id: str):
        """Remove agent instance"""
        if instance_id in self.instances:
            instance = self.instances.pop(instance_id)
            logger.info(f"Removed agent instance {instance_id}")

    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """Update metrics for an instance"""
        if instance_id in self.instances:
            instance = self.instances[instance_id]

            # Update metrics
            instance.cpu_usage = metrics.get("cpu_usage", instance.cpu_usage)
            instance.memory_usage_mb = metrics.get("memory_usage_mb", instance.memory_usage_mb)
            instance.avg_response_time_ms = metrics.get("avg_response_time_ms", instance.avg_response_time_ms)
            instance.error_rate = metrics.get("error_rate", instance.error_rate)
            instance.tasks_completed = metrics.get("tasks_completed", instance.tasks_completed)
            instance.last_activity = time.time()

            # Update cost
            instance.cost_per_hour = self.cost_optimizer.calculate_instance_cost(instance)

            # Add to historical data for predictive scaling
            self.predictive_scaler.add_metrics(instance.agent_type, metrics)

    def _scaling_loop(self):
        """Main scaling decision loop"""
        while self.running:
            try:
                current_time = time.time()

                # Check each agent type
                for agent_type, policy in self.policies.items():
                    # Check cooldown period
                    last_scale_time = self.last_scale_times.get(agent_type, 0)
                    if current_time - last_scale_time < policy.cooldown_period_seconds:
                        continue

                    # Make scaling decision
                    decision = self._make_scaling_decision(agent_type, policy)

                    if decision.direction != ScalingDirection.NO_CHANGE:
                        self._execute_scaling_decision(decision)

                # Sleep for next cycle
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(30)

    def _make_scaling_decision(self, agent_type: str, policy: ScalingPolicy) -> ScalingDecision:
        """Make scaling decision for agent type"""
        current_instances = [i for i in self.instances.values() if i.agent_type == agent_type]
        current_count = len(current_instances)

        # Collect current metrics
        metrics = self._collect_metrics(agent_type, current_instances)

        # Determine scaling direction and target
        direction, target_count, trigger, reason, confidence = self._determine_scaling_action(
            agent_type, policy, current_instances, metrics
        )

        # Calculate impact
        estimated_impact = self._calculate_scaling_impact(agent_type, current_count, target_count, metrics)

        # Optimize for cost
        target_count, cost_impact = self.cost_optimizer.optimize_scaling(
            current_instances, policy, target_count
        )

        decision = ScalingDecision(
            direction=direction,
            trigger=trigger,
            current_instances=current_count,
            target_instances=target_count,
            reason=reason,
            confidence=confidence,
            estimated_impact=estimated_impact,
            cost_impact=cost_impact
        )

        # Record decision
        self.scaling_history.append(decision)
        self.last_scale_times[agent_type] = time.time()

        logger.info(f"Scaling decision for {agent_type}: {direction.value} from {current_count} to {target_count} instances")
        return decision

    def _collect_metrics(self, agent_type: str, instances: List[AgentInstance]) -> Dict[str, Any]:
        """Collect metrics for scaling decision"""
        if not instances:
            return {
                "cpu_utilization": 0.0,
                "memory_utilization_mb": 0.0,
                "queue_length": 0,
                "avg_response_time_ms": 0.0,
                "error_rate": 0.0,
                "throughput_per_sec": 0.0
            }

        # Calculate aggregate metrics
        total_cpu = sum(i.cpu_usage for i in instances)
        total_memory = sum(i.memory_usage_mb for i in instances)
        total_response_time = sum(i.avg_response_time_ms for i in instances)
        total_tasks = sum(i.tasks_completed for i in instances)
        total_errors = sum(i.error_rate * i.tasks_completed for i in instances) if sum(i.tasks_completed for i in instances) > 0 else 0

        avg_cpu = total_cpu / len(instances)
        avg_memory = total_memory / len(instances)
        avg_response_time = total_response_time / len(instances)
        error_rate = (total_errors / max(total_tasks, 1)) * 100

        # Calculate throughput (tasks per second)
        uptime = min((time.time() - min(i.created_at for i in instances)), 1)
        throughput = total_tasks / uptime

        return {
            "cpu_utilization": avg_cpu,
            "memory_utilization_mb": avg_memory,
            "queue_length": self._get_queue_length(agent_type),
            "avg_response_time_ms": avg_response_time,
            "error_rate": error_rate,
            "throughput_per_sec": throughput
        }

    def _get_queue_length(self, agent_type: str) -> int:
        """Get current queue length for agent type"""
        # This would interface with the actual message queue system
        # For now, return a mock value
        return 0

    def _determine_scaling_action(self,
                                 agent_type: str,
                                 policy: ScalingPolicy,
                                 instances: List[AgentInstance],
                                 metrics: Dict[str, Any]) -> Tuple[ScalingDirection, int, ScalingTrigger, str, float]:

        """Determine scaling action based on metrics"""
        current_count = len(instances)

        # Check scaling up conditions
        scale_up_triggers = []

        if metrics["cpu_utilization"] > policy.scale_up_threshold:
            scale_up_triggers.append((ScalingTrigger.CPU_THRESHOLD, f"High CPU utilization: {metrics['cpu_utilization']:.1f}%"))

        if metrics["memory_utilization_mb"] > policy.target_memory_utilization_mb * 1.5:
            scale_up_triggers.append((ScalingTrigger.MEMORY_THRESHOLD, f"High memory usage: {metrics['memory_utilization_mb']:.1f}MB"))

        if metrics["avg_response_time_ms"] > policy.max_response_time_ms:
            scale_up_triggers.append((ScalingTrigger.RESPONSE_TIME, f"High response time: {metrics['avg_response_time_ms']:.1f}ms"))

        if metrics["error_rate"] > policy.max_error_rate:
            scale_up_triggers.append((ScalingTrigger.ERROR_RATE, f"High error rate: {metrics['error_rate']:.1f}%"))

        if scale_up_triggers:
            # Scale up
            target_count = min(current_count + policy.scale_up_increment, policy.max_instances)
            trigger, reason = scale_up_triggers[0]
            return ScalingDirection.SCALE_UP, target_count, trigger, reason, 0.9

        # Check scaling down conditions
        if current_count > policy.min_instances:
            scale_down_triggers = []

            if metrics["cpu_utilization"] < policy.scale_down_threshold:
                scale_down_triggers.append((ScalingTrigger.CPU_THRESHOLD, f"Low CPU utilization: {metrics['cpu_utilization']:.1f}%"))

            if metrics["memory_utilization_mb"] < policy.target_memory_utilization_mb * 0.5:
                scale_down_triggers.append((ScalingTrigger.MEMORY_THRESHOLD, f"Low memory usage: {metrics['memory_utilization_mb']:.1f}MB"))

            if scale_down_triggers and current_count > policy.min_instances:
                # Scale down
                target_count = max(current_count - policy.scale_down_decrement, policy.min_instances)
                trigger, reason = scale_down_triggers[0]
                return ScalingDirection.SCALE_DOWN, target_count, trigger, reason, 0.8

        # Check predictive scaling
        if policy.predictive_scaling_enabled:
            predicted_instances, confidence = self.predictive_scaler.get_required_instances(agent_type, policy)

            if predicted_instances > current_count and confidence > 0.7:
                return ScalingDirection.SCALE_UP, predicted_instances, ScalingTrigger.PREDICTIVE, f"Predictive scaling based on patterns", confidence

        # No scaling needed
        return ScalingDirection.NO_CHANGE, current_count, ScalingTrigger.MANUAL, "No scaling required", 1.0

    def _calculate_scaling_impact(self,
                                 agent_type: str,
                                 current_count: int,
                                 target_count: int,
                                 current_metrics: Dict[str, Any]) -> Dict[str, Any]:

        """Calculate estimated impact of scaling"""
        if target_count == current_count:
            return {"no_change": True}

        # Estimate new metrics
        scale_factor = target_count / max(current_count, 1)

        estimated_metrics = {
            "cpu_utilization": current_metrics["cpu_utilization"] / scale_factor,
            "memory_utilization_mb": current_metrics["memory_utilization_mb"] / scale_factor,
            "avg_response_time_ms": current_metrics["avg_response_time_ms"] / scale_factor,
            "throughput_per_sec": current_metrics["throughput_per_sec"] * scale_factor
        }

        return {
            "estimated_metrics": estimated_metrics,
            "scale_factor": scale_factor,
            "instance_change": target_count - current_count
        }

    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision"""
        logger.info(f"Executing scaling decision: {decision.direction.value} {decision.agent_type} to {decision.target_instances} instances")

        if decision.direction == ScalingDirection.SCALE_UP:
            self._scale_up(decision)
        elif decision.direction == ScalingDirection.SCALE_DOWN:
            self._scale_down(decision)

    def _scale_up(self, decision: ScalingDecision):
        """Scale up agent instances"""
        agent_type = decision.agent_type
        current_instances = [i for i in self.instances.values() if i.agent_type == agent_type]
        current_count = len(current_instances)
        target_count = decision.target_instances

        instances_to_create = target_count - current_count

        for i in range(instances_to_create):
            instance_id = f"{agent_type}_instance_{int(time.time())}_{i}"
            instance = AgentInstance(
                instance_id=instance_id,
                agent_type=agent_type,
                model_tier="sonnet",  # Default tier
                state=AgentState.CREATING,
                created_at=time.time(),
                last_activity=time.time()
            )

            self.add_instance(instance)

            # In a real implementation, this would trigger actual agent creation
            # For now, mark as active after a short delay
            def activate_instance(instance_id):
                time.sleep(2)  # Simulate creation time
                if instance_id in self.instances:
                    self.instances[instance_id].state = AgentState.ACTIVE

            threading.Thread(target=activate_instance, args=(instance_id,)).start()

    def _scale_down(self, decision: ScalingDecision):
        """Scale down agent instances"""
        agent_type = decision.agent_type
        instances = [i for i in self.instances.values() if i.agent_type == agent_type and i.state == AgentState.ACTIVE]

        # Sort by efficiency (least efficient first)
        instances.sort(key=lambda x: x.efficiency_score)

        instances_to_remove = len(instances) - decision.target_instances

        for i in range(min(instances_to_remove, len(instances))):
            instance = instances[i]
            instance.state = AgentState.SCALING_DOWN

            # Remove instance after termination
            def remove_instance(instance_id):
                time.sleep(5)  # Simulate termination time
                self.remove_instance(instance_id)

            threading.Thread(target=remove_instance, args=(instance.instance_id,)).start()

    def get_scaling_metrics(self, agent_type: str) -> Optional[ScalingMetrics]:
        """Get comprehensive scaling metrics for agent type"""
        instances = [i for i in self.instances.values() if i.agent_type == agent_type]
        if not instances:
            return None

        policy = self.policies.get(agent_type)
        if not policy:
            return None

        # Collect metrics
        metrics = self._collect_metrics(agent_type, instances)

        # Calculate total cost
        total_cost = sum(i.cost_per_hour for i in instances)

        # Calculate efficiency score
        avg_efficiency = sum(i.efficiency_score for i in instances) / len(instances)

        # Count recent scaling events
        recent_events = [d for d in self.scaling_history if d.timestamp > time.time() - 3600 and any(i.agent_type == agent_type for i in self.instances.values())]

        last_scale_time = self.last_scale_times.get(agent_type, 0)

        return ScalingMetrics(
            agent_type=agent_type,
            current_instances=len(instances),
            target_instances=policy.max_instances,  # Could be from predictive scaling
            cpu_utilization=metrics["cpu_utilization"],
            memory_utilization_mb=metrics["memory_utilization_mb"],
            queue_length=metrics["queue_length"],
            avg_response_time_ms=metrics["avg_response_time_ms"],
            error_rate=metrics["error_rate"],
            throughput_per_sec=metrics["throughput_per_sec"],
            cost_per_hour=total_cost,
            scaling_events_last_hour=len(recent_events),
            last_scale_time=last_scale_time,
            efficiency_score=avg_efficiency
        )

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all agent types"""
        all_metrics = {}
        for agent_type in self.policies.keys():
            metrics = self.get_scaling_metrics(agent_type)
            if metrics:
                all_metrics[agent_type] = metrics.__dict__

        # Calculate global metrics
        total_instances = len(self.instances)
        total_cost = sum(i.cost_per_hour for i in self.instances.values())
        total_scaling_events = len([d for d in self.scaling_history if d.timestamp > time.time() - 3600])

        return {
            "agent_metrics": all_metrics,
            "global_metrics": {
                "total_instances": total_instances,
                "total_cost_per_hour": total_cost,
                "scaling_events_last_hour": total_scaling_events,
                "registered_agent_types": len(self.policies)
            }
        }

# Example usage
if __name__ == "__main__":
    # Create agent scaler
    scaler = AgentScaler()

    try:
        # Register scaling policy
        policy = ScalingPolicy(
            agent_type="python_backend_coder",
            min_instances=1,
            max_instances=10,
            target_cpu_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            predictive_scaling_enabled=True
        )

        scaler.register_policy("python_backend_coder", policy)

        # Start scaler
        scaler.start()

        # Add some initial instances
        for i in range(2):
            instance = AgentInstance(
                instance_id=f"python_backend_coder_{i}",
                agent_type="python_backend_coder",
                model_tier="sonnet",
                state=AgentState.ACTIVE,
                created_at=time.time(),
                last_activity=time.time()
            )
            scaler.add_instance(instance)

        # Simulate some load
        time.sleep(10)

        # Update metrics (simulating high load)
        instances = list(scaler.instances.values())
        for instance in instances:
            scaler.update_instance_metrics(instance.instance_id, {
                "cpu_usage": 85.0,
                "memory_usage_mb": 600,
                "avg_response_time_ms": 1200,
                "error_rate": 2.0,
                "tasks_completed": 50
            })

        # Wait for scaling decision
        time.sleep(65)

        # Get metrics
        metrics = scaler.get_all_metrics()
        print(f"Scaling metrics: {metrics}")

    finally:
        scaler.stop()