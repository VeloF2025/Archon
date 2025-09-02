#!/usr/bin/env python3
"""
Meta-Agent for Archon+ System
Dynamic agent spawning, management, and orchestration system
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Union
import uuid

logger = logging.getLogger(__name__)

class MetaAgentDecision(Enum):
    SPAWN_AGENT = "spawn_agent"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REASSIGN_TASK = "reassign_task"
    MERGE_WORKFLOWS = "merge_workflows"
    SPLIT_WORKFLOW = "split_workflow"
    OPTIMIZE_RESOURCES = "optimize_resources"
    DEFER_TASK = "defer_task"

@dataclass
class WorkflowContext:
    """Context for a workflow managed by meta-agent"""
    workflow_id: str
    project_type: str
    complexity_score: float
    required_skills: List[str]
    current_agents: List[str]
    resource_requirements: Dict[str, Any]
    priority: int = 1
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass 
class AgentCapability:
    """Capability profile for an agent"""
    role: str
    skills: List[str]
    complexity_handling: float  # 0.0-1.0
    specialization_areas: List[str]
    performance_metrics: Dict[str, float]
    resource_cost: float
    scaling_factor: float = 1.0

@dataclass
class MetaDecision:
    """Meta-agent decision with reasoning"""
    decision_id: str
    decision_type: MetaAgentDecision
    reasoning: str
    confidence: float
    parameters: Dict[str, Any]
    expected_outcome: str
    risk_assessment: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

class ArchonMetaAgent:
    """
    Meta-Agent for dynamic agent spawning, management, and workflow optimization
    Provides intelligent orchestration and resource management
    """
    
    def __init__(self,
                 agent_pool_manager,
                 orchestrator,
                 config_path: str = "python/src/agents/configs",
                 max_total_agents: int = 50,
                 decision_interval: int = 30):
        
        self.agent_pool_manager = agent_pool_manager
        self.orchestrator = orchestrator
        self.config_path = Path(config_path)
        self.max_total_agents = max_total_agents
        self.decision_interval = decision_interval
        
        # Agent capabilities and profiles
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.workflow_contexts: Dict[str, WorkflowContext] = {}
        self.active_decisions: Dict[str, MetaDecision] = {}
        
        # Decision-making state
        self.decision_history: List[MetaDecision] = []
        self.performance_metrics: Dict[str, float] = {}
        self.resource_optimization_score: float = 0.0
        
        # Meta-agent intelligence
        self.learning_enabled = True
        self.prediction_models: Dict[str, Any] = {}
        self.decision_weights: Dict[str, float] = {
            "efficiency": 0.3,
            "quality": 0.25,
            "cost": 0.2,
            "time": 0.15,
            "risk": 0.1
        }
        
        # Load agent capabilities
        self._load_agent_capabilities()
        
        # Start meta-agent loop
        self.meta_loop_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(f"ArchonMetaAgent initialized with {len(self.agent_capabilities)} agent capabilities")
    
    def _load_agent_capabilities(self):
        """Load agent capability profiles from configurations"""
        if not self.config_path.exists():
            logger.error(f"Config path does not exist: {self.config_path}")
            return
        
        # Load agent registry
        registry_path = self.config_path / "agent_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        # Load individual agent configs
        config_files = list(self.config_path.glob("*.json"))
        config_files = [f for f in config_files if f.name not in ["agent_registry.json", "template_registry.json"]]
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                role = config['role']
                
                # Create capability profile
                capability = AgentCapability(
                    role=role,
                    skills=config.get('skills', []),
                    complexity_handling=self._calculate_complexity_handling(config),
                    specialization_areas=config.get('memory_scope', []),
                    performance_metrics=self._get_performance_baseline(role),
                    resource_cost=self._calculate_resource_cost(config),
                    scaling_factor=self._get_scaling_factor(config)
                )
                
                self.agent_capabilities[role] = capability
                logger.debug(f"Loaded capability profile for {role}")
                
            except Exception as e:
                logger.error(f"Failed to load capability for {config_file}: {e}")
        
        logger.info(f"Loaded {len(self.agent_capabilities)} agent capability profiles")
    
    def _calculate_complexity_handling(self, config: Dict[str, Any]) -> float:
        """Calculate complexity handling score for agent"""
        priority_map = {"critical": 0.9, "high": 0.8, "medium": 0.6, "low": 0.4}
        base_score = priority_map.get(config.get('priority', 'medium'), 0.6)
        
        # Adjust based on execution context
        execution_context = config.get('execution_context', {})
        if execution_context.get('requires_isolation', False):
            base_score += 0.1
        if execution_context.get('timeout_minutes', 30) > 60:
            base_score += 0.05
        
        return min(base_score, 1.0)
    
    def _get_performance_baseline(self, role: str) -> Dict[str, float]:
        """Get performance baseline metrics for agent role"""
        # Mock performance metrics - in real implementation would be learned
        return {
            "success_rate": 0.85 + hash(role) % 10 / 100,  # 0.85-0.94
            "avg_completion_time": 300 + hash(role) % 1800,  # 5-35 minutes
            "resource_efficiency": 0.7 + hash(role) % 25 / 100,  # 0.70-0.94
            "quality_score": 0.8 + hash(role) % 15 / 100  # 0.80-0.94
        }
    
    def _calculate_resource_cost(self, config: Dict[str, Any]) -> float:
        """Calculate relative resource cost for agent"""
        execution_context = config.get('execution_context', {})
        base_cost = 1.0
        
        # Adjust based on resource requirements
        max_parallel = execution_context.get('max_parallel_tasks', 1)
        timeout = execution_context.get('timeout_minutes', 30)
        
        cost_factor = (max_parallel * 0.2) + (timeout / 100)
        return base_cost + cost_factor
    
    def _get_scaling_factor(self, config: Dict[str, Any]) -> float:
        """Get scaling factor for agent type"""
        # Some agents scale better than others
        role = config['role']
        
        high_scaling_roles = [
            "test_generator", "documentation_writer", "code_reviewer", 
            "performance_optimizer", "quality_assurance"
        ]
        
        medium_scaling_roles = [
            "python_backend_coder", "typescript_frontend_agent", 
            "api_integrator", "refactoring_specialist"
        ]
        
        if role in high_scaling_roles:
            return 1.5
        elif role in medium_scaling_roles:
            return 1.2
        else:
            return 1.0
    
    async def start_meta_agent(self):
        """Start the meta-agent decision loop"""
        if self.is_running:
            logger.warning("Meta-agent already running")
            return
        
        self.is_running = True
        self.meta_loop_task = asyncio.create_task(self._meta_decision_loop())
        logger.info("Meta-agent started")
    
    async def stop_meta_agent(self):
        """Stop the meta-agent decision loop"""
        self.is_running = False
        if self.meta_loop_task:
            self.meta_loop_task.cancel()
            try:
                await self.meta_loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Meta-agent stopped")
    
    async def _meta_decision_loop(self):
        """Main meta-agent decision-making loop"""
        while self.is_running:
            try:
                # Analyze current system state
                system_state = await self._analyze_system_state()
                
                # Make meta-decisions
                decisions = await self._make_meta_decisions(system_state)
                
                # Execute approved decisions
                for decision in decisions:
                    await self._execute_decision(decision)
                
                # Learn from outcomes
                if self.learning_enabled:
                    await self._update_learning_models()
                
                # Wait for next decision cycle
                await asyncio.sleep(self.decision_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in meta-agent decision loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _analyze_system_state(self) -> Dict[str, Any]:
        """Analyze current system state for decision-making"""
        try:
            # Get system status from components
            pool_status = self.agent_pool_manager.get_pool_status()
            orchestrator_status = self.orchestrator.get_system_status()
            
            # Calculate derived metrics
            system_state = {
                "timestamp": time.time(),
                "total_agents": pool_status["total_agents"],
                "agent_utilization": self._calculate_agent_utilization(pool_status),
                "resource_usage": pool_status["resource_usage"],
                "task_queue_depth": orchestrator_status["executor"]["queued_tasks"],
                "active_workflows": len(self.workflow_contexts),
                "system_health": self._assess_system_health(pool_status, orchestrator_status),
                "performance_trends": self._analyze_performance_trends(),
                "bottlenecks": self._identify_bottlenecks(pool_status, orchestrator_status),
                "scaling_opportunities": self._identify_scaling_opportunities(pool_status)
            }
            
            return system_state
            
        except Exception as e:
            logger.error(f"Failed to analyze system state: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def _calculate_agent_utilization(self, pool_status: Dict[str, Any]) -> Dict[str, float]:
        """Calculate utilization metrics for different agent roles"""
        utilization = {}
        
        for role, stats in pool_status.get("role_distribution", {}).items():
            total = stats["count"]
            busy = stats["busy"]
            
            if total > 0:
                utilization[role] = busy / total
            else:
                utilization[role] = 0.0
        
        # Overall utilization
        total_agents = pool_status["total_agents"]
        busy_agents = pool_status["task_statistics"]["currently_busy"]
        utilization["overall"] = busy_agents / total_agents if total_agents > 0 else 0.0
        
        return utilization
    
    def _assess_system_health(self, pool_status: Dict[str, Any], orchestrator_status: Dict[str, Any]) -> Dict[str, float]:
        """Assess overall system health metrics"""
        # Calculate health scores (0.0 to 1.0)
        task_stats = pool_status["task_statistics"]
        
        success_rate = 1.0
        if (task_stats["total_completed"] + task_stats["total_failed"]) > 0:
            success_rate = task_stats["total_completed"] / (task_stats["total_completed"] + task_stats["total_failed"])
        
        error_agents = pool_status["agent_states"].get("error", 0)
        total_agents = pool_status["total_agents"]
        agent_health = 1.0 - (error_agents / total_agents if total_agents > 0 else 0)
        
        resource_health = 1.0 - min(
            pool_status["resource_usage"]["average_cpu_percent"] / 100,
            pool_status["resource_usage"]["total_memory_mb"] / 8192  # Assume 8GB limit
        )
        
        return {
            "overall": (success_rate + agent_health + resource_health) / 3,
            "task_success": success_rate,
            "agent_health": agent_health,
            "resource_health": resource_health
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from recent decisions"""
        if not self.decision_history:
            return {"trend": "no_data", "confidence": 0.0}
        
        recent_decisions = [d for d in self.decision_history if time.time() - float(d.decision_id.split('_')[-1]) < 3600]
        
        if not recent_decisions:
            return {"trend": "stable", "confidence": 0.5}
        
        # Analyze decision outcomes (simplified)
        success_count = len([d for d in recent_decisions if d.confidence > 0.8])
        total_decisions = len(recent_decisions)
        
        trend_score = success_count / total_decisions if total_decisions > 0 else 0.5
        
        return {
            "trend": "improving" if trend_score > 0.7 else "declining" if trend_score < 0.3 else "stable",
            "confidence": trend_score,
            "recent_decisions": total_decisions
        }
    
    def _identify_bottlenecks(self, pool_status: Dict[str, Any], orchestrator_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # Check for overutilized roles
        for role, stats in pool_status.get("role_distribution", {}).items():
            if stats["count"] > 0:
                utilization = stats["busy"] / stats["count"]
                if utilization > 0.8:  # 80% utilization threshold
                    bottlenecks.append({
                        "type": "high_utilization",
                        "role": role,
                        "severity": min(utilization, 1.0),
                        "recommendation": "scale_up"
                    })
        
        # Check for queued tasks
        queued_tasks = orchestrator_status["executor"]["queued_tasks"]
        if queued_tasks > 5:
            bottlenecks.append({
                "type": "task_queue_backlog",
                "count": queued_tasks,
                "severity": min(queued_tasks / 20, 1.0),
                "recommendation": "increase_capacity"
            })
        
        # Check for resource constraints
        memory_usage = pool_status["resource_usage"]["total_memory_mb"]
        if memory_usage > 6144:  # 6GB threshold
            bottlenecks.append({
                "type": "memory_pressure",
                "usage_mb": memory_usage,
                "severity": min(memory_usage / 8192, 1.0),
                "recommendation": "optimize_memory"
            })
        
        return bottlenecks
    
    def _identify_scaling_opportunities(self, pool_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for scaling agents"""
        opportunities = []
        
        for role, stats in pool_status.get("role_distribution", {}).items():
            capability = self.agent_capabilities.get(role)
            if not capability:
                continue
            
            # Check if role can be scaled up beneficially
            utilization = stats["busy"] / stats["count"] if stats["count"] > 0 else 0
            
            if utilization > 0.6 and capability.scaling_factor > 1.0:
                benefit_score = utilization * capability.scaling_factor * capability.performance_metrics.get("success_rate", 0.8)
                
                opportunities.append({
                    "type": "scale_up_opportunity",
                    "role": role,
                    "current_count": stats["count"],
                    "utilization": utilization,
                    "benefit_score": benefit_score,
                    "scaling_factor": capability.scaling_factor
                })
        
        # Sort by benefit score
        opportunities.sort(key=lambda x: x["benefit_score"], reverse=True)
        
        return opportunities
    
    async def _make_meta_decisions(self, system_state: Dict[str, Any]) -> List[MetaDecision]:
        """Make intelligent meta-decisions based on system state"""
        decisions = []
        
        try:
            # Decision 1: Address bottlenecks
            for bottleneck in system_state.get("bottlenecks", []):
                decision = await self._create_bottleneck_decision(bottleneck, system_state)
                if decision:
                    decisions.append(decision)
            
            # Decision 2: Optimize scaling opportunities
            opportunities = system_state.get("scaling_opportunities", [])
            for opportunity in opportunities[:3]:  # Top 3 opportunities
                decision = await self._create_scaling_decision(opportunity, system_state)
                if decision:
                    decisions.append(decision)
            
            # Decision 3: Resource optimization
            if system_state.get("system_health", {}).get("resource_health", 1.0) < 0.7:
                decision = await self._create_optimization_decision(system_state)
                if decision:
                    decisions.append(decision)
            
            # Decision 4: Proactive scaling based on trends
            trends = system_state.get("performance_trends", {})
            if trends.get("trend") == "declining":
                decision = await self._create_proactive_decision(system_state)
                if decision:
                    decisions.append(decision)
            
            # Filter and prioritize decisions
            decisions = self._prioritize_decisions(decisions, system_state)
            
            logger.info(f"Meta-agent generated {len(decisions)} decisions")
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to make meta-decisions: {e}")
            return []
    
    async def _create_bottleneck_decision(self, bottleneck: Dict[str, Any], system_state: Dict[str, Any]) -> Optional[MetaDecision]:
        """Create decision to address a specific bottleneck"""
        try:
            decision_id = f"bottleneck_{int(time.time() * 1000)}"
            
            if bottleneck["type"] == "high_utilization":
                role = bottleneck["role"]
                current_count = system_state.get("agent_utilization", {}).get(role, 0)
                
                return MetaDecision(
                    decision_id=decision_id,
                    decision_type=MetaAgentDecision.SPAWN_AGENT,
                    reasoning=f"High utilization ({bottleneck['severity']:.2f}) detected for {role}, spawning additional agent",
                    confidence=0.8 + (bottleneck["severity"] * 0.1),
                    parameters={"role": role, "count": 1},
                    expected_outcome=f"Reduce {role} utilization to <80%",
                    risk_assessment=0.2,
                    alternatives=[
                        {"action": "reassign_tasks", "confidence": 0.6},
                        {"action": "defer_low_priority", "confidence": 0.4}
                    ]
                )
            
            elif bottleneck["type"] == "task_queue_backlog":
                return MetaDecision(
                    decision_id=decision_id,
                    decision_type=MetaAgentDecision.SCALE_UP,
                    reasoning=f"Task queue backlog ({bottleneck['count']} tasks) requires additional capacity",
                    confidence=0.75,
                    parameters={"target_roles": ["python_backend_coder", "typescript_frontend_agent"], "scale_factor": 1.5},
                    expected_outcome="Reduce task queue to <5 tasks",
                    risk_assessment=0.3
                )
                
        except Exception as e:
            logger.error(f"Failed to create bottleneck decision: {e}")
            return None
    
    async def _create_scaling_decision(self, opportunity: Dict[str, Any], system_state: Dict[str, Any]) -> Optional[MetaDecision]:
        """Create decision for scaling opportunity"""
        try:
            decision_id = f"scaling_{int(time.time() * 1000)}"
            role = opportunity["role"]
            
            return MetaDecision(
                decision_id=decision_id,
                decision_type=MetaAgentDecision.SCALE_UP,
                reasoning=f"High-benefit scaling opportunity for {role} (benefit score: {opportunity['benefit_score']:.2f})",
                confidence=min(opportunity["benefit_score"], 0.9),
                parameters={"role": role, "target_count": opportunity["current_count"] + 1},
                expected_outcome=f"Improve {role} throughput by {opportunity['scaling_factor']*20:.0f}%",
                risk_assessment=0.1
            )
            
        except Exception as e:
            logger.error(f"Failed to create scaling decision: {e}")
            return None
    
    async def _create_optimization_decision(self, system_state: Dict[str, Any]) -> Optional[MetaDecision]:
        """Create decision for resource optimization"""
        try:
            decision_id = f"optimization_{int(time.time() * 1000)}"
            
            return MetaDecision(
                decision_id=decision_id,
                decision_type=MetaAgentDecision.OPTIMIZE_RESOURCES,
                reasoning="System resource health below 70%, optimization needed",
                confidence=0.7,
                parameters={"cleanup_idle": True, "memory_optimization": True},
                expected_outcome="Improve resource health to >80%",
                risk_assessment=0.2
            )
            
        except Exception as e:
            logger.error(f"Failed to create optimization decision: {e}")
            return None
    
    async def _create_proactive_decision(self, system_state: Dict[str, Any]) -> Optional[MetaDecision]:
        """Create proactive decision based on performance trends"""
        try:
            decision_id = f"proactive_{int(time.time() * 1000)}"
            
            return MetaDecision(
                decision_id=decision_id,
                decision_type=MetaAgentDecision.SCALE_UP,
                reasoning="Declining performance trend detected, proactive scaling recommended",
                confidence=0.6,
                parameters={"preventive_scaling": True, "monitoring_enhanced": True},
                expected_outcome="Prevent further performance degradation",
                risk_assessment=0.4
            )
            
        except Exception as e:
            logger.error(f"Failed to create proactive decision: {e}")
            return None
    
    def _prioritize_decisions(self, decisions: List[MetaDecision], system_state: Dict[str, Any]) -> List[MetaDecision]:
        """Prioritize decisions based on confidence, risk, and system needs"""
        def decision_score(decision: MetaDecision) -> float:
            # Calculate priority score based on multiple factors
            base_score = decision.confidence
            
            # Adjust for risk (lower risk is better)
            risk_adjustment = 1.0 - decision.risk_assessment
            
            # Adjust for system health (more urgent when health is poor)
            health_urgency = 1.0 - system_state.get("system_health", {}).get("overall", 1.0)
            
            # Combine factors
            return (base_score * 0.5) + (risk_adjustment * 0.3) + (health_urgency * 0.2)
        
        # Sort by priority score
        prioritized = sorted(decisions, key=decision_score, reverse=True)
        
        # Limit to top 3 decisions to prevent overwhelming the system
        return prioritized[:3]
    
    async def _execute_decision(self, decision: MetaDecision):
        """Execute a meta-agent decision"""
        try:
            logger.info(f"Executing meta-decision: {decision.decision_type.value} - {decision.reasoning}")
            
            self.active_decisions[decision.decision_id] = decision
            
            if decision.decision_type == MetaAgentDecision.SPAWN_AGENT:
                await self._execute_spawn_decision(decision)
            
            elif decision.decision_type == MetaAgentDecision.SCALE_UP:
                await self._execute_scale_up_decision(decision)
            
            elif decision.decision_type == MetaAgentDecision.SCALE_DOWN:
                await self._execute_scale_down_decision(decision)
            
            elif decision.decision_type == MetaAgentDecision.OPTIMIZE_RESOURCES:
                await self._execute_optimization_decision(decision)
            
            elif decision.decision_type == MetaAgentDecision.REASSIGN_TASK:
                await self._execute_reassign_decision(decision)
            
            # Record decision in history
            self.decision_history.append(decision)
            
            # Keep history bounded
            if len(self.decision_history) > 1000:
                self.decision_history.pop(0)
            
            logger.info(f"Successfully executed decision: {decision.decision_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute decision {decision.decision_id}: {e}")
        
        finally:
            # Remove from active decisions
            if decision.decision_id in self.active_decisions:
                del self.active_decisions[decision.decision_id]
    
    async def _execute_spawn_decision(self, decision: MetaDecision):
        """Execute agent spawning decision"""
        role = decision.parameters["role"]
        count = decision.parameters.get("count", 1)
        
        for _ in range(count):
            agent_id = self.agent_pool_manager.spawn_agent(role)
            if agent_id:
                logger.info(f"Meta-agent spawned {role}: {agent_id}")
            else:
                logger.warning(f"Failed to spawn {role}")
    
    async def _execute_scale_up_decision(self, decision: MetaDecision):
        """Execute scale-up decision"""
        if "target_roles" in decision.parameters:
            # Scale specific roles
            for role in decision.parameters["target_roles"]:
                self.agent_pool_manager.spawn_agent(role)
        
        elif "role" in decision.parameters:
            # Scale specific role to target count
            role = decision.parameters["role"]
            target_count = decision.parameters.get("target_count", 1)
            
            pool_status = self.agent_pool_manager.get_pool_status()
            current_count = pool_status["role_distribution"].get(role, {}).get("count", 0)
            
            for _ in range(max(0, target_count - current_count)):
                self.agent_pool_manager.spawn_agent(role)
    
    async def _execute_scale_down_decision(self, decision: MetaDecision):
        """Execute scale-down decision"""
        # Implementation for scaling down agents
        logger.info("Scale-down decision executed (placeholder)")
    
    async def _execute_optimization_decision(self, decision: MetaDecision):
        """Execute resource optimization decision"""
        if decision.parameters.get("cleanup_idle"):
            # Trigger cleanup of idle agents
            self.agent_pool_manager.scale_agents(target_load=0.8)
        
        logger.info("Resource optimization executed")
    
    async def _execute_reassign_decision(self, decision: MetaDecision):
        """Execute task reassignment decision"""
        # Implementation for task reassignment
        logger.info("Task reassignment decision executed (placeholder)")
    
    async def _update_learning_models(self):
        """Update learning models based on decision outcomes"""
        # Placeholder for machine learning model updates
        # In real implementation, would analyze decision success rates
        # and adjust decision-making parameters
        pass
    
    def register_workflow(self, workflow_context: WorkflowContext):
        """Register a new workflow with the meta-agent"""
        self.workflow_contexts[workflow_context.workflow_id] = workflow_context
        logger.info(f"Registered workflow: {workflow_context.workflow_id}")
    
    def unregister_workflow(self, workflow_id: str):
        """Unregister a completed workflow"""
        if workflow_id in self.workflow_contexts:
            del self.workflow_contexts[workflow_id]
            logger.info(f"Unregistered workflow: {workflow_id}")
    
    def get_meta_status(self) -> Dict[str, Any]:
        """Get comprehensive meta-agent status"""
        return {
            "is_running": self.is_running,
            "total_workflows": len(self.workflow_contexts),
            "active_decisions": len(self.active_decisions),
            "decision_history_length": len(self.decision_history),
            "agent_capabilities": len(self.agent_capabilities),
            "resource_optimization_score": self.resource_optimization_score,
            "learning_enabled": self.learning_enabled,
            "performance_metrics": self.performance_metrics,
            "recent_decisions": [
                {
                    "decision_id": d.decision_id,
                    "type": d.decision_type.value,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning
                }
                for d in self.decision_history[-5:]  # Last 5 decisions
            ]
        }

# Example usage
if __name__ == "__main__":
    async def main():
        from ..orchestration.agent_pool import AgentPool
        from ..orchestration.orchestrator import ArchonOrchestrator
        
        # Initialize components
        agent_pool = AgentPool()
        orchestrator = ArchonOrchestrator()
        
        # Initialize meta-agent
        meta_agent = ArchonMetaAgent(
            agent_pool_manager=agent_pool,
            orchestrator=orchestrator,
            max_total_agents=30,
            decision_interval=15  # 15 second decision cycles
        )
        
        # Start meta-agent
        await meta_agent.start_meta_agent()
        
        try:
            # Let it run for a while
            await asyncio.sleep(120)  # 2 minutes
            
            # Show status
            status = meta_agent.get_meta_status()
            print(f"Meta-agent status: {json.dumps(status, indent=2)}")
            
        finally:
            await meta_agent.stop_meta_agent()
    
    asyncio.run(main())