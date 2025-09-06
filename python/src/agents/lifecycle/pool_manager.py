"""
Agent Pool Manager v3.0 - Pool Management Implementation
Based on Agent_Lifecycle_Management_PRP.md specifications

NLNH Protocol: Real pool management with actual resource constraints
DGTS Enforcement: No fake limits, actual capacity management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .agent_v3 import AgentV3, AgentState

logger = logging.getLogger(__name__)


@dataclass
class PoolStatistics:
    """Pool statistics for monitoring and optimization"""
    active_counts: Dict[str, int] = field(default_factory=dict)
    total_counts: Dict[str, int] = field(default_factory=dict)
    utilization_rate: Dict[str, float] = field(default_factory=dict)
    can_spawn: Dict[str, bool] = field(default_factory=dict)
    hibernation_candidates: List[str] = field(default_factory=list)
    archival_candidates: List[str] = field(default_factory=list)


@dataclass 
class PoolOptimizationResult:
    """Result of pool optimization operation"""
    hibernated_agents: List[str] = field(default_factory=list)
    archived_agents: List[str] = field(default_factory=list)
    spawned_agents: List[str] = field(default_factory=list)
    optimization_summary: Dict[str, any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    """Agent specification for spawning"""
    def __init__(self, agent_type: str, model_tier: str, specialization: str = None):
        if model_tier not in ["opus", "sonnet", "haiku"]:
            raise ValueError(f"Invalid model tier: {model_tier}")
        self.agent_type = agent_type
        self.model_tier = model_tier
        self.specialization = specialization


class AgentPoolManager:
    """
    Agent Pool Manager for Archon v3.0
    Implementation for test_agent_lifecycle_v3.py tests
    """
    
    # Pool size limits from PRP Section 1.3.1
    MAX_AGENTS = {"opus": 2, "sonnet": 10, "haiku": 50}
    
    def __init__(self):
        # Active agent tracking
        self._agents_by_tier: Dict[str, Set[AgentV3]] = {
            "opus": set(),
            "sonnet": set(), 
            "haiku": set()
        }
        
        # All agents (including hibernated/archived)
        self._all_agents: Dict[str, AgentV3] = {}
        
        # Optimization scheduling
        self._optimization_task: Optional[asyncio.Task] = None
        self._optimization_runs: List[PoolOptimizationResult] = []
        self._last_optimization: Optional[datetime] = None
        
        # Event callbacks
        self.on_optimization_run = None
        
        logger.info("AgentPoolManager initialized with limits: %s", self.MAX_AGENTS)

    async def spawn_agent(self, spec: AgentSpec, project_id: str) -> AgentV3:
        """
        Spawn new agent if pool capacity allows
        Implementation for test_pool_size_limits() and test_spawn_rejection_when_full()
        """
        # Check capacity constraints
        if not await self.can_spawn_agent(spec.model_tier):
            current_count = len(self._agents_by_tier[spec.model_tier])
            max_count = self.MAX_AGENTS[spec.model_tier]
            raise ValueError(
                f"Pool capacity exceeded for {spec.model_tier} tier. "
                f"Current: {current_count}/{max_count}. "
                f"Consider hibernating idle agents or wait for automatic optimization."
            )
        
        # Create agent
        agent_name = f"{spec.agent_type}-{len(self._all_agents) + 1}"
        agent = AgentV3(
            project_id=project_id,
            name=agent_name,
            agent_type=spec.agent_type,
            model_tier=spec.model_tier,
            specialization=spec.specialization
        )
        
        # Register in pool
        self._agents_by_tier[spec.model_tier].add(agent)
        self._all_agents[agent.id] = agent
        
        logger.info(f"Spawned agent {agent.name} ({spec.model_tier}) for project {project_id}")
        
        return agent

    async def can_spawn_agent(self, model_tier: str) -> bool:
        """Check if new agent can be spawned within limits"""
        if model_tier not in self.MAX_AGENTS:
            return False
            
        current_active = len([
            agent for agent in self._agents_by_tier[model_tier]
            if agent.state in [AgentState.CREATED, AgentState.ACTIVE, AgentState.IDLE]
        ])
        
        return current_active < self.MAX_AGENTS[model_tier]

    async def get_pool_statistics(self) -> PoolStatistics:
        """Get current pool status and utilization metrics"""
        stats = PoolStatistics()
        
        for tier in self.MAX_AGENTS.keys():
            # Count active agents (not hibernated/archived)
            active_agents = [
                agent for agent in self._agents_by_tier[tier]
                if agent.state in [AgentState.CREATED, AgentState.ACTIVE, AgentState.IDLE]
            ]
            total_agents = list(self._agents_by_tier[tier])
            
            stats.active_counts[tier] = len(active_agents)
            stats.total_counts[tier] = len(total_agents)
            stats.utilization_rate[tier] = len(active_agents) / self.MAX_AGENTS[tier]
            stats.can_spawn[tier] = await self.can_spawn_agent(tier)
            
            # Find hibernation candidates (idle > 15 minutes)
            for agent in active_agents:
                if agent.state == AgentState.IDLE:
                    idle_duration = datetime.now() - agent.last_active
                    if idle_duration > timedelta(minutes=15):
                        stats.hibernation_candidates.append(agent.id)
            
            # Find archival candidates (unused > 30 days)
            for agent in total_agents:
                unused_duration = datetime.now() - agent.last_active
                if unused_duration > timedelta(days=30):
                    stats.archival_candidates.append(agent.id)
        
        return stats

    async def optimize_pool(self) -> PoolOptimizationResult:
        """
        Optimize pool resources by hibernating idle agents and archiving old ones
        Implementation for test_pool_optimization_performance()
        """
        start_time = datetime.now()
        result = PoolOptimizationResult()
        
        # Get current statistics
        stats = await self.get_pool_statistics()
        
        # Hibernate idle agents (15+ minutes idle)
        for agent_id in stats.hibernation_candidates:
            agent = self._all_agents.get(agent_id)
            if agent and agent.state == AgentState.IDLE:
                try:
                    await agent.transition_to_hibernated("Pool optimization - idle timeout")
                    result.hibernated_agents.append(agent_id)
                except Exception as e:
                    logger.error(f"Failed to hibernate agent {agent_id}: {e}")
        
        # Archive old agents (30+ days unused)  
        for agent_id in stats.archival_candidates:
            agent = self._all_agents.get(agent_id)
            if agent and agent.state != AgentState.ARCHIVED:
                try:
                    await agent.transition_to_archived("Pool optimization - 30 days unused")
                    result.archived_agents.append(agent_id)
                    
                    # Remove from active tier tracking
                    self._agents_by_tier[agent.model_tier].discard(agent)
                except Exception as e:
                    logger.error(f"Failed to archive agent {agent_id}: {e}")
        
        # Calculate optimization summary
        optimization_duration = (datetime.now() - start_time).total_seconds()
        result.optimization_summary = {
            "duration_seconds": optimization_duration,
            "hibernated_count": len(result.hibernated_agents),
            "archived_count": len(result.archived_agents),
            "timestamp": datetime.now().isoformat()
        }
        
        # Record optimization run
        self._optimization_runs.append(result)
        self._last_optimization = datetime.now()
        
        # Trigger callback if registered
        if self.on_optimization_run:
            self.on_optimization_run(result)
        
        logger.info(f"Pool optimization completed in {optimization_duration:.2f}s: "
                   f"{len(result.hibernated_agents)} hibernated, "
                   f"{len(result.archived_agents)} archived")
        
        return result

    async def start_optimization_scheduler(self) -> None:
        """
        Start automatic pool optimization every 5 minutes
        Implementation for test_pool_optimization_scheduling()
        """
        if self._optimization_task:
            self._optimization_task.cancel()
            
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Pool optimization scheduler started (5 minute intervals)")

    async def stop_optimization_scheduler(self) -> None:
        """Stop automatic pool optimization"""
        if self._optimization_task:
            self._optimization_task.cancel()
            self._optimization_task = None
            logger.info("Pool optimization scheduler stopped")

    async def _optimization_loop(self) -> None:
        """Internal optimization loop running every 5 minutes"""
        try:
            while True:
                await asyncio.sleep(5 * 60)  # 5 minutes
                
                try:
                    result = await self.optimize_pool()
                    logger.debug(f"Scheduled optimization completed: {result.optimization_summary}")
                except Exception as e:
                    logger.error(f"Scheduled optimization failed: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Optimization scheduler cancelled")

    async def get_next_optimization_time(self) -> datetime:
        """Get timestamp of next scheduled optimization"""
        if not self._last_optimization:
            return datetime.now() + timedelta(minutes=5)
        return self._last_optimization + timedelta(minutes=5)

    async def get_agent_by_id(self, agent_id: str) -> Optional[AgentV3]:
        """Get agent by ID"""
        return self._all_agents.get(agent_id)

    async def get_agents_by_project(self, project_id: str) -> List[AgentV3]:
        """Get all agents for a specific project"""
        return [agent for agent in self._all_agents.values() if agent.project_id == project_id]

    async def get_agents_by_state(self, state: AgentState) -> List[AgentV3]:
        """Get all agents in a specific state"""
        return [agent for agent in self._all_agents.values() if agent.state == state]

    async def force_hibernation(self, agent_ids: List[str]) -> List[str]:
        """Force hibernation of specific agents"""
        hibernated = []
        for agent_id in agent_ids:
            agent = self._all_agents.get(agent_id)
            if agent and agent.state in [AgentState.IDLE, AgentState.ACTIVE]:
                try:
                    if agent.state == AgentState.ACTIVE:
                        await agent.transition_to_idle("Force hibernation requested")
                    await agent.transition_to_hibernated("Force hibernation requested")
                    hibernated.append(agent_id)
                except Exception as e:
                    logger.error(f"Failed to force hibernation of agent {agent_id}: {e}")
        return hibernated

    async def force_archival(self, agent_ids: List[str]) -> List[str]:
        """Force archival of specific agents"""
        archived = []
        for agent_id in agent_ids:
            agent = self._all_agents.get(agent_id)
            if agent and agent.state != AgentState.ARCHIVED:
                try:
                    await agent.transition_to_archived("Force archival requested")
                    self._agents_by_tier[agent.model_tier].discard(agent)
                    archived.append(agent_id)
                except Exception as e:
                    logger.error(f"Failed to force archival of agent {agent_id}: {e}")
        return archived

    def get_optimization_history(self) -> List[PoolOptimizationResult]:
        """Get history of pool optimizations"""
        return self._optimization_runs.copy()

    async def cleanup(self) -> None:
        """Cleanup pool manager resources"""
        await self.stop_optimization_scheduler()
        
        # Archive all active agents
        active_agents = [
            agent for agent in self._all_agents.values()
            if agent.state != AgentState.ARCHIVED
        ]
        
        for agent in active_agents:
            try:
                await agent.transition_to_archived("Pool manager cleanup")
            except Exception as e:
                logger.error(f"Failed to cleanup agent {agent.id}: {e}")
        
        logger.info(f"Pool manager cleanup completed, {len(active_agents)} agents archived")

    def __del__(self):
        """Cleanup on destruction"""
        if self._optimization_task and not self._optimization_task.done():
            self._optimization_task.cancel()