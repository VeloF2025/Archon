"""
Dynamic Agent Manager for Phase 2 Meta-Agent Orchestration
Manages agent lifecycle: spawning, monitoring, termination
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class ManagedAgent:
    """Managed agent instance"""
    agent_id: str
    agent_role: str
    specialization: Dict[str, Any]
    state: AgentState
    created_at: datetime
    last_active: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task_id: Optional[str] = None
    idle_time: float = 0.0
    total_execution_time: float = 0.0
    health_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadMetrics:
    """Current workload metrics"""
    total_tasks_queued: int = 0
    total_tasks_active: int = 0
    average_wait_time: float = 0.0
    average_execution_time: float = 0.0
    peak_load: int = 0
    current_load: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class DynamicAgentManager:
    """
    Manages agent lifecycle: spawning, monitoring, termination.
    """
    
    def __init__(self, 
                 max_agents: int = 50,
                 idle_timeout_seconds: float = 60.0,
                 spawn_threshold_queue_depth: int = 5,
                 min_agents: int = 3):
        """
        Initialize dynamic agent manager.
        
        Args:
            max_agents: Maximum number of agents allowed
            idle_timeout_seconds: Time before terminating idle agents
            spawn_threshold_queue_depth: Queue depth to trigger spawning
            min_agents: Minimum number of agents to maintain
        """
        self.max_agents = max_agents
        self.min_agents = min_agents
        self.idle_timeout = idle_timeout_seconds
        self.spawn_threshold = spawn_threshold_queue_depth
        
        # Agent tracking
        self.agents: Dict[str, ManagedAgent] = {}
        self.agent_pool: List[str] = []  # Available agent IDs
        self.agent_roles: Dict[str, Set[str]] = {}  # role -> set of agent_ids
        
        # Monitoring
        self.workload_metrics = WorkloadMetrics()
        self.health_check_interval = 10.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._shutdown = False
        
        logger.info(f"Initialized DynamicAgentManager with max_agents={max_agents}, min_agents={min_agents}")
    
    async def start_monitoring(self):
        """Start background health monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            return
        
        self.monitoring_task = asyncio.create_task(self.monitor_health())
        logger.info("Started agent health monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self._shutdown = True
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped agent health monitoring")
    
    async def spawn_agent(self, role: str, specialization: Dict[str, Any]) -> str:
        """
        Create new agent with specific capabilities.
        
        Args:
            role: Agent role (e.g., 'python_backend', 'typescript_frontend')
            specialization: Agent specialization details
            
        Returns:
            Agent ID of spawned agent
        """
        async with self._lock:
            # Check if we can spawn more agents
            if len(self.agents) >= self.max_agents:
                logger.warning(f"Cannot spawn agent: max_agents ({self.max_agents}) reached")
                # Try to terminate an idle agent first
                idle_agent = self._find_idle_agent_to_terminate()
                if idle_agent:
                    await self._terminate_agent(idle_agent)
                else:
                    raise RuntimeError(f"Cannot spawn new agent: limit of {self.max_agents} reached")
            
            # Generate unique agent ID
            agent_id = f"{role}_{uuid.uuid4().hex[:8]}"
            
            # Create managed agent
            agent = ManagedAgent(
                agent_id=agent_id,
                agent_role=role,
                specialization=specialization,
                state=AgentState.INITIALIZING,
                created_at=datetime.now(),
                last_active=datetime.now()
            )
            
            # Initialize the actual agent (simulate for now)
            await self._initialize_agent(agent)
            
            # Register agent
            self.agents[agent_id] = agent
            self.agent_pool.append(agent_id)
            
            # Track by role
            if role not in self.agent_roles:
                self.agent_roles[role] = set()
            self.agent_roles[role].add(agent_id)
            
            # Update state
            agent.state = AgentState.READY
            
            logger.info(f"Spawned agent {agent_id} with role {role}")
            return agent_id
    
    async def _initialize_agent(self, agent: ManagedAgent):
        """Initialize the actual agent instance"""
        # Simulate agent initialization
        await asyncio.sleep(0.1)
        
        # In real implementation, this would:
        # 1. Create agent instance based on role
        # 2. Load agent-specific models/resources
        # 3. Configure agent with specialization
        # 4. Validate agent is working
        
        agent.metadata['initialized'] = True
        agent.metadata['init_time'] = datetime.now().isoformat()
    
    async def get_available_agent(self, role: Optional[str] = None) -> Optional[str]:
        """
        Get an available agent, optionally filtered by role.
        
        Args:
            role: Optional role filter
            
        Returns:
            Agent ID if available, None otherwise
        """
        async with self._lock:
            candidates = []
            
            if role and role in self.agent_roles:
                # Filter by role
                candidates = [
                    aid for aid in self.agent_roles[role]
                    if self.agents[aid].state == AgentState.READY
                ]
            else:
                # Any available agent
                candidates = [
                    aid for aid, agent in self.agents.items()
                    if agent.state == AgentState.READY
                ]
            
            if candidates:
                # Return agent with lowest load
                best_agent = min(
                    candidates,
                    key=lambda aid: self.agents[aid].tasks_completed
                )
                return best_agent
            
            return None
    
    async def assign_task(self, agent_id: str, task_id: str):
        """
        Assign a task to an agent.
        
        Args:
            agent_id: Agent to assign to
            task_id: Task ID being assigned
        """
        async with self._lock:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.agents[agent_id]
            agent.state = AgentState.BUSY
            agent.current_task_id = task_id
            agent.last_active = datetime.now()
            
            logger.debug(f"Assigned task {task_id} to agent {agent_id}")
    
    async def release_task(self, agent_id: str, success: bool, execution_time: float):
        """
        Release agent after task completion.
        
        Args:
            agent_id: Agent that completed task
            success: Whether task succeeded
            execution_time: Task execution time
        """
        async with self._lock:
            if agent_id not in self.agents:
                return
            
            agent = self.agents[agent_id]
            
            # Update metrics
            if success:
                agent.tasks_completed += 1
            else:
                agent.tasks_failed += 1
            
            agent.total_execution_time += execution_time
            agent.current_task_id = None
            agent.state = AgentState.IDLE
            agent.last_active = datetime.now()
            
            # Update health score
            total_tasks = agent.tasks_completed + agent.tasks_failed
            if total_tasks > 0:
                agent.health_score = agent.tasks_completed / total_tasks
            
            logger.debug(f"Released agent {agent_id} after task (success={success})")
    
    async def monitor_health(self):
        """
        Background task for health monitoring.
        Runs continuously until shutdown.
        """
        logger.info("Health monitoring started")
        
        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                async with self._lock:
                    current_time = datetime.now()
                    agents_to_terminate = []
                    
                    for agent_id, agent in self.agents.items():
                        # Update idle time
                        if agent.state == AgentState.IDLE:
                            idle_duration = (current_time - agent.last_active).total_seconds()
                            agent.idle_time = idle_duration
                            
                            # Check for termination
                            if idle_duration > self.idle_timeout and len(self.agents) > self.min_agents:
                                agents_to_terminate.append(agent_id)
                        
                        # Check for stuck agents
                        elif agent.state == AgentState.BUSY:
                            busy_duration = (current_time - agent.last_active).total_seconds()
                            if busy_duration > 300:  # 5 minutes
                                logger.warning(f"Agent {agent_id} stuck in BUSY state for {busy_duration:.0f}s")
                                agent.health_score *= 0.9  # Reduce health score
                        
                        # Check error state
                        elif agent.state == AgentState.ERROR:
                            error_duration = (current_time - agent.last_active).total_seconds()
                            if error_duration > 60:  # 1 minute in error
                                agents_to_terminate.append(agent_id)
                    
                    # Terminate idle agents
                    for agent_id in agents_to_terminate:
                        await self._terminate_agent(agent_id)
                
                # Log health summary
                if len(self.agents) > 0:
                    ready_count = sum(1 for a in self.agents.values() if a.state == AgentState.READY)
                    busy_count = sum(1 for a in self.agents.values() if a.state == AgentState.BUSY)
                    idle_count = sum(1 for a in self.agents.values() if a.state == AgentState.IDLE)
                    
                    logger.debug(f"Agent health: total={len(self.agents)}, ready={ready_count}, busy={busy_count}, idle={idle_count}")
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}", exc_info=True)
        
        logger.info("Health monitoring stopped")
    
    async def scale_agents(self, workload_metrics: Dict[str, Any]):
        """
        Auto-scale based on workload.
        
        Args:
            workload_metrics: Current workload metrics
        """
        async with self._lock:
            queue_depth = workload_metrics.get('queue_depth', 0)
            active_tasks = workload_metrics.get('active_tasks', 0)
            
            # Update metrics
            self.workload_metrics.total_tasks_queued = queue_depth
            self.workload_metrics.total_tasks_active = active_tasks
            self.workload_metrics.current_load = queue_depth + active_tasks
            self.workload_metrics.timestamp = datetime.now()
            
            # Track peak load
            if self.workload_metrics.current_load > self.workload_metrics.peak_load:
                self.workload_metrics.peak_load = self.workload_metrics.current_load
            
            # Determine if scaling needed
            ready_agents = sum(1 for a in self.agents.values() if a.state == AgentState.READY)
            
            # Scale up if high queue depth and no ready agents
            if queue_depth > self.spawn_threshold and ready_agents == 0:
                agents_to_spawn = min(
                    3,  # Spawn up to 3 at a time
                    self.max_agents - len(self.agents)
                )
                
                if agents_to_spawn > 0:
                    logger.info(f"Scaling up: spawning {agents_to_spawn} agents (queue_depth={queue_depth})")
                    
                    # Spawn generic agents
                    for _ in range(agents_to_spawn):
                        try:
                            await self.spawn_agent(
                                role="fullstack",
                                specialization={"auto_scaled": True}
                            )
                        except Exception as e:
                            logger.error(f"Failed to spawn agent during scaling: {e}")
                            break
            
            # Scale down if many idle agents
            elif queue_depth == 0 and ready_agents > self.min_agents:
                idle_agents = [
                    aid for aid, agent in self.agents.items()
                    if agent.state == AgentState.IDLE and agent.idle_time > 30
                ]
                
                # Terminate excess idle agents
                agents_to_terminate = idle_agents[:max(0, ready_agents - self.min_agents)]
                
                if agents_to_terminate:
                    logger.info(f"Scaling down: terminating {len(agents_to_terminate)} idle agents")
                    
                    for agent_id in agents_to_terminate:
                        await self._terminate_agent(agent_id)
    
    async def terminate_idle_agents(self):
        """Clean up unused agents"""
        async with self._lock:
            current_time = datetime.now()
            terminated = []
            
            for agent_id, agent in list(self.agents.items()):
                if agent.state == AgentState.IDLE:
                    idle_duration = (current_time - agent.last_active).total_seconds()
                    
                    # Keep minimum agents
                    if len(self.agents) <= self.min_agents:
                        continue
                    
                    # Terminate if idle too long
                    if idle_duration > self.idle_timeout:
                        await self._terminate_agent(agent_id)
                        terminated.append(agent_id)
            
            if terminated:
                logger.info(f"Terminated {len(terminated)} idle agents: {terminated}")
    
    async def _terminate_agent(self, agent_id: str):
        """
        Terminate an agent.
        
        Args:
            agent_id: Agent to terminate
        """
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        agent.state = AgentState.TERMINATING
        
        try:
            # Cleanup agent resources (simulate)
            await asyncio.sleep(0.05)
            
            # Remove from tracking
            del self.agents[agent_id]
            self.agent_pool.remove(agent_id)
            
            # Remove from role tracking
            if agent.agent_role in self.agent_roles:
                self.agent_roles[agent.agent_role].discard(agent_id)
                if not self.agent_roles[agent.agent_role]:
                    del self.agent_roles[agent.agent_role]
            
            logger.info(f"Terminated agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error terminating agent {agent_id}: {e}")
            agent.state = AgentState.ERROR
    
    def _find_idle_agent_to_terminate(self) -> Optional[str]:
        """Find the best idle agent to terminate"""
        idle_agents = [
            (aid, agent) for aid, agent in self.agents.items()
            if agent.state == AgentState.IDLE
        ]
        
        if not idle_agents:
            return None
        
        # Sort by idle time (longest idle first) and health score (lowest first)
        idle_agents.sort(key=lambda x: (x[1].idle_time, -x[1].health_score), reverse=True)
        
        return idle_agents[0][0]
    
    async def terminate_all(self):
        """Terminate all agents (for shutdown)"""
        async with self._lock:
            logger.info(f"Terminating all {len(self.agents)} agents")
            
            agent_ids = list(self.agents.keys())
            for agent_id in agent_ids:
                await self._terminate_agent(agent_id)
            
            self.agents.clear()
            self.agent_pool.clear()
            self.agent_roles.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        stats = {
            'total_agents': len(self.agents),
            'max_agents': self.max_agents,
            'min_agents': self.min_agents,
            'agent_states': {},
            'agent_roles': {},
            'workload': {
                'queued': self.workload_metrics.total_tasks_queued,
                'active': self.workload_metrics.total_tasks_active,
                'current_load': self.workload_metrics.current_load,
                'peak_load': self.workload_metrics.peak_load
            },
            'agents': []
        }
        
        # Count states
        for agent in self.agents.values():
            state_name = agent.state.value
            stats['agent_states'][state_name] = stats['agent_states'].get(state_name, 0) + 1
            
            # Count roles
            stats['agent_roles'][agent.agent_role] = stats['agent_roles'].get(agent.agent_role, 0) + 1
            
            # Agent details
            stats['agents'].append({
                'id': agent.agent_id,
                'role': agent.agent_role,
                'state': agent.state.value,
                'health_score': agent.health_score,
                'tasks_completed': agent.tasks_completed,
                'tasks_failed': agent.tasks_failed,
                'idle_time': agent.idle_time,
                'average_execution_time': agent.total_execution_time / max(1, agent.tasks_completed + agent.tasks_failed)
            })
        
        return stats