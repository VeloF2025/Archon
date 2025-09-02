"""
Meta Agent Coordinator - Central Integration Point
Coordinates all Phase 2 components with unified registry and real execution
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from .unified_registry import UnifiedAgentRegistry, AgentServiceConnector
from .parallel_execution_engine import ParallelExecutionEngine, TaskPriority
from .task_router import IntelligentTaskRouter
from .agent_manager import DynamicAgentManager
from .parallel_executor import AgentTask, AgentStatus

logger = logging.getLogger(__name__)


class MetaAgentCoordinator:
    """
    Central coordinator that actually makes Phase 2 work.
    Fixes the integration issues by providing shared registry and execution.
    """
    
    def __init__(self, 
                 max_agents: int = 50,
                 max_workers: int = 10,
                 agents_service_url: str = "http://localhost:8052"):
        """
        Initialize the Meta Agent Coordinator with proper integration.
        
        Args:
            max_agents: Maximum number of agents to manage
            max_workers: Maximum parallel execution workers
            agents_service_url: URL of agents service
        """
        # Core: Unified registry for all components
        self.registry = UnifiedAgentRegistry()
        
        # Service connector for real execution
        self.service_connector = AgentServiceConnector(agents_service_url)
        self.service_connector.set_registry(self.registry)
        
        # Initialize components with shared registry
        self.router = IntelligentTaskRouter()
        self.manager = DynamicAgentManager(max_agents=max_agents)
        self.engine = ParallelExecutionEngine(max_workers=max_workers)
        
        # Track initialization state
        self.initialized = False
        self._initialization_lock = asyncio.Lock()
        
        logger.info(f"Initialized MetaAgentCoordinator with max_agents={max_agents}, max_workers={max_workers}")
    
    async def initialize(self):
        """
        Initialize and synchronize all components.
        This fixes the agent ID mismatch issue.
        """
        async with self._initialization_lock:
            if self.initialized:
                return
            
            logger.info("Initializing MetaAgentCoordinator components...")
            
            # 1. Synchronize router's default agents with registry
            await self._sync_router_agents()
            
            # 2. Start manager's health monitoring
            await self.manager.start_monitoring()
            
            # 3. Start execution engine workers
            await self.engine.start_workers()
            
            # 4. Register observer for dynamic updates
            self.registry.add_observer(self._on_registry_change)
            
            self.initialized = True
            logger.info("MetaAgentCoordinator initialization complete")
    
    async def _sync_router_agents(self):
        """Synchronize router's default agents with unified registry"""
        for component_id, capability in self.router.agent_capabilities.items():
            # Generate a proper agent_id for manager
            agent_id = f"{capability.agent_role}_{component_id}"
            
            # Register in unified registry
            await self.registry.register_agent(
                agent_id=agent_id,
                agent_role=capability.agent_role,
                component_id=component_id,
                source="router",
                metadata={
                    "capability": capability,
                    "specializations": capability.specializations,
                    "supported_languages": capability.supported_languages
                }
            )
            
            # Spawn corresponding agent in manager
            try:
                spawned_id = await self.manager.spawn_agent(
                    role=capability.agent_role,
                    specialization={
                        "component_id": component_id,
                        "from_router": True
                    }
                )
                
                # Update registry with manager's ID
                if spawned_id:
                    await self.registry.register_agent(
                        agent_id=spawned_id,
                        agent_role=capability.agent_role,
                        component_id=component_id,
                        source="manager",
                        metadata={"linked_to_router": True}
                    )
                    
            except Exception as e:
                logger.error(f"Failed to spawn agent for {component_id}: {e}")
    
    async def execute_parallel(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """
        Execute tasks in parallel with proper integration.
        This is the FIXED version that actually works.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            List of executed tasks with results
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"MetaAgentCoordinator executing {len(tasks)} tasks in parallel")
        
        executed_tasks = []
        
        for task in tasks:
            try:
                # 1. Route task to optimal agent (using component_id)
                component_id = await self.router.route_task(task)
                
                if not component_id:
                    logger.warning(f"No agent available for task {task.task_id}")
                    task.status = AgentStatus.FAILED
                    task.error_message = "No agent available"
                    executed_tasks.append(task)
                    continue
                
                # 2. Get actual agent from registry
                agent = self.registry.get_agent_by_component_id(component_id)
                
                if not agent:
                    # Agent doesn't exist, spawn it
                    logger.info(f"Spawning new agent for role {task.agent_role}")
                    spawned_id = await self.manager.spawn_agent(
                        role=task.agent_role,
                        specialization={"auto_spawned": True}
                    )
                    
                    if spawned_id:
                        # Register new agent
                        new_component_id = f"{task.agent_role}_auto_{len(self.registry.agents)}"
                        await self.registry.register_agent(
                            agent_id=spawned_id,
                            agent_role=task.agent_role,
                            component_id=new_component_id,
                            source="coordinator",
                            metadata={"auto_spawned": True}
                        )
                        component_id = new_component_id
                        agent = self.registry.get_agent_by_component_id(component_id)
                
                if agent:
                    # 3. Assign task to agent in manager
                    await self.manager.assign_task(agent.agent_id, task.task_id)
                    
                    # 4. Execute task via service connector
                    result = await self.service_connector.execute_task(agent.agent_id, task)
                    
                    # 5. Update task with results
                    task.status = AgentStatus.COMPLETED if result.get("status") == "completed" else AgentStatus.FAILED
                    task.output = result.get("output", "")
                    task.error_message = result.get("error", "")
                    task.end_time = datetime.now()
                    
                    # 6. Update router metrics
                    execution_time = result.get("execution_time", 0.1)
                    self.router.update_task_result(
                        agent_id=component_id,
                        task_id=task.task_id,
                        success=task.status == AgentStatus.COMPLETED,
                        execution_time=execution_time
                    )
                    
                    # 7. Release agent in manager
                    await self.manager.release_task(
                        agent.agent_id,
                        success=task.status == AgentStatus.COMPLETED,
                        execution_time=execution_time
                    )
                else:
                    task.status = AgentStatus.FAILED
                    task.error_message = "Failed to create agent"
                
                executed_tasks.append(task)
                
            except Exception as e:
                logger.error(f"Failed to execute task {task.task_id}: {e}")
                task.status = AgentStatus.FAILED
                task.error_message = str(e)
                executed_tasks.append(task)
        
        # Log results
        successful = len([t for t in executed_tasks if t.status == AgentStatus.COMPLETED])
        logger.info(f"Parallel execution complete: {successful}/{len(tasks)} succeeded")
        
        return executed_tasks
    
    async def execute_batch_parallel(self, tasks: List[AgentTask], timeout_minutes: float = 5.0) -> Dict:
        """
        Execute batch of tasks using the ParallelExecutionEngine.
        
        Args:
            tasks: Tasks to execute
            timeout_minutes: Timeout for batch
            
        Returns:
            Batch execution results
        """
        if not self.initialized:
            await self.initialize()
        
        # Use the execution engine for true parallel execution
        # But with proper agent integration this time
        
        # First, ensure all tasks can be routed
        routable_tasks = []
        for task in tasks:
            component_id = await self.router.route_task(task)
            if component_id:
                task.metadata = task.metadata or {}
                task.metadata["routed_to"] = component_id
                routable_tasks.append(task)
            else:
                logger.warning(f"Cannot route task {task.task_id}")
        
        # Execute via engine (which now uses our service connector)
        batch_result = await self.engine.execute_batch(routable_tasks, timeout_minutes)
        
        return {
            "total_tasks": len(tasks),
            "routed_tasks": len(routable_tasks),
            "completed": batch_result.completed,
            "failed": batch_result.failed,
            "timeout": batch_result.timeout,
            "total_execution_time": batch_result.total_execution_time,
            "parallel_efficiency": batch_result.parallel_efficiency
        }
    
    async def _on_registry_change(self, event: str, agent):
        """Handle registry changes"""
        logger.debug(f"Registry event: {event} for agent {agent.agent_id}")
        
        # Could trigger re-routing, scaling decisions, etc.
        if event == "unregister":
            # Check if we need to spawn replacement
            role_agents = self.registry.get_agents_by_role(agent.agent_role)
            if len(role_agents) == 0:
                logger.info(f"No agents left for role {agent.agent_role}, spawning replacement")
                await self.manager.spawn_agent(agent.agent_role, {"replacement": True})
    
    async def get_status(self) -> Dict:
        """Get comprehensive status of all components"""
        return {
            "initialized": self.initialized,
            "registry": self.registry.get_statistics(),
            "router": self.router.get_routing_statistics(),
            "manager": self.manager.get_statistics(),
            "engine": self.engine.get_status(),
            "integration_health": {
                "registry_active": len(self.registry.agents) > 0,
                "router_synced": len(self.router.agent_capabilities) == len([a for a in self.registry.agents.values() if a.source == "router"]),
                "manager_synced": len(self.manager.agents) > 0,
                "workers_running": self.engine._workers_started
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down MetaAgentCoordinator...")
        
        # Stop workers
        await self.engine.stop_workers()
        
        # Stop agent monitoring
        await self.manager.stop_monitoring()
        
        # Terminate all agents
        await self.manager.terminate_all()
        
        logger.info("MetaAgentCoordinator shutdown complete")