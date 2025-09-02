"""
Unified Agent Registry for Phase 2 Integration
Single source of truth for agent identification across all components
"""

import logging
from typing import Dict, Optional, Set, Any
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class RegisteredAgent:
    """Unified agent registration"""
    agent_id: str
    agent_role: str
    component_id: str  # ID used by specific component (router, manager, etc)
    source: str  # Which component registered this
    metadata: Dict[str, Any]


class UnifiedAgentRegistry:
    """
    Single source of truth for agent IDs across all Phase 2 components.
    Solves the ID mismatch between router, manager, and executor.
    """
    
    def __init__(self):
        # Main registry: component_id -> RegisteredAgent
        self.agents: Dict[str, RegisteredAgent] = {}
        
        # Reverse lookups
        self.role_to_agents: Dict[str, Set[str]] = {}  # role -> set of agent_ids
        self.agent_to_component: Dict[str, str] = {}  # agent_id -> component_id
        
        # Observers for changes
        self.observers = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Initialized UnifiedAgentRegistry")
    
    async def register_agent(self, 
                            agent_id: str,
                            agent_role: str, 
                            component_id: str,
                            source: str = "unknown",
                            metadata: Optional[Dict] = None) -> bool:
        """
        Register an agent with unified ID mapping.
        
        Args:
            agent_id: Unique agent identifier (e.g., UUID from manager)
            agent_role: Agent role/type (e.g., "python_backend")
            component_id: ID used by component (e.g., "python_backend_1" from router)
            source: Component that registered this (router/manager/executor)
            metadata: Additional agent information
            
        Returns:
            Success of registration
        """
        async with self._lock:
            try:
                # Create registration
                agent = RegisteredAgent(
                    agent_id=agent_id,
                    agent_role=agent_role,
                    component_id=component_id,
                    source=source,
                    metadata=metadata or {}
                )
                
                # Store in main registry
                self.agents[component_id] = agent
                
                # Update role mapping
                if agent_role not in self.role_to_agents:
                    self.role_to_agents[agent_role] = set()
                self.role_to_agents[agent_role].add(component_id)
                
                # Update reverse lookup
                self.agent_to_component[agent_id] = component_id
                
                logger.info(f"Registered agent: {agent_id} as {component_id} (role: {agent_role}, source: {source})")
                
                # Notify observers
                await self._notify_observers('register', agent)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to register agent {agent_id}: {e}")
                return False
    
    async def unregister_agent(self, component_id: str) -> bool:
        """Remove agent from registry"""
        async with self._lock:
            if component_id not in self.agents:
                return False
            
            agent = self.agents[component_id]
            
            # Remove from main registry
            del self.agents[component_id]
            
            # Remove from role mapping
            if agent.agent_role in self.role_to_agents:
                self.role_to_agents[agent.agent_role].discard(component_id)
                if not self.role_to_agents[agent.agent_role]:
                    del self.role_to_agents[agent.agent_role]
            
            # Remove from reverse lookup
            if agent.agent_id in self.agent_to_component:
                del self.agent_to_component[agent.agent_id]
            
            logger.info(f"Unregistered agent: {component_id}")
            
            # Notify observers
            await self._notify_observers('unregister', agent)
            
            return True
    
    def get_agent_by_component_id(self, component_id: str) -> Optional[RegisteredAgent]:
        """Get agent by component ID (e.g., router's ID)"""
        return self.agents.get(component_id)
    
    def get_agent_by_agent_id(self, agent_id: str) -> Optional[RegisteredAgent]:
        """Get agent by actual agent ID (e.g., manager's UUID)"""
        component_id = self.agent_to_component.get(agent_id)
        if component_id:
            return self.agents.get(component_id)
        return None
    
    def get_agents_by_role(self, role: str) -> Set[RegisteredAgent]:
        """Get all agents with specific role"""
        agents = set()
        if role in self.role_to_agents:
            for component_id in self.role_to_agents[role]:
                if component_id in self.agents:
                    agents.add(self.agents[component_id])
        return agents
    
    def translate_id(self, id_value: str, to_format: str = "agent_id") -> Optional[str]:
        """
        Translate between different ID formats.
        
        Args:
            id_value: The ID to translate
            to_format: Target format ("agent_id" or "component_id")
            
        Returns:
            Translated ID or None if not found
        """
        # Check if it's a component_id
        if id_value in self.agents:
            agent = self.agents[id_value]
            return agent.agent_id if to_format == "agent_id" else id_value
        
        # Check if it's an agent_id
        if id_value in self.agent_to_component:
            component_id = self.agent_to_component[id_value]
            return id_value if to_format == "agent_id" else component_id
        
        return None
    
    def has_agent(self, id_value: str) -> bool:
        """Check if agent exists by any ID format"""
        return id_value in self.agents or id_value in self.agent_to_component
    
    def get_all_agents(self) -> Dict[str, RegisteredAgent]:
        """Get all registered agents"""
        return self.agents.copy()
    
    def add_observer(self, callback):
        """Add observer for registry changes"""
        self.observers.append(callback)
    
    async def _notify_observers(self, event: str, agent: RegisteredAgent):
        """Notify all observers of changes"""
        for observer in self.observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(event, agent)
                else:
                    observer(event, agent)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        role_counts = {}
        for role, agents in self.role_to_agents.items():
            role_counts[role] = len(agents)
        
        source_counts = {}
        for agent in self.agents.values():
            source = agent.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_agents": len(self.agents),
            "unique_agent_ids": len(self.agent_to_component),
            "roles": role_counts,
            "sources": source_counts,
            "observers": len(self.observers)
        }
    
    async def synchronize_with_router(self, router):
        """Synchronize registry with IntelligentTaskRouter"""
        # Register router's default agents
        for component_id, capability in router.agent_capabilities.items():
            await self.register_agent(
                agent_id=f"{capability.agent_role}_{component_id}",  # Generate agent_id
                agent_role=capability.agent_role,
                component_id=component_id,
                source="router",
                metadata={"capability": capability}
            )
    
    async def synchronize_with_manager(self, manager):
        """Synchronize registry with DynamicAgentManager"""
        # Register manager's spawned agents
        for agent_id, managed_agent in manager.agents.items():
            # Create component_id for router compatibility
            component_id = f"{managed_agent.agent_role}_{len(self.role_to_agents.get(managed_agent.agent_role, [])) + 1}"
            
            await self.register_agent(
                agent_id=agent_id,
                agent_role=managed_agent.agent_role,
                component_id=component_id,
                source="manager",
                metadata={"managed_agent": managed_agent}
            )


class AgentServiceConnector:
    """
    Connects to actual agents service for real task execution.
    Replaces the broken circular dependency with real HTTP calls.
    """
    
    def __init__(self, agents_service_url: str = "http://localhost:8052"):
        self.service_url = agents_service_url
        self.registry = None  # Will be set by coordinator
        
    def set_registry(self, registry: UnifiedAgentRegistry):
        """Set the unified registry"""
        self.registry = registry
    
    async def execute_task(self, agent_id: str, task: Any) -> Dict:
        """
        Execute task via actual agents service using the /agents/run endpoint.
        
        Args:
            agent_id: Agent identifier (any format)
            task: Task to execute
            
        Returns:
            Execution result from agents service
        """
        try:
            import httpx
            import os
            
            # Map agent role to agent type that the service understands
            def map_agent_role_to_type(agent_role: str) -> str:
                """Map agent role to actual agent type available in service"""
                if "document" in agent_role.lower() or "writer" in agent_role.lower():
                    return "document"
                else:
                    return "rag"  # Default to rag for all other agents
            
            # Get agent type from role
            agent_type = map_agent_role_to_type(task.agent_role)
            
            # Create context with required fields
            context = {
                "task_id": task.task_id,
                "agent_role": task.agent_role,
                "input_data": task.input_data,
                "file_patterns": getattr(task, 'file_patterns', []),
                "project_id": task.input_data.get("project_id", "archon-plus"),
                "user_id": "system",
                "trace_id": f"task_{task.task_id}",
                "request_id": f"req_{task.task_id}"
            }
            
            # Add agent-specific fields
            if agent_type == "rag":
                context.update({
                    "source_filter": None,
                    "match_count": 5
                })
            elif agent_type == "document":
                context.update({
                    "current_document_id": f"doc_{task.task_id}",
                    "document_type": "technical_documentation", 
                    "content": task.description,
                    "metadata": {
                        "task_type": "documentation",
                        "project": task.input_data.get("project_name", "archon-plus"),
                        "component": task.agent_role
                    }
                })
            
            # Create proper agent request format
            agent_request = {
                "agent_type": agent_type,
                "prompt": task.description,
                "context": context,
                "options": {
                    "timeout_minutes": getattr(task, 'timeout_minutes', 30),
                    "priority": getattr(task, 'priority', 1)
                }
            }
            
            # Call agents service using the correct endpoint
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.service_url}/agents/run",
                    json=agent_request
                )
                
                if response.status_code != 200:
                    raise Exception(f"Agent service error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                if not result.get("success", False):
                    raise Exception(f"Agent execution failed: {result.get('error', 'Unknown error')}")
                
                return {
                    "status": "completed",
                    "output": result.get("result", ""),
                    "metadata": result.get("metadata", {})
                }
            
        except Exception as e:
            logger.error(f"Failed to execute task via agents service: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }