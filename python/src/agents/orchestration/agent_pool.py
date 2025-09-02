#!/usr/bin/env python3
"""
Agent Pool Manager for Archon+ Sub-Agents
Manages agent lifecycle, spawning, and resource allocation
"""

import asyncio
import json
import logging
import psutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
import threading

logger = logging.getLogger(__name__)

class AgentState(Enum):
    SPAWNING = "spawning"
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"

@dataclass
class AgentInstance:
    """Running agent instance representation"""
    agent_id: str
    role: str
    name: str
    process_id: Optional[int] = None
    state: AgentState = AgentState.SPAWNING
    spawn_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    current_task_id: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    completed_tasks: int = 0
    failed_tasks: int = 0
    error_message: Optional[str] = None
    memory_scope: List[str] = field(default_factory=list)
    resource_limits: Dict = field(default_factory=dict)

class AgentPool:
    """Manages a pool of specialized agents with dynamic spawning"""
    
    def __init__(self, 
                 config_path: str = "python/src/agents/configs",
                 min_agents_per_role: int = 1,
                 max_agents_per_role: int = 3,
                 max_total_agents: int = 20,
                 auto_scale: bool = True):
        
        self.config_path = Path(config_path)
        self.min_agents_per_role = min_agents_per_role
        self.max_agents_per_role = max_agents_per_role
        self.max_total_agents = max_total_agents
        self.auto_scale = auto_scale
        
        # Agent management
        self.agent_configs: Dict[str, Dict] = {}
        self.active_agents: Dict[str, AgentInstance] = {}
        self.agent_roles: Set[str] = set()
        
        # Resource management
        self.resource_monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active: bool = False
        
        # Load configurations
        self._load_agent_configs()
        
        # Start resource monitoring
        self.start_monitoring()
        
        logger.info(f"AgentPool initialized with {len(self.agent_configs)} agent types")
    
    def _load_agent_configs(self):
        """Load agent configurations from JSON files"""
        if not self.config_path.exists():
            logger.error(f"Config path does not exist: {self.config_path}")
            return
        
        config_files = list(self.config_path.glob("*.json"))
        config_files = [f for f in config_files if f.name != "agent_registry.json"]
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                role = config['role']
                self.agent_configs[role] = config
                self.agent_roles.add(role)
                
                logger.debug(f"Loaded config for agent role: {role}")
                
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
        
        logger.info(f"Loaded configurations for {len(self.agent_configs)} agent roles")
    
    def _generate_agent_id(self, role: str) -> str:
        """Generate unique agent ID"""
        existing_ids = [aid for aid in self.active_agents.keys() if aid.startswith(f"{role}_")]
        count = len(existing_ids) + 1
        return f"{role}_{count}_{int(time.time())}"
    
    def _spawn_agent_process(self, agent: AgentInstance) -> bool:
        """Spawn actual agent process (simulated for now)"""
        try:
            # In real implementation, this would spawn actual agent process
            # For now, we simulate with a mock process
            
            # Simulate process creation
            mock_pid = hash(agent.agent_id) % 10000 + 1000
            agent.process_id = mock_pid
            agent.state = AgentState.IDLE
            
            logger.info(f"Spawned agent {agent.agent_id} (PID: {mock_pid})")
            return True
            
        except Exception as e:
            agent.state = AgentState.ERROR
            agent.error_message = f"Failed to spawn: {e}"
            logger.error(f"Failed to spawn agent {agent.agent_id}: {e}")
            return False
    
    def spawn_agent(self, role: str) -> Optional[str]:
        """Spawn new agent instance of specified role"""
        if role not in self.agent_configs:
            logger.error(f"Unknown agent role: {role}")
            return None
        
        # Check limits
        role_agents = [a for a in self.active_agents.values() if a.role == role]
        if len(role_agents) >= self.max_agents_per_role:
            logger.warning(f"Maximum agents reached for role {role}")
            return None
        
        if len(self.active_agents) >= self.max_total_agents:
            logger.warning("Maximum total agents reached")
            return None
        
        # Create agent instance
        config = self.agent_configs[role]
        agent_id = self._generate_agent_id(role)
        
        agent = AgentInstance(
            agent_id=agent_id,
            role=role,
            name=config['name'],
            memory_scope=config.get('memory_scope', []),
            resource_limits=config.get('execution_context', {})
        )
        
        # Spawn process
        if self._spawn_agent_process(agent):
            self.active_agents[agent_id] = agent
            logger.info(f"Successfully spawned agent {agent_id} for role {role}")
            return agent_id
        else:
            logger.error(f"Failed to spawn agent for role {role}")
            return None
    
    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate specific agent instance"""
        if agent_id not in self.active_agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent = self.active_agents[agent_id]
        
        try:
            # In real implementation, would terminate actual process
            # For now, simulate termination
            
            agent.state = AgentState.TERMINATED
            logger.info(f"Terminated agent {agent_id} (PID: {agent.process_id})")
            
            # Remove from active pool
            del self.active_agents[agent_id]
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate agent {agent_id}: {e}")
            return False
    
    def get_available_agent(self, role: str) -> Optional[str]:
        """Get available agent for specified role"""
        # Find idle agent of the role
        available_agents = [
            a for a in self.active_agents.values() 
            if a.role == role and a.state == AgentState.IDLE
        ]
        
        if available_agents:
            # Return least recently used agent
            agent = min(available_agents, key=lambda a: a.last_activity)
            return agent.agent_id
        
        # Try to spawn new agent if under limits
        if self.auto_scale:
            role_agents = [a for a in self.active_agents.values() if a.role == role]
            if len(role_agents) < self.max_agents_per_role:
                return self.spawn_agent(role)
        
        return None
    
    def assign_task(self, agent_id: str, task_id: str) -> bool:
        """Assign task to specific agent"""
        if agent_id not in self.active_agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        agent = self.active_agents[agent_id]
        
        if agent.state not in [AgentState.IDLE, AgentState.ACTIVE]:
            logger.warning(f"Agent {agent_id} not available for task assignment (state: {agent.state})")
            return False
        
        agent.current_task_id = task_id
        agent.state = AgentState.BUSY
        agent.last_activity = time.time()
        
        logger.info(f"Assigned task {task_id} to agent {agent_id}")
        return True
    
    def complete_task(self, agent_id: str, success: bool = True) -> bool:
        """Mark task completion for agent"""
        if agent_id not in self.active_agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        agent = self.active_agents[agent_id]
        
        if success:
            agent.completed_tasks += 1
        else:
            agent.failed_tasks += 1
        
        agent.current_task_id = None
        agent.state = AgentState.IDLE
        agent.last_activity = time.time()
        
        logger.info(f"Task completed for agent {agent_id} (success: {success})")
        return True
    
    def _update_resource_usage(self, agent: AgentInstance):
        """Update resource usage for agent (simulated)"""
        try:
            # In real implementation, would query actual process
            # For now, simulate resource usage
            import random
            
            agent.memory_usage_mb = random.uniform(50, 200)  # 50-200 MB
            agent.cpu_usage_percent = random.uniform(0, 25)  # 0-25% CPU
            
        except Exception as e:
            logger.error(f"Failed to update resources for {agent.agent_id}: {e}")
    
    def _monitor_resources(self):
        """Background thread to monitor agent resources"""
        while self.monitoring_active:
            try:
                for agent in self.active_agents.values():
                    if agent.state in [AgentState.IDLE, AgentState.ACTIVE, AgentState.BUSY]:
                        self._update_resource_usage(agent)
                
                # Check for unresponsive agents
                current_time = time.time()
                for agent in list(self.active_agents.values()):
                    if (current_time - agent.last_activity) > 300:  # 5 minutes
                        logger.warning(f"Agent {agent.agent_id} appears unresponsive")
                        agent.state = AgentState.ERROR
                        agent.error_message = "Unresponsive"
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(30)  # Wait longer on error
    
    def start_monitoring(self):
        """Start resource monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.resource_monitor_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self.resource_monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring thread"""
        self.monitoring_active = False
        if self.resource_monitor_thread:
            self.resource_monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def ensure_minimum_agents(self):
        """Ensure minimum number of agents per role"""
        for role in self.agent_roles:
            active_role_agents = [a for a in self.active_agents.values() if a.role == role]
            active_count = len(active_role_agents)
            
            if active_count < self.min_agents_per_role:
                needed = self.min_agents_per_role - active_count
                logger.info(f"Spawning {needed} agents for role {role}")
                
                for _ in range(needed):
                    self.spawn_agent(role)
    
    def scale_agents(self, target_load: float = 0.8):
        """Auto-scale agents based on load"""
        if not self.auto_scale:
            return
        
        # Calculate current load per role
        for role in self.agent_roles:
            role_agents = [a for a in self.active_agents.values() if a.role == role]
            if not role_agents:
                continue
            
            busy_agents = [a for a in role_agents if a.state == AgentState.BUSY]
            current_load = len(busy_agents) / len(role_agents) if role_agents else 0
            
            # Scale up if load is high
            if current_load > target_load and len(role_agents) < self.max_agents_per_role:
                logger.info(f"Scaling up role {role} (load: {current_load:.2f})")
                self.spawn_agent(role)
            
            # Scale down if load is low (but keep minimum)
            elif current_load < 0.2 and len(role_agents) > self.min_agents_per_role:
                # Terminate least active agent
                idle_agents = [a for a in role_agents if a.state == AgentState.IDLE]
                if idle_agents:
                    agent_to_terminate = min(idle_agents, key=lambda a: a.last_activity)
                    logger.info(f"Scaling down role {role} (load: {current_load:.2f})")
                    self.terminate_agent(agent_to_terminate.agent_id)
    
    def get_pool_status(self) -> Dict:
        """Get comprehensive pool status"""
        status = {
            "total_agents": len(self.active_agents),
            "max_total_agents": self.max_total_agents,
            "agent_states": {},
            "role_distribution": {},
            "resource_usage": {
                "total_memory_mb": 0,
                "average_cpu_percent": 0
            },
            "task_statistics": {
                "total_completed": 0,
                "total_failed": 0,
                "currently_busy": 0
            }
        }
        
        # Collect statistics
        for state in AgentState:
            status["agent_states"][state.value] = 0
        
        total_memory = 0
        total_cpu = 0
        
        for agent in self.active_agents.values():
            # State distribution
            status["agent_states"][agent.state.value] += 1
            
            # Role distribution
            role = agent.role
            if role not in status["role_distribution"]:
                status["role_distribution"][role] = {
                    "count": 0,
                    "idle": 0,
                    "busy": 0,
                    "error": 0
                }
            
            status["role_distribution"][role]["count"] += 1
            
            if agent.state == AgentState.IDLE:
                status["role_distribution"][role]["idle"] += 1
            elif agent.state == AgentState.BUSY:
                status["role_distribution"][role]["busy"] += 1
            elif agent.state == AgentState.ERROR:
                status["role_distribution"][role]["error"] += 1
            
            # Resource usage
            total_memory += agent.memory_usage_mb
            total_cpu += agent.cpu_usage_percent
            
            # Task statistics
            status["task_statistics"]["total_completed"] += agent.completed_tasks
            status["task_statistics"]["total_failed"] += agent.failed_tasks
            
            if agent.state == AgentState.BUSY:
                status["task_statistics"]["currently_busy"] += 1
        
        # Calculate averages
        agent_count = len(self.active_agents)
        if agent_count > 0:
            status["resource_usage"]["total_memory_mb"] = total_memory
            status["resource_usage"]["average_cpu_percent"] = total_cpu / agent_count
        
        return status
    
    def shutdown(self):
        """Graceful shutdown of agent pool"""
        logger.info("Shutting down AgentPool")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Terminate all agents
        agent_ids = list(self.active_agents.keys())
        for agent_id in agent_ids:
            self.terminate_agent(agent_id)
        
        logger.info("AgentPool shutdown complete")

# Example usage
if __name__ == "__main__":
    def main():
        # Initialize pool
        pool = AgentPool(max_total_agents=10, auto_scale=True)
        
        # Ensure minimum agents
        pool.ensure_minimum_agents()
        
        # Print status
        status = pool.get_pool_status()
        print(f"\nAgent Pool Status:")
        print(f"Total Agents: {status['total_agents']}")
        print(f"Role Distribution: {status['role_distribution']}")
        
        # Simulate some work
        time.sleep(2)
        
        # Get an agent
        agent_id = pool.get_available_agent("python_backend_coder")
        if agent_id:
            print(f"\nGot agent: {agent_id}")
            
            # Assign task
            pool.assign_task(agent_id, "test_task_1")
            
            # Simulate work
            time.sleep(1)
            
            # Complete task
            pool.complete_task(agent_id, success=True)
            
            print(f"Task completed by {agent_id}")
        
        # Final status
        status = pool.get_pool_status()
        print(f"\nFinal Status:")
        print(f"Completed Tasks: {status['task_statistics']['total_completed']}")
        
        # Shutdown
        pool.shutdown()
    
    main()