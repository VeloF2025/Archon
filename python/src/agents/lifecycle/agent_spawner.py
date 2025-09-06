"""
Agent Spawner v3.0 - Agent Spawning Implementation  
Based on Agent_Lifecycle_Management_PRP.md specifications

NLNH Protocol: Real agent spawning with actual knowledge inheritance
DGTS Enforcement: No fake spawning, actual agent creation with validation
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .agent_v3 import AgentV3, KnowledgeItem
from .pool_manager import AgentSpec, AgentPoolManager
from .project_analyzer import ProjectAnalyzer, ProjectAnalysis

logger = logging.getLogger(__name__)


class AgentSpawner:
    """
    Agent Spawner for Archon v3.0
    Implementation for test_knowledge_inheritance() and related tests
    """
    
    def __init__(self, pool_manager: Optional[AgentPoolManager] = None):
        self.pool_manager = pool_manager or AgentPoolManager()
        self.project_analyzer = ProjectAnalyzer()

    async def spawn_agent(
        self,
        spec: AgentSpec,
        project_id: str,
        inherit_from: Optional[List[str]] = None
    ) -> AgentV3:
        """
        Spawn new agent with knowledge inheritance
        Implementation for test_knowledge_inheritance() and test_cross_project_knowledge_isolation()
        """
        # Validate inheritance constraints
        if inherit_from:
            await self._validate_inheritance(inherit_from, project_id)
        
        # Use pool manager to create agent (respects capacity limits)
        agent = await self.pool_manager.spawn_agent(spec, project_id)
        
        # Wait for global rules to be loaded (give it a moment)
        await asyncio.sleep(0.1)
        
        # Ensure global rules are loaded
        if not agent._rules_loaded:
            await agent._initialize_global_rules()
        
        # Inherit knowledge if specified
        if inherit_from:
            await self.inherit_knowledge(agent, inherit_from)
        
        # Log rules compliance status
        rules_count = 0
        if agent.rules_profile:
            rules_count = len(agent.rules_profile.global_rules) + len(agent.rules_profile.project_rules)
        
        logger.info(f"Agent {agent.name} spawned for project {project_id} "
                   f"with {len(inherit_from or [])} inheritance sources "
                   f"and {rules_count} applicable rules")
        
        return agent

    async def inherit_knowledge(self, new_agent: AgentV3, source_agent_ids: List[str]) -> None:
        """
        Transfer relevant knowledge from existing agents
        Implementation for test_knowledge_inheritance()
        """
        if not source_agent_ids:
            return
            
        inherited_count = 0
        
        for source_agent_id in source_agent_ids:
            source_agent = await self.pool_manager.get_agent_by_id(source_agent_id)
            if not source_agent:
                logger.warning(f"Source agent {source_agent_id} not found for inheritance")
                continue
            
            # Get knowledge items from source agent
            source_knowledge = await source_agent.get_knowledge_items()
            
            for knowledge_item in source_knowledge:
                # Only inherit high-confidence knowledge (> 0.7)
                if knowledge_item.confidence > 0.7:
                    # Create new knowledge item for the new agent
                    inherited_item = await new_agent.add_knowledge_item({
                        "type": knowledge_item.item_type,
                        "content": knowledge_item.content,
                        "confidence": knowledge_item.confidence,
                        "usage_count": 0,  # Reset usage for new agent
                        "success_count": 0,
                        "failure_count": 0,
                        "metadata": {
                            **knowledge_item.metadata,
                            "inherited_from": source_agent_id,
                            "inherited_at": datetime.now().isoformat()
                        }
                    })
                    inherited_count += 1
        
        logger.info(f"Agent {new_agent.name} inherited {inherited_count} knowledge items")

    async def _validate_inheritance(self, source_agent_ids: List[str], target_project_id: str) -> None:
        """
        Validate that knowledge inheritance is allowed
        Implementation for test_cross_project_knowledge_isolation()
        """
        for source_agent_id in source_agent_ids:
            source_agent = await self.pool_manager.get_agent_by_id(source_agent_id)
            if not source_agent:
                continue
                
            # Prevent cross-project knowledge inheritance
            if source_agent.project_id != target_project_id:
                raise ValueError(
                    f"Cannot inherit knowledge across projects. "
                    f"Source agent {source_agent_id} belongs to project {source_agent.project_id}, "
                    f"target project is {target_project_id}"
                )

    async def spawn_agents_for_project(self, project_id: str, project_files: Optional[Dict[str, Any]] = None) -> List[AgentV3]:
        """
        Analyze project and spawn all required agents
        Implementation for test_required_agents_determination()
        """
        # Analyze project to determine required agents
        analysis = await self.project_analyzer.analyze_project(project_id, project_files)
        required_specs = await self.project_analyzer.determine_required_agents(analysis)
        
        spawned_agents = []
        failed_spawns = []
        
        # Spawn agents based on analysis
        for spec in required_specs:
            try:
                agent = await self.spawn_agent(spec, project_id)
                spawned_agents.append(agent)
            except ValueError as e:
                # Pool capacity exceeded, log and continue
                failed_spawns.append({"spec": spec, "error": str(e)})
                logger.warning(f"Failed to spawn {spec.agent_type} ({spec.model_tier}): {e}")
        
        logger.info(f"Project {project_id} setup complete: "
                   f"{len(spawned_agents)} agents spawned, "
                   f"{len(failed_spawns)} failed due to capacity")
        
        if failed_spawns:
            # Log detailed failure information
            for failure in failed_spawns:
                logger.warning(f"Failed spawn: {failure['spec'].agent_type} - {failure['error']}")
        
        return spawned_agents

    async def get_similar_agents(self, agent_type: str, project_id: str) -> List[AgentV3]:
        """Get existing agents of similar type for knowledge inheritance"""
        all_project_agents = await self.pool_manager.get_agents_by_project(project_id)
        
        # Find agents with same or similar type
        similar_agents = [
            agent for agent in all_project_agents
            if agent.agent_type == agent_type and agent.knowledge_items
        ]
        
        # Sort by knowledge quality (agents with high-confidence items first)
        similar_agents.sort(
            key=lambda a: sum(item.confidence for item in a.knowledge_items),
            reverse=True
        )
        
        return similar_agents

    async def spawn_specialized_agent(
        self,
        base_type: str,
        specialization: str,
        model_tier: str,
        project_id: str,
        inherit_knowledge: bool = True
    ) -> AgentV3:
        """
        Spawn a specialized variant of an existing agent type
        """
        # Create specialized spec
        spec = AgentSpec(base_type, model_tier, specialization)
        
        # Find similar agents for knowledge inheritance
        inherit_from = None
        if inherit_knowledge:
            similar_agents = await self.get_similar_agents(base_type, project_id)
            if similar_agents:
                # Inherit from the most knowledgeable similar agent
                inherit_from = [similar_agents[0].id]
        
        # Spawn the specialized agent
        agent = await self.spawn_agent(spec, project_id, inherit_from)
        
        logger.info(f"Specialized agent {agent.name} spawned with specialization: {specialization}")
        
        return agent

    async def clone_agent(self, source_agent_id: str, new_name: Optional[str] = None) -> AgentV3:
        """
        Create a clone of an existing agent with identical knowledge
        """
        source_agent = await self.pool_manager.get_agent_by_id(source_agent_id)
        if not source_agent:
            raise ValueError(f"Source agent {source_agent_id} not found")
        
        # Create clone spec
        clone_spec = AgentSpec(
            source_agent.agent_type,
            source_agent.model_tier,
            source_agent.specialization
        )
        
        # Spawn clone
        clone_name = new_name or f"{source_agent.name}-clone"
        clone_agent = await self.spawn_agent(clone_spec, source_agent.project_id)
        clone_agent.name = clone_name
        
        # Copy all knowledge (not just high-confidence items)
        for knowledge_item in source_agent.knowledge_items:
            await clone_agent.add_knowledge_item({
                "type": knowledge_item.item_type,
                "content": knowledge_item.content,
                "confidence": knowledge_item.confidence,
                "usage_count": 0,  # Reset usage stats
                "success_count": 0,
                "failure_count": 0,
                "metadata": {
                    **knowledge_item.metadata,
                    "cloned_from": source_agent_id,
                    "cloned_at": datetime.now().isoformat()
                }
            })
        
        logger.info(f"Agent {clone_agent.name} cloned from {source_agent.name} "
                   f"with {len(source_agent.knowledge_items)} knowledge items")
        
        return clone_agent

    async def get_spawn_recommendations(self, project_id: str) -> Dict[str, Any]:
        """
        Get recommendations for agents to spawn based on project needs
        """
        # Get current agents for the project
        existing_agents = await self.pool_manager.get_agents_by_project(project_id)
        
        # Get pool statistics to check capacity
        pool_stats = await self.pool_manager.get_pool_statistics()
        
        # Simple analysis of what might be needed
        existing_types = {agent.agent_type for agent in existing_agents}
        
        recommendations = {
            "existing_agents": len(existing_agents),
            "available_capacity": {
                tier: self.pool_manager.MAX_AGENTS[tier] - pool_stats.active_counts.get(tier, 0)
                for tier in self.pool_manager.MAX_AGENTS
            },
            "recommended_spawns": [],
            "capacity_constraints": []
        }
        
        # Basic recommendations based on common needs
        basic_needs = [
            ("code-formatter", "haiku", "Essential for code quality"),
            ("documentation-writer", "sonnet", "Important for project documentation"),
            ("test-engineer", "sonnet", "Critical for quality assurance")
        ]
        
        for agent_type, tier, reason in basic_needs:
            if agent_type not in existing_types:
                if pool_stats.can_spawn.get(tier, False):
                    recommendations["recommended_spawns"].append({
                        "agent_type": agent_type,
                        "model_tier": tier,
                        "reason": reason,
                        "priority": "high" if agent_type == "code-formatter" else "medium"
                    })
                else:
                    recommendations["capacity_constraints"].append({
                        "agent_type": agent_type,
                        "model_tier": tier,
                        "reason": f"No capacity available for {tier} tier"
                    })
        
        return recommendations

    async def cleanup_project_agents(self, project_id: str) -> Dict[str, int]:
        """
        Archive all agents for a project (project cleanup)
        """
        project_agents = await self.pool_manager.get_agents_by_project(project_id)
        
        archived_count = 0
        failed_count = 0
        
        for agent in project_agents:
            try:
                if agent.state != agent.state.ARCHIVED:
                    await agent.transition_to_archived(f"Project {project_id} cleanup")
                    archived_count += 1
            except Exception as e:
                logger.error(f"Failed to archive agent {agent.id} during cleanup: {e}")
                failed_count += 1
        
        logger.info(f"Project {project_id} cleanup: {archived_count} agents archived, "
                   f"{failed_count} failed")
        
        return {"archived": archived_count, "failed": failed_count}