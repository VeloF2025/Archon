"""
Database Service for Archon 3.0 Intelligence-Tiered Agent Management

This service provides database operations for the Intelligence-Tiered Adaptive Agent Management System:
- Agent CRUD operations with lifecycle management
- Intelligence tier routing and task complexity assessment  
- Knowledge management with confidence evolution
- Cost tracking and budget enforcement
- Real-time collaboration and pub/sub messaging
- Global rules integration and violation tracking

Integrates with existing Archon Supabase database.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import UUID

import asyncpg
from supabase import create_client, Client

from .agent_models import (
    AgentV3, AgentState, ModelTier, AgentType,
    AgentStateHistory, AgentPool, TaskComplexity, RoutingRule,
    AgentKnowledge, KnowledgeEvolution, SharedKnowledge,
    CostTracking, BudgetConstraint, ROIAnalysis,
    SharedContext, BroadcastMessage, TopicSubscription, MessageAcknowledgment,
    RulesProfile, RuleViolation,
    AgentPerformanceMetrics, ProjectIntelligenceOverview, CostOptimizationRecommendation,
    calculate_complexity_score, recommend_tier, calculate_tier_cost, evolve_confidence
)


logger = logging.getLogger(__name__)


class AgentDatabaseService:
    """Database service for Intelligence-Tiered Agent Management system"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize with Supabase connection"""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.connection_pool: Optional[asyncpg.Pool] = None
    
    async def initialize_connection_pool(self, database_url: str):
        """Initialize asyncpg connection pool for complex operations"""
        self.connection_pool = await asyncpg.create_pool(
            database_url, 
            min_size=2, 
            max_size=10,
            command_timeout=30
        )
        logger.info("Initialized agent database connection pool")
    
    async def close(self):
        """Clean up database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Closed agent database connection pool")

    # =====================================================
    # AGENT LIFECYCLE MANAGEMENT
    # =====================================================
    
    async def create_agent(self, agent: AgentV3) -> AgentV3:
        """Create a new agent with lifecycle tracking"""
        try:
            # Insert agent into database
            # Handle both enum objects and string values
            agent_type_value = agent.agent_type.value if hasattr(agent.agent_type, 'value') else agent.agent_type
            model_tier_value = agent.model_tier.value if hasattr(agent.model_tier, 'value') else agent.model_tier
            state_value = agent.state.value if hasattr(agent.state, 'value') else agent.state
            
            result = self.supabase.table("archon_agents_v3").insert({
                "id": str(agent.id),
                "name": agent.name,
                "agent_type": agent_type_value,
                "model_tier": model_tier_value,
                "project_id": str(agent.project_id),
                "state": state_value,
                "state_changed_at": agent.state_changed_at.isoformat(),
                "tasks_completed": agent.tasks_completed,
                "success_rate": float(agent.success_rate),
                "avg_completion_time_seconds": agent.avg_completion_time_seconds,
                "memory_usage_mb": agent.memory_usage_mb,
                "cpu_usage_percent": float(agent.cpu_usage_percent),
                "capabilities": agent.capabilities,
                "rules_profile_id": str(agent.rules_profile_id) if agent.rules_profile_id else None,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat()
            }).execute()
            
            if result.data:
                # Ensure agent pool exists for project
                await self._ensure_agent_pool_exists(agent.project_id)
                logger.info(f"Created agent {agent.name} ({agent.id}) for project {agent.project_id}")
                return agent
            else:
                raise Exception("Failed to create agent in database")
                
        except Exception as e:
            logger.error(f"Failed to create agent {agent.name}: {e}")
            raise
    
    async def get_agent_by_id(self, agent_id: UUID) -> Optional[AgentV3]:
        """Retrieve agent by ID"""
        try:
            result = self.supabase.table("archon_agents_v3").select("*").eq("id", str(agent_id)).execute()
            
            if result.data:
                agent_data = result.data[0]
                return AgentV3(
                    id=UUID(agent_data["id"]),
                    name=agent_data["name"],
                    agent_type=AgentType(agent_data["agent_type"]),
                    model_tier=ModelTier(agent_data["model_tier"]),
                    project_id=UUID(agent_data["project_id"]),
                    state=AgentState(agent_data["state"]),
                    state_changed_at=datetime.fromisoformat(agent_data["state_changed_at"]),
                    tasks_completed=agent_data["tasks_completed"],
                    success_rate=Decimal(str(agent_data["success_rate"])),
                    avg_completion_time_seconds=agent_data["avg_completion_time_seconds"],
                    last_active_at=datetime.fromisoformat(agent_data["last_active_at"]) if agent_data["last_active_at"] else None,
                    memory_usage_mb=agent_data["memory_usage_mb"],
                    cpu_usage_percent=Decimal(str(agent_data["cpu_usage_percent"])),
                    capabilities=agent_data["capabilities"] or {},
                    rules_profile_id=UUID(agent_data["rules_profile_id"]) if agent_data["rules_profile_id"] else None,
                    created_at=datetime.fromisoformat(agent_data["created_at"]),
                    updated_at=datetime.fromisoformat(agent_data["updated_at"])
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def update_agent_state(self, agent_id: UUID, new_state: AgentState, reason: str = "System transition") -> bool:
        """Update agent state with history tracking"""
        try:
            # Get current agent data
            current_agent = await self.get_agent_by_id(agent_id)
            if not current_agent:
                return False
            
            # Update agent state
            result = self.supabase.table("archon_agents_v3").update({
                "state": new_state.value,
                "state_changed_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }).eq("id", str(agent_id)).execute()
            
            if result.data:
                # Log state transition (trigger will handle this, but we can also do it manually)
                await self._log_state_transition(agent_id, current_agent.state, new_state, reason)
                logger.info(f"Updated agent {agent_id} state: {current_agent.state} -> {new_state}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to update agent state for {agent_id}: {e}")
            return False
    
    async def get_agents_by_project(self, project_id: UUID, state_filter: Optional[AgentState] = None) -> List[AgentV3]:
        """Get all agents for a project, optionally filtered by state"""
        try:
            query = self.supabase.table("archon_agents_v3").select("*").eq("project_id", str(project_id))
            
            if state_filter:
                query = query.eq("state", state_filter.value)
            
            result = query.execute()
            
            agents = []
            for agent_data in result.data:
                agents.append(AgentV3(
                    id=UUID(agent_data["id"]),
                    name=agent_data["name"],
                    agent_type=AgentType(agent_data["agent_type"]),
                    model_tier=ModelTier(agent_data["model_tier"]),
                    project_id=UUID(agent_data["project_id"]),
                    state=AgentState(agent_data["state"]),
                    state_changed_at=datetime.fromisoformat(agent_data["state_changed_at"]),
                    tasks_completed=agent_data["tasks_completed"],
                    success_rate=Decimal(str(agent_data["success_rate"])),
                    avg_completion_time_seconds=agent_data["avg_completion_time_seconds"],
                    last_active_at=datetime.fromisoformat(agent_data["last_active_at"]) if agent_data["last_active_at"] else None,
                    memory_usage_mb=agent_data["memory_usage_mb"],
                    cpu_usage_percent=Decimal(str(agent_data["cpu_usage_percent"])),
                    capabilities=agent_data["capabilities"] or {},
                    rules_profile_id=UUID(agent_data["rules_profile_id"]) if agent_data["rules_profile_id"] else None,
                    created_at=datetime.fromisoformat(agent_data["created_at"]),
                    updated_at=datetime.fromisoformat(agent_data["updated_at"])
                ))
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to get agents for project {project_id}: {e}")
            return []

    async def hibernate_idle_agents(self, project_id: UUID, idle_timeout_minutes: int = 30) -> int:
        """Hibernate agents that have been idle for too long"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=idle_timeout_minutes)
            
            # Find idle agents to hibernate
            idle_agents = self.supabase.table("archon_agents_v3").select("id").eq("project_id", str(project_id)).eq("state", "IDLE").lt("last_active_at", cutoff_time.isoformat()).execute()
            
            hibernated_count = 0
            for agent in idle_agents.data:
                if await self.update_agent_state(UUID(agent["id"]), AgentState.HIBERNATED, "Automatic hibernation due to inactivity"):
                    hibernated_count += 1
            
            logger.info(f"Hibernated {hibernated_count} idle agents in project {project_id}")
            return hibernated_count
            
        except Exception as e:
            logger.error(f"Failed to hibernate idle agents for project {project_id}: {e}")
            return 0

    # =====================================================
    # INTELLIGENCE TIER ROUTING
    # =====================================================
    
    async def assess_task_complexity(self, task_id: UUID, technical: float, domain: float, code_volume: float, integration: float) -> TaskComplexity:
        """Assess task complexity and recommend tier"""
        try:
            overall_complexity = calculate_complexity_score(technical, domain, code_volume, integration)
            recommended_tier = recommend_tier(overall_complexity)
            
            complexity = TaskComplexity(
                task_id=task_id,
                technical_complexity=Decimal(str(technical)),
                domain_expertise_required=Decimal(str(domain)),
                code_volume_complexity=Decimal(str(code_volume)),
                integration_complexity=Decimal(str(integration)),
                overall_complexity=Decimal(str(overall_complexity)),
                recommended_tier=recommended_tier,
                assigned_tier=recommended_tier,  # Default to recommended
                tier_justification=f"Complexity score: {overall_complexity:.3f} -> {recommended_tier.value}"
            )
            
            # Store in database
            result = self.supabase.table("archon_task_complexity").insert({
                "id": str(complexity.id),
                "task_id": str(task_id),
                "technical_complexity": float(complexity.technical_complexity),
                "domain_expertise_required": float(complexity.domain_expertise_required),
                "code_volume_complexity": float(complexity.code_volume_complexity),
                "integration_complexity": float(complexity.integration_complexity),
                "recommended_tier": recommended_tier.value,
                "assigned_tier": recommended_tier.value,
                "tier_justification": complexity.tier_justification,
                "assessed_at": complexity.assessed_at.isoformat()
            }).execute()
            
            if result.data:
                logger.info(f"Assessed task {task_id} complexity: {overall_complexity:.3f} -> {recommended_tier.value}")
                return complexity
            else:
                raise Exception("Failed to store task complexity assessment")
                
        except Exception as e:
            logger.error(f"Failed to assess task complexity for {task_id}: {e}")
            raise

    async def get_optimal_agent_for_task(self, project_id: UUID, task_complexity: TaskComplexity, agent_type: AgentType) -> Optional[AgentV3]:
        """Find the optimal agent for a task based on complexity and availability"""
        try:
            # Get agents of the right type and tier
            agents = await self.get_agents_by_project(project_id)
            
            # Filter by agent type and tier
            suitable_agents = [
                agent for agent in agents 
                if agent.agent_type == agent_type 
                and agent.model_tier == task_complexity.assigned_tier
                and agent.state in [AgentState.IDLE, AgentState.ACTIVE]
            ]
            
            if not suitable_agents:
                # Try to find agents of different tiers if none available
                suitable_agents = [
                    agent for agent in agents
                    if agent.agent_type == agent_type
                    and agent.state in [AgentState.IDLE, AgentState.ACTIVE]
                ]
            
            if suitable_agents:
                # Prefer IDLE agents, then sort by success rate and availability
                idle_agents = [a for a in suitable_agents if a.state == AgentState.IDLE]
                if idle_agents:
                    # Sort by success rate (descending) and tasks completed (ascending for load balancing)
                    idle_agents.sort(key=lambda x: (-x.success_rate, x.tasks_completed))
                    return idle_agents[0]
                else:
                    # All are active, pick least loaded
                    suitable_agents.sort(key=lambda x: x.tasks_completed)
                    return suitable_agents[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find optimal agent for task: {e}")
            return None

    # =====================================================
    # KNOWLEDGE MANAGEMENT  
    # =====================================================
    
    async def store_agent_knowledge(self, knowledge: AgentKnowledge) -> bool:
        """Store knowledge item for an agent"""
        try:
            result = self.supabase.table("archon_agent_knowledge").insert({
                "id": str(knowledge.id),
                "agent_id": str(knowledge.agent_id),
                "knowledge_type": knowledge.knowledge_type,
                "title": knowledge.title,
                "content": knowledge.content,
                "confidence": float(knowledge.confidence),
                "success_count": knowledge.success_count,
                "failure_count": knowledge.failure_count,
                "last_used_at": knowledge.last_used_at.isoformat() if knowledge.last_used_at else None,
                "context_tags": knowledge.context_tags,
                "project_id": str(knowledge.project_id),
                "task_context": knowledge.task_context,
                "storage_layer": knowledge.storage_layer,
                "created_at": knowledge.created_at.isoformat()
            }).execute()
            
            if result.data:
                logger.info(f"Stored knowledge '{knowledge.title}' for agent {knowledge.agent_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to store agent knowledge: {e}")
            return False

    async def search_agent_knowledge(self, agent_id: UUID, query_embedding: List[float], limit: int = 10) -> List[AgentKnowledge]:
        """Search agent knowledge using vector similarity"""
        try:
            if not self.connection_pool:
                logger.warning("Connection pool not initialized for vector search")
                return []
            
            async with self.connection_pool.acquire() as conn:
                # Use pgvector for similarity search
                query = """
                    SELECT id, agent_id, knowledge_type, title, content, confidence, 
                           success_count, failure_count, last_used_at, context_tags,
                           project_id, task_context, storage_layer, created_at, updated_at,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM archon_agent_knowledge 
                    WHERE agent_id = $2 AND storage_layer != 'temporary'
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """
                
                rows = await conn.fetch(query, query_embedding, str(agent_id), limit)
                
                knowledge_items = []
                for row in rows:
                    knowledge_items.append(AgentKnowledge(
                        id=row["id"],
                        agent_id=UUID(row["agent_id"]),
                        knowledge_type=row["knowledge_type"],
                        title=row["title"],
                        content=row["content"],
                        confidence=Decimal(str(row["confidence"])),
                        success_count=row["success_count"],
                        failure_count=row["failure_count"],
                        last_used_at=row["last_used_at"],
                        context_tags=row["context_tags"],
                        project_id=UUID(row["project_id"]),
                        task_context=row["task_context"],
                        storage_layer=row["storage_layer"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"]
                    ))
                
                logger.info(f"Found {len(knowledge_items)} knowledge items for agent {agent_id}")
                return knowledge_items
                
        except Exception as e:
            logger.error(f"Failed to search agent knowledge: {e}")
            return []

    async def update_knowledge_confidence(self, knowledge_id: UUID, success: bool) -> bool:
        """Update knowledge confidence based on usage outcome"""
        try:
            # Get current knowledge
            result = self.supabase.table("archon_agent_knowledge").select("*").eq("id", str(knowledge_id)).execute()
            
            if not result.data:
                return False
            
            current_data = result.data[0]
            current_confidence = Decimal(str(current_data["confidence"]))
            
            # Update counts and confidence
            if success:
                new_success_count = current_data["success_count"] + 1
                new_failure_count = current_data["failure_count"]
            else:
                new_success_count = current_data["success_count"]
                new_failure_count = current_data["failure_count"] + 1
            
            # Confidence will be evolved by database trigger
            update_result = self.supabase.table("archon_agent_knowledge").update({
                "success_count": new_success_count,
                "failure_count": new_failure_count,
                "last_used_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }).eq("id", str(knowledge_id)).execute()
            
            if update_result.data:
                logger.info(f"Updated knowledge confidence for {knowledge_id}: success={success}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update knowledge confidence: {e}")
            return False

    # =====================================================
    # COST TRACKING AND OPTIMIZATION
    # =====================================================
    
    async def track_agent_cost(self, agent_id: UUID, project_id: UUID, task_id: Optional[UUID], 
                              input_tokens: int, output_tokens: int, model_tier: ModelTier,
                              task_duration_seconds: Optional[int] = None, success: bool = True) -> bool:
        """Track cost for agent task execution"""
        try:
            # Calculate costs
            input_cost, output_cost = calculate_tier_cost(model_tier, input_tokens, output_tokens)
            
            cost_record = CostTracking(
                agent_id=agent_id,
                project_id=project_id,
                task_id=task_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                model_tier=model_tier,
                task_duration_seconds=task_duration_seconds,
                success=success
            )
            
            # Store in database
            result = self.supabase.table("archon_cost_tracking").insert({
                "id": str(cost_record.id),
                "agent_id": str(agent_id),
                "project_id": str(project_id),
                "task_id": str(task_id) if task_id else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": float(input_cost),
                "output_cost": float(output_cost),
                "model_tier": model_tier.value,
                "task_duration_seconds": task_duration_seconds,
                "success": success,
                "recorded_at": cost_record.recorded_at.isoformat()
            }).execute()
            
            if result.data:
                total_cost = input_cost + output_cost
                logger.info(f"Tracked cost ${total_cost:.4f} for agent {agent_id} ({input_tokens}+{output_tokens} tokens)")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to track agent cost: {e}")
            return False

    async def check_budget_constraints(self, project_id: UUID) -> Dict[str, Any]:
        """Check if project is within budget constraints"""
        try:
            # Get budget constraints
            budget_result = self.supabase.table("archon_budget_constraints").select("*").eq("project_id", str(project_id)).execute()
            
            if not budget_result.data:
                return {"status": "no_budget_set", "within_limits": True}
            
            budget = budget_result.data[0]
            
            # Check current spending
            today = datetime.now().date()
            month_start = today.replace(day=1)
            
            # Get daily spending
            daily_cost_result = self.supabase.table("archon_cost_tracking").select("total_cost").eq("project_id", str(project_id)).gte("recorded_at", today.isoformat()).execute()
            
            daily_spend = sum(Decimal(str(row["total_cost"])) for row in daily_cost_result.data)
            
            # Get monthly spending
            monthly_cost_result = self.supabase.table("archon_cost_tracking").select("total_cost").eq("project_id", str(project_id)).gte("recorded_at", month_start.isoformat()).execute()
            
            monthly_spend = sum(Decimal(str(row["total_cost"])) for row in monthly_cost_result.data)
            
            # Check constraints
            constraints = {
                "daily_budget": Decimal(str(budget["daily_budget"])) if budget["daily_budget"] else None,
                "monthly_budget": Decimal(str(budget["monthly_budget"])) if budget["monthly_budget"] else None,
                "warning_threshold": Decimal(str(budget["warning_threshold"])),
                "critical_threshold": Decimal(str(budget["critical_threshold"])),
                "daily_spend": daily_spend,
                "monthly_spend": monthly_spend,
                "within_limits": True,
                "alerts": []
            }
            
            # Check daily budget
            if constraints["daily_budget"]:
                daily_usage_pct = (daily_spend / constraints["daily_budget"]) * 100
                if daily_usage_pct >= constraints["critical_threshold"]:
                    constraints["alerts"].append({"type": "critical", "message": f"Daily budget {daily_usage_pct:.1f}% used"})
                    constraints["within_limits"] = False
                elif daily_usage_pct >= constraints["warning_threshold"]:
                    constraints["alerts"].append({"type": "warning", "message": f"Daily budget {daily_usage_pct:.1f}% used"})
            
            # Check monthly budget
            if constraints["monthly_budget"]:
                monthly_usage_pct = (monthly_spend / constraints["monthly_budget"]) * 100
                if monthly_usage_pct >= constraints["critical_threshold"]:
                    constraints["alerts"].append({"type": "critical", "message": f"Monthly budget {monthly_usage_pct:.1f}% used"})
                    constraints["within_limits"] = False
                elif monthly_usage_pct >= constraints["warning_threshold"]:
                    constraints["alerts"].append({"type": "warning", "message": f"Monthly budget {monthly_usage_pct:.1f}% used"})
            
            return constraints
            
        except Exception as e:
            logger.error(f"Failed to check budget constraints: {e}")
            return {"status": "error", "within_limits": True}

    async def generate_cost_optimization_recommendations(self, project_id: UUID) -> List[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations for project agents"""
        try:
            if not self.connection_pool:
                logger.warning("Connection pool not initialized for cost optimization")
                return []
            
            async with self.connection_pool.acquire() as conn:
                # Get agent performance and cost data
                query = """
                    SELECT 
                        a.id as agent_id,
                        a.agent_type,
                        a.model_tier,
                        COUNT(ct.id) as task_count,
                        AVG(ct.total_cost) as avg_cost_per_task,
                        SUM(ct.total_cost) as total_cost,
                        AVG(CASE WHEN ct.success THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM archon_agents_v3 a
                    LEFT JOIN archon_cost_tracking ct ON a.id = ct.agent_id
                    WHERE a.project_id = $1 
                    AND ct.recorded_at > NOW() - INTERVAL '30 days'
                    GROUP BY a.id, a.agent_type, a.model_tier
                    HAVING COUNT(ct.id) >= 5
                """
                
                rows = await conn.fetch(query, str(project_id))
                
                recommendations = []
                for row in rows:
                    # Determine recommendation based on performance
                    current_tier = ModelTier(row["model_tier"])
                    success_rate = Decimal(str(row["success_rate"] or 0))
                    avg_cost = Decimal(str(row["avg_cost_per_task"] or 0))
                    total_cost = Decimal(str(row["total_cost"] or 0))
                    
                    recommendation = "OPTIMAL"
                    potential_savings = Decimal("0")
                    
                    if current_tier == ModelTier.OPUS and success_rate > Decimal("0.95") and avg_cost > Decimal("0.50"):
                        recommendation = "CONSIDER_SONNET"
                        potential_savings = total_cost * Decimal("0.8")  # ~80% cost reduction
                    elif current_tier == ModelTier.SONNET and success_rate < Decimal("0.80"):
                        recommendation = "CONSIDER_OPUS"
                    elif current_tier == ModelTier.SONNET and success_rate > Decimal("0.98") and avg_cost < Decimal("0.05"):
                        recommendation = "CONSIDER_HAIKU"
                        potential_savings = total_cost * Decimal("0.83")  # ~83% cost reduction
                    elif current_tier == ModelTier.HAIKU and success_rate < Decimal("0.70"):
                        recommendation = "CONSIDER_SONNET"
                    
                    recommendations.append(CostOptimizationRecommendation(
                        agent_id=UUID(row["agent_id"]),
                        agent_type=AgentType(row["agent_type"]),
                        current_tier=current_tier,
                        total_cost=total_cost,
                        success_rate=success_rate,
                        avg_cost_per_task=avg_cost,
                        recommendation=recommendation,
                        potential_monthly_savings=potential_savings
                    ))
                
                logger.info(f"Generated {len(recommendations)} cost optimization recommendations for project {project_id}")
                return recommendations
                
        except Exception as e:
            logger.error(f"Failed to generate cost optimization recommendations: {e}")
            return []

    # =====================================================
    # REAL-TIME COLLABORATION
    # =====================================================
    
    async def create_shared_context(self, task_id: UUID, project_id: UUID, context_name: str) -> SharedContext:
        """Create a shared collaboration context"""
        try:
            context = SharedContext(
                task_id=task_id,
                project_id=project_id,
                context_name=context_name
            )
            
            result = self.supabase.table("archon_shared_contexts").insert({
                "id": str(context.id),
                "task_id": str(task_id),
                "project_id": str(project_id),
                "context_name": context_name,
                "discoveries": context.discoveries,
                "blockers": context.blockers,
                "patterns": context.patterns,
                "participants": [str(p) for p in context.participants],
                "is_active": context.is_active,
                "created_at": context.created_at.isoformat()
            }).execute()
            
            if result.data:
                logger.info(f"Created shared context '{context_name}' for task {task_id}")
                return context
            else:
                raise Exception("Failed to create shared context in database")
                
        except Exception as e:
            logger.error(f"Failed to create shared context: {e}")
            raise

    async def broadcast_message(self, message: BroadcastMessage) -> bool:
        """Broadcast a message to subscribed agents"""
        try:
            result = self.supabase.table("archon_broadcast_messages").insert({
                "id": str(message.id),
                "message_id": message.message_id,
                "topic": message.topic,
                "content": message.content,
                "message_type": message.message_type,
                "priority": message.priority,
                "sender_id": str(message.sender_id) if message.sender_id else None,
                "target_agents": [str(a) for a in message.target_agents],
                "target_topics": message.target_topics,
                "sent_at": message.sent_at.isoformat(),
                "expires_at": message.expires_at.isoformat() if message.expires_at else None
            }).execute()
            
            if result.data:
                # Find subscribers and deliver message
                await self._deliver_broadcast_message(message)
                logger.info(f"Broadcast message '{message.message_id}' to topic '{message.topic}'")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False

    async def subscribe_agent_to_topic(self, agent_id: UUID, topic: str, priority_filter: int = 1) -> bool:
        """Subscribe an agent to a collaboration topic"""
        try:
            subscription = TopicSubscription(
                agent_id=agent_id,
                topic=topic,
                priority_filter=priority_filter
            )
            
            result = self.supabase.table("archon_topic_subscriptions").insert({
                "id": str(subscription.id),
                "agent_id": str(agent_id),
                "topic": topic,
                "priority_filter": priority_filter,
                "content_filters": subscription.content_filters,
                "is_active": subscription.is_active,
                "subscription_type": subscription.subscription_type,
                "callback_endpoint": subscription.callback_endpoint,
                "callback_timeout_seconds": subscription.callback_timeout_seconds,
                "created_at": subscription.created_at.isoformat()
            }).execute()
            
            if result.data:
                logger.info(f"Subscribed agent {agent_id} to topic '{topic}'")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe agent to topic: {e}")
            return False

    # =====================================================
    # ANALYTICS AND MONITORING
    # =====================================================
    
    async def get_agent_performance_metrics(self, agent_id: UUID) -> Optional[AgentPerformanceMetrics]:
        """Get comprehensive performance metrics for an agent"""
        try:
            if not self.connection_pool:
                return None
            
            async with self.connection_pool.acquire() as conn:
                # Get agent performance data (using the dashboard view)
                query = """
                    SELECT * FROM archon_agent_performance_dashboard 
                    WHERE id = $1
                """
                
                row = await conn.fetchrow(query, str(agent_id))
                
                if row:
                    return AgentPerformanceMetrics(
                        agent_id=UUID(row["id"]),
                        tasks_completed=row["tasks_completed"],
                        success_rate=Decimal(str(row["success_rate"])),
                        avg_completion_time_seconds=row["avg_completion_time_seconds"],
                        cost_last_30_days=Decimal(str(row["cost_last_30_days"])),
                        knowledge_items_count=row["knowledge_items_count"],
                        activity_level=row["activity_level"]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get agent performance metrics: {e}")
            return None

    async def get_project_intelligence_overview(self, project_id: UUID) -> Optional[ProjectIntelligenceOverview]:
        """Get project-level intelligence and performance overview"""
        try:
            if not self.connection_pool:
                return None
            
            async with self.connection_pool.acquire() as conn:
                # Get project overview data (using the dashboard view)
                query = """
                    SELECT * FROM archon_project_intelligence_overview 
                    WHERE project_id = $1
                """
                
                row = await conn.fetchrow(query, str(project_id))
                
                if row:
                    return ProjectIntelligenceOverview(
                        project_id=UUID(row["project_id"]),
                        project_name=row["project_name"],
                        total_agents=row["total_agents"],
                        active_agents=row["active_agents"],
                        opus_agents=row["opus_agents"],
                        sonnet_agents=row["sonnet_agents"],
                        haiku_agents=row["haiku_agents"],
                        avg_success_rate=Decimal(str(row["avg_success_rate"] or 0)),
                        total_tasks_completed=row["total_tasks_completed"],
                        monthly_cost=Decimal(str(row["monthly_cost"])),
                        monthly_budget=Decimal(str(row["monthly_budget"])),
                        budget_utilization_percent=Decimal(str(row["budget_utilization_percent"])),
                        active_shared_contexts=row["active_shared_contexts"],
                        recent_broadcasts=row["recent_broadcasts"]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get project intelligence overview: {e}")
            return None

    # =====================================================
    # PRIVATE HELPER METHODS
    # =====================================================
    
    async def _ensure_agent_pool_exists(self, project_id: UUID) -> bool:
        """Ensure an agent pool exists for the project"""
        try:
            # Check if pool exists
            result = self.supabase.table("archon_agent_pools").select("id").eq("project_id", str(project_id)).execute()
            
            if not result.data:
                # Create default pool
                pool = AgentPool(project_id=project_id)
                
                create_result = self.supabase.table("archon_agent_pools").insert({
                    "id": str(pool.id),
                    "project_id": str(project_id),
                    "opus_limit": pool.opus_limit,
                    "sonnet_limit": pool.sonnet_limit,
                    "haiku_limit": pool.haiku_limit,
                    "auto_scaling_enabled": pool.auto_scaling_enabled,
                    "hibernation_timeout_minutes": pool.hibernation_timeout_minutes,
                    "created_at": pool.created_at.isoformat()
                }).execute()
                
                if create_result.data:
                    logger.info(f"Created agent pool for project {project_id}")
                    return True
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure agent pool exists: {e}")
            return False
    
    async def _log_state_transition(self, agent_id: UUID, from_state: AgentState, to_state: AgentState, reason: str):
        """Log agent state transition"""
        try:
            history = AgentStateHistory(
                agent_id=agent_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason
            )
            
            # Convert string states to enum if needed
            if isinstance(from_state, str):
                from_state = AgentState[from_state] if from_state else None
            if isinstance(to_state, str):
                to_state = AgentState[to_state]
            
            result = self.supabase.table("archon_agent_state_history").insert({
                "id": str(history.id),
                "agent_id": str(agent_id),
                "from_state": from_state.value if from_state else None,
                "to_state": to_state.value,
                "reason": reason,
                "changed_at": history.changed_at.isoformat()
            }).execute()
            
            if result.data:
                logger.debug(f"Logged state transition for agent {agent_id}: {from_state} -> {to_state}")
                
        except Exception as e:
            logger.error(f"Failed to log state transition: {e}")
    
    async def _deliver_broadcast_message(self, message: BroadcastMessage):
        """Deliver broadcast message to subscribed agents"""
        try:
            # Find relevant subscriptions
            subscriptions_result = self.supabase.table("archon_topic_subscriptions").select("agent_id, priority_filter").eq("topic", message.topic).eq("is_active", True).execute()
            
            delivered_count = 0
            for sub in subscriptions_result.data:
                # Check priority filter
                if message.priority >= sub["priority_filter"]:
                    # In a real implementation, this would trigger the callback or queue the message
                    # For now, we'll just increment the count
                    delivered_count += 1
            
            # Update delivery count
            if delivered_count > 0:
                self.supabase.table("archon_broadcast_messages").update({
                    "delivered_count": delivered_count
                }).eq("id", str(message.id)).execute()
                
                logger.info(f"Delivered message {message.message_id} to {delivered_count} agents")
            
        except Exception as e:
            logger.error(f"Failed to deliver broadcast message: {e}")


# =====================================================
# USAGE EXAMPLE AND SERVICE INITIALIZATION
# =====================================================

async def create_agent_service(supabase_url: str, supabase_key: str, database_url: Optional[str] = None) -> AgentDatabaseService:
    """Create and initialize the agent database service"""
    service = AgentDatabaseService(supabase_url, supabase_key)
    
    if database_url:
        await service.initialize_connection_pool(database_url)
    
    return service


# Example usage:
"""
# Initialize service
agent_service = await create_agent_service(
    supabase_url="https://your-project.supabase.co",
    supabase_key="your-service-key",
    database_url="postgresql://user:pass@host:port/db"
)

# Create a new agent
agent = AgentV3(
    name="Code Quality Reviewer",
    agent_type=AgentType.CODE_QUALITY_REVIEWER,
    model_tier=ModelTier.SONNET,
    project_id=UUID("project-uuid")
)

# Store in database
created_agent = await agent_service.create_agent(agent)

# Assess task complexity and find optimal agent
complexity = await agent_service.assess_task_complexity(
    task_id=UUID("task-uuid"),
    technical=0.7, domain=0.5, code_volume=0.8, integration=0.6
)

optimal_agent = await agent_service.get_optimal_agent_for_task(
    project_id=UUID("project-uuid"),
    task_complexity=complexity,
    agent_type=AgentType.CODE_QUALITY_REVIEWER
)

# Track costs
await agent_service.track_agent_cost(
    agent_id=optimal_agent.id,
    project_id=UUID("project-uuid"),
    task_id=UUID("task-uuid"),
    input_tokens=1000,
    output_tokens=500,
    model_tier=ModelTier.SONNET,
    success=True
)

# Check budget status
budget_status = await agent_service.check_budget_constraints(UUID("project-uuid"))

# Create shared context for collaboration
context = await agent_service.create_shared_context(
    task_id=UUID("task-uuid"),
    project_id=UUID("project-uuid"),
    context_name="Code Review Session"
)

# Get performance metrics
metrics = await agent_service.get_agent_performance_metrics(agent.id)
overview = await agent_service.get_project_intelligence_overview(UUID("project-uuid"))

# Generate cost optimization recommendations
recommendations = await agent_service.generate_cost_optimization_recommendations(UUID("project-uuid"))

# Clean up
await agent_service.close()
"""