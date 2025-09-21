"""
Agent Execution Service - Bridge between Database Agents and AI Execution

This service connects the Intelligence-Tiered Agent Management System database records
to actual AI model execution through the existing PydanticAI agents and specialized agents.

Key Features:
- Routes tasks from database agents to appropriate execution engines
- Tracks token usage and costs in real-time
- Updates agent performance metrics
- Manages agent state transitions during execution
- Integrates with existing specialized agents
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum

from ...database.agent_service import AgentDatabaseService
from ...database.agent_models import (
    AgentV3, AgentState, ModelTier, AgentType,
    TaskComplexity, CostTracking
)
from ...agents.specialized.agent_factory import (
    SpecializedAgentFactory, 
    SpecializedAgentType,
    AgentExecutionContext,
    AgentExecutionResult
)
from .token_tracking_service import get_token_tracking_service
from .knowledge_embedding_service import get_knowledge_embedding_service, KnowledgeContext

logger = logging.getLogger(__name__)

# Map database agent types to specialized agent types
AGENT_TYPE_MAPPING = {
    AgentType.CODE_IMPLEMENTER: SpecializedAgentType.PYTHON_BACKEND_CODER,
    AgentType.SYSTEM_ARCHITECT: SpecializedAgentType.SYSTEM_ARCHITECT,
    AgentType.CODE_QUALITY_REVIEWER: SpecializedAgentType.CODE_REVIEWER,
    AgentType.TEST_COVERAGE_VALIDATOR: SpecializedAgentType.TEST_GENERATOR,
    AgentType.SECURITY_AUDITOR: SpecializedAgentType.SECURITY_AUDITOR,
    AgentType.PERFORMANCE_OPTIMIZER: SpecializedAgentType.PERFORMANCE_OPTIMIZER,
    AgentType.DEPLOYMENT_AUTOMATION: SpecializedAgentType.DEVOPS_ENGINEER,
    AgentType.DATABASE_ARCHITECT: SpecializedAgentType.DATABASE_DESIGNER,
    AgentType.DOCUMENTATION_GENERATOR: SpecializedAgentType.DOCUMENTATION_WRITER,
    AgentType.UI_UX_OPTIMIZER: SpecializedAgentType.UI_UX_DESIGNER,
    AgentType.API_DESIGN_ARCHITECT: SpecializedAgentType.API_INTEGRATOR,
    AgentType.CODE_REFACTORING_OPTIMIZER: SpecializedAgentType.REFACTORING_SPECIALIST,
    AgentType.STRATEGIC_PLANNER: SpecializedAgentType.STRATEGIC_PLANNER,
}

# Model tier to actual model mapping
MODEL_TIER_TO_MODEL = {
    ModelTier.OPUS: "claude-3-opus",
    ModelTier.SONNET: "claude-3-sonnet", 
    ModelTier.HAIKU: "claude-3-haiku"
}

# Token pricing per million tokens (input/output)
TOKEN_PRICING = {
    ModelTier.OPUS: {"input": 15.0, "output": 75.0},      # $15/$75 per million
    ModelTier.SONNET: {"input": 3.0, "output": 15.0},      # $3/$15 per million  
    ModelTier.HAIKU: {"input": 0.25, "output": 1.25}       # $0.25/$1.25 per million
}

class TaskExecutionRequest:
    """Request for task execution"""
    def __init__(
        self,
        agent_id: UUID,
        task_description: str,
        input_data: Dict[str, Any] = None,
        project_context: Dict[str, Any] = None,
        complexity_assessment: Optional[TaskComplexity] = None,
        require_approval: bool = False,
        timeout_minutes: int = 10
    ):
        self.task_id = uuid4()
        self.agent_id = agent_id
        self.task_description = task_description
        self.input_data = input_data or {}
        self.project_context = project_context or {}
        self.complexity_assessment = complexity_assessment
        self.require_approval = require_approval
        self.timeout_minutes = timeout_minutes
        self.created_at = datetime.now()

class TaskExecutionResponse:
    """Response from task execution"""
    def __init__(
        self,
        task_id: UUID,
        agent_id: UUID,
        status: str,
        output: Optional[str] = None,
        files_modified: List[str] = None,
        execution_time: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_cost: Decimal = Decimal("0.00"),
        error_message: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.task_id = task_id
        self.agent_id = agent_id
        self.status = status
        self.output = output
        self.files_modified = files_modified or []
        self.execution_time = execution_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_cost = total_cost
        self.error_message = error_message
        self.metadata = metadata or {}
        self.completed_at = datetime.now()

class AgentExecutionService:
    """Service for executing tasks with database-managed agents"""
    
    def __init__(self, agent_db_service: AgentDatabaseService):
        self.agent_db_service = agent_db_service
        self.specialized_factory = SpecializedAgentFactory()
        self.token_tracker = get_token_tracking_service()
        self.knowledge_service = get_knowledge_embedding_service(agent_db_service.supabase)
        self.active_executions: Dict[UUID, TaskExecutionRequest] = {}
        self.execution_history: List[TaskExecutionResponse] = []
        
        logger.info("Initialized Agent Execution Service with token tracking and knowledge embeddings")
    
    async def execute_task(self, request: TaskExecutionRequest) -> TaskExecutionResponse:
        """
        Execute a task with a database-managed agent
        
        This bridges the database agent to actual AI execution:
        1. Get agent from database
        2. Update agent state to ACTIVE
        3. Route to appropriate execution engine
        4. Track token usage and costs
        5. Update agent performance metrics
        6. Return execution results
        """
        start_time = time.time()
        
        try:
            # Get agent from database
            agent = await self.agent_db_service.get_agent_by_id(request.agent_id)
            if not agent:
                raise ValueError(f"Agent {request.agent_id} not found")
            
            # Check if agent is available
            if agent.state not in [AgentState.CREATED, AgentState.IDLE]:
                raise ValueError(f"Agent {agent.name} is not available (state: {agent.state})")
            
            # Update agent state to ACTIVE
            await self.agent_db_service.update_agent_state(
                agent.id, 
                AgentState.ACTIVE,
                f"Starting task: {request.task_description[:100]}"
            )
            
            # Track active execution
            self.active_executions[request.task_id] = request
            
            try:
                # Retrieve relevant knowledge for the task
                knowledge_context = KnowledgeContext(
                    agent_id=agent.id,
                    task_id=request.task_id,
                    query=request.task_description,
                    max_results=5,
                    similarity_threshold=0.6
                )
                
                relevant_knowledge = await self.knowledge_service.retrieve_relevant_knowledge(knowledge_context)
                
                # Add relevant knowledge to the request context
                if relevant_knowledge:
                    if not request.project_context:
                        request.project_context = {}
                    request.project_context["relevant_knowledge"] = [
                        {
                            "content": item.content,
                            "relevance": item.relevance_score,
                            "context": item.context
                        }
                        for item in relevant_knowledge
                    ]
                    logger.info(f"Added {len(relevant_knowledge)} relevant knowledge items to task context")
                
                # Route to appropriate execution engine based on agent type
                if agent.agent_type in AGENT_TYPE_MAPPING:
                    # Use specialized agent factory
                    result = await self._execute_with_specialized_agent(agent, request)
                else:
                    # Fallback to general purpose execution
                    result = await self._execute_with_general_agent(agent, request)
                
                # Track real token usage using token tracking service
                model_name = MODEL_TIER_TO_MODEL.get(agent.model_tier, "gpt-4o")
                
                # Track token usage
                token_usage = await self.token_tracker.track_usage(
                    task_id=request.task_id,
                    agent_id=agent.id,
                    model=model_name,
                    input_text=request.task_description,
                    output_text=result.output,
                    metadata={
                        "agent_type": agent.agent_type if isinstance(agent.agent_type, str) else agent.agent_type.value,
                        "project_id": str(agent.project_id) if agent.project_id else None
                    }
                )
                
                # Use real token counts and costs
                input_tokens = token_usage.input_tokens
                output_tokens = token_usage.output_tokens
                cost = token_usage.total_cost
                
                # Track cost in database
                await self.agent_db_service.track_agent_cost(
                    agent_id=agent.id,
                    project_id=agent.project_id,
                    task_id=request.task_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_tier=agent.model_tier,
                    task_duration_seconds=int(result.execution_time),
                    success=(result.status == "completed")
                )
                
                # Update agent metrics
                await self._update_agent_metrics(agent, result.status == "completed", result.execution_time)
                
                # Store successful execution as knowledge for future use
                if result.status == "completed" and result.output:
                    knowledge_content = f"Task: {request.task_description}\nSolution: {result.output}"
                    knowledge_context = {
                        "task_type": agent.agent_type if isinstance(agent.agent_type, str) else agent.agent_type.value,
                        "success": True,
                        "execution_time": result.execution_time
                    }
                    
                    await self.knowledge_service.store_knowledge(
                        agent_id=agent.id,
                        content=knowledge_content,
                        context=knowledge_context,
                        metadata={
                            "task_id": str(request.task_id),
                            "tokens_used": input_tokens + output_tokens,
                            "cost": float(cost)
                        }
                    )
                    logger.info(f"Stored task solution as knowledge for agent {agent.id}")
                
                # Create response
                response = TaskExecutionResponse(
                    task_id=request.task_id,
                    agent_id=agent.id,
                    status=result.status,
                    output=result.output,
                    files_modified=result.files_modified,
                    execution_time=time.time() - start_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_cost=cost,
                    metadata={
                        "agent_name": agent.name,
                        "agent_type": agent.agent_type if isinstance(agent.agent_type, str) else agent.agent_type.value,
                        "model_tier": agent.model_tier if isinstance(agent.model_tier, str) else agent.model_tier.value,
                        "specialized_execution": agent.agent_type in AGENT_TYPE_MAPPING
                    }
                )
                
                # Store in history
                self.execution_history.append(response)
                
                return response
                
            finally:
                # Update agent state back to IDLE
                await self.agent_db_service.update_agent_state(
                    agent.id,
                    AgentState.IDLE,
                    "Task completed"
                )
                
                # Remove from active executions
                del self.active_executions[request.task_id]
                
        except Exception as e:
            import traceback
            logger.error(f"Task execution failed for agent {request.agent_id}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try to update agent state to IDLE on error
            try:
                await self.agent_db_service.update_agent_state(
                    request.agent_id,
                    AgentState.IDLE,
                    f"Task failed: {str(e)[:100]}"
                )
            except:
                pass
            
            return TaskExecutionResponse(
                task_id=request.task_id,
                agent_id=request.agent_id,
                status="failed",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _execute_with_specialized_agent(
        self, 
        agent: AgentV3, 
        request: TaskExecutionRequest
    ) -> AgentExecutionResult:
        """Execute task using specialized agent factory"""
        
        # Map database agent type to specialized agent type
        specialized_type = AGENT_TYPE_MAPPING[agent.agent_type]
        
        # Add agent capabilities to context
        # Handle model_tier as either string or enum
        model_tier_value = agent.model_tier if isinstance(agent.model_tier, str) else agent.model_tier.value
        
        enriched_context = {
            **request.project_context,
            "agent_capabilities": agent.capabilities,
            "model_tier": model_tier_value,
            "agent_name": agent.name
        }
        
        # Execute with specialized agent
        result = await self.specialized_factory.execute_task_with_agent(
            agent_type=specialized_type,
            task_description=request.task_description,
            input_data=request.input_data,
            project_context=enriched_context
        )
        
        return result
    
    async def _execute_with_general_agent(
        self,
        agent: AgentV3,
        request: TaskExecutionRequest
    ) -> AgentExecutionResult:
        """Execute task using general purpose agent"""
        
        # For agents without specialized mapping, use general execution
        # This would integrate with the base PydanticAI agents
        
        from ...agents.base_agent import BaseAgent
        
        # Create a general purpose agent
        base_agent = BaseAgent(
            name=agent.name,
            model=MODEL_TIER_TO_MODEL.get(agent.model_tier, "openai:gpt-4")
        )
        
        # Execute task
        try:
            response = await base_agent.run(
                request.task_description,
                deps={
                    "request_id": str(request.task_id),
                    "agent_role": agent.agent_type if isinstance(agent.agent_type, str) else agent.agent_type.value,
                    "context": request.input_data
                }
            )
            
            return AgentExecutionResult(
                task_id=str(request.task_id),
                agent_type=SpecializedAgentType.PYTHON_BACKEND_CODER,  # Default
                status="completed",
                output=response,
                execution_time=0.0,
                metadata={"general_execution": True}
            )
            
        except Exception as e:
            return AgentExecutionResult(
                task_id=str(request.task_id),
                agent_type=SpecializedAgentType.PYTHON_BACKEND_CODER,
                status="failed",
                error_message=str(e),
                execution_time=0.0
            )
    
    def _calculate_token_usage(
        self,
        model_tier: ModelTier,
        input_text: str,
        output_text: Optional[str]
    ) -> Tuple[int, int, Decimal]:
        """Calculate token usage and cost"""
        
        # Simple estimation: ~4 characters per token
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4 if output_text else 0
        
        # Get pricing for tier
        pricing = TOKEN_PRICING[model_tier]
        
        # Calculate cost (price is per million tokens)
        input_cost = Decimal(str(input_tokens * pricing["input"] / 1_000_000))
        output_cost = Decimal(str(output_tokens * pricing["output"] / 1_000_000))
        total_cost = input_cost + output_cost
        
        return input_tokens, output_tokens, total_cost
    
    async def _update_agent_metrics(self, agent: AgentV3, success: bool, execution_time: float):
        """Update agent performance metrics"""
        
        # Calculate new metrics
        new_tasks_completed = agent.tasks_completed + 1
        new_success_count = int(agent.success_rate * agent.tasks_completed) + (1 if success else 0)
        new_success_rate = Decimal(str(new_success_count / new_tasks_completed))
        
        # Update average completion time
        total_time = agent.avg_completion_time_seconds * agent.tasks_completed + execution_time
        new_avg_time = int(total_time / new_tasks_completed)
        
        # Update in database (would need to add this method to agent_service)
        update_data = {
            "tasks_completed": new_tasks_completed,
            "success_rate": float(new_success_rate),
            "avg_completion_time_seconds": new_avg_time,
            "last_active_at": datetime.now().isoformat()
        }
        
        # For now, log the update
        logger.info(f"Updated metrics for agent {agent.name}: {update_data}")
    
    async def get_agent_workload(self, agent_id: UUID) -> Dict[str, Any]:
        """Get current workload for an agent"""
        
        active_tasks = [
            task for task_id, task in self.active_executions.items()
            if task.agent_id == agent_id
        ]
        
        return {
            "agent_id": str(agent_id),
            "active_tasks": len(active_tasks),
            "task_ids": [str(t.task_id) for t in active_tasks],
            "can_accept_tasks": len(active_tasks) == 0
        }
    
    async def get_execution_history(
        self,
        agent_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get execution history for an agent or all agents"""
        
        history = self.execution_history
        
        if agent_id:
            history = [h for h in history if h.agent_id == agent_id]
        
        # Sort by completion time (most recent first)
        history.sort(key=lambda x: x.completed_at, reverse=True)
        
        # Convert to dict and limit
        return [
            {
                "task_id": str(h.task_id),
                "agent_id": str(h.agent_id),
                "status": h.status,
                "execution_time": h.execution_time,
                "input_tokens": h.input_tokens,
                "output_tokens": h.output_tokens,
                "total_cost": str(h.total_cost),
                "completed_at": h.completed_at.isoformat(),
                "metadata": h.metadata
            }
            for h in history[:limit]
        ]
    
    async def execute_with_optimal_agent(
        self,
        project_id: UUID,
        agent_type: AgentType,
        task_description: str,
        input_data: Dict[str, Any] = None,
        complexity_assessment: Optional[TaskComplexity] = None
    ) -> TaskExecutionResponse:
        """Find and execute with the optimal agent for a task"""
        
        # If no complexity assessment, use default
        if not complexity_assessment:
            complexity_assessment = await self.agent_db_service.assess_task_complexity(
                task_id=uuid4(),
                technical=0.5,
                domain=0.5,
                code_volume=0.5,
                integration=0.5
            )
        
        # Find optimal agent
        optimal_agent = await self.agent_db_service.get_optimal_agent_for_task(
            project_id=project_id,
            task_complexity=complexity_assessment,
            agent_type=agent_type
        )
        
        if not optimal_agent:
            raise ValueError(f"No available agent of type {agent_type} for project {project_id}")
        
        # Create execution request
        request = TaskExecutionRequest(
            agent_id=optimal_agent.id,
            task_description=task_description,
            input_data=input_data,
            complexity_assessment=complexity_assessment
        )
        
        # Execute task
        return await self.execute_task(request)
    
    async def hibernate_idle_agents(self, project_id: UUID, idle_timeout_minutes: int = 30):
        """Hibernate agents that have been idle for too long"""
        
        hibernated_count = await self.agent_db_service.hibernate_idle_agents(
            project_id, 
            idle_timeout_minutes
        )
        
        logger.info(f"Hibernated {hibernated_count} idle agents in project {project_id}")
        return hibernated_count
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get metrics for the execution service"""
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h.status == "completed")
        
        total_cost = sum(h.total_cost for h in self.execution_history)
        total_tokens = sum(h.input_tokens + h.output_tokens for h in self.execution_history)
        
        return {
            "active_executions": len(self.active_executions),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
            "total_cost": str(total_cost),
            "total_tokens": total_tokens,
            "average_execution_time": sum(h.execution_time for h in self.execution_history) / total_executions if total_executions > 0 else 0.0,
            "factory_metrics": await self.specialized_factory.get_factory_metrics()
        }

# Global instance for easy access
_execution_service: Optional[AgentExecutionService] = None

async def get_execution_service(agent_db_service: AgentDatabaseService) -> AgentExecutionService:
    """Get or create the execution service instance"""
    global _execution_service
    
    if _execution_service is None:
        _execution_service = AgentExecutionService(agent_db_service)
    
    return _execution_service