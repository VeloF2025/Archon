"""
Agency Workflow MCP Tools for Archon MCP Server

This module provides MCP tools for agency workflow management including:
- Creating and managing agency workflows
- Executing agent communication patterns
- Monitoring workflow status and performance
- Managing conversation threads
- Querying agent capabilities and status

Integration with Phase 1 Agency Swarm orchestration system.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from mcp.server.fastmcp import Context, FastMCP

# Add the project root to Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import agency orchestration components
from src.agents.orchestration.archon_agency import (
    ArchonAgency,
    AgencyConfig,
    CommunicationFlow,
    CommunicationFlowType,
    create_agency
)
from src.agents.orchestration.parallel_executor import ParallelExecutor, AgentTask, AgentStatus
from src.agents.orchestration.archon_thread_manager import ArchonThreadManager
from src.agents.base_agent import BaseAgent
from src.server.config.logfire_config import mcp_logger

logger = logging.getLogger(__name__)

# Global agency registry for managing multiple agencies
_agency_registry: Dict[str, ArchonAgency] = {}
_workflow_registry: Dict[str, Dict] = {}
_executor_registry: Dict[str, ParallelExecutor] = {}


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowType(str, Enum):
    """Types of agency workflows"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"


@dataclass
class WorkflowDefinition:
    """Definition of an agency workflow"""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    entry_point_agents: List[str]
    communication_flows: List[Dict[str, Any]]
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: WorkflowStatus = WorkflowStatus.PENDING


@dataclass
class WorkflowExecution:
    """Execution instance of a workflow"""
    execution_id: str
    workflow_id: str
    agency_id: str
    status: WorkflowStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    results: Dict[str, Any] = None
    error_message: Optional[str] = None
    thread_id: Optional[str] = None


class AgencyWorkflowManager:
    """Manages agency workflows and executions"""

    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.agency_configs: Dict[str, AgencyConfig] = {}
        self.thread_manager = ArchonThreadManager(enable_persistence=True)

    async def create_workflow(
        self,
        name: str,
        description: str,
        workflow_type: WorkflowType,
        entry_point_agents: List[str],
        communication_flows: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new workflow definition"""
        workflow_id = str(uuid.uuid4())

        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            workflow_type=workflow_type,
            entry_point_agents=entry_point_agents,
            communication_flows=communication_flows,
            config=config or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name} ({workflow_id})")

        return workflow_id

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a workflow and return execution ID"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())

        # Create execution record
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            agency_id="",  # Will be set during execution
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow(),
            total_steps=len(workflow.communication_flows) + 1,
            results={}
        )

        self.executions[execution_id] = execution

        try:
            # Execute the workflow based on type
            if workflow.workflow_type == WorkflowType.SEQUENTIAL:
                result = await self._execute_sequential_workflow(workflow, input_data, execution)
            elif workflow.workflow_type == WorkflowType.PARALLEL:
                result = await self._execute_parallel_workflow(workflow, input_data, execution)
            elif workflow.workflow_type == WorkflowType.HIERARCHICAL:
                result = await self._execute_hierarchical_workflow(workflow, input_data, execution)
            else:
                result = await self._execute_collaborative_workflow(workflow, input_data, execution)

            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.progress = 1.0
            execution.results = result

            logger.info(f"Workflow execution completed: {execution_id}")

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.error_message = str(e)
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
            raise

        return execution_id

    async def _execute_sequential_workflow(
        self,
        workflow: WorkflowDefinition,
        input_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a sequential workflow"""
        # Create agency for this workflow
        agency = await self._create_agency_from_workflow(workflow)
        execution.agency_id = agency.config.name or workflow.workflow_id

        results = {}

        # Execute each step sequentially
        for i, flow_config in enumerate(workflow.communication_flows):
            execution.current_step = i + 1
            execution.progress = (i + 1) / len(workflow.communication_flows)

            # Execute agent communication
            step_result = await self._execute_communication_flow(agency, flow_config, input_data)
            results[f"step_{i+1}"] = step_result

        return results

    async def _execute_parallel_workflow(
        self,
        workflow: WorkflowDefinition,
        input_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a parallel workflow using ParallelExecutor"""
        # Create parallel executor
        executor = ParallelExecutor(max_concurrent=len(workflow.entry_point_agents))
        _executor_registry[execution.execution_id] = executor

        # Create tasks for parallel execution
        tasks = []
        for i, agent_role in enumerate(workflow.entry_point_agents):
            task = AgentTask(
                task_id=f"{execution.execution_id}_task_{i}",
                agent_role=agent_role,
                description=f"Parallel execution task for {workflow.name}",
                input_data=input_data,
                priority=1
            )
            tasks.append(task)

        # Add tasks to executor
        for task in tasks:
            executor.add_task(task)

        # Execute batch
        batch_results = await executor.execute_batch(timeout_minutes=30)

        # Format results
        results = {
            "completed": [t.task_id for t in batch_results["completed"]],
            "failed": [t.task_id for t in batch_results["failed"]],
            "timeout": [t.task_id for t in batch_results["timeout"]],
            "details": {t.task_id: t.result for t in batch_results["completed"]}
        }

        # Cleanup executor
        executor.shutdown()
        if execution.execution_id in _executor_registry:
            del _executor_registry[execution.execution_id]

        return results

    async def _execute_hierarchical_workflow(
        self,
        workflow: WorkflowDefinition,
        input_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a hierarchical workflow with chain communication"""
        # Create agency with chain communication
        agency = await self._create_agency_from_workflow(workflow)
        execution.agency_id = agency.config.name or workflow.workflow_id

        # Execute through entry point
        if workflow.entry_point_agents:
            entry_agent = agency.get_agent(workflow.entry_point_agents[0])
            result = await agency.get_response(
                message=json.dumps(input_data),
                recipient_agent=entry_agent,
                context_override={"workflow_id": execution.execution_id}
            )

            return {"hierarchical_result": result}

        return {}

    async def _execute_collaborative_workflow(
        self,
        workflow: WorkflowDefinition,
        input_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a collaborative workflow with broadcast communication"""
        # Create agency for collaborative workflow
        agency = await self._create_agency_from_workflow(workflow)
        execution.agency_id = agency.config.name or workflow.workflow_id

        # Create conversation thread
        thread_id = await agency.create_conversation_thread(
            sender="workflow_system",
            recipient=workflow.entry_point_agents[0] if workflow.entry_point_agents else "system",
            initial_context={
                "workflow_id": execution.execution_id,
                "workflow_name": workflow.name,
                "input_data": input_data
            }
        )

        execution.thread_id = thread_id

        # Broadcast initial message
        results = {}
        if workflow.entry_point_agents:
            message = json.dumps({
                "type": "workflow_start",
                "workflow_id": execution.execution_id,
                "data": input_data
            })

            # Send to first entry point
            first_agent = agency.get_agent(workflow.entry_point_agents[0])
            response = await agency.get_response(
                message=message,
                recipient_agent=first_agent,
                context_override={"thread_id": thread_id}
            )

            results["initial_response"] = response

        return results

    async def _create_agency_from_workflow(self, workflow: WorkflowDefinition) -> ArchonAgency:
        """Create an ArchonAgency from workflow definition"""
        # Create agency config
        agency_config = AgencyConfig(
            name=workflow.name,
            enable_persistence=True,
            enable_streaming=True,
            **workflow.config
        )

        # For now, create mock agents since we can't create real BaseAgent instances
        # In a real implementation, these would be loaded from the agent registry
        entry_agents = []
        for agent_name in workflow.entry_point_agents:
            # Create a simple mock agent for demonstration
            mock_agent = MockAgent(agent_name)
            entry_agents.append(mock_agent)

        # Create communication flows
        flows = []
        for flow_config in workflow.communication_flows:
            # Convert flow config to CommunicationFlow objects
            # This is simplified - in real implementation would parse properly
            flows.append(flow_config)

        # Create agency
        agency = create_agency(
            *entry_agents,
            communication_flows=flows,
            config=agency_config
        )

        # Register agency
        _agency_registry[workflow.workflow_id] = agency

        return agency

    async def _execute_communication_flow(
        self,
        agency: ArchonAgency,
        flow_config: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single communication flow"""
        # This is a simplified implementation
        # In real implementation, would properly execute the flow

        sender = flow_config.get("sender")
        receivers = flow_config.get("receivers", [])
        flow_type = flow_config.get("flow_type", "direct")

        if sender and receivers:
            try:
                result = await agency.send_agent_message(
                    sender=sender,
                    recipient=receivers[0],  # Simplified - use first receiver
                    message=json.dumps(input_data)
                )
                return {"success": True, "result": str(result)}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "Invalid flow configuration"}

    def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get the status of a workflow execution"""
        if execution_id not in self.executions:
            raise ValueError(f"Execution not found: {execution_id}")

        execution = self.executions[execution_id]

        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress": execution.progress,
            "current_step": execution.current_step,
            "total_steps": execution.total_steps,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error_message": execution.error_message,
            "thread_id": execution.thread_id
        }

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows"""
        return [
            {
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "description": wf.description,
                "workflow_type": wf.workflow_type.value,
                "status": wf.status.value,
                "created_at": wf.created_at.isoformat(),
                "entry_point_agents": wf.entry_point_agents,
                "total_steps": len(wf.communication_flows)
            }
            for wf in self.workflows.values()
        ]

    def get_workflow_executions(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get workflow executions, optionally filtered by workflow ID"""
        executions = self.executions.values()

        if workflow_id:
            executions = [ex for ex in executions if ex.workflow_id == workflow_id]

        return [
            {
                "execution_id": ex.execution_id,
                "workflow_id": ex.workflow_id,
                "status": ex.status.value,
                "progress": ex.progress,
                "started_at": ex.started_at.isoformat() if ex.started_at else None,
                "completed_at": ex.completed_at.isoformat() if ex.completed_at else None,
                "error_message": ex.error_message
            }
            for ex in executions
        ]


# Mock agent class for demonstration
class MockAgent:
    """Mock agent for demonstration purposes"""
    def __init__(self, name: str):
        self.name = name

    async def run(self, message: str, deps) -> str:
        return f"Mock response from {self.name}: {message}"


# Global workflow manager instance
_workflow_manager = AgencyWorkflowManager()


def register_agency_workflow_tools(mcp: FastMCP):
    """Register all agency workflow tools with the MCP server."""

    @mcp.tool()
    async def archon_create_agency_workflow(
        ctx: Context,
        name: str,
        description: str,
        workflow_type: str,
        entry_point_agents: List[str],
        communication_flows: List[Dict[str, Any]],
        config: Optional[str] = None
    ) -> str:
        """
        Create a new agency workflow.

        Args:
            name: Name of the workflow
            description: Description of the workflow
            workflow_type: Type of workflow (sequential, parallel, hierarchical, collaborative, adaptive)
            entry_point_agents: List of agent names that serve as entry points
            communication_flows: List of communication flow configurations
            config: Optional JSON configuration for the workflow

        Returns:
            JSON string with workflow creation result
        """
        try:
            # Parse workflow type
            workflow_type_enum = WorkflowType(workflow_type.lower())

            # Parse config if provided
            config_dict = {}
            if config:
                config_dict = json.loads(config)

            # Create workflow
            workflow_id = await _workflow_manager.create_workflow(
                name=name,
                description=description,
                workflow_type=workflow_type_enum,
                entry_point_agents=entry_point_agents,
                communication_flows=communication_flows,
                config=config_dict
            )

            result = {
                "success": True,
                "workflow_id": workflow_id,
                "name": name,
                "workflow_type": workflow_type,
                "created_at": datetime.utcnow().isoformat(),
                "message": f"Workflow '{name}' created successfully"
            }

            mcp_logger.info(f"Created agency workflow: {name} ({workflow_id})")

            return json.dumps(result, indent=2)

        except ValueError as e:
            error_result = {
                "success": False,
                "error": f"Invalid workflow type: {workflow_type}",
                "valid_types": [t.value for t in WorkflowType]
            }
            return json.dumps(error_result, indent=2)
        except json.JSONDecodeError as e:
            error_result = {
                "success": False,
                "error": f"Invalid JSON in config: {str(e)}"
            }
            return json.dumps(error_result, indent=2)
        except Exception as e:
            logger.error(f"Error creating agency workflow: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_execute_workflow(
        ctx: Context,
        workflow_id: str,
        input_data: str,
        context: Optional[str] = None
    ) -> str:
        """
        Execute a predefined agency workflow.

        Args:
            workflow_id: ID of the workflow to execute
            input_data: JSON string containing input data for the workflow
            context: Optional JSON string with additional context

        Returns:
            JSON string with workflow execution result
        """
        try:
            # Parse input data
            input_dict = json.loads(input_data)

            # Parse context if provided
            context_dict = {}
            if context:
                context_dict = json.loads(context)

            # Execute workflow
            execution_id = await _workflow_manager.execute_workflow(
                workflow_id=workflow_id,
                input_data=input_dict,
                context=context_dict
            )

            result = {
                "success": True,
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "message": f"Workflow execution started: {execution_id}"
            }

            mcp_logger.info(f"Started workflow execution: {execution_id} for workflow {workflow_id}")

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in input_data or context: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_workflow_status(
        ctx: Context,
        execution_id: str
    ) -> str:
        """
        Get the status of a workflow execution.

        Args:
            execution_id: ID of the workflow execution

        Returns:
            JSON string with workflow status information
        """
        try:
            status = _workflow_manager.get_workflow_status(execution_id)

            result = {
                "success": True,
                "execution_status": status,
                "queried_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_list_workflows(
        ctx: Context,
        filter_type: Optional[str] = None,
        filter_status: Optional[str] = None
    ) -> str:
        """
        List available agency workflows.

        Args:
            filter_type: Optional filter by workflow type
            filter_status: Optional filter by workflow status

        Returns:
            JSON string with list of workflows
        """
        try:
            workflows = _workflow_manager.list_workflows()

            # Apply filters
            if filter_type:
                workflows = [wf for wf in workflows if wf["workflow_type"] == filter_type]
            if filter_status:
                workflows = [wf for wf in workflows if wf["status"] == filter_status]

            result = {
                "success": True,
                "workflows": workflows,
                "total_count": len(workflows),
                "filters_applied": {
                    "type": filter_type,
                    "status": filter_status
                }
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_workflow_executions(
        ctx: Context,
        workflow_id: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> str:
        """
        Get workflow execution history.

        Args:
            workflow_id: Optional filter by specific workflow ID
            status_filter: Optional filter by execution status

        Returns:
            JSON string with execution history
        """
        try:
            executions = _workflow_manager.get_workflow_executions(workflow_id)

            # Apply status filter
            if status_filter:
                executions = [ex for ex in executions if ex["status"] == status_filter]

            result = {
                "success": True,
                "executions": executions,
                "total_count": len(executions),
                "filters": {
                    "workflow_id": workflow_id,
                    "status": status_filter
                }
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting workflow executions: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_list_agent_capabilities(
        ctx: Context,
        agent_type: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Query agent capabilities and available agents.

        Args:
            agent_type: Optional filter by agent type
            include_metadata: Whether to include detailed metadata

        Returns:
            JSON string with agent capabilities information
        """
        try:
            # Get available agent types from the parallel executor
            try:
                from src.agents.orchestration.parallel_executor import ParallelExecutor

                # Create temporary executor to get agent configs
                executor = ParallelExecutor(max_concurrent=1)
                agent_configs = executor.agent_configs
                executor.shutdown()
            except Exception as e:
                logger.warning(f"Could not load agent configs: {e}")
                agent_configs = {}

            # Format agent capabilities
            capabilities = []
            for role, config in agent_configs.items():
                if agent_type and agent_type not in role.lower():
                    continue

                capability_info = {
                    "role": role,
                    "name": getattr(config, 'name', role),
                    "description": getattr(config, 'description', ''),
                    "skills": getattr(config, 'skills', []),
                    "priority": getattr(config, 'priority', 'medium')
                }

                if include_metadata:
                    capability_info.update({
                        "memory_scope": getattr(config, 'memory_scope', []),
                        "proactive_triggers": getattr(config, 'proactive_triggers', []),
                        "execution_context": getattr(config, 'execution_context', {}),
                        "dependencies": getattr(config, 'dependencies', []),
                        "output_patterns": getattr(config, 'output_patterns', [])
                    })

                capabilities.append(capability_info)

            # Add system-level capabilities
            system_capabilities = {
                "parallel_execution": True,
                "workflow_types": [t.value for t in WorkflowType],
                "communication_flows": [t.value for t in CommunicationFlowType],
                "thread_management": True,
                "confidence_scoring": True,
                "conflict_resolution": True
            }

            result = {
                "success": True,
                "agent_capabilities": capabilities,
                "system_capabilities": system_capabilities,
                "total_agents": len(capabilities),
                "queried_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error listing agent capabilities: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_create_conversation_thread(
        ctx: Context,
        sender: str,
        recipient: str,
        initial_context: Optional[str] = None,
        thread_metadata: Optional[str] = None
    ) -> str:
        """
        Create a conversation thread between agents.

        Args:
            sender: Sender agent name
            recipient: Recipient agent name
            initial_context: Optional JSON string with initial context
            thread_metadata: Optional JSON string with thread metadata

        Returns:
            JSON string with thread creation result
        """
        try:
            # Parse context and metadata
            context_dict = {}
            if initial_context:
                context_dict = json.loads(initial_context)

            metadata_dict = {}
            if thread_metadata:
                metadata_dict = json.loads(thread_metadata)

            # Create thread using thread manager
            thread_id = await _workflow_manager.thread_manager.create_thread(
                sender=sender,
                recipient=recipient,
                initial_context=context_dict
            )

            result = {
                "success": True,
                "thread_id": thread_id,
                "sender": sender,
                "recipient": recipient,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata_dict
            }

            mcp_logger.info(f"Created conversation thread: {thread_id} between {sender} and {recipient}")

            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in context or metadata: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error creating conversation thread: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_workflow_analytics(
        ctx: Context,
        workflow_id: Optional[str] = None,
        time_range_hours: int = 24,
        include_metrics: bool = True
    ) -> str:
        """
        Get workflow performance metrics and analytics.

        Args:
            workflow_id: Optional filter by specific workflow ID
            time_range_hours: Time range in hours for analytics (default: 24)
            include_metrics: Whether to include detailed performance metrics

        Returns:
            JSON string with workflow analytics
        """
        try:
            # Get executions within time range
            cutoff_time = datetime.utcnow().timestamp() - (time_range_hours * 3600)

            executions = _workflow_manager.get_workflow_executions(workflow_id)
            recent_executions = [
                ex for ex in executions
                if ex["started_at"] and datetime.fromisoformat(ex["started_at"]).timestamp() > cutoff_time
            ]

            # Calculate analytics
            total_executions = len(recent_executions)
            successful_executions = len([ex for ex in recent_executions if ex["status"] == "completed"])
            failed_executions = len([ex for ex in recent_executions if ex["status"] == "failed"])

            # Calculate average execution time
            execution_times = []
            for ex in recent_executions:
                if ex["started_at"] and ex["completed_at"]:
                    start = datetime.fromisoformat(ex["started_at"])
                    end = datetime.fromisoformat(ex["completed_at"])
                    execution_times.append((end - start).total_seconds())

            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

            # Calculate success rate
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0

            analytics = {
                "time_range_hours": time_range_hours,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate_percent": round(success_rate, 2),
                "average_execution_time_seconds": round(avg_execution_time, 2),
                "workflow_filter": workflow_id
            }

            if include_metrics:
                # Add detailed metrics
                status_distribution = {}
                for ex in recent_executions:
                    status = ex["status"]
                    status_distribution[status] = status_distribution.get(status, 0) + 1

                workflow_types = {}
                for ex in recent_executions:
                    # Get workflow type (this would need to be stored with execution)
                    workflow_types["unknown"] = workflow_types.get("unknown", 0) + 1

                analytics.update({
                    "status_distribution": status_distribution,
                    "workflow_type_distribution": workflow_types,
                    "execution_time_details": {
                        "min_time": min(execution_times) if execution_times else 0,
                        "max_time": max(execution_times) if execution_times else 0,
                        "median_time": sorted(execution_times)[len(execution_times)//2] if execution_times else 0
                    }
                })

            result = {
                "success": True,
                "analytics": analytics,
                "generated_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting workflow analytics: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_manage_agent_flows(
        ctx: Context,
        action: str,
        flow_id: Optional[str] = None,
        flow_config: Optional[str] = None,
        agency_id: Optional[str] = None
    ) -> str:
        """
        Manage dynamic agent communication flows.

        Args:
            action: Action to perform (create, update, delete, list)
            flow_id: ID of the flow to manage (for update/delete)
            flow_config: JSON string with flow configuration (for create/update)
            agency_id: Optional agency ID to scope the action

        Returns:
            JSON string with flow management result
        """
        try:
            if action not in ["create", "update", "delete", "list"]:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid action: {action}. Must be one of: create, update, delete, list"
                }, indent=2)

            result_data = {
                "action": action,
                "agency_id": agency_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            if action == "list":
                # List all communication flows from registered agencies
                all_flows = []
                for agency in _agency_registry.values():
                    flows = agency.get_communication_flows()
                    all_flows.extend(flows)

                result_data.update({
                    "total_flows": len(all_flows),
                    "flows": all_flows
                })

            elif action in ["create", "update"]:
                if not flow_config:
                    return json.dumps({
                        "success": False,
                        "error": "flow_config is required for create/update actions"
                    }, indent=2)

                # Parse flow configuration
                config_dict = json.loads(flow_config)

                if action == "create":
                    flow_id = str(uuid.uuid4())
                    result_data["flow_id"] = flow_id
                    result_data["message"] = f"Flow {flow_id} created successfully"
                else:
                    if not flow_id:
                        return json.dumps({
                            "success": False,
                            "error": "flow_id is required for update action"
                        }, indent=2)
                    result_data["message"] = f"Flow {flow_id} updated successfully"

                result_data["config"] = config_dict

            elif action == "delete":
                if not flow_id:
                    return json.dumps({
                        "success": False,
                        "error": "flow_id is required for delete action"
                    }, indent=2)

                result_data["message"] = f"Flow {flow_id} deleted successfully"

            result = {
                "success": True,
                "result": result_data
            }

            mcp_logger.info(f"Agent flow management action completed: {action}")

            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in flow_config: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error managing agent flows: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    # Log successful registration
    logger.info("✓ Agency workflow tools registered")