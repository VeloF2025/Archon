"""
MCP Tools for Agency Workflow Management

Provides MCP tools for workflow operations that integrate with:
- ReactFlow workflow visualization components
- Agency Swarm agent execution
- Real-time workflow monitoring
- Analytics and optimization

Tools follow MCP standards and integrate with existing Archon MCP architecture
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from mcp.server.fastmcp import Context

from ...database import get_db
from ...database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, ExecutionStatus,
    WorkflowCreateRequest, WorkflowUpdateRequest, WorkflowExecutionRequest,
    ReactFlowData, validate_reactflow_data, WorkflowStatus, TriggerType,
    AgentType, ModelTier
)
from ...server.services.workflow_service import get_workflow_service
from ...server.services.workflow_analytics_service import get_workflow_analytics_service

logger = logging.getLogger(__name__)

def register_workflow_tools(mcp):
    """Register all workflow management MCP tools"""

    @mcp.tool()
    async def archon_create_workflow(
        ctx: Context,
        name: str,
        project_id: str,
        flow_data: Dict[str, Any],
        description: Optional[str] = None,
        trigger_type: Optional[str] = "MANUAL",
        default_agent_type: Optional[str] = None,
        default_model_tier: Optional[str] = "SONNET",
        timeout_seconds: Optional[int] = 3600,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = "MCP User"
    ) -> str:
        """
        Create a new agency workflow with ReactFlow compatibility

        Creates a workflow definition that can be visualized and edited using ReactFlow components.
        The workflow supports agent execution, parallel steps, and various trigger types.

        Args:
            name: Workflow name (max 255 chars)
            project_id: Project UUID for the workflow
            flow_data: ReactFlow compatible nodes and edges data
            description: Optional workflow description
            trigger_type: How workflow is triggered (MANUAL, SCHEDULED, EVENT, WEBHOOK, API)
            default_agent_type: Default agent type for steps
            default_model_tier: Default model tier (OPUS, SONNET, HAIKU)
            timeout_seconds: Workflow execution timeout in seconds
            tags: List of workflow tags
            created_by: User creating the workflow

        Returns:
            JSON with workflow ID and creation status
        """
        try:
            # Validate ReactFlow data
            is_valid, errors = validate_reactflow_data(flow_data)
            if not is_valid:
                return json.dumps({
                    "success": False,
                    "error": "Invalid ReactFlow data",
                    "validation_errors": errors
                })

            # Parse enums
            try:
                workflow_trigger_type = TriggerType(trigger_type.upper())
                if default_agent_type:
                    default_agent_type = AgentType(default_agent_type.upper())
                default_model_tier = ModelTier(default_model_tier.upper())
            except ValueError as e:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid enum value: {str(e)}"
                })

            # Create request
            request = WorkflowCreateRequest(
                name=name,
                description=description,
                flow_data=flow_data,
                trigger_type=workflow_trigger_type,
                default_agent_type=default_agent_type,
                default_model_tier=default_model_tier,
                timeout_seconds=timeout_seconds,
                tags=tags or []
            )

            # Create workflow
            db = next(get_db())
            workflow_service = await get_workflow_service()
            success, result = await workflow_service.create_workflow(
                request, created_by, project_id, db
            )

            if success:
                workflow = result
                return json.dumps({
                    "success": True,
                    "workflow_id": str(workflow.id),
                    "name": workflow.name,
                    "status": workflow.status.value,
                    "created_at": workflow.created_at.isoformat(),
                    "message": "Workflow created successfully"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result
                })

        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to create workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_update_workflow(
        ctx: Context,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        flow_data: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = None,
        updated_by: Optional[str] = "MCP User"
    ) -> str:
        """
        Update an existing workflow definition

        Updates workflow configuration including ReactFlow data, status, and metadata.
        Changes are version tracked for rollback capabilities.

        Args:
            workflow_id: UUID of the workflow to update
            name: New workflow name
            description: New workflow description
            flow_data: Updated ReactFlow nodes and edges
            status: New workflow status (DRAFT, PUBLISHED, etc.)
            tags: Updated workflow tags
            timeout_seconds: New execution timeout
            updated_by: User making the update

        Returns:
            JSON with update status and changes made
        """
        try:
            # Parse status enum
            workflow_status = None
            if status:
                try:
                    workflow_status = WorkflowStatus(status.upper())
                except ValueError:
                    return json.dumps({
                        "success": False,
                        "error": f"Invalid status: {status}"
                    })

            # Create request
            request = WorkflowUpdateRequest(
                name=name,
                description=description,
                flow_data=flow_data,
                status=workflow_status,
                tags=tags,
                timeout_seconds=timeout_seconds
            )

            # Update workflow
            db = next(get_db())
            workflow_service = await get_workflow_service()
            success, result = await workflow_service.update_workflow(
                workflow_id, request, updated_by, db
            )

            if success:
                workflow = result
                return json.dumps({
                    "success": True,
                    "workflow_id": str(workflow.id),
                    "name": workflow.name,
                    "status": workflow.status.value,
                    "updated_at": workflow.updated_at.isoformat(),
                    "message": "Workflow updated successfully"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result
                })

        except Exception as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to update workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_delete_workflow(
        ctx: Context,
        workflow_id: str
    ) -> str:
        """
        Delete a workflow definition

        Permanently deletes a workflow and all related data including
        execution history and versions. Cannot be undone.

        Args:
            workflow_id: UUID of the workflow to delete

        Returns:
            JSON with deletion status
        """
        try:
            db = next(get_db())
            workflow_service = await get_workflow_service()
            success, result = await workflow_service.delete_workflow(workflow_id, db)

            if success:
                return json.dumps({
                    "success": True,
                    "workflow_id": workflow_id,
                    "message": "Workflow deleted successfully"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result
                })

        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to delete workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_execute_workflow(
        ctx: Context,
        workflow_id: str,
        trigger_data: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        async_execution: Optional[bool] = True,
        priority: Optional[int] = 0,
        triggered_by: Optional[str] = "MCP User"
    ) -> str:
        """
        Execute a workflow

        Starts workflow execution with the specified parameters.
        Supports both synchronous and asynchronous execution modes.

        Args:
            workflow_id: UUID of the workflow to execute
            trigger_data: Data that triggered the workflow
            parameters: Execution parameters
            async_execution: Whether to execute asynchronously (default: True)
            priority: Execution priority (-10 to 10)
            triggered_by: User/agent triggering the execution

        Returns:
            JSON with execution ID and status
        """
        try:
            # Create request
            request = WorkflowExecutionRequest(
                workflow_id=uuid.UUID(workflow_id),
                trigger_data=trigger_data or {},
                parameters=parameters or {},
                async_execution=async_execution,
                priority=priority
            )

            # Execute workflow
            db = next(get_db())
            workflow_service = await get_workflow_service()
            success, result = await workflow_service.execute_workflow(
                request, triggered_by, db
            )

            if success:
                execution = result
                return json.dumps({
                    "success": True,
                    "execution_id": execution.execution_id,
                    "workflow_id": str(execution.workflow_id),
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "estimated_duration": execution.estimated_duration,
                    "message": "Workflow execution started" if async_execution else "Workflow executed"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result
                })

        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to execute workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_list_workflows(
        ctx: Context,
        project_id: str,
        status: Optional[str] = None,
        trigger_type: Optional[str] = None,
        skip: Optional[int] = 0,
        limit: Optional[int] = 100
    ) -> str:
        """
        List workflows in a project

        Retrieves workflow definitions with optional filtering by status and trigger type.
        Supports pagination for large lists.

        Args:
            project_id: Project UUID to filter workflows
            status: Filter by workflow status
            trigger_type: Filter by trigger type
            skip: Number of workflows to skip (pagination)
            limit: Maximum number of workflows to return

        Returns:
            JSON with workflow list and pagination info
        """
        try:
            # Parse filter enums
            workflow_status = None
            if status:
                try:
                    workflow_status = WorkflowStatus(status.upper())
                except ValueError:
                    pass

            workflow_trigger_type = None
            if trigger_type:
                try:
                    workflow_trigger_type = TriggerType(trigger_type.upper())
                except ValueError:
                    pass

            # List workflows
            db = next(get_db())
            workflow_service = await get_workflow_service()
            workflows, total = await workflow_service.list_workflows(
                project_id, workflow_status, workflow_trigger_type, skip, limit, db
            )

            return json.dumps({
                "success": True,
                "total": total,
                "skip": skip,
                "limit": limit,
                "workflows": [
                    {
                        "workflow_id": str(w.id),
                        "name": w.name,
                        "description": w.description,
                        "status": w.status.value,
                        "trigger_type": w.trigger_type.value,
                        "version": w.version,
                        "tags": w.tags,
                        "created_at": w.created_at.isoformat(),
                        "updated_at": w.updated_at.isoformat()
                    }
                    for w in workflows
                ]
            })

        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to list workflows: {str(e)}"
            })

    @mcp.tool()
    async def archon_get_workflow_status(
        ctx: Context,
        execution_id: str
    ) -> str:
        """
        Get workflow execution status

        Retrieves detailed status information for a workflow execution including
        progress, current step, results, and errors.

        Args:
            execution_id: Execution ID to check status for

        Returns:
            JSON with detailed execution status
        """
        try:
            db = next(get_db())
            workflow_service = await get_workflow_service()
            status = await workflow_service.get_execution_status(execution_id, db)

            if status:
                return json.dumps({
                    "success": True,
                    "execution": {
                        "execution_id": status.execution_id,
                        "workflow_id": str(status.workflow_id),
                        "status": status.status.value,
                        "progress": status.progress,
                        "current_step_id": status.current_step_id,
                        "started_at": status.started_at.isoformat() if status.started_at else None,
                        "completed_at": status.completed_at.isoformat() if status.completed_at else None,
                        "results": status.results,
                        "errors": status.errors,
                        "metrics": status.metrics
                    }
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": "Execution not found"
                })

        except Exception as e:
            logger.error(f"Failed to get workflow status for {execution_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to get workflow status: {str(e)}"
            })

    @mcp.tool()
    async def archon_pause_workflow(
        ctx: Context,
        execution_id: str
    ) -> str:
        """
        Pause a running workflow execution

        Temporarily pauses a workflow execution that can be resumed later.
        Useful for debugging or resource management.

        Args:
            execution_id: Execution ID to pause

        Returns:
            JSON with pause status
        """
        try:
            db = next(get_db())
            workflow_service = await get_workflow_service()
            success, result = await workflow_service.pause_execution(execution_id, db)

            if success:
                return json.dumps({
                    "success": True,
                    "execution_id": execution_id,
                    "message": "Workflow execution paused"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result
                })

        except Exception as e:
            logger.error(f"Failed to pause workflow execution {execution_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to pause workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_resume_workflow(
        ctx: Context,
        execution_id: str
    ) -> str:
        """
        Resume a paused workflow execution

        Resumes a previously paused workflow execution from where it left off.

        Args:
            execution_id: Execution ID to resume

        Returns:
            JSON with resume status
        """
        try:
            db = next(get_db())
            workflow_service = await get_workflow_service()
            success, result = await workflow_service.resume_execution(execution_id, db)

            if success:
                return json.dumps({
                    "success": True,
                    "execution_id": execution_id,
                    "message": "Workflow execution resumed"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result
                })

        except Exception as e:
            logger.error(f"Failed to resume workflow execution {execution_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to resume workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_cancel_workflow(
        ctx: Context,
        execution_id: str
    ) -> str:
        """
        Cancel a workflow execution

        Permanently stops a workflow execution and marks it as cancelled.
        Cannot be resumed.

        Args:
            execution_id: Execution ID to cancel

        Returns:
            JSON with cancellation status
        """
        try:
            db = next(get_db())
            workflow_service = await get_workflow_service()
            success, result = await workflow_service.cancel_execution(execution_id, db)

            if success:
                return json.dumps({
                    "success": True,
                    "execution_id": execution_id,
                    "message": "Workflow execution cancelled"
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result
                })

        except Exception as e:
            logger.error(f"Failed to cancel workflow execution {execution_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to cancel workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_visualize_workflow(
        ctx: Context,
        workflow_id: str,
        include_execution_data: Optional[bool] = False
    ) -> str:
        """
        Generate workflow visualization data for ReactFlow

        Returns ReactFlow-compatible data structure for workflow visualization.
        Can include current execution state for running workflows.

        Args:
            workflow_id: Workflow ID to visualize
            include_execution_data: Whether to include current execution data

        Returns:
            JSON with ReactFlow visualization data
        """
        try:
            db = next(get_db())
            workflow_service = await get_workflow_service()
            workflow = await workflow_service.get_workflow(workflow_id, db)

            if not workflow:
                return json.dumps({
                    "success": False,
                    "error": "Workflow not found"
                })

            # Get ReactFlow data
            reactflow_data = workflow.to_reactflow_format()

            # Add execution data if requested
            if include_execution_data:
                # Get most recent execution
                recent_execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.workflow_id == workflow.id
                ).order_by(WorkflowExecution.created_at.desc()).first()

                if recent_execution:
                    reactflow_data["execution_data"] = {
                        "execution_id": recent_execution.execution_id,
                        "status": recent_execution.status.value,
                        "progress": recent_execution.progress,
                        "current_step_id": recent_execution.current_step_id,
                        "started_at": recent_execution.started_at.isoformat() if recent_execution.started_at else None
                    }

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "reactflow_data": reactflow_data,
                "metadata": {
                    "name": workflow.name,
                    "description": workflow.description,
                    "status": workflow.status.value,
                    "version": workflow.version,
                    "created_at": workflow.created_at.isoformat(),
                    "updated_at": workflow.updated_at.isoformat()
                }
            })

        except Exception as e:
            logger.error(f"Failed to visualize workflow {workflow_id}: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to visualize workflow: {str(e)}"
            })

    @mcp.tool()
    async def archon_workflow_analytics(
        ctx: Context,
        workflow_id: Optional[str] = None,
        project_id: Optional[str] = None,
        period_days: Optional[int] = 30,
        include_recommendations: Optional[bool] = True
    ) -> str:
        """
        Get workflow analytics and insights

        Returns comprehensive analytics including performance metrics,
        cost analysis, bottleneck identification, and optimization recommendations.

        Args:
            workflow_id: Specific workflow to analyze (optional)
            project_id: Project to analyze (optional, overrides workflow_id)
            period_days: Analysis period in days (default: 30)
            include_recommendations: Whether to include optimization recommendations

        Returns:
            JSON with comprehensive analytics data
        """
        try:
            analytics_service = await get_workflow_analytics_service()

            if project_id:
                # Project-level analytics
                analytics = await analytics_service.get_project_workflow_analytics(
                    project_id, period_days
                )
            elif workflow_id:
                # Single workflow analytics
                analytics = await analytics_service.get_workflow_performance(
                    workflow_id, period_days
                )
            else:
                return json.dumps({
                    "success": False,
                    "error": "Either workflow_id or project_id must be specified"
                })

            # Add recommendations if requested
            if include_recommendations and workflow_id:
                recommendations = await analytics_service.generate_workflow_recommendations(workflow_id)
                analytics["recommendations"] = recommendations

            # Add trends
            if workflow_id:
                trends = await analytics_service.get_workflow_trends(workflow_id, period_days)
                analytics["trends"] = trends

            return json.dumps({
                "success": True,
                "analytics": analytics,
                "period_days": period_days,
                "generated_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to get workflow analytics: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to get workflow analytics: {str(e)}"
            })

    @mcp.tool()
    async def archon_real_time_metrics(
        ctx: Context,
        project_id: Optional[str] = None
    ) -> str:
        """
        Get real-time workflow metrics

        Returns live metrics for dashboard display including active executions,
        recent activity, and system health status.

        Args:
            project_id: Filter metrics by project (optional)

        Returns:
            JSON with real-time metrics data
        """
        try:
            analytics_service = await get_workflow_analytics_service()
            metrics = await analytics_service.get_real_time_metrics(project_id)

            return json.dumps({
                "success": True,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to get real-time metrics: {str(e)}"
            })

    @mcp.tool()
    async def archon_validate_workflow(
        ctx: Context,
        flow_data: Dict[str, Any]
    ) -> str:
        """
        Validate ReactFlow workflow data

        Validates workflow structure for ReactFlow compatibility and
        execution readiness. Checks for circular dependencies, missing nodes, etc.

        Args:
            flow_data: ReactFlow nodes and edges data to validate

        Returns:
            JSON with validation results and any errors found
        """
        try:
            # Validate basic ReactFlow structure
            is_valid, errors = validate_reactflow_data(flow_data)

            # Additional workflow-specific validation
            workflow_errors = []
            nodes = flow_data.get("nodes", [])
            edges = flow_data.get("edges", [])

            # Check for nodes without connections
            connected_nodes = set()
            for edge in edges:
                connected_nodes.add(edge.get("source"))
                connected_nodes.add(edge.get("target"))

            disconnected_nodes = []
            for node in nodes:
                node_id = node.get("id")
                if node_id not in connected_nodes and len(nodes) > 1:
                    disconnected_nodes.append(node_id)

            if disconnected_nodes:
                workflow_errors.append(f"Disconnected nodes found: {', '.join(disconnected_nodes)}")

            # Check for circular dependencies
            if edges:
                has_cycles = self._detect_cycles(nodes, edges)
                if has_cycles:
                    workflow_errors.append("Circular dependencies detected in workflow")

            # Check for unreachable end nodes
            end_nodes = [n for n in nodes if not any(e.get("source") == n.get("id") for e in edges)]
            start_nodes = [n for n in nodes if not any(e.get("target") == n.get("id") for e in edges)]

            if len(end_nodes) == 0 and len(nodes) > 0:
                workflow_errors.append("No end nodes found - workflow cannot complete")

            all_errors = errors + workflow_errors

            return json.dumps({
                "success": True,
                "is_valid": len(all_errors) == 0,
                "validation_errors": all_errors,
                "validation_summary": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "connected_nodes": len(connected_nodes),
                    "disconnected_nodes": len(disconnected_nodes),
                    "start_nodes": len(start_nodes),
                    "end_nodes": len(end_nodes),
                    "has_cycles": has_cycles if edges else False
                }
            })

        except Exception as e:
            logger.error(f"Failed to validate workflow: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to validate workflow: {str(e)}"
            })

    def _detect_cycles(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> bool:
        """Detect cycles in workflow graph using DFS"""
        # Build adjacency list
        graph = {}
        for node in nodes:
            graph[node["id"]] = []

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                graph[source].append(target)

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in graph:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    logger.info("✅ Workflow MCP tools registered successfully")