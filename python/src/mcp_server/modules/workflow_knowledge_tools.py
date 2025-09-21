"""
MCP Tools for Workflow Knowledge Integration

Provides MCP tools that connect workflow execution with knowledge management,
enabling automatic knowledge capture, contextual search, and pattern-based
workflow optimization.

Tools integrate with existing Archon knowledge base and RAG systems.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from mcp.server.fastmcp import Context

from ...database import get_db
from ...database.workflow_models import WorkflowDefinition, WorkflowExecution
from ...server.services.knowledge_agent_bridge import KnowledgeAgentBridge
from ...server.services.workflow_knowledge_capture import WorkflowKnowledgeCapture
from ...server.services.knowledge_driven_workflow import KnowledgeDrivenWorkflow
from ...server.services.knowledge_embedding_service import KnowledgeEmbeddingService

logger = logging.getLogger(__name__)

def register_workflow_knowledge_tools(mcp):
    """Register all workflow knowledge integration MCP tools"""

    @mcp.tool()
    async def archon_start_workflow_knowledge_session(
        ctx: Context,
        workflow_id: str,
        project_id: str,
        capture_config: Optional[Dict[str, Any]] = None,
        context_tags: Optional[List[str]] = None
    ) -> str:
        """
        Start a knowledge capture session for workflow execution

        Initializes a session that automatically captures insights, patterns,
        and learnings during workflow execution for future knowledge retrieval.

        Args:
            workflow_id: UUID of the workflow being executed
            project_id: UUID of the project containing the workflow
            capture_config: Configuration for knowledge capture (auto_capture, pattern_detection, etc.)
            context_tags: Tags to associate with captured knowledge

        Returns:
            JSON with session ID and capture configuration
        """
        try:
            # Initialize knowledge bridge
            bridge = KnowledgeAgentBridge()

            # Set default capture configuration
            default_config = {
                "auto_capture": True,
                "capture_insights": True,
                "capture_patterns": True,
                "capture_errors": True,
                "capture_successes": True,
                "real_time_analysis": True,
                "embedding_generation": True
            }

            if capture_config:
                default_config.update(capture_config)

            # Start knowledge session
            session_id = await bridge.start_workflow_session(
                workflow_id=workflow_id,
                project_id=project_id,
                capture_config=default_config,
                context_tags=context_tags or []
            )

            return json.dumps({
                "success": True,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "project_id": project_id,
                "capture_config": default_config,
                "context_tags": context_tags or [],
                "started_at": datetime.now().isoformat(),
                "message": "Workflow knowledge session started successfully"
            })

        except Exception as e:
            logger.error(f"Failed to start workflow knowledge session: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to start knowledge session: {str(e)}"
            })

    @mcp.tool()
    async def archon_capture_workflow_insight(
        ctx: Context,
        session_id: str,
        insight_type: str,
        insight_data: Dict[str, Any],
        step_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        importance_score: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Capture an insight during workflow execution

        Manually or automatically capture insights, patterns, and learnings
        during workflow execution for future reference and optimization.

        Args:
            session_id: Knowledge session ID
            insight_type: Type of insight (performance_optimization, error_pattern, success_pattern, best_practice)
            insight_data: The insight content and metadata
            step_id: Workflow step where insight was captured
            execution_id: Workflow execution ID
            importance_score: Importance score (0.0-1.0)
            tags: Additional tags for the insight

        Returns:
            JSON with captured insight details and storage status
        """
        try:
            bridge = KnowledgeAgentBridge()

            # Validate insight type
            valid_types = [
                "performance_optimization", "error_pattern", "success_pattern",
                "best_practice", "bottleneck_identified", "efficiency_gain",
                "cost_optimization", "quality_improvement", "risk_mitigation"
            ]

            if insight_type not in valid_types:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid insight type: {insight_type}. Valid types: {valid_types}"
                })

            # Capture the insight
            insight_id = await bridge.capture_execution_insight(
                session_id=session_id,
                insight_type=insight_type,
                insight_data=insight_data,
                step_id=step_id,
                execution_id=execution_id,
                importance_score=importance_score or 0.5,
                tags=tags or []
            )

            return json.dumps({
                "success": True,
                "insight_id": insight_id,
                "session_id": session_id,
                "insight_type": insight_type,
                "step_id": step_id,
                "execution_id": execution_id,
                "importance_score": importance_score or 0.5,
                "tags": tags or [],
                "captured_at": datetime.now().isoformat(),
                "message": "Workflow insight captured successfully"
            })

        except Exception as e:
            logger.error(f"Failed to capture workflow insight: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to capture insight: {str(e)}"
            })

    @mcp.tool()
    async def archon_get_contextual_knowledge(
        ctx: Context,
        session_id: str,
        query: str,
        context_type: Optional[str] = "execution_context",
        max_results: Optional[int] = 10,
        similarity_threshold: Optional[float] = 0.7
    ) -> str:
        """
        Retrieve contextual knowledge for workflow execution

        Search the knowledge base for relevant information based on current
        workflow context, execution state, and specific queries.

        Args:
            session_id: Knowledge session ID
            query: Search query for knowledge retrieval
            context_type: Type of context (execution_context, step_context, project_context)
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results

        Returns:
            JSON with relevant knowledge items and contextual information
        """
        try:
            bridge = KnowledgeAgentBridge()

            # Validate context type
            valid_contexts = ["execution_context", "step_context", "project_context", "global_context"]
            if context_type not in valid_contexts:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid context type: {context_type}. Valid types: {valid_contexts}"
                })

            # Get contextual knowledge
            knowledge = await bridge.get_contextual_knowledge(
                session_id=session_id,
                query=query,
                context_type=context_type,
                max_results=max_results or 10,
                similarity_threshold=similarity_threshold or 0.7
            )

            return json.dumps({
                "success": True,
                "session_id": session_id,
                "query": query,
                "context_type": context_type,
                "knowledge_items": knowledge,
                "total_results": len(knowledge),
                "max_results": max_results or 10,
                "similarity_threshold": similarity_threshold or 0.7,
                "retrieved_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to get contextual knowledge: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to retrieve knowledge: {str(e)}"
            })

    @mcp.tool()
    async def archon_store_workflow_template(
        ctx: Context,
        workflow_id: str,
        template_name: str,
        template_description: str,
        use_cases: Optional[List[str]] = None,
        best_practices: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        is_public: Optional[bool] = False
    ) -> str:
        """
        Store a workflow as a reusable template in knowledge base

        Saves successful workflow patterns as templates that can be
        reused and adapted for future similar scenarios.

        Args:
            workflow_id: UUID of the workflow to template
            template_name: Name for the template
            template_description: Description of template purpose and use
            use_cases: List of applicable use cases
            best_practices: Best practices learned from this workflow
            tags: Tags for template categorization
            is_public: Whether template is available to all projects

        Returns:
            JSON with template storage status and metadata
        """
        try:
            bridge = KnowledgeAgentBridge()
            db = next(get_db())

            # Get workflow details
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == uuid.UUID(workflow_id)
            ).first()

            if not workflow:
                return json.dumps({
                    "success": False,
                    "error": "Workflow not found"
                })

            # Store as template
            template_id = await bridge.store_workflow_template(
                workflow_id=workflow_id,
                template_name=template_name,
                template_description=template_description,
                workflow_data=workflow.to_dict(),
                use_cases=use_cases or [],
                best_practices=best_practices or [],
                tags=tags or [],
                is_public=is_public or False
            )

            return json.dumps({
                "success": True,
                "template_id": template_id,
                "workflow_id": workflow_id,
                "template_name": template_name,
                "template_description": template_description,
                "use_cases": use_cases or [],
                "best_practices": best_practices or [],
                "tags": tags or [],
                "is_public": is_public or False,
                "stored_at": datetime.now().isoformat(),
                "message": "Workflow template stored successfully"
            })

        except Exception as e:
            logger.error(f"Failed to store workflow template: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to store template: {str(e)}"
            })

    @mcp.tool()
    async def archon_search_workflow_templates(
        ctx: Context,
        query: str,
        project_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = 20
    ) -> str:
        """
        Search for workflow templates in knowledge base

        Find reusable workflow templates based on search criteria,
        tags, and project context.

        Args:
            query: Search query for template matching
            project_id: Filter by project (optional)
            tags: Filter by template tags
            limit: Maximum number of templates to return

        Returns:
            JSON with matching workflow templates
        """
        try:
            bridge = KnowledgeAgentBridge()

            # Search templates
            templates = await bridge.search_workflow_templates(
                query=query,
                project_id=project_id,
                tags=tags or [],
                limit=limit or 20
            )

            return json.dumps({
                "success": True,
                "query": query,
                "project_id": project_id,
                "tags": tags or [],
                "templates": templates,
                "total_results": len(templates),
                "limit": limit or 20,
                "searched_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to search workflow templates: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to search templates: {str(e)}"
            })

    @mcp.tool()
    async def archon_end_workflow_knowledge_session(
        ctx: Context,
        session_id: str,
        generate_summary: Optional[bool] = True,
        extract_patterns: Optional[bool] = True
    ) -> str:
        """
        End a workflow knowledge session and generate insights

        Completes the knowledge capture session and optionally generates
        a summary and extracts patterns from the captured insights.

        Args:
            session_id: Knowledge session ID to end
            generate_summary: Whether to generate an execution summary
            extract_patterns: Whether to extract patterns from captured data

        Returns:
            JSON with session completion status and generated insights
        """
        try:
            bridge = KnowledgeAgentBridge()

            # End session and get summary
            result = await bridge.end_workflow_session(
                session_id=session_id,
                generate_summary=generate_summary,
                extract_patterns=extract_patterns
            )

            return json.dumps({
                "success": True,
                "session_id": session_id,
                "session_summary": result.get("summary") if generate_summary else None,
                "extracted_patterns": result.get("patterns") if extract_patterns else None,
                "total_insights_captured": result.get("total_insights", 0),
                "ended_at": datetime.now().isoformat(),
                "message": "Workflow knowledge session completed successfully"
            })

        except Exception as e:
            logger.error(f"Failed to end workflow knowledge session: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to end session: {str(e)}"
            })

    @mcp.tool()
    async def archon_analyze_workflow_performance_patterns(
        ctx: Context,
        workflow_id: str,
        period_days: Optional[int] = 30,
        include_recommendations: Optional[bool] = True
    ) -> str:
        """
        Analyze performance patterns from workflow executions

        Examines historical execution data to identify performance patterns,
        bottlenecks, optimization opportunities, and best practices.

        Args:
            workflow_id: UUID of workflow to analyze
            period_days: Analysis period in days
            include_recommendations: Whether to include optimization recommendations

        Returns:
            JSON with performance pattern analysis and insights
        """
        try:
            knowledge_capture = WorkflowKnowledgeCapture()

            # Analyze performance patterns
            patterns = await knowledge_capture.analyze_workflow_performance_patterns(
                workflow_id=workflow_id,
                period_days=period_days or 30
            )

            # Generate recommendations if requested
            recommendations = None
            if include_recommendations:
                recommendations = await knowledge_capture.generate_performance_recommendations(
                    workflow_id=workflow_id,
                    patterns=patterns
                )

            return json.dumps({
                "success": True,
                "workflow_id": workflow_id,
                "period_days": period_days or 30,
                "performance_patterns": patterns,
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to analyze workflow performance patterns: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to analyze patterns: {str(e)}"
            })

    @mcp.tool()
    async def archon_get_workflow_execution_context(
        ctx: Context,
        execution_id: str,
        include_knowledge: Optional[bool] = True,
        max_context_items: Optional[int] = 15
    ) -> str:
        """
        Get comprehensive context for workflow execution

        Retrieves execution context including current state, history,
        and relevant knowledge items for informed decision making.

        Args:
            execution_id: Workflow execution ID
            include_knowledge: Whether to include relevant knowledge items
            max_context_items: Maximum number of knowledge items to include

        Returns:
            JSON with comprehensive execution context
        """
        try:
            db = next(get_db())

            # Get execution details
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.execution_id == execution_id
            ).first()

            if not execution:
                return json.dumps({
                    "success": False,
                    "error": "Execution not found"
                })

            # Build context
            context = {
                "execution_id": execution_id,
                "workflow_id": str(execution.workflow_id),
                "status": execution.status.value,
                "progress": execution.progress,
                "current_step_id": execution.current_step_id,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "results": execution.results,
                "errors": execution.errors,
                "metrics": execution.metrics
            }

            # Add relevant knowledge if requested
            if include_knowledge:
                bridge = KnowledgeAgentBridge()

                # Create a session for this execution
                session_id = await bridge.start_workflow_session(
                    workflow_id=str(execution.workflow_id),
                    project_id="context_retrieval",
                    capture_config={"auto_capture": False}
                )

                # Get contextual knowledge
                query = f"Workflow execution {execution_id} context"
                knowledge = await bridge.get_contextual_knowledge(
                    session_id=session_id,
                    query=query,
                    context_type="execution_context",
                    max_results=max_context_items or 15
                )

                context["relevant_knowledge"] = knowledge

                # Clean up session
                await bridge.end_workflow_session(session_id, generate_summary=False)

            return json.dumps({
                "success": True,
                "execution_context": context,
                "included_knowledge": include_knowledge,
                "retrieved_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to get workflow execution context: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to get execution context: {str(e)}"
            })

    logger.info("✅ Workflow Knowledge MCP tools registered successfully")