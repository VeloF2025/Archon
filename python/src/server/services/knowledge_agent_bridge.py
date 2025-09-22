"""
Knowledge-Agent Bridge Service

Provides seamless integration between workflow execution and knowledge management,
enabling automatic knowledge capture, retrieval, and context-aware agent communication.

Key Features:
- Automatic knowledge capture during workflow execution
- Workflow template storage and retrieval
- Context-aware agent communication using RAG
- Performance insights as knowledge artifacts
- Integration with existing knowledge systems
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...database.agent_models import AgentV3, AgentState
from ...database.workflow_models import WorkflowDefinition, WorkflowExecution, StepExecution
from ..config.config import get_config
from .knowledge_embedding_service import KnowledgeEmbeddingService, KnowledgeItem, KnowledgeContext
from .search.rag_service import RAGService

# Temporary inline get_supabase_client function to avoid import issues
import os
import re
from supabase import Client, create_client
from ..config.logfire_config import search_logger

def get_supabase_client() -> Client:
    """Get a Supabase client instance."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
        )

    try:
        client = create_client(url, key)
        match = re.match(r"https://([^.]+)\.supabase\.co", url)
        if match:
            project_id = match.group(1)
            search_logger.info(f"Supabase client initialized - project_id={project_id}")
        return client
    except Exception as e:
        search_logger.error(f"Failed to create Supabase client: {e}")
        raise

logger = logging.getLogger(__name__)

class KnowledgeIntegrationType(Enum):
    """Types of knowledge integration"""
    WORKFLOW_TEMPLATE = "workflow_template"
    EXECUTION_INSIGHT = "execution_insight"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_PATTERN = "error_pattern"
    SUCCESS_PATTERN = "success_pattern"
    AGENT_COMMUNICATION = "agent_communication"
    CONTEXT_INSIGHT = "context_insight"

@dataclass
class WorkflowKnowledge:
    """Knowledge artifact from workflow execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    execution_id: str = ""
    knowledge_type: KnowledgeIntegrationType = KnowledgeIntegrationType.EXECUTION_INSIGHT
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    relevance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class KnowledgeCaptureConfig:
    """Configuration for automatic knowledge capture"""
    capture_execution_insights: bool = True
    capture_performance_metrics: bool = True
    capture_error_patterns: bool = True
    capture_success_patterns: bool = True
    capture_agent_communications: bool = True
    min_relevance_threshold: float = 0.7
    max_knowledge_items_per_execution: int = 10
    enable_context_awareness: bool = True

class KnowledgeAgentBridge:
    """
    Bridge between workflow execution and knowledge management systems

    This service provides intelligent integration between workflows and knowledge,
    enabling automatic learning and improvement through knowledge capture and retrieval.
    """

    def __init__(self, supabase_client=None):
        """Initialize the knowledge-agent bridge"""
        self.supabase = supabase_client or get_supabase_client()
        self.settings = get_config()

        # Initialize knowledge services
        self.knowledge_embedding_service = KnowledgeEmbeddingService(self.supabase)
        self.rag_service = RAGService(self.supabase)

        # Configuration
        self.capture_config = KnowledgeCaptureConfig()

        # Active knowledge capture sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Knowledge cache for workflow contexts
        self.workflow_knowledge_cache: Dict[str, List[WorkflowKnowledge]] = {}

        # Agent communication patterns
        self.communication_patterns: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("Knowledge-Agent Bridge initialized")

    async def start_workflow_session(self,
                                   workflow_id: str,
                                   execution_id: str,
                                   workflow_definition: Dict[str, Any]) -> str:
        """
        Start a knowledge capture session for workflow execution

        Args:
            workflow_id: ID of the workflow being executed
            execution_id: ID of the execution instance
            workflow_definition: Workflow definition data

        Returns:
            Session ID for knowledge tracking
        """
        session_id = f"session_{execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create session context
        session_context = {
            "session_id": session_id,
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "workflow_definition": workflow_definition,
            "start_time": datetime.now(),
            "knowledge_captured": [],
            "context_insights": [],
            "performance_metrics": {},
            "agent_communications": [],
            "error_patterns": [],
            "success_patterns": []
        }

        self.active_sessions[session_id] = session_context

        # Search for relevant existing knowledge
        relevant_knowledge = await self._search_relevant_workflow_knowledge(workflow_id)
        session_context["relevant_knowledge"] = relevant_knowledge

        logger.info(f"Started knowledge capture session: {session_id} for workflow: {workflow_id}")
        return session_id

    async def capture_execution_insight(self,
                                      session_id: str,
                                      step_id: str,
                                      insight_type: str,
                                      content: str,
                                      context: Optional[Dict[str, Any]] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Capture insights during workflow execution

        Args:
            session_id: Knowledge session ID
            step_id: Workflow step that generated the insight
            insight_type: Type of insight (performance, error, success, etc.)
            content: The insight content
            context: Context information
            metadata: Additional metadata

        Returns:
            Success status
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session ID: {session_id}")
            return False

        session = self.active_sessions[session_id]

        # Create knowledge item
        knowledge_item = WorkflowKnowledge(
            workflow_id=session["workflow_id"],
            execution_id=session["execution_id"],
            knowledge_type=KnowledgeIntegrationType.EXECUTION_INSIGHT,
            content=content,
            context=context or {},
            metadata={
                **(metadata or {}),
                "step_id": step_id,
                "insight_type": insight_type,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        )

        # Generate embedding
        knowledge_item.embedding = await self._generate_embedding(content)

        # Store in session
        session["knowledge_captured"].append(knowledge_item)

        # Categorize insight
        if insight_type == "error":
            session["error_patterns"].append(knowledge_item)
        elif insight_type == "success":
            session["success_patterns"].append(knowledge_item)
        elif insight_type == "performance":
            session["performance_metrics"][step_id] = knowledge_item

        logger.debug(f"Captured execution insight: {insight_type} for step: {step_id}")
        return True

    async def capture_agent_communication(self,
                                         session_id: str,
                                         agent_id: str,
                                         message: str,
                                         response: str,
                                         context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Capture agent communication patterns

        Args:
            session_id: Knowledge session ID
            agent_id: ID of the agent
            message: Agent message
            response: Response received
            context: Communication context

        Returns:
            Success status
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Create communication record
        communication = {
            "agent_id": agent_id,
            "message": message,
            "response": response,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }

        session["agent_communications"].append(communication)

        # Extract communication patterns
        pattern = await self._extract_communication_pattern(
            agent_id, message, response, context or {}
        )

        if pattern:
            knowledge_item = WorkflowKnowledge(
                workflow_id=session["workflow_id"],
                execution_id=session["execution_id"],
                knowledge_type=KnowledgeIntegrationType.AGENT_COMMUNICATION,
                content=pattern["content"],
                context=pattern.get("context", {}),
                metadata={
                    **pattern.get("metadata", {}),
                    "agent_id": agent_id,
                    "pattern_type": pattern["type"],
                    "session_id": session_id
                }
            )

            knowledge_item.embedding = await self._generate_embedding(pattern["content"])
            session["knowledge_captured"].append(knowledge_item)

        return True

    async def get_contextual_knowledge(self,
                                      session_id: str,
                                      query: str,
                                      context_type: str = "execution") -> List[Dict[str, Any]]:
        """
        Get knowledge relevant to current execution context

        Args:
            session_id: Knowledge session ID
            query: Search query
            context_type: Type of context (execution, workflow, step)

        Returns:
            List of relevant knowledge items
        """
        if session_id not in self.active_sessions:
            return []

        session = self.active_sessions[session_id]

        # Create knowledge context
        knowledge_context = KnowledgeContext(
            agent_id=uuid.UUID(session["workflow_id"]),  # Use workflow_id as agent_id
            task_id=uuid.UUID(session["execution_id"]),
            query=query,
            max_results=5,
            similarity_threshold=self.capture_config.min_relevance_threshold,
            include_global=True
        )

        # Search for relevant knowledge
        relevant_knowledge = await self.knowledge_embedding_service.retrieve_relevant_knowledge(
            knowledge_context
        )

        # Add session-specific knowledge
        session_knowledge = [
            {
                "content": item.content,
                "context": item.context,
                "metadata": item.metadata,
                "relevance_score": item.relevance_score,
                "source": "session"
            }
            for item in session["knowledge_captured"]
            if self._is_relevant_to_query(item.content, query)
        ]

        # Combine and rank results
        all_knowledge = []

        # Add RAG results
        for item in relevant_knowledge:
            all_knowledge.append({
                "content": item.content,
                "context": item.context,
                "metadata": item.metadata,
                "relevance_score": item.relevance_score,
                "source": "global"
            })

        # Add session knowledge
        all_knowledge.extend(session_knowledge)

        # Sort by relevance
        all_knowledge.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return all_knowledge[:5]  # Return top 5 results

    async def store_workflow_template(self,
                                    workflow_id: str,
                                    template_data: Dict[str, Any],
                                    description: str,
                                    tags: List[str] = None) -> Tuple[bool, str]:
        """
        Store workflow template as knowledge artifact

        Args:
            workflow_id: ID of the workflow
            template_data: Template data (ReactFlow format)
            description: Template description
            tags: Template tags

        Returns:
            Success status and message
        """
        try:
            # Create knowledge item for template
            content = f"Workflow Template: {description}\n\n"
            content += f"Workflow ID: {workflow_id}\n"
            content += f"Nodes: {len(template_data.get('nodes', []))}\n"
            content += f"Edges: {len(template_data.get('edges', []))}\n\n"

            # Add template structure information
            nodes = template_data.get("nodes", [])
            if nodes:
                content += "Node Types:\n"
                node_types = {}
                for node in nodes:
                    node_type = node.get("type", "unknown")
                    node_types[node_type] = node_types.get(node_type, 0) + 1

                for node_type, count in node_types.items():
                    content += f"- {node_type}: {count}\n"

            # Create knowledge item
            knowledge_item = WorkflowKnowledge(
                workflow_id=workflow_id,
                execution_id="template",
                knowledge_type=KnowledgeIntegrationType.WORKFLOW_TEMPLATE,
                content=content,
                context={
                    "template_data": template_data,
                    "description": description,
                    "tags": tags or []
                },
                metadata={
                    "template_id": f"template_{workflow_id}",
                    "node_count": len(nodes),
                    "edge_count": len(template_data.get("edges", [])),
                    "tags": tags or [],
                    "created_at": datetime.now().isoformat()
                }
            )

            # Generate embedding and store
            knowledge_item.embedding = await self._generate_embedding(content)
            success = await self._store_knowledge_item(knowledge_item)

            if success:
                logger.info(f"Stored workflow template: {workflow_id}")
                return True, "Template stored successfully"
            else:
                return False, "Failed to store template"

        except Exception as e:
            logger.error(f"Failed to store workflow template: {e}")
            return False, f"Storage failed: {str(e)}"

    async def get_workflow_templates(self,
                                   search_query: str = None,
                                   tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve workflow templates from knowledge base

        Args:
            search_query: Optional search query
            tags: Filter by tags

        Returns:
            List of workflow templates
        """
        try:
            # Build search context
            query = search_query or "workflow template"
            if tags:
                query += " " + " ".join(tags)

            knowledge_context = KnowledgeContext(
                agent_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),  # System agent
                task_id=uuid.uuid4(),
                query=query,
                max_results=20,
                similarity_threshold=0.6,
                include_global=True
            )

            # Search for templates
            relevant_items = await self.knowledge_embedding_service.retrieve_relevant_knowledge(
                knowledge_context
            )

            # Filter for workflow templates
            templates = []
            for item in relevant_items:
                if (item.metadata.get("template_id") and
                    item.metadata.get("template_id", "").startswith("template_")):

                    # Apply tag filter if specified
                    if tags:
                        item_tags = item.metadata.get("tags", [])
                        if not any(tag in item_tags for tag in tags):
                            continue

                    templates.append({
                        "template_id": item.metadata["template_id"],
                        "workflow_id": item.workflow_id,
                        "description": item.context.get("description", ""),
                        "content": item.content,
                        "node_count": item.metadata.get("node_count", 0),
                        "edge_count": item.metadata.get("edge_count", 0),
                        "tags": item.metadata.get("tags", []),
                        "relevance_score": item.relevance_score,
                        "created_at": item.created_at.isoformat()
                    })

            return templates

        except Exception as e:
            logger.error(f"Failed to retrieve workflow templates: {e}")
            return []

    async def complete_workflow_session(self,
                                       session_id: str,
                                       execution_results: Dict[str, Any]) -> Tuple[bool, List[WorkflowKnowledge]]:
        """
        Complete knowledge capture session and process insights

        Args:
            session_id: Knowledge session ID
            execution_results: Final execution results

        Returns:
            Success status and list of captured knowledge items
        """
        if session_id not in self.active_sessions:
            return False, []

        session = self.active_sessions[session_id]

        try:
            # Process session data and generate final insights
            final_insights = await self._generate_final_insights(session, execution_results)

            # Add final insights to captured knowledge
            for insight in final_insights:
                session["knowledge_captured"].append(insight)

            # Store all captured knowledge items
            stored_items = []
            for knowledge_item in session["knowledge_captured"]:
                success = await self._store_knowledge_item(knowledge_item)
                if success:
                    stored_items.append(knowledge_item)

            # Update workflow knowledge cache
            workflow_id = session["workflow_id"]
            if workflow_id not in self.workflow_knowledge_cache:
                self.workflow_knowledge_cache[workflow_id] = []
            self.workflow_knowledge_cache[workflow_id].extend(stored_items)

            # Clean up session
            del self.active_sessions[session_id]

            logger.info(f"Completed knowledge capture session: {session_id}")
            logger.info(f"Stored {len(stored_items)} knowledge items")

            return True, stored_items

        except Exception as e:
            logger.error(f"Failed to complete knowledge session {session_id}: {e}")
            return False, []

    async def get_workflow_knowledge_summary(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get summary of knowledge associated with a workflow

        Args:
            workflow_id: Workflow ID

        Returns:
            Knowledge summary
        """
        try:
            # Get cached knowledge
            cached_items = self.workflow_knowledge_cache.get(workflow_id, [])

            # Get agent knowledge summary
            agent_summary = await self.knowledge_embedding_service.get_agent_knowledge_summary(
                uuid.UUID(workflow_id)
            )

            # Compile summary
            summary = {
                "workflow_id": workflow_id,
                "cached_knowledge_items": len(cached_items),
                "agent_knowledge_summary": agent_summary,
                "knowledge_types": {},
                "recent_insights": [],
                "performance_trends": {}
            }

            # Analyze knowledge types
            for item in cached_items:
                knowledge_type = item.knowledge_type.value
                if knowledge_type not in summary["knowledge_types"]:
                    summary["knowledge_types"][knowledge_type] = 0
                summary["knowledge_types"][knowledge_type] += 1

                # Add recent insights
                if (datetime.now() - item.created_at).days <= 7:
                    summary["recent_insights"].append({
                        "content": item.content[:200],
                        "type": knowledge_type,
                        "created_at": item.created_at.isoformat()
                    })

            return summary

        except Exception as e:
            logger.error(f"Failed to get workflow knowledge summary: {e}")
            return {"workflow_id": workflow_id, "error": str(e)}

    # Private helper methods

    async def _search_relevant_workflow_knowledge(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Search for knowledge relevant to a specific workflow"""
        try:
            knowledge_context = KnowledgeContext(
                agent_id=uuid.UUID(workflow_id),
                task_id=uuid.uuid4(),
                query=f"workflow {workflow_id} patterns insights best practices",
                max_results=10,
                similarity_threshold=0.6,
                include_global=True
            )

            relevant_items = await self.knowledge_embedding_service.retrieve_relevant_knowledge(
                knowledge_context
            )

            return [
                {
                    "content": item.content,
                    "context": item.context,
                    "metadata": item.metadata,
                    "relevance_score": item.relevance_score
                }
                for item in relevant_items
            ]

        except Exception as e:
            logger.error(f"Failed to search relevant knowledge: {e}")
            return []

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.knowledge_embedding_service.generate_embedding(text)

    async def _store_knowledge_item(self, knowledge_item: WorkflowKnowledge) -> bool:
        """Store knowledge item in knowledge base"""
        try:
            # Convert to KnowledgeItem for embedding service
            knowledge_context = {
                "workflow_id": knowledge_item.workflow_id,
                "execution_id": knowledge_item.execution_id,
                "knowledge_type": knowledge_item.knowledge_type.value
            }

            await self.knowledge_embedding_service.store_knowledge(
                agent_id=uuid.UUID(knowledge_item.workflow_id),
                content=knowledge_item.content,
                context=knowledge_context,
                metadata=knowledge_item.metadata
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store knowledge item: {e}")
            return False

    async def _extract_communication_pattern(self,
                                           agent_id: str,
                                           message: str,
                                           response: str,
                                           context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract patterns from agent communications"""
        try:
            # Simple pattern extraction - can be enhanced with ML
            if "error" in response.lower():
                return {
                    "type": "error_response",
                    "content": f"Agent {agent_id} error pattern: {message[:100]} -> {response[:100]}",
                    "context": context,
                    "metadata": {
                        "agent_id": agent_id,
                        "pattern_category": "error_handling"
                    }
                }
            elif "success" in response.lower() or "completed" in response.lower():
                return {
                    "type": "success_response",
                    "content": f"Agent {agent_id} success pattern: {message[:100]} -> {response[:100]}",
                    "context": context,
                    "metadata": {
                        "agent_id": agent_id,
                        "pattern_category": "success_handling"
                    }
                }

            return None

        except Exception as e:
            logger.error(f"Failed to extract communication pattern: {e}")
            return None

    async def _generate_final_insights(self,
                                      session: Dict[str, Any],
                                      execution_results: Dict[str, Any]) -> List[WorkflowKnowledge]:
        """Generate final insights from completed workflow session"""
        insights = []

        try:
            # Performance insight
            if execution_results.get("execution_time_seconds"):
                performance_insight = WorkflowKnowledge(
                    workflow_id=session["workflow_id"],
                    execution_id=session["execution_id"],
                    knowledge_type=KnowledgeIntegrationType.PERFORMANCE_METRIC,
                    content=f"Workflow {session['workflow_id']} completed in {execution_results['execution_time_seconds']} seconds",
                    context={"execution_results": execution_results},
                    metadata={
                        "insight_type": "performance_summary",
                        "execution_time": execution_results.get("execution_time_seconds"),
                        "status": execution_results.get("status")
                    }
                )
                performance_insight.embedding = await self._generate_embedding(performance_insight.content)
                insights.append(performance_insight)

            # Error pattern analysis
            if session["error_patterns"]:
                error_summary = f"Workflow {session['workflow_id']} encountered {len(session['error_patterns'])} errors"
                error_insight = WorkflowKnowledge(
                    workflow_id=session["workflow_id"],
                    execution_id=session["execution_id"],
                    knowledge_type=KnowledgeIntegrationType.ERROR_PATTERN,
                    content=error_summary,
                    context={"error_patterns": [item.content for item in session["error_patterns"]]},
                    metadata={
                        "insight_type": "error_summary",
                        "error_count": len(session["error_patterns"])
                    }
                )
                error_insight.embedding = await self._generate_embedding(error_insight.content)
                insights.append(error_insight)

            # Success pattern analysis
            if session["success_patterns"]:
                success_summary = f"Workflow {session['workflow_id']} had {len(session['success_patterns'])} successful patterns"
                success_insight = WorkflowKnowledge(
                    workflow_id=session["workflow_id"],
                    execution_id=session["execution_id"],
                    knowledge_type=KnowledgeIntegrationType.SUCCESS_PATTERN,
                    content=success_summary,
                    context={"success_patterns": [item.content for item in session["success_patterns"]]},
                    metadata={
                        "insight_type": "success_summary",
                        "success_count": len(session["success_patterns"])
                    }
                )
                success_insight.embedding = await self._generate_embedding(success_insight.content)
                insights.append(success_insight)

        except Exception as e:
            logger.error(f"Failed to generate final insights: {e}")

        return insights

    def _is_relevant_to_query(self, content: str, query: str) -> bool:
        """Simple relevance check between content and query"""
        if not query:
            return True

        query_lower = query.lower()
        content_lower = content.lower()

        # Simple keyword matching
        query_keywords = query_lower.split()
        content_words = content_lower.split()

        match_count = sum(1 for keyword in query_keywords if keyword in content_words)
        return match_count >= len(query_keywords) * 0.3  # 30% keyword match

# Global instance
_knowledge_agent_bridge = None

def get_knowledge_agent_bridge(supabase_client=None) -> KnowledgeAgentBridge:
    """Get or create the global knowledge-agent bridge instance"""
    global _knowledge_agent_bridge
    if _knowledge_agent_bridge is None:
        _knowledge_agent_bridge = KnowledgeAgentBridge(supabase_client)
    return _knowledge_agent_bridge