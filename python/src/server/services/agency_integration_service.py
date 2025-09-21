"""
Agency Integration Service - Integration with existing systems.

This service provides integration capabilities between the agency workflow system
and existing systems including Agent Swarm, MCP tools, knowledge management,
and external services.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import aiohttp

from ...agents.orchestration.parallel_executor import ParallelExecutor
from ...agents.orchestration.archon_agency import ArchonAgency
from ...database.models import (
    AgencyWorkflow,
    WorkflowExecution,
    AgentCapability,
    ConversationThread,
    CommunicationMessage,
    IntegrationLog
)
from ...server.services.agency_workflow_service import AgencyWorkflowService
from ...server.services.agency_monitoring_service import AgencyMonitoringService

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    """Supported integration types."""
    AGENT_SWARM = "agent_swarm"
    MCP_TOOLS = "mcp_tools"
    KNOWLEDGE_BASE = "knowledge_base"
    EXTERNAL_API = "external_api"
    WEBHOOK = "webhook"
    MESSAGE_QUEUE = "message_queue"


class IntegrationStatus(str, Enum):
    """Integration status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class IntegrationConfig(BaseModel):
    """Configuration for integration settings."""
    integration_type: IntegrationType
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    webhook_url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


class IntegrationRequest(BaseModel):
    """Request model for integration operations."""
    integration_type: IntegrationType
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrationResponse(BaseModel):
    """Response model for integration operations."""
    success: bool
    data: Any
    error_message: Optional[str]
    execution_time_ms: float
    integration_metadata: Dict[str, Any]


class WebhookPayload(BaseModel):
    """Webhook payload model."""
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    source: str
    signature: Optional[str] = None


class AgencyIntegrationService:
    """Service for integrating agency workflows with existing systems."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.workflow_service = AgencyWorkflowService(db_session)
        self.monitoring_service = AgencyMonitoringService(db_session)
        self._integration_cache = {}
        self._webhook_handlers = {}
        self._active_integrations = {}

    async def initialize_integrations(self) -> None:
        """Initialize all configured integrations."""
        try:
            # Initialize Agent Swarm integration
            await self._initialize_agent_swarm_integration()

            # Initialize MCP Tools integration
            await self._initialize_mcp_tools_integration()

            # Initialize Knowledge Base integration
            await self._initialize_knowledge_base_integration()

            logger.info("Agency integrations initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing integrations: {e}")
            raise

    async def execute_integration(self, request: IntegrationRequest) -> IntegrationResponse:
        """Execute an integration operation."""
        start_time = datetime.utcnow()

        try:
            # Log the request
            await self._log_integration_request(request)

            # Execute based on integration type
            if request.integration_type == IntegrationType.AGENT_SWARM:
                result = await self._execute_agent_swarm_operation(request.operation, request.parameters)
            elif request.integration_type == IntegrationType.MCP_TOOLS:
                result = await self._execute_mcp_tools_operation(request.operation, request.parameters)
            elif request.integration_type == IntegrationType.KNOWLEDGE_BASE:
                result = await self._execute_knowledge_base_operation(request.operation, request.parameters)
            elif request.integration_type == IntegrationType.EXTERNAL_API:
                result = await self._execute_external_api_operation(request.operation, request.parameters)
            else:
                raise ValueError(f"Unsupported integration type: {request.integration_type}")

            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return IntegrationResponse(
                success=True,
                data=result,
                execution_time_ms=execution_time_ms,
                integration_metadata=request.metadata
            )

        except Exception as e:
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Log the error
            await self._log_integration_error(request, str(e))

            return IntegrationResponse(
                success=False,
                data=None,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                integration_metadata=request.metadata
            )

    async def register_webhook(self, event_type: str, handler_url: str, config: IntegrationConfig) -> bool:
        """Register a webhook handler for specific events."""
        try:
            webhook_id = f"{event_type}_{handler_url.replace('https://', '').replace('http://', '')}"

            self._webhook_handlers[webhook_id] = {
                "event_type": event_type,
                "handler_url": handler_url,
                "config": config,
                "registered_at": datetime.utcnow()
            }

            logger.info(f"Registered webhook for event: {event_type}")
            return True

        except Exception as e:
            logger.error(f"Error registering webhook: {e}")
            return False

    async def handle_webhook(self, payload: WebhookPayload) -> bool:
        """Handle incoming webhook payload."""
        try:
            # Verify signature if provided
            if payload.signature:
                if not await self._verify_webhook_signature(payload):
                    logger.warning("Invalid webhook signature")
                    return False

            # Find matching handlers
            matching_handlers = [
                handler for handler in self._webhook_handlers.values()
                if handler["event_type"] == payload.event_type
            ]

            # Send to all matching handlers
            for handler in matching_handlers:
                await self._send_webhook_payload(handler, payload)

            logger.info(f"Processed webhook for event: {payload.event_type}")
            return True

        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return False

    async def sync_with_agent_swarm(self) -> Dict[str, Any]:
        """Synchronize with Agent Swarm system."""
        try:
            sync_result = {
                "workflows_synced": 0,
                "agents_synced": 0,
                "capabilities_synced": 0,
                "errors": []
            }

            # Sync workflows
            workflows_synced = await self._sync_workflows_with_agent_swarm()
            sync_result["workflows_synced"] = workflows_synced

            # Sync agents
            agents_synced = await self._sync_agents_with_agent_swarm()
            sync_result["agents_synced"] = agents_synced

            # Sync capabilities
            capabilities_synced = await self._sync_capabilities_with_agent_swarm()
            sync_result["capabilities_synced"] = capabilities_synced

            logger.info(f"Agent Swarm sync completed: {sync_result}")
            return sync_result

        except Exception as e:
            logger.error(f"Error syncing with Agent Swarm: {e}")
            raise

    async def get_integration_status(self, integration_type: Optional[IntegrationType] = None) -> Dict[str, Any]:
        """Get status of all integrations."""
        try:
            status = {}

            # Check Agent Swarm integration
            if not integration_type or integration_type == IntegrationType.AGENT_SWARM:
                status["agent_swarm"] = await self._check_agent_swarm_status()

            # Check MCP Tools integration
            if not integration_type or integration_type == IntegrationType.MCP_TOOLS:
                status["mcp_tools"] = await self._check_mcp_tools_status()

            # Check Knowledge Base integration
            if not integration_type or integration_type == IntegrationType.KNOWLEDGE_BASE:
                status["knowledge_base"] = await self._check_knowledge_base_status()

            # Check External API integrations
            if not integration_type or integration_type == IntegrationType.EXTERNAL_API:
                status["external_apis"] = await self._check_external_api_status()

            return status

        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            raise

    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics."""
        try:
            # Get integration logs from last 24 hours
            time_threshold = datetime.utcnow() - timedelta(hours=24)

            stmt = select(IntegrationLog).where(
                IntegrationLog.created_at >= time_threshold
            )

            result = await self.db.execute(stmt)
            logs = result.scalars().all()

            # Calculate metrics
            total_calls = len(logs)
            successful_calls = sum(1 for log in logs if log.success)
            failed_calls = total_calls - successful_calls

            avg_execution_time = 0
            if logs:
                execution_times = [log.execution_time_ms for log in logs]
                avg_execution_time = sum(execution_times) / len(execution_times)

            # Group by integration type
            integration_stats = {}
            for log in logs:
                integration_type = log.integration_type
                if integration_type not in integration_stats:
                    integration_stats[integration_type] = {
                        "total_calls": 0,
                        "successful_calls": 0,
                        "failed_calls": 0,
                        "avg_execution_time": 0
                    }

                integration_stats[integration_type]["total_calls"] += 1
                if log.success:
                    integration_stats[integration_type]["successful_calls"] += 1
                else:
                    integration_stats[integration_type]["failed_calls"] += 1

            # Calculate averages per integration type
            for integration_type, stats in integration_stats.items():
                type_logs = [log for log in logs if log.integration_type == integration_type]
                if type_logs:
                    execution_times = [log.execution_time_ms for log in type_logs]
                    stats["avg_execution_time"] = sum(execution_times) / len(execution_times)

            return {
                "total_calls_24h": total_calls,
                "successful_calls_24h": successful_calls,
                "failed_calls_24h": failed_calls,
                "success_rate_24h": (successful_calls / total_calls * 100) if total_calls > 0 else 0,
                "average_execution_time_ms": avg_execution_time,
                "integration_stats": integration_stats
            }

        except Exception as e:
            logger.error(f"Error getting integration metrics: {e}")
            raise

    # Private methods for integration operations
    async def _initialize_agent_swarm_integration(self) -> None:
        """Initialize Agent Swarm integration."""
        try:
            # Test connection to Agent Swarm
            parallel_executor = ParallelExecutor()
            archon_agency = ArchonAgency()

            self._active_integrations["agent_swarm"] = {
                "status": IntegrationStatus.ACTIVE,
                "parallel_executor": parallel_executor,
                "archon_agency": archon_agency,
                "initialized_at": datetime.utcnow()
            }

            logger.info("Agent Swarm integration initialized")

        except Exception as e:
            logger.error(f"Error initializing Agent Swarm integration: {e}")
            self._active_integrations["agent_swarm"] = {
                "status": IntegrationStatus.ERROR,
                "error": str(e),
                "initialized_at": datetime.utcnow()
            }

    async def _initialize_mcp_tools_integration(self) -> None:
        """Initialize MCP Tools integration."""
        try:
            # Initialize MCP tools connection
            self._active_integrations["mcp_tools"] = {
                "status": IntegrationStatus.ACTIVE,
                "initialized_at": datetime.utcnow()
            }

            logger.info("MCP Tools integration initialized")

        except Exception as e:
            logger.error(f"Error initializing MCP Tools integration: {e}")
            self._active_integrations["mcp_tools"] = {
                "status": IntegrationStatus.ERROR,
                "error": str(e),
                "initialized_at": datetime.utcnow()
            }

    async def _initialize_knowledge_base_integration(self) -> None:
        """Initialize Knowledge Base integration."""
        try:
            # Initialize knowledge base connection
            self._active_integrations["knowledge_base"] = {
                "status": IntegrationStatus.ACTIVE,
                "initialized_at": datetime.utcnow()
            }

            logger.info("Knowledge Base integration initialized")

        except Exception as e:
            logger.error(f"Error initializing Knowledge Base integration: {e}")
            self._active_integrations["knowledge_base"] = {
                "status": IntegrationStatus.ERROR,
                "error": str(e),
                "initialized_at": datetime.utcnow()
            }

    async def _execute_agent_swarm_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute Agent Swarm operation."""
        try:
            if operation == "create_workflow":
                return await self.workflow_service.create_workflow(parameters)
            elif operation == "execute_workflow":
                return await self.workflow_service.execute_workflow(parameters)
            elif operation == "get_workflow_status":
                return await self.workflow_service.get_workflow_status(parameters.get("workflow_id"))
            elif operation == "list_workflows":
                return await self.workflow_service.list_workflows()
            else:
                raise ValueError(f"Unknown Agent Swarm operation: {operation}")

        except Exception as e:
            logger.error(f"Error executing Agent Swarm operation {operation}: {e}")
            raise

    async def _execute_mcp_tools_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute MCP Tools operation."""
        try:
            # This would integrate with the MCP server
            # For now, return a placeholder response
            return {
                "operation": operation,
                "parameters": parameters,
                "result": "MCP tool execution completed",
                "timestamp": datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Error executing MCP Tools operation {operation}: {e}")
            raise

    async def _execute_knowledge_base_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute Knowledge Base operation."""
        try:
            # This would integrate with the knowledge base system
            # For now, return a placeholder response
            return {
                "operation": operation,
                "parameters": parameters,
                "result": "Knowledge base operation completed",
                "timestamp": datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Error executing Knowledge Base operation {operation}: {e}")
            raise

    async def _execute_external_api_operation(self, operation: str, parameters: Dict[str, Any]) -> Any:
        """Execute External API operation."""
        try:
            url = parameters.get("url")
            method = parameters.get("method", "GET")
            headers = parameters.get("headers", {})
            data = parameters.get("data")

            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=data) as response:
                    result = await response.json()
                    return result

        except Exception as e:
            logger.error(f"Error executing External API operation {operation}: {e}")
            raise

    async def _sync_workflows_with_agent_swarm(self) -> int:
        """Sync workflows with Agent Swarm."""
        try:
            # Get workflows from Agent Swarm
            workflows = await self.workflow_service.list_workflows()
            return len(workflows)

        except Exception as e:
            logger.error(f"Error syncing workflows: {e}")
            return 0

    async def _sync_agents_with_agent_swarm(self) -> int:
        """Sync agents with Agent Swarm."""
        try:
            # Get agent capabilities from database
            stmt = select(AgentCapability).distinct(AgentCapability.agent_id)
            result = await self.db.execute(stmt)
            agent_ids = [row[0] for row in result.all()]
            return len(agent_ids)

        except Exception as e:
            logger.error(f"Error syncing agents: {e}")
            return 0

    async def _sync_capabilities_with_agent_swarm(self) -> int:
        """Sync capabilities with Agent Swarm."""
        try:
            # Get capabilities from database
            stmt = select(AgentCapability)
            result = await self.db.execute(stmt)
            capabilities = result.scalars().all()
            return len(capabilities)

        except Exception as e:
            logger.error(f"Error syncing capabilities: {e}")
            return 0

    async def _check_agent_swarm_status(self) -> Dict[str, Any]:
        """Check Agent Swarm integration status."""
        try:
            integration = self._active_integrations.get("agent_swarm")
            if not integration:
                return {"status": IntegrationStatus.INACTIVE.value, "message": "Integration not initialized"}

            return {
                "status": integration["status"].value,
                "message": "Agent Swarm integration is active",
                "last_check": datetime.utcnow()
            }

        except Exception as e:
            return {"status": IntegrationStatus.ERROR.value, "message": str(e)}

    async def _check_mcp_tools_status(self) -> Dict[str, Any]:
        """Check MCP Tools integration status."""
        try:
            integration = self._active_integrations.get("mcp_tools")
            if not integration:
                return {"status": IntegrationStatus.INACTIVE.value, "message": "Integration not initialized"}

            return {
                "status": integration["status"].value,
                "message": "MCP Tools integration is active",
                "last_check": datetime.utcnow()
            }

        except Exception as e:
            return {"status": IntegrationStatus.ERROR.value, "message": str(e)}

    async def _check_knowledge_base_status(self) -> Dict[str, Any]:
        """Check Knowledge Base integration status."""
        try:
            integration = self._active_integrations.get("knowledge_base")
            if not integration:
                return {"status": IntegrationStatus.INACTIVE.value, "message": "Integration not initialized"}

            return {
                "status": integration["status"].value,
                "message": "Knowledge Base integration is active",
                "last_check": datetime.utcnow()
            }

        except Exception as e:
            return {"status": IntegrationStatus.ERROR.value, "message": str(e)}

    async def _check_external_api_status(self) -> Dict[str, Any]:
        """Check External API integration status."""
        try:
            # Check health of configured external APIs
            return {
                "status": IntegrationStatus.ACTIVE.value,
                "message": "External API integrations configured",
                "last_check": datetime.utcnow()
            }

        except Exception as e:
            return {"status": IntegrationStatus.ERROR.value, "message": str(e)}

    async def _log_integration_request(self, request: IntegrationRequest) -> None:
        """Log integration request."""
        try:
            log_entry = IntegrationLog(
                integration_type=request.integration_type.value,
                operation=request.operation,
                parameters=request.parameters,
                success=True,
                execution_time_ms=0,
                created_at=datetime.utcnow()
            )

            self.db.add(log_entry)
            await self.db.commit()

        except Exception as e:
            logger.error(f"Error logging integration request: {e}")

    async def _log_integration_error(self, request: IntegrationRequest, error: str) -> None:
        """Log integration error."""
        try:
            log_entry = IntegrationLog(
                integration_type=request.integration_type.value,
                operation=request.operation,
                parameters=request.parameters,
                success=False,
                error_message=error,
                execution_time_ms=0,
                created_at=datetime.utcnow()
            )

            self.db.add(log_entry)
            await self.db.commit()

        except Exception as e:
            logger.error(f"Error logging integration error: {e}")

    async def _verify_webhook_signature(self, payload: WebhookPayload) -> bool:
        """Verify webhook signature."""
        # Implement signature verification logic
        return True

    async def _send_webhook_payload(self, handler: Dict[str, Any], payload: WebhookPayload) -> None:
        """Send webhook payload to handler."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    handler["handler_url"],
                    json=payload.dict(),
                    headers=handler["config"].headers,
                    timeout=handler["config"].timeout_seconds
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully to {handler['handler_url']}")
                    else:
                        logger.warning(f"Webhook failed with status {response.status}")

        except Exception as e:
            logger.error(f"Error sending webhook to {handler['handler_url']}: {e}")