#!/usr/bin/env python3
"""
Agency Swarm Integration Validation Script

This script performs comprehensive integration validation to ensure
all Agency Swarm components work together seamlessly.
"""

import asyncio
import sys
import json
import time
import subprocess
import requests
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IntegrationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    PARTIAL = "PARTIAL"
    SKIP = "SKIP"


@dataclass
class IntegrationTest:
    name: str
    description: str
    status: IntegrationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    critical: bool = False


class IntegrationValidator:
    """Comprehensive integration validation system."""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.test_results: List[IntegrationTest] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "http://localhost:8181"  # Default backend URL
        self.frontend_url = "http://localhost:3737"  # Default frontend URL
        self.mcp_url = "http://localhost:8051"  # Default MCP URL
        self.agents_url = "http://localhost:8052"  # Default agents URL

    async def run_all_validations(self) -> List[IntegrationTest]:
        """Run all integration validation tests."""
        print(f"🔗 Starting integration validation for environment: {self.environment}")
        print("=" * 60)

        # Initialize HTTP session
        self.session = aiohttp.ClientSession()

        try:
            # Service Health Integration Tests
            await self._test_service_health_integration()
            await self._test_service_communication()
            await self._test_service_discovery()

            # API Integration Tests
            await self._test_api_endpoint_integration()
            await self._test_authentication_integration()
            await self._test_authorization_integration()

            # Database Integration Tests
            await self._test_database_service_integration()
            await self._test_database_schema_integration()
            await self._test_database_performance_integration()

            # Agent Integration Tests
            await self._test_agent_service_integration()
            await self._test_agent_communication_integration()
            await self._test_agent_lifecycle_integration()

            # MCP Integration Tests
            await self._test_mcp_service_integration()
            await self._test_mcp_tool_integration()
            await self._test_mcp_agent_integration()

            # Real-time Integration Tests
            await self._test_websocket_integration()
            await self._test_message_queue_integration()
            await self._test_event_streaming_integration()

            # Knowledge Base Integration Tests
            await self._test_knowledge_service_integration()
            await self._test_knowledge_search_integration()
            await self._test_knowledge_update_integration()

            # Frontend Integration Tests
            await self._test_frontend_backend_integration()
            await self._test_frontend_api_integration()
            await self._test_frontend_realtime_integration()

            # Monitoring Integration Tests
            await self._test_monitoring_integration()
            await self._test_logging_integration()
            await self._test_metrics_integration()

            # Security Integration Tests
            await self._test_authentication_flow_integration()
            await self._test_authorization_flow_integration()
            await self._test_security_policy_integration()

            # Performance Integration Tests
            await self._test_performance_integration()
            await self._test_scaling_integration()
            await self._test_recovery_integration()

            # Generate integration report
            await self._generate_integration_report()

            return self.test_results

        finally:
            if self.session:
                await self.session.close()

    async def _test_service_health_integration(self) -> None:
        """Test health integration across all services."""
        start_time = time.time()

        services = [
            ("Backend API", self.base_url + "/health"),
            ("Frontend", self.frontend_url + "/health"),
            ("MCP Service", self.mcp_url + "/health"),
            ("Agents Service", self.agents_url + "/health")
        ]

        healthy_services = []
        unhealthy_services = []

        for service_name, health_url in services:
            try:
                async with self.session.get(health_url, timeout=10) as response:
                    if response.status == 200:
                        healthy_services.append(service_name)
                    else:
                        unhealthy_services.append(service_name)
            except:
                unhealthy_services.append(service_name)

        if len(healthy_services) == len(services):
            status = IntegrationStatus.PASS
            message = f"All services are healthy: {', '.join(healthy_services)}"
        elif len(healthy_services) > 0:
            status = IntegrationStatus.PARTIAL
            message = f"Some services are unhealthy. Healthy: {', '.join(healthy_services)}, Unhealthy: {', '.join(unhealthy_services)}"
        else:
            status = IntegrationStatus.FAIL
            message = f"All services are unhealthy: {', '.join(unhealthy_services)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Service Health Integration",
            description="Test health check integration across all services",
            status=status,
            message=message,
            details={
                "healthy_services": healthy_services,
                "unhealthy_services": unhealthy_services,
                "total_services": len(services)
            },
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_service_communication(self) -> None:
        """Test inter-service communication."""
        start_time = time.time()

        communication_tests = [
            ("Backend to MCP", self.base_url + "/api/mcp/health"),
            ("Backend to Agents", self.base_url + "/agent-management/agents"),
            ("Frontend to Backend", self.frontend_url + "/api/health"),
            ("MCP to Agents", self.mcp_url + "/agents/health")
        ]

        successful_communications = []
        failed_communications = []

        for test_name, url in communication_tests:
            try:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        successful_communications.append(test_name)
                    else:
                        failed_communications.append(test_name)
            except:
                failed_communications.append(test_name)

        if len(successful_communications) == len(communication_tests):
            status = IntegrationStatus.PASS
            message = f"All service communications working: {', '.join(successful_communications)}"
        elif len(successful_communications) > 0:
            status = IntegrationStatus.PARTIAL
            message = f"Partial service communication. Working: {', '.join(successful_communications)}, Failed: {', '.join(failed_communications)}"
        else:
            status = IntegrationStatus.FAIL
            message = f"All service communications failed: {', '.join(failed_communications)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Service Communication",
            description="Test inter-service communication",
            status=status,
            message=message,
            details={
                "successful": successful_communications,
                "failed": failed_communications,
                "total_tests": len(communication_tests)
            },
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_service_discovery(self) -> None:
        """Test service discovery mechanisms."""
        start_time = time.time()

        # Test if services can discover each other
        try:
            # Check if backend can discover MCP
            async with self.session.get(self.base_url + "/api/mcp/tools", timeout=10) as response:
                if response.status == 200:
                    mcp_discovered = True
                else:
                    mcp_discovered = False
        except:
            mcp_discovered = False

        try:
            # Check if backend can discover agents
            async with self.session.get(self.base_url + "/agent-management/agents", timeout=10) as response:
                if response.status == 200:
                    agents_discovered = True
                else:
                    agents_discovered = False
        except:
            agents_discovered = False

        if mcp_discovered and agents_discovered:
            status = IntegrationStatus.PASS
            message = "Service discovery working for all services"
        elif mcp_discovered or agents_discovered:
            status = IntegrationStatus.PARTIAL
            message = "Partial service discovery working"
        else:
            status = IntegrationStatus.FAIL
            message = "Service discovery not working"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Service Discovery",
            description="Test service discovery mechanisms",
            status=status,
            message=message,
            details={
                "mcp_discovered": mcp_discovered,
                "agents_discovered": agents_discovered
            },
            duration=duration
        )
        self.test_results.append(test)

    async def _test_api_endpoint_integration(self) -> None:
        """Test API endpoint integration and consistency."""
        start_time = time.time()

        api_endpoints = [
            ("Health Check", "/health", "GET"),
            ("Agent List", "/agent-management/agents", "GET"),
            ("Project Overview", "/agent-management/analytics/project-overview", "GET"),
            ("Cost Summary", "/agent-management/costs/summary", "GET"),
            ("Knowledge Search", "/api/knowledge/search", "POST")
        ]

        working_endpoints = []
        failing_endpoints = []

        for endpoint_name, endpoint_path, method in api_endpoints:
            try:
                if method == "GET":
                    async with self.session.get(self.base_url + endpoint_path, timeout=10) as response:
                        if response.status in [200, 201]:
                            working_endpoints.append(endpoint_name)
                        else:
                            failing_endpoints.append(f"{endpoint_name} (status: {response.status})")
                else:  # POST
                    async with self.session.post(self.base_url + endpoint_path, json={"query": "test"}, timeout=10) as response:
                        if response.status in [200, 201]:
                            working_endpoints.append(endpoint_name)
                        else:
                            failing_endpoints.append(f"{endpoint_name} (status: {response.status})")
            except Exception as e:
                failing_endpoints.append(f"{endpoint_name} (error: {str(e)})")

        if len(working_endpoints) == len(api_endpoints):
            status = IntegrationStatus.PASS
            message = f"All API endpoints working: {', '.join(working_endpoints)}"
        elif len(working_endpoints) > 0:
            status = IntegrationStatus.PARTIAL
            message = f"Partial API endpoint functionality. Working: {', '.join(working_endpoints)}, Failing: {', '.join(failing_endpoints)}"
        else:
            status = IntegrationStatus.FAIL
            message = f"All API endpoints failing: {', '.join(failing_endpoints)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="API Endpoint Integration",
            description="Test API endpoint integration and consistency",
            status=status,
            message=message,
            details={
                "working_endpoints": working_endpoints,
                "failing_endpoints": failing_endpoints,
                "total_endpoints": len(api_endpoints)
            },
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_authentication_integration(self) -> None:
        """Test authentication integration across services."""
        start_time = time.time()

        # Test authentication endpoints
        auth_tests = [
            ("Backend Auth", self.base_url + "/auth/test"),
            ("Frontend Auth", self.frontend_url + "/auth/test"),
            ("MCP Auth", self.mcp_url + "/auth/test"),
            ("Agents Auth", self.agents_url + "/auth/test")
        ]

        working_auth = []
        failing_auth = []

        for auth_name, auth_url in auth_tests:
            try:
                async with self.session.get(auth_url, timeout=10) as response:
                    if response.status in [200, 401]:  # 401 is acceptable for auth endpoints
                        working_auth.append(auth_name)
                    else:
                        failing_auth.append(f"{auth_name} (status: {response.status})")
            except:
                failing_auth.append(f"{auth_name} (connection failed)")

        if len(working_auth) == len(auth_tests):
            status = IntegrationStatus.PASS
            message = f"Authentication integration working: {', '.join(working_auth)}"
        elif len(working_auth) > 0:
            status = IntegrationStatus.PARTIAL
            message = f"Partial authentication integration. Working: {', '.join(working_auth)}, Failing: {', '.join(failing_auth)}"
        else:
            status = IntegrationStatus.FAIL
            message = f"Authentication integration failing: {', '.join(failing_auth)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Authentication Integration",
            description="Test authentication integration across services",
            status=status,
            message=message,
            details={
                "working_auth": working_auth,
                "failing_auth": failing_auth
            },
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_authorization_integration(self) -> None:
        """Test authorization integration across services."""
        start_time = time.time()

        # Test that authorization is consistently applied
        protected_endpoints = [
            ("Agent Creation", self.base_url + "/agent-management/agents", "POST"),
            ("Cost Tracking", self.base_url + "/agent-management/costs/track", "POST"),
            ("Knowledge Upload", self.base_url + "/api/knowledge/upload", "POST")
        ]

        consistent_authorization = []
        inconsistent_authorization = []

        for endpoint_name, endpoint_url, method in protected_endpoints:
            try:
                if method == "POST":
                    async with self.session.post(endpoint_url, json={"test": "data"}, timeout=10) as response:
                        if response.status in [401, 403]:  # Expected for unauthorized requests
                            consistent_authorization.append(endpoint_name)
                        else:
                            inconsistent_authorization.append(f"{endpoint_name} (status: {response.status})")
            except:
                inconsistent_authorization.append(f"{endpoint_name} (connection failed)")

        if len(consistent_authorization) == len(protected_endpoints):
            status = IntegrationStatus.PASS
            message = f"Authorization integration consistent: {', '.join(consistent_authorization)}"
        elif len(consistent_authorization) > 0:
            status = IntegrationStatus.PARTIAL
            message = f"Partial authorization integration. Consistent: {', '.join(consistent_authorization)}, Inconsistent: {', '.join(inconsistent_authorization)}"
        else:
            status = IntegrationStatus.FAIL
            message = f"Authorization integration inconsistent: {', '.join(inconsistent_authorization)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Authorization Integration",
            description="Test authorization integration across services",
            status=status,
            message=message,
            details={
                "consistent_authorization": consistent_authorization,
                "inconsistent_authorization": inconsistent_authorization
            },
            duration=duration
        )
        self.test_results.append(test)

    async def _test_database_service_integration(self) -> None:
        """Test database service integration."""
        start_time = time.time()

        try:
            # Test database connectivity through API
            async with self.session.get(self.base_url + "/agent-management/agents", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list):
                        status = IntegrationStatus.PASS
                        message = "Database service integration working correctly"
                    else:
                        status = IntegrationStatus.FAIL
                        message = "Database service returning invalid data format"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Database service integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Database service integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Database Service Integration",
            description="Test database service integration",
            status=status,
            message=message,
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_database_schema_integration(self) -> None:
        """Test database schema integration."""
        start_time = time.time()

        # Test that all required tables are accessible
        schema_tests = [
            ("Agents Table", self.base_url + "/agent-management/agents"),
            ("Cost Tracking", self.base_url + "/agent-management/costs/summary"),
            ("Knowledge Base", self.base_url + "/api/knowledge/items")
        ]

        working_schema = []
        failing_schema = []

        for table_name, endpoint_url in schema_tests:
            try:
                async with self.session.get(endpoint_url, timeout=10) as response:
                    if response.status == 200:
                        working_schema.append(table_name)
                    else:
                        failing_schema.append(f"{table_name} (status: {response.status})")
            except:
                failing_schema.append(f"{table_name} (connection failed)")

        if len(working_schema) == len(schema_tests):
            status = IntegrationStatus.PASS
            message = f"Database schema integration working: {', '.join(working_schema)}"
        elif len(working_schema) > 0:
            status = IntegrationStatus.PARTIAL
            message = f"Partial database schema integration. Working: {', '.join(working_schema)}, Failing: {', '.join(failing_schema)}"
        else:
            status = IntegrationStatus.FAIL
            message = f"Database schema integration failing: {', '.join(failing_schema)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Database Schema Integration",
            description="Test database schema integration",
            status=status,
            message=message,
            details={
                "working_schema": working_schema,
                "failing_schema": failing_schema
            },
            duration=duration
        )
        self.test_results.append(test)

    async def _test_database_performance_integration(self) -> None:
        """Test database performance integration."""
        start_time = time.time()

        try:
            # Test database query performance
            start_query = time.time()
            async with self.session.get(self.base_url + "/agent-management/agents", timeout=10) as response:
                if response.status == 200:
                    query_time = time.time() - start_query
                    if query_time < 1.0:  # Less than 1 second is acceptable
                        status = IntegrationStatus.PASS
                        message = f"Database performance integration working (query time: {query_time:.2f}s)"
                    else:
                        status = IntegrationStatus.WARNING
                        message = f"Database performance slow (query time: {query_time:.2f}s)"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Database performance query failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Database performance integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Database Performance Integration",
            description="Test database performance integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_agent_service_integration(self) -> None:
        """Test agent service integration."""
        start_time = time.time()

        try:
            # Test agent service health
            async with self.session.get(self.agents_url + "/health", timeout=10) as response:
                if response.status == 200:
                    # Test agent service through backend API
                    async with self.session.get(self.base_url + "/agent-management/agents", timeout=10) as response:
                        if response.status == 200:
                            status = IntegrationStatus.PASS
                            message = "Agent service integration working correctly"
                        else:
                            status = IntegrationStatus.FAIL
                            message = f"Agent service integration through backend failed with status {response.status}"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Agent service health check failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Agent service integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Agent Service Integration",
            description="Test agent service integration",
            status=status,
            message=message,
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_agent_communication_integration(self) -> None:
        """Test agent communication integration."""
        start_time = time.time()

        try:
            # Test agent communication endpoints
            async with self.session.get(self.agents_url + "/communications", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Agent communication integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Agent communication integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Agent communication integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Agent Communication Integration",
            description="Test agent communication integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_agent_lifecycle_integration(self) -> None:
        """Test agent lifecycle integration."""
        start_time = time.time()

        try:
            # Test agent lifecycle operations
            # Create agent
            create_response = await self.session.post(
                self.base_url + "/agent-management/agents",
                json={
                    "name": "Test Agent",
                    "agent_type": "CODE_IMPLEMENTER",
                    "model_tier": "SONNET"
                },
                timeout=10
            )

            if create_response.status == 200:
                agent_data = await create_response.json()
                agent_id = agent_data.get("id")

                if agent_id:
                    # Test agent state change
                    update_response = await self.session.patch(
                        self.base_url + f"/agent-management/agents/{agent_id}/state",
                        json={"state": "ACTIVE", "reason": "Test activation"},
                        timeout=10
                    )

                    if update_response.status == 200:
                        status = IntegrationStatus.PASS
                        message = "Agent lifecycle integration working correctly"
                    else:
                        status = IntegrationStatus.FAIL
                        message = f"Agent state change failed with status {update_response.status}"
                else:
                    status = IntegrationStatus.FAIL
                    message = "Created agent but no ID returned"
            else:
                status = IntegrationStatus.FAIL
                message = f"Agent creation failed with status {create_response.status}"

        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Agent lifecycle integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Agent Lifecycle Integration",
            description="Test agent lifecycle integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_mcp_service_integration(self) -> None:
        """Test MCP service integration."""
        start_time = time.time()

        try:
            # Test MCP service health
            async with self.session.get(self.mcp_url + "/health", timeout=10) as response:
                if response.status == 200:
                    # Test MCP tools through backend API
                    async with self.session.get(self.base_url + "/api/mcp/tools", timeout=10) as response:
                        if response.status == 200:
                            status = IntegrationStatus.PASS
                            message = "MCP service integration working correctly"
                        else:
                            status = IntegrationStatus.FAIL
                            message = f"MCP service integration through backend failed with status {response.status}"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"MCP service health check failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"MCP service integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="MCP Service Integration",
            description="Test MCP service integration",
            status=status,
            message=message,
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_mcp_tool_integration(self) -> None:
        """Test MCP tool integration."""
        start_time = time.time()

        try:
            # Test MCP tool execution
            async with self.session.post(
                self.mcp_url + "/tools/archon:perform_rag_query",
                json={"query": "test", "match_count": 5},
                timeout=10
            ) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "MCP tool integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"MCP tool execution failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"MCP tool integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="MCP Tool Integration",
            description="Test MCP tool integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_mcp_agent_integration(self) -> None:
        """Test MCP and agent service integration."""
        start_time = time.time()

        try:
            # Test MCP agent tools
            async with self.session.get(self.mcp_url + "/tools", timeout=10) as response:
                if response.status == 200:
                    tools = await response.json()
                    if isinstance(tools, list):
                        status = IntegrationStatus.PASS
                        message = f"MCP agent integration working ({len(tools)} tools available)"
                    else:
                        status = IntegrationStatus.FAIL
                        message = "MCP agent integration returning invalid data format"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"MCP agent integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"MCP agent integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="MCP Agent Integration",
            description="Test MCP and agent service integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_websocket_integration(self) -> None:
        """Test WebSocket integration for real-time features."""
        start_time = time.time()

        try:
            # Test WebSocket endpoint availability
            async with self.session.get(self.base_url + "/socket.io/", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "WebSocket integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"WebSocket integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"WebSocket integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="WebSocket Integration",
            description="Test WebSocket integration for real-time features",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_message_queue_integration(self) -> None:
        """Test message queue integration."""
        start_time = time.time()

        try:
            # Test message queue through agents service
            async with self.session.get(self.agents_url + "/messages", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Message queue integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Message queue integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Message queue integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Message Queue Integration",
            description="Test message queue integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_event_streaming_integration(self) -> None:
        """Test event streaming integration."""
        start_time = time.time()

        try:
            # Test event streaming
            async with self.session.get(self.base_url + "/events/stream", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Event streaming integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Event streaming integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Event streaming integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Event Streaming Integration",
            description="Test event streaming integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_knowledge_service_integration(self) -> None:
        """Test knowledge service integration."""
        start_time = time.time()

        try:
            # Test knowledge service through API
            async with self.session.get(self.base_url + "/api/knowledge/items", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Knowledge service integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Knowledge service integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Knowledge service integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Knowledge Service Integration",
            description="Test knowledge service integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_knowledge_search_integration(self) -> None:
        """Test knowledge search integration."""
        start_time = time.time()

        try:
            # Test knowledge search
            async with self.session.post(
                self.base_url + "/api/knowledge/search",
                json={"query": "test", "match_count": 5},
                timeout=10
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    if isinstance(results, dict):
                        status = IntegrationStatus.PASS
                        message = "Knowledge search integration working correctly"
                    else:
                        status = IntegrationStatus.FAIL
                        message = "Knowledge search integration returning invalid data format"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Knowledge search integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Knowledge search integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Knowledge Search Integration",
            description="Test knowledge search integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_knowledge_update_integration(self) -> None:
        """Test knowledge update integration."""
        start_time = time.time()

        try:
            # Test knowledge update
            async with self.session.post(
                self.base_url + "/api/knowledge/update",
                json={"id": "test", "content": "test content"},
                timeout=10
            ) as response:
                if response.status in [200, 201]:
                    status = IntegrationStatus.PASS
                    message = "Knowledge update integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Knowledge update integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Knowledge update integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Knowledge Update Integration",
            description="Test knowledge update integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_frontend_backend_integration(self) -> None:
        """Test frontend-backend integration."""
        start_time = time.time()

        try:
            # Test that frontend can communicate with backend
            async with self.session.get(self.frontend_url + "/api/health", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Frontend-backend integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Frontend-backend integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Frontend-backend integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Frontend-Backend Integration",
            description="Test frontend-backend integration",
            status=status,
            message=message,
            duration=duration,
            critical=True
        )
        self.test_results.append(test)

    async def _test_frontend_api_integration(self) -> None:
        """Test frontend API integration."""
        start_time = time.time()

        try:
            # Test frontend API integration
            async with self.session.get(self.frontend_url + "/api/agent-management/agents", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Frontend API integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Frontend API integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Frontend API integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Frontend API Integration",
            description="Test frontend API integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_frontend_realtime_integration(self) -> None:
        """Test frontend real-time integration."""
        start_time = time.time()

        try:
            # Test frontend real-time features
            async with self.session.get(self.frontend_url + "/socket.io/", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Frontend real-time integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Frontend real-time integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Frontend real-time integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Frontend Real-time Integration",
            description="Test frontend real-time integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_monitoring_integration(self) -> None:
        """Test monitoring integration."""
        start_time = time.time()

        try:
            # Test monitoring endpoints
            async with self.session.get(self.base_url + "/metrics", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Monitoring integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Monitoring integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Monitoring integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Monitoring Integration",
            description="Test monitoring integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_logging_integration(self) -> None:
        """Test logging integration."""
        start_time = time.time()

        try:
            # Test logging endpoints
            async with self.session.get(self.base_url + "/logs", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Logging integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Logging integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Logging integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Logging Integration",
            description="Test logging integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_metrics_integration(self) -> None:
        """Test metrics integration."""
        start_time = time.time()

        try:
            # Test metrics collection
            async with self.session.get(self.base_url + "/metrics", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Metrics integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Metrics integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Metrics integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Metrics Integration",
            description="Test metrics integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_authentication_flow_integration(self) -> None:
        """Test authentication flow integration."""
        start_time = time.time()

        try:
            # Test authentication flow
            async with self.session.get(self.base_url + "/auth/login", timeout=10) as response:
                if response.status in [200, 302]:
                    status = IntegrationStatus.PASS
                    message = "Authentication flow integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Authentication flow integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Authentication flow integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Authentication Flow Integration",
            description="Test authentication flow integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_authorization_flow_integration(self) -> None:
        """Test authorization flow integration."""
        start_time = time.time()

        try:
            # Test authorization flow
            async with self.session.get(self.base_url + "/auth/authorize", timeout=10) as response:
                if response.status in [200, 302]:
                    status = IntegrationStatus.PASS
                    message = "Authorization flow integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Authorization flow integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Authorization flow integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Authorization Flow Integration",
            description="Test authorization flow integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_security_policy_integration(self) -> None:
        """Test security policy integration."""
        start_time = time.time()

        try:
            # Test security policy endpoints
            async with self.session.get(self.base_url + "/security/policies", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Security policy integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Security policy integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Security policy integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Security Policy Integration",
            description="Test security policy integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_performance_integration(self) -> None:
        """Test performance integration."""
        start_time = time.time()

        try:
            # Test performance endpoints
            async with self.session.get(self.base_url + "/performance", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Performance integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Performance integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Performance integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Performance Integration",
            description="Test performance integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_scaling_integration(self) -> None
    """Test scaling integration."""
        start_time = time.time()

        try:
            # Test scaling endpoints
            async with self.session.get(self.base_url + "/scaling", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Scaling integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Scaling integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Scaling integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Scaling Integration",
            description="Test scaling integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _test_recovery_integration(self) -> None:
        """Test recovery integration."""
        start_time = time.time()

        try:
            # Test recovery endpoints
            async with self.session.get(self.base_url + "/recovery", timeout=10) as response:
                if response.status == 200:
                    status = IntegrationStatus.PASS
                    message = "Recovery integration working correctly"
                else:
                    status = IntegrationStatus.FAIL
                    message = f"Recovery integration failed with status {response.status}"
        except Exception as e:
            status = IntegrationStatus.FAIL
            message = f"Recovery integration error: {str(e)}"

        duration = time.time() - start_time
        test = IntegrationTest(
            name="Recovery Integration",
            description="Test recovery integration",
            status=status,
            message=message,
            duration=duration
        )
        self.test_results.append(test)

    async def _generate_integration_report(self) -> None:
        """Generate comprehensive integration report."""
        print(f"\n🔗 Integration Validation Report")
        print("=" * 60)
        print(f"Environment: {self.environment}")
        print(f"Total Tests: {len(self.test_results)}")

        # Count results by status
        status_counts = {}
        total_tests = len(self.test_results)

        for test in self.test_results:
            status_counts[test.status.value] = status_counts.get(test.status.value, 0) + 1

        print(f"Test Results: {status_counts}")

        # Print detailed results
        print("\nDetailed Results:")
        print("-" * 40)

        for test in self.test_results:
            status_icon = {
                IntegrationStatus.PASS: "✅",
                IntegrationStatus.FAIL: "❌",
                IntegrationStatus.WARNING: "⚠️",
                IntegrationStatus.PARTIAL: "🟡",
                IntegrationStatus.SKIP: "⏭️"
            }.get(test.status, "❓")

            critical_marker = "🔴 " if test.critical else ""
            print(f"{status_icon} {critical_marker}{test.name}: {test.message}")
            if test.duration > 0:
                print(f"   Duration: {test.duration:.2f}s")

        # Summary
        critical_failures = [t for t in self.test_results if t.status == IntegrationStatus.FAIL and t.critical]

        print("\n" + "=" * 60)
        if critical_failures:
            print("❌ CRITICAL INTEGRATION FAILURES DETECTED:")
            for failure in critical_failures:
                print(f"   - {failure.name}: {failure.message}")
            print("\n   These issues must be addressed before deployment.")
        else:
            print("✅ No critical integration failures detected.")

        # Integration recommendations
        print("\n💡 Integration Recommendations:")
        print("   1. Address all critical integration failures")
        print("   2. Monitor partial integrations for potential issues")
        print("   3. Regular integration testing to prevent regressions")
        print("   4. Implement integration monitoring and alerting")
        print("   5. Document integration points and dependencies")
        print("   6. Consider integration testing in CI/CD pipeline")

        print("=" * 60)

        # Save report to file
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": self.environment,
            "total_tests": total_tests,
            "status_counts": status_counts,
            "tests": [
                {
                    "name": test.name,
                    "description": test.description,
                    "status": test.status.value,
                    "message": test.message,
                    "details": test.details,
                    "duration": test.duration,
                    "critical": test.critical
                }
                for test in self.test_results
            ]
        }

        report_file = f"integration_report_{self.environment}_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"📄 Detailed report saved to: {report_file}")


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Agency Swarm Integration Validation")
    parser.add_argument("--environment", default="production", help="Environment to validate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    validator = IntegrationValidator(args.environment)
    results = await validator.run_all_validations()

    # Exit with appropriate code based on critical failures
    critical_failures = [t for t in results if t.status == IntegrationStatus.FAIL and t.critical]
    sys.exit(1 if critical_failures else 0)


if __name__ == "__main__":
    asyncio.run(main())