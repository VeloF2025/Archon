#!/usr/bin/env python3
"""
Integration Validation for Agency Swarm Components
Validates all integration points between Phase 1-3 systems
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pytest
from pathlib import Path

logger = logging.getLogger(__name__)

class IntegrationValidator:
    """Validates integration points between Agency Swarm components"""

    def __init__(self):
        self.services = {
            "frontend": "http://localhost:3737",
            "api": "http://localhost:8181",
            "mcp": "http://localhost:8051",
            "agents": "http://localhost:8052"
        }
        self.integration_results = []

    async def validate_service_connectivity(self):
        """Validate connectivity between all services"""
        logger.info("Validating service connectivity...")

        connectivity_matrix = {}
        for service_name, service_url in self.services.items():
            connectivity_matrix[service_name] = {}

            for target_name, target_url in self.services.items():
                if service_name != target_name:
                    try:
                        # Test if service can reach target
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{service_url}/health") as response:
                                connectivity_matrix[service_name][target_name] = {
                                    "status": "connected" if response.status == 200 else "failed",
                                    "response_time": response.headers.get("X-Response-Time", "N/A")
                                }
                    except Exception as e:
                        connectivity_matrix[service_name][target_name] = {
                            "status": "failed",
                            "error": str(e)
                        }

        return {
            "test_name": "Service Connectivity",
            "status": "passed" if all(
                all(conn["status"] == "connected" for conn in service.values())
                for service in connectivity_matrix.values()
            ) else "failed",
            "connectivity_matrix": connectivity_matrix
        }

    async def validate_database_integration(self):
        """Validate database connectivity and consistency across services"""
        logger.info("Validating database integration...")

        # Test database access from each service
        db_tests = []
        for service_name, service_url in self.services.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service_url}/api/db/health") as response:
                        data = await response.json()
                        db_tests.append({
                            "service": service_name,
                            "status": "connected" if response.status == 200 else "failed",
                            "details": data
                        })
            except Exception as e:
                db_tests.append({
                    "service": service_name,
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "test_name": "Database Integration",
            "status": "passed" if all(test["status"] == "connected" for test in db_tests) else "failed",
            "db_tests": db_tests
        }

    async def validate_knowledge_to_agent_flow(self):
        """Validate knowledge flow from Phase 1 to Phase 3 agents"""
        logger.info("Validating knowledge-to-agent integration flow...")

        # 1. Create knowledge item
        knowledge_data = {
            "title": "Integration Test Knowledge",
            "content": "Test content for integration validation",
            "type": "document"
        }

        async with aiohttp.ClientSession() as session:
            # Create knowledge
            async with session.post(f"{self.services['api']}/api/knowledge/upload", json=knowledge_data) as response:
                knowledge_result = await response.json()
                knowledge_id = knowledge_result.get("id")

            # 2. Access knowledge from agents service
            async with session.post(f"{self.services['agents']}/agents/knowledge_query", json={
                "knowledge_id": knowledge_id,
                "query": "test integration"
            }) as response:
                agent_result = await response.json()

            # 3. Verify consistency
            success = (
                response.status == 200 and
                agent_result.get("knowledge_found") and
                agent_result.get("knowledge_id") == knowledge_id
            )

            return {
                "test_name": "Knowledge-to-Agent Flow",
                "status": "passed" if success else "failed",
                "knowledge_id": knowledge_id,
                "agent_response": agent_result
            }

    async def validate_template_to_agent_creation(self):
        """Validate template usage in agent creation (Phase 2 to Phase 3)"""
        logger.info("Validating template-to-agent creation flow...")

        # 1. Create template
        template_data = {
            "name": "Integration Test Template",
            "description": "Template for integration testing",
            "type": "agent_template",
            "config": {
                "model": "gpt-4",
                "capabilities": ["knowledge_processing", "collaboration"]
            }
        }

        async with aiohttp.ClientSession() as session:
            # Create template
            async with session.post(f"{self.services['api']}/api/templates", json=template_data) as response:
                template_result = await response.json()
                template_id = template_result.get("id")

            # 2. Create agent from template
            async with session.post(f"{self.services['agents']}/agents/create_from_template", json={
                "template_id": template_id,
                "name": "Integration Test Agent"
            }) as response:
                agent_result = await response.json()

            # 3. Validate agent has template properties
            success = (
                response.status == 200 and
                agent_result.get("template_id") == template_id and
                agent_result.get("config", {}).get("model") == "gpt-4"
            )

            return {
                "test_name": "Template-to-Agent Creation",
                "status": "passed" if success else "failed",
                "template_id": template_id,
                "agent_id": agent_result.get("id"),
                "agent_result": agent_result
            }

    async def validate_pattern_to_workflow_integration(self):
        """Validate pattern usage in workflows (Phase 3 patterns to workflows)"""
        logger.info("Validating pattern-to-workflow integration...")

        # 1. Create pattern
        pattern_data = {
            "name": "Integration Test Pattern",
            "description": "Pattern for testing integration",
            "category": "agent_communication",
            "steps": [
                {"action": "initialize", "parameters": {}},
                {"action": "process", "parameters": {}},
                {"action": "finalize", "parameters": {}}
            ]
        }

        async with aiohttp.ClientSession() as session:
            # Create pattern
            async with session.post(f"{self.services['api']}/api/patterns", json=pattern_data) as response:
                pattern_result = await response.json()
                pattern_id = pattern_result.get("id")

            # 2. Create workflow using pattern
            async with session.post(f"{self.services['agents']}/api/workflows", json={
                "pattern_id": pattern_id,
                "name": "Integration Test Workflow",
                "agents": ["agent1", "agent2"]
            }) as response:
                workflow_result = await response.json()

            # 3. Validate workflow structure
            success = (
                response.status == 200 and
                workflow_result.get("pattern_id") == pattern_id and
                len(workflow_result.get("steps", [])) == 3
            )

            return {
                "test_name": "Pattern-to-Workflow Integration",
                "status": "passed" if success else "failed",
                "pattern_id": pattern_id,
                "workflow_id": workflow_result.get("id"),
                "workflow_result": workflow_result
            }

    async def validate_mcp_tool_access(self):
        """Validate MCP tool access across all services"""
        logger.info("Validating MCP tool access integration...")

        async with aiohttp.ClientSession() as session:
            # Get available tools from MCP
            async with session.get(f"{self.services['mcp']}/tools") as response:
                tools_response = await response.json()
                available_tools = tools_response.get("tools", [])

            # Test tool access from different services
            tool_access_results = []
            for tool in available_tools[:3]:  # Test first 3 tools
                for service_name, service_url in self.services.items():
                    if service_name != "mcp":  # Skip MCP itself
                        try:
                            async with session.post(f"{service_url}/api/mcp/execute", json={
                                "tool_name": tool["name"],
                                "parameters": tool.get("parameters", {})
                            }) as response:
                                tool_result = await response.json()
                                tool_access_results.append({
                                    "tool": tool["name"],
                                    "service": service_name,
                                    "status": "accessible" if response.status == 200 else "failed",
                                    "result": tool_result
                                })
                        except Exception as e:
                            tool_access_results.append({
                                "tool": tool["name"],
                                "service": service_name,
                                "status": "failed",
                                "error": str(e)
                            })

            return {
                "test_name": "MCP Tool Access Integration",
                "status": "passed" if all(r["status"] == "accessible" for r in tool_access_results) else "failed",
                "available_tools": available_tools,
                "tool_access_results": tool_access_results
            }

    async def validate_realtime_updates(self):
        """Validate real-time update propagation across services"""
        logger.info("Validating real-time update propagation...")

        try:
            import websockets

            # Test WebSocket connections to all services
            websocket_results = []
            for service_name, service_url in self.services.items():
                try:
                    ws_url = service_url.replace("http", "ws") + "/ws"
                    async with websockets.connect(ws_url) as websocket:
                        # Send test message
                        await websocket.send(json.dumps({
                            "type": "test",
                            "data": {"service": service_name}
                        }))

                        # Wait for response
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        response_data = json.loads(response)

                        websocket_results.append({
                            "service": service_name,
                            "status": "connected" if response_data.get("status") == "ok" else "failed",
                            "response": response_data
                        })
                except Exception as e:
                    websocket_results.append({
                        "service": service_name,
                        "status": "failed",
                        "error": str(e)
                    })

            return {
                "test_name": "Real-time Update Propagation",
                "status": "passed" if all(r["status"] == "connected" for r in websocket_results) else "failed",
                "websocket_results": websocket_results
            }

        except ImportError:
            return {
                "test_name": "Real-time Update Propagation",
                "status": "skipped",
                "reason": "websockets library not available"
            }

    async def validate_data_consistency(self):
        """Validate data consistency across all services"""
        logger.info("Validating data consistency across services...")

        # Create test data
        test_data = {
            "id": "integration_test_data",
            "content": "Test data for consistency validation",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"test": True, "integration": True}
        }

        # Store data in each service and verify consistency
        consistency_results = []
        async with aiohttp.ClientSession() as session:
            for service_name, service_url in self.services.items():
                try:
                    # Store data
                    async with session.post(f"{service_url}/api/test_data", json=test_data) as response:
                        store_result = await response.json()

                    # Retrieve data
                    async with session.get(f"{service_url}/api/test_data/{test_data['id']}") as response:
                        retrieve_result = await response.json()
                        retrieved_data = retrieve_result.get("data", {})

                    # Check consistency
                    is_consistent = (
                        retrieved_data.get("id") == test_data["id"] and
                        retrieved_data.get("content") == test_data["content"] and
                        retrieved_data.get("metadata") == test_data["metadata"]
                    )

                    consistency_results.append({
                        "service": service_name,
                        "status": "consistent" if is_consistent else "inconsistent",
                        "store_result": store_result,
                        "retrieve_result": retrieve_result
                    })

                except Exception as e:
                    consistency_results.append({
                        "service": service_name,
                        "status": "failed",
                        "error": str(e)
                    })

        return {
            "test_name": "Data Consistency",
            "status": "passed" if all(r["status"] == "consistent" for r in consistency_results) else "failed",
            "consistency_results": consistency_results
        }

    async def run_all_integration_tests(self):
        """Run all integration validation tests"""
        logger.info("Running all integration validation tests...")

        test_functions = [
            self.validate_service_connectivity,
            self.validate_database_integration,
            self.validate_knowledge_to_agent_flow,
            self.validate_template_to_agent_creation,
            self.validate_pattern_to_workflow_integration,
            self.validate_mcp_tool_access,
            self.validate_realtime_updates,
            self.validate_data_consistency
        ]

        for test_func in test_functions:
            try:
                result = await test_func()
                self.integration_results.append(result)
                logger.info(f"✓ {test_func.__name__}: {result['status']}")
            except Exception as e:
                logger.error(f"✗ {test_func.__name__} failed: {e}")
                self.integration_results.append({
                    "test_name": test_func.__name__,
                    "status": "failed",
                    "error": str(e)
                })

        return self.integration_results

    def generate_integration_report(self):
        """Generate integration validation report"""
        total_tests = len(self.integration_results)
        passed_tests = sum(1 for r in self.integration_results if r["status"] == "passed")

        report = {
            "test_suite": "Agency Swarm Integration Validation",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            "detailed_results": self.integration_results,
            "integration_status": "healthy" if passed_tests == total_tests else "issues_detected"
        }

        # Save report
        report_path = Path("agency_swarm_integration_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Integration report saved to {report_path}")
        return report

async def main():
    """Main function to run integration validation"""
    validator = IntegrationValidator()
    await validator.run_all_integration_tests()
    return validator.generate_integration_report()

if __name__ == "__main__":
    asyncio.run(main())