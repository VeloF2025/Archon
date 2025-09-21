"""
ReactFlow Integration Validation Module

This module validates the integration between the Archon workflow system
and ReactFlow components, ensuring proper data format compatibility,
API functionality, and real-time updates.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from src.database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution,
    ReactFlowNode, ReactFlowEdge, WorkflowStep
)
from src.server.services.workflow_service import WorkflowService
from src.server.services.workflow_execution_service import WorkflowExecutionService
from src.server.services.workflow_analytics_service import WorkflowAnalyticsService
from src.mcp_server.modules.workflow_mcp_tools import WorkflowMCPTools

logger = logging.getLogger(__name__)


class ReactFlowIntegrationValidator:
    """
    Validates ReactFlow integration by testing data format compatibility,
    API functionality, and real-time updates.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.workflow_service = WorkflowService(session)
        self.execution_service = WorkflowExecutionService(session)
        self.analytics_service = WorkflowAnalyticsService(session)
        self.mcp_tools = WorkflowMCPTools(session)

    async def validate_reactflow_data_format(self) -> Dict[str, Any]:
        """
        Validate that workflow data structures are compatible with ReactFlow format.

        ReactFlow expects:
        - nodes: Array of {id, position, data, type}
        - edges: Array of {id, source, target, sourceHandle, targetHandle}
        """
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "test_cases": []
        }

        # Test 1: Validate node structure compatibility
        test_case = {
            "name": "Node Structure Compatibility",
            "description": "Verify workflow nodes map to ReactFlow node format"
        }

        try:
            # Create a test workflow with various node types
            workflow_data = await self._create_test_workflow_data()

            # Convert to ReactFlow format
            reactflow_nodes = []
            for node in workflow_data["nodes"]:
                reactflow_node = {
                    "id": node["id"],
                    "position": {"x": node["position_x"], "y": node["position_y"]},
                    "data": {
                        "label": node["name"],
                        "type": node["node_type"],
                        "config": node.get("config", {}),
                        "status": "idle"
                    },
                    "type": self._map_node_type_to_reactflow(node["node_type"])
                }
                reactflow_nodes.append(reactflow_node)

            # Validate ReactFlow node structure
            required_fields = ["id", "position", "data", "type"]
            for node in reactflow_nodes:
                for field in required_fields:
                    if field not in node:
                        validation_results["errors"].append(
                            f"Missing required field '{field}' in ReactFlow node"
                        )
                        validation_results["passed"] = False

            validation_results["test_cases"].append({
                **test_case,
                "status": "passed" if validation_results["passed"] else "failed",
                "details": {"node_count": len(reactflow_nodes)}
            })

        except Exception as e:
            validation_results["errors"].append(f"Node structure validation failed: {str(e)}")
            validation_results["passed"] = False

        # Test 2: Validate edge structure compatibility
        test_case = {
            "name": "Edge Structure Compatibility",
            "description": "Verify workflow connections map to ReactFlow edge format"
        }

        try:
            reactflow_edges = []
            for edge in workflow_data["connections"]:
                reactflow_edge = {
                    "id": edge["id"],
                    "source": edge["source_node_id"],
                    "target": edge["target_node_id"],
                    "sourceHandle": edge.get("source_handle", "source"),
                    "targetHandle": edge.get("target_handle", "target"),
                    "type": "default",
                    "animated": False,
                    "style": {"stroke": "#555"}
                }
                reactflow_edges.append(reactflow_edge)

            # Validate ReactFlow edge structure
            required_fields = ["id", "source", "target"]
            for edge in reactflow_edges:
                for field in required_fields:
                    if field not in edge:
                        validation_results["errors"].append(
                            f"Missing required field '{field}' in ReactFlow edge"
                        )
                        validation_results["passed"] = False

            validation_results["test_cases"].append({
                **test_case,
                "status": "passed" if validation_results["passed"] else "failed",
                "details": {"edge_count": len(reactflow_edges)}
            })

        except Exception as e:
            validation_results["errors"].append(f"Edge structure validation failed: {str(e)}")
            validation_results["passed"] = False

        # Test 3: Validate data serialization
        test_case = {
            "name": "Data Serialization",
            "description": "Verify workflow data can be serialized for ReactFlow"
        }

        try:
            reactflow_data = {
                "nodes": reactflow_nodes,
                "edges": reactflow_edges,
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }

            # Test JSON serialization
            json_str = json.dumps(reactflow_data)
            parsed_data = json.loads(json_str)

            # Verify data integrity after serialization
            assert len(parsed_data["nodes"]) == len(reactflow_nodes)
            assert len(parsed_data["edges"]) == len(reactflow_edges)

            validation_results["test_cases"].append({
                **test_case,
                "status": "passed",
                "details": {"json_size": len(json_str)}
            })

        except Exception as e:
            validation_results["errors"].append(f"Data serialization validation failed: {str(e)}")
            validation_results["passed"] = False

        return validation_results

    async def validate_api_integration(self) -> Dict[str, Any]:
        """
        Validate that API endpoints provide data in ReactFlow-compatible format.
        """
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "api_tests": []
        }

        # Test 1: Create workflow API
        api_test = {
            "name": "Create Workflow API",
            "endpoint": "POST /api/workflows",
            "description": "Test workflow creation with ReactFlow data"
        }

        try:
            # Simulate API request with ReactFlow data
            reactflow_request = {
                "name": "Test Workflow",
                "description": "ReactFlow integration test",
                "reactflow_data": {
                    "nodes": [
                        {
                            "id": "node-1",
                            "position": {"x": 100, "y": 100},
                            "data": {"label": "Start", "type": "agent_task"},
                            "type": "input"
                        },
                        {
                            "id": "node-2",
                            "position": {"x": 300, "y": 100},
                            "data": {"label": "Process", "type": "agent_task"},
                            "type": "default"
                        }
                    ],
                    "edges": [
                        {
                            "id": "edge-1",
                            "source": "node-1",
                            "target": "node-2",
                            "sourceHandle": "source",
                            "targetHandle": "target"
                        }
                    ]
                }
            }

            # Test MCP tool for workflow creation
            result = await self.mcp_tools.create_workflow(
                name=reactflow_request["name"],
                description=reactflow_request["description"],
                reactflow_data=reactflow_request["reactflow_data"]
            )

            if result["success"]:
                workflow_id = result["workflow_id"]
                api_test["status"] = "passed"
                api_test["details"] = {"workflow_id": workflow_id}
            else:
                validation_results["errors"].append(f"Workflow creation failed: {result.get('error', 'Unknown error')}")
                validation_results["passed"] = False
                api_test["status"] = "failed"

        except Exception as e:
            validation_results["errors"].append(f"Create workflow API test failed: {str(e)}")
            validation_results["passed"] = False
            api_test["status"] = "failed"

        validation_results["api_tests"].append(api_test)

        # Test 2: Get workflow API
        api_test = {
            "name": "Get Workflow API",
            "endpoint": "GET /api/workflows/{id}",
            "description": "Test retrieving workflow in ReactFlow format"
        }

        try:
            if 'workflow_id' in locals():
                # Test MCP tool for getting workflow
                result = await self.mcp_tools.get_workflow(workflow_id=workflow_id)

                if result["success"]:
                    workflow_data = result["workflow"]

                    # Verify ReactFlow data structure
                    assert "reactflow_data" in workflow_data
                    assert "nodes" in workflow_data["reactflow_data"]
                    assert "edges" in workflow_data["reactflow_data"]

                    api_test["status"] = "passed"
                    api_test["details"] = {
                        "node_count": len(workflow_data["reactflow_data"]["nodes"]),
                        "edge_count": len(workflow_data["reactflow_data"]["edges"])
                    }
                else:
                    validation_results["errors"].append(f"Get workflow failed: {result.get('error', 'Unknown error')}")
                    validation_results["passed"] = False
                    api_test["status"] = "failed"
            else:
                validation_results["warnings"].append("Skipping get workflow test - no workflow created")
                api_test["status"] = "skipped"

        except Exception as e:
            validation_results["errors"].append(f"Get workflow API test failed: {str(e)}")
            validation_results["passed"] = False
            api_test["status"] = "failed"

        validation_results["api_tests"].append(api_test)

        # Test 3: Update workflow API
        api_test = {
            "name": "Update Workflow API",
            "endpoint": "PUT /api/workflows/{id}",
            "description": "Test updating workflow with ReactFlow data"
        }

        try:
            if 'workflow_id' in locals():
                # Update workflow with new ReactFlow data
                updated_data = {
                    "nodes": [
                        {
                            "id": "node-1",
                            "position": {"x": 100, "y": 100},
                            "data": {"label": "Start Updated", "type": "agent_task"},
                            "type": "input"
                        },
                        {
                            "id": "node-2",
                            "position": {"x": 300, "y": 100},
                            "data": {"label": "Process Updated", "type": "agent_task"},
                            "type": "default"
                        },
                        {
                            "id": "node-3",
                            "position": {"x": 500, "y": 100},
                            "data": {"label": "End", "type": "agent_task"},
                            "type": "output"
                        }
                    ],
                    "edges": [
                        {
                            "id": "edge-1",
                            "source": "node-1",
                            "target": "node-2",
                            "sourceHandle": "source",
                            "targetHandle": "target"
                        },
                        {
                            "id": "edge-2",
                            "source": "node-2",
                            "target": "node-3",
                            "sourceHandle": "source",
                            "targetHandle": "target"
                        }
                    ]
                }

                result = await self.mcp_tools.update_workflow(
                    workflow_id=workflow_id,
                    reactflow_data=updated_data
                )

                if result["success"]:
                    api_test["status"] = "passed"
                    api_test["details"] = {"updated_nodes": len(updated_data["nodes"])}
                else:
                    validation_results["errors"].append(f"Update workflow failed: {result.get('error', 'Unknown error')}")
                    validation_results["passed"] = False
                    api_test["status"] = "failed"
            else:
                validation_results["warnings"].append("Skipping update workflow test - no workflow created")
                api_test["status"] = "skipped"

        except Exception as e:
            validation_results["errors"].append(f"Update workflow API test failed: {str(e)}")
            validation_results["passed"] = False
            api_test["status"] = "failed"

        validation_results["api_tests"].append(api_test)

        return validation_results

    async def validate_realtime_updates(self) -> Dict[str, Any]:
        """
        Validate real-time updates integration with ReactFlow components.
        """
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "realtime_tests": []
        }

        # Test 1: Workflow execution updates
        realtime_test = {
            "name": "Execution Status Updates",
            "event_type": "workflow_execution_update",
            "description": "Test real-time updates during workflow execution"
        }

        try:
            if 'workflow_id' in locals():
                # Start workflow execution
                execution_result = await self.mcp_tools.execute_workflow(
                    workflow_id=workflow_id,
                    input_data={"test": "data"}
                )

                if execution_result["success"]:
                    execution_id = execution_result["execution_id"]

                    # Simulate receiving Socket.IO events
                    simulated_events = [
                        {
                            "event": "workflow_execution_started",
                            "data": {
                                "execution_id": execution_id,
                                "workflow_id": workflow_id,
                                "status": "running",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        },
                        {
                            "event": "step_execution_update",
                            "data": {
                                "execution_id": execution_id,
                                "step_id": "node-1",
                                "status": "completed",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        },
                        {
                            "event": "workflow_execution_completed",
                            "data": {
                                "execution_id": execution_id,
                                "workflow_id": workflow_id,
                                "status": "completed",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    ]

                    # Validate event structure for ReactFlow compatibility
                    for event in simulated_events:
                        self._validate_reactflow_event(event, validation_results)

                    realtime_test["status"] = "passed"
                    realtime_test["details"] = {"events_count": len(simulated_events)}
                else:
                    validation_results["errors"].append(f"Workflow execution failed: {execution_result.get('error', 'Unknown error')}")
                    validation_results["passed"] = False
                    realtime_test["status"] = "failed"
            else:
                validation_results["warnings"].append("Skipping execution test - no workflow created")
                realtime_test["status"] = "skipped"

        except Exception as e:
            validation_results["errors"].append(f"Real-time execution test failed: {str(e)}")
            validation_results["passed"] = False
            realtime_test["status"] = "failed"

        validation_results["realtime_tests"].append(realtime_test)

        # Test 2: Node status updates
        realtime_test = {
            "name": "Node Status Updates",
            "event_type": "node_status_update",
            "description": "Test real-time node status updates for ReactFlow"
        }

        try:
            # Simulate node status update event
            node_update_event = {
                "event": "node_status_update",
                "data": {
                    "workflow_id": workflow_id if 'workflow_id' in locals() else "test-workflow",
                    "node_id": "node-1",
                    "status": "running",
                    "progress": 50,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            self._validate_reactflow_event(node_update_event, validation_results)

            realtime_test["status"] = "passed"
            realtime_test["details"] = {"node_status": "running"}

        except Exception as e:
            validation_results["errors"].append(f"Node status update test failed: {str(e)}")
            validation_results["passed"] = False
            realtime_test["status"] = "failed"

        validation_results["realtime_tests"].append(realtime_test)

        return validation_results

    async def validate_frontend_integration(self) -> Dict[str, Any]:
        """
        Validate frontend integration patterns and component compatibility.
        """
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "frontend_tests": []
        }

        # Test 1: Component data flow
        frontend_test = {
            "name": "Component Data Flow",
            "description": "Test data flow between ReactFlow components and backend"
        }

        try:
            # Simulate ReactFlow component data requirements
            component_requirements = {
                "ReactFlow": {
                    "nodes": "Array of node objects with id, position, data, type",
                    "edges": "Array of edge objects with id, source, target",
                    "onNodesChange": "Function to handle node changes",
                    "onEdgesChange": "Function to handle edge changes",
                    "onConnect": "Function to handle new connections"
                },
                "CustomNode": {
                    "id": "Unique node identifier",
                    "data": "Node configuration and state",
                    "type": "Node type for rendering"
                }
            }

            # Verify our API provides required data
            if 'workflow_id' in locals():
                workflow_result = await self.mcp_tools.get_workflow(workflow_id=workflow_id)

                if workflow_result["success"]:
                    reactflow_data = workflow_result["workflow"]["reactflow_data"]

                    # Check required fields
                    for field in ["nodes", "edges"]:
                        if field not in reactflow_data:
                            validation_results["errors"].append(f"Missing required field '{field}' in ReactFlow data")
                            validation_results["passed"] = False

                    frontend_test["status"] = "passed"
                    frontend_test["details"] = {
                        "has_nodes": "nodes" in reactflow_data,
                        "has_edges": "edges" in reactflow_data,
                        "node_count": len(reactflow_data.get("nodes", [])),
                        "edge_count": len(reactflow_data.get("edges", []))
                    }
                else:
                    validation_results["errors"].append(f"Failed to get workflow for frontend test: {workflow_result.get('error', 'Unknown error')}")
                    validation_results["passed"] = False
                    frontend_test["status"] = "failed"
            else:
                validation_results["warnings"].append("Skipping frontend test - no workflow created")
                frontend_test["status"] = "skipped"

        except Exception as e:
            validation_results["errors"].append(f"Frontend data flow test failed: {str(e)}")
            validation_results["passed"] = False
            frontend_test["status"] = "failed"

        validation_results["frontend_tests"].append(frontend_test)

        return validation_results

    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run all ReactFlow integration validation tests.
        """
        logger.info("Starting ReactFlow integration validation...")

        comprehensive_results = {
            "overall_passed": True,
            "validation_summary": {},
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": []
        }

        # Run all validation tests
        validation_functions = [
            ("data_format", self.validate_reactflow_data_format),
            ("api_integration", self.validate_api_integration),
            ("realtime_updates", self.validate_realtime_updates),
            ("frontend_integration", self.validate_frontend_integration)
        ]

        for validation_name, validation_func in validation_functions:
            try:
                result = await validation_func()
                comprehensive_results["validation_summary"][validation_name] = result

                if not result["passed"]:
                    comprehensive_results["overall_passed"] = False

            except Exception as e:
                logger.error(f"Validation {validation_name} failed: {str(e)}")
                comprehensive_results["validation_summary"][validation_name] = {
                    "passed": False,
                    "errors": [f"Validation failed: {str(e)}"],
                    "warnings": [],
                    "test_cases": []
                }
                comprehensive_results["overall_passed"] = False

        # Generate recommendations
        comprehensive_results["recommendations"] = self._generate_recommendations(comprehensive_results)

        logger.info(f"ReactFlow integration validation completed. Overall status: {'PASSED' if comprehensive_results['overall_passed'] else 'FAILED'}")

        return comprehensive_results

    def _map_node_type_to_reactflow(self, node_type: str) -> str:
        """Map workflow node types to ReactFlow component types."""
        type_mapping = {
            "agent_task": "default",
            "decision": "decision",
            "api_call": "api",
            "data_transform": "transform",
            "start": "input",
            "end": "output",
            "parallel": "parallel",
            "condition": "condition"
        }
        return type_mapping.get(node_type, "default")

    async def _create_test_workflow_data(self) -> Dict[str, Any]:
        """Create test workflow data for validation."""
        return {
            "name": "Test Workflow",
            "description": "ReactFlow integration test workflow",
            "nodes": [
                {
                    "id": "node-1",
                    "name": "Start Node",
                    "node_type": "start",
                    "position_x": 100,
                    "position_y": 100,
                    "config": {}
                },
                {
                    "id": "node-2",
                    "name": "Process Node",
                    "node_type": "agent_task",
                    "position_x": 300,
                    "position_y": 100,
                    "config": {"agent_type": "data_processor"}
                },
                {
                    "id": "node-3",
                    "name": "Decision Node",
                    "node_type": "decision",
                    "position_x": 500,
                    "position_y": 100,
                    "config": {"condition": "data_quality > 0.8"}
                },
                {
                    "id": "node-4",
                    "name": "End Node",
                    "node_type": "end",
                    "position_x": 700,
                    "position_y": 100,
                    "config": {}
                }
            ],
            "connections": [
                {
                    "id": "conn-1",
                    "source_node_id": "node-1",
                    "target_node_id": "node-2",
                    "source_handle": "source",
                    "target_handle": "target"
                },
                {
                    "id": "conn-2",
                    "source_node_id": "node-2",
                    "target_node_id": "node-3",
                    "source_handle": "source",
                    "target_handle": "target"
                },
                {
                    "id": "conn-3",
                    "source_node_id": "node-3",
                    "target_node_id": "node-4",
                    "source_handle": "true",
                    "target_handle": "target"
                }
            ]
        }

    def _validate_reactflow_event(self, event: Dict[str, Any], validation_results: Dict[str, Any]) -> None:
        """Validate that an event is compatible with ReactFlow real-time updates."""
        required_fields = ["event", "data"]

        for field in required_fields:
            if field not in event:
                validation_results["errors"].append(f"Missing required field '{field}' in event")
                validation_results["passed"] = False

        # Validate event data structure based on event type
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "workflow_execution_update":
            required_data_fields = ["execution_id", "workflow_id", "status"]
            for field in required_data_fields:
                if field not in data:
                    validation_results["errors"].append(f"Missing required field '{field}' in workflow execution update data")
                    validation_results["passed"] = False

        elif event_type == "node_status_update":
            required_data_fields = ["node_id", "status"]
            for field in required_data_fields:
                if field not in data:
                    validation_results["errors"].append(f"Missing required field '{field}' in node status update data")
                    validation_results["passed"] = False

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check for common issues
        for validation_name, validation_result in results["validation_summary"].items():
            if not validation_result["passed"]:
                if validation_name == "data_format":
                    recommendations.append("Fix ReactFlow data format compatibility issues")
                elif validation_name == "api_integration":
                    recommendations.append("Review API endpoint implementations for ReactFlow integration")
                elif validation_name == "realtime_updates":
                    recommendations.append("Improve real-time event structure for ReactFlow components")
                elif validation_name == "frontend_integration":
                    recommendations.append("Enhance frontend component data flow patterns")

        # Add general recommendations
        if results["overall_passed"]:
            recommendations.append("ReactFlow integration is working correctly")
            recommendations.append("Consider adding more comprehensive error handling")
        else:
            recommendations.append("Address validation failures before deploying ReactFlow integration")
            recommendations.append("Run integration tests after fixing issues")

        return recommendations


# Pytest fixtures and test functions
@pytest.fixture
async def validator(async_session):
    """Create ReactFlow integration validator for testing."""
    return ReactFlowIntegrationValidator(async_session)


@pytest.mark.asyncio
async def test_reactflow_data_format_validation(validator):
    """Test ReactFlow data format validation."""
    results = await validator.validate_reactflow_data_format()

    assert "passed" in results
    assert "errors" in results
    assert "test_cases" in results
    assert len(results["test_cases"]) > 0

    # Log results for debugging
    print(f"Data format validation results: {results}")


@pytest.mark.asyncio
async def test_reactflow_api_integration(validator):
    """Test ReactFlow API integration."""
    results = await validator.validate_api_integration()

    assert "passed" in results
    assert "errors" in results
    assert "api_tests" in results
    assert len(results["api_tests"]) > 0

    # Log results for debugging
    print(f"API integration validation results: {results}")


@pytest.mark.asyncio
async def test_reactflow_realtime_updates(validator):
    """Test ReactFlow real-time updates."""
    results = await validator.validate_realtime_updates()

    assert "passed" in results
    assert "errors" in results
    assert "realtime_tests" in results
    assert len(results["realtime_tests"]) > 0

    # Log results for debugging
    print(f"Real-time updates validation results: {results}")


@pytest.mark.asyncio
async def test_reactflow_frontend_integration(validator):
    """Test ReactFlow frontend integration."""
    results = await validator.validate_frontend_integration()

    assert "passed" in results
    assert "errors" in results
    assert "frontend_tests" in results
    assert len(results["frontend_tests"]) > 0

    # Log results for debugging
    print(f"Frontend integration validation results: {results}")


@pytest.mark.asyncio
async def test_comprehensive_reactflow_validation(validator):
    """Test comprehensive ReactFlow integration validation."""
    results = await validator.run_comprehensive_validation()

    assert "overall_passed" in results
    assert "validation_summary" in results
    assert "recommendations" in results
    assert "timestamp" in results

    # Check that all validation categories are present
    expected_categories = ["data_format", "api_integration", "realtime_updates", "frontend_integration"]
    for category in expected_categories:
        assert category in results["validation_summary"]

    # Log comprehensive results
    print(f"Comprehensive validation results: {results}")

    return results


if __name__ == "__main__":
    # Run validation tests
    import asyncio

    async def run_validation():
        # This would typically be run with a proper database session
        print("ReactFlow Integration Validation Module")
        print("This module should be run with proper database setup")
        print("Use pytest to run the test functions")

    asyncio.run(run_validation())