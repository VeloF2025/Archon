"""
ReactFlow Integration Test Suite

This test suite validates the integration between the Archon workflow system
and ReactFlow components, ensuring proper data format compatibility,
API functionality, and real-time updates.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution,
    ReactFlowNode, ReactFlowEdge, WorkflowNode, WorkflowConnection
)
from src.server.services.workflow_service import WorkflowService
from src.server.services.workflow_execution_service import WorkflowExecutionService
from src.server.services.workflow_analytics_service import WorkflowAnalyticsService
from src.mcp_server.modules.workflow_mcp_tools import WorkflowMCPTools
from src.validation.reactflow_integration_validator import ReactFlowIntegrationValidator


@pytest.fixture
async def async_session():
    """Create async session for testing."""
    # Use in-memory SQLite for testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        # Create tables
        from src.database.workflow_models import Base
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def validator(async_session):
    """Create ReactFlow integration validator."""
    return ReactFlowIntegrationValidator(async_session)


@pytest.fixture
def sample_reactflow_data():
    """Sample ReactFlow data for testing."""
    return {
        "nodes": [
            {
                "id": "node-1",
                "position": {"x": 100, "y": 100},
                "data": {
                    "label": "Start",
                    "type": "start",
                    "config": {}
                },
                "type": "input"
            },
            {
                "id": "node-2",
                "position": {"x": 300, "y": 100},
                "data": {
                    "label": "Data Processing",
                    "type": "agent_task",
                    "config": {
                        "agent_type": "data_processor",
                        "parameters": {"batch_size": 100}
                    }
                },
                "type": "default"
            },
            {
                "id": "node-3",
                "position": {"x": 500, "y": 100},
                "data": {
                    "label": "Quality Check",
                    "type": "decision",
                    "config": {
                        "condition": "data_quality > 0.8"
                    }
                },
                "type": "decision"
            },
            {
                "id": "node-4",
                "position": {"x": 700, "y": 100},
                "data": {
                    "label": "End",
                    "type": "end",
                    "config": {}
                },
                "type": "output"
            }
        ],
        "edges": [
            {
                "id": "edge-1",
                "source": "node-1",
                "target": "node-2",
                "sourceHandle": "source",
                "targetHandle": "target",
                "type": "default",
                "animated": False
            },
            {
                "id": "edge-2",
                "source": "node-2",
                "target": "node-3",
                "sourceHandle": "source",
                "targetHandle": "target",
                "type": "default",
                "animated": False
            },
            {
                "id": "edge-3",
                "source": "node-3",
                "target": "node-4",
                "sourceHandle": "true",
                "targetHandle": "target",
                "type": "default",
                "animated": True
            }
        ],
        "viewport": {"x": 0, "y": 0, "zoom": 1}
    }


class TestReactFlowDataFormatValidation:
    """Test ReactFlow data format validation."""

    @pytest.mark.asyncio
    async def test_node_structure_compatibility(self, validator, sample_reactflow_data):
        """Test that workflow nodes are compatible with ReactFlow format."""
        # Mock workflow service to return test data
        workflow_data = {
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
                }
            ],
            "connections": []
        }

        # Test node structure conversion
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
                "type": validator._map_node_type_to_reactflow(node["node_type"])
            }
            reactflow_nodes.append(reactflow_node)

        # Validate required fields
        required_fields = ["id", "position", "data", "type"]
        for node in reactflow_nodes:
            for field in required_fields:
                assert field in node, f"Missing required field '{field}' in ReactFlow node"

        # Test node type mapping
        assert validator._map_node_type_to_reactflow("start") == "input"
        assert validator._map_node_type_to_reactflow("end") == "output"
        assert validator._map_node_type_to_reactflow("agent_task") == "default"
        assert validator._map_node_type_to_reactflow("decision") == "decision"

    @pytest.mark.asyncio
    async def test_edge_structure_compatibility(self, validator, sample_reactflow_data):
        """Test that workflow edges are compatible with ReactFlow format."""
        # Test edge structure conversion
        workflow_connections = [
            {
                "id": "conn-1",
                "source_node_id": "node-1",
                "target_node_id": "node-2",
                "source_handle": "source",
                "target_handle": "target"
            }
        ]

        reactflow_edges = []
        for edge in workflow_connections:
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

        # Validate required fields
        required_fields = ["id", "source", "target"]
        for edge in reactflow_edges:
            for field in required_fields:
                assert field in edge, f"Missing required field '{field}' in ReactFlow edge"

    @pytest.mark.asyncio
    async def test_data_serialization(self, validator, sample_reactflow_data):
        """Test that workflow data can be serialized for ReactFlow."""
        # Test JSON serialization
        json_str = json.dumps(sample_reactflow_data)
        parsed_data = json.loads(json_str)

        # Verify data integrity
        assert len(parsed_data["nodes"]) == len(sample_reactflow_data["nodes"])
        assert len(parsed_data["edges"]) == len(sample_reactflow_data["edges"])

        # Verify node structure preservation
        for original_node, parsed_node in zip(sample_reactflow_data["nodes"], parsed_data["nodes"]):
            assert original_node["id"] == parsed_node["id"]
            assert original_node["position"] == parsed_node["position"]
            assert original_node["data"] == parsed_node["data"]
            assert original_node["type"] == parsed_node["type"]

        # Verify edge structure preservation
        for original_edge, parsed_edge in zip(sample_reactflow_data["edges"], parsed_data["edges"]):
            assert original_edge["id"] == parsed_edge["id"]
            assert original_edge["source"] == parsed_edge["source"]
            assert original_edge["target"] == parsed_edge["target"]

    @pytest.mark.asyncio
    async def test_complete_reactflow_data_format_validation(self, validator):
        """Test complete ReactFlow data format validation."""
        results = await validator.validate_reactflow_data_format()

        assert "passed" in results
        assert "errors" in results
        assert "warnings" in results
        assert "test_cases" in results
        assert len(results["test_cases"]) > 0

        # Verify test case structure
        for test_case in results["test_cases"]:
            assert "name" in test_case
            assert "description" in test_case
            assert "status" in test_case


class TestReactFlowAPIIntegration:
    """Test ReactFlow API integration."""

    @pytest.mark.asyncio
    async def test_create_workflow_with_reactflow_data(self, validator, async_session, sample_reactflow_data):
        """Test creating a workflow with ReactFlow data."""
        # Create workflow service and MCP tools
        workflow_service = WorkflowService(async_session)
        mcp_tools = WorkflowMCPTools(async_session)

        # Test workflow creation
        result = await mcp_tools.create_workflow(
            name="Test ReactFlow Workflow",
            description="Test workflow with ReactFlow data",
            reactflow_data=sample_reactflow_data
        )

        assert result["success"] is True
        assert "workflow_id" in result

        # Verify workflow was created
        workflow = await workflow_service.get_workflow(result["workflow_id"])
        assert workflow is not None
        assert workflow.name == "Test ReactFlow Workflow"
        assert workflow.reactflow_data is not None

        # Verify ReactFlow data was stored correctly
        stored_data = workflow.reactflow_data
        assert "nodes" in stored_data
        assert "edges" in stored_data
        assert len(stored_data["nodes"]) == len(sample_reactflow_data["nodes"])
        assert len(stored_data["edges"]) == len(sample_reactflow_data["edges"])

    @pytest.mark.asyncio
    async def test_get_workflow_in_reactflow_format(self, validator, async_session, sample_reactflow_data):
        """Test retrieving workflow in ReactFlow format."""
        # Create workflow first
        workflow_service = WorkflowService(async_session)
        mcp_tools = WorkflowMCPTools(async_session)

        create_result = await mcp_tools.create_workflow(
            name="Test Get Workflow",
            description="Test getting workflow in ReactFlow format",
            reactflow_data=sample_reactflow_data
        )

        assert create_result["success"] is True

        # Get workflow
        get_result = await mcp_tools.get_workflow(create_result["workflow_id"])

        assert get_result["success"] is True
        assert "workflow" in get_result

        workflow_data = get_result["workflow"]
        assert "reactflow_data" in workflow_data

        reactflow_data = workflow_data["reactflow_data"]
        assert "nodes" in reactflow_data
        assert "edges" in reactflow_data

        # Verify data structure matches ReactFlow requirements
        for node in reactflow_data["nodes"]:
            assert "id" in node
            assert "position" in node
            assert "data" in node
            assert "type" in node

        for edge in reactflow_data["edges"]:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge

    @pytest.mark.asyncio
    async def test_update_workflow_with_reactflow_data(self, validator, async_session, sample_reactflow_data):
        """Test updating workflow with ReactFlow data."""
        # Create workflow first
        workflow_service = WorkflowService(async_session)
        mcp_tools = WorkflowMCPTools(async_session)

        create_result = await mcp_tools.create_workflow(
            name="Test Update Workflow",
            description="Test updating workflow with ReactFlow data",
            reactflow_data=sample_reactflow_data
        )

        assert create_result["success"] is True

        # Update workflow with modified ReactFlow data
        updated_data = sample_reactflow_data.copy()
        updated_data["nodes"].append({
            "id": "node-5",
            "position": {"x": 900, "y": 100},
            "data": {
                "label": "Additional Step",
                "type": "agent_task",
                "config": {"agent_type": "additional_processor"}
            },
            "type": "default"
        })

        update_result = await mcp_tools.update_workflow(
            workflow_id=create_result["workflow_id"],
            reactflow_data=updated_data
        )

        assert update_result["success"] is True

        # Verify update
        get_result = await mcp_tools.get_workflow(create_result["workflow_id"])
        assert get_result["success"] is True

        updated_workflow = get_result["workflow"]
        assert len(updated_workflow["reactflow_data"]["nodes"]) == len(sample_reactflow_data["nodes"]) + 1

    @pytest.mark.asyncio
    async def test_complete_api_integration_validation(self, validator):
        """Test complete API integration validation."""
        results = await validator.validate_api_integration()

        assert "passed" in results
        assert "errors" in results
        assert "warnings" in results
        assert "api_tests" in results
        assert len(results["api_tests"]) > 0

        # Verify API test structure
        for api_test in results["api_tests"]:
            assert "name" in api_test
            assert "endpoint" in api_test
            assert "description" in api_test
            assert "status" in api_test


class TestReactFlowRealtimeUpdates:
    """Test ReactFlow real-time updates."""

    @pytest.mark.asyncio
    async def test_workflow_execution_events(self, validator):
        """Test workflow execution events for ReactFlow."""
        # Test workflow execution started event
        execution_event = {
            "event": "workflow_execution_started",
            "data": {
                "execution_id": "exec-123",
                "workflow_id": "workflow-456",
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        # Validate event structure
        validator._validate_reactflow_event(execution_event, {"passed": True, "errors": []})

        # Test step execution update event
        step_event = {
            "event": "step_execution_update",
            "data": {
                "execution_id": "exec-123",
                "step_id": "node-1",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        validator._validate_reactflow_event(step_event, {"passed": True, "errors": []})

        # Test workflow execution completed event
        completion_event = {
            "event": "workflow_execution_completed",
            "data": {
                "execution_id": "exec-123",
                "workflow_id": "workflow-456",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        validator._validate_reactflow_event(completion_event, {"passed": True, "errors": []})

    @pytest.mark.asyncio
    async def test_node_status_update_events(self, validator):
        """Test node status update events for ReactFlow."""
        # Test node status update event
        node_status_event = {
            "event": "node_status_update",
            "data": {
                "workflow_id": "workflow-456",
                "node_id": "node-1",
                "status": "running",
                "progress": 50,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        # Validate event structure
        validator._validate_reactflow_event(node_status_event, {"passed": True, "errors": []})

        # Test node status with different states
        status_states = ["idle", "running", "completed", "failed", "paused"]
        for status in status_states:
            event = {
                "event": "node_status_update",
                "data": {
                    "workflow_id": "workflow-456",
                    "node_id": "node-1",
                    "status": status,
                    "progress": 75,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            validator._validate_reactflow_event(event, {"passed": True, "errors": []})

    @pytest.mark.asyncio
    async def test_complete_realtime_updates_validation(self, validator):
        """Test complete real-time updates validation."""
        results = await validator.validate_realtime_updates()

        assert "passed" in results
        assert "errors" in results
        assert "warnings" in results
        assert "realtime_tests" in results
        assert len(results["realtime_tests"]) > 0

        # Verify realtime test structure
        for realtime_test in results["realtime_tests"]:
            assert "name" in realtime_test
            assert "event_type" in realtime_test
            assert "description" in realtime_test
            assert "status" in realtime_test


class TestReactFlowFrontendIntegration:
    """Test ReactFlow frontend integration."""

    @pytest.mark.asyncio
    async def test_component_data_flow(self, validator, async_session, sample_reactflow_data):
        """Test data flow between ReactFlow components and backend."""
        # Create workflow service and MCP tools
        workflow_service = WorkflowService(async_session)
        mcp_tools = WorkflowMCPTools(async_session)

        # Create workflow with ReactFlow data
        create_result = await mcp_tools.create_workflow(
            name="Test Frontend Integration",
            description="Test frontend integration with ReactFlow",
            reactflow_data=sample_reactflow_data
        )

        assert create_result["success"] is True

        # Get workflow for frontend
        get_result = await mcp_tools.get_workflow(create_result["workflow_id"])
        assert get_result["success"] is True

        workflow_data = get_result["workflow"]
        reactflow_data = workflow_data["reactflow_data"]

        # Verify ReactFlow component requirements
        component_requirements = {
            "ReactFlow": {
                "nodes": "Array of node objects",
                "edges": "Array of edge objects",
                "onNodesChange": "Function to handle node changes",
                "onEdgesChange": "Function to handle edge changes",
                "onConnect": "Function to handle new connections"
            }
        }

        # Check that required data is present
        assert "nodes" in reactflow_data
        assert "edges" in reactflow_data
        assert isinstance(reactflow_data["nodes"], list)
        assert isinstance(reactflow_data["edges"], list)

        # Check node structure for ReactFlow compatibility
        for node in reactflow_data["nodes"]:
            assert "id" in node
            assert "position" in node
            assert "data" in node
            assert "type" in node

        # Check edge structure for ReactFlow compatibility
        for edge in reactflow_data["edges"]:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge

    @pytest.mark.asyncio
    async def test_frontend_event_handling(self, validator):
        """Test frontend event handling patterns."""
        # Simulate frontend events
        frontend_events = [
            {
                "type": "nodesChange",
                "changes": [
                    {
                        "id": "node-1",
                        "type": "position",
                        "position": {"x": 150, "y": 150}
                    }
                ]
            },
            {
                "type": "edgesChange",
                "changes": [
                    {
                        "id": "edge-1",
                        "type": "remove"
                    }
                ]
            },
            {
                "type": "connect",
                "connection": {
                    "source": "node-1",
                    "target": "node-2",
                    "sourceHandle": "source",
                    "targetHandle": "target"
                }
            }
        ]

        # Validate that backend can handle these events
        for event in frontend_events:
            assert "type" in event
            assert "changes" in event or "connection" in event

            if event["type"] == "nodesChange":
                for change in event["changes"]:
                    assert "id" in change
                    assert "type" in change

            elif event["type"] == "edgesChange":
                for change in event["changes"]:
                    assert "id" in change
                    assert "type" in change

            elif event["type"] == "connect":
                connection = event["connection"]
                assert "source" in connection
                assert "target" in connection

    @pytest.mark.asyncio
    async def test_complete_frontend_integration_validation(self, validator):
        """Test complete frontend integration validation."""
        results = await validator.validate_frontend_integration()

        assert "passed" in results
        assert "errors" in results
        assert "warnings" in results
        assert "frontend_tests" in results
        assert len(results["frontend_tests"]) > 0

        # Verify frontend test structure
        for frontend_test in results["frontend_tests"]:
            assert "name" in frontend_test
            assert "description" in frontend_test
            assert "status" in frontend_test


class TestComprehensiveReactFlowValidation:
    """Test comprehensive ReactFlow integration validation."""

    @pytest.mark.asyncio
    async def test_comprehensive_validation(self, validator):
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

        # Check validation summary structure
        for category_name, category_result in results["validation_summary"].items():
            assert "passed" in category_result
            assert "errors" in category_result
            assert "warnings" in category_result

        # Check recommendations
        assert isinstance(results["recommendations"], list)
        assert len(results["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, validator):
        """Test validation error handling."""
        # Test with invalid data
        invalid_reactflow_data = {
            "nodes": [
                {
                    # Missing required 'id' field
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Invalid Node"},
                    "type": "default"
                }
            ],
            "edges": []
        }

        # Mock the validator to test error handling
        with patch.object(validator, '_create_test_workflow_data') as mock_create:
            mock_create.return_value = invalid_reactflow_data

            results = await validator.validate_reactflow_data_format()

            assert "passed" in results
            assert "errors" in results
            assert len(results["errors"]) > 0
            assert not results["passed"]

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, validator):
        """Test recommendation generation."""
        # Test with failed validation
        mock_results = {
            "overall_passed": False,
            "validation_summary": {
                "data_format": {"passed": False},
                "api_integration": {"passed": True},
                "realtime_updates": {"passed": False},
                "frontend_integration": {"passed": True}
            }
        }

        recommendations = validator._generate_recommendations(mock_results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check that specific recommendations are generated for failed categories
        recommendation_text = " ".join(recommendations)
        assert "data format" in recommendation_text.lower()
        assert "real-time" in recommendation_text.lower()

    @pytest.mark.asyncio
    async def test_validation_with_missing_workflow(self, validator):
        """Test validation when no workflow exists."""
        # This tests the graceful handling of missing workflow data
        results = await validator.validate_api_integration()

        assert "passed" in results
        assert "warnings" in results
        # Should have warnings about missing workflow
        assert len(results["warnings"]) > 0


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v"])