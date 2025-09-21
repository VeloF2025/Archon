"""
Tests for workflow MCP tools

Comprehensive test suite covering:
- All MCP workflow tool implementations
- Request validation and error handling
- Response formatting and data integrity
- Integration with workflow services
- Security and access control
- Performance and scalability
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution, ReactFlowNode, ReactFlowEdge,
    ReactFlowData, ExecutionStatus, StepType, AgentType, ModelTier
)
from src.database import Base
from src.mcp_server.modules.workflow_mcp_tools import (
    WorkflowMCPTools, WorkflowCreateRequest, WorkflowUpdateRequest,
    WorkflowExecuteRequest, WorkflowValidationRequest
)


@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def workflow_mcp_tools():
    """Create workflow MCP tools instance"""
    return WorkflowMCPTools()


@pytest.fixture
def sample_reactflow_nodes():
    """Sample ReactFlow nodes"""
    return [
        {
            "id": "node1",
            "type": "input",
            "position": {"x": 0, "y": 0},
            "data": {"label": "Start", "input_text": "Hello World"}
        },
        {
            "id": "node2",
            "type": "agentTask",
            "position": {"x": 200, "y": 0},
            "data": {
                "label": "Process Data",
                "agent_type": "analyst",
                "model_tier": "standard",
                "prompt": "Analyze the input text"
            }
        },
        {
            "id": "node3",
            "type": "output",
            "position": {"x": 400, "y": 0},
            "data": {"label": "End"}
        }
    ]


@pytest.fixture
def sample_reactflow_edges():
    """Sample ReactFlow edges"""
    return [
        {
            "id": "edge1",
            "source": "node1",
            "target": "node2",
            "type": "default"
        },
        {
            "id": "edge2",
            "source": "node2",
            "target": "node3",
            "type": "default"
        }
    ]


@pytest.fixture
def sample_workflow(test_db, sample_reactflow_nodes, sample_reactflow_edges):
    """Create sample workflow for testing"""
    nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
    edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

    workflow = WorkflowDefinition(
        name="Test Workflow",
        description="For MCP testing",
        nodes=nodes,
        edges=edges,
        variables={"input_text": "default_value"}
    )
    test_db.add(workflow)
    test_db.commit()
    return workflow


@pytest.fixture
def sample_execution(test_db, sample_workflow):
    """Create sample execution for testing"""
    execution = WorkflowExecution(
        workflow_id=sample_workflow.id,
        status=ExecutionStatus.RUNNING,
        inputs={"input_text": "test input"},
        started_at=datetime.now(),
        metrics={"duration": 5.0, "steps_executed": 2}
    )
    test_db.add(execution)
    test_db.commit()
    return execution


class TestWorkflowCreationTools:
    """Test workflow creation MCP tools"""

    @pytest.mark.asyncio
    async def test_create_workflow_basic(self, workflow_mcp_tools, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test basic workflow creation"""
        request = WorkflowCreateRequest(
            name="Test Workflow",
            description="Created via MCP",
            nodes=sample_reactflow_nodes,
            edges=sample_reactflow_edges,
            variables={"input": "default"},
            tags=["mcp", "test"]
        )

        result = await workflow_mcp_tools.create_workflow(request, db=test_db)

        assert result["success"] is True
        assert "workflow_id" in result
        assert result["name"] == "Test Workflow"
        assert result["description"] == "Created via MCP"
        assert result["status"] == "draft"
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2

    @pytest.mark.asyncio
    async def test_create_workflow_minimal(self, workflow_mcp_tools, test_db):
        """Test creating workflow with minimal data"""
        request = WorkflowCreateRequest(
            name="Minimal Workflow",
            description="Minimal MCP workflow",
            nodes=[],
            edges=[]
        )

        result = await workflow_mcp_tools.create_workflow(request, db=test_db)

        assert result["success"] is True
        assert result["name"] == "Minimal Workflow"
        assert result["nodes"] == []
        assert result["edges"] == []

    @pytest.mark.asyncio
    async def test_create_workflow_validation_error(self, workflow_mcp_tools, test_db):
        """Test workflow creation with validation error"""
        request = WorkflowCreateRequest(
            name="",  # Empty name should fail validation
            description="Invalid workflow",
            nodes=[],
            edges=[]
        )

        result = await workflow_mcp_tools.create_workflow(request, db=test_db)

        assert result["success"] is False
        assert "error" in result
        assert "name" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_workflow_database_error(self, workflow_mcp_tools, test_db):
        """Test handling database errors during workflow creation"""
        request = WorkflowCreateRequest(
            name="Error Workflow",
            description="Should fail",
            nodes=[],
            edges=[]
        )

        # Mock database to raise exception
        with patch.object(test_db, 'commit', side_effect=Exception("Database error")):
            result = await workflow_mcp_tools.create_workflow(request, db=test_db)

            assert result["success"] is False
            assert "error" in result


class TestWorkflowUpdateTools:
    """Test workflow update MCP tools"""

    @pytest.mark.asyncio
    async def test_update_workflow_basic(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test basic workflow update"""
        request = WorkflowUpdateRequest(
            name="Updated Workflow",
            description="Updated via MCP",
            variables={"new_var": "new_value"},
            tags=["updated", "mcp"],
            status="active"
        )

        result = await workflow_mcp_tools.update_workflow(
            workflow_id=str(sample_workflow.id),
            request=request,
            db=test_db
        )

        assert result["success"] is True
        assert result["workflow_id"] == str(sample_workflow.id)
        assert result["name"] == "Updated Workflow"
        assert result["description"] == "Updated via MCP"
        assert result["variables"] == {"new_var": "new_value"}
        assert result["tags"] == ["updated", "mcp"]
        assert result["status"] == "active"
        assert result["version"] == 2

    @pytest.mark.asyncio
    async def test_update_workflow_partial(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test partial workflow update"""
        request = WorkflowUpdateRequest(
            name="Name Update Only"
        )

        result = await workflow_mcp_tools.update_workflow(
            workflow_id=str(sample_workflow.id),
            request=request,
            db=test_db
        )

        assert result["success"] is True
        assert result["name"] == "Name Update Only"
        # Other fields should remain unchanged
        assert result["description"] == sample_workflow.description

    @pytest.mark.asyncio
    async def test_update_workflow_not_exists(self, workflow_mcp_tools, test_db):
        """Test updating non-existent workflow"""
        non_existent_id = str(uuid4())
        request = WorkflowUpdateRequest(name="Updated")

        result = await workflow_mcp_tools.update_workflow(
            workflow_id=non_existent_id,
            request=request,
            db=test_db
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_update_workflow_reactflow_data(self, workflow_mcp_tools, test_db, sample_workflow, sample_reactflow_nodes, sample_reactflow_edges):
        """Test updating workflow ReactFlow data"""
        # Add new node
        new_nodes = sample_reactflow_nodes + [
            {
                "id": "node4",
                "type": "agentTask",
                "position": {"x": 600, "y": 0},
                "data": {"label": "Additional Step", "agent_type": "tester"}
            }
        ]

        request = WorkflowUpdateRequest(nodes=new_nodes)

        result = await workflow_mcp_tools.update_workflow(
            workflow_id=str(sample_workflow.id),
            request=request,
            db=test_db
        )

        assert result["success"] is True
        assert len(result["nodes"]) == 4
        assert result["nodes"][-1]["id"] == "node4"


class TestWorkflowDeletionTools:
    """Test workflow deletion MCP tools"""

    @pytest.mark.asyncio
    async def test_delete_workflow_exists(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test deleting existing workflow"""
        result = await workflow_mcp_tools.delete_workflow(
            workflow_id=str(sample_workflow.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["workflow_id"] == str(sample_workflow.id)
        assert result["message"] == "Workflow deleted successfully"

        # Verify workflow is soft deleted
        deleted = test_db.query(WorkflowDefinition).filter(WorkflowDefinition.id == sample_workflow.id).first()
        assert deleted is not None
        assert deleted.deleted_at is not None

    @pytest.mark.asyncio
    async def test_delete_workflow_not_exists(self, workflow_mcp_tools, test_db):
        """Test deleting non-existent workflow"""
        non_existent_id = str(uuid4())
        result = await workflow_mcp_tools.delete_workflow(
            workflow_id=non_existent_id,
            db=test_db
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_workflow_with_executions(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test deleting workflow with executions"""
        # Create execution for workflow
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now()
        )
        test_db.add(execution)
        test_db.commit()

        # Try to delete workflow (should fail)
        result = await workflow_mcp_tools.delete_workflow(
            workflow_id=str(sample_workflow.id),
            db=test_db
        )

        assert result["success"] is False
        assert "execution history" in result["error"].lower()


class TestWorkflowExecutionTools:
    """Test workflow execution MCP tools"""

    @pytest.mark.asyncio
    async def test_execute_workflow_basic(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test basic workflow execution"""
        request = WorkflowExecuteRequest(
            workflow_id=str(sample_workflow.id),
            inputs={"input_text": "test input"},
            async_execution=True,
            timeout=300
        )

        result = await workflow_mcp_tools.execute_workflow(request, db=test_db)

        assert result["success"] is True
        assert "execution_id" in result
        assert result["status"] == "started"
        assert result["message"] == "Workflow execution started"

    @pytest.mark.asyncio
    async def test_execute_workflow_sync(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test synchronous workflow execution"""
        request = WorkflowExecuteRequest(
            workflow_id=str(sample_workflow.id),
            inputs={"input_text": "test input"},
            async_execution=False,
            timeout=60
        )

        result = await workflow_mcp_tools.execute_workflow(request, db=test_db)

        assert result["success"] is True
        assert "execution_id" in result
        # Should have execution results due to sync execution
        assert "results" in result or "status" in result

    @pytest.mark.asyncio
    async def test_execute_workflow_not_exists(self, workflow_mcp_tools, test_db):
        """Test executing non-existent workflow"""
        non_existent_id = str(uuid4())
        request = WorkflowExecuteRequest(
            workflow_id=non_existent_id,
            inputs={"input": "test"}
        )

        result = await workflow_mcp_tools.execute_workflow(request, db=test_db)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_pause_resume_cancel_execution(self, workflow_mcp_tools, test_db, sample_workflow, sample_execution):
        """Test pause, resume, and cancel execution"""
        # Pause execution
        result = await workflow_mcp_tools.pause_workflow(
            execution_id=str(sample_execution.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["execution_id"] == str(sample_execution.id)
        assert result["status"] == "paused"

        # Resume execution
        result = await workflow_mcp_tools.resume_workflow(
            execution_id=str(sample_execution.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["status"] == "resumed"

        # Cancel execution
        result = await workflow_mcp_tools.cancel_workflow(
            execution_id=str(sample_execution.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["status"] == "cancelled"


class TestWorkflowRetrievalTools:
    """Test workflow retrieval MCP tools"""

    @pytest.mark.asyncio
    async def test_list_workflows_basic(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test basic workflow listing"""
        result = await workflow_mcp_tools.list_workflows(
            skip=0,
            limit=10,
            db=test_db
        )

        assert result["success"] is True
        assert "workflows" in result
        assert "total" in result
        assert result["total"] == 1
        assert len(result["workflows"]) == 1
        assert result["workflows"][0]["id"] == str(sample_workflow.id)

    @pytest.mark.asyncio
    async def test_list_workflows_with_filters(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test workflow listing with filters"""
        # Update workflow to have specific status and tags
        sample_workflow.status = "active"
        sample_workflow.tags = ["filtered", "test"]
        test_db.commit()

        # Filter by status
        result = await workflow_mcp_tools.list_workflows(
            status="active",
            skip=0,
            limit=10,
            db=test_db
        )

        assert result["success"] is True
        assert result["total"] == 1
        assert result["workflows"][0]["status"] == "active"

        # Filter by tags
        result = await workflow_mcp_tools.list_workflows(
            tags=["filtered"],
            skip=0,
            limit=10,
            db=test_db
        )

        assert result["success"] is True
        assert result["total"] == 1
        assert "filtered" in result["workflows"][0]["tags"]

        # Search by name
        result = await workflow_mcp_tools.list_workflows(
            search="Test",
            skip=0,
            limit=10,
            db=test_db
        )

        assert result["success"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_list_workflows_pagination(self, workflow_mcp_tools, test_db):
        """Test workflow listing pagination"""
        # Create multiple workflows
        for i in range(5):
            workflow = WorkflowDefinition(
                name=f"Workflow {i}",
                description=f"Test workflow {i}",
                nodes=[],
                edges=[]
            )
            test_db.add(workflow)

        test_db.commit()

        # Test pagination
        result = await workflow_mcp_tools.list_workflows(
            skip=2,
            limit=2,
            db=test_db
        )

        assert result["success"] is True
        assert result["total"] == 6  # 5 new + 1 original
        assert len(result["workflows"]) == 2

    @pytest.mark.asyncio
    async def test_get_workflow_details(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test getting workflow details"""
        result = await workflow_mcp_tools.get_workflow(
            workflow_id=str(sample_workflow.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["workflow"]["id"] == str(sample_workflow.id)
        assert result["workflow"]["name"] == sample_workflow.name
        assert result["workflow"]["description"] == sample_workflow.description
        assert "nodes" in result["workflow"]
        assert "edges" in result["workflow"]

    @pytest.mark.asyncio
    async def test_get_workflow_details_not_exists(self, workflow_mcp_tools, test_db):
        """Test getting details of non-existent workflow"""
        non_existent_id = str(uuid4())
        result = await workflow_mcp_tools.get_workflow(
            workflow_id=non_existent_id,
            db=test_db
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_workflow_status(self, workflow_mcp_tools, test_db, sample_workflow, sample_execution):
        """Test getting workflow status"""
        result = await workflow_mcp_tools.get_workflow_status(
            workflow_id=str(sample_workflow.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["workflow_id"] == str(sample_workflow.id)
        assert "status" in result
        assert "has_active_execution" in result
        assert "latest_execution" in result


class TestWorkflowValidationTools:
    """Test workflow validation MCP tools"""

    @pytest.mark.asyncio
    async def test_validate_workflow_valid(self, workflow_mcp_tools, sample_reactflow_nodes, sample_reactflow_edges):
        """Test validating valid workflow"""
        request = WorkflowValidationRequest(
            nodes=sample_reactflow_nodes,
            edges=sample_reactflow_edges
        )

        result = await workflow_mcp_tools.validate_workflow(request)

        assert result["success"] is True
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["node_count"] == 3
        assert result["edge_count"] == 2

    @pytest.mark.asyncio
    async def test_validate_workflow_invalid(self, workflow_mcp_tools, sample_reactflow_nodes):
        """Test validating invalid workflow"""
        # Create edge with non-existent source
        invalid_edges = [
            {
                "id": "invalid_edge",
                "source": "nonexistent_node",
                "target": "node2"
            }
        ]

        request = WorkflowValidationRequest(
            nodes=sample_reactflow_nodes,
            edges=invalid_edges
        )

        result = await workflow_mcp_tools.validate_workflow(request)

        assert result["success"] is True
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("source node" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_existing_workflow(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test validating existing workflow"""
        result = await workflow_mcp_tools.validate_workflow(
            workflow_id=str(sample_workflow.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["valid"] is True
        assert result["workflow_id"] == str(sample_workflow.id)


class TestWorkflowAnalyticsTools:
    """Test workflow analytics MCP tools"""

    @pytest.mark.asyncio
    async def test_get_workflow_analytics(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test getting workflow analytics"""
        result = await workflow_mcp_tools.get_workflow_analytics(
            workflow_id=str(sample_workflow.id),
            days=7,
            db=test_db
        )

        assert result["success"] is True
        assert result["workflow_id"] == str(sample_workflow.id)
        assert "executions" in result
        assert "performance" in result
        assert "cost_analysis" in result
        assert "bottlenecks" in result

    @pytest.mark.asyncio
    async def test_get_workflow_analytics_not_exists(self, workflow_mcp_tools, test_db):
        """Test getting analytics for non-existent workflow"""
        non_existent_id = str(uuid4())
        result = await workflow_mcp_tools.get_workflow_analytics(
            workflow_id=non_existent_id,
            days=7,
            db=test_db
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_visualize_workflow(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test workflow visualization"""
        result = await workflow_mcp_tools.visualize_workflow(
            workflow_id=str(sample_workflow.id),
            format="reactflow",
            db=test_db
        )

        assert result["success"] is True
        assert result["workflow_id"] == str(sample_workflow.id)
        assert "visualization" in result
        assert "nodes" in result["visualization"]
        assert "edges" in result["visualization"]


class TestTemplateManagementTools:
    """Test template management MCP tools"""

    @pytest.mark.asyncio
    async def test_list_workflow_templates(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test listing workflow templates"""
        # Create template
        sample_workflow.is_template = True
        sample_workflow.category = "analysis"
        test_db.commit()

        result = await workflow_mcp_tools.list_workflow_templates(
            category="analysis",
            skip=0,
            limit=10,
            db=test_db
        )

        assert result["success"] is True
        assert "templates" in result
        assert "total" in result
        assert result["total"] == 1
        assert result["templates"][0]["id"] == str(sample_workflow.id)

    @pytest.mark.asyncio
    async def test_get_workflow_template(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test getting workflow template"""
        # Mark as template
        sample_workflow.is_template = True
        sample_workflow.category = "analysis"
        test_db.commit()

        result = await workflow_mcp_tools.get_workflow_template(
            template_id=str(sample_workflow.id),
            db=test_db
        )

        assert result["success"] is True
        assert result["template"]["id"] == str(sample_workflow.id)
        assert result["template"]["is_template"] is True
        assert result["template"]["category"] == "analysis"


class TestRealTimeMetricsTools:
    """Test real-time metrics MCP tools"""

    @pytest.mark.asyncio
    async def test_get_real_time_metrics(self, workflow_mcp_tools, test_db):
        """Test getting real-time metrics"""
        result = await workflow_mcp_tools.get_real_time_metrics(db=test_db)

        assert result["success"] is True
        assert "timestamp" in result
        assert "active_executions" in result
        assert "system_load" in result
        assert "performance_indicators" in result


class TestErrorHandlingAndValidation:
    """Test error handling and validation"""

    @pytest.mark.asyncio
    async def test_invalid_request_format(self, workflow_mcp_tools, test_db):
        """Test handling invalid request format"""
        # Test with malformed request (missing required fields)
        request_data = {"description": "Missing name"}  # Missing name field

        result = await workflow_mcp_tools.create_workflow(request_data, db=test_db)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_database_connection_error(self, workflow_mcp_tools):
        """Test handling database connection errors"""
        request = WorkflowCreateRequest(
            name="Test Workflow",
            description="Database error test",
            nodes=[],
            edges=[]
        )

        # Mock database to raise connection error
        mock_db = Mock()
        mock_db.query.side_effect = Exception("Connection failed")

        result = await workflow_mcp_tools.create_workflow(request, mock_db)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_service_unavailable_error(self, workflow_mcp_tools, test_db):
        """Test handling service unavailable errors"""
        request = WorkflowExecuteRequest(
            workflow_id=str(uuid4()),
            inputs={"test": "input"}
        )

        # Mock execution service to raise unavailable error
        with patch('src.mcp_server.modules.workflow_mcp_tools.WorkflowExecutionService') as mock_service:
            mock_service.return_value.execute_workflow.side_effect = Exception("Service unavailable")

            result = await workflow_mcp_tools.execute_workflow(request, test_db)

            assert result["success"] is False
            assert "unavailable" in result["error"].lower()


class TestSecurityAndAccessControl:
    """Test security and access control"""

    @pytest.mark.asyncio
    async def test_input_sanitization(self, workflow_mcp_tools, test_db):
        """Test input sanitization"""
        # Test with potentially malicious input
        malicious_nodes = [
            {
                "id": "node1",
                "type": "input",
                "position": {"x": 0, "y": 0},
                "data": {"label": "<script>alert('xss')</script>", "input": "normal"}
            }
        ]

        request = WorkflowCreateRequest(
            name="Sanitization Test",
            description="Test input sanitization",
            nodes=malicious_nodes,
            edges=[]
        )

        result = await workflow_mcp_tools.create_workflow(request, db=test_db)

        # Should handle safely without script execution
        assert result["success"] is True
        assert "<script>" not in str(result)  # Script should be escaped or removed

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, workflow_mcp_tools, test_db):
        """Test SQL injection prevention"""
        request = WorkflowCreateRequest(
            name="SQL Injection Test",
            description="Test'; DROP TABLE workflows; --",
            nodes=[],
            edges=[]
        )

        result = await workflow_mcp_tools.create_workflow(request, db=test_db)

        # Should handle safely without executing SQL injection
        assert result["success"] is True
        # Verify table still exists
        tables = test_db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        workflow_table_exists = any("workflow" in str(table).lower() for table in tables)
        assert workflow_table_exists


class TestPerformanceAndScalability:
    """Test performance and scalability"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, workflow_mcp_tools, test_db):
        """Test handling concurrent requests"""
        import asyncio

        # Create multiple concurrent requests
        requests = []
        for i in range(10):
            request = WorkflowCreateRequest(
                name=f"Concurrent Workflow {i}",
                description=f"Test concurrent execution {i}",
                nodes=[],
                edges=[]
            )
            requests.append(request)

        # Execute all requests concurrently
        tasks = [
            workflow_mcp_tools.create_workflow(request, test_db)
            for request in requests
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successful_results) == 10

    @pytest.mark.asyncio
    async def test_large_workflow_handling(self, workflow_mcp_tools, test_db):
        """Test handling of large workflows"""
        # Create large workflow with many nodes and edges
        large_nodes = []
        large_edges = []

        for i in range(100):
            large_nodes.append({
                "id": f"node{i}",
                "type": "agentTask",
                "position": {"x": i * 50, "y": 0},
                "data": {"label": f"Task {i}", "agent_type": "analyst"}
            })

            if i > 0:
                large_edges.append({
                    "id": f"edge{i-1}-{i}",
                    "source": f"node{i-1}",
                    "target": f"node{i}"
                })

        request = WorkflowCreateRequest(
            name="Large Workflow",
            description="Test large workflow handling",
            nodes=large_nodes,
            edges=large_edges
        )

        result = await workflow_mcp_tools.create_workflow(request, db=test_db)

        assert result["success"] is True
        assert len(result["nodes"]) == 100
        assert len(result["edges"]) == 99

    @pytest.mark.asyncio
    async def test_response_size_limits(self, workflow_mcp_tools, test_db):
        """Test response size limits"""
        # Create workflow with large data
        large_data = {"large_field": "x" * 10000}  # 10KB of data

        request = WorkflowCreateRequest(
            name="Large Data Test",
            description="Test response size limits",
            nodes=[],
            edges=[],
            variables=large_data
        )

        result = await workflow_mcp_tools.create_workflow(request, test_db)

        assert result["success"] is True
        # Response should be manageable size
        result_str = json.dumps(result)
        assert len(result_str) < 1024 * 1024  # Less than 1MB


class TestIntegrationTesting:
    """Test integration with other services"""

    @pytest.mark.asyncio
    async def test_service_integration(self, workflow_mcp_tools, test_db, sample_workflow):
        """Test integration with workflow services"""
        # Mock services to verify integration
        with patch('src.mcp_server.modules.workflow_mcp_tools.WorkflowService') as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance
            mock_instance.get_workflow.return_value = sample_workflow

            result = await workflow_mcp_tools.get_workflow(
                workflow_id=str(sample_workflow.id),
                db=test_db
            )

            assert result["success"] is True
            # Verify service was called
            mock_instance.get_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, workflow_mcp_tools, test_db):
        """Test database transaction rollback on errors"""
        request = WorkflowCreateRequest(
            name="Transaction Test",
            description="Test transaction rollback",
            nodes=[],
            edges=[]
        )

        # Mock database to raise error during commit
        with patch.object(test_db, 'commit', side_effect=Exception("Rollback test")):
            result = await workflow_mcp_tools.create_workflow(request, test_db)

            assert result["success"] is False

            # Verify no partial data was created
            workflows = test_db.query(WorkflowDefinition).filter(
                WorkflowDefinition.name == "Transaction Test"
            ).all()
            assert len(workflows) == 0