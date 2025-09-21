"""
Tests for workflow service layer

Comprehensive test suite covering:
- Workflow CRUD operations
- ReactFlow data handling
- Validation and error handling
- Template management
- Search and filtering
- Business logic and edge cases
"""

import pytest
import json
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution, ReactFlowNode, ReactFlowEdge,
    ReactFlowData, ExecutionStatus, StepType, WorkflowCreateRequest, WorkflowUpdateRequest
)
from src.database import Base
from src.server.services.workflow_service import WorkflowService
from src.database.agent_models import AgentV3


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
def workflow_service():
    """Create workflow service instance"""
    return WorkflowService()


@pytest.fixture
def sample_reactflow_nodes():
    """Sample ReactFlow nodes"""
    return [
        {
            "id": "node1",
            "type": "input",
            "position": {"x": 0, "y": 0},
            "data": {"label": "Start", "input_type": "text"}
        },
        {
            "id": "node2",
            "type": "agentTask",
            "position": {"x": 200, "y": 0},
            "data": {
                "label": "Process Data",
                "agent_type": "analyst",
                "model_tier": "standard",
                "prompt": "Analyze the input data"
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
def sample_workflow_data():
    """Sample workflow data"""
    return {
        "name": "Test Workflow",
        "description": "A comprehensive test workflow",
        "variables": {"input_text": "default_value", "max_length": 100},
        "tags": ["test", "sample", "validation"]
    }


class TestWorkflowCreation:
    """Test workflow creation operations"""

    def test_create_workflow_basic(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges, sample_workflow_data):
        """Test basic workflow creation"""
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        workflow = workflow_service.create_workflow(
            name=sample_workflow_data["name"],
            description=sample_workflow_data["description"],
            react_flow_data=react_flow_data,
            variables=sample_workflow_data["variables"],
            tags=sample_workflow_data["tags"],
            db=test_db
        )

        assert workflow.id is not None
        assert workflow.name == sample_workflow_data["name"]
        assert workflow.description == sample_workflow_data["description"]
        assert workflow.status == "draft"
        assert workflow.version == 1
        assert workflow.variables == sample_workflow_data["variables"]
        assert workflow.tags == sample_workflow_data["tags"]
        assert len(workflow.nodes) == 3
        assert len(workflow.edges) == 2

    def test_create_workflow_minimal(self, workflow_service, test_db):
        """Test creating workflow with minimal data"""
        workflow = workflow_service.create_workflow(
            name="Minimal Workflow",
            description="Minimal description",
            react_flow_data=ReactFlowData(nodes=[], edges=[]),
            db=test_db
        )

        assert workflow.id is not None
        assert workflow.name == "Minimal Workflow"
        assert workflow.status == "draft"
        assert workflow.version == 1
        assert workflow.variables == {}
        assert workflow.tags == []
        assert len(workflow.nodes) == 0
        assert len(workflow.edges) == 0

    def test_create_workflow_as_template(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test creating workflow as template"""
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        workflow = workflow_service.create_workflow(
            name="Template Workflow",
            description="A workflow template",
            react_flow_data=react_flow_data,
            is_template=True,
            db=test_db
        )

        assert workflow.is_template is True
        assert workflow.category == "general"  # Default category

    def test_create_workflow_validation_error(self, workflow_service, test_db):
        """Test workflow creation with invalid data"""
        with pytest.raises(ValueError, match="Workflow name is required"):
            workflow_service.create_workflow(
                name="",  # Empty name
                description="Invalid workflow",
                react_flow_data=ReactFlowData(nodes=[], edges=[]),
                db=test_db
            )

    def test_create_workflow_database_error(self, workflow_service, test_db):
        """Test handling database errors during workflow creation"""
        # Mock database session to raise exception
        with patch.object(test_db, 'commit', side_effect=Exception("Database error")):
            with pytest.raises(Exception, match="Database error"):
                workflow_service.create_workflow(
                    name="Error Workflow",
                    description="Should fail",
                    react_flow_data=ReactFlowData(nodes=[], edges=[]),
                    db=test_db
                )


class TestWorkflowRetrieval:
    """Test workflow retrieval operations"""

    def test_get_workflow_exists(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test retrieving existing workflow"""
        # Create workflow first
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Test Workflow",
            description="For retrieval",
            react_flow_data=react_flow_data,
            db=test_db
        )

        # Retrieve workflow
        retrieved = workflow_service.get_workflow(workflow.id, db=test_db)

        assert retrieved is not None
        assert retrieved.id == workflow.id
        assert retrieved.name == "Test Workflow"
        assert len(retrieved.nodes) == 3
        assert len(retrieved.edges) == 2

    def test_get_workflow_not_exists(self, workflow_service, test_db):
        """Test retrieving non-existent workflow"""
        non_existent_id = str(uuid4())
        retrieved = workflow_service.get_workflow(non_existent_id, db=test_db)

        assert retrieved is None

    def test_list_workflows_empty(self, workflow_service, test_db):
        """Test listing workflows when none exist"""
        result = workflow_service.list_workflows(db=test_db)

        assert result["total"] == 0
        assert result["workflows"] == []

    def test_list_workflows_with_data(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test listing workflows with data"""
        # Create multiple workflows
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        workflow1 = workflow_service.create_workflow(
            name="Workflow 1",
            description="First workflow",
            react_flow_data=react_flow_data,
            tags=["tag1", "common"],
            db=test_db
        )

        workflow2 = workflow_service.create_workflow(
            name="Workflow 2",
            description="Second workflow",
            react_flow_data=react_flow_data,
            tags=["tag2", "common"],
            status="active",
            db=test_db
        )

        workflow3 = workflow_service.create_workflow(
            name="Workflow 3",
            description="Third workflow",
            react_flow_data=react_flow_data,
            tags=["tag3"],
            status="draft",
            db=test_db
        )

        # List all workflows
        result = workflow_service.list_workflows(db=test_db)
        assert result["total"] == 3
        assert len(result["workflows"]) == 3

        # List with pagination
        result_paginated = workflow_service.list_workflows(skip=1, limit=2, db=test_db)
        assert result_paginated["total"] == 3
        assert len(result_paginated["workflows"]) == 2

        # List with status filter
        result_active = workflow_service.list_workflows(status="active", db=test_db)
        assert result_active["total"] == 1
        assert result_active["workflows"][0]["name"] == "Workflow 2"

        # List with tags filter
        result_tagged = workflow_service.list_workflows(tags=["common"], db=test_db)
        assert result_tagged["total"] == 2

        # List with search
        result_search = workflow_service.list_workflows(search="First", db=test_db)
        assert result_search["total"] == 1
        assert result_search["workflows"][0]["name"] == "Workflow 1"

    def test_list_templates(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test listing workflow templates"""
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        # Create regular workflow
        workflow_service.create_workflow(
            name="Regular Workflow",
            description="Not a template",
            react_flow_data=react_flow_data,
            is_template=False,
            db=test_db
        )

        # Create template
        template = workflow_service.create_workflow(
            name="Template Workflow",
            description="This is a template",
            react_flow_data=react_flow_data,
            is_template=True,
            category="analysis",
            db=test_db
        )

        # List templates
        templates = workflow_service.list_templates(db=test_db)
        assert templates["total"] == 1
        assert templates["templates"][0]["id"] == template.id

        # List templates by category
        analysis_templates = workflow_service.list_templates(category="analysis", db=test_db)
        assert analysis_templates["total"] == 1

        empty_category = workflow_service.list_templates(category="nonexistent", db=test_db)
        assert empty_category["total"] == 0


class TestWorkflowUpdate:
    """Test workflow update operations"""

    def test_update_workflow_basic(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test basic workflow update"""
        # Create workflow first
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Original Workflow",
            description="Original description",
            react_flow_data=react_flow_data,
            db=test_db
        )

        # Update workflow
        update_data = {
            "name": "Updated Workflow",
            "description": "Updated description",
            "variables": {"new_var": "new_value"},
            "tags": ["updated", "workflow"],
            "status": "active"
        }

        updated = workflow_service.update_workflow(workflow.id, update_data, db=test_db)

        assert updated is not None
        assert updated.name == "Updated Workflow"
        assert updated.description == "Updated description"
        assert updated.variables == {"new_var": "new_value"}
        assert updated.tags == ["updated", "workflow"]
        assert updated.status == "active"
        assert updated.version == 2  # Version should increment

    def test_update_workflow_partial(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test partial workflow update"""
        # Create workflow first
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Original Workflow",
            description="Original description",
            react_flow_data=react_flow_data,
            variables={"original": "value"},
            tags=["original"],
            db=test_db
        )

        # Update only name and description
        update_data = {
            "name": "Updated Name Only",
            "description": "Updated description only"
        }

        updated = workflow_service.update_workflow(workflow.id, update_data, db=test_db)

        assert updated is not None
        assert updated.name == "Updated Name Only"
        assert updated.description == "Updated description only"
        assert updated.variables == {"original": "value"}  # Unchanged
        assert updated.tags == ["original"]  # Unchanged

    def test_update_workflow_reactflow_data(self, workflow_service, test_db):
        """Test updating workflow ReactFlow data"""
        # Create workflow with simple data
        simple_nodes = [ReactFlowNode(id="node1", type="input", position={"x": 0, "y": 0}, data={"label": "Start"})]
        simple_edges = []
        react_flow_data = ReactFlowData(nodes=simple_nodes, edges=simple_edges)

        workflow = workflow_service.create_workflow(
            name="Workflow to Update",
            description="Will update ReactFlow data",
            react_flow_data=react_flow_data,
            db=test_db
        )

        # Update with new ReactFlow data
        new_nodes = [
            ReactFlowNode(id="node1", type="input", position={"x": 0, "y": 0}, data={"label": "Start"}),
            ReactFlowNode(id="node2", type="output", position={"x": 200, "y": 0}, data={"label": "End"})
        ]
        new_edges = [ReactFlowEdge(id="edge1", source="node1", target="node2")]
        new_reactflow_data = ReactFlowData(nodes=new_nodes, edges=new_edges)

        update_data = {"react_flow_data": new_reactflow_data}
        updated = workflow_service.update_workflow(workflow.id, update_data, db=test_db)

        assert updated is not None
        assert len(updated.nodes) == 2
        assert len(updated.edges) == 1
        assert updated.nodes[1].id == "node2"

    def test_update_workflow_not_exists(self, workflow_service, test_db):
        """Test updating non-existent workflow"""
        non_existent_id = str(uuid4())
        update_data = {"name": "Updated Name"}

        updated = workflow_service.update_workflow(non_existent_id, update_data, db=test_db)
        assert updated is None

    def test_update_workflow_validation_error(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test workflow update with invalid data"""
        # Create workflow first
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Original Workflow",
            description="Original description",
            react_flow_data=react_flow_data,
            db=test_db
        )

        # Try to update with empty name
        update_data = {"name": ""}
        updated = workflow_service.update_workflow(workflow.id, update_data, db=test_db)

        # Should not update
        assert updated.name == "Original Workflow"


class TestWorkflowDeletion:
    """Test workflow deletion operations"""

    def test_delete_workflow_exists(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test deleting existing workflow"""
        # Create workflow first
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Workflow to Delete",
            description="Will be deleted",
            react_flow_data=react_flow_data,
            db=test_db
        )

        workflow_id = workflow.id

        # Delete workflow
        success = workflow_service.delete_workflow(workflow_id, db=test_db)

        assert success is True

        # Verify workflow is soft deleted
        deleted = test_db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id).first()
        assert deleted is not None
        assert deleted.deleted_at is not None

    def test_delete_workflow_not_exists(self, workflow_service, test_db):
        """Test deleting non-existent workflow"""
        non_existent_id = str(uuid4())
        success = workflow_service.delete_workflow(non_existent_id, db=test_db)

        assert success is False

    def test_delete_workflow_with_executions(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test deleting workflow with executions"""
        # Create workflow first
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Workflow with Executions",
            description="Has execution history",
            react_flow_data=react_flow_data,
            db=test_db
        )

        # Create execution
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now()
        )
        test_db.add(execution)
        test_db.commit()

        # Try to delete workflow (should fail due to execution history)
        success = workflow_service.delete_workflow(workflow.id, db=test_db)

        assert success is False

        # Verify workflow still exists
        existing = test_db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow.id).first()
        assert existing is not None
        assert existing.deleted_at is None


class TestWorkflowValidation:
    """Test workflow validation operations"""

    def test_validate_workflow_valid(self, workflow_service, sample_reactflow_nodes, sample_reactflow_edges):
        """Test validating valid workflow"""
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        is_valid, errors = workflow_service.validate_workflow(react_flow_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_workflow_invalid_edges(self, workflow_service, sample_reactflow_nodes):
        """Test validating workflow with invalid edges"""
        # Create edge with non-existent source
        invalid_edge = ReactFlowEdge(
            id="invalid_edge",
            source="nonexistent_node",
            target="node2"
        )

        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[invalid_edge]
        )

        is_valid, errors = workflow_service.validate_workflow(react_flow_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("source node" in error for error in errors)

    def test_validate_workflow_disconnected_nodes(self, workflow_service):
        """Test validating workflow with disconnected nodes"""
        nodes = [
            ReactFlowNode(id="node1", type="input", position={"x": 0, "y": 0}, data={"label": "Start"}),
            ReactFlowNode(id="node2", type="output", position={"x": 200, "y": 0}, data={"label": "End"}),
            ReactFlowNode(id="node3", type="agentTask", position={"x": 400, "y": 0}, data={"label": "Disconnected"})
        ]
        edges = [
            ReactFlowEdge(id="edge1", source="node1", target="node2")
        ]

        react_flow_data = ReactFlowData(nodes=nodes, edges=edges)

        is_valid, errors = workflow_service.validate_workflow(react_flow_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("disconnected" in error.lower() for error in errors)

    def test_validate_workflow_circular_references(self, workflow_service):
        """Test validating workflow with circular references"""
        nodes = [
            ReactFlowNode(id="node1", type="input", position={"x": 0, "y": 0}, data={"label": "Start"}),
            ReactFlowNode(id="node2", type="agentTask", position={"x": 200, "y": 0}, data={"label": "Process"}),
            ReactFlowNode(id="node3", type="condition", position={"x": 400, "y": 0}, data={"label": "Check"})
        ]
        edges = [
            ReactFlowEdge(id="edge1", source="node1", target="node2"),
            ReactFlowEdge(id="edge2", source="node2", target="node3"),
            ReactFlowEdge(id="edge3", source="node3", target="node2")  # Circular reference
        ]

        react_flow_data = ReactFlowData(nodes=nodes, edges=edges)

        is_valid, errors = workflow_service.validate_workflow(react_flow_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("circular" in error.lower() for error in errors)


class TestWorkflowTemplates:
    """Test workflow template operations"""

    def test_create_workflow_from_template(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test creating workflow from template"""
        # Create template
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        template = workflow_service.create_workflow(
            name="Template Workflow",
            description="A workflow template",
            react_flow_data=react_flow_data,
            is_template=True,
            category="analysis",
            parameters={"input_param": {"type": "string", "default": "default_value"}},
            db=test_db
        )

        # Create workflow from template
        workflow = workflow_service.create_workflow_from_template(
            template_id=template.id,
            name="Workflow from Template",
            description="Created from template",
            parameters={"input_param": "custom_value"},
            db=test_db
        )

        assert workflow is not None
        assert workflow.name == "Workflow from Template"
        assert workflow.description == "Created from template"
        assert workflow.is_template is False
        assert len(workflow.nodes) == 3
        assert len(workflow.edges) == 2
        assert workflow.variables.get("input_param") == "custom_value"

    def test_create_workflow_from_template_not_exists(self, workflow_service, test_db):
        """Test creating workflow from non-existent template"""
        non_existent_id = str(uuid4())

        workflow = workflow_service.create_workflow_from_template(
            template_id=non_existent_id,
            name="Should Fail",
            description="Template doesn't exist",
            db=test_db
        )

        assert workflow is None

    def test_get_template_exists(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test retrieving existing template"""
        # Create template
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        template = workflow_service.create_workflow(
            name="Template Workflow",
            description="A workflow template",
            react_flow_data=react_flow_data,
            is_template=True,
            category="analysis",
            db=test_db
        )

        # Get template
        retrieved = workflow_service.get_template(template.id, db=test_db)

        assert retrieved is not None
        assert retrieved.id == template.id
        assert retrieved.is_template is True
        assert retrieved.category == "analysis"

    def test_get_template_not_exists(self, workflow_service, test_db):
        """Test retrieving non-existent template"""
        non_existent_id = str(uuid4())
        retrieved = workflow_service.get_template(non_existent_id, db=test_db)

        assert retrieved is None

    def test_get_template_from_regular_workflow(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test getting template from regular workflow (should return None)"""
        # Create regular workflow
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Regular Workflow",
            description="Not a template",
            react_flow_data=react_flow_data,
            is_template=False,
            db=test_db
        )

        # Try to get as template
        template = workflow_service.get_template(workflow.id, db=test_db)

        assert template is None


class TestWorkflowMetrics:
    """Test workflow metrics and statistics"""

    def test_get_workflow_count(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test getting workflow count"""
        # Initially no workflows
        count = workflow_service.get_workflow_count(db=test_db)
        assert count == 0

        # Create workflows
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        workflow_service.create_workflow(
            name="Workflow 1",
            description="First workflow",
            react_flow_data=react_flow_data,
            db=test_db
        )

        workflow_service.create_workflow(
            name="Workflow 2",
            description="Second workflow",
            react_flow_data=react_flow_data,
            db=test_db
        )

        # Check count increased
        count = workflow_service.get_workflow_count(db=test_db)
        assert count == 2

        # Create a deleted workflow
        workflow_service.create_workflow(
            name="Deleted Workflow",
            description="Will be deleted",
            react_flow_data=react_flow_data,
            db=test_db
        )

        workflow_service.delete_workflow(
            str(workflow_service.list_workflows(db=test_db)["workflows"][-1]["id"]),
            db=test_db
        )

        # Count should not include deleted workflows
        count = workflow_service.get_workflow_count(db=test_db)
        assert count == 2

    def test_update_execution_count(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test updating execution count"""
        # Create workflow
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )
        workflow = workflow_service.create_workflow(
            name="Workflow",
            description="For execution counting",
            react_flow_data=react_flow_data,
            db=test_db
        )

        # Initially no executions
        assert workflow.execution_count == 0

        # Update execution count
        workflow_service.update_execution_count(workflow.id, db=test_db)

        # Verify count increased
        updated = test_db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow.id).first()
        assert updated.execution_count == 1
        assert updated.last_executed is not None


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_handle_database_errors(self, workflow_service, test_db):
        """Test handling of database errors"""
        # Mock database session to raise exception
        with patch.object(test_db, 'query', side_effect=Exception("Database connection error")):
            result = workflow_service.list_workflows(db=test_db)
            assert result["total"] == 0
            assert result["workflows"] == []

    def test_handle_invalid_reactflow_data(self, workflow_service, test_db):
        """Test handling of invalid ReactFlow data"""
        # Try to create workflow with invalid ReactFlow data
        invalid_nodes = [
            ReactFlowNode(id="", type="input", position={"x": 0, "y": 0}, data={"label": "Invalid"})  # Empty ID
        ]
        invalid_edges = []
        react_flow_data = ReactFlowData(nodes=invalid_nodes, edges=invalid_edges)

        with pytest.raises(Exception):  # Should raise validation error
            workflow_service.create_workflow(
                name="Invalid Workflow",
                description="Has invalid ReactFlow data",
                react_flow_data=react_flow_data,
                db=test_db
            )

    def test_concurrent_workflow_creation(self, workflow_service, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test concurrent workflow creation"""
        import asyncio
        import threading

        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        results = []

        def create_workflow(name):
            try:
                workflow = workflow_service.create_workflow(
                    name=name,
                    description="Concurrent creation",
                    react_flow_data=react_flow_data,
                    db=test_db
                )
                results.append(workflow)
            except Exception as e:
                results.append(e)

        # Create multiple workflows concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_workflow, args=[f"Concurrent Workflow {i}"])
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all workflows were created successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, WorkflowDefinition)

        # Verify all workflows exist in database
        count = workflow_service.get_workflow_count(db=test_db)
        assert count == 5


class TestReactFlowCompatibility:
    """Test ReactFlow compatibility and conversions"""

    def test_convert_to_reactflow_format(self, workflow_service, sample_reactflow_nodes, sample_reactflow_edges):
        """Test converting internal format to ReactFlow"""
        internal_nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
        internal_edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

        reactflow_data = workflow_service._convert_to_reactflow_format(internal_nodes, internal_edges)

        assert "nodes" in reactflow_data
        assert "edges" in reactflow_data
        assert len(reactflow_data["nodes"]) == 3
        assert len(reactflow_data["edges"]) == 2
        assert reactflow_data["nodes"][0]["id"] == "node1"
        assert reactflow_data["edges"][0]["source"] == "node1"

    def test_convert_from_reactflow_format(self, workflow_service, sample_reactflow_nodes, sample_reactflow_edges):
        """Test converting ReactFlow to internal format"""
        reactflow_data = {
            "nodes": sample_reactflow_nodes,
            "edges": sample_reactflow_edges
        }

        nodes, edges = workflow_service._convert_from_reactflow_format(reactflow_data)

        assert len(nodes) == 3
        assert len(edges) == 2
        assert nodes[0].id == "node1"
        assert edges[0].source == "node1"