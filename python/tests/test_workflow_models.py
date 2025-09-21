"""
Tests for workflow database models and Pydantic schemas

Comprehensive test suite covering:
- Workflow definition models
- Execution models
- ReactFlow data compatibility
- Validation utilities
- Type safety and serialization
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution, WorkflowAnalytics,
    ReactFlowNode, ReactFlowEdge, ReactFlowData, ExecutionStatus, StepType,
    AgentType, ModelTier, WorkflowCreateRequest, WorkflowUpdateRequest,
    WorkflowExecuteRequest, ReactFlowCompatibility
)
from src.database import Base
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
def sample_reactflow_nodes():
    """Sample ReactFlow nodes for testing"""
    return [
        {
            "id": "node1",
            "type": "input",
            "position": {"x": 0, "y": 0},
            "data": {"label": "Start", "param1": "value1"}
        },
        {
            "id": "node2",
            "type": "agentTask",
            "position": {"x": 200, "y": 0},
            "data": {
                "label": "Process Data",
                "agent_type": "analyst",
                "model_tier": "standard",
                "prompt": "Analyze the following data"
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
    """Sample ReactFlow edges for testing"""
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
    """Sample workflow data for testing"""
    return {
        "name": "Test Workflow",
        "description": "A test workflow for validation",
        "variables": {"input_data": "default_value"},
        "tags": ["test", "validation"]
    }


class TestReactFlowModels:
    """Test ReactFlow data models"""

    def test_reactflow_node_creation(self, sample_reactflow_nodes):
        """Test ReactFlowNode model creation"""
        node_data = sample_reactflow_nodes[0]
        node = ReactFlowNode(**node_data)

        assert node.id == "node1"
        assert node.type == "input"
        assert node.position == {"x": 0, "y": 0}
        assert node.data == {"label": "Start", "param1": "value1"}

    def test_reactflow_edge_creation(self, sample_reactflow_edges):
        """Test ReactFlowEdge model creation"""
        edge_data = sample_reactflow_edges[0]
        edge = ReactFlowEdge(**edge_data)

        assert edge.id == "edge1"
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.type == "default"

    def test_reactflow_data_creation(self, sample_reactflow_nodes, sample_reactflow_edges):
        """Test ReactFlowData model creation"""
        nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
        edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

        data = ReactFlowData(nodes=nodes, edges=edges)

        assert len(data.nodes) == 3
        assert len(data.edges) == 2
        assert data.nodes[0].id == "node1"
        assert data.edges[0].source == "node1"

    def test_reactflow_data_serialization(self, sample_reactflow_nodes, sample_reactflow_edges):
        """Test ReactFlowData serialization"""
        nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
        edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

        data = ReactFlowData(nodes=nodes, edges=edges)
        serialized = data.model_dump()

        assert "nodes" in serialized
        assert "edges" in serialized
        assert len(serialized["nodes"]) == 3
        assert len(serialized["edges"]) == 2


class TestWorkflowDefinition:
    """Test WorkflowDefinition model"""

    def test_workflow_creation(self, test_db, sample_reactflow_nodes, sample_reactflow_edges, sample_workflow_data):
        """Test workflow definition creation"""
        nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
        edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

        workflow = WorkflowDefinition(
            name=sample_workflow_data["name"],
            description=sample_workflow_data["description"],
            variables=sample_workflow_data["variables"],
            tags=sample_workflow_data["tags"],
            nodes=nodes,
            edges=edges
        )

        test_db.add(workflow)
        test_db.commit()

        assert workflow.id is not None
        assert workflow.name == "Test Workflow"
        assert workflow.status == "draft"
        assert workflow.version == 1
        assert len(workflow.nodes) == 3
        assert len(workflow.edges) == 2
        assert workflow.created_at is not None

    def test_workflow_update(self, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test workflow update"""
        # Create workflow
        nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
        edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

        workflow = WorkflowDefinition(
            name="Original Name",
            description="Original description",
            nodes=nodes,
            edges=edges
        )
        test_db.add(workflow)
        test_db.commit()

        # Update workflow
        workflow.name = "Updated Name"
        workflow.description = "Updated description"
        workflow.status = "active"
        workflow.version = 2
        test_db.commit()

        updated_workflow = test_db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow.id).first()
        assert updated_workflow.name == "Updated Name"
        assert updated_workflow.description == "Updated description"
        assert updated_workflow.status == "active"
        assert updated_workflow.version == 2

    def test_workflow_soft_delete(self, test_db, sample_reactflow_nodes, sample_reactflow_edges):
        """Test workflow soft delete"""
        # Create workflow
        nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
        edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

        workflow = WorkflowDefinition(
            name="To Delete",
            description="Will be deleted",
            nodes=nodes,
            edges=edges
        )
        test_db.add(workflow)
        test_db.commit()

        # Soft delete
        workflow.deleted_at = datetime.now()
        test_db.commit()

        # Check it's marked as deleted but still in database
        deleted_workflow = test_db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow.id).first()
        assert deleted_workflow is not None
        assert deleted_workflow.deleted_at is not None


class TestWorkflowExecution:
    """Test WorkflowExecution model"""

    def test_execution_creation(self, test_db):
        """Test workflow execution creation"""
        # Create workflow first
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For execution testing"
        )
        test_db.add(workflow)
        test_db.commit()

        # Create execution
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING,
            inputs={"test_input": "test_value"},
            started_at=datetime.now()
        )
        test_db.add(execution)
        test_db.commit()

        assert execution.id is not None
        assert execution.workflow_id == workflow.id
        assert execution.status == ExecutionStatus.RUNNING
        assert execution.inputs == {"test_input": "test_value"}
        assert execution.started_at is not None
        assert execution.completed_at is None

    def test_execution_completion(self, test_db):
        """Test execution completion"""
        # Create workflow and execution
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For execution testing"
        )
        test_db.add(workflow)
        test_db.commit()

        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now()
        )
        test_db.add(execution)
        test_db.commit()

        # Complete execution
        execution.status = ExecutionStatus.COMPLETED
        execution.completed_at = datetime.now()
        execution.results = {"output": "success"}
        execution.metrics = {"duration": 5.2, "steps_executed": 3}
        test_db.commit()

        completed_execution = test_db.query(WorkflowExecution).filter(WorkflowExecution.id == execution.id).first()
        assert completed_execution.status == ExecutionStatus.COMPLETED
        assert completed_execution.completed_at is not None
        assert completed_execution.results == {"output": "success"}
        assert completed_execution.metrics == {"duration": 5.2, "steps_executed": 3}

    def test_execution_failure(self, test_db):
        """Test execution failure"""
        # Create workflow and execution
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For execution testing"
        )
        test_db.add(workflow)
        test_db.commit()

        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now()
        )
        test_db.add(execution)
        test_db.commit()

        # Fail execution
        execution.status = ExecutionStatus.FAILED
        execution.completed_at = datetime.now()
        execution.errors = ["Step 1 failed: timeout error"]
        test_db.commit()

        failed_execution = test_db.query(WorkflowExecution).filter(WorkflowExecution.id == execution.id).first()
        assert failed_execution.status == ExecutionStatus.FAILED
        assert failed_execution.completed_at is not None
        assert failed_execution.errors == ["Step 1 failed: timeout error"]


class TestStepExecution:
    """Test StepExecution model"""

    def test_step_creation(self, test_db):
        """Test step execution creation"""
        # Create workflow execution first
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For step testing"
        )
        test_db.add(workflow)
        test_db.commit()

        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING
        )
        test_db.add(execution)
        test_db.commit()

        # Create step execution
        step = StepExecution(
            execution_id=execution.id,
            step_id="step1",
            step_name="Test Step",
            step_type=StepType.AGENT_TASK,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(),
            inputs={"param1": "value1"}
        )
        test_db.add(step)
        test_db.commit()

        assert step.id is not None
        assert step.execution_id == execution.id
        assert step.step_id == "step1"
        assert step.step_name == "Test Step"
        assert step.step_type == StepType.AGENT_TASK
        assert step.status == ExecutionStatus.RUNNING
        assert step.inputs == {"param1": "value1"}

    def test_step_completion(self, test_db):
        """Test step execution completion"""
        # Create workflow execution and step
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For step testing"
        )
        test_db.add(workflow)
        test_db.commit()

        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING
        )
        test_db.add(execution)
        test_db.commit()

        step = StepExecution(
            execution_id=execution.id,
            step_id="step1",
            step_name="Test Step",
            status=ExecutionStatus.RUNNING
        )
        test_db.add(step)
        test_db.commit()

        # Complete step
        step.status = ExecutionStatus.COMPLETED
        step.completed_at = datetime.now()
        step.result = {"output": "step completed"}
        step.metrics = {"duration": 1.5, "tokens_used": 250}
        test_db.commit()

        completed_step = test_db.query(StepExecution).filter(StepExecution.id == step.id).first()
        assert completed_step.status == ExecutionStatus.COMPLETED
        assert completed_step.completed_at is not None
        assert completed_step.result == {"output": "step completed"}
        assert completed_step.metrics == {"duration": 1.5, "tokens_used": 250}


class TestWorkflowAnalytics:
    """Test WorkflowAnalytics model"""

    def test_analytics_creation(self, test_db):
        """Test analytics creation"""
        # Create workflow first
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For analytics testing"
        )
        test_db.add(workflow)
        test_db.commit()

        # Create analytics
        analytics = WorkflowAnalytics(
            workflow_id=workflow.id,
            date=datetime.now().date(),
            executions_count=10,
            successful_executions=8,
            failed_executions=2,
            average_duration=45.5,
            total_tokens=5000,
            total_cost=0.25,
            performance_metrics={"throughput": 2.0, "error_rate": 0.2}
        )
        test_db.add(analytics)
        test_db.commit()

        assert analytics.id is not None
        assert analytics.workflow_id == workflow.id
        assert analytics.executions_count == 10
        assert analytics.successful_executions == 8
        assert analytics.failed_executions == 2
        assert analytics.average_duration == 45.5
        assert analytics.total_tokens == 5000
        assert analytics.total_cost == 0.25


class TestPydanticSchemas:
    """Test Pydantic request/response schemas"""

    def test_workflow_create_request(self, sample_reactflow_nodes, sample_reactflow_edges):
        """Test WorkflowCreateRequest schema"""
        request_data = {
            "name": "New Workflow",
            "description": "A new workflow",
            "nodes": sample_reactflow_nodes,
            "edges": sample_reactflow_edges,
            "variables": {"input": "default"},
            "tags": ["new", "workflow"]
        }

        request = WorkflowCreateRequest(**request_data)

        assert request.name == "New Workflow"
        assert request.description == "A new workflow"
        assert len(request.nodes) == 3
        assert len(request.edges) == 2
        assert request.variables == {"input": "default"}
        assert request.tags == ["new", "workflow"]

    def test_workflow_update_request(self):
        """Test WorkflowUpdateRequest schema"""
        request_data = {
            "name": "Updated Workflow",
            "description": "Updated description",
            "variables": {"new_input": "new_value"},
            "status": "active"
        }

        request = WorkflowUpdateRequest(**request_data)

        assert request.name == "Updated Workflow"
        assert request.description == "Updated description"
        assert request.variables == {"new_input": "new_value"}
        assert request.status == "active"

    def test_workflow_execute_request(self):
        """Test WorkflowExecuteRequest schema"""
        request_data = {
            "workflow_id": str(uuid4()),
            "inputs": {"param1": "value1", "param2": "value2"},
            "async_execution": True,
            "timeout": 300
        }

        request = WorkflowExecuteRequest(**request_data)

        assert request.inputs == {"param1": "value1", "param2": "value2"}
        assert request.async_execution is True
        assert request.timeout == 300


class TestReactFlowCompatibility:
    """Test ReactFlow compatibility utilities"""

    def test_convert_to_reactflow_format(self, sample_reactflow_nodes, sample_reactflow_edges):
        """Test converting internal format to ReactFlow"""
        from src.database.workflow_models import convert_to_reactflow_format

        internal_nodes = [ReactFlowNode(**node) for node in sample_reactflow_nodes]
        internal_edges = [ReactFlowEdge(**edge) for edge in sample_reactflow_edges]

        reactflow_data = convert_to_reactflow_format(internal_nodes, internal_edges)

        assert "nodes" in reactflow_data
        assert "edges" in reactflow_data
        assert len(reactflow_data["nodes"]) == 3
        assert len(reactflow_data["edges"]) == 2
        assert reactflow_data["nodes"][0]["id"] == "node1"
        assert reactflow_data["edges"][0]["source"] == "node1"

    def test_convert_from_reactflow_format(self, sample_reactflow_nodes, sample_reactflow_edges):
        """Test converting ReactFlow to internal format"""
        from src.database.workflow_models import convert_from_reactflow_format

        reactflow_data = {
            "nodes": sample_reactflow_nodes,
            "edges": sample_reactflow_edges
        }

        nodes, edges = convert_from_reactflow_format(reactflow_data)

        assert len(nodes) == 3
        assert len(edges) == 2
        assert nodes[0].id == "node1"
        assert edges[0].source == "node1"

    def test_validate_reactflow_data_valid(self, sample_reactflow_nodes, sample_reactflow_edges):
        """Test validation of valid ReactFlow data"""
        from src.database.workflow_models import validate_reactflow_data

        reactflow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        is_valid, errors = validate_reactflow_data(reactflow_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_reactflow_data_invalid_edges(self, sample_reactflow_nodes):
        """Test validation of ReactFlow data with invalid edges"""
        from src.database.workflow_models import validate_reactflow_data

        # Create edge with non-existent source
        invalid_edge = ReactFlowEdge(
            id="invalid_edge",
            source="nonexistent_node",
            target="node2"
        )

        reactflow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in sample_reactflow_nodes],
            edges=[invalid_edge]
        )

        is_valid, errors = validate_reactflow_data(reactflow_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("source node" in error for error in errors)

    def test_validate_reactflow_data_duplicate_nodes(self, sample_reactflow_edges):
        """Test validation of ReactFlow data with duplicate node IDs"""
        from src.database.workflow_models import validate_reactflow_data

        # Create nodes with duplicate IDs
        nodes = [
            ReactFlowNode(id="duplicate", type="input", position={"x": 0, "y": 0}, data={"label": "Node 1"}),
            ReactFlowNode(id="duplicate", type="output", position={"x": 100, "y": 0}, data={"label": "Node 2"})
        ]

        reactflow_data = ReactFlowData(
            nodes=nodes,
            edges=[ReactFlowEdge(**edge) for edge in sample_reactflow_edges]
        )

        is_valid, errors = validate_reactflow_data(reactflow_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("duplicate" in error.lower() for error in errors)


class TestModelRelationships:
    """Test model relationships"""

    def test_workflow_execution_relationship(self, test_db):
        """Test workflow-execution relationship"""
        # Create workflow
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For relationship testing"
        )
        test_db.add(workflow)
        test_db.commit()

        # Create multiple executions
        execution1 = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now()
        )
        execution2 = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now()
        )

        test_db.add(execution1)
        test_db.add(execution2)
        test_db.commit()

        # Query workflow with executions
        workflow_with_executions = test_db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow.id).first()
        assert len(workflow_with_executions.executions) == 2

    def test_execution_steps_relationship(self, test_db):
        """Test execution-steps relationship"""
        # Create workflow and execution
        workflow = WorkflowDefinition(
            name="Test Workflow",
            description="For relationship testing"
        )
        test_db.add(workflow)
        test_db.commit()

        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING
        )
        test_db.add(execution)
        test_db.commit()

        # Create multiple steps
        step1 = StepExecution(
            execution_id=execution.id,
            step_id="step1",
            step_name="Step 1",
            status=ExecutionStatus.COMPLETED
        )
        step2 = StepExecution(
            execution_id=execution.id,
            step_id="step2",
            step_name="Step 2",
            status=ExecutionStatus.RUNNING
        )

        test_db.add(step1)
        test_db.add(step2)
        test_db.commit()

        # Query execution with steps
        execution_with_steps = test_db.query(WorkflowExecution).filter(WorkflowExecution.id == execution.id).first()
        assert len(execution_with_steps.step_executions) == 2


class TestEnumValues:
    """Test enum values and constraints"""

    def test_execution_status_values(self):
        """Test ExecutionStatus enum values"""
        assert ExecutionStatus.RUNNING == "running"
        assert ExecutionStatus.COMPLETED == "completed"
        assert ExecutionStatus.FAILED == "failed"
        assert ExecutionStatus.CANCELLED == "cancelled"
        assert ExecutionStatus.PAUSED == "paused"

    def test_step_type_values(self):
        """Test StepType enum values"""
        assert StepType.AGENT_TASK == "agent_task"
        assert StepType.API_CALL == "api_call"
        assert StepType.CONDITION == "condition"
        assert StepType.DELAY == "delay"
        assert StepType.NOTIFICATION == "notification"

    def test_agent_type_values(self):
        """Test AgentType enum values"""
        assert AgentType.ANALYST == "analyst"
        assert AgentType.DEVELOPER == "developer"
        assert AgentType.DESIGNER == "designer"
        assert AgentType.TESTER == "tester"
        assert AgentType.MANAGER == "manager"

    def test_model_tier_values(self):
        """Test ModelTier enum values"""
        assert ModelTier.BASIC == "basic"
        assert ModelTier.STANDARD == "standard"
        assert ModelTier.ADVANCED == "advanced"
        assert ModelTier.EXPERT == "expert"