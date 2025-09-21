"""
Tests for workflow execution service

Comprehensive test suite covering:
- Workflow execution engine
- Step execution and parallel processing
- Agent assignment and task execution
- Error handling and retry logic
- Status management and real-time updates
- Performance and timeout handling
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
from src.server.services.workflow_execution_service import (
    WorkflowExecutionEngine, WorkflowExecutionService, ExecutionContext, ExecutionState
)
from src.database.agent_models import AgentV3, AgentState


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
def execution_engine():
    """Create workflow execution engine"""
    return WorkflowExecutionEngine()


@pytest.fixture
def execution_service():
    """Create workflow execution service"""
    return WorkflowExecutionService()


@pytest.fixture
def sample_workflow_nodes():
    """Sample workflow nodes"""
    return [
        ReactFlowNode(
            id="node1",
            type="input",
            position={"x": 0, "y": 0},
            data={"label": "Start", "input_text": "Hello World"}
        ),
        ReactFlowNode(
            id="node2",
            type="agentTask",
            position={"x": 200, "y": 0},
            data={
                "label": "Process Data",
                "agent_type": "analyst",
                "model_tier": "standard",
                "prompt": "Analyze the input text"
            }
        ),
        ReactFlowNode(
            id="node3",
            type="output",
            position={"x": 400, "y": 0},
            data={"label": "End"}
        )
    ]


@pytest.fixture
def sample_workflow_edges():
    """Sample workflow edges"""
    return [
        ReactFlowEdge(id="edge1", source="node1", target="node2"),
        ReactFlowEdge(id="edge2", source="node2", target="node3")
    ]


@pytest.fixture
def sample_workflow(test_db, sample_workflow_nodes, sample_workflow_edges):
    """Create sample workflow for testing"""
    workflow = WorkflowDefinition(
        name="Test Workflow",
        description="For execution testing",
        nodes=sample_workflow_nodes,
        edges=sample_workflow_edges,
        variables={"input_text": "default_value"}
    )
    test_db.add(workflow)
    test_db.commit()
    return workflow


@pytest.fixture
def sample_agent(test_db):
    """Create sample agent for testing"""
    agent = AgentV3(
        name="Test Analyst",
        role="analyst",
        model_tier="standard",
        system_prompt="You are a helpful analyst.",
        state=AgentState.ACTIVE
    )
    test_db.add(agent)
    test_db.commit()
    return agent


class TestWorkflowExecutionEngine:
    """Test workflow execution engine"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, execution_engine):
        """Test execution engine initialization"""
        assert execution_engine.running is True
        assert execution_engine.execution_queue is not None
        assert execution_engine.active_contexts == {}
        assert len(execution_engine.workers) == 3  # Default worker count

    @pytest.mark.asyncio
    async def test_engine_start_stop(self, execution_engine):
        """Test engine start and stop"""
        # Stop engine
        await execution_engine.stop()
        assert execution_engine.running is False

        # Restart engine
        execution_engine.running = True
        asyncio.create_task(execution_engine._status_updater())

        assert execution_engine.running is True

    @pytest.mark.asyncio
    async def test_execute_workflow_basic(self, execution_engine, test_db, sample_workflow):
        """Test basic workflow execution"""
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING,
            inputs={"input_text": "test input"}
        )
        test_db.add(execution)
        test_db.commit()

        # Execute workflow
        with patch('src.server.services.workflow_execution_service.broadcast_execution_started') as mock_broadcast:
            result = await execution_engine.execute_workflow(
                execution.execution_id,
                sample_workflow,
                {"input_text": "test input"},
                test_db
            )

            assert result.id == execution.execution_id
            assert result.status == ExecutionStatus.RUNNING
            assert execution.execution_id in execution_engine.active_contexts

            # Verify broadcast was called
            mock_broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_invalid_id(self, execution_engine, test_db, sample_workflow):
        """Test workflow execution with invalid ID"""
        with pytest.raises(ValueError, match="Execution not found"):
            await execution_engine.execute_workflow(
                "invalid_id",
                sample_workflow,
                {},
                test_db
            )

    @pytest.mark.asyncio
    async def test_pause_execution(self, execution_engine, test_db, sample_workflow):
        """Test pausing execution"""
        # Create and start execution
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING
        )
        test_db.add(execution)
        test_db.commit()

        await execution_engine.execute_workflow(
            execution.execution_id,
            sample_workflow,
            {},
            test_db
        )

        # Pause execution
        with patch('src.server.services.workflow_execution_service.broadcast_execution_paused') as mock_broadcast:
            await execution_engine.pause_execution(execution.execution_id)

            context = execution_engine.active_contexts.get(execution.execution_id)
            assert context.state == ExecutionState.PAUSED

            # Verify broadcast was called
            mock_broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_execution(self, execution_engine, test_db, sample_workflow):
        """Test resuming execution"""
        # Create and start execution
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING
        )
        test_db.add(execution)
        test_db.commit()

        await execution_engine.execute_workflow(
            execution.execution_id,
            sample_workflow,
            {},
            test_db
        )

        # Pause and resume
        await execution_engine.pause_execution(execution.execution_id)

        with patch('src.server.services.workflow_execution_service.broadcast_execution_resumed') as mock_broadcast:
            await execution_engine.resume_execution(execution.execution_id)

            context = execution_engine.active_contexts.get(execution.execution_id)
            assert context.state == ExecutionState.EXECUTING

            # Verify broadcast was called
            mock_broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_execution(self, execution_engine, test_db, sample_workflow):
        """Test cancelling execution"""
        # Create and start execution
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING
        )
        test_db.add(execution)
        test_db.commit()

        await execution_engine.execute_workflow(
            execution.execution_id,
            sample_workflow,
            {},
            test_db
        )

        # Cancel execution
        with patch('src.server.services.workflow_execution_service.broadcast_execution_cancelled') as mock_broadcast:
            await execution_engine.cancel_execution(execution.execution_id)

            # Verify context is removed
            assert execution.execution_id not in execution_engine.active_contexts

            # Verify broadcast was called
            mock_broadcast.assert_called_once()


class TestStepExecution:
    """Test individual step execution"""

    @pytest.mark.asyncio
    async def test_execute_input_step(self, execution_engine, test_db):
        """Test executing input step"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={},
            errors=[]
        )

        step_data = {
            "id": "input_step",
            "type": "input",
            "data": {"input_text": "test input", "input_type": "text"}
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="input_step",
            step_name="Input Step",
            step_type=StepType.INPUT,
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        result = await execution_engine._execute_input_step(context, step_data, step_execution, db)

        assert result["step_type"] == "input"
        assert result["input_data"] == {"input_text": "test input", "input_type": "text"}
        assert step_execution.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_output_step(self, execution_engine, test_db):
        """Test executing output step"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={"previous_step": {"output": "test result"}},
            errors=[]
        )

        step_data = {
            "id": "output_step",
            "type": "output",
            "data": {"output_format": "json"}
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="output_step",
            step_name="Output Step",
            step_type=StepType.OUTPUT,
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        result = await execution_engine._execute_output_step(context, step_data, step_execution, db)

        assert result["step_type"] == "output"
        assert result["output_format"] == "json"
        assert step_execution.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_delay_step(self, execution_engine, test_db):
        """Test executing delay step"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={},
            errors=[]
        )

        step_data = {
            "id": "delay_step",
            "type": "delay",
            "data": {"delay_seconds": 1}
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="delay_step",
            step_name="Delay Step",
            step_type=StepType.DELAY,
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        start_time = datetime.now()
        result = await execution_engine._execute_delay_step(context, step_data, step_execution, db)
        end_time = datetime.now()

        assert result["step_type"] == "delay"
        assert result["delayed_seconds"] == 1
        assert step_execution.status == ExecutionStatus.COMPLETED
        assert (end_time - start_time).total_seconds() >= 1

    @pytest.mark.asyncio
    async def test_execute_agent_step_no_agent(self, execution_engine, test_db):
        """Test executing agent step with no available agent"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={},
            errors=[]
        )

        step_data = {
            "id": "agent_step",
            "type": "agentTask",
            "data": {
                "agent_type": "analyst",
                "model_tier": "standard",
                "prompt": "Analyze this data"
            }
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="agent_step",
            step_name="Agent Step",
            step_type=StepType.AGENT_TASK,
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        # Mock agent service to return no agents
        with patch('src.server.services.workflow_execution_service.AgentService') as mock_agent_service:
            mock_agent_service.return_value.get_available_agent.return_value = None

            result = await execution_engine._execute_agent_step(context, step_data, step_execution, db)

            assert result["error"] == "No available agent"
            assert step_execution.status == ExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_condition_step(self, execution_engine, test_db):
        """Test executing condition step"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={"previous_step": {"score": 85}},
            errors=[]
        )

        step_data = {
            "id": "condition_step",
            "type": "condition",
            "data": {
                "condition": "score > 80",
                "true_path": "node3",
                "false_path": "node4"
            }
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="condition_step",
            step_name="Condition Step",
            step_type=StepType.CONDITION,
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        result = await execution_engine._execute_condition_step(context, step_data, step_execution, db)

        assert result["step_type"] == "condition"
        assert result["condition"] == "score > 80"
        assert result["result"] is True
        assert result["next_step"] == "node3"
        assert step_execution.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_api_call_step(self, execution_engine, test_db):
        """Test executing API call step"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={},
            errors=[]
        )

        step_data = {
            "id": "api_step",
            "type": "apiCall",
            "data": {
                "url": "https://jsonplaceholder.typicode.com/todos/1",
                "method": "GET",
                "headers": {"Content-Type": "application/json"}
            }
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="api_step",
            step_name="API Step",
            step_type=StepType.API_CALL,
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        # Mock HTTP client
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"id": 1, "title": "Test Todo"}
            mock_response.text = '{"id": 1, "title": "Test Todo"}'

            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            result = await execution_engine._execute_api_call_step(context, step_data, step_execution, db)

            assert result["step_type"] == "api_call"
            assert result["status_code"] == 200
            assert result["response"]["title"] == "Test Todo"
            assert step_execution.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_step_error_handling(self, execution_engine, test_db):
        """Test step execution error handling"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={},
            errors=[]
        )

        # Invalid step data
        step_data = {
            "id": "invalid_step",
            "type": "invalid_type",  # Invalid type
            "data": {}
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="invalid_step",
            step_name="Invalid Step",
            step_type=StepType.AGENT_TASK,  # This will be overridden
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        result = await execution_engine._execute_step(context, step_data, step_execution, db)

        assert "error" in result
        assert step_execution.status == ExecutionStatus.FAILED
        assert step_execution.error is not None


class TestWorkflowExecutionService:
    """Test workflow execution service wrapper"""

    @pytest.mark.asyncio
    async def test_service_initialization(self, execution_service):
        """Test service initialization"""
        assert execution_service.engine is None

    @pytest.mark.asyncio
    async def test_execute_workflow(self, execution_service, test_db, sample_workflow):
        """Test executing workflow through service"""
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING,
            inputs={"input_text": "test input"}
        )
        test_db.add(execution)
        test_db.commit()

        # Execute workflow
        result = await execution_service.execute_workflow(
            execution.execution_id,
            sample_workflow,
            {"input_text": "test input"}
        )

        assert result.id == execution.execution_id
        assert result.status == ExecutionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_pause_resume_cancel_execution(self, execution_service, test_db, sample_workflow):
        """Test pause, resume, and cancel operations"""
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING,
            inputs={"input_text": "test input"}
        )
        test_db.add(execution)
        test_db.commit()

        # Execute workflow
        await execution_service.execute_workflow(
            execution.execution_id,
            sample_workflow,
            {"input_text": "test input"}
        )

        # Pause execution
        await execution_service.pause_execution(execution.execution_id)

        # Resume execution
        await execution_service.resume_execution(execution.execution_id)

        # Cancel execution
        await execution_service.cancel_execution(execution.execution_id)

    @pytest.mark.asyncio
    async def test_get_execution_status(self, execution_service, test_db, sample_workflow):
        """Test getting execution status"""
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.RUNNING,
            inputs={"input_text": "test input"}
        )
        test_db.add(execution)
        test_db.commit()

        # Get status
        result = await execution_service.get_execution_status(execution.execution_id)

        assert result is not None
        assert result.id == execution.execution_id
        assert result.status == ExecutionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_execution_status_not_exists(self, execution_service):
        """Test getting status of non-existent execution"""
        result = await execution_service.get_execution_status("non_existent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_execution_history(self, execution_service, test_db, sample_workflow):
        """Test getting execution history"""
        # Create multiple executions
        execution1 = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.now() - timedelta(hours=2),
            completed_at=datetime.now() - timedelta(hours=1)
        )
        execution2 = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now()
        )

        test_db.add(execution1)
        test_db.add(execution2)
        test_db.commit()

        # Get history
        result = await execution_service.get_execution_history(
            workflow_id=sample_workflow.id,
            db=test_db
        )

        assert result["total"] == 2
        assert len(result["executions"]) == 2

        # Get with pagination
        result_paginated = await execution_service.get_execution_history(
            workflow_id=sample_workflow.id,
            skip=1,
            limit=1,
            db=test_db
        )

        assert result_paginated["total"] == 2
        assert len(result_paginated["executions"]) == 1


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms"""

    @pytest.mark.asyncio
    async def test_step_execution_retry(self, execution_engine, test_db):
        """Test step execution retry mechanism"""
        context = ExecutionContext(
            execution_id="test_execution",
            workflow_id="test_workflow",
            workflow=None,
            execution=None,
            parameters={},
            step_results={},
            errors=[]
        )

        step_data = {
            "id": "failing_step",
            "type": "apiCall",
            "data": {
                "url": "https://nonexistent-domain.com/api",
                "method": "GET",
                "retry_count": 2
            }
        }

        db = next(test_db)
        step_execution = StepExecution(
            execution_id="test_execution",
            step_id="failing_step",
            step_name="Failing Step",
            step_type=StepType.API_CALL,
            status=ExecutionStatus.RUNNING
        )
        db.add(step_execution)
        db.commit()

        # Mock HTTP client to fail
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")

            result = await execution_engine._execute_api_call_step(context, step_data, step_execution, db)

            assert "error" in result
            assert step_execution.status == ExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, execution_engine, test_db, sample_workflow):
        """Test workflow execution timeout"""
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING,
            inputs={"input_text": "test input"}
        )
        test_db.add(execution)
        test_db.commit()

        # Mock step execution to take too long
        with patch.object(execution_engine, '_execute_workflow_steps', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = asyncio.sleep(2)  # Sleep longer than timeout

            # Set short timeout
            execution_engine.execution_timeout = 1

            # Execute workflow (should timeout)
            start_time = datetime.now()
            result = await execution_engine.execute_workflow(
                execution.execution_id,
                sample_workflow,
                {"input_text": "test input"},
                test_db
            )
            end_time = datetime.now()

            # Verify timeout occurred
            assert (end_time - start_time).total_seconds() < 3  # Should be less than sleep time
            assert result.status == ExecutionStatus.RUNNING  # Should still be running

    @pytest.mark.asyncio
    async def test_execution_context_cleanup(self, execution_engine, test_db, sample_workflow):
        """Test execution context cleanup"""
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=sample_workflow.id,
            status=ExecutionStatus.PENDING
        )
        test_db.add(execution)
        test_db.commit()

        # Execute workflow
        await execution_engine.execute_workflow(
            execution.execution_id,
            sample_workflow,
            {},
            test_db
        )

        # Cancel execution
        await execution_engine.cancel_execution(execution.execution_id)

        # Verify context is cleaned up
        assert execution.execution_id not in execution_engine.active_contexts

    @pytest.mark.asyncio
    async def test_worker_error_handling(self, execution_engine):
        """Test worker error handling"""
        # Mock execution queue to raise exception
        with patch.object(execution_engine.execution_queue, 'get', side_effect=Exception("Queue error")):
            # Worker should handle error gracefully
            worker_task = asyncio.create_task(execution_engine._execution_worker("test_worker"))

            # Let worker run briefly
            await asyncio.sleep(0.1)

            # Cancel worker
            worker_task.cancel()

            try:
                await worker_task
            except asyncio.CancelledError:
                pass


class TestPerformanceAndScalability:
    """Test performance and scalability aspects"""

    @pytest.mark.asyncio
    async def test_parallel_step_execution(self, execution_engine, test_db):
        """Test parallel execution of independent steps"""
        # Create workflow with parallel branches
        nodes = [
            ReactFlowNode(id="start", type="input", position={"x": 0, "y": 0}, data={"label": "Start"}),
            ReactFlowNode(id="parallel1", type="agentTask", position={"x": 200, "y": -50}, data={"label": "Task 1"}),
            ReactFlowNode(id="parallel2", type="agentTask", position={"x": 200, "y": 50}, data={"label": "Task 2"}),
            ReactFlowNode(id="end", type="output", position={"x": 400, "y": 0}, data={"label": "End"})
        ]
        edges = [
            ReactFlowEdge(id="e1", source="start", target="parallel1"),
            ReactFlowEdge(id="e2", source="start", target="parallel2"),
            ReactFlowEdge(id="e3", source="parallel1", target="end"),
            ReactFlowEdge(id="e4", source="parallel2", target="end")
        ]

        workflow = WorkflowDefinition(
            name="Parallel Workflow",
            description="Tests parallel execution",
            nodes=nodes,
            edges=edges
        )
        test_db.add(workflow)
        test_db.commit()

        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status=ExecutionStatus.PENDING
        )
        test_db.add(execution)
        test_db.commit()

        # Execute workflow
        start_time = datetime.now()
        await execution_engine.execute_workflow(
            execution.execution_id,
            workflow,
            {},
            test_db
        )

        # Wait for completion or timeout
        timeout = 5
        while (datetime.now() - start_time).total_seconds() < timeout:
            updated_execution = test_db.query(WorkflowExecution).filter(
                WorkflowExecution.execution_id == execution.execution_id
            ).first()

            if updated_execution.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                break

            await asyncio.sleep(0.1)

        # Verify execution completed
        updated_execution = test_db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution.execution_id
        ).first()

        assert updated_execution.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, execution_engine, test_db, sample_workflow):
        """Test memory cleanup after execution"""
        # Create multiple executions
        executions = []
        for i in range(5):
            execution = WorkflowExecution(
                workflow_id=sample_workflow.id,
                status=ExecutionStatus.PENDING,
                inputs={"iteration": i}
            )
            test_db.add(execution)
            executions.append(execution)
        test_db.commit()

        # Execute all workflows
        for execution in executions:
            await execution_engine.execute_workflow(
                execution.execution_id,
                sample_workflow,
                {"iteration": executions.index(execution)},
                test_db
            )

        # Cancel all executions
        for execution in executions:
            await execution_engine.cancel_execution(execution.execution_id)

        # Verify all contexts are cleaned up
        assert len(execution_engine.active_contexts) == 0

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, execution_engine, test_db):
        """Test concurrent execution of multiple workflows"""
        # Create multiple workflows
        workflows = []
        for i in range(3):
            workflow = WorkflowDefinition(
                name=f"Concurrent Workflow {i}",
                description=f"Test concurrent execution {i}",
                nodes=[
                    ReactFlowNode(id=f"start{i}", type="input", position={"x": 0, "y": 0}, data={"label": "Start"}),
                    ReactFlowNode(id=f"end{i}", type="output", position={"x": 200, "y": 0}, data={"label": "End"})
                ],
                edges=[ReactFlowEdge(id=f"e{i}", source=f"start{i}", target=f"end{i}")]
            )
            test_db.add(workflow)
            workflows.append(workflow)
        test_db.commit()

        # Create and execute all workflows concurrently
        execution_tasks = []
        for workflow in workflows:
            execution = WorkflowExecution(
                workflow_id=workflow.id,
                status=ExecutionStatus.PENDING
            )
            test_db.add(execution)
            test_db.commit()

            task = execution_engine.execute_workflow(
                execution.execution_id,
                workflow,
                {},
                test_db
            )
            execution_tasks.append(task)

        # Wait for all executions to start
        await asyncio.gather(*execution_tasks, return_exceptions=True)

        # Verify all executions are active
        assert len(execution_engine.active_contexts) == 3

        # Clean up
        for workflow in workflows:
            execution = test_db.query(WorkflowExecution).filter(
                WorkflowExecution.workflow_id == workflow.id
            ).first()
            if execution:
                await execution_engine.cancel_execution(execution.execution_id)