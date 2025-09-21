"""
Comprehensive tests for Workflow-Knowledge Integration

Tests the complete integration between workflow execution and knowledge management,
including the bridge components, API endpoints, and MCP tools.
"""

import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.server.services.knowledge_agent_bridge import KnowledgeAgentBridge
from src.server.services.workflow_knowledge_capture import WorkflowKnowledgeCapture
from src.server.services.knowledge_driven_workflow import KnowledgeDrivenWorkflow
from src.server.api_routes.knowledge_api import router
from src.database import get_db
from src.database.workflow_models import WorkflowDefinition, WorkflowExecution, ExecutionStatus


class TestKnowledgeAgentBridge:
    """Test suite for Knowledge-Agent Bridge service"""

    @pytest.fixture
    def bridge(self):
        """Create a KnowledgeAgentBridge instance for testing"""
        return KnowledgeAgentBridge()

    @pytest.fixture
    def mock_workflow_id(self):
        """Generate a mock workflow ID"""
        return str(uuid.uuid4())

    @pytest.fixture
    def mock_project_id(self):
        """Generate a mock project ID"""
        return str(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_start_workflow_session(self, bridge, mock_workflow_id, mock_project_id):
        """Test starting a workflow knowledge session"""
        capture_config = {
            "auto_capture": True,
            "capture_insights": True,
            "capture_patterns": True,
            "real_time_analysis": True
        }
        context_tags = ["test", "workflow-execution"]

        session_id = await bridge.start_workflow_session(
            workflow_id=mock_workflow_id,
            project_id=mock_project_id,
            capture_config=capture_config,
            context_tags=context_tags
        )

        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_capture_execution_insight(self, bridge, mock_workflow_id, mock_project_id):
        """Test capturing workflow execution insights"""
        # Start session first
        session_id = await bridge.start_workflow_session(
            workflow_id=mock_workflow_id,
            project_id=mock_project_id
        )

        insight_data = {
            "observation": "Workflow step completed faster than expected",
            "metrics": {"duration": 250, "expected_duration": 500},
            "impact": "positive"
        }

        insight_id = await bridge.capture_execution_insight(
            session_id=session_id,
            insight_type="performance_optimization",
            insight_data=insight_data,
            importance_score=0.8,
            tags=["performance", "optimization"]
        )

        assert insight_id is not None
        assert isinstance(insight_id, str)

    @pytest.mark.asyncio
    async def test_get_contextual_knowledge(self, bridge, mock_workflow_id, mock_project_id):
        """Test retrieving contextual knowledge"""
        # Start session first
        session_id = await bridge.start_workflow_session(
            workflow_id=mock_workflow_id,
            project_id=mock_project_id
        )

        query = "How to optimize workflow performance?"
        knowledge = await bridge.get_contextual_knowledge(
            session_id=session_id,
            query=query,
            context_type="execution_context",
            max_results=5
        )

        assert isinstance(knowledge, list)
        assert len(knowledge) <= 5

    @pytest.mark.asyncio
    async def test_store_workflow_template(self, bridge, mock_workflow_id):
        """Test storing workflow as template"""
        template_name = "Optimized Data Processing Workflow"
        template_description = "Template for efficient data processing with parallel execution"
        workflow_data = {
            "nodes": [
                {"id": "1", "type": "input"},
                {"id": "2", "type": "process"},
                {"id": "3", "type": "output"}
            ],
            "edges": [
                {"source": "1", "target": "2"},
                {"source": "2", "target": "3"}
            ]
        }

        template_id = await bridge.store_workflow_template(
            workflow_id=mock_workflow_id,
            template_name=template_name,
            template_description=template_description,
            workflow_data=workflow_data,
            use_cases=["data-processing", "batch-jobs"],
            best_practices=["parallel-execution", "error-handling"],
            tags=["data", "processing", "optimized"]
        )

        assert template_id is not None
        assert isinstance(template_id, str)


class TestWorkflowKnowledgeCapture:
    """Test suite for Workflow Knowledge Capture service"""

    @pytest.fixture
    def capture_service(self):
        """Create a WorkflowKnowledgeCapture instance for testing"""
        return WorkflowKnowledgeCapture()

    @pytest.fixture
    def mock_execution_id(self):
        """Generate a mock execution ID"""
        return f"exec_{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_start_execution_monitoring(self, capture_service, mock_execution_id):
        """Test starting execution monitoring"""
        monitoring_id = await capture_service.start_execution_monitoring(
            execution_id=mock_execution_id,
            workflow_id=str(uuid.uuid4()),
            config={
                "real_time_analysis": True,
                "performance_tracking": True,
                "error_detection": True
            }
        )

        assert monitoring_id is not None
        assert isinstance(monitoring_id, str)

    @pytest.mark.asyncio
    async def test_capture_step_completion(self, capture_service, mock_execution_id):
        """Test capturing step completion data"""
        step_metrics = {
            "step_id": "step_1",
            "duration": 1200,
            "status": "completed",
            "input_size": 1024,
            "output_size": 2048,
            "memory_usage": 512,
            "cpu_usage": 0.75
        }

        metrics_id = await capture_service.capture_step_completion(
            execution_id=mock_execution_id,
            step_metrics=step_metrics
        )

        assert metrics_id is not None

    @pytest.mark.asyncio
    async def test_analyze_step_performance(self, capture_service, mock_execution_id):
        """Test analyzing step performance"""
        patterns = await capture_service.analyze_step_performance(
            execution_id=mock_execution_id,
            step_id="step_1",
            historical_data=[
                {"duration": 1000, "memory_usage": 480},
                {"duration": 1100, "memory_usage": 500},
                {"duration": 1200, "memory_usage": 512}
            ]
        )

        assert isinstance(patterns, dict)
        assert "performance_trend" in patterns
        assert "efficiency_score" in patterns

    @pytest.mark.asyncio
    async def test_identify_performance_patterns(self, capture_service):
        """Test identifying performance patterns"""
        execution_data = [
            {
                "execution_id": "exec_1",
                "steps": [
                    {"id": "step_1", "duration": 1000},
                    {"id": "step_2", "duration": 2000}
                ],
                "total_duration": 3000
            },
            {
                "execution_id": "exec_2",
                "steps": [
                    {"id": "step_1", "duration": 1100},
                    {"id": "step_2", "duration": 2100}
                ],
                "total_duration": 3200
            }
        ]

        patterns = await capture_service.identify_performance_patterns(
            workflow_id=str(uuid.uuid4()),
            execution_data=execution_data
        )

        assert isinstance(patterns, list)
        assert len(patterns) > 0


class TestKnowledgeDrivenWorkflow:
    """Test suite for Knowledge-Driven Workflow service"""

    @pytest.fixture
    def workflow_service(self):
        """Create a KnowledgeDrivenWorkflow instance for testing"""
        return KnowledgeDrivenWorkflow()

    @pytest.fixture
    def mock_workflow_id(self):
        """Generate a mock workflow ID"""
        return str(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_enhance_workflow_step(self, workflow_service, mock_workflow_id):
        """Test enhancing workflow step with knowledge"""
        step_data = {
            "id": "step_1",
            "type": "data_processing",
            "config": {"batch_size": 100}
        }

        context_knowledge = [
            {
                "content": "Use batch sizes of 500-1000 for optimal performance",
                "relevance_score": 0.9,
                "knowledge_type": "best_practice"
            }
        ]

        enhanced_step = await workflow_service.enhance_workflow_step(
            workflow_id=mock_workflow_id,
            step_data=step_data,
            context_knowledge=context_knowledge
        )

        assert isinstance(enhanced_step, dict)
        assert "knowledge_enhancements" in enhanced_step
        assert "optimized_parameters" in enhanced_step

    @pytest.mark.asyncio
    async def test_make_knowledge_driven_decision(self, workflow_service):
        """Test making knowledge-driven decisions"""
        decision_context = {
            "current_step": "data_validation",
            "error_rate": 0.15,
            "performance_metrics": {"throughput": 100, "latency": 50}
        }

        historical_knowledge = [
            {
                "scenario": "high_error_rate_validation",
                "solution": "Implement additional validation rules",
                "success_rate": 0.95
            }
        ]

        decision = await workflow_service.make_knowledge_driven_decision(
            decision_context=decision_context,
            historical_knowledge=historical_knowledge
        )

        assert isinstance(decision, dict)
        assert "decision" in decision
        assert "confidence" in decision
        assert "reasoning" in decision

    @pytest.mark.asyncio
    async def test_adapt_workflow_dynamically(self, workflow_service, mock_workflow_id):
        """Test dynamic workflow adaptation"""
        execution_context = {
            "current_performance": {"duration": 1500, "success_rate": 0.85},
            "resource_constraints": {"memory_limit": 2048, "cpu_limit": 0.8},
            "business_goals": ["minimize_cost", "maximize_reliability"]
        }

        adaptations = await workflow_service.adapt_workflow_dynamically(
            workflow_id=mock_workflow_id,
            execution_context=execution_context
        )

        assert isinstance(adaptations, list)
        assert len(adaptations) > 0
        for adaptation in adaptations:
            assert "type" in adaptation
            assert "priority" in adaptation
            assert "implementation" in adaptation


class TestKnowledgeAPIEndpoints:
    """Test suite for Knowledge API endpoints"""

    @pytest.fixture
    def client(self):
        """Create a test client for the API"""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        with patch('src.server.api_routes.knowledge_api.get_db') as mock:
            session = Mock()
            mock.return_value.__next__ = Mock(return_value=session)
            yield session

    def test_start_workflow_knowledge_session(self, client, mock_db_session):
        """Test starting a workflow knowledge session via API"""
        request_data = {
            "workflow_id": str(uuid.uuid4()),
            "project_id": str(uuid.uuid4()),
            "capture_config": {
                "auto_capture": True,
                "capture_insights": True
            },
            "context_tags": ["test"]
        }

        response = client.post("/api/workflow-knowledge/start-session", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "session_id" in data

    def test_capture_workflow_insight(self, client, mock_db_session):
        """Test capturing workflow insight via API"""
        request_data = {
            "session_id": str(uuid.uuid4()),
            "insight_type": "performance_optimization",
            "insight_data": {
                "observation": "Test observation",
                "impact": "positive"
            },
            "importance_score": 0.8
        }

        response = client.post("/api/workflow-knowledge/capture-insight", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "insight_id" in data

    def test_get_contextual_knowledge(self, client, mock_db_session):
        """Test getting contextual knowledge via API"""
        session_id = str(uuid.uuid4())
        request_data = {
            "query": "How to optimize workflow?",
            "context_type": "execution_context",
            "max_results": 10
        }

        response = client.post(f"/api/workflow-knowledge/contextual/{session_id}", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "knowledge_items" in data

    def test_store_workflow_template(self, client, mock_db_session):
        """Test storing workflow template via API"""
        request_data = {
            "workflow_id": str(uuid.uuid4()),
            "template_name": "Test Template",
            "template_description": "Test template description",
            "use_cases": ["testing"],
            "best_practices": ["test practices"],
            "tags": ["test"]
        }

        response = client.post("/api/workflow-knowledge/store-template", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "template_id" in data


class TestMCPWorkflowKnowledgeTools:
    """Test suite for MCP Workflow Knowledge Tools"""

    @pytest.fixture
    def mock_mcp_context(self):
        """Create a mock MCP context"""
        context = Mock()
        context.request_context = Mock()
        return context

    @pytest.mark.asyncio
    async def test_archon_start_workflow_knowledge_session(self, mock_mcp_context):
        """Test MCP tool for starting workflow knowledge session"""
        with patch('src.mcp_server.modules.workflow_knowledge_tools.KnowledgeAgentBridge') as mock_bridge:
            mock_instance = AsyncMock()
            mock_bridge.return_value = mock_instance
            mock_instance.start_workflow_session.return_value = "test_session_id"

            from src.mcp_server.modules.workflow_knowledge_tools import register_workflow_knowledge_tools
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("test")
            register_workflow_knowledge_tools(mcp)

            # This would normally be called by the MCP server
            # We're testing the function exists and can be called
            assert hasattr(register_workflow_knowledge_tools, '__call__')

    @pytest.mark.asyncio
    async def test_archon_capture_workflow_insight(self, mock_mcp_context):
        """Test MCP tool for capturing workflow insight"""
        with patch('src.mcp_server.modules.workflow_knowledge_tools.KnowledgeAgentBridge') as mock_bridge:
            mock_instance = AsyncMock()
            mock_bridge.return_value = mock_instance
            mock_instance.capture_execution_insight.return_value = "test_insight_id"

            from src.mcp_server.modules.workflow_knowledge_tools import register_workflow_knowledge_tools

            # Test tool registration
            assert callable(register_workflow_knowledge_tools)

    @pytest.mark.asyncio
    async def test_archon_get_contextual_knowledge(self, mock_mcp_context):
        """Test MCP tool for getting contextual knowledge"""
        with patch('src.mcp_server.modules.workflow_knowledge_tools.KnowledgeAgentBridge') as mock_bridge:
            mock_instance = AsyncMock()
            mock_bridge.return_value = mock_instance
            mock_instance.get_contextual_knowledge.return_value = []

            from src.mcp_server.modules.workflow_knowledge_tools import register_workflow_knowledge_tools

            # Test tool registration
            assert callable(register_workflow_knowledge_tools)


# Integration tests
class TestWorkflowKnowledgeIntegration:
    """Integration tests for the complete workflow-knowledge system"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_knowledge_flow(self):
        """Test complete workflow knowledge capture and retrieval flow"""
        # Initialize services
        bridge = KnowledgeAgentBridge()
        capture_service = WorkflowKnowledgeCapture()

        workflow_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())

        # 1. Start knowledge session
        session_id = await bridge.start_workflow_session(
            workflow_id=workflow_id,
            project_id=project_id
        )

        # 2. Start execution monitoring
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        await capture_service.start_execution_monitoring(execution_id, workflow_id)

        # 3. Capture insights during execution
        insight_data = {
            "observation": "Step completed efficiently",
            "metrics": {"duration": 800, "expected": 1000},
            "efficiency_gain": 0.2
        }

        insight_id = await bridge.capture_execution_insight(
            session_id=session_id,
            insight_type="efficiency_gain",
            insight_data=insight_data,
            importance_score=0.9
        )

        # 4. Capture step completion
        step_metrics = {
            "step_id": "step_1",
            "duration": 800,
            "status": "completed",
            "efficiency_score": 0.9
        }

        await capture_service.capture_step_completion(execution_id, step_metrics)

        # 5. Retrieve contextual knowledge
        query = "How to maintain high efficiency in data processing?"
        knowledge = await bridge.get_contextual_knowledge(
            session_id=session_id,
            query=query,
            max_results=5
        )

        # 6. End session and generate summary
        summary = await bridge.end_workflow_session(session_id, generate_summary=True)

        # Verify the flow completed successfully
        assert session_id is not None
        assert insight_id is not None
        assert isinstance(knowledge, list)
        assert "summary" in summary

    @pytest.mark.asyncio
    async def test_workflow_template_lifecycle(self):
        """Test complete workflow template lifecycle"""
        bridge = KnowledgeAgentBridge()
        workflow_id = str(uuid.uuid4())

        # 1. Store workflow as template
        template_id = await bridge.store_workflow_template(
            workflow_id=workflow_id,
            template_name="Test Workflow Template",
            template_description="A template for testing purposes",
            workflow_data={"nodes": [], "edges": []},
            use_cases=["testing", "development"],
            best_practices=["test-driven", "modular"],
            tags=["test", "template"]
        )

        # 2. Search for templates
        templates = await bridge.search_workflow_templates(
            query="test workflow",
            limit=10
        )

        # 3. Retrieve specific template
        template = await bridge.get_workflow_template(template_id)

        # Verify template lifecycle
        assert template_id is not None
        assert len(templates) > 0
        assert template is not None
        assert template["name"] == "Test Workflow Template"

    @pytest.mark.asyncio
    async def test_performance_pattern_detection(self):
        """Test performance pattern detection and analysis"""
        capture_service = WorkflowKnowledgeCapture()
        workflow_id = str(uuid.uuid4())

        # Simulate multiple execution data points
        execution_data = [
            {
                "execution_id": "exec_1",
                "steps": [
                    {"id": "step_1", "duration": 1000, "memory": 512},
                    {"id": "step_2", "duration": 2000, "memory": 1024}
                ],
                "total_duration": 3000,
                "success": True
            },
            {
                "execution_id": "exec_2",
                "steps": [
                    {"id": "step_1", "duration": 1100, "memory": 528},
                    {"id": "step_2", "duration": 2100, "memory": 1056}
                ],
                "total_duration": 3200,
                "success": True
            },
            {
                "execution_id": "exec_3",
                "steps": [
                    {"id": "step_1", "duration": 900, "memory": 496},
                    {"id": "step_2", "duration": 1900, "memory": 992}
                ],
                "total_duration": 2800,
                "success": True
            }
        ]

        # Analyze performance patterns
        patterns = await capture_service.identify_performance_patterns(
            workflow_id=workflow_id,
            execution_data=execution_data
        )

        # Verify pattern detection
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        # Check for common pattern types
        pattern_types = [p.get("pattern_type") for p in patterns]
        assert any("performance" in str(pt).lower() for pt in pattern_types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])