"""
Comprehensive tests for DeepConf Advanced Debugging API Routes - Phase 7 PRD Implementation

Tests all API endpoints for advanced debugging functionality:
- Health check endpoint
- Low confidence analysis endpoint
- Confidence factor tracing endpoint
- Performance bottleneck analysis endpoint
- Optimization suggestions endpoint
- Debug session management endpoints
- Data export endpoints

Author: Archon AI System
Version: 1.0.0
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the API routes and dependencies
from src.server.api_routes.deepconf_debug_api import router, get_debugger
from src.agents.deepconf.debugger import DeepConfDebugger, DebugSession
from src.agents.deepconf.engine import DeepConfEngine

class TestDeepConfDebugAPI:
    """Test suite for DeepConf debugging API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test app with debug routes"""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_debugger(self):
        """Create mock debugger for testing"""
        debugger = Mock(spec=DeepConfDebugger)
        debugger.config = {
            'max_active_sessions': 10,
            'session_timeout': 3600,
            'export_formats': ['json', 'csv', 'pdf']
        }
        debugger.active_sessions = {}
        debugger.session_history = []
        debugger.performance_monitors = {}
        debugger.engine = Mock(spec=DeepConfEngine)
        return debugger
    
    @pytest.fixture
    def sample_task_request(self):
        """Sample task request data"""
        return {
            "task_id": "test_task_001",
            "content": "Test debugging task content",
            "domain": "debugging",
            "complexity": "moderate",
            "priority": "normal",
            "model_source": "test_model",
            "context_size": 1000
        }
    
    @pytest.fixture
    def sample_confidence_request(self):
        """Sample confidence request data"""
        return {
            "overall_confidence": 0.25,
            "factual_confidence": 0.3,
            "reasoning_confidence": 0.2,
            "contextual_confidence": 0.25,
            "epistemic_uncertainty": 0.4,
            "aleatoric_uncertainty": 0.3,
            "uncertainty_bounds": [0.15, 0.35],
            "confidence_factors": {
                "technical_complexity": 0.3,
                "domain_expertise": 0.2,
                "data_availability": 0.4,
                "model_capability": 0.25
            },
            "primary_factors": ["data_availability", "domain_expertise"],
            "confidence_reasoning": "Low confidence due to multiple issues",
            "model_source": "test_model",
            "task_id": "test_task_001"
        }

class TestHealthEndpoint(TestDeepConfDebugAPI):
    """Test health check endpoint"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_health_check_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful health check"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        mock_session = Mock(spec=DebugSession)
        mock_session.session_id = "test_session_001"
        mock_debugger.create_debug_session.return_value = mock_session
        
        # Act
        response = client.get("/api/deepconf/debug/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["debugger_initialized"] is True
        assert data["engine_attached"] is True
        assert "active_sessions" in data
        assert "max_sessions" in data
        assert "session_timeout" in data
        assert "supported_export_formats" in data
        assert "test_session_created" in data
        assert "timestamp" in data
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_health_check_failure(self, mock_get_debugger, client):
        """Test health check when debugger fails"""
        # Setup - make get_debugger raise an exception
        mock_get_debugger.side_effect = Exception("Debugger initialization failed")
        
        # Act
        response = client.get("/api/deepconf/debug/health")
        
        # Assert
        assert response.status_code == 200  # Health endpoint should not fail
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert data["debugger_initialized"] is False
        assert "error" in data
        assert "timestamp" in data

class TestLowConfidenceAnalysisEndpoint(TestDeepConfDebugAPI):
    """Test low confidence analysis endpoint"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    @pytest.mark.asyncio
    async def test_analyze_low_confidence_success(self, mock_get_debugger, client, mock_debugger, sample_task_request, sample_confidence_request):
        """Test successful low confidence analysis"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        # Create mock debug report
        mock_report = Mock()
        mock_report.report_id = "test_report_001"
        mock_report.task_id = "test_task_001"
        mock_report.confidence_score = 0.25
        mock_report.analysis_timestamp = Mock()
        mock_report.analysis_timestamp.isoformat.return_value = "2024-01-01T00:00:00"
        mock_report.issues = []
        mock_report.recommendations = ["Improve data quality", "Add more context"]
        mock_report.severity_summary = {"critical": 1, "high": 2}
        mock_report.factor_analysis = {"technical_complexity": {"score": 0.3}}
        mock_report.performance_profile = {"computation_time": 0.15}
        mock_report.optimization_opportunities = []
        mock_report.confidence_projection = {
            "current_confidence": 0.25,
            "projected_confidence": 0.65,
            "potential_improvement": 0.4
        }
        
        mock_debugger.analyze_low_confidence = AsyncMock(return_value=mock_report)
        
        # Act
        response = client.post(
            "/api/deepconf/debug/analyze/low-confidence",
            json={
                "task": sample_task_request,
                "confidence": sample_confidence_request
            }
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "analysis_time" in data
        assert "report" in data
        assert data["report"]["report_id"] == "test_report_001"
        assert data["report"]["task_id"] == "test_task_001"
        assert data["report"]["confidence_score"] == 0.25
        assert len(data["report"]["recommendations"]) == 2
        assert "timestamp" in data
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_analyze_low_confidence_invalid_task(self, mock_get_debugger, client, sample_confidence_request):
        """Test analysis with invalid task data"""
        # Setup
        mock_get_debugger.return_value = Mock()
        
        invalid_task = {
            "task_id": "",  # Invalid empty task_id
            "content": "",  # Invalid empty content
        }
        
        # Act
        response = client.post(
            "/api/deepconf/debug/analyze/low-confidence",
            json={
                "task": invalid_task,
                "confidence": sample_confidence_request
            }
        )
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_analyze_low_confidence_invalid_confidence(self, mock_get_debugger, client, sample_task_request):
        """Test analysis with invalid confidence data"""
        # Setup
        mock_get_debugger.return_value = Mock()
        
        invalid_confidence = {
            "overall_confidence": 1.5,  # Invalid - out of range
            "uncertainty_bounds": [0.5],  # Invalid - wrong length
        }
        
        # Act
        response = client.post(
            "/api/deepconf/debug/analyze/low-confidence",
            json={
                "task": sample_task_request,
                "confidence": invalid_confidence
            }
        )
        
        # Assert
        assert response.status_code == 422  # Validation error

class TestConfidenceTracingEndpoint(TestDeepConfDebugAPI):
    """Test confidence factor tracing endpoint"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    @pytest.mark.asyncio
    async def test_trace_confidence_factors_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful confidence factor tracing"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        # Create mock trace
        mock_trace = Mock()
        mock_trace.factor_name = "technical_complexity"
        mock_trace.raw_score = 0.65
        mock_trace.weighted_score = 0.1625
        mock_trace.weight = 0.25
        mock_trace.calculation_steps = [
            {
                "step": 1,
                "operation": "complexity_assessment",
                "calculation": "base_score + domain_adjustment",
                "result": 0.65,
                "computation_time": 0.001
            }
        ]
        mock_trace.dependencies = ["domain_expertise", "historical_performance"]
        mock_trace.computation_time = 0.002
        mock_trace.confidence_contribution = 0.23
        mock_trace.trace_id = "test_trace_001"
        mock_trace.timestamp = Mock()
        mock_trace.timestamp.isoformat.return_value = "2024-01-01T00:00:00"
        
        mock_debugger.trace_confidence_factors = AsyncMock(return_value=mock_trace)
        
        # Act
        response = client.get("/api/deepconf/debug/trace/confidence-factors/test_confidence_001")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "trace_time" in data
        assert "trace" in data
        assert data["trace"]["factor_name"] == "technical_complexity"
        assert data["trace"]["raw_score"] == 0.65
        assert data["trace"]["weighted_score"] == 0.1625
        assert len(data["trace"]["calculation_steps"]) == 1
        assert len(data["trace"]["dependencies"]) == 2
        assert "metadata" in data
        assert data["metadata"]["contribution_percentage"] == 23.0
    
    def test_trace_confidence_factors_invalid_id(self, client):
        """Test tracing with invalid confidence ID"""
        # Act - test with very long ID that should be rejected
        response = client.get("/api/deepconf/debug/trace/confidence-factors/" + "x" * 200)
        
        # Assert
        assert response.status_code in [400, 422]  # Should reject invalid ID

class TestPerformanceBottleneckEndpoint(TestDeepConfDebugAPI):
    """Test performance bottleneck analysis endpoint"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    @pytest.mark.asyncio
    async def test_analyze_performance_bottlenecks_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful bottleneck analysis"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        # Create mock bottleneck analysis
        mock_analysis = Mock()
        mock_analysis.bottleneck_id = "bottleneck_001"
        mock_analysis.category = Mock()
        mock_analysis.category.value = "computation"
        mock_analysis.severity = Mock()
        mock_analysis.severity.value = "high"
        mock_analysis.description = "High computation time detected"
        mock_analysis.affected_operations = ["confidence_calculation", "factor_analysis"]
        mock_analysis.performance_impact = {"latency": 2.5, "throughput_reduction": 0.4}
        mock_analysis.root_causes = ["Complex algorithms", "Lack of caching"]
        mock_analysis.optimization_suggestions = ["Implement caching", "Optimize algorithms"]
        mock_analysis.estimated_improvement = {"latency_reduction": 0.4}
        mock_analysis.timestamp = Mock()
        mock_analysis.timestamp.isoformat.return_value = "2024-01-01T00:00:00"
        
        mock_debugger.identify_performance_bottlenecks = AsyncMock(return_value=mock_analysis)
        
        # Prepare request data
        task_history_request = {
            "task_id": "test_task_001",
            "execution_records": [
                {"timestamp": time.time(), "duration": 2.5, "success": True}
            ],
            "performance_metrics": [
                {"computation_time": 2.5, "memory_usage": 85}
            ],
            "total_executions": 1,
            "average_confidence": 0.65,
            "performance_trends": {
                "computation_time": [2.5, 3.0, 2.8]
            }
        }
        
        # Act
        response = client.post(
            "/api/deepconf/debug/analyze/performance-bottlenecks",
            json=task_history_request
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "analysis_time" in data
        assert "bottleneck_analysis" in data
        assert data["bottleneck_analysis"]["bottleneck_id"] == "bottleneck_001"
        assert data["bottleneck_analysis"]["category"] == "computation"
        assert data["bottleneck_analysis"]["severity"] == "high"
        assert len(data["bottleneck_analysis"]["optimization_suggestions"]) == 2
        assert "recommendations" in data

class TestOptimizationSuggestionsEndpoint(TestDeepConfDebugAPI):
    """Test optimization suggestions endpoint"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    @pytest.mark.asyncio
    async def test_get_optimization_suggestions_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful optimization suggestions generation"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        # Create mock optimization suggestions
        mock_strategy = Mock()
        mock_strategy.strategy_id = "cache_optimization_001"
        mock_strategy.title = "Improve Caching"
        mock_strategy.description = "Implement advanced caching strategies"
        mock_strategy.implementation_complexity = "medium"
        mock_strategy.expected_improvement = {"latency_reduction": 0.4, "throughput_increase": 0.6}
        mock_strategy.implementation_steps = ["Add cache layer", "Implement invalidation"]
        mock_strategy.risks = ["Memory usage increase"]
        mock_strategy.prerequisites = ["Memory monitoring"]
        mock_strategy.estimated_effort = "days"
        mock_strategy.confidence = 0.85
        
        mock_suggestions = Mock()
        mock_suggestions.strategies = [mock_strategy]
        mock_suggestions.priority_ranking = ["cache_optimization_001"]
        mock_suggestions.implementation_roadmap = [
            {
                "phase": 1,
                "strategy_id": "cache_optimization_001",
                "title": "Improve Caching",
                "estimated_duration": "days"
            }
        ]
        mock_suggestions.resource_requirements = {"development_effort": {"medium_complexity": 1}}
        mock_suggestions.risk_assessment = {"overall_risk_score": 0.5}
        mock_suggestions.success_metrics = ["Latency reduction ≥ 40%"]
        
        mock_debugger.suggest_optimization_strategies = AsyncMock(return_value=mock_suggestions)
        
        # Prepare request data
        performance_data_request = {
            "operation_times": {
                "confidence_calculation": [0.15, 0.18, 0.16]
            },
            "memory_usage": {
                "peak_usage": [45, 48, 50]
            },
            "cache_hit_rates": {
                "confidence_cache": 0.65
            },
            "error_rates": {
                "calculation_errors": 0.02
            },
            "throughput_metrics": {
                "requests_per_second": 25
            },
            "bottleneck_indicators": {
                "computation_intensive": True
            }
        }
        
        # Act
        response = client.post(
            "/api/deepconf/debug/optimize/suggestions",
            json={"performance_data": performance_data_request}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "analysis_time" in data
        assert "optimization_suggestions" in data
        assert len(data["optimization_suggestions"]["strategies"]) == 1
        assert data["optimization_suggestions"]["strategies"][0]["title"] == "Improve Caching"
        assert len(data["optimization_suggestions"]["priority_ranking"]) == 1
        assert "recommendations" in data
        assert len(data["recommendations"]["quick_wins"]) >= 0

class TestSessionManagementEndpoints(TestDeepConfDebugAPI):
    """Test debug session management endpoints"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_create_debug_session_success(self, mock_get_debugger, client, mock_debugger, sample_task_request):
        """Test successful debug session creation"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        mock_session = Mock(spec=DebugSession)
        mock_session.session_id = "test_session_001"
        mock_session.task = Mock()
        mock_session.task.task_id = "test_task_001"
        mock_session.start_time = Mock()
        mock_session.start_time.isoformat.return_value = "2024-01-01T00:00:00"
        mock_session.is_active = True
        mock_session.session_state = {"initialized": True}
        
        mock_debugger.create_debug_session.return_value = mock_session
        mock_debugger.active_sessions = {"test_session_001": mock_session}
        
        # Act
        response = client.post(
            "/api/deepconf/debug/sessions",
            json=sample_task_request
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "session" in data
        assert data["session"]["session_id"] == "test_session_001"
        assert data["session"]["task_id"] == "test_task_001"
        assert data["session"]["is_active"] is True
        assert "config" in data
        assert data["config"]["max_active_sessions"] == 10
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_create_debug_session_max_sessions_exceeded(self, mock_get_debugger, client, mock_debugger, sample_task_request):
        """Test session creation when max sessions exceeded"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        mock_debugger.create_debug_session.side_effect = RuntimeError("Maximum active sessions (10) exceeded")
        
        # Act
        response = client.post(
            "/api/deepconf/debug/sessions",
            json=sample_task_request
        )
        
        # Assert
        assert response.status_code == 429  # Too Many Requests
        data = response.json()
        assert "Maximum active sessions" in data["detail"]
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_get_debug_session_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful session retrieval"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        mock_session = Mock(spec=DebugSession)
        mock_session.session_id = "test_session_001"
        mock_session.task = Mock()
        mock_session.task.task_id = "test_task_001"
        mock_session.task.content = "Test content"
        mock_session.task.domain = "testing"
        mock_session.task.complexity = "moderate"
        mock_session.task.priority = "normal"
        mock_session.start_time = Mock()
        mock_session.start_time.isoformat.return_value = "2024-01-01T00:00:00"
        mock_session.end_time = None
        mock_session.is_active = True
        mock_session.debug_reports = []
        mock_session.performance_snapshots = []
        mock_session.debug_actions = []
        mock_session.session_state = {"initialized": True}
        
        mock_debugger.active_sessions = {"test_session_001": mock_session}
        
        # Act
        response = client.get("/api/deepconf/debug/sessions/test_session_001")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "session" in data
        assert data["session"]["session_id"] == "test_session_001"
        assert data["session"]["task"]["task_id"] == "test_task_001"
        assert data["session"]["is_active"] is True
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_get_debug_session_not_found(self, mock_get_debugger, client, mock_debugger):
        """Test session retrieval when session doesn't exist"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        mock_debugger.active_sessions = {}
        
        # Act
        response = client.get("/api/deepconf/debug/sessions/nonexistent_session")
        
        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_list_debug_sessions_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful session listing"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        mock_session1 = Mock(spec=DebugSession)
        mock_session1.session_id = "session_001"
        mock_session1.task = Mock()
        mock_session1.task.task_id = "task_001"
        mock_session1.start_time = Mock()
        mock_session1.start_time.isoformat.return_value = "2024-01-01T00:00:00"
        mock_session1.end_time = None
        mock_session1.is_active = True
        mock_session1.debug_reports = []
        mock_session1.duration_minutes = 15.5
        
        mock_session2 = Mock(spec=DebugSession)
        mock_session2.session_id = "session_002"
        mock_session2.task = Mock()
        mock_session2.task.task_id = "task_002"
        mock_session2.start_time = Mock()
        mock_session2.start_time.isoformat.return_value = "2024-01-01T01:00:00"
        mock_session2.end_time = Mock()
        mock_session2.end_time.isoformat.return_value = "2024-01-01T01:30:00"
        mock_session2.is_active = False
        mock_session2.debug_reports = []
        mock_session2.duration_minutes = 30.0
        
        mock_debugger.active_sessions = {"session_001": mock_session1}
        mock_debugger.session_history = [mock_session2]
        
        # Act
        response = client.get("/api/deepconf/debug/sessions?active_only=false&limit=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "sessions" in data
        assert len(data["sessions"]) == 2
        assert "metadata" in data
        assert data["metadata"]["total_active"] == 1
        assert data["metadata"]["returned_count"] == 2

class TestDataExportEndpoint(TestDeepConfDebugAPI):
    """Test data export endpoint"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_export_debug_data_json_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful JSON data export"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        mock_session = Mock(spec=DebugSession)
        mock_session.session_id = "test_session_001"
        
        mock_export = Mock()
        mock_export.export_id = "export_001"
        mock_export.session = mock_session
        mock_export.export_format = "json"
        mock_export.export_timestamp = Mock()
        mock_export.export_timestamp.isoformat.return_value = "2024-01-01T00:00:00"
        mock_export.analysis_summary = {"session_overview": {"session_id": "test_session_001"}}
        mock_export.raw_data = {"session": {"session_id": "test_session_001"}}
        mock_export.visualizations = [{"type": "confidence_timeline"}]
        mock_export.metadata = {"export_format": "json"}
        
        mock_debugger.active_sessions = {"test_session_001": mock_session}
        mock_debugger.export_debug_data.return_value = mock_export
        
        # Act
        response = client.post("/api/deepconf/debug/export/test_session_001?export_format=json")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "export_time" in data
        assert "export" in data
        assert data["export"]["export_id"] == "export_001"
        assert data["export"]["export_format"] == "json"
        assert "download_info" in data
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_export_debug_data_session_not_found(self, mock_get_debugger, client, mock_debugger):
        """Test export when session doesn't exist"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        mock_debugger.active_sessions = {}
        mock_debugger.session_history = []
        
        # Act
        response = client.post("/api/deepconf/debug/export/nonexistent_session?export_format=json")
        
        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    def test_export_debug_data_invalid_format(self, client):
        """Test export with invalid format"""
        # Act
        response = client.post("/api/deepconf/debug/export/test_session?export_format=invalid")
        
        # Assert
        assert response.status_code == 422  # Validation error

class TestConfigurationEndpoints(TestDeepConfDebugAPI):
    """Test configuration endpoints"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_get_debug_config_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful config retrieval"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        # Act
        response = client.get("/api/deepconf/debug/config")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "config" in data
        assert "runtime_info" in data
        assert data["runtime_info"]["active_sessions"] == 0
        assert data["runtime_info"]["engine_attached"] is True
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_update_debug_config_success(self, mock_get_debugger, client, mock_debugger):
        """Test successful config update"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        config_updates = {
            "session_timeout": 7200,
            "confidence_threshold_critical": 0.2,
            "trace_depth": 10
        }
        
        # Act
        response = client.post(
            "/api/deepconf/debug/config",
            json=config_updates
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "updated_config" in data
        assert data["updated_config"]["session_timeout"] == 7200
        assert "current_config" in data
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_update_debug_config_restricted_keys(self, mock_get_debugger, client, mock_debugger):
        """Test config update with restricted keys"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        
        config_updates = {
            "max_active_sessions": 20,  # Not allowed to change at runtime
            "session_timeout": 7200,    # Allowed
            "dangerous_setting": True   # Not in allowed list
        }
        
        # Act
        response = client.post(
            "/api/deepconf/debug/config",
            json=config_updates
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        # Should only update allowed keys
        assert "session_timeout" in data["updated_config"]
        assert "max_active_sessions" not in data["updated_config"]
        assert "dangerous_setting" not in data["updated_config"]

class TestAPIErrorHandling(TestDeepConfDebugAPI):
    """Test API error handling"""
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    def test_debugger_initialization_failure(self, mock_get_debugger, client):
        """Test API behavior when debugger initialization fails"""
        # Setup
        mock_get_debugger.side_effect = RuntimeError("Debugger initialization failed")
        
        # Act
        response = client.post("/api/deepconf/debug/sessions", json={"task_id": "test"})
        
        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "initialization failed" in data["detail"]
    
    @patch('src.server.api_routes.deepconf_debug_api.get_debugger')
    @pytest.mark.asyncio
    async def test_analysis_method_failure(self, mock_get_debugger, client, mock_debugger, sample_task_request, sample_confidence_request):
        """Test API behavior when analysis method fails"""
        # Setup
        mock_get_debugger.return_value = mock_debugger
        mock_debugger.analyze_low_confidence = AsyncMock(side_effect=Exception("Analysis failed"))
        
        # Act
        response = client.post(
            "/api/deepconf/debug/analyze/low-confidence",
            json={
                "task": sample_task_request,
                "confidence": sample_confidence_request
            }
        )
        
        # Assert
        assert response.status_code == 500
        data = response.json()
        assert "Analysis failed" in data["detail"]

if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "--tb=short"])