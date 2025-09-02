"""
Pytest configuration and fixtures for Archon PM System Enhancement tests.
These fixtures support TDD development by providing mock objects and test data.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import tempfile
import os
import json


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_git_repo():
    """Mock git repository with test history."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock git structure
        git_dir = os.path.join(temp_dir, '.git')
        os.makedirs(git_dir)
        
        # Mock git log data representing missed implementations
        mock_commits = [
            {
                'hash': 'abc123',
                'message': 'Implement MANIFEST integration for enhanced discovery',
                'author': 'agent-system-architect',
                'date': '2024-01-15 10:30:00',
                'files': ['src/manifest_integration.py', 'config/manifest.json']
            },
            {
                'hash': 'def456', 
                'message': 'Fix Socket.IO connection timeouts and reliability',
                'author': 'agent-code-implementer',
                'date': '2024-01-16 14:45:00',
                'files': ['src/socketio_handler.py', 'tests/test_socketio.py']
            },
            {
                'hash': 'ghi789',
                'message': 'Add backend health check endpoints',
                'author': 'agent-system-architect', 
                'date': '2024-01-17 09:15:00',
                'files': ['src/health_check.py', 'routes/health.py']
            },
            {
                'hash': 'jkl012',
                'message': 'Implement chunks count API with validation',
                'author': 'agent-api-designer',
                'date': '2024-01-18 16:20:00', 
                'files': ['src/api/chunks.py', 'validators/chunks_validator.py']
            },
            {
                'hash': 'mno345',
                'message': 'Add confidence scoring system for implementations',
                'author': 'agent-code-implementer',
                'date': '2024-01-19 11:00:00',
                'files': ['src/confidence_api.py', 'models/confidence_model.py']
            }
        ]
        
        yield {
            'repo_path': temp_dir,
            'commits': mock_commits,
            'total_implementations': 25,
            'tracked_implementations': 2,  # Current poor tracking
            'missing_implementations': 23
        }


@pytest.fixture
def mock_agent_activity():
    """Mock real-time agent activity data."""
    return {
        'current_agents': [
            {
                'id': 'agent-001',
                'type': 'system-architect', 
                'status': 'working',
                'current_task': 'Designing database schema',
                'project_id': 'ff-react-neon',
                'start_time': datetime.now() - timedelta(minutes=15)
            },
            {
                'id': 'agent-002',
                'type': 'code-implementer',
                'status': 'completed',
                'current_task': 'User authentication system',
                'project_id': 'ff-react-neon', 
                'completion_time': datetime.now() - timedelta(minutes=5),
                'files_modified': ['src/auth.py', 'tests/test_auth.py']
            },
            {
                'id': 'agent-003',
                'type': 'test-coverage-validator',
                'status': 'testing',
                'current_task': 'API endpoint validation',
                'project_id': 'ff-react-neon',
                'start_time': datetime.now() - timedelta(minutes=8)
            }
        ],
        'completed_work': [
            {
                'agent_id': 'agent-002',
                'task_title': 'User authentication system',
                'completion_time': datetime.now() - timedelta(minutes=5),
                'confidence_score': 0.95,
                'implementation_verified': True,
                'files_created': ['src/auth.py', 'src/auth_middleware.py'],
                'tests_created': ['tests/test_auth.py', 'tests/test_auth_integration.py']
            }
        ]
    }


@pytest.fixture
def mock_implementation_data():
    """Mock data for implementation verification tests."""
    return {
        'verified_implementations': [
            {
                'name': 'User Authentication',
                'status': 'verified',
                'confidence': 0.98,
                'health_check_passed': True,
                'api_endpoints_working': True,
                'files_exist': True,
                'tests_passing': True
            },
            {
                'name': 'Database Connection',
                'status': 'verified', 
                'confidence': 0.92,
                'health_check_passed': True,
                'api_endpoints_working': True,
                'files_exist': True,
                'tests_passing': True
            }
        ],
        'unverified_implementations': [
            {
                'name': 'MANIFEST Integration',
                'status': 'unverified',
                'confidence': 0.0,
                'health_check_passed': False,
                'api_endpoints_working': False, 
                'files_exist': False,
                'tests_passing': False,
                'expected_files': ['src/manifest_integration.py', 'config/manifest.json']
            },
            {
                'name': 'Socket.IO Handler',
                'status': 'unverified',
                'confidence': 0.0,
                'health_check_passed': False,
                'api_endpoints_working': False,
                'files_exist': False, 
                'tests_passing': False,
                'expected_files': ['src/socketio_handler.py']
            }
        ]
    }


@pytest.fixture 
def mock_task_data():
    """Mock data for task management tests."""
    return {
        'existing_tasks': [
            {
                'id': 'task-001',
                'title': 'User Authentication',
                'status': 'completed',
                'project_id': 'ff-react-neon',
                'created_date': datetime.now() - timedelta(days=2),
                'completion_date': datetime.now() - timedelta(hours=6)
            },
            {
                'id': 'task-002', 
                'title': 'Database Setup',
                'status': 'completed',
                'project_id': 'ff-react-neon',
                'created_date': datetime.now() - timedelta(days=1),
                'completion_date': datetime.now() - timedelta(hours=2)
            }
        ],
        'missing_tasks': [
            'MANIFEST Integration',
            'Socket.IO Connection Handler', 
            'Backend Health Checks',
            'Chunks Count API',
            'Confidence Scoring System',
            'API Timeout Configuration',
            'Database Migration Scripts',
            'User Profile Management',
            'File Upload Handler',
            'Email Notification System',
            'Logging Infrastructure',
            'Rate Limiting Middleware',
            'Data Validation Layer',
            'Caching Implementation',
            'Background Job Processor',
            'API Documentation Generator',
            'Performance Monitoring',
            'Security Audit Implementation',
            'Backup System',
            'Error Reporting System',
            'Session Management',
            'OAuth Integration',
            'Real-time Updates'
        ]
    }


@pytest.fixture
def mock_archon_pm_system():
    """Mock the current Archon PM system (showing inadequacies)."""
    return Mock(
        get_tracked_implementations=Mock(return_value=2),  # Only 2/25+ tracked
        get_total_implementations=Mock(return_value=25),
        get_implementation_status=Mock(return_value='incomplete'),
        update_task_status=Mock(return_value=False),  # No real-time updates
        discover_historical_work=Mock(return_value=[]),  # No discovery capability
        verify_implementation=Mock(return_value=None),  # No verification
        get_confidence_score=Mock(return_value=0.0),  # No confidence scoring
        monitor_agent_activity=Mock(return_value=None),  # No monitoring
        create_task_from_work=Mock(return_value=None)  # No automatic task creation
    )


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark targets."""
    return {
        'discovery_operation_max_time': 0.5,  # 500ms max
        'real_time_update_max_delay': 30,  # 30 seconds max
        'implementation_verification_max_time': 1.0,  # 1 second max
        'confidence_calculation_max_time': 0.2,  # 200ms max
        'task_creation_max_time': 0.1,  # 100ms max
        'max_concurrent_tasks': 1000,
        'uptime_requirement': 0.999  # 99.9% uptime
    }


@pytest.fixture
def accuracy_requirements():
    """Data accuracy requirements."""
    return {
        'work_tracking_accuracy': 0.95,  # 95% accuracy
        'implementation_status_accuracy': 0.98,  # 98% accuracy
        'false_positive_rate_max': 0.02,  # Max 2% false positives
        'false_negative_rate_max': 0.05,  # Max 5% false negatives
        'confidence_score_accuracy': 0.90  # 90% confidence accuracy
    }


@pytest.fixture 
def current_pm_failures():
    """Document current PM system failures."""
    return {
        'tracked_vs_actual': {
            'tracked': 2,
            'actual': 25,
            'missing': 23,
            'accuracy': 0.08  # Only 8% accuracy
        },
        'missing_implementations': [
            'MANIFEST integration',
            'Socket.IO fixes', 
            'Backend health checks',
            'Chunks count API',
            'Confidence API',
            'API timeout handling',
            'Database migrations',
            'User profile system',
            'File upload service',
            'Email notifications',
            'Logging system',
            'Rate limiting',
            'Data validation',
            'Caching layer', 
            'Background jobs',
            'API documentation',
            'Performance monitoring',
            'Security auditing',
            'Backup systems',
            'Error reporting',
            'Session management',
            'OAuth integration',
            'Real-time updates'
        ],
        'real_time_failures': {
            'update_delay': float('inf'),  # No real-time updates
            'agent_monitoring': False,
            'automatic_task_creation': False
        },
        'verification_failures': {
            'implementation_verification': False,
            'health_checks': False,
            'api_endpoint_testing': False,
            'confidence_scoring': False
        }
    }


class MockArchonPMSystemEnhanced:
    """Enhanced mock that will pass tests after implementation."""
    
    def __init__(self):
        self.implementations = {}
        self.tasks = {}
        self.agent_activities = []
        self.confidence_scores = {}
        
    async def discover_historical_work(self) -> List[Dict[str, Any]]:
        """Mock enhanced discovery that finds all 25+ implementations."""
        # This will initially return empty list (causing tests to fail)
        # After implementation, should return all discovered work
        return []
    
    async def monitor_agent_activity(self) -> List[Dict[str, Any]]:
        """Mock real-time agent monitoring."""
        # Initially returns empty (tests fail)
        # After implementation, returns real-time data
        return []
    
    async def verify_implementation(self, name: str) -> Dict[str, Any]:
        """Mock implementation verification."""
        # Initially returns None (tests fail)
        # After implementation, returns verification results
        return {}
    
    async def create_task_from_work(self, work_data: Dict[str, Any]) -> Optional[str]:
        """Mock automatic task creation."""
        # Initially returns None (tests fail)
        # After implementation, creates and returns task ID
        return None
    
    def get_confidence_score(self, implementation: str) -> float:
        """Mock confidence scoring."""
        # Initially returns 0.0 (tests fail)
        # After implementation, returns calculated confidence
        return 0.0


@pytest.fixture
def mock_enhanced_pm_system():
    """Mock of the enhanced PM system (initially failing)."""
    return MockArchonPMSystemEnhanced()