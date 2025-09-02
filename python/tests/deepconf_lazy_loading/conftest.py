"""
Pytest configuration and fixtures for DeepConf lazy loading tests
================================================================

Shared fixtures and configuration for all lazy loading tests.
"""

import pytest
import sys
import os
import gc
import time
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

# Add the project source to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root path"""
    return project_root


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment before and after each test"""
    # Clean before test
    gc.collect()
    
    # Clear any cached modules related to deepconf
    modules_to_remove = [
        module for module in sys.modules.keys() 
        if 'deepconf' in module.lower() or 'agents.deepconf' in module
    ]
    
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
    
    yield
    
    # Clean after test
    gc.collect()


@pytest.fixture
def memory_tracker():
    """Track memory usage during tests"""
    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.baseline_memory = None
            self.measurements = []
        
        def start_tracking(self):
            """Start memory tracking"""
            self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            return self.baseline_memory
        
        def measure(self, label: str = ""):
            """Take a memory measurement"""
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            if self.baseline_memory:
                delta = current_memory - self.baseline_memory
            else:
                delta = 0
            
            measurement = {
                'label': label,
                'timestamp': time.time(),
                'memory_mb': current_memory,
                'delta_mb': delta
            }
            self.measurements.append(measurement)
            return measurement
        
        def get_peak_memory(self) -> float:
            """Get peak memory usage"""
            if not self.measurements:
                return 0.0
            return max(m['memory_mb'] for m in self.measurements)
        
        def get_memory_delta(self) -> float:
            """Get total memory delta"""
            if not self.measurements or not self.baseline_memory:
                return 0.0
            final_memory = self.measurements[-1]['memory_mb']
            return final_memory - self.baseline_memory
    
    return MemoryTracker()


@pytest.fixture
def performance_timer():
    """Timer for performance measurements"""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.measurements = {}
        
        def start(self, label: str = "default"):
            """Start timing"""
            self.start_time = time.perf_counter()
            return self.start_time
        
        def stop(self, label: str = "default") -> float:
            """Stop timing and return duration"""
            if self.start_time is None:
                return 0.0
            
            duration = time.perf_counter() - self.start_time
            self.measurements[label] = duration
            self.start_time = None
            return duration
        
        def get_duration(self, label: str = "default") -> Optional[float]:
            """Get recorded duration"""
            return self.measurements.get(label)
    
    return PerformanceTimer()


@pytest.fixture
def mock_task_factory():
    """Factory for creating mock tasks with various properties"""
    def create_mock_task(
        task_id: str = "test_task",
        content: str = "Test task content",
        complexity: str = "moderate",
        domain: str = "testing",
        model_source: str = "claude-3.5-sonnet",
        priority: str = "normal",
        **kwargs
    ) -> MagicMock:
        """Create a mock task with specified properties"""
        mock_task = MagicMock()
        mock_task.task_id = task_id
        mock_task.content = content
        mock_task.complexity = complexity
        mock_task.domain = domain
        mock_task.model_source = model_source
        mock_task.priority = priority
        
        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(mock_task, key, value)
        
        return mock_task
    
    return create_mock_task


@pytest.fixture 
def mock_context_factory():
    """Factory for creating mock contexts with various properties"""
    def create_mock_context(
        environment: str = "test",
        user_id: str = "test_user",
        session_id: str = "test_session",
        timestamp: Optional[float] = None,
        **kwargs
    ) -> MagicMock:
        """Create a mock context with specified properties"""
        mock_context = MagicMock()
        mock_context.environment = environment
        mock_context.user_id = user_id
        mock_context.session_id = session_id
        mock_context.timestamp = timestamp or time.time()
        
        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(mock_context, key, value)
        
        return mock_context
    
    return create_mock_context


@pytest.fixture
def test_data_directory(tmp_path):
    """Provide a temporary directory for test data"""
    test_dir = tmp_path / "deepconf_test_data"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def baseline_metrics_file(test_data_directory):
    """Provide path for baseline metrics storage"""
    return test_data_directory / "baseline_confidence_metrics.json"


# Pytest hooks for custom reporting
def pytest_runtest_setup(item):
    """Setup hook for each test"""
    # Mark slow tests
    if hasattr(item.function, '__name__'):
        if any(marker in item.function.__name__ for marker in ['stress', 'concurrent', 'exhaustion']):
            item.add_marker(pytest.mark.slow)


def pytest_runtest_teardown(item, nextitem):
    """Teardown hook for each test"""
    # Force garbage collection after each test
    gc.collect()


def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "functional: Functional tests for feature behavior" 
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test names/paths
    for item in items:
        # Mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark integration tests  
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
            
        # Mark regression tests
        if "regression" in item.nodeid.lower():
            item.add_marker(pytest.mark.regression)
            
        # Mark edge case tests
        if "edge" in item.nodeid.lower() or "error" in item.nodeid.lower():
            item.add_marker(pytest.mark.edge_case)


# Custom assertions for test validation
class DeepConfAssertions:
    """Custom assertions for DeepConf testing"""
    
    @staticmethod
    def assert_valid_confidence_score(confidence_score, task_id: Optional[str] = None):
        """Assert that a confidence score is valid"""
        assert hasattr(confidence_score, 'overall_confidence'), "Missing overall_confidence"
        assert hasattr(confidence_score, 'factual_confidence'), "Missing factual_confidence"
        assert hasattr(confidence_score, 'reasoning_confidence'), "Missing reasoning_confidence"
        assert hasattr(confidence_score, 'contextual_confidence'), "Missing contextual_confidence"
        assert hasattr(confidence_score, 'uncertainty_bounds'), "Missing uncertainty_bounds"
        
        # Validate confidence ranges
        assert 0.0 <= confidence_score.overall_confidence <= 1.0, f"Invalid overall_confidence: {confidence_score.overall_confidence}"
        assert 0.0 <= confidence_score.factual_confidence <= 1.0, f"Invalid factual_confidence: {confidence_score.factual_confidence}"
        assert 0.0 <= confidence_score.reasoning_confidence <= 1.0, f"Invalid reasoning_confidence: {confidence_score.reasoning_confidence}"
        assert 0.0 <= confidence_score.contextual_confidence <= 1.0, f"Invalid contextual_confidence: {confidence_score.contextual_confidence}"
        
        # Validate uncertainty bounds
        assert len(confidence_score.uncertainty_bounds) == 2, "Uncertainty bounds should be a tuple of 2 values"
        assert confidence_score.uncertainty_bounds[0] <= confidence_score.uncertainty_bounds[1], "Invalid uncertainty bounds order"
        
        # Validate task ID if provided
        if task_id:
            assert confidence_score.task_id == task_id, f"Task ID mismatch: expected {task_id}, got {confidence_score.task_id}"
    
    @staticmethod
    def assert_performance_within_limits(duration_seconds: float, max_duration: float, operation_name: str = "operation"):
        """Assert that performance is within acceptable limits"""
        assert duration_seconds <= max_duration, (
            f"{operation_name} took {duration_seconds:.3f}s, exceeds limit of {max_duration}s"
        )
    
    @staticmethod
    def assert_memory_within_limits(memory_mb: float, max_memory: float, operation_name: str = "operation"):
        """Assert that memory usage is within acceptable limits"""
        assert memory_mb <= max_memory, (
            f"{operation_name} used {memory_mb:.2f}MB, exceeds limit of {max_memory}MB"
        )


@pytest.fixture
def deepconf_assertions():
    """Provide custom DeepConf assertions"""
    return DeepConfAssertions()