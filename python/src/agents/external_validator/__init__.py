"""
Phase 5: External Validator Agent
Independent validation service for Archon system
"""

from .main import app
from .config import ValidatorConfig
from .validation_engine import ValidationEngine
from .models import (
    ValidationRequest, 
    ValidationResponse,
    ValidationStatus,
    ValidationSeverity
)
from .llm_client import LLMClient
from .deterministic import DeterministicChecker
from .cross_check import CrossChecker
from .mcp_integration import MCPIntegration

__all__ = [
    "app",
    "ValidatorConfig",
    "ValidationEngine",
    "ValidationRequest",
    "ValidationResponse",
    "ValidationStatus",
    "ValidationSeverity",
    "LLMClient",
    "DeterministicChecker",
    "CrossChecker",
    "MCPIntegration",
]

__version__ = "1.0.0"