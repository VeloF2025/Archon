"""
TDD Enforcement Module - Phase 9 Implementation

This module provides Test-Driven Development enforcement with browserbase-stagehand integration.
Ensures NO feature can be built without tests first, with natural language test generation.

Components:
- stagehand_test_engine: Natural language test generation using Stagehand
- browserbase_executor: Cloud test execution management  
- tdd_enforcement_gate: Mandatory test-first validation
- enhanced_dgts_validator: Extended DGTS for Stagehand-specific gaming patterns

Integration with existing Archon validation systems for zero-tolerance quality enforcement.
"""

from .stagehand_test_engine import StagehandTestEngine, TestGenerationResult
from .browserbase_executor import BrowserbaseExecutor, ExecutionResult
from .tdd_enforcement_gate import TDDEnforcementGate, EnforcementResult
from .enhanced_dgts_validator import EnhancedDGTSValidator, StagehandGamingViolation

__all__ = [
    'StagehandTestEngine',
    'TestGenerationResult', 
    'BrowserbaseExecutor',
    'ExecutionResult',
    'TDDEnforcementGate',
    'EnforcementResult',
    'EnhancedDGTSValidator',
    'StagehandGamingViolation'
]

__version__ = "1.0.0"
__author__ = "Archon TDD Enforcement System"