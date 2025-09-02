"""
PM Enhancement Package

This package implements the Archon PM Enhancement System to address
the critical 8% work visibility problem and move to 95%+ tracking accuracy.

Modules:
- historical_work_discovery: Discovers 25+ missing implementations from git and system state
- real_time_activity_monitor: Monitors agent activity and auto-creates tasks (<30s)  
- implementation_verification: Verifies implementations with health checks and confidence scoring

This package supports the TDD test suite by providing the implementation layer
that makes the failing tests pass (RED to GREEN transition).
"""

from .historical_work_discovery import (
    HistoricalWorkDiscoveryEngine,
    ImplementationDiscovery,
    get_historical_discovery_engine
)

from .real_time_activity_monitor import (
    RealTimeActivityMonitor,
    AgentActivity,
    WorkCompletion,
    get_activity_monitor,
    initialize_monitoring,
    cleanup_monitoring
)

from .implementation_verification import (
    ImplementationVerificationSystem,
    VerificationResult,
    get_verification_system
)

__all__ = [
    # Discovery engine
    'HistoricalWorkDiscoveryEngine',
    'ImplementationDiscovery', 
    'get_historical_discovery_engine',
    
    # Activity monitoring
    'RealTimeActivityMonitor',
    'AgentActivity',
    'WorkCompletion',
    'get_activity_monitor',
    'initialize_monitoring',
    'cleanup_monitoring',
    
    # Verification system
    'ImplementationVerificationSystem',
    'VerificationResult',
    'get_verification_system'
]