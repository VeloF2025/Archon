#!/usr/bin/env python3
"""
TDD Enforcement API - Test-Driven Development Gate Management

This API provides endpoints for activating and managing TDD enforcement gates,
ensuring all feature development follows strict test-first principles.

CRITICAL: TDD enforcement is mandatory for all feature development.
No implementation is allowed without passing TDD compliance validation.
"""

import os
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ...agents.tdd.tdd_enforcement_gate import (
    TDDEnforcementGate,
    EnforcementResult,
    EnforcementLevel,
    FeatureStatus,
    enforce_tdd_compliance,
    is_feature_implementation_allowed
)
from ...agents.tdd.stagehand_test_engine import StagehandTestEngine
from ...agents.tdd.browserbase_executor import BrowserbaseExecutor
from ...agents.tdd.browserbase_config import test_browserbase_connection
from ...agents.tdd.file_monitor import get_file_monitor, start_tdd_file_monitoring

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tdd", tags=["TDD Enforcement"])

# Global TDD gate instance for persistent enforcement
_tdd_gate: Optional[TDDEnforcementGate] = None
_gate_active = False

# Request models
class FeatureValidationRequest(BaseModel):
    """Request to validate TDD compliance for a feature"""
    feature_name: str = Field(..., description="Name of the feature to validate")
    implementation_files: Optional[List[str]] = Field(None, description="List of implementation files")
    test_files: Optional[List[str]] = Field(None, description="List of test files") 
    force_validation: bool = Field(False, description="Force validation even if recently validated")
    project_path: str = Field(".", description="Path to project root")

class TestGenerationRequest(BaseModel):
    """Request to generate tests for a feature"""
    feature_name: str = Field(..., description="Name of the feature")
    requirements: List[str] = Field(..., description="List of feature requirements")
    acceptance_criteria: List[str] = Field(..., description="Acceptance criteria for the feature")
    test_framework: str = Field("playwright", description="Test framework to use")
    generate_e2e: bool = Field(True, description="Generate end-to-end tests")
    generate_unit: bool = Field(True, description="Generate unit tests")
    generate_integration: bool = Field(True, description="Generate integration tests")

class EnforcementConfigRequest(BaseModel):
    """Request to configure TDD enforcement settings"""
    enforcement_level: str = Field("strict", description="Enforcement level: strict, moderate, development")
    min_coverage_percentage: float = Field(95.0, description="Minimum test coverage percentage")
    enable_gaming_detection: bool = Field(True, description="Enable DGTS gaming detection")
    project_path: str = Field(".", description="Path to project root")

class FileMonitoringRequest(BaseModel):
    """Request to configure file monitoring for TDD enforcement"""
    watch_patterns: List[str] = Field(["**/*.py", "**/*.js", "**/*.ts", "**/*.tsx"], description="File patterns to monitor")
    ignore_patterns: List[str] = Field(["node_modules/**", ".git/**", "**/__pycache__/**"], description="Patterns to ignore")
    enable_real_time: bool = Field(True, description="Enable real-time file monitoring")
    block_commits: bool = Field(True, description="Block git commits without tests")

# Response models  
class EnforcementStatusResponse(BaseModel):
    """Current TDD enforcement status"""
    active: bool = Field(..., description="Whether TDD enforcement is active")
    level: str = Field(..., description="Current enforcement level")
    features_tracked: int = Field(..., description="Number of features being tracked")
    total_validations: int = Field(..., description="Total validations performed")
    success_rate: float = Field(..., description="Validation success rate percentage")
    gaming_attempts: int = Field(..., description="Number of gaming attempts detected")
    file_monitoring_active: bool = Field(..., description="Whether file monitoring is active")
    browserbase_connected: bool = Field(..., description="Whether Browserbase API is connected")

class ValidationResponse(BaseModel):
    """TDD validation result response"""
    allowed: bool = Field(..., description="Whether implementation is allowed")
    message: str = Field(..., description="Validation message")
    feature_name: str = Field(..., description="Feature being validated")
    violations: int = Field(..., description="Number of violations found")
    critical_violations: int = Field(..., description="Number of critical violations")
    gaming_score: float = Field(..., description="Gaming detection score")
    coverage_percentage: float = Field(..., description="Test coverage percentage")
    tests_exist: bool = Field(..., description="Whether tests exist")
    tests_passing: bool = Field(..., description="Whether tests are passing")

class TestGenerationResponse(BaseModel):
    """Test generation result response"""
    success: bool = Field(..., description="Whether test generation succeeded")
    message: str = Field(..., description="Generation result message")
    tests_generated: int = Field(..., description="Number of tests generated")
    files_created: List[str] = Field(..., description="List of test files created")
    coverage_estimate: float = Field(..., description="Estimated test coverage")
    stagehand_actions: int = Field(..., description="Number of Stagehand actions generated")

def get_tdd_gate() -> TDDEnforcementGate:
    """Get or create the global TDD enforcement gate"""
    global _tdd_gate, _gate_active
    
    if _tdd_gate is None or not _gate_active:
        _tdd_gate = TDDEnforcementGate(
            project_path=".",
            enforcement_level=EnforcementLevel.STRICT,
            min_coverage_percentage=95.0,
            enable_gaming_detection=True
        )
        _gate_active = True
        logger.info("🔒 TDD Enforcement Gate activated - STRICT mode")
    
    return _tdd_gate

@router.post("/enforce", response_model=EnforcementStatusResponse)
async def activate_tdd_enforcement(
    config: EnforcementConfigRequest,
    background_tasks: BackgroundTasks
):
    """
    Activate TDD enforcement gate with specified configuration
    
    This endpoint initializes and activates the TDD enforcement system,
    setting up file monitoring, gaming detection, and test-first validation.
    """
    global _tdd_gate, _gate_active
    
    try:
        logger.info(f"🚀 Activating TDD enforcement for project: {config.project_path}")
        
        # Parse enforcement level
        enforcement_level = EnforcementLevel(config.enforcement_level.lower())
        
        # Create and configure TDD gate
        _tdd_gate = TDDEnforcementGate(
            project_path=config.project_path,
            enforcement_level=enforcement_level,
            min_coverage_percentage=config.min_coverage_percentage,
            enable_gaming_detection=config.enable_gaming_detection
        )
        _gate_active = True
        
        # Start background file monitoring
        background_tasks.add_task(start_file_monitoring_task, config.project_path)
        
        # Get current stats
        stats = _tdd_gate.get_enforcement_stats()
        
        logger.info("✅ TDD Enforcement Gate activated successfully")
        
        return EnforcementStatusResponse(
            active=True,
            level=enforcement_level.value,
            features_tracked=stats.get("active_features", 0),
            total_validations=stats.get("total_validations", 0),
            success_rate=0.0 if stats.get("total_validations", 0) == 0 else 
                        (stats.get("allowed_features", 0) / stats.get("total_validations", 1)) * 100,
            gaming_attempts=stats.get("gaming_attempts", 0),
            file_monitoring_active=True,
            browserbase_connected=await check_browserbase_connection()
        )
        
    except Exception as e:
        error_msg = f"Failed to activate TDD enforcement: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/status", response_model=EnforcementStatusResponse)
async def get_tdd_status():
    """
    Get current TDD enforcement status
    
    Returns the current state of TDD enforcement including active features,
    validation statistics, and system health.
    """
    global _gate_active
    
    try:
        if not _gate_active or _tdd_gate is None:
            return EnforcementStatusResponse(
                active=False,
                level="disabled",
                features_tracked=0,
                total_validations=0,
                success_rate=0.0,
                gaming_attempts=0,
                file_monitoring_active=False,
                browserbase_connected=False
            )
        
        stats = _tdd_gate.get_enforcement_stats()
        
        return EnforcementStatusResponse(
            active=True,
            level=_tdd_gate.enforcement_level.value,
            features_tracked=stats.get("active_features", 0),
            total_validations=stats.get("total_validations", 0),
            success_rate=0.0 if stats.get("total_validations", 0) == 0 else 
                        (stats.get("allowed_features", 0) / stats.get("total_validations", 1)) * 100,
            gaming_attempts=stats.get("gaming_attempts", 0),
            file_monitoring_active=True,
            browserbase_connected=await check_browserbase_connection()
        )
        
    except Exception as e:
        error_msg = f"Failed to get TDD status: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/validate", response_model=ValidationResponse)
async def validate_feature(validation_request: FeatureValidationRequest):
    """
    Validate TDD compliance for a specific feature
    
    Performs comprehensive validation including test existence, test quality,
    coverage requirements, and gaming detection.
    """
    try:
        gate = get_tdd_gate()
        
        logger.info(f"🔍 Validating TDD compliance for feature: {validation_request.feature_name}")
        
        # Perform TDD validation
        result = await gate.validate_feature_development(
            feature_name=validation_request.feature_name,
            implementation_files=validation_request.implementation_files,
            test_files=validation_request.test_files,
            force_validation=validation_request.force_validation
        )
        
        # Extract feature validation details
        feature_validation = result.feature_validations[0] if result.feature_validations else None
        
        response = ValidationResponse(
            allowed=result.allowed,
            message=result.message,
            feature_name=validation_request.feature_name,
            violations=result.total_violations,
            critical_violations=result.critical_violations,
            gaming_score=result.gaming_score,
            coverage_percentage=feature_validation.coverage_percentage if feature_validation else 0.0,
            tests_exist=feature_validation.tests_exist if feature_validation else False,
            tests_passing=feature_validation.tests_passing if feature_validation else False
        )
        
        if result.allowed:
            logger.info(f"✅ TDD validation passed for {validation_request.feature_name}")
        else:
            logger.warning(f"❌ TDD validation failed for {validation_request.feature_name}: {result.message}")
        
        return response
        
    except Exception as e:
        error_msg = f"TDD validation failed for {validation_request.feature_name}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/generate-tests", response_model=TestGenerationResponse)
async def generate_tests_for_feature(generation_request: TestGenerationRequest):
    """
    Generate tests for a feature using Stagehand test engine
    
    Creates comprehensive test suites from requirements using natural language
    processing and browser automation patterns.
    """
    try:
        gate = get_tdd_gate()
        
        logger.info(f"🧪 Generating tests for feature: {generation_request.feature_name}")
        
        # Generate tests using the test engine
        result = await gate.generate_tests_for_feature(
            feature_name=generation_request.feature_name,
            requirements=generation_request.requirements,
            acceptance_criteria=generation_request.acceptance_criteria,
            test_framework=generation_request.test_framework,
            include_unit_tests=generation_request.generate_unit,
            include_integration_tests=generation_request.generate_integration,
            include_e2e_tests=generation_request.generate_e2e
        )
        
        # Count Stagehand actions in generated tests
        stagehand_actions = 0
        for test in result.tests_generated:
            stagehand_actions += len(test.stagehand_actions)
        
        response = TestGenerationResponse(
            success=result.success,
            message=result.message,
            tests_generated=result.total_tests,
            files_created=[test.file_path for test in result.tests_generated],
            coverage_estimate=result.coverage_percentage,
            stagehand_actions=stagehand_actions
        )
        
        if result.success:
            logger.info(f"✅ Generated {result.total_tests} tests for {generation_request.feature_name}")
        else:
            logger.warning(f"❌ Test generation failed for {generation_request.feature_name}: {result.message}")
        
        return response
        
    except Exception as e:
        error_msg = f"Test generation failed for {generation_request.feature_name}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/quick-check/{feature_name}")
async def quick_tdd_check(feature_name: str, project_path: str = "."):
    """
    Quick TDD compliance check for a feature
    
    Fast validation to check if feature implementation is allowed without
    full validation process. Used for pre-commit hooks and CI checks.
    """
    try:
        logger.info(f"⚡ Quick TDD check for feature: {feature_name}")
        
        # Perform quick validation
        allowed = is_feature_implementation_allowed(
            feature_name=feature_name,
            project_path=project_path,
            enforcement_level=EnforcementLevel.STRICT,
            enable_gaming_detection=True
        )
        
        return {
            "feature_name": feature_name,
            "implementation_allowed": allowed,
            "message": "Implementation allowed" if allowed else "TDD compliance required",
            "check_type": "quick",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Quick TDD check failed for {feature_name}: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/report")
async def get_enforcement_report():
    """
    Get comprehensive TDD enforcement report
    
    Returns detailed report with feature status, violation history,
    gaming detection results, and system metrics.
    """
    try:
        if not _gate_active or _tdd_gate is None:
            return {
                "message": "TDD enforcement not active",
                "active": False,
                "timestamp": datetime.now().isoformat()
            }
        
        report = await _tdd_gate.create_enforcement_report()
        return report
        
    except Exception as e:
        error_msg = f"Failed to generate enforcement report: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/file-monitoring/start")
async def start_file_monitoring_endpoint(monitoring_request: FileMonitoringRequest):
    """
    Start file monitoring for TDD enforcement
    
    Activates real-time file change monitoring to enforce test-first development.
    Blocks commits and file changes that violate TDD principles.
    """
    try:
        logger.info("🔍 Starting TDD file monitoring...")
        
        # This would integrate with file system watchers
        # For now, return success status
        
        # Start file monitoring
        success = await start_tdd_file_monitoring()
        
        return {
            "monitoring_active": success,
            "watch_patterns": monitoring_request.watch_patterns,
            "ignore_patterns": monitoring_request.ignore_patterns,
            "real_time_enabled": monitoring_request.enable_real_time,
            "commit_blocking_enabled": monitoring_request.block_commits,
            "message": "File monitoring activated for TDD enforcement" if success else "File monitoring activation failed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Failed to start file monitoring: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/browserbase/status")
async def get_browserbase_status():
    """
    Check Browserbase API connection status
    
    Verifies connection to Browserbase cloud testing infrastructure
    for Stagehand test execution.
    """
    try:
        result = await test_browserbase_connection()
        
        return {
            **result,
            "service": "browserbase",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Failed to check Browserbase status: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Background task functions
async def start_file_monitoring_task(project_path: str):
    """Background task to start file monitoring"""
    try:
        logger.info(f"🔍 Starting file monitoring for project: {project_path}")
        
        success = await start_tdd_file_monitoring(project_path)
        
        if success:
            logger.info("✅ File monitoring active for TDD enforcement")
        else:
            logger.warning("⚠️  File monitoring activation failed - watchdog may not be available")
        
    except Exception as e:
        logger.error(f"File monitoring startup failed: {str(e)}")

async def check_browserbase_connection() -> bool:
    """Check if Browserbase API is accessible"""
    try:
        result = await test_browserbase_connection()
        return result.get("connected", False)
    except Exception as e:
        logger.error(f"Browserbase connection check failed: {str(e)}")
        return False

# Pre-commit hook integration
@router.post("/pre-commit/validate")
async def validate_pre_commit():
    """
    Validate TDD compliance before commit
    
    Used by git pre-commit hooks to ensure all changes comply with
    TDD requirements before allowing commits.
    """
    try:
        if not _gate_active or _tdd_gate is None:
            return {
                "allowed": False,
                "message": "TDD enforcement not active - activation required",
                "violations": ["TDD_GATE_INACTIVE"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Get all tracked features and validate them
        violations = []
        blocked_features = []
        
        for feature_name, status in _tdd_gate.feature_status.items():
            if status not in [FeatureStatus.IMPLEMENTATION_ALLOWED, FeatureStatus.APPROVED]:
                blocked_features.append(feature_name)
                violations.append(f"Feature '{feature_name}' status: {status.value}")
        
        allowed = len(blocked_features) == 0
        
        return {
            "allowed": allowed,
            "message": "Commit allowed" if allowed else f"Blocked features: {', '.join(blocked_features)}",
            "violations": violations,
            "blocked_features": blocked_features,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Pre-commit validation failed: {str(e)}"
        logger.error(error_msg)
        return {
            "allowed": False,
            "message": error_msg,
            "violations": ["VALIDATION_ERROR"],
            "timestamp": datetime.now().isoformat()
        }