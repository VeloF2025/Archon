"""
Pydantic models for External Validator
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ValidationStatus(str, Enum):
    """Validation status enumeration"""
    PASS = "PASS"
    FAIL = "FAIL"
    UNSURE = "UNSURE"
    ERROR = "ERROR"


class ValidationSeverity(str, Enum):
    """Issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    """Individual validation issue"""
    
    severity: ValidationSeverity = Field(
        description="Severity of the issue"
    )
    category: str = Field(
        description="Category of issue (e.g., 'hallucination', 'syntax', 'logic')"
    )
    message: str = Field(
        description="Description of the issue"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="File path where issue was found"
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number where issue was found"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Evidence or context for the issue"
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested fix for the issue"
    )


class ValidationEvidence(BaseModel):
    """Evidence supporting validation verdict"""
    
    source: str = Field(
        description="Source of evidence (e.g., 'pytest', 'ruff', 'cross-check')"
    )
    content: str = Field(
        description="Evidence content"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this evidence"
    )
    provenance: Optional[str] = Field(
        default=None,
        description="Provenance tracking (e.g., 'path@commit:span')"
    )


class ValidationRequest(BaseModel):
    """Request for validation"""
    
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Original prompt to validate"
    )
    output: str = Field(
        description="Output to validate"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="PRP context (max 5k tokens)"
    )
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="File paths to validate"
    )
    validation_type: Literal["code", "documentation", "prompt", "output", "full"] = Field(
        default="full",
        description="Type of validation to perform"
    )
    enable_deterministic: bool = Field(
        default=True,
        description="Enable deterministic checks"
    )
    enable_cross_check: bool = Field(
        default=True,
        description="Enable cross-checking"
    )
    temperature_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.2,
        description="Override default temperature"
    )


class ValidationMetrics(BaseModel):
    """Metrics from validation"""
    
    total_checks: int = Field(
        default=0,
        description="Total number of checks performed"
    )
    passed_checks: int = Field(
        default=0,
        description="Number of passed checks"
    )
    failed_checks: int = Field(
        default=0,
        description="Number of failed checks"
    )
    hallucination_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Detected hallucination rate"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )
    token_count: int = Field(
        default=0,
        description="Total tokens processed"
    )
    validation_time_ms: int = Field(
        default=0,
        description="Validation time in milliseconds"
    )


class ValidationResponse(BaseModel):
    """Response from validation"""
    
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier"
    )
    status: ValidationStatus = Field(
        description="Overall validation status"
    )
    issues: List[ValidationIssue] = Field(
        default_factory=list,
        description="List of validation issues found"
    )
    evidence: List[ValidationEvidence] = Field(
        default_factory=list,
        description="Supporting evidence for verdict"
    )
    fixes: List[str] = Field(
        default_factory=list,
        description="Suggested fixes"
    )
    metrics: ValidationMetrics = Field(
        default_factory=ValidationMetrics,
        description="Validation metrics"
    )
    summary: str = Field(
        description="Human-readable summary of validation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Validation timestamp"
    )
    validator_version: str = Field(
        default="1.0.0",
        description="Validator version"
    )


class ConfigureValidatorRequest(BaseModel):
    """Request to configure validator"""
    
    provider: Optional[Literal["deepseek", "openai"]] = Field(
        default=None,
        description="LLM provider"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM provider"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.2,
        description="Temperature setting"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold"
    )
    enable_proactive_triggers: Optional[bool] = Field(
        default=None,
        description="Enable proactive validation triggers"
    )


class ValidatorHealthResponse(BaseModel):
    """Health check response"""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Service health status"
    )
    version: str = Field(
        description="Validator version"
    )
    llm_provider: str = Field(
        description="Current LLM provider"
    )
    llm_connected: bool = Field(
        description="LLM connection status"
    )
    deterministic_available: bool = Field(
        description="Deterministic checks availability"
    )
    uptime_seconds: int = Field(
        description="Service uptime in seconds"
    )
    total_validations: int = Field(
        description="Total validations performed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )