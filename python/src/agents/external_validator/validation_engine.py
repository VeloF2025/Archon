"""
Core validation engine for External Validator
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import logging
import subprocess
from pathlib import Path

from .config import ValidatorConfig
from .models import (
    ValidationRequest,
    ValidationResponse,
    ValidationIssue,
    ValidationEvidence,
    ValidationStatus,
    ValidationSeverity,
    ValidationMetrics
)
from .llm_client import LLMClient
from .deterministic import DeterministicChecker
from .cross_check import CrossChecker

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Main validation engine orchestrating all checks"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.deterministic_checker = DeterministicChecker()
        self.cross_checker = CrossChecker(config, self.llm_client)
        
        # Metrics tracking
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
    
    async def initialize(self):
        """Initialize validation engine"""
        
        logger.info("Initializing validation engine")
        
        # Initialize LLM client
        await self.llm_client.initialize()
        
        # Verify deterministic tools
        self.deterministic_checker.verify_tools()
        
        logger.info("Validation engine initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        
        await self.llm_client.cleanup()
    
    async def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Perform validation on request"""
        
        start_time = time.time()
        issues: List[ValidationIssue] = []
        evidence: List[ValidationEvidence] = []
        
        # Run deterministic checks if enabled
        if request.enable_deterministic and self.config.validation_config.enable_deterministic:
            det_issues, det_evidence = await self.run_deterministic_checks(request)
            issues.extend(det_issues)
            evidence.extend(det_evidence)
        
        # Run cross-checks if enabled
        if request.enable_cross_check and self.config.validation_config.enable_cross_check:
            cross_issues, cross_evidence = await self.run_cross_checks(request)
            issues.extend(cross_issues)
            evidence.extend(cross_evidence)
        
        # Calculate metrics
        validation_time = int((time.time() - start_time) * 1000)
        
        # Determine overall status
        status = self.determine_status(issues)
        
        # Generate summary
        summary = self.generate_summary(status, issues, evidence)
        
        # Calculate hallucination rate
        hallucination_rate = self.calculate_hallucination_rate(issues)
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence_score(evidence)
        
        # Generate fixes
        fixes = self.generate_fixes(issues)
        
        # Create metrics
        metrics = ValidationMetrics(
            total_checks=len(evidence),
            passed_checks=sum(1 for e in evidence if e.confidence >= 0.9),
            failed_checks=len(issues),
            hallucination_rate=hallucination_rate,
            confidence_score=confidence_score,
            token_count=0,  # Will be updated by LLM client
            validation_time_ms=validation_time
        )
        
        # Update global metrics
        self.total_checks += metrics.total_checks
        self.passed_checks += metrics.passed_checks
        self.failed_checks += metrics.failed_checks
        
        return ValidationResponse(
            request_id=request.request_id,
            status=status,
            issues=issues,
            evidence=evidence,
            fixes=fixes,
            metrics=metrics,
            summary=summary
        )
    
    async def run_deterministic_checks(
        self, 
        request: ValidationRequest
    ) -> tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Run deterministic validation checks"""
        
        issues = []
        evidence = []
        
        try:
            # Run checks based on validation type
            if request.validation_type in ["code", "full"]:
                # Run code quality checks
                if request.file_paths:
                    for file_path in request.file_paths:
                        file_issues, file_evidence = await self.deterministic_checker.check_file(
                            Path(file_path)
                        )
                        issues.extend(file_issues)
                        evidence.extend(file_evidence)
                
                # Check output as code
                if request.output:
                    code_issues, code_evidence = await self.deterministic_checker.check_code(
                        request.output
                    )
                    issues.extend(code_issues)
                    evidence.extend(code_evidence)
            
            # Add evidence for successful checks
            if not issues:
                evidence.append(ValidationEvidence(
                    source="deterministic",
                    content="All deterministic checks passed",
                    confidence=1.0
                ))
                
        except Exception as e:
            logger.error(f"Deterministic check error: {e}", exc_info=True)
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="system",
                message=f"Deterministic check failed: {str(e)}"
            ))
        
        return issues, evidence
    
    async def run_cross_checks(
        self,
        request: ValidationRequest
    ) -> tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Run cross-checking with context"""
        
        issues = []
        evidence = []
        
        try:
            # Perform cross-checks
            cross_issues, cross_evidence = await self.cross_checker.validate(
                output=request.output,
                context=request.context,
                prompt=request.prompt
            )
            
            issues.extend(cross_issues)
            evidence.extend(cross_evidence)
            
        except Exception as e:
            logger.error(f"Cross-check error: {e}", exc_info=True)
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="system",
                message=f"Cross-check failed: {str(e)}"
            ))
        
        return issues, evidence
    
    def determine_status(self, issues: List[ValidationIssue]) -> ValidationStatus:
        """Determine overall validation status - DECISIVE for DGTS/NLNH compliance"""
        
        if not issues:
            return ValidationStatus.PASS
        
        # Check for gaming patterns first - these are always failures
        gaming_issues = [i for i in issues if i.category == "gaming"]
        if gaming_issues:
            # Any gaming pattern should fail validation
            return ValidationStatus.FAIL
        
        # Check for critical issues - immediate fail
        critical_count = sum(1 for i in issues if i.severity == ValidationSeverity.CRITICAL)
        if critical_count > 0:
            return ValidationStatus.FAIL
        
        # Check for security issues - always fail
        security_issues = [i for i in issues if i.category == "security" or "security" in i.message.lower() or "eval" in i.message.lower()]
        if security_issues:
            return ValidationStatus.FAIL
        
        # Check for errors
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        if error_count > 0:
            # Any error should fail
            return ValidationStatus.FAIL
        
        # Check for performance issues (like unbounded recursion)
        performance_issues = [i for i in issues if i.category == "performance" or "recursion" in i.message.lower() or "performance" in i.message.lower()]
        if performance_issues:
            # Performance issues in production code should fail
            return ValidationStatus.FAIL
        
        # Check for warnings - be more decisive
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        if warning_count > 3:
            # Many warnings indicate poor code quality
            return ValidationStatus.FAIL
        elif warning_count > 0:
            # Check if warnings are about actual problems
            serious_warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING and 
                              any(keyword in i.message.lower() for keyword in ["error", "fail", "invalid", "missing", "undefined"])]
            if serious_warnings:
                return ValidationStatus.FAIL
            # Minor warnings (style, info) can pass
            return ValidationStatus.PASS
        
        return ValidationStatus.PASS
    
    def generate_summary(
        self,
        status: ValidationStatus,
        issues: List[ValidationIssue],
        evidence: List[ValidationEvidence]
    ) -> str:
        """Generate human-readable summary"""
        
        if status == ValidationStatus.PASS:
            return f"Validation passed with {len(evidence)} supporting evidence points."
        
        issue_summary = []
        for severity in ValidationSeverity:
            count = sum(1 for i in issues if i.severity == severity)
            if count > 0:
                issue_summary.append(f"{count} {severity.value}")
        
        return f"Validation {status.value.lower()} with {', '.join(issue_summary)} issues found."
    
    def calculate_hallucination_rate(self, issues: List[ValidationIssue]) -> float:
        """Calculate hallucination rate from issues"""
        
        hallucination_issues = sum(
            1 for i in issues 
            if i.category == "hallucination" or "hallucination" in i.message.lower()
        )
        
        if not issues:
            return 0.0
        
        return min(hallucination_issues / len(issues), 1.0)
    
    def calculate_confidence_score(self, evidence: List[ValidationEvidence]) -> float:
        """Calculate overall confidence score"""
        
        if not evidence:
            return 0.0
        
        total_confidence = sum(e.confidence for e in evidence)
        return total_confidence / len(evidence)
    
    def generate_fixes(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate suggested fixes from issues"""
        
        fixes = []
        seen_fixes = set()
        
        for issue in issues:
            if issue.suggested_fix and issue.suggested_fix not in seen_fixes:
                fixes.append(issue.suggested_fix)
                seen_fixes.add(issue.suggested_fix)
        
        return fixes
    
    async def check_llm_connection(self) -> bool:
        """Check if LLM is connected"""
        
        return await self.llm_client.check_connection()
    
    def check_deterministic_tools(self) -> bool:
        """Check if deterministic tools are available"""
        
        return self.deterministic_checker.tools_available()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get validation engine metrics"""
        
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "success_rate": self.passed_checks / max(self.total_checks, 1),
            "llm_metrics": await self.llm_client.get_metrics()
        }