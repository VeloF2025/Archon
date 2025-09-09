"""
Community Pattern Validator

Provides comprehensive validation for community-submitted patterns,
including security validation, quality assessment, and community review coordination.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from .pattern_models import (
    Pattern, PatternValidationResult, PatternSubmission,
    PatternType, PatternComplexity, PatternTechnology
)
import logging

logger = logging.getLogger(__name__)


class SecurityPatternValidator:
    """Validates patterns for security vulnerabilities and malicious content."""
    
    # Dangerous patterns in pattern definitions
    DANGEROUS_PATTERNS = [
        # Command injection risks
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess\.call',
        r'os\.system',
        r'shell=True',
        
        # File system risks
        r'\.\./|\.\.\)',
        r'/etc/passwd',
        r'/etc/shadow',
        r'~/.ssh',
        r'rm\s+-rf\s+/',
        
        # Network security risks
        r'curl.*\|.*bash',
        r'wget.*\|.*bash',
        r'http://[^/]*:[^/]*@',  # Credentials in URLs
        
        # Hardcoded secrets patterns
        r'password\s*=\s*["\'][^"\']{8,}["\']',
        r'secret\s*=\s*["\'][^"\']{16,}["\']',
        r'api_key\s*=\s*["\'][^"\']{16,}["\']',
        r'token\s*=\s*["\'][^"\']{20,}["\']',
        
        # Docker security risks
        r'--privileged',
        r'user.*root',
        r'chmod\s+777',
        
        # Database security
        r'SELECT\s+\*\s+FROM\s+\w+\s+WHERE\s+\w+\s*=\s*["\']?\$',  # SQL injection pattern
    ]
    
    # Suspicious workflow commands
    DANGEROUS_COMMANDS = [
        'sudo rm -rf /',
        'chmod 777',
        'chown -R root',
        'dd if=/dev/zero',
        'fork bomb',
        ':(){ :|:& };:',
        'curl | sh',
        'wget | sh',
        'eval $(curl',
        'cat /etc/passwd'
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
    
    async def validate_pattern_security(self, pattern: Pattern) -> Dict[str, Any]:
        """
        Comprehensive security validation of a pattern.
        
        Returns:
            Dict with security validation results
        """
        security_issues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'warnings': []
        }
        
        # Check pattern metadata
        await self._check_metadata_security(pattern.metadata, security_issues)
        
        # Check workflow commands
        await self._check_workflow_security(pattern.workflows, security_issues)
        
        # Check template variables
        await self._check_variable_security(pattern.template_variables, security_issues)
        
        # Check component configurations
        await self._check_component_security(pattern.components, security_issues)
        
        # Calculate overall security score
        security_score = self._calculate_security_score(security_issues)
        
        return {
            'security_score': security_score,
            'issues': security_issues,
            'passed': security_score >= 0.7,  # 70% threshold for passing
            'requires_review': security_score < 0.9,  # Manual review if < 90%
        }
    
    async def _check_metadata_security(self, metadata: Any, issues: Dict[str, List]):
        """Check pattern metadata for security issues."""
        # Check for suspicious URLs
        suspicious_urls = []
        for url in [metadata.documentation_url] + getattr(metadata, 'examples', []) + getattr(metadata, 'tutorials', []):
            if url and self._is_suspicious_url(url):
                suspicious_urls.append(url)
        
        if suspicious_urls:
            issues['medium'].append(f"Suspicious URLs detected: {suspicious_urls}")
        
        # Check for inappropriate content in description
        if self._contains_inappropriate_content(metadata.description):
            issues['high'].append("Pattern description contains inappropriate content")
    
    async def _check_workflow_security(self, workflows: List, issues: Dict[str, List]):
        """Check workflow commands for security risks."""
        for workflow in workflows:
            for command in workflow.commands:
                # Check for dangerous commands
                for dangerous_cmd in self.DANGEROUS_COMMANDS:
                    if dangerous_cmd.lower() in command.lower():
                        issues['critical'].append(f"Dangerous command in workflow '{workflow.name}': {command}")
                
                # Check against regex patterns
                for pattern in self.compiled_patterns:
                    if pattern.search(command):
                        issues['high'].append(f"Security pattern match in workflow '{workflow.name}': {command}")
                
                # Check for unvalidated input usage
                if '$' in command and not self._is_safe_variable_usage(command):
                    issues['medium'].append(f"Potentially unsafe variable usage in command: {command}")
    
    async def _check_variable_security(self, variables: Dict[str, Any], issues: Dict[str, List]):
        """Check template variables for security risks."""
        for var_name, var_value in variables.items():
            if isinstance(var_value, str):
                # Check for hardcoded secrets
                for pattern in self.compiled_patterns:
                    if pattern.search(var_value):
                        issues['high'].append(f"Potential secret in variable '{var_name}': {var_value[:50]}...")
                
                # Check for command injection risks
                if any(char in var_value for char in [';', '|', '&', '`']):
                    issues['medium'].append(f"Variable '{var_name}' contains shell metacharacters")
    
    async def _check_component_security(self, components: List, issues: Dict[str, List]):
        """Check component configurations for security issues."""
        for component in components:
            config = getattr(component, 'configuration', {})
            
            # Check for insecure configurations
            insecure_configs = {
                'debug': True,
                'ssl_verify': False,
                'auth_required': False,
                'cors_allow_all': True
            }
            
            for key, insecure_value in insecure_configs.items():
                if config.get(key) == insecure_value:
                    issues['medium'].append(f"Insecure configuration in component '{component.name}': {key}={insecure_value}")
    
    def _is_suspicious_url(self, url: str) -> bool:
        """Check if a URL is suspicious."""
        suspicious_domains = [
            'bit.ly', 'tinyurl.com', 'short.link', 'goo.gl',  # URL shorteners
            'dropbox.com/s/', 'drive.google.com',  # File sharing (can be risky)
        ]
        
        suspicious_patterns = [
            r'https?://\d+\.\d+\.\d+\.\d+',  # Raw IP addresses
            r'[a-zA-Z0-9]\.tk$',  # Free domains
            r'[a-zA-Z0-9]\.ml$',
            r'[a-zA-Z0-9]\.cf$',
        ]
        
        return any(domain in url for domain in suspicious_domains) or \
               any(re.search(pattern, url) for pattern in suspicious_patterns)
    
    def _contains_inappropriate_content(self, text: str) -> bool:
        """Check for inappropriate content in text."""
        # This is a basic implementation - in production, you'd use more sophisticated content filtering
        inappropriate_words = ['hack', 'crack', 'exploit', 'backdoor', 'malware']
        return any(word in text.lower() for word in inappropriate_words)
    
    def _is_safe_variable_usage(self, command: str) -> bool:
        """Check if variable usage in command is safe."""
        # Safe patterns: ${VAR}, $VAR (with word boundaries)
        safe_patterns = [
            r'\$\{[A-Z_][A-Z0-9_]*\}',
            r'\$[A-Z_][A-Z0-9_]*\b'
        ]
        
        # Find all variable usages
        var_usages = re.findall(r'\$[^;\|&\s]+', command)
        
        for usage in var_usages:
            if not any(re.match(pattern, usage) for pattern in safe_patterns):
                return False
        
        return True
    
    def _calculate_security_score(self, issues: Dict[str, List]) -> float:
        """Calculate overall security score based on issues."""
        base_score = 1.0
        
        # Deduct points based on issue severity
        base_score -= len(issues['critical']) * 0.3
        base_score -= len(issues['high']) * 0.2
        base_score -= len(issues['medium']) * 0.1
        base_score -= len(issues['low']) * 0.05
        
        return max(0.0, base_score)


class QualityPatternValidator:
    """Validates patterns for quality, completeness, and best practices."""
    
    def __init__(self):
        pass
    
    async def validate_pattern_quality(self, pattern: Pattern) -> Dict[str, Any]:
        """
        Comprehensive quality validation of a pattern.
        
        Returns:
            Dict with quality validation results
        """
        quality_metrics = {
            'completeness': await self._check_completeness(pattern),
            'documentation': await self._check_documentation(pattern),
            'best_practices': await self._check_best_practices(pattern),
            'maintainability': await self._check_maintainability(pattern),
            'usability': await self._check_usability(pattern)
        }
        
        # Calculate overall quality score
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            'quality_score': quality_score,
            'metrics': quality_metrics,
            'passed': quality_score >= 0.7,
            'recommendations': await self._generate_quality_recommendations(pattern, quality_metrics)
        }
    
    async def _check_completeness(self, pattern: Pattern) -> float:
        """Check if pattern is complete."""
        score = 0.0
        total_checks = 8
        
        # Required metadata
        if pattern.metadata.name and len(pattern.metadata.name) >= 5:
            score += 1
        
        if pattern.metadata.description and len(pattern.metadata.description) >= 20:
            score += 1
        
        if pattern.metadata.version and re.match(r'^\d+\.\d+\.\d+', pattern.metadata.version):
            score += 1
        
        if pattern.metadata.author and len(pattern.metadata.author) >= 2:
            score += 1
        
        # Components
        if pattern.components and len(pattern.components) > 0:
            score += 1
        
        # Workflows
        if pattern.workflows and len(pattern.workflows) > 0:
            score += 1
        
        # Technologies
        if pattern.metadata.technologies and len(pattern.metadata.technologies) > 0:
            score += 1
        
        # Tags
        if pattern.metadata.tags and len(pattern.metadata.tags) >= 3:
            score += 1
        
        return score / total_checks
    
    async def _check_documentation(self, pattern: Pattern) -> float:
        """Check documentation quality."""
        score = 0.0
        total_checks = 6
        
        # Description quality
        description = pattern.metadata.description
        if description:
            if len(description) >= 50:
                score += 0.5
            if len(description.split('.')) >= 2:  # Multiple sentences
                score += 0.5
        
        # Documentation URL
        if pattern.documentation_url:
            score += 1
        
        # Examples
        if pattern.examples and len(pattern.examples) > 0:
            score += 1
        
        # Tutorials
        if pattern.tutorials and len(pattern.tutorials) > 0:
            score += 1
        
        # Component descriptions
        if pattern.components:
            described_components = sum(
                1 for comp in pattern.components 
                if comp.description and len(comp.description) >= 10
            )
            score += (described_components / len(pattern.components))
        
        # Workflow descriptions
        if pattern.workflows:
            described_workflows = sum(
                1 for workflow in pattern.workflows 
                if workflow.description and len(workflow.description) >= 10
            )
            score += (described_workflows / len(pattern.workflows))
        
        return score / total_checks
    
    async def _check_best_practices(self, pattern: Pattern) -> float:
        """Check adherence to best practices."""
        score = 0.0
        total_checks = 5
        
        # Proper naming convention
        if pattern.metadata.name:
            # Should be descriptive and follow naming conventions
            if re.match(r'^[A-Z][a-zA-Z\s]+Pattern$', pattern.metadata.name):
                score += 1
            elif len(pattern.metadata.name) >= 10:  # At least descriptive
                score += 0.5
        
        # Technology versioning
        if pattern.metadata.technologies:
            versioned_techs = sum(
                1 for tech in pattern.metadata.technologies 
                if tech.version is not None
            )
            score += (versioned_techs / len(pattern.metadata.technologies))
        
        # Component dependencies properly defined
        if pattern.components:
            properly_defined_deps = sum(
                1 for comp in pattern.components
                if isinstance(comp.depends_on, list)
            )
            score += (properly_defined_deps / len(pattern.components))
        
        # Workflow ordering
        if pattern.workflows:
            properly_ordered = all(
                i == 0 or pattern.workflows[i].step > pattern.workflows[i-1].step
                for i in range(len(pattern.workflows))
            )
            if properly_ordered:
                score += 1
        
        # License specified
        if hasattr(pattern.metadata, 'license') and pattern.metadata.license:
            score += 1
        
        return score / total_checks
    
    async def _check_maintainability(self, pattern: Pattern) -> float:
        """Check pattern maintainability."""
        score = 0.0
        total_checks = 4
        
        # Technology alternatives provided
        if pattern.metadata.technologies:
            with_alternatives = sum(
                1 for tech in pattern.metadata.technologies
                if tech.alternatives and len(tech.alternatives) > 0
            )
            score += (with_alternatives / len(pattern.metadata.technologies))
        
        # Validation rules defined
        if pattern.validation_rules and len(pattern.validation_rules) > 0:
            score += 1
        
        # Test suite available
        if pattern.test_suite_url:
            score += 1
        
        # Update frequency (based on metadata)
        if hasattr(pattern.metadata, 'updated_at') and pattern.metadata.updated_at:
            # Recently updated is good for maintainability
            days_since_update = (datetime.utcnow() - pattern.metadata.updated_at).days
            if days_since_update <= 30:
                score += 1
            elif days_since_update <= 90:
                score += 0.5
        
        return score / total_checks
    
    async def _check_usability(self, pattern: Pattern) -> float:
        """Check pattern usability."""
        score = 0.0
        total_checks = 5
        
        # Complexity appropriate for pattern type
        complexity_scores = {
            PatternComplexity.BEGINNER: 1.0,
            PatternComplexity.INTERMEDIATE: 0.8,
            PatternComplexity.ADVANCED: 0.6,
            PatternComplexity.EXPERT: 0.4
        }
        score += complexity_scores.get(pattern.metadata.complexity, 0.5)
        
        # Clear workflow steps
        if pattern.workflows:
            clear_workflows = sum(
                1 for workflow in pattern.workflows
                if workflow.commands and len(workflow.commands) > 0
            )
            score += (clear_workflows / len(pattern.workflows))
        
        # Estimated durations provided
        if pattern.workflows:
            timed_workflows = sum(
                1 for workflow in pattern.workflows
                if workflow.estimated_duration is not None
            )
            score += (timed_workflows / len(pattern.workflows))
        
        # Multiple provider support
        provider_count = len(pattern.metadata.providers) if pattern.metadata.providers else 0
        if provider_count >= 3:
            score += 1
        elif provider_count >= 2:
            score += 0.5
        
        # Template integration
        if pattern.generates_templates and len(pattern.generates_templates) > 0:
            score += 1
        
        return score / total_checks
    
    async def _generate_quality_recommendations(self, pattern: Pattern, metrics: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if metrics['completeness'] < 0.8:
            recommendations.append("Add more comprehensive metadata and component descriptions")
        
        if metrics['documentation'] < 0.7:
            recommendations.append("Provide better documentation with examples and tutorials")
        
        if metrics['best_practices'] < 0.7:
            recommendations.append("Follow naming conventions and provide technology versions")
        
        if metrics['maintainability'] < 0.6:
            recommendations.append("Add validation rules and consider providing a test suite")
        
        if metrics['usability'] < 0.7:
            recommendations.append("Simplify workflows and provide time estimates for steps")
        
        return recommendations


class CommunityPatternValidator:
    """Main validator that coordinates security, quality, and community validation."""
    
    def __init__(self):
        self.security_validator = SecurityPatternValidator()
        self.quality_validator = QualityPatternValidator()
    
    async def validate_pattern_submission(self, submission: PatternSubmission) -> PatternValidationResult:
        """
        Comprehensive validation of a pattern submission.
        
        Args:
            submission: PatternSubmission object
        
        Returns:
            PatternValidationResult with validation details
        """
        try:
            logger.info(f"Validating pattern submission | id={submission.id} | pattern={submission.pattern.id}")
            
            # Security validation
            security_result = await self.security_validator.validate_pattern_security(submission.pattern)
            
            # Quality validation
            quality_result = await self.quality_validator.validate_pattern_quality(submission.pattern)
            
            # Combine results
            validation_result = await self._combine_validation_results(
                submission, security_result, quality_result
            )
            
            logger.info(f"Pattern validation complete | id={submission.id} | valid={validation_result.valid} | confidence={validation_result.confidence_score}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Pattern validation failed | id={submission.id} | error={str(e)}")
            
            # Return failed validation
            return PatternValidationResult(
                pattern_id=submission.pattern.id,
                valid=False,
                confidence_score=0.0,
                errors=[f"Validation failed: {str(e)}"]
            )
    
    async def _combine_validation_results(
        self, 
        submission: PatternSubmission,
        security_result: Dict[str, Any],
        quality_result: Dict[str, Any]
    ) -> PatternValidationResult:
        """Combine security and quality validation results."""
        
        # Collect all errors and warnings
        errors = []
        warnings = []
        suggestions = []
        
        # Security issues
        for severity, issues in security_result['issues'].items():
            if severity in ['critical', 'high']:
                errors.extend(issues)
            elif severity == 'medium':
                warnings.extend(issues)
            else:
                suggestions.extend(issues)
        
        # Quality recommendations
        suggestions.extend(quality_result.get('recommendations', []))
        
        # Calculate overall confidence score
        security_score = security_result['security_score']
        quality_score = quality_result['quality_score']
        
        # Weighted combination (security is more important)
        confidence_score = (security_score * 0.6 + quality_score * 0.4)
        
        # Pattern is valid if both security and quality pass minimum thresholds
        is_valid = (
            security_result['passed'] and 
            quality_result['passed'] and 
            len(errors) == 0
        )
        
        return PatternValidationResult(
            pattern_id=submission.pattern.id,
            valid=is_valid,
            confidence_score=confidence_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            implementation_tested=False,  # Would need actual testing infrastructure
            performance_benchmarks={
                'security_score': security_score,
                'quality_score': quality_score,
                'combined_score': confidence_score
            }
        )
    
    async def validate_community_pattern(self, pattern: Pattern) -> PatternValidationResult:
        """Validate a pattern (convenience method without submission wrapper)."""
        # Create temporary submission
        temp_submission = PatternSubmission(
            pattern=pattern,
            submitter="system"
        )
        
        return await self.validate_pattern_submission(temp_submission)
    
    async def batch_validate_patterns(self, patterns: List[Pattern]) -> List[PatternValidationResult]:
        """Validate multiple patterns concurrently."""
        validation_tasks = [
            self.validate_community_pattern(pattern)
            for pattern in patterns
        ]
        
        return await asyncio.gather(*validation_tasks, return_exceptions=True)