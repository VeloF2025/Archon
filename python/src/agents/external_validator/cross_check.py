"""
Cross-checking validation with context
"""

import re
from typing import List, Tuple, Dict, Any, Optional
import logging

from .models import ValidationIssue, ValidationEvidence, ValidationSeverity
from .config import ValidatorConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class CrossChecker:
    """Performs cross-checking validation against context"""
    
    def __init__(self, config: ValidatorConfig, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
    
    async def validate(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Validate output against context"""
        
        issues = []
        evidence = []
        
        # Perform different types of cross-checks
        if context:
            # Check against PRP context
            if "prp" in context:
                prp_issues, prp_evidence = await self._check_against_prp(output, context["prp"])
                issues.extend(prp_issues)
                evidence.extend(prp_evidence)
            
            # Check against file context
            if "files" in context:
                file_issues, file_evidence = await self._check_against_files(output, context["files"])
                issues.extend(file_issues)
                evidence.extend(file_evidence)
            
            # Check against entity context (Graphiti)
            if "entities" in context:
                entity_issues, entity_evidence = await self._check_entities(output, context["entities"])
                issues.extend(entity_issues)
                evidence.extend(entity_evidence)
            
            # Check against documentation (REF)
            if "docs" in context:
                doc_issues, doc_evidence = await self._check_documentation(output, context["docs"])
                issues.extend(doc_issues)
                evidence.extend(doc_evidence)
        
        # Perform LLM-based validation if configured
        if self.llm_client.client:
            llm_issues, llm_evidence = await self._llm_validation(output, context, prompt)
            issues.extend(llm_issues)
            evidence.extend(llm_evidence)
        
        # Apply confidence filtering (DeepConf)
        filtered_issues, filtered_evidence = self._apply_confidence_filter(issues, evidence)
        
        return filtered_issues, filtered_evidence
    
    async def _check_against_prp(
        self,
        output: str,
        prp_context: str
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check output against PRP requirements"""
        
        issues = []
        evidence = []
        
        # Extract requirements from PRP
        requirements = self._extract_requirements(prp_context)
        
        # Check if output addresses requirements
        for req in requirements:
            if not self._requirement_addressed(output, req):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="requirement",
                    message=f"Requirement not addressed: {req[:100]}...",
                    suggested_fix="Ensure all PRP requirements are implemented"
                ))
            else:
                evidence.append(ValidationEvidence(
                    source="prp-check",
                    content=f"Requirement addressed: {req[:50]}...",
                    confidence=0.8
                ))
        
        return issues, evidence
    
    async def _check_against_files(
        self,
        output: str,
        file_context: List[str]
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check output against file context"""
        
        issues = []
        evidence = []
        
        # Check for file path references
        file_paths_in_output = re.findall(r'[\w/\\]+\.\w+', output)
        
        for path in file_paths_in_output:
            if path not in file_context:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="file-reference",
                    message=f"Referenced file not in context: {path}",
                    suggested_fix="Verify file exists and is accessible"
                ))
            else:
                evidence.append(ValidationEvidence(
                    source="file-check",
                    content=f"Valid file reference: {path}",
                    confidence=1.0,
                    provenance=f"{path}@current"
                ))
        
        return issues, evidence
    
    async def _check_entities(
        self,
        output: str,
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check entities mentioned in output (Graphiti integration)"""
        
        issues = []
        evidence = []
        
        # Extract entity names from context
        entity_names = {e.get("name", "") for e in entities if e.get("name")}
        
        # Check for entity references in output
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', output)
        
        for word in set(words):
            if word in entity_names:
                evidence.append(ValidationEvidence(
                    source="graphiti",
                    content=f"Valid entity reference: {word}",
                    confidence=0.9
                ))
            elif len(word) > 3 and word not in ["True", "False", "None"]:
                # Potential unrecognized entity
                if any(word.lower() in e.get("name", "").lower() for e in entities):
                    evidence.append(ValidationEvidence(
                        source="graphiti",
                        content=f"Possible entity match: {word}",
                        confidence=0.6
                    ))
        
        return issues, evidence
    
    async def _check_documentation(
        self,
        output: str,
        docs: str
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check output against documentation (REF integration)"""
        
        issues = []
        evidence = []
        
        # Check for technical terms that should match documentation
        technical_terms = self._extract_technical_terms(output)
        doc_terms = self._extract_technical_terms(docs)
        
        for term in technical_terms:
            if term in doc_terms:
                evidence.append(ValidationEvidence(
                    source="ref-docs",
                    content=f"Term verified in documentation: {term}",
                    confidence=0.95
                ))
            else:
                # Check for similar terms (typos, variations)
                similar = self._find_similar_terms(term, doc_terms)
                if similar:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="terminology",
                        message=f"Term '{term}' not exact match. Did you mean '{similar}'?",
                        suggested_fix=f"Use documented term: {similar}"
                    ))
        
        return issues, evidence
    
    async def _llm_validation(
        self,
        output: str,
        context: Optional[Dict[str, Any]],
        prompt: Optional[str]
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Perform LLM-based validation"""
        
        issues = []
        evidence = []
        
        try:
            # Build validation prompt
            validation_content = output
            if prompt:
                validation_content = f"PROMPT: {prompt}\n\nOUTPUT: {output}"
            
            # Call LLM for validation
            result = await self.llm_client.validate_with_llm(
                validation_content,
                context,
                temperature=self.config.llm_config.temperature
            )
            
            # Parse LLM response
            if not result.get("valid", True):
                for issue in result.get("issues", []):
                    issues.append(ValidationIssue(
                        severity=self._map_severity(issue.get("severity", "warning")),
                        category=issue.get("type", "llm-check"),
                        message=issue.get("description", "LLM validation issue"),
                        evidence=issue.get("evidence"),
                        suggested_fix=issue.get("suggestion")
                    ))
            
            # Add verified claims as evidence
            for claim in result.get("verified_claims", []):
                evidence.append(ValidationEvidence(
                    source="llm-verification",
                    content=claim,
                    confidence=result.get("confidence", 0.7)
                ))
            
            # Add unverified claims as issues
            for claim in result.get("unverified_claims", []):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="unverified",
                    message=f"Unverified claim: {claim}",
                    suggested_fix="Provide evidence or context for this claim"
                ))
                
        except Exception as e:
            logger.error(f"LLM validation error: {e}", exc_info=True)
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="system",
                message="LLM validation unavailable",
                suggested_fix="Check LLM configuration"
            ))
        
        return issues, evidence
    
    def _apply_confidence_filter(
        self,
        issues: List[ValidationIssue],
        evidence: List[ValidationEvidence]
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Apply confidence filtering (DeepConf)"""
        
        threshold = self.config.validation_config.confidence_threshold
        
        # Filter evidence by confidence
        filtered_evidence = [
            e for e in evidence
            if e.confidence >= threshold
        ]
        
        # Adjust issue severity based on evidence confidence
        high_confidence_evidence = [e for e in filtered_evidence if e.confidence >= 0.9]
        
        if len(high_confidence_evidence) > len(issues) * 2:
            # Lots of high-confidence evidence, reduce issue severity
            for issue in issues:
                if issue.severity == ValidationSeverity.WARNING:
                    issue.severity = ValidationSeverity.INFO
        
        return issues, filtered_evidence
    
    def _extract_requirements(self, prp_context: str) -> List[str]:
        """Extract requirements from PRP context"""
        
        requirements = []
        
        # Look for requirement patterns
        patterns = [
            r"- \[ \] (.+)",  # Checkbox items
            r"\d+\. (.+)",     # Numbered lists
            r"MUST (.+)",      # MUST requirements
            r"SHALL (.+)",     # SHALL requirements
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, prp_context)
            requirements.extend(matches)
        
        return requirements
    
    def _requirement_addressed(self, output: str, requirement: str) -> bool:
        """Check if a requirement is addressed in output"""
        
        # Simple keyword matching
        req_keywords = set(re.findall(r'\b\w+\b', requirement.lower()))
        output_keywords = set(re.findall(r'\b\w+\b', output.lower()))
        
        # Check for keyword overlap
        overlap = len(req_keywords & output_keywords)
        coverage = overlap / max(len(req_keywords), 1)
        
        return coverage > 0.3
    
    def _extract_technical_terms(self, text: str) -> set:
        """Extract technical terms from text"""
        
        # Look for CamelCase, snake_case, and technical keywords
        terms = set()
        
        # CamelCase
        terms.update(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text))
        
        # snake_case
        terms.update(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text))
        
        # Technical keywords
        keywords = ["API", "HTTP", "JSON", "REST", "GraphQL", "OAuth", "JWT"]
        for keyword in keywords:
            if keyword in text:
                terms.add(keyword)
        
        return terms
    
    def _find_similar_terms(self, term: str, term_set: set) -> Optional[str]:
        """Find similar terms (simple string similarity)"""
        
        term_lower = term.lower()
        for candidate in term_set:
            if term_lower in candidate.lower() or candidate.lower() in term_lower:
                return candidate
        
        return None
    
    def _map_severity(self, severity_str: str) -> ValidationSeverity:
        """Map string severity to enum"""
        
        mapping = {
            "critical": ValidationSeverity.CRITICAL,
            "error": ValidationSeverity.ERROR,
            "warning": ValidationSeverity.WARNING,
            "info": ValidationSeverity.INFO
        }
        
        return mapping.get(severity_str.lower(), ValidationSeverity.WARNING)