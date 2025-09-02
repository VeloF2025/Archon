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
        
        # Check for performance issues in code
        perf_issues, perf_evidence = self._check_performance_issues(output)
        issues.extend(perf_issues)
        evidence.extend(perf_evidence)
        
        # Perform LLM-based validation if configured
        if self.llm_client.client:
            llm_issues, llm_evidence = await self._llm_validation(output, context, prompt)
            issues.extend(llm_issues)
            evidence.extend(llm_evidence)
        
        # Apply confidence filtering (DeepConf)
        filtered_issues, filtered_evidence = self._apply_confidence_filter(issues, evidence)
        
        return filtered_issues, filtered_evidence
    
    def _check_performance_issues(
        self,
        output: str
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check for performance issues in code"""
        
        issues = []
        evidence = []
        
        # Check for unbounded recursion (like naive fibonacci)
        if "def fibonacci" in output or "def fib" in output:
            if "return fibonacci(n-1) + fibonacci(n-2)" in output or "return fib(n-1) + fib(n-2)" in output:
                # This is the classic inefficient recursive fibonacci
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="performance",
                    message="Unbounded recursive fibonacci has O(2^n) complexity - will fail for n>30",
                    suggested_fix="Use iterative approach or memoization for better performance"
                ))
        
        # Check for eval() usage - security issue
        if "eval(" in output:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="security",
                message="eval() usage detected - critical security vulnerability allowing code injection",
                suggested_fix="Never use eval() with user input. Use ast.literal_eval() for safe evaluation of literals"
            ))
        
        # Check for other recursive patterns without memoization
        if "def " in output and "return " in output:
            # Simple pattern matching for recursive calls without memoization
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if 'def ' in line:
                    func_name = line.split('def ')[1].split('(')[0]
                    # Check if function calls itself
                    for j in range(i+1, min(i+20, len(lines))):
                        if f'{func_name}(' in lines[j] and 'return' in lines[j]:
                            # Check if there's memoization
                            has_cache = any('@cache' in lines[k] or '@lru_cache' in lines[k] or 'memo' in lines[k].lower() 
                                          for k in range(max(0, i-3), min(i+20, len(lines))))
                            if not has_cache and func_name in ['factorial', 'fibonacci', 'fib']:
                                issues.append(ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    category="performance",
                                    message=f"Recursive {func_name} without memoization detected",
                                    suggested_fix="Add @lru_cache or implement iterative version"
                                ))
                                break
        
        # If code looks secure and performant, add positive evidence
        if not issues:
            if "hashlib" in output and "sha256" in output:
                evidence.append(ValidationEvidence(
                    source="security-check",
                    content="Using secure SHA-256 hashing",
                    confidence=1.0
                ))
            if "isinstance" in output and "TypeError" in output:
                evidence.append(ValidationEvidence(
                    source="validation-check", 
                    content="Proper input validation detected",
                    confidence=0.9
                ))
        
        return issues, evidence
    
    async def _check_against_prp(
        self,
        output: str,
        prp_context: str | List[str]
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check output against PRP requirements"""
        
        issues = []
        evidence = []
        
        # Handle PRP as string or list
        if isinstance(prp_context, list):
            prp_text = " ".join(prp_context)
        else:
            prp_text = prp_context
        
        # Extract requirements from PRP
        requirements = self._extract_requirements(prp_text)
        
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
        
        # Improved regex to detect actual file paths, not Python attributes
        # Match paths with extensions but exclude Python object attributes (self.x, this.x)
        # Also exclude common Python patterns like module.function
        file_path_pattern = r'(?<![a-zA-Z_\.])["\']?(?:[./\\][\w/\\]+|[\w/\\]+/[\w/\\]+)\.(?:py|js|ts|tsx|jsx|json|yaml|yml|txt|md|csv|html|css|xml|sql|sh|bash|env|conf|cfg|ini|log)["\']?'
        file_paths_in_output = re.findall(file_path_pattern, output)
        
        # Also check for import statements
        import_pattern = r'(?:from|import)\s+([\w.]+)'
        imports = re.findall(import_pattern, output)
        
        # Clean up extracted paths
        cleaned_paths = []
        for path in file_paths_in_output:
            # Remove quotes if present
            path = path.strip('"\'')
            cleaned_paths.append(path)
        
        # Check actual file paths only
        for path in cleaned_paths:
            if path not in file_context:
                # Only report as issue if it looks like a real file path
                # Skip if it's clearly a Python attribute or method
                if not path.startswith('self.') and not path.startswith('this.') and '/' in path or '\\' in path:
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
        
        # Check imports separately - these are module references, not file paths
        for imp in imports:
            # Don't flag standard library or common modules as missing files
            standard_modules = {'os', 'sys', 'time', 'json', 'math', 'random', 'datetime', 
                              'typing', 'pathlib', 'asyncio', 'logging', 're', 'hashlib',
                              'sqlite3', 'secrets', 'collections', 'functools', 'itertools'}
            
            if imp.split('.')[0] not in standard_modules:
                # This is a custom module - could be worth checking
                evidence.append(ValidationEvidence(
                    source="import-check",
                    content=f"Import statement found: {imp}",
                    confidence=0.7
                ))
        
        return issues, evidence
    
    async def _check_entities(
        self,
        output: str,
        entities: List[str] | List[Dict[str, Any]]
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check entities mentioned in output (Graphiti integration)"""
        
        issues = []
        evidence = []
        
        # Extract entity names from context (handle both string list and dict list)
        if not entities:
            entity_names = set()
        elif isinstance(entities[0], str):
            # Simple list of entity names
            entity_names = set(entities)
        else:
            # List of entity dictionaries
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
            elif len(word) > 3 and word not in ["True", "False", "None"] and entities:
                # Potential unrecognized entity
                if isinstance(entities[0], str):
                    # Check against string list
                    if any(word.lower() in e.lower() for e in entities):
                        evidence.append(ValidationEvidence(
                            source="graphiti",
                            content=f"Possible entity match: {word}",
                            confidence=0.6
                        ))
                else:
                    # Check against dict list
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
        docs: str | List[str]
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check output against documentation (REF integration)"""
        
        issues = []
        evidence = []
        
        # Handle docs as string or list
        if isinstance(docs, list):
            docs_text = " ".join(docs)
        else:
            docs_text = docs
        
        # Check for technical terms that should match documentation
        technical_terms = self._extract_technical_terms(output)
        doc_terms = self._extract_technical_terms(docs_text)
        
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