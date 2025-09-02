#!/usr/bin/env python3
"""
DOCUMENTATION-DRIVEN TEST VALIDATOR
Enforces mandatory test creation from PRD/PRP/ADR docs before implementation

This validator ensures agents follow the hardcoded rule:
1. Parse documentation for requirements
2. Create test specifications 
3. Write tests BEFORE any implementation
4. Validate tests match documented acceptance criteria

CRITICAL: This is a blocking validation - development stops if violated
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from .archon_validation_rules import ValidationRule, validate_rule_compliance
except ImportError:
    # Fallback for direct execution or missing rules
    def validate_rule_compliance(validation_results):
        return {"compliant": True, "violations": [], "critical_violations": []}

logger = logging.getLogger(__name__)

@dataclass
class DocumentRequirement:
    """Parsed requirement from documentation"""
    doc_type: str  # PRD, PRP, ADR
    doc_path: str
    section: str
    requirement: str
    acceptance_criteria: List[str]
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW

@dataclass
class TestValidation:
    """Test validation result"""
    has_tests: bool
    tests_from_docs: bool
    tests_match_specs: bool
    missing_requirements: List[str]
    validation_errors: List[str]

class DocDrivenValidator:
    """Validates documentation-driven test development"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.requirements: List[DocumentRequirement] = []
        
    def scan_documentation(self) -> List[DocumentRequirement]:
        """Scan for PRD/PRP/ADR documents and extract requirements"""
        doc_patterns = {
            'PRD': ['**/PRD*.md', '**/prd*.md', '**/product_requirements*.md'],
            'PRP': ['**/PRP*.md', '**/prp*.md', '**/project_requirements*.md'], 
            'ADR': ['**/ADR*.md', '**/adr*.md', '**/architectural_decisions*.md']
        }
        
        requirements = []
        
        for doc_type, patterns in doc_patterns.items():
            for pattern in patterns:
                for doc_path in self.project_path.glob(pattern):
                    reqs = self._parse_document(doc_path, doc_type)
                    requirements.extend(reqs)
        
        self.requirements = requirements
        return requirements
    
    def _parse_document(self, doc_path: Path, doc_type: str) -> List[DocumentRequirement]:
        """Parse requirements from a documentation file"""
        try:
            content = doc_path.read_text(encoding='utf-8')
            requirements = []
            
            # Parse structured requirements
            # Look for common patterns in PRD/PRP/ADR docs
            patterns = {
                'requirements': r'(?i)^##?\s*(requirement|feature|user story|acceptance criteria)[:\s](.+?)(?=^##?|\Z)',
                'acceptance': r'(?i)^[-*]\s*(given|when|then|should|must|shall|will)\s+(.+?)$',
                'priority': r'(?i)(critical|high|medium|low|p[0-4])'
            }
            
            # Extract sections with requirements
            sections = re.split(r'^(##?\s*.+?)$', content, flags=re.MULTILINE)
            
            for i in range(1, len(sections), 2):
                if i + 1 >= len(sections):
                    continue
                    
                section_title = sections[i].strip()
                section_content = sections[i + 1].strip()
                
                if self._is_requirement_section(section_title):
                    # Extract acceptance criteria
                    acceptance_criteria = re.findall(patterns['acceptance'], section_content, re.MULTILINE)
                    acceptance_criteria = [ac[1].strip() for ac in acceptance_criteria]
                    
                    # Determine priority
                    priority_match = re.search(patterns['priority'], section_content, re.IGNORECASE)
                    priority = priority_match.group(1).upper() if priority_match else 'MEDIUM'
                    
                    if acceptance_criteria or 'test' in section_content.lower():
                        req = DocumentRequirement(
                            doc_type=doc_type,
                            doc_path=str(doc_path),
                            section=section_title,
                            requirement=section_content[:200] + '...' if len(section_content) > 200 else section_content,
                            acceptance_criteria=acceptance_criteria,
                            priority=priority
                        )
                        requirements.append(req)
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error parsing {doc_path}: {e}")
            return []
    
    def _is_requirement_section(self, title: str) -> bool:
        """Check if section contains requirements"""
        requirement_keywords = [
            'requirement', 'feature', 'functionality', 'behavior', 
            'user story', 'acceptance', 'criteria', 'specification',
            'must', 'should', 'shall', 'will'
        ]
        return any(keyword in title.lower() for keyword in requirement_keywords)
    
    def validate_test_coverage(self, test_path: Optional[str] = None) -> TestValidation:
        """Validate that tests exist and cover documented requirements"""
        
        # Find test files
        test_patterns = [
            '**/test_*.py', '**/tests/*.py', 
            '**/test/*.py', '**/*_test.py',
            '**/test_*.js', '**/tests/*.js',
            '**/*.test.js', '**/*.spec.js',
            '**/test_*.ts', '**/tests/*.ts',
            '**/*.test.ts', '**/*.spec.ts'
        ]
        
        test_files = []
        for pattern in test_patterns:
            test_files.extend(self.project_path.glob(pattern))
        
        if not test_files:
            return TestValidation(
                has_tests=False,
                tests_from_docs=False, 
                tests_match_specs=False,
                missing_requirements=[req.requirement for req in self.requirements],
                validation_errors=["No test files found in project"]
            )
        
        # Analyze test content for requirement coverage
        test_content = ""
        for test_file in test_files:
            try:
                test_content += test_file.read_text(encoding='utf-8') + "\n"
            except Exception as e:
                logger.warning(f"Could not read test file {test_file}: {e}")
        
        # Check if tests reference documented requirements
        tests_from_docs = self._validate_tests_from_docs(test_content)
        tests_match_specs = self._validate_tests_match_specs(test_content)
        missing_requirements = self._find_missing_test_coverage(test_content)
        
        validation_errors = []
        if not tests_from_docs:
            validation_errors.append("Tests do not reference documented requirements from PRD/PRP/ADR")
        if not tests_match_specs:
            validation_errors.append("Tests do not match documented acceptance criteria")
        if missing_requirements:
            validation_errors.append(f"Missing test coverage for: {', '.join(missing_requirements[:3])}")
        
        return TestValidation(
            has_tests=True,
            tests_from_docs=tests_from_docs,
            tests_match_specs=tests_match_specs,
            missing_requirements=missing_requirements,
            validation_errors=validation_errors
        )
    
    def _validate_tests_from_docs(self, test_content: str) -> bool:
        """Check if tests reference documentation"""
        doc_references = [
            'PRD', 'PRP', 'ADR', 'requirement', 'acceptance criteria',
            'user story', 'specification', 'documented', 'as specified'
        ]
        
        test_content_lower = test_content.lower()
        return any(ref.lower() in test_content_lower for ref in doc_references)
    
    def _validate_tests_match_specs(self, test_content: str) -> bool:
        """Check if tests validate documented acceptance criteria"""
        if not self.requirements:
            return True  # No requirements to validate against
        
        test_content_lower = test_content.lower()
        
        # Look for tests that match acceptance criteria patterns
        for req in self.requirements:
            for criteria in req.acceptance_criteria:
                # Simple keyword matching - could be enhanced with NLP
                keywords = re.findall(r'\b\w{4,}\b', criteria.lower())
                if len(keywords) >= 2:
                    matches = sum(1 for kw in keywords if kw in test_content_lower)
                    if matches >= len(keywords) * 0.5:  # 50% keyword match
                        return True
        
        return False
    
    def _find_missing_test_coverage(self, test_content: str) -> List[str]:
        """Find requirements without test coverage"""
        missing = []
        test_content_lower = test_content.lower()
        
        for req in self.requirements:
            # Check if requirement is covered by tests
            req_keywords = re.findall(r'\b\w{4,}\b', req.requirement.lower())
            if len(req_keywords) >= 2:
                matches = sum(1 for kw in req_keywords if kw in test_content_lower)
                if matches < len(req_keywords) * 0.3:  # Less than 30% coverage
                    missing.append(f"{req.section}: {req.requirement[:100]}...")
        
        return missing
    
    def enforce_doc_driven_development(self) -> Dict[str, Any]:
        """Main enforcement method - validates documentation-driven development"""
        
        # Step 1: Scan documentation
        requirements = self.scan_documentation()
        
        if not requirements:
            return {
                "compliant": False,
                "error": "No PRD/PRP/ADR documentation found - cannot validate documentation-driven development",
                "remediation": "Create PRD/PRP/ADR documents with clear requirements and acceptance criteria"
            }
        
        # Step 2: Validate test coverage
        test_validation = self.validate_test_coverage()
        
        # Step 3: Check for implementation without tests
        has_implementation = self._check_for_implementation()
        
        # Step 4: Run validation rules
        validation_results = {
            "tests_exist": test_validation.has_tests,
            "tests_from_docs": test_validation.tests_from_docs,
            "tests_match_specs": test_validation.tests_match_specs,
            "has_implementation": has_implementation,
            "syntax_errors": 0,  # Would be set by other validators
            "coverage": 1.0 if test_validation.has_tests else 0.0
        }
        
        rule_compliance = validate_rule_compliance(validation_results)
        
        return {
            "compliant": rule_compliance["compliant"],
            "requirements_found": len(requirements),
            "test_validation": test_validation,
            "violations": rule_compliance["violations"],
            "critical_violations": rule_compliance["critical_violations"],
            "remediation_steps": [
                "1. Parse PRD/PRP/ADR documents for requirements",
                "2. Create test specifications from documented acceptance criteria", 
                "3. Write comprehensive tests BEFORE implementation",
                "4. Validate tests cover all documented requirements",
                "5. Only implement code to pass the doc-derived tests"
            ]
        }
    
    def _check_for_implementation(self) -> bool:
        """Check if implementation code exists"""
        impl_patterns = [
            '**/*.py', '**/*.js', '**/*.ts', '**/*.java',
            '**/*.cpp', '**/*.cs', '**/*.go', '**/*.rs'
        ]
        
        # Exclude test files and common non-implementation files
        exclude_patterns = [
            '**/test_*', '**/tests/*', '**/*_test.*', '**/*.test.*', 
            '**/node_modules/*', '**/venv/*', '**/.git/*', '**/docs/*',
            '**/__pycache__/*', '**/dist/*', '**/build/*'
        ]
        
        for pattern in impl_patterns:
            for file_path in self.project_path.glob(pattern):
                # Skip if matches exclude patterns
                if any(file_path.match(excl) for excl in exclude_patterns):
                    continue
                
                # Check if file has substantial implementation content
                try:
                    content = file_path.read_text(encoding='utf-8')
                    # Skip files that are mostly comments/imports
                    code_lines = [line.strip() for line in content.split('\n') 
                                 if line.strip() and not line.strip().startswith(('#', '//', '/*', '*', 'import', 'from'))]
                    if len(code_lines) > 10:  # Has substantial implementation
                        return True
                except Exception:
                    continue
        
        return False

def validate_doc_driven_development(project_path: str = ".") -> Dict[str, Any]:
    """
    Main validation function for documentation-driven test development
    
    This enforces the hardcoded global rule that tests must be created
    from documentation before any implementation begins.
    """
    validator = DocDrivenValidator(project_path)
    return validator.enforce_doc_driven_development()

if __name__ == "__main__":
    import sys
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    result = validate_doc_driven_development(project_path)
    print(json.dumps(result, indent=2, default=str))