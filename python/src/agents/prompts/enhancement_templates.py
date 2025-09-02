"""
Enhancement Templates Engine for Archon Phase 3
Role-specific templates and formatting for prompt enhancement
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class TemplateType(str, Enum):
    """Types of enhancement templates"""
    ROLE_SPECIFIC = "role_specific"
    USER_FRIENDLY = "user_friendly"
    FORMATTING = "formatting"
    CONTEXT_WRAPPER = "context_wrapper"


@dataclass
class TemplateApplication:
    """Result of template application"""
    enhanced_prompt: str
    template_name: str
    variables_used: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemplateEngine:
    """
    Enhanced template engine for prompt enhancement.
    
    Features:
    - Role-specific prompt templates
    - User-friendly response formatting
    - Context-aware template selection
    - Dynamic variable substitution
    - Template confidence scoring
    """

    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the template engine.
        
        Args:
            templates_path: Path to template files directory
        """
        self.templates_path = Path(templates_path or "python/src/agents/prompts/templates")
        self.role_templates: Dict[str, Dict[str, Any]] = {}
        self.formatting_templates: Dict[str, str] = {}
        self.context_templates: Dict[str, str] = {}
        
        # Load templates
        self._load_builtin_templates()
        self._load_external_templates()
        
        logger.info(f"TemplateEngine initialized with {len(self.role_templates)} role templates")

    def _load_builtin_templates(self):
        """Load built-in templates for common scenarios."""
        
        # Role-specific templates
        self.role_templates = {
            "code_implementer": {
                "template": """You are an expert code implementer focused on zero-error, production-ready solutions.

## Task Context
{task_description}

## Project Information
- **Type**: {project_type}
- **Technologies**: {technologies}
- **Constraints**: {constraints}

## Quality Requirements (MANDATORY)
- Zero TypeScript/ESLint errors
- Zero console.log statements (use proper logging)
- Comprehensive error handling
- 100% type safety (no 'any' types)
- Minimum 90% test coverage
- Performance targets: <200ms API responses

## Implementation Guidelines
1. **Analysis First**: Understand requirements completely
2. **Pattern Recognition**: Use established project patterns
3. **Error Handling**: Try-catch with specific error types
4. **Type Safety**: Full TypeScript typing
5. **Testing**: Write tests alongside implementation
6. **Documentation**: Clear comments for complex logic

## Expected Deliverables
- Complete, working implementation
- Comprehensive unit tests
- Integration tests where applicable
- Clear documentation of design decisions

{additional_context}

Please implement this with zero tolerance for errors and full adherence to quality standards.""",
                "variables": ["task_description", "project_type", "technologies", "constraints", "additional_context"],
                "complexity_adaptations": {
                    "simple": "Focus on straightforward implementation with basic error handling.",
                    "medium": "Include comprehensive error handling and edge case management.",
                    "complex": "Design for scalability, maintainability, and extensive testing."
                }
            },
            
            "system_architect": {
                "template": """You are a senior system architect specializing in scalable, maintainable solutions.

## Architecture Challenge
{task_description}

## System Context
- **Current Architecture**: {current_architecture}
- **Scale Requirements**: {scale_requirements}
- **Technology Stack**: {technologies}
- **Constraints**: {constraints}

## Architectural Principles
1. **Scalability**: Design for growth and load
2. **Maintainability**: Clear separation of concerns
3. **Reliability**: Fault tolerance and graceful degradation
4. **Security**: Security-first design principles
5. **Performance**: Optimize for speed and efficiency
6. **Testability**: Design for comprehensive testing

## Analysis Framework
1. **Requirements Analysis**: Functional and non-functional requirements
2. **Architecture Patterns**: Choose appropriate patterns (microservices, event-driven, etc.)
3. **Data Architecture**: Database design and data flow
4. **API Design**: RESTful/GraphQL API specifications
5. **Security Architecture**: Authentication, authorization, data protection
6. **Deployment Strategy**: CI/CD, containerization, cloud services

## Expected Output
- Detailed architectural design
- Component diagrams and relationships
- API specifications
- Database schema design
- Security considerations
- Performance optimization strategies
- Migration and deployment plan

{additional_context}

Provide a comprehensive architectural solution with clear rationale for all design decisions.""",
                "variables": ["task_description", "current_architecture", "scale_requirements", "technologies", "constraints", "additional_context"],
                "complexity_adaptations": {
                    "simple": "Focus on basic architectural patterns and straightforward design.",
                    "medium": "Include scalability considerations and moderate complexity patterns.",
                    "complex": "Design enterprise-grade architecture with full scalability and reliability."
                }
            },

            "test_coverage_validator": {
                "template": """You are a testing specialist focused on comprehensive test coverage and quality validation.

## Testing Objective
{task_description}

## Code Context
- **Modules to Test**: {modules}
- **Test Framework**: {test_framework}
- **Coverage Target**: {coverage_target}% minimum
- **Testing Types**: {testing_types}

## Testing Standards (MANDATORY)
- Minimum 3 test cases per function (expected, edge, failure)
- Unit tests for all business logic
- Integration tests for API endpoints
- Mock external dependencies appropriately
- Test both success and failure paths
- Performance tests for critical operations

## Test Categories Required
1. **Unit Tests**: Individual function/method testing
2. **Integration Tests**: Component interaction testing
3. **Edge Case Tests**: Boundary conditions and edge cases
4. **Error Tests**: Exception handling and error scenarios
5. **Performance Tests**: Load and stress testing
6. **Security Tests**: Input validation and security vulnerabilities

## Coverage Analysis
- **Statement Coverage**: Every line executed
- **Branch Coverage**: All conditional branches tested
- **Function Coverage**: All functions called
- **Critical Path Coverage**: 100% for core business logic

## Test Structure Template
```typescript
describe('ComponentName', () => {{
  // Setup and teardown
  beforeEach(() => {{ /* setup */ }});
  afterEach(() => {{ /* cleanup */ }});
  
  describe('methodName', () => {{
    it('should handle expected behavior', () => {{ /* expected case */ }});
    it('should handle edge cases', () => {{ /* edge case */ }});
    it('should handle errors gracefully', () => {{ /* error case */ }});
  }});
}});
```

{additional_context}

Create comprehensive tests that ensure robust, reliable code with proper error handling and edge case coverage.""",
                "variables": ["task_description", "modules", "test_framework", "coverage_target", "testing_types", "additional_context"],
                "complexity_adaptations": {
                    "simple": "Focus on basic unit tests and straightforward test cases.",
                    "medium": "Include integration tests and moderate edge case coverage.",
                    "complex": "Comprehensive test suite with performance, security, and stress testing."
                }
            },

            "security_auditor": {
                "template": """You are a security expert conducting comprehensive security audits and vulnerability assessments.

## Security Audit Scope
{task_description}

## Application Context
- **Application Type**: {application_type}
- **Technology Stack**: {technologies}
- **Data Sensitivity**: {data_sensitivity}
- **Compliance Requirements**: {compliance}

## Security Assessment Framework
1. **Authentication & Authorization**
   - Identity verification mechanisms
   - Access control implementation
   - Session management
   - Multi-factor authentication

2. **Data Protection**
   - Encryption at rest and in transit
   - Data sanitization and validation
   - PII handling and privacy
   - Backup security

3. **Input Validation & Output Encoding**
   - SQL injection prevention
   - XSS protection
   - CSRF protection
   - Command injection prevention

4. **API Security**
   - Rate limiting and throttling
   - API authentication
   - Input validation
   - Response data filtering

5. **Infrastructure Security**
   - Network security configuration
   - Container security
   - Cloud security best practices
   - Monitoring and logging

## Vulnerability Categories
- **OWASP Top 10** compliance
- **CWE Common Weaknesses**
- **Application-specific vulnerabilities**
- **Infrastructure vulnerabilities**

## Security Checklist
- [ ] Input validation on all endpoints
- [ ] Parameterized queries (no SQL injection)
- [ ] Proper authentication checks
- [ ] Authorization for each operation
- [ ] Encrypted sensitive data storage
- [ ] Secure communication (HTTPS/TLS)
- [ ] Error handling without information leakage
- [ ] Logging security events
- [ ] Rate limiting implementation
- [ ] Security headers configuration

{additional_context}

Provide a comprehensive security assessment with specific remediation steps and implementation guidance.""",
                "variables": ["task_description", "application_type", "technologies", "data_sensitivity", "compliance", "additional_context"],
                "complexity_adaptations": {
                    "simple": "Focus on basic security practices and common vulnerabilities.",
                    "medium": "Include comprehensive OWASP compliance and security best practices.",
                    "complex": "Enterprise-grade security audit with advanced threat modeling."
                }
            },

            "rag_agent": {
                "template": """You are a RAG (Retrieval-Augmented Generation) specialist focused on knowledge retrieval and contextual responses.

## Query Context
{task_description}

## Knowledge Base Information
- **Sources**: {knowledge_sources}
- **Domain**: {domain}
- **Search Context**: {search_context}
- **Retrieved Documents**: {retrieved_docs}

## RAG Processing Framework
1. **Query Understanding**
   - Intent classification
   - Entity extraction
   - Context identification
   - Scope determination

2. **Knowledge Retrieval**
   - Semantic search execution
   - Document ranking
   - Context filtering
   - Relevance scoring

3. **Response Generation**
   - Context synthesis
   - Factual accuracy verification
   - Source attribution
   - Coherent response construction

## Quality Standards
- **Accuracy**: Responses must be factually correct based on sources
- **Relevance**: All information must be relevant to the query
- **Completeness**: Address all aspects of the query
- **Attribution**: Cite sources for claims and information
- **Clarity**: Present information in clear, understandable format

## Response Structure
1. **Direct Answer**: Immediate response to the query
2. **Supporting Details**: Additional relevant information
3. **Source References**: Clear citations and references
4. **Related Information**: Contextually relevant additional insights
5. **Confidence Indicators**: Indicate certainty level of responses

## Source Handling
- Prioritize authoritative and recent sources
- Cross-reference multiple sources when possible
- Flag conflicting information from sources
- Indicate when information is incomplete or uncertain

{additional_context}

Provide accurate, well-sourced responses based on the retrieved knowledge while maintaining transparency about sources and confidence levels.""",
                "variables": ["task_description", "knowledge_sources", "domain", "search_context", "retrieved_docs", "additional_context"],
                "complexity_adaptations": {
                    "simple": "Focus on straightforward information retrieval and basic responses.",
                    "medium": "Include source analysis and comprehensive contextual responses.",
                    "complex": "Advanced knowledge synthesis with multi-source analysis and expert-level responses."
                }
            }
        }
        
        # User-friendly formatting templates
        self.formatting_templates = {
            "structured_response": """## Summary
{summary}

## Details
{details}

## Key Points
{key_points}

## Next Steps
{next_steps}""",
            
            "code_explanation": """## Solution Overview
{overview}

## Implementation
{code_block}

## Explanation
{explanation}

## Usage Example
{usage}

## Important Notes
{notes}""",
            
            "error_solution": """## Problem Identified
{problem}

## Root Cause
{cause}

## Solution
{solution}

## Prevention
{prevention}""",
            
            "step_by_step": """## Objective
{objective}

## Prerequisites
{prerequisites}

## Steps
{steps}

## Verification
{verification}

## Troubleshooting
{troubleshooting}"""
        }
        
        # Context wrapper templates
        self.context_templates = {
            "project_context": """
## Project Context
- **Name**: {project_name}
- **Type**: {project_type}
- **Technologies**: {technologies}
- **Current Phase**: {phase}

{content}
""",
            
            "technical_context": """
## Technical Context
- **Framework**: {framework}
- **Language**: {language}
- **Database**: {database}
- **Architecture**: {architecture}

{content}
""",
            
            "user_context": """
## Background
{background}

{content}

## Additional Resources
{resources}
"""
        }

    def _load_external_templates(self):
        """Load templates from external files if available."""
        if not self.templates_path.exists():
            logger.info(f"Templates directory not found: {self.templates_path}")
            return
        
        try:
            # Load role-specific templates
            role_templates_file = self.templates_path / "role_templates.json"
            if role_templates_file.exists():
                with open(role_templates_file, 'r') as f:
                    external_roles = json.load(f)
                    self.role_templates.update(external_roles)
            
            # Load formatting templates
            formatting_file = self.templates_path / "formatting_templates.json"
            if formatting_file.exists():
                with open(formatting_file, 'r') as f:
                    external_formatting = json.load(f)
                    self.formatting_templates.update(external_formatting)
                    
            logger.info("External templates loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load external templates: {e}")

    async def apply_role_template(self, 
                                 role: str,
                                 prompt: str,
                                 context: Dict[str, Any],
                                 complexity: Optional[Any] = None) -> Optional[TemplateApplication]:
        """
        Apply role-specific template to enhance prompt.
        
        Args:
            role: Agent role (must match template name)
            prompt: Original prompt to enhance
            context: Context variables for template
            complexity: Task complexity level for adaptation
            
        Returns:
            Template application result or None if no template
        """
        if role not in self.role_templates:
            logger.warning(f"No template found for role: {role}")
            return None
        
        try:
            template_config = self.role_templates[role]
            template_content = template_config["template"]
            
            # Prepare variables for substitution
            variables = self._prepare_template_variables(
                template_config.get("variables", []),
                context,
                prompt
            )
            
            # Apply complexity adaptation if available
            if complexity and "complexity_adaptations" in template_config:
                complexity_key = complexity.value if hasattr(complexity, 'value') else str(complexity)
                if complexity_key in template_config["complexity_adaptations"]:
                    adaptation = template_config["complexity_adaptations"][complexity_key]
                    variables["additional_context"] = f"{variables.get('additional_context', '')}\n\n**Complexity Note**: {adaptation}"
            
            # Substitute variables in template
            enhanced_prompt = self._substitute_variables(template_content, variables)
            
            # Calculate confidence based on variable coverage
            confidence = self._calculate_template_confidence(
                template_config.get("variables", []),
                variables
            )
            
            return TemplateApplication(
                enhanced_prompt=enhanced_prompt,
                template_name=f"{role}_template",
                variables_used=list(variables.keys()),
                confidence=confidence,
                metadata={
                    "role": role,
                    "complexity": str(complexity) if complexity else None,
                    "variable_coverage": len(variables) / len(template_config.get("variables", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to apply role template for {role}: {e}")
            return None

    async def apply_user_friendly_formatting(self,
                                           response: str,
                                           analysis: Dict[str, Any],
                                           context: Dict[str, Any]) -> str:
        """
        Apply user-friendly formatting to agent responses.
        
        Args:
            response: Original agent response
            analysis: Response analysis results
            context: Additional context for formatting
            
        Returns:
            Formatted response
        """
        try:
            response_type = analysis.get("type", "general")
            
            # Select appropriate formatting template
            if response_type == "code_solution" and analysis.get("has_code"):
                return self._apply_code_explanation_format(response, context)
            elif response_type == "error_response":
                return self._apply_error_solution_format(response, context)
            elif analysis.get("has_lists") or analysis.get("has_numbers"):
                return self._apply_structured_format(response, context)
            else:
                return self._apply_general_format(response, context)
                
        except Exception as e:
            logger.error(f"Failed to apply user-friendly formatting: {e}")
            return response

    def _prepare_template_variables(self,
                                  required_vars: List[str],
                                  context: Dict[str, Any],
                                  prompt: str) -> Dict[str, str]:
        """Prepare variables for template substitution."""
        variables = {}
        
        # Map context to template variables
        variable_mappings = {
            "task_description": prompt,
            "project_type": context.get("project_type", "Unknown"),
            "technologies": self._format_list(context.get("technologies", [])),
            "constraints": self._format_list(context.get("constraints", [])),
            "additional_context": context.get("additional_context", ""),
            "current_architecture": context.get("current_architecture", "Not specified"),
            "scale_requirements": context.get("scale_requirements", "Standard"),
            "modules": self._format_list(context.get("modules", [])),
            "test_framework": context.get("test_framework", "Jest/Vitest"),
            "coverage_target": str(context.get("coverage_target", 90)),
            "testing_types": self._format_list(context.get("testing_types", ["unit", "integration"])),
            "application_type": context.get("application_type", "Web Application"),
            "data_sensitivity": context.get("data_sensitivity", "Medium"),
            "compliance": context.get("compliance", "Standard"),
            "knowledge_sources": self._format_list(context.get("knowledge_sources", [])),
            "domain": context.get("domain", "General"),
            "search_context": context.get("search_context", ""),
            "retrieved_docs": str(context.get("retrieved_docs", ""))
        }
        
        # Fill in available variables
        for var in required_vars:
            if var in variable_mappings:
                variables[var] = variable_mappings[var]
            elif var in context:
                variables[var] = str(context[var])
            else:
                variables[var] = f"[{var}]"  # Placeholder for missing variables
        
        return variables

    def _substitute_variables(self, template: str, variables: Dict[str, str]) -> str:
        """Safely substitute variables in template."""
        try:
            # Use format() for variable substitution
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            # Fallback: use simple string replacement
            result = template
            for var, value in variables.items():
                result = result.replace(f"{{{var}}}", value)
            return result

    def _calculate_template_confidence(self, 
                                     required_vars: List[str],
                                     provided_vars: Dict[str, str]) -> float:
        """Calculate confidence score for template application."""
        if not required_vars:
            return 1.0
        
        # Count variables with actual values (not placeholders)
        actual_vars = sum(1 for var in required_vars 
                         if var in provided_vars and not provided_vars[var].startswith('['))
        
        return actual_vars / len(required_vars)

    def _format_list(self, items: List[Any]) -> str:
        """Format list items for template display."""
        if not items:
            return "None specified"
        
        if len(items) == 1:
            return str(items[0])
        
        return ", ".join(str(item) for item in items)

    def _apply_code_explanation_format(self, response: str, context: Dict[str, Any]) -> str:
        """Apply code explanation formatting."""
        # Extract code blocks
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', response, re.DOTALL)
        
        # Remove code blocks from response to get explanation
        explanation = re.sub(r'```[\w]*\n.*?\n```', '[CODE_BLOCK]', response, flags=re.DOTALL)
        
        if code_blocks:
            formatted_code = f"```\n{code_blocks[0]}\n```"
            
            return self.formatting_templates["code_explanation"].format(
                overview="Code solution provided below",
                code_block=formatted_code,
                explanation=explanation.replace('[CODE_BLOCK]', '').strip(),
                usage="See code comments for usage instructions",
                notes="Ensure all dependencies are installed before running"
            )
        
        return response

    def _apply_error_solution_format(self, response: str, context: Dict[str, Any]) -> str:
        """Apply error solution formatting."""
        # Try to extract problem and solution sections
        lines = response.split('\n')
        
        problem = "Error identified in the system"
        cause = "Root cause analysis needed"
        solution = response
        prevention = "Follow best practices to prevent similar issues"
        
        # Simple heuristic to identify sections
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ['error', 'problem', 'issue']):
                problem = line.strip()
                break
        
        return self.formatting_templates["error_solution"].format(
            problem=problem,
            cause=cause,
            solution=solution,
            prevention=prevention
        )

    def _apply_structured_format(self, response: str, context: Dict[str, Any]) -> str:
        """Apply structured formatting for list-based responses."""
        # Extract structured elements
        lines = response.split('\n')
        
        summary = "Structured response with multiple components"
        details = response
        key_points = []
        next_steps = []
        
        # Extract bullet points and numbered lists
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                next_steps.append(line)
            elif re.match(r'^[-*+]', line):
                key_points.append(line)
        
        return self.formatting_templates["structured_response"].format(
            summary=summary,
            details=details,
            key_points='\n'.join(key_points) if key_points else "See details above",
            next_steps='\n'.join(next_steps) if next_steps else "Follow the provided guidance"
        )

    def _apply_general_format(self, response: str, context: Dict[str, Any]) -> str:
        """Apply general formatting for standard responses."""
        # For general responses, add helpful structure
        if len(response.split('\n')) > 5:
            # Multi-paragraph response
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                summary = paragraphs[0]
                details = '\n\n'.join(paragraphs[1:])
                
                return f"## Summary\n{summary}\n\n## Details\n{details}"
        
        return response

    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get list of available templates by category."""
        return {
            "role_templates": list(self.role_templates.keys()),
            "formatting_templates": list(self.formatting_templates.keys()),
            "context_templates": list(self.context_templates.keys())
        }

    def validate_template(self, template_name: str, template_type: TemplateType) -> Dict[str, Any]:
        """Validate a template for correctness and completeness."""
        validation_result = {
            "valid": False,
            "issues": [],
            "suggestions": []
        }
        
        template_source = None
        
        if template_type == TemplateType.ROLE_SPECIFIC:
            template_source = self.role_templates.get(template_name)
        elif template_type == TemplateType.FORMATTING:
            template_source = self.formatting_templates.get(template_name)
        elif template_type == TemplateType.CONTEXT_WRAPPER:
            template_source = self.context_templates.get(template_name)
        
        if not template_source:
            validation_result["issues"].append(f"Template {template_name} not found")
            return validation_result
        
        # Validate template structure
        if isinstance(template_source, dict):
            template_content = template_source.get("template", "")
        else:
            template_content = str(template_source)
        
        # Check for variable placeholders
        variables = re.findall(r'\{(\w+)\}', template_content)
        if not variables:
            validation_result["suggestions"].append("Consider adding variable placeholders for dynamic content")
        
        # Check for required sections in role templates
        if template_type == TemplateType.ROLE_SPECIFIC:
            required_sections = ["Task Context", "Quality Requirements", "Expected"]
            missing_sections = [section for section in required_sections 
                              if section not in template_content]
            if missing_sections:
                validation_result["issues"].extend([f"Missing section: {section}" for section in missing_sections])
        
        validation_result["valid"] = len(validation_result["issues"]) == 0
        return validation_result