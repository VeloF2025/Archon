#!/usr/bin/env python3
"""
ENHANCED SPECIFICATION PARSER
Integrates Spec Kit's structured approach with Archon's validation systems

This parser enhances Archon's documentation-driven development by:
1. Parsing Spec Kit's structured specification format
2. Converting specifications to Archon's DocumentRequirement format
3. Validating against both Spec Kit and Archon quality standards
4. Generating test specifications from structured requirements
5. Supporting multi-AI agent task generation
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .doc_driven_validator import DocumentRequirement, TestValidation
from .archon_validation_rules import ValidationRule, validate_rule_compliance

logger = logging.getLogger(__name__)

class SpecSection(Enum):
    """Standard sections in enhanced specifications"""
    USER_SCENARIOS = "User Scenarios & Testing"
    REQUIREMENTS = "Requirements"
    TECHNICAL_CONTEXT = "Technical Context"
    CONSTITUTION_CHECK = "Constitution Check"
    REVIEW_CHECKLIST = "Review & Acceptance Checklist"

class RequirementType(Enum):
    """Types of requirements in enhanced specifications"""
    FUNCTIONAL = "Functional Requirements"
    ARCHON_INTEGRATION = "Archon Integration Requirements"
    QUALITY_GATE = "Quality Gates"
    MULTI_AGENT = "Multi-Agent Support"

@dataclass
class TestSpecification:
    """Enhanced test specification from structured requirements"""
    requirement_id: str
    description: str
    test_category: str  # Happy Path, Edge Cases, Error Handling, etc.
    acceptance_criteria: List[str]
    dgts_compliance: bool
    automation_level: str  # Full, Partial, Manual
    priority: str
    estimated_complexity: int  # 1-10

@dataclass
class AgentTask:
    """Task for specialized Archon agents"""
    task_id: str
    agent_type: str
    description: str
    requirements: List[str]
    dependencies: List[str]
    estimated_duration: int  # minutes
    parallelizable: bool
    quality_gates: List[str]

@dataclass
class EnhancedSpec:
    """Enhanced specification combining Spec Kit and Archon approaches"""
    metadata: Dict[str, Any]
    user_scenarios: Dict[str, Any]
    requirements: List[DocumentRequirement]
    test_specifications: List[TestSpecification]
    technical_context: Optional[Dict[str, Any]]
    constitution_check: Dict[str, Any]
    agent_tasks: List[AgentTask]
    validation_errors: List[str] = field(default_factory=list)
    complexity_score: int = 0

class EnhancedSpecParser:
    """Parses enhanced specifications and converts to Archon format"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.templates_path = Path(__file__).parent.parent.parent / "spec-kit" / "templates"

    def parse_specification(self, spec_path: Union[str, Path]) -> EnhancedSpec:
        """Parse enhanced specification file"""
        spec_path = Path(spec_path)

        try:
            content = spec_path.read_text(encoding='utf-8')

            # Parse metadata
            metadata = self._parse_metadata(content)

            # Parse user scenarios
            user_scenarios = self._parse_user_scenarios(content)

            # Parse requirements
            requirements = self._parse_requirements(content)

            # Generate test specifications
            test_specifications = self._generate_test_specifications(requirements)

            # Parse technical context (if present)
            technical_context = self._parse_technical_context(content)

            # Parse constitution check (if present)
            constitution_check = self._parse_constitution_check(content)

            # Generate agent tasks
            agent_tasks = self._generate_agent_tasks(requirements, test_specifications)

            # Calculate complexity score
            complexity_score = self._calculate_complexity(requirements, test_specifications)

            # Validate specification
            validation_errors = self._validate_specification(content)

            return EnhancedSpec(
                metadata=metadata,
                user_scenarios=user_scenarios,
                requirements=requirements,
                test_specifications=test_specifications,
                technical_context=technical_context,
                constitution_check=constitution_check,
                agent_tasks=agent_tasks,
                validation_errors=validation_errors,
                complexity_score=complexity_score
            )

        except Exception as e:
            logger.error(f"Error parsing specification {spec_path}: {e}")
            raise

    def _parse_metadata(self, content: str) -> Dict[str, Any]:
        """Parse specification metadata"""
        metadata = {}

        # Extract basic metadata
        patterns = {
            'feature_name': r'^#\s*Enhanced Feature Specification:\s*(.+)$',
            'branch': r'\*\*Feature Branch\*\*:\s*`(.+?)`',
            'created': r'\*\*Created\*\*:\s*(.+)',
            'status': r'\*\*Status\*\*:\s*(.+)',
            'input': r'\*\*Input\*\*:\s*(.+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

        return metadata

    def _parse_user_scenarios(self, content: str) -> Dict[str, Any]:
        """Parse user scenarios section"""
        scenarios = {}

        # Find user scenarios section
        section_match = re.search(
            r'^## User Scenarios & Testing.*?$(.*?)(?=^##|\Z)',
            content,
            re.MULTILINE | re.DOTALL
        )

        if not section_match:
            return scenarios

        section_content = section_match.group(1)

        # Extract primary user story
        story_match = re.search(
            r'^### Primary User Story\s*\n+(.+?)$',
            section_content,
            re.MULTILINE
        )
        if story_match:
            scenarios['primary_story'] = story_match.group(1).strip()

        # Extract acceptance scenarios
        acceptance_matches = re.findall(
            r'^\d+\.\s*\*\*Given\*\*(.+?)\s*\*\*When\*\*(.+?)\s*\*\*Then\*\*(.+?)$',
            section_content,
            re.MULTILINE
        )
        scenarios['acceptance_scenarios'] = [
            {
                'given': match[0].strip(),
                'when': match[1].strip(),
                'then': match[2].strip()
            }
            for match in acceptance_matches
        ]

        # Extract edge cases
        edge_cases = re.findall(
            r'^-\s*(.+?)$',
            section_content,
            re.MULTILINE
        )
        scenarios['edge_cases'] = [case.strip() for case in edge_cases if case.strip()]

        return scenarios

    def _parse_requirements(self, content: str) -> List[DocumentRequirement]:
        """Parse requirements and convert to DocumentRequirement format"""
        requirements = []

        # Find requirements section
        section_match = re.search(
            r'^## Requirements.*?$(.*?)(?=^##|\Z)',
            content,
            re.MULTILINE | re.DOTALL
        )

        if not section_match:
            return requirements

        section_content = section_match.group(1)

        # Parse different requirement types
        requirement_types = [
            RequirementType.FUNCTIONAL.value,
            RequirementType.ARCHON_INTEGRATION.value,
            RequirementType.QUALITY_GATE.value,
            RequirementType.MULTI_AGENT.value
        ]

        for req_type in requirement_types:
            # Find subsection for this requirement type
            subsection_match = re.search(
                f'^### {re.escape(req_type)}.*?$(.*?)(?=^###|\\Z)',
                section_content,
                re.MULTILINE | re.DOTALL
            )

            if not subsection_match:
                continue

            subsection_content = subsection_match.group(1)

            # Parse individual requirements
            req_matches = re.findall(
                r'^-\s*\*\*([A-Z]+-\d+)\*\*:\s*(.+?)$',
                subsection_content,
                re.MULTILINE
            )

            for req_id, req_description in req_matches:
                # Extract acceptance criteria from requirement description
                acceptance_criteria = self._extract_acceptance_criteria(req_description)

                # Determine priority
                priority = self._determine_requirement_priority(req_description, req_type)

                requirement = DocumentRequirement(
                    doc_type="SPEC",  # Enhanced specification
                    doc_path="",  # Will be set by caller
                    section=req_type,
                    requirement=req_description,
                    acceptance_criteria=acceptance_criteria,
                    priority=priority
                )
                requirements.append(requirement)

        return requirements

    def _extract_acceptance_criteria(self, description: str) -> List[str]:
        """Extract acceptance criteria from requirement description"""
        criteria = []

        # Look for criteria patterns
        patterns = [
            r'MUST\s+(.+?)(?:\.|$)',
            r'SHOULD\s+(.+?)(?:\.|$)',
            r'REQUIRED\s+TO\s+(.+?)(?:\.|$)',
            r'ABLE\s+TO\s+(.+?)(?:\.|$)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            criteria.extend(matches)

        return list(set(criteria))  # Remove duplicates

    def _determine_requirement_priority(self, description: str, req_type: str) -> str:
        """Determine requirement priority based on content and type"""
        priority_keywords = {
            'CRITICAL': ['critical', 'must', 'required', 'mandatory', 'security'],
            'HIGH': ['high', 'important', 'essential', 'primary'],
            'MEDIUM': ['medium', 'should', 'recommended', 'good'],
            'LOW': ['low', 'optional', 'nice-to-have', 'enhancement']
        }

        description_lower = description.lower()

        # Check type-based priority
        if req_type in [RequirementType.ARCHON_INTEGRATION.value, RequirementType.QUALITY_GATE.value]:
            return 'HIGH'

        # Check keyword-based priority
        for priority, keywords in priority_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return priority

        return 'MEDIUM'

    def _generate_test_specifications(self, requirements: List[DocumentRequirement]) -> List[TestSpecification]:
        """Generate test specifications from requirements"""
        test_specs = []

        for req in requirements:
            # Generate test specification for each requirement
            test_spec = TestSpecification(
                requirement_id=req.requirement.split(':')[0] if ':' in req.requirement else f"REQ-{len(test_specs)+1}",
                description=req.requirement,
                test_category=self._categorize_requirement(req.requirement),
                acceptance_criteria=req.acceptance_criteria,
                dgts_compliance=self._check_dgts_compliance(req.requirement),
                automation_level="Full",  # Default to full automation
                priority=req.priority,
                estimated_complexity=self._estimate_test_complexity(req)
            )
            test_specs.append(test_spec)

        return test_specs

    def _categorize_requirement(self, requirement: str) -> str:
        """Categorize requirement for testing"""
        req_lower = requirement.lower()

        if any(word in req_lower for word in ['error', 'exception', 'failure', 'invalid']):
            return "Error Handling"
        elif any(word in req_lower for word in ['performance', 'speed', 'latency', 'response']):
            return "Performance"
        elif any(word in req_lower for word in ['security', 'auth', 'permission', 'access']):
            return "Security"
        elif any(word in req_lower for word in ['user', 'interface', 'ui', 'display']):
            return "User Interface"
        else:
            return "Happy Path"

    def _check_dgts_compliance(self, requirement: str) -> bool:
        """Check if requirement can be implemented without gaming"""
        # Simple heuristic - requirements that demand real functionality are DGTS compliant
        anti_patterns = [
            'display', 'show', 'present', 'render',  # UI-heavy
            'must work', 'should function', 'actually'  # Real functionality
        ]

        req_lower = requirement.lower()
        return any(pattern in req_lower for pattern in anti_patterns)

    def _estimate_test_complexity(self, requirement: DocumentRequirement) -> int:
        """Estimate test complexity (1-10)"""
        complexity = 1

        # Base complexity on acceptance criteria count
        complexity += len(req.acceptance_criteria) * 2

        # Increase for complex requirements
        if any(word in requirement.requirement.lower() for word in ['multiple', 'various', 'complex']):
            complexity += 2

        # Increase for integration requirements
        if 'integration' in requirement.section.lower():
            complexity += 3

        return min(complexity, 10)  # Cap at 10

    def _parse_technical_context(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse technical context section if present"""
        section_match = re.search(
            r'^## Technical Context.*?$(.*?)(?=^##|\Z)',
            content,
            re.MULTILINE | re.DOTALL
        )

        if not section_match:
            return None

        section_content = section_match.group(1)

        context = {}
        patterns = {
            'language': r'\*\*Language/Version\*\*:\s*(.+)',
            'dependencies': r'\*\*Primary Dependencies\*\*:\s*(.+)',
            'storage': r'\*\*Storage\*\*:\s*(.+)',
            'testing': r'\*\*Testing\*\*:\s*(.+)',
            'platform': r'\*\*Target Platform\*\*:\s*(.+)',
            'project_type': r'\*\*Project Type\*\*:\s*(.+)',
            'performance': r'\*\*Performance Goals\*\*:\s*(.+)',
            'constraints': r'\*\*Constraints\*\*:\s*(.+)',
            'scale': r'\*\*Scale/Scope\*\*:\s*(.+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, section_content)
            if match:
                context[key] = match.group(1).strip()

        return context

    def _parse_constitution_check(self, content: str) -> Dict[str, Any]:
        """Parse constitution check section if present"""
        section_match = re.search(
            r'^## Constitution Check.*?$(.*?)(?=^##|\Z)',
            content,
            re.MULTILINE | re.DOTALL
        )

        if not section_match:
            return {}

        section_content = section_match.group(1)

        # Parse checklist items
        checklist_items = re.findall(r'^- \[([ x])\] (.+)$', section_content, re.MULTILINE)

        return {
            'items': [
                {
                    'checked': item[0] == 'x',
                    'text': item[1].strip()
                }
                for item in checklist_items
            ]
        }

    def _generate_agent_tasks(self, requirements: List[DocumentRequirement],
                            test_specs: List[TestSpecification]) -> List[AgentTask]:
        """Generate tasks for specialized Archon agents"""
        tasks = []

        # Generate research tasks for unclear requirements
        unclear_reqs = [req for req in requirements if '[NEEDS CLARIFICATION' in req.requirement]
        if unclear_reqs:
            tasks.append(AgentTask(
                task_id="research-unclear-requirements",
                agent_type="strategic-planner",
                description="Research and clarify unclear requirements",
                requirements=[req.requirement for req in unclear_reqs],
                dependencies=[],
                estimated_duration=30,
                parallelizable=True,
                quality_gates=["all_requirements_clear"]
            ))

        # Generate design tasks based on complexity
        if test_specs:
            avg_complexity = sum(spec.estimated_complexity for spec in test_specs) / len(test_specs)
            if avg_complexity > 5:
                tasks.append(AgentTask(
                    task_id="system-architecture-design",
                    agent_type="system-architect",
                    description="Design system architecture for complex feature",
                    requirements=[f"Average test complexity: {avg_complexity:.1f}"],
                    dependencies=["research-unclear-requirements"] if unclear_reqs else [],
                    estimated_duration=60,
                    parallelizable=False,
                    quality_gates=["architecture_approved", "performance_targets_met"]
                ))

        # Generate implementation tasks
        tasks.append(AgentTask(
            task_id="implementation",
            agent_type="code-implementer",
            description="Implement feature with zero-error compliance",
            requirements=[req.requirement for req in requirements],
            dependencies=["system-architecture-design"] if avg_complexity > 5 else [],
            estimated_duration=120,
            parallelizable=True,
            quality_gates=["zero_typerorrors", "dgts_compliance", "95%_coverage"]
        ))

        # Generate testing tasks
        tasks.append(AgentTask(
            task_id="test-implementation",
            agent_type="test-coverage-validator",
            description="Implement comprehensive tests from specifications",
            requirements=[f"Test {spec.requirement_id}: {spec.description}" for spec in test_specs],
            dependencies=["research-unclear-requirements"] if unclear_reqs else [],
            estimated_duration=90,
            parallelizable=True,
            quality_gates=["95%_coverage", "dgts_compliance"]
        ))

        # Generate quality validation tasks
        tasks.append(AgentTask(
            task_id="quality-validation",
            agent_type="code-quality-reviewer",
            description="Validate code quality and compliance",
            requirements=["All quality gates must pass"],
            dependencies=["implementation", "test-implementation"],
            estimated_duration=30,
            parallelizable=False,
            quality_gates=["zero_typerorrors", "no_console_log", "bundle_size_compliance"]
        ))

        return tasks

    def _calculate_complexity(self, requirements: List[DocumentRequirement],
                           test_specs: List[TestSpecification]) -> int:
        """Calculate feature complexity score (0-10)"""
        complexity = 0

        # Base complexity from requirements
        complexity += len(requirements) * 0.5

        # Test complexity factor
        if test_specs:
            avg_test_complexity = sum(spec.estimated_complexity for spec in test_specs) / len(test_specs)
            complexity += avg_test_complexity * 0.3

        # Integration complexity
        integration_reqs = [req for req in requirements if 'integration' in req.section.lower()]
        complexity += len(integration_reqs) * 2

        # Quality gate complexity
        quality_reqs = [req for req in requirements if 'quality' in req.section.lower()]
        complexity += len(quality_reqs) * 1.5

        return min(int(complexity), 10)

    def _validate_specification(self, content: str) -> List[str]:
        """Validate specification for completeness and quality"""
        errors = []

        # Check for required sections
        required_sections = [
            SpecSection.USER_SCENARIOS.value,
            SpecSection.REQUIREMENTS.value,
            SpecSection.REVIEW_CHECKLIST.value
        ]

        for section in required_sections:
            if f"## {section}" not in content:
                errors.append(f"Missing required section: {section}")

        # Check for unclear requirements
        unclear_count = content.count('[NEEDS CLARIFICATION')
        if unclear_count > 0:
            errors.append(f"Found {unclear_count} unclear requirements that need clarification")

        # Check for acceptance criteria
        if 'acceptance criteria' not in content.lower():
            errors.append("Missing acceptance criteria for requirements")

        # Check for testability
        if 'test' not in content.lower():
            errors.append("Specification does not address testing requirements")

        return errors

    def convert_to_archon_format(self, enhanced_spec: EnhancedSpec) -> List[DocumentRequirement]:
        """Convert enhanced specification to Archon's DocumentRequirement format"""
        return enhanced_spec.requirements

    def generate_test_plan(self, enhanced_spec: EnhancedSpec) -> Dict[str, Any]:
        """Generate comprehensive test plan from enhanced specification"""
        return {
            'test_specifications': [
                {
                    'requirement_id': spec.requirement_id,
                    'description': spec.description,
                    'category': spec.test_category,
                    'acceptance_criteria': spec.acceptance_criteria,
                    'automation_level': spec.automation_level,
                    'priority': spec.priority,
                    'complexity': spec.estimated_complexity,
                    'dgts_compliance': spec.dgts_compliance
                }
                for spec in enhanced_spec.test_specifications
            ],
            'agent_tasks': [
                {
                    'task_id': task.task_id,
                    'agent_type': task.agent_type,
                    'description': task.description,
                    'estimated_duration': task.estimated_duration,
                    'parallelizable': task.parallelizable,
                    'dependencies': task.dependencies,
                    'quality_gates': task.quality_gates
                }
                for task in enhanced_spec.agent_tasks
            ],
            'complexity_score': enhanced_spec.complexity_score,
            'validation_errors': enhanced_spec.validation_errors
        }

    def validate_agent_compatibility(self, enhanced_spec: EnhancedSpec,
                                   project_agents_path: str) -> Dict[str, Any]:
        """Validate that required agents are available in the project"""
        try:
            # Load project agents configuration
            agents_config = Path(project_agents_path)
            if not agents_config.exists():
                return {'compatible': False, 'error': 'No project agents configuration found'}

            # This would parse the YAML and check agent availability
            # For now, return a simple compatibility check
            available_agents = ['strategic-planner', 'system-architect', 'code-implementer',
                              'test-coverage-validator', 'code-quality-reviewer']

            required_agents = set(task.agent_type for task in enhanced_spec.agent_tasks)
            missing_agents = required_agents - set(available_agents)

            return {
                'compatible': len(missing_agents) == 0,
                'missing_agents': list(missing_agents),
                'available_agents': available_agents
            }

        except Exception as e:
            logger.error(f"Error validating agent compatibility: {e}")
            return {'compatible': False, 'error': str(e)}

def parse_enhanced_specification(spec_path: str, project_path: str = ".") -> EnhancedSpec:
    """
    Main function to parse enhanced specification files

    Args:
        spec_path: Path to the enhanced specification file
        project_path: Path to the project root

    Returns:
        EnhancedSpec: Parsed and validated specification
    """
    parser = EnhancedSpecParser(project_path)
    return parser.parse_specification(spec_path)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_spec_parser.py <spec_path> [project_path]")
        sys.exit(1)

    spec_path = sys.argv[1]
    project_path = sys.argv[2] if len(sys.argv) > 2 else "."

    try:
        enhanced_spec = parse_enhanced_specification(spec_path, project_path)

        # Convert to Archon format
        archon_requirements = enhanced_spec.requirements

        # Generate test plan
        test_plan = parser.generate_test_plan(enhanced_spec)

        # Output results
        result = {
            'specification': {
                'metadata': enhanced_spec.metadata,
                'requirements_count': len(enhanced_spec.requirements),
                'test_specs_count': len(enhanced_spec.test_specifications),
                'complexity_score': enhanced_spec.complexity_score,
                'agent_tasks_count': len(enhanced_spec.agent_tasks)
            },
            'archon_requirements': [
                {
                    'doc_type': req.doc_type,
                    'section': req.section,
                    'requirement': req.requirement[:100] + '...' if len(req.requirement) > 100 else req.requirement,
                    'acceptance_criteria_count': len(req.acceptance_criteria),
                    'priority': req.priority
                }
                for req in archon_requirements
            ],
            'test_plan': test_plan,
            'validation_errors': enhanced_spec.validation_errors
        }

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)