#!/usr/bin/env python3
"""
ENHANCED SPECIFICATION CLI - Simple Version
Integrates Spec Kit's structured approach with Archon's validation and agent systems
"""

import os
import sys
import json
import logging
import re
from pathlib import Path

# Add Archon imports
sys.path.insert(0, str(Path(__file__).parent / "python" / "src"))

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    if command == "help" or command == "--help" or command == "-h":
        print_help()
    elif command == "init":
        handle_init(sys.argv[2:])
    elif command == "specify":
        handle_specify(sys.argv[2:])
    elif command == "plan":
        handle_plan(sys.argv[2:])
    elif command == "tasks":
        handle_tasks(sys.argv[2:])
    elif command == "validate":
        handle_validate(sys.argv[2:])
    elif command == "status":
        handle_status(sys.argv[2:])
    else:
        print(f"Unknown command: {command}")
        print_help()

def print_help():
    """Print help information"""
    help_text = """
Enhanced Spec-Driven Development CLI for Archon

Usage:
    python archon-spec-simple.py <command> [options]

Commands:
    init <project-name>      Initialize a new enhanced spec project
    specify <feature-name>   Create an enhanced feature specification
    plan <spec-file>         Generate implementation plan from specification
    tasks <plan-dir>         Create task list from plan
    validate <path>          Validate specification, plan, or project
    status                   Show project status

Options:
    --input <text>          User description/requirements (for specify)
    --template <type>        Template type: enhanced, standard
    --output-dir <path>      Output directory
    --verbose                Verbose output

Examples:
    # Initialize a new project
    python archon-spec-simple.py init my-project

    # Create a feature specification
    python archon-spec-simple.py specify user-auth --input "Implement secure login system"

    # Generate plan
    python archon-spec-simple.py plan specs/user-auth-spec/spec.md

    # Create tasks
    python archon-spec-simple.py tasks specs/user-auth-spec-plan/

    # Validate project
    python archon-spec-simple.py validate .

Features:
- Enhanced Specifications with Archon integration
- Agent Orchestration and task assignment
- Quality Gates and DGTS compliance
- Multi-AI support (Claude, Copilot, Gemini, Cursor)
"""
    print(help_text)

def handle_init(args):
    """Handle init command"""
    if not args:
        print("Error: Project name required")
        print("Usage: python archon-spec-simple.py init <project-name>")
        return

    project_name = args[0]
    output_dir = "."

    # Parse options
    for i, arg in enumerate(args):
        if arg == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]

    print(f"Initializing enhanced spec project: {project_name}")

    # Create project directory
    project_dir = Path(output_dir) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create basic structure
    directories = [
        "specs",
        "contracts",
        "docs",
        "tests/unit",
        "tests/integration",
        "tests/e2e"
    ]

    for directory in directories:
        (project_dir / directory).mkdir(parents=True, exist_ok=True)

    # Create README
    readme_content = f"""# {project_name}

Enhanced specification-driven development project.

## Getting Started
1. Create a feature specification:
   ```bash
   python archon-spec-simple.py specify feature-name --input "User description"
   ```

2. Generate implementation plan:
   ```bash
   python archon-spec-simple.py plan specs/feature-spec/spec.md
   ```

3. Create tasks:
   ```bash
   python archon-spec-simple.py tasks specs/feature-spec-plan/
   ```

## Commands
- `init` - Initialize new project
- `specify` - Create specification
- `plan` - Generate plan
- `tasks` - Create tasks
- `validate` - Validate compliance
- `status` - Show project status

## Features
- Enhanced Specifications with Archon integration requirements
- Agent Orchestration with specialized Archon agents
- Quality Gates and DGTS compliance
- Multi-AI support (Claude, Copilot, Gemini, Cursor)
"""
    (project_dir / "README.md").write_text(readme_content)

    print(f"[OK] Project initialized: {project_dir}")
    print("Next steps:")
    print("1. Create a feature specification:")
    print(f"   python archon-spec-simple.py specify feature-name --input 'User requirements'")
    print("2. Generate implementation plan:")
    print("   python archon-spec-simple.py plan specs/feature-spec/spec.md")

def handle_specify(args):
    """Handle specify command"""
    if not args:
        print("Error: Feature name required")
        print("Usage: python archon-spec-simple.py specify <feature-name> [options]")
        return

    feature_name = args[0]
    user_input = ""
    output_dir = "specs"
    template = "enhanced"

    # Parse options
    for i, arg in enumerate(args):
        if arg == "--input" and i + 1 < len(args):
            user_input = args[i + 1]
        elif arg == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
        elif arg == "--template" and i + 1 < len(args):
            template = args[i + 1]

    print(f"Creating enhanced specification: {feature_name}")

    # Generate branch name
    branch = f"feat-{feature_name.lower().replace(' ', '-')}"

    # Create output directory
    spec_dir = Path(output_dir) / f"{branch}-spec"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Get template path
    templates_path = Path(__file__).parent / "spec-kit" / "templates"
    template_file = templates_path / f"enhanced-spec-template.md"

    if not template_file.exists():
        print(f"[ERROR] Template not found: {template_file}")
        return

    # Read template
    template_content = template_file.read_text(encoding='utf-8')

    # Replace placeholders
    spec_content = template_content.replace("[FEATURE NAME]", feature_name)
    spec_content = spec_content.replace("[###-feature-name]", branch)
    spec_content = spec_content.replace("[DATE]", "2025-09-17")
    spec_content = spec_content.replace("$ARGUMENTS", user_input or f"Implement {feature_name}")

    # Write specification
    spec_file = spec_dir / "spec.md"
    spec_file.write_text(spec_content, encoding='utf-8')

    print(f"[OK] Enhanced specification created: {spec_file}")

    # Show specification structure
    print("\nSpecification structure:")
    content = spec_file.read_text(encoding='utf-8')
    sections = re.findall(r'^## (.+)$', content, re.MULTILINE)
    for section in sections:
        # Remove Unicode characters for display
        clean_section = re.sub(r'[^\x00-\x7F]+', '', section)
        print(f"  - {clean_section}")

def handle_plan(args):
    """Handle plan command"""
    if not args:
        print("Error: Specification file required")
        print("Usage: python archon-spec-simple.py plan <spec-file>")
        return

    spec_path = args[0]
    output_dir = "."

    # Parse options
    for i, arg in enumerate(args):
        if arg == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]

    spec_path = Path(spec_path)
    if not spec_path.exists():
        print(f"[ERROR] Specification not found: {spec_path}")
        return

    print(f"Generating enhanced plan: {spec_path}")

    # Parse specification (simplified)
    try:
        content = spec_path.read_text(encoding='utf-8')

        # Extract feature name
        feature_match = re.search(r'^# .*?: (.+)$', content, re.MULTILINE)
        feature_name = feature_match.group(1) if feature_match else "Unknown"

        # Create plan directory
        spec_name = spec_path.stem
        plan_dir = Path(output_dir) / f"{spec_name}-plan"
        plan_dir.mkdir(parents=True, exist_ok=True)

        # Create basic plan structure
        plan_content = f"""# Implementation Plan: {feature_name}

**Branch**: {spec_name.replace('-spec', '')} | **Date**: 2025-09-17 | **Spec**: {spec_path.name}

## Summary
Extracted from enhanced specification with Archon integration requirements.

## Technical Context
*To be filled during planning phase*

**Language/Version**: [NEEDS CLARIFICATION]
**Primary Dependencies**: [NEEDS CLARIFICATION]
**Storage**: [NEEDS CLARIFICATION]
**Testing**: [NEEDS CLARIFICATION]
**Target Platform**: [NEEDS CLARIFICATION]
**Project Type**: [single/web/mobile]
**Performance Goals**: [NEEDS CLARIFICATION]
**Constraints**: [NEEDS CLARIFICATION]
**Scale/Scope**: [NEEDS CLARIFICATION]

## Constitution Check
*To be validated during planning*

## Phase 0: Outline & Research
1. Extract unknowns from Technical Context above
2. Generate and dispatch research agents
3. Consolidate findings in research.md

## Phase 1: Design & Contracts
1. Extract entities from feature specification
2. Generate API contracts
3. Generate contract tests
4. Extract test scenarios
5. Update agent files incrementally

## Phase 2: Task Planning
1. Generate tasks from Phase 1 design docs
2. Plan parallel execution where possible
3. Assign tasks to specialized agents

## Phase 3+: Implementation
*Beyond scope of planning command*

---
*Generated with Enhanced Spec CLI*
"""

        plan_file = plan_dir / "plan.md"
        plan_file.write_text(plan_content, encoding='utf-8')

        # Create research.md
        research_content = f"""# Research: {feature_name}

## Technical Unknowns
- List technical dependencies requiring research
- [NEEDS CLARIFICATION] items from specification

## Dependencies to Research
- List technical dependencies requiring research

## Integration Points
- List integration points requiring investigation

## Best Practices
- List best practices to research

## Research Results
<!-- Update with research findings -->
"""
        research_file = plan_dir / "research.md"
        research_file.write_text(research_content, encoding='utf-8')

        print(f"[OK] Enhanced plan generated: {plan_dir}")
        print("\nNext steps:")
        print("1. Complete research phase:")
        print(f"   Edit {plan_dir}/research.md")
        print("2. Generate tasks:")
        print(f"   python archon-spec-simple.py tasks {plan_dir}")
        print("3. Execute tasks with ForgeFlow:")
        print("   @FF execute tasks.md")

    except Exception as e:
        print(f"[ERROR] Error generating plan: {e}")

def handle_tasks(args):
    """Handle tasks command"""
    if not args:
        print("Error: Plan directory required")
        print("Usage: python archon-spec-simple.py tasks <plan-dir>")
        return

    plan_dir = args[0]
    output_file = "tasks.md"

    # Parse options
    for i, arg in enumerate(args):
        if arg == "--output-file" and i + 1 < len(args):
            output_file = args[i + 1]

    plan_path = Path(plan_dir)
    if not plan_path.exists():
        print(f"[ERROR] Plan directory not found: {plan_dir}")
        return

    print(f"Generating enhanced tasks: {plan_dir}")

    # Generate basic task list
    tasks_content = """# Enhanced Task List
*Generated from enhanced specification plan*

## Phase 0: Research & Planning
1. [ ] Complete technical research
2. [ ] Validate agent compatibility
3. [ ] Review architecture decisions
4. [ ] Confirm quality gates

## Phase 1: Design & Contracts
5. [ ] Create data models
6. [ ] Design API contracts
7. [ ] Generate contract tests
8. [ ] Setup testing infrastructure

## Phase 2: Implementation
9. [ ] Implement core functionality
10. [ ] Implement API endpoints
11. [ ] Create user interface
12. [ ] Implement error handling

## Phase 3: Testing & Validation
13. [ ] Write unit tests
14. [ ] Write integration tests
15. [ ] Write end-to-end tests
16. [ ] Validate DGTS compliance

## Phase 4: Quality Assurance
17. [ ] Code quality review
18. [ ] Security audit
19. [ ] Performance testing
20. [ ] Documentation updates

## Agent Assignments
*Based on enhanced specification analysis*

**strategic-planner**: Tasks 1-4
**system-architect**: Tasks 5-8
**code-implementer**: Tasks 9-12
**test-coverage-validator**: Tasks 13-16
**code-quality-reviewer**: Tasks 17-20

## Quality Gates
- Zero TypeScript/ESLint errors
- >95% test coverage
- DGTS compliance
- Performance targets met
- Security validation passed

## Execution Notes
- Tasks 5-8, 9-12, 13-16 can be executed in parallel where possible
- Each phase depends on previous phases
- All quality gates must be satisfied before deployment
"""

    # Write tasks file
    tasks_file = Path(output_file)
    tasks_file.write_text(tasks_content, encoding='utf-8')

    print(f"[OK] Enhanced tasks generated: {tasks_file}")
    print("\nTask Summary:")
    print("  - Total Tasks: 20")
    print("  - Phases: 4")
    print("  - Agents: 5")
    print("\nNext steps:")
    print("1. Execute tasks with ForgeFlow:")
    print("   @FF execute tasks.md")
    print("2. Monitor progress:")
    print("   python archon-spec-simple.py status")

def handle_validate(args):
    """Handle validate command"""
    if not args:
        print("Error: Path to validate required")
        print("Usage: python archon-spec-simple.py validate <path>")
        return

    target_path = args[0]
    target = Path(target_path)

    if not target.exists():
        print(f"[ERROR] Target not found: {target}")
        return

    print(f"Validating: {target}")

    # Simple validation checks
    issues = []

    if target.is_file() and target.name.endswith(".md"):
        # Validate specification file
        content = target.read_text(encoding='utf-8')

        # Check for required sections
        required_sections = ["User Scenarios", "Requirements", "Review Checklist"]
        for section in required_sections:
            if section not in content:
                issues.append(f"Missing required section: {section}")

        # Check for unclear requirements
        unclear_count = content.count('[NEEDS CLARIFICATION')
        if unclear_count > 0:
            issues.append(f"Found {unclear_count} unclear requirements")

    elif target.is_dir():
        # Validate project directory
        specs = list(target.glob("specs/**/spec.md"))
        plans = list(target.glob("**/plan.md")) + list(target.glob("spec*/plan.md"))
        tasks = list(target.glob("**/tasks.md")) + list(target.glob("tasks.md"))

        if not specs:
            issues.append("No specifications found")

        if not plans:
            issues.append("No implementation plans found")

        if not tasks:
            issues.append("No task lists found")

    # Display results
    if not issues:
        print("[OK] Validation passed")
    else:
        print(f"[ERROR] Validation failed ({len(issues)} issues)")
        for issue in issues:
            print(f"  - {issue}")

def handle_status(args):
    """Handle status command"""
    project_path = "."

    # Parse options
    for i, arg in enumerate(args):
        if i == 0:
            project_path = arg

    project_dir = Path(project_path)

    print(f"Project Status: {project_path}")

    # Check for specifications
    specs = list(project_dir.glob("specs/**/spec.md"))
    specs.extend(project_dir.glob("**/*spec*.md"))

    # Check for plans
    plans = list(project_dir.glob("specs/**/plan.md"))
    plans.extend(project_dir.glob("**/*plan*.md"))

    # Check for tasks
    tasks = list(project_dir.glob("specs/**/tasks.md"))
    tasks.extend(project_dir.glob("**/tasks.md"))

    # Check Archon integration
    archon_integration = {
        "has_agents_config": (project_dir / ".archon" / "project_agents.yaml").exists(),
        "has_prd_docs": len(list(project_dir.glob("PRDs/*.md"))) > 0,
        "has_validation": (project_dir / "python" / "src" / "agents" / "validation").exists(),
        "dgts_enabled": (project_dir / "python" / "src" / "agents" / "validation" / "dgts_validator.py").exists()
    }

    # Display status
    print("\nEnhanced Spec Metrics:")
    print(f"  Specifications: {len(specs)}")
    print(f"  Plans: {len(plans)}")
    print(f"  Task Lists: {len(tasks)}")

    print("\nArchon Integration:")
    for integration, enabled in archon_integration.items():
        status = "[OK]" if enabled else "[ERROR]"
        print(f"  {status} {integration.replace('_', ' ').title()}")

    if specs:
        print(f"\nRecent Specifications:")
        for spec in sorted(specs)[-3:]:
            print(f"  - {spec}")

if __name__ == "__main__":
    main()