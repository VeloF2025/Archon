#!/usr/bin/env python3
"""
ENHANCED SPECIFICATION CLI
Integrates Spec Kit's structured approach with Archon's validation and agent systems

This CLI provides enhanced commands for spec-driven development:
- /specify-enhanced: Create enhanced specifications with Archon integration
- /plan-enhanced: Generate plans with agent orchestration
- /tasks-enhanced: Create tasks with multi-agent support
- /validate-enhanced: Comprehensive validation with DGTS compliance
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.live import Live

# Add Archon imports
sys.path.append(str(Path(__file__).parent.parent))
from agents.validation.enhanced_spec_parser import EnhancedSpecParser, parse_enhanced_specification
from agents.validation.doc_driven_validator import DocDrivenValidator, validate_doc_driven_development
from agents.validation.archon_validation_rules import get_validation_rules

# CLI App
app = typer.Typer(
    name="archon-spec",
    help="Enhanced Spec-Driven Development CLI for Archon",
    add_completion=False
)
console = Console()

# Global logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CLIConfig:
    """Configuration for the enhanced CLI"""
    project_path: str = "."
    templates_path: Optional[str] = None
    agents_config_path: Optional[str] = None
    verbose: bool = False
    output_format: str = "rich"  # rich, json, yaml

def get_default_config() -> CLIConfig:
    """Get default CLI configuration"""
    project_path = Path.cwd()

    # Find templates path
    templates_paths = [
        project_path / "spec-kit" / "templates",
        Path(__file__).parent.parent.parent / "spec-kit" / "templates",
        project_path / ".specify" / "templates"
    ]

    templates_path = None
    for path in templates_paths:
        if path.exists():
            templates_path = str(path)
            break

    # Find agents config
    agents_config_path = project_path / ".archon" / "project_agents.yaml"
    if not agents_config_path.exists():
        agents_config_path = None

    return CLIConfig(
        project_path=str(project_path),
        templates_path=templates_path,
        agents_config_path=str(agents_config_path) if agents_config_path else None
    )

@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option("enhanced", help="Template type: enhanced, standard"),
    ai_assistant: str = typer.Option("claude", help="AI assistant: claude, copilot, gemini, cursor"),
    output_dir: str = typer.Option(".", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Initialize a new enhanced specification project"""

    config = get_default_config()
    config.verbose = verbose

    console.print(f"[bold blue]🚀 Initializing Enhanced Spec Project: {project_name}[/]")

    # Create project directory
    project_dir = Path(output_dir) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        ("Creating project structure", "create_structure"),
        ("Copying enhanced templates", "copy_templates"),
        ("Setting up AI assistant", "setup_ai"),
        ("Validating setup", "validate_setup")
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:

        for step_desc, step_key in steps:
            task = progress.add_task(step_desc, total=None)

            if step_key == "create_structure":
                _create_project_structure(project_dir, project_name)
            elif step_key == "copy_templates":
                _copy_enhanced_templates(project_dir, config.templates_path, template)
            elif step_key == "setup_ai":
                _setup_ai_assistant(project_dir, ai_assistant)
            elif step_key == "validate_setup":
                _validate_project_setup(project_dir)

            progress.update(task, completed=True)

    console.print(f"[bold green]✅ Enhanced spec project initialized: {project_dir}[/]")

    # Show next steps
    _show_next_steps(project_dir, ai_assistant)

@app.command()
def specify(
    feature_name: str = typer.Argument(..., help="Feature name"),
    user_input: str = typer.Option("", "--input", "-i", help="User description/requirements"),
    template: str = typer.Option("enhanced", help="Template: enhanced, standard"),
    output_dir: str = typer.Option("specs", help="Output directory"),
    branch: str = typer.Option("", help="Feature branch name"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Create an enhanced feature specification"""

    config = get_default_config()
    config.verbose = verbose

    console.print(f"[bold blue]📝 Creating Enhanced Specification: {feature_name}[/]")

    # Generate branch name if not provided
    if not branch:
        branch = f"feat-{feature_name.lower().replace(' ', '-')}"

    # Create output directory
    spec_dir = Path(output_dir) / f"{branch}-spec"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Get template path
    template_file = Path(config.templates_path) / f"enhanced-spec-template.md" if template == "enhanced" else \
                    Path(config.templates_path) / "spec-template.md"

    if not template_file.exists():
        console.print(f"[bold red]❌ Template not found: {template_file}[/]")
        raise typer.Exit(1)

    # Read template
    template_content = template_file.read_text(encoding='utf-8')

    # Replace placeholders
    spec_content = template_content.replace("[FEATURE NAME]", feature_name)
    spec_content = spec_content.replace("[###-feature-name]", branch)
    spec_content = spec_content.replace("[DATE]", "2025-09-17")  # TODO: Use current date
    spec_content = spec_content.replace("$ARGUMENTS", user_input or f"Implement {feature_name}")

    # Write specification
    spec_file = spec_dir / "spec.md"
    spec_file.write_text(spec_content, encoding='utf-8')

    console.print(f"[bold green]✅ Enhanced specification created: {spec_file}[/]")

    # Show specification structure
    _show_spec_structure(spec_file)

@app.command()
def plan(
    spec_path: str = typer.Argument(..., help="Path to specification file"),
    output_dir: str = typer.Option(".", help="Output directory"),
    agents: str = typer.Option("auto", help="Agent selection: auto, manual"),
    constitution_check: bool = typer.Option(True, help="Run constitution check"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Generate enhanced implementation plan from specification"""

    config = get_default_config()
    config.verbose = verbose

    console.print(f"[bold blue]📋 Generating Enhanced Plan: {spec_path}[/]")

    spec_path = Path(spec_path)
    if not spec_path.exists():
        console.print(f"[bold red]❌ Specification not found: {spec_path}[/]")
        raise typer.Exit(1)

    # Parse specification
    try:
        enhanced_spec = parse_enhanced_specification(str(spec_path), config.project_path)

        if enhanced_spec.validation_errors:
            console.print("[bold yellow]⚠️  Specification validation errors:[/]")
            for error in enhanced_spec.validation_errors:
                console.print(f"  - {error}")

            if not typer.confirm("Continue with plan generation?"):
                raise typer.Exit(1)

        # Show specification summary
        _show_spec_summary(enhanced_spec)

        # Generate plan directory
        spec_name = spec_path.stem
        plan_dir = Path(output_dir) / f"{spec_name}-plan"
        plan_dir.mkdir(parents=True, exist_ok=True)

        # Generate plan files
        _generate_plan_files(enhanced_spec, plan_dir, config)

        console.print(f"[bold green]✅ Enhanced plan generated: {plan_dir}[/]")

        # Show next steps
        _show_plan_next_steps(plan_dir, enhanced_spec)

    except Exception as e:
        console.print(f"[bold red]❌ Error parsing specification: {e}[/]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

@app.command()
def tasks(
    plan_path: str = typer.Argument(..., help="Path to plan directory"),
    output_file: str = typer.Option("tasks.md", help="Output tasks file"),
    parallel: bool = typer.Option(True, help="Enable parallel execution planning"),
    agent_assignment: str = typer.Option("auto", help="Agent assignment: auto, manual"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Generate enhanced task list from plan"""

    config = get_default_config()
    config.verbose = verbose

    console.print(f"[bold blue]📝 Generating Enhanced Tasks: {plan_path}[/]")

    plan_dir = Path(plan_path)
    if not plan_dir.exists():
        console.print(f"[bold red]❌ Plan directory not found: {plan_dir}[/]")
        raise typer.Exit(1)

    # Parse plan files
    try:
        # Read plan.md
        plan_file = plan_dir / "plan.md"
        if not plan_file.exists():
            console.print(f"[bold red]❌ Plan file not found: {plan_file}[/]")
            raise typer.Exit(1)

        plan_content = plan_file.read_text(encoding='utf-8')

        # Read research.md if available
        research_file = plan_dir / "research.md"
        research_content = ""
        if research_file.exists():
            research_content = research_file.read_text(encoding='utf-8')

        # Read test specifications if available
        test_specs_file = plan_dir / "test-specifications.md"
        test_specs_content = ""
        if test_specs_file.exists():
            test_specs_content = test_specs_file.read_text(encoding='utf-8')

        # Generate tasks
        tasks = _generate_enhanced_tasks(
            plan_content,
            research_content,
            test_specs_content,
            parallel,
            agent_assignment,
            config
        )

        # Write tasks file
        tasks_file = Path(output_file)
        tasks_file.write_text(tasks, encoding='utf-8')

        console.print(f"[bold green]✅ Enhanced tasks generated: {tasks_file}[/]")

        # Show task summary
        _show_task_summary(tasks)

    except Exception as e:
        console.print(f"[bold red]❌ Error generating tasks: {e}[/]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

@app.command()
def validate(
    target_path: str = typer.Argument(..., help="Path to validate (spec, plan, or project)"),
    validation_type: str = typer.Option("auto", help="Validation type: auto, spec, plan, archon"),
    strict: bool = typer.Option(False, help="Strict validation (fail on warnings)"),
    output_format: str = typer.Option("rich", help="Output format: rich, json"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Validate specification, plan, or project against Archon standards"""

    config = get_default_config()
    config.verbose = verbose

    console.print(f"[bold blue]🔍 Validating: {target_path}[/]")

    target = Path(target_path)
    if not target.exists():
        console.print(f"[bold red]❌ Target not found: {target}[/]")
        raise typer.Exit(1)

    # Determine validation type
    if validation_type == "auto":
        if target.is_file() and target.name.endswith(".md"):
            if "spec" in target.name.lower():
                validation_type = "spec"
            elif "plan" in target.name.lower():
                validation_type = "plan"
            else:
                validation_type = "archon"
        else:
            validation_type = "archon"

    try:
        if validation_type == "spec":
            result = _validate_specification(target, config)
        elif validation_type == "plan":
            result = _validate_plan(target.parent, config)
        elif validation_type == "archon":
            result = _validate_archon_project(target, config)
        else:
            console.print(f"[bold red]❌ Unknown validation type: {validation_type}[/]")
            raise typer.Exit(1)

        # Display results
        if output_format == "json":
            console.print(json.dumps(result, indent=2, default=str))
        else:
            _display_validation_results(result, strict)

        # Exit with appropriate code
        if strict and result.get("critical_issues", 0) > 0:
            raise typer.Exit(1)
        elif not result.get("valid", True):
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]❌ Validation error: {e}[/]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

@app.command()
def status(
    project_path: str = typer.Argument(".", help="Project path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Show project status and enhanced spec compliance"""

    config = get_default_config()
    config.verbose = verbose

    console.print(f"[bold blue]📊 Project Status: {project_path}[/]")

    project_dir = Path(project_path)

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
    _display_project_status(
        specs_count=len(specs),
        plans_count=len(plans),
        tasks_count=len(tasks),
        archon_integration=archon_integration
    )

# Helper functions
def _create_project_structure(project_dir: Path, project_name: str):
    """Create standard project structure"""
    directories = [
        "specs",
        ".specify/templates",
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

## Structure
- `specs/` - Feature specifications and plans
- `contracts/` - API contracts and schemas
- `docs/` - Project documentation
- `tests/` - Test suites (unit, integration, e2e)

## Getting Started
1. Create a feature specification:
   ```bash
   archon-spec specify feature-name --input "User description"
   ```

2. Generate implementation plan:
   ```bash
   archon-spec plan specs/feature-spec/spec.md
   ```

3. Create tasks:
   ```bash
   archon-spec tasks specs/feature-spec-plan/
   ```

## Commands
- `archon-spec init` - Initialize new project
- `archon-spec specify` - Create specification
- `archon-spec plan` - Generate plan
- `archon-spec tasks` - Create tasks
- `archon-spec validate` - Validate compliance
- `archon-spec status` - Show project status
"""

    (project_dir / "README.md").write_text(readme_content)

def _copy_enhanced_templates(project_dir: Path, templates_path: Optional[str], template_type: str):
    """Copy enhanced templates to project"""
    if not templates_path:
        return

    templates_source = Path(templates_path)
    templates_dest = project_dir / ".specify" / "templates"
    templates_dest.mkdir(parents=True, exist_ok=True)

    # Copy enhanced templates
    enhanced_templates = [
        "enhanced-spec-template.md",
        "enhanced-plan-template.md"
    ]

    for template in enhanced_templates:
        source_file = templates_source / template
        if source_file.exists():
            dest_file = templates_dest / template
            dest_file.write_text(source_file.read_text(encoding='utf-8'))

def _setup_ai_assistant(project_dir: Path, ai_assistant: str):
    """Setup AI assistant configuration"""
    # Create AI assistant configuration
    ai_config = {
        "assistant": ai_assistant,
        "created_at": "2025-09-17",  # TODO: Use current date
        "project_type": "enhanced-spec"
    }

    config_file = project_dir / ".specify" / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        json.dump(ai_config, f, indent=2)

def _validate_project_setup(project_dir: Path):
    """Validate that project setup is complete"""
    required_files = [
        "README.md",
        ".specify/config.json"
    ]

    for file_path in required_files:
        if not (project_dir / file_path).exists():
            raise RuntimeError(f"Missing required file: {file_path}")

def _show_next_steps(project_dir: Path, ai_assistant: str):
    """Show next steps for project setup"""
    console.print("\n[bold cyan]Next Steps:[/]")
    console.print("1. Create your first feature specification:")
    console.print(f"   archon-spec specify feature-name --input \"User requirements\"")
    console.print("2. Generate implementation plan:")
    console.print("   archon-spec plan specs/feature-spec/spec.md")
    console.print("3. Create and execute tasks:")
    console.print("   archon-spec tasks specs/feature-spec-plan/")
    console.print("4. Validate compliance:")
    console.print("   archon-spec validate .")

def _show_spec_structure(spec_file: Path):
    """Show specification file structure"""
    console.print("\n[bold cyan]Specification Structure:[/]")

    # Parse and show key sections
    content = spec_file.read_text(encoding='utf-8')
    sections = re.findall(r'^## (.+)$', content, re.MULTILINE)

    for section in sections:
        console.print(f"  - {section}")

def _show_spec_summary(enhanced_spec):
    """Show specification summary"""
    table = Table(title="Specification Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Requirements", str(len(enhanced_spec.requirements)))
    table.add_row("Test Specifications", str(len(enhanced_spec.test_specifications)))
    table.add_row("Agent Tasks", str(len(enhanced_spec.agent_tasks)))
    table.add_row("Complexity Score", str(enhanced_spec.complexity_score))
    table.add_row("Validation Errors", str(len(enhanced_spec.validation_errors)))

    console.print(table)

def _generate_plan_files(enhanced_spec, plan_dir, config):
    """Generate plan files from enhanced specification"""
    # Create plan.md
    plan_template = Path(config.templates_path) / "enhanced-plan-template.md"
    if plan_template.exists():
        plan_content = plan_template.read_text(encoding='utf-8')

        # Replace placeholders
        plan_content = plan_content.replace("[FEATURE]", enhanced_spec.metadata.get('feature_name', 'Unknown'))

        plan_file = plan_dir / "plan.md"
        plan_file.write_text(plan_content, encoding='utf-8')

    # Create research.md placeholder
    research_file = plan_dir / "research.md"
    research_content = f"""# Research: {enhanced_spec.metadata.get('feature_name', 'Unknown')}

## Technical Unknowns
{chr(10).join(f'- {req.requirement}' for req in enhanced_spec.requirements if '[NEEDS CLARIFICATION' in req.requirement)}

## Dependencies to Research
- [List technical dependencies requiring research]

## Integration Points
- [List integration points requiring investigation]

## Best Practices
- [List best practices to research]

## Research Results
<!-- Update with research findings -->
"""
    research_file.write_text(research_content, encoding='utf-8')

    # Create test-specifications.md
    test_specs_file = plan_dir / "test-specifications.md"
    test_specs_content = f"""# Test Specifications: {enhanced_spec.metadata.get('feature_name', 'Unknown')}

## Test Specifications by Requirement

{chr(10).join(f'''
### {spec.requirement_id}: {spec.description}
**Category**: {spec.test_category}
**Priority**: {spec.priority}
**Complexity**: {spec.estimated_complexity}/10
**DGTS Compliant**: {'Yes' if spec.dgts_compliance else 'No'}

**Acceptance Criteria**:
{chr(10).join(f'- {criteria}' for criteria in spec.acceptance_criteria)}
''' for spec in enhanced_spec.test_specifications)}

## Test Coverage Requirements
- **Minimum Coverage**: 95%
- **Test Quality**: No mocked implementations for core functionality
- **Automation Level**: Full automation where possible
- **Parallel Execution**: Enabled for independent tests
"""
    test_specs_file.write_text(test_specs_content, encoding='utf-8')

def _show_plan_next_steps(plan_dir, enhanced_spec):
    """Show next steps for plan execution"""
    console.print("\n[bold cyan]Next Steps:[/]")
    console.print("1. Complete research phase:")
    console.print(f"   Edit {plan_dir}/research.md")
    console.print("2. Generate tasks:")
    console.print(f"   archon-spec tasks {plan_dir}")
    console.print("3. Execute tasks with ForgeFlow:")
    console.print("   @FF execute tasks.md")
    console.print("4. Validate implementation:")
    console.print("   archon-spec validate .")

def _generate_enhanced_tasks(plan_content, research_content, test_specs_content, parallel, agent_assignment, config):
    """Generate enhanced task list"""
    # This is a simplified task generation
    # In a full implementation, this would parse the plan content more intelligently

    tasks = f"""# Enhanced Task List
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

## Phase 5: Deployment
21. [ ] Prepare deployment
22. [ ] Deploy to staging
23. [ ] Validate production readiness
24. [ ] Deploy to production

## Agent Assignments
*Based on enhanced specification analysis*

**strategic-planner**: Tasks 1-4
**system-architect**: Tasks 5-8
**code-implementer**: Tasks 9-12
**test-coverage-validator**: Tasks 13-16
**code-quality-reviewer**: Tasks 17-20
**deployment-automation**: Tasks 21-24

## Parallel Execution
- **High Parallelization**: Tasks 5-8, 9-12, 13-16
- **Sequential**: Tasks 1-4, 17-20, 21-24
- **Dependencies**: Each phase depends on previous phases

## Quality Gates
- Zero TypeScript/ESLint errors
- >95% test coverage
- DGTS compliance
- Performance targets met
- Security validation passed
"""

    return tasks

def _show_task_summary(tasks):
    """Show task summary"""
    # Count tasks by phase
    phases = re.findall(r'## Phase \d+: (.+)', tasks)
    task_lines = [line for line in tasks.split('\n') if line.strip().startswith('- [ ]')]

    console.print(f"[bold cyan]Task Summary:[/]")
    console.print(f"  - Total Tasks: {len(task_lines)}")
    console.print(f"  - Phases: {len(phases)}")

    # Show agent assignments
    agent_matches = re.findall(r'\*\*(.+?)\*\*:', tasks)
    if agent_matches:
        console.print(f"  - Agents: {len(agent_matches)}")

def _validate_specification(spec_file, config):
    """Validate specification file"""
    try:
        enhanced_spec = parse_enhanced_specification(str(spec_file), config.project_path)

        return {
            "valid": len(enhanced_spec.validation_errors) == 0,
            "critical_issues": len(enhanced_spec.validation_errors),
            "warnings": 0,  # TODO: Add warning detection
            "specification": {
                "requirements_count": len(enhanced_spec.requirements),
                "test_specs_count": len(enhanced_spec.test_specifications),
                "complexity_score": enhanced_spec.complexity_score
            },
            "validation_errors": enhanced_spec.validation_errors
        }
    except Exception as e:
        return {
            "valid": False,
            "critical_issues": 1,
            "error": str(e)
        }

def _validate_plan(plan_dir, config):
    """Validate plan directory"""
    plan_file = plan_dir / "plan.md"
    research_file = plan_dir / "research.md"

    issues = []

    if not plan_file.exists():
        issues.append("Missing plan.md")

    if not research_file.exists():
        issues.append("Missing research.md")

    return {
        "valid": len(issues) == 0,
        "critical_issues": len(issues),
        "issues": issues
    }

def _validate_archon_project(project_dir, config):
    """Validate Archon project compliance"""
    # Run Archon's validation
    try:
        validator = DocDrivenValidator(str(project_dir))
        result = validator.enforce_doc_driven_development()

        return {
            "valid": result.get("compliant", False),
            "critical_issues": len(result.get("critical_violations", [])),
            "archon_compliance": result
        }
    except Exception as e:
        return {
            "valid": False,
            "critical_issues": 1,
            "error": str(e)
        }

def _display_validation_results(result, strict):
    """Display validation results"""
    if result.get("valid", False):
        console.print("[bold green]✅ Validation passed[/]")
    else:
        console.print(f"[bold red]❌ Validation failed ({result.get('critical_issues', 0)} critical issues)[/]")

    if "validation_errors" in result:
        for error in result["validation_errors"]:
            console.print(f"  - {error}")

    if "issues" in result:
        for issue in result["issues"]:
            console.print(f"  - {issue}")

def _display_project_status(specs_count, plans_count, tasks_count, archon_integration):
    """Display project status"""
    console.print("\n[bold cyan]Project Status:[/]")

    # Enhanced Spec Metrics
    table = Table(title="Enhanced Spec Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")

    table.add_row("Specifications", str(specs_count))
    table.add_row("Plans", str(plans_count))
    table.add_row("Task Lists", str(tasks_count))

    console.print(table)

    # Archon Integration
    console.print("\n[bold cyan]Archon Integration:[/]")
    for integration, enabled in archon_integration.items():
        status = "✅" if enabled else "❌"
        console.print(f"  {status} {integration.replace('_', ' ').title()}")

if __name__ == "__main__":
    app()