#!/usr/bin/env python3
"""
Setup script for Enhanced Spec CLI
Integrates Spec Kit commands with Archon's systems
"""

import os
import sys
import shutil
from pathlib import Path

def setup_enhanced_cli():
    """Setup the enhanced CLI for Archon"""

    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent

    # CLI source file
    cli_source = Path(__file__).parent / "enhanced_spec_cli.py"

    # Installation targets
    install_locations = [
        project_root / "archon-spec",  # Direct in project root
        project_root / "scripts" / "archon-spec",  # Scripts directory
    ]

    print("Setting up Enhanced Spec CLI for Archon...")
    print(f"Project root: {project_root}")
    print(f"CLI source: {cli_source}")

    # Create scripts directory if needed
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Copy CLI to project root
    target_cli = project_root / "archon-spec.py"
    if not target_cli.exists():
        shutil.copy2(cli_source, target_cli)
        print(f"[OK] CLI installed: {target_cli}")
    else:
        print(f"[OK] CLI already exists: {target_cli}")

    # Create scripts version
    scripts_cli = scripts_dir / "archon-spec.py"
    if not scripts_cli.exists():
        shutil.copy2(cli_source, scripts_cli)
        print(f"[OK] CLI installed in scripts: {scripts_cli}")

    # Create executable version (Unix-like systems)
    if os.name != 'nt':  # Not Windows
        exec_target = project_root / "archon-spec"
        if not exec_target.exists():
            # Create executable script
            with open(exec_target, 'w') as f:
                f.write('#!/usr/bin/env python3\n')
                f.write('import sys\n')
                f.write('import os\n')
                f.write('sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "src"))\n')
                f.write('from cli.enhanced_spec_cli import app\n')
                f.write('if __name__ == "__main__":\n')
                f.write('    app()\n')

            # Make executable
            os.chmod(exec_target, 0o755)
            print(f"[OK] Executable created: {exec_target}")

    # Create Windows batch file
    if os.name == 'nt':  # Windows
        batch_target = project_root / "archon-spec.bat"
        if not batch_target.exists():
            with open(batch_target, 'w') as f:
                f.write('@echo off\n')
                f.write('python "%~dp0archon-spec.py" %*\n')
            print(f"[OK] Batch file created: {batch_target}")

    # Update project README if it exists
    readme_file = project_root / "README.md"
    if readme_file.exists():
        readme_content = readme_file.read_text(encoding='utf-8')

        # Check if CLI section already exists
        if "## Enhanced Spec CLI" not in readme_content:
            # Add CLI section to README
            cli_section = """

## Enhanced Spec CLI

Archon now includes an enhanced specification-driven development CLI that integrates Spec Kit's structured approach with Archon's validation and agent systems.

### Installation

The CLI is automatically installed in the project root as `archon-spec.py`.

### Commands

```bash
# Initialize a new enhanced spec project
python archon-spec.py init my-project

# Create an enhanced feature specification
python archon-spec.py specify feature-name --input "User requirements"

# Generate implementation plan
python archon-spec.py plan specs/feature-spec/spec.md

# Create task list
python archon-spec.py tasks specs/feature-spec-plan/

# Validate compliance
python archon-spec.py validate .

# Show project status
python archon-spec.py status
```

### Features

- **Enhanced Specifications**: Structured specs with Archon integration requirements
- **Agent Orchestration**: Automatic task assignment to specialized Archon agents
- **Quality Gates**: Built-in validation against Archon's strict quality standards
- **Multi-AI Support**: Compatible with Claude, Copilot, Gemini, and Cursor
- **DGTS Compliance**: Prevents test gaming and ensures real functionality
"""

            # Add to README
            with open(readme_file, 'a') as f:
                f.write(cli_section)

            print("[OK] README updated with CLI documentation")

    # Create example usage
    examples_dir = project_root / "examples" / "enhanced-spec"
    examples_dir.mkdir(parents=True, exist_ok=True)

    example_spec = examples_dir / "example-spec.md"
    if not example_spec.exists():
        example_content = """# Enhanced Feature Specification: User Authentication System

**Feature Branch**: `feat-user-authentication`
**Created**: 2025-09-17
**Status**: Draft
**Input**: User description: "Implement secure user authentication with email/password and OAuth"

## User Scenarios & Testing

### Primary User Story
Users need to securely authenticate with the system using email/password credentials or OAuth providers to access protected features.

### Acceptance Scenarios
1. **Given** a new user visits the login page, **When** they enter valid credentials, **Then** they should be logged in and redirected to dashboard
2. **Given** an existing user, **When** they click "Login with Google", **Then** they should be authenticated via OAuth

## Requirements

### Functional Requirements
- **FR-001**: System MUST allow users to register with email and password
- **FR-002**: System MUST validate email format and password strength
- **FR-003**: Users MUST be able to login with existing credentials
- **FR-004**: System MUST support OAuth authentication with Google and GitHub
- **FR-005**: System MUST provide password reset functionality

### Archon Integration Requirements
- **ADR-001**: Feature MUST be documented with comprehensive test specifications
- **ADR-002**: All authentication logic MUST have >95% test coverage
- **ADR-003**: Implementation MUST NOT use console.log statements
- **QG-001**: All authentication endpoints MUST validate input
- **DGTS-001**: Authentication tests MUST validate real security functionality

---

*Generated with Enhanced Spec CLI*
"""
        example_spec.write_text(example_content)
        print(f"[OK] Example specification created: {example_spec}")

    print("\nEnhanced Spec CLI setup complete!")
    print("\nQuick Start:")
    print("1. Create a feature specification:")
    print("   python archon-spec.py specify my-feature --input 'Feature description'")
    print("2. Generate implementation plan:")
    print("   python archon-spec.py plan specs/my-feature-spec/spec.md")
    print("3. See all commands:")
    print("   python archon-spec.py --help")

if __name__ == "__main__":
    setup_enhanced_cli()