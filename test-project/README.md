# test-project

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
