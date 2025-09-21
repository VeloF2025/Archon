# AGENTS.md

Guidelines for agentic coding agents operating in the Archon repository.

## Build/Lint/Test Commands

### Frontend (archon-ui-main/)
- Dev server: npm run dev (port 3737)
- Build: npm run build
- Lint: npm run lint
- Test all: npm run test
- Test with coverage: npm run test:coverage
- Single test: npm run test path/to/test.file.ts -- --run

### Backend (python/)
- Sync dependencies: uv sync
- Run server: uv run python -m src.server.main (port 8181)
- Lint: uv run ruff check .
- Typecheck: uv run mypy src/
- Test all: uv run pytest
- Single test: uv run pytest tests/test_specific.py -k "test_function_name"

## Code Style Guidelines
- **Python**: Use Python 3.12, max 120 char lines. Enforce with Ruff (linting) and Mypy (types).
- **JavaScript/TypeScript**: Use ESLint (from .eslintrc.cjs), camelCase naming, type everything.
- **Imports**: Group standard/third-party/local; alphabetize within groups.
- **Formatting**: 4-space indentation, auto-format on save. No trailing commas in Python.
- **Types**: Full type hints in Python (Pydantic for models); strict TypeScript in frontend.
- **Naming**: snake_case (Python vars/functions), PascalCase (classes), camelCase (JS/TS).
- **Error Handling**: Fail fast for startup/config/auth/DB issues. For batch/background: log detailed errors, skip failed items, never store corrupted data. Preserve stack traces.
- **General**: Remove dead code immediately. Follow 75% confidence rule: don't confirm <75% sure; say "I don't know" and collaborate.
- **Cursor/Copilot Rules**: No specific .cursor/rules or .github/copilot-instructions.md found; adhere to CLAUDE.md guidelines.

(Note: No Cursor or Copilot specific rules detected in codebase.)
