"""
Deterministic validation checks for External Validator
"""

import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import ast
import re

from .models import ValidationIssue, ValidationEvidence, ValidationSeverity

logger = logging.getLogger(__name__)


class DeterministicChecker:
    """Performs deterministic validation checks"""
    
    def __init__(self):
        self.available_tools = self._check_available_tools()
    
    def _check_available_tools(self) -> dict:
        """Check which validation tools are available"""
        
        tools = {
            "pytest": False,
            "ruff": False,
            "mypy": False,
            "eslint": False,
            "semgrep": False
        }
        
        for tool in tools:
            try:
                result = subprocess.run(
                    [tool, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                tools[tool] = result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                tools[tool] = False
        
        logger.info(f"Available tools: {[k for k, v in tools.items() if v]}")
        return tools
    
    def verify_tools(self):
        """Verify that essential tools are available"""
        
        if not any(self.available_tools.values()):
            logger.warning("No deterministic validation tools available")
    
    def tools_available(self) -> bool:
        """Check if any tools are available"""
        
        return any(self.available_tools.values())
    
    async def check_file(
        self,
        file_path: Path
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check a specific file"""
        
        issues = []
        evidence = []
        
        if not file_path.exists():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="file",
                message=f"File not found: {file_path}",
                file_path=str(file_path)
            ))
            return issues, evidence
        
        # Determine file type and run appropriate checks
        if file_path.suffix == ".py":
            py_issues, py_evidence = await self._check_python_file(file_path)
            issues.extend(py_issues)
            evidence.extend(py_evidence)
            
        elif file_path.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            js_issues, js_evidence = await self._check_javascript_file(file_path)
            issues.extend(js_issues)
            evidence.extend(js_evidence)
        
        return issues, evidence
    
    async def check_code(
        self,
        code: str
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check code snippet"""
        
        issues = []
        evidence = []
        
        # Try to detect language
        if self._is_python_code(code):
            issues, evidence = await self._check_python_code(code)
        elif self._is_javascript_code(code):
            issues, evidence = await self._check_javascript_code(code)
        else:
            # Generic checks
            issues.extend(self._check_generic_patterns(code))
        
        return issues, evidence
    
    def _is_python_code(self, code: str) -> bool:
        """Detect if code is Python"""
        
        python_indicators = [
            "def ", "class ", "import ", "from ", "if __name__",
            "print(", "self.", "__init__", "async def", "await "
        ]
        return any(indicator in code for indicator in python_indicators)
    
    def _is_javascript_code(self, code: str) -> bool:
        """Detect if code is JavaScript/TypeScript"""
        
        js_indicators = [
            "function ", "const ", "let ", "var ", "=>",
            "console.log", "export ", "import ", "class ",
            "interface ", "type ", "async function"
        ]
        return any(indicator in code for indicator in js_indicators)
    
    async def _check_python_file(
        self,
        file_path: Path
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check Python file with available tools"""
        
        issues = []
        evidence = []
        
        # Run pytest if available and it's a test file
        if self.available_tools["pytest"] and "test" in file_path.name:
            test_issues = self._run_pytest(file_path)
            issues.extend(test_issues)
        
        # Run ruff if available
        if self.available_tools["ruff"]:
            ruff_issues = self._run_ruff(file_path)
            issues.extend(ruff_issues)
        
        # Run mypy if available
        if self.available_tools["mypy"]:
            mypy_issues = self._run_mypy(file_path)
            issues.extend(mypy_issues)
        
        # Add evidence for successful checks
        if not issues:
            evidence.append(ValidationEvidence(
                source="deterministic-python",
                content=f"Python file {file_path.name} passed all checks",
                confidence=1.0
            ))
        
        return issues, evidence
    
    async def _check_python_code(
        self,
        code: str
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check Python code snippet"""
        
        issues = []
        evidence = []
        
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="syntax",
                message=f"Python syntax error: {e.msg}",
                line_number=e.lineno,
                suggested_fix="Fix the syntax error"
            ))
        
        # Check for common anti-patterns
        anti_patterns = [
            (r"except:\s*pass", "Bare except with pass", "Handle exceptions properly"),
            (r"eval\(", "Use of eval()", "Avoid eval() for security"),
            (r"exec\(", "Use of exec()", "Avoid exec() for security"),
            (r"assert\s+True", "Meaningless assertion", "Write meaningful test assertions"),
            (r"return\s+[\"']mock", "Mock data return", "Implement real functionality"),
            (r"#\s*TODO", "TODO comment found", "Complete the implementation")
        ]
        
        for pattern, message, fix in anti_patterns:
            if re.search(pattern, code):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="anti-pattern",
                    message=message,
                    suggested_fix=fix
                ))
        
        # Check for gaming patterns (DGTS)
        gaming_patterns = [
            (r"#\s*validation.*=.*False", "Commented validation", ValidationSeverity.CRITICAL),
            (r"if\s+False:", "Unreachable code block", ValidationSeverity.ERROR),
            (r"assert\s+True", "Meaningless assertion (assert True)", ValidationSeverity.CRITICAL),
            (r"assert\s+1\s*==\s*1", "Tautological assertion", ValidationSeverity.ERROR),
            (r"def\s+\w+\([^)]*\):\s*pass", "Stub implementation", ValidationSeverity.WARNING),
            (r"return\s+['\"]mock", "Mock data return", ValidationSeverity.ERROR)
        ]
        
        for pattern, message, severity in gaming_patterns:
            if re.search(pattern, code):
                issues.append(ValidationIssue(
                    severity=severity,
                    category="gaming",
                    message=f"Gaming pattern detected: {message}",
                    suggested_fix="Implement real functionality, don't game the system"
                ))
        
        if not issues:
            evidence.append(ValidationEvidence(
                source="deterministic-python-snippet",
                content="Python code passed validation",
                confidence=0.9
            ))
        
        return issues, evidence
    
    async def _check_javascript_file(
        self,
        file_path: Path
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check JavaScript/TypeScript file"""
        
        issues = []
        evidence = []
        
        # Run eslint if available
        if self.available_tools["eslint"]:
            eslint_issues = self._run_eslint(file_path)
            issues.extend(eslint_issues)
        
        if not issues:
            evidence.append(ValidationEvidence(
                source="deterministic-javascript",
                content=f"JavaScript file {file_path.name} passed all checks",
                confidence=1.0
            ))
        
        return issues, evidence
    
    async def _check_javascript_code(
        self,
        code: str
    ) -> Tuple[List[ValidationIssue], List[ValidationEvidence]]:
        """Check JavaScript/TypeScript code snippet"""
        
        issues = []
        evidence = []
        
        # Check for common issues
        js_patterns = [
            (r"console\.log", "Console.log statement", "Remove console statements"),
            (r"debugger;", "Debugger statement", "Remove debugger statements"),
            (r"//\s*TODO", "TODO comment found", "Complete the implementation"),
            (r"return\s+[\"']mock", "Mock data return", "Implement real functionality")
        ]
        
        for pattern, message, fix in js_patterns:
            if re.search(pattern, code):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="code-quality",
                    message=message,
                    suggested_fix=fix
                ))
        
        if not issues:
            evidence.append(ValidationEvidence(
                source="deterministic-javascript-snippet",
                content="JavaScript code passed validation",
                confidence=0.9
            ))
        
        return issues, evidence
    
    def _check_generic_patterns(self, code: str) -> List[ValidationIssue]:
        """Check for generic code patterns"""
        
        issues = []
        
        # Check for placeholder content
        placeholders = [
            "lorem ipsum",
            "placeholder",
            "TODO",
            "FIXME",
            "XXX",
            "HACK"
        ]
        
        for placeholder in placeholders:
            if placeholder.lower() in code.lower():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="placeholder",
                    message=f"Placeholder content found: {placeholder}",
                    suggested_fix="Replace placeholder with real content"
                ))
        
        return issues
    
    def _run_pytest(self, file_path: Path) -> List[ValidationIssue]:
        """Run pytest on file"""
        
        issues = []
        
        try:
            result = subprocess.run(
                ["pytest", str(file_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Parse pytest output for failures
                for line in result.stdout.split("\n"):
                    if "FAILED" in line:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="test",
                            message=f"Test failure: {line}",
                            file_path=str(file_path),
                            suggested_fix="Fix the failing test"
                        ))
                        
        except subprocess.SubprocessError as e:
            logger.error(f"Pytest execution error: {e}")
        
        return issues
    
    def _run_ruff(self, file_path: Path) -> List[ValidationIssue]:
        """Run ruff linter on file"""
        
        issues = []
        
        try:
            result = subprocess.run(
                ["ruff", "check", str(file_path), "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.stdout:
                violations = json.loads(result.stdout)
                for violation in violations:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="lint",
                        message=f"{violation.get('code', '')}: {violation.get('message', '')}",
                        file_path=str(file_path),
                        line_number=violation.get("location", {}).get("row"),
                        suggested_fix=violation.get("fix", {}).get("message")
                    ))
                    
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.error(f"Ruff execution error: {e}")
        
        return issues
    
    def _run_mypy(self, file_path: Path) -> List[ValidationIssue]:
        """Run mypy type checker on file"""
        
        issues = []
        
        try:
            result = subprocess.run(
                ["mypy", str(file_path), "--no-error-summary"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                for line in result.stdout.split("\n"):
                    if "error:" in line:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="type",
                            message=line,
                            file_path=str(file_path),
                            suggested_fix="Fix the type error"
                        ))
                        
        except subprocess.SubprocessError as e:
            logger.error(f"Mypy execution error: {e}")
        
        return issues
    
    def _run_eslint(self, file_path: Path) -> List[ValidationIssue]:
        """Run eslint on file"""
        
        issues = []
        
        try:
            result = subprocess.run(
                ["eslint", str(file_path), "--format", "json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.stdout:
                reports = json.loads(result.stdout)
                for report in reports:
                    for message in report.get("messages", []):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING if message["severity"] == 1 else ValidationSeverity.ERROR,
                            category="lint",
                            message=f"{message.get('ruleId', '')}: {message.get('message', '')}",
                            file_path=str(file_path),
                            line_number=message.get("line"),
                            suggested_fix=message.get("fix", {}).get("text")
                        ))
                        
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.error(f"ESLint execution error: {e}")
        
        return issues