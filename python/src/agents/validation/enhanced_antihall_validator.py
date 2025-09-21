"""
Enhanced Anti-Hallucination Validator for Archon
Prevents AI agents from referencing non-existent code, methods, or components
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Validation result types"""
    EXISTS = "exists"
    NOT_FOUND = "not_found"
    PARTIAL = "partial_match"
    DEPRECATED = "deprecated"
    UNCERTAIN = "uncertain"

@dataclass
class CodeReference:
    """Represents a code reference to validate"""
    reference_type: str  # 'function', 'class', 'method', 'variable', 'import', 'file'
    name: str
    context: Optional[str] = None  # e.g., class name for methods
    file_hint: Optional[str] = None  # Suggested file location
    line_number: Optional[int] = None

@dataclass
class ValidationReport:
    """Detailed validation report for a code reference"""
    reference: CodeReference
    result: ValidationResult
    actual_location: Optional[str] = None
    similar_matches: List[str] = None
    confidence: float = 0.0
    suggestion: Optional[str] = None
    evidence: List[str] = None

class EnhancedAntiHallValidator:
    """
    Advanced anti-hallucination system that validates all code references
    before allowing AI agents to suggest or implement them
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.code_index: Dict[str, Set[str]] = {}
        self.import_map: Dict[str, str] = {}
        self.class_hierarchy: Dict[str, Dict[str, Any]] = {}
        self.function_signatures: Dict[str, str] = {}
        self.deprecated_items: Set[str] = set()
        self._build_code_index()
        
    def _build_code_index(self):
        """Build comprehensive index of all code in the project"""
        logger.info(f"Building code index for {self.project_root}")
        
        for file_path in self.project_root.rglob("*.py"):
            if "venv" in file_path.parts or "__pycache__" in file_path.parts:
                continue
                
            try:
                self._index_python_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")
                
        for file_path in self.project_root.rglob("*.ts"):
            if "node_modules" in file_path.parts:
                continue
                
            try:
                self._index_typescript_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")
                
        logger.info(f"Code index built: {len(self.code_index)} files indexed")
        
    def _index_python_file(self, file_path: Path):
        """Index Python file contents"""
        relative_path = str(file_path.relative_to(self.project_root))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            
            # Index all definitions in this file
            definitions = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    definitions.add(node.name)
                    self.function_signatures[node.name] = self._get_function_signature(node)
                    
                elif isinstance(node, ast.ClassDef):
                    definitions.add(node.name)
                    class_methods = {}
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_methods[item.name] = self._get_function_signature(item)
                            
                    self.class_hierarchy[node.name] = {
                        'file': relative_path,
                        'methods': class_methods,
                        'line': node.lineno
                    }
                    
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.import_map[alias.name] = relative_path
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            full_name = f"{node.module}.{alias.name}"
                            self.import_map[full_name] = relative_path
                            
            self.code_index[relative_path] = definitions
            
            # Check for deprecation markers
            if "@deprecated" in content or "DEPRECATED" in content:
                for definition in definitions:
                    if f"@deprecated" in content or f"DEPRECATED.*{definition}" in content:
                        self.deprecated_items.add(definition)
                        
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            
    def _index_typescript_file(self, file_path: Path):
        """Index TypeScript file contents"""
        relative_path = str(file_path.relative_to(self.project_root))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        definitions = set()
        
        # Simple regex-based extraction for TypeScript
        # Function definitions
        functions = re.findall(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', content)
        definitions.update(functions)
        
        # Arrow functions
        arrow_funcs = re.findall(r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>', content)
        definitions.update(arrow_funcs)
        
        # Classes
        classes = re.findall(r'(?:export\s+)?class\s+(\w+)', content)
        definitions.update(classes)
        
        # Interfaces
        interfaces = re.findall(r'(?:export\s+)?interface\s+(\w+)', content)
        definitions.update(interfaces)
        
        # Types
        types = re.findall(r'(?:export\s+)?type\s+(\w+)', content)
        definitions.update(types)
        
        self.code_index[relative_path] = definitions
        
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node"""
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        return f"{node.name}({', '.join(params)})"
        
    def validate_reference(self, reference: CodeReference) -> ValidationReport:
        """
        Validate a single code reference
        Returns detailed report about whether the reference exists
        """
        logger.debug(f"Validating reference: {reference}")
        
        if reference.reference_type == 'file':
            return self._validate_file_reference(reference)
        elif reference.reference_type == 'class':
            return self._validate_class_reference(reference)
        elif reference.reference_type == 'method':
            return self._validate_method_reference(reference)
        elif reference.reference_type == 'function':
            return self._validate_function_reference(reference)
        elif reference.reference_type == 'import':
            return self._validate_import_reference(reference)
        else:
            return self._validate_generic_reference(reference)
            
    def _validate_file_reference(self, reference: CodeReference) -> ValidationReport:
        """Validate file path reference"""
        file_path = self.project_root / reference.name
        
        if file_path.exists():
            return ValidationReport(
                reference=reference,
                result=ValidationResult.EXISTS,
                actual_location=str(file_path.relative_to(self.project_root)),
                confidence=1.0,
                evidence=[f"File exists at {file_path}"]
            )
            
        # Try to find similar files
        similar = self._find_similar_files(reference.name)
        
        if similar:
            return ValidationReport(
                reference=reference,
                result=ValidationResult.NOT_FOUND,
                similar_matches=similar[:3],
                confidence=0.0,
                suggestion=f"Did you mean: {similar[0]}?",
                evidence=[f"File not found: {reference.name}"]
            )
            
        return ValidationReport(
            reference=reference,
            result=ValidationResult.NOT_FOUND,
            confidence=0.0,
            evidence=[f"File not found: {reference.name}"]
        )
        
    def _validate_class_reference(self, reference: CodeReference) -> ValidationReport:
        """Validate class reference"""
        if reference.name in self.class_hierarchy:
            class_info = self.class_hierarchy[reference.name]
            
            if reference.name in self.deprecated_items:
                return ValidationReport(
                    reference=reference,
                    result=ValidationResult.DEPRECATED,
                    actual_location=class_info['file'],
                    confidence=1.0,
                    suggestion="This class is marked as deprecated",
                    evidence=[f"Class found in {class_info['file']} (DEPRECATED)"]
                )
                
            return ValidationReport(
                reference=reference,
                result=ValidationResult.EXISTS,
                actual_location=f"{class_info['file']}:{class_info['line']}",
                confidence=1.0,
                evidence=[f"Class found in {class_info['file']}"]
            )
            
        # Search in all files
        for file_path, definitions in self.code_index.items():
            if reference.name in definitions:
                return ValidationReport(
                    reference=reference,
                    result=ValidationResult.EXISTS,
                    actual_location=file_path,
                    confidence=0.9,
                    evidence=[f"Found in {file_path}"]
                )
                
        # Find similar class names
        similar = self._find_similar_names(reference.name, self.class_hierarchy.keys())
        
        if similar:
            return ValidationReport(
                reference=reference,
                result=ValidationResult.NOT_FOUND,
                similar_matches=similar[:3],
                confidence=0.0,
                suggestion=f"Did you mean: {similar[0]}?",
                evidence=[f"Class not found: {reference.name}"]
            )
            
        return ValidationReport(
            reference=reference,
            result=ValidationResult.NOT_FOUND,
            confidence=0.0,
            evidence=[f"Class not found: {reference.name}"]
        )
        
    def _validate_method_reference(self, reference: CodeReference) -> ValidationReport:
        """Validate method reference (requires class context)"""
        if not reference.context:
            # Try to find method in any class
            found_in_classes = []
            
            for class_name, class_info in self.class_hierarchy.items():
                if reference.name in class_info['methods']:
                    found_in_classes.append(class_name)
                    
            if found_in_classes:
                return ValidationReport(
                    reference=reference,
                    result=ValidationResult.PARTIAL,
                    similar_matches=found_in_classes,
                    confidence=0.7,
                    suggestion=f"Method found in classes: {', '.join(found_in_classes)}",
                    evidence=[f"Method {reference.name} exists but class context missing"]
                )
                
        else:
            # Validate with class context
            if reference.context in self.class_hierarchy:
                class_info = self.class_hierarchy[reference.context]
                
                if reference.name in class_info['methods']:
                    return ValidationReport(
                        reference=reference,
                        result=ValidationResult.EXISTS,
                        actual_location=f"{class_info['file']}",
                        confidence=1.0,
                        evidence=[f"Method {reference.context}.{reference.name} found"]
                    )
                    
                # Find similar method names in this class
                similar = self._find_similar_names(reference.name, class_info['methods'].keys())
                
                if similar:
                    return ValidationReport(
                        reference=reference,
                        result=ValidationResult.NOT_FOUND,
                        similar_matches=similar[:3],
                        confidence=0.0,
                        suggestion=f"Did you mean: {reference.context}.{similar[0]}?",
                        evidence=[f"Method not found in class {reference.context}"]
                    )
                    
        return ValidationReport(
            reference=reference,
            result=ValidationResult.NOT_FOUND,
            confidence=0.0,
            evidence=[f"Method {reference.name} not found"]
        )
        
    def _validate_function_reference(self, reference: CodeReference) -> ValidationReport:
        """Validate function reference"""
        if reference.name in self.function_signatures:
            # Find which file contains this function
            for file_path, definitions in self.code_index.items():
                if reference.name in definitions:
                    
                    if reference.name in self.deprecated_items:
                        return ValidationReport(
                            reference=reference,
                            result=ValidationResult.DEPRECATED,
                            actual_location=file_path,
                            confidence=1.0,
                            suggestion="This function is marked as deprecated",
                            evidence=[f"Function found in {file_path} (DEPRECATED)"]
                        )
                        
                    return ValidationReport(
                        reference=reference,
                        result=ValidationResult.EXISTS,
                        actual_location=file_path,
                        confidence=1.0,
                        evidence=[f"Function {self.function_signatures[reference.name]} found"]
                    )
                    
        # Search in all files
        for file_path, definitions in self.code_index.items():
            if reference.name in definitions:
                return ValidationReport(
                    reference=reference,
                    result=ValidationResult.EXISTS,
                    actual_location=file_path,
                    confidence=0.9,
                    evidence=[f"Found in {file_path}"]
                )
                
        # Find similar function names
        similar = self._find_similar_names(reference.name, self.function_signatures.keys())
        
        if similar:
            return ValidationReport(
                reference=reference,
                result=ValidationResult.NOT_FOUND,
                similar_matches=similar[:3],
                confidence=0.0,
                suggestion=f"Did you mean: {similar[0]}?",
                evidence=[f"Function not found: {reference.name}"]
            )
            
        return ValidationReport(
            reference=reference,
            result=ValidationResult.NOT_FOUND,
            confidence=0.0,
            evidence=[f"Function not found: {reference.name}"]
        )
        
    def _validate_import_reference(self, reference: CodeReference) -> ValidationReport:
        """Validate import reference"""
        if reference.name in self.import_map:
            return ValidationReport(
                reference=reference,
                result=ValidationResult.EXISTS,
                actual_location=self.import_map[reference.name],
                confidence=1.0,
                evidence=[f"Import found in {self.import_map[reference.name]}"]
            )
            
        # Check if it's a standard library import
        standard_libs = {'os', 'sys', 'json', 'datetime', 'pathlib', 'typing', 're', 'ast'}
        if reference.name in standard_libs:
            return ValidationReport(
                reference=reference,
                result=ValidationResult.EXISTS,
                confidence=1.0,
                evidence=[f"Standard library import: {reference.name}"]
            )
            
        return ValidationReport(
            reference=reference,
            result=ValidationResult.NOT_FOUND,
            confidence=0.0,
            evidence=[f"Import not found: {reference.name}"]
        )
        
    def _validate_generic_reference(self, reference: CodeReference) -> ValidationReport:
        """Generic validation for any code reference"""
        # Search across all indexed code
        found_locations = []
        
        for file_path, definitions in self.code_index.items():
            if reference.name in definitions:
                found_locations.append(file_path)
                
        if found_locations:
            return ValidationReport(
                reference=reference,
                result=ValidationResult.EXISTS,
                actual_location=found_locations[0],
                similar_matches=found_locations[1:] if len(found_locations) > 1 else None,
                confidence=0.8,
                evidence=[f"Found in {len(found_locations)} file(s)"]
            )
            
        return ValidationReport(
            reference=reference,
            result=ValidationResult.NOT_FOUND,
            confidence=0.0,
            evidence=[f"Reference not found: {reference.name}"]
        )
        
    def _find_similar_files(self, target: str) -> List[str]:
        """Find files with similar names"""
        from difflib import SequenceMatcher
        
        all_files = []
        for file_path in self.code_index.keys():
            similarity = SequenceMatcher(None, target, file_path).ratio()
            if similarity > 0.6:
                all_files.append((file_path, similarity))
                
        all_files.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in all_files]
        
    def _find_similar_names(self, target: str, candidates: set) -> List[str]:
        """Find similar names using edit distance"""
        from difflib import SequenceMatcher
        
        similar = []
        for candidate in candidates:
            similarity = SequenceMatcher(None, target.lower(), candidate.lower()).ratio()
            if similarity > 0.7:
                similar.append((candidate, similarity))
                
        similar.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similar]
        
    def validate_code_snippet(self, code: str, language: str = 'python') -> List[ValidationReport]:
        """
        Validate all references in a code snippet
        Returns list of validation reports for each reference found
        """
        reports = []
        
        if language == 'python':
            reports.extend(self._validate_python_snippet(code))
        elif language in ['typescript', 'javascript']:
            reports.extend(self._validate_typescript_snippet(code))
            
        return reports
        
    def _validate_python_snippet(self, code: str) -> List[ValidationReport]:
        """Extract and validate references from Python code"""
        reports = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Function call
                    if isinstance(node.func, ast.Name):
                        ref = CodeReference('function', node.func.id)
                        reports.append(self.validate_reference(ref))
                        
                    elif isinstance(node.func, ast.Attribute):
                        # Method call
                        if isinstance(node.func.value, ast.Name):
                            ref = CodeReference(
                                'method',
                                node.func.attr,
                                context=node.func.value.id
                            )
                            reports.append(self.validate_reference(ref))
                            
                elif isinstance(node, ast.Name):
                    # Variable or class reference
                    ref = CodeReference('generic', node.id)
                    reports.append(self.validate_reference(ref))
                    
        except SyntaxError:
            # If code doesn't parse, extract what we can with regex
            # Function calls
            func_calls = re.findall(r'(\w+)\s*\(', code)
            for func in func_calls:
                ref = CodeReference('function', func)
                reports.append(self.validate_reference(ref))
                
            # Method calls
            method_calls = re.findall(r'(\w+)\.(\w+)\s*\(', code)
            for obj, method in method_calls:
                ref = CodeReference('method', method, context=obj)
                reports.append(self.validate_reference(ref))
                
        return reports
        
    def _validate_typescript_snippet(self, code: str) -> List[ValidationReport]:
        """Extract and validate references from TypeScript code"""
        reports = []
        
        # Function calls
        func_calls = re.findall(r'(\w+)\s*\(', code)
        for func in func_calls:
            ref = CodeReference('function', func)
            reports.append(self.validate_reference(ref))
            
        # Method calls
        method_calls = re.findall(r'(\w+)\.(\w+)\s*\(', code)
        for obj, method in method_calls:
            ref = CodeReference('method', method, context=obj)
            reports.append(self.validate_reference(ref))
            
        # Class instantiation
        new_calls = re.findall(r'new\s+(\w+)', code)
        for class_name in new_calls:
            ref = CodeReference('class', class_name)
            reports.append(self.validate_reference(ref))
            
        # Import statements
        imports = re.findall(r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]', code)
        for import_path in imports:
            ref = CodeReference('import', import_path)
            reports.append(self.validate_reference(ref))
            
        return reports
        
    def generate_validation_summary(self, reports: List[ValidationReport]) -> Dict[str, Any]:
        """Generate summary of validation results"""
        summary = {
            'total_references': len(reports),
            'valid_references': 0,
            'invalid_references': 0,
            'deprecated_references': 0,
            'uncertain_references': 0,
            'suggestions': [],
            'critical_errors': [],
            'warnings': []
        }
        
        for report in reports:
            if report.result == ValidationResult.EXISTS:
                summary['valid_references'] += 1
            elif report.result == ValidationResult.NOT_FOUND:
                summary['invalid_references'] += 1
                summary['critical_errors'].append(
                    f"❌ {report.reference.name} does not exist"
                )
                if report.suggestion:
                    summary['suggestions'].append(report.suggestion)
                    
            elif report.result == ValidationResult.DEPRECATED:
                summary['deprecated_references'] += 1
                summary['warnings'].append(
                    f"⚠️ {report.reference.name} is deprecated"
                )
                
            elif report.result in [ValidationResult.PARTIAL, ValidationResult.UNCERTAIN]:
                summary['uncertain_references'] += 1
                summary['warnings'].append(
                    f"❓ {report.reference.name} validation uncertain"
                )
                
        summary['validation_score'] = (
            summary['valid_references'] / summary['total_references']
            if summary['total_references'] > 0 else 0.0
        )
        
        summary['safe_to_proceed'] = (
            summary['invalid_references'] == 0 and
            summary['validation_score'] >= 0.95
        )
        
        return summary
        
    def enforce_validation(self, code: str, language: str = 'python', min_confidence: float = 0.75) -> Tuple[bool, Dict[str, Any]]:
        """
        Enforce validation - returns whether code is safe to use
        Requires minimum 75% confidence by default
        """
        reports = self.validate_code_snippet(code, language)
        summary = self.generate_validation_summary(reports)
        
        # Calculate average confidence
        total_confidence = sum(r.confidence for r in reports)
        avg_confidence = total_confidence / len(reports) if reports else 0.0
        summary['average_confidence'] = avg_confidence
        
        # Check if we meet minimum confidence threshold
        if avg_confidence < min_confidence:
            summary['safe_to_proceed'] = False
            summary['confidence_too_low'] = True
            summary['critical_errors'].append(
                f"⚠️ Confidence too low ({avg_confidence:.1%}) - minimum required: {min_confidence:.1%}"
            )
            logger.warning(f"Validation confidence too low: {avg_confidence:.1%} < {min_confidence:.1%}")
        
        if not summary['safe_to_proceed']:
            logger.error(f"Validation failed: {summary['critical_errors']}")
            
        return summary['safe_to_proceed'], summary


class RealTimeValidator:
    """
    Real-time validation that runs as code is being generated
    """
    
    def __init__(self, validator: EnhancedAntiHallValidator):
        self.validator = validator
        self.validation_cache = {}
        
    def validate_line(self, line: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Validate a single line of code in real-time
        Returns error message if invalid, None if valid
        """
        # Quick checks for common hallucination patterns
        if self._contains_suspicious_pattern(line):
            return self._validate_suspicious_line(line, context)
            
        return None
        
    def _contains_suspicious_pattern(self, line: str) -> bool:
        """Check for patterns that often indicate hallucination"""
        suspicious_patterns = [
            r'\.do_something\(',  # Generic method names
            r'\.process_data\(',   # Too generic
            r'\.handle_.*\(',      # Generic handlers
            r'TODO:',              # Incomplete implementation
            r'pass\s*#.*implement', # Stub implementation
            r'raise NotImplementedError',  # Unimplemented
            r'your_.*_here',       # Placeholder text
            r'example_',           # Example code
            r'dummy_',             # Dummy implementations
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
                
        return False
        
    def _validate_suspicious_line(self, line: str, context: Dict[str, Any]) -> Optional[str]:
        """Validate a suspicious line of code"""
        # Extract potential references
        references = []
        
        # Method calls
        methods = re.findall(r'(\w+)\.(\w+)\s*\(', line)
        for obj, method in methods:
            ref = CodeReference('method', method, context=obj)
            references.append(ref)
            
        # Function calls
        functions = re.findall(r'(\w+)\s*\(', line)
        for func in functions:
            if func not in ['print', 'len', 'str', 'int', 'float']:  # Skip builtins
                ref = CodeReference('function', func)
                references.append(ref)
                
        # Validate all references
        for ref in references:
            report = self.validator.validate_reference(ref)
            
            if report.result == ValidationResult.NOT_FOUND:
                return f"Reference not found: {ref.name}"
                
            elif report.result == ValidationResult.DEPRECATED:
                return f"Deprecated reference: {ref.name}"
                
        return None


# Integration with Archon agents
class AgentValidationWrapper:
    """
    Wraps AI agents to enforce validation before code generation
    """
    
    def __init__(self, agent, validator: EnhancedAntiHallValidator):
        self.agent = agent
        self.validator = validator
        self.real_time_validator = RealTimeValidator(validator)
        
    async def generate_code(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code with validation enforcement
        """
        # First, analyze the prompt for references
        references_in_prompt = self._extract_references_from_prompt(prompt)
        
        # Validate all references before generation
        invalid_refs = []
        suggestions = []
        
        for ref in references_in_prompt:
            report = self.validator.validate_reference(ref)
            
            if report.result == ValidationResult.NOT_FOUND:
                invalid_refs.append(ref.name)
                if report.suggestion:
                    suggestions.append(report.suggestion)
                    
        if invalid_refs:
            return {
                'success': False,
                'error': f"Cannot proceed - invalid references: {', '.join(invalid_refs)}",
                'suggestions': suggestions,
                'code': None
            }
            
        # Generate code with the agent
        result = await self.agent.generate_code(prompt, context)
        
        if 'code' in result:
            # Validate generated code
            is_valid, validation_summary = self.validator.enforce_validation(
                result['code'],
                context.get('language', 'python')
            )
            
            if not is_valid:
                # Try to fix hallucinations
                fixed_code = self._attempt_fix(result['code'], validation_summary)
                
                if fixed_code:
                    result['code'] = fixed_code
                    result['validation_fixes_applied'] = True
                else:
                    result['validation_failed'] = True
                    result['validation_errors'] = validation_summary['critical_errors']
                    
            result['validation_summary'] = validation_summary
            
        return result
        
    def _extract_references_from_prompt(self, prompt: str) -> List[CodeReference]:
        """Extract code references from natural language prompt"""
        references = []
        
        # Look for code-like references in backticks
        code_refs = re.findall(r'`([^`]+)`', prompt)
        for ref in code_refs:
            if '.' in ref:
                parts = ref.split('.')
                if len(parts) == 2:
                    references.append(CodeReference('method', parts[1], context=parts[0]))
            else:
                references.append(CodeReference('generic', ref))
                
        # Look for "use the X" or "call the Y" patterns
        use_patterns = re.findall(r'(?:use|call|invoke|reference)\s+(?:the\s+)?(\w+)', prompt, re.IGNORECASE)
        for pattern in use_patterns:
            references.append(CodeReference('generic', pattern))
            
        return references
        
    def _attempt_fix(self, code: str, validation_summary: Dict[str, Any]) -> Optional[str]:
        """Attempt to fix hallucinated references"""
        fixed_code = code
        
        # Try to replace invalid references with suggestions
        for i, error in enumerate(validation_summary['critical_errors']):
            if i < len(validation_summary['suggestions']):
                suggestion = validation_summary['suggestions'][i]
                
                # Extract the invalid reference and suggested replacement
                match = re.search(r'(\w+) does not exist', error)
                if match:
                    invalid_ref = match.group(1)
                    
                    # Extract suggested name
                    suggest_match = re.search(r'Did you mean: (\w+)', suggestion)
                    if suggest_match:
                        suggested_ref = suggest_match.group(1)
                        
                        # Replace in code
                        fixed_code = re.sub(
                            r'\b' + re.escape(invalid_ref) + r'\b',
                            suggested_ref,
                            fixed_code
                        )
                        
        # Re-validate fixed code
        is_valid, _ = self.validator.enforce_validation(fixed_code)
        
        if is_valid:
            return fixed_code
            
        return None