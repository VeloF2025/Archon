"""
Pattern Detection Engine using ML and AST analysis
Identifies recurring code patterns and anti-patterns
"""

import ast
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class CodePattern(BaseModel):
    """Represents a detected code pattern"""
    id: str = Field(description="Unique pattern identifier")
    name: str = Field(description="Pattern name")
    category: str = Field(description="Pattern category (e.g., structural, behavioral)")
    language: str = Field(description="Programming language")
    signature: str = Field(description="Pattern signature hash")
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    frequency: int = Field(default=1)
    confidence: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    detected_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    is_antipattern: bool = Field(default=False)
    performance_impact: Optional[str] = None
    suggested_alternative: Optional[str] = None


class PatternDetector:
    """ML-based pattern detection engine"""
    
    def __init__(self):
        self.patterns_cache = {}
        self.antipatterns = self._load_antipatterns()
        self.pattern_threshold = 0.7  # Confidence threshold
        
    def _load_antipatterns(self) -> Dict[str, Dict]:
        """Load known anti-patterns database"""
        return {
            # Python anti-patterns
            "mutable_default_argument": {
                "description": "Using mutable default arguments in functions",
                "languages": ["python"],
                "severity": "high",
                "alternative": "Use None as default and create new instance in function"
            },
            "catch_all_exception": {
                "description": "Catching all exceptions without specific handling",
                "languages": ["python", "javascript", "typescript"],
                "severity": "high",
                "alternative": "Catch specific exceptions and handle appropriately"
            },
            "god_function": {
                "description": "Function with too many responsibilities (>100 lines)",
                "languages": ["all"],
                "severity": "medium",
                "alternative": "Split into smaller, focused functions"
            },
            "nested_loops_deep": {
                "description": "Deeply nested loops (>3 levels)",
                "languages": ["all"],
                "severity": "medium",
                "alternative": "Extract inner loops into separate functions"
            },
            "console_log_production": {
                "description": "Console.log statements in production code",
                "languages": ["javascript", "typescript"],
                "severity": "medium",
                "alternative": "Use proper logging service with levels"
            },
            "synchronous_io_blocking": {
                "description": "Blocking I/O operations without async",
                "languages": ["python", "javascript", "typescript"],
                "severity": "high",
                "alternative": "Use async/await for I/O operations"
            }
        }
    
    async def detect_patterns(self, code: str, language: str = "python") -> List[CodePattern]:
        """
        Detect patterns in provided code
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        try:
            if language == "python":
                patterns.extend(await self._detect_python_patterns(code))
            elif language in ["javascript", "typescript"]:
                patterns.extend(await self._detect_js_patterns(code))
            
            # Detect anti-patterns
            antipatterns = await self._detect_antipatterns(code, language)
            patterns.extend(antipatterns)
            
            # Calculate pattern confidence scores
            for pattern in patterns:
                pattern.confidence = self._calculate_confidence(pattern)
            
            # Filter by confidence threshold
            patterns = [p for p in patterns if p.confidence >= self.pattern_threshold]
            
            logger.info(f"Detected {len(patterns)} patterns in {language} code")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    async def _detect_python_patterns(self, code: str) -> List[CodePattern]:
        """Detect Python-specific patterns using AST analysis"""
        patterns = []
        
        try:
            tree = ast.parse(code)
            
            # Detect decorator patterns
            decorators = self._find_decorators(tree)
            if decorators:
                patterns.append(CodePattern(
                    id=self._generate_pattern_id("decorator_pattern"),
                    name="Decorator Pattern",
                    category="structural",
                    language="python",
                    signature=self._generate_signature(str(decorators)),
                    examples=[{"decorators": decorators}],
                    frequency=len(decorators),
                    metadata={"decorator_types": list(set(decorators))}
                ))
            
            # Detect context manager patterns
            context_managers = self._find_context_managers(tree)
            if context_managers:
                patterns.append(CodePattern(
                    id=self._generate_pattern_id("context_manager"),
                    name="Context Manager Pattern",
                    category="behavioral",
                    language="python",
                    signature=self._generate_signature(str(context_managers)),
                    examples=context_managers[:3],
                    frequency=len(context_managers)
                ))
            
            # Detect factory patterns
            factories = self._find_factory_patterns(tree)
            if factories:
                patterns.append(CodePattern(
                    id=self._generate_pattern_id("factory_pattern"),
                    name="Factory Pattern",
                    category="creational",
                    language="python",
                    signature=self._generate_signature(str(factories)),
                    examples=factories[:3],
                    frequency=len(factories)
                ))
            
            # Detect singleton patterns
            singletons = self._find_singleton_patterns(tree)
            if singletons:
                patterns.append(CodePattern(
                    id=self._generate_pattern_id("singleton_pattern"),
                    name="Singleton Pattern",
                    category="creational",
                    language="python",
                    signature=self._generate_signature(str(singletons)),
                    examples=singletons[:3],
                    frequency=len(singletons)
                ))
                
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing Python code: {e}")
        
        return patterns
    
    async def _detect_js_patterns(self, code: str) -> List[CodePattern]:
        """Detect JavaScript/TypeScript patterns"""
        patterns = []
        
        # Simple regex-based detection for JS patterns
        # In production, use a proper JS parser like Babel
        
        # Detect Promise patterns
        if "new Promise" in code or ".then(" in code or "async " in code:
            patterns.append(CodePattern(
                id=self._generate_pattern_id("promise_pattern"),
                name="Promise/Async Pattern",
                category="behavioral",
                language="javascript",
                signature=self._generate_signature("promise"),
                frequency=code.count("Promise") + code.count("async")
            ))
        
        # Detect module patterns
        if "export " in code or "import " in code:
            patterns.append(CodePattern(
                id=self._generate_pattern_id("module_pattern"),
                name="ES6 Module Pattern",
                category="structural",
                language="javascript",
                signature=self._generate_signature("module"),
                frequency=code.count("export") + code.count("import")
            ))
        
        # Detect React patterns
        if "useState" in code or "useEffect" in code:
            patterns.append(CodePattern(
                id=self._generate_pattern_id("react_hooks"),
                name="React Hooks Pattern",
                category="behavioral",
                language="javascript",
                signature=self._generate_signature("react_hooks"),
                frequency=code.count("use")
            ))
        
        return patterns
    
    async def _detect_antipatterns(self, code: str, language: str) -> List[CodePattern]:
        """Detect anti-patterns in code"""
        antipatterns = []
        
        for pattern_key, pattern_info in self.antipatterns.items():
            if language not in pattern_info["languages"] and "all" not in pattern_info["languages"]:
                continue
            
            detected = False
            
            # Pattern-specific detection logic
            if pattern_key == "mutable_default_argument" and language == "python":
                detected = self._detect_mutable_defaults(code)
            elif pattern_key == "catch_all_exception":
                detected = "except:" in code or "catch (e)" in code or "catch {" in code
            elif pattern_key == "god_function":
                detected = self._detect_god_function(code)
            elif pattern_key == "nested_loops_deep":
                detected = self._detect_deep_nesting(code)
            elif pattern_key == "console_log_production":
                detected = "console.log" in code or "console.error" in code
            elif pattern_key == "synchronous_io_blocking":
                detected = self._detect_blocking_io(code, language)
            
            if detected:
                antipatterns.append(CodePattern(
                    id=self._generate_pattern_id(f"anti_{pattern_key}"),
                    name=f"Anti-pattern: {pattern_key.replace('_', ' ').title()}",
                    category="antipattern",
                    language=language,
                    signature=self._generate_signature(pattern_key),
                    is_antipattern=True,
                    performance_impact=pattern_info["severity"],
                    suggested_alternative=pattern_info["alternative"],
                    metadata={"description": pattern_info["description"]}
                ))
        
        return antipatterns
    
    def _find_decorators(self, tree: ast.AST) -> List[str]:
        """Find decorator usage in Python AST"""
        decorators = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        decorators.append(decorator.func.id)
        return decorators
    
    def _find_context_managers(self, tree: ast.AST) -> List[Dict]:
        """Find context manager usage (with statements)"""
        contexts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    context = {"type": "context_manager"}
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            context["name"] = item.context_expr.func.id
                    contexts.append(context)
        return contexts
    
    def _find_factory_patterns(self, tree: ast.AST) -> List[Dict]:
        """Detect factory pattern implementations"""
        factories = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function creates and returns objects
                if "create" in node.name.lower() or "factory" in node.name.lower():
                    factories.append({
                        "name": node.name,
                        "type": "factory_function"
                    })
        return factories
    
    def _find_singleton_patterns(self, tree: ast.AST) -> List[Dict]:
        """Detect singleton pattern implementations"""
        singletons = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for singleton indicators
                has_instance = False
                has_new = False
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__new__":
                            has_new = True
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and "_instance" in target.id:
                                has_instance = True
                
                if has_instance or has_new:
                    singletons.append({
                        "name": node.name,
                        "type": "singleton_class"
                    })
        return singletons
    
    def _detect_mutable_defaults(self, code: str) -> bool:
        """Detect mutable default arguments in Python"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            return True
        except:
            pass
        return False
    
    def _detect_god_function(self, code: str) -> bool:
        """Detect overly complex functions"""
        lines = code.split('\n')
        in_function = False
        function_lines = 0
        
        for line in lines:
            if 'def ' in line or 'function ' in line or 'const ' in line:
                if in_function and function_lines > 100:
                    return True
                in_function = True
                function_lines = 0
            elif in_function:
                function_lines += 1
        
        return function_lines > 100
    
    def _detect_deep_nesting(self, code: str) -> bool:
        """Detect deeply nested code blocks"""
        max_indent = 0
        for line in code.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        # Assuming 4 spaces per indent level
        return max_indent > 12  # More than 3 levels of nesting
    
    def _detect_blocking_io(self, code: str, language: str) -> bool:
        """Detect blocking I/O operations"""
        if language == "python":
            blocking_patterns = ["open(", "requests.get", "requests.post", "time.sleep"]
            async_patterns = ["async def", "await ", "aiohttp", "asyncio"]
        else:  # JavaScript/TypeScript
            blocking_patterns = ["fs.readFileSync", "fs.writeFileSync", "XMLHttpRequest"]
            async_patterns = ["async ", "await ", "Promise", "fetch"]
        
        has_blocking = any(pattern in code for pattern in blocking_patterns)
        has_async = any(pattern in code for pattern in async_patterns)
        
        return has_blocking and not has_async
    
    def _calculate_confidence(self, pattern: CodePattern) -> float:
        """Calculate confidence score for detected pattern"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on frequency
        if pattern.frequency > 5:
            confidence += 0.2
        elif pattern.frequency > 2:
            confidence += 0.1
        
        # Anti-patterns have high confidence
        if pattern.is_antipattern:
            confidence = 0.9
        
        # Well-known patterns have higher confidence
        known_patterns = ["decorator", "context_manager", "factory", "singleton", "promise", "module"]
        if any(p in pattern.name.lower() for p in known_patterns):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _generate_pattern_id(self, pattern_type: str) -> str:
        """Generate unique pattern ID"""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(f"{pattern_type}_{timestamp}".encode()).hexdigest()[:12]
    
    def _generate_signature(self, content: str) -> str:
        """Generate pattern signature hash"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def learn_from_feedback(self, pattern_id: str, useful: bool):
        """Update pattern detection based on user feedback"""
        if pattern_id in self.patterns_cache:
            pattern = self.patterns_cache[pattern_id]
            if useful:
                pattern.confidence = min(pattern.confidence * 1.1, 1.0)
            else:
                pattern.confidence = max(pattern.confidence * 0.9, 0.1)
            
            # Adjust threshold if needed
            if pattern.confidence < 0.5 and not pattern.is_antipattern:
                logger.info(f"Pattern {pattern_id} confidence dropped below threshold")
    
    async def export_patterns(self) -> Dict[str, Any]:
        """Export detected patterns for persistence"""
        return {
            "patterns": [p.dict() for p in self.patterns_cache.values()],
            "antipatterns": self.antipatterns,
            "threshold": self.pattern_threshold,
            "exported_at": datetime.utcnow().isoformat()
        }