"""
Quality Metrics Module
Code quality measurement and technical debt tracking
"""

import asyncio
import ast
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
from collections import defaultdict

from .analytics_engine import AnalyticsEngine


class QualityMetricType(Enum):
    """Types of quality metrics"""
    CODE_COVERAGE = "code_coverage"
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    MAINTAINABILITY_INDEX = "maintainability_index"
    TECHNICAL_DEBT = "technical_debt"
    CODE_DUPLICATION = "code_duplication"
    DOCUMENTATION_COVERAGE = "documentation_coverage"
    TEST_QUALITY = "test_quality"
    SECURITY_SCORE = "security_score"
    PERFORMANCE_SCORE = "performance_score"
    DEPENDENCY_HEALTH = "dependency_health"


class CodeSmellType(Enum):
    """Types of code smells"""
    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    DUPLICATE_CODE = "duplicate_code"
    DEAD_CODE = "dead_code"
    COMPLEX_METHOD = "complex_method"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    SWITCH_STATEMENTS = "switch_statements"
    PARALLEL_INHERITANCE = "parallel_inheritance"
    LAZY_CLASS = "lazy_class"
    SPECULATIVE_GENERALITY = "speculative_generality"
    MESSAGE_CHAINS = "message_chains"
    MIDDLE_MAN = "middle_man"
    INAPPROPRIATE_INTIMACY = "inappropriate_intimacy"


class DebtCategory(Enum):
    """Categories of technical debt"""
    ARCHITECTURE = "architecture"
    BUILD = "build"
    CODE = "code"
    DEFECT = "defect"
    DESIGN = "design"
    DOCUMENTATION = "documentation"
    INFRASTRUCTURE = "infrastructure"
    PEOPLE = "people"
    PROCESS = "process"
    REQUIREMENT = "requirement"
    TEST = "test"
    TEST_AUTOMATION = "test_automation"


@dataclass
class QualityMetric:
    """Individual quality metric"""
    metric_id: str
    metric_type: QualityMetricType
    value: float
    threshold: float
    status: str  # good, warning, critical
    file_path: Optional[str] = None
    component: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeSmell:
    """Detected code smell"""
    smell_id: str
    smell_type: CodeSmellType
    severity: str  # low, medium, high, critical
    file_path: str
    line_start: int
    line_end: int
    description: str
    suggested_fix: str
    estimated_effort: int  # minutes
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnicalDebt:
    """Technical debt item"""
    debt_id: str
    category: DebtCategory
    principal: float  # effort to fix (hours)
    interest: float  # ongoing cost (hours/month)
    description: str
    file_paths: List[str]
    priority: int  # 1-5, 1 being highest
    created_at: datetime
    last_updated: datetime
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    timestamp: datetime
    overall_score: float  # 0-100
    metrics: Dict[QualityMetricType, QualityMetric]
    code_smells: List[CodeSmell]
    technical_debt: List[TechnicalDebt]
    total_debt_hours: float
    debt_ratio: float  # debt vs productive code
    trends: Dict[str, List[float]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityMetrics:
    """
    Comprehensive code quality analysis and tracking system
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics_engine = analytics_engine
        self.metrics: Dict[str, QualityMetric] = {}
        self.code_smells: List[CodeSmell] = []
        self.technical_debt: Dict[str, TechnicalDebt] = {}
        self.quality_history: List[QualityReport] = []
        
        # Thresholds for metrics
        self.thresholds = {
            QualityMetricType.CODE_COVERAGE: 80.0,
            QualityMetricType.CYCLOMATIC_COMPLEXITY: 10.0,
            QualityMetricType.MAINTAINABILITY_INDEX: 20.0,
            QualityMetricType.TECHNICAL_DEBT: 100.0,  # hours
            QualityMetricType.CODE_DUPLICATION: 5.0,  # percentage
            QualityMetricType.DOCUMENTATION_COVERAGE: 70.0,
            QualityMetricType.TEST_QUALITY: 75.0,
            QualityMetricType.SECURITY_SCORE: 80.0,
            QualityMetricType.PERFORMANCE_SCORE: 75.0,
            QualityMetricType.DEPENDENCY_HEALTH: 90.0
        }
        
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start quality monitoring background tasks"""
        asyncio.create_task(self._monitor_quality())
        asyncio.create_task(self._detect_code_smells())
        asyncio.create_task(self._calculate_technical_debt())
    
    async def _monitor_quality(self):
        """Monitor code quality continuously"""
        while True:
            try:
                await self.analyze_codebase()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                print(f"Quality monitoring error: {e}")
                await asyncio.sleep(3600)
    
    async def _detect_code_smells(self):
        """Detect code smells periodically"""
        while True:
            try:
                await self.scan_for_code_smells()
                await asyncio.sleep(7200)  # Check every 2 hours
            except Exception as e:
                print(f"Code smell detection error: {e}")
                await asyncio.sleep(7200)
    
    async def _calculate_technical_debt(self):
        """Calculate technical debt"""
        while True:
            try:
                await self.calculate_debt()
                await asyncio.sleep(86400)  # Daily calculation
            except Exception as e:
                print(f"Technical debt calculation error: {e}")
                await asyncio.sleep(86400)
    
    async def analyze_codebase(self, path: str = ".") -> QualityReport:
        """Analyze entire codebase quality"""
        import uuid
        
        report_id = str(uuid.uuid4())
        metrics = {}
        
        # Analyze various quality aspects
        metrics[QualityMetricType.CODE_COVERAGE] = await self._measure_code_coverage(path)
        metrics[QualityMetricType.CYCLOMATIC_COMPLEXITY] = await self._measure_complexity(path)
        metrics[QualityMetricType.MAINTAINABILITY_INDEX] = await self._calculate_maintainability(path)
        metrics[QualityMetricType.CODE_DUPLICATION] = await self._detect_duplication(path)
        metrics[QualityMetricType.DOCUMENTATION_COVERAGE] = await self._measure_documentation(path)
        metrics[QualityMetricType.TEST_QUALITY] = await self._assess_test_quality(path)
        metrics[QualityMetricType.SECURITY_SCORE] = await self._calculate_security_score(path)
        metrics[QualityMetricType.PERFORMANCE_SCORE] = await self._calculate_performance_score(path)
        metrics[QualityMetricType.DEPENDENCY_HEALTH] = await self._check_dependency_health(path)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics)
        
        # Calculate debt
        total_debt_hours = sum(debt.principal for debt in self.technical_debt.values())
        debt_ratio = self._calculate_debt_ratio(total_debt_hours, path)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, self.code_smells)
        
        # Create report
        report = QualityReport(
            report_id=report_id,
            timestamp=datetime.now(),
            overall_score=overall_score,
            metrics=metrics,
            code_smells=self.code_smells[-100:],  # Last 100 smells
            technical_debt=list(self.technical_debt.values()),
            total_debt_hours=total_debt_hours,
            debt_ratio=debt_ratio,
            trends=await self._calculate_quality_trends(),
            recommendations=recommendations
        )
        
        self.quality_history.append(report)
        
        # Record metrics
        for metric_type, metric in metrics.items():
            await self.analytics_engine.record_metric(
                f"quality.{metric_type.value}",
                metric.value,
                tags={"status": metric.status}
            )
        
        return report
    
    async def _measure_code_coverage(self, path: str) -> QualityMetric:
        """Measure code coverage"""
        import uuid
        
        try:
            # Run coverage tool
            result = subprocess.run(
                ["coverage", "report", "--format=json"],
                capture_output=True,
                text=True,
                cwd=path
            )
            
            if result.returncode == 0:
                import json
                coverage_data = json.loads(result.stdout)
                coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
            else:
                # Fallback: estimate coverage
                coverage_percent = 75.0  # Placeholder
            
        except Exception:
            coverage_percent = 75.0  # Default placeholder
        
        threshold = self.thresholds[QualityMetricType.CODE_COVERAGE]
        status = "good" if coverage_percent >= threshold else \
                "warning" if coverage_percent >= threshold * 0.8 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.CODE_COVERAGE,
            value=coverage_percent,
            threshold=threshold,
            status=status,
            details={"coverage_percent": coverage_percent}
        )
    
    async def _measure_complexity(self, path: str) -> QualityMetric:
        """Measure cyclomatic complexity"""
        import uuid
        
        total_complexity = 0
        file_count = 0
        complex_functions = []
        
        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        total_complexity += complexity
                        file_count += 1
                        
                        if complexity > 10:
                            complex_functions.append({
                                "file": str(py_file),
                                "function": node.name,
                                "complexity": complexity
                            })
            except Exception:
                continue
        
        avg_complexity = total_complexity / file_count if file_count > 0 else 0
        
        threshold = self.thresholds[QualityMetricType.CYCLOMATIC_COMPLEXITY]
        status = "good" if avg_complexity <= threshold else \
                "warning" if avg_complexity <= threshold * 1.5 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.CYCLOMATIC_COMPLEXITY,
            value=avg_complexity,
            threshold=threshold,
            status=status,
            details={
                "average_complexity": avg_complexity,
                "complex_functions": complex_functions[:10]
            }
        )
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    async def _calculate_maintainability(self, path: str) -> QualityMetric:
        """Calculate maintainability index"""
        import uuid
        import math
        
        total_loc = 0
        total_complexity = 0
        total_halstead = 0
        file_count = 0
        
        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    # Count lines of code (excluding blanks and comments)
                    loc = sum(1 for line in lines 
                            if line.strip() and not line.strip().startswith('#'))
                    total_loc += loc
                    
                    # Parse AST for complexity
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_complexity += self._calculate_cyclomatic_complexity(node)
                    
                    file_count += 1
                    
            except Exception:
                continue
        
        # Simplified Maintainability Index calculation
        # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * Cyclomatic Complexity - 16.2 * ln(LOC)
        if file_count > 0 and total_loc > 0:
            avg_complexity = total_complexity / file_count
            mi = 171 - 0.23 * avg_complexity - 16.2 * math.log(total_loc)
            mi = max(0, min(100, mi))  # Normalize to 0-100
        else:
            mi = 50.0
        
        threshold = self.thresholds[QualityMetricType.MAINTAINABILITY_INDEX]
        status = "good" if mi >= threshold else \
                "warning" if mi >= threshold * 0.8 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.MAINTAINABILITY_INDEX,
            value=mi,
            threshold=threshold,
            status=status,
            details={
                "maintainability_index": mi,
                "total_loc": total_loc,
                "average_complexity": total_complexity / file_count if file_count > 0 else 0
            }
        )
    
    async def _detect_duplication(self, path: str) -> QualityMetric:
        """Detect code duplication"""
        import uuid
        
        duplicates = []
        total_lines = 0
        duplicate_lines = 0
        
        # Collect all code snippets
        code_snippets: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                    # Create sliding window of code blocks
                    for i in range(len(lines) - 5):
                        block = ''.join(lines[i:i+6]).strip()
                        if len(block) > 100:  # Minimum block size
                            normalized = self._normalize_code(block)
                            code_snippets[normalized].append((str(py_file), i))
                            
            except Exception:
                continue
        
        # Find duplicates
        for block_hash, locations in code_snippets.items():
            if len(locations) > 1:
                duplicate_lines += 6 * (len(locations) - 1)
                duplicates.append({
                    "locations": locations[:5],  # First 5 locations
                    "count": len(locations)
                })
        
        duplication_percent = (duplicate_lines / total_lines * 100) if total_lines > 0 else 0
        
        threshold = self.thresholds[QualityMetricType.CODE_DUPLICATION]
        status = "good" if duplication_percent <= threshold else \
                "warning" if duplication_percent <= threshold * 2 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.CODE_DUPLICATION,
            value=duplication_percent,
            threshold=threshold,
            status=status,
            details={
                "duplication_percent": duplication_percent,
                "duplicate_blocks": len(duplicates),
                "top_duplicates": duplicates[:10]
            }
        )
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for duplication detection"""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        # Remove string literals
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)
        return code.strip()
    
    async def _measure_documentation(self, path: str) -> QualityMetric:
        """Measure documentation coverage"""
        import uuid
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            documented_classes += 1
                            
            except Exception:
                continue
        
        total_items = total_functions + total_classes
        documented_items = documented_functions + documented_classes
        
        doc_coverage = (documented_items / total_items * 100) if total_items > 0 else 0
        
        threshold = self.thresholds[QualityMetricType.DOCUMENTATION_COVERAGE]
        status = "good" if doc_coverage >= threshold else \
                "warning" if doc_coverage >= threshold * 0.8 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.DOCUMENTATION_COVERAGE,
            value=doc_coverage,
            threshold=threshold,
            status=status,
            details={
                "documentation_coverage": doc_coverage,
                "documented_functions": documented_functions,
                "total_functions": total_functions,
                "documented_classes": documented_classes,
                "total_classes": total_classes
            }
        )
    
    async def _assess_test_quality(self, path: str) -> QualityMetric:
        """Assess test quality"""
        import uuid
        
        test_files = 0
        test_functions = 0
        assertion_count = 0
        mock_usage = 0
        
        for py_file in Path(path).rglob("test_*.py"):
            test_files += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith('test_'):
                            test_functions += 1
                    elif isinstance(node, ast.Assert):
                        assertion_count += 1
                    elif isinstance(node, ast.Name) and 'mock' in node.id.lower():
                        mock_usage += 1
                        
            except Exception:
                continue
        
        # Calculate test quality score
        if test_functions > 0:
            avg_assertions = assertion_count / test_functions
            mock_ratio = mock_usage / test_functions
            test_quality = min(100, (avg_assertions * 10 + mock_ratio * 20 + test_files))
        else:
            test_quality = 0
        
        threshold = self.thresholds[QualityMetricType.TEST_QUALITY]
        status = "good" if test_quality >= threshold else \
                "warning" if test_quality >= threshold * 0.8 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.TEST_QUALITY,
            value=test_quality,
            threshold=threshold,
            status=status,
            details={
                "test_quality_score": test_quality,
                "test_files": test_files,
                "test_functions": test_functions,
                "total_assertions": assertion_count,
                "mock_usage": mock_usage
            }
        )
    
    async def _calculate_security_score(self, path: str) -> QualityMetric:
        """Calculate security score"""
        import uuid
        
        security_issues = []
        total_files = 0
        
        # Common security patterns to check
        security_patterns = [
            (r'eval\(', 'Use of eval() is dangerous'),
            (r'exec\(', 'Use of exec() is dangerous'),
            (r'pickle\.loads', 'Pickle deserialization can be unsafe'),
            (r'os\.system', 'Direct system calls can be unsafe'),
            (r'subprocess\..*shell=True', 'Shell injection vulnerability'),
            (r'password\s*=\s*["\']', 'Hardcoded password detected'),
            (r'api_key\s*=\s*["\']', 'Hardcoded API key detected'),
            (r'SECRET.*=\s*["\']', 'Hardcoded secret detected'),
        ]
        
        for py_file in Path(path).rglob("*.py"):
            total_files += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in security_patterns:
                    if re.search(pattern, content):
                        security_issues.append({
                            "file": str(py_file),
                            "issue": description
                        })
                        
            except Exception:
                continue
        
        # Calculate security score (100 - issues ratio)
        if total_files > 0:
            security_score = max(0, 100 - (len(security_issues) / total_files * 100))
        else:
            security_score = 100
        
        threshold = self.thresholds[QualityMetricType.SECURITY_SCORE]
        status = "good" if security_score >= threshold else \
                "warning" if security_score >= threshold * 0.8 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.SECURITY_SCORE,
            value=security_score,
            threshold=threshold,
            status=status,
            details={
                "security_score": security_score,
                "issues_found": len(security_issues),
                "issues": security_issues[:10]
            }
        )
    
    async def _calculate_performance_score(self, path: str) -> QualityMetric:
        """Calculate performance score"""
        import uuid
        
        performance_issues = []
        total_functions = 0
        
        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check for performance anti-patterns
                        for child in ast.walk(node):
                            # Nested loops
                            if isinstance(child, ast.For):
                                for grandchild in ast.walk(child):
                                    if isinstance(grandchild, ast.For) and grandchild != child:
                                        performance_issues.append({
                                            "file": str(py_file),
                                            "function": node.name,
                                            "issue": "Nested loops detected"
                                        })
                                        break
                                        
            except Exception:
                continue
        
        # Calculate performance score
        if total_functions > 0:
            performance_score = max(0, 100 - (len(performance_issues) / total_functions * 100))
        else:
            performance_score = 100
        
        threshold = self.thresholds[QualityMetricType.PERFORMANCE_SCORE]
        status = "good" if performance_score >= threshold else \
                "warning" if performance_score >= threshold * 0.8 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.PERFORMANCE_SCORE,
            value=performance_score,
            threshold=threshold,
            status=status,
            details={
                "performance_score": performance_score,
                "issues_found": len(performance_issues),
                "issues": performance_issues[:10]
            }
        )
    
    async def _check_dependency_health(self, path: str) -> QualityMetric:
        """Check dependency health"""
        import uuid
        
        try:
            # Check for outdated dependencies
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                cwd=path
            )
            
            if result.returncode == 0:
                import json
                outdated = json.loads(result.stdout)
                outdated_count = len(outdated)
                
                # Check total dependencies
                result = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                    cwd=path
                )
                
                if result.returncode == 0:
                    total = len(json.loads(result.stdout))
                    health_score = ((total - outdated_count) / total * 100) if total > 0 else 100
                else:
                    health_score = 75
            else:
                health_score = 75
                
        except Exception:
            health_score = 75
        
        threshold = self.thresholds[QualityMetricType.DEPENDENCY_HEALTH]
        status = "good" if health_score >= threshold else \
                "warning" if health_score >= threshold * 0.8 else "critical"
        
        return QualityMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=QualityMetricType.DEPENDENCY_HEALTH,
            value=health_score,
            threshold=threshold,
            status=status,
            details={"dependency_health": health_score}
        )
    
    async def scan_for_code_smells(self, path: str = ".") -> List[CodeSmell]:
        """Scan codebase for code smells"""
        import uuid
        
        smells = []
        
        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    tree = ast.parse(content)
                    
                # Check for long methods
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        
                        if method_lines > 50:
                            smells.append(CodeSmell(
                                smell_id=str(uuid.uuid4()),
                                smell_type=CodeSmellType.LONG_METHOD,
                                severity="high" if method_lines > 100 else "medium",
                                file_path=str(py_file),
                                line_start=node.lineno,
                                line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                                description=f"Method '{node.name}' is {method_lines} lines long",
                                suggested_fix="Break down into smaller methods",
                                estimated_effort=30
                            ))
                        
                        # Check for complex methods
                        complexity = self._calculate_cyclomatic_complexity(node)
                        if complexity > 10:
                            smells.append(CodeSmell(
                                smell_id=str(uuid.uuid4()),
                                smell_type=CodeSmellType.COMPLEX_METHOD,
                                severity="high" if complexity > 20 else "medium",
                                file_path=str(py_file),
                                line_start=node.lineno,
                                line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                                description=f"Method '{node.name}' has complexity {complexity}",
                                suggested_fix="Simplify logic or extract methods",
                                estimated_effort=45
                            ))
                    
                    # Check for large classes
                    elif isinstance(node, ast.ClassDef):
                        class_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        
                        if class_lines > 200:
                            smells.append(CodeSmell(
                                smell_id=str(uuid.uuid4()),
                                smell_type=CodeSmellType.LARGE_CLASS,
                                severity="high" if class_lines > 500 else "medium",
                                file_path=str(py_file),
                                line_start=node.lineno,
                                line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                                description=f"Class '{node.name}' is {class_lines} lines long",
                                suggested_fix="Consider splitting into multiple classes",
                                estimated_effort=120
                            ))
                            
            except Exception:
                continue
        
        self.code_smells.extend(smells)
        return smells
    
    async def calculate_debt(self, path: str = ".") -> float:
        """Calculate technical debt"""
        import uuid
        
        total_debt = 0.0
        
        # Calculate debt from code smells
        for smell in self.code_smells:
            debt_hours = smell.estimated_effort / 60
            
            debt_item = TechnicalDebt(
                debt_id=str(uuid.uuid4()),
                category=DebtCategory.CODE,
                principal=debt_hours,
                interest=debt_hours * 0.1,  # 10% monthly interest
                description=f"{smell.smell_type.value}: {smell.description}",
                file_paths=[smell.file_path],
                priority=2 if smell.severity == "critical" else 3 if smell.severity == "high" else 4,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.technical_debt[debt_item.debt_id] = debt_item
            total_debt += debt_hours
        
        # Calculate debt from low test coverage
        coverage_metric = await self._measure_code_coverage(path)
        if coverage_metric.value < 80:
            coverage_debt = (80 - coverage_metric.value) * 2  # 2 hours per percent
            
            debt_item = TechnicalDebt(
                debt_id=str(uuid.uuid4()),
                category=DebtCategory.TEST,
                principal=coverage_debt,
                interest=coverage_debt * 0.15,
                description=f"Low test coverage: {coverage_metric.value:.1f}%",
                file_paths=[],
                priority=2,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.technical_debt[debt_item.debt_id] = debt_item
            total_debt += coverage_debt
        
        return total_debt
    
    def _calculate_overall_score(self, metrics: Dict[QualityMetricType, QualityMetric]) -> float:
        """Calculate overall quality score"""
        weights = {
            QualityMetricType.CODE_COVERAGE: 0.20,
            QualityMetricType.CYCLOMATIC_COMPLEXITY: 0.15,
            QualityMetricType.MAINTAINABILITY_INDEX: 0.15,
            QualityMetricType.CODE_DUPLICATION: 0.10,
            QualityMetricType.DOCUMENTATION_COVERAGE: 0.10,
            QualityMetricType.TEST_QUALITY: 0.10,
            QualityMetricType.SECURITY_SCORE: 0.10,
            QualityMetricType.PERFORMANCE_SCORE: 0.05,
            QualityMetricType.DEPENDENCY_HEALTH: 0.05
        }
        
        score = 0.0
        for metric_type, metric in metrics.items():
            if metric_type in weights:
                # Normalize metric value to 0-100
                if metric_type == QualityMetricType.CYCLOMATIC_COMPLEXITY:
                    # Lower is better for complexity
                    normalized = max(0, 100 - metric.value * 5)
                elif metric_type == QualityMetricType.CODE_DUPLICATION:
                    # Lower is better for duplication
                    normalized = max(0, 100 - metric.value * 10)
                else:
                    normalized = min(100, metric.value)
                
                score += normalized * weights[metric_type]
        
        return score
    
    def _calculate_debt_ratio(self, debt_hours: float, path: str) -> float:
        """Calculate technical debt ratio"""
        # Estimate total codebase size
        total_loc = 0
        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_loc += len(f.readlines())
            except Exception:
                continue
        
        # Assume 10 LOC = 1 hour of development
        development_hours = total_loc / 10
        
        return (debt_hours / development_hours * 100) if development_hours > 0 else 0
    
    def _generate_recommendations(self, metrics: Dict[QualityMetricType, QualityMetric],
                                 smells: List[CodeSmell]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        for metric_type, metric in metrics.items():
            if metric.status == "critical":
                if metric_type == QualityMetricType.CODE_COVERAGE:
                    recommendations.append(f"Critical: Increase test coverage from {metric.value:.1f}% to at least {metric.threshold}%")
                elif metric_type == QualityMetricType.CYCLOMATIC_COMPLEXITY:
                    recommendations.append(f"Critical: Reduce average complexity from {metric.value:.1f} to below {metric.threshold}")
                elif metric_type == QualityMetricType.SECURITY_SCORE:
                    recommendations.append(f"Critical: Address security vulnerabilities to improve score from {metric.value:.1f}")
            elif metric.status == "warning":
                if metric_type == QualityMetricType.DOCUMENTATION_COVERAGE:
                    recommendations.append(f"Warning: Improve documentation coverage from {metric.value:.1f}% to {metric.threshold}%")
        
        # Add smell-based recommendations
        smell_counts = defaultdict(int)
        for smell in smells:
            smell_counts[smell.smell_type] += 1
        
        for smell_type, count in smell_counts.items():
            if count > 5:
                recommendations.append(f"Address {count} instances of {smell_type.value.replace('_', ' ')}")
        
        return recommendations[:10]  # Top 10 recommendations
    
    async def _calculate_quality_trends(self) -> Dict[str, List[float]]:
        """Calculate quality trends from history"""
        trends = {
            "overall_score": [],
            "coverage": [],
            "complexity": [],
            "debt": []
        }
        
        for report in self.quality_history[-30:]:  # Last 30 reports
            trends["overall_score"].append(report.overall_score)
            
            if QualityMetricType.CODE_COVERAGE in report.metrics:
                trends["coverage"].append(report.metrics[QualityMetricType.CODE_COVERAGE].value)
            
            if QualityMetricType.CYCLOMATIC_COMPLEXITY in report.metrics:
                trends["complexity"].append(report.metrics[QualityMetricType.CYCLOMATIC_COMPLEXITY].value)
            
            trends["debt"].append(report.total_debt_hours)
        
        return trends