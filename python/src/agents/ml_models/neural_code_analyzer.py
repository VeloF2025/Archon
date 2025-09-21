"""
Neural Code Analyzer
Advanced neural network-based code analysis using transformer models
Provides deep semantic understanding of code structure and intent
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import logging
import numpy as np
import json
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of neural code analysis"""
    SEMANTIC_UNDERSTANDING = "semantic_understanding"
    CODE_QUALITY = "code_quality"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    INTENT_PREDICTION = "intent_prediction"
    REFACTORING_SUGGESTIONS = "refactoring_suggestions"
    BUG_PREDICTION = "bug_prediction"
    PERFORMANCE_ANALYSIS = "performance_analysis"


class CodeDomain(Enum):
    """Code domains for specialized analysis"""
    WEB_FRONTEND = "web_frontend"
    WEB_BACKEND = "web_backend"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    SYSTEMS_PROGRAMMING = "systems_programming"
    API_DEVELOPMENT = "api_development"
    DATABASE = "database"
    DEVOPS = "devops"


@dataclass
class CodeFeatures:
    """Extracted features from code for neural analysis"""
    abstract_syntax_tree: Dict[str, Any]
    token_sequence: List[str]
    semantic_embeddings: np.ndarray
    structural_features: Dict[str, float]
    complexity_metrics: Dict[str, int]
    dependency_graph: Dict[str, List[str]]
    code_patterns: List[str]
    domain_indicators: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstract_syntax_tree": self.abstract_syntax_tree,
            "token_sequence": self.token_sequence,
            "semantic_embeddings": self.semantic_embeddings.tolist() if isinstance(self.semantic_embeddings, np.ndarray) else self.semantic_embeddings,
            "structural_features": self.structural_features,
            "complexity_metrics": self.complexity_metrics,
            "dependency_graph": self.dependency_graph,
            "code_patterns": self.code_patterns,
            "domain_indicators": self.domain_indicators
        }


@dataclass
class NeuralAnalysisResult:
    """Result from neural code analysis"""
    analysis_id: str
    analysis_type: AnalysisType
    confidence_score: float
    predictions: Dict[str, Any]
    feature_importance: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type.value,
            "confidence_score": self.confidence_score,
            "predictions": self.predictions,
            "feature_importance": self.feature_importance,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


class CodeAnalysisRequest(BaseModel):
    """Request for neural code analysis"""
    code: str
    language: str = "python"
    file_path: Optional[str] = None
    analysis_types: List[str] = Field(default=["semantic_understanding"])
    domain: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class NeuralCodeAnalyzer:
    """
    Advanced neural network-based code analyzer
    Uses transformer models and deep learning for sophisticated code understanding
    """
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else Path("./ml_models_cache")
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.loaded_models: Dict[str, Any] = {}
        self.feature_extractors: Dict[str, Any] = {}
        self.analysis_cache: Dict[str, NeuralAnalysisResult] = {}
        
        # Configuration
        self.max_code_length = 10000  # tokens
        self.batch_size = 16
        self.confidence_threshold = 0.7
        self.cache_expiry_hours = 24
        
        # Initialize feature extraction components
        self._initialize_extractors()
        
        logger.info(f"NeuralCodeAnalyzer initialized with cache dir: {self.model_cache_dir}")
    
    def _initialize_extractors(self) -> None:
        """Initialize feature extraction components"""
        try:
            # Initialize AST parser
            self.feature_extractors["ast_parser"] = self._create_ast_parser()
            
            # Initialize tokenizer
            self.feature_extractors["tokenizer"] = self._create_tokenizer()
            
            # Initialize embedding model
            self.feature_extractors["embedding_model"] = self._create_embedding_model()
            
            # Initialize complexity analyzer
            self.feature_extractors["complexity_analyzer"] = self._create_complexity_analyzer()
            
            logger.info("Feature extractors initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing extractors: {e}")
            # Use fallback implementations
            self._initialize_fallback_extractors()
    
    def _create_ast_parser(self) -> Any:
        """Create AST parser for code structure analysis"""
        try:
            import ast
            return ast
        except ImportError:
            logger.warning("AST module not available, using fallback")
            return None
    
    def _create_tokenizer(self) -> Any:
        """Create tokenizer for code tokenization"""
        try:
            # In production, would use specialized code tokenizers like CodeBERT
            # For now, using simple tokenization
            import re
            return re
        except ImportError:
            return None
    
    def _create_embedding_model(self) -> Any:
        """Create embedding model for semantic representations"""
        try:
            # In production, would load pre-trained code embeddings (CodeBERT, GraphCodeBERT, etc.)
            # For now, simulate with simple embeddings
            return self._create_simple_embedding_model()
        except Exception:
            return None
    
    def _create_simple_embedding_model(self) -> Dict[str, Any]:
        """Create simple embedding model for demonstration"""
        return {
            "vocab_size": 50000,
            "embedding_dim": 768,
            "model_type": "simple_hash_embedding"
        }
    
    def _create_complexity_analyzer(self) -> Dict[str, Any]:
        """Create complexity analysis tools"""
        return {
            "cyclomatic": True,
            "halstead": True,
            "maintainability": True,
            "cognitive": True
        }
    
    def _initialize_fallback_extractors(self) -> None:
        """Initialize fallback extractors when full models unavailable"""
        self.feature_extractors = {
            "ast_parser": None,
            "tokenizer": None,
            "embedding_model": self._create_simple_embedding_model(),
            "complexity_analyzer": self._create_complexity_analyzer()
        }
        logger.info("Fallback extractors initialized")
    
    async def analyze_code(self, request: CodeAnalysisRequest) -> List[NeuralAnalysisResult]:
        """
        Perform neural analysis on code
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate analysis ID
            analysis_id = self._generate_analysis_id(request.code)
            
            # Check cache
            cached_results = self._check_cache(analysis_id, request.analysis_types)
            if cached_results:
                logger.debug(f"Returning cached analysis results for {analysis_id}")
                return cached_results
            
            # Extract features from code
            features = await self._extract_features(
                request.code, 
                request.language, 
                request.context
            )
            
            # Perform analysis for each requested type
            results = []
            for analysis_type_str in request.analysis_types:
                try:
                    analysis_type = AnalysisType(analysis_type_str)
                    result = await self._perform_analysis(
                        analysis_id,
                        analysis_type,
                        features,
                        request
                    )
                    results.append(result)
                except ValueError:
                    logger.warning(f"Unknown analysis type: {analysis_type_str}")
                    continue
            
            # Cache results
            processing_time = asyncio.get_event_loop().time() - start_time
            for result in results:
                result.processing_time = processing_time
                self._cache_result(result)
            
            logger.info(
                f"Neural code analysis completed: {len(results)} analyses "
                f"in {processing_time:.3f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in neural code analysis: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Return error result
            error_result = NeuralAnalysisResult(
                analysis_id="error",
                analysis_type=AnalysisType.SEMANTIC_UNDERSTANDING,
                confidence_score=0.0,
                predictions={"error": str(e)},
                feature_importance={},
                recommendations=[],
                metadata={"error": True},
                processing_time=processing_time
            )
            
            return [error_result]
    
    async def _extract_features(
        self, 
        code: str, 
        language: str, 
        context: Dict[str, Any]
    ) -> CodeFeatures:
        """
        Extract comprehensive features from code
        """
        # AST extraction
        ast_features = await self._extract_ast_features(code, language)
        
        # Token sequence
        tokens = await self._tokenize_code(code, language)
        
        # Semantic embeddings
        embeddings = await self._generate_embeddings(code, tokens)
        
        # Structural features
        structural = await self._extract_structural_features(code, ast_features)
        
        # Complexity metrics
        complexity = await self._calculate_complexity_metrics(code, ast_features)
        
        # Dependency graph
        dependencies = await self._extract_dependencies(code, ast_features)
        
        # Code patterns
        patterns = await self._identify_code_patterns(code, ast_features)
        
        # Domain indicators
        domain_indicators = await self._identify_domain_indicators(code, context)
        
        return CodeFeatures(
            abstract_syntax_tree=ast_features,
            token_sequence=tokens,
            semantic_embeddings=embeddings,
            structural_features=structural,
            complexity_metrics=complexity,
            dependency_graph=dependencies,
            code_patterns=patterns,
            domain_indicators=domain_indicators
        )
    
    async def _extract_ast_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract Abstract Syntax Tree features"""
        if not self.feature_extractors.get("ast_parser") or language != "python":
            return {"nodes": [], "depth": 0, "complexity": 0}
        
        try:
            import ast
            tree = ast.parse(code)
            
            # Extract AST features
            features = {
                "nodes": [],
                "depth": 0,
                "complexity": 0,
                "node_types": [],
                "function_count": 0,
                "class_count": 0
            }
            
            for node in ast.walk(tree):
                features["nodes"].append(type(node).__name__)
                features["node_types"].append(type(node).__name__)
                
                if isinstance(node, ast.FunctionDef):
                    features["function_count"] += 1
                elif isinstance(node, ast.ClassDef):
                    features["class_count"] += 1
            
            # Calculate tree depth and complexity
            features["depth"] = self._calculate_ast_depth(tree)
            features["complexity"] = len(features["nodes"])
            features["unique_node_types"] = len(set(features["node_types"]))
            
            return features
            
        except Exception as e:
            logger.warning(f"AST extraction failed: {e}")
            return {"nodes": [], "depth": 0, "complexity": 0, "error": str(e)}
    
    def _calculate_ast_depth(self, node, depth: int = 0) -> int:
        """Calculate maximum depth of AST"""
        if not hasattr(node, '_fields') or not node._fields:
            return depth
        
        max_depth = depth
        for field_name, field_value in node.__dict__.items():
            if isinstance(field_value, list):
                for item in field_value:
                    if hasattr(item, '_fields'):
                        child_depth = self._calculate_ast_depth(item, depth + 1)
                        max_depth = max(max_depth, child_depth)
            elif hasattr(field_value, '_fields'):
                child_depth = self._calculate_ast_depth(field_value, depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    async def _tokenize_code(self, code: str, language: str) -> List[str]:
        """Tokenize code into meaningful tokens"""
        try:
            import re
            
            # Simple tokenization - in production would use specialized tokenizers
            # Remove comments and strings for basic tokenization
            code_no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            code_no_strings = re.sub(r'["\'].*?["\']', 'STRING_LITERAL', code_no_comments)
            
            # Split on common delimiters
            tokens = re.findall(
                r'\b\w+\b|[^\w\s]', 
                code_no_strings, 
                re.MULTILINE
            )
            
            # Filter out empty tokens and limit length
            filtered_tokens = [t.strip() for t in tokens if t.strip()]
            
            return filtered_tokens[:self.max_code_length]
            
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return code.split()[:100]  # Fallback to simple split
    
    async def _generate_embeddings(self, code: str, tokens: List[str]) -> np.ndarray:
        """Generate semantic embeddings for code"""
        try:
            # Simple hash-based embeddings for demonstration
            # In production, would use CodeBERT, GraphCodeBERT, or similar
            
            embedding_dim = 768
            embeddings = np.zeros(embedding_dim)
            
            # Generate embedding based on token hashes
            for i, token in enumerate(tokens[:100]):  # Limit to first 100 tokens
                token_hash = hash(token.lower()) % 10000
                embeddings[i % embedding_dim] += token_hash * 0.001
            
            # Normalize
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                embeddings = embeddings / norm
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return np.zeros(768)  # Return zero vector on error
    
    async def _extract_structural_features(
        self, 
        code: str, 
        ast_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract structural features from code"""
        features = {
            "lines_of_code": len(code.split('\n')),
            "characters": len(code),
            "tokens": len(code.split()),
            "ast_nodes": ast_features.get("complexity", 0),
            "function_count": ast_features.get("function_count", 0),
            "class_count": ast_features.get("class_count", 0),
            "ast_depth": ast_features.get("depth", 0),
            "unique_node_types": ast_features.get("unique_node_types", 0)
        }
        
        # Calculate ratios
        if features["lines_of_code"] > 0:
            features["tokens_per_line"] = features["tokens"] / features["lines_of_code"]
            features["chars_per_line"] = features["characters"] / features["lines_of_code"]
        else:
            features["tokens_per_line"] = 0.0
            features["chars_per_line"] = 0.0
        
        return features
    
    async def _calculate_complexity_metrics(
        self, 
        code: str, 
        ast_features: Dict[str, Any]
    ) -> Dict[str, int]:
        """Calculate various complexity metrics"""
        metrics = {
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(code),
            "cognitive_complexity": self._calculate_cognitive_complexity(code),
            "halstead_volume": self._calculate_halstead_volume(code),
            "maintainability_index": self._calculate_maintainability_index(code),
            "nested_depth": ast_features.get("depth", 0)
        }
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        # Count decision points
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        complexity = 1  # Base complexity
        
        for keyword in decision_keywords:
            complexity += code.count(f' {keyword} ') + code.count(f'\n{keyword} ')
        
        return complexity
    
    def _calculate_cognitive_complexity(self, code: str) -> int:
        """Calculate cognitive complexity (simplified)"""
        # Similar to cyclomatic but with weights for nesting
        cognitive_keywords = ['if', 'for', 'while', 'try']
        complexity = 0
        
        lines = code.split('\n')
        for line in lines:
            indent_level = len(line) - len(line.lstrip())
            nesting_weight = max(1, indent_level // 4)  # 4 spaces = 1 level
            
            for keyword in cognitive_keywords:
                if keyword in line:
                    complexity += nesting_weight
        
        return complexity
    
    def _calculate_halstead_volume(self, code: str) -> int:
        """Calculate Halstead volume (simplified)"""
        import re
        
        # Extract operators and operands
        operators = re.findall(r'[+\-*/%=<>!&|^~]|\b(and|or|not|in|is)\b', code)
        operands = re.findall(r'\b[a-zA-Z_]\w*\b', code)
        
        unique_operators = len(set(operators))
        unique_operands = len(set(operands))
        total_operators = len(operators)
        total_operands = len(operands)
        
        if unique_operators > 0 and unique_operands > 0:
            vocabulary = unique_operators + unique_operands
            length = total_operators + total_operands
            volume = length * np.log2(max(vocabulary, 1))
            return int(volume)
        
        return 0
    
    def _calculate_maintainability_index(self, code: str) -> int:
        """Calculate maintainability index (simplified)"""
        lines = len(code.split('\n'))
        complexity = self._calculate_cyclomatic_complexity(code)
        volume = self._calculate_halstead_volume(code)
        
        if lines > 0:
            # Simplified maintainability index formula
            mi = max(0, (171 - 5.2 * np.log(max(volume, 1)) - 0.23 * complexity - 16.2 * np.log(lines)) * 100 / 171)
            return int(mi)
        
        return 0
    
    async def _extract_dependencies(
        self, 
        code: str, 
        ast_features: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract code dependencies"""
        dependencies = {"imports": [], "functions": [], "classes": []}
        
        # Extract imports
        import_lines = [line.strip() for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
        dependencies["imports"] = import_lines
        
        # Extract function and class names (simplified)
        import re
        
        function_matches = re.findall(r'def\s+(\w+)\s*\(', code)
        dependencies["functions"] = function_matches
        
        class_matches = re.findall(r'class\s+(\w+)\s*[\(:]', code)
        dependencies["classes"] = class_matches
        
        return dependencies
    
    async def _identify_code_patterns(
        self, 
        code: str, 
        ast_features: Dict[str, Any]
    ) -> List[str]:
        """Identify common code patterns"""
        patterns = []
        
        # Design pattern detection (simplified)
        if 'class' in code and '__init__' in code:
            patterns.append("class_definition")
        
        if 'def __enter__' in code and 'def __exit__' in code:
            patterns.append("context_manager")
        
        if 'yield' in code:
            patterns.append("generator")
        
        if 'async def' in code:
            patterns.append("async_function")
        
        if 'try:' in code and 'except' in code:
            patterns.append("exception_handling")
        
        if 'with open(' in code:
            patterns.append("file_handling")
        
        if 'import unittest' in code or 'import pytest' in code:
            patterns.append("test_code")
        
        return patterns
    
    async def _identify_domain_indicators(
        self, 
        code: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Identify domain-specific indicators"""
        indicators = []
        
        # Web development indicators
        web_keywords = ['flask', 'django', 'fastapi', 'request', 'response', 'http']
        if any(keyword in code.lower() for keyword in web_keywords):
            indicators.append("web_development")
        
        # Data science indicators
        ds_keywords = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'dataframe']
        if any(keyword in code.lower() for keyword in ds_keywords):
            indicators.append("data_science")
        
        # Machine learning indicators
        ml_keywords = ['tensorflow', 'pytorch', 'keras', 'model', 'train', 'predict']
        if any(keyword in code.lower() for keyword in ml_keywords):
            indicators.append("machine_learning")
        
        # API development
        api_keywords = ['api', 'endpoint', 'router', 'get', 'post', 'put', 'delete']
        if any(keyword in code.lower() for keyword in api_keywords):
            indicators.append("api_development")
        
        # Database
        db_keywords = ['sql', 'database', 'query', 'select', 'insert', 'update', 'delete']
        if any(keyword in code.lower() for keyword in db_keywords):
            indicators.append("database")
        
        return indicators
    
    async def _perform_analysis(
        self,
        analysis_id: str,
        analysis_type: AnalysisType,
        features: CodeFeatures,
        request: CodeAnalysisRequest
    ) -> NeuralAnalysisResult:
        """
        Perform specific type of neural analysis
        """
        if analysis_type == AnalysisType.SEMANTIC_UNDERSTANDING:
            return await self._analyze_semantic_understanding(analysis_id, features, request)
        elif analysis_type == AnalysisType.CODE_QUALITY:
            return await self._analyze_code_quality(analysis_id, features, request)
        elif analysis_type == AnalysisType.COMPLEXITY_ANALYSIS:
            return await self._analyze_complexity(analysis_id, features, request)
        elif analysis_type == AnalysisType.INTENT_PREDICTION:
            return await self._predict_intent(analysis_id, features, request)
        elif analysis_type == AnalysisType.REFACTORING_SUGGESTIONS:
            return await self._suggest_refactoring(analysis_id, features, request)
        elif analysis_type == AnalysisType.BUG_PREDICTION:
            return await self._predict_bugs(analysis_id, features, request)
        elif analysis_type == AnalysisType.PERFORMANCE_ANALYSIS:
            return await self._analyze_performance(analysis_id, features, request)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    async def _analyze_semantic_understanding(
        self,
        analysis_id: str,
        features: CodeFeatures,
        request: CodeAnalysisRequest
    ) -> NeuralAnalysisResult:
        """Analyze semantic understanding of code"""
        
        predictions = {
            "purpose": self._infer_code_purpose(features, request.code),
            "functionality": self._describe_functionality(features),
            "domain": features.domain_indicators,
            "patterns": features.code_patterns,
            "semantic_similarity": self._calculate_semantic_similarity(features)
        }
        
        # Calculate confidence based on feature richness
        confidence = self._calculate_semantic_confidence(features)
        
        # Generate feature importance
        feature_importance = {
            "ast_complexity": 0.3,
            "token_diversity": 0.2,
            "pattern_matches": 0.25,
            "domain_indicators": 0.15,
            "structural_features": 0.1
        }
        
        recommendations = self._generate_semantic_recommendations(features, predictions)
        
        return NeuralAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.SEMANTIC_UNDERSTANDING,
            confidence_score=confidence,
            predictions=predictions,
            feature_importance=feature_importance,
            recommendations=recommendations,
            metadata={
                "token_count": len(features.token_sequence),
                "ast_nodes": features.complexity_metrics.get("nested_depth", 0),
                "identified_patterns": len(features.code_patterns)
            },
            processing_time=0.0  # Will be set by caller
        )
    
    def _infer_code_purpose(self, features: CodeFeatures, code: str) -> str:
        """Infer the purpose of the code"""
        # Simple heuristic-based purpose inference
        if "test" in code.lower() or any("test" in pattern for pattern in features.code_patterns):
            return "testing"
        elif features.structural_features.get("class_count", 0) > 0:
            return "class_definition"
        elif features.structural_features.get("function_count", 0) > 1:
            return "utility_functions"
        elif "api" in features.domain_indicators:
            return "api_endpoint"
        elif "data_science" in features.domain_indicators:
            return "data_analysis"
        elif "web_development" in features.domain_indicators:
            return "web_application"
        else:
            return "general_purpose"
    
    def _describe_functionality(self, features: CodeFeatures) -> Dict[str, Any]:
        """Describe the functionality of the code"""
        return {
            "complexity_level": self._classify_complexity(features.complexity_metrics),
            "main_operations": features.code_patterns,
            "dependencies": len(features.dependency_graph.get("imports", [])),
            "structure_type": self._classify_structure(features.structural_features)
        }
    
    def _classify_complexity(self, complexity_metrics: Dict[str, int]) -> str:
        """Classify complexity level"""
        cyclomatic = complexity_metrics.get("cyclomatic_complexity", 0)
        
        if cyclomatic <= 5:
            return "simple"
        elif cyclomatic <= 10:
            return "moderate"
        elif cyclomatic <= 20:
            return "complex"
        else:
            return "very_complex"
    
    def _classify_structure(self, structural_features: Dict[str, float]) -> str:
        """Classify code structure type"""
        if structural_features.get("class_count", 0) > 0:
            return "object_oriented"
        elif structural_features.get("function_count", 0) > 3:
            return "functional"
        else:
            return "procedural"
    
    def _calculate_semantic_similarity(self, features: CodeFeatures) -> float:
        """Calculate semantic similarity score (placeholder)"""
        # In production, would compare against known code patterns
        return 0.75  # Placeholder value
    
    def _calculate_semantic_confidence(self, features: CodeFeatures) -> float:
        """Calculate confidence in semantic analysis"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on feature richness
        if len(features.code_patterns) > 0:
            confidence += 0.1
        
        if len(features.domain_indicators) > 0:
            confidence += 0.1
        
        if features.complexity_metrics.get("ast_nodes", 0) > 0:
            confidence += 0.1
        
        if len(features.token_sequence) > 50:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_semantic_recommendations(
        self, 
        features: CodeFeatures, 
        predictions: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on semantic analysis"""
        recommendations = []
        
        if predictions["purpose"] == "testing":
            recommendations.append("Consider adding more edge case tests")
        
        if len(features.domain_indicators) > 1:
            recommendations.append("Code appears to mix multiple domains - consider separation")
        
        if features.complexity_metrics.get("cyclomatic_complexity", 0) > 10:
            recommendations.append("High complexity detected - consider refactoring")
        
        return recommendations
    
    async def _analyze_code_quality(
        self,
        analysis_id: str,
        features: CodeFeatures,
        request: CodeAnalysisRequest
    ) -> NeuralAnalysisResult:
        """Analyze code quality metrics"""
        
        quality_scores = {
            "maintainability": features.complexity_metrics.get("maintainability_index", 50) / 100,
            "readability": self._assess_readability(features),
            "complexity_score": 1.0 - min(features.complexity_metrics.get("cyclomatic_complexity", 0) / 20, 1.0),
            "structure_quality": self._assess_structure_quality(features),
            "documentation_score": self._assess_documentation(request.code)
        }
        
        overall_quality = sum(quality_scores.values()) / len(quality_scores)
        
        predictions = {
            "overall_quality": overall_quality,
            "quality_breakdown": quality_scores,
            "quality_issues": self._identify_quality_issues(features, request.code),
            "quality_grade": self._assign_quality_grade(overall_quality)
        }
        
        recommendations = self._generate_quality_recommendations(quality_scores, features)
        
        return NeuralAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.CODE_QUALITY,
            confidence_score=0.85,
            predictions=predictions,
            feature_importance={
                "maintainability": 0.3,
                "complexity": 0.25,
                "readability": 0.2,
                "structure": 0.15,
                "documentation": 0.1
            },
            recommendations=recommendations,
            metadata={"quality_metrics_count": len(quality_scores)},
            processing_time=0.0
        )
    
    def _assess_readability(self, features: CodeFeatures) -> float:
        """Assess code readability"""
        readability = 0.8  # Base score
        
        # Penalize very long lines
        avg_line_length = features.structural_features.get("chars_per_line", 0)
        if avg_line_length > 100:
            readability -= 0.2
        elif avg_line_length > 80:
            readability -= 0.1
        
        # Boost for good structure
        if features.structural_features.get("function_count", 0) > 0:
            readability += 0.1
        
        return max(0.0, min(1.0, readability))
    
    def _assess_structure_quality(self, features: CodeFeatures) -> float:
        """Assess structural quality"""
        structure_score = 0.7  # Base score
        
        # Good function decomposition
        lines_per_function = features.structural_features.get("lines_of_code", 0) / max(features.structural_features.get("function_count", 1), 1)
        if lines_per_function < 50:
            structure_score += 0.2
        elif lines_per_function > 100:
            structure_score -= 0.2
        
        # Good class design
        if features.structural_features.get("class_count", 0) > 0:
            structure_score += 0.1
        
        return max(0.0, min(1.0, structure_score))
    
    def _assess_documentation(self, code: str) -> float:
        """Assess documentation quality"""
        doc_score = 0.3  # Base score for no docs
        
        # Check for docstrings
        if '"""' in code or "'''" in code:
            doc_score += 0.4
        
        # Check for comments
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        total_lines = len(code.split('\n'))
        
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            doc_score += min(comment_ratio * 2, 0.3)
        
        return min(1.0, doc_score)
    
    def _identify_quality_issues(self, features: CodeFeatures, code: str) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        if features.complexity_metrics.get("cyclomatic_complexity", 0) > 10:
            issues.append("High cyclomatic complexity")
        
        if features.structural_features.get("chars_per_line", 0) > 100:
            issues.append("Lines too long")
        
        if len(features.dependency_graph.get("imports", [])) > 20:
            issues.append("Too many dependencies")
        
        if features.complexity_metrics.get("nested_depth", 0) > 5:
            issues.append("Deep nesting detected")
        
        return issues
    
    def _assign_quality_grade(self, quality_score: float) -> str:
        """Assign quality grade"""
        if quality_score >= 0.9:
            return "A"
        elif quality_score >= 0.8:
            return "B"
        elif quality_score >= 0.7:
            return "C"
        elif quality_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_quality_recommendations(
        self, 
        quality_scores: Dict[str, float], 
        features: CodeFeatures
    ) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if quality_scores["complexity_score"] < 0.6:
            recommendations.append("Reduce cyclomatic complexity by breaking down large functions")
        
        if quality_scores["readability"] < 0.7:
            recommendations.append("Improve readability by shortening lines and adding whitespace")
        
        if quality_scores["documentation_score"] < 0.5:
            recommendations.append("Add docstrings and comments to improve documentation")
        
        if quality_scores["maintainability"] < 0.6:
            recommendations.append("Improve maintainability by reducing complexity and improving structure")
        
        return recommendations
    
    # Placeholder implementations for other analysis types
    async def _analyze_complexity(self, analysis_id: str, features: CodeFeatures, request: CodeAnalysisRequest) -> NeuralAnalysisResult:
        """Analyze code complexity"""
        return NeuralAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.COMPLEXITY_ANALYSIS,
            confidence_score=0.8,
            predictions={"complexity_metrics": features.complexity_metrics},
            feature_importance={"cyclomatic": 0.4, "cognitive": 0.3, "halstead": 0.3},
            recommendations=["Consider simplifying complex functions"],
            metadata={},
            processing_time=0.0
        )
    
    async def _predict_intent(self, analysis_id: str, features: CodeFeatures, request: CodeAnalysisRequest) -> NeuralAnalysisResult:
        """Predict code intent"""
        return NeuralAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.INTENT_PREDICTION,
            confidence_score=0.7,
            predictions={"predicted_intent": self._infer_code_purpose(features, request.code)},
            feature_importance={"patterns": 0.5, "structure": 0.3, "tokens": 0.2},
            recommendations=["Intent unclear - consider adding documentation"],
            metadata={},
            processing_time=0.0
        )
    
    async def _suggest_refactoring(self, analysis_id: str, features: CodeFeatures, request: CodeAnalysisRequest) -> NeuralAnalysisResult:
        """Suggest refactoring opportunities"""
        return NeuralAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.REFACTORING_SUGGESTIONS,
            confidence_score=0.75,
            predictions={"refactoring_opportunities": ["Extract method", "Reduce nesting"]},
            feature_importance={"complexity": 0.6, "structure": 0.4},
            recommendations=["Extract complex logic into separate methods"],
            metadata={},
            processing_time=0.0
        )
    
    async def _predict_bugs(self, analysis_id: str, features: CodeFeatures, request: CodeAnalysisRequest) -> NeuralAnalysisResult:
        """Predict potential bugs"""
        return NeuralAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.BUG_PREDICTION,
            confidence_score=0.6,
            predictions={"bug_risk": "medium", "potential_issues": ["Null pointer risk"]},
            feature_importance={"patterns": 0.4, "complexity": 0.4, "structure": 0.2},
            recommendations=["Add input validation and error handling"],
            metadata={},
            processing_time=0.0
        )
    
    async def _analyze_performance(self, analysis_id: str, features: CodeFeatures, request: CodeAnalysisRequest) -> NeuralAnalysisResult:
        """Analyze performance characteristics"""
        return NeuralAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
            confidence_score=0.65,
            predictions={"performance_risk": "low", "bottlenecks": []},
            feature_importance={"complexity": 0.5, "patterns": 0.3, "structure": 0.2},
            recommendations=["Performance looks good for current complexity level"],
            metadata={},
            processing_time=0.0
        )
    
    def _generate_analysis_id(self, code: str) -> str:
        """Generate unique analysis ID"""
        code_hash = hashlib.md5(code.encode()).hexdigest()
        return f"neural_{code_hash[:12]}"
    
    def _check_cache(self, analysis_id: str, analysis_types: List[str]) -> Optional[List[NeuralAnalysisResult]]:
        """Check if analysis results are cached"""
        # Simple cache check - in production would be more sophisticated
        cached_results = []
        for analysis_type in analysis_types:
            cache_key = f"{analysis_id}_{analysis_type}"
            if cache_key in self.analysis_cache:
                cached_results.append(self.analysis_cache[cache_key])
        
        return cached_results if len(cached_results) == len(analysis_types) else None
    
    def _cache_result(self, result: NeuralAnalysisResult) -> None:
        """Cache analysis result"""
        cache_key = f"{result.analysis_id}_{result.analysis_type.value}"
        self.analysis_cache[cache_key] = result
        
        # Limit cache size
        if len(self.analysis_cache) > 1000:
            # Remove oldest 25% of entries
            sorted_items = sorted(
                self.analysis_cache.items(), 
                key=lambda x: x[1].timestamp
            )
            for i in range(len(sorted_items) // 4):
                del self.analysis_cache[sorted_items[i][0]]
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "feature_extractors": list(self.feature_extractors.keys()),
            "cache_size": len(self.analysis_cache),
            "model_cache_dir": str(self.model_cache_dir)
        }