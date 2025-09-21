"""
ML Model Coordinator
Central coordinator for all ML models in the Archon system
Provides unified intelligence, model orchestration, and cross-model learning
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import logging
from collections import defaultdict, Counter
import json

from pydantic import BaseModel, Field

from .neural_code_analyzer import NeuralCodeAnalyzer, CodeAnalysisRequest, NeuralAnalysisResult
from .pattern_prediction_model import PatternPredictionModel, PatternPredictionRequest, PatternPrediction
from .semantic_similarity_engine import SemanticSimilarityEngine, SimilaritySearchRequest, SimilarityMatch, CodeSegment
from .adaptive_learning_system import AdaptiveLearningSystem, UserInteraction, UserFeedback, LearnableModel

logger = logging.getLogger(__name__)


class IntelligenceLevel(Enum):
    """Levels of intelligence processing"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AnalysisScope(Enum):
    """Scope of analysis across models"""
    SINGLE_FILE = "single_file"
    MULTI_FILE = "multi_file"
    PROJECT_WIDE = "project_wide"
    CROSS_PROJECT = "cross_project"


class ModelPriority(Enum):
    """Priority levels for model execution"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelCapability:
    """Describes capabilities of a registered model"""
    model_name: str
    model_type: str
    capabilities: List[str]
    performance_metrics: Dict[str, float]
    resource_requirements: Dict[str, Any]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "capabilities": self.capabilities,
            "performance_metrics": self.performance_metrics,
            "resource_requirements": self.resource_requirements,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class UnifiedAnalysisResult:
    """Unified result from multiple ML models"""
    analysis_id: str
    request_context: Dict[str, Any]
    individual_results: Dict[str, Any]
    unified_insights: Dict[str, Any]
    confidence_score: float
    processing_time: float
    models_used: List[str]
    cross_model_correlations: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "request_context": self.request_context,
            "individual_results": self.individual_results,
            "unified_insights": self.unified_insights,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "models_used": self.models_used,
            "cross_model_correlations": self.cross_model_correlations,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat()
        }


class UnifiedAnalysisRequest(BaseModel):
    """Request for unified analysis across all ML models"""
    code: str
    file_path: Optional[str] = None
    project_context: Dict[str, Any] = Field(default_factory=dict)
    analysis_types: List[str] = Field(default=["all"])
    intelligence_level: str = "enhanced"
    include_patterns: bool = True
    include_similarity: bool = True
    include_learning: bool = True
    max_processing_time: int = 30  # seconds
    user_id: Optional[str] = None


class MLModelCoordinator:
    """
    Central coordinator for all ML models in Archon
    Provides unified intelligence and cross-model learning
    """
    
    def __init__(self):
        # Core ML models
        self.neural_analyzer: Optional[NeuralCodeAnalyzer] = None
        self.pattern_predictor: Optional[PatternPredictionModel] = None
        self.similarity_engine: Optional[SemanticSimilarityEngine] = None
        self.learning_system: Optional[AdaptiveLearningSystem] = None
        
        # Model registry
        self.registered_models: Dict[str, ModelCapability] = {}
        self.model_performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Coordination state
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        self.model_load_balancer: ModelLoadBalancer = ModelLoadBalancer()
        self.cross_model_learner: CrossModelLearner = CrossModelLearner()
        
        # Configuration
        self.enable_cross_model_learning = True
        self.enable_performance_optimization = True
        self.max_concurrent_analyses = 10
        
        # Statistics
        self.coordination_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "model_usage_counts": defaultdict(int)
        }
        
        logger.info("MLModelCoordinator initialized")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the ML model coordinator with all models"""
        try:
            # Initialize core models
            self.neural_analyzer = NeuralCodeAnalyzer()
            self.pattern_predictor = PatternPredictionModel()
            self.similarity_engine = SemanticSimilarityEngine()
            self.learning_system = AdaptiveLearningSystem()
            
            # Register models with learning system
            await self._register_models_with_learning_system()
            
            # Register model capabilities
            await self._register_model_capabilities()
            
            # Initialize cross-model learning
            await self.cross_model_learner.initialize(
                {
                    "neural_analyzer": self.neural_analyzer,
                    "pattern_predictor": self.pattern_predictor,
                    "similarity_engine": self.similarity_engine
                }
            )
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("MLModelCoordinator successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MLModelCoordinator: {e}")
            return False
    
    async def analyze_code_unified(
        self, 
        request: UnifiedAnalysisRequest
    ) -> UnifiedAnalysisResult:
        """
        Perform unified analysis using multiple ML models
        """
        start_time = asyncio.get_event_loop().time()
        analysis_id = f"unified_{hash(request.code)%10000}_{int(start_time)}"
        
        try:
            # Initialize analysis tracking
            self.active_analyses[analysis_id] = {
                "start_time": start_time,
                "request": request,
                "status": "running"
            }
            
            # Determine which models to use
            models_to_use = await self._select_models_for_analysis(request)
            
            # Execute analyses in parallel
            individual_results = await self._execute_parallel_analyses(
                request, models_to_use
            )
            
            # Process results and generate unified insights
            unified_insights = await self._generate_unified_insights(
                individual_results, request
            )
            
            # Calculate cross-model correlations
            correlations = await self._calculate_cross_model_correlations(
                individual_results
            )
            
            # Generate unified recommendations
            recommendations = await self._generate_unified_recommendations(
                unified_insights, correlations, request
            )
            
            # Calculate overall confidence
            confidence = await self._calculate_unified_confidence(
                individual_results, correlations
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create unified result
            result = UnifiedAnalysisResult(
                analysis_id=analysis_id,
                request_context={
                    "code_length": len(request.code),
                    "file_path": request.file_path,
                    "intelligence_level": request.intelligence_level,
                    "user_id": request.user_id
                },
                individual_results=individual_results,
                unified_insights=unified_insights,
                confidence_score=confidence,
                processing_time=processing_time,
                models_used=models_to_use,
                cross_model_correlations=correlations,
                recommendations=recommendations
            )
            
            # Record for learning if user provided
            if request.user_id:
                await self._record_analysis_for_learning(request, result)
            
            # Update statistics
            self._update_coordination_stats(result, True)
            
            # Clean up tracking
            del self.active_analyses[analysis_id]
            
            logger.info(f"Unified analysis completed: {analysis_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in unified analysis: {e}")
            
            # Create error result
            processing_time = asyncio.get_event_loop().time() - start_time
            error_result = UnifiedAnalysisResult(
                analysis_id=analysis_id,
                request_context={},
                individual_results={"error": str(e)},
                unified_insights={"error": "Analysis failed"},
                confidence_score=0.0,
                processing_time=processing_time,
                models_used=[],
                cross_model_correlations={},
                recommendations=["Analysis failed - please try again"]
            )
            
            self._update_coordination_stats(error_result, False)
            
            if analysis_id in self.active_analyses:
                del self.active_analyses[analysis_id]
            
            return error_result
    
    async def _register_models_with_learning_system(self) -> None:
        """Register models with the adaptive learning system"""
        class NeuralAnalyzerAdapter(LearnableModel):
            def __init__(self, analyzer):
                self.analyzer = analyzer
            
            async def adapt(self, adaptations):
                # Simplified adaptation for neural analyzer
                return True
            
            async def get_current_performance(self):
                return 0.85  # Placeholder performance metric
            
            async def rollback_adaptation(self, adaptation_id):
                return True
        
        class PatternPredictorAdapter(LearnableModel):
            def __init__(self, predictor):
                self.predictor = predictor
            
            async def adapt(self, adaptations):
                # Simplified adaptation for pattern predictor
                return True
            
            async def get_current_performance(self):
                return 0.82  # Placeholder performance metric
            
            async def rollback_adaptation(self, adaptation_id):
                return True
        
        # Register adapters
        if self.neural_analyzer:
            await self.learning_system.register_learnable_model(
                "neural_analyzer", 
                NeuralAnalyzerAdapter(self.neural_analyzer)
            )
        
        if self.pattern_predictor:
            await self.learning_system.register_learnable_model(
                "pattern_predictor", 
                PatternPredictorAdapter(self.pattern_predictor)
            )
    
    async def _register_model_capabilities(self) -> None:
        """Register capabilities of all models"""
        # Neural Code Analyzer
        if self.neural_analyzer:
            self.registered_models["neural_analyzer"] = ModelCapability(
                model_name="neural_analyzer",
                model_type="neural_network",
                capabilities=[
                    "semantic_understanding", "code_quality", "complexity_analysis",
                    "intent_prediction", "bug_prediction", "performance_analysis"
                ],
                performance_metrics={"accuracy": 0.85, "speed": 0.9},
                resource_requirements={"memory": "512MB", "cpu": "medium"},
                last_updated=datetime.now(timezone.utc)
            )
        
        # Pattern Prediction Model
        if self.pattern_predictor:
            self.registered_models["pattern_predictor"] = ModelCapability(
                model_name="pattern_predictor",
                model_type="pattern_matcher",
                capabilities=[
                    "design_pattern_prediction", "refactoring_suggestions",
                    "architectural_recommendations", "anti_pattern_detection"
                ],
                performance_metrics={"accuracy": 0.82, "speed": 0.95},
                resource_requirements={"memory": "256MB", "cpu": "low"},
                last_updated=datetime.now(timezone.utc)
            )
        
        # Semantic Similarity Engine
        if self.similarity_engine:
            self.registered_models["similarity_engine"] = ModelCapability(
                model_name="similarity_engine",
                model_type="similarity_matcher",
                capabilities=[
                    "duplicate_detection", "semantic_similarity", "code_clustering",
                    "refactor_opportunities"
                ],
                performance_metrics={"accuracy": 0.88, "speed": 0.85},
                resource_requirements={"memory": "768MB", "cpu": "high"},
                last_updated=datetime.now(timezone.utc)
            )
        
        # Adaptive Learning System
        if self.learning_system:
            self.registered_models["learning_system"] = ModelCapability(
                model_name="learning_system",
                model_type="adaptive_learner",
                capabilities=[
                    "user_preference_learning", "model_adaptation",
                    "feedback_processing", "pattern_extraction"
                ],
                performance_metrics={"accuracy": 0.78, "speed": 0.75},
                resource_requirements={"memory": "1GB", "cpu": "medium"},
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _select_models_for_analysis(
        self, 
        request: UnifiedAnalysisRequest
    ) -> List[str]:
        """Select appropriate models based on request requirements"""
        selected_models = []
        
        # Always include neural analyzer for basic analysis
        selected_models.append("neural_analyzer")
        
        # Include pattern predictor if patterns requested
        if request.include_patterns:
            selected_models.append("pattern_predictor")
        
        # Include similarity engine if similarity analysis requested
        if request.include_similarity:
            selected_models.append("similarity_engine")
        
        # Include learning system for personalization if user provided
        if request.include_learning and request.user_id:
            selected_models.append("learning_system")
        
        # Filter based on intelligence level
        if request.intelligence_level == "basic":
            selected_models = ["neural_analyzer"]
        elif request.intelligence_level == "expert":
            # Use all available models for expert level
            selected_models = list(self.registered_models.keys())
        
        # Apply load balancing
        selected_models = await self.model_load_balancer.balance_model_selection(
            selected_models, self.model_performance_history
        )
        
        return selected_models
    
    async def _execute_parallel_analyses(
        self, 
        request: UnifiedAnalysisRequest, 
        models_to_use: List[str]
    ) -> Dict[str, Any]:
        """Execute analyses in parallel across selected models"""
        tasks = []
        
        # Neural analyzer task
        if "neural_analyzer" in models_to_use and self.neural_analyzer:
            analysis_request = CodeAnalysisRequest(
                code=request.code,
                file_path=request.file_path,
                analysis_types=["semantic_understanding", "code_quality", "complexity_analysis"],
                context=request.project_context
            )
            task = self._run_neural_analysis(analysis_request)
            tasks.append(("neural_analyzer", task))
        
        # Pattern predictor task
        if "pattern_predictor" in models_to_use and self.pattern_predictor:
            pattern_request = PatternPredictionRequest(
                current_code=request.code,
                file_path=request.file_path,
                project_context=request.project_context,
                prediction_types=["design_pattern", "refactoring_pattern", "anti_pattern"]
            )
            task = self._run_pattern_prediction(pattern_request)
            tasks.append(("pattern_predictor", task))
        
        # Similarity engine task
        if "similarity_engine" in models_to_use and self.similarity_engine:
            similarity_request = SimilaritySearchRequest(
                query_code=request.code,
                similarity_types=["semantic", "structural"],
                threshold=0.7
            )
            task = self._run_similarity_analysis(similarity_request)
            tasks.append(("similarity_engine", task))
        
        # Learning system task
        if "learning_system" in models_to_use and self.learning_system and request.user_id:
            task = self._run_adaptive_suggestions(request.user_id, request.project_context)
            tasks.append(("learning_system", task))
        
        # Execute all tasks in parallel
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for i, (model_name, _) in enumerate(tasks):
                result = completed_tasks[i]
                if isinstance(result, Exception):
                    logger.error(f"Error in {model_name}: {result}")
                    results[model_name] = {"error": str(result)}
                else:
                    results[model_name] = result
                    self.coordination_stats["model_usage_counts"][model_name] += 1
        
        return results
    
    async def _run_neural_analysis(self, request: CodeAnalysisRequest) -> Dict[str, Any]:
        """Run neural code analysis"""
        try:
            results = await self.neural_analyzer.analyze_code(request)
            return {
                "success": True,
                "analyses": [result.to_dict() for result in results],
                "count": len(results)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_pattern_prediction(self, request: PatternPredictionRequest) -> Dict[str, Any]:
        """Run pattern prediction"""
        try:
            predictions = await self.pattern_predictor.predict_patterns(request)
            return {
                "success": True,
                "predictions": [pred.to_dict() for pred in predictions],
                "count": len(predictions)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_similarity_analysis(self, request: SimilaritySearchRequest) -> Dict[str, Any]:
        """Run similarity analysis"""
        try:
            matches = await self.similarity_engine.find_similar_code(request)
            return {
                "success": True,
                "matches": [match.to_dict() for match in matches],
                "count": len(matches)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_adaptive_suggestions(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run adaptive learning suggestions"""
        try:
            suggestions = await self.learning_system.get_adaptive_suggestions(user_id, context)
            return {
                "success": True,
                "suggestions": suggestions,
                "count": len(suggestions)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_unified_insights(
        self, 
        individual_results: Dict[str, Any], 
        request: UnifiedAnalysisRequest
    ) -> Dict[str, Any]:
        """Generate unified insights from individual model results"""
        insights = {
            "code_quality_score": 0.0,
            "complexity_assessment": "unknown",
            "recommended_patterns": [],
            "potential_issues": [],
            "refactoring_opportunities": [],
            "similarity_findings": [],
            "personalized_suggestions": []
        }
        
        # Process neural analyzer results
        if "neural_analyzer" in individual_results and individual_results["neural_analyzer"].get("success"):
            neural_results = individual_results["neural_analyzer"]["analyses"]
            
            for analysis in neural_results:
                if analysis.get("analysis_type") == "code_quality":
                    predictions = analysis.get("predictions", {})
                    if "overall_quality" in predictions:
                        insights["code_quality_score"] = predictions["overall_quality"]
                
                elif analysis.get("analysis_type") == "complexity_analysis":
                    predictions = analysis.get("predictions", {})
                    insights["complexity_assessment"] = predictions.get("complexity_level", "unknown")
                
                # Extract potential issues
                recommendations = analysis.get("recommendations", [])
                insights["potential_issues"].extend(recommendations)
        
        # Process pattern predictor results
        if "pattern_predictor" in individual_results and individual_results["pattern_predictor"].get("success"):
            pattern_results = individual_results["pattern_predictor"]["predictions"]
            
            for prediction in pattern_results:
                if prediction.get("confidence_score", 0) > 0.7:
                    insights["recommended_patterns"].append({
                        "pattern": prediction.get("pattern_name"),
                        "confidence": prediction.get("confidence_score"),
                        "reasoning": prediction.get("reasoning", [])
                    })
                
                # Extract refactoring opportunities
                if "refactoring" in prediction.get("pattern_type", "").lower():
                    insights["refactoring_opportunities"].append({
                        "opportunity": prediction.get("pattern_name"),
                        "confidence": prediction.get("confidence_score"),
                        "effort": prediction.get("implementation_effort")
                    })
        
        # Process similarity engine results
        if "similarity_engine" in individual_results and individual_results["similarity_engine"].get("success"):
            similarity_results = individual_results["similarity_engine"]["matches"]
            
            for match in similarity_results:
                if match.get("similarity_score", 0) > 0.8:
                    insights["similarity_findings"].append({
                        "type": match.get("match_type"),
                        "similarity": match.get("similarity_score"),
                        "file": match.get("target_segment", {}).get("file_path"),
                        "suggestions": match.get("refactor_suggestions", [])
                    })
        
        # Process learning system results
        if "learning_system" in individual_results and individual_results["learning_system"].get("success"):
            learning_results = individual_results["learning_system"]["suggestions"]
            insights["personalized_suggestions"] = learning_results
        
        return insights
    
    async def _calculate_cross_model_correlations(
        self, 
        individual_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate correlations between model results"""
        correlations = {}
        
        # Correlation between quality score and complexity
        neural_quality = 0.5
        neural_complexity = "unknown"
        
        if "neural_analyzer" in individual_results:
            neural_results = individual_results["neural_analyzer"]
            if neural_results.get("success"):
                for analysis in neural_results.get("analyses", []):
                    predictions = analysis.get("predictions", {})
                    if "overall_quality" in predictions:
                        neural_quality = predictions["overall_quality"]
                    if "complexity_level" in predictions:
                        neural_complexity = predictions["complexity_level"]
        
        # Map complexity to numeric for correlation
        complexity_map = {"simple": 1, "moderate": 2, "complex": 3, "very_complex": 4}
        complexity_numeric = complexity_map.get(neural_complexity, 2)
        
        # Calculate quality-complexity correlation
        # Generally, higher complexity should correlate with lower quality
        expected_quality = 1.0 - (complexity_numeric - 1) * 0.25
        quality_complexity_correlation = 1.0 - abs(neural_quality - expected_quality)
        
        correlations["quality_complexity"] = {
            "correlation_score": quality_complexity_correlation,
            "quality": neural_quality,
            "complexity": neural_complexity,
            "alignment": "good" if quality_complexity_correlation > 0.7 else "poor"
        }
        
        # Correlation between patterns and quality
        pattern_count = 0
        if "pattern_predictor" in individual_results:
            pattern_results = individual_results["pattern_predictor"]
            if pattern_results.get("success"):
                pattern_count = len([p for p in pattern_results.get("predictions", []) 
                                  if p.get("confidence_score", 0) > 0.7])
        
        pattern_quality_correlation = min(pattern_count * 0.1 + neural_quality, 1.0)
        correlations["pattern_quality"] = {
            "correlation_score": pattern_quality_correlation,
            "pattern_count": pattern_count,
            "quality": neural_quality
        }
        
        return correlations
    
    async def _generate_unified_recommendations(
        self, 
        unified_insights: Dict[str, Any],
        correlations: Dict[str, Any],
        request: UnifiedAnalysisRequest
    ) -> List[str]:
        """Generate unified recommendations from all models"""
        recommendations = []
        
        # Quality-based recommendations
        quality_score = unified_insights.get("code_quality_score", 0.5)
        if quality_score < 0.6:
            recommendations.append("Consider refactoring to improve code quality")
        elif quality_score > 0.9:
            recommendations.append("Excellent code quality - consider sharing as a template")
        
        # Complexity-based recommendations
        complexity = unified_insights.get("complexity_assessment", "unknown")
        if complexity in ["complex", "very_complex"]:
            recommendations.append("High complexity detected - consider breaking into smaller functions")
        
        # Pattern-based recommendations
        recommended_patterns = unified_insights.get("recommended_patterns", [])
        if recommended_patterns:
            top_pattern = max(recommended_patterns, key=lambda p: p.get("confidence", 0))
            recommendations.append(f"Consider implementing {top_pattern['pattern']} pattern")
        
        # Similarity-based recommendations
        similarity_findings = unified_insights.get("similarity_findings", [])
        high_similarity = [f for f in similarity_findings if f.get("similarity", 0) > 0.9]
        if high_similarity:
            recommendations.append("High code similarity detected - consider extracting common functionality")
        
        # Refactoring opportunities
        refactor_ops = unified_insights.get("refactoring_opportunities", [])
        if refactor_ops:
            top_refactor = max(refactor_ops, key=lambda r: r.get("confidence", 0))
            recommendations.append(f"Refactoring opportunity: {top_refactor['opportunity']}")
        
        # Cross-model correlation insights
        quality_complexity = correlations.get("quality_complexity", {})
        if quality_complexity.get("alignment") == "poor":
            recommendations.append("Quality-complexity alignment is poor - investigate potential issues")
        
        # Personalized recommendations
        personal_suggestions = unified_insights.get("personalized_suggestions", [])
        if personal_suggestions:
            top_personal = personal_suggestions[0] if personal_suggestions else {}
            if top_personal.get("confidence", 0) > 0.8:
                recommendations.append(f"Personalized suggestion: {top_personal.get('type', 'improvement')}")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def _calculate_unified_confidence(
        self, 
        individual_results: Dict[str, Any],
        correlations: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score for unified analysis"""
        confidence_scores = []
        
        # Extract individual model confidences
        for model_name, results in individual_results.items():
            if results.get("success"):
                if model_name == "neural_analyzer":
                    analyses = results.get("analyses", [])
                    model_confidences = [a.get("confidence_score", 0.5) for a in analyses]
                    if model_confidences:
                        confidence_scores.append(sum(model_confidences) / len(model_confidences))
                
                elif model_name == "pattern_predictor":
                    predictions = results.get("predictions", [])
                    model_confidences = [p.get("confidence_score", 0.5) for p in predictions]
                    if model_confidences:
                        confidence_scores.append(sum(model_confidences) / len(model_confidences))
                
                elif model_name == "similarity_engine":
                    matches = results.get("matches", [])
                    model_confidences = [m.get("confidence", 0.5) for m in matches]
                    if model_confidences:
                        confidence_scores.append(sum(model_confidences) / len(model_confidences))
                
                elif model_name == "learning_system":
                    suggestions = results.get("suggestions", [])
                    model_confidences = [s.get("confidence", 0.5) for s in suggestions]
                    if model_confidences:
                        confidence_scores.append(sum(model_confidences) / len(model_confidences))
        
        # Base confidence from individual models
        if confidence_scores:
            base_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            base_confidence = 0.5
        
        # Boost confidence based on cross-model correlations
        correlation_boost = 0.0
        for correlation_name, correlation_data in correlations.items():
            correlation_score = correlation_data.get("correlation_score", 0.5)
            if correlation_score > 0.8:
                correlation_boost += 0.05  # Small boost for good correlations
        
        # Final confidence
        unified_confidence = min(base_confidence + correlation_boost, 1.0)
        
        return unified_confidence
    
    async def _record_analysis_for_learning(
        self, 
        request: UnifiedAnalysisRequest, 
        result: UnifiedAnalysisResult
    ) -> None:
        """Record analysis for adaptive learning"""
        if not request.user_id or not self.learning_system:
            return
        
        try:
            interaction = UserInteraction(
                interaction_id=result.analysis_id,
                user_id=request.user_id,
                timestamp=result.timestamp,
                context={
                    "code_length": len(request.code),
                    "file_path": request.file_path,
                    "intelligence_level": request.intelligence_level,
                    "models_used": result.models_used
                },
                action_taken="unified_code_analysis",
                outcome="success" if result.confidence_score > 0.7 else "mixed"
            )
            
            await self.learning_system.record_interaction(interaction)
            
        except Exception as e:
            logger.error(f"Error recording analysis for learning: {e}")
    
    def _update_coordination_stats(self, result: UnifiedAnalysisResult, success: bool) -> None:
        """Update coordination statistics"""
        self.coordination_stats["total_analyses"] += 1
        
        if success:
            self.coordination_stats["successful_analyses"] += 1
        else:
            self.coordination_stats["failed_analyses"] += 1
        
        # Update average processing time
        current_avg = self.coordination_stats["average_processing_time"]
        total = self.coordination_stats["total_analyses"]
        new_avg = ((current_avg * (total - 1)) + result.processing_time) / total
        self.coordination_stats["average_processing_time"] = new_avg
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for optimization and maintenance"""
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_task())
        
        # Start model optimization
        asyncio.create_task(self._model_optimization_task())
    
    async def _performance_monitoring_task(self) -> None:
        """Background task for monitoring model performance"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Monitor active analyses
                current_time = asyncio.get_event_loop().time()
                for analysis_id, analysis_data in self.active_analyses.items():
                    analysis_time = current_time - analysis_data["start_time"]
                    if analysis_time > 60:  # 1 minute timeout
                        logger.warning(f"Long-running analysis detected: {analysis_id}")
                
                # Update model performance history
                for model_name, capability in self.registered_models.items():
                    performance_data = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "usage_count": self.coordination_stats["model_usage_counts"].get(model_name, 0),
                        "metrics": capability.performance_metrics
                    }
                    self.model_performance_history[model_name].append(performance_data)
                    
                    # Keep only last 100 entries
                    if len(self.model_performance_history[model_name]) > 100:
                        self.model_performance_history[model_name] = self.model_performance_history[model_name][-50:]
                
            except Exception as e:
                logger.error(f"Error in performance monitoring task: {e}")
    
    async def _model_optimization_task(self) -> None:
        """Background task for model optimization"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                if self.enable_performance_optimization:
                    # Trigger adaptive learning for models
                    if self.learning_system:
                        await self.learning_system.adapt_models()
                    
                    # Update model load balancing
                    await self.model_load_balancer.update_load_metrics(
                        self.model_performance_history
                    )
                
            except Exception as e:
                logger.error(f"Error in model optimization task: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the ML model coordination system"""
        status = {
            "coordinator_status": "active",
            "registered_models": {
                name: capability.to_dict() 
                for name, capability in self.registered_models.items()
            },
            "active_analyses": len(self.active_analyses),
            "coordination_stats": self.coordination_stats.copy(),
            "model_performance": {},
            "system_health": "good"
        }
        
        # Get model performance summaries
        for model_name, history in self.model_performance_history.items():
            if history:
                recent_usage = [entry["usage_count"] for entry in history[-10:]]
                status["model_performance"][model_name] = {
                    "recent_usage_avg": sum(recent_usage) / len(recent_usage) if recent_usage else 0,
                    "total_history_points": len(history)
                }
        
        # Determine system health
        success_rate = 0.0
        if self.coordination_stats["total_analyses"] > 0:
            success_rate = self.coordination_stats["successful_analyses"] / self.coordination_stats["total_analyses"]
        
        if success_rate > 0.9:
            status["system_health"] = "excellent"
        elif success_rate > 0.8:
            status["system_health"] = "good"
        elif success_rate > 0.6:
            status["system_health"] = "fair"
        else:
            status["system_health"] = "poor"
        
        return status
    
    async def shutdown(self) -> None:
        """Shutdown the ML model coordinator gracefully"""
        try:
            # Save learning state
            if self.learning_system:
                await self.learning_system.save_learning_state()
            
            # Clear active analyses
            self.active_analyses.clear()
            
            logger.info("MLModelCoordinator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during MLModelCoordinator shutdown: {e}")


# Helper classes

class ModelLoadBalancer:
    """Load balancer for ML models"""
    
    def __init__(self):
        self.load_metrics = defaultdict(float)
    
    async def balance_model_selection(
        self, 
        requested_models: List[str],
        performance_history: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Balance model selection based on load and performance"""
        # Simple load balancing - prioritize models with better performance
        model_scores = {}
        
        for model_name in requested_models:
            score = 1.0  # Base score
            
            # Adjust based on recent usage (lower usage = higher score)
            current_load = self.load_metrics.get(model_name, 0.0)
            score *= (1.0 - min(current_load, 0.5))  # Max 50% penalty for high load
            
            model_scores[model_name] = score
        
        # Sort by score and return
        sorted_models = sorted(
            model_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [model_name for model_name, _ in sorted_models]
    
    async def update_load_metrics(
        self, 
        performance_history: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Update load metrics based on recent usage"""
        for model_name, history in performance_history.items():
            if history and len(history) >= 2:
                recent_usage = [entry["usage_count"] for entry in history[-5:]]
                avg_usage = sum(recent_usage) / len(recent_usage)
                
                # Normalize load metric (0.0 = low load, 1.0 = high load)
                max_usage = max(recent_usage) if recent_usage else 1
                normalized_load = avg_usage / max(max_usage, 1)
                
                self.load_metrics[model_name] = normalized_load


class CrossModelLearner:
    """Learns from correlations between different models"""
    
    def __init__(self):
        self.correlation_history = defaultdict(list)
        self.learned_patterns = {}
    
    async def initialize(self, models: Dict[str, Any]) -> None:
        """Initialize cross-model learning"""
        self.models = models
        logger.info("CrossModelLearner initialized")
    
    async def learn_from_correlation(
        self, 
        model_results: Dict[str, Any],
        correlations: Dict[str, Any]
    ) -> None:
        """Learn from cross-model correlations"""
        # Record correlation patterns
        for correlation_name, correlation_data in correlations.items():
            correlation_score = correlation_data.get("correlation_score", 0.5)
            
            self.correlation_history[correlation_name].append({
                "score": correlation_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": correlation_data
            })
            
            # Keep only recent history
            if len(self.correlation_history[correlation_name]) > 100:
                self.correlation_history[correlation_name] = self.correlation_history[correlation_name][-50:]
    
    async def get_learned_insights(self) -> Dict[str, Any]:
        """Get insights learned from cross-model analysis"""
        insights = {
            "correlation_patterns": {},
            "model_synergies": {},
            "prediction_improvements": {}
        }
        
        # Analyze correlation patterns
        for correlation_name, history in self.correlation_history.items():
            if len(history) >= 10:
                scores = [entry["score"] for entry in history[-20:]]
                avg_score = sum(scores) / len(scores)
                trend = "improving" if scores[-5:] > scores[-10:-5] else "stable"
                
                insights["correlation_patterns"][correlation_name] = {
                    "average_correlation": avg_score,
                    "trend": trend,
                    "data_points": len(history)
                }
        
        return insights