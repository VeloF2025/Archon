"""
Pattern Marketplace API

Comprehensive API for pattern discovery, submission, validation,
and AI-powered recommendations with multi-provider deployment support.
"""

import asyncio
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...agents.patterns import (
    Pattern, PatternSearchRequest, PatternSubmission, PatternValidationResult,
    PatternRecommendation, PatternProvider, PatternType, PatternCategory,
    PatternComplexity, MultiProviderConfig
)
from ...agents.patterns import PatternAnalyzer, CommunityPatternValidator, MultiProviderEngine
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/patterns", tags=["patterns"])

# Initialize pattern services
pattern_analyzer = PatternAnalyzer()
pattern_validator = CommunityPatternValidator()
multi_provider_engine = MultiProviderEngine()

# In-memory storage for demo (in production, use database)
pattern_storage: Dict[str, Pattern] = {}
submission_storage: Dict[str, PatternSubmission] = {}
active_analyses: Dict[str, asyncio.Task] = {}


# API Models
class PatternAnalysisRequest(BaseModel):
    project_path: str
    source_info: Optional[Dict[str, Any]] = None


class PatternSubmissionRequest(BaseModel):
    pattern: Pattern
    submitter: str
    submitter_email: Optional[str] = None
    submission_notes: Optional[str] = None


class MultiProviderDeploymentRequest(BaseModel):
    pattern_id: str
    target_providers: List[PatternProvider]
    provider_configs: Optional[Dict[str, Dict[str, Any]]] = None


class PatternRecommendationRequest(BaseModel):
    project_technologies: List[str]
    project_type: Optional[PatternType] = None
    complexity_preference: Optional[PatternComplexity] = None
    budget_limit: Optional[float] = None
    provider_preference: Optional[List[PatternProvider]] = None


# Pattern Discovery and Search
@router.get("/", response_model=Dict[str, Any])
async def list_patterns(
    category: Optional[PatternCategory] = Query(None),
    type_: Optional[PatternType] = Query(None, alias="type"),
    complexity: Optional[PatternComplexity] = Query(None),
    provider: Optional[PatternProvider] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List patterns with optional filtering."""
    try:
        # Filter patterns based on criteria
        filtered_patterns = []
        
        for pattern in pattern_storage.values():
            if category and pattern.metadata.category != category:
                continue
            if type_ and pattern.metadata.type != type_:
                continue
            if complexity and pattern.metadata.complexity != complexity:
                continue
            if provider and provider not in pattern.metadata.providers:
                continue
            
            filtered_patterns.append(pattern)
        
        # Sort by rating (descending)
        filtered_patterns.sort(
            key=lambda p: p.metadata.metrics.rating,
            reverse=True
        )
        
        # Apply pagination
        total = len(filtered_patterns)
        paginated = filtered_patterns[offset:offset + limit]
        
        return {
            "success": True,
            "patterns": [p.dict() for p in paginated],
            "pagination": {
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total
            }
        }
        
    except Exception as e:
        logger.error(f"Pattern listing failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Pattern listing failed: {str(e)}")


@router.post("/search", response_model=Dict[str, Any])
async def search_patterns(request: PatternSearchRequest):
    """Advanced pattern search with multiple criteria."""
    try:
        filtered_patterns = []
        
        for pattern in pattern_storage.values():
            # Text search in name and description
            if request.query:
                query_lower = request.query.lower()
                if (query_lower not in pattern.metadata.name.lower() and
                    query_lower not in pattern.metadata.description.lower() and
                    not any(query_lower in tag.lower() for tag in pattern.metadata.tags)):
                    continue
            
            # Filter by criteria
            if request.type and pattern.metadata.type != request.type:
                continue
            if request.category and pattern.metadata.category != request.category:
                continue
            if request.complexity and pattern.metadata.complexity != request.complexity:
                continue
            if request.providers and not any(p in pattern.metadata.providers for p in request.providers):
                continue
            if request.technologies and not any(
                tech.name.lower() in [t.lower() for t in request.technologies]
                for tech in pattern.metadata.technologies
            ):
                continue
            if request.tags and not any(tag in pattern.metadata.tags for tag in request.tags):
                continue
            if request.min_rating and pattern.metadata.metrics.rating < request.min_rating:
                continue
            
            filtered_patterns.append(pattern)
        
        # Sort patterns
        sort_key_map = {
            "rating": lambda p: p.metadata.metrics.rating,
            "downloads": lambda p: p.metadata.metrics.downloads,
            "created_at": lambda p: p.metadata.created_at,
            "updated_at": lambda p: p.metadata.updated_at,
            "name": lambda p: p.metadata.name.lower()
        }
        
        if request.sort_by in sort_key_map:
            filtered_patterns.sort(
                key=sort_key_map[request.sort_by],
                reverse=(request.sort_order == "desc")
            )
        
        # Apply pagination
        total = len(filtered_patterns)
        paginated = filtered_patterns[request.offset:request.offset + request.limit]
        
        return {
            "success": True,
            "results": [p.dict() for p in paginated],
            "pagination": {
                "total": total,
                "offset": request.offset,
                "limit": request.limit,
                "has_more": request.offset + request.limit < total
            }
        }
        
    except Exception as e:
        logger.error(f"Pattern search failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/{pattern_id}", response_model=Pattern)
async def get_pattern(pattern_id: str):
    """Get a specific pattern by ID."""
    try:
        if pattern_id not in pattern_storage:
            raise HTTPException(status_code=404, detail=f"Pattern '{pattern_id}' not found")
        
        pattern = pattern_storage[pattern_id]
        
        # Increment download count
        pattern.metadata.metrics.downloads += 1
        
        return pattern
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pattern | id={pattern_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pattern: {str(e)}")


# Pattern Analysis and Extraction
@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_project_for_patterns(
    request: PatternAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze a project and extract architectural patterns."""
    try:
        analysis_id = str(uuid.uuid4())
        
        logger.info(f"Starting pattern analysis | analysis_id={analysis_id} | path={request.project_path}")
        
        # Start background analysis task
        task = asyncio.create_task(
            _perform_pattern_analysis(analysis_id, request.project_path, request.source_info)
        )
        active_analyses[analysis_id] = task
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "message": f"Pattern analysis started for project: {request.project_path}",
            "estimated_time": "30-90 seconds"
        }
        
    except Exception as e:
        logger.error(f"Failed to start pattern analysis | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed to start: {str(e)}")


@router.get("/analyze/{analysis_id}/status", response_model=Dict[str, Any])
async def get_analysis_status(analysis_id: str):
    """Get the status of a pattern analysis."""
    try:
        if analysis_id not in active_analyses:
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found")
        
        task = active_analyses[analysis_id]
        
        if task.done():
            # Analysis completed
            try:
                result = task.result()
                # Clean up completed task
                del active_analyses[analysis_id]
                
                return {
                    "success": True,
                    "status": "completed",
                    "result": result
                }
            except Exception as e:
                del active_analyses[analysis_id]
                return {
                    "success": False,
                    "status": "failed",
                    "error": str(e)
                }
        else:
            return {
                "success": True,
                "status": "running",
                "message": "Pattern analysis in progress..."
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis status | id={analysis_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# Pattern Submission and Community Features
@router.post("/submit", response_model=Dict[str, Any])
async def submit_pattern(request: PatternSubmissionRequest):
    """Submit a pattern for community validation."""
    try:
        submission = PatternSubmission(
            pattern=request.pattern,
            submitter=request.submitter,
            submitter_email=request.submitter_email,
            submission_notes=request.submission_notes
        )
        
        logger.info(f"Pattern submitted | submission_id={submission.id} | pattern={request.pattern.id}")
        
        # Validate the submitted pattern
        validation_result = await pattern_validator.validate_pattern_submission(submission)
        submission.validation_result = validation_result
        
        # Update submission status based on validation
        if validation_result.valid:
            if validation_result.confidence_score >= 0.9:
                submission.status = "approved"
                # Add to pattern storage
                pattern_storage[request.pattern.id] = request.pattern
            else:
                submission.status = "under_review"
        else:
            submission.status = "needs_revision"
        
        # Store submission
        submission_storage[submission.id] = submission
        
        return {
            "success": True,
            "submission_id": submission.id,
            "status": submission.status,
            "validation_result": validation_result.dict(),
            "message": f"Pattern submitted successfully with status: {submission.status}"
        }
        
    except Exception as e:
        logger.error(f"Pattern submission failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


@router.get("/submissions/{submission_id}", response_model=PatternSubmission)
async def get_submission(submission_id: str):
    """Get a pattern submission by ID."""
    try:
        if submission_id not in submission_storage:
            raise HTTPException(status_code=404, detail=f"Submission '{submission_id}' not found")
        
        return submission_storage[submission_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get submission | id={submission_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve submission: {str(e)}")


@router.post("/{pattern_id}/validate", response_model=PatternValidationResult)
async def validate_pattern(pattern_id: str):
    """Validate a pattern for quality and security."""
    try:
        if pattern_id not in pattern_storage:
            raise HTTPException(status_code=404, detail=f"Pattern '{pattern_id}' not found")
        
        pattern = pattern_storage[pattern_id]
        validation_result = await pattern_validator.validate_community_pattern(pattern)
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern validation failed | id={pattern_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# Multi-Provider Deployment
@router.post("/deploy", response_model=Dict[str, Any])
async def generate_deployment_plans(request: MultiProviderDeploymentRequest):
    """Generate deployment plans for multiple providers."""
    try:
        if request.pattern_id not in pattern_storage:
            raise HTTPException(status_code=404, detail=f"Pattern '{request.pattern_id}' not found")
        
        pattern = pattern_storage[request.pattern_id]
        
        logger.info(f"Generating deployment plans | pattern={request.pattern_id} | providers={len(request.target_providers)}")
        
        # Generate deployment plans
        deployment_plans = await multi_provider_engine.generate_deployment_plans(
            pattern=pattern,
            target_providers=request.target_providers,
            provider_configs=request.provider_configs
        )
        
        # Convert to serializable format
        plans_dict = {}
        for provider, plan in deployment_plans.items():
            plans_dict[provider.value] = {
                "provider": provider.value,
                "resources": [
                    {
                        "name": r.name,
                        "type": r.type,
                        "provider_specific_name": r.provider_specific_name,
                        "cost_estimate": r.cost_estimate
                    }
                    for r in plan.resources
                ],
                "deployment_scripts": plan.deployment_scripts,
                "configuration_files": plan.configuration_files,
                "estimated_cost": plan.estimated_cost,
                "deployment_time_estimate": plan.deployment_time_estimate,
                "prerequisites": plan.prerequisites,
                "post_deployment_steps": plan.post_deployment_steps
            }
        
        return {
            "success": True,
            "pattern_id": request.pattern_id,
            "deployment_plans": plans_dict,
            "total_providers": len(deployment_plans)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deployment plan generation failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Deployment planning failed: {str(e)}")


@router.get("/deploy/{pattern_id}/cost-comparison", response_model=Dict[str, Any])
async def compare_provider_costs(pattern_id: str):
    """Compare costs across different providers for a pattern."""
    try:
        if pattern_id not in pattern_storage:
            raise HTTPException(status_code=404, detail=f"Pattern '{pattern_id}' not found")
        
        pattern = pattern_storage[pattern_id]
        
        # Compare costs across all supported providers
        providers = [PatternProvider.AWS, PatternProvider.GCP, PatternProvider.AZURE]
        cost_comparison = await multi_provider_engine.compare_provider_costs(pattern, providers)
        
        # Get provider recommendations
        recommendations = await multi_provider_engine.get_provider_recommendations(pattern)
        
        return {
            "success": True,
            "pattern_id": pattern_id,
            "cost_comparison": {
                provider.value: cost for provider, cost in cost_comparison.items()
            },
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost comparison failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Cost comparison failed: {str(e)}")


# AI-Powered Recommendations
@router.post("/recommend", response_model=Dict[str, Any])
async def get_pattern_recommendations(request: PatternRecommendationRequest):
    """Get AI-powered pattern recommendations based on project requirements."""
    try:
        logger.info(f"Generating pattern recommendations | technologies={len(request.project_technologies)}")
        
        # Analyze available patterns and match to requirements
        recommendations = []
        
        for pattern in pattern_storage.values():
            score = 0.0
            reasons = []
            
            # Technology compatibility
            pattern_techs = {tech.name.lower() for tech in pattern.metadata.technologies}
            request_techs = {tech.lower() for tech in request.project_technologies}
            
            tech_overlap = len(pattern_techs.intersection(request_techs))
            tech_compatibility = tech_overlap / max(len(request_techs), 1) if request_techs else 0
            
            if tech_compatibility > 0:
                score += tech_compatibility * 0.4
                reasons.append(f"Technology match: {tech_overlap}/{len(request_techs)} technologies")
            
            # Pattern type match
            if request.project_type and pattern.metadata.type == request.project_type:
                score += 0.2
                reasons.append(f"Exact pattern type match: {request.project_type.value}")
            
            # Complexity preference
            if request.complexity_preference and pattern.metadata.complexity == request.complexity_preference:
                score += 0.1
                reasons.append(f"Complexity preference match: {request.complexity_preference.value}")
            
            # Provider preference
            if request.provider_preference:
                provider_overlap = len(
                    set(pattern.metadata.providers).intersection(set(request.provider_preference))
                )
                if provider_overlap > 0:
                    score += 0.1
                    reasons.append(f"Provider compatibility: {provider_overlap} providers")
            
            # Budget consideration (if provided)
            if request.budget_limit and pattern.metadata.providers:
                # Estimate cost for the cheapest provider
                provider_costs = await multi_provider_engine.compare_provider_costs(
                    pattern, list(pattern.metadata.providers)
                )
                if provider_costs:
                    min_cost = min(provider_costs.values())
                    if min_cost <= request.budget_limit:
                        score += 0.1
                        reasons.append(f"Within budget: ${min_cost:.2f}/month")
                    else:
                        score *= 0.5  # Penalize over-budget patterns
            
            # Quality factors
            quality_score = (
                pattern.metadata.metrics.rating / 5.0 * 0.1 +
                min(pattern.metadata.metrics.downloads / 100, 1.0) * 0.05 +
                pattern.metadata.metrics.success_rate * 0.05
            )
            score += quality_score
            
            if pattern.metadata.metrics.rating >= 4.0:
                reasons.append(f"High quality: {pattern.metadata.metrics.rating}/5.0 rating")
            
            # Only include patterns with reasonable scores
            if score >= 0.2:
                recommendations.append({
                    "pattern_id": pattern.id,
                    "pattern_name": pattern.metadata.name,
                    "confidence_score": score,
                    "reasons": reasons,
                    "pattern_type": pattern.metadata.type.value,
                    "complexity": pattern.metadata.complexity.value,
                    "technologies": [tech.name for tech in pattern.metadata.technologies],
                    "rating": pattern.metadata.metrics.rating,
                    "estimated_setup_time": pattern.metadata.metrics.avg_setup_time
                })
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        # Limit to top 10 recommendations
        recommendations = recommendations[:10]
        
        return {
            "success": True,
            "recommendations": recommendations,
            "total_patterns_analyzed": len(pattern_storage),
            "matching_patterns": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Pattern recommendations failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


# Pattern Statistics and Analytics
@router.get("/stats", response_model=Dict[str, Any])
async def get_pattern_statistics():
    """Get pattern marketplace statistics."""
    try:
        stats = {
            "total_patterns": len(pattern_storage),
            "total_submissions": len(submission_storage),
            "patterns_by_category": {},
            "patterns_by_type": {},
            "patterns_by_complexity": {},
            "patterns_by_provider": {},
            "top_technologies": {},
            "average_rating": 0.0,
            "total_downloads": 0
        }
        
        # Calculate statistics
        category_counts = {}
        type_counts = {}
        complexity_counts = {}
        provider_counts = {}
        technology_counts = {}
        total_rating = 0.0
        total_downloads = 0
        
        for pattern in pattern_storage.values():
            # Category stats
            category = pattern.metadata.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Type stats
            type_val = pattern.metadata.type.value
            type_counts[type_val] = type_counts.get(type_val, 0) + 1
            
            # Complexity stats
            complexity = pattern.metadata.complexity.value
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
            # Provider stats
            for provider in pattern.metadata.providers:
                provider_counts[provider.value] = provider_counts.get(provider.value, 0) + 1
            
            # Technology stats
            for tech in pattern.metadata.technologies:
                technology_counts[tech.name] = technology_counts.get(tech.name, 0) + 1
            
            # Rating and downloads
            total_rating += pattern.metadata.metrics.rating
            total_downloads += pattern.metadata.metrics.downloads
        
        stats["patterns_by_category"] = category_counts
        stats["patterns_by_type"] = type_counts
        stats["patterns_by_complexity"] = complexity_counts
        stats["patterns_by_provider"] = provider_counts
        stats["top_technologies"] = dict(sorted(technology_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        stats["average_rating"] = total_rating / max(len(pattern_storage), 1)
        stats["total_downloads"] = total_downloads
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get pattern statistics | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")


# Helper Functions
async def _perform_pattern_analysis(analysis_id: str, project_path: str, source_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform pattern analysis in background."""
    try:
        patterns = await pattern_analyzer.analyze_project_for_patterns(project_path, source_info)
        
        # Store discovered patterns (in production, save to database)
        for pattern in patterns:
            pattern_storage[pattern.id] = pattern
        
        logger.info(f"Pattern analysis completed | analysis_id={analysis_id} | patterns_found={len(patterns)}")
        
        return {
            "analysis_id": analysis_id,
            "patterns_discovered": len(patterns),
            "patterns": [p.dict() for p in patterns]
        }
        
    except Exception as e:
        logger.error(f"Pattern analysis failed | analysis_id={analysis_id} | error={str(e)}")
        raise


# Health Check
@router.get("/health", response_model=Dict[str, Any])
async def pattern_health():
    """Pattern API health check."""
    try:
        return {
            "status": "healthy",
            "service": "pattern-api",
            "patterns_count": len(pattern_storage),
            "submissions_count": len(submission_storage),
            "active_analyses": len(active_analyses)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "pattern-api",
            "error": str(e)
        }