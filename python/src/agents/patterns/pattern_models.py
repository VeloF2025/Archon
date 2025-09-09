"""
Pattern Models for Architectural Pattern Recognition

Defines data structures for patterns extracted from crawled projects
and submitted by the community.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator


class PatternType(str, Enum):
    """Types of architectural patterns."""
    
    # Architecture Patterns
    MICROSERVICES = "microservices"
    MONOLITHIC = "monolithic"
    SERVERLESS = "serverless"
    JAMSTACK = "jamstack"
    
    # Design Patterns
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    REPOSITORY = "repository"
    FACTORY = "factory"
    OBSERVER = "observer"
    COMMAND = "command"
    
    # Infrastructure Patterns
    CONTAINER_ORCHESTRATION = "container_orchestration"
    CI_CD_PIPELINE = "cicd_pipeline"
    INFRASTRUCTURE_AS_CODE = "infrastructure_as_code"
    
    # Data Patterns
    DATABASE_PER_SERVICE = "database_per_service"
    EVENT_SOURCING = "event_sourcing"
    CQRS = "cqrs"
    
    # Security Patterns
    OAUTH2_FLOW = "oauth2_flow"
    JWT_AUTHENTICATION = "jwt_authentication"
    API_GATEWAY = "api_gateway"
    
    # Frontend Patterns
    COMPONENT_LIBRARY = "component_library"
    STATE_MANAGEMENT = "state_management"
    ROUTING_PATTERN = "routing_pattern"
    
    # Custom/Community
    CUSTOM = "custom"


class PatternComplexity(str, Enum):
    """Complexity levels for patterns."""
    
    BEGINNER = "beginner"        # Simple, single-service patterns
    INTERMEDIATE = "intermediate" # Multi-service, moderate complexity
    ADVANCED = "advanced"        # Complex, enterprise-level patterns
    EXPERT = "expert"           # Highly specialized, cutting-edge patterns


class PatternCategory(str, Enum):
    """Categories for organizing patterns."""
    
    ARCHITECTURE = "architecture"
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    DEVOPS = "devops"
    SECURITY = "security"
    TESTING = "testing"
    MONITORING = "monitoring"
    MOBILE = "mobile"
    AI_ML = "ai_ml"


class PatternProvider(str, Enum):
    """Supported cloud/infrastructure providers."""
    
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    VERCEL = "vercel"
    NETLIFY = "netlify"
    SUPABASE = "supabase"
    FIREBASE = "firebase"
    PLANETSCALE = "planetscale"
    RAILWAY = "railway"
    RENDER = "render"
    AGNOSTIC = "provider_agnostic"  # Works with any provider


class PatternTechnology(BaseModel):
    """Technology used in a pattern."""
    
    name: str
    version: Optional[str] = None
    category: str  # e.g., "framework", "database", "language"
    required: bool = True
    alternatives: List[str] = Field(default_factory=list)


class PatternMetrics(BaseModel):
    """Metrics for pattern usage and performance."""
    
    downloads: int = 0
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    rating_count: int = 0
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    community_votes: int = 0
    
    # Performance metrics
    avg_setup_time: Optional[float] = None  # minutes
    avg_build_time: Optional[float] = None  # minutes
    complexity_score: Optional[float] = None  # 0.0-1.0


class PatternDetectionSource(BaseModel):
    """Information about where a pattern was detected."""
    
    source_url: Optional[str] = None
    repository: Optional[str] = None
    project_name: Optional[str] = None
    detection_confidence: float = Field(ge=0.0, le=1.0)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    analyzer_version: str = "1.0.0"


class PatternMetadata(BaseModel):
    """Metadata for architectural patterns."""
    
    name: str = Field(min_length=3, max_length=100)
    description: str = Field(min_length=10, max_length=500)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9\-]+)?$")
    
    # Classification
    type: PatternType
    category: PatternCategory
    complexity: PatternComplexity
    
    # Authorship
    author: str
    organization: Optional[str] = None
    license: str = "MIT"
    
    # Provider support
    providers: List[PatternProvider] = Field(default_factory=lambda: [PatternProvider.AGNOSTIC])
    
    # Technologies
    technologies: List[PatternTechnology] = Field(default_factory=list)
    
    # Tagging and discovery
    tags: List[str] = Field(default_factory=list, max_items=20)
    keywords: List[str] = Field(default_factory=list, max_items=30)
    
    # Metrics
    metrics: PatternMetrics = Field(default_factory=PatternMetrics)
    
    # Detection info
    detection_source: Optional[PatternDetectionSource] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified: Optional[datetime] = None
    
    @validator('tags', 'keywords')
    def validate_lists(cls, v):
        """Ensure tag and keyword lists contain unique, non-empty strings."""
        if not v:
            return v
        # Remove duplicates and empty strings
        unique_items = list(set(item.strip() for item in v if item.strip()))
        return unique_items


class PatternComponent(BaseModel):
    """A component/service within a pattern."""
    
    name: str
    type: str  # e.g., "service", "database", "frontend", "worker"
    description: str
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)
    optional: bool = False
    
    # Implementation
    technologies: List[PatternTechnology] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    
    # Deployment
    scaling: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None


class PatternWorkflow(BaseModel):
    """Workflow step in pattern implementation."""
    
    step: int
    name: str
    description: str
    commands: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # Validation
    validation_commands: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    
    # Timing
    estimated_duration: Optional[int] = None  # seconds
    can_parallel: bool = False


class Pattern(BaseModel):
    """Complete architectural pattern definition."""
    
    id: str = Field(default_factory=lambda: f"pattern_{uuid.uuid4().hex[:12]}")
    metadata: PatternMetadata
    
    # Structure
    components: List[PatternComponent] = Field(default_factory=list)
    relationships: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Implementation
    workflows: List[PatternWorkflow] = Field(default_factory=list)
    
    # Template integration
    generates_templates: List[str] = Field(default_factory=list)  # Template IDs
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Documentation
    documentation_url: Optional[str] = None
    examples: List[str] = Field(default_factory=list)  # URLs to example implementations
    tutorials: List[str] = Field(default_factory=list)  # URLs to tutorials
    
    # Validation
    validation_rules: List[str] = Field(default_factory=list)
    test_suite_url: Optional[str] = None
    
    @validator('id')
    def validate_id(cls, v):
        """Ensure ID follows pattern naming convention."""
        if not v.startswith('pattern_'):
            raise ValueError("Pattern ID must start with 'pattern_'")
        return v


class PatternValidationResult(BaseModel):
    """Result of pattern validation."""
    
    pattern_id: str
    valid: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    
    # Validation details
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    
    # Community validation
    community_votes: Dict[str, int] = Field(default_factory=dict)  # "approve", "reject", "needs_work"
    expert_reviews: List[str] = Field(default_factory=list)
    
    # Performance validation
    implementation_tested: bool = False
    performance_benchmarks: Optional[Dict[str, Any]] = None
    
    # Validation timestamp
    validated_at: datetime = Field(default_factory=datetime.utcnow)
    validator_version: str = "1.0.0"


class PatternSubmission(BaseModel):
    """Community pattern submission."""
    
    id: str = Field(default_factory=lambda: f"submission_{uuid.uuid4().hex[:12]}")
    pattern: Pattern
    
    # Submission details
    submitter: str
    submitter_email: Optional[str] = None
    submission_notes: Optional[str] = None
    
    # Review status
    status: str = "pending"  # pending, under_review, approved, rejected
    assigned_reviewer: Optional[str] = None
    
    # Validation
    validation_result: Optional[PatternValidationResult] = None
    
    # Timeline
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    
    @validator('status')
    def validate_status(cls, v):
        """Ensure status is valid."""
        valid_statuses = ["pending", "under_review", "approved", "rejected", "needs_revision"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v


class PatternSearchRequest(BaseModel):
    """Request model for pattern search."""
    
    query: Optional[str] = None
    type: Optional[PatternType] = None
    category: Optional[PatternCategory] = None
    complexity: Optional[PatternComplexity] = None
    providers: Optional[List[PatternProvider]] = None
    technologies: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    
    # Sorting
    sort_by: str = "rating"  # rating, downloads, created_at, updated_at, name
    sort_order: str = "desc"  # asc, desc
    
    # Pagination
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class PatternRecommendation(BaseModel):
    """AI-powered pattern recommendation."""
    
    pattern_id: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    reason: str
    
    # Context that led to recommendation
    project_analysis: Optional[Dict[str, Any]] = None
    similar_patterns: List[str] = Field(default_factory=list)
    
    # Customization suggestions
    variable_suggestions: Dict[str, Any] = Field(default_factory=dict)
    provider_recommendations: List[PatternProvider] = Field(default_factory=list)


class MultiProviderConfig(BaseModel):
    """Multi-provider configuration for patterns."""
    
    provider: PatternProvider
    configuration: Dict[str, Any] = Field(default_factory=dict)
    
    # Provider-specific resources
    resource_mapping: Dict[str, str] = Field(default_factory=dict)
    deployment_scripts: List[str] = Field(default_factory=list)
    
    # Cost and performance estimates
    estimated_cost: Optional[Dict[str, float]] = None  # monthly estimates
    performance_characteristics: Optional[Dict[str, Any]] = None
    
    # Limitations or special considerations
    limitations: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)