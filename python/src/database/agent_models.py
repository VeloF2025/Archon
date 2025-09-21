"""
Pydantic Models for Archon 3.0 Intelligence-Tiered Agent Management

This module defines Pydantic models for the Intelligence-Tiered Adaptive Agent Management System:
- Agent Lifecycle Management with 5-state system
- Intelligence Tier Routing (Opus/Sonnet/Haiku) 
- Knowledge Management with confidence-based learning
- Cost Optimization with budget tracking and ROI analysis
- Real-Time Collaboration with pub/sub messaging
- Global Rules Integration from configuration files

Created from: archon_3_0_agent_management_schema.sql
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


# =====================================================
# ENUMS FOR TYPE SAFETY
# =====================================================

class AgentState(str, Enum):
    """Agent lifecycle states"""
    CREATED = "CREATED"          # Initial state after agent creation
    ACTIVE = "ACTIVE"            # Currently processing tasks
    IDLE = "IDLE"                # Available but not processing
    HIBERNATED = "HIBERNATED"    # Temporarily suspended to save resources
    ARCHIVED = "ARCHIVED"        # Permanently deactivated


class ModelTier(str, Enum):
    """Intelligence model tiers for task routing"""
    OPUS = "OPUS"       # Highest capability, most expensive
    SONNET = "SONNET"   # Balanced performance and cost (default)
    HAIKU = "HAIKU"     # Basic tasks only, most cost-effective


class AgentType(str, Enum):
    """Specialized agent types for different development tasks"""
    CODE_IMPLEMENTER = "CODE_IMPLEMENTER"
    SYSTEM_ARCHITECT = "SYSTEM_ARCHITECT"
    CODE_QUALITY_REVIEWER = "CODE_QUALITY_REVIEWER"
    TEST_COVERAGE_VALIDATOR = "TEST_COVERAGE_VALIDATOR"
    SECURITY_AUDITOR = "SECURITY_AUDITOR"
    PERFORMANCE_OPTIMIZER = "PERFORMANCE_OPTIMIZER"
    DEPLOYMENT_AUTOMATION = "DEPLOYMENT_AUTOMATION"
    ANTIHALLUCINATION_VALIDATOR = "ANTIHALLUCINATION_VALIDATOR"
    UI_UX_OPTIMIZER = "UI_UX_OPTIMIZER"
    DATABASE_ARCHITECT = "DATABASE_ARCHITECT"
    DOCUMENTATION_GENERATOR = "DOCUMENTATION_GENERATOR"
    CODE_REFACTORING_OPTIMIZER = "CODE_REFACTORING_OPTIMIZER"
    STRATEGIC_PLANNER = "STRATEGIC_PLANNER"
    API_DESIGN_ARCHITECT = "API_DESIGN_ARCHITECT"
    GENERAL_PURPOSE = "GENERAL_PURPOSE"


# =====================================================
# AGENT LIFECYCLE MANAGEMENT MODELS
# =====================================================

class AgentV3(BaseModel):
    """Core agent model for Intelligence-Tiered system"""
    
    # Basic identification
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., max_length=255, description="Agent display name")
    agent_type: AgentType = Field(..., description="Specialized agent type")
    model_tier: ModelTier = Field(default=ModelTier.SONNET, description="Intelligence tier assignment")
    project_id: UUID = Field(..., description="Associated project UUID")
    
    # Lifecycle management
    state: AgentState = Field(default=AgentState.CREATED, description="Current agent state")
    state_changed_at: datetime = Field(default_factory=datetime.now)
    
    # Performance tracking
    tasks_completed: int = Field(default=0, ge=0)
    success_rate: Decimal = Field(default=Decimal('0.0000'), ge=Decimal('0'), le=Decimal('1'))
    avg_completion_time_seconds: int = Field(default=0, ge=0)
    last_active_at: Optional[datetime] = None
    
    # Resource management
    memory_usage_mb: int = Field(default=0, ge=0)
    cpu_usage_percent: Decimal = Field(default=Decimal('0.00'), ge=Decimal('0'), le=Decimal('100'))
    
    # Configuration
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    rules_profile_id: Optional[UUID] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
            Decimal: str
        }


class AgentStateHistory(BaseModel):
    """Tracks agent lifecycle state transitions"""
    
    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID = Field(..., description="Agent that changed state")
    
    from_state: Optional[AgentState] = None
    to_state: AgentState = Field(..., description="New state")
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    changed_at: datetime = Field(default_factory=datetime.now)
    changed_by: str = Field(default="system", description="Who/what triggered the change")


class AgentPool(BaseModel):
    """Manages agent pool limits and scaling per project"""
    
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID = Field(..., description="Project this pool serves")
    
    # Pool limits per tier
    opus_limit: int = Field(default=2, ge=0, description="Max Opus agents")
    sonnet_limit: int = Field(default=10, ge=0, description="Max Sonnet agents")
    haiku_limit: int = Field(default=50, ge=0, description="Max Haiku agents")
    
    # Current active counts
    opus_active: int = Field(default=0, ge=0)
    sonnet_active: int = Field(default=0, ge=0)
    haiku_active: int = Field(default=0, ge=0)
    
    # Pool settings
    auto_scaling_enabled: bool = Field(default=True)
    hibernation_timeout_minutes: int = Field(default=30, ge=1)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('opus_active')
    def opus_within_limit(cls, v, values):
        if 'opus_limit' in values and v > values['opus_limit']:
            raise ValueError(f"Opus active ({v}) cannot exceed limit ({values['opus_limit']})")
        return v
    
    @validator('sonnet_active')
    def sonnet_within_limit(cls, v, values):
        if 'sonnet_limit' in values and v > values['sonnet_limit']:
            raise ValueError(f"Sonnet active ({v}) cannot exceed limit ({values['sonnet_limit']})")
        return v
    
    @validator('haiku_active')
    def haiku_within_limit(cls, v, values):
        if 'haiku_limit' in values and v > values['haiku_limit']:
            raise ValueError(f"Haiku active ({v}) cannot exceed limit ({values['haiku_limit']})")
        return v


# =====================================================
# INTELLIGENCE TIER ROUTING MODELS
# =====================================================

class TaskComplexity(BaseModel):
    """Assesses task complexity for tier routing"""
    
    id: UUID = Field(default_factory=uuid4)
    task_id: UUID = Field(..., description="Task being assessed")
    
    # Complexity metrics (0.0 to 1.0)
    technical_complexity: Decimal = Field(default=Decimal('0.5000'), ge=Decimal('0'), le=Decimal('1'))
    domain_expertise_required: Decimal = Field(default=Decimal('0.5000'), ge=Decimal('0'), le=Decimal('1'))
    code_volume_complexity: Decimal = Field(default=Decimal('0.5000'), ge=Decimal('0'), le=Decimal('1'))
    integration_complexity: Decimal = Field(default=Decimal('0.5000'), ge=Decimal('0'), le=Decimal('1'))
    
    # Automatically calculated overall complexity (computed in database)
    overall_complexity: Optional[Decimal] = None
    
    # Tier assignment
    recommended_tier: ModelTier = Field(..., description="Recommended tier based on complexity")
    assigned_tier: ModelTier = Field(..., description="Actually assigned tier")
    tier_justification: Optional[str] = None
    
    # Assessment metadata
    assessed_by: str = Field(default="system")
    assessed_at: datetime = Field(default_factory=datetime.now)


class RoutingRule(BaseModel):
    """Intelligence tier routing rules and thresholds"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Rule definition
    rule_name: str = Field(..., max_length=255, description="Unique rule identifier")
    rule_description: Optional[str] = None
    
    # Tier thresholds (complexity scores 0.0-1.0)
    opus_threshold: Decimal = Field(default=Decimal('0.7500'), ge=Decimal('0'), le=Decimal('1'))
    sonnet_threshold: Decimal = Field(default=Decimal('0.1500'), ge=Decimal('0'), le=Decimal('1'))  # Sonnet-first
    haiku_threshold: Decimal = Field(default=Decimal('0.0000'), ge=Decimal('0'), le=Decimal('1'))
    
    # Special routing conditions
    agent_type_preferences: Dict[str, str] = Field(default_factory=dict)
    project_tier_override: Dict[str, str] = Field(default_factory=dict)
    
    # Rule status
    is_active: bool = Field(default=True)
    priority_order: int = Field(default=1, ge=1)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# =====================================================
# KNOWLEDGE MANAGEMENT MODELS
# =====================================================

class AgentKnowledge(BaseModel):
    """Knowledge items learned by agents with confidence evolution"""
    
    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID = Field(..., description="Agent that owns this knowledge")
    
    # Knowledge content
    knowledge_type: str = Field(..., max_length=100, description="Type: pattern, solution, error, context")
    title: str = Field(..., max_length=500)
    content: str = Field(..., description="Detailed knowledge content")
    
    # Confidence and learning
    confidence: Decimal = Field(default=Decimal('0.5000'), ge=Decimal('0'), le=Decimal('1'))
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    last_used_at: Optional[datetime] = None
    
    # Context and categorization
    context_tags: List[str] = Field(default_factory=list)
    project_id: UUID = Field(..., description="Project context")
    task_context: Optional[str] = None
    
    # Storage layer (temporary, working, long_term)
    storage_layer: str = Field(default="temporary", description="Knowledge persistence layer")
    
    # Vector embedding for similarity search (handled by database)
    embedding: Optional[List[float]] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class KnowledgeEvolution(BaseModel):
    """Tracks how agent knowledge confidence evolves over time"""
    
    id: UUID = Field(default_factory=uuid4)
    knowledge_id: UUID = Field(..., description="Knowledge item that evolved")
    
    # Evolution tracking
    previous_confidence: Optional[Decimal] = None
    new_confidence: Optional[Decimal] = None
    evolution_reason: str = Field(..., max_length=255, description="success, failure, validation, consolidation")
    
    # Context
    task_id: Optional[UUID] = None
    agent_feedback: Optional[str] = None
    
    evolved_at: datetime = Field(default_factory=datetime.now)


class SharedKnowledge(BaseModel):
    """Cross-agent validated knowledge patterns"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Knowledge identification
    knowledge_pattern: str = Field(..., max_length=500)
    solution_approach: str = Field(..., description="Detailed solution")
    
    # Multi-agent validation
    contributing_agents: List[UUID] = Field(default_factory=list)
    validation_count: int = Field(default=0, ge=0)
    success_rate: Decimal = Field(default=Decimal('0.0000'), ge=Decimal('0'), le=Decimal('1'))
    
    # Applicability
    applicable_agent_types: List[AgentType] = Field(default_factory=list)
    applicable_contexts: List[str] = Field(default_factory=list)
    
    # Vector search (handled by database)
    embedding: Optional[List[float]] = None
    
    # Status
    is_verified: bool = Field(default=False)
    verification_threshold: int = Field(default=3, ge=1)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# =====================================================
# COST OPTIMIZATION MODELS
# =====================================================

class CostTracking(BaseModel):
    """Tracks token usage and costs per agent/task"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Cost association
    agent_id: UUID = Field(..., description="Agent that incurred the cost")
    project_id: UUID = Field(..., description="Project context")
    task_id: Optional[UUID] = None
    
    # Token usage
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: Optional[int] = None  # Computed in database
    
    # Cost calculation (in USD)
    input_cost: Decimal = Field(..., ge=Decimal('0'))
    output_cost: Decimal = Field(..., ge=Decimal('0'))
    total_cost: Optional[Decimal] = None  # Computed in database
    
    # Model information
    model_tier: ModelTier = Field(..., description="Tier used for this task")
    model_name: Optional[str] = None
    
    # Performance metrics
    task_duration_seconds: Optional[int] = None
    success: bool = Field(default=True)
    
    recorded_at: datetime = Field(default_factory=datetime.now)


class BudgetConstraint(BaseModel):
    """Project budget limits and enforcement"""
    
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID = Field(..., description="Project this budget applies to")
    
    # Budget limits (in USD)
    monthly_budget: Optional[Decimal] = None
    daily_budget: Optional[Decimal] = None
    per_task_budget: Optional[Decimal] = None
    
    # Alert thresholds (as percentages)
    warning_threshold: Decimal = Field(default=Decimal('80.00'), ge=Decimal('0'), le=Decimal('100'))
    critical_threshold: Decimal = Field(default=Decimal('95.00'), ge=Decimal('0'), le=Decimal('100'))
    
    # Current usage tracking
    current_monthly_spend: Decimal = Field(default=Decimal('0.00'), ge=Decimal('0'))
    current_daily_spend: Decimal = Field(default=Decimal('0.00'), ge=Decimal('0'))
    spend_reset_date: date = Field(default_factory=lambda: datetime.now().date())
    
    # Budget enforcement
    auto_downgrade_enabled: bool = Field(default=True)
    emergency_stop_enabled: bool = Field(default=False)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ROIAnalysis(BaseModel):
    """Return on Investment analysis for agents"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Analysis scope
    agent_id: UUID = Field(..., description="Agent being analyzed")
    project_id: UUID = Field(..., description="Project context")
    analysis_period_days: int = Field(default=30, ge=1)
    
    # Cost metrics
    total_cost: Decimal = Field(..., ge=Decimal('0'))
    cost_per_task: Optional[Decimal] = None
    
    # Value metrics
    tasks_completed: int = Field(..., ge=0)
    success_rate: Decimal = Field(..., ge=Decimal('0'), le=Decimal('1'))
    avg_completion_time_hours: Optional[Decimal] = None
    
    # ROI calculation
    estimated_value_delivered: Optional[Decimal] = None
    roi_ratio: Optional[Decimal] = None  # value/cost ratio
    
    # Optimization recommendations
    recommended_tier: Optional[ModelTier] = None
    tier_change_rationale: Optional[str] = None
    potential_savings: Optional[Decimal] = None
    
    analyzed_at: datetime = Field(default_factory=datetime.now)
    analyst: str = Field(default="system")


# =====================================================
# REAL-TIME COLLABORATION MODELS
# =====================================================

class SharedContext(BaseModel):
    """Shared collaboration context between agents"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Context identification
    task_id: UUID = Field(..., description="Task this context serves")
    project_id: UUID = Field(..., description="Project context")
    context_name: str = Field(..., max_length=255)
    
    # Collaboration data
    discoveries: List[Dict[str, Any]] = Field(default_factory=list, description="Agent discoveries")
    blockers: List[Dict[str, Any]] = Field(default_factory=list, description="Reported blockers")
    patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Successful patterns")
    participants: List[UUID] = Field(default_factory=list, description="Participating agent IDs")
    
    # Context status
    is_active: bool = Field(default=True)
    last_updated_by: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class BroadcastMessage(BaseModel):
    """Messages broadcast in the pub/sub system"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Message identification
    message_id: str = Field(..., max_length=255, description="Unique message identifier")
    topic: str = Field(..., max_length=255, description="Broadcast topic")
    
    # Message content
    content: Dict[str, Any] = Field(..., description="Message payload")
    message_type: str = Field(..., max_length=100, description="discovery, blocker, pattern, update")
    
    # Priority and delivery
    priority: int = Field(default=1, ge=1, le=4, description="1=low, 2=medium, 3=high, 4=critical")
    sender_id: Optional[UUID] = None
    
    # Targeting
    target_agents: List[UUID] = Field(default_factory=list, description="Specific agent IDs")
    target_topics: List[str] = Field(default_factory=list, description="Topic subscriptions")
    
    # Status tracking
    delivered_count: int = Field(default=0, ge=0)
    acknowledgment_count: int = Field(default=0, ge=0)
    
    # Timestamps
    sent_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class TopicSubscription(BaseModel):
    """Agent subscriptions to collaboration topics"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Subscription details
    agent_id: UUID = Field(..., description="Subscribing agent")
    topic: str = Field(..., max_length=255, description="Topic to subscribe to")
    
    # Filtering
    priority_filter: int = Field(default=1, ge=1, le=4, description="Min priority to receive")
    content_filters: Dict[str, Any] = Field(default_factory=dict, description="JSONPath filters")
    
    # Subscription management
    is_active: bool = Field(default=True)
    subscription_type: str = Field(default="standard", max_length=50)
    
    # Callback configuration
    callback_endpoint: Optional[str] = None
    callback_timeout_seconds: int = Field(default=30, ge=1)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class MessageAcknowledgment(BaseModel):
    """Tracks message delivery acknowledgments"""
    
    id: UUID = Field(default_factory=uuid4)
    
    message_id: UUID = Field(..., description="Message being acknowledged")
    agent_id: UUID = Field(..., description="Agent acknowledging")
    
    # Acknowledgment details
    acknowledged_at: datetime = Field(default_factory=datetime.now)
    processing_status: str = Field(default="received", max_length=50)
    response_data: Dict[str, Any] = Field(default_factory=dict)


# =====================================================
# GLOBAL RULES INTEGRATION MODELS
# =====================================================

class RulesProfile(BaseModel):
    """Parsed rules from CLAUDE.md, RULES.md, MANIFEST.md"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Profile identification
    profile_name: str = Field(..., max_length=255, description="Unique profile identifier")
    agent_type: Optional[AgentType] = None
    model_tier: Optional[ModelTier] = None
    
    # Rule sources (parsed from configuration files)
    global_rules: List[str] = Field(default_factory=list, description="From global CLAUDE.md, RULES.md")
    project_rules: List[str] = Field(default_factory=list, description="From project-specific rules")
    manifest_rules: List[str] = Field(default_factory=list, description="From MANIFEST.md")
    
    # Rule categories
    quality_gates: List[str] = Field(default_factory=list, description="Quality enforcement rules")
    security_rules: List[str] = Field(default_factory=list, description="Security constraints")
    performance_rules: List[str] = Field(default_factory=list, description="Performance requirements")
    coding_standards: List[str] = Field(default_factory=list, description="Coding style and standards")
    
    # Profile metadata
    rule_count: int = Field(default=0, ge=0)
    last_parsed_at: Optional[datetime] = None
    source_file_hashes: Dict[str, str] = Field(default_factory=dict, description="Track file changes")
    
    # Profile status
    is_active: bool = Field(default=True)
    validation_status: str = Field(default="pending", max_length=50)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class RuleViolation(BaseModel):
    """Tracks rule enforcement violations"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Violation context
    agent_id: UUID = Field(..., description="Agent that violated the rule")
    task_id: Optional[UUID] = None
    rules_profile_id: Optional[UUID] = None
    
    # Violation details
    rule_name: str = Field(..., max_length=255)
    rule_category: Optional[str] = None
    violation_type: str = Field(..., max_length=100, description="WARNING, ERROR, CRITICAL")
    description: str = Field(..., description="Detailed violation description")
    
    # Resolution tracking
    status: str = Field(default="open", max_length=50, description="open, resolved, acknowledged, suppressed")
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    detected_at: datetime = Field(default_factory=datetime.now)


# =====================================================
# REQUEST/RESPONSE MODELS FOR API
# =====================================================

class CreateAgentRequest(BaseModel):
    """Request to create a new agent"""
    
    name: str = Field(..., max_length=255)
    agent_type: AgentType
    model_tier: ModelTier = Field(default=ModelTier.SONNET)
    project_id: UUID
    capabilities: Dict[str, Any] = Field(default_factory=dict)


class UpdateAgentRequest(BaseModel):
    """Request to update agent properties"""
    
    name: Optional[str] = None
    state: Optional[AgentState] = None
    model_tier: Optional[ModelTier] = None
    capabilities: Optional[Dict[str, Any]] = None


class AgentPerformanceMetrics(BaseModel):
    """Agent performance summary"""
    
    agent_id: UUID
    tasks_completed: int
    success_rate: Decimal
    avg_completion_time_seconds: int
    cost_last_30_days: Decimal
    knowledge_items_count: int
    activity_level: str  # RECENT, TODAY, WEEK, INACTIVE


class ProjectIntelligenceOverview(BaseModel):
    """Project-level intelligence metrics"""
    
    project_id: UUID
    project_name: str
    total_agents: int
    active_agents: int
    opus_agents: int
    sonnet_agents: int
    haiku_agents: int
    avg_success_rate: Decimal
    total_tasks_completed: int
    monthly_cost: Decimal
    monthly_budget: Decimal
    budget_utilization_percent: Decimal
    active_shared_contexts: int
    recent_broadcasts: int


class CostOptimizationRecommendation(BaseModel):
    """Cost optimization recommendation for an agent"""
    
    agent_id: UUID
    agent_type: AgentType
    current_tier: ModelTier
    total_cost: Decimal
    success_rate: Decimal
    avg_cost_per_task: Decimal
    recommendation: str  # CONSIDER_SONNET, CONSIDER_OPUS, CONSIDER_HAIKU, OPTIMAL
    potential_monthly_savings: Decimal


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def calculate_complexity_score(
    technical: float,
    domain_expertise: float,
    code_volume: float,
    integration: float
) -> float:
    """Calculate overall task complexity score"""
    return (technical + domain_expertise + code_volume + integration) / 4.0


def recommend_tier(complexity_score: float) -> ModelTier:
    """Recommend model tier based on complexity (Sonnet-first preference)"""
    if complexity_score >= 0.75:
        return ModelTier.OPUS
    elif complexity_score >= 0.15:  # Sonnet-first: lower threshold
        return ModelTier.SONNET
    else:
        return ModelTier.HAIKU


def calculate_tier_cost(tier: ModelTier, input_tokens: int, output_tokens: int) -> tuple[Decimal, Decimal]:
    """Calculate cost for input/output tokens based on tier"""
    
    # Pricing per 1M tokens (in USD)
    tier_pricing = {
        ModelTier.OPUS: {"input": Decimal("15.00"), "output": Decimal("75.00")},
        ModelTier.SONNET: {"input": Decimal("3.00"), "output": Decimal("15.00")},
        ModelTier.HAIKU: {"input": Decimal("0.25"), "output": Decimal("1.25")}
    }
    
    pricing = tier_pricing[tier]
    input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * pricing["input"]
    output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * pricing["output"]
    
    return input_cost, output_cost


def evolve_confidence(current_confidence: Decimal, success: bool) -> Decimal:
    """Evolve knowledge confidence based on success/failure"""
    if success:
        # Success increases confidence (max 0.99)
        new_confidence = current_confidence * Decimal("1.1")
        return min(new_confidence, Decimal("0.99"))
    else:
        # Failure decreases confidence (min 0.1)
        new_confidence = current_confidence * Decimal("0.9")
        return max(new_confidence, Decimal("0.1"))


# =====================================================
# MODEL REGISTRY
# =====================================================

# Export all models for easy importing
__all__ = [
    # Enums
    "AgentState", "ModelTier", "AgentType",
    
    # Core models
    "AgentV3", "AgentStateHistory", "AgentPool",
    "TaskComplexity", "RoutingRule",
    "AgentKnowledge", "KnowledgeEvolution", "SharedKnowledge",
    "CostTracking", "BudgetConstraint", "ROIAnalysis",
    "SharedContext", "BroadcastMessage", "TopicSubscription", "MessageAcknowledgment",
    "RulesProfile", "RuleViolation",
    
    # API models
    "CreateAgentRequest", "UpdateAgentRequest",
    "AgentPerformanceMetrics", "ProjectIntelligenceOverview",
    "CostOptimizationRecommendation",
    
    # Utility functions
    "calculate_complexity_score", "recommend_tier", "calculate_tier_cost", "evolve_confidence"
]