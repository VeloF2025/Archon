"""
Database Models for Agency Workflow Management

Extended models for workflow management that integrate with the existing Archon agent system
and support ReactFlow visualization components. These models extend the existing agent_models.py
to provide comprehensive workflow management capabilities.

Key features:
- Workflow definitions compatible with ReactFlow data format
- Workflow execution tracking and monitoring
- Agent communication flow control
- Performance metrics and analytics
- Integration with existing AgentV3 and Agency Swarm components
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, json
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Table
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .agent_models import AgentV3, ModelTier, AgentType, AgentState

# SQLAlchemy Base
class Base(DeclarativeBase):
    pass

# =====================================================
# ENUMS FOR WORKFLOW MANAGEMENT
# =====================================================

class WorkflowStatus(str, Enum):
    """Workflow lifecycle states"""
    DRAFT = "DRAFT"              # Being designed/configured
    PUBLISHED = "PUBLISHED"      # Available for execution
    RUNNING = "RUNNING"          # Currently executing
    PAUSED = "PAUSED"            # Temporarily paused
    COMPLETED = "COMPLETED"      # Successfully finished
    FAILED = "FAILED"            # Execution failed
    CANCELLED = "CANCELLED"      # Cancelled by user
    ARCHIVED = "ARCHIVED"        # No longer active

class StepType(str, Enum):
    """Types of workflow steps"""
    AGENT_TASK = "AGENT_TASK"        # Execute agent task
    DECISION = "DECISION"            # Conditional branching
    PARALLEL = "PARALLEL"            # Parallel execution
    DELAY = "DELAY"                  # Wait/delay step
    API_CALL = "API_CALL"            # External API call
    DATA_TRANSFORM = "DATA_TRANSFORM"  # Data transformation
    HUMAN_INPUT = "HUMAN_INPUT"      # Wait for human input
    NOTIFICATION = "NOTIFICATION"    # Send notification
    SUB_WORKFLOW = "SUB_WORKFLOW"    # Execute sub-workflow

class ExecutionStatus(str, Enum):
    """Execution instance status"""
    PENDING = "PENDING"            # Waiting to start
    RUNNING = "RUNNING"            # Currently executing
    COMPLETED = "COMPLETED"        # Successfully finished
    FAILED = "FAILED"              # Execution failed
    CANCELLED = "CANCELLED"        # Cancelled
    TIMEOUT = "TIMEOUT"            # Execution timed out
    SKIPPED = "SKIPPED"            # Step was skipped

class TriggerType(str, Enum):
    """Workflow trigger types"""
    MANUAL = "MANUAL"              # Manual trigger
    SCHEDULED = "SCHEDULED"        # Time-based trigger
    EVENT = "EVENT"                # Event-based trigger
    WEBHOOK = "WEBHOOK"            # HTTP webhook trigger
    API = "API"                    # API call trigger

# =====================================================
# WORKFLOW DEFINITION MODELS
# =====================================================

class WorkflowDefinition(Base):
    """Main workflow definition compatible with ReactFlow format"""

    __tablename__ = "workflow_definitions"

    # Identification
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    tags: Mapped[List[str]] = mapped_column(JSONB, default=list)

    # ReactFlow compatibility
    flow_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)  # ReactFlow nodes and edges

    # Workflow configuration
    status: Mapped[WorkflowStatus] = mapped_column(String(50), default=WorkflowStatus.DRAFT)
    trigger_type: Mapped[TriggerType] = mapped_column(String(50), default=TriggerType.MANUAL)

    # Execution settings
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=3600)  # 1 hour default
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_delay_seconds: Mapped[int] = mapped_column(Integer, default=60)

    # Agent assignment
    default_agent_type: Mapped[Optional[AgentType]] = mapped_column(String(50))
    default_model_tier: Mapped[ModelTier] = mapped_column(String(50), default=ModelTier.SONNET)

    # Access control
    project_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")
    versions = relationship("WorkflowVersion", back_populates="workflow", cascade="all, delete-orphan")

    def to_reactflow_format(self) -> Dict[str, Any]:
        """Convert to ReactFlow compatible format"""
        return {
            "workflowId": str(self.id),
            "name": self.name,
            "description": self.description,
            "nodes": self.flow_data.get("nodes", []),
            "edges": self.flow_data.get("edges", []),
            "viewport": self.flow_data.get("viewport", {"x": 0, "y": 0, "zoom": 1})
        }

class WorkflowVersion(Base):
    """Workflow version tracking for change management"""

    __tablename__ = "workflow_versions"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)
    version: str = Column(String(50), nullable=False)

    # Version data
    flow_data: Dict[str, Any] = Column(JSONB, nullable=False)
    changelog: str = Column(Text)
    created_by: str = Column(String(255), nullable=False)

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    workflow = relationship("WorkflowDefinition", back_populates="versions")

class WorkflowStep(Base):
    """Individual workflow step definition"""

    __tablename__ = "workflow_steps"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)
    step_id: str = Column(String(255), nullable=False)  # ReactFlow node ID

    # Step configuration
    name: str = Column(String(255), nullable=False)
    step_type: StepType = Column(String(50), nullable=False)
    description: str = Column(Text)

    # Step parameters
    parameters: Dict[str, Any] = Column(JSONB, default=dict)
    conditions: Dict[str, Any] = Column(JSONB, default=dict)  # Execution conditions

    # Execution settings
    timeout_seconds: int = Column(Integer, default=300)
    retry_count: int = Column(Integer, default=0)

    # Agent assignment
    agent_type: Optional[AgentType] = Column(String(50))
    model_tier: Optional[ModelTier] = Column(String(50))
    required_capabilities: List[str] = Column(JSONB, default=list)

    # Position (ReactFlow)
    position: Dict[str, float] = Column(JSONB)  # {"x": 0, "y": 0}

    # Dependencies
    depends_on: List[str] = Column(JSONB, default=list)  # Step IDs this depends on
    next_steps: List[str] = Column(JSONB, default=list)   # Possible next step IDs

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# =====================================================
# WORKFLOW EXECUTION MODELS
# =====================================================

class WorkflowExecution(Base):
    """Workflow execution instance"""

    __tablename__ = "workflow_executions"

    # Identification
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)
    execution_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Execution context
    triggered_by: Mapped[str] = mapped_column(String(255), nullable=False)  # User or system that triggered
    trigger_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)  # Data that triggered the workflow

    # Execution status
    status: Mapped[ExecutionStatus] = mapped_column(String(50), default=ExecutionStatus.PENDING)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 to 1.0

    # Execution details
    current_step_id: Mapped[Optional[str]] = mapped_column(String(255))
    execution_path: Mapped[List[str]] = mapped_column(JSONB, default=list)  # Path of executed steps

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    estimated_duration: Mapped[Optional[int]] = mapped_column(Integer)  # Estimated seconds

    # Results
    results: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    errors: Mapped[List[str]] = mapped_column(JSONB, default=list)
    output_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Performance metrics
    execution_time_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(Float)
    cpu_usage_percent: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    workflow = relationship("WorkflowDefinition", back_populates="executions")
    step_executions = relationship("StepExecution", back_populates="workflow_execution", cascade="all, delete-orphan")

class StepExecution(Base):
    """Individual step execution tracking"""

    __tablename__ = "step_executions"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_execution_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_executions.id"), nullable=False)
    step_id: str = Column(String(255), nullable=False)

    # Agent assignment
    assigned_agent_id: Optional[UUID] = Column(PG_UUID(as_uuid=True))
    agent_type: Optional[AgentType] = Column(String(50))
    model_tier: Optional[ModelTier] = Column(String(50))

    # Execution status
    status: ExecutionStatus = Column(String(50), default=ExecutionStatus.PENDING)

    # Input/Output
    input_data: Dict[str, Any] = Column(JSONB, default=dict)
    output_data: Dict[str, Any] = Column(JSONB, default=dict)

    # Results
    result_data: Dict[str, Any] = Column(JSONB, default=dict)
    error_message: Optional[str] = Column(Text)
    retry_count: int = Column(Integer, default=0)

    # Timing
    started_at: Optional[datetime] = Column(DateTime(timezone=True))
    completed_at: Optional[datetime] = Column(DateTime(timezone=True))
    execution_time_seconds: Optional[int] = Column(Integer)

    # Performance
    tokens_used: Optional[Integer] = Column(Integer)
    cost_usd: Optional[Float] = Column(Float)

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="step_executions")

# =====================================================
# WORKFLOW ANALYTICS MODELS
# =====================================================

class WorkflowMetrics(Base):
    """Aggregated workflow performance metrics"""

    __tablename__ = "workflow_metrics"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)

    # Time period
    date: datetime = Column(DateTime(timezone=True), nullable=False)  # Daily aggregation
    hour: Optional[int] = Column(Integer)  # For hourly aggregation

    # Execution metrics
    total_executions: int = Column(Integer, default=0)
    successful_executions: int = Column(Integer, default=0)
    failed_executions: int = Column(Integer, default=0)

    # Performance metrics
    avg_execution_time_seconds: Optional[Float] = Column(Float)
    max_execution_time_seconds: Optional[Float] = Column(Float)
    min_execution_time_seconds: Optional[Float] = Column(Float)

    # Cost metrics
    total_cost_usd: Optional[Float] = Column(Float, default=0.0)
    avg_cost_usd: Optional[Float] = Column(Float)

    # Resource usage
    avg_memory_mb: Optional[Float] = Column(Float)
    avg_cpu_percent: Optional[Float] = Column(Float)

    # Agent usage
    agent_types_used: List[str] = Column(JSONB, default=list)
    model_tiers_used: List[str] = Column(JSONB, default=list)

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())

class WorkflowAnalytics(Base):
    """Real-time workflow analytics and insights"""

    __tablename__ = "workflow_analytics"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)

    # Analytics period
    period_start: datetime = Column(DateTime(timezone=True), nullable=False)
    period_end: datetime = Column(DateTime(timezone=True), nullable=False)

    # Performance insights
    performance_score: Optional[Float] = Column(Float)  # 0.0 to 1.0
    reliability_score: Optional[Float] = Column(Float)    # 0.0 to 1.0
    efficiency_score: Optional[Float] = Column(Float)     # 0.0 to 1.0

    # Bottleneck analysis
    bottleneck_steps: List[Dict[str, Any]] = Column(JSONB, default=list)
    slowest_steps: List[Dict[str, Any]] = Column(JSONB, default=list)

    # Cost optimization
    cost_optimization_opportunities: List[Dict[str, Any]] = Column(JSONB, default=list)
    estimated_monthly_savings: Optional[Float] = Column(Float)

    # Usage patterns
    peak_usage_times: List[Dict[str, Any]] = Column(JSONB, default=list)
    common_failure_patterns: List[Dict[str, Any]] = Column(JSONB, default=list)

    # Recommendations
    recommendations: List[str] = Column(JSONB, default=list)

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# =====================================================
# WORKFLOW SCHEDULING MODELS
# =====================================================

class WorkflowSchedule(Base):
    """Workflow scheduling configuration"""

    __tablename__ = "workflow_schedules"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)

    # Schedule configuration
    name: str = Column(String(255), nullable=False)
    enabled: bool = Column(Boolean, default=True)

    # Timing
    cron_expression: Optional[str] = Column(String(255))
    interval_seconds: Optional[int] = Column(Integer)
    start_date: Optional[datetime] = Column(DateTime(timezone=True))
    end_date: Optional[datetime] = Column(DateTime(timezone=True))

    # Execution parameters
    parameters: Dict[str, Any] = Column(JSONB, default=dict)

    # Notifications
    notify_on_success: bool = Column(Boolean, default=False)
    notify_on_failure: bool = Column(Boolean, default=True)
    notification_recipients: List[str] = Column(JSONB, default=list)

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    workflow = relationship("WorkflowDefinition")

# =====================================================
# WORKFLOW TRIGGER MODELS
# =====================================================

class WebhookTrigger(Base):
    """Webhook trigger configuration"""

    __tablename__ = "webhook_triggers"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)

    # Webhook configuration
    endpoint: str = Column(String(500), nullable=False, unique=True)
    secret: Optional[str] = Column(String(255))  # HMAC secret for validation

    # Request filtering
    allowed_methods: List[str] = Column(JSONB, default=["POST"])  # HTTP methods
    allowed_ips: List[str] = Column(JSONB, default=list)  # IP whitelist

    # Event processing
    event_type_filter: Optional[str] = Column(String(255))
    data_validation_schema: Optional[Dict[str, Any]] = Column(JSONB)

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    workflow = relationship("WorkflowDefinition")

class EventTrigger(Base):
    """Event-based trigger configuration"""

    __tablename__ = "event_triggers"

    # Identification
    id: UUID = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id: UUID = Column(PG_UUID(as_uuid=True), ForeignKey("workflow_definitions.id"), nullable=False)

    # Event configuration
    event_type: str = Column(String(255), nullable=False)
    event_source: Optional[str] = Column(String(255))

    # Event filtering
    conditions: Dict[str, Any] = Column(JSONB, default=dict)
    correlation_id: Optional[str] = Column(String(255))

    # Debouncing
    debounce_seconds: int = Column(Integer, default=0)  # 0 = no debouncing

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    workflow = relationship("WorkflowDefinition")

# =====================================================
# PYDANTIC MODELS FOR API
# =====================================================

class WorkflowCreateRequest(BaseModel):
    """Request model for creating a workflow"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    flow_data: Dict[str, Any] = Field(..., description="ReactFlow nodes and edges")
    trigger_type: TriggerType = Field(default=TriggerType.MANUAL)
    default_agent_type: Optional[AgentType] = None
    default_model_tier: ModelTier = Field(default=ModelTier.SONNET)
    timeout_seconds: int = Field(default=3600, ge=30, le=86400)
    tags: List[str] = Field(default_factory=list)
    is_public: bool = Field(default=False)

class WorkflowUpdateRequest(BaseModel):
    """Request model for updating a workflow"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    flow_data: Optional[Dict[str, Any]] = Field(None)
    status: Optional[WorkflowStatus] = None
    tags: Optional[List[str]] = None
    timeout_seconds: Optional[int] = Field(None, ge=30, le=86400)

class WorkflowExecutionRequest(BaseModel):
    """Request model for executing a workflow"""
    workflow_id: UUID = Field(..., description="Workflow to execute")
    trigger_data: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = Field(default=True)
    priority: int = Field(default=0, ge=-10, le=10)

class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution"""
    execution_id: str
    workflow_id: UUID
    status: ExecutionStatus
    started_at: datetime
    estimated_duration: Optional[int] = None
    message: str

class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    execution_id: str
    workflow_id: UUID
    status: ExecutionStatus
    progress: float
    current_step_id: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    results: Dict[str, Any]
    errors: List[str]
    metrics: Dict[str, Any]

class WorkflowAnalyticsResponse(BaseModel):
    """Response model for workflow analytics"""
    workflow_id: UUID
    period_start: datetime
    period_end: datetime
    total_executions: int
    success_rate: float
    avg_execution_time: float
    total_cost: float
    performance_score: float
    bottleneck_steps: List[Dict[str, Any]]
    recommendations: List[str]

class WorkflowStepRequest(BaseModel):
    """Request model for creating a workflow step"""
    step_id: str = Field(..., min_length=1, max_length=255)
    name: str = Field(..., min_length=1, max_length=255)
    step_type: StepType = Field(...)
    description: Optional[str] = Field(None, max_length=500)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    agent_type: Optional[AgentType] = None
    model_tier: Optional[ModelTier] = None
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
    position: Dict[str, float] = Field(..., description="ReactFlow position")
    depends_on: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

class ReactFlowNode(BaseModel):
    """ReactFlow node model"""
    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type")
    position: Dict[str, float] = Field(..., description="Node position")
    data: Dict[str, Any] = Field(..., description="Node data")
    style: Optional[Dict[str, Any]] = Field(None)
    className: Optional[str] = Field(None)

class ReactFlowEdge(BaseModel):
    """ReactFlow edge model"""
    id: str = Field(..., description="Unique edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: Optional[str] = Field(None)
    type: Optional[str] = Field(None)
    animated: bool = Field(default=False)
    style: Optional[Dict[str, Any]] = Field(None)
    data: Optional[Dict[str, Any]] = Field(None)

class ReactFlowData(BaseModel):
    """Complete ReactFlow data structure"""
    nodes: List[ReactFlowNode]
    edges: List[ReactFlowEdge]
    viewport: Dict[str, Any] = Field(default={"x": 0, "y": 0, "zoom": 1})

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def calculate_workflow_performance_score(
    success_rate: float,
    avg_execution_time: float,
    cost_efficiency: float,
    reliability_score: float
) -> float:
    """Calculate overall workflow performance score (0.0 to 1.0)"""
    weights = {
        "success_rate": 0.4,
        "execution_time": 0.2,
        "cost_efficiency": 0.2,
        "reliability": 0.2
    }

    # Normalize metrics to 0-1 scale
    normalized_time = max(0, 1 - (avg_execution_time / 3600))  # Normalize by 1 hour
    normalized_cost = max(0, 1 - cost_efficiency) if cost_efficiency > 1 else 1.0

    return (
        weights["success_rate"] * success_rate +
        weights["execution_time"] * normalized_time +
        weights["cost_efficiency"] * normalized_cost +
        weights["reliability"] * reliability_score
    )

def identify_workflow_bottlenecks(
    step_executions: List[StepExecution]
) -> List[Dict[str, Any]]:
    """Identify bottleneck steps in workflow execution"""
    if not step_executions:
        return []

    # Calculate average execution time
    avg_time = sum(
        se.execution_time_seconds or 0
        for se in step_executions
    ) / len(step_executions)

    bottlenecks = []

    for step_exec in step_executions:
        if step_exec.execution_time_seconds and step_exec.execution_time_seconds > avg_time * 1.5:
            bottlenecks.append({
                "step_id": step_exec.step_id,
                "execution_time": step_exec.execution_time_seconds,
                "avg_time": avg_time,
                "slowness_factor": step_exec.execution_time_seconds / avg_time,
                "error_rate": 1.0 if step_exec.status == ExecutionStatus.FAILED else 0.0
            })

    # Sort by slowness factor
    bottlenecks.sort(key=lambda x: x["slowness_factor"], reverse=True)

    return bottlenecks[:5]  # Return top 5 bottlenecks

def validate_reactflow_data(flow_data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate ReactFlow data structure"""
    errors = []

    # Check required fields
    if "nodes" not in flow_data:
        errors.append("Missing required field: nodes")
    elif not isinstance(flow_data["nodes"], list):
        errors.append("nodes must be a list")

    if "edges" not in flow_data:
        errors.append("Missing required field: edges")
    elif not isinstance(flow_data["edges"], list):
        errors.append("edges must be a list")

    # Validate nodes
    if "nodes" in flow_data and isinstance(flow_data["nodes"], list):
        for i, node in enumerate(flow_data["nodes"]):
            if not isinstance(node, dict):
                errors.append(f"Node {i} must be an object")
                continue

            if "id" not in node:
                errors.append(f"Node {i} missing required field: id")
            if "type" not in node:
                errors.append(f"Node {i} missing required field: type")
            if "position" not in node:
                errors.append(f"Node {i} missing required field: position")
            if "data" not in node:
                errors.append(f"Node {i} missing required field: data")

    # Validate edges
    if "edges" in flow_data and isinstance(flow_data["edges"], list):
        for i, edge in enumerate(flow_data["edges"]):
            if not isinstance(edge, dict):
                errors.append(f"Edge {i} must be an object")
                continue

            if "id" not in edge:
                errors.append(f"Edge {i} missing required field: id")
            if "source" not in edge:
                errors.append(f"Edge {i} missing required field: source")
            if "target" not in edge:
                errors.append(f"Edge {i} missing required field: target")

    return len(errors) == 0, errors

# =====================================================
# MODEL REGISTRY
# =====================================================

# Export all models for easy importing
__all__ = [
    # Enums
    "WorkflowStatus", "StepType", "ExecutionStatus", "TriggerType",

    # SQLAlchemy models
    "WorkflowDefinition", "WorkflowVersion", "WorkflowStep",
    "WorkflowExecution", "StepExecution",
    "WorkflowMetrics", "WorkflowAnalytics",
    "WorkflowSchedule", "WebhookTrigger", "EventTrigger",

    # Pydantic models
    "WorkflowCreateRequest", "WorkflowUpdateRequest",
    "WorkflowExecutionRequest", "WorkflowExecutionResponse",
    "WorkflowStatusResponse", "WorkflowAnalyticsResponse",
    "WorkflowStepRequest", "ReactFlowNode", "ReactFlowEdge", "ReactFlowData",

    # Utility functions
    "calculate_workflow_performance_score", "identify_workflow_bottlenecks", "validate_reactflow_data"
]