"""
Actor/Critic Quality Gate Pattern

Implements the separation of generation (Actor) from evaluation (Critic)
for automated quality assurance without human review of every step.

Based on 2025 best practices for agentic quality control.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 75-89%
    ACCEPTABLE = "acceptable"  # 60-74%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 40-59%
    UNACCEPTABLE = "unacceptable"  # <40%


class CriticDecision(str, Enum):
    """Critic's decision on Actor's output"""
    APPROVE = "approve"  # Accept as-is
    REQUEST_CHANGES = "request_changes"  # Needs revisions
    REJECT = "reject"  # Unacceptable, start over


class QualityMetric(BaseModel):
    """Individual quality metric"""
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    passed: bool
    threshold: float = Field(..., ge=0.0, le=1.0)
    details: Optional[str] = None


class ActorOutput(BaseModel):
    """
    Output from Actor (generator) agent.

    The Actor generates code, documentation, or other artifacts
    without self-evaluation.
    """
    actor_id: str
    task_id: str
    artifact_type: str  # "code", "documentation", "test", etc.
    content: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CriticReview(BaseModel):
    """
    Review from Critic (evaluator) agent.

    The Critic evaluates Actor's output against quality rubrics.
    """
    critic_id: str
    task_id: str
    actor_output: ActorOutput

    # Quality assessment
    metrics: list[QualityMetric]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    quality_level: QualityLevel

    # Decision
    decision: CriticDecision
    reasoning: str = Field(..., min_length=20, max_length=2000)

    # Issues found
    critical_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    # For REQUEST_CHANGES decision
    required_changes: list[str] = Field(default_factory=list)

    # Metadata
    review_time_ms: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QualityGate(BaseModel):
    """
    Quality gate configuration.

    Defines thresholds and criteria for automated quality control.
    """
    name: str
    description: str

    # Metrics and thresholds
    required_metrics: list[str]
    metric_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Metric name → minimum score (0.0-1.0)"
    )

    # Gate behavior
    overall_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum weighted average score to pass"
    )
    allow_critical_issues: bool = Field(
        default=False,
        description="Whether to pass with critical issues"
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum Actor→Critic cycles before escalation"
    )


class ActorCriticCycle(BaseModel):
    """
    Complete Actor→Critic→(Revise) cycle tracking.

    Tracks iterations until quality gate passes or max iterations reached.
    """
    cycle_id: str
    task_id: str
    gate: QualityGate

    # Participants
    actor_id: str
    critic_id: str

    # Iterations
    iterations: list[tuple[ActorOutput, CriticReview]] = Field(
        default_factory=list
    )
    current_iteration: int = Field(default=0, ge=0)

    # Status
    passed: bool = False
    escalated: bool = Field(
        default=False,
        description="Escalated to human review after max iterations"
    )

    # Final result
    final_output: Optional[ActorOutput] = None
    final_review: Optional[CriticReview] = None

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ActorCriticSystem:
    """
    Actor/Critic quality gate system.

    Manages the separation of generation from evaluation with
    automated quality control.
    """

    def __init__(self):
        self._quality_gates: dict[str, QualityGate] = {}
        self._metric_evaluators: dict[str, Callable] = {}
        self._active_cycles: dict[str, ActorCriticCycle] = {}

    def register_quality_gate(self, gate: QualityGate) -> None:
        """
        Register a quality gate configuration.

        Args:
            gate: Quality gate to register
        """
        self._quality_gates[gate.name] = gate
        logger.info(f"Registered quality gate: {gate.name}")

    def register_metric_evaluator(
        self,
        metric_name: str,
        evaluator: Callable[[ActorOutput], float]
    ) -> None:
        """
        Register metric evaluation function.

        Args:
            metric_name: Name of the metric
            evaluator: Function that takes ActorOutput and returns score (0.0-1.0)
        """
        self._metric_evaluators[metric_name] = evaluator
        logger.info(f"Registered metric evaluator: {metric_name}")

    async def start_cycle(
        self,
        task_id: str,
        gate_name: str,
        actor_id: str,
        critic_id: str
    ) -> ActorCriticCycle:
        """
        Start new Actor/Critic quality cycle.

        Args:
            task_id: Task identifier
            gate_name: Name of quality gate to use
            actor_id: ID of Actor agent
            critic_id: ID of Critic agent

        Returns:
            Started cycle

        Raises:
            ValueError: If gate not registered
        """
        import uuid

        gate = self._quality_gates.get(gate_name)
        if gate is None:
            raise ValueError(f"Quality gate not registered: {gate_name}")

        cycle = ActorCriticCycle(
            cycle_id=str(uuid.uuid4()),
            task_id=task_id,
            gate=gate,
            actor_id=actor_id,
            critic_id=critic_id
        )

        self._active_cycles[cycle.cycle_id] = cycle

        logger.info(
            f"Started Actor/Critic cycle {cycle.cycle_id} "
            f"for task {task_id} with gate {gate_name}"
        )

        return cycle

    async def evaluate_output(
        self,
        cycle_id: str,
        actor_output: ActorOutput
    ) -> CriticReview:
        """
        Critic evaluates Actor's output.

        Args:
            cycle_id: Cycle identifier
            actor_output: Output from Actor to evaluate

        Returns:
            Critic's review

        Raises:
            ValueError: If cycle not found
        """
        import time

        cycle = self._active_cycles.get(cycle_id)
        if cycle is None:
            raise ValueError(f"Cycle not found: {cycle_id}")

        start_time = time.time()

        # Evaluate all metrics
        metrics = []
        for metric_name in cycle.gate.required_metrics:
            evaluator = self._metric_evaluators.get(metric_name)
            if evaluator is None:
                logger.warning(f"No evaluator for metric: {metric_name}")
                continue

            score = await evaluator(actor_output)
            threshold = cycle.gate.metric_thresholds.get(metric_name, 0.6)

            metric = QualityMetric(
                name=metric_name,
                score=score,
                passed=score >= threshold,
                threshold=threshold
            )
            metrics.append(metric)

        # Calculate overall score (weighted average)
        if metrics:
            total_weight = sum(m.weight for m in metrics)
            overall_score = sum(m.score * m.weight for m in metrics) / total_weight
        else:
            overall_score = 0.0

        # Determine quality level
        if overall_score >= 0.9:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.75:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 0.6:
            quality_level = QualityLevel.ACCEPTABLE
        elif overall_score >= 0.4:
            quality_level = QualityLevel.NEEDS_IMPROVEMENT
        else:
            quality_level = QualityLevel.UNACCEPTABLE

        # Collect issues
        critical_issues = []
        warnings = []
        for metric in metrics:
            if not metric.passed:
                if metric.score < 0.4:
                    critical_issues.append(
                        f"{metric.name}: {metric.score:.2f} (threshold: {metric.threshold:.2f})"
                    )
                else:
                    warnings.append(
                        f"{metric.name}: {metric.score:.2f} (threshold: {metric.threshold:.2f})"
                    )

        # Make decision
        if overall_score >= cycle.gate.overall_threshold:
            if critical_issues and not cycle.gate.allow_critical_issues:
                decision = CriticDecision.REQUEST_CHANGES
                reasoning = (
                    f"Overall score {overall_score:.2f} meets threshold, "
                    f"but critical issues present: {', '.join(critical_issues)}"
                )
            else:
                decision = CriticDecision.APPROVE
                reasoning = (
                    f"Quality level {quality_level.value} with score {overall_score:.2f} "
                    f"meets requirements. {len(metrics)} metrics passed."
                )
        elif overall_score >= 0.4:
            decision = CriticDecision.REQUEST_CHANGES
            reasoning = (
                f"Quality level {quality_level.value} with score {overall_score:.2f}. "
                f"Improvements needed in: {', '.join([m.name for m in metrics if not m.passed])}"
            )
        else:
            decision = CriticDecision.REJECT
            reasoning = (
                f"Quality level {quality_level.value} with score {overall_score:.2f} "
                f"below acceptable threshold. Recommend starting over."
            )

        # Required changes for REQUEST_CHANGES decision
        required_changes = []
        if decision == CriticDecision.REQUEST_CHANGES:
            for metric in metrics:
                if not metric.passed:
                    required_changes.append(
                        f"Improve {metric.name} from {metric.score:.2f} to {metric.threshold:.2f}"
                    )

        review_time_ms = int((time.time() - start_time) * 1000)

        review = CriticReview(
            critic_id=cycle.critic_id,
            task_id=cycle.task_id,
            actor_output=actor_output,
            metrics=metrics,
            overall_score=overall_score,
            quality_level=quality_level,
            decision=decision,
            reasoning=reasoning,
            critical_issues=critical_issues,
            warnings=warnings,
            required_changes=required_changes,
            review_time_ms=review_time_ms,
            confidence=0.9  # High confidence in automated evaluation
        )

        # Update cycle
        cycle.iterations.append((actor_output, review))
        cycle.current_iteration += 1

        if decision == CriticDecision.APPROVE:
            cycle.passed = True
            cycle.final_output = actor_output
            cycle.final_review = review
            cycle.completed_at = datetime.utcnow()
        elif cycle.current_iteration >= cycle.gate.max_iterations:
            cycle.escalated = True
            cycle.completed_at = datetime.utcnow()
            logger.warning(
                f"Cycle {cycle_id} escalated after {cycle.current_iteration} iterations"
            )

        logger.info(
            f"Critic review: {decision.value} "
            f"(score: {overall_score:.2f}, iteration: {cycle.current_iteration})"
        )

        return review

    def get_cycle_status(self, cycle_id: str) -> Optional[ActorCriticCycle]:
        """Get current status of a cycle"""
        return self._active_cycles.get(cycle_id)

    def get_quality_gate(self, gate_name: str) -> Optional[QualityGate]:
        """Get quality gate configuration"""
        return self._quality_gates.get(gate_name)


# Predefined quality gates for Archon

CODE_QUALITY_GATE = QualityGate(
    name="code_quality",
    description="Quality gate for code implementation",
    required_metrics=[
        "type_safety",
        "test_coverage",
        "lint_compliance",
        "documentation_quality",
        "error_handling",
        "performance"
    ],
    metric_thresholds={
        "type_safety": 1.0,  # 100% type safety required
        "test_coverage": 0.95,  # 95% coverage required
        "lint_compliance": 1.0,  # Zero lint errors
        "documentation_quality": 0.8,  # 80% doc coverage
        "error_handling": 0.9,  # 90% error handling
        "performance": 0.85  # 85% performance target
    },
    overall_threshold=0.90,  # 90% overall quality
    allow_critical_issues=False,
    max_iterations=3
)

SECURITY_GATE = QualityGate(
    name="security_audit",
    description="Security audit quality gate",
    required_metrics=[
        "owasp_compliance",
        "input_validation",
        "authentication_security",
        "secrets_management",
        "vulnerability_scan"
    ],
    metric_thresholds={
        "owasp_compliance": 1.0,  # 100% OWASP compliance
        "input_validation": 1.0,  # 100% input validation
        "authentication_security": 1.0,  # 100% auth security
        "secrets_management": 1.0,  # 100% secrets handled
        "vulnerability_scan": 1.0  # Zero vulnerabilities
    },
    overall_threshold=1.0,  # 100% security required
    allow_critical_issues=False,
    max_iterations=2
)

DOCUMENTATION_GATE = QualityGate(
    name="documentation",
    description="Documentation quality gate",
    required_metrics=[
        "completeness",
        "clarity",
        "examples",
        "accuracy"
    ],
    metric_thresholds={
        "completeness": 0.9,  # 90% complete
        "clarity": 0.85,  # 85% clear
        "examples": 0.8,  # 80% have examples
        "accuracy": 0.95  # 95% accurate
    },
    overall_threshold=0.85,
    allow_critical_issues=False,
    max_iterations=3
)

# Global system instance
_global_system = ActorCriticSystem()
_global_system.register_quality_gate(CODE_QUALITY_GATE)
_global_system.register_quality_gate(SECURITY_GATE)
_global_system.register_quality_gate(DOCUMENTATION_GATE)


# Convenience functions
async def start_quality_cycle(
    task_id: str,
    gate_name: str,
    actor_id: str,
    critic_id: str
) -> ActorCriticCycle:
    """Start quality cycle using global system"""
    return await _global_system.start_cycle(task_id, gate_name, actor_id, critic_id)


async def evaluate_actor_output(
    cycle_id: str,
    actor_output: ActorOutput
) -> CriticReview:
    """Evaluate Actor output using global system"""
    return await _global_system.evaluate_output(cycle_id, actor_output)


def register_metric_evaluator(
    metric_name: str,
    evaluator: Callable[[ActorOutput], float]
) -> None:
    """Register metric evaluator with global system"""
    _global_system.register_metric_evaluator(metric_name, evaluator)
