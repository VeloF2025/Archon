"""
SCWT Metrics tracking and calculation
Based on PRD Section 9: Success Metrics
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..models import ValidationResponse, ValidationStatus


@dataclass
class SCWTMetrics:
    """
    Metrics for SCWT benchmark based on PRD requirements
    
    Target Metrics:
    - Hallucination rate: ≤10%
    - Knowledge reuse: ≥30%
    - Efficiency: 70-85% token/compute savings
    - Task efficiency: ≥30% reduction
    - Communication efficiency: ≥20% fewer iterations
    - Precision: ≥85%
    - Verdict accuracy: ≥90%
    """
    
    # Test identification
    test_id: str
    test_name: str
    phase: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Core metrics
    hallucination_rate: float = 0.0  # Target: ≤10%
    knowledge_reuse_rate: float = 0.0  # Target: ≥30%
    token_savings: float = 0.0  # Target: 70-85%
    task_time_reduction: float = 0.0  # Target: ≥30%
    iteration_reduction: float = 0.0  # Target: ≥20%
    precision: float = 0.0  # Target: ≥85%
    verdict_accuracy: float = 0.0  # Target: ≥90%
    
    # Performance metrics
    validation_time_ms: int = 0
    total_tokens_used: int = 0
    deterministic_checks_run: int = 0
    cross_checks_run: int = 0
    
    # Detection metrics
    issues_detected: int = 0
    gaming_patterns_detected: int = 0
    hallucinations_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Evidence metrics
    evidence_count: int = 0
    high_confidence_evidence: int = 0  # confidence ≥ 0.9
    entities_validated: int = 0
    requirements_validated: int = 0
    
    # Baseline comparison (for calculating improvements)
    baseline_time_ms: Optional[int] = None
    baseline_tokens: Optional[int] = None
    baseline_iterations: Optional[int] = None
    baseline_hallucination_rate: Optional[float] = None
    
    def calculate_from_response(self, response: ValidationResponse):
        """Calculate metrics from validation response"""
        
        # Basic metrics from response
        self.validation_time_ms = response.metrics.validation_time_ms
        self.total_tokens_used = response.metrics.token_count
        self.hallucination_rate = response.metrics.hallucination_rate
        
        # Count issues and evidence
        self.issues_detected = len(response.issues)
        self.evidence_count = len(response.evidence)
        
        # Calculate high confidence evidence
        self.high_confidence_evidence = sum(
            1 for e in response.evidence if e.confidence >= 0.9
        )
        
        # Detect gaming patterns
        self.gaming_patterns_detected = sum(
            1 for issue in response.issues 
            if issue.category == "gaming"
        )
        
        # Detect hallucinations
        self.hallucinations_detected = sum(
            1 for issue in response.issues
            if issue.category == "hallucination" or "hallucination" in issue.message.lower()
        )
        
        # Calculate precision if evidence exists
        if self.evidence_count > 0:
            self.precision = self.high_confidence_evidence / self.evidence_count
        
        # Calculate knowledge reuse from evidence sources
        evidence_sources = set(e.source for e in response.evidence)
        if len(evidence_sources) > 0:
            # Assume maximum of 5 knowledge sources
            self.knowledge_reuse_rate = min(len(evidence_sources) / 5, 1.0)
    
    def calculate_improvements(self):
        """Calculate improvement metrics against baseline"""
        
        # Calculate token savings
        if self.baseline_tokens and self.baseline_tokens > 0:
            self.token_savings = 1 - (self.total_tokens_used / self.baseline_tokens)
        
        # Calculate task time reduction
        if self.baseline_time_ms and self.baseline_time_ms > 0:
            self.task_time_reduction = 1 - (self.validation_time_ms / self.baseline_time_ms)
        
        # Calculate iteration reduction
        if self.baseline_iterations and self.baseline_iterations > 0:
            current_iterations = 1  # Assume single validation pass
            self.iteration_reduction = 1 - (current_iterations / self.baseline_iterations)
        
        # Calculate hallucination reduction
        if self.baseline_hallucination_rate and self.baseline_hallucination_rate > 0:
            reduction = 1 - (self.hallucination_rate / self.baseline_hallucination_rate)
            # Convert to positive improvement metric
            self.hallucination_reduction = max(reduction, 0)
    
    def calculate_accuracy_metrics(
        self,
        expected_issues: List[str],
        found_issues: List[str]
    ):
        """Calculate precision, recall, and accuracy"""
        
        expected_set = set(expected_issues)
        found_set = set(found_issues)
        
        # True positives: correctly identified issues
        self.true_positives = len(expected_set & found_set)
        
        # False positives: incorrectly identified issues
        self.false_positives = len(found_set - expected_set)
        
        # False negatives: missed issues
        self.false_negatives = len(expected_set - found_set)
        
        # Calculate precision
        if (self.true_positives + self.false_positives) > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        
        # Calculate recall
        if (self.true_positives + self.false_negatives) > 0:
            recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            recall = 0
        
        # Calculate F1 score as accuracy proxy
        if (self.precision + recall) > 0:
            self.verdict_accuracy = 2 * (self.precision * recall) / (self.precision + recall)
    
    def meets_targets(self) -> Dict[str, bool]:
        """Check if metrics meet PRD targets"""
        
        return {
            "hallucination_rate": self.hallucination_rate <= 0.1,  # ≤10%
            "knowledge_reuse": self.knowledge_reuse_rate >= 0.3,  # ≥30%
            "token_savings": 0.7 <= self.token_savings <= 0.85,  # 70-85%
            "task_efficiency": self.task_time_reduction >= 0.3,  # ≥30%
            "communication_efficiency": self.iteration_reduction >= 0.2,  # ≥20%
            "precision": self.precision >= 0.85,  # ≥85%
            "verdict_accuracy": self.verdict_accuracy >= 0.9,  # ≥90%
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        
        targets_met = self.meets_targets()
        success_rate = sum(targets_met.values()) / len(targets_met)
        
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat(),
            "success_rate": success_rate,
            "targets_met": targets_met,
            "metrics": {
                "hallucination_rate": f"{self.hallucination_rate:.2%}",
                "knowledge_reuse_rate": f"{self.knowledge_reuse_rate:.2%}",
                "token_savings": f"{self.token_savings:.2%}",
                "task_time_reduction": f"{self.task_time_reduction:.2%}",
                "iteration_reduction": f"{self.iteration_reduction:.2%}",
                "precision": f"{self.precision:.2%}",
                "verdict_accuracy": f"{self.verdict_accuracy:.2%}",
            },
            "performance": {
                "validation_time_ms": self.validation_time_ms,
                "tokens_used": self.total_tokens_used,
                "issues_detected": self.issues_detected,
                "evidence_count": self.evidence_count,
            },
            "detection": {
                "gaming_patterns": self.gaming_patterns_detected,
                "hallucinations": self.hallucinations_detected,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
            }
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report"""
        
        targets = self.meets_targets()
        
        report = f"""# SCWT Metrics Report

## Test: {self.test_name}
- **ID**: {self.test_id}
- **Phase**: {self.phase}
- **Timestamp**: {self.timestamp.isoformat()}

## Success Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Hallucination Rate | {self.hallucination_rate:.2%} | ≤10% | {'✅' if targets['hallucination_rate'] else '❌'} |
| Knowledge Reuse | {self.knowledge_reuse_rate:.2%} | ≥30% | {'✅' if targets['knowledge_reuse'] else '❌'} |
| Token Savings | {self.token_savings:.2%} | 70-85% | {'✅' if targets['token_savings'] else '❌'} |
| Task Efficiency | {self.task_time_reduction:.2%} | ≥30% | {'✅' if targets['task_efficiency'] else '❌'} |
| Communication Efficiency | {self.iteration_reduction:.2%} | ≥20% | {'✅' if targets['communication_efficiency'] else '❌'} |
| Precision | {self.precision:.2%} | ≥85% | {'✅' if targets['precision'] else '❌'} |
| Verdict Accuracy | {self.verdict_accuracy:.2%} | ≥90% | {'✅' if targets['verdict_accuracy'] else '❌'} |

## Performance Metrics

- **Validation Time**: {self.validation_time_ms}ms
- **Tokens Used**: {self.total_tokens_used}
- **Issues Detected**: {self.issues_detected}
- **Evidence Count**: {self.evidence_count}
- **High Confidence Evidence**: {self.high_confidence_evidence}

## Detection Results

- **Gaming Patterns**: {self.gaming_patterns_detected}
- **Hallucinations**: {self.hallucinations_detected}
- **True Positives**: {self.true_positives}
- **False Positives**: {self.false_positives}
- **False Negatives**: {self.false_negatives}

## Overall Success Rate: {sum(targets.values()) / len(targets):.1%}
"""
        return report


class SCWTMetricsAggregator:
    """Aggregate metrics across multiple SCWT runs"""
    
    def __init__(self):
        self.metrics_history: List[SCWTMetrics] = []
    
    def add_metrics(self, metrics: SCWTMetrics):
        """Add metrics from a test run"""
        self.metrics_history.append(metrics)
    
    def get_phase_summary(self, phase: int) -> Dict[str, Any]:
        """Get aggregated metrics for a specific phase"""
        
        phase_metrics = [m for m in self.metrics_history if m.phase == phase]
        
        if not phase_metrics:
            return {"error": f"No metrics for phase {phase}"}
        
        # Calculate averages
        avg_hallucination = sum(m.hallucination_rate for m in phase_metrics) / len(phase_metrics)
        avg_knowledge_reuse = sum(m.knowledge_reuse_rate for m in phase_metrics) / len(phase_metrics)
        avg_token_savings = sum(m.token_savings for m in phase_metrics) / len(phase_metrics)
        avg_precision = sum(m.precision for m in phase_metrics) / len(phase_metrics)
        avg_accuracy = sum(m.verdict_accuracy for m in phase_metrics) / len(phase_metrics)
        
        # Count tests meeting targets
        tests_meeting_targets = sum(
            1 for m in phase_metrics
            if sum(m.meets_targets().values()) >= 5  # At least 5/7 targets met
        )
        
        return {
            "phase": phase,
            "total_tests": len(phase_metrics),
            "tests_passing": tests_meeting_targets,
            "pass_rate": tests_meeting_targets / len(phase_metrics),
            "averages": {
                "hallucination_rate": avg_hallucination,
                "knowledge_reuse_rate": avg_knowledge_reuse,
                "token_savings": avg_token_savings,
                "precision": avg_precision,
                "verdict_accuracy": avg_accuracy,
            }
        }
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall SCWT summary across all phases"""
        
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        phase_summaries = {}
        for phase in [1, 2, 3]:
            summary = self.get_phase_summary(phase)
            if "error" not in summary:
                phase_summaries[f"phase_{phase}"] = summary
        
        # Calculate overall success
        all_targets_met = [m.meets_targets() for m in self.metrics_history]
        overall_success = sum(
            sum(targets.values()) / len(targets)
            for targets in all_targets_met
        ) / len(all_targets_met)
        
        return {
            "total_tests_run": len(self.metrics_history),
            "overall_success_rate": overall_success,
            "phases": phase_summaries,
            "recommendation": self._get_recommendation(overall_success)
        }
    
    def _get_recommendation(self, success_rate: float) -> str:
        """Get recommendation based on success rate"""
        
        if success_rate >= 0.9:
            return "✅ Validator meets all PRD requirements. Ready for production."
        elif success_rate >= 0.7:
            return "⚠️ Validator meets most requirements. Minor improvements needed."
        elif success_rate >= 0.5:
            return "🔧 Validator partially meets requirements. Significant improvements needed."
        else:
            return "❌ Validator does not meet requirements. Major rework required."