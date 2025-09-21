"""
Handoff Strategies - Different handoff strategies and patterns.

This module implements various handoff strategies for different scenarios:
- Collaborative handoffs where agents work together
- Sequential handoffs where agents work in sequence
- Conditional handoffs based on specific conditions
- Parallel handoffs for concurrent processing
- Delegation handoffs for task transfer
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .agent_handoff_engine import (
    HandoffStrategyBase, HandoffDecision, HandoffResult, HandoffStatus,
    HandoffContext, HandoffStrategy
)

# Type hint for ArchonAgency to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..orchestration.archon_agency import ArchonAgency
from ...agents.base_agent import ArchonDependencies

logger = logging.getLogger(__name__)


class SequentialHandoffStrategy(HandoffStrategyBase):
    """Sequential handoff where agents work in sequence."""

    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute sequential handoff."""
        start_time = datetime.utcnow()
        handoff_id = str(uuid.uuid4())

        try:
            # Execute handoff through agency
            result = await agency.execute_handoff(decision)

            return HandoffResult(
                handoff_id=handoff_id,
                status=HandoffStatus.COMPLETED,
                source_agent_id=decision.source_agent_id,
                target_agent_id=decision.target_agent_id,
                execution_time=(datetime.utcnow() - start_time).total_seconds() * 1000,
                context_package_id=result.context_package_id if hasattr(result, 'context_package_id') else None,
                confidence_score=decision.confidence_score,
                metrics={"strategy": "sequential", "sequential_steps": 1}
            )
        except Exception as e:
            logger.error(f"Sequential handoff failed: {e}")
            return HandoffResult(
                handoff_id=handoff_id,
                status=HandoffStatus.FAILED,
                source_agent_id=decision.source_agent_id,
                target_agent_id=decision.target_agent_id,
                execution_time=(datetime.utcnow() - start_time).total_seconds() * 1000,
                error_message=str(e),
                confidence_score=decision.confidence_score,
                metrics={"strategy": "sequential", "error": str(e)}
            )


class CollaborativeHandoffStrategy(HandoffStrategyBase):
    """Collaborative handoff where agents work together."""

    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute collaborative handoff."""
        start_time = datetime.utcnow()
        handoff_id = str(uuid.uuid4())

        try:
            # Execute collaborative handoff
            result = await agency.execute_handoff(decision)

            return HandoffResult(
                handoff_id=handoff_id,
                status=HandoffStatus.COMPLETED,
                source_agent_id=decision.source_agent_id,
                target_agent_id=decision.target_agent_id,
                execution_time=(datetime.utcnow() - start_time).total_seconds() * 1000,
                context_package_id=result.context_package_id if hasattr(result, 'context_package_id') else None,
                confidence_score=decision.confidence_score,
                metrics={"strategy": "collaborative", "collaboration_mode": "real-time"}
            )
        except Exception as e:
            logger.error(f"Collaborative handoff failed: {e}")
            return HandoffResult(
                handoff_id=handoff_id,
                status=HandoffStatus.FAILED,
                source_agent_id=decision.source_agent_id,
                target_agent_id=decision.target_agent_id,
                execution_time=(datetime.utcnow() - start_time).total_seconds() * 1000,
                error_message=str(e),
                confidence_score=decision.confidence_score,
                metrics={"strategy": "collaborative", "error": str(e)}
            )


class ConditionalHandoffStrategy(HandoffStrategyBase):
    """Conditional handoff based on specific conditions."""

    def __init__(self, conditions: Dict[str, Any]):
        """Initialize with handoff conditions."""
        self.conditions = conditions

    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute conditional handoff."""
        start_time = datetime.utcnow()
        handoff_id = str(uuid.uuid4())

        try:
            # Check if conditions are met
            if not self._check_conditions(decision):
                return HandoffResult(
                    handoff_id=handoff_id,
                    decision=decision,
                    status=HandoffStatus.CANCELLED,
                    error_message="Handoff conditions not met",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    metrics={"strategy": "conditional", "conditions_met": False}
                )

            logger.info(f"Executing conditional handoff from {decision.source_agent} to {decision.target_agent}")

            # Proceed with handoff if conditions are met
            target_agent = agency.agents[decision.target_agent]

            handoff_message = f"""
            [CONDITIONAL HANDOFF] Conditions met for transfer from {decision.source_agent}

            Task: {decision.context.task_description}
            Original message: {decision.context.original_message}

            Conditions that triggered this handoff:
            {self._format_conditions()}

            Context: {decision.context.metadata}

            Please continue the task.
            """

            deps = ArchonDependencies(
                request_id=str(uuid.uuid4()),
                user_id=decision.context.dependencies.user_id if decision.context.dependencies else None,
                trace_id=str(uuid.uuid4()),
                context={
                    **decision.context.to_dict(),
                    "conditional_handoff": True,
                    "conditions": self.conditions
                }
            )

            if hasattr(target_agent, 'run_with_confidence'):
                response, confidence = await target_agent.run_with_confidence(
                    handoff_message,
                    deps,
                    task_description=f"Conditional handoff task: {decision.context.task_description}"
                )
            else:
                response = await target_agent.run(handoff_message, deps)
                confidence = None

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.COMPLETED,
                response_content=str(response) if response else "",
                execution_time=execution_time,
                metrics={
                    "confidence_score": confidence,
                    "strategy": "conditional",
                    "conditions_met": True
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Conditional handoff failed: {e}")

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                metrics={"strategy": "conditional", "conditions_met": False}
            )

    def _check_conditions(self, decision: HandoffDecision) -> bool:
        """Check if handoff conditions are met."""
        try:
            # Example conditions check
            if "confidence_threshold" in self.conditions:
                if decision.confidence_score < self.conditions["confidence_threshold"]:
                    return False

            if "task_complexity" in self.conditions:
                # Check task complexity (placeholder logic)
                complexity_score = len(decision.context.task_description.split())
                if complexity_score < self.conditions["task_complexity"]:
                    return False

            if "agent_workload" in self.conditions:
                # Check if target agent is overloaded (placeholder logic)
                target_capability = decision.context.metadata.get("target_capability", {})
                if target_capability.get("current_tasks", 0) > self.conditions["agent_workload"]:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking handoff conditions: {e}")
            return False

    def _format_conditions(self) -> str:
        """Format conditions for display."""
        formatted = []
        for key, value in self.conditions.items():
            formatted.append(f"  - {key}: {value}")
        return "\n".join(formatted)


class ParallelHandoffStrategy(HandoffStrategyBase):
    """Parallel handoff where multiple agents work concurrently."""

    def __init__(self, target_agents: List[str]):
        """Initialize with target agents for parallel execution."""
        self.target_agents = target_agents

    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute parallel handoff."""
        start_time = datetime.utcnow()
        handoff_id = str(uuid.uuid4())

        try:
            logger.info(f"Executing parallel handoff from {decision.source_agent} to {len(self.target_agents)} agents")

            # Prepare parallel tasks
            tasks = []
            for target_agent_name in self.target_agents:
                if target_agent_name not in agency.agents:
                    logger.warning(f"Target agent {target_agent_name} not found, skipping")
                    continue

                task = self._execute_single_handoff(target_agent_name, decision, agency)
                tasks.append(task)

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful_results = []
            failed_results = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_results.append({
                        "agent": self.target_agents[i],
                        "error": str(result)
                    })
                else:
                    successful_results.append(result)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Combine results
            combined_response = self._combine_parallel_results(successful_results)

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.COMPLETED if successful_results else HandoffStatus.FAILED,
                response_content=combined_response,
                execution_time=execution_time,
                metrics={
                    "strategy": "parallel",
                    "successful_agents": len(successful_results),
                    "failed_agents": len(failed_results),
                    "total_agents": len(self.target_agents),
                    "parallel_results": {
                        "successful": [r.response_content for r in successful_results],
                        "failed": failed_results
                    }
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Parallel handoff failed: {e}")

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                metrics={"strategy": "parallel"}
            )

    async def _execute_single_handoff(
        self,
        target_agent_name: str,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> Dict[str, Any]:
        """Execute handoff to a single target agent."""
        target_agent = agency.agents[target_agent_name]

        handoff_message = f"""
        [PARALLEL HANDOFF] Working concurrently on task from {decision.source_agent}

        Task: {decision.context.task_description}
        Original message: {decision.context.original_message}

        Context: {decision.context.metadata}

        Please provide your expertise on this task. Multiple agents are working concurrently.
        """

        deps = ArchonDependencies(
            request_id=str(uuid.uuid4()),
            user_id=decision.context.dependencies.user_id if decision.context.dependencies else None,
            trace_id=str(uuid.uuid4()),
            context={
                **decision.context.to_dict(),
                "parallel_mode": True,
                "parallel_agent_id": target_agent_name
            }
        )

        if hasattr(target_agent, 'run_with_confidence'):
            response, confidence = await target_agent.run_with_confidence(
                handoff_message,
                deps,
                task_description=f"Parallel task from {decision.source_agent}: {decision.context.task_description}"
            )
        else:
            response = await target_agent.run(handoff_message, deps)
            confidence = None

        return {
            "agent": target_agent_name,
            "response": str(response) if response else "",
            "confidence": confidence
        }

    def _combine_parallel_results(self, results: List[Dict[str, Any]]) -> str:
        """Combine results from parallel executions."""
        if not results:
            return "No results from parallel execution"

        combined = []
        for result in results:
            agent_name = result["agent"]
            response = result["response"]
            confidence = result.get("confidence")

            combined.append(f"=== {agent_name} ===")
            if confidence:
                combined.append(f"Confidence: {confidence:.2f}")
            combined.append(response)
            combined.append("")

        return "\n".join(combined)


class DelegationHandoffStrategy(HandoffStrategyBase):
    """Delegation handoff where one agent delegates complete responsibility."""

    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute delegation handoff."""
        start_time = datetime.utcnow()
        handoff_id = str(uuid.uuid4())

        try:
            logger.info(f"Executing delegation handoff from {decision.source_agent} to {decision.target_agent}")

            target_agent = agency.agents[decision.target_agent]

            delegation_message = f"""
            [DELEGATION HANDOFF] Full responsibility transferred from {decision.source_agent}

            You are now fully responsible for this task:

            Task: {decision.context.task_description}
            Original request: {decision.context.original_message}

            Context: {decision.context.metadata}

            Conversation history: {decision.context.conversation_history}

            Please take complete ownership of this task and provide a comprehensive solution.
            The original agent ({decision.source_agent}) will not be involved further.
            """

            deps = ArchonDependencies(
                request_id=str(uuid.uuid4()),
                user_id=decision.context.dependencies.user_id if decision.context.dependencies else None,
                trace_id=str(uuid.uuid4()),
                context={
                    **decision.context.to_dict(),
                    "delegation_mode": True,
                    "full_ownership": True
                }
            )

            if hasattr(target_agent, 'run_with_confidence'):
                response, confidence = await target_agent.run_with_confidence(
                    delegation_message,
                    deps,
                    task_description=f"Delegated task from {decision.source_agent}: {decision.context.task_description}"
                )
            else:
                response = await target_agent.run(delegation_message, deps)
                confidence = None

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.COMPLETED,
                response_content=str(response) if response else "",
                execution_time=execution_time,
                metrics={
                    "confidence_score": confidence,
                    "strategy": "delegation"
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Delegation handoff failed: {e}")

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                metrics={"strategy": "delegation"}
            )


@dataclass
class HandoffStrategyConfig:
    """Configuration for handoff strategies."""
    default_strategy: HandoffStrategy = HandoffStrategy.SEQUENTIAL
    fallback_strategy: HandoffStrategy = HandoffStrategy.SEQUENTIAL
    timeout_seconds: float = 30.0
    max_retries: int = 3
    enable_confidence_scoring: bool = True
    parallel_agent_limit: int = 5
    confidence_threshold: float = 0.7


class AdaptiveHandoffStrategy:
    """Adaptive handoff strategy that learns and adapts over time."""

    def __init__(self, config: HandoffStrategyConfig):
        """Initialize with configuration."""
        self.config = config
        self.strategy_performance: Dict[HandoffStrategy, Dict[str, float]] = {
            strategy: {"success_rate": 0.0, "avg_time": 0.0, "usage_count": 0}
            for strategy in HandoffStrategy
        }

    async def execute_adaptive_handoff(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute handoff with adaptive strategy selection."""
        # Select best strategy based on performance history
        selected_strategy = self._select_best_strategy(decision)

        # Update decision with selected strategy
        decision.strategy = selected_strategy

        # Execute the handoff
        strategy_instance = self._get_strategy_instance(selected_strategy)
        result = await strategy_instance.execute(decision, agency)

        # Update performance metrics
        self._update_strategy_performance(selected_strategy, result)

        return result

    def _select_best_strategy(self, decision: HandoffDecision) -> HandoffStrategy:
        """Select the best strategy based on context and performance."""
        # Simple selection logic - can be enhanced with ML
        task_complexity = len(decision.context.task_description.split())
        num_agents = len(agency.agents) if hasattr(agency, 'agents') else 1

        # Select based on task characteristics
        if task_complexity > 50 and num_agents > 2:
            return HandoffStrategy.COLLABORATIVE
        elif task_complexity > 100:
            return HandoffStrategy.PARALLEL
        elif decision.confidence_score < self.config.confidence_threshold:
            return HandoffStrategy.CONDITIONAL
        else:
            return self.config.default_strategy

    def _get_strategy_instance(self, strategy: HandoffStrategy) -> HandoffStrategyBase:
        """Get strategy instance for execution."""
        strategy_map = {
            HandoffStrategy.SEQUENTIAL: SequentialHandoffStrategy(),
            HandoffStrategy.COLLABORATIVE: CollaborativeHandoffStrategy(),
            HandoffStrategy.CONDITIONAL: ConditionalHandoffStrategy({"confidence_threshold": 0.7}),
            HandoffStrategy.PARALLEL: ParallelHandoffStrategy([]),  # Will be configured
            HandoffStrategy.DELEGATION: DelegationHandoffStrategy()
        }

        return strategy_map.get(strategy, SequentialHandoffStrategy())

    def _update_strategy_performance(self, strategy: HandoffStrategy, result: HandoffResult) -> None:
        """Update performance metrics for the strategy."""
        metrics = self.strategy_performance[strategy]
        metrics["usage_count"] += 1

        if result.status == HandoffStatus.COMPLETED:
            # Update success rate
            current_rate = metrics["success_rate"]
            new_rate = ((current_rate * (metrics["usage_count"] - 1)) + 1.0) / metrics["usage_count"]
            metrics["success_rate"] = new_rate

            # Update average time
            current_avg = metrics["avg_time"]
            new_avg = ((current_avg * (metrics["usage_count"] - 1)) + result.execution_time) / metrics["usage_count"]
            metrics["avg_time"] = new_avg

    def get_strategy_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get strategy recommendations based on current context."""
        recommendations = []

        for strategy, metrics in self.strategy_performance.items():
            if metrics["usage_count"] > 0:
                recommendation = {
                    "strategy": strategy.value,
                    "success_rate": metrics["success_rate"],
                    "avg_time": metrics["avg_time"],
                    "usage_count": metrics["usage_count"],
                    "recommended": metrics["success_rate"] > 0.8
                }
                recommendations.append(recommendation)

        # Sort by success rate
        recommendations.sort(key=lambda x: x["success_rate"], reverse=True)

        return recommendations