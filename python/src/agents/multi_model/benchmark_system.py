"""
Cross-Provider Performance Benchmarking System
Comprehensive benchmarking across all AI providers to optimize model selection.
"""

import asyncio
import time
import logging
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pandas as pd

from .model_ensemble import ModelEnsemble, TaskType, TaskRequest, ModelResponse, ModelProvider
from ..monitoring.metrics import track_agent_execution

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks to run."""
    QUALITY = "quality"
    SPEED = "speed"
    COST = "cost"
    RELIABILITY = "reliability"
    CONSISTENCY = "consistency"
    COMPLEX_REASONING = "complex_reasoning"
    CODE_GENERATION = "code_generation"
    CREATIVE_TASKS = "creative_tasks"


@dataclass
class BenchmarkTask:
    """Individual benchmark task."""
    id: str
    task_type: TaskType
    benchmark_type: BenchmarkType
    prompt: str
    expected_output: Optional[str] = None
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard, expert
    timeout: int = 30  # seconds
    system_prompt: Optional[str] = None
    reference_answer: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    task_id: str
    model_id: str
    provider: ModelProvider
    response: ModelResponse
    quality_score: float
    speed_score: float
    cost_score: float
    reliability_score: float
    overall_score: float
    evaluation_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderComparison:
    """Comparison across providers."""
    task_type: TaskType
    benchmark_type: BenchmarkType
    results: Dict[str, BenchmarkResult]  # model_id -> result
    winner: str
    winner_score: float
    score_differences: Dict[str, float]
    statistical_significance: bool
    timestamp: datetime = field(default_factory=datetime.now)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for AI providers.
    """
    
    def __init__(self, model_ensemble: ModelEnsemble, redis_client=None):
        self.ensemble = model_ensemble
        self.redis_client = redis_client
        
        # Benchmark tasks library
        self.benchmark_tasks: Dict[str, BenchmarkTask] = {}
        self.benchmark_results: Dict[str, List[BenchmarkResult]] = {}
        self.provider_comparisons: List[ProviderComparison] = []
        
        # Load benchmark tasks
        self._initialize_benchmark_tasks()
        
        # Evaluation models for quality assessment
        self.evaluation_models = [
            "anthropic:claude-3-sonnet-20240229",  # Primary evaluator
            "openai:gpt-4-turbo-preview"           # Secondary evaluator
        ]
        
        # Benchmark configuration
        self.benchmark_config = {
            "concurrent_tests": 3,
            "retry_failed": True,
            "max_retries": 2,
            "quality_weight": 0.4,
            "speed_weight": 0.2,
            "cost_weight": 0.2,
            "reliability_weight": 0.2,
            "min_samples": 5,  # Minimum samples for statistical significance
        }
        
        # Performance baselines (updated regularly)
        self.performance_baselines = {
            "response_time": 2.0,  # seconds
            "cost_per_1k_tokens": 5.0,  # dollars
            "quality_score": 0.8,  # 0-1 scale
            "reliability": 0.95,  # success rate
        }
    
    def _initialize_benchmark_tasks(self):
        """Initialize comprehensive benchmark task library."""
        
        # Coding benchmarks
        self._add_coding_benchmarks()
        
        # Creative writing benchmarks
        self._add_creative_benchmarks()
        
        # Analysis benchmarks
        self._add_analysis_benchmarks()
        
        # Reasoning benchmarks
        self._add_reasoning_benchmarks()
        
        # Speed benchmarks
        self._add_speed_benchmarks()
        
        logger.info(f"Initialized {len(self.benchmark_tasks)} benchmark tasks")
    
    def _add_coding_benchmarks(self):
        """Add coding-specific benchmark tasks."""
        
        # Basic coding task
        self.benchmark_tasks["coding_basic_function"] = BenchmarkTask(
            id="coding_basic_function",
            task_type=TaskType.CODING,
            benchmark_type=BenchmarkType.CODE_GENERATION,
            prompt="Write a Python function that calculates the factorial of a number using recursion.",
            expected_output=None,
            evaluation_criteria={
                "correctness": "Function works correctly for test inputs",
                "efficiency": "Uses appropriate algorithm",
                "style": "Follows Python conventions"
            },
            difficulty="easy",
            reference_answer="""def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
        )
        
        # Complex coding task
        self.benchmark_tasks["coding_algorithm_design"] = BenchmarkTask(
            id="coding_algorithm_design",
            task_type=TaskType.CODING,
            benchmark_type=BenchmarkType.CODE_GENERATION,
            prompt="""Design and implement a Python class for a LRU (Least Recently Used) cache with the following requirements:
- Initialize with maximum capacity
- get(key): Get value by key, return -1 if not found
- put(key, value): Insert or update key-value pair
- Both operations should be O(1) average time complexity""",
            evaluation_criteria={
                "correctness": "Implementation works correctly",
                "complexity": "Achieves O(1) time complexity",
                "completeness": "Handles all requirements"
            },
            difficulty="hard"
        )
        
        # Code review task
        self.benchmark_tasks["code_review_quality"] = BenchmarkTask(
            id="code_review_quality",
            task_type=TaskType.CODE_REVIEW,
            benchmark_type=BenchmarkType.QUALITY,
            prompt="""Review this Python code and suggest improvements:
            
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] != None and data[i] > 0:
            result.append(data[i] * 2)
    return result""",
            evaluation_criteria={
                "identifies_issues": "Finds style and efficiency problems",
                "suggests_improvements": "Provides concrete improvements",
                "explains_reasoning": "Explains why changes are needed"
            },
            difficulty="medium"
        )
    
    def _add_creative_benchmarks(self):
        """Add creative writing benchmark tasks."""
        
        self.benchmark_tasks["creative_story"] = BenchmarkTask(
            id="creative_story",
            task_type=TaskType.CREATIVE_WRITING,
            benchmark_type=BenchmarkType.CREATIVE_TASKS,
            prompt="Write a short story (200-300 words) about a robot discovering emotions for the first time.",
            evaluation_criteria={
                "creativity": "Original and imaginative content",
                "coherence": "Story flows logically",
                "emotional_impact": "Evokes appropriate emotions",
                "length": "Meets word count requirement"
            },
            difficulty="medium"
        )
        
        self.benchmark_tasks["creative_marketing"] = BenchmarkTask(
            id="creative_marketing",
            task_type=TaskType.CREATIVE_WRITING,
            benchmark_type=BenchmarkType.CREATIVE_TASKS,
            prompt="Create a compelling marketing email subject line and first paragraph for a new AI-powered productivity app.",
            evaluation_criteria={
                "persuasiveness": "Compelling and action-oriented",
                "clarity": "Clear value proposition",
                "engagement": "Captures attention"
            },
            difficulty="easy"
        )
    
    def _add_analysis_benchmarks(self):
        """Add analytical reasoning benchmark tasks."""
        
        self.benchmark_tasks["analysis_data_interpretation"] = BenchmarkTask(
            id="analysis_data_interpretation",
            task_type=TaskType.ANALYSIS,
            benchmark_type=BenchmarkType.COMPLEX_REASONING,
            prompt="""Analyze this sales data and provide insights:
Q1: Revenue $100K, Customers 1000, Avg Order $100
Q2: Revenue $120K, Customers 1100, Avg Order $109
Q3: Revenue $150K, Customers 1200, Avg Order $125
Q4: Revenue $180K, Customers 1250, Avg Order $144

What trends do you observe and what might be causing them?""",
            evaluation_criteria={
                "accuracy": "Mathematical calculations are correct",
                "insights": "Identifies meaningful trends",
                "reasoning": "Logical explanations for trends"
            },
            difficulty="medium"
        )
    
    def _add_reasoning_benchmarks(self):
        """Add logical reasoning benchmark tasks."""
        
        self.benchmark_tasks["reasoning_logic_puzzle"] = BenchmarkTask(
            id="reasoning_logic_puzzle",
            task_type=TaskType.REASONING,
            benchmark_type=BenchmarkType.COMPLEX_REASONING,
            prompt="""Solve this logic puzzle:
Three friends - Alice, Bob, and Charlie - each have a different pet (cat, dog, fish) and live in different colored houses (red, blue, green).

Clues:
1. Alice doesn't live in the red house
2. The person with the cat lives in the blue house
3. Bob doesn't have a fish
4. Charlie lives in the green house
5. The person in the red house has a dog

Who has which pet and lives in which house?""",
            evaluation_criteria={
                "correctness": "Solution is logically correct",
                "reasoning": "Shows clear logical steps",
                "completeness": "Answers all parts of the question"
            },
            difficulty="hard",
            reference_answer="Alice: fish/blue, Bob: dog/red, Charlie: cat/green"
        )
    
    def _add_speed_benchmarks(self):
        """Add speed-focused benchmark tasks."""
        
        self.benchmark_tasks["speed_simple_query"] = BenchmarkTask(
            id="speed_simple_query",
            task_type=TaskType.SIMPLE_QUERY,
            benchmark_type=BenchmarkType.SPEED,
            prompt="What is the capital of France?",
            expected_output="Paris",
            evaluation_criteria={
                "speed": "Response time under 1 second",
                "accuracy": "Correct answer"
            },
            difficulty="easy",
            timeout=5
        )
        
        self.benchmark_tasks["speed_calculation"] = BenchmarkTask(
            id="speed_calculation",
            task_type=TaskType.SIMPLE_QUERY,
            benchmark_type=BenchmarkType.SPEED,
            prompt="Calculate 15% of 240 and show your work.",
            expected_output="36",
            evaluation_criteria={
                "speed": "Fast response",
                "accuracy": "Correct calculation",
                "clarity": "Shows calculation steps"
            },
            difficulty="easy",
            timeout=5
        )
    
    async def run_comprehensive_benchmark(
        self,
        models_to_test: Optional[List[str]] = None,
        benchmark_types: Optional[List[BenchmarkType]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all providers."""
        logger.info("Starting comprehensive benchmark suite")
        
        if models_to_test is None:
            models_to_test = list(self.ensemble.model_configs.keys())
        
        if benchmark_types is None:
            benchmark_types = list(BenchmarkType)
        
        # Filter tasks by benchmark types
        tasks_to_run = [
            task for task in self.benchmark_tasks.values()
            if task.benchmark_type in benchmark_types
        ]
        
        logger.info(f"Running {len(tasks_to_run)} tasks on {len(models_to_test)} models")
        
        # Run benchmarks
        all_results = []
        
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=self.benchmark_config["concurrent_tests"]) as executor:
            
            for task in tasks_to_run:
                task_results = await self._run_task_benchmark(task, models_to_test)
                all_results.extend(task_results)
        
        # Analyze results
        analysis = self._analyze_benchmark_results(all_results)
        
        # Store results
        await self._store_benchmark_results(all_results, analysis)
        
        logger.info(f"Benchmark completed with {len(all_results)} total results")
        
        return {
            "total_tasks": len(tasks_to_run),
            "total_results": len(all_results),
            "models_tested": len(models_to_test),
            "analysis": analysis,
            "raw_results": all_results
        }
    
    async def _run_task_benchmark(
        self,
        task: BenchmarkTask,
        models_to_test: List[str]
    ) -> List[BenchmarkResult]:
        """Run benchmark task on all specified models."""
        results = []
        
        for model_id in models_to_test:
            if model_id not in self.ensemble.model_configs:
                continue
            
            try:
                result = await self._run_single_benchmark(task, model_id)
                results.append(result)
                
                # Track in monitoring system
                track_agent_execution(
                    agent_type=f"benchmark_{task.benchmark_type.value}",
                    tier=model_id.split(":")[1] if ":" in model_id else model_id,
                    status="success" if result.response.success else "failure",
                    duration=result.response.response_time,
                    cost=result.response.cost
                )
                
            except Exception as e:
                logger.error(f"Benchmark failed for {model_id} on {task.id}: {e}")
                
                # Create failed result
                failed_result = BenchmarkResult(
                    task_id=task.id,
                    model_id=model_id,
                    provider=self.ensemble.model_configs[model_id].provider,
                    response=ModelResponse(
                        content="",
                        provider=self.ensemble.model_configs[model_id].provider,
                        model_name=model_id,
                        tokens_used=0,
                        cost=0.0,
                        response_time=0.0,
                        quality_score=0.0,
                        success=False,
                        error_message=str(e)
                    ),
                    quality_score=0.0,
                    speed_score=0.0,
                    cost_score=0.0,
                    reliability_score=0.0,
                    overall_score=0.0
                )
                results.append(failed_result)
        
        return results
    
    async def _run_single_benchmark(
        self,
        task: BenchmarkTask,
        model_id: str
    ) -> BenchmarkResult:
        """Run single benchmark test."""
        
        # Create task request
        request = TaskRequest(
            prompt=task.prompt,
            task_type=task.task_type,
            max_tokens=1000,
            temperature=0.7,
            system_prompt=task.system_prompt
        )
        
        # Execute with specific model
        start_time = time.time()
        response = await self.ensemble._execute_with_model(model_id, request)
        execution_time = time.time() - start_time
        
        # Evaluate response
        quality_score = await self._evaluate_quality(task, response)
        speed_score = self._evaluate_speed(response.response_time, task.timeout)
        cost_score = self._evaluate_cost(response.cost, task.benchmark_type)
        reliability_score = 1.0 if response.success else 0.0
        
        # Calculate overall score
        overall_score = (
            quality_score * self.benchmark_config["quality_weight"] +
            speed_score * self.benchmark_config["speed_weight"] +
            cost_score * self.benchmark_config["cost_weight"] +
            reliability_score * self.benchmark_config["reliability_weight"]
        )
        
        return BenchmarkResult(
            task_id=task.id,
            model_id=model_id,
            provider=self.ensemble.model_configs[model_id].provider,
            response=response,
            quality_score=quality_score,
            speed_score=speed_score,
            cost_score=cost_score,
            reliability_score=reliability_score,
            overall_score=overall_score,
            evaluation_details={
                "execution_time": execution_time,
                "task_difficulty": task.difficulty,
                "benchmark_type": task.benchmark_type.value
            }
        )
    
    async def _evaluate_quality(
        self,
        task: BenchmarkTask,
        response: ModelResponse
    ) -> float:
        """Evaluate response quality using multiple methods."""
        if not response.success:
            return 0.0
        
        quality_scores = []
        
        # 1. Reference answer comparison (if available)
        if task.reference_answer:
            similarity_score = self._calculate_similarity(
                response.content, task.reference_answer
            )
            quality_scores.append(similarity_score)
        
        # 2. Expected output matching (if available)
        if task.expected_output:
            if task.expected_output.lower() in response.content.lower():
                quality_scores.append(1.0)
            else:
                quality_scores.append(0.0)
        
        # 3. AI-powered evaluation (for complex tasks)
        if len(quality_scores) == 0 or task.benchmark_type in [
            BenchmarkType.CREATIVE_TASKS,
            BenchmarkType.COMPLEX_REASONING,
            BenchmarkType.CODE_GENERATION
        ]:
            ai_score = await self._ai_powered_evaluation(task, response)
            quality_scores.append(ai_score)
        
        # 4. Task-specific criteria
        criteria_score = self._evaluate_task_criteria(task, response)
        quality_scores.append(criteria_score)
        
        # Return average of all scores
        return statistics.mean(quality_scores) if quality_scores else 0.5
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _ai_powered_evaluation(
        self,
        task: BenchmarkTask,
        response: ModelResponse
    ) -> float:
        """Use AI model to evaluate response quality."""
        evaluation_prompt = f"""
Evaluate the following response to this task on a scale of 0.0 to 1.0:

Task: {task.prompt}

Response: {response.content}

Evaluation Criteria:
{json.dumps(task.evaluation_criteria, indent=2)}

Provide only a numerical score between 0.0 and 1.0, where:
- 1.0 = Excellent, exceeds expectations
- 0.8 = Good, meets expectations well
- 0.6 = Adequate, meets basic requirements
- 0.4 = Poor, partially meets requirements
- 0.2 = Very poor, barely addresses task
- 0.0 = Completely inadequate

Score:"""
        
        try:
            # Use a reliable model for evaluation
            eval_request = TaskRequest(
                prompt=evaluation_prompt,
                task_type=TaskType.ANALYSIS,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=100
            )
            
            eval_response = await self.ensemble._execute_with_model(
                self.evaluation_models[0], eval_request
            )
            
            if eval_response.success:
                # Extract score from response
                score_text = eval_response.content.strip()
                try:
                    score = float(score_text)
                    return max(0.0, min(1.0, score))  # Clamp to valid range
                except ValueError:
                    # Try to extract first number from response
                    import re
                    numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
                    if numbers:
                        return float(numbers[0])
            
        except Exception as e:
            logger.warning(f"AI evaluation failed: {e}")
        
        return 0.5  # Neutral score on failure
    
    def _evaluate_task_criteria(self, task: BenchmarkTask, response: ModelResponse) -> float:
        """Evaluate based on task-specific criteria."""
        if not task.evaluation_criteria:
            return 0.5
        
        # Simple heuristic-based evaluation
        score = 0.0
        total_criteria = len(task.evaluation_criteria)
        
        content = response.content.lower()
        
        # Check for basic quality indicators
        if len(content) > 50:  # Substantial response
            score += 0.3
        
        if len(content.split()) > 10:  # Detailed response
            score += 0.2
        
        # Task type specific checks
        if task.task_type == TaskType.CODING:
            if 'def ' in content or 'function' in content or 'class ' in content:
                score += 0.3
            if '```' in response.content:  # Code formatting
                score += 0.2
        
        elif task.task_type == TaskType.CREATIVE_WRITING:
            if len(content.split()) >= 100:  # Adequate length
                score += 0.4
            if any(word in content for word in ['emotion', 'feel', 'thought']):
                score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_speed(self, response_time: float, timeout: int) -> float:
        """Evaluate response speed."""
        if response_time <= 0:
            return 0.0
        
        # Normalize against baseline and timeout
        baseline = self.performance_baselines["response_time"]
        
        if response_time <= baseline / 2:
            return 1.0  # Excellent speed
        elif response_time <= baseline:
            return 0.8  # Good speed
        elif response_time <= baseline * 2:
            return 0.6  # Acceptable speed
        elif response_time <= timeout:
            return 0.3  # Slow but within timeout
        else:
            return 0.0  # Too slow
    
    def _evaluate_cost(self, cost: float, benchmark_type: BenchmarkType) -> float:
        """Evaluate cost efficiency."""
        if cost <= 0:
            return 1.0  # Free is best
        
        baseline_cost = self.performance_baselines["cost_per_1k_tokens"]
        
        # Adjust baseline based on benchmark type
        if benchmark_type in [BenchmarkType.SPEED, BenchmarkType.SIMPLE_QUERY]:
            # Expect lower cost for simple tasks
            target_cost = baseline_cost * 0.2
        elif benchmark_type in [BenchmarkType.COMPLEX_REASONING, BenchmarkType.CREATIVE_TASKS]:
            # Accept higher cost for complex tasks
            target_cost = baseline_cost * 2.0
        else:
            target_cost = baseline_cost
        
        if cost <= target_cost * 0.5:
            return 1.0  # Excellent cost efficiency
        elif cost <= target_cost:
            return 0.8  # Good cost efficiency
        elif cost <= target_cost * 2:
            return 0.6  # Acceptable cost
        elif cost <= target_cost * 5:
            return 0.3  # Expensive
        else:
            return 0.0  # Too expensive
    
    def _analyze_benchmark_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results to find patterns and winners."""
        if not results:
            return {}
        
        # Group results by different dimensions
        by_provider = {}
        by_model = {}
        by_task_type = {}
        by_benchmark_type = {}
        
        for result in results:
            # By provider
            provider = result.provider.value
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(result)
            
            # By model
            if result.model_id not in by_model:
                by_model[result.model_id] = []
            by_model[result.model_id].append(result)
            
            # By task type (from benchmark tasks)
            task = self.benchmark_tasks.get(result.task_id)
            if task:
                task_type = task.task_type.value
                if task_type not in by_task_type:
                    by_task_type[task_type] = []
                by_task_type[task_type].append(result)
                
                # By benchmark type
                bench_type = task.benchmark_type.value
                if bench_type not in by_benchmark_type:
                    by_benchmark_type[bench_type] = []
                by_benchmark_type[bench_type].append(result)
        
        # Calculate statistics
        analysis = {
            "overall": self._calculate_statistics(results),
            "by_provider": {},
            "by_model": {},
            "by_task_type": {},
            "by_benchmark_type": {},
            "winners": {},
            "recommendations": []
        }
        
        # Provider analysis
        for provider, provider_results in by_provider.items():
            analysis["by_provider"][provider] = self._calculate_statistics(provider_results)
        
        # Model analysis
        for model_id, model_results in by_model.items():
            analysis["by_model"][model_id] = self._calculate_statistics(model_results)
        
        # Task type analysis
        for task_type, type_results in by_task_type.items():
            analysis["by_task_type"][task_type] = self._calculate_statistics(type_results)
        
        # Benchmark type analysis
        for bench_type, bench_results in by_benchmark_type.items():
            analysis["by_benchmark_type"][bench_type] = self._calculate_statistics(bench_results)
        
        # Find winners in each category
        analysis["winners"] = self._find_category_winners(by_model, by_task_type)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _calculate_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate statistics for a group of results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.response.success]
        
        overall_scores = [r.overall_score for r in successful_results]
        quality_scores = [r.quality_score for r in successful_results]
        speed_scores = [r.speed_score for r in successful_results]
        cost_scores = [r.cost_score for r in successful_results]
        response_times = [r.response.response_time for r in successful_results]
        costs = [r.response.cost for r in successful_results]
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "avg_overall_score": statistics.mean(overall_scores) if overall_scores else 0,
            "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0,
            "avg_speed_score": statistics.mean(speed_scores) if speed_scores else 0,
            "avg_cost_score": statistics.mean(cost_scores) if cost_scores else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "avg_cost": statistics.mean(costs) if costs else 0,
            "median_overall_score": statistics.median(overall_scores) if overall_scores else 0,
            "std_overall_score": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            "min_score": min(overall_scores) if overall_scores else 0,
            "max_score": max(overall_scores) if overall_scores else 0,
        }
    
    def _find_category_winners(
        self,
        by_model: Dict[str, List[BenchmarkResult]],
        by_task_type: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, Any]:
        """Find winners in different categories."""
        winners = {}
        
        # Overall winner
        model_scores = {}
        for model_id, results in by_model.items():
            successful = [r for r in results if r.response.success]
            if successful:
                model_scores[model_id] = statistics.mean([r.overall_score for r in successful])
        
        if model_scores:
            overall_winner = max(model_scores.keys(), key=lambda k: model_scores[k])
            winners["overall"] = {
                "model": overall_winner,
                "score": model_scores[overall_winner]
            }
        
        # Winners by task type
        winners["by_task_type"] = {}
        for task_type, results in by_task_type.items():
            task_model_scores = {}
            for result in results:
                if result.response.success:
                    if result.model_id not in task_model_scores:
                        task_model_scores[result.model_id] = []
                    task_model_scores[result.model_id].append(result.overall_score)
            
            # Average scores per model for this task type
            avg_scores = {
                model: statistics.mean(scores)
                for model, scores in task_model_scores.items()
            }
            
            if avg_scores:
                winner = max(avg_scores.keys(), key=lambda k: avg_scores[k])
                winners["by_task_type"][task_type] = {
                    "model": winner,
                    "score": avg_scores[winner]
                }
        
        return winners
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Overall performance recommendation
        if "winners" in analysis and "overall" in analysis["winners"]:
            winner = analysis["winners"]["overall"]
            recommendations.append(
                f"For general-purpose tasks, use {winner['model']} "
                f"(overall score: {winner['score']:.3f})"
            )
        
        # Task-specific recommendations
        if "by_task_type" in analysis["winners"]:
            for task_type, winner_info in analysis["winners"]["by_task_type"].items():
                recommendations.append(
                    f"For {task_type} tasks, use {winner_info['model']} "
                    f"(score: {winner_info['score']:.3f})"
                )
        
        # Performance insights
        if "by_provider" in analysis:
            # Find most reliable provider
            provider_reliability = {
                provider: stats.get("success_rate", 0)
                for provider, stats in analysis["by_provider"].items()
            }
            if provider_reliability:
                most_reliable = max(provider_reliability.keys(), key=lambda k: provider_reliability[k])
                recommendations.append(
                    f"Most reliable provider: {most_reliable} "
                    f"({provider_reliability[most_reliable]:.1%} success rate)"
                )
            
            # Find most cost-effective
            provider_cost_scores = {
                provider: stats.get("avg_cost_score", 0)
                for provider, stats in analysis["by_provider"].items()
            }
            if provider_cost_scores:
                most_cost_effective = max(provider_cost_scores.keys(), key=lambda k: provider_cost_scores[k])
                recommendations.append(
                    f"Most cost-effective provider: {most_cost_effective} "
                    f"(cost score: {provider_cost_scores[most_cost_effective]:.3f})"
                )
        
        return recommendations
    
    async def _store_benchmark_results(
        self,
        results: List[BenchmarkResult],
        analysis: Dict[str, Any]
    ):
        """Store benchmark results for future reference."""
        if self.redis_client:
            try:
                # Store results summary
                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "total_results": len(results),
                    "analysis": analysis
                }
                
                self.redis_client.setex(
                    "benchmark_results_latest",
                    3600 * 24,  # 24 hours
                    json.dumps(summary)
                )
                
                # Store detailed results (with shorter TTL)
                detailed_results = [
                    {
                        "task_id": r.task_id,
                        "model_id": r.model_id,
                        "provider": r.provider.value,
                        "overall_score": r.overall_score,
                        "quality_score": r.quality_score,
                        "speed_score": r.speed_score,
                        "cost_score": r.cost_score,
                        "reliability_score": r.reliability_score,
                        "response_time": r.response.response_time,
                        "cost": r.response.cost,
                        "success": r.response.success,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in results
                ]
                
                self.redis_client.setex(
                    "benchmark_results_detailed",
                    3600,  # 1 hour
                    json.dumps(detailed_results)
                )
                
                logger.info("Benchmark results stored successfully")
                
            except Exception as e:
                logger.error(f"Failed to store benchmark results: {e}")
    
    async def get_latest_benchmark_results(self) -> Optional[Dict[str, Any]]:
        """Get the latest benchmark results."""
        if self.redis_client:
            try:
                data = self.redis_client.get("benchmark_results_latest")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Failed to retrieve benchmark results: {e}")
        
        return None
    
    async def schedule_regular_benchmarks(self, interval_hours: int = 24):
        """Schedule regular benchmarking to track performance over time."""
        logger.info(f"Scheduling regular benchmarks every {interval_hours} hours")
        
        while True:
            try:
                # Run subset of benchmarks regularly
                critical_benchmarks = [
                    BenchmarkType.SPEED,
                    BenchmarkType.RELIABILITY,
                    BenchmarkType.QUALITY
                ]
                
                await self.run_comprehensive_benchmark(
                    benchmark_types=critical_benchmarks
                )
                
                # Wait for next scheduled run
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Scheduled benchmark failed: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour