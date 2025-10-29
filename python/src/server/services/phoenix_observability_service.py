"""
Enhanced Phoenix Observability Integration Service for Archon v6.0.0+

Provides comprehensive AI/ML observability using the latest Phoenix platform:
- Advanced LLM call tracing and monitoring with OpenTelemetry
- Real-time agent behavior tracking and collaboration analysis
- Performance analytics with cost optimization insights
- Comprehensive error tracking and debugging capabilities
- Vector database performance monitoring
- Multi-modal AI workflow observability

Integration Features:
- OpenTelemetry-based distributed tracing
- OpenInference instrumentation for AI/ML libraries
- Real-time dashboard at http://localhost:6006
- Automated performance insights and recommendations
- Agent collaboration pattern analysis
- RAG pipeline performance monitoring
- Cost tracking and optimization suggestions
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from enum import Enum
from datetime import datetime, timedelta

import phoenix as px
from phoenix.otel import register
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.llamaindex import LlamaIndexInstrumentor

from ..config.logfire_config import get_logger

logger = get_logger(__name__)


class ObservabilityLevel(Enum):
    """Observability levels for different components"""
    BASIC = "basic"          # Essential metrics and traces
    STANDARD = "standard"    # Detailed monitoring with analytics
    COMPREHENSIVE = "comprehensive"  # Full observability with deep insights


@dataclass
class PhoenixConfig:
    """Phoenix configuration"""
    enabled: bool = True
    collector_endpoint: str = "http://localhost:6006"
    project_name: str = "archon-ai-workflows"
    observability_level: ObservabilityLevel = ObservabilityLevel.STANDARD
    batch_size: int = 512
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000
    sampling_rate: float = 1.0  # Sample 100% of traces for comprehensive monitoring
    auto_instrumentation: bool = True


@dataclass
class AIMetrics:
    """Enhanced AI/ML metrics collection"""
    llm_calls_total: int = 0
    llm_calls_successful: int = 0
    llm_calls_failed: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    rag_queries_total: int = 0
    rag_queries_successful: int = 0
    embedding_generations: int = 0
    knowledge_base_hits: int = 0
    model_usage_counts: Dict[str, int] = field(default_factory=dict)
    provider_costs: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Enhanced agent behavior metrics"""
    agents_active: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration: float = 0.0
    collaboration_events: int = 0
    handoff_success_rate: float = 0.0
    agent_types: Dict[str, int] = field(default_factory=dict)
    agent_success_rates: Dict[str, float] = field(default_factory=dict)
    collaboration_patterns: Dict[str, int] = field(default_factory=dict)


@dataclass
class VectorDBMetrics:
    """Vector database performance metrics"""
    query_count: int = 0
    query_success_count: int = 0
    average_query_time_ms: float = 0.0
    index_size_mb: float = 0.0
    total_vectors: int = 0
    embedding_dimension: int = 0
    cache_hit_rate: float = 0.0
    indexing_performance_ms: float = 0.0
    similarity_threshold_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class RAGPipelineMetrics:
    """RAG pipeline performance metrics"""
    retrieval_count: int = 0
    retrieval_success_count: int = 0
    average_retrieval_time_ms: float = 0.0
    average_documents_retrieved: float = 0.0
    reranking_count: int = 0
    reranking_success_count: int = 0
    average_reranking_time_ms: float = 0.0
    context_relevance_scores: List[float] = field(default_factory=list)
    answer_quality_scores: List[float] = field(default_factory=list)


class PhoenixObservabilityService:
    """Enhanced Phoenix observability service for Archon AI workflows"""

    def __init__(self, config: Optional[PhoenixConfig] = None):
        self.config = config or PhoenixConfig()
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self.initialized = False
        self.session_start_time = time.time()

        # Enhanced metrics tracking
        self.ai_metrics = AIMetrics()
        self.agent_metrics = AgentMetrics()
        self.vector_db_metrics = VectorDBMetrics()
        self.rag_pipeline_metrics = RAGPipelineMetrics()

        # Performance tracking
        self.performance_buffer: List[Dict[str, Any]] = []
        self.error_buffer: List[Dict[str, Any]] = []
        self.collaboration_events: List[Dict[str, Any]] = []
        self.knowledge_base_usage: List[Dict[str, Any]] = []

        # Advanced features
        self.cost_tracking_enabled = True
        self.agent_graph_data: Dict[str, List[str]] = {}
        self.performance_alerts: List[Dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize Phoenix observability"""
        try:
            if not self.config.enabled:
                logger.info("Phoenix observability disabled")
                return True

            logger.info("Initializing Phoenix observability...")

            # Launch Phoenix if not running
            try:
                px.launch_app(
                    host="localhost",
                    port=6006,
                    log_level="WARNING"
                )
                logger.info("Phoenix app launched at http://localhost:6006")
            except Exception as e:
                logger.warning(f"Phoenix may already be running: {e}")

            # Configure OpenTelemetry
            self.tracer_provider = trace.get_tracer_provider()

            # Add OTLP exporter for Phoenix
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.collector_endpoint + "/v1/traces",
                insecure=True
            )

            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_export_batch_size=self.config.max_export_batch_size,
                export_timeout_millis=self.config.export_timeout_millis
            )

            self.tracer_provider.add_span_processor(span_processor)

            # Register with Phoenix
            tracer_provider = register(
                project_name=self.config.project_name,
                endpoint=self.config.collector_endpoint,
                batch_size=self.config.batch_size
            )

            self.tracer = tracer_provider.get_tracer(__name__)

            # Auto-instrumentation
            if self.config.auto_instrumentation:
                await self._setup_auto_instrumentation()

            self.initialized = True
            logger.info(f"Phoenix observability initialized: {self.config.collector_endpoint}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Phoenix: {e}")
            return False

    async def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for AI/ML libraries"""
        try:
            # Instrument OpenAI
            try:
                OpenAIInstrumentor().instrument()
                logger.info("OpenAI instrumentation enabled")
            except Exception as e:
                logger.warning(f"OpenAI instrumentation failed: {e}")

            # Instrument LlamaIndex
            try:
                LlamaIndexInstrumentor().instrument()
                logger.info("LlamaIndex instrumentation enabled")
            except Exception as e:
                logger.warning(f"LlamaIndex instrumentation failed: {e}")

        except Exception as e:
            logger.warning(f"Auto-instrumentation setup failed: {e}")

    @asynccontextmanager
    async def trace_llm_call(
        self,
        provider: str,
        model: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Trace an LLM call with comprehensive monitoring"""
        if not self.initialized:
            yield {}
            return

        span_name = f"llm.{provider}.{operation}"
        start_time = time.time()

        with self.tracer.start_as_current_span(span_name) as span:
            try:
                # Set span attributes
                span.set_attribute("llm.provider", provider)
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.operation", operation)
                span.set_attribute("observability.level", self.config.observability_level.value)

                if metadata:
                    for key, value in metadata.items():
                        span.set_attribute(f"llm.{key}", str(value))

                # Update metrics
                self.ai_metrics.llm_calls_total += 1

                yield {
                    "span": span,
                    "start_time": start_time,
                    "trace_id": span.get_span_context().trace_id,
                    "span_id": span.get_span_context().span_id
                }

                # Record success
                self.ai_metrics.llm_calls_successful += 1

            except Exception as e:
                # Record failure
                self.ai_metrics.llm_calls_failed += 1
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

                # Add to error buffer
                self.error_buffer.append({
                    "timestamp": time.time(),
                    "type": "llm_error",
                    "provider": provider,
                    "model": model,
                    "operation": operation,
                    "error": str(e),
                    "trace_id": span.get_span_context().trace_id if span else None
                })

                raise
            finally:
                # Calculate duration
                duration = time.time() - start_time

                # Update performance metrics
                self.ai_metrics.average_response_time = (
                    (self.ai_metrics.average_response_time * (self.ai_metrics.llm_calls_total - 1) + duration) /
                    self.ai_metrics.llm_calls_total
                )

                # Update error rate
                self.ai_metrics.error_rate = (
                    self.ai_metrics.llm_calls_failed / self.ai_metrics.llm_calls_total
                ) if self.ai_metrics.llm_calls_total > 0 else 0.0

                span.set_attribute("llm.duration", duration)
                span.set_attribute("llm.tokens_used", metadata.get("tokens", 0) if metadata else 0)
                span.set_attribute("llm.cost_usd", metadata.get("cost", 0.0) if metadata else 0.0)

    @asynccontextmanager
    async def trace_agent_operation(
        self,
        agent_name: str,
        operation: str,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Trace an agent operation with behavior monitoring"""
        if not self.initialized:
            yield {}
            return

        span_name = f"agent.{agent_name}.{operation}"
        start_time = time.time()

        with self.tracer.start_as_current_span(span_name) as span:
            try:
                # Set span attributes
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.operation", operation)
                span.set_attribute("agent.task_type", task_type)

                if metadata:
                    for key, value in metadata.items():
                        span.set_attribute(f"agent.{key}", str(value))

                # Update metrics
                if operation == "start":
                    self.agent_metrics.agents_active += 1
                elif operation == "complete":
                    self.agent_metrics.tasks_completed += 1
                    self.agent_metrics.agents_active = max(0, self.agent_metrics.agents_active - 1)

                yield {
                    "span": span,
                    "start_time": start_time,
                    "trace_id": span.get_span_context().trace_id,
                    "span_id": span.get_span_context().span_id
                }

            except Exception as e:
                # Record failure
                self.agent_metrics.tasks_failed += 1
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

                # Add to error buffer
                self.error_buffer.append({
                    "timestamp": time.time(),
                    "type": "agent_error",
                    "agent_name": agent_name,
                    "operation": operation,
                    "task_type": task_type,
                    "error": str(e),
                    "trace_id": span.get_span_context().trace_id if span else None
                })

                raise
            finally:
                # Calculate duration
                duration = time.time() - start_time

                # Update agent metrics
                if operation == "complete":
                    self.agent_metrics.average_task_duration = (
                        (self.agent_metrics.average_task_duration * (self.agent_metrics.tasks_completed - 1) + duration) /
                        self.agent_metrics.tasks_completed
                    ) if self.agent_metrics.tasks_completed > 0 else duration

    def record_performance_event(
        self,
        event_type: str,
        component: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance event"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "component": component,
            "metrics": metrics,
            "metadata": metadata or {}
        }

        self.performance_buffer.append(event)

        # Keep buffer size manageable
        if len(self.performance_buffer) > 1000:
            self.performance_buffer = self.performance_buffer[-500:]

    def record_agent_collaboration(
        self,
        from_agent: str,
        to_agent: str,
        collaboration_type: str,
        context: Dict[str, Any]
    ) -> None:
        """Record agent collaboration events"""
        self.agent_metrics.collaboration_events += 1

        # Update collaboration patterns
        pattern_key = f"{from_agent}->{to_agent}"
        self.agent_metrics.collaboration_patterns[pattern_key] = \
            self.agent_metrics.collaboration_patterns.get(pattern_key, 0) + 1

        # Update agent graph data
        if from_agent not in self.agent_graph_data:
            self.agent_graph_data[from_agent] = []
        if to_agent not in self.agent_graph_data[from_agent]:
            self.agent_graph_data[from_agent].append(to_agent)

        event = {
            "timestamp": time.time(),
            "type": "agent_collaboration",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "collaboration_type": collaboration_type,
            "context": context
        }

        self.collaboration_events.append(event)
        self.performance_buffer.append(event)

    @asynccontextmanager
    async def trace_vector_db_query(
        self,
        operation: str,
        vector_store: str,
        query_metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Trace vector database operations"""
        if not self.initialized:
            yield {}
            return

        span_name = f"vector_db.{vector_store}.{operation}"
        start_time = time.time()

        with self.tracer.start_as_current_span(span_name) as span:
            try:
                # Set span attributes
                span.set_attribute("vector_db.store", vector_store)
                span.set_attribute("vector_db.operation", operation)

                if query_metadata:
                    for key, value in query_metadata.items():
                        span.set_attribute(f"vector_db.{key}", str(value))

                # Update metrics
                self.vector_db_metrics.query_count += 1

                yield {
                    "span": span,
                    "start_time": start_time,
                    "trace_id": span.get_span_context().trace_id,
                    "span_id": span.get_span_context().span_id
                }

                # Record success
                self.vector_db_metrics.query_success_count += 1

            except Exception as e:
                # Record failure
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

                # Add to error buffer
                self.error_buffer.append({
                    "timestamp": time.time(),
                    "type": "vector_db_error",
                    "vector_store": vector_store,
                    "operation": operation,
                    "error": str(e),
                    "trace_id": span.get_span_context().trace_id if span else None
                })

                raise
            finally:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Update vector DB metrics
                if self.vector_db_metrics.query_count > 0:
                    self.vector_db_metrics.average_query_time_ms = (
                        (self.vector_db_metrics.average_query_time_ms * (self.vector_db_metrics.query_count - 1) + duration_ms) /
                        self.vector_db_metrics.query_count
                    )

                span.set_attribute("vector_db.duration_ms", duration_ms)

    @asynccontextmanager
    async def trace_rag_pipeline(
        self,
        query: str,
        retrieval_config: Dict[str, Any],
        pipeline_stage: str = "full_pipeline"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Trace RAG pipeline operations"""
        if not self.initialized:
            yield {}
            return

        span_name = f"rag.{pipeline_stage}"
        start_time = time.time()

        with self.tracer.start_as_current_span(span_name) as span:
            try:
                # Set span attributes
                span.set_attribute("rag.query", query)
                span.set_attribute("rag.stage", pipeline_stage)
                span.set_attribute("rag.query_length", len(query))

                if retrieval_config:
                    for key, value in retrieval_config.items():
                        span.set_attribute(f"rag.{key}", str(value))

                # Update metrics
                if pipeline_stage == "retrieval":
                    self.rag_pipeline_metrics.retrieval_count += 1
                elif pipeline_stage == "reranking":
                    self.rag_pipeline_metrics.reranking_count += 1

                yield {
                    "span": span,
                    "start_time": start_time,
                    "trace_id": span.get_span_context().trace_id,
                    "span_id": span.get_span_context().span_id
                }

                # Record success
                if pipeline_stage == "retrieval":
                    self.rag_pipeline_metrics.retrieval_success_count += 1
                elif pipeline_stage == "reranking":
                    self.rag_pipeline_metrics.reranking_success_count += 1

                # Update AI metrics for RAG queries
                self.ai_metrics.rag_queries_total += 1
                self.ai_metrics.rag_queries_successful += 1

            except Exception as e:
                # Record failure
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

                # Update failure metrics
                if pipeline_stage == "retrieval":
                    self.rag_pipeline_metrics.retrieval_success_count -= 1  # Don't increment success
                elif pipeline_stage == "reranking":
                    self.rag_pipeline_metrics.reranking_success_count -= 1  # Don't increment success

                self.ai_metrics.rag_queries_successful -= 1  # Don't increment success

                # Add to error buffer
                self.error_buffer.append({
                    "timestamp": time.time(),
                    "type": "rag_error",
                    "stage": pipeline_stage,
                    "query": query,
                    "error": str(e),
                    "trace_id": span.get_span_context().trace_id if span else None
                })

                raise
            finally:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Update RAG pipeline metrics
                if pipeline_stage == "retrieval":
                    if self.rag_pipeline_metrics.retrieval_count > 0:
                        self.rag_pipeline_metrics.average_retrieval_time_ms = (
                            (self.rag_pipeline_metrics.average_retrieval_time_ms *
                             (self.rag_pipeline_metrics.retrieval_count - 1) + duration_ms) /
                            self.rag_pipeline_metrics.retrieval_count
                        )
                elif pipeline_stage == "reranking":
                    if self.rag_pipeline_metrics.reranking_count > 0:
                        self.rag_pipeline_metrics.average_reranking_time_ms = (
                            (self.rag_pipeline_metrics.average_reranking_time_ms *
                             (self.rag_pipeline_metrics.reranking_count - 1) + duration_ms) /
                            self.rag_pipeline_metrics.reranking_count
                        )

                span.set_attribute("rag.duration_ms", duration_ms)

    def record_knowledge_base_usage(
        self,
        source_id: str,
        source_type: str,
        documents_accessed: int,
        relevance_score: float,
        query: str
    ) -> None:
        """Record knowledge base usage patterns"""
        self.ai_metrics.knowledge_base_hits += 1

        usage_event = {
            "timestamp": time.time(),
            "source_id": source_id,
            "source_type": source_type,
            "documents_accessed": documents_accessed,
            "relevance_score": relevance_score,
            "query": query,
            "query_length": len(query)
        }

        self.knowledge_base_usage.append(usage_event)

        # Keep buffer manageable
        if len(self.knowledge_base_usage) > 1000:
            self.knowledge_base_usage = self.knowledge_base_usage[-500:]

    def record_embedding_generation(
        self,
        model: str,
        text_count: int,
        total_tokens: int,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """Record embedding generation metrics"""
        self.ai_metrics.embedding_generations += 1

        # Update model usage counts
        self.ai_metrics.model_usage_counts[model] = \
            self.ai_metrics.model_usage_counts.get(model, 0) + 1

        if success:
            self.ai_metrics.total_tokens_used += total_tokens

        event = {
            "timestamp": time.time(),
            "type": "embedding_generation",
            "model": model,
            "text_count": text_count,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "success": success
        }

        self.performance_buffer.append(event)

    async def get_ai_metrics(self) -> AIMetrics:
        """Get current AI/ML metrics"""
        return self.ai_metrics

    async def get_agent_metrics(self) -> AgentMetrics:
        """Get current agent metrics"""
        return self.agent_metrics

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary and insights"""
        uptime = time.time() - self.session_start_time

        return {
            "uptime_seconds": uptime,
            "ai_metrics": asdict(self.ai_metrics),
            "agent_metrics": asdict(self.agent_metrics),
            "vector_db_metrics": asdict(self.vector_db_metrics),
            "rag_pipeline_metrics": asdict(self.rag_pipeline_metrics),
            "performance_events_count": len(self.performance_buffer),
            "error_events_count": len(self.error_buffer),
            "collaboration_events_count": len(self.collaboration_events),
            "knowledge_base_usage_count": len(self.knowledge_base_usage),
            "error_rate": self.ai_metrics.error_rate,
            "average_response_time": self.ai_metrics.average_response_time,
            "observability_level": self.config.observability_level.value,
            "phoenix_url": "http://localhost:6006" if self.config.enabled else None,
            "cost_tracking_enabled": self.cost_tracking_enabled,
            "performance_alerts_count": len(self.performance_alerts)
        }

    async def get_vector_db_metrics(self) -> VectorDBMetrics:
        """Get vector database performance metrics"""
        return self.vector_db_metrics

    async def get_rag_pipeline_metrics(self) -> RAGPipelineMetrics:
        """Get RAG pipeline performance metrics"""
        return self.rag_pipeline_metrics

    async def get_agent_collaboration_graph(self) -> Dict[str, Any]:
        """Get agent collaboration graph data"""
        return {
            "graph_data": self.agent_graph_data,
            "collaboration_patterns": self.agent_metrics.collaboration_patterns,
            "total_collaborations": self.agent_metrics.collaboration_events,
            "most_active_collaborations": sorted(
                self.agent_metrics.collaboration_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    async def get_knowledge_base_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get knowledge base usage analytics"""
        cutoff_time = time.time() - (hours * 3600)
        recent_usage = [event for event in self.knowledge_base_usage if event["timestamp"] > cutoff_time]

        if not recent_usage:
            return {"message": "No recent knowledge base usage"}

        # Analyze usage patterns
        source_types = {}
        total_documents = 0
        total_relevance = 0
        avg_relevance_scores = []

        for event in recent_usage:
            source_type = event["source_type"]
            source_types[source_type] = source_types.get(source_type, 0) + 1
            total_documents += event["documents_accessed"]
            total_relevance += event["relevance_score"]
            avg_relevance_scores.append(event["relevance_score"])

        return {
            "period_hours": hours,
            "total_queries": len(recent_usage),
            "source_type_distribution": source_types,
            "total_documents_accessed": total_documents,
            "average_documents_per_query": total_documents / len(recent_usage) if recent_usage else 0,
            "average_relevance_score": sum(avg_relevance_scores) / len(avg_relevance_scores) if avg_relevance_scores else 0,
            "most_used_sources": sorted(source_types.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    async def get_cost_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost analysis and optimization insights"""
        if not self.cost_tracking_enabled:
            return {"message": "Cost tracking is disabled"}

        cutoff_time = time.time() - (hours * 3600)
        recent_events = [event for event in self.performance_buffer if event["timestamp"] > cutoff_time]

        # Calculate costs by provider
        provider_costs = {}
        model_costs = {}

        # Simple cost estimation (would need actual pricing data in production)
        for event in recent_events:
            if event["type"] == "llm_call":
                provider = event.get("provider", "unknown")
                model = event.get("model", "unknown")
                tokens = event.get("tokens", 0)

                # Estimate costs (simplified)
                if "gpt-4" in model:
                    cost_per_token = 0.00003  # ~$0.03 per 1K tokens
                elif "gpt-3.5" in model:
                    cost_per_token = 0.000002  # ~$0.002 per 1K tokens
                else:
                    cost_per_token = 0.000001  # Default estimate

                cost = (tokens / 1000) * cost_per_token

                provider_costs[provider] = provider_costs.get(provider, 0) + cost
                model_costs[model] = model_costs.get(model, 0) + cost

        return {
            "period_hours": hours,
            "total_estimated_cost": sum(provider_costs.values()),
            "costs_by_provider": provider_costs,
            "costs_by_model": dict(sorted(model_costs.items(), key=lambda x: x[1], reverse=True)[:10]),
            "total_tokens": self.ai_metrics.total_tokens_used,
            "average_cost_per_token": sum(provider_costs.values()) / max(self.ai_metrics.total_tokens_used, 1),
            "optimization_suggestions": await self._get_cost_optimization_suggestions()
        }

    async def _get_cost_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate cost optimization suggestions"""
        suggestions = []

        # Check for high-cost models
        if "gpt-4" in self.ai_metrics.model_usage_counts:
            gpt4_usage = self.ai_metrics.model_usage_counts["gpt-4"]
            total_calls = sum(self.ai_metrics.model_usage_counts.values())
            if total_calls > 0 and gpt4_usage / total_calls > 0.3:
                suggestions.append({
                    "type": "model_optimization",
                    "priority": "high",
                    "message": "High usage of GPT-4 (expensive model)",
                    "suggestion": "Consider using GPT-3.5 Turbo for less critical tasks",
                    "potential_savings": "40-60%"
                })

        # Check for high token usage
        if self.ai_metrics.total_tokens_used > 100000:
            suggestions.append({
                "type": "token_optimization",
                "priority": "medium",
                "message": f"High token usage: {self.ai_metrics.total_tokens_used:,} tokens",
                "suggestion": "Implement token optimization strategies (prompt engineering, caching)",
                "potential_savings": "20-30%"
            })

        return suggestions

    async def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors for debugging"""
        return sorted(self.error_buffer, key=lambda x: x["timestamp"], reverse=True)[:limit]

    async def get_insights(self) -> Dict[str, Any]:
        """Generate AI/ML insights from collected data"""
        insights = {
            "performance_insights": [],
            "cost_optimization": [],
            "error_patterns": [],
            "recommendations": []
        }

        # Performance insights
        if self.ai_metrics.average_response_time > 5.0:
            insights["performance_insights"].append({
                "type": "slow_response",
                "message": f"Average LLM response time is {self.ai_metrics.average_response_time:.2f}s",
                "suggestion": "Consider using faster models or caching"
            })

        # Cost optimization
        if self.ai_metrics.total_tokens_used > 100000:
            insights["cost_optimization"].append({
                "type": "high_usage",
                "message": f"High token usage: {self.ai_metrics.total_tokens_used:,} tokens",
                "suggestion": "Consider implementing token optimization strategies"
            })

        # Error patterns
        if self.ai_metrics.error_rate > 0.1:
            insights["error_patterns"].append({
                "type": "high_error_rate",
                "message": f"Error rate is {self.ai_metrics.error_rate:.1%}",
                "suggestion": "Investigate recent errors for common patterns"
            })

        return insights

    async def shutdown(self) -> None:
        """Shutdown Phoenix observability"""
        try:
            if self.tracer_provider:
                self.tracer_provider.shutdown()
            logger.info("Phoenix observability shutdown")
        except Exception as e:
            logger.error(f"Phoenix shutdown error: {e}")


# Global Phoenix service instance
phoenix_service = PhoenixObservabilityService()


async def get_phoenix_service() -> PhoenixObservabilityService:
    """Get the global Phoenix service instance"""
    if not phoenix_service.initialized:
        await phoenix_service.initialize()
    return phoenix_service


async def create_llm_trace(
    provider: str,
    model: str,
    operation: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Context manager for LLM tracing"""
    service = await get_phoenix_service()
    return service.trace_llm_call(provider, model, operation, metadata)


async def create_agent_trace(
    agent_name: str,
    operation: str,
    task_type: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Context manager for agent tracing"""
    service = await get_phoenix_service()
    return service.trace_agent_operation(agent_name, operation, task_type, metadata)


# Convenience functions for common operations
async def record_llm_call(
    provider: str,
    model: str,
    tokens: int,
    cost: float,
    duration: float,
    success: bool = True
) -> None:
    """Record an LLM call for metrics"""
    service = await get_phoenix_service()

    service.ai_metrics.llm_calls_total += 1
    if success:
        service.ai_metrics.llm_calls_successful += 1
    else:
        service.ai_metrics.llm_calls_failed += 1

    service.ai_metrics.total_tokens_used += tokens
    service.ai_metrics.total_cost_usd += cost

    # Update running average
    total_calls = service.ai_metrics.llm_calls_total
    service.ai_metrics.average_response_time = (
        (service.ai_metrics.average_response_time * (total_calls - 1) + duration) / total_calls
    )


async def record_agent_task(
    agent_name: str,
    task_type: str,
    duration: float,
    success: bool = True
) -> None:
    """Record an agent task for metrics"""
    service = await get_phoenix_service()

    if success:
        service.agent_metrics.tasks_completed += 1
    else:
        service.agent_metrics.tasks_failed += 1

    # Update running average
    completed_tasks = service.agent_metrics.tasks_completed
    if completed_tasks > 0:
        service.agent_metrics.average_task_duration = (
            (service.agent_metrics.average_task_duration * (completed_tasks - 1) + duration) / completed_tasks
        )