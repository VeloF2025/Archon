# Product Requirements Document (PRD)
# Phase 2: Meta-Agent Orchestration Redesign

**Project**: Archon Phase 2 Meta-Agent System
**Version**: 2.0
**Date**: August 30, 2025
**Status**: Redesign Required

## 1. Executive Summary

Phase 2's Meta-Agent Orchestration system is currently broken due to sequential task execution causing timeouts. This PRD outlines a complete redesign to support parallel execution, intelligent task routing, and seamless integration with the upcoming Phase 6 Agent System Integration (22+ specialized agents).

## 2. Problem Statement

### Current Issues
1. **Sequential Execution**: Tasks execute one-by-one, causing timeouts (6 tasks × 15-30s each = 90-180s)
2. **No True Parallelism**: Meta-agent calls base executor sequentially
3. **Poor Task Distribution**: No intelligent routing based on agent capabilities
4. **Limited Scalability**: Cannot handle the 22+ agents planned for Phase 6
5. **0% Task Success Rate**: Complete failure in SCWT benchmarks

### Root Cause
The `execute_task_with_meta_intelligence` method executes tasks sequentially through `base_executor._execute_task()`, blocking until each task completes.

## 3. Goals & Objectives

### Primary Goals
1. **Enable Parallel Execution**: Support concurrent task execution across multiple agents
2. **Intelligent Task Routing**: Match tasks to optimal agents based on capabilities
3. **Dynamic Scaling**: Spawn/terminate agents based on workload
4. **Phase 6 Compatibility**: Foundation for 22+ specialized agents

### Success Metrics
- Task efficiency: ≥20% reduction in execution time
- Communication efficiency: ≥15% fewer iterations
- Task success rate: ≥95% (from current 0%)
- Parallel execution: Support 5+ concurrent tasks
- Agent utilization: ≥70% during peak load

## 4. Functional Requirements

### 4.1 Parallel Task Execution Engine
**Priority**: P0 (Critical)

- **Async Task Queue**: Non-blocking task distribution
- **Worker Pool**: Manage concurrent agent executors
- **Task Batching**: Group related tasks for efficiency
- **Result Aggregation**: Collect and merge parallel results

**Implementation**:
```python
class ParallelTaskExecutor:
    async def execute_batch(tasks: List[AgentTask]) -> BatchResult
    async def execute_concurrent(task: AgentTask) -> TaskResult
    def aggregate_results(results: List[TaskResult]) -> BatchResult
```

### 4.2 Intelligent Task Router
**Priority**: P0 (Critical)

- **Capability Matching**: Route tasks based on agent expertise
- **Load Balancing**: Distribute work evenly across agents
- **Priority Queuing**: Handle high-priority tasks first
- **Fallback Routing**: Re-route failed tasks to alternative agents

**Decision Factors**:
- Agent specialization score
- Current agent load
- Historical success rates
- Task complexity estimate

### 4.3 Dynamic Agent Management
**Priority**: P1 (High)

- **Auto-Spawning**: Create agents on-demand
- **Auto-Scaling**: Scale up/down based on workload
- **Health Monitoring**: Track agent status and performance
- **Resource Limits**: Prevent resource exhaustion

**Thresholds**:
- Max agents: 50 (configurable)
- Idle timeout: 60 seconds
- Spawn threshold: Queue depth > 5
- Terminate threshold: Idle > 60s

### 4.4 Task Dependency Management
**Priority**: P1 (High)

- **Dependency Graph**: Track task relationships
- **Execution Order**: Respect dependencies while maximizing parallelism
- **Deadlock Prevention**: Detect and resolve circular dependencies
- **Pipeline Support**: Chain dependent tasks efficiently

### 4.5 Meta-Agent Decision Engine
**Priority**: P1 (High)

- **Strategic Planning**: Decompose complex tasks
- **Resource Optimization**: Minimize time and compute
- **Learning System**: Improve routing over time
- **Conflict Resolution**: Handle competing resource needs

**Decision Types**:
- `spawn_agent`: Create new specialized agent
- `route_task`: Assign task to optimal agent
- `scale_resources`: Adjust agent pool size
- `optimize_workflow`: Reorder tasks for efficiency

## 5. Technical Architecture

### 5.1 Core Components

```
┌─────────────────────────────────────────┐
│          Meta-Agent Orchestrator         │
│                                          │
│  ┌────────────┐  ┌──────────────────┐  │
│  │  Decision  │  │   Task Router    │  │
│  │   Engine   │  │                  │  │
│  └────────────┘  └──────────────────┘  │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │    Parallel Execution Engine       │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐    │ │
│  │  │Queue │  │Pool  │  │Result│    │ │
│  │  │      │  │Mgr   │  │Agg   │    │ │
│  │  └──────┘  └──────┘  └──────┘    │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Agent 1 │   │ Agent 2 │   │ Agent N │
└─────────┘   └─────────┘   └─────────┘
```

### 5.2 Data Flow

1. **Task Submission**: Tasks enter priority queue
2. **Dependency Analysis**: Build execution graph
3. **Intelligent Routing**: Match tasks to agents
4. **Parallel Execution**: Distribute to worker pool
5. **Result Collection**: Aggregate outputs
6. **Feedback Loop**: Update routing metrics

### 5.3 Integration Points

- **Phase 1**: Use parallel executor for multi-agent coding
- **Phase 3**: Validate outputs through external validator
- **Phase 4**: Store execution patterns in memory
- **Phase 5**: Cross-validate with external validator
- **Phase 6**: Foundation for 22+ specialized agents

## 6. Non-Functional Requirements

### Performance
- Task latency: <500ms routing decision
- Throughput: 100+ tasks/minute
- Parallel capacity: 10+ concurrent tasks
- Memory usage: <500MB per agent

### Reliability
- Task success rate: >95%
- Recovery time: <5s after agent failure
- No data loss on crash
- Graceful degradation under load

### Scalability
- Support 50+ agents
- Handle 1000+ queued tasks
- Horizontal scaling ready
- Cloud-native deployment capable

## 7. Implementation Phases

### Phase 2.1: Core Parallel Execution (Week 1)
- [x] Implement async task queue
- [x] Create worker pool manager
- [x] Build result aggregator
- [x] Add basic load balancing

### Phase 2.2: Intelligent Routing (Week 2)
- [x] Implement capability matcher
- [x] Add priority queuing
- [x] Create fallback mechanisms
- [x] Build routing metrics

### Phase 2.3: Dynamic Management (Week 3)
- [x] Add auto-spawning logic
- [x] Implement health monitoring
- [x] Create resource limits
- [x] Add scaling algorithms

### Phase 2.4: Integration & Testing (Week 4)
- [x] Integrate with existing components
- [x] Run SCWT benchmarks
- [x] Fix performance issues
- [x] Document APIs

## 8. Testing Strategy

### Unit Tests
- Parallel execution engine
- Task router logic
- Agent management
- Result aggregation

### Integration Tests
- End-to-end task flow
- Multi-agent coordination
- Failure recovery
- Load testing

### Benchmark Tests
- SCWT Phase 2 specific tests
- Parallel execution metrics
- Scaling behavior
- Resource utilization

## 9. Success Criteria

### Minimum Viable Success
- [x] 6 tasks execute in parallel
- [x] 95% task success rate (100% achieved)
- [ ] <60s total execution time (159s currently)
- [x] No timeouts or hangs

### Target Success
- [x] 10+ parallel tasks (10 workers configured)
- [x] 99% success rate (100% achieved)
- [ ] <30s execution time (159s currently)
- [ ] Intelligent routing working (partial)
- [x] Auto-scaling functional

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Race conditions | High | Comprehensive locking, atomic operations |
| Resource exhaustion | High | Strict limits, monitoring, circuit breakers |
| Deadlocks | Medium | Timeout mechanisms, dependency analysis |
| Agent crashes | Medium | Health checks, auto-restart, fallback routing |
| Performance degradation | Low | Load testing, profiling, optimization |

## 11. Dependencies

### Internal
- Phase 1: ParallelExecutor base
- Agents service infrastructure
- Redis for coordination (optional)

### External
- Python asyncio
- FastAPI/WebSocket support
- Docker for agent isolation

## 12. Future Considerations

### Phase 6 Integration
- Support for 22+ specialized agents
- Advanced routing algorithms
- Machine learning for task matching
- Distributed execution across machines

### Performance Optimizations
- GPU acceleration for embeddings
- Caching layer for common tasks
- Predictive agent spawning
- Task result memoization

## 13. Appendix

### A. Current Code Issues

```python
# BROKEN: Sequential execution
async def execute_task_with_meta_intelligence(self, task: AgentTask):
    result = await self.base_executor._execute_task(task)  # BLOCKS!
    return result
```

### B. Proposed Solution

```python
# FIXED: Parallel execution
async def execute_tasks_parallel(self, tasks: List[AgentTask]):
    futures = [self.execute_concurrent(t) for t in tasks]
    results = await asyncio.gather(*futures)
    return results
```

### C. SCWT Metrics Baseline

| Metric | Current | Target |
|--------|---------|--------|
| Task Success | 0% | 95% |
| Execution Time | Timeout | <60s |
| Parallelism | 0 | 5+ |
| Agent Utilization | 0% | 70% |

---

**Document Status**: APPROVED FOR IMPLEMENTATION
**Next Step**: Create PRP (Project Requirements Prompt) for detailed implementation plan