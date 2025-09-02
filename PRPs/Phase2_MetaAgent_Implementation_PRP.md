# Project Requirements Prompt (PRP)
# Phase 2: Meta-Agent Orchestration Implementation

**Project**: Archon Phase 2 Meta-Agent System Implementation
**Version**: 2.0
**Date**: August 30, 2025
**PRD Reference**: Phase2_MetaAgent_Redesign_PRD.md

## 1. Implementation Overview

This PRP provides detailed implementation requirements for fixing the Phase 2 Meta-Agent Orchestration system to enable parallel task execution, intelligent routing, and Phase 6 compatibility.

## 2. Core Implementation Tasks

### 2.1 Parallel Execution Engine

**File**: `python/src/agents/orchestration/parallel_execution_engine.py`

```python
class ParallelExecutionEngine:
    """
    Manages concurrent task execution across multiple agents.
    Replaces sequential execution with async parallel processing.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.worker_pool = []
        self.active_tasks = {}
        
    async def submit_task(self, task: AgentTask) -> str:
        """Add task to execution queue"""
        
    async def execute_batch(self, tasks: List[AgentTask]) -> BatchResult:
        """Execute multiple tasks in parallel"""
        
    async def _worker(self):
        """Worker coroutine for processing tasks"""
        
    async def aggregate_results(self) -> Dict:
        """Collect and merge parallel execution results"""
```

**Key Requirements**:
- Non-blocking task submission
- Concurrent execution up to max_workers limit
- Result aggregation with order preservation
- Error handling per task without affecting others
- Progress tracking via WebSocket updates

### 2.2 Intelligent Task Router

**File**: `python/src/agents/orchestration/task_router.py`

```python
class IntelligentTaskRouter:
    """
    Routes tasks to optimal agents based on capabilities,
    load, and historical performance.
    """
    
    def __init__(self):
        self.agent_capabilities = {}
        self.agent_metrics = {}
        self.routing_history = []
        
    async def route_task(self, task: AgentTask) -> str:
        """Select optimal agent for task"""
        
    def calculate_agent_score(self, agent_id: str, task: AgentTask) -> float:
        """Score agent suitability for task"""
        
    async def get_fallback_agent(self, task: AgentTask) -> str:
        """Find alternative agent if primary fails"""
```

**Routing Algorithm**:
1. Extract task requirements (language, framework, complexity)
2. Score each available agent:
   - Specialization match: 40%
   - Current load: 30%
   - Historical success: 20%
   - Response time: 10%
3. Select highest scoring agent
4. Track routing decision for learning

### 2.3 Meta-Agent Orchestrator Refactor

**File**: `python/src/agents/orchestration/meta_agent.py`

```python
class MetaAgentOrchestrator:
    """
    Refactored meta-agent with parallel execution support.
    """
    
    def __init__(self):
        self.execution_engine = ParallelExecutionEngine()
        self.task_router = IntelligentTaskRouter()
        self.agent_manager = DynamicAgentManager()
        
    async def execute_task_with_meta_intelligence(self, task: AgentTask):
        """DEPRECATED - Use execute_parallel instead"""
        
    async def execute_parallel(self, tasks: List[AgentTask]) -> List[TaskResult]:
        """New parallel execution method"""
        # 1. Analyze task dependencies
        # 2. Build execution graph
        # 3. Route tasks to agents
        # 4. Execute in parallel
        # 5. Aggregate results
        
    async def spawn_specialized_agent(self, role: str) -> str:
        """Dynamically create specialized agent"""
```

### 2.4 Dynamic Agent Manager

**File**: `python/src/agents/orchestration/agent_manager.py`

```python
class DynamicAgentManager:
    """
    Manages agent lifecycle: spawning, monitoring, termination.
    """
    
    def __init__(self, max_agents: int = 50):
        self.max_agents = max_agents
        self.agents = {}
        self.agent_pool = []
        
    async def spawn_agent(self, role: str, specialization: Dict) -> str:
        """Create new agent with specific capabilities"""
        
    async def monitor_health(self):
        """Background task for health monitoring"""
        
    async def scale_agents(self, workload_metrics: Dict):
        """Auto-scale based on workload"""
        
    async def terminate_idle_agents(self):
        """Clean up unused agents"""
```

## 3. Testing Requirements

### 3.1 Phase 2 Specific Tests

**File**: `benchmarks/phase2_meta_agent_scwt.py`

```python
class Phase2ComprehensiveSCWT:
    """
    Phase 2 specific SCWT tests for meta-agent orchestration.
    """
    
    async def test_parallel_execution(self):
        """Verify 6+ tasks execute concurrently"""
        # Create 6 diverse tasks
        # Submit to meta-agent
        # Verify parallel execution (time < sequential_time * 0.5)
        # Assert all tasks complete
        
    async def test_intelligent_routing(self):
        """Verify tasks route to optimal agents"""
        # Create specialized tasks (Python, TypeScript, SQL)
        # Submit to router
        # Verify correct agent selection
        # Assert routing metrics collected
        
    async def test_dynamic_spawning(self):
        """Verify agents spawn on demand"""
        # Submit task requiring unavailable agent
        # Verify agent spawns
        # Verify task completes
        # Verify agent terminates after idle
        
    async def test_load_balancing(self):
        """Verify even distribution across agents"""
        # Submit 20 similar tasks
        # Verify distribution across available agents
        # Assert no agent overloaded
        
    async def test_failure_recovery(self):
        """Verify system recovers from agent failures"""
        # Submit tasks
        # Simulate agent crash
        # Verify task re-routing
        # Assert eventual completion
```

**Success Criteria**:
- Parallel execution: 6+ concurrent tasks
- Execution time: <60s for 6 tasks
- Task success rate: >95%
- Routing accuracy: >80%
- Auto-scaling: Functional

### 3.2 Integration Tests

**File**: `python/tests/test_phase2_integration.py`

```python
class TestPhase2Integration:
    """
    Integration tests with other phases.
    """
    
    async def test_phase1_compatibility(self):
        """Verify Phase 1 multi-agent coding still works"""
        
    async def test_phase3_validation_integration(self):
        """Verify outputs pass through validator"""
        
    async def test_phase4_memory_integration(self):
        """Verify execution patterns stored in memory"""
        
    async def test_phase5_external_validation(self):
        """Verify external validator integration"""
        
    async def test_phase6_preparation(self):
        """Verify foundation for 22+ agents"""
```

### 3.3 Global SCWT Benchmark

**File**: `benchmarks/comprehensive_scwt_benchmark.py`

```python
class ComprehensiveSCWTBenchmark:
    """
    Global benchmark tracking all phase improvements.
    """
    
    def __init__(self):
        self.phase_tests = {
            1: Phase1SCWTBenchmark(),
            2: Phase2ComprehensiveSCWT(),
            3: Phase3ValidationSCWT(),
            4: Phase4MemorySCWT(),
            5: Phase5ExternalValidatorSCWT(),
            6: Phase6AgentSystemSCWT()  # Future
        }
        
    async def run_comprehensive_benchmark(self):
        """
        Execute all phase tests and track progression.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "global_metrics": {},
            "regression_check": {}
        }
        
        for phase_num, test in self.phase_tests.items():
            if phase_num <= 5:  # Only run implemented phases
                phase_results = await test.run()
                results["phases"][phase_num] = phase_results
                
                # Check for regressions
                if phase_num > 1:
                    regression = self.check_regression(
                        results["phases"][phase_num - 1],
                        phase_results
                    )
                    results["regression_check"][phase_num] = regression
        
        # Calculate global metrics
        results["global_metrics"] = {
            "hallucination_reduction": self.calculate_hallucination_trend(),
            "knowledge_reuse_improvement": self.calculate_reuse_trend(),
            "efficiency_gain": self.calculate_efficiency_trend(),
            "overall_progression": self.calculate_overall_progression()
        }
        
        return results
        
    def check_regression(self, previous: Dict, current: Dict) -> Dict:
        """
        Verify no regression from previous phase.
        """
        regressions = []
        
        # Check key metrics
        metrics_to_check = [
            "task_success_rate",
            "execution_time",
            "precision",
            "hallucination_rate"
        ]
        
        for metric in metrics_to_check:
            if metric in previous and metric in current:
                if self.is_regression(metric, previous[metric], current[metric]):
                    regressions.append({
                        "metric": metric,
                        "previous": previous[metric],
                        "current": current[metric],
                        "regression": True
                    })
        
        return {
            "has_regression": len(regressions) > 0,
            "regressions": regressions
        }
```

### 3.4 Performance Benchmarks

**File**: `benchmarks/phase2_performance_benchmark.py`

```python
class Phase2PerformanceBenchmark:
    """
    Performance-specific tests for Phase 2.
    """
    
    async def test_scaling_behavior(self):
        """Test with increasing number of tasks (1, 5, 10, 20, 50)"""
        
    async def test_memory_usage(self):
        """Monitor memory during extended execution"""
        
    async def test_latency_distribution(self):
        """Measure task routing and execution latencies"""
        
    async def test_throughput(self):
        """Measure tasks/minute at various loads"""
```

## 4. Implementation Steps

### Week 1: Core Parallel Execution
1. [x] Implement ParallelExecutionEngine class
2. [x] Create async task queue system
3. [x] Build worker pool manager
4. [x] Implement result aggregator
5. [x] Write unit tests for parallel execution
6. [x] Run phase2_meta_agent_scwt.py baseline

### Week 2: Intelligent Routing
1. [x] Implement IntelligentTaskRouter class
2. [x] Create capability matching algorithm
3. [x] Add load balancing logic
4. [x] Implement fallback mechanisms
5. [x] Write routing tests
6. [x] Integrate with execution engine

### Week 3: Meta-Agent Refactor
1. [x] Refactor MetaAgentOrchestrator
2. [x] Implement execute_parallel method
3. [x] Add dependency analysis
4. [x] Create DynamicAgentManager
5. [x] Implement auto-spawning
6. [x] Add health monitoring

### Week 4: Integration & Testing
1. [x] Run Phase 2 SCWT tests
2. [x] Run integration tests with other phases
3. [x] Execute global SCWT benchmark
4. [x] Fix any regressions
5. [ ] Performance optimization (execution time still >30s)
6. [x] Documentation update

## 5. Validation Criteria

### Phase 2 Specific
- [ ] 6 tasks execute in parallel successfully
- [ ] Total execution time <60s (from timeout/180s)
- [ ] Task success rate >95% (from 0%)
- [ ] Intelligent routing accuracy >80%
- [ ] Auto-scaling functional

### Global SCWT
- [ ] No regression in Phase 1 metrics
- [ ] No regression in Phase 3 validation
- [ ] No regression in Phase 4 memory
- [ ] No regression in Phase 5 external validation
- [ ] Overall progression score improved

### Performance
- [ ] Task routing latency <500ms
- [ ] Memory per agent <500MB
- [ ] Support 10+ concurrent tasks
- [ ] Throughput >100 tasks/minute

## 6. Code Examples

### Example: Parallel Task Execution
```python
# OLD (BROKEN)
for task in tasks:
    result = await self.meta_orchestrator.execute_task_with_meta_intelligence(task)
    results.append(result)

# NEW (FIXED)
results = await self.meta_orchestrator.execute_parallel(tasks)
```

### Example: Intelligent Routing
```python
# Route task to optimal agent
router = IntelligentTaskRouter()
agent_id = await router.route_task(task)

# If agent not available, spawn it
if not agent_id:
    agent_id = await agent_manager.spawn_agent(task.agent_role)
```

### Example: Result Aggregation
```python
# Aggregate parallel results
batch_result = await execution_engine.aggregate_results()
successful = [r for r in batch_result.results if r.status == "completed"]
failed = [r for r in batch_result.results if r.status == "failed"]
```

## 7. File Structure

```
python/src/agents/orchestration/
├── meta_agent.py (refactor)
├── parallel_execution_engine.py (new)
├── task_router.py (new)
├── agent_manager.py (new)
└── __init__.py

benchmarks/
├── phase2_meta_agent_scwt.py (update)
├── phase2_performance_benchmark.py (new)
└── comprehensive_scwt_benchmark.py (update)

python/tests/
├── test_phase2_integration.py (new)
├── test_parallel_execution.py (new)
└── test_task_router.py (new)
```

## 8. Dependencies

### Internal
- `ParallelExecutor` from Phase 1
- `ValidationEngine` from Phase 3
- `MemoryService` from Phase 4
- `ExternalValidator` from Phase 5

### External
- Python 3.11+ (asyncio improvements)
- Redis (optional, for coordination)
- Docker (for agent isolation)

## 9. Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| Race conditions | Use asyncio locks, atomic operations |
| Memory leaks | Implement proper cleanup, resource limits |
| Agent crashes | Health monitoring, auto-restart |
| Task starvation | Priority queuing, fair scheduling |
| Infinite loops | Task timeouts, circuit breakers |

## 10. Success Metrics Summary

### Before (Current State)
- Task success: 0%
- Execution: Sequential, timeouts
- Parallelism: None
- Routing: Random
- Scaling: None

### After (Target State)
- Task success: >95%
- Execution: Parallel, <60s
- Parallelism: 10+ concurrent
- Routing: Intelligent (>80% accuracy)
- Scaling: Auto (up to 50 agents)

---

**Document Status**: READY FOR IMPLEMENTATION
**Next Step**: Create ADR for architectural decisions