"""
Comprehensive test suite for Phase 7: Autonomous AI Agents & Orchestration
Tests all components to ensure 100% Phase 7 completion
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List
import json
import numpy as np

# Import Phase 7 components
from src.agents.autonomous.autonomous_agent_architecture import (
    AutonomousAgent, AgentCapability, AgentGoal, AgentBehavior,
    AgentType, AutonomousAgentOrchestrator
)
from src.agents.autonomous.multi_agent_coordination import (
    MultiAgentCoordinator, CoordinationStrategy, TaskAllocation,
    AllocationStrategy, CoordinationMessage
)
from src.agents.autonomous.agent_communication_protocols import (
    AgentCommunicationManager, CommunicationProtocol, MessageType,
    CommunicationMessage, MessagePriority
)
from src.agents.autonomous.agent_task_planning_execution import (
    TaskPlanningEngine, TaskExecutor, AgentTask, TaskType,
    TaskStatus, PlanningStrategy, ExecutionMode
)
from src.agents.autonomous.swarm_intelligence_framework import (
    SwarmIntelligenceFramework, SwarmAlgorithm, SwarmAgent,
    SwarmObjective, OptimizationType
)
from src.agents.autonomous.agent_learning_adaptation import (
    AgentLearningSystem, LearningStrategy, ExperienceReplay,
    KnowledgeBase, AdaptationMechanism
)


class TestAutonomousAgentArchitecture:
    """Test the core autonomous agent architecture"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        return AutonomousAgent(
            agent_id="test_agent_001",
            name="Test Agent",
            agent_type=AgentType.TASK_EXECUTOR,
            capabilities=[
                AgentCapability.PLANNING,
                AgentCapability.LEARNING,
                AgentCapability.COLLABORATION
            ]
        )
    
    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator"""
        return AutonomousAgentOrchestrator()
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent is properly initialized"""
        assert agent.agent_id == "test_agent_001"
        assert agent.name == "Test Agent"
        assert agent.agent_type == AgentType.TASK_EXECUTOR
        assert AgentCapability.PLANNING in agent.capabilities
        assert agent.status == "idle"
    
    @pytest.mark.asyncio
    async def test_agent_goal_setting(self, agent):
        """Test agent can set and pursue goals"""
        goal = AgentGoal(
            goal_id="goal_001",
            description="Complete test task",
            priority=1,
            deadline=datetime.now()
        )
        
        agent.set_goal(goal)
        assert goal in agent.goals
        assert agent.current_goal == goal
    
    @pytest.mark.asyncio
    async def test_agent_behavior_execution(self, agent):
        """Test agent can execute behaviors"""
        behavior = AgentBehavior(
            behavior_id="behavior_001",
            name="test_behavior",
            action_sequence=["observe", "plan", "execute"],
            conditions={"environment": "test"}
        )
        
        result = await agent.execute_behavior(behavior)
        assert result is not None
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_orchestrator_agent_management(self, orchestrator):
        """Test orchestrator can manage multiple agents"""
        # Register agents
        agent1 = await orchestrator.create_agent(
            name="Agent 1",
            agent_type=AgentType.COORDINATOR
        )
        agent2 = await orchestrator.create_agent(
            name="Agent 2",
            agent_type=AgentType.TASK_EXECUTOR
        )
        
        assert len(orchestrator.agents) == 2
        assert agent1.agent_id in orchestrator.agents
        assert agent2.agent_id in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_orchestrator_task_distribution(self, orchestrator):
        """Test orchestrator can distribute tasks to agents"""
        # Create agents
        agents = []
        for i in range(3):
            agent = await orchestrator.create_agent(
                name=f"Worker {i}",
                agent_type=AgentType.TASK_EXECUTOR
            )
            agents.append(agent)
        
        # Create and distribute tasks
        tasks = [
            {"task_id": f"task_{i}", "type": "process", "data": f"data_{i}"}
            for i in range(5)
        ]
        
        distributions = await orchestrator.distribute_tasks(tasks)
        assert len(distributions) == 5
        assert all(d["assigned_agent"] in orchestrator.agents for d in distributions)


class TestMultiAgentCoordination:
    """Test multi-agent coordination system"""
    
    @pytest.fixture
    def coordinator(self):
        """Create test coordinator"""
        return MultiAgentCoordinator(
            strategy=CoordinationStrategy.CONTRACT_NET
        )
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator is properly initialized"""
        assert coordinator.strategy == CoordinationStrategy.CONTRACT_NET
        assert coordinator.agents == {}
        assert coordinator.active_tasks == {}
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, coordinator):
        """Test agents can register with coordinator"""
        agent_id = await coordinator.register_agent(
            agent_id="agent_001",
            capabilities=["planning", "execution"],
            capacity=10
        )
        
        assert agent_id in coordinator.agents
        assert coordinator.agents[agent_id]["capacity"] == 10
    
    @pytest.mark.asyncio
    async def test_task_allocation_contract_net(self, coordinator):
        """Test contract net protocol task allocation"""
        # Register agents with different capabilities
        await coordinator.register_agent("agent_001", ["planning"], 5)
        await coordinator.register_agent("agent_002", ["execution"], 8)
        await coordinator.register_agent("agent_003", ["planning", "execution"], 10)
        
        # Allocate task
        allocation = TaskAllocation(
            task_id="task_001",
            task_type="complex",
            required_capabilities=["planning", "execution"],
            priority=1
        )
        
        result = await coordinator.allocate_task(allocation)
        assert result["status"] == "allocated"
        assert result["agent_id"] == "agent_003"  # Most capable agent
    
    @pytest.mark.asyncio
    async def test_coordination_message_broadcast(self, coordinator):
        """Test coordination message broadcasting"""
        # Register multiple agents
        agent_ids = []
        for i in range(3):
            agent_id = await coordinator.register_agent(
                f"agent_{i:03d}",
                ["communication"],
                5
            )
            agent_ids.append(agent_id)
        
        # Broadcast message
        message = CoordinationMessage(
            message_id="msg_001",
            sender="coordinator",
            message_type="announcement",
            content={"announcement": "System update"}
        )
        
        results = await coordinator.broadcast_message(message)
        assert len(results) == 3
        assert all(r["received"] for r in results)
    
    @pytest.mark.asyncio
    async def test_consensus_mechanism(self, coordinator):
        """Test consensus-based coordination"""
        coordinator.strategy = CoordinationStrategy.CONSENSUS
        
        # Register voting agents
        for i in range(5):
            await coordinator.register_agent(f"voter_{i}", ["voting"], 1)
        
        # Request consensus
        proposal = {
            "proposal_id": "prop_001",
            "action": "migrate_database",
            "required_consensus": 0.6
        }
        
        result = await coordinator.reach_consensus(proposal)
        assert "consensus_reached" in result
        assert "votes" in result
        assert result["participation_rate"] >= 0.6


class TestAgentCommunicationProtocols:
    """Test agent communication protocols"""
    
    @pytest.fixture
    def comm_manager(self):
        """Create communication manager"""
        return AgentCommunicationManager()
    
    @pytest.mark.asyncio
    async def test_communication_manager_init(self, comm_manager):
        """Test communication manager initialization"""
        assert comm_manager.agents == {}
        assert comm_manager.message_queue == []
        assert comm_manager.protocol_handlers is not None
    
    @pytest.mark.asyncio
    async def test_direct_message_protocol(self, comm_manager):
        """Test direct messaging between agents"""
        # Register agents
        await comm_manager.register_agent("sender_001")
        await comm_manager.register_agent("receiver_001")
        
        # Send direct message
        message = CommunicationMessage(
            message_id="msg_001",
            sender_id="sender_001",
            receiver_id="receiver_001",
            protocol=CommunicationProtocol.DIRECT_MESSAGE,
            message_type=MessageType.REQUEST,
            content={"request": "status_update"}
        )
        
        result = await comm_manager.send_message(message)
        assert result["status"] == "delivered"
        assert result["receiver"] == "receiver_001"
    
    @pytest.mark.asyncio
    async def test_broadcast_protocol(self, comm_manager):
        """Test broadcast messaging"""
        # Register multiple agents
        agent_ids = [f"agent_{i:03d}" for i in range(5)]
        for agent_id in agent_ids:
            await comm_manager.register_agent(agent_id)
        
        # Broadcast message
        message = CommunicationMessage(
            message_id="broadcast_001",
            sender_id="agent_000",
            receiver_id=None,  # Broadcast to all
            protocol=CommunicationProtocol.BROADCAST,
            message_type=MessageType.ANNOUNCEMENT,
            content={"announcement": "System maintenance"}
        )
        
        results = await comm_manager.send_message(message)
        assert len(results["recipients"]) == 4  # All except sender
        assert all(r["delivered"] for r in results["recipients"])
    
    @pytest.mark.asyncio
    async def test_publish_subscribe_protocol(self, comm_manager):
        """Test pub-sub messaging"""
        # Register agents and subscribe to topics
        await comm_manager.register_agent("publisher_001")
        await comm_manager.register_agent("subscriber_001")
        await comm_manager.register_agent("subscriber_002")
        
        # Subscribe to topic
        await comm_manager.subscribe_to_topic("subscriber_001", "updates")
        await comm_manager.subscribe_to_topic("subscriber_002", "updates")
        
        # Publish message
        message = CommunicationMessage(
            message_id="pub_001",
            sender_id="publisher_001",
            receiver_id=None,
            protocol=CommunicationProtocol.PUBLISH_SUBSCRIBE,
            message_type=MessageType.PUBLISH,
            content={"topic": "updates", "data": "New version available"}
        )
        
        result = await comm_manager.send_message(message)
        assert result["subscribers_notified"] == 2
    
    @pytest.mark.asyncio
    async def test_message_priority_handling(self, comm_manager):
        """Test message priority queue handling"""
        await comm_manager.register_agent("agent_001")
        await comm_manager.register_agent("agent_002")
        
        # Send messages with different priorities
        high_priority = CommunicationMessage(
            message_id="high_001",
            sender_id="agent_001",
            receiver_id="agent_002",
            protocol=CommunicationProtocol.DIRECT_MESSAGE,
            message_type=MessageType.URGENT,
            priority=MessagePriority.HIGH,
            content={"alert": "Critical issue"}
        )
        
        low_priority = CommunicationMessage(
            message_id="low_001",
            sender_id="agent_001",
            receiver_id="agent_002",
            protocol=CommunicationProtocol.DIRECT_MESSAGE,
            message_type=MessageType.INFO,
            priority=MessagePriority.LOW,
            content={"info": "Status update"}
        )
        
        # Add to queue
        comm_manager.message_queue.append(low_priority)
        comm_manager.message_queue.append(high_priority)
        
        # Process queue - high priority should be processed first
        processed = await comm_manager.process_message_queue()
        assert processed[0]["message_id"] == "high_001"


class TestTaskPlanningAndExecution:
    """Test task planning and execution engine"""
    
    @pytest.fixture
    def planning_engine(self):
        """Create planning engine"""
        return TaskPlanningEngine(
            strategy=PlanningStrategy.HIERARCHICAL
        )
    
    @pytest.fixture
    def executor(self):
        """Create task executor"""
        return TaskExecutor(
            mode=ExecutionMode.PARALLEL
        )
    
    @pytest.mark.asyncio
    async def test_task_planning_initialization(self, planning_engine):
        """Test planning engine initialization"""
        assert planning_engine.strategy == PlanningStrategy.HIERARCHICAL
        assert planning_engine.task_graph == {}
        assert planning_engine.execution_plan == []
    
    @pytest.mark.asyncio
    async def test_hierarchical_task_decomposition(self, planning_engine):
        """Test hierarchical task decomposition"""
        # Create complex task
        main_task = AgentTask(
            task_id="main_001",
            agent_id="agent_001",
            task_type=TaskType.COMPLEX,
            name="Build Feature",
            description="Implement new feature with multiple components"
        )
        
        # Decompose into subtasks
        subtasks = await planning_engine.decompose_task(main_task)
        assert len(subtasks) > 0
        assert all(t.parent_task_id == "main_001" for t in subtasks)
        assert all(t.task_type == TaskType.SIMPLE for t in subtasks)
    
    @pytest.mark.asyncio
    async def test_task_dependency_resolution(self, planning_engine):
        """Test task dependency resolution"""
        # Create tasks with dependencies
        tasks = [
            AgentTask("task_001", "agent_001", TaskType.SIMPLE, "Setup", "Setup environment"),
            AgentTask("task_002", "agent_001", TaskType.SIMPLE, "Build", "Build components"),
            AgentTask("task_003", "agent_001", TaskType.SIMPLE, "Test", "Run tests"),
            AgentTask("task_004", "agent_001", TaskType.SIMPLE, "Deploy", "Deploy application")
        ]
        
        # Define dependencies
        dependencies = {
            "task_002": ["task_001"],
            "task_003": ["task_002"],
            "task_004": ["task_003"]
        }
        
        # Resolve execution order
        execution_order = await planning_engine.resolve_dependencies(tasks, dependencies)
        assert execution_order == ["task_001", "task_002", "task_003", "task_004"]
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution(self, executor):
        """Test parallel task execution"""
        # Create independent tasks
        tasks = [
            AgentTask(f"task_{i:03d}", "agent_001", TaskType.SIMPLE, f"Task {i}", f"Execute task {i}")
            for i in range(5)
        ]
        
        # Execute in parallel
        start_time = datetime.now()
        results = await executor.execute_parallel(tasks)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        assert len(results) == 5
        assert all(r["status"] == "completed" for r in results)
        # Parallel execution should be faster than sequential
        assert execution_time < 5.0
    
    @pytest.mark.asyncio
    async def test_reactive_planning_adaptation(self, planning_engine):
        """Test reactive planning adaptation to changes"""
        planning_engine.strategy = PlanningStrategy.REACTIVE
        
        # Initial task
        task = AgentTask("task_001", "agent_001", TaskType.SIMPLE, "Process", "Process data")
        
        # Simulate environment change
        environment_change = {
            "event": "resource_unavailable",
            "resource": "database",
            "impact": "high"
        }
        
        # Adapt plan
        adapted_plan = await planning_engine.adapt_to_change(task, environment_change)
        assert adapted_plan["strategy"] == "alternative_resource"
        assert "fallback_tasks" in adapted_plan


class TestSwarmIntelligence:
    """Test swarm intelligence framework"""
    
    @pytest.fixture
    def swarm_framework(self):
        """Create swarm intelligence framework"""
        return SwarmIntelligenceFramework(
            algorithm=SwarmAlgorithm.PARTICLE_SWARM
        )
    
    @pytest.mark.asyncio
    async def test_swarm_initialization(self, swarm_framework):
        """Test swarm framework initialization"""
        assert swarm_framework.algorithm == SwarmAlgorithm.PARTICLE_SWARM
        assert swarm_framework.swarm_agents == []
        assert swarm_framework.global_best is None
    
    @pytest.mark.asyncio
    async def test_particle_swarm_optimization(self, swarm_framework):
        """Test PSO algorithm implementation"""
        # Define optimization objective
        objective = SwarmObjective(
            objective_id="opt_001",
            optimization_type=OptimizationType.MINIMIZE,
            fitness_function=lambda x: sum(xi**2 for xi in x),  # Sphere function
            dimensions=3,
            bounds=[(-10, 10)] * 3
        )
        
        # Initialize swarm
        swarm = await swarm_framework.initialize_swarm(
            num_agents=20,
            objective=objective
        )
        assert len(swarm) == 20
        
        # Run optimization
        result = await swarm_framework.optimize(
            objective=objective,
            max_iterations=100
        )
        
        assert result["best_fitness"] < 1.0  # Should find near-optimal solution
        assert len(result["best_position"]) == 3
    
    @pytest.mark.asyncio
    async def test_ant_colony_optimization(self, swarm_framework):
        """Test ACO algorithm for path finding"""
        swarm_framework.algorithm = SwarmAlgorithm.ANT_COLONY
        
        # Define graph for path finding
        graph = {
            "nodes": ["A", "B", "C", "D", "E"],
            "edges": [
                ("A", "B", 1.0), ("A", "C", 2.0),
                ("B", "D", 1.5), ("C", "D", 1.0),
                ("D", "E", 1.0)
            ]
        }
        
        # Find optimal path
        result = await swarm_framework.find_optimal_path(
            graph=graph,
            start="A",
            end="E",
            num_ants=10,
            iterations=50
        )
        
        assert result["path"] is not None
        assert result["path"][0] == "A"
        assert result["path"][-1] == "E"
        assert result["total_cost"] < 4.0
    
    @pytest.mark.asyncio
    async def test_bee_colony_optimization(self, swarm_framework):
        """Test BCO algorithm for resource allocation"""
        swarm_framework.algorithm = SwarmAlgorithm.BEE_COLONY
        
        # Define resource allocation problem
        resources = {
            "sources": ["S1", "S2", "S3"],
            "capacities": [100, 150, 200],
            "demands": [80, 70, 90, 60, 50]
        }
        
        # Optimize allocation
        result = await swarm_framework.optimize_resource_allocation(
            resources=resources,
            num_bees=15,
            iterations=100
        )
        
        assert result["allocation"] is not None
        assert sum(result["allocation"].values()) == sum(resources["demands"])
        assert result["efficiency"] > 0.8


class TestAgentLearningAdaptation:
    """Test agent learning and adaptation system"""
    
    @pytest.fixture
    def learning_system(self):
        """Create learning system"""
        return AgentLearningSystem(
            strategy=LearningStrategy.Q_LEARNING
        )
    
    @pytest.mark.asyncio
    async def test_learning_system_initialization(self, learning_system):
        """Test learning system initialization"""
        assert learning_system.strategy == LearningStrategy.Q_LEARNING
        assert learning_system.knowledge_base is not None
        assert learning_system.experience_replay is not None
    
    @pytest.mark.asyncio
    async def test_q_learning_implementation(self, learning_system):
        """Test Q-learning algorithm"""
        # Define state-action space
        states = ["s1", "s2", "s3", "s4"]
        actions = ["a1", "a2", "a3"]
        
        # Initialize Q-table
        q_table = await learning_system.initialize_q_table(states, actions)
        assert q_table.shape == (4, 3)
        
        # Simulate learning episodes
        for episode in range(100):
            state = "s1"
            for step in range(10):
                action = await learning_system.select_action(state, epsilon=0.1)
                next_state, reward = await learning_system.simulate_environment(state, action)
                await learning_system.update_q_value(state, action, reward, next_state)
                state = next_state
        
        # Check if Q-values have been updated
        final_q_table = learning_system.get_q_table()
        assert not np.allclose(final_q_table, q_table)  # Values should have changed
    
    @pytest.mark.asyncio
    async def test_experience_replay_buffer(self, learning_system):
        """Test experience replay mechanism"""
        replay_buffer = ExperienceReplay(capacity=1000)
        
        # Add experiences
        for i in range(100):
            experience = {
                "state": f"s_{i}",
                "action": f"a_{i%3}",
                "reward": np.random.random(),
                "next_state": f"s_{i+1}",
                "done": i == 99
            }
            await replay_buffer.add(experience)
        
        assert len(replay_buffer) == 100
        
        # Sample batch
        batch = await replay_buffer.sample(batch_size=32)
        assert len(batch) == 32
        assert all("state" in exp for exp in batch)
    
    @pytest.mark.asyncio
    async def test_knowledge_transfer_learning(self, learning_system):
        """Test knowledge transfer between agents"""
        # Create source agent with knowledge
        source_knowledge = KnowledgeBase()
        await source_knowledge.add_knowledge(
            domain="navigation",
            knowledge={"optimal_paths": {"A->B": ["A", "C", "B"]}}
        )
        
        # Create target agent
        target_system = AgentLearningSystem(strategy=LearningStrategy.TRANSFER_LEARNING)
        
        # Transfer knowledge
        result = await target_system.transfer_knowledge_from(source_knowledge, "navigation")
        assert result["transferred_items"] > 0
        assert "navigation" in target_system.knowledge_base.domains
    
    @pytest.mark.asyncio
    async def test_adaptation_mechanism(self, learning_system):
        """Test agent adaptation to environment changes"""
        adaptation = AdaptationMechanism(threshold=0.7)
        
        # Simulate performance metrics
        performance_history = [0.5, 0.6, 0.65, 0.68, 0.69, 0.68]
        
        # Check if adaptation is needed
        needs_adaptation = await adaptation.evaluate_performance(performance_history)
        assert needs_adaptation  # Performance below threshold
        
        # Adapt strategy
        new_strategy = await adaptation.adapt_strategy(
            current_strategy="exploitative",
            performance=0.68
        )
        assert new_strategy == "exploratory"  # Should switch to exploration


@pytest.mark.integration
class TestPhase7Integration:
    """Integration tests for complete Phase 7 system"""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self):
        """Test complete agent lifecycle from creation to task completion"""
        # Create orchestrator
        orchestrator = AutonomousAgentOrchestrator()
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(
            strategy=CoordinationStrategy.CONTRACT_NET
        )
        
        # Create communication manager
        comm_manager = AgentCommunicationManager()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = await orchestrator.create_agent(
                name=f"Worker_{i}",
                agent_type=AgentType.TASK_EXECUTOR
            )
            await coordinator.register_agent(
                agent.agent_id,
                ["execution", "communication"],
                capacity=10
            )
            await comm_manager.register_agent(agent.agent_id)
            agents.append(agent)
        
        # Create and distribute tasks
        tasks = []
        for i in range(5):
            task = AgentTask(
                task_id=f"task_{i:03d}",
                agent_id=None,  # To be assigned
                task_type=TaskType.SIMPLE,
                name=f"Task {i}",
                description=f"Execute task number {i}"
            )
            tasks.append(task)
        
        # Allocate tasks
        allocations = []
        for task in tasks:
            allocation = TaskAllocation(
                task_id=task.task_id,
                task_type="execution",
                required_capabilities=["execution"],
                priority=1
            )
            result = await coordinator.allocate_task(allocation)
            task.agent_id = result["agent_id"]
            allocations.append(result)
        
        assert len(allocations) == 5
        assert all(a["status"] == "allocated" for a in allocations)
        
        # Execute tasks
        executor = TaskExecutor(mode=ExecutionMode.PARALLEL)
        execution_results = await executor.execute_parallel(tasks)
        
        assert len(execution_results) == 5
        assert all(r["status"] == "completed" for r in execution_results)
    
    @pytest.mark.asyncio
    async def test_swarm_based_optimization_with_agents(self):
        """Test swarm intelligence integrated with agent system"""
        # Create swarm framework
        swarm = SwarmIntelligenceFramework(
            algorithm=SwarmAlgorithm.PARTICLE_SWARM
        )
        
        # Create orchestrator
        orchestrator = AutonomousAgentOrchestrator()
        
        # Create swarm agents
        swarm_agents = []
        for i in range(10):
            agent = await orchestrator.create_agent(
                name=f"SwarmAgent_{i}",
                agent_type=AgentType.SWARM_MEMBER
            )
            swarm_agent = SwarmAgent(
                agent_id=agent.agent_id,
                position=np.random.randn(3),
                velocity=np.random.randn(3)
            )
            swarm_agents.append(swarm_agent)
        
        # Define optimization problem
        objective = SwarmObjective(
            objective_id="swarm_opt_001",
            optimization_type=OptimizationType.MINIMIZE,
            fitness_function=lambda x: sum(xi**2 for xi in x),
            dimensions=3,
            bounds=[(-10, 10)] * 3
        )
        
        # Run swarm optimization
        swarm.swarm_agents = swarm_agents
        result = await swarm.optimize(objective, max_iterations=50)
        
        assert result["best_fitness"] < 1.0
        assert result["convergence_achieved"]
    
    @pytest.mark.asyncio
    async def test_learning_agents_with_communication(self):
        """Test learning agents that communicate to share knowledge"""
        # Create learning systems for multiple agents
        learning_agents = []
        comm_manager = AgentCommunicationManager()
        
        for i in range(3):
            agent = AgentLearningSystem(
                strategy=LearningStrategy.Q_LEARNING if i < 2 else LearningStrategy.POLICY_GRADIENT
            )
            agent.agent_id = f"learner_{i:03d}"
            await comm_manager.register_agent(agent.agent_id)
            learning_agents.append(agent)
        
        # Agents learn independently
        for agent in learning_agents:
            for episode in range(10):
                state = "start"
                for step in range(5):
                    action = await agent.select_action(state, epsilon=0.2)
                    next_state, reward = await agent.simulate_environment(state, action)
                    await agent.update_q_value(state, action, reward, next_state)
                    state = next_state
        
        # Share knowledge through communication
        knowledge_message = CommunicationMessage(
            message_id="knowledge_001",
            sender_id="learner_000",
            receiver_id=None,
            protocol=CommunicationProtocol.BROADCAST,
            message_type=MessageType.KNOWLEDGE_SHARE,
            content={"knowledge": learning_agents[0].get_q_table().tolist()}
        )
        
        result = await comm_manager.send_message(knowledge_message)
        assert result["recipients"] is not None
        assert len(result["recipients"]) == 2  # Broadcast to other agents


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])