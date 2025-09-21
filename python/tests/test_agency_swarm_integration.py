"""
Comprehensive tests for Agency Swarm integration into Archon.

This test suite validates that the Agency Swarm enhancement works correctly
with Archon's existing systems while maintaining backward compatibility.

Phase 1 Tests: Core Communication Foundation
"""

import asyncio
import pytest
import uuid
from datetime import datetime
from typing import Dict, Any

# Import the agency components
from src.agents.orchestration.archon_agency import (
    ArchonAgency,
    AgencyConfig,
    CommunicationFlow,
    CommunicationFlowType,
    create_agency
)
from src.agents.orchestration.archon_thread_manager import (
    ArchonThreadManager,
    ThreadContext,
    ThreadMessage,
    ThreadStatus,
    InMemoryThreadStorage
)
from src.agents.orchestration.archon_send_message import (
    ArchonSendMessageTool,
    SendMessageOutput,
    MessagePriority,
    MessageStatus
)
from src.agents.base_agent import BaseAgent, ArchonDependencies


# Mock agent for testing
class MockAgent(BaseAgent):
    """Mock agent for testing agency communication."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, enable_agency=True, **kwargs)

    def _create_agent(self, **kwargs):
        from pydantic_ai import Agent
        return Agent(
            'openai:gpt-4o-mini',
            deps_type=ArchonDependencies,
            result_type=str,
            system_prompt=f"You are a test agent named {self.name}. Respond with 'Mock response from {self.name}'."
        )

    def get_system_prompt(self) -> str:
        return f"You are a test agent named {self.name}. Respond with 'Mock response from {self.name}'."


class TestAgencyCoreFunctionality:
    """Test core agency functionality."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        ceo = MockAgent("CEO")
        developer = MockAgent("Developer")
        tester = MockAgent("Tester")
        return {"ceo": ceo, "developer": developer, "tester": tester}

    @pytest.fixture
    def agency_config(self):
        """Create agency configuration for testing."""
        return AgencyConfig(
            name="Test Agency",
            shared_instructions="Test shared instructions",
            enable_persistence=True,
            enable_streaming=True,
            default_timeout=30.0,
            max_message_retries=2
        )

    @pytest.fixture
    def thread_manager(self):
        """Create thread manager for testing."""
        return ArchonThreadManager(
            storage=InMemoryThreadStorage(),
            enable_persistence=True,
            max_threads_per_agent=100,
            thread_timeout=3600,
            enable_cleanup=False  # Disable for testing
        )

    @pytest.mark.asyncio
    async def test_agency_creation(self, mock_agents, agency_config):
        """Test agency creation with basic configuration."""
        agency = ArchonAgency(
            mock_agents["ceo"],
            config=agency_config
        )

        assert agency.config.name == "Test Agency"
        assert len(agency.agents) == 1
        assert mock_agents["ceo"].name in agency.agents
        assert len(agency.entry_points) == 1
        assert agency.entry_points[0] == mock_agents["ceo"]

    @pytest.mark.asyncio
    async def test_agency_with_communication_flows(self, mock_agents, agency_config):
        """Test agency creation with communication flows."""
        agency = ArchonAgency(
            mock_agents["ceo"],
            communication_flows=[
                (mock_agents["ceo"], mock_agents["developer"]),
                (mock_agents["developer"], mock_agents["tester"])
            ],
            config=agency_config
        )

        assert len(agency.agents) == 3
        assert len(agency.communication_flows) == 2
        assert agency.communication_flows[0].sender == mock_agents["ceo"]
        assert agency.communication_flows[0].receivers[0] == mock_agents["developer"]

    @pytest.mark.asyncio
    async def test_agency_operator_syntax(self, mock_agents, agency_config):
        """Test the `>` operator syntax for creating agencies."""
        agency = mock_agents["ceo"] > mock_agents["developer"] > mock_agents["tester"]
        agency.config = agency_config  # Set config for testing

        assert len(agency.agents) == 3
        assert len(agency.communication_flows) == 2
        assert agency.communication_flows[0].flow_type == CommunicationFlowType.CHAIN
        assert agency.communication_flows[1].flow_type == CommunicationFlowType.CHAIN

    @pytest.mark.asyncio
    async def test_agency_structure_generation(self, mock_agents, agency_config):
        """Test agency structure generation for visualization."""
        agency = ArchonAgency(
            mock_agents["ceo"],
            communication_flows=[
                (mock_agents["ceo"], mock_agents["developer"]),
                (mock_agents["developer"], mock_agents["tester"])
            ],
            config=agency_config
        )

        structure = agency.get_agency_structure()

        assert "nodes" in structure
        assert "edges" in structure
        assert len(structure["nodes"]) == 3
        assert len(structure["edges"]) == 2

        # Check nodes structure
        node_names = [node["id"] for node in structure["nodes"]]
        assert "CEO" in node_names
        assert "Developer" in node_names
        assert "Tester" in node_names

        # Check edges structure
        edge_sources = [edge["source"] for edge in structure["edges"]]
        assert "CEO" in edge_sources
        assert "Developer" in edge_sources


class TestThreadManager:
    """Test thread management functionality."""

    @pytest.fixture
    def thread_manager(self):
        """Create thread manager for testing."""
        return ArchonThreadManager(
            storage=InMemoryThreadStorage(),
            enable_persistence=True,
            enable_cleanup=False
        )

    @pytest.mark.asyncio
    async def test_thread_creation(self, thread_manager):
        """Test thread creation."""
        thread_id = await thread_manager.create_thread("agent1", "agent2")

        assert thread_id is not None
        assert len(thread_id) > 0

        # Verify thread was created
        thread = await thread_manager.get_thread(thread_id)
        assert thread.thread_id == thread_id
        assert thread.sender == "agent1"
        assert thread.recipient == "agent2"
        assert thread.status == ThreadStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_thread_message_management(self, thread_manager):
        """Test adding and retrieving messages from threads."""
        thread_id = await thread_manager.create_thread("agent1", "agent2")

        # Add messages
        message1 = ThreadMessage(
            role="user",
            content="Hello world",
            sender="agent1",
            recipient="agent2",
            timestamp=datetime.utcnow().isoformat()
        )

        message2 = ThreadMessage(
            role="assistant",
            content="Hello back",
            sender="agent2",
            recipient="agent1",
            timestamp=datetime.utcnow().isoformat()
        )

        await thread_manager.update_thread(thread_id, messages=[message1, message2])

        # Retrieve thread
        thread = await thread_manager.get_thread(thread_id)
        assert len(thread.messages) == 2
        assert thread.messages[0].content == "Hello world"
        assert thread.messages[1].content == "Hello back"

    @pytest.mark.asyncio
    async def test_thread_filtering(self, thread_manager):
        """Test thread filtering capabilities."""
        # Create multiple threads
        thread_id1 = await thread_manager.create_thread("agent1", "agent2")
        thread_id2 = await thread_manager.create_thread("agent1", "agent3")
        thread_id3 = await thread_manager.create_thread("agent2", "agent3")

        # Test filtering by sender
        agent1_threads = await thread_manager.list_threads(sender="agent1")
        assert len(agent1_threads) >= 2

        # Test filtering by recipient
        agent3_threads = await thread_manager.list_threads(recipient="agent3")
        assert len(agent3_threads) >= 2

        # Test filtering by both
        specific_threads = await thread_manager.list_threads(sender="agent1", recipient="agent2")
        assert len(specific_threads) >= 1

    @pytest.mark.asyncio
    async def test_thread_statistics(self, thread_manager):
        """Test thread statistics generation."""
        # Create some threads
        await thread_manager.create_thread("agent1", "agent2")
        await thread_manager.create_thread("agent1", "agent3")

        stats = await thread_manager.get_thread_statistics()

        assert "active_threads" in stats
        assert "total_threads" in stats
        assert "status_distribution" in stats
        assert stats["active_threads"] >= 2


class TestSendMessageTool:
    """Test SendMessage tool functionality."""

    @pytest.fixture
    def mock_agency(self):
        """Create mock agency for testing."""
        ceo = MockAgent("CEO")
        developer = MockAgent("Developer")

        agency = ArchonAgency(
            ceo,
            communication_flows=[(ceo, developer)],
            config=AgencyConfig(name="Test Agency", default_timeout=30.0)
        )

        return agency

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_agency):
        """Test successful message sending."""
        tool = ArchonSendMessageTool(mock_agency)

        result = await tool.send_to_agent(
            recipient_agent="Developer",
            message="Hello from CEO",
            sender_agent="CEO"
        )

        assert result.success is True
        assert "Successfully sent message" in result.message
        assert result.message_id is not None
        assert result.thread_id is not None

    @pytest.mark.asyncio
    async def test_send_message_invalid_recipient(self, mock_agency):
        """Test message sending to invalid recipient."""
        tool = ArchonSendMessageTool(mock_agency)

        result = await tool.send_to_agent(
            recipient_agent="NonExistentAgent",
            message="Hello",
            sender_agent="CEO"
        )

        assert result.success is False
        assert "not found in agency" in result.message
        assert result.error == "AGENT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_send_message_with_thread_continuity(self, mock_agency):
        """Test message sending with thread continuity."""
        tool = ArchonSendMessageTool(mock_agency)

        # Send first message
        result1 = await tool.send_to_agent(
            recipient_agent="Developer",
            message="First message",
            sender_agent="CEO"
        )

        assert result1.success is True
        thread_id = result1.thread_id

        # Send second message in same thread
        result2 = await tool.send_to_agent(
            recipient_agent="Developer",
            message="Second message",
            sender_agent="CEO",
            thread_id=thread_id
        )

        assert result2.success is True
        assert result2.thread_id == thread_id

        # Verify conversation history
        history = await mock_agency.get_conversation_history(thread_id)
        assert len(history) >= 4  # 2 messages + 2 responses

    @pytest.mark.asyncio
    async def test_broadcast_message(self, mock_agency):
        """Test message broadcasting."""
        tool = ArchonSendMessageTool(mock_agency)

        results = await tool.broadcast_message(
            message="Broadcast message",
            sender_agent="CEO",
            recipient_agents=["Developer"]
        )

        assert "Developer" in results
        assert results["Developer"].success is True

    @pytest.mark.asyncio
    async def test_message_statistics(self, mock_agency):
        """Test message statistics generation."""
        tool = ArchonSendMessageTool(mock_agency)

        # Send some messages
        await tool.send_to_agent("Developer", "Message 1", "CEO")
        await tool.send_to_agent("Developer", "Message 2", "CEO")

        stats = tool.get_statistics()

        assert "total_messages" in stats
        assert "pending_messages" in stats
        assert "status_distribution" in stats
        assert "available_agents" in stats
        assert stats["total_messages"] >= 2


class TestBaseAgentIntegration:
    """Test BaseAgent integration with agency features."""

    @pytest.fixture
    def integrated_agents(self):
        """Create agents with agency integration."""
        ceo = MockAgent("CEO")
        developer = MockAgent("Developer")

        agency = ArchonAgency(
            ceo,
            communication_flows=[(ceo, developer)],
            config=AgencyConfig(name="Test Agency")
        )

        return {"ceo": ceo, "developer": developer, "agency": agency}

    @pytest.mark.asyncio
    async def test_agent_agency_membership(self, integrated_agents):
        """Test that agents are properly integrated with agency."""
        ceo = integrated_agents["ceo"]
        developer = integrated_agents["developer"]

        assert ceo.is_in_agency() is True
        assert developer.is_in_agency() is True
        assert ceo.agency == integrated_agents["agency"]
        assert developer.agency == integrated_agents["agency"]

    @pytest.mark.asyncio
    async def test_agent_send_message_method(self, integrated_agents):
        """Test agent send_message_to_agent method."""
        ceo = integrated_agents["ceo"]

        result = await ceo.send_message_to_agent(
            recipient_agent_name="Developer",
            message="Test message"
        )

        assert isinstance(result, SendMessageOutput)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_agent_broadcast_method(self, integrated_agents):
        """Test agent broadcast_message method."""
        ceo = integrated_agents["ceo"]

        results = await ceo.broadcast_message(
            message="Broadcast test",
            recipient_agents=["Developer"]
        )

        assert "Developer" in results
        assert results["Developer"].success is True

    @pytest.mark.asyncio
    async def test_agent_thread_management(self, integrated_agents):
        """Test agent thread management methods."""
        ceo = integrated_agents["ceo"]

        # Create conversation thread
        thread_id = await ceo.create_conversation_thread("Developer")

        assert thread_id is not None

        # Get conversation history
        history = await ceo.get_conversation_history(thread_id)
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_agent_status_methods(self, integrated_agents):
        """Test agent status and information methods."""
        ceo = integrated_agents["ceo"]

        # Get available agents
        available_agents = ceo.get_available_agents()
        assert "CEO" in available_agents
        assert "Developer" in available_agents

        # Get communication flows
        flows = ceo.get_communication_flows()
        assert len(flows) >= 1

        # Get agency status
        status = await ceo.get_agency_status()
        assert "agency_name" in status
        assert "total_agents" in status
        assert status["agency_name"] == "Test Agency"


class TestBackwardCompatibility:
    """Test that existing agents work without agency features."""

    @pytest.mark.asyncio
    async def test_agent_without_agency(self):
        """Test that agents work normally when not in an agency."""
        agent = MockAgent("StandaloneAgent", enable_agency=False)

        assert agent.is_in_agency() is False
        assert agent.get_available_agents() == []
        assert agent.get_communication_flows() == []

        # Test that agent still works normally
        deps = ArchonDependencies(
            request_id=str(uuid.uuid4()),
            user_id="test_user",
            trace_id=str(uuid.uuid4())
        )

        # This should work without agency features
        try:
            result = await agent.run("Test prompt", deps)
            assert result is not None
        except Exception as e:
            # If it fails due to API keys, that's expected in testing
            assert "API key" in str(e) or "OPENAI_API_KEY" in str(e)

    @pytest.mark.asyncio
    async def test_agency_methods_error_handling(self):
        """Test that agency methods properly handle errors when not in agency."""
        agent = MockAgent("StandaloneAgent", enable_agency=False)

        with pytest.raises(ValueError, match="not part of an agency"):
            await agent.send_message_to_agent("other", "message")

        with pytest.raises(ValueError, match="not part of an agency"):
            await agent.broadcast_message("message")

        with pytest.raises(ValueError, match="not part of an agency"):
            await agent.create_conversation_thread("other")

        with pytest.raises(ValueError, match="not part of an agency"):
            await agent.get_conversation_history("thread_id")

        with pytest.raises(ValueError, match="not part of an agency"):
            await agent.get_agency_status()


class TestPerformanceAndReliability:
    """Test performance and reliability aspects."""

    @pytest.mark.asyncio
    async def test_concurrent_message_sending(self):
        """Test concurrent message sending doesn't cause conflicts."""
        ceo = MockAgent("CEO")
        developer = MockAgent("Developer")
        tester = MockAgent("Tester")

        agency = ArchonAgency(
            ceo,
            communication_flows=[
                (ceo, developer),
                (ceo, tester)
            ],
            config=AgencyConfig(name="Concurrent Test Agency")
        )

        tool = ArchonSendMessageTool(agency)

        # Send multiple messages concurrently
        tasks = [
            tool.send_to_agent("Developer", f"Message {i}", "CEO")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (or fail gracefully due to API limits)
        successful_results = [r for r in results if isinstance(r, SendMessageOutput) and r.success]
        assert len(successful_results) > 0

    @pytest.mark.asyncio
    async def test_thread_cleanup(self):
        """Test thread cleanup functionality."""
        thread_manager = ArchonThreadManager(
            storage=InMemoryThreadStorage(),
            max_threads_per_agent=3,
            enable_cleanup=False
        )

        # Create more threads than the limit
        thread_ids = []
        for i in range(5):
            thread_id = await thread_manager.create_thread(f"agent{i}", f"agent{i+1}")
            thread_ids.append(thread_id)

        # Should have cleaned up old threads
        active_threads = len(thread_manager._active_threads)
        assert active_threads <= 3  # Max threads per agent pair


# Integration test for the complete workflow
@pytest.mark.asyncio
async def test_complete_agency_workflow():
    """Test a complete agency workflow from creation to message exchange."""
    # Create agents
    ceo = MockAgent("CEO")
    developer = MockAgent("Developer")
    tester = MockAgent("Tester")

    # Create agency with communication flows
    agency = ArchonAgency(
        ceo,
        communication_flows=[
            (ceo, developer),
            (developer, tester),
            (ceo, [developer, tester])  # CEO can broadcast to both
        ],
        config=AgencyConfig(
            name="Complete Workflow Test Agency",
            enable_persistence=True,
            enable_streaming=False  # Disable for testing
        )
    )

    # Verify agency structure
    assert len(agency.agents) == 3
    assert len(agency.communication_flows) == 3

    # Test direct message
    result1 = await agency.get_response("Hello from external", recipient_agent="CEO")
    assert result1 is not None

    # Test inter-agent communication
    send_result = await ceo.send_message_to_agent("Developer", "Please implement feature X")
    assert send_result.success is True

    # Test conversation continuity
    thread_id = await ceo.create_conversation_thread("Developer")
    await ceo.send_message_to_agent("Developer", "Follow up message", thread_id=thread_id)

    history = await agency.get_conversation_history(thread_id)
    assert len(history) >= 2

    # Test broadcasting
    broadcast_results = await ceo.broadcast_message(
        "Team meeting at 3 PM",
        recipient_agents=["Developer", "Tester"]
    )

    assert "Developer" in broadcast_results
    assert "Tester" in broadcast_results

    # Test agency status
    status = await ceo.get_agency_status()
    assert status["agency_name"] == "Complete Workflow Test Agency"
    assert status["total_agents"] == 3
    assert status["communication_flows"] == 3

    print("✅ Complete agency workflow test passed!")