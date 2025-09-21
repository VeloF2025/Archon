#!/usr/bin/env python3
"""
End-to-End Kafka Integration Test
Tests the complete Kafka integration including messaging, streaming, and agent communication
"""

import asyncio
import logging
import time
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_kafka_messaging():
    """Test basic Kafka messaging functionality"""
    try:
        from python.src.server.services.kafka_integration_service import get_kafka_service
        
        logger.info("🔧 Testing Kafka Integration Service...")
        
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            logger.warning("⚠️  Kafka service not initialized - starting initialization...")
            from python.src.server.services.kafka_integration_service import initialize_kafka_service
            success = await initialize_kafka_service()
            if not success:
                logger.error("❌ Failed to initialize Kafka service")
                return False
        
        # Test system event publishing
        logger.info("📤 Testing system event publishing...")
        event_success = await kafka_service.publish_system_event(
            event_type="test_system_event",
            data={
                "test_id": "kafka_integration_test",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "End-to-end Kafka integration test"
            }
        )
        
        if event_success:
            logger.info("✅ System event published successfully")
        else:
            logger.error("❌ Failed to publish system event")
            return False
        
        # Test analytics event publishing
        logger.info("📊 Testing analytics event publishing...")
        analytics_success = await kafka_service.publish_analytics_event(
            metric_type="test_metric",
            value=42,
            tags={"test": "kafka_integration", "component": "end_to_end_test"}
        )
        
        if analytics_success:
            logger.info("✅ Analytics event published successfully")
        else:
            logger.error("❌ Failed to publish analytics event")
            return False
        
        # Test agent command publishing
        logger.info("🤖 Testing agent command publishing...")
        command_success = await kafka_service.publish_agent_command(
            agent_id="test_agent",
            command="test_command",
            params={"test_param": "test_value"}
        )
        
        if command_success:
            logger.info("✅ Agent command published successfully")
        else:
            logger.error("❌ Failed to publish agent command")
            return False
        
        # Test system alert publishing
        logger.info("🚨 Testing system alert publishing...")
        alert_success = await kafka_service.publish_system_alert(
            alert_type="test_alert",
            message="Test alert for Kafka integration",
            severity="info",
            details={"test": True, "integration": "kafka"}
        )
        
        if alert_success:
            logger.info("✅ System alert published successfully")
        else:
            logger.error("❌ Failed to publish system alert")
            return False
        
        # Test streaming data publishing
        logger.info("📡 Testing streaming data publishing...")
        stream_success = await kafka_service.publish_live_stream_data(
            stream_id="test_stream",
            data={
                "stream_data": "test_streaming_data",
                "value": 123.45,
                "timestamp": datetime.utcnow().isoformat()
            },
            stream_type="test"
        )
        
        if stream_success:
            logger.info("✅ Streaming data published successfully")
        else:
            logger.error("❌ Failed to publish streaming data")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Kafka messaging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_streaming_service():
    """Test real-time streaming service"""
    try:
        from python.src.server.services.real_time_streaming_service import get_streaming_service
        from python.src.agents.analytics.streaming_analytics import StreamType
        
        logger.info("🌊 Testing Real-Time Streaming Service...")
        
        streaming_service = get_streaming_service()
        
        if not streaming_service.is_running:
            logger.info("🔄 Initializing streaming service...")
            await streaming_service.initialize()
        
        # Test stream creation
        logger.info("📺 Testing stream creation...")
        test_stream_name = await streaming_service.create_live_stream(
            stream_name="test_integration_stream",
            kafka_topic="test.integration.stream",
            stream_type=StreamType.EVENTS,
            aggregation_window_seconds=10,
            enable_pattern_detection=True
        )
        
        if test_stream_name:
            logger.info(f"✅ Stream created successfully: {test_stream_name}")
        else:
            logger.error("❌ Failed to create stream")
            return False
        
        # Test data publishing to stream
        logger.info("📤 Testing data publishing to stream...")
        event_id = await streaming_service.publish_to_stream(
            stream_name="test_integration_stream",
            data={
                "test_data": "streaming_integration_test",
                "value": 999,
                "timestamp": datetime.utcnow().isoformat()
            },
            event_type="test_event"
        )
        
        if event_id:
            logger.info(f"✅ Data published to stream successfully: {event_id}")
        else:
            logger.error("❌ Failed to publish data to stream")
            return False
        
        # Test stream metrics
        logger.info("📊 Testing stream metrics...")
        metrics = await streaming_service.get_stream_metrics("test_integration_stream")
        
        if "error" not in metrics:
            logger.info("✅ Stream metrics retrieved successfully")
            logger.info(f"   Stream metrics: {json.dumps(metrics['real_time_metrics'], indent=2)}")
        else:
            logger.error(f"❌ Failed to get stream metrics: {metrics['error']}")
            return False
        
        # Test stream listing
        logger.info("📋 Testing stream listing...")
        streams = await streaming_service.list_active_streams()
        
        if streams:
            logger.info(f"✅ Active streams retrieved: {len(streams)} streams")
            for stream in streams:
                logger.info(f"   - {stream['stream_name']}: {stream['total_events']} events, {stream['events_per_second']:.2f} EPS")
        else:
            logger.error("❌ Failed to list active streams")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_kafka_communication():
    """Test agent Kafka communication"""
    try:
        from python.src.agents.base_agent import BaseAgent, ArchonDependencies, BaseAgentOutput
        from python.src.agents.specialized.agent_factory import AgentFactory
        from pydantic_ai import Agent
        
        logger.info("🤖 Testing Agent Kafka Communication...")
        
        # Create a simple test agent
        class TestKafkaAgent(BaseAgent[ArchonDependencies, BaseAgentOutput]):
            def _create_agent(self, **kwargs) -> Agent:
                return Agent(
                    model=self.model,
                    result_type=BaseAgentOutput,
                    system_prompt=self.get_manifest_enhanced_system_prompt()
                )
            
            def get_system_prompt(self) -> str:
                return "You are a test agent for Kafka integration testing. Respond with success messages."
        
        # Initialize test agent
        test_agent = TestKafkaAgent(
            name="TestKafkaAgent",
            enable_confidence=False
        )
        
        logger.info(f"🤖 Test agent created: {test_agent.name}")
        logger.info(f"   Kafka enabled: {test_agent.kafka_enabled}")
        
        # Test Kafka status
        kafka_status = await test_agent.get_kafka_status()
        logger.info(f"📊 Agent Kafka status: {json.dumps(kafka_status, indent=2)}")
        
        # Test event publishing
        logger.info("📤 Testing agent event publishing...")
        event_success = await test_agent.publish_agent_event(
            "test_agent_event",
            {
                "test_data": "agent_kafka_integration_test",
                "agent_test": True,
                "value": 789
            },
            priority="normal"
        )
        
        if event_success:
            logger.info("✅ Agent event published successfully")
        else:
            logger.error("❌ Failed to publish agent event")
            return False
        
        # Test analytics publishing
        logger.info("📊 Testing agent analytics publishing...")
        analytics_success = await test_agent.publish_analytics(
            "test_execution_time",
            1.23,
            {"test_tag": "kafka_integration", "agent": "TestKafkaAgent"}
        )
        
        if analytics_success:
            logger.info("✅ Agent analytics published successfully")
        else:
            logger.error("❌ Failed to publish agent analytics")
            return False
        
        # Test command sending (to another agent)
        logger.info("🔄 Testing agent command sending...")
        command_success = await test_agent.send_agent_command(
            "target_test_agent",
            "test_command",
            {"param1": "value1", "param2": 42}
        )
        
        if command_success:
            logger.info("✅ Agent command sent successfully")
        else:
            logger.error("❌ Failed to send agent command")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent Kafka communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_kafka_backend():
    """Test Kafka messaging backend"""
    try:
        from python.src.agents.messaging.distributed_messaging_system import DistributedMessagingSystem
        
        logger.info("⚙️  Testing Kafka Messaging Backend...")
        
        # Initialize messaging system
        messaging_system = DistributedMessagingSystem()
        await messaging_system.initialize()
        
        logger.info("✅ Messaging system initialized")
        
        # Get backend status
        backend_status = messaging_system.get_backend_status()
        logger.info(f"📊 Backend status: {json.dumps(backend_status, indent=2)}")
        
        # Get system metrics
        system_metrics = messaging_system.get_system_metrics()
        logger.info(f"📈 System metrics:")
        logger.info(f"   Messages sent: {system_metrics.messages_sent}")
        logger.info(f"   Messages received: {system_metrics.messages_received}")
        logger.info(f"   Messages failed: {system_metrics.messages_failed}")
        logger.info(f"   Active connections: {system_metrics.connections_active}")
        
        # Test message sending
        logger.info("📤 Testing message sending...")
        success = await messaging_system.send_message(
            "test.topic",
            {
                "test_message": "kafka_backend_integration_test",
                "timestamp": datetime.utcnow().isoformat(),
                "backend": "kafka"
            }
        )
        
        if success:
            logger.info("✅ Message sent successfully via Kafka backend")
        else:
            logger.error("❌ Failed to send message via Kafka backend")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Kafka backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_end_to_end_test():
    """Run complete end-to-end Kafka integration test"""
    logger.info("🚀 Starting End-to-End Kafka Integration Test")
    logger.info("=" * 60)
    
    test_results = {}
    start_time = time.time()
    
    # Test 1: Kafka Messaging Service
    logger.info("🔧 TEST 1: Kafka Messaging Service")
    logger.info("-" * 40)
    test_results["kafka_messaging"] = await test_kafka_messaging()
    
    await asyncio.sleep(2)  # Brief pause between tests
    
    # Test 2: Real-Time Streaming Service
    logger.info("🌊 TEST 2: Real-Time Streaming Service")
    logger.info("-" * 40)
    test_results["streaming_service"] = await test_streaming_service()
    
    await asyncio.sleep(2)  # Brief pause between tests
    
    # Test 3: Agent Kafka Communication
    logger.info("🤖 TEST 3: Agent Kafka Communication")
    logger.info("-" * 40)
    test_results["agent_communication"] = await test_agent_kafka_communication()
    
    await asyncio.sleep(2)  # Brief pause between tests
    
    # Test 4: Kafka Messaging Backend
    logger.info("⚙️  TEST 4: Kafka Messaging Backend")
    logger.info("-" * 40)
    test_results["kafka_backend"] = await test_kafka_backend()
    
    # Summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🏁 END-TO-END TEST RESULTS")
    logger.info("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed_tests += 1
    
    logger.info("-" * 40)
    logger.info(f"📊 Test Results: {passed_tests}/{total_tests} passed")
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Kafka integration is fully functional.")
        return True
    else:
        logger.error(f"💥 {total_tests - passed_tests} tests failed. Integration needs attention.")
        return False

async def main():
    """Main test runner"""
    try:
        success = await run_end_to_end_test()
        exit_code = 0 if success else 1
        
        logger.info("=" * 60)
        if success:
            logger.info("✅ KAFKA INTEGRATION FULLY OPERATIONAL")
        else:
            logger.error("❌ KAFKA INTEGRATION ISSUES DETECTED")
        logger.info("=" * 60)
        
        return exit_code
        
    except Exception as e:
        logger.error(f"💥 Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)