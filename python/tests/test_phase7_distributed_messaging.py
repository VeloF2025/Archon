"""
Test suite for Phase 7 Distributed Messaging Systems
Validates Redis, RabbitMQ, and Kafka integrations
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from typing import Dict, Any, List
from datetime import datetime

# Import distributed messaging module
from src.agents.messaging.distributed_messaging_system import (
    DistributedMessagingSystem, MessagingBackend, MessageQueue,
    MessageTopic, MessagePriority, DeliveryGuarantee,
    ConnectionConfig, Message
)


class TestDistributedMessagingSystem:
    """Test distributed messaging system with multiple backends"""
    
    @pytest.fixture
    def messaging_system(self):
        """Create messaging system instance"""
        return DistributedMessagingSystem(
            backend=MessagingBackend.REDIS
        )
    
    @pytest.fixture
    def redis_config(self):
        """Redis connection configuration"""
        return ConnectionConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            timeout=5
        )
    
    @pytest.fixture
    def rabbitmq_config(self):
        """RabbitMQ connection configuration"""
        return ConnectionConfig(
            host="localhost",
            port=5672,
            username="guest",
            password="guest",
            virtual_host="/",
            timeout=10
        )
    
    def test_messaging_system_initialization(self, messaging_system):
        """Test messaging system initialization"""
        assert messaging_system is not None
        assert messaging_system.backend == MessagingBackend.REDIS
        assert messaging_system.connection is None
        assert messaging_system.queues == {}
    
    def test_backend_availability_detection(self, messaging_system):
        """Test backend availability detection"""
        redis_available = messaging_system._check_redis_available()
        rabbitmq_available = messaging_system._check_rabbitmq_available()
        kafka_available = messaging_system._check_kafka_available()
        
        assert isinstance(redis_available, bool)
        assert isinstance(rabbitmq_available, bool)
        assert isinstance(kafka_available, bool)
        
        print(f"Redis available: {redis_available}")
        print(f"RabbitMQ available: {rabbitmq_available}")
        print(f"Kafka available: {kafka_available}")
    
    @pytest.mark.asyncio
    async def test_redis_connection(self, messaging_system, redis_config):
        """Test Redis connection establishment"""
        if not messaging_system._check_redis_available():
            pytest.skip("Redis not available")
        
        connected = await messaging_system.connect(redis_config)
        assert connected
        assert messaging_system.connection is not None
        
        # Disconnect
        await messaging_system.disconnect()
        assert messaging_system.connection is None
    
    @pytest.mark.asyncio
    async def test_rabbitmq_connection(self, rabbitmq_config):
        """Test RabbitMQ connection establishment"""
        messaging_system = DistributedMessagingSystem(
            backend=MessagingBackend.RABBITMQ
        )
        
        if not messaging_system._check_rabbitmq_available():
            pytest.skip("RabbitMQ not available")
        
        connected = await messaging_system.connect(rabbitmq_config)
        assert connected
        assert messaging_system.connection is not None
        
        await messaging_system.disconnect()
    
    @pytest.mark.asyncio
    async def test_create_message_queue(self, messaging_system):
        """Test message queue creation"""
        queue = MessageQueue(
            name="test_queue",
            durable=True,
            max_size=1000,
            ttl=3600
        )
        
        created = await messaging_system.create_queue(queue)
        assert created
        assert "test_queue" in messaging_system.queues
        assert messaging_system.queues["test_queue"].durable
    
    @pytest.mark.asyncio
    async def test_create_message_topic(self, messaging_system):
        """Test message topic creation for pub-sub"""
        topic = MessageTopic(
            name="events",
            partitions=3,
            replication_factor=1,
            retention_ms=86400000  # 1 day
        )
        
        created = await messaging_system.create_topic(topic)
        assert created
        assert "events" in messaging_system.topics
    
    @pytest.mark.asyncio
    async def test_send_receive_message(self, messaging_system):
        """Test sending and receiving messages"""
        # Create queue
        queue = MessageQueue(name="test_messages")
        await messaging_system.create_queue(queue)
        
        # Send message
        message = Message(
            id="msg_001",
            content={"data": "test_data", "timestamp": datetime.now().isoformat()},
            priority=MessagePriority.NORMAL,
            headers={"sender": "test_agent"}
        )
        
        sent = await messaging_system.send_message("test_messages", message)
        assert sent
        
        # Receive message
        received = await messaging_system.receive_message("test_messages", timeout=5)
        assert received is not None
        assert received.id == "msg_001"
        assert received.content["data"] == "test_data"
    
    @pytest.mark.asyncio
    async def test_message_priority_ordering(self, messaging_system):
        """Test message priority queue ordering"""
        # Create priority queue
        queue = MessageQueue(
            name="priority_queue",
            enable_priority=True
        )
        await messaging_system.create_queue(queue)
        
        # Send messages with different priorities
        messages = [
            Message("low_1", {"data": "low"}, MessagePriority.LOW),
            Message("high_1", {"data": "high"}, MessagePriority.HIGH),
            Message("normal_1", {"data": "normal"}, MessagePriority.NORMAL),
            Message("critical_1", {"data": "critical"}, MessagePriority.CRITICAL),
        ]
        
        for msg in messages:
            await messaging_system.send_message("priority_queue", msg)
        
        # Receive messages - should get critical first
        first_msg = await messaging_system.receive_message("priority_queue")
        assert first_msg.priority == MessagePriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_publish_subscribe_pattern(self, messaging_system):
        """Test pub-sub messaging pattern"""
        # Create topic
        topic = MessageTopic(name="news_updates")
        await messaging_system.create_topic(topic)
        
        # Subscribe multiple consumers
        subscriptions = []
        for i in range(3):
            sub_id = await messaging_system.subscribe(
                topic="news_updates",
                subscriber_id=f"subscriber_{i}",
                callback=None  # Would be async callback in production
            )
            subscriptions.append(sub_id)
        
        assert len(subscriptions) == 3
        
        # Publish message
        message = Message(
            id="news_001",
            content={"headline": "Breaking news!", "category": "tech"}
        )
        
        published = await messaging_system.publish("news_updates", message)
        assert published
        assert published["subscribers_notified"] == 3
    
    @pytest.mark.asyncio
    async def test_message_acknowledgment(self, messaging_system):
        """Test message acknowledgment and redelivery"""
        queue = MessageQueue(
            name="ack_queue",
            require_ack=True,
            redelivery_delay=1
        )
        await messaging_system.create_queue(queue)
        
        # Send message
        message = Message("ack_001", {"important": "data"})
        await messaging_system.send_message("ack_queue", message)
        
        # Receive without acknowledging
        received = await messaging_system.receive_message(
            "ack_queue",
            auto_ack=False
        )
        assert received is not None
        
        # Message should be redelivered after timeout
        await asyncio.sleep(2)
        redelivered = await messaging_system.receive_message("ack_queue")
        assert redelivered.id == received.id
        
        # Acknowledge message
        acked = await messaging_system.acknowledge_message(
            "ack_queue",
            redelivered.id
        )
        assert acked
    
    @pytest.mark.asyncio
    async def test_batch_message_processing(self, messaging_system):
        """Test batch message sending and receiving"""
        queue = MessageQueue(name="batch_queue")
        await messaging_system.create_queue(queue)
        
        # Send batch of messages
        messages = [
            Message(f"batch_{i}", {"index": i, "data": f"data_{i}"})
            for i in range(10)
        ]
        
        sent_count = await messaging_system.send_batch(
            "batch_queue",
            messages
        )
        assert sent_count == 10
        
        # Receive batch
        received_batch = await messaging_system.receive_batch(
            "batch_queue",
            batch_size=5,
            timeout=5
        )
        assert len(received_batch) == 5
        assert all(msg.id.startswith("batch_") for msg in received_batch)
    
    @pytest.mark.asyncio
    async def test_message_filtering(self, messaging_system):
        """Test message filtering and routing"""
        # Create filtered queue
        queue = MessageQueue(
            name="filtered_queue",
            filter_expression="category = 'important'"
        )
        await messaging_system.create_queue(queue)
        
        # Send messages with different categories
        messages = [
            Message("msg_1", {"data": "A"}, headers={"category": "important"}),
            Message("msg_2", {"data": "B"}, headers={"category": "normal"}),
            Message("msg_3", {"data": "C"}, headers={"category": "important"}),
        ]
        
        for msg in messages:
            await messaging_system.send_message("filtered_queue", msg)
        
        # Receive filtered messages
        received = []
        while True:
            msg = await messaging_system.receive_message(
                "filtered_queue",
                timeout=1
            )
            if msg is None:
                break
            received.append(msg)
        
        # Should only receive "important" messages
        assert len(received) == 2
        assert all(msg.headers.get("category") == "important" for msg in received)
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, messaging_system):
        """Test dead letter queue for failed messages"""
        # Create main queue with DLQ
        main_queue = MessageQueue(
            name="main_queue",
            max_retries=2,
            dead_letter_queue="dlq"
        )
        dlq = MessageQueue(name="dlq")
        
        await messaging_system.create_queue(main_queue)
        await messaging_system.create_queue(dlq)
        
        # Send message that will fail processing
        message = Message("fail_001", {"poison": "pill"})
        await messaging_system.send_message("main_queue", message)
        
        # Simulate failed processing attempts
        for _ in range(3):
            msg = await messaging_system.receive_message("main_queue")
            if msg:
                # Simulate processing failure
                await messaging_system.reject_message(
                    "main_queue",
                    msg.id,
                    requeue=True
                )
        
        # Message should be in DLQ after max retries
        dlq_message = await messaging_system.receive_message("dlq")
        assert dlq_message is not None
        assert dlq_message.id == "fail_001"
    
    @pytest.mark.asyncio
    async def test_message_ttl_expiration(self, messaging_system):
        """Test message TTL and expiration"""
        queue = MessageQueue(
            name="ttl_queue",
            message_ttl=2  # 2 seconds
        )
        await messaging_system.create_queue(queue)
        
        # Send message with TTL
        message = Message(
            "ttl_001",
            {"ephemeral": "data"},
            ttl=1  # 1 second override
        )
        await messaging_system.send_message("ttl_queue", message)
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Message should be expired
        expired_msg = await messaging_system.receive_message(
            "ttl_queue",
            timeout=1
        )
        assert expired_msg is None
    
    @pytest.mark.asyncio
    async def test_connection_resilience(self, messaging_system):
        """Test connection resilience and auto-reconnect"""
        # Simulate connection failure
        messaging_system.connection = None
        
        # Enable auto-reconnect
        messaging_system.auto_reconnect = True
        messaging_system.reconnect_delay = 1
        messaging_system.max_reconnect_attempts = 3
        
        # Try to send message - should trigger reconnect
        message = Message("reconnect_001", {"test": "data"})
        
        with patch.object(messaging_system, '_reconnect', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = True
            
            sent = await messaging_system.send_message(
                "test_queue",
                message,
                retry_on_failure=True
            )
            
            mock_reconnect.assert_called()
    
    @pytest.mark.asyncio
    async def test_message_compression(self, messaging_system):
        """Test message compression for large payloads"""
        queue = MessageQueue(
            name="compressed_queue",
            enable_compression=True,
            compression_threshold=1024  # Compress messages > 1KB
        )
        await messaging_system.create_queue(queue)
        
        # Send large message
        large_data = "x" * 10000  # 10KB of data
        message = Message(
            "large_001",
            {"data": large_data}
        )
        
        sent = await messaging_system.send_message(
            "compressed_queue",
            message,
            compress=True
        )
        assert sent
        
        # Receive and decompress
        received = await messaging_system.receive_message(
            "compressed_queue",
            auto_decompress=True
        )
        assert received is not None
        assert received.content["data"] == large_data
    
    @pytest.mark.asyncio
    async def test_distributed_transactions(self, messaging_system):
        """Test distributed transaction support"""
        # Begin transaction
        tx_id = await messaging_system.begin_transaction()
        assert tx_id is not None
        
        try:
            # Send messages in transaction
            messages = [
                Message(f"tx_{i}", {"tx_data": i})
                for i in range(3)
            ]
            
            for msg in messages:
                await messaging_system.send_message(
                    "tx_queue",
                    msg,
                    transaction_id=tx_id
                )
            
            # Commit transaction
            committed = await messaging_system.commit_transaction(tx_id)
            assert committed
            
        except Exception as e:
            # Rollback on error
            await messaging_system.rollback_transaction(tx_id)
            raise
    
    @pytest.mark.asyncio
    async def test_message_deduplication(self, messaging_system):
        """Test message deduplication"""
        queue = MessageQueue(
            name="dedup_queue",
            enable_deduplication=True,
            dedup_window=60  # 60 second window
        )
        await messaging_system.create_queue(queue)
        
        # Send duplicate messages
        message = Message(
            "dedup_001",
            {"unique": "data"},
            dedup_id="unique_key_123"
        )
        
        # First send should succeed
        sent1 = await messaging_system.send_message("dedup_queue", message)
        assert sent1
        
        # Duplicate should be rejected
        sent2 = await messaging_system.send_message("dedup_queue", message)
        assert not sent2


@pytest.mark.integration
class TestMessagingIntegration:
    """Integration tests for messaging system with real backends"""
    
    @pytest.mark.asyncio
    async def test_redis_pubsub_integration(self):
        """Test Redis pub-sub with real connection"""
        system = DistributedMessagingSystem(backend=MessagingBackend.REDIS)
        
        if not system._check_redis_available():
            pytest.skip("Redis not available for integration test")
        
        config = ConnectionConfig(host="localhost", port=6379)
        await system.connect(config)
        
        # Create pub-sub channel
        channel = "test_channel"
        received_messages = []
        
        async def callback(msg):
            received_messages.append(msg)
        
        # Subscribe
        await system.subscribe(channel, "test_subscriber", callback)
        
        # Publish messages
        for i in range(5):
            message = Message(f"pub_{i}", {"index": i})
            await system.publish(channel, message)
        
        # Wait for messages
        await asyncio.sleep(1)
        
        assert len(received_messages) == 5
        await system.disconnect()
    
    @pytest.mark.asyncio
    async def test_rabbitmq_work_queue_integration(self):
        """Test RabbitMQ work queue pattern"""
        system = DistributedMessagingSystem(backend=MessagingBackend.RABBITMQ)
        
        if not system._check_rabbitmq_available():
            pytest.skip("RabbitMQ not available for integration test")
        
        config = ConnectionConfig(
            host="localhost",
            port=5672,
            username="guest",
            password="guest"
        )
        await system.connect(config)
        
        # Create work queue
        queue = MessageQueue(
            name="work_queue",
            durable=True,
            fair_dispatch=True
        )
        await system.create_queue(queue)
        
        # Send work items
        for i in range(10):
            work = Message(f"work_{i}", {"task": f"process_{i}"})
            await system.send_message("work_queue", work)
        
        # Simulate workers
        processed = []
        for _ in range(10):
            work_item = await system.receive_message("work_queue")
            if work_item:
                processed.append(work_item)
                await system.acknowledge_message("work_queue", work_item.id)
        
        assert len(processed) == 10
        await system.disconnect()
    
    @pytest.mark.asyncio
    async def test_fallback_to_in_memory(self):
        """Test fallback to in-memory backend when others unavailable"""
        system = DistributedMessagingSystem(backend=MessagingBackend.REDIS)
        
        # Force fallback by not connecting
        system.use_fallback = True
        await system.initialize_in_memory_backend()
        
        # Should work with in-memory backend
        queue = MessageQueue(name="memory_queue")
        await system.create_queue(queue)
        
        message = Message("mem_001", {"in_memory": "test"})
        await system.send_message("memory_queue", message)
        
        received = await system.receive_message("memory_queue")
        assert received is not None
        assert received.id == "mem_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])