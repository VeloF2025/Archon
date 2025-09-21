#!/usr/bin/env python3
"""
Test script for verifying the Real-time Collaboration Pub/Sub System
"""

import asyncio
import httpx
import json
from uuid import uuid4

BASE_URL = "http://localhost:8181"

async def test_collaboration_api():
    """Test the collaboration REST API endpoints"""
    async with httpx.AsyncClient() as client:
        print("Testing Collaboration API Endpoints...")
        
        # Test data
        project_id = str(uuid4())
        agent1_id = str(uuid4())
        agent2_id = str(uuid4())
        
        # Test 1: Create collaboration session
        print("\n1. Creating collaboration session...")
        session_data = {
            "project_id": project_id,
            "agents": [agent1_id, agent2_id],
            "pattern": "PEER_TO_PEER",
            "context": {"test": "data"}
        }
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/agents/collaboration/sessions",
                json=session_data
            )
            if response.status_code == 200:
                session = response.json()
                session_id = session["session_id"]
                print(f"✅ Session created: {session_id}")
            else:
                print(f"❌ Failed to create session: {response.status_code}")
                print(response.text)
                return
        except Exception as e:
            print(f"❌ Error creating session: {e}")
            return
        
        # Test 2: Get session details
        print("\n2. Getting session details...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/agents/collaboration/sessions/{session_id}"
            )
            if response.status_code == 200:
                print(f"✅ Session retrieved: {response.json()}")
            else:
                print(f"❌ Failed to get session: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting session: {e}")
        
        # Test 3: Publish collaboration event
        print("\n3. Publishing collaboration event...")
        event_data = {
            "event_type": "TASK_STARTED",
            "source_agent_id": agent1_id,
            "target_agent_id": agent2_id,
            "project_id": project_id,
            "task_id": str(uuid4()),
            "payload": {"message": "Starting task"},
            "metadata": {"priority": "high"}
        }
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/agents/collaboration/events",
                json=event_data
            )
            if response.status_code == 200:
                print(f"✅ Event published: {response.json()}")
            else:
                print(f"❌ Failed to publish event: {response.status_code}")
        except Exception as e:
            print(f"❌ Error publishing event: {e}")
        
        # Test 4: Get agent subscriptions
        print("\n4. Getting agent subscriptions...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/agents/collaboration/agents/{agent1_id}/subscriptions"
            )
            if response.status_code == 200:
                print(f"✅ Subscriptions retrieved: {response.json()}")
            else:
                print(f"❌ Failed to get subscriptions: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting subscriptions: {e}")
        
        # Test 5: End collaboration session
        print("\n5. Ending collaboration session...")
        try:
            response = await client.delete(
                f"{BASE_URL}/api/agents/collaboration/sessions/{session_id}"
            )
            if response.status_code == 200:
                print(f"✅ Session ended: {response.json()}")
            else:
                print(f"❌ Failed to end session: {response.status_code}")
        except Exception as e:
            print(f"❌ Error ending session: {e}")

async def test_socket_io():
    """Test Socket.IO connectivity (basic test without client library)"""
    print("\n\nTesting Socket.IO Connectivity...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test Socket.IO endpoint availability
            response = await client.get(f"{BASE_URL}/socket.io/")
            if response.status_code in [200, 400]:
                print("✅ Socket.IO endpoint is accessible")
            else:
                print(f"❌ Socket.IO endpoint returned: {response.status_code}")
        except Exception as e:
            print(f"❌ Error accessing Socket.IO: {e}")

async def main():
    print("=" * 60)
    print("Real-time Collaboration Pub/Sub System Test")
    print("=" * 60)
    
    await test_collaboration_api()
    await test_socket_io()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())