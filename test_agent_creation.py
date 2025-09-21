#!/usr/bin/env python3
"""
Test script to create real agents and verify the agent management system works end-to-end.
This will test the actual functionality, not just the API plumbing.
"""

import asyncio
import json
import logging
import requests
from datetime import datetime
from uuid import uuid4, UUID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8181/api/agent-management"

def test_api_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        print(f"[{method}] {endpoint} -> {response.status_code}")
        if response.status_code != 200:
            print(f"  Error: {response.text}")
            return None
        
        result = response.json()
        print(f"  Response: {json.dumps(result, indent=2)[:200]}...")
        return result
        
    except Exception as e:
        print(f"  Exception: {e}")
        return None

def create_test_agent(name, agent_type, model_tier, project_id=None):
    """Create a test agent"""
    # Use the Archon Enhancement 2025 project ID from the real database
    if project_id is None:
        project_id = "1dee0f47-d5fd-4c8d-8c55-b36d34647280"  # Archon Enhancement 2025
    
    agent_data = {
        "name": name,
        "agent_type": agent_type,
        "model_tier": model_tier,
        "project_id": project_id,
        "capabilities": {
            "languages": ["python", "typescript"],
            "frameworks": ["fastapi", "react"],
            "specialties": ["api_development", "frontend"]
        }
    }
    
    return test_api_endpoint("/agents", method="POST", data=agent_data)

def main():
    print("🧪 Testing Agent Management System End-to-End")
    print("=" * 60)
    
    # 1. Test initial state (should be empty)
    print("\n1. Testing initial state...")
    agents = test_api_endpoint("/agents")
    initial_count = len(agents) if agents else 0
    print(f"   Initial agent count: {initial_count}")
    
    # 2. Create test agents
    print("\n2. Creating test agents...")
    
    test_agents = [
        ("Python API Specialist", "CODE_IMPLEMENTER", "SONNET"),
        ("Frontend React Expert", "UI_UX_OPTIMIZER", "HAIKU"),
        ("Database Architect", "DATABASE_ARCHITECT", "OPUS"),
        ("Security Auditor", "SECURITY_AUDITOR", "SONNET"),
        ("Performance Optimizer", "PERFORMANCE_OPTIMIZER", "HAIKU")
    ]
    
    created_agents = []
    for name, agent_type, model_tier in test_agents:
        print(f"\n   Creating: {name} ({agent_type}, {model_tier})")
        agent = create_test_agent(name, agent_type, model_tier)
        if agent:
            created_agents.append(agent)
            print(f"   ✅ Created agent: {agent.get('id', 'unknown')}")
        else:
            print(f"   ❌ Failed to create agent: {name}")
    
    # 3. Test agent retrieval
    print(f"\n3. Testing agent retrieval...")
    agents = test_api_endpoint("/agents")
    final_count = len(agents) if agents else 0
    print(f"   Final agent count: {final_count}")
    print(f"   Created agents: {len(created_agents)}")
    
    if final_count != initial_count + len(created_agents):
        print(f"   ❌ Count mismatch! Expected {initial_count + len(created_agents)}, got {final_count}")
    else:
        print(f"   ✅ Agent count matches expected")
    
    # 4. Test individual agent retrieval
    print(f"\n4. Testing individual agent retrieval...")
    if created_agents:
        first_agent = created_agents[0]
        agent_id = first_agent.get('id')
        if agent_id:
            agent = test_api_endpoint(f"/agents/{agent_id}")
            if agent:
                print(f"   ✅ Successfully retrieved agent: {agent.get('name')}")
            else:
                print(f"   ❌ Failed to retrieve agent: {agent_id}")
    
    # 5. Test analytics endpoints
    print(f"\n5. Testing analytics endpoints...")
    
    # Performance metrics
    performance = test_api_endpoint("/analytics/performance")
    if performance is not None:
        print(f"   ✅ Performance metrics: {len(performance)} entries")
    else:
        print(f"   ❌ Performance metrics failed")
    
    # Project overview
    overview = test_api_endpoint("/analytics/project-overview")
    if overview is not None:
        print(f"   ✅ Project overview: {type(overview)}")
    else:
        print(f"   ❌ Project overview failed")
    
    # Cost recommendations
    costs = test_api_endpoint("/costs/recommendations")
    if costs is not None:
        print(f"   ✅ Cost recommendations: {len(costs)} entries")
    else:
        print(f"   ❌ Cost recommendations failed")
    
    # 6. Test agent state updates
    print(f"\n6. Testing agent state updates...")
    if created_agents:
        first_agent = created_agents[0]
        agent_id = first_agent.get('id')
        if agent_id:
            state_data = {
                "state": "ACTIVE",
                "reason": "Test activation"
            }
            result = test_api_endpoint(f"/agents/{agent_id}/state", method="POST", data=state_data)
            if result:
                print(f"   ✅ State update successful")
            else:
                print(f"   ❌ State update failed")
    
    # 7. Summary
    print(f"\n" + "=" * 60)
    print(f"📊 TEST SUMMARY")
    print(f"   Total agents created: {len(created_agents)}")
    print(f"   Current agent count: {final_count}")
    
    if len(created_agents) > 0:
        print(f"   ✅ Agent creation: WORKING")
    else:
        print(f"   ❌ Agent creation: BROKEN")
    
    if final_count > 0:
        print(f"   ✅ Agent retrieval: WORKING") 
    else:
        print(f"   ❌ Agent retrieval: BROKEN")
    
    print(f"\n🎯 DGTS STATUS: {'REAL DATA VERIFIED' if len(created_agents) > 0 else 'SYSTEM STILL BROKEN'}")

if __name__ == "__main__":
    main()