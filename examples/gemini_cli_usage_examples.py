#!/usr/bin/env python3
"""
Example usage of Gemini CLI integration in Archon.

This script demonstrates practical use cases for the Gemini CLI within free tier limits.
"""

import asyncio
import httpx
import json
from pathlib import Path
from typing import Dict, Any

API_BASE = "http://localhost:8181/api/gemini"


async def example_ui_mockup_to_code():
    """Convert a UI mockup sketch to React component code."""
    print("\n🎨 Example 1: UI Mockup to React Component")
    print("-" * 40)
    
    # In a real scenario, you'd have an actual image file
    # For demo, we'll use the endpoint structure
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Simulate uploading a UI mockup
        print("📤 Uploading UI mockup image...")
        print("🔄 Converting to React TypeScript component...")
        
        # Example request structure
        request_data = {
            "image_type": "ui_mockup",
            "output_language": "react",
            "additional_instructions": """
            Create a modern dashboard component with:
            - Responsive grid layout
            - Dark mode support
            - Loading states
            - Error boundaries
            - TypeScript interfaces
            - Tailwind CSS styling
            """
        }
        
        print(f"📋 Request: {json.dumps(request_data, indent=2)}")
        print("\n✨ Generated code would include:")
        print("   - Full React component with hooks")
        print("   - TypeScript interfaces for props")
        print("   - Tailwind CSS classes for styling")
        print("   - Loading and error states")
        print("   - Responsive design patterns")


async def example_pdf_spec_to_implementation():
    """Convert a PDF specification to implementation code."""
    print("\n📄 Example 2: PDF Specification to Code")
    print("-" * 40)
    
    # Example: Converting an API specification PDF to client library
    request_data = {
        "pdf_type": "api_docs",
        "output_type": "client_library",
        "additional_context": """
        Generate a TypeScript client library with:
        - Full type safety
        - Automatic retry logic
        - Request/response interceptors
        - Error handling
        - Rate limiting support
        - WebSocket support for real-time updates
        """
    }
    
    print("📤 Processing API specification PDF...")
    print(f"📋 Configuration: {json.dumps(request_data, indent=2)}")
    print("\n✨ Generated client library would include:")
    print("   - Fully typed API client class")
    print("   - All endpoint methods with parameters")
    print("   - Response type interfaces")
    print("   - Error handling and retry logic")
    print("   - Authentication helpers")
    print("   - WebSocket event handlers")


async def example_codebase_analysis():
    """Analyze entire Archon codebase for improvements."""
    print("\n🔍 Example 3: Codebase Analysis (1M Token Context)")
    print("-" * 40)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        request = {
            "path": "/mnt/c/Jarvis/AI Workspace/Archon",
            "analysis_type": "architecture",
            "specific_questions": [
                "What are the main architectural patterns used?",
                "Are there any circular dependencies?",
                "What improvements would enhance scalability?",
                "Are there security vulnerabilities?",
                "What technical debt should be prioritized?"
            ],
            "include_patterns": ["*.py", "*.ts", "*.tsx"],
            "exclude_patterns": ["node_modules/*", "__pycache__/*", "*.test.*"]
        }
        
        print("🔍 Analyzing Archon codebase architecture...")
        print(f"📋 Analysis request: {json.dumps(request, indent=2)}")
        
        # In production, this would actually call the endpoint
        try:
            response = await client.post(
                f"{API_BASE}/analyze-codebase",
                json=request
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Analysis complete!")
                print(f"   Model: {result.get('model')}")
                print(f"   Context: {result.get('context_window')}")
                print(f"   Time: {result.get('execution_time', 0):.2f}s")
            else:
                print(f"⚠️ Analysis returned status {response.status_code}")
                
        except Exception as e:
            print(f"ℹ️ Demo mode - actual analysis would process entire codebase")


async def example_intelligent_routing():
    """Demonstrate intelligent task routing to optimize costs."""
    print("\n🧠 Example 4: Intelligent Task Routing")
    print("-" * 40)
    
    tasks = [
        {
            "description": "Generate unit tests for authentication module",
            "routed_to": "Gemini CLI",
            "reason": "Code generation within daily budget",
            "cost": "Free (within 1000/day limit)"
        },
        {
            "description": "Real-time code completion with <100ms latency",
            "routed_to": "OpenAI GPT-4",
            "reason": "Requires low latency streaming",
            "cost": "Paid (performance critical)"
        },
        {
            "description": "Analyze 500KB codebase for refactoring",
            "routed_to": "Gemini CLI",
            "reason": "Large context task, perfect for 1M token window",
            "cost": "Free (within daily limit)"
        },
        {
            "description": "Convert architecture diagram to code structure",
            "routed_to": "Gemini CLI",
            "reason": "Multimodal task requiring image processing",
            "cost": "Free (multimodal allocation)"
        },
        {
            "description": "Execute function calls for IDE integration",
            "routed_to": "OpenAI GPT-4",
            "reason": "Requires function calling support",
            "cost": "Paid (feature requirement)"
        }
    ]
    
    print("📊 Task Routing Examples:\n")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. Task: {task['description']}")
        print(f"   → Routed to: {task['routed_to']}")
        print(f"   → Reason: {task['reason']}")
        print(f"   → Cost: {task['cost']}")
        print()


async def example_daily_budget_management():
    """Show how daily budget is managed across different task types."""
    print("\n💰 Example 5: Daily Budget Management (1000 requests/day)")
    print("-" * 40)
    
    allocations = {
        "🖼️ Multimodal (images/PDFs)": {"budget": 200, "used": 45, "examples": [
            "UI mockup → code",
            "Architecture diagram → implementation",
            "PDF specs → code"
        ]},
        "📚 Large Context Analysis": {"budget": 300, "used": 120, "examples": [
            "Codebase analysis",
            "Security audits",
            "Performance profiling"
        ]},
        "⚡ Code Generation": {"budget": 300, "used": 89, "examples": [
            "Test generation",
            "Boilerplate creation",
            "Refactoring"
        ]},
        "📝 Documentation": {"budget": 100, "used": 32, "examples": [
            "API docs",
            "README generation",
            "Comment generation"
        ]},
        "🔧 General Tasks": {"budget": 100, "used": 15, "examples": [
            "Code review",
            "Bug analysis",
            "Quick fixes"
        ]}
    }
    
    total_used = sum(a["used"] for a in allocations.values())
    
    print(f"📅 Daily Allocation (Total: 1000 requests)\n")
    for category, data in allocations.items():
        percentage = (data["used"] / data["budget"]) * 100
        bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
        
        print(f"{category}:")
        print(f"   Budget: {data['budget']} | Used: {data['used']} | {bar} {percentage:.0f}%")
        print(f"   Examples: {', '.join(data['examples'][:2])}")
        print()
    
    print(f"📊 Total Daily Usage: {total_used}/1000 ({(total_used/1000)*100:.1f}%)")
    print(f"💡 Remaining: {1000 - total_used} requests")
    
    # Rate limit status
    print(f"\n⏱️ Rate Limiting:")
    print(f"   Per minute: 60 requests max")
    print(f"   Per day: 1000 requests max")
    print(f"   Overflow: Queued for next cycle")


async def example_practical_workflow():
    """Show a practical development workflow using Gemini CLI."""
    print("\n🔄 Example 6: Practical Development Workflow")
    print("-" * 40)
    
    workflow = [
        {
            "step": "Morning: Architecture Review",
            "action": "Analyze codebase for architectural issues",
            "uses": "Gemini CLI (large context)",
            "requests": "~5-10"
        },
        {
            "step": "Design Phase: UI Mockup",
            "action": "Convert Figma mockup to React components",
            "uses": "Gemini CLI (multimodal)",
            "requests": "~20-30"
        },
        {
            "step": "Implementation: Code Generation",
            "action": "Generate boilerplate and tests",
            "uses": "Gemini CLI (code generation)",
            "requests": "~50-100"
        },
        {
            "step": "Real-time Coding: Completions",
            "action": "IDE autocompletions and suggestions",
            "uses": "OpenAI (low latency required)",
            "requests": "Unlimited (paid)"
        },
        {
            "step": "Documentation: Auto-generate",
            "action": "Create API docs and READMEs",
            "uses": "Gemini CLI (documentation)",
            "requests": "~20-30"
        },
        {
            "step": "Review: Security Audit",
            "action": "Scan for vulnerabilities",
            "uses": "Gemini CLI (analysis)",
            "requests": "~10-20"
        }
    ]
    
    total_gemini = 0
    print("📋 Typical Day Workflow:\n")
    for i, phase in enumerate(workflow, 1):
        print(f"{i}. {phase['step']}")
        print(f"   Action: {phase['action']}")
        print(f"   Provider: {phase['uses']}")
        print(f"   Est. Requests: {phase['requests']}")
        
        if "Gemini" in phase['uses']:
            # Parse request range
            if "~" in phase['requests']:
                avg = sum(map(int, phase['requests'].replace("~", "").split("-"))) // 2
                total_gemini += avg
        print()
    
    print(f"📊 Estimated Daily Gemini Usage: ~{total_gemini} requests")
    print(f"✅ Well within free tier limit of 1000/day")


async def show_usage_stats():
    """Display current Gemini CLI usage statistics."""
    print("\n📊 Current Usage Statistics")
    print("-" * 40)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE}/usage-stats")
            if response.status_code == 200:
                data = response.json()
                if data.get("available"):
                    stats = data.get("stats", {})
                    print(f"✅ Gemini CLI Status: Active")
                    print(f"📈 Today's Usage: {stats.get('daily_used', 0)}/1000")
                    print(f"⏱️ Current Minute: {stats.get('minute_used', 0)}/60")
                    print(f"📦 Queued Tasks: {stats.get('queue_size', 0)}")
                    print(f"💾 Cache Hits: {stats.get('cache_hits', 0)}")
                else:
                    print("⚠️ Gemini CLI not initialized")
            else:
                print(f"❌ Could not fetch usage stats (status: {response.status_code})")
        except Exception as e:
            print(f"ℹ️ Stats endpoint not available: {e}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("🚀 GEMINI CLI INTEGRATION - USAGE EXAMPLES")
    print("=" * 60)
    print("Free Tier Limits: 60/min, 1000/day")
    print("Strategy: Intelligent routing for cost optimization")
    
    await example_ui_mockup_to_code()
    await example_pdf_spec_to_implementation()
    await example_codebase_analysis()
    await example_intelligent_routing()
    await example_daily_budget_management()
    await example_practical_workflow()
    await show_usage_stats()
    
    print("\n" + "=" * 60)
    print("💡 KEY BENEFITS OF GEMINI CLI INTEGRATION:")
    print("=" * 60)
    print("✅ 1M token context for entire codebase analysis")
    print("✅ Multimodal: Images/PDFs to code conversion")
    print("✅ Free tier: 1000 requests/day at zero cost")
    print("✅ Intelligent routing: Use Gemini for bulk, OpenAI for real-time")
    print("✅ Cache optimization: Reuse responses for 24 hours")
    print("✅ Queue management: Never lose requests due to rate limits")
    print("✅ Cost savings: ~$50-100/month for typical developer")
    print()


if __name__ == "__main__":
    asyncio.run(main())