#!/usr/bin/env python3
"""
Archon Self-Enhancement Project Setup
Generate comprehensive task breakdown and create tasks in the Archon system
"""

import json
import asyncio
import httpx
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from enum import Enum

class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

class TaskCategory(Enum):
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    FEATURES = "features"
    UI_UX = "ui_ux"
    INFRASTRUCTURE = "infrastructure"
    TESTING = "testing"
    DOCUMENTATION = "documentation"

@dataclass
class TaskDefinition:
    title: str
    description: str
    category: TaskCategory
    priority: TaskPriority
    estimated_hours: float
    dependencies: List[str]
    assigned_agent: str
    success_criteria: List[str]
    risk_level: str

def get_comprehensive_task_definitions():
    """Get complete task definitions for Archon Self-Enhancement project"""
    return [
        # CRITICAL PRIORITY - Performance Fixes
        TaskDefinition(
            title="Implement DeepConf Lazy Loading",
            description="Remove 1,417ms startup penalty by implementing lazy initialization of DeepConf engine only when confidence scoring is actually needed",
            category=TaskCategory.PERFORMANCE,
            priority=TaskPriority.CRITICAL,
            estimated_hours=8.0,
            dependencies=[],
            assigned_agent="performance-optimizer",
            success_criteria=[
                "Startup time reduced from >1400ms to <100ms",
                "DeepConf initializes only when first confidence request made",
                "All DeepConf functionality works when loaded",
                "No regression in confidence scoring accuracy"
            ],
            risk_level="low"
        ),
        TaskDefinition(
            title="Simplify Meta-Agent Orchestration",
            description="Reduce meta-agent execution time from 159s to <30s by simplifying decision cycles and removing unnecessary coordination overhead",
            category=TaskCategory.PERFORMANCE,
            priority=TaskPriority.CRITICAL,
            estimated_hours=12.0,
            dependencies=[],
            assigned_agent="system-architect",
            success_criteria=[
                "Meta-agent execution time <30 seconds consistently",
                "Maintain parallel execution capabilities", 
                "No reduction in task success rate (keep 100%)",
                "Maintain agent coordination quality"
            ],
            risk_level="medium"
        ),
        
        # HIGH PRIORITY - Integration Fixes
        TaskDefinition(
            title="Connect DeepConf to Main Workflow",
            description="Integrate DeepConf confidence scoring into the main agent execution pipeline to provide real-time confidence metrics",
            category=TaskCategory.INTEGRATION,
            priority=TaskPriority.HIGH,
            estimated_hours=16.0,
            dependencies=["implement-deepconf-lazy-loading"],
            assigned_agent="code-implementer",
            success_criteria=[
                "Confidence scores appear in agent execution logs",
                "Real-time confidence updates during task execution", 
                "Confidence data available via API endpoints",
                "Integration doesn't impact execution performance"
            ],
            risk_level="medium"
        ),
        TaskDefinition(
            title="Complete Claude Code Task Tool Integration",
            description="Fix the Claude Code bridge to enable seamless task tool integration with 95% agent compatibility",
            category=TaskCategory.INTEGRATION,
            priority=TaskPriority.HIGH,
            estimated_hours=20.0,
            dependencies=[],
            assigned_agent="code-implementer",
            success_criteria=[
                "95% of 22 agents integrated with Claude Code",
                "Task tool bridge working (currently 0% integration)",
                "Autonomous workflows triggering correctly",
                "File monitoring and agent spawning functional"
            ],
            risk_level="high"
        ),
        TaskDefinition(
            title="Activate TDD Enforcement Gate",
            description="Enable active TDD enforcement monitoring with file change detection and test-first validation",
            category=TaskCategory.INTEGRATION,
            priority=TaskPriority.HIGH,
            estimated_hours=14.0,
            dependencies=[],
            assigned_agent="code-implementer",
            success_criteria=[
                "TDD gate actively monitoring file changes",
                "Browserbase API integration working",
                "Test-first enforcement blocking untested code",
                "DGTS anti-gaming validation active"
            ],
            risk_level="medium"
        ),
        
        # MEDIUM PRIORITY - UI and UX Improvements
        TaskDefinition(
            title="Connect DeepConf UI to Real Data",
            description="Connect DeepConf visualization components to real confidence data streams instead of mock data",
            category=TaskCategory.UI_UX,
            priority=TaskPriority.MEDIUM,
            estimated_hours=10.0,
            dependencies=["connect-deepconf-to-main-workflow"],
            assigned_agent="ui-ux-optimizer",
            success_criteria=[
                "DeepConf dashboard shows real confidence metrics",
                "Real-time confidence charts update with live data",
                "Uncertainty bounds visualization working",
                "Performance metrics display actual values"
            ],
            risk_level="low"
        ),
        TaskDefinition(
            title="Improve Dashboard Usability",
            description="Redesign main dashboard to improve usability from 7.3% to >60% with better navigation and information architecture",
            category=TaskCategory.UI_UX,
            priority=TaskPriority.MEDIUM,
            estimated_hours=18.0,
            dependencies=["connect-deepconf-ui-to-real-data"],
            assigned_agent="ui-ux-optimizer",
            success_criteria=[
                "UI usability score >60% (currently 7.3%)",
                "Improved navigation and information hierarchy",
                "Better responsive design across devices", 
                "Reduced cognitive load for users"
            ],
            risk_level="low"
        ),
        
        # MEDIUM PRIORITY - Infrastructure Optimization
        TaskDefinition(
            title="Optimize Docker Memory Usage",
            description="Reduce total Docker memory footprint from 1GB+ to <500MB through container optimization and resource management",
            category=TaskCategory.INFRASTRUCTURE,
            priority=TaskPriority.MEDIUM,
            estimated_hours=12.0,
            dependencies=["implement-deepconf-lazy-loading"],
            assigned_agent="performance-optimizer",
            success_criteria=[
                "Total Docker memory usage <500MB",
                "Server container <300MB (currently 872MB)",
                "UI container <100MB (currently 182MB)",
                "No functionality loss or performance regression"
            ],
            risk_level="low"
        ),
        
        # TESTING AND VALIDATION
        TaskDefinition(
            title="Validate All Integration Changes",
            description="Comprehensive testing of all integration fixes to ensure no regressions in existing functionality",
            category=TaskCategory.TESTING,
            priority=TaskPriority.HIGH,
            estimated_hours=16.0,
            dependencies=[
                "connect-deepconf-to-main-workflow",
                "complete-claude-code-task-tool-integration",
                "activate-tdd-enforcement-gate"
            ],
            assigned_agent="test-coverage-validator",
            success_criteria=[
                ">95% test coverage maintained",
                "All SCWT benchmarks passing",
                "No regression in core functionality",
                "Integration tests covering new connections"
            ],
            risk_level="low"
        ),
        
        # DOCUMENTATION AND KNOWLEDGE TRANSFER
        TaskDefinition(
            title="Document All System Changes",
            description="Create comprehensive documentation for all fixes and optimizations for future Archon development",
            category=TaskCategory.DOCUMENTATION,
            priority=TaskPriority.MEDIUM,
            estimated_hours=8.0,
            dependencies=["validate-all-integration-changes"],
            assigned_agent="documentation-generator",
            success_criteria=[
                "Architecture decision records (ADRs) for major changes",
                "Performance optimization guide",
                "Integration troubleshooting documentation",
                "Updated deployment and setup instructions"
            ],
            risk_level="low"
        )
    ]

async def create_tasks_in_archon(project_id: str, task_definitions: List[TaskDefinition]):
    """Create all tasks in the Archon system via API"""
    
    # Create a mapping for dependency resolution
    title_to_id = {}
    created_tasks = []
    
    print(f"📋 Creating {len(task_definitions)} tasks in Archon project {project_id}")
    
    async with httpx.AsyncClient() as client:
        for task_def in task_definitions:
            try:
                # Convert dependencies from titles to snake_case IDs
                dependency_ids = [dep.replace("-", "_").replace(" ", "_").lower() for dep in task_def.dependencies]
                
                task_data = {
                    "title": task_def.title,
                    "description": task_def.description,
                    "status": "todo", 
                    "priority": task_def.priority.value,
                    "category": task_def.category.value,
                    "assigned_agent": task_def.assigned_agent,
                    "estimated_hours": task_def.estimated_hours,
                    "dependencies": dependency_ids,
                    "success_criteria": task_def.success_criteria,
                    "risk_level": task_def.risk_level
                }
                
                # Try to create task in Archon
                response = await client.post(
                    f"http://localhost:8181/api/projects/{project_id}/tasks",
                    json=task_data,
                    timeout=10.0
                )
                
                if response.status_code in [200, 201]:
                    print(f"✅ Created: {task_def.title}")
                    created_tasks.append(task_def)
                else:
                    print(f"❌ Failed to create {task_def.title}: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print(f"❌ Error creating {task_def.title}: {str(e)}")
    
    return created_tasks

def generate_project_summary(task_definitions: List[TaskDefinition]):
    """Generate a project summary report"""
    
    total_hours = sum(task.estimated_hours for task in task_definitions)
    
    # Count by priority
    priority_counts = {}
    priority_hours = {}
    for task in task_definitions:
        priority = task.priority.value
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        priority_hours[priority] = priority_hours.get(priority, 0) + task.estimated_hours
    
    # Count by category  
    category_counts = {}
    category_hours = {}
    for task in task_definitions:
        category = task.category.value
        category_counts[category] = category_counts.get(category, 0) + 1
        category_hours[category] = category_hours.get(category, 0) + task.estimated_hours
    
    print(f"""
🎯 ARCHON SELF-ENHANCEMENT PROJECT SUMMARY
{'='*50}

📊 Overall Metrics:
   Total Tasks: {len(task_definitions)}
   Total Estimated Hours: {total_hours:.1f}
   Average Hours per Task: {total_hours/len(task_definitions):.1f}

🚨 By Priority:""")
    
    for priority in ['critical', 'high', 'medium', 'low']:
        if priority in priority_counts:
            count = priority_counts[priority]
            hours = priority_hours[priority]
            print(f"   {priority.upper()}: {count} tasks ({hours:.1f}h)")
    
    print(f"""
📋 By Category:""")
    for category, count in category_counts.items():
        hours = category_hours[category]
        print(f"   {category.replace('_', ' ').title()}: {count} tasks ({hours:.1f}h)")
    
    print(f"""
🎯 Next Steps:
   1. Start with CRITICAL priority tasks (can run in parallel)
   2. Focus on performance optimizations first  
   3. Then integration fixes
   4. Finally UI/UX and documentation
   
📈 Expected Timeline:
   With 2-3 agents working in parallel: ~2-3 weeks
   Sequential execution: ~4-6 weeks
""")

async def main():
    """Main setup function"""
    project_id = "422c4712-8619-4788-804c-3016cbc37478"  # Archon Self-Enhancement
    
    print("🚀 Setting up Archon Self-Enhancement Project")
    print("="*50)
    
    # Get task definitions
    task_definitions = get_comprehensive_task_definitions()
    
    # Generate project summary
    generate_project_summary(task_definitions)
    
    # Create tasks in Archon system
    try:
        created_tasks = await create_tasks_in_archon(project_id, task_definitions)
        
        print(f"\n✅ Successfully created {len(created_tasks)}/{len(task_definitions)} tasks")
        
        if len(created_tasks) < len(task_definitions):
            print("⚠️  Some tasks failed to create - they may need to be created manually")
        
        print(f"\n🎯 Project ready! Tasks available at:")
        print(f"   http://localhost:3737/projects/{project_id}")
        
    except Exception as e:
        print(f"\n❌ Error setting up project: {str(e)}")
        print("📋 Task definitions are ready - create tasks manually if API fails")

if __name__ == "__main__":
    asyncio.run(main())