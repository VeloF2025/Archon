#!/usr/bin/env python3
"""
Archon Project Manager Agent - Task Creation, Monitoring & Completion
Specialized agent for managing the Archon Self-Enhancement meta-development project.

Core Responsibilities:
1. Create and break down complex tasks from high-level requirements
2. Monitor task progress and workflow dependencies
3. Track completion status and mark tasks as done
4. Manage project timeline and resource allocation
5. Report progress and identify bottlenecks

Author: Archon Meta-Development System
Version: 1.0.0 - Self-Enhancement Protocol
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

from ..base_agent import BaseAgent, ArchonDependencies, BaseAgentOutput
from pydantic import BaseModel
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

class TaskStatus(Enum):
    """Task status values"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TaskCategory(Enum):
    """Task categorization"""
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    FEATURES = "features"
    UI_UX = "ui_ux"
    INFRASTRUCTURE = "infrastructure"
    TESTING = "testing"
    DOCUMENTATION = "documentation"

@dataclass
class Task:
    """Individual task structure"""
    id: str
    title: str
    description: str
    category: TaskCategory
    priority: TaskPriority
    status: TaskStatus
    estimated_hours: float
    dependencies: List[str]  # List of task IDs this task depends on
    assigned_agent: str
    success_criteria: List[str]
    risk_level: str  # low, medium, high
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    notes: List[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []

@dataclass
class ProjectStatus:
    """Overall project status"""
    project_id: str
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    blocked_tasks: int
    completion_percentage: float
    estimated_remaining_hours: float
    critical_path_issues: List[str]
    next_milestone: str
    risk_assessment: str

@dataclass
class TaskCreationRequest:
    """Request structure for creating tasks"""
    high_level_requirement: str
    context: str
    priority_level: TaskPriority
    category: TaskCategory
    estimated_complexity: str  # simple, moderate, complex
    dependencies: List[str] = None

class ArchonProjectManager(BaseAgent):
    """
    Specialized Project Manager Agent for Archon Self-Enhancement
    
    Manages the complete lifecycle of tasks in the meta-development project
    where Archon improves itself using its own capabilities.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        project_id: str = "422c4712-8619-4788-804c-3016cbc37478",  # Archon Self-Enhancement project
        **kwargs
    ):
        self.project_id = project_id
        self.tasks: Dict[str, Task] = {}
        self.project_context = {
            "name": "Archon Self-Enhancement",
            "type": "Meta-Development", 
            "base_system": "Archon V2 Alpha",
            "goal": "Fix integration gaps and optimize performance using Archon's own agents"
        }
        
        super().__init__(model=model, name="ArchonProjectManager", **kwargs)
        
        # Initialize with current system status from audit
        self._initialize_system_context()
        
        logger.info("Archon Project Manager initialized for project: %s", project_id)

    def _create_agent(self, **kwargs) -> Agent:
        """Create the PydanticAI agent for project management"""
        return Agent(
            model=self.model,
            result_type=BaseAgentOutput,
            system_prompt=self.get_system_prompt(),
            **kwargs
        )

    def get_system_prompt(self) -> str:
        """System prompt for the Project Manager Agent"""
        return """You are the Archon Project Manager Agent - an elite project management AI 
        specialized in orchestrating complex software development tasks for the Archon system.

        CORE MISSION: Manage the "Archon Self-Enhancement" meta-development project where 
        Archon uses its own specialized agents to fix, optimize, and enhance itself.

        KEY RESPONSIBILITIES:
        1. TASK CREATION: Break down complex requirements into actionable, specific tasks
        2. WORKFLOW MANAGEMENT: Track dependencies, critical paths, and resource allocation
        3. PROGRESS MONITORING: Continuously assess completion status and identify bottlenecks  
        4. QUALITY ASSURANCE: Ensure all tasks meet success criteria before marking complete
        5. RISK MANAGEMENT: Identify and mitigate project risks proactively

        PROJECT CONTEXT - ARCHON SYSTEM STATUS:
        - Base: 22 working agents, multi-service Docker architecture
        - Performance: 98.3% efficiency in core workflows (20s execution)
        - Issues: Integration gaps, performance bottlenecks, UI disconnections
        - Goal: "Fix what we have" strategy vs rebuild from scratch

        CURRENT PRIORITY ISSUES TO ADDRESS:
        1. CRITICAL: DeepConf 1.4s startup penalty (lazy loading needed)
        2. CRITICAL: Meta-agent 159s vs 20s execution (simplification needed)
        3. HIGH: Claude Code Task tool integration incomplete
        4. HIGH: TDD enforcement not actively monitoring
        5. MEDIUM: UI usability at 7.3% (needs major improvement)
        6. MEDIUM: Memory usage 1GB+ (optimization needed)

        TASK CREATION PRINCIPLES:
        - Create SPECIFIC, ACTIONABLE tasks with clear success criteria
        - Include estimated effort and risk assessment
        - Identify dependencies and critical path items
        - Assign appropriate specialized agents (system-architect, code-implementer, etc.)
        - Focus on high-impact, low-risk improvements first
        - Maintain existing functionality while adding enhancements

        WORKFLOW MANAGEMENT:
        - Track task status in real-time
        - Identify blocked tasks and resolve dependencies
        - Monitor critical path for project timeline
        - Coordinate multiple agents working in parallel
        - Escalate issues that require architectural decisions

        SUCCESS CRITERIA:
        - All integration gaps resolved and working in production
        - Performance optimized (sub-30s execution times consistently)
        - UI usability improved to >60%
        - Memory usage optimized to <500MB total
        - All features working together seamlessly
        - Comprehensive test coverage maintained

        Be decisive, detail-oriented, and focused on measurable outcomes. 
        Your role is to ensure Archon successfully enhances itself using its own capabilities."""

    def _initialize_system_context(self):
        """Initialize project context with current system audit findings"""
        self.system_audit = {
            "working_components": [
                "22 agent system (100% operational)",
                "Multi-agent parallel execution (98.3% efficiency)", 
                "External validation system (88% precision)",
                "Knowledge base and RAG (functional)",
                "Docker infrastructure (all services healthy)"
            ],
            "performance_issues": [
                "DeepConf initialization: 1,417ms penalty",
                "Meta-agent orchestration: 159s vs 20s target", 
                "UI usability: 7.3% (extremely poor)",
                "Memory usage: 1GB+ Docker footprint"
            ],
            "integration_gaps": [
                "DeepConf not connected to main workflow",
                "TDD enforcement not actively monitoring",
                "Claude Code Task tool bridge incomplete",
                "UI components not showing real data"
            ],
            "fix_strategy": "Optimize existing implementations rather than rebuild"
        }

    async def create_comprehensive_task_breakdown(self, project_scope: str) -> List[Task]:
        """
        Create comprehensive task breakdown for the entire Archon Self-Enhancement project
        
        Args:
            project_scope: High-level description of what needs to be accomplished
            
        Returns:
            List[Task]: Complete task breakdown with dependencies and priorities
        """
        logger.info("Creating comprehensive task breakdown for project scope: %s", project_scope)
        
        # Define the complete task structure based on audit findings
        task_definitions = [
            # CRITICAL PRIORITY - Performance Fixes
            {
                "title": "Implement DeepConf Lazy Loading",
                "description": "Remove 1,417ms startup penalty by implementing lazy initialization of DeepConf engine only when confidence scoring is actually needed",
                "category": TaskCategory.PERFORMANCE,
                "priority": TaskPriority.CRITICAL,
                "estimated_hours": 8.0,
                "dependencies": [],
                "assigned_agent": "performance-optimizer",
                "success_criteria": [
                    "Startup time reduced from >1400ms to <100ms",
                    "DeepConf initializes only when first confidence request made",
                    "All DeepConf functionality works when loaded",
                    "No regression in confidence scoring accuracy"
                ],
                "risk_level": "low"
            },
            {
                "title": "Simplify Meta-Agent Orchestration",
                "description": "Reduce meta-agent execution time from 159s to <30s by simplifying decision cycles and removing unnecessary coordination overhead",
                "category": TaskCategory.PERFORMANCE,
                "priority": TaskPriority.CRITICAL,
                "estimated_hours": 12.0,
                "dependencies": [],
                "assigned_agent": "system-architect",
                "success_criteria": [
                    "Meta-agent execution time <30 seconds consistently",
                    "Maintain parallel execution capabilities",
                    "No reduction in task success rate (keep 100%)",
                    "Maintain agent coordination quality"
                ],
                "risk_level": "medium"
            },
            
            # HIGH PRIORITY - Integration Fixes
            {
                "title": "Connect DeepConf to Main Workflow",
                "description": "Integrate DeepConf confidence scoring into the main agent execution pipeline to provide real-time confidence metrics",
                "category": TaskCategory.INTEGRATION,
                "priority": TaskPriority.HIGH,
                "estimated_hours": 16.0,
                "dependencies": ["implement-deepconf-lazy-loading"],
                "assigned_agent": "code-implementer",
                "success_criteria": [
                    "Confidence scores appear in agent execution logs",
                    "Real-time confidence updates during task execution",
                    "Confidence data available via API endpoints",
                    "Integration doesn't impact execution performance"
                ],
                "risk_level": "medium"
            },
            {
                "title": "Complete Claude Code Task Tool Integration",
                "description": "Fix the Claude Code bridge to enable seamless task tool integration with 95% agent compatibility",
                "category": TaskCategory.INTEGRATION,
                "priority": TaskPriority.HIGH,
                "estimated_hours": 20.0,
                "dependencies": [],
                "assigned_agent": "code-implementer",
                "success_criteria": [
                    "95% of 22 agents integrated with Claude Code",
                    "Task tool bridge working (currently 0% integration)",
                    "Autonomous workflows triggering correctly",
                    "File monitoring and agent spawning functional"
                ],
                "risk_level": "high"
            },
            {
                "title": "Activate TDD Enforcement Gate",
                "description": "Enable active TDD enforcement monitoring with file change detection and test-first validation",
                "category": TaskCategory.INTEGRATION,
                "priority": TaskPriority.HIGH,
                "estimated_hours": 14.0,
                "dependencies": [],
                "assigned_agent": "code-implementer",
                "success_criteria": [
                    "TDD gate actively monitoring file changes",
                    "Browserbase API integration working",
                    "Test-first enforcement blocking untested code",
                    "DGTS anti-gaming validation active"
                ],
                "risk_level": "medium"
            },
            
            # MEDIUM PRIORITY - UI and UX Improvements
            {
                "title": "Connect DeepConf UI to Real Data",
                "description": "Connect DeepConf visualization components to real confidence data streams instead of mock data",
                "category": TaskCategory.UI_UX,
                "priority": TaskPriority.MEDIUM,
                "estimated_hours": 10.0,
                "dependencies": ["connect-deepconf-to-main-workflow"],
                "assigned_agent": "ui-ux-optimizer",
                "success_criteria": [
                    "DeepConf dashboard shows real confidence metrics",
                    "Real-time confidence charts update with live data",
                    "Uncertainty bounds visualization working",
                    "Performance metrics display actual values"
                ],
                "risk_level": "low"
            },
            {
                "title": "Improve Dashboard Usability",
                "description": "Redesign main dashboard to improve usability from 7.3% to >60% with better navigation and information architecture",
                "category": TaskCategory.UI_UX,
                "priority": TaskPriority.MEDIUM,
                "estimated_hours": 18.0,
                "dependencies": ["connect-deepconf-ui-to-real-data"],
                "assigned_agent": "ui-ux-optimizer", 
                "success_criteria": [
                    "UI usability score >60% (currently 7.3%)",
                    "Improved navigation and information hierarchy",
                    "Better responsive design across devices",
                    "Reduced cognitive load for users"
                ],
                "risk_level": "low"
            },
            
            # MEDIUM PRIORITY - Infrastructure Optimization
            {
                "title": "Optimize Docker Memory Usage",
                "description": "Reduce total Docker memory footprint from 1GB+ to <500MB through container optimization and resource management",
                "category": TaskCategory.INFRASTRUCTURE,
                "priority": TaskPriority.MEDIUM,
                "estimated_hours": 12.0,
                "dependencies": ["implement-deepconf-lazy-loading"],
                "assigned_agent": "performance-optimizer",
                "success_criteria": [
                    "Total Docker memory usage <500MB",
                    "Server container <300MB (currently 872MB)",
                    "UI container <100MB (currently 182MB)",
                    "No functionality loss or performance regression"
                ],
                "risk_level": "low"
            },
            
            # TESTING AND VALIDATION
            {
                "title": "Validate All Integration Changes",
                "description": "Comprehensive testing of all integration fixes to ensure no regressions in existing functionality",
                "category": TaskCategory.TESTING,
                "priority": TaskPriority.HIGH,
                "estimated_hours": 16.0,
                "dependencies": [
                    "connect-deepconf-to-main-workflow",
                    "complete-claude-code-task-tool-integration", 
                    "activate-tdd-enforcement-gate"
                ],
                "assigned_agent": "test-coverage-validator",
                "success_criteria": [
                    ">95% test coverage maintained",
                    "All SCWT benchmarks passing",
                    "No regression in core functionality",
                    "Integration tests covering new connections"
                ],
                "risk_level": "low"
            },
            
            # DOCUMENTATION AND KNOWLEDGE TRANSFER
            {
                "title": "Document All System Changes", 
                "description": "Create comprehensive documentation for all fixes and optimizations for future Archon development",
                "category": TaskCategory.DOCUMENTATION,
                "priority": TaskPriority.MEDIUM,
                "estimated_hours": 8.0,
                "dependencies": ["validate-all-integration-changes"],
                "assigned_agent": "documentation-generator",
                "success_criteria": [
                    "Architecture decision records (ADRs) for major changes",
                    "Performance optimization guide",
                    "Integration troubleshooting documentation",
                    "Updated deployment and setup instructions"
                ],
                "risk_level": "low"
            }
        ]
        
        # Create Task objects
        tasks = []
        for task_def in task_definitions:
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                title=task_def["title"],
                description=task_def["description"],
                category=task_def["category"],
                priority=task_def["priority"],
                status=TaskStatus.PENDING,
                estimated_hours=task_def["estimated_hours"],
                dependencies=task_def["dependencies"],
                assigned_agent=task_def["assigned_agent"],
                success_criteria=task_def["success_criteria"],
                risk_level=task_def["risk_level"],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            tasks.append(task)
            self.tasks[task_id] = task
        
        logger.info("Created %d tasks for Archon Self-Enhancement project", len(tasks))
        return tasks

    async def create_project_tasks_in_archon(self, tasks: List[Task]) -> bool:
        """
        Create all project tasks in the Archon system via API
        
        Args:
            tasks: List of tasks to create
            
        Returns:
            bool: True if all tasks created successfully
        """
        import httpx
        
        success_count = 0
        
        for task in tasks:
            try:
                # Create task via Archon API
                task_data = {
                    "title": task.title,
                    "description": task.description,
                    "status": "todo",
                    "priority": task.priority.value,
                    "category": task.category.value,
                    "assigned_agent": task.assigned_agent,
                    "estimated_hours": task.estimated_hours,
                    "dependencies": task.dependencies,
                    "success_criteria": task.success_criteria,
                    "risk_level": task.risk_level
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://localhost:8181/api/projects/{self.project_id}/tasks",
                        json=task_data,
                        timeout=10.0
                    )
                    
                    if response.status_code in [200, 201]:
                        success_count += 1
                        logger.info("Created task in Archon: %s", task.title)
                    else:
                        logger.error("Failed to create task %s: %s", task.title, response.text)
                        
            except Exception as e:
                logger.error("Error creating task %s: %s", task.title, str(e))
        
        logger.info("Successfully created %d/%d tasks in Archon system", success_count, len(tasks))
        return success_count == len(tasks)

    async def get_project_status(self) -> ProjectStatus:
        """Get current project status and progress metrics"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        in_progress_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS)
        blocked_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.BLOCKED)
        
        completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate estimated remaining hours
        remaining_hours = sum(
            task.estimated_hours for task in self.tasks.values()
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
        )
        
        # Identify critical path issues
        critical_issues = [
            task.title for task in self.tasks.values()
            if task.priority == TaskPriority.CRITICAL and task.status != TaskStatus.COMPLETED
        ]
        
        return ProjectStatus(
            project_id=self.project_id,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            in_progress_tasks=in_progress_tasks,
            blocked_tasks=blocked_tasks,
            completion_percentage=completion_percentage,
            estimated_remaining_hours=remaining_hours,
            critical_path_issues=critical_issues,
            next_milestone="Complete all CRITICAL priority tasks",
            risk_assessment="MEDIUM - Integration complexity manageable with existing architecture"
        )

    async def update_task_status(self, task_id: str, new_status: TaskStatus, notes: str = None) -> bool:
        """
        Update task status and track progress
        
        Args:
            task_id: Task identifier
            new_status: New status to set
            notes: Optional notes about the status change
            
        Returns:
            bool: True if update successful
        """
        if task_id not in self.tasks:
            logger.error("Task not found: %s", task_id)
            return False
        
        task = self.tasks[task_id]
        old_status = task.status
        
        task.status = new_status
        task.updated_at = datetime.now()
        
        if new_status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now()
        
        if notes:
            task.notes.append(f"{datetime.now().isoformat()}: {notes}")
        
        logger.info("Task %s status changed: %s -> %s", task.title, old_status.value, new_status.value)
        
        # Check if this unblocks any dependent tasks
        await self._check_dependency_completion(task_id)
        
        return True

    async def _check_dependency_completion(self, completed_task_id: str):
        """Check if completing a task unblocks dependent tasks"""
        completed_task = self.tasks[completed_task_id]
        if completed_task.status != TaskStatus.COMPLETED:
            return
        
        # Find tasks that depend on this completed task
        for task in self.tasks.values():
            if completed_task_id in task.dependencies and task.status == TaskStatus.BLOCKED:
                # Check if all dependencies are now completed
                all_deps_completed = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED 
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                
                if all_deps_completed:
                    await self.update_task_status(task.id, TaskStatus.PENDING, 
                                                f"Unblocked - all dependencies completed")

    async def get_next_available_tasks(self) -> List[Task]:
        """Get tasks that can be started immediately (no blocking dependencies)"""
        available_tasks = []
        
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                if not task.dependencies:
                    available_tasks.append(task)
                else:
                    deps_completed = all(
                        dep_id in self.tasks and self.tasks[dep_id].status == TaskStatus.COMPLETED
                        for dep_id in task.dependencies
                    )
                    if deps_completed:
                        available_tasks.append(task)
        
        # Sort by priority
        priority_order = {TaskPriority.CRITICAL: 0, TaskPriority.HIGH: 1, TaskPriority.MEDIUM: 2, TaskPriority.LOW: 3}
        available_tasks.sort(key=lambda t: priority_order[t.priority])
        
        return available_tasks

    async def generate_progress_report(self) -> str:
        """Generate detailed progress report for the project"""
        status = await self.get_project_status()
        next_tasks = await self.get_next_available_tasks()
        
        report = f"""
# Archon Self-Enhancement Project Progress Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Status
- **Completion**: {status.completion_percentage:.1f}% ({status.completed_tasks}/{status.total_tasks} tasks)
- **In Progress**: {status.in_progress_tasks} tasks
- **Blocked**: {status.blocked_tasks} tasks
- **Estimated Remaining**: {status.estimated_remaining_hours:.1f} hours

## 🚨 Critical Path Issues
{chr(10).join(f"- {issue}" for issue in status.critical_path_issues) if status.critical_path_issues else "None - all critical tasks completed!"}

## 🎯 Next Available Tasks
{chr(10).join(f"- **{task.priority.value.upper()}**: {task.title} (Est: {task.estimated_hours}h, Agent: {task.assigned_agent})" for task in next_tasks[:5])}

## 📈 Progress by Category
{self._generate_category_progress()}

## 🔍 Risk Assessment
**Level**: {status.risk_assessment}

**Mitigation Strategy**: Focus on critical path items, maintain parallel execution where possible, validate each integration thoroughly before moving to next task.
"""
        return report

    def _generate_category_progress(self) -> str:
        """Generate progress breakdown by task category"""
        category_stats = {}
        
        for task in self.tasks.values():
            cat = task.category.value
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "completed": 0}
            
            category_stats[cat]["total"] += 1
            if task.status == TaskStatus.COMPLETED:
                category_stats[cat]["completed"] += 1
        
        report = ""
        for category, stats in category_stats.items():
            completion = (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report += f"- **{category.title()}**: {completion:.0f}% ({stats['completed']}/{stats['total']})\n"
        
        return report

# Convenience functions for easy access
async def create_project_manager() -> ArchonProjectManager:
    """Create and initialize the Archon Project Manager"""
    pm = ArchonProjectManager()
    logger.info("Archon Project Manager created and ready")
    return pm

async def setup_archon_self_enhancement_project() -> Tuple[ArchonProjectManager, List[Task]]:
    """Complete setup for Archon Self-Enhancement project"""
    pm = await create_project_manager()
    
    # Create comprehensive task breakdown
    tasks = await pm.create_comprehensive_task_breakdown(
        "Fix integration gaps, optimize performance, and complete feature implementations for Archon system"
    )
    
    # Create tasks in Archon system
    success = await pm.create_project_tasks_in_archon(tasks)
    
    if success:
        logger.info("Archon Self-Enhancement project setup completed successfully")
    else:
        logger.warning("Some tasks may not have been created in Archon system")
    
    return pm, tasks