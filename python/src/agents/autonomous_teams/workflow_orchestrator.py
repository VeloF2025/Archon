"""
Workflow Orchestration Engine for Phase 9 Autonomous Development Teams

This module manages the complete SDLC workflow for autonomous development teams,
handling task dependencies, quality gates, and seamless handoffs between specialized agents.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from dataclasses import dataclass, field
from uuid import uuid4, UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkflowPhase(str, Enum):
    """Development lifecycle phases managed by the orchestrator."""
    PLANNING = "planning"
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    IMPLEMENTATION = "implementation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityGateType(str, Enum):
    """Types of quality gates in the workflow."""
    CODE_QUALITY = "code_quality"
    TEST_COVERAGE = "test_coverage"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_CHECK = "performance_check"
    DOCUMENTATION_REVIEW = "documentation_review"
    ARCHITECTURE_REVIEW = "architecture_review"
    PEER_REVIEW = "peer_review"


@dataclass
class WorkflowTask:
    """Individual task within the development workflow."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    phase: WorkflowPhase = WorkflowPhase.PLANNING
    assigned_agent: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    priority: int = 5  # 1-10, higher = more priority
    quality_gates: List[QualityGateType] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGate:
    """Quality gate checkpoint in the workflow."""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: QualityGateType = QualityGateType.CODE_QUALITY
    name: str = ""
    description: str = ""
    criteria: Dict[str, Any] = field(default_factory=dict)
    automated: bool = True
    required_approvers: List[str] = field(default_factory=list)
    blocking: bool = True  # If True, workflow cannot proceed without passing
    timeout_minutes: int = 60
    retry_attempts: int = 3


class WorkflowExecution(BaseModel):
    """Complete workflow execution context."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str
    name: str
    description: str = ""
    tasks: List[WorkflowTask] = Field(default_factory=list)
    quality_gates: List[QualityGate] = Field(default_factory=list)
    current_phase: WorkflowPhase = WorkflowPhase.PLANNING
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration_hours: float = 0.0
    actual_duration_hours: float = 0.0
    success_rate: float = 0.0
    team_composition: Dict[str, Any] = Field(default_factory=dict)
    configuration: Dict[str, Any] = Field(default_factory=dict)


class WorkflowOrchestrator:
    """
    Main orchestrator for autonomous development team workflows.
    
    Manages the complete SDLC process with intelligent task scheduling,
    quality gates, and seamless handoffs between specialized agents.
    """
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.quality_gate_handlers: Dict[QualityGateType, Callable] = {}
        self.agent_registry: Dict[str, Any] = {}
        self.task_executors: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._running = False
        self._orchestration_task: Optional[asyncio.Task] = None
        
        # Initialize workflow templates
        self._initialize_workflow_templates()
        self._initialize_quality_gates()
    
    def _initialize_workflow_templates(self):
        """Initialize standard workflow templates for different project types."""
        
        # Standard web application development workflow
        self.workflow_templates["web_application"] = {
            "phases": [
                {
                    "phase": WorkflowPhase.REQUIREMENTS_ANALYSIS,
                    "tasks": [
                        {
                            "name": "Analyze Requirements",
                            "description": "Parse and validate project requirements",
                            "required_skills": ["business_analysis", "requirements_engineering"],
                            "estimated_hours": 4.0,
                            "quality_gates": [QualityGateType.PEER_REVIEW]
                        },
                        {
                            "name": "Create User Stories",
                            "description": "Transform requirements into user stories",
                            "required_skills": ["agile_methodology", "user_experience"],
                            "estimated_hours": 6.0,
                            "quality_gates": [QualityGateType.PEER_REVIEW]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.ARCHITECTURE_DESIGN,
                    "tasks": [
                        {
                            "name": "System Architecture Design",
                            "description": "Design high-level system architecture",
                            "required_skills": ["system_architecture", "design_patterns"],
                            "estimated_hours": 8.0,
                            "quality_gates": [QualityGateType.ARCHITECTURE_REVIEW]
                        },
                        {
                            "name": "Database Schema Design",
                            "description": "Design database schema and relationships",
                            "required_skills": ["database_design", "data_modeling"],
                            "estimated_hours": 6.0,
                            "quality_gates": [QualityGateType.ARCHITECTURE_REVIEW]
                        },
                        {
                            "name": "API Contract Definition",
                            "description": "Define API endpoints and contracts",
                            "required_skills": ["api_design", "openapi_specification"],
                            "estimated_hours": 4.0,
                            "quality_gates": [QualityGateType.ARCHITECTURE_REVIEW]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.IMPLEMENTATION,
                    "tasks": [
                        {
                            "name": "Backend Development",
                            "description": "Implement backend services and APIs",
                            "required_skills": ["backend_development", "api_development"],
                            "estimated_hours": 40.0,
                            "quality_gates": [QualityGateType.CODE_QUALITY, QualityGateType.TEST_COVERAGE]
                        },
                        {
                            "name": "Frontend Development",
                            "description": "Implement user interface components",
                            "required_skills": ["frontend_development", "ui_development"],
                            "estimated_hours": 32.0,
                            "quality_gates": [QualityGateType.CODE_QUALITY, QualityGateType.TEST_COVERAGE]
                        },
                        {
                            "name": "Database Implementation",
                            "description": "Implement database schema and migrations",
                            "required_skills": ["database_implementation", "sql"],
                            "estimated_hours": 12.0,
                            "quality_gates": [QualityGateType.CODE_QUALITY]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.TESTING,
                    "tasks": [
                        {
                            "name": "Unit Testing",
                            "description": "Create and execute unit tests",
                            "required_skills": ["unit_testing", "test_automation"],
                            "estimated_hours": 20.0,
                            "quality_gates": [QualityGateType.TEST_COVERAGE]
                        },
                        {
                            "name": "Integration Testing",
                            "description": "Create and execute integration tests",
                            "required_skills": ["integration_testing", "api_testing"],
                            "estimated_hours": 16.0,
                            "quality_gates": [QualityGateType.TEST_COVERAGE]
                        },
                        {
                            "name": "End-to-End Testing",
                            "description": "Create and execute E2E tests",
                            "required_skills": ["e2e_testing", "browser_automation"],
                            "estimated_hours": 12.0,
                            "quality_gates": [QualityGateType.TEST_COVERAGE]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.SECURITY_AUDIT,
                    "tasks": [
                        {
                            "name": "Security Vulnerability Scan",
                            "description": "Automated security vulnerability scanning",
                            "required_skills": ["security_scanning", "vulnerability_assessment"],
                            "estimated_hours": 4.0,
                            "quality_gates": [QualityGateType.SECURITY_SCAN]
                        },
                        {
                            "name": "Authentication & Authorization Review",
                            "description": "Review authentication and authorization implementation",
                            "required_skills": ["security_architecture", "auth_implementation"],
                            "estimated_hours": 6.0,
                            "quality_gates": [QualityGateType.SECURITY_SCAN]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.DEPLOYMENT,
                    "tasks": [
                        {
                            "name": "CI/CD Pipeline Setup",
                            "description": "Configure continuous integration and deployment",
                            "required_skills": ["devops", "ci_cd", "automation"],
                            "estimated_hours": 8.0,
                            "quality_gates": [QualityGateType.PEER_REVIEW]
                        },
                        {
                            "name": "Production Deployment",
                            "description": "Deploy application to production environment",
                            "required_skills": ["devops", "deployment", "monitoring"],
                            "estimated_hours": 6.0,
                            "quality_gates": [QualityGateType.PERFORMANCE_CHECK]
                        }
                    ]
                }
            ]
        }
        
        # API service development workflow
        self.workflow_templates["api_service"] = {
            "phases": [
                {
                    "phase": WorkflowPhase.REQUIREMENTS_ANALYSIS,
                    "tasks": [
                        {
                            "name": "API Requirements Analysis",
                            "description": "Analyze API requirements and constraints",
                            "required_skills": ["api_design", "requirements_analysis"],
                            "estimated_hours": 3.0,
                            "quality_gates": [QualityGateType.PEER_REVIEW]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.ARCHITECTURE_DESIGN,
                    "tasks": [
                        {
                            "name": "API Architecture Design",
                            "description": "Design API architecture and endpoints",
                            "required_skills": ["api_architecture", "rest_design"],
                            "estimated_hours": 6.0,
                            "quality_gates": [QualityGateType.ARCHITECTURE_REVIEW]
                        },
                        {
                            "name": "OpenAPI Specification",
                            "description": "Create detailed OpenAPI specification",
                            "required_skills": ["openapi", "api_documentation"],
                            "estimated_hours": 4.0,
                            "quality_gates": [QualityGateType.ARCHITECTURE_REVIEW]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.IMPLEMENTATION,
                    "tasks": [
                        {
                            "name": "API Implementation",
                            "description": "Implement API endpoints and business logic",
                            "required_skills": ["backend_development", "api_implementation"],
                            "estimated_hours": 24.0,
                            "quality_gates": [QualityGateType.CODE_QUALITY, QualityGateType.TEST_COVERAGE]
                        }
                    ]
                },
                {
                    "phase": WorkflowPhase.TESTING,
                    "tasks": [
                        {
                            "name": "API Testing",
                            "description": "Comprehensive API testing suite",
                            "required_skills": ["api_testing", "test_automation"],
                            "estimated_hours": 16.0,
                            "quality_gates": [QualityGateType.TEST_COVERAGE]
                        }
                    ]
                }
            ]
        }
    
    def _initialize_quality_gates(self):
        """Initialize quality gate configurations."""
        
        self.quality_gate_handlers = {
            QualityGateType.CODE_QUALITY: self._check_code_quality,
            QualityGateType.TEST_COVERAGE: self._check_test_coverage,
            QualityGateType.SECURITY_SCAN: self._check_security,
            QualityGateType.PERFORMANCE_CHECK: self._check_performance,
            QualityGateType.DOCUMENTATION_REVIEW: self._check_documentation,
            QualityGateType.ARCHITECTURE_REVIEW: self._check_architecture,
            QualityGateType.PEER_REVIEW: self._check_peer_review
        }
    
    async def create_workflow(
        self,
        project_id: str,
        workflow_name: str,
        template_name: str = "web_application",
        customizations: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Create a new workflow execution from template."""
        
        logger.info(f"Creating workflow '{workflow_name}' for project {project_id}")
        
        # Get workflow template
        template = self.workflow_templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown workflow template: {template_name}")
        
        # Create workflow execution
        workflow = WorkflowExecution(
            project_id=project_id,
            name=workflow_name,
            description=f"Autonomous development workflow for {workflow_name}"
        )
        
        # Create tasks from template
        task_id_map: Dict[str, str] = {}
        
        for phase_config in template["phases"]:
            phase = WorkflowPhase(phase_config["phase"])
            
            for task_config in phase_config["tasks"]:
                task = WorkflowTask(
                    name=task_config["name"],
                    description=task_config["description"],
                    phase=phase,
                    required_skills=task_config["required_skills"],
                    estimated_hours=task_config["estimated_hours"],
                    quality_gates=[QualityGateType(qg) for qg in task_config.get("quality_gates", [])]
                )
                
                # Apply customizations
                if customizations and task.name in customizations:
                    custom_config = customizations[task.name]
                    for key, value in custom_config.items():
                        if hasattr(task, key):
                            setattr(task, key, value)
                
                workflow.tasks.append(task)
                task_id_map[task.name] = task.id
        
        # Set up task dependencies based on phases
        previous_phase_tasks: List[str] = []
        current_phase = None
        
        for task in workflow.tasks:
            if task.phase != current_phase:
                # New phase - depend on all tasks from previous phase
                if previous_phase_tasks:
                    task.dependencies.extend(previous_phase_tasks)
                previous_phase_tasks = [task.id]
                current_phase = task.phase
            else:
                # Same phase - tasks can run in parallel
                previous_phase_tasks.append(task.id)
        
        # Calculate total estimated duration
        workflow.estimated_duration_hours = sum(task.estimated_hours for task in workflow.tasks)
        
        # Store workflow
        self.active_workflows[workflow.id] = workflow
        
        logger.info(f"Created workflow {workflow.id} with {len(workflow.tasks)} tasks")
        return workflow
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution."""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            logger.error(f"Workflow {workflow_id} not found")
            return False
        
        if workflow.status != TaskStatus.PENDING:
            logger.error(f"Workflow {workflow_id} is not in pending status")
            return False
        
        workflow.status = TaskStatus.IN_PROGRESS
        workflow.started_at = datetime.now()
        
        logger.info(f"Started workflow {workflow_id}")
        
        # Start orchestration if not already running
        if not self._running:
            await self.start_orchestration()
        
        return True
    
    async def start_orchestration(self):
        """Start the main orchestration loop."""
        
        if self._running:
            return
        
        self._running = True
        self._orchestration_task = asyncio.create_task(self._orchestration_loop())
        logger.info("Started workflow orchestration")
    
    async def stop_orchestration(self):
        """Stop the orchestration loop."""
        
        if not self._running:
            return
        
        self._running = False
        
        if self._orchestration_task:
            self._orchestration_task.cancel()
            try:
                await self._orchestration_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped workflow orchestration")
    
    async def _orchestration_loop(self):
        """Main orchestration loop that manages workflow execution."""
        
        while self._running:
            try:
                # Process all active workflows
                for workflow_id, workflow in list(self.active_workflows.items()):
                    if workflow.status == TaskStatus.IN_PROGRESS:
                        await self._process_workflow(workflow)
                
                # Sleep before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _process_workflow(self, workflow: WorkflowExecution):
        """Process a single workflow execution."""
        
        # Check if workflow is complete
        if all(task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED] for task in workflow.tasks):
            await self._complete_workflow(workflow)
            return
        
        # Find ready tasks
        ready_tasks = self._find_ready_tasks(workflow)
        
        # Execute ready tasks
        for task in ready_tasks:
            if task.status == TaskStatus.READY:
                asyncio.create_task(self._execute_task(workflow, task))
    
    def _find_ready_tasks(self, workflow: WorkflowExecution) -> List[WorkflowTask]:
        """Find tasks that are ready to execute."""
        
        ready_tasks: List[WorkflowTask] = []
        
        for task in workflow.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_completed = all(
                    any(dep_task.id == dep_id and dep_task.status == TaskStatus.COMPLETED 
                        for dep_task in workflow.tasks)
                    for dep_id in task.dependencies
                )
                
                if not task.dependencies or dependencies_completed:
                    task.status = TaskStatus.READY
                    ready_tasks.append(task)
        
        return ready_tasks
    
    async def _execute_task(self, workflow: WorkflowExecution, task: WorkflowTask):
        """Execute a single task."""
        
        logger.info(f"Executing task {task.name} in workflow {workflow.id}")
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Find appropriate agent for the task
            agent = await self._assign_agent(task)
            if not agent:
                logger.error(f"No suitable agent found for task {task.name}")
                task.status = TaskStatus.FAILED
                return
            
            task.assigned_agent = agent
            
            # Execute the task
            result = await self._run_task_execution(workflow, task, agent)
            
            if result.get("success", False):
                # Run quality gates
                quality_passed = await self._run_quality_gates(workflow, task)
                
                if quality_passed:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.actual_hours = (task.completed_at - task.started_at).total_seconds() / 3600
                    
                    logger.info(f"Task {task.name} completed successfully")
                else:
                    task.status = TaskStatus.REVIEW_REQUIRED
                    logger.warning(f"Task {task.name} failed quality gates")
            else:
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.name} execution failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error executing task {task.name}: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
    
    async def _assign_agent(self, task: WorkflowTask) -> Optional[str]:
        """Assign the most suitable agent for a task based on required skills."""
        
        # This would integrate with the team assembly system
        # For now, return a placeholder based on task phase
        agent_mapping = {
            WorkflowPhase.REQUIREMENTS_ANALYSIS: "business_analyst_agent",
            WorkflowPhase.ARCHITECTURE_DESIGN: "architect_agent",
            WorkflowPhase.IMPLEMENTATION: "developer_agent",
            WorkflowPhase.CODE_REVIEW: "reviewer_agent",
            WorkflowPhase.TESTING: "tester_agent",
            WorkflowPhase.SECURITY_AUDIT: "security_agent",
            WorkflowPhase.PERFORMANCE_OPTIMIZATION: "performance_agent",
            WorkflowPhase.DOCUMENTATION: "technical_writer_agent",
            WorkflowPhase.DEPLOYMENT: "devops_agent"
        }
        
        return agent_mapping.get(task.phase, "generalist_agent")
    
    async def _run_task_execution(
        self,
        workflow: WorkflowExecution,
        task: WorkflowTask,
        agent: str
    ) -> Dict[str, Any]:
        """Run the actual task execution with the assigned agent."""
        
        # Placeholder for actual agent execution
        # This would call the actual agent implementation
        await asyncio.sleep(1)  # Simulate work
        
        return {
            "success": True,
            "deliverables": task.deliverables,
            "duration": 1.0,
            "agent": agent
        }
    
    async def _run_quality_gates(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Run quality gates for a completed task."""
        
        if not task.quality_gates:
            return True
        
        logger.info(f"Running {len(task.quality_gates)} quality gates for task {task.name}")
        
        for gate_type in task.quality_gates:
            handler = self.quality_gate_handlers.get(gate_type)
            if not handler:
                logger.warning(f"No handler found for quality gate {gate_type}")
                continue
            
            try:
                passed = await handler(workflow, task)
                if not passed:
                    logger.warning(f"Quality gate {gate_type} failed for task {task.name}")
                    return False
            except Exception as e:
                logger.error(f"Error in quality gate {gate_type}: {e}", exc_info=True)
                return False
        
        logger.info(f"All quality gates passed for task {task.name}")
        return True
    
    async def _complete_workflow(self, workflow: WorkflowExecution):
        """Complete a workflow execution."""
        
        workflow.status = TaskStatus.COMPLETED
        workflow.completed_at = datetime.now()
        
        if workflow.started_at:
            workflow.actual_duration_hours = (
                workflow.completed_at - workflow.started_at
            ).total_seconds() / 3600
        
        # Calculate success rate
        completed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED)
        workflow.success_rate = completed_tasks / len(workflow.tasks) if workflow.tasks else 0.0
        
        logger.info(f"Workflow {workflow.id} completed with {workflow.success_rate:.1%} success rate")
    
    # Quality gate implementations
    async def _check_code_quality(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Check code quality standards."""
        # Placeholder - would integrate with actual code quality tools
        return True
    
    async def _check_test_coverage(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Check test coverage requirements."""
        # Placeholder - would integrate with coverage tools
        return True
    
    async def _check_security(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Check security requirements."""
        # Placeholder - would integrate with security scanning tools
        return True
    
    async def _check_performance(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Check performance requirements."""
        # Placeholder - would integrate with performance testing tools
        return True
    
    async def _check_documentation(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Check documentation completeness."""
        # Placeholder - would check for required documentation
        return True
    
    async def _check_architecture(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Check architectural compliance."""
        # Placeholder - would validate architectural decisions
        return True
    
    async def _check_peer_review(self, workflow: WorkflowExecution, task: WorkflowTask) -> bool:
        """Check peer review requirements."""
        # Placeholder - would integrate with code review systems
        return True
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed workflow status."""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "project_id": workflow.project_id,
            "status": workflow.status,
            "current_phase": workflow.current_phase,
            "progress": {
                "total_tasks": len(workflow.tasks),
                "completed_tasks": sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED),
                "in_progress_tasks": sum(1 for task in workflow.tasks if task.status == TaskStatus.IN_PROGRESS),
                "failed_tasks": sum(1 for task in workflow.tasks if task.status == TaskStatus.FAILED)
            },
            "estimated_duration_hours": workflow.estimated_duration_hours,
            "actual_duration_hours": workflow.actual_duration_hours,
            "success_rate": workflow.success_rate,
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at
        }
    
    async def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get status of all workflows."""
        
        workflows = []
        for workflow_id in self.active_workflows.keys():
            status = await self.get_workflow_status(workflow_id)
            if status:
                workflows.append(status)
        
        return workflows


async def main():
    """Test the workflow orchestrator."""
    
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = WorkflowOrchestrator()
    
    # Create a test workflow
    workflow = await orchestrator.create_workflow(
        project_id="test-project-1",
        workflow_name="Test Web Application",
        template_name="web_application"
    )
    
    print(f"Created workflow: {workflow.id}")
    print(f"Tasks: {len(workflow.tasks)}")
    
    # Start the workflow
    await orchestrator.start_workflow(workflow.id)
    
    # Monitor for a bit
    await asyncio.sleep(10)
    
    # Get status
    status = await orchestrator.get_workflow_status(workflow.id)
    print(f"Workflow status: {status}")
    
    await orchestrator.stop_orchestration()


if __name__ == "__main__":
    asyncio.run(main())