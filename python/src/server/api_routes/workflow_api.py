"""
Workflow Automation API Routes
RESTful API for workflow management and execution
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio

from ...agents.workflow import (
    WorkflowEngine, Workflow, WorkflowStep, WorkflowStatus, StepType, TriggerType,
    TaskScheduler, ScheduledTask, Schedule, ScheduleType, TaskPriority,
    AutomationRulesEngine, Rule, RuleType, Condition, ConditionOperator, Action, ActionType,
    TemplateLibrary, WorkflowTemplate, TemplateCategory
)

router = APIRouter(prefix="/api/workflow", tags=["workflow"])

# Initialize workflow components
workflow_engine = WorkflowEngine()
task_scheduler = TaskScheduler()
rules_engine = AutomationRulesEngine()
template_library = TemplateLibrary()


# Pydantic models for API
class WorkflowCreateRequest(BaseModel):
    """Workflow creation request"""
    name: str
    description: str
    template_id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    triggers: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStepRequest(BaseModel):
    """Workflow step request"""
    name: str
    step_type: str
    action: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    timeout: Optional[int] = None


class WorkflowExecuteRequest(BaseModel):
    """Workflow execution request"""
    workflow_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = True


class TaskCreateRequest(BaseModel):
    """Task creation request"""
    name: str
    description: str
    schedule_type: str
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: str = "normal"
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RuleCreateRequest(BaseModel):
    """Rule creation request"""
    name: str
    description: str
    rule_type: str
    conditions: Optional[Dict[str, Any]] = None
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    priority: int = 0
    enabled: bool = True


class EventRequest(BaseModel):
    """Event processing request"""
    event_type: str
    data: Dict[str, Any]
    source: Optional[str] = None
    timestamp: Optional[datetime] = None


# Workflow Management Endpoints
@router.post("/workflows")
async def create_workflow(request: WorkflowCreateRequest) -> Dict[str, Any]:
    """Create a new workflow"""
    try:
        if request.template_id:
            # Create from template
            workflow = template_library.instantiate_workflow(
                request.template_id,
                request.parameters
            )
            if not workflow:
                raise HTTPException(status_code=404, detail="Template not found")
        else:
            # Create new workflow
            workflow = workflow_engine.create_workflow(
                request.name,
                request.description
            )
            
            # Add custom steps if provided
            for step_data in request.steps:
                step = WorkflowStep(
                    step_id=step_data.get("id", str(uuid.uuid4())),
                    name=step_data["name"],
                    step_type=StepType(step_data["step_type"]),
                    action=step_data.get("action"),
                    parameters=step_data.get("parameters", {}),
                    next_steps=step_data.get("next_steps", [])
                )
                workflow_engine.add_step(workflow.workflow_id, step)
            
            # Set variables
            workflow.variables = request.variables
        
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workflows")
async def list_workflows(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """List all workflows"""
    workflows = list(workflow_engine.workflows.values())
    
    return {
        "total": len(workflows),
        "workflows": [
            {
                "workflow_id": w.workflow_id,
                "name": w.name,
                "description": w.description,
                "status": w.status.value,
                "created_at": w.created_at.isoformat(),
                "updated_at": w.updated_at.isoformat(),
                "version": w.version
            }
            for w in workflows[skip:skip + limit]
        ]
    }


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get workflow details"""
    if workflow_id not in workflow_engine.workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = workflow_engine.workflows[workflow_id]
    
    return {
        "workflow_id": workflow.workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "status": workflow.status.value,
        "version": workflow.version,
        "triggers": [
            {
                "trigger_id": t.trigger_id,
                "type": t.trigger_type.value,
                "enabled": t.enabled
            }
            for t in workflow.triggers
        ],
        "steps": [
            {
                "step_id": s.step_id,
                "name": s.name,
                "type": s.step_type.value,
                "action": s.action,
                "next_steps": s.next_steps
            }
            for s in workflow.steps
        ],
        "variables": workflow.variables,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat()
    }


@router.post("/workflows/{workflow_id}/steps")
async def add_workflow_step(
    workflow_id: str,
    request: WorkflowStepRequest
) -> Dict[str, Any]:
    """Add a step to workflow"""
    if workflow_id not in workflow_engine.workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    import uuid
    step = WorkflowStep(
        step_id=str(uuid.uuid4()),
        name=request.name,
        step_type=StepType(request.step_type),
        action=request.action,
        parameters=request.parameters,
        conditions=request.conditions,
        next_steps=request.next_steps,
        timeout=request.timeout
    )
    
    success = workflow_engine.add_step(workflow_id, step)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add step")
    
    return {
        "step_id": step.step_id,
        "message": "Step added successfully"
    }


@router.post("/workflows/{workflow_id}/validate")
async def validate_workflow(workflow_id: str) -> Dict[str, Any]:
    """Validate workflow configuration"""
    is_valid, errors = workflow_engine.validate_workflow(workflow_id)
    
    return {
        "valid": is_valid,
        "errors": errors
    }


# Workflow Execution Endpoints
@router.post("/workflows/execute")
async def execute_workflow(
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Execute a workflow"""
    if request.workflow_id not in workflow_engine.workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    try:
        if request.async_execution:
            # Execute asynchronously
            execution_id = await workflow_engine.execute_workflow(
                request.workflow_id,
                request.inputs
            )
            
            return {
                "execution_id": execution_id,
                "status": "started",
                "message": "Workflow execution started"
            }
        else:
            # Execute synchronously (wait for completion)
            execution_id = await workflow_engine.execute_workflow(
                request.workflow_id,
                request.inputs
            )
            
            # Wait for completion (with timeout)
            max_wait = 300  # 5 minutes
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < max_wait:
                execution = workflow_engine.get_execution_status(execution_id)
                if execution and execution.status in [
                    WorkflowStatus.COMPLETED,
                    WorkflowStatus.FAILED,
                    WorkflowStatus.CANCELLED
                ]:
                    return {
                        "execution_id": execution_id,
                        "status": execution.status.value,
                        "results": execution.results,
                        "errors": execution.errors
                    }
                await asyncio.sleep(1)
            
            return {
                "execution_id": execution_id,
                "status": "timeout",
                "message": "Execution timeout - check status separately"
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str) -> Dict[str, Any]:
    """Get workflow execution status"""
    execution = workflow_engine.get_execution_status(execution_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return {
        "execution_id": execution.execution_id,
        "workflow_id": execution.workflow_id,
        "status": execution.status.value,
        "current_step": execution.current_step,
        "execution_path": execution.execution_path,
        "started_at": execution.started_at.isoformat(),
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "results": execution.results,
        "errors": execution.errors,
        "metrics": execution.metrics
    }


@router.post("/executions/{execution_id}/pause")
async def pause_execution(execution_id: str) -> Dict[str, Any]:
    """Pause workflow execution"""
    await workflow_engine.pause_execution(execution_id)
    return {"message": "Execution paused"}


@router.post("/executions/{execution_id}/resume")
async def resume_execution(execution_id: str) -> Dict[str, Any]:
    """Resume workflow execution"""
    await workflow_engine.resume_execution(execution_id)
    return {"message": "Execution resumed"}


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str) -> Dict[str, Any]:
    """Cancel workflow execution"""
    await workflow_engine.cancel_execution(execution_id)
    return {"message": "Execution cancelled"}


# Task Scheduling Endpoints
@router.post("/tasks")
async def create_scheduled_task(request: TaskCreateRequest) -> Dict[str, Any]:
    """Create a scheduled task"""
    try:
        import uuid
        
        # Create schedule
        schedule = Schedule(
            schedule_type=ScheduleType(request.schedule_type),
            start_time=request.start_time or datetime.now(),
            end_time=request.end_time,
            cron_expression=request.cron_expression,
            interval_seconds=request.interval_seconds
        )
        
        # Create task
        task = ScheduledTask(
            task_id=str(uuid.uuid4()),
            name=request.name,
            description=request.description,
            schedule=schedule,
            action=lambda **kwargs: asyncio.create_task(
                workflow_engine.execute_workflow(request.action, kwargs)
            ),
            parameters=request.parameters,
            priority=TaskPriority[request.priority.upper()]
        )
        
        success = task_scheduler.add_task(task)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create task")
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "next_run": task.next_run.isoformat() if task.next_run else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tasks")
async def list_scheduled_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """List scheduled tasks"""
    tasks = list(task_scheduler.tasks.values())
    
    return {
        "total": len(tasks),
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "description": t.description,
                "enabled": t.enabled,
                "priority": t.priority.name,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "next_run": t.next_run.isoformat() if t.next_run else None
            }
            for t in tasks[skip:skip + limit]
        ]
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get task details"""
    task = task_scheduler.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task.task_id,
        "name": task.name,
        "description": task.description,
        "schedule_type": task.schedule.schedule_type.value,
        "enabled": task.enabled,
        "priority": task.priority.name,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "last_run": task.last_run.isoformat() if task.last_run else None,
        "next_run": task.next_run.isoformat() if task.next_run else None,
        "parameters": task.parameters
    }


@router.post("/tasks/{task_id}/execute")
async def execute_task_now(task_id: str) -> Dict[str, Any]:
    """Execute a task immediately"""
    try:
        execution_id = await task_scheduler.execute_task_now(task_id)
        return {
            "execution_id": execution_id,
            "message": "Task execution started"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tasks/{task_id}/enable")
async def enable_task(task_id: str) -> Dict[str, Any]:
    """Enable a scheduled task"""
    success = task_scheduler.enable_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task enabled"}


@router.post("/tasks/{task_id}/disable")
async def disable_task(task_id: str) -> Dict[str, Any]:
    """Disable a scheduled task"""
    success = task_scheduler.disable_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task disabled"}


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, Any]:
    """Delete a scheduled task"""
    success = task_scheduler.remove_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted"}


# Automation Rules Endpoints
@router.post("/rules")
async def create_rule(request: RuleCreateRequest) -> Dict[str, Any]:
    """Create an automation rule"""
    try:
        import uuid
        
        # Parse conditions
        conditions = None
        if request.conditions:
            from ...agents.workflow.automation_rules import ConditionGroup, LogicalOperator
            conditions = ConditionGroup(
                operator=LogicalOperator(request.conditions.get("operator", "and")),
                conditions=[]
            )
            
            for cond_data in request.conditions.get("conditions", []):
                condition = Condition(
                    field=cond_data["field"],
                    operator=ConditionOperator(cond_data["operator"]),
                    value=cond_data["value"]
                )
                conditions.conditions.append(condition)
        
        # Parse actions
        actions = []
        for action_data in request.actions:
            action = Action(
                action_type=ActionType(action_data["type"]),
                target=action_data.get("target"),
                parameters=action_data.get("parameters", {})
            )
            actions.append(action)
        
        # Create rule
        rule = Rule(
            rule_id=str(uuid.uuid4()),
            name=request.name,
            description=request.description,
            rule_type=RuleType(request.rule_type),
            conditions=conditions,
            actions=actions,
            priority=request.priority,
            enabled=request.enabled
        )
        
        success = rules_engine.add_rule(rule)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create rule")
        
        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "enabled": rule.enabled
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rules")
async def list_rules(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """List automation rules"""
    rules = list(rules_engine.rules.values())
    
    return {
        "total": len(rules),
        "rules": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "description": r.description,
                "rule_type": r.rule_type.value,
                "priority": r.priority,
                "enabled": r.enabled,
                "trigger_count": r.trigger_count,
                "last_triggered": r.last_triggered.isoformat() if r.last_triggered else None
            }
            for r in rules[skip:skip + limit]
        ]
    }


@router.post("/rules/evaluate")
async def evaluate_rules(data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate rules against data"""
    results = await rules_engine.evaluate(data)
    
    return {
        "evaluated_count": len(results),
        "triggered_count": sum(1 for r in results if r.triggered),
        "results": [
            {
                "rule_id": r.rule_id,
                "triggered": r.triggered,
                "conditions_met": r.conditions_met,
                "actions_executed": r.actions_executed,
                "execution_time": r.execution_time,
                "errors": r.errors
            }
            for r in results
        ]
    }


@router.post("/rules/{rule_id}/enable")
async def enable_rule(rule_id: str) -> Dict[str, Any]:
    """Enable a rule"""
    success = rules_engine.update_rule(rule_id, {"enabled": True})
    if not success:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"message": "Rule enabled"}


@router.post("/rules/{rule_id}/disable")
async def disable_rule(rule_id: str) -> Dict[str, Any]:
    """Disable a rule"""
    success = rules_engine.update_rule(rule_id, {"enabled": False})
    if not success:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"message": "Rule disabled"}


# Template Management Endpoints
@router.get("/templates")
async def list_templates(
    category: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """List workflow templates"""
    if category:
        templates = template_library.get_templates_by_category(
            TemplateCategory(category)
        )
    else:
        templates = list(template_library.templates.values())
    
    return {
        "total": len(templates),
        "templates": [
            {
                "template_id": t.template_id,
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "version": t.version,
                "author": t.author,
                "tags": t.tags,
                "usage_count": t.usage_count,
                "rating": t.rating
            }
            for t in templates[skip:skip + limit]
        ]
    }


@router.get("/templates/{template_id}")
async def get_template(template_id: str) -> Dict[str, Any]:
    """Get template details"""
    template = template_library.get_template(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {
        "template_id": template.template_id,
        "name": template.name,
        "description": template.description,
        "category": template.category.value,
        "version": template.version,
        "author": template.author,
        "tags": template.tags,
        "parameters": template.parameters,
        "variables": template.variables,
        "workflow_definition": template.workflow_definition,
        "usage_count": template.usage_count,
        "rating": template.rating,
        "created_at": template.created_at.isoformat(),
        "updated_at": template.updated_at.isoformat()
    }


@router.get("/templates/search")
async def search_templates(
    q: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=100)
) -> Dict[str, Any]:
    """Search workflow templates"""
    results = template_library.search_templates(q)
    
    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "template_id": t.template_id,
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "tags": t.tags
            }
            for t in results[:limit]
        ]
    }


# Event Processing Endpoints
@router.post("/events")
async def process_event(request: EventRequest) -> Dict[str, Any]:
    """Process an event through the workflow engine"""
    event_data = {
        "type": request.event_type,
        "data": request.data,
        "source": request.source or "api",
        "timestamp": (request.timestamp or datetime.now()).isoformat()
    }
    
    # Process through rules engine
    rule_results = await rules_engine.process_event(event_data)
    
    # Add to workflow engine event queue
    await workflow_engine.event_queue.put(event_data)
    
    return {
        "event_id": str(uuid.uuid4()),
        "processed": True,
        "rules_triggered": len(rule_results),
        "message": "Event processed successfully"
    }


# Metrics and Statistics Endpoints
@router.get("/metrics")
async def get_workflow_metrics() -> Dict[str, Any]:
    """Get workflow system metrics"""
    workflow_metrics = workflow_engine.get_workflow_metrics()
    task_metrics = task_scheduler.get_metrics()
    rule_metrics = rules_engine.get_metrics()
    rule_stats = rules_engine.get_rule_statistics()
    
    return {
        "workflows": workflow_metrics,
        "tasks": task_metrics,
        "rules": rule_metrics,
        "statistics": rule_stats,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for workflow system"""
    return {
        "status": "healthy",
        "components": {
            "workflow_engine": "active",
            "task_scheduler": "active",
            "rules_engine": "active",
            "template_library": "active"
        },
        "timestamp": datetime.now().isoformat()
    }


# Upcoming tasks endpoint
@router.get("/tasks/upcoming")
async def get_upcoming_tasks(
    hours: int = Query(24, ge=1, le=168)
) -> Dict[str, Any]:
    """Get upcoming scheduled tasks"""
    upcoming = task_scheduler.get_upcoming_tasks(hours)
    
    return {
        "hours": hours,
        "count": len(upcoming),
        "tasks": [
            {
                "task_id": task_id,
                "scheduled_time": scheduled_time.isoformat(),
                "task_name": task_scheduler.tasks[task_id].name if task_id in task_scheduler.tasks else "Unknown"
            }
            for scheduled_time, task_id in upcoming
        ]
    }


# Export/Import endpoints
@router.get("/workflows/{workflow_id}/export")
async def export_workflow(workflow_id: str) -> Dict[str, Any]:
    """Export workflow as JSON"""
    if workflow_id not in workflow_engine.workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = workflow_engine.workflows[workflow_id]
    
    return {
        "name": workflow.name,
        "description": workflow.description,
        "version": workflow.version,
        "triggers": [
            {
                "type": t.trigger_type.value,
                "conditions": t.conditions,
                "enabled": t.enabled
            }
            for t in workflow.triggers
        ],
        "steps": [
            {
                "id": s.step_id,
                "name": s.name,
                "type": s.step_type.value,
                "action": s.action,
                "parameters": s.parameters,
                "conditions": s.conditions,
                "next_steps": s.next_steps,
                "timeout": s.timeout
            }
            for s in workflow.steps
        ],
        "variables": workflow.variables,
        "tags": workflow.tags
    }


@router.post("/templates/import")
async def import_template(template_data: Dict[str, Any]) -> Dict[str, Any]:
    """Import a workflow template"""
    try:
        import json
        template_json = json.dumps(template_data)
        template = template_library.import_template(template_json)
        
        if not template:
            raise HTTPException(status_code=400, detail="Failed to import template")
        
        return {
            "template_id": template.template_id,
            "name": template.name,
            "message": "Template imported successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))