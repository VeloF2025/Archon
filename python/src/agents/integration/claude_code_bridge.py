"""
Claude Code Task Tool Integration Bridge for Phase 6
Enables specialized agents to be called through Claude Code's Task tool
REAL EXECUTION ONLY - NO SIMULATION
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import httpx
from pathlib import Path

# Import AgentTask for Phase 2 integration
from ..orchestration.parallel_executor import AgentTask

logger = logging.getLogger(__name__)

class TaskToolType(Enum):
    """Available Claude Code Task tool types"""
    GENERAL_PURPOSE = "general-purpose"
    STATUSLINE_SETUP = "statusline-setup"
    OUTPUT_STYLE_SETUP = "output-style-setup"
    UI_UX_OPTIMIZER = "ui-ux-optimizer"
    TEST_COVERAGE_VALIDATOR = "test-coverage-validator"
    SYSTEM_ARCHITECT = "system-architect"
    STRATEGIC_PLANNER = "strategic-planner"
    SECURITY_AUDITOR = "security-auditor"
    PERFORMANCE_OPTIMIZER = "performance-optimizer"
    DOCUMENTATION_GENERATOR = "documentation-generator"
    DEVOPS_AUTOMATION = "devops-automation"
    DEPLOYMENT_AUTOMATION = "deployment-automation"
    DATABASE_ARCHITECT = "database-architect"
    CODE_REFACTORING_OPTIMIZER = "code-refactoring-optimizer"
    CODE_QUALITY_REVIEWER = "code-quality-reviewer"
    CODE_IMPLEMENTER = "code-implementer"
    API_DESIGN_ARCHITECT = "api-design-architect"
    ANTIHALLUCINATION_VALIDATOR = "antihallucination-validator"

@dataclass
class AgentMapping:
    """Mapping between Archon agents and Claude Code Task tool types"""
    archon_agent: str
    claude_code_tool: TaskToolType
    tool_permissions: List[str]
    autonomous_triggers: List[str] = field(default_factory=list)
    requires_human_approval: bool = False
    max_concurrent: int = 2

@dataclass
class TaskBridgeRequest:
    """Request to execute task through Claude Code bridge"""
    task_id: str
    agent_role: str
    task_description: str
    context: Dict[str, Any]
    tool_permissions: List[str]
    timeout_minutes: int = 10
    real_execution: bool = True

@dataclass
class TaskBridgeResult:
    """Result from Claude Code task execution"""
    task_id: str
    status: str
    output: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

class ClaudeCodeAgentBridge:
    """
    Bridge between Archon specialized agents and Claude Code Task tool.
    Enables real agent execution through Claude Code's sub-agent system.
    """
    
    def __init__(self, claude_code_api_url: str = "http://localhost:8181"):
        self.api_url = claude_code_api_url
        self.agent_mappings = self._initialize_agent_mappings()
        self.active_tasks: Dict[str, TaskBridgeRequest] = {}
        self.execution_history: List[TaskBridgeResult] = []
        
        logger.info("Initialized ClaudeCodeAgentBridge")
    
    def _initialize_agent_mappings(self) -> Dict[str, AgentMapping]:
        """Initialize mappings between Archon agents and Claude Code tools - REAL MAPPINGS"""
        mappings = {
            # Development agents
            "python_backend_coder": AgentMapping(
                archon_agent="python_backend_coder",
                claude_code_tool=TaskToolType.CODE_IMPLEMENTER,
                tool_permissions=["Read", "Write", "Edit", "Bash", "Grep"],
                autonomous_triggers=["*.py", "requirements.txt", "pyproject.toml"],
                max_concurrent=3
            ),
            "typescript_frontend_agent": AgentMapping(
                archon_agent="typescript_frontend_agent", 
                claude_code_tool=TaskToolType.CODE_IMPLEMENTER,
                tool_permissions=["Read", "Write", "Edit", "Bash", "Grep"],
                autonomous_triggers=["*.tsx", "*.ts", "package.json"],
                max_concurrent=3
            ),
            "api_integrator": AgentMapping(
                archon_agent="api_integrator",
                claude_code_tool=TaskToolType.API_DESIGN_ARCHITECT,
                tool_permissions=["Read", "Write", "Edit", "Bash"],
                autonomous_triggers=["*api*", "*.openapi.json"],
                max_concurrent=2
            ),
            
            # Quality agents
            "security_auditor": AgentMapping(
                archon_agent="security_auditor",
                claude_code_tool=TaskToolType.SECURITY_AUDITOR,
                tool_permissions=["Read", "Grep", "Bash"],  # NO Write/Edit for security
                autonomous_triggers=["*.py", "*.ts", "*.tsx", "*.js"],
                max_concurrent=2
            ),
            "test_generator": AgentMapping(
                archon_agent="test_generator",
                claude_code_tool=TaskToolType.TEST_COVERAGE_VALIDATOR,
                tool_permissions=["Read", "Write", "Edit", "Bash"],
                autonomous_triggers=["test_*.py", "*test*.js", "*.spec.ts"],
                max_concurrent=4
            ),
            "code_reviewer": AgentMapping(
                archon_agent="code_reviewer",
                claude_code_tool=TaskToolType.CODE_QUALITY_REVIEWER,
                tool_permissions=["Read", "Grep", "Bash"],  # Read-only for reviews
                autonomous_triggers=["*.py", "*.ts", "*.tsx", "*.js"],
                max_concurrent=2
            ),
            
            # Documentation agents
            "documentation_writer": AgentMapping(
                archon_agent="documentation_writer",
                claude_code_tool=TaskToolType.DOCUMENTATION_GENERATOR,
                tool_permissions=["Read", "Write", "Edit"],
                autonomous_triggers=["*.md", "README*", "docs/*"],
                requires_human_approval=True,  # Documentation changes need approval
                max_concurrent=1
            ),
            "technical_writer": AgentMapping(
                archon_agent="technical_writer",
                claude_code_tool=TaskToolType.DOCUMENTATION_GENERATOR,
                tool_permissions=["Read", "Write", "Edit"],
                autonomous_triggers=["*.md", "docs/*"],
                requires_human_approval=True,
                max_concurrent=1
            ),
            
            # Architecture agents
            "system_architect": AgentMapping(
                archon_agent="system_architect",
                claude_code_tool=TaskToolType.SYSTEM_ARCHITECT,
                tool_permissions=["Read", "Write", "Edit", "Grep"],
                autonomous_triggers=[],  # Manual only
                requires_human_approval=True,
                max_concurrent=1
            ),
            "database_designer": AgentMapping(
                archon_agent="database_designer",
                claude_code_tool=TaskToolType.DATABASE_ARCHITECT,
                tool_permissions=["Read", "Write", "Edit", "Bash"],
                autonomous_triggers=["*.sql", "migrations/*", "schema.*"],
                max_concurrent=1
            ),
            
            # Operations agents
            "devops_engineer": AgentMapping(
                archon_agent="devops_engineer",
                claude_code_tool=TaskToolType.DEVOPS_AUTOMATION,
                tool_permissions=["Read", "Write", "Edit", "Bash"],
                autonomous_triggers=["Dockerfile", "docker-compose.*", ".github/*"],
                max_concurrent=2
            ),
            "deployment_coordinator": AgentMapping(
                archon_agent="deployment_coordinator",
                claude_code_tool=TaskToolType.DEPLOYMENT_AUTOMATION,
                tool_permissions=["Read", "Bash", "Grep"],
                autonomous_triggers=[],  # Manual deployment only
                requires_human_approval=True,
                max_concurrent=1
            ),
            
            # Optimization agents
            "performance_optimizer": AgentMapping(
                archon_agent="performance_optimizer",
                claude_code_tool=TaskToolType.PERFORMANCE_OPTIMIZER,
                tool_permissions=["Read", "Edit", "Bash", "Grep"],
                autonomous_triggers=["*.py", "*.ts", "*.tsx"],
                max_concurrent=2
            ),
            "refactoring_specialist": AgentMapping(
                archon_agent="refactoring_specialist",
                claude_code_tool=TaskToolType.CODE_REFACTORING_OPTIMIZER,
                tool_permissions=["Read", "Edit", "Grep"],
                autonomous_triggers=["*.py", "*.ts", "*.tsx"],
                max_concurrent=2
            ),
            
            # Design agents
            "ui_ux_designer": AgentMapping(
                archon_agent="ui_ux_designer",
                claude_code_tool=TaskToolType.UI_UX_OPTIMIZER,
                tool_permissions=["Read", "Write", "Edit"],
                autonomous_triggers=["*.tsx", "*.css", "*.scss"],
                max_concurrent=1
            )
        }
        
        logger.info(f"Initialized {len(mappings)} agent mappings")
        return mappings
    
    async def execute_task_via_claude_code(self, request: TaskBridgeRequest) -> TaskBridgeResult:
        """
        Execute task through Claude Code Task tool - REAL EXECUTION
        
        Args:
            request: Task execution request
            
        Returns:
            Real execution result from Claude Code
        """
        start_time = time.time()
        logger.info(f"Executing task {request.task_id} via Claude Code Task tool")
        
        try:
            # Get agent mapping
            if request.agent_role not in self.agent_mappings:
                raise ValueError(f"No mapping found for agent {request.agent_role}")
            
            mapping = self.agent_mappings[request.agent_role]
            
            # Validate tool permissions BEFORE execution
            await self._validate_tool_permissions(request.agent_role, request.tool_permissions)
            
            # Create Claude Code Task tool request
            claude_task_request = {
                "description": f"Execute {request.agent_role} task",
                "prompt": f"""
                Agent Role: {request.agent_role}
                Task: {request.task_description}
                
                Context:
                {json.dumps(request.context, indent=2)}
                
                IMPORTANT: This is a REAL execution for Phase 6 agent integration.
                NO SIMULATION OR MOCKING allowed.
                Use only the following tools: {', '.join(mapping.tool_permissions)}
                
                Complete the task according to the specialized agent's capabilities.
                """,
                "subagent_type": mapping.claude_code_tool.value
            }
            
            # REAL: Call Claude Code Task tool API
            async with httpx.AsyncClient(timeout=request.timeout_minutes * 60) as client:
                # Create proper API request format
                api_request = {
                    "agent_type": mapping.archon_agent,  # Use Archon agent name
                    "prompt": claude_task_request["prompt"],
                    "context": request.context,
                    "timeout": request.timeout_minutes * 60
                }
                
                logger.info(f"Calling Claude task API with agent {mapping.archon_agent}")
                
                response = await client.post(
                    f"{self.api_url}/api/claude-code/task",
                    json={
                        "subagent_type": mapping.archon_agent,
                        "description": request.task_description,
                        "prompt": claude_task_request["prompt"],
                        "context": request.context,
                        "timeout": request.timeout_minutes * 60
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Claude Code API error: {response.status_code} - {response.text}")
                
                result_data = response.json()
                
                execution_time = time.time() - start_time
                
                # Extract REAL results
                result = TaskBridgeResult(
                    task_id=request.task_id,
                    status="completed" if result_data.get("success", False) else "failed",
                    output=result_data.get("result", ""),
                    error_message=result_data.get("error"),
                    execution_time=execution_time,
                    tools_used=result_data.get("tools_used", []),
                    files_modified=result_data.get("files_modified", [])
                )
                
                # Track execution
                self.execution_history.append(result)
                if request.task_id in self.active_tasks:
                    del self.active_tasks[request.task_id]
                
                logger.info(f"Task {request.task_id} completed via Claude Code in {execution_time:.2f}s")
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = TaskBridgeResult(
                task_id=request.task_id,
                status="failed",
                error_message=str(e),
                execution_time=execution_time
            )
            
            self.execution_history.append(error_result)
            logger.error(f"Task {request.task_id} failed: {e}")
            return error_result
    
    async def _validate_tool_permissions(self, agent_role: str, requested_tools: List[str]) -> bool:
        """Validate agent tool permissions - REAL SECURITY CHECK"""
        
        if agent_role not in self.agent_mappings:
            raise ValueError(f"Unknown agent role: {agent_role}")
        
        mapping = self.agent_mappings[agent_role]
        
        # Check each requested tool against permissions
        for tool in requested_tools:
            if tool not in mapping.tool_permissions:
                raise PermissionError(f"Agent {agent_role} not authorized for tool {tool}")
        
        logger.debug(f"Tool permissions validated for {agent_role}")
        return True
    
    async def spawn_agent_via_task_tool(self, agent_role: str, task_description: str, 
                                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Spawn specialized agent through Claude Code Task tool - REAL SPAWNING
        
        Args:
            agent_role: Archon agent role to spawn
            task_description: Task for the agent to execute
            context: Additional context for the task
            
        Returns:
            Task ID for tracking execution
        """
        
        if agent_role not in self.agent_mappings:
            raise ValueError(f"No Claude Code mapping for agent {agent_role}")
        
        mapping = self.agent_mappings[agent_role]
        task_id = str(uuid.uuid4())
        
        # Create bridge request
        request = TaskBridgeRequest(
            task_id=task_id,
            agent_role=agent_role,
            task_description=task_description,
            context=context or {},
            tool_permissions=mapping.tool_permissions,
            timeout_minutes=10
        )
        
        # Track active task
        self.active_tasks[task_id] = request
        
        logger.info(f"Spawning {agent_role} via Claude Code Task tool (ID: {task_id})")
        
        # Execute task asynchronously
        asyncio.create_task(self.execute_task_via_claude_code(request))
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout_seconds: int = 60) -> Optional[TaskBridgeResult]:
        """Get result from executed task - REAL RESULT RETRIEVAL"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check if task completed
            for result in self.execution_history:
                if result.task_id == task_id:
                    return result
            
            # Check if task still active
            if task_id not in self.active_tasks:
                # Task completed but result not found - error
                break
                
            await asyncio.sleep(0.5)
        
        logger.warning(f"Task {task_id} result not available within timeout")
        return None
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent roles - REAL AGENT LIST"""
        return list(self.agent_mappings.keys())
    
    def get_agent_capabilities(self, agent_role: str) -> Dict[str, Any]:
        """Get agent capabilities and permissions - REAL CAPABILITIES"""
        
        if agent_role not in self.agent_mappings:
            return {}
        
        mapping = self.agent_mappings[agent_role]
        
        return {
            "agent_role": agent_role,
            "claude_code_tool": mapping.claude_code_tool.value,
            "tool_permissions": mapping.tool_permissions,
            "autonomous_triggers": mapping.autonomous_triggers,
            "requires_approval": mapping.requires_human_approval,
            "max_concurrent": mapping.max_concurrent
        }
    
    async def test_integration_health(self) -> Dict[str, Any]:
        """Test Claude Code integration health - REAL HEALTH CHECK"""
        logger.info("Testing Claude Code integration health...")
        
        health_results = {
            "claude_code_api_available": False,
            "task_tool_accessible": False,
            "agent_mappings_valid": False,
            "total_agents_mapped": len(self.agent_mappings),
            "test_timestamp": datetime.now().isoformat()
        }
        
        try:
            # REAL: Test Claude Code API availability
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_url}/health")
                health_results["claude_code_api_available"] = response.status_code == 200
                
                # REAL: Test Task tool endpoint
                if health_results["claude_code_api_available"]:
                    # Test with a simple task to validate integration
                    test_task = {
                        "subagent_type": "documentation_writer",
                        "description": "Health check test",
                        "prompt": "Return 'HEALTH_CHECK_SUCCESS' to confirm Task tool is working",
                        "context": {"test": True}
                    }
                    
                    task_response = await client.post(
                        f"{self.api_url}/api/claude-code/task",
                        json=test_task
                    )
                    
                    health_results["task_tool_accessible"] = task_response.status_code == 200
                    
                    if health_results["task_tool_accessible"]:
                        task_result = task_response.json()
                        health_results["task_tool_working"] = "HEALTH_CHECK_SUCCESS" in str(task_result.get("result", ""))
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_results["error"] = str(e)
        
        # Validate agent mappings
        health_results["agent_mappings_valid"] = all(
            mapping.claude_code_tool in TaskToolType for mapping in self.agent_mappings.values()
        )
        
        logger.info(f"Integration health: API={health_results['claude_code_api_available']}, "
                   f"Task Tool={health_results['task_tool_accessible']}, "
                   f"Mappings={health_results['agent_mappings_valid']}")
        
        return health_results

class AutonomousWorkflowTrigger:
    """
    Autonomous workflow trigger system for file-based agent spawning.
    REAL FILE MONITORING - NO SIMULATION
    """
    
    def __init__(self, bridge: ClaudeCodeAgentBridge, watch_directory: str = "."):
        self.bridge = bridge
        self.watch_directory = Path(watch_directory)
        self.monitoring_active = False
        self.trigger_history: List[Dict[str, Any]] = []
        
        # File pattern to agent mappings
        self.trigger_patterns = self._build_trigger_patterns()
        
        logger.info(f"Initialized AutonomousWorkflowTrigger for {self.watch_directory}")
    
    def _build_trigger_patterns(self) -> Dict[str, List[str]]:
        """Build file pattern to agent role mappings - REAL PATTERNS"""
        patterns = {}
        
        for agent_role, mapping in self.bridge.agent_mappings.items():
            for pattern in mapping.autonomous_triggers:
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(agent_role)
        
        logger.info(f"Built {len(patterns)} trigger patterns for autonomous workflows")
        return patterns
    
    async def start_monitoring(self):
        """Start autonomous file monitoring - REAL FILE WATCHING"""
        if self.monitoring_active:
            logger.warning("File monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting autonomous file monitoring...")
        
        # REAL: This would use a proper file watching system
        # For now, implement basic polling until real file watcher is added
        asyncio.create_task(self._monitor_file_changes())
        
        logger.info("Autonomous file monitoring started")
    
    async def stop_monitoring(self):
        """Stop autonomous file monitoring"""
        self.monitoring_active = False
        logger.info("Stopped autonomous file monitoring")
    
    async def _monitor_file_changes(self):
        """Monitor for file changes and trigger agents - REAL MONITORING"""
        
        # Enhanced file monitoring with proper change detection
        file_states = {}  # Track file modification times
        check_interval = 1.0  # Check every second
        
        # Initial scan to establish baseline
        for file_path in self.watch_directory.rglob("*"):
            if file_path.is_file():
                try:
                    file_states[str(file_path)] = file_path.stat().st_mtime
                except OSError:
                    continue
        
        logger.info(f"Monitoring {len(file_states)} files for changes")
        
        while self.monitoring_active:
            try:
                current_files = {}
                changes_detected = False
                
                # Scan all files in directory
                for file_path in self.watch_directory.rglob("*"):
                    if not file_path.is_file():
                        continue
                        
                    try:
                        file_str = str(file_path)
                        current_mtime = file_path.stat().st_mtime
                        current_files[file_str] = current_mtime
                        
                        # Check for modifications
                        if file_str in file_states:
                            if current_mtime > file_states[file_str]:
                                logger.info(f"File modified: {file_path.name}")
                                await self._handle_file_change(file_path)
                                changes_detected = True
                        else:
                            # New file created
                            logger.info(f"New file created: {file_path.name}")
                            await self._handle_file_change(file_path)
                            changes_detected = True
                            
                    except OSError as e:
                        logger.debug(f"Could not stat file {file_path}: {e}")
                        continue
                
                # Update file states
                file_states = current_files
                
                # Shorter sleep if changes detected (reactive monitoring)
                sleep_time = 0.5 if changes_detected else check_interval
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"File monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_file_change(self, file_path: Path):
        """Handle individual file change - REAL AGENT TRIGGERING"""
        
        file_name = file_path.name
        file_relative = str(file_path.relative_to(self.watch_directory))
        
        # Find matching trigger patterns
        triggered_agents = []
        
        for pattern, agent_roles in self.trigger_patterns.items():
            if self._matches_pattern(file_name, pattern) or self._matches_pattern(file_relative, pattern):
                triggered_agents.extend(agent_roles)
        
        # Remove duplicates
        triggered_agents = list(set(triggered_agents))
        
        if triggered_agents:
            logger.info(f"File change {file_relative} triggered agents: {triggered_agents}")
            
            # REAL: Spawn agents for file change
            for agent_role in triggered_agents:
                await self._trigger_agent_for_file(agent_role, file_path)
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches trigger pattern with enhanced pattern matching"""
        import fnmatch
        import re
        
        # Direct match
        if fnmatch.fnmatch(filename, pattern):
            return True
        
        # Check if pattern contains directory separators
        if '/' in pattern or '\\' in pattern:
            # Pattern includes path - check full relative path
            return fnmatch.fnmatch(filename, pattern)
        
        # Check if filename contains pattern as substring (for patterns like *api*)
        if '*' in pattern:
            # Convert glob pattern to regex for better matching
            regex_pattern = pattern.replace('*', '.*').replace('?', '.')
            return bool(re.search(regex_pattern, filename, re.IGNORECASE))
        
        # Exact match or case-insensitive substring
        return pattern.lower() in filename.lower()
    
    async def _trigger_agent_for_file(self, agent_role: str, file_path: Path):
        """Trigger specific agent for file change - REAL AGENT EXECUTION"""
        
        try:
            # Create context-aware task description
            task_description = f"Process file change in {file_path.name}"
            
            if agent_role == "security_auditor":
                task_description = f"Security audit of modified file: {file_path}"
            elif agent_role == "test_generator":
                task_description = f"Generate or update tests for: {file_path}"
            elif "frontend" in agent_role:
                task_description = f"Review and optimize frontend component: {file_path}"
            elif "backend" in agent_role:
                task_description = f"Review and optimize backend code: {file_path}"
            
            context = {
                "trigger_file": str(file_path),
                "trigger_type": "file_change",
                "autonomous": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # REAL: Spawn agent via bridge
            task_id = await self.bridge.spawn_agent_via_task_tool(
                agent_role=agent_role,
                task_description=task_description,
                context=context
            )
            
            # Record trigger
            self.trigger_history.append({
                "timestamp": datetime.now().isoformat(),
                "file_path": str(file_path),
                "agent_role": agent_role,
                "task_id": task_id,
                "trigger_type": "autonomous"
            })
            
            logger.info(f"Triggered {agent_role} for {file_path} (task: {task_id})")
            
        except Exception as e:
            logger.error(f"Failed to trigger {agent_role} for {file_path}: {e}")

class ToolAccessController:
    """
    Tool access control and security enforcement for agents.
    REAL SECURITY - NO BYPASSES
    """
    
    def __init__(self):
        self.access_rules: Dict[str, List[str]] = {}
        self.violation_log: List[Dict[str, Any]] = []
        self.blocked_attempts: int = 0
        
        self._initialize_access_rules()
        logger.info("Initialized ToolAccessController")
    
    def _initialize_access_rules(self):
        """Initialize tool access rules for each agent type - REAL SECURITY RULES"""
        
        self.access_rules = {
            # Development agents - full access
            "python_backend_coder": ["Read", "Write", "Edit", "Bash", "Grep", "MultiEdit"],
            "typescript_frontend_agent": ["Read", "Write", "Edit", "Bash", "Grep", "MultiEdit"],
            "api_integrator": ["Read", "Write", "Edit", "Bash", "Grep"],
            
            # Security agents - read-only + analysis tools
            "security_auditor": ["Read", "Grep", "Bash"],
            "code_reviewer": ["Read", "Grep", "Bash"],
            
            # Test agents - test-focused access
            "test_generator": ["Read", "Write", "Edit", "Bash", "Grep"],
            "integration_tester": ["Read", "Bash", "Grep"],
            
            # Documentation agents - documentation-focused
            "documentation_writer": ["Read", "Write", "Edit"],
            "technical_writer": ["Read", "Write", "Edit"],
            
            # Architecture agents - design access
            "system_architect": ["Read", "Write", "Edit", "Grep"],
            "database_designer": ["Read", "Write", "Edit", "Bash"],
            
            # Operations agents - deployment access
            "devops_engineer": ["Read", "Write", "Edit", "Bash", "Grep"],
            "deployment_coordinator": ["Read", "Bash", "Grep"],
            "monitoring_agent": ["Read", "Bash", "Grep"],
            
            # Optimization agents - analysis + modification
            "performance_optimizer": ["Read", "Edit", "Bash", "Grep"],
            "refactoring_specialist": ["Read", "Edit", "Grep"],
            
            # Design agents - UI/UX access
            "ui_ux_designer": ["Read", "Write", "Edit"],
            
            # Analysis agents - read-only
            "data_analyst": ["Read", "Grep", "Bash"],
            "hrm_reasoning_agent": ["Read", "Grep"]
        }
        
        logger.info(f"Initialized access rules for {len(self.access_rules)} agent types")
    
    async def validate_tool_access(self, agent_role: str, tool: str, context: Dict[str, Any]) -> bool:
        """Validate if agent can access specific tool - REAL VALIDATION"""
        
        # Check if agent has permission
        if agent_role not in self.access_rules:
            await self._log_violation(agent_role, tool, "Unknown agent role", context)
            return False
        
        if tool not in self.access_rules[agent_role]:
            await self._log_violation(agent_role, tool, "Tool not permitted", context)
            return False
        
        # Additional context-based validation
        if not await self._validate_context_permissions(agent_role, tool, context):
            await self._log_violation(agent_role, tool, "Context violation", context)
            return False
        
        logger.debug(f"Tool access granted: {agent_role} -> {tool}")
        return True
    
    async def _validate_context_permissions(self, agent_role: str, tool: str, context: Dict[str, Any]) -> bool:
        """Validate context-specific permissions - REAL CONTEXT VALIDATION"""
        
        # Security agents cannot write to production files
        if agent_role == "security_auditor" and tool in ["Write", "Edit"]:
            return False
        
        # Documentation agents cannot execute system commands
        if "documentation" in agent_role and tool == "Bash":
            return False
        
        # Deployment agents need approval for production operations
        if agent_role == "deployment_coordinator" and context.get("environment") == "production":
            return context.get("human_approved", False)
        
        return True
    
    async def _log_violation(self, agent_role: str, tool: str, reason: str, context: Dict[str, Any]):
        """Log security violation - REAL AUDIT LOGGING"""
        
        violation = {
            "timestamp": datetime.now().isoformat(),
            "agent_role": agent_role,
            "tool": tool,
            "reason": reason,
            "context": context,
            "blocked": True
        }
        
        self.violation_log.append(violation)
        self.blocked_attempts += 1
        
        logger.warning(f"SECURITY VIOLATION BLOCKED: {agent_role} attempted {tool} - {reason}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics - REAL METRICS"""
        return {
            "total_violations": len(self.violation_log),
            "blocked_attempts": self.blocked_attempts,
            "agents_with_violations": len(set(v["agent_role"] for v in self.violation_log)),
            "most_violated_tools": self._get_most_violated_tools(),
            "recent_violations": self.violation_log[-10:] if self.violation_log else []
        }
    
    def _get_most_violated_tools(self) -> Dict[str, int]:
        """Get most frequently violated tools"""
        tool_counts = {}
        for violation in self.violation_log:
            tool = violation["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        return dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5])

# Integration with existing Phase 2 system
class Phase6MetaAgentIntegration:
    """Integration layer between Phase 6 and existing Phase 2 parallel system"""
    
    def __init__(self):
        self.claude_bridge = ClaudeCodeAgentBridge()
        self.autonomous_trigger = AutonomousWorkflowTrigger(self.claude_bridge)
        self.access_controller = ToolAccessController()
        
        logger.info("Initialized Phase 6 Meta-Agent Integration")
    
    async def execute_task_with_specialized_agent(self, task: AgentTask) -> TaskBridgeResult:
        """Execute task using specialized agent through Claude Code - REAL EXECUTION"""
        
        # Validate agent exists and tools are permitted
        agent_capabilities = self.claude_bridge.get_agent_capabilities(task.agent_role)
        if not agent_capabilities:
            raise ValueError(f"Specialized agent {task.agent_role} not available")
        
        # Create bridge request
        request = TaskBridgeRequest(
            task_id=task.task_id,
            agent_role=task.agent_role,
            task_description=task.description,
            context=task.input_data,
            tool_permissions=agent_capabilities["tool_permissions"]
        )
        
        # Execute through Claude Code
        return await self.claude_bridge.execute_task_via_claude_code(request)
    
    async def start_autonomous_workflows(self):
        """Start autonomous workflow monitoring - REAL AUTOMATION"""
        await self.autonomous_trigger.start_monitoring()
        logger.info("Autonomous workflows started")
    
    async def stop_autonomous_workflows(self):
        """Stop autonomous workflow monitoring"""
        await self.autonomous_trigger.stop_monitoring()
        logger.info("Autonomous workflows stopped")