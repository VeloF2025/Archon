#!/usr/bin/env python3
"""
Dynamic Task Tracking System for Archon+
Updates tasks in real-time via Archon's API and Socket.IO integration

Features:
- Real-time task status updates
- Progress tracking with metrics
- Phase gate validation
- SCWT benchmark integration
- Automatic task transitions
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArchonTaskTracker:
    """Dynamic task tracking system integrated with Archon API"""
    
    def __init__(self, archon_api_base: str = "http://localhost:8181/api"):
        self.api_base = archon_api_base
        self.project_id = "85bc9bf7-465e-4235-9990-969adac869e5"  # Archon+ project ID
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Task status mappings
        self.status_map = {
            "todo": "pending",
            "doing": "in_progress", 
            "done": "completed",
            "blocked": "blocked"
        }
        
        # AI Agent assignments based on task type
        self.agent_assignments = {
            # Phase-level orchestration
            "phase": "Archon Meta-Agent",
            
            # Development tasks
            "fork_repo": "Claude Code IDE",
            "design_agents": "Archon System Architect",
            "implement_engine": "Python Backend Coder Agent",
            "create_prp": "Documentation Writer Agent", 
            "build_triggers": "Python Backend Coder Agent",
            "develop_ui": "TypeScript Frontend Agent",
            "setup_tests": "Test Framework Agent",
            "run_benchmark": "SCWT Benchmark Agent",
            
            # Architecture & Design
            "meta_agent": "Archon Meta-Agent",
            "validator": "External Validator Agent (DeepSeek)",
            "prompt_enhancer": "Prompt Enhancement Agent",
            "memory_service": "Memory Management Agent",
            "graphiti": "Graphiti Knowledge Graph Agent",
            "deepconf": "DeepConf Reasoning Agent",
            
            # Quality & Testing
            "test": "Unit Test Generator Agent",
            "security": "Security Auditor Agent",
            "performance": "Performance Optimizer Agent",
            "integration": "Integration Test Agent",
            
            # Documentation & Deployment
            "documentation": "Technical Writer Agent",
            "deployment": "DevOps Deployment Agent",
            "ui_polish": "UI/UX Designer Agent",
            
            # Default fallbacks
            "subtask": "Specialized Sub-Agent",
            "general": "Claude Code IDE"
        }
        
        # Phase task IDs (populated on initialization)
        self.phase_tasks = {}
        self._load_phase_tasks()
    
    def _load_phase_tasks(self):
        """Load phase task IDs from Archon system"""
        try:
            response = self.session.get(f"{self.api_base}/projects/{self.project_id}/tasks")
            if response.status_code == 200:
                tasks = response.json()
                for task in tasks:
                    title = task.get("title", "")
                    if "Phase" in title:
                        phase_num = self._extract_phase_number(title)
                        if phase_num:
                            self.phase_tasks[phase_num] = task["id"]
                            logger.info(f"Loaded Phase {phase_num} task ID: {task['id']}")
            else:
                logger.error(f"Failed to load tasks: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error loading phase tasks: {e}")
    
    def _extract_phase_number(self, title: str) -> Optional[int]:
        """Extract phase number from task title"""
        import re
        match = re.search(r'Phase (\d+)', title)
        return int(match.group(1)) if match else None
    
    def _determine_agent_assignment(self, task_title: str, task_description: str) -> str:
        """Determine the appropriate AI agent for a task"""
        title_lower = task_title.lower()
        desc_lower = task_description.lower()
        
        # Phase-level tasks
        if "phase" in title_lower and any(word in title_lower for word in [":", "fork", "meta-agent", "validator", "memory", "deepconf", "polish"]):
            return self.agent_assignments["phase"]
        
        # Specific task type matching
        if "fork" in title_lower and "repository" in title_lower:
            return self.agent_assignments["fork_repo"]
        elif "design" in title_lower and "agent" in title_lower:
            return self.agent_assignments["design_agents"] 
        elif "implement" in title_lower and "engine" in title_lower:
            return self.agent_assignments["implement_engine"]
        elif "prp" in title_lower or ("prompt" in title_lower and "system" in title_lower):
            return self.agent_assignments["create_prp"]
        elif "trigger" in title_lower:
            return self.agent_assignments["build_triggers"]
        elif "ui" in title_lower and ("dashboard" in title_lower or "develop" in title_lower):
            return self.agent_assignments["develop_ui"]
        elif "test" in title_lower and "environment" in title_lower:
            return self.agent_assignments["setup_tests"]
        elif "scwt" in title_lower and "benchmark" in title_lower:
            return self.agent_assignments["run_benchmark"]
        elif "meta-agent" in title_lower:
            return self.agent_assignments["meta_agent"]
        elif "validator" in title_lower:
            return self.agent_assignments["validator"]
        elif "prompt enhancer" in title_lower:
            return self.agent_assignments["prompt_enhancer"]
        elif "memory" in title_lower:
            return self.agent_assignments["memory_service"]
        elif "graphiti" in title_lower:
            return self.agent_assignments["graphiti"]
        elif "deepconf" in title_lower:
            return self.agent_assignments["deepconf"]
        elif "test" in title_lower:
            return self.agent_assignments["test"]
        elif "security" in title_lower:
            return self.agent_assignments["security"]
        elif "performance" in title_lower:
            return self.agent_assignments["performance"]
        elif "integration" in title_lower:
            return self.agent_assignments["integration"]
        elif "documentation" in title_lower or "docs" in title_lower:
            return self.agent_assignments["documentation"]
        elif "deployment" in title_lower or "deploy" in title_lower:
            return self.agent_assignments["deployment"]
        elif "polish" in title_lower and "ui" in title_lower:
            return self.agent_assignments["ui_polish"]
        elif "phase" in title_lower:
            return self.agent_assignments["subtask"]
        else:
            return self.agent_assignments["general"]
    
    def update_task_status(self, task_id: str, status: str, progress_notes: str = "") -> bool:
        """Update task status in Archon system"""
        try:
            update_data = {
                "status": status,
                "description": f"{progress_notes}\n\nLast updated: {datetime.now().isoformat()}" if progress_notes else None
            }
            
            # Remove None values
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            response = self.session.put(f"{self.api_base}/tasks/{task_id}", json=update_data)
            
            if response.status_code == 200:
                logger.info(f"✓ Updated task {task_id} to {status}")
                return True
            else:
                logger.error(f"Failed to update task {task_id}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            return False
    
    def start_phase(self, phase: int, notes: str = "") -> bool:
        """Start a specific phase and update task status"""
        task_id = self.phase_tasks.get(phase)
        if not task_id:
            logger.error(f"No task ID found for Phase {phase}")
            return False
        
        start_notes = f"PHASE {phase} STARTED\n\n{notes}\n\nStarted at: {datetime.now().isoformat()}"
        return self.update_task_status(task_id, "doing", start_notes)
    
    def complete_phase(self, phase: int, scwt_results: Dict[str, Any], notes: str = "") -> bool:
        """Complete a phase with SCWT benchmark results"""
        task_id = self.phase_tasks.get(phase)
        if not task_id:
            logger.error(f"No task ID found for Phase {phase}")
            return False
        
        # Format SCWT results for task description
        metrics_summary = self._format_scwt_metrics(scwt_results)
        gate_status = scwt_results.get("gate_status", {})
        gate_decision = gate_status.get("decision", "UNKNOWN")
        
        completion_notes = f"""PHASE {phase} COMPLETED

📊 SCWT BENCHMARK RESULTS:
{metrics_summary}

🚪 GATE DECISION: {gate_decision}
{self._format_gate_criteria(gate_status)}

📝 NOTES:
{notes}

✅ Completed at: {datetime.now().isoformat()}"""
        
        status = "done" if gate_decision == "PROCEED" else "blocked"
        return self.update_task_status(task_id, status, completion_notes)
    
    def update_phase_progress(self, phase: int, progress_percentage: float, current_task: str, notes: str = "") -> bool:
        """Update phase progress with current task and completion percentage"""
        task_id = self.phase_tasks.get(phase)
        if not task_id:
            logger.error(f"No task ID found for Phase {phase}")
            return False
        
        progress_notes = f"""PHASE {phase} IN PROGRESS

📈 Progress: {progress_percentage:.1f}% complete
🔨 Current Task: {current_task}

📝 Progress Notes:
{notes}

🕐 Last Updated: {datetime.now().isoformat()}"""
        
        return self.update_task_status(task_id, "doing", progress_notes)
    
    def block_phase(self, phase: int, blocking_issues: List[str], notes: str = "") -> bool:
        """Block a phase due to issues or failed gate criteria"""
        task_id = self.phase_tasks.get(phase)
        if not task_id:
            logger.error(f"No task ID found for Phase {phase}")
            return False
        
        issues_list = "\n".join([f"• {issue}" for issue in blocking_issues])
        blocking_notes = f"""PHASE {phase} BLOCKED

🚫 BLOCKING ISSUES:
{issues_list}

📝 NOTES:
{notes}

⏸️ Blocked at: {datetime.now().isoformat()}

ACTION REQUIRED: Resolve blocking issues before proceeding to next phase."""
        
        return self.update_task_status(task_id, "blocked", blocking_notes)
    
    def _format_scwt_metrics(self, scwt_results: Dict[str, Any]) -> str:
        """Format SCWT metrics for task description"""
        metrics = scwt_results.get("metrics", {})
        
        formatted_metrics = []
        for metric, value in metrics.items():
            if isinstance(value, float):
                if "rate" in metric or "efficiency" in metric or "reuse" in metric or "precision" in metric or "accuracy" in metric or "usability" in metric:
                    formatted_metrics.append(f"  • {metric.replace('_', ' ').title()}: {value:.1%}")
                else:
                    formatted_metrics.append(f"  • {metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                formatted_metrics.append(f"  • {metric.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted_metrics)
    
    def _format_gate_criteria(self, gate_status: Dict[str, Any]) -> str:
        """Format gate criteria evaluation"""
        criteria = gate_status.get("criteria", {})
        if not criteria:
            return ""
        
        formatted_criteria = []
        for metric, details in criteria.items():
            target = details.get("target", 0)
            actual = details.get("actual", 0)
            passed = details.get("passed", False)
            status_icon = "✅" if passed else "❌"
            
            if isinstance(target, float) and isinstance(actual, float):
                formatted_criteria.append(
                    f"  {status_icon} {metric.replace('_', ' ').title()}: {actual:.1%} (target: {target:.1%})"
                )
        
        return "\nGate Criteria:\n" + "\n".join(formatted_criteria) if formatted_criteria else ""
    
    def create_subtask(self, phase: int, subtask_title: str, subtask_description: str, priority: str = "medium") -> Optional[str]:
        """Create a subtask for a specific phase with proper AI agent assignment"""
        try:
            full_title = f"Phase {phase} - {subtask_title}"
            assigned_agent = self._determine_agent_assignment(full_title, subtask_description)
            
            subtask_data = {
                "title": full_title,
                "description": f"🤖 ASSIGNED TO: {assigned_agent}\n\n{subtask_description}\n\n⚙️ This task will be executed by the {assigned_agent}, not the user.",
                "project_id": self.project_id,
                "priority": priority,
                "status": "todo",
                "type": "subtask"
            }
            
            response = self.session.post(f"{self.api_base}/tasks", json=subtask_data)
            
            if response.status_code == 200:
                task_data = response.json()
                task_id = task_data["task"]["id"]
                logger.info(f"✓ Created subtask: {subtask_title} (ID: {task_id})")
                return task_id
            else:
                logger.error(f"Failed to create subtask: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating subtask: {e}")
            return None
    
    def get_project_status(self) -> Dict[str, Any]:
        """Get overall project status with phase progress"""
        try:
            response = self.session.get(f"{self.api_base}/projects/{self.project_id}/tasks")
            if response.status_code != 200:
                logger.error(f"Failed to get project status: {response.status_code}")
                return {}
            
            tasks = response.json()
            status_summary = {
                "total_phases": len(self.phase_tasks),
                "phases_completed": 0,
                "phases_in_progress": 0,
                "phases_blocked": 0,
                "phases_pending": 0,
                "current_phase": None,
                "last_updated": datetime.now().isoformat()
            }
            
            for task in tasks:
                if task["id"] in self.phase_tasks.values():
                    status = task["status"]
                    if status == "done":
                        status_summary["phases_completed"] += 1
                    elif status == "doing":
                        status_summary["phases_in_progress"] += 1
                        phase_num = self._extract_phase_number(task["title"])
                        status_summary["current_phase"] = phase_num
                    elif status == "blocked":
                        status_summary["phases_blocked"] += 1
                    else:
                        status_summary["phases_pending"] += 1
            
            return status_summary
            
        except Exception as e:
            logger.error(f"Error getting project status: {e}")
            return {}
    
    def update_all_task_assignments(self):
        """Update all existing tasks with proper AI agent assignments"""
        logger.info("Updating all task assignments to reflect AI agents...")
        
        try:
            response = self.session.get(f"{self.api_base}/projects/{self.project_id}/tasks")
            if response.status_code != 200:
                logger.error(f"Failed to get tasks: {response.status_code}")
                return False
            
            tasks = response.json()
            updated_count = 0
            
            for task in tasks:
                task_id = task["id"]
                title = task["title"]
                description = task["description"]
                
                # Determine the correct agent assignment
                assigned_agent = self._determine_agent_assignment(title, description)
                
                # Update task description with agent assignment
                updated_description = f"🤖 ASSIGNED TO: {assigned_agent}\n\n"
                
                # Keep existing description but remove old agent assignments
                existing_desc = description
                if "🤖 ASSIGNED TO:" in existing_desc:
                    # Remove existing assignment and keep the rest
                    parts = existing_desc.split("\n\n", 1)
                    if len(parts) > 1:
                        existing_desc = parts[1]
                
                if "⚙️ This task will be executed by" in existing_desc:
                    # Remove old execution note
                    parts = existing_desc.split("\n\n⚙️ This task will be executed by")[0]
                    existing_desc = parts
                
                updated_description += existing_desc
                updated_description += f"\n\n⚙️ This task will be executed by the {assigned_agent}, not the user."
                
                # Update the task
                update_data = {"description": updated_description}
                update_response = self.session.put(f"{self.api_base}/tasks/{task_id}", json=update_data)
                
                if update_response.status_code == 200:
                    updated_count += 1
                    logger.info(f"Updated task '{title}' -> {assigned_agent}")
                else:
                    logger.error(f"Failed to update task {task_id}: {update_response.text}")
            
            logger.info(f"Successfully updated {updated_count} tasks with AI agent assignments")
            return True
            
        except Exception as e:
            logger.error(f"Error updating task assignments: {e}")
            return False
    
    def initialize_phase_1(self):
        """Initialize Phase 1 with subtasks based on PRP requirements"""
        logger.info("Initializing Phase 1 with detailed subtasks...")
        
        phase_1_subtasks = [
            {
                "title": "Fork Archon Repository",
                "description": "Clone coleam00/Archon to VeloF2025/Archon repository with proper setup and configuration",
                "priority": "high"
            },
            {
                "title": "Design 20+ Sub-Agent Roles",
                "description": "Define JSON configurations for specialized agents: Python Backend Coder, TS Frontend Linter, Unit Test Generator, Security Auditor, Doc Writer, API Integrator, HRM Reasoning Agent, etc.",
                "priority": "high"
            },
            {
                "title": "Implement Parallel Execution Engine",
                "description": "Build agent pool management with conflict resolution using Redis locks or Git worktrees",
                "priority": "high"
            },
            {
                "title": "Create PRP-Based Prompt System",
                "description": "Develop role-specific PRP templates with examples, file paths, and test patterns",
                "priority": "medium"
            },
            {
                "title": "Build Proactive Trigger System",
                "description": "Implement file watcher with pattern matching for auto-invoking agents (e.g., Security Auditor on code changes)",
                "priority": "medium"
            },
            {
                "title": "Develop UI Agent Dashboard", 
                "description": "Create real-time agent monitoring interface showing roles, statuses, PIDs, and controls",
                "priority": "high"
            },
            {
                "title": "Setup SCWT Test Environment",
                "description": "Prepare benchmark testing repository and framework for Phase 1 validation",
                "priority": "medium"
            },
            {
                "title": "Run Phase 1 SCWT Benchmark",
                "description": "Execute SCWT test and validate against Phase 1 gate criteria (≥10% efficiency gain, ≥85% precision)",
                "priority": "high"
            }
        ]
        
        for subtask in phase_1_subtasks:
            self.create_subtask(1, subtask["title"], subtask["description"], subtask["priority"])
        
        # Start Phase 1
        self.start_phase(1, "Phase 1 initialization complete with 8 detailed subtasks created. Ready to begin implementation.")


def main():
    """Initialize dynamic task tracking system"""
    tracker = ArchonTaskTracker()
    
    # Get current project status
    status = tracker.get_project_status()
    logger.info(f"Current project status: {status}")
    
    # Initialize Phase 1 if no phases are in progress
    if status.get("phases_in_progress", 0) == 0 and status.get("phases_completed", 0) == 0:
        tracker.initialize_phase_1()
        logger.info("✓ Phase 1 initialized with subtasks and started")
    
    # Example of updating phase progress
    # tracker.update_phase_progress(1, 25.0, "Implementing agent configurations", "Created 12 of 20 agent role definitions")
    
    # Example of completing a phase with SCWT results
    # sample_scwt_results = {
    #     "metrics": {"task_efficiency_time": 0.16, "communication_efficiency": 0.12, "precision": 0.87},
    #     "gate_status": {"decision": "PROCEED", "criteria": {"precision": {"target": 0.85, "actual": 0.87, "passed": True}}}
    # }
    # tracker.complete_phase(1, sample_scwt_results, "Phase 1 implementation successful")
    
    return tracker


if __name__ == "__main__":
    tracker = main()
    print("DYNAMIC TASK TRACKING SYSTEM INITIALIZED AND READY")