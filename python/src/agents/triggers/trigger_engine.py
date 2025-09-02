#!/usr/bin/env python3
"""
Trigger Engine for Archon+ Proactive Agent System
Advanced trigger system with pattern matching, conditions, and agent orchestration
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Union
import re

from .file_watcher import ProactiveTriggerManager, TriggerEvent, TriggerEventType

logger = logging.getLogger(__name__)

class TriggerConditionType(Enum):
    FILE_SIZE = "file_size"
    FILE_AGE = "file_age"
    FILE_CONTENT_MATCH = "file_content_match"
    DEPENDENCY_CHANGE = "dependency_change"
    TIME_BASED = "time_based"
    BATCH_SIZE = "batch_size"
    CUSTOM_CONDITION = "custom_condition"

@dataclass
class TriggerCondition:
    """Advanced condition for trigger evaluation"""
    condition_type: TriggerConditionType
    parameters: Dict[str, Any]
    description: str = ""
    
class TriggerResult(Enum):
    TRIGGERED = "triggered"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    FAILED = "failed"

@dataclass
class TriggerExecution:
    """Result of a trigger execution"""
    trigger_id: str
    agent_role: str
    result: TriggerResult
    timestamp: float
    execution_time: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class AdvancedTriggerRule:
    """Advanced trigger rule with conditions and constraints"""
    rule_id: str
    agent_role: str
    patterns: List[str]
    event_types: List[TriggerEventType]
    conditions: List[TriggerCondition] = field(default_factory=list)
    priority: int = 1
    cooldown_seconds: int = 30
    batch_size: int = 1  # Number of events to batch before triggering
    max_concurrent: int = 1  # Max concurrent executions
    timeout_seconds: int = 300  # 5 minutes
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class ArchonTriggerEngine:
    """
    Advanced trigger engine for Archon+ agent system
    Provides sophisticated triggering with conditions, batching, and orchestration
    """
    
    def __init__(self, 
                 config_path: str = "python/src/agents/configs",
                 watch_paths: List[str] = None,
                 orchestrator_callback: Optional[Callable] = None):
        
        self.config_path = Path(config_path)
        self.watch_paths = watch_paths or ["."]
        self.orchestrator_callback = orchestrator_callback
        
        # Core components
        self.trigger_manager = ProactiveTriggerManager(
            watch_paths=self.watch_paths,
            config_path=str(self.config_path),
            agent_callback=self._handle_agent_trigger
        )
        
        # Advanced trigger rules
        self.advanced_rules: Dict[str, AdvancedTriggerRule] = {}
        self.rule_execution_counts: Dict[str, int] = {}
        self.rule_last_execution: Dict[str, float] = {}
        
        # Batching and queuing
        self.event_batches: Dict[str, List[TriggerEvent]] = {}
        self.batch_timers: Dict[str, asyncio.Task] = {}
        self.pending_executions: Dict[str, Set[str]] = {}  # agent -> set of rule_ids
        
        # Execution tracking
        self.execution_history: List[TriggerExecution] = []
        self.active_executions: Dict[str, TriggerExecution] = {}
        
        # Load advanced rules
        self._load_advanced_rules()
        
        logger.info(f"ArchonTriggerEngine initialized with {len(self.advanced_rules)} advanced rules")
    
    def _load_advanced_rules(self):
        """Load advanced trigger rules from configuration"""
        # Create some default advanced rules based on agent configs
        self._create_default_advanced_rules()
        
        # Load custom advanced rules if they exist
        advanced_rules_file = self.config_path / "advanced_trigger_rules.json"
        if advanced_rules_file.exists():
            try:
                with open(advanced_rules_file, 'r') as f:
                    rules_config = json.load(f)
                
                for rule_config in rules_config.get("rules", []):
                    rule = self._parse_advanced_rule(rule_config)
                    if rule:
                        self.advanced_rules[rule.rule_id] = rule
                
                logger.info(f"Loaded {len(rules_config.get('rules', []))} custom advanced rules")
                
            except Exception as e:
                logger.error(f"Failed to load advanced rules: {e}")
    
    def _create_default_advanced_rules(self):
        """Create default advanced rules from basic agent configs"""
        # Security-focused rules
        security_rule = AdvancedTriggerRule(
            rule_id="security_critical_files",
            agent_role="security_auditor",
            patterns=["*.py", "*.js", "*.ts", "requirements.txt", "package.json", ".env*"],
            event_types=[TriggerEventType.FILE_MODIFIED, TriggerEventType.FILE_CREATED],
            conditions=[
                TriggerCondition(
                    condition_type=TriggerConditionType.FILE_CONTENT_MATCH,
                    parameters={"patterns": ["password", "secret", "token", "api_key"]},
                    description="Trigger on potential security-sensitive content"
                )
            ],
            priority=1,
            cooldown_seconds=60,
            batch_size=3,  # Batch security events
            metadata={"category": "security", "criticality": "high"}
        )
        self.advanced_rules[security_rule.rule_id] = security_rule
        
        # Test generation rule
        test_rule = AdvancedTriggerRule(
            rule_id="auto_test_generation",
            agent_role="test_generator",
            patterns=["*.py", "*.js", "*.ts"],
            event_types=[TriggerEventType.FILE_CREATED, TriggerEventType.FILE_MODIFIED],
            conditions=[
                TriggerCondition(
                    condition_type=TriggerConditionType.FILE_SIZE,
                    parameters={"min_size": 100, "max_size": 50000},  # 100 bytes to 50KB
                    description="Only trigger for reasonably sized files"
                ),
                TriggerCondition(
                    condition_type=TriggerConditionType.CUSTOM_CONDITION,
                    parameters={"function": "exclude_test_files"},
                    description="Don't generate tests for test files"
                )
            ],
            priority=2,
            cooldown_seconds=120,
            batch_size=5,  # Batch test generation
            metadata={"category": "testing", "auto_generated": True}
        )
        self.advanced_rules[test_rule.rule_id] = test_rule
        
        # Documentation rule
        doc_rule = AdvancedTriggerRule(
            rule_id="auto_documentation",
            agent_role="documentation_writer",
            patterns=["*.py", "*.js", "*.ts", "README.md", "*.md"],
            event_types=[TriggerEventType.FILE_MODIFIED],
            conditions=[
                TriggerCondition(
                    condition_type=TriggerConditionType.TIME_BASED,
                    parameters={"delay_minutes": 5},  # Wait 5 minutes before generating docs
                    description="Delay documentation to avoid frequent updates"
                )
            ],
            priority=3,
            cooldown_seconds=300,  # 5 minutes
            batch_size=10,  # Batch documentation updates
            metadata={"category": "documentation", "delay_enabled": True}
        )
        self.advanced_rules[doc_rule.rule_id] = doc_rule
    
    def _parse_advanced_rule(self, rule_config: Dict[str, Any]) -> Optional[AdvancedTriggerRule]:
        """Parse advanced rule from JSON configuration"""
        try:
            conditions = []
            for cond_config in rule_config.get("conditions", []):
                condition = TriggerCondition(
                    condition_type=TriggerConditionType(cond_config["type"]),
                    parameters=cond_config["parameters"],
                    description=cond_config.get("description", "")
                )
                conditions.append(condition)
            
            rule = AdvancedTriggerRule(
                rule_id=rule_config["rule_id"],
                agent_role=rule_config["agent_role"],
                patterns=rule_config["patterns"],
                event_types=[TriggerEventType(t) for t in rule_config["event_types"]],
                conditions=conditions,
                priority=rule_config.get("priority", 2),
                cooldown_seconds=rule_config.get("cooldown_seconds", 30),
                batch_size=rule_config.get("batch_size", 1),
                max_concurrent=rule_config.get("max_concurrent", 1),
                timeout_seconds=rule_config.get("timeout_seconds", 300),
                enabled=rule_config.get("enabled", True),
                metadata=rule_config.get("metadata", {})
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"Failed to parse advanced rule: {e}")
            return None
    
    async def _handle_agent_trigger(self, agent_role: str, context: Dict[str, Any]):
        """Handle agent trigger from basic trigger manager"""
        # Find applicable advanced rules
        applicable_rules = [
            rule for rule in self.advanced_rules.values()
            if rule.agent_role == agent_role and rule.enabled
        ]
        
        if not applicable_rules:
            # Fallback to basic triggering
            await self._execute_basic_trigger(agent_role, context)
            return
        
        # Process with advanced rules
        trigger_event = self._create_trigger_event_from_context(context)
        
        for rule in applicable_rules:
            if await self._should_trigger_rule(rule, trigger_event):
                await self._process_advanced_rule(rule, trigger_event)
    
    def _create_trigger_event_from_context(self, context: Dict[str, Any]) -> TriggerEvent:
        """Create trigger event from agent callback context"""
        trigger_info = context.get("trigger_event", {})
        return TriggerEvent(
            event_id=f"advanced_{int(time.time() * 1000)}",
            event_type=TriggerEventType(trigger_info.get("type", "file_modified")),
            file_path=trigger_info.get("file_path", ""),
            timestamp=trigger_info.get("timestamp", time.time()),
            metadata=trigger_info.get("metadata", {})
        )
    
    async def _should_trigger_rule(self, rule: AdvancedTriggerRule, event: TriggerEvent) -> bool:
        """Evaluate if rule should be triggered based on conditions"""
        try:
            # Check basic pattern matching
            if not self._matches_patterns(event.file_path, rule.patterns):
                return False
            
            # Check event type
            if event.event_type not in rule.event_types:
                return False
            
            # Check cooldown
            if self._is_rule_on_cooldown(rule.rule_id):
                return False
            
            # Check concurrent execution limit
            if self._exceeds_concurrent_limit(rule):
                return False
            
            # Evaluate conditions
            for condition in rule.conditions:
                if not await self._evaluate_condition(condition, event):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return False
    
    def _matches_patterns(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file path matches any pattern"""
        import fnmatch
        
        file_name = Path(file_path).name
        file_path_normalized = file_path.replace("\\", "/")
        
        for pattern in patterns:
            if (fnmatch.fnmatch(file_name, pattern) or 
                fnmatch.fnmatch(file_path_normalized, pattern) or
                fnmatch.fnmatch(file_path_normalized, f"*/{pattern}")):
                return True
        
        return False
    
    def _is_rule_on_cooldown(self, rule_id: str) -> bool:
        """Check if rule is on cooldown"""
        if rule_id not in self.rule_last_execution:
            return False
        
        rule = self.advanced_rules[rule_id]
        last_execution = self.rule_last_execution[rule_id]
        
        return (time.time() - last_execution) < rule.cooldown_seconds
    
    def _exceeds_concurrent_limit(self, rule: AdvancedTriggerRule) -> bool:
        """Check if rule exceeds concurrent execution limit"""
        agent_pending = self.pending_executions.get(rule.agent_role, set())
        return len(agent_pending) >= rule.max_concurrent
    
    async def _evaluate_condition(self, condition: TriggerCondition, event: TriggerEvent) -> bool:
        """Evaluate a specific trigger condition"""
        try:
            if condition.condition_type == TriggerConditionType.FILE_SIZE:
                return await self._check_file_size_condition(condition, event)
            
            elif condition.condition_type == TriggerConditionType.FILE_CONTENT_MATCH:
                return await self._check_content_match_condition(condition, event)
            
            elif condition.condition_type == TriggerConditionType.FILE_AGE:
                return await self._check_file_age_condition(condition, event)
            
            elif condition.condition_type == TriggerConditionType.TIME_BASED:
                return await self._check_time_based_condition(condition, event)
            
            elif condition.condition_type == TriggerConditionType.CUSTOM_CONDITION:
                return await self._check_custom_condition(condition, event)
            
            else:
                logger.warning(f"Unknown condition type: {condition.condition_type}")
                return True  # Default to allow
                
        except Exception as e:
            logger.error(f"Error evaluating condition {condition.condition_type}: {e}")
            return True  # Default to allow on error
    
    async def _check_file_size_condition(self, condition: TriggerCondition, event: TriggerEvent) -> bool:
        """Check file size condition"""
        try:
            file_path = Path(event.file_path)
            if not file_path.exists():
                return False
            
            file_size = file_path.stat().st_size
            min_size = condition.parameters.get("min_size", 0)
            max_size = condition.parameters.get("max_size", float('inf'))
            
            return min_size <= file_size <= max_size
            
        except Exception:
            return True  # Default to allow if we can't check
    
    async def _check_content_match_condition(self, condition: TriggerCondition, event: TriggerEvent) -> bool:
        """Check file content matching condition"""
        try:
            file_path = Path(event.file_path)
            if not file_path.exists() or file_path.is_dir():
                return False
            
            # Read file content (limit to prevent huge files)
            max_read_size = condition.parameters.get("max_read_size", 10000)  # 10KB
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(max_read_size).lower()
            except (UnicodeDecodeError, PermissionError):
                return False
            
            patterns = condition.parameters.get("patterns", [])
            for pattern in patterns:
                if pattern.lower() in content:
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _check_file_age_condition(self, condition: TriggerCondition, event: TriggerEvent) -> bool:
        """Check file age condition"""
        try:
            file_path = Path(event.file_path)
            if not file_path.exists():
                return False
            
            file_mtime = file_path.stat().st_mtime
            file_age_minutes = (time.time() - file_mtime) / 60
            
            min_age = condition.parameters.get("min_age_minutes", 0)
            max_age = condition.parameters.get("max_age_minutes", float('inf'))
            
            return min_age <= file_age_minutes <= max_age
            
        except Exception:
            return True
    
    async def _check_time_based_condition(self, condition: TriggerCondition, event: TriggerEvent) -> bool:
        """Check time-based condition (delays)"""
        delay_minutes = condition.parameters.get("delay_minutes", 0)
        if delay_minutes <= 0:
            return True
        
        # For time-based conditions, we defer the execution
        # This is a simplified implementation - in practice would use a scheduler
        event_age_minutes = (time.time() - event.timestamp) / 60
        return event_age_minutes >= delay_minutes
    
    async def _check_custom_condition(self, condition: TriggerCondition, event: TriggerEvent) -> bool:
        """Check custom condition function"""
        function_name = condition.parameters.get("function")
        
        if function_name == "exclude_test_files":
            # Don't trigger on test files
            file_path = event.file_path.lower()
            return not (
                "test_" in file_path or 
                "_test" in file_path or
                "/tests/" in file_path or
                ".test." in file_path or
                ".spec." in file_path
            )
        
        # Add more custom conditions as needed
        return True
    
    async def _process_advanced_rule(self, rule: AdvancedTriggerRule, event: TriggerEvent):
        """Process advanced rule with batching and orchestration"""
        try:
            # Handle batching
            if rule.batch_size > 1:
                await self._handle_batch_processing(rule, event)
            else:
                # Execute immediately
                await self._execute_rule(rule, [event])
                
        except Exception as e:
            logger.error(f"Error processing advanced rule {rule.rule_id}: {e}")
    
    async def _handle_batch_processing(self, rule: AdvancedTriggerRule, event: TriggerEvent):
        """Handle batched event processing"""
        batch_key = f"{rule.rule_id}_{rule.agent_role}"
        
        # Add event to batch
        if batch_key not in self.event_batches:
            self.event_batches[batch_key] = []
        
        self.event_batches[batch_key].append(event)
        
        # Check if batch is full
        if len(self.event_batches[batch_key]) >= rule.batch_size:
            # Execute batch immediately
            events = self.event_batches[batch_key].copy()
            self.event_batches[batch_key] = []
            
            # Cancel any pending timer
            if batch_key in self.batch_timers:
                self.batch_timers[batch_key].cancel()
                del self.batch_timers[batch_key]
            
            await self._execute_rule(rule, events)
        
        else:
            # Set timer for batch timeout (30 seconds)
            if batch_key not in self.batch_timers:
                self.batch_timers[batch_key] = asyncio.create_task(
                    self._batch_timeout(rule, batch_key, 30)
                )
    
    async def _batch_timeout(self, rule: AdvancedTriggerRule, batch_key: str, timeout_seconds: int):
        """Handle batch timeout"""
        await asyncio.sleep(timeout_seconds)
        
        if batch_key in self.event_batches and self.event_batches[batch_key]:
            # Execute partial batch
            events = self.event_batches[batch_key].copy()
            self.event_batches[batch_key] = []
            
            logger.info(f"Executing partial batch for {rule.rule_id}: {len(events)} events")
            await self._execute_rule(rule, events)
        
        # Clean up timer
        if batch_key in self.batch_timers:
            del self.batch_timers[batch_key]
    
    async def _execute_rule(self, rule: AdvancedTriggerRule, events: List[TriggerEvent]):
        """Execute advanced rule with events"""
        execution_id = f"{rule.rule_id}_{int(time.time() * 1000)}"
        
        execution = TriggerExecution(
            trigger_id=execution_id,
            agent_role=rule.agent_role,
            result=TriggerResult.TRIGGERED,
            timestamp=time.time(),
            context={
                "rule_id": rule.rule_id,
                "events": [
                    {
                        "file_path": e.file_path,
                        "event_type": e.event_type.value,
                        "timestamp": e.timestamp
                    }
                    for e in events
                ],
                "batch_size": len(events)
            }
        )
        
        try:
            # Track execution
            self.active_executions[execution_id] = execution
            
            # Add to pending for concurrency control
            if rule.agent_role not in self.pending_executions:
                self.pending_executions[rule.agent_role] = set()
            self.pending_executions[rule.agent_role].add(rule.rule_id)
            
            start_time = time.time()
            
            # Execute via orchestrator callback
            if self.orchestrator_callback:
                context = {
                    "rule": rule,
                    "events": events,
                    "execution_id": execution_id,
                    "batch_size": len(events),
                    "file_paths": [e.file_path for e in events]
                }
                
                await asyncio.wait_for(
                    self.orchestrator_callback(rule.agent_role, context),
                    timeout=rule.timeout_seconds
                )
            
            # Mark as completed
            execution.execution_time = time.time() - start_time
            execution.result = TriggerResult.TRIGGERED
            
            # Update statistics
            self.rule_execution_counts[rule.rule_id] = self.rule_execution_counts.get(rule.rule_id, 0) + 1
            self.rule_last_execution[rule.rule_id] = time.time()
            
            logger.info(f"Executed rule {rule.rule_id} for {rule.agent_role} with {len(events)} events")
            
        except asyncio.TimeoutError:
            execution.result = TriggerResult.FAILED
            execution.error_message = "Execution timeout"
            logger.error(f"Rule execution timeout: {rule.rule_id}")
            
        except Exception as e:
            execution.result = TriggerResult.FAILED
            execution.error_message = str(e)
            logger.error(f"Rule execution failed: {rule.rule_id}: {e}")
        
        finally:
            # Clean up tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            if rule.agent_role in self.pending_executions:
                self.pending_executions[rule.agent_role].discard(rule.rule_id)
            
            # Store in history
            self.execution_history.append(execution)
            
            # Keep history limited
            if len(self.execution_history) > 1000:
                self.execution_history.pop(0)
    
    async def _execute_basic_trigger(self, agent_role: str, context: Dict[str, Any]):
        """Execute basic trigger without advanced rules"""
        logger.info(f"Basic trigger: {agent_role} - {context.get('trigger_reason', 'Unknown')}")
        
        if self.orchestrator_callback:
            await self.orchestrator_callback(agent_role, context)
    
    def start(self):
        """Start the trigger engine"""
        self.trigger_manager.start_watching()
        logger.info("ArchonTriggerEngine started")
    
    def stop(self):
        """Stop the trigger engine"""
        self.trigger_manager.stop_watching()
        
        # Cancel all batch timers
        for timer in self.batch_timers.values():
            timer.cancel()
        self.batch_timers.clear()
        
        logger.info("ArchonTriggerEngine stopped")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive trigger engine status"""
        basic_stats = self.trigger_manager.get_trigger_statistics()
        
        return {
            "basic_triggers": basic_stats,
            "advanced_rules": {
                "total_rules": len(self.advanced_rules),
                "enabled_rules": len([r for r in self.advanced_rules.values() if r.enabled]),
                "execution_counts": self.rule_execution_counts,
                "active_executions": len(self.active_executions),
                "pending_batches": {k: len(v) for k, v in self.event_batches.items()},
                "batch_timers": len(self.batch_timers)
            },
            "execution_history": {
                "total_executions": len(self.execution_history),
                "recent_executions": len([e for e in self.execution_history if time.time() - e.timestamp < 3600]),
                "success_rate": len([e for e in self.execution_history if e.result == TriggerResult.TRIGGERED]) / len(self.execution_history) if self.execution_history else 0
            }
        }
    
    def get_rule_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each rule"""
        performance = {}
        
        for rule_id, rule in self.advanced_rules.items():
            executions = [e for e in self.execution_history if e.context.get("rule_id") == rule_id]
            
            if executions:
                success_count = len([e for e in executions if e.result == TriggerResult.TRIGGERED])
                avg_execution_time = sum(e.execution_time for e in executions) / len(executions)
                
                performance[rule_id] = {
                    "total_executions": len(executions),
                    "success_count": success_count,
                    "success_rate": success_count / len(executions),
                    "avg_execution_time": avg_execution_time,
                    "last_execution": max(e.timestamp for e in executions),
                    "agent_role": rule.agent_role,
                    "priority": rule.priority
                }
            else:
                performance[rule_id] = {
                    "total_executions": 0,
                    "success_count": 0,
                    "success_rate": 0,
                    "avg_execution_time": 0,
                    "last_execution": None,
                    "agent_role": rule.agent_role,
                    "priority": rule.priority
                }
        
        return performance

# Example usage
if __name__ == "__main__":
    async def example_orchestrator_callback(agent_role: str, context: Dict[str, Any]):
        """Example orchestrator callback"""
        rule = context.get("rule")
        events = context.get("events", [])
        
        print(f"🚀 ORCHESTRATOR TRIGGER: {agent_role}")
        if rule:
            print(f"   Rule: {rule.rule_id} (Priority: {rule.priority})")
            print(f"   Batch: {len(events)} events")
        
        for event in events[:3]:  # Show first 3 events
            print(f"   • {event.event_type.value}: {event.file_path}")
        
        if len(events) > 3:
            print(f"   • ... and {len(events) - 3} more")
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        print(f"   ✅ Completed {agent_role}")
        print()
    
    async def main():
        # Initialize trigger engine
        engine = ArchonTriggerEngine(
            orchestrator_callback=example_orchestrator_callback
        )
        
        # Start engine
        engine.start()
        
        print("Advanced Trigger Engine started!")
        print("Monitoring for file changes with advanced rules...")
        print("Status:", engine.get_engine_status())
        print("\nPress Ctrl+C to stop")
        
        try:
            while True:
                await asyncio.sleep(5)
                
                # Show periodic status
                status = engine.get_engine_status()
                if status["basic_triggers"]["last_activity"]:
                    print(f"Activity: {status['advanced_rules']['active_executions']} active, "
                          f"{sum(status['advanced_rules']['pending_batches'].values())} batched")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            engine.stop()
    
    asyncio.run(main())