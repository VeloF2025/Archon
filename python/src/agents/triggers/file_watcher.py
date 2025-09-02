#!/usr/bin/env python3
"""
File Watcher for Archon+ Proactive Trigger System
Monitors file system changes and triggers appropriate agents
"""

import asyncio
import fnmatch
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

logger = logging.getLogger(__name__)

class TriggerEventType(Enum):
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    DIRECTORY_CREATED = "directory_created"
    DIRECTORY_DELETED = "directory_deleted"

@dataclass
class TriggerEvent:
    """Represents a trigger event that should invoke agents"""
    event_id: str
    event_type: TriggerEventType
    file_path: str
    timestamp: float
    matched_patterns: List[str] = field(default_factory=list)
    triggered_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentTriggerRule:
    """Rule for triggering agents based on file patterns"""
    agent_role: str
    patterns: List[str]
    event_types: List[TriggerEventType]
    cooldown_seconds: int = 30  # Prevent spam triggering
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=high, 2=medium, 3=low

class ArchonFileWatcher(FileSystemEventHandler):
    """
    File system event handler for Archon+ proactive triggers
    Monitors file changes and triggers appropriate agents
    """
    
    def __init__(self, trigger_manager: 'ProactiveTriggerManager'):
        super().__init__()
        self.trigger_manager = trigger_manager
        self.logger = logging.getLogger(f"{__name__}.ArchonFileWatcher")
    
    def on_created(self, event: FileSystemEvent):
        """Handle file/directory creation events"""
        if event.is_directory:
            self._handle_event(event.src_path, TriggerEventType.DIRECTORY_CREATED, event)
        else:
            self._handle_event(event.src_path, TriggerEventType.FILE_CREATED, event)
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events"""
        if not event.is_directory:
            self._handle_event(event.src_path, TriggerEventType.FILE_MODIFIED, event)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file/directory deletion events"""
        if event.is_directory:
            self._handle_event(event.src_path, TriggerEventType.DIRECTORY_DELETED, event)
        else:
            self._handle_event(event.src_path, TriggerEventType.FILE_DELETED, event)
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move events"""
        if not event.is_directory:
            self._handle_event(event.dest_path, TriggerEventType.FILE_MOVED, event, 
                             metadata={"source_path": event.src_path})
    
    def _handle_event(self, file_path: str, event_type: TriggerEventType, 
                     original_event: FileSystemEvent, metadata: Dict[str, Any] = None):
        """Process file system event and check for agent triggers"""
        try:
            # Create trigger event
            trigger_event = TriggerEvent(
                event_id=f"{event_type.value}_{int(time.time() * 1000)}",
                event_type=event_type,
                file_path=file_path,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # Let trigger manager handle the event
            asyncio.create_task(self.trigger_manager.process_trigger_event(trigger_event))
            
        except Exception as e:
            self.logger.error(f"Error handling file event {file_path}: {e}")

class ProactiveTriggerManager:
    """
    Manages proactive triggers for Archon+ agents
    Watches file system and invokes appropriate agents based on patterns
    """
    
    def __init__(self, 
                 watch_paths: List[str] = None,
                 config_path: str = "python/src/agents/configs",
                 agent_callback: Optional[Callable] = None):
        
        self.watch_paths = watch_paths or ["."]
        self.config_path = Path(config_path)
        self.agent_callback = agent_callback
        
        # File watching
        self.observer: Optional[Observer] = None
        self.file_watcher = ArchonFileWatcher(self)
        self.is_watching = False
        
        # Trigger rules and state
        self.trigger_rules: Dict[str, AgentTriggerRule] = {}
        self.recent_events: List[TriggerEvent] = []
        self.agent_cooldowns: Dict[str, float] = {}
        self.ignored_patterns = [
            "*.pyc", "__pycache__/*", ".git/*", ".venv/*", "node_modules/*",
            "*.log", ".DS_Store", "Thumbs.db", "*.tmp", "*.swp"
        ]
        
        # Statistics
        self.stats = {
            "events_processed": 0,
            "agents_triggered": 0,
            "triggers_blocked_by_cooldown": 0,
            "last_activity": None
        }
        
        # Load trigger rules from agent configs
        self._load_trigger_rules()
        
        logger.info(f"ProactiveTriggerManager initialized with {len(self.trigger_rules)} rules")
    
    def _load_trigger_rules(self):
        """Load trigger rules from agent configurations"""
        if not self.config_path.exists():
            logger.error(f"Config path does not exist: {self.config_path}")
            return
        
        config_files = list(self.config_path.glob("*.json"))
        config_files = [f for f in config_files if f.name not in ["agent_registry.json", "template_registry.json"]]
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                agent_role = config['role']
                proactive_triggers = config.get('proactive_triggers', [])
                
                if proactive_triggers:
                    # Convert to trigger rule
                    trigger_rule = AgentTriggerRule(
                        agent_role=agent_role,
                        patterns=proactive_triggers,
                        event_types=[TriggerEventType.FILE_CREATED, TriggerEventType.FILE_MODIFIED],
                        cooldown_seconds=self._get_agent_cooldown(config),
                        priority=self._get_agent_priority(config)
                    )
                    
                    self.trigger_rules[agent_role] = trigger_rule
                    logger.debug(f"Loaded trigger rule for {agent_role}: {len(proactive_triggers)} patterns")
                
            except Exception as e:
                logger.error(f"Failed to load trigger rules from {config_file}: {e}")
        
        logger.info(f"Loaded trigger rules for {len(self.trigger_rules)} agents")
    
    def _get_agent_cooldown(self, config: Dict[str, Any]) -> int:
        """Extract cooldown setting from agent config"""
        execution_context = config.get('execution_context', {})
        return execution_context.get('trigger_cooldown', 30)
    
    def _get_agent_priority(self, config: Dict[str, Any]) -> int:
        """Extract priority from agent config"""
        priority_map = {"critical": 1, "high": 1, "medium": 2, "low": 3}
        return priority_map.get(config.get('priority', 'medium'), 2)
    
    def start_watching(self):
        """Start file system monitoring"""
        if self.is_watching:
            logger.warning("Already watching file system")
            return
        
        try:
            self.observer = Observer()
            
            # Add watch paths
            for watch_path in self.watch_paths:
                path = Path(watch_path).resolve()
                if path.exists():
                    self.observer.schedule(self.file_watcher, str(path), recursive=True)
                    logger.info(f"Watching path: {path}")
                else:
                    logger.warning(f"Watch path does not exist: {path}")
            
            self.observer.start()
            self.is_watching = True
            logger.info("File system monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start file system monitoring: {e}")
            self.is_watching = False
    
    def stop_watching(self):
        """Stop file system monitoring"""
        if not self.is_watching or not self.observer:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.is_watching = False
            logger.info("File system monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping file system monitoring: {e}")
    
    def should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns"""
        file_name = Path(file_path).name
        file_path_normalized = file_path.replace("\\", "/")
        
        for pattern in self.ignored_patterns:
            if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_path_normalized, pattern):
                return True
        
        return False
    
    def match_trigger_patterns(self, file_path: str, event_type: TriggerEventType) -> List[str]:
        """Find agents that should be triggered by this file event"""
        if self.should_ignore_file(file_path):
            return []
        
        matched_agents = []
        file_name = Path(file_path).name
        file_path_normalized = file_path.replace("\\", "/")
        
        for agent_role, rule in self.trigger_rules.items():
            # Check if event type matches
            if event_type not in rule.event_types:
                continue
            
            # Check if any pattern matches
            for pattern in rule.patterns:
                if (fnmatch.fnmatch(file_name, pattern) or 
                    fnmatch.fnmatch(file_path_normalized, pattern) or
                    fnmatch.fnmatch(file_path_normalized, f"*/{pattern}")):
                    matched_agents.append(agent_role)
                    break
        
        return matched_agents
    
    def is_agent_on_cooldown(self, agent_role: str) -> bool:
        """Check if agent is on cooldown"""
        if agent_role not in self.agent_cooldowns:
            return False
        
        rule = self.trigger_rules.get(agent_role)
        if not rule:
            return False
        
        last_triggered = self.agent_cooldowns[agent_role]
        cooldown_period = rule.cooldown_seconds
        
        return (time.time() - last_triggered) < cooldown_period
    
    async def process_trigger_event(self, event: TriggerEvent):
        """Process a trigger event and invoke appropriate agents"""
        try:
            self.stats["events_processed"] += 1
            self.stats["last_activity"] = time.time()
            
            # Find matching agents
            matched_agents = self.match_trigger_patterns(event.file_path, event.event_type)
            event.triggered_agents = matched_agents
            
            if not matched_agents:
                logger.debug(f"No agents matched for {event.file_path}")
                return
            
            # Check cooldowns and trigger agents
            for agent_role in matched_agents:
                if self.is_agent_on_cooldown(agent_role):
                    logger.debug(f"Agent {agent_role} on cooldown, skipping trigger")
                    self.stats["triggers_blocked_by_cooldown"] += 1
                    continue
                
                # Trigger agent
                await self._trigger_agent(agent_role, event)
                
                # Update cooldown
                self.agent_cooldowns[agent_role] = time.time()
                self.stats["agents_triggered"] += 1
            
            # Store recent event
            self.recent_events.append(event)
            
            # Keep only last 100 events
            if len(self.recent_events) > 100:
                self.recent_events.pop(0)
            
            logger.info(f"Processed trigger event: {event.event_type.value} {event.file_path} -> {matched_agents}")
            
        except Exception as e:
            logger.error(f"Error processing trigger event: {e}")
    
    async def _trigger_agent(self, agent_role: str, event: TriggerEvent):
        """Trigger a specific agent with context from the event"""
        try:
            # Prepare agent context
            context = {
                "trigger_event": {
                    "type": event.event_type.value,
                    "file_path": event.file_path,
                    "timestamp": event.timestamp,
                    "metadata": event.metadata
                },
                "project_files": [event.file_path],
                "trigger_reason": f"File {event.event_type.value}: {event.file_path}"
            }
            
            # Call agent callback if provided
            if self.agent_callback:
                await self.agent_callback(agent_role, context)
            else:
                logger.info(f"Would trigger agent {agent_role} for {event.file_path}")
                # In real implementation, this would invoke the actual agent
            
        except Exception as e:
            logger.error(f"Failed to trigger agent {agent_role}: {e}")
    
    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get trigger system statistics"""
        stats = self.stats.copy()
        stats.update({
            "active_rules": len(self.trigger_rules),
            "watching_paths": len(self.watch_paths),
            "is_watching": self.is_watching,
            "recent_events_count": len(self.recent_events),
            "agents_on_cooldown": len([
                agent for agent, last_triggered in self.agent_cooldowns.items()
                if self.is_agent_on_cooldown(agent)
            ])
        })
        return stats
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trigger events"""
        events = self.recent_events[-limit:] if limit else self.recent_events
        return [
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "file_path": event.file_path,
                "timestamp": event.timestamp,
                "triggered_agents": event.triggered_agents
            }
            for event in events
        ]
    
    def add_trigger_rule(self, rule: AgentTriggerRule):
        """Add a new trigger rule"""
        self.trigger_rules[rule.agent_role] = rule
        logger.info(f"Added trigger rule for {rule.agent_role}")
    
    def remove_trigger_rule(self, agent_role: str):
        """Remove a trigger rule"""
        if agent_role in self.trigger_rules:
            del self.trigger_rules[agent_role]
            logger.info(f"Removed trigger rule for {agent_role}")
    
    def test_pattern_matching(self, file_path: str) -> List[str]:
        """Test which agents would be triggered by a file path"""
        return self.match_trigger_patterns(file_path, TriggerEventType.FILE_MODIFIED)
    
    def shutdown(self):
        """Graceful shutdown of trigger system"""
        logger.info("Shutting down ProactiveTriggerManager")
        self.stop_watching()
        logger.info("ProactiveTriggerManager shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    async def example_agent_callback(agent_role: str, context: Dict[str, Any]):
        """Example callback function for agent triggering"""
        print(f"🤖 TRIGGERED: {agent_role}")
        print(f"   Reason: {context['trigger_reason']}")
        print(f"   File: {context['trigger_event']['file_path']}")
        print(f"   Type: {context['trigger_event']['type']}")
        print()
    
    async def main():
        # Initialize trigger manager
        trigger_manager = ProactiveTriggerManager(
            watch_paths=["."],
            agent_callback=example_agent_callback
        )
        
        # Start watching
        trigger_manager.start_watching()
        
        # Test pattern matching
        test_files = [
            "app.py",
            "main.py", 
            "requirements.txt",
            "package.json",
            "src/components/LoginForm.tsx",
            "tests/test_user.py",
            "alembic/versions/001_initial.py"
        ]
        
        print("Testing pattern matching:")
        for file_path in test_files:
            matches = trigger_manager.test_pattern_matching(file_path)
            if matches:
                print(f"  {file_path} -> {matches}")
        
        print(f"\nWatching for file changes... (Press Ctrl+C to stop)")
        print(f"Statistics: {trigger_manager.get_trigger_statistics()}")
        
        try:
            # Keep running until interrupted
            while trigger_manager.is_watching:
                await asyncio.sleep(1)
                
                # Show recent activity
                stats = trigger_manager.get_trigger_statistics()
                if stats["last_activity"]:
                    recent_events = trigger_manager.get_recent_events(3)
                    if recent_events:
                        print(f"\nRecent events: {len(recent_events)}")
                        for event in recent_events[-1:]:  # Show just the latest
                            print(f"  {event['event_type']} {event['file_path']}")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            trigger_manager.shutdown()
    
    # Run example
    asyncio.run(main())