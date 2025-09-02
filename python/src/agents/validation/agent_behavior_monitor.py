#!/usr/bin/env python3
"""
AGENT BEHAVIOR MONITOR
Real-time monitoring and analysis of agent actions to prevent gaming

This monitor tracks all agent activities and detects:
1. Agents commenting out validation rules
2. Agents creating fake implementations to pass tests  
3. Agents bypassing quality gates
4. Agents manipulating metrics or test results
5. Patterns of gaming behavior across multiple actions

CRITICAL: This runs alongside agents and blocks gaming behavior in real-time
"""

import os
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of agent actions to monitor"""
    CODE_CHANGE = "code_change"
    TEST_CREATION = "test_creation" 
    TEST_MODIFICATION = "test_modification"
    VALIDATION_BYPASS = "validation_bypass"
    FEATURE_IMPLEMENTATION = "feature_implementation"
    COMMENT_ADDITION = "comment_addition"
    FILE_DELETION = "file_deletion"
    RULE_MODIFICATION = "rule_modification"

class SuspicionLevel(Enum):
    """Levels of suspicious behavior"""
    CLEAN = "clean"
    MINOR = "minor"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentAction:
    """Single agent action record"""
    agent_name: str
    action_type: ActionType
    timestamp: datetime
    file_path: str
    content_before: str
    content_after: str
    explanation: str
    suspicion_level: SuspicionLevel
    gaming_indicators: List[str]
    action_hash: str

@dataclass
class BehaviorPattern:
    """Pattern of behavior across multiple actions"""
    pattern_type: str
    agent_name: str
    actions_count: int
    first_seen: datetime
    last_seen: datetime
    suspicion_score: float
    indicators: List[str]
    blocked: bool

class AgentBehaviorMonitor:
    """Monitors and analyzes agent behavior for gaming patterns"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.monitor_dir = self.project_path / ".archon_monitor"
        self.monitor_dir.mkdir(exist_ok=True)
        
        self.actions_log = self.monitor_dir / "agent_actions.jsonl"
        self.patterns_file = self.monitor_dir / "behavior_patterns.json"
        self.blocked_agents_file = self.monitor_dir / "blocked_agents.json"
        
        # Gaming detection patterns
        self.gaming_patterns = {
            'validation_bypass': [
                'commented out validation',
                'disabled enforcement',
                'skipped required check',
                'bypassed quality gate'
            ],
            'fake_implementation': [
                'returns mock data',
                'stub implementation',
                'placeholder function',
                'fake response'
            ],
            'test_gaming': [
                'meaningless assertion',
                'always passes test',
                'mocked real functionality',
                'fake test coverage'
            ],
            'metric_manipulation': [
                'artificially increased coverage',
                'hidden failing tests',
                'fake success metrics',
                'masked errors'
            ]
        }
        
        # Load existing data
        self.actions_history = self._load_actions_history()
        self.behavior_patterns = self._load_behavior_patterns()
        self.blocked_agents = self._load_blocked_agents()
    
    def record_agent_action(self, 
                           agent_name: str,
                           action_type: ActionType,
                           file_path: str,
                           content_before: str,
                           content_after: str,
                           explanation: str = "") -> Dict[str, Any]:
        """
        Record an agent action and analyze for gaming behavior
        
        Returns analysis result with blocking decision
        """
        
        # Create action record
        action = AgentAction(
            agent_name=agent_name,
            action_type=action_type,
            timestamp=datetime.now(),
            file_path=file_path,
            content_before=content_before,
            content_after=content_after,
            explanation=explanation,
            suspicion_level=SuspicionLevel.CLEAN,
            gaming_indicators=[],
            action_hash=self._calculate_action_hash(agent_name, file_path, content_before, content_after)
        )
        
        # Analyze action for gaming behavior
        self._analyze_action_gaming(action)
        
        # Check if agent should be blocked
        should_block = self._should_block_agent(agent_name, action)
        
        if should_block:
            self._block_agent(agent_name, action)
            
        # Record action
        self._save_action(action)
        
        # Update behavior patterns
        self._update_behavior_patterns(agent_name, action)
        
        return {
            "action_recorded": True,
            "agent_name": agent_name,
            "suspicion_level": action.suspicion_level.value,
            "gaming_indicators": action.gaming_indicators,
            "blocked": should_block,
            "action_hash": action.action_hash,
            "message": self._get_action_message(action, should_block)
        }
    
    def _analyze_action_gaming(self, action: AgentAction):
        """Analyze single action for gaming behavior"""
        
        content_diff = self._get_content_diff(action.content_before, action.content_after)
        
        # Check for gaming indicators
        gaming_indicators = []
        max_suspicion = SuspicionLevel.CLEAN
        
        # Check for validation bypasses
        validation_bypasses = self._detect_validation_bypass(content_diff)
        if validation_bypasses:
            gaming_indicators.extend(validation_bypasses)
            max_suspicion = max(max_suspicion, SuspicionLevel.CRITICAL, key=lambda x: x.value)
        
        # Check for fake implementations
        fake_implementations = self._detect_fake_implementation(content_diff)
        if fake_implementations:
            gaming_indicators.extend(fake_implementations)
            max_suspicion = max(max_suspicion, SuspicionLevel.HIGH, key=lambda x: x.value)
        
        # Check for test gaming
        test_gaming = self._detect_test_gaming(action.file_path, content_diff)
        if test_gaming:
            gaming_indicators.extend(test_gaming)
            max_suspicion = max(max_suspicion, SuspicionLevel.HIGH, key=lambda x: x.value)
        
        # Check for suspicious comments
        suspicious_comments = self._detect_suspicious_comments(content_diff)
        if suspicious_comments:
            gaming_indicators.extend(suspicious_comments)
            max_suspicion = max(max_suspicion, SuspicionLevel.MODERATE, key=lambda x: x.value)
        
        # Update action with findings
        action.gaming_indicators = gaming_indicators
        action.suspicion_level = max_suspicion
    
    def _detect_validation_bypass(self, content_diff: Dict[str, Any]) -> List[str]:
        """Detect attempts to bypass validation"""
        
        indicators = []
        added_lines = content_diff.get('added', [])
        removed_lines = content_diff.get('removed', [])
        
        # Check for commented validation
        validation_keywords = ['validation', 'enforce', 'mandatory', 'critical', 'required']
        
        for line in added_lines:
            if line.strip().startswith('#') or line.strip().startswith('//'):
                if any(keyword in line.lower() for keyword in validation_keywords):
                    indicators.append(f"Commented out validation: {line.strip()}")
        
        # Check for disabled enforcement
        disable_patterns = [
            r'if\s+False:', r'return\s+True\s*#.*skip', r'pass\s*#.*bypass'
        ]
        
        for line in added_lines:
            for pattern in disable_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    indicators.append(f"Disabled enforcement: {line.strip()}")
        
        return indicators
    
    def _detect_fake_implementation(self, content_diff: Dict[str, Any]) -> List[str]:
        """Detect fake or stub implementations"""
        
        indicators = []
        added_lines = content_diff.get('added', [])
        
        fake_patterns = [
            r'return\s+["\']mock["\']',
            r'return\s+["\']fake["\']',
            r'return\s+["\']stub["\']',
            r'return\s+\{\s*["\']status["\']:\s*["\']ok["\']',
            r'return\s+True\s*#.*fake',
            r'return\s+\[\]\s*#.*placeholder'
        ]
        
        for line in added_lines:
            for pattern in fake_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    indicators.append(f"Fake implementation: {line.strip()}")
        
        return indicators
    
    def _detect_test_gaming(self, file_path: str, content_diff: Dict[str, Any]) -> List[str]:
        """Detect test gaming in test files"""
        
        indicators = []
        
        # Only check test files
        if not any(test_indicator in file_path.lower() for test_indicator in ['test', 'spec']):
            return indicators
        
        added_lines = content_diff.get('added', [])
        
        test_gaming_patterns = [
            r'assert\s+True\s*$',
            r'assert\s+1\s*==\s*1',
            r'assert\s+not\s+False',
            r'@mock\.',
            r'\.return_value\s*='
        ]
        
        for line in added_lines:
            for pattern in test_gaming_patterns:
                if re.search(pattern, line):
                    indicators.append(f"Test gaming: {line.strip()}")
        
        return indicators
    
    def _detect_suspicious_comments(self, content_diff: Dict[str, Any]) -> List[str]:
        """Detect suspicious comments that might indicate gaming"""
        
        indicators = []
        added_lines = content_diff.get('added', [])
        
        suspicious_keywords = [
            'hack', 'temp', 'bypass', 'skip', 'disable', 'fake',
            'cheat', 'workaround', 'quick fix', 'dirty'
        ]
        
        for line in added_lines:
            if line.strip().startswith('#') or line.strip().startswith('//'):
                if any(keyword in line.lower() for keyword in suspicious_keywords):
                    indicators.append(f"Suspicious comment: {line.strip()}")
        
        return indicators
    
    def _should_block_agent(self, agent_name: str, action: AgentAction) -> bool:
        """Determine if agent should be blocked based on action and history"""
        
        # Always block critical suspicion level
        if action.suspicion_level == SuspicionLevel.CRITICAL:
            return True
        
        # Check if agent is already blocked
        if agent_name in self.blocked_agents:
            return True
        
        # Check recent pattern of suspicious behavior
        recent_actions = self._get_recent_actions(agent_name, hours=24)
        suspicious_actions = [a for a in recent_actions if a.suspicion_level.value >= 3]  # MODERATE or higher
        
        # Block if too many suspicious actions
        if len(suspicious_actions) >= 3:
            return True
        
        # Block if gaming indicators exceed threshold
        total_indicators = sum(len(a.gaming_indicators) for a in recent_actions)
        if total_indicators >= 5:
            return True
        
        return False
    
    def _block_agent(self, agent_name: str, action: AgentAction):
        """Block agent and record reason"""
        
        block_reason = {
            "agent_name": agent_name,
            "blocked_at": datetime.now().isoformat(),
            "reason": f"Gaming behavior detected: {action.suspicion_level.value}",
            "gaming_indicators": action.gaming_indicators,
            "trigger_action": action.action_hash,
            "auto_unblock_at": (datetime.now() + timedelta(hours=2)).isoformat()  # Auto-unblock after 2 hours
        }
        
        self.blocked_agents[agent_name] = block_reason
        self._save_blocked_agents()
        
        logger.critical(f"AGENT BLOCKED: {agent_name} - Reason: {block_reason['reason']}")
    
    def is_agent_blocked(self, agent_name: str) -> Dict[str, Any]:
        """Check if agent is currently blocked"""
        
        if agent_name not in self.blocked_agents:
            return {"blocked": False}
        
        block_info = self.blocked_agents[agent_name]
        
        # Check if auto-unblock time has passed
        auto_unblock_time = datetime.fromisoformat(block_info["auto_unblock_at"])
        if datetime.now() > auto_unblock_time:
            # Auto-unblock
            del self.blocked_agents[agent_name]
            self._save_blocked_agents()
            return {"blocked": False, "message": "Auto-unblocked after timeout"}
        
        return {
            "blocked": True,
            "blocked_at": block_info["blocked_at"],
            "reason": block_info["reason"],
            "gaming_indicators": block_info["gaming_indicators"],
            "auto_unblock_at": block_info["auto_unblock_at"]
        }
    
    def unblock_agent(self, agent_name: str, reason: str = "Manual unblock") -> Dict[str, Any]:
        """Manually unblock an agent"""
        
        if agent_name not in self.blocked_agents:
            return {"unblocked": False, "message": "Agent was not blocked"}
        
        del self.blocked_agents[agent_name]
        self._save_blocked_agents()
        
        logger.info(f"AGENT UNBLOCKED: {agent_name} - Reason: {reason}")
        
        return {"unblocked": True, "agent": agent_name, "reason": reason}
    
    def get_agent_behavior_report(self, agent_name: str, days: int = 7) -> Dict[str, Any]:
        """Generate behavior report for specific agent"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        agent_actions = [a for a in self.actions_history if a.agent_name == agent_name and a.timestamp > cutoff_date]
        
        if not agent_actions:
            return {"agent": agent_name, "actions": 0, "message": "No recent actions"}
        
        # Calculate behavior metrics
        total_actions = len(agent_actions)
        suspicious_actions = [a for a in agent_actions if a.suspicion_level != SuspicionLevel.CLEAN]
        gaming_indicators_count = sum(len(a.gaming_indicators) for a in agent_actions)
        
        suspicion_breakdown = {}
        for level in SuspicionLevel:
            count = len([a for a in agent_actions if a.suspicion_level == level])
            suspicion_breakdown[level.value] = count
        
        gaming_risk = "LOW"
        if gaming_indicators_count > 10:
            gaming_risk = "CRITICAL"
        elif gaming_indicators_count > 5:
            gaming_risk = "HIGH" 
        elif gaming_indicators_count > 2:
            gaming_risk = "MODERATE"
        
        return {
            "agent": agent_name,
            "period_days": days,
            "total_actions": total_actions,
            "suspicious_actions": len(suspicious_actions),
            "gaming_indicators_count": gaming_indicators_count,
            "suspicion_breakdown": suspicion_breakdown,
            "gaming_risk": gaming_risk,
            "blocked": agent_name in self.blocked_agents,
            "recent_indicators": suspicious_actions[-5:] if suspicious_actions else []
        }
    
    def _get_content_diff(self, before: str, after: str) -> Dict[str, Any]:
        """Get difference between content versions"""
        
        before_lines = before.split('\n') if before else []
        after_lines = after.split('\n') if after else []
        
        # Simple diff - added and removed lines
        added = [line for line in after_lines if line not in before_lines]
        removed = [line for line in before_lines if line not in after_lines]
        
        return {"added": added, "removed": removed}
    
    def _get_recent_actions(self, agent_name: str, hours: int = 24) -> List[AgentAction]:
        """Get recent actions for an agent"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.actions_history if a.agent_name == agent_name and a.timestamp > cutoff_time]
    
    def _calculate_action_hash(self, agent_name: str, file_path: str, before: str, after: str) -> str:
        """Calculate unique hash for action"""
        
        content = f"{agent_name}:{file_path}:{before}:{after}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_action_message(self, action: AgentAction, blocked: bool) -> str:
        """Get human-readable message for action result"""
        
        if blocked:
            return f"AGENT BLOCKED: {action.agent_name} engaged in gaming behavior ({action.suspicion_level.value})"
        elif action.suspicion_level == SuspicionLevel.CLEAN:
            return f"Action recorded: {action.agent_name} - clean behavior"
        else:
            return f"Suspicious behavior detected: {action.agent_name} - {action.suspicion_level.value} level"
    
    def _update_behavior_patterns(self, agent_name: str, action: AgentAction):
        """Update behavior pattern analysis"""
        # Implementation for pattern tracking would go here
        pass
    
    def _save_action(self, action: AgentAction):
        """Save action to log file"""
        
        action_dict = asdict(action)
        action_dict['timestamp'] = action.timestamp.isoformat()
        action_dict['action_type'] = action.action_type.value
        action_dict['suspicion_level'] = action.suspicion_level.value
        
        with open(self.actions_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(action_dict) + '\n')
        
        # Keep in memory (limited to last 1000 actions)
        self.actions_history.append(action)
        if len(self.actions_history) > 1000:
            self.actions_history = self.actions_history[-1000:]
    
    def _load_actions_history(self) -> List[AgentAction]:
        """Load actions from log file"""
        
        actions = []
        
        if self.actions_log.exists():
            try:
                with open(self.actions_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            action_dict = json.loads(line)
                            action_dict['timestamp'] = datetime.fromisoformat(action_dict['timestamp'])
                            action_dict['action_type'] = ActionType(action_dict['action_type'])
                            action_dict['suspicion_level'] = SuspicionLevel(action_dict['suspicion_level'])
                            actions.append(AgentAction(**action_dict))
            except Exception as e:
                logger.warning(f"Error loading actions history: {e}")
        
        return actions[-1000:]  # Keep last 1000
    
    def _load_behavior_patterns(self) -> Dict[str, Any]:
        """Load behavior patterns"""
        
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading behavior patterns: {e}")
        
        return {}
    
    def _load_blocked_agents(self) -> Dict[str, Any]:
        """Load blocked agents list"""
        
        if self.blocked_agents_file.exists():
            try:
                with open(self.blocked_agents_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading blocked agents: {e}")
        
        return {}
    
    def _save_blocked_agents(self):
        """Save blocked agents list"""
        
        try:
            with open(self.blocked_agents_file, 'w', encoding='utf-8') as f:
                json.dump(self.blocked_agents, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving blocked agents: {e}")

# Global monitor instance (singleton pattern)
_global_monitor: Optional[AgentBehaviorMonitor] = None

def get_behavior_monitor(project_path: str = ".") -> AgentBehaviorMonitor:
    """Get global behavior monitor instance"""
    global _global_monitor
    
    if _global_monitor is None or str(_global_monitor.project_path) != str(Path(project_path).resolve()):
        _global_monitor = AgentBehaviorMonitor(project_path)
    
    return _global_monitor

def monitor_agent_action(agent_name: str, 
                        action_type: ActionType,
                        file_path: str,
                        content_before: str,
                        content_after: str,
                        explanation: str = "",
                        project_path: str = ".") -> Dict[str, Any]:
    """
    Monitor and analyze agent action for gaming behavior
    
    This function should be called by all systems that modify code on behalf of agents
    """
    
    monitor = get_behavior_monitor(project_path)
    return monitor.record_agent_action(agent_name, action_type, file_path, content_before, content_after, explanation)

def check_agent_blocked(agent_name: str, project_path: str = ".") -> Dict[str, Any]:
    """Check if agent is currently blocked"""
    
    monitor = get_behavior_monitor(project_path)
    return monitor.is_agent_blocked(agent_name)