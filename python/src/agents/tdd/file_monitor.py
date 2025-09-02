#!/usr/bin/env python3
"""
File Monitor - Real-time TDD Enforcement File Watching

This module provides real-time file system monitoring to enforce TDD principles
by detecting code changes and ensuring tests exist before implementation.

CRITICAL: Blocks all file changes that violate test-first development principles.
"""

import os
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    # Fallback for systems without watchdog
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None
    FileSystemEvent = None

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of file system changes"""
    CREATED = "created"
    MODIFIED = "modified"  
    DELETED = "deleted"
    MOVED = "moved"

class ViolationSeverity(Enum):
    """Severity levels for TDD violations"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKING = "blocking"

@dataclass
class FileChange:
    """Individual file change event"""
    file_path: str
    change_type: ChangeType
    timestamp: datetime
    file_extension: str
    is_test_file: bool
    is_implementation_file: bool
    related_feature: Optional[str]
    size_bytes: int

@dataclass
class TDDViolationEvent:
    """TDD violation detected by file monitoring"""
    violation_id: str
    file_path: str
    change_type: ChangeType
    severity: ViolationSeverity
    description: str
    remediation: str
    detected_at: datetime
    blocking: bool
    feature_name: Optional[str]

if WATCHDOG_AVAILABLE:
    class TDDFileMonitor(FileSystemEventHandler):
        """
        Real-time file monitor for TDD enforcement
        
        Watches for file changes and enforces test-first development principles
        by blocking implementation changes without corresponding tests.
        """
else:
    class TDDFileMonitor:
        """
        Fallback TDD file monitor (watchdog not available)
        
        Provides basic file monitoring functionality without real-time watching.
        """
    
    def __init__(
        self,
        project_path: str = ".",
        watch_patterns: List[str] = None,
        ignore_patterns: List[str] = None,
        enable_blocking: bool = True,
        violation_callback: Callable = None
    ):
        if WATCHDOG_AVAILABLE:
            super().__init__()
        
        self.project_path = Path(project_path).resolve()
        self.watch_patterns = watch_patterns or ["**/*.py", "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx"]
        self.ignore_patterns = ignore_patterns or [
            "node_modules/**", ".git/**", "**/__pycache__/**", 
            "**/venv/**", "**/env/**", "**/.venv/**",
            "**/dist/**", "**/build/**", "**/.coverage"
        ]
        self.enable_blocking = enable_blocking
        self.violation_callback = violation_callback
        
        # Tracking
        self.recent_changes: List[FileChange] = []
        self.violations: List[TDDViolationEvent] = []
        self.blocked_files: Set[str] = set()
        self.feature_test_mapping: Dict[str, Set[str]] = {}
        
        # Observer instance
        self.observer: Optional[Observer] = None
        self.is_monitoring = False
        
        # Test file patterns
        self.test_patterns = [
            "test_*.py", "*_test.py", "*.test.js", "*.test.ts", 
            "*.spec.js", "*.spec.ts", "**/tests/**", "**/test/**",
            "**/__tests__/**", "**.test.**", "**.spec.**"
        ]
        
        # Implementation file extensions
        self.implementation_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp", ".c", ".cs"
        }
        
        logger.info(f"🔍 TDD File Monitor initialized for: {self.project_path}")
    
    def start_monitoring(self) -> bool:
        """Start real-time file monitoring"""
        if not WATCHDOG_AVAILABLE:
            logger.error("❌ Watchdog not available - install with: pip install watchdog")
            return False
        
        if self.is_monitoring:
            logger.warning("⚠️  File monitoring already active")
            return True
        
        try:
            self.observer = Observer()
            self.observer.schedule(self, str(self.project_path), recursive=True)
            self.observer.start()
            self.is_monitoring = True
            
            logger.info(f"✅ TDD file monitoring started - watching: {self.project_path}")
            logger.info(f"🔒 Blocking mode: {'ENABLED' if self.enable_blocking else 'DISABLED'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start file monitoring: {str(e)}")
            return False
    
    def stop_monitoring(self):
        """Stop file monitoring"""
        if self.observer and self.is_monitoring:
            self.observer.stop()
            self.observer.join()
            self.is_monitoring = False
            logger.info("🔍 TDD file monitoring stopped")
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events"""
        if not event.is_directory:
            self._handle_file_change(event.src_path, ChangeType.CREATED)
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events"""  
        if not event.is_directory:
            self._handle_file_change(event.src_path, ChangeType.MODIFIED)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events"""
        if not event.is_directory:
            self._handle_file_change(event.src_path, ChangeType.DELETED)
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move events"""
        if not event.is_directory:
            self._handle_file_change(event.dest_path, ChangeType.MOVED)
    
    def _handle_file_change(self, file_path: str, change_type: ChangeType):
        """Process individual file changes for TDD compliance"""
        
        try:
            file_path = str(Path(file_path).resolve())
            
            # Check if file should be monitored
            if not self._should_monitor_file(file_path):
                return
            
            # Create file change record
            file_change = self._create_file_change(file_path, change_type)
            self.recent_changes.append(file_change)
            
            # Trim recent changes (keep last 1000)
            if len(self.recent_changes) > 1000:
                self.recent_changes = self.recent_changes[-1000:]
            
            # Check for TDD violations
            violation = self._check_tdd_violation(file_change)
            
            if violation:
                self.violations.append(violation)
                
                # Notify callback if provided
                if self.violation_callback:
                    try:
                        self.violation_callback(violation)
                    except Exception as e:
                        logger.error(f"Violation callback failed: {str(e)}")
                
                # Handle blocking if enabled
                if self.enable_blocking and violation.blocking:
                    self._handle_blocking_violation(violation)
                
                logger.warning(
                    f"🚫 TDD Violation: {violation.description} "
                    f"[{violation.severity.value.upper()}] - {file_path}"
                )
            else:
                logger.debug(f"✅ TDD Compliant change: {file_path} ({change_type.value})")
                
        except Exception as e:
            logger.error(f"Error handling file change {file_path}: {str(e)}")
    
    def _should_monitor_file(self, file_path: str) -> bool:
        """Check if file should be monitored for TDD compliance"""
        
        path = Path(file_path)
        
        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if path.match(pattern):
                return False
        
        # Check watch patterns
        for pattern in self.watch_patterns:
            if path.match(pattern):
                return True
        
        return False
    
    def _create_file_change(self, file_path: str, change_type: ChangeType) -> FileChange:
        """Create FileChange object from file path and change type"""
        
        path = Path(file_path)
        
        # Determine file size
        size_bytes = 0
        try:
            if change_type != ChangeType.DELETED and path.exists():
                size_bytes = path.stat().st_size
        except Exception:
            pass
        
        # Determine if test file
        is_test_file = any(path.match(pattern) for pattern in self.test_patterns)
        
        # Determine if implementation file
        is_implementation_file = (
            path.suffix in self.implementation_extensions and 
            not is_test_file
        )
        
        # Extract related feature name
        related_feature = self._extract_feature_name(path)
        
        return FileChange(
            file_path=file_path,
            change_type=change_type,
            timestamp=datetime.now(),
            file_extension=path.suffix,
            is_test_file=is_test_file,
            is_implementation_file=is_implementation_file,
            related_feature=related_feature,
            size_bytes=size_bytes
        )
    
    def _extract_feature_name(self, path: Path) -> Optional[str]:
        """Extract feature name from file path"""
        
        # Remove test indicators and file extension
        name = path.stem
        
        # Remove common test prefixes/suffixes
        for test_indicator in ["test_", "_test", ".test", ".spec"]:
            name = name.replace(test_indicator, "")
        
        # Clean up the name
        name = name.strip("_-.")
        
        return name if name else None
    
    def _check_tdd_violation(self, file_change: FileChange) -> Optional[TDDViolationEvent]:
        """Check if file change violates TDD principles"""
        
        # Only check implementation files
        if not file_change.is_implementation_file:
            return None
        
        # Skip deleted files (they don't add new functionality)
        if file_change.change_type == ChangeType.DELETED:
            return None
        
        # Check if corresponding tests exist
        if file_change.related_feature:
            test_files = self._find_test_files_for_feature(file_change.related_feature)
            
            if not test_files:
                return TDDViolationEvent(
                    violation_id=f"tdd_{int(time.time())}_{hash(file_change.file_path)}",
                    file_path=file_change.file_path,
                    change_type=file_change.change_type,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"Implementation change without tests for feature '{file_change.related_feature}'",
                    remediation=f"Create tests for feature '{file_change.related_feature}' before implementation",
                    detected_at=file_change.timestamp,
                    blocking=True,
                    feature_name=file_change.related_feature
                )
        
        # Check for large implementation changes (> 1KB) without recent test changes
        if file_change.size_bytes > 1024:
            recent_test_changes = [
                change for change in self.recent_changes[-20:]  # Last 20 changes
                if (change.is_test_file and 
                    change.related_feature == file_change.related_feature and
                    (file_change.timestamp - change.timestamp).total_seconds() < 3600)  # Within 1 hour
            ]
            
            if not recent_test_changes:
                return TDDViolationEvent(
                    violation_id=f"tdd_{int(time.time())}_{hash(file_change.file_path)}",
                    file_path=file_change.file_path,
                    change_type=file_change.change_type,
                    severity=ViolationSeverity.ERROR,
                    description=f"Large implementation change ({file_change.size_bytes} bytes) without recent test updates",
                    remediation="Update or create tests before making large implementation changes",
                    detected_at=file_change.timestamp,
                    blocking=True,
                    feature_name=file_change.related_feature
                )
        
        return None
    
    def _find_test_files_for_feature(self, feature_name: str) -> List[str]:
        """Find test files related to a feature"""
        
        test_files = []
        
        # Search for test files with matching names
        for pattern in [
            f"test_{feature_name}.py",
            f"{feature_name}_test.py", 
            f"{feature_name}.test.js",
            f"{feature_name}.test.ts",
            f"{feature_name}.spec.js",
            f"{feature_name}.spec.ts"
        ]:
            matches = list(self.project_path.glob(f"**/{pattern}"))
            test_files.extend(str(match) for match in matches)
        
        return test_files
    
    def _handle_blocking_violation(self, violation: TDDViolationEvent):
        """Handle blocking TDD violations"""
        
        self.blocked_files.add(violation.file_path)
        
        if self.enable_blocking:
            logger.error(
                f"🚫 BLOCKING TDD VIOLATION: {violation.description}\n"
                f"   File: {violation.file_path}\n"
                f"   Remediation: {violation.remediation}\n"
                f"   🔒 Further changes to this file are BLOCKED until TDD compliance is achieved"
            )
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get file monitoring statistics"""
        
        return {
            "monitoring_active": self.is_monitoring,
            "total_changes": len(self.recent_changes),
            "total_violations": len(self.violations),
            "blocked_files": len(self.blocked_files),
            "critical_violations": len([v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]),
            "recent_changes_24h": len([
                c for c in self.recent_changes 
                if (datetime.now() - c.timestamp).total_seconds() < 86400
            ]),
            "test_file_changes": len([c for c in self.recent_changes if c.is_test_file]),
            "implementation_changes": len([c for c in self.recent_changes if c.is_implementation_file]),
            "watch_patterns": self.watch_patterns,
            "ignore_patterns": self.ignore_patterns,
            "blocking_enabled": self.enable_blocking
        }
    
    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent TDD violations"""
        
        recent = sorted(self.violations, key=lambda v: v.detected_at, reverse=True)[:limit]
        
        return [
            {
                "violation_id": v.violation_id,
                "file_path": v.file_path,
                "change_type": v.change_type.value,
                "severity": v.severity.value,
                "description": v.description,
                "remediation": v.remediation,
                "detected_at": v.detected_at.isoformat(),
                "blocking": v.blocking,
                "feature_name": v.feature_name
            }
            for v in recent
        ]
    
    def clear_violation_history(self):
        """Clear violation history (for testing/reset)"""
        self.violations.clear()
        self.blocked_files.clear()
        logger.info("🧹 TDD violation history cleared")

# Global monitor instance
_file_monitor: Optional[TDDFileMonitor] = None

def get_file_monitor(project_path: str = ".") -> TDDFileMonitor:
    """Get or create global file monitor instance"""
    global _file_monitor
    
    if _file_monitor is None or not _file_monitor.is_monitoring:
        _file_monitor = TDDFileMonitor(
            project_path=project_path,
            enable_blocking=True
        )
    
    return _file_monitor

async def start_tdd_file_monitoring(project_path: str = ".") -> bool:
    """Start TDD file monitoring for a project"""
    monitor = get_file_monitor(project_path)
    return monitor.start_monitoring()

def stop_tdd_file_monitoring():
    """Stop TDD file monitoring"""
    global _file_monitor
    if _file_monitor:
        _file_monitor.stop_monitoring()
        _file_monitor = None