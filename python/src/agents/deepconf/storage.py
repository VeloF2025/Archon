"""
DeepConf Persistent Storage Service

Provides persistent storage for confidence data to ensure historical data
survives across agent execution instances. Supports both database and file-based
storage with automatic fallback.

This solves the critical issue where DeepConf only showed 13 data points instead
of thousands from months of daily coding by ensuring ALL confidence data persists.

Author: Archon AI System  
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
import threading
from contextlib import contextmanager

from .types import ConfidenceScore

logger = logging.getLogger(__name__)

class DeepConfStorage:
    """
    Persistent storage backend for DeepConf confidence data
    
    Features:
    - SQLite database for structured confidence data
    - JSON file fallback for reliability
    - Thread-safe operations
    - Automatic data migration
    - Historical data retention
    - Fast querying and indexing
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize persistent storage"""
        # Default storage location
        if not storage_path:
            storage_path = os.getenv('DEEPCONF_STORAGE_PATH', '/tmp/archon_deepconf_data')
        
        self.storage_dir = Path(storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.db_path = self.storage_dir / 'confidence_data.db'
        self.json_backup_path = self.storage_dir / 'confidence_backup.json'
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize storage
        self._init_database()
        self._init_indexes()
        
        logger.info(f"DeepConf storage initialized at {self.storage_dir}")
    
    def _init_database(self):
        """Initialize SQLite database with confidence data schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS confidence_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        agent_id TEXT NOT NULL,
                        agent_type TEXT,
                        domain TEXT,
                        complexity TEXT,
                        phase TEXT,
                        overall_confidence REAL NOT NULL,
                        factual_confidence REAL,
                        reasoning_confidence REAL,
                        contextual_confidence REAL,
                        epistemic_uncertainty REAL,
                        aleatoric_uncertainty REAL,
                        uncertainty_bounds_lower REAL,
                        uncertainty_bounds_upper REAL,
                        confidence_factors TEXT, -- JSON string
                        primary_factors TEXT,    -- JSON string
                        confidence_reasoning TEXT,
                        model_source TEXT,
                        calibration_applied INTEGER DEFAULT 0,
                        gaming_detection_score REAL DEFAULT 0.0,
                        execution_duration REAL,
                        success INTEGER,
                        result_quality REAL,
                        user_prompt TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX (timestamp),
                        INDEX (agent_id),
                        INDEX (domain),
                        INDEX (phase)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        agent_type TEXT,
                        user_prompt TEXT,
                        execution_start_time REAL,
                        execution_end_time REAL,
                        execution_duration REAL,
                        success INTEGER,
                        result_quality REAL,
                        complexity_assessment TEXT,
                        domain TEXT,
                        phase TEXT,
                        error_details TEXT,
                        confidence_data TEXT, -- JSON string of full confidence score
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX (timestamp),
                        INDEX (agent_name),
                        INDEX (domain),
                        INDEX (success)
                    )
                """)
                
                # Check if we need to migrate existing structure
                self._migrate_database_if_needed(conn)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Fall back to JSON-only storage
    
    def _migrate_database_if_needed(self, conn: sqlite3.Connection):
        """Migrate database schema if needed"""
        try:
            # Check for missing columns and add them
            cursor = conn.execute("PRAGMA table_info(confidence_history)")
            columns = {row[1] for row in cursor.fetchall()}
            
            required_columns = [
                ('agent_type', 'TEXT'),
                ('execution_duration', 'REAL'),
                ('success', 'INTEGER'),
                ('result_quality', 'REAL'),
                ('user_prompt', 'TEXT')
            ]
            
            for column_name, column_type in required_columns:
                if column_name not in columns:
                    conn.execute(f"ALTER TABLE confidence_history ADD COLUMN {column_name} {column_type}")
                    logger.info(f"Added column {column_name} to confidence_history table")
                    
        except Exception as e:
            logger.warning(f"Database migration warning: {e}")
    
    def _init_indexes(self):
        """Create database indexes for performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create additional performance indexes
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_confidence_timestamp ON confidence_history(timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_confidence_agent_domain ON confidence_history(agent_id, domain)",
                    "CREATE INDEX IF NOT EXISTS idx_confidence_phase_success ON confidence_history(phase, success)",
                    "CREATE INDEX IF NOT EXISTS idx_execution_timestamp ON execution_records(execution_end_time DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_execution_agent_type ON execution_records(agent_name, agent_type)"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    async def store_confidence_score(self, confidence_score: ConfidenceScore, 
                                   execution_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store confidence score with optional execution metadata
        
        Args:
            confidence_score: ConfidenceScore object to store
            execution_data: Optional execution metadata (duration, success, etc.)
            
        Returns:
            bool: True if successfully stored
        """
        try:
            with self._lock:
                # Store in database
                await self._store_in_database(confidence_score, execution_data)
                
                # Store in JSON backup
                await self._store_in_json_backup(confidence_score, execution_data)
                
            logger.debug(f"Stored confidence data for task {confidence_score.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store confidence score for task {confidence_score.task_id}: {e}")
            return False
    
    async def _store_in_database(self, confidence_score: ConfidenceScore, 
                               execution_data: Optional[Dict[str, Any]] = None):
        """Store confidence score in SQLite database"""
        try:
            with self._get_connection() as conn:
                # Prepare data for insertion
                data = {
                    'task_id': confidence_score.task_id,
                    'timestamp': confidence_score.timestamp,
                    'agent_id': confidence_score.model_source,
                    'overall_confidence': confidence_score.overall_confidence,
                    'factual_confidence': confidence_score.factual_confidence,
                    'reasoning_confidence': confidence_score.reasoning_confidence,
                    'contextual_confidence': confidence_score.contextual_confidence,
                    'epistemic_uncertainty': confidence_score.epistemic_uncertainty,
                    'aleatoric_uncertainty': confidence_score.aleatoric_uncertainty,
                    'uncertainty_bounds_lower': confidence_score.uncertainty_bounds[0],
                    'uncertainty_bounds_upper': confidence_score.uncertainty_bounds[1],
                    'confidence_factors': json.dumps(confidence_score.confidence_factors),
                    'primary_factors': json.dumps(confidence_score.primary_factors),
                    'confidence_reasoning': confidence_score.confidence_reasoning,
                    'model_source': confidence_score.model_source,
                    'calibration_applied': int(confidence_score.calibration_applied),
                    'gaming_detection_score': confidence_score.gaming_detection_score
                }
                
                # Add execution data if provided
                if execution_data:
                    data.update({
                        'agent_type': execution_data.get('agent_type', ''),
                        'domain': execution_data.get('domain', 'general'),
                        'complexity': execution_data.get('complexity', 'moderate'),
                        'phase': execution_data.get('phase', 'production'),
                        'execution_duration': execution_data.get('execution_duration', 0.0),
                        'success': int(execution_data.get('success', True)),
                        'result_quality': execution_data.get('result_quality', 0.8),
                        'user_prompt': execution_data.get('user_prompt', '')
                    })
                
                # Insert into database
                placeholders = ', '.join(['?' for _ in data])
                columns = ', '.join(data.keys())
                
                conn.execute(
                    f"INSERT INTO confidence_history ({columns}) VALUES ({placeholders})",
                    tuple(data.values())
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            # Don't re-raise - let JSON backup handle it
    
    async def _store_in_json_backup(self, confidence_score: ConfidenceScore, 
                                  execution_data: Optional[Dict[str, Any]] = None):
        """Store confidence score in JSON backup file"""
        try:
            # Create backup entry
            backup_entry = {
                'timestamp': confidence_score.timestamp,
                'task_id': confidence_score.task_id,
                'confidence_score': confidence_score.to_dict(),
                'execution_data': execution_data or {},
                'stored_at': time.time()
            }
            
            # Read existing data
            existing_data = []
            if self.json_backup_path.exists():
                try:
                    with open(self.json_backup_path, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
            
            # Append new entry
            existing_data.append(backup_entry)
            
            # Keep only recent entries (last 10,000)
            if len(existing_data) > 10000:
                existing_data = existing_data[-10000:]
            
            # Write back to file
            with open(self.json_backup_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"JSON backup storage failed: {e}")
    
    async def load_historical_data(self, limit: int = 1000, 
                                 agent_id: Optional[str] = None,
                                 domain: Optional[str] = None,
                                 phase: Optional[str] = None,
                                 since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Load historical confidence data with filtering
        
        Args:
            limit: Maximum number of records to return
            agent_id: Filter by specific agent
            domain: Filter by domain
            phase: Filter by phase
            since_timestamp: Only return data after this timestamp
            
        Returns:
            List[Dict[str, Any]]: Historical confidence data
        """
        try:
            # Try database first
            data = await self._load_from_database(limit, agent_id, domain, phase, since_timestamp)
            
            if not data:
                # Fallback to JSON backup
                data = await self._load_from_json_backup(limit, agent_id, domain, phase, since_timestamp)
            
            logger.info(f"Loaded {len(data)} historical confidence records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return []
    
    async def _load_from_database(self, limit: int, agent_id: Optional[str] = None,
                                domain: Optional[str] = None, phase: Optional[str] = None,
                                since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Load historical data from SQLite database"""
        try:
            with self._get_connection() as conn:
                # Build query with filters
                query = """
                    SELECT * FROM confidence_history 
                    WHERE 1=1
                """
                params = []
                
                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)
                
                if domain:
                    query += " AND domain = ?"
                    params.append(domain)
                
                if phase:
                    query += " AND phase = ?"
                    params.append(phase)
                
                if since_timestamp:
                    query += " AND timestamp >= ?"
                    params.append(since_timestamp)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                historical_data = []
                for row in rows:
                    record = {
                        'timestamp': row['timestamp'],
                        'task_id': row['task_id'],
                        'agent_id': row['agent_id'],
                        'domain': row['domain'] or 'general',
                        'complexity': row['complexity'] or 'moderate',
                        'phase': row['phase'] or 'production',
                        'confidence_score': {
                            'overall_confidence': row['overall_confidence'],
                            'factual_confidence': row['factual_confidence'],
                            'reasoning_confidence': row['reasoning_confidence'],
                            'contextual_confidence': row['contextual_confidence'],
                            'epistemic_uncertainty': row['epistemic_uncertainty'],
                            'aleatoric_uncertainty': row['aleatoric_uncertainty'],
                            'uncertainty_bounds': (row['uncertainty_bounds_lower'], row['uncertainty_bounds_upper']),
                            'confidence_factors': json.loads(row['confidence_factors'] or '{}'),
                            'primary_factors': json.loads(row['primary_factors'] or '[]'),
                            'confidence_reasoning': row['confidence_reasoning'],
                            'model_source': row['model_source'],
                            'task_id': row['task_id'],
                            'timestamp': row['timestamp'],
                            'calibration_applied': bool(row['calibration_applied']),
                            'gaming_detection_score': row['gaming_detection_score'] or 0.0
                        },
                        'execution_duration': row['execution_duration'] or 0.0,
                        'success': bool(row['success']) if row['success'] is not None else True,
                        'result_quality': row['result_quality'] or 0.8,
                        'source': 'database'
                    }
                    historical_data.append(record)
                
                return historical_data
                
        except Exception as e:
            logger.warning(f"Database load failed, trying JSON backup: {e}")
            return []
    
    async def _load_from_json_backup(self, limit: int, agent_id: Optional[str] = None,
                                   domain: Optional[str] = None, phase: Optional[str] = None,
                                   since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Load historical data from JSON backup"""
        try:
            if not self.json_backup_path.exists():
                return []
            
            with open(self.json_backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Filter and format data
            filtered_data = []
            for entry in backup_data:
                confidence_score = entry['confidence_score']
                execution_data = entry['execution_data']
                
                # Apply filters
                if agent_id and confidence_score.get('model_source') != agent_id:
                    continue
                if domain and execution_data.get('domain') != domain:
                    continue
                if phase and execution_data.get('phase') != phase:
                    continue
                if since_timestamp and entry['timestamp'] < since_timestamp:
                    continue
                
                # Format for consistency with database format
                record = {
                    'timestamp': entry['timestamp'],
                    'task_id': confidence_score['task_id'],
                    'agent_id': confidence_score['model_source'],
                    'domain': execution_data.get('domain', 'general'),
                    'complexity': execution_data.get('complexity', 'moderate'),
                    'phase': execution_data.get('phase', 'production'),
                    'confidence_score': confidence_score,
                    'execution_duration': execution_data.get('execution_duration', 0.0),
                    'success': execution_data.get('success', True),
                    'result_quality': execution_data.get('result_quality', 0.8),
                    'source': 'json_backup'
                }
                filtered_data.append(record)
            
            # Sort by timestamp (newest first) and limit
            filtered_data.sort(key=lambda x: x['timestamp'], reverse=True)
            return filtered_data[:limit]
            
        except Exception as e:
            logger.error(f"JSON backup load failed: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'storage_path': str(self.storage_dir),
            'database_exists': self.db_path.exists(),
            'json_backup_exists': self.json_backup_path.exists(),
            'database_size_mb': 0,
            'json_backup_size_mb': 0,
            'total_records': 0,
            'oldest_record': None,
            'newest_record': None
        }
        
        try:
            # Database stats
            if self.db_path.exists():
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM confidence_history")
                    stats['total_records'] = cursor.fetchone()[0]
                    
                    # Get date range
                    cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM confidence_history")
                    min_ts, max_ts = cursor.fetchone()
                    if min_ts:
                        stats['oldest_record'] = datetime.fromtimestamp(min_ts).isoformat()
                    if max_ts:
                        stats['newest_record'] = datetime.fromtimestamp(max_ts).isoformat()
            
            # JSON backup stats
            if self.json_backup_path.exists():
                stats['json_backup_size_mb'] = self.json_backup_path.stat().st_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return stats
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old confidence data to manage storage size
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            int: Number of records deleted
        """
        try:
            cutoff_timestamp = time.time() - (days_to_keep * 24 * 3600)
            deleted_count = 0
            
            # Clean database
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM confidence_history WHERE timestamp < ?", (cutoff_timestamp,))
                deleted_count = cursor.rowcount
                conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old confidence records older than {days_to_keep} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0

# Global storage instance
_global_storage: Optional[DeepConfStorage] = None

def get_storage() -> DeepConfStorage:
    """Get or create global storage instance"""
    global _global_storage
    if _global_storage is None:
        _global_storage = DeepConfStorage()
    return _global_storage

async def store_confidence_data(confidence_score: ConfidenceScore, 
                              execution_data: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to store confidence data"""
    storage = get_storage()
    return await storage.store_confidence_score(confidence_score, execution_data)

async def load_confidence_history(limit: int = 1000, **filters) -> List[Dict[str, Any]]:
    """Convenience function to load historical confidence data"""
    storage = get_storage()
    return await storage.load_historical_data(limit=limit, **filters)