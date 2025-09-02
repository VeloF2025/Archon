"""
DeepConf Supabase Storage Integration
=====================================

Enhanced storage backend that integrates with the new Phase 7 DeepConf database schema
in Supabase. Provides high-performance confidence scoring with proper database integration.

This replaces the SQLite-based storage with a proper PostgreSQL backend while maintaining
backward compatibility with the existing DeepConf API.

Author: Database Architect via Claude Code
Version: 2.0.0 (Phase 7 Integration)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
import threading
from contextlib import asynccontextmanager

from .types import ConfidenceScore

logger = logging.getLogger(__name__)

class SupabaseDeepConfStorage:
    """
    Supabase-backed storage for DeepConf confidence data
    
    Features:
    - PostgreSQL database with optimized indexing
    - Real-time confidence scoring integration
    - Performance metrics tracking
    - Calibration data collection
    - Horizontal scaling support
    - Advanced analytics queries
    """
    
    def __init__(self, supabase_client=None):
        """Initialize Supabase storage"""
        self.client = supabase_client
        self._lock = threading.RLock()
        
        # Lazy initialization of Supabase client
        if not self.client:
            self._init_supabase_client()
        
        # Configuration
        self.batch_size = int(os.getenv('DEEPCONF_BATCH_SIZE', '100'))
        self.enable_performance_tracking = os.getenv('DEEPCONF_TRACK_PERFORMANCE', 'true').lower() == 'true'
        
        logger.info("DeepConf Supabase storage initialized")
    
    def _init_supabase_client(self):
        """Initialize Supabase client if not provided"""
        try:
            from supabase import create_client
            
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
            
            self.client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    async def store_confidence_score(self, confidence_score: ConfidenceScore, 
                                   execution_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store confidence score in Supabase archon_confidence_scores table
        
        Args:
            confidence_score: ConfidenceScore object to store
            execution_data: Optional execution metadata
            
        Returns:
            bool: True if successfully stored
        """
        try:
            start_time = time.time()
            
            with self._lock:
                # Prepare confidence score data for Supabase
                confidence_data = self._prepare_confidence_data(confidence_score, execution_data)
                
                # Insert into archon_confidence_scores table
                result = self.client.table('archon_confidence_scores').insert(confidence_data).execute()
                
                # Track performance metrics if enabled
                if self.enable_performance_tracking and execution_data:
                    await self._track_performance_metrics(confidence_score, execution_data, start_time)
                
                logger.debug(f"Stored confidence score for task {confidence_score.task_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store confidence score for task {confidence_score.task_id}: {e}")
            # Fallback to local storage if configured
            return await self._fallback_storage(confidence_score, execution_data)
    
    def _prepare_confidence_data(self, confidence_score: ConfidenceScore, 
                               execution_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare confidence data for Supabase insertion"""
        
        # Generate request_id if not provided
        request_id = str(uuid.uuid4())
        if execution_data and 'request_id' in execution_data:
            request_id = execution_data['request_id']
        
        # Map ConfidenceScore to Supabase schema
        data = {
            'request_id': request_id,
            'factual_confidence': round(confidence_score.factual_confidence, 4),
            'reasoning_confidence': round(confidence_score.reasoning_confidence, 4),
            'contextual_relevance': round(confidence_score.contextual_confidence, 4),
            'uncertainty_lower': round(confidence_score.uncertainty_bounds[0], 4),
            'uncertainty_upper': round(confidence_score.uncertainty_bounds[1], 4),
            'model_consensus': self._create_model_consensus(confidence_score),
            'model_version': confidence_score.model_source or 'unknown',
            'request_type': self._determine_request_type(execution_data),
            'prompt_hash': self._calculate_prompt_hash(execution_data)
        }
        
        # Add optional fields if available
        if execution_data:
            if 'user_id' in execution_data:
                data['user_id'] = execution_data['user_id']
            if 'session_id' in execution_data:
                data['session_id'] = execution_data['session_id']
            if 'temperature' in execution_data:
                data['temperature'] = execution_data['temperature']
            if 'max_tokens' in execution_data:
                data['max_tokens'] = execution_data['max_tokens']
        
        return data
    
    def _create_model_consensus(self, confidence_score: ConfidenceScore) -> Dict[str, Any]:
        """Create model consensus JSON from confidence score"""
        consensus = {
            'primary_model': confidence_score.model_source or 'unknown',
            'confidence_factors': confidence_score.confidence_factors,
            'primary_factors': confidence_score.primary_factors,
            'gaming_score': confidence_score.gaming_detection_score,
            'calibration_applied': confidence_score.calibration_applied
        }
        
        # Add reasoning if available
        if confidence_score.confidence_reasoning:
            consensus['reasoning'] = confidence_score.confidence_reasoning[:500]  # Limit length
        
        return consensus
    
    def _determine_request_type(self, execution_data: Optional[Dict[str, Any]] = None) -> str:
        """Determine request type from execution data"""
        if not execution_data:
            return 'general'
        
        # Map from common execution data fields
        if 'agent_type' in execution_data:
            return execution_data['agent_type']
        elif 'domain' in execution_data:
            return execution_data['domain']
        elif 'request_type' in execution_data:
            return execution_data['request_type']
        else:
            return 'general'
    
    def _calculate_prompt_hash(self, execution_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Calculate SHA-256 hash of prompt for caching"""
        if not execution_data or 'user_prompt' not in execution_data:
            return None
        
        import hashlib
        prompt = execution_data['user_prompt']
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    async def _track_performance_metrics(self, confidence_score: ConfidenceScore,
                                       execution_data: Dict[str, Any], 
                                       start_time: float):
        """Track performance metrics in archon_performance_metrics table"""
        try:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate performance metrics
            metrics_data = {
                'token_efficiency': self._calculate_token_efficiency(execution_data),
                'response_time_ms': response_time_ms,
                'confidence_accuracy': self._estimate_confidence_accuracy(confidence_score),
                'hallucination_rate': confidence_score.gaming_detection_score,
                'system_load': self._get_system_load(),
                'request_type': self._determine_request_type(execution_data),
                'model_version': confidence_score.model_source,
                'endpoint': execution_data.get('endpoint', '/deepconf/confidence'),
                'metadata': {
                    'agent_type': execution_data.get('agent_type'),
                    'domain': execution_data.get('domain'),
                    'complexity': execution_data.get('complexity'),
                    'task_id': confidence_score.task_id
                }
            }
            
            # Add optional performance fields
            if 'memory_usage_mb' in execution_data:
                metrics_data['memory_usage_mb'] = execution_data['memory_usage_mb']
            if 'cpu_usage_percent' in execution_data:
                metrics_data['cpu_usage_percent'] = execution_data['cpu_usage_percent']
            
            # Insert performance metrics
            result = self.client.table('archon_performance_metrics').insert(metrics_data).execute()
            
        except Exception as e:
            logger.warning(f"Failed to track performance metrics: {e}")
    
    def _calculate_token_efficiency(self, execution_data: Dict[str, Any]) -> float:
        """Calculate token efficiency ratio"""
        total_tokens = execution_data.get('total_tokens', 0)
        useful_tokens = execution_data.get('useful_tokens', total_tokens)
        
        if total_tokens == 0:
            return 1.0
        
        return min(1.0, max(0.0, useful_tokens / total_tokens))
    
    def _estimate_confidence_accuracy(self, confidence_score: ConfidenceScore) -> float:
        """Estimate confidence accuracy based on calibration"""
        if confidence_score.calibration_applied:
            # Use calibrated confidence as accuracy estimate
            return confidence_score.overall_confidence
        else:
            # Conservative estimate for uncalibrated scores
            return max(0.5, confidence_score.overall_confidence * 0.8)
    
    def _get_system_load(self) -> float:
        """Get current system load factor"""
        try:
            import psutil
            return min(2.0, psutil.cpu_percent() / 100.0)
        except ImportError:
            return 0.5  # Default moderate load
        except Exception:
            return 1.0  # Default high load on error
    
    async def load_historical_data(self, limit: int = 1000, 
                                 agent_id: Optional[str] = None,
                                 domain: Optional[str] = None,
                                 phase: Optional[str] = None,
                                 since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Load historical confidence data from Supabase
        
        Args:
            limit: Maximum number of records to return
            agent_id: Filter by specific agent (maps to model_version)
            domain: Filter by domain (maps to request_type)
            phase: Filter by phase (maps to request_type)
            since_timestamp: Only return data after this timestamp
            
        Returns:
            List[Dict[str, Any]]: Historical confidence data
        """
        try:
            # Build query with filters
            query = self.client.table('archon_confidence_scores').select(
                'id, request_id, factual_confidence, reasoning_confidence, contextual_relevance, '
                'uncertainty_lower, uncertainty_upper, overall_confidence, model_consensus, '
                'model_version, request_type, user_id, session_id, created_at'
            )
            
            # Apply filters
            if agent_id:
                query = query.eq('model_version', agent_id)
            if domain:
                query = query.eq('request_type', domain)
            if phase:
                query = query.eq('request_type', phase)
            if since_timestamp:
                since_datetime = datetime.fromtimestamp(since_timestamp, tz=timezone.utc)
                query = query.gte('created_at', since_datetime.isoformat())
            
            # Execute query with limit and ordering
            result = query.order('created_at', desc=True).limit(limit).execute()
            
            # Transform data to match legacy format
            historical_data = []
            for record in result.data:
                transformed_record = self._transform_to_legacy_format(record)
                historical_data.append(transformed_record)
            
            logger.info(f"Loaded {len(historical_data)} historical confidence records from Supabase")
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to load historical data from Supabase: {e}")
            # Fallback to legacy storage if available
            return await self._fallback_load_historical_data(limit, agent_id, domain, phase, since_timestamp)
    
    def _transform_to_legacy_format(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Supabase record to legacy format for backward compatibility"""
        # Extract model consensus data
        consensus = record.get('model_consensus', {})
        
        return {
            'timestamp': datetime.fromisoformat(record['created_at'].replace('Z', '+00:00')).timestamp(),
            'task_id': record.get('request_id', str(record['id'])),
            'agent_id': record.get('model_version', 'unknown'),
            'domain': record.get('request_type', 'general'),
            'complexity': 'moderate',  # Not stored in new schema
            'phase': 'production',      # Not stored in new schema
            'confidence_score': {
                'overall_confidence': float(record['overall_confidence']),
                'factual_confidence': float(record['factual_confidence']),
                'reasoning_confidence': float(record['reasoning_confidence']),
                'contextual_confidence': float(record['contextual_relevance']),
                'epistemic_uncertainty': 0.0,  # Calculated from bounds
                'aleatoric_uncertainty': 0.0,   # Calculated from bounds
                'uncertainty_bounds': (
                    float(record['uncertainty_lower']),
                    float(record['uncertainty_upper'])
                ),
                'confidence_factors': consensus.get('confidence_factors', {}),
                'primary_factors': consensus.get('primary_factors', []),
                'confidence_reasoning': consensus.get('reasoning', ''),
                'model_source': record.get('model_version', 'unknown'),
                'task_id': record.get('request_id', str(record['id'])),
                'timestamp': datetime.fromisoformat(record['created_at'].replace('Z', '+00:00')).timestamp(),
                'calibration_applied': consensus.get('calibration_applied', False),
                'gaming_detection_score': consensus.get('gaming_score', 0.0)
            },
            'execution_duration': 0.0,  # Would need to be tracked separately
            'success': True,            # Would need to be tracked separately
            'result_quality': float(record['overall_confidence']),  # Use overall confidence as proxy
            'source': 'supabase'
        }
    
    async def store_calibration_data(self, confidence_score_id: str, 
                                   actual_accuracy: float,
                                   validation_method: str = 'automated_test',
                                   validator_id: Optional[str] = None,
                                   feedback_score: Optional[int] = None,
                                   feedback_comments: Optional[str] = None) -> bool:
        """
        Store confidence calibration data for model improvement
        
        Args:
            confidence_score_id: ID of the confidence score being calibrated
            actual_accuracy: Measured actual accuracy (0.0-1.0)
            validation_method: Method used for validation
            validator_id: Optional validator ID
            feedback_score: Optional feedback score (1-5)
            feedback_comments: Optional feedback comments
            
        Returns:
            bool: True if successfully stored
        """
        try:
            calibration_data = {
                'confidence_score_id': confidence_score_id,
                'actual_accuracy': round(actual_accuracy, 4),
                'validation_method': validation_method
            }
            
            # Add optional fields
            if validator_id:
                calibration_data['validator_id'] = validator_id
            if feedback_score:
                calibration_data['feedback_score'] = feedback_score
            if feedback_comments:
                calibration_data['feedback_comments'] = feedback_comments
            
            # Insert calibration record
            result = self.client.table('archon_confidence_calibration').insert(calibration_data).execute()
            
            logger.debug(f"Stored calibration data for confidence score {confidence_score_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store calibration data: {e}")
            return False
    
    async def get_confidence_trends(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get confidence trends using the archon_confidence_trends view"""
        try:
            # Query the confidence trends view
            result = self.client.table('archon_confidence_trends').select('*').execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to get confidence trends: {e}")
            return []
    
    async def get_performance_dashboard(self, minutes_back: int = 60) -> List[Dict[str, Any]]:
        """Get performance metrics using the archon_performance_dashboard view"""
        try:
            # Query the performance dashboard view
            result = self.client.table('archon_performance_dashboard').select('*').execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to get performance dashboard: {e}")
            return []
    
    async def get_calibration_analysis(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get calibration analysis using the archon_calibration_analysis view"""
        try:
            # Query the calibration analysis view
            result = self.client.table('archon_calibration_analysis').select('*').execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to get calibration analysis: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics from Supabase tables"""
        try:
            # Get table counts
            confidence_count = self.client.table('archon_confidence_scores').select('count', count='exact').execute().count
            performance_count = self.client.table('archon_performance_metrics').select('count', count='exact').execute().count
            calibration_count = self.client.table('archon_confidence_calibration').select('count', count='exact').execute().count
            
            # Get date ranges
            oldest_result = self.client.table('archon_confidence_scores').select('created_at').order('created_at', desc=False).limit(1).execute()
            newest_result = self.client.table('archon_confidence_scores').select('created_at').order('created_at', desc=True).limit(1).execute()
            
            stats = {
                'storage_type': 'supabase',
                'total_confidence_records': confidence_count,
                'total_performance_records': performance_count,
                'total_calibration_records': calibration_count,
                'oldest_record': oldest_result.data[0]['created_at'] if oldest_result.data else None,
                'newest_record': newest_result.data[0]['created_at'] if newest_result.data else None,
                'database_status': 'connected'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                'storage_type': 'supabase',
                'database_status': 'error',
                'error': str(e)
            }
    
    async def _fallback_storage(self, confidence_score: ConfidenceScore, 
                              execution_data: Optional[Dict[str, Any]] = None) -> bool:
        """Fallback to legacy storage if Supabase fails"""
        try:
            # Try to use legacy storage as fallback
            from .storage import DeepConfStorage
            fallback_storage = DeepConfStorage()
            return await fallback_storage.store_confidence_score(confidence_score, execution_data)
        except Exception as e:
            logger.error(f"Fallback storage also failed: {e}")
            return False
    
    async def _fallback_load_historical_data(self, limit: int, agent_id: Optional[str] = None,
                                           domain: Optional[str] = None, phase: Optional[str] = None,
                                           since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Fallback to legacy storage for historical data"""
        try:
            from .storage import DeepConfStorage
            fallback_storage = DeepConfStorage()
            return await fallback_storage.load_historical_data(limit, agent_id, domain, phase, since_timestamp)
        except Exception as e:
            logger.error(f"Fallback load also failed: {e}")
            return []

# Global storage instance
_global_supabase_storage: Optional[SupabaseDeepConfStorage] = None

def get_supabase_storage() -> SupabaseDeepConfStorage:
    """Get or create global Supabase storage instance"""
    global _global_supabase_storage
    if _global_supabase_storage is None:
        _global_supabase_storage = SupabaseDeepConfStorage()
    return _global_supabase_storage

async def store_confidence_data_supabase(confidence_score: ConfidenceScore, 
                                        execution_data: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to store confidence data in Supabase"""
    storage = get_supabase_storage()
    return await storage.store_confidence_score(confidence_score, execution_data)

async def load_confidence_history_supabase(limit: int = 1000, **filters) -> List[Dict[str, Any]]:
    """Convenience function to load historical confidence data from Supabase"""
    storage = get_supabase_storage()
    return await storage.load_historical_data(limit=limit, **filters)