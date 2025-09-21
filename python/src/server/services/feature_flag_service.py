"""
Feature Flag Service for Archon

Provides runtime feature toggle capabilities for gradual rollouts,
A/B testing, and safe feature deployment without code changes.

This service enables:
- Dynamic feature enabling/disabling
- User-specific feature targeting
- Percentage-based rollouts
- A/B testing experiments
- Feature flag analytics
"""

import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"          # Simple on/off toggle
    PERCENTAGE = "percentage"     # Percentage-based rollout
    USER_LIST = "user_list"       # Specific user targeting
    VARIANT = "variant"           # A/B testing variants
    SCHEDULE = "schedule"         # Time-based activation


class FlagStatus(Enum):
    """Feature flag lifecycle status"""
    DRAFT = "draft"               # Not yet active
    ACTIVE = "active"             # Currently in use
    ARCHIVED = "archived"         # No longer in use
    EMERGENCY_OFF = "emergency"   # Disabled due to issues


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    key: str                      # Unique identifier
    name: str                     # Human-readable name
    description: str              # What this flag controls
    flag_type: FlagType          # Type of flag
    status: FlagStatus           # Current status
    default_value: Any           # Default when not targeted
    
    # Targeting configuration
    enabled_for_all: bool = False
    percentage: float = 0.0       # For percentage rollouts
    user_ids: List[str] = None   # For user targeting
    variants: Dict[str, Any] = None  # For A/B testing
    
    # Scheduling
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = None
    updated_at: datetime = None
    created_by: str = None
    tags: List[str] = None
    
    # Analytics
    evaluation_count: int = 0
    last_evaluated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.user_ids is None:
            self.user_ids = []
        if self.variants is None:
            self.variants = {}
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class FeatureFlagService:
    """
    Service for managing feature flags in Archon
    
    This provides a simple in-memory implementation that can be
    extended with database persistence or external services like
    LaunchDarkly or Unleash.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the feature flag service
        
        Args:
            config_file: Optional JSON file with flag configurations
        """
        self._flags: Dict[str, FeatureFlag] = {}
        self._evaluation_cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_clear = datetime.utcnow()
        
        # Load configuration if provided
        if config_file and Path(config_file).exists():
            self._load_config(config_file)
        else:
            self._initialize_default_flags()
    
    def _initialize_default_flags(self):
        """Initialize with default Archon feature flags"""
        default_flags = [
            FeatureFlag(
                key="youtube_integration",
                name="YouTube Integration",
                description="Enable YouTube video processing and knowledge extraction",
                flag_type=FlagType.BOOLEAN,
                status=FlagStatus.ACTIVE,
                default_value=True,
                enabled_for_all=True,
                tags=["knowledge", "video", "new"]
            ),
            FeatureFlag(
                key="advanced_tdd_enforcement",
                name="Advanced TDD Enforcement",
                description="Strict test-first development enforcement with gaming detection",
                flag_type=FlagType.PERCENTAGE,
                status=FlagStatus.ACTIVE,
                default_value=False,
                percentage=100.0,  # Fully rolled out
                tags=["quality", "testing"]
            ),
            FeatureFlag(
                key="deepconf_scoring",
                name="DeepConf Confidence Scoring",
                description="Enable advanced confidence scoring for AI responses",
                flag_type=FlagType.BOOLEAN,
                status=FlagStatus.ACTIVE,
                default_value=True,
                enabled_for_all=True,
                tags=["ai", "confidence"]
            ),
            FeatureFlag(
                key="parallel_agent_execution",
                name="Parallel Agent Execution",
                description="Enable parallel execution of independent agent tasks",
                flag_type=FlagType.PERCENTAGE,
                status=FlagStatus.ACTIVE,
                default_value=False,
                percentage=75.0,  # 75% rollout
                tags=["performance", "agents"]
            ),
            FeatureFlag(
                key="ui_theme",
                name="UI Theme Selection",
                description="Control which UI theme variant users see",
                flag_type=FlagType.VARIANT,
                status=FlagStatus.ACTIVE,
                default_value="modern",
                variants={
                    "classic": 0.2,    # 20% of users
                    "modern": 0.6,     # 60% of users
                    "experimental": 0.2 # 20% of users
                },
                tags=["ui", "experiment"]
            ),
            FeatureFlag(
                key="beta_features",
                name="Beta Features Access",
                description="Enable access to beta features for specific users",
                flag_type=FlagType.USER_LIST,
                status=FlagStatus.ACTIVE,
                default_value=False,
                user_ids=["beta_tester_1", "beta_tester_2"],
                tags=["beta", "experimental"]
            ),
            FeatureFlag(
                key="holiday_mode",
                name="Holiday Mode",
                description="Special holiday theme and features",
                flag_type=FlagType.SCHEDULE,
                status=FlagStatus.DRAFT,
                default_value=False,
                start_date=datetime(2025, 12, 20),
                end_date=datetime(2026, 1, 5),
                tags=["seasonal", "ui"]
            )
        ]
        
        for flag in default_flags:
            self._flags[flag.key] = flag
    
    def _load_config(self, config_file: str):
        """Load flags from configuration file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                for flag_data in config.get('flags', []):
                    flag = FeatureFlag(**flag_data)
                    self._flags[flag.key] = flag
            logger.info(f"Loaded {len(self._flags)} feature flags from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load feature flags from {config_file}: {e}")
            self._initialize_default_flags()
    
    def is_enabled(self, 
                   flag_key: str, 
                   user_id: Optional[str] = None,
                   attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a feature flag is enabled
        
        Args:
            flag_key: The feature flag key
            user_id: Optional user identifier for targeting
            attributes: Optional user attributes for advanced targeting
            
        Returns:
            Boolean indicating if the feature is enabled
        """
        flag = self._flags.get(flag_key)
        if not flag:
            logger.warning(f"Feature flag '{flag_key}' not found")
            return False
        
        if flag.status != FlagStatus.ACTIVE:
            return False
        
        # Check schedule
        if flag.flag_type == FlagType.SCHEDULE:
            now = datetime.utcnow()
            if flag.start_date and now < flag.start_date:
                return False
            if flag.end_date and now > flag.end_date:
                return False
            return True
        
        # Check if enabled for all
        if flag.enabled_for_all:
            return True
        
        # Check user list
        if flag.flag_type == FlagType.USER_LIST and user_id:
            return user_id in flag.user_ids
        
        # Check percentage rollout
        if flag.flag_type == FlagType.PERCENTAGE and user_id:
            return self._check_percentage(flag.key, user_id, flag.percentage)
        
        # Boolean flag
        if flag.flag_type == FlagType.BOOLEAN:
            return flag.default_value
        
        # Variant flag - just check if user gets any variant
        if flag.flag_type == FlagType.VARIANT and user_id:
            variant = self.get_variant(flag_key, user_id)
            return variant is not None
        
        return flag.default_value
    
    def get_variant(self, 
                    flag_key: str, 
                    user_id: str,
                    attributes: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get the variant for an A/B test flag
        
        Args:
            flag_key: The feature flag key
            user_id: User identifier for consistent bucketing
            attributes: Optional user attributes
            
        Returns:
            The variant name or None if not applicable
        """
        flag = self._flags.get(flag_key)
        if not flag or flag.flag_type != FlagType.VARIANT:
            return None
        
        if flag.status != FlagStatus.ACTIVE:
            return None
        
        if not flag.variants:
            return flag.default_value
        
        # Use consistent hashing for variant assignment
        hash_input = f"{flag_key}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) / 100.0
        
        cumulative = 0.0
        for variant, percentage in flag.variants.items():
            cumulative += percentage
            if bucket < cumulative:
                return variant
        
        return flag.default_value
    
    def _check_percentage(self, flag_key: str, user_id: str, percentage: float) -> bool:
        """Check if user falls within percentage rollout"""
        hash_input = f"{flag_key}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        user_bucket = (hash_value % 10000) / 100.0
        return user_bucket < percentage
    
    async def get_all_flags(self, 
                           user_id: Optional[str] = None,
                           include_archived: bool = False) -> Dict[str, Any]:
        """
        Get all feature flags and their values
        
        Args:
            user_id: Optional user for evaluation
            include_archived: Include archived flags
            
        Returns:
            Dictionary of flag keys to evaluated values
        """
        result = {}
        for key, flag in self._flags.items():
            if not include_archived and flag.status == FlagStatus.ARCHIVED:
                continue
            
            if flag.flag_type == FlagType.VARIANT:
                result[key] = self.get_variant(key, user_id) if user_id else flag.default_value
            else:
                result[key] = self.is_enabled(key, user_id)
        
        return result
    
    async def create_flag(self, flag: FeatureFlag) -> bool:
        """Create a new feature flag"""
        if flag.key in self._flags:
            logger.warning(f"Feature flag '{flag.key}' already exists")
            return False
        
        self._flags[flag.key] = flag
        logger.info(f"Created feature flag '{flag.key}'")
        return True
    
    async def update_flag(self, flag_key: str, updates: Dict[str, Any]) -> bool:
        """Update an existing feature flag"""
        if flag_key not in self._flags:
            logger.warning(f"Feature flag '{flag_key}' not found")
            return False
        
        flag = self._flags[flag_key]
        for key, value in updates.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        
        flag.updated_at = datetime.utcnow()
        self._clear_cache()
        logger.info(f"Updated feature flag '{flag_key}'")
        return True
    
    async def delete_flag(self, flag_key: str) -> bool:
        """Delete a feature flag (archives it)"""
        if flag_key not in self._flags:
            return False
        
        self._flags[flag_key].status = FlagStatus.ARCHIVED
        logger.info(f"Archived feature flag '{flag_key}'")
        return True
    
    async def emergency_disable(self, flag_key: str) -> bool:
        """Emergency disable a feature flag"""
        if flag_key not in self._flags:
            return False
        
        self._flags[flag_key].status = FlagStatus.EMERGENCY_OFF
        self._clear_cache()
        logger.warning(f"Emergency disabled feature flag '{flag_key}'")
        return True
    
    def _clear_cache(self):
        """Clear the evaluation cache"""
        self._evaluation_cache.clear()
        self._last_cache_clear = datetime.utcnow()
    
    def get_flag_config(self, flag_key: str) -> Optional[Dict[str, Any]]:
        """Get the full configuration for a flag"""
        flag = self._flags.get(flag_key)
        if not flag:
            return None
        return asdict(flag)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics for all flags"""
        analytics = {
            "total_flags": len(self._flags),
            "active_flags": sum(1 for f in self._flags.values() if f.status == FlagStatus.ACTIVE),
            "flags_by_type": {},
            "flags_by_status": {},
            "most_evaluated": [],
            "recently_updated": []
        }
        
        # Count by type and status
        for flag in self._flags.values():
            flag_type = flag.flag_type.value
            status = flag.status.value
            
            analytics["flags_by_type"][flag_type] = analytics["flags_by_type"].get(flag_type, 0) + 1
            analytics["flags_by_status"][status] = analytics["flags_by_status"].get(status, 0) + 1
        
        # Most evaluated flags
        sorted_by_evaluation = sorted(
            self._flags.values(), 
            key=lambda f: f.evaluation_count, 
            reverse=True
        )[:5]
        analytics["most_evaluated"] = [
            {"key": f.key, "count": f.evaluation_count} 
            for f in sorted_by_evaluation
        ]
        
        # Recently updated flags
        sorted_by_update = sorted(
            self._flags.values(), 
            key=lambda f: f.updated_at, 
            reverse=True
        )[:5]
        analytics["recently_updated"] = [
            {"key": f.key, "updated_at": f.updated_at.isoformat()} 
            for f in sorted_by_update
        ]
        
        return analytics


# Singleton instance
_feature_flag_service: Optional[FeatureFlagService] = None


def get_feature_flag_service() -> FeatureFlagService:
    """Get or create the feature flag service singleton"""
    global _feature_flag_service
    if _feature_flag_service is None:
        # Try to load from config file if it exists
        config_file = Path(__file__).parent.parent / "config" / "feature_flags.json"
        _feature_flag_service = FeatureFlagService(
            str(config_file) if config_file.exists() else None
        )
    return _feature_flag_service


# Convenience functions for easy usage
def is_feature_enabled(flag_key: str, user_id: Optional[str] = None) -> bool:
    """Quick check if a feature is enabled"""
    service = get_feature_flag_service()
    return service.is_enabled(flag_key, user_id)


def get_feature_variant(flag_key: str, user_id: str) -> Optional[str]:
    """Quick get variant for A/B testing"""
    service = get_feature_flag_service()
    return service.get_variant(flag_key, user_id)