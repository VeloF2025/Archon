"""
Usage Tracker Module
User behavior analytics and feature usage tracking
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import hashlib
import numpy as np
from scipy import stats

from .analytics_engine import AnalyticsEngine


class EventType(Enum):
    """Types of usage events"""
    PAGE_VIEW = "page_view"
    FEATURE_USE = "feature_use"
    API_CALL = "api_call"
    BUTTON_CLICK = "button_click"
    FORM_SUBMIT = "form_submit"
    ERROR = "error"
    LOGIN = "login"
    LOGOUT = "logout"
    SEARCH = "search"
    EXPORT = "export"
    IMPORT = "import"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"


class UserSegment(Enum):
    """User segmentation categories"""
    NEW_USER = "new_user"
    ACTIVE_USER = "active_user"
    POWER_USER = "power_user"
    CASUAL_USER = "casual_user"
    DORMANT_USER = "dormant_user"
    CHURNED_USER = "churned_user"
    RETURNING_USER = "returning_user"
    TRIAL_USER = "trial_user"
    PAID_USER = "paid_user"
    ENTERPRISE_USER = "enterprise_user"


class FeatureCategory(Enum):
    """Feature categories for tracking"""
    CORE = "core"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"
    PREMIUM = "premium"
    ADMIN = "admin"
    INTEGRATION = "integration"
    AUTOMATION = "automation"
    ANALYTICS = "analytics"
    COLLABORATION = "collaboration"
    SECURITY = "security"


@dataclass
class UsageEvent:
    """Individual usage event"""
    event_id: str
    event_type: EventType
    user_id: str
    session_id: str
    timestamp: datetime
    feature_name: Optional[str] = None
    page_url: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    device_info: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int = 0
    page_views: int = 0
    events_count: int = 0
    features_used: Set[str] = field(default_factory=set)
    errors_encountered: int = 0
    device_info: Dict[str, Any] = field(default_factory=dict)
    entry_page: Optional[str] = None
    exit_page: Optional[str] = None
    bounce: bool = False


@dataclass
class UserProfile:
    """User behavior profile"""
    user_id: str
    first_seen: datetime
    last_seen: datetime
    total_sessions: int = 0
    total_events: int = 0
    average_session_duration: float = 0.0
    favorite_features: List[str] = field(default_factory=list)
    usage_pattern: str = "casual"  # casual, regular, power
    segment: UserSegment = UserSegment.NEW_USER
    lifetime_value: float = 0.0
    churn_risk: float = 0.0
    engagement_score: float = 0.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    cohort: Optional[str] = None


@dataclass
class FeatureUsage:
    """Feature usage statistics"""
    feature_name: str
    category: FeatureCategory
    total_uses: int = 0
    unique_users: int = 0
    average_duration_ms: float = 0.0
    success_rate: float = 100.0
    error_rate: float = 0.0
    adoption_rate: float = 0.0
    retention_rate: float = 0.0
    usage_trend: str = "stable"  # growing, stable, declining
    peak_usage_time: Optional[datetime] = None
    user_segments: Dict[UserSegment, int] = field(default_factory=dict)


@dataclass
class UsageReport:
    """Comprehensive usage report"""
    report_id: str
    period_start: datetime
    period_end: datetime
    total_users: int
    active_users: int
    new_users: int
    returning_users: int
    total_sessions: int
    total_events: int
    average_session_duration: float
    bounce_rate: float
    feature_adoption: Dict[str, float]
    user_segments: Dict[UserSegment, int]
    top_features: List[Tuple[str, int]]
    user_flow: List[Dict[str, Any]]
    conversion_funnel: Dict[str, float]
    engagement_metrics: Dict[str, float]


class UsageTracker:
    """
    Comprehensive usage tracking and user behavior analytics
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics_engine = analytics_engine
        self.events: List[UsageEvent] = []
        self.sessions: Dict[str, UserSession] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.feature_usage: Dict[str, FeatureUsage] = {}
        self.usage_reports: List[UsageReport] = []
        
        # Real-time tracking
        self.active_sessions: Set[str] = set()
        self.event_buffer: List[UsageEvent] = []
        self.buffer_size = 1000
        
        # Analytics cache
        self.cache_ttl = timedelta(minutes=15)
        self.analytics_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        self._start_tracking()
    
    def _start_tracking(self):
        """Start background tracking tasks"""
        asyncio.create_task(self._process_event_buffer())
        asyncio.create_task(self._update_user_segments())
        asyncio.create_task(self._calculate_usage_metrics())
        asyncio.create_task(self._detect_usage_patterns())
    
    async def _process_event_buffer(self):
        """Process buffered events"""
        while True:
            try:
                if len(self.event_buffer) >= self.buffer_size:
                    await self._flush_event_buffer()
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                print(f"Event buffer processing error: {e}")
                await asyncio.sleep(10)
    
    async def _update_user_segments(self):
        """Update user segmentation"""
        while True:
            try:
                for user_profile in self.user_profiles.values():
                    user_profile.segment = self._calculate_user_segment(user_profile)
                
                await asyncio.sleep(3600)  # Update hourly
                
            except Exception as e:
                print(f"User segmentation error: {e}")
                await asyncio.sleep(3600)
    
    async def _calculate_usage_metrics(self):
        """Calculate usage metrics"""
        while True:
            try:
                await self._update_feature_metrics()
                await self._calculate_engagement_scores()
                await self._predict_churn_risk()
                
                await asyncio.sleep(1800)  # Calculate every 30 minutes
                
            except Exception as e:
                print(f"Usage metrics calculation error: {e}")
                await asyncio.sleep(1800)
    
    async def _detect_usage_patterns(self):
        """Detect usage patterns and anomalies"""
        while True:
            try:
                await self._analyze_user_flows()
                await self._identify_feature_correlations()
                await self._detect_usage_anomalies()
                
                await asyncio.sleep(7200)  # Analyze every 2 hours
                
            except Exception as e:
                print(f"Pattern detection error: {e}")
                await asyncio.sleep(7200)
    
    async def track_event(self, event_type: EventType, user_id: str,
                         session_id: str, feature_name: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> UsageEvent:
        """Track a usage event"""
        import uuid
        
        event = UsageEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            feature_name=feature_name,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.event_buffer.append(event)
        
        # Update session
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.events_count += 1
            if feature_name:
                session.features_used.add(feature_name)
            if event_type == EventType.ERROR:
                session.errors_encountered += 1
        
        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            )
        else:
            self.user_profiles[user_id].last_seen = datetime.now()
            self.user_profiles[user_id].total_events += 1
        
        # Update feature usage
        if feature_name:
            if feature_name not in self.feature_usage:
                self.feature_usage[feature_name] = FeatureUsage(
                    feature_name=feature_name,
                    category=self._categorize_feature(feature_name)
                )
            self.feature_usage[feature_name].total_uses += 1
        
        # Record metric
        await self.analytics_engine.record_metric(
            f"usage.{event_type.value}",
            1,
            tags={"user_id": user_id, "feature": feature_name or "unknown"}
        )
        
        return event
    
    async def start_session(self, user_id: str,
                          device_info: Optional[Dict[str, Any]] = None) -> str:
        """Start a new user session"""
        import uuid
        
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            device_info=device_info or {}
        )
        
        self.sessions[session_id] = session
        self.active_sessions.add(session_id)
        
        # Update user profile
        if user_id in self.user_profiles:
            self.user_profiles[user_id].total_sessions += 1
        
        return session_id
    
    async def end_session(self, session_id: str):
        """End a user session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session.end_time = datetime.now()
        session.duration_seconds = int(
            (session.end_time - session.start_time).total_seconds()
        )
        
        # Check for bounce
        if session.page_views <= 1 and session.duration_seconds < 30:
            session.bounce = True
        
        self.active_sessions.discard(session_id)
        
        # Update user profile
        user_id = session.user_id
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            # Update average session duration
            total_duration = (profile.average_session_duration * 
                            (profile.total_sessions - 1) + 
                            session.duration_seconds)
            profile.average_session_duration = total_duration / profile.total_sessions
    
    async def track_page_view(self, user_id: str, session_id: str,
                            page_url: str, referrer: Optional[str] = None):
        """Track a page view"""
        event = await self.track_event(
            EventType.PAGE_VIEW,
            user_id,
            session_id,
            metadata={"page_url": page_url, "referrer": referrer}
        )
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.page_views += 1
            
            if not session.entry_page:
                session.entry_page = page_url
            session.exit_page = page_url
        
        return event
    
    async def track_feature_use(self, user_id: str, session_id: str,
                              feature_name: str, duration_ms: Optional[int] = None,
                              success: bool = True):
        """Track feature usage"""
        event = await self.track_event(
            EventType.FEATURE_USE,
            user_id,
            session_id,
            feature_name,
            metadata={"duration_ms": duration_ms, "success": success}
        )
        
        if feature_name in self.feature_usage:
            feature = self.feature_usage[feature_name]
            
            # Update duration
            if duration_ms:
                total_duration = feature.average_duration_ms * feature.total_uses
                feature.average_duration_ms = (total_duration + duration_ms) / (feature.total_uses + 1)
            
            # Update success rate
            if not success:
                feature.error_rate = ((feature.error_rate * feature.total_uses + 1) / 
                                     (feature.total_uses + 1))
                feature.success_rate = 100 - feature.error_rate
        
        return event
    
    async def get_user_journey(self, user_id: str,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get user journey/flow"""
        user_events = [e for e in self.events if e.user_id == user_id]
        user_events.sort(key=lambda x: x.timestamp)
        
        journey = []
        for event in user_events[-limit:]:
            journey.append({
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "feature": event.feature_name,
                "metadata": event.metadata
            })
        
        return journey
    
    async def get_feature_adoption(self) -> Dict[str, float]:
        """Calculate feature adoption rates"""
        total_users = len(self.user_profiles)
        if total_users == 0:
            return {}
        
        adoption = {}
        
        for feature_name, usage in self.feature_usage.items():
            # Count unique users who used this feature
            unique_users = len(set(
                e.user_id for e in self.events 
                if e.feature_name == feature_name
            ))
            
            adoption[feature_name] = (unique_users / total_users) * 100
            usage.adoption_rate = adoption[feature_name]
        
        return adoption
    
    async def get_user_segments(self) -> Dict[UserSegment, List[str]]:
        """Get users by segment"""
        segments = defaultdict(list)
        
        for user_id, profile in self.user_profiles.items():
            segments[profile.segment].append(user_id)
        
        return dict(segments)
    
    async def get_conversion_funnel(self, funnel_steps: List[str]) -> Dict[str, float]:
        """Calculate conversion funnel"""
        funnel = {}
        previous_users = set(self.user_profiles.keys())
        
        for step in funnel_steps:
            # Find users who completed this step
            step_users = set(
                e.user_id for e in self.events
                if e.feature_name == step or 
                   e.metadata.get("page_url") == step
            )
            
            # Calculate conversion rate
            if previous_users:
                conversion_rate = (len(step_users & previous_users) / 
                                 len(previous_users)) * 100
            else:
                conversion_rate = 0
            
            funnel[step] = conversion_rate
            previous_users = step_users
        
        return funnel
    
    async def get_engagement_metrics(self) -> Dict[str, float]:
        """Calculate engagement metrics"""
        metrics = {}
        
        # Daily Active Users (DAU)
        today = datetime.now().date()
        dau = len([p for p in self.user_profiles.values()
                  if p.last_seen.date() == today])
        metrics["dau"] = dau
        
        # Weekly Active Users (WAU)
        week_ago = datetime.now() - timedelta(days=7)
        wau = len([p for p in self.user_profiles.values()
                  if p.last_seen >= week_ago])
        metrics["wau"] = wau
        
        # Monthly Active Users (MAU)
        month_ago = datetime.now() - timedelta(days=30)
        mau = len([p for p in self.user_profiles.values()
                  if p.last_seen >= month_ago])
        metrics["mau"] = mau
        
        # Stickiness (DAU/MAU)
        metrics["stickiness"] = (dau / mau * 100) if mau > 0 else 0
        
        # Average session duration
        if self.sessions:
            durations = [s.duration_seconds for s in self.sessions.values()
                        if s.duration_seconds > 0]
            metrics["avg_session_duration"] = np.mean(durations) if durations else 0
        else:
            metrics["avg_session_duration"] = 0
        
        # Bounce rate
        bounced = len([s for s in self.sessions.values() if s.bounce])
        total = len(self.sessions)
        metrics["bounce_rate"] = (bounced / total * 100) if total > 0 else 0
        
        # Feature adoption rate
        if self.feature_usage:
            adoption_rates = [f.adoption_rate for f in self.feature_usage.values()]
            metrics["avg_feature_adoption"] = np.mean(adoption_rates)
        else:
            metrics["avg_feature_adoption"] = 0
        
        return metrics
    
    async def generate_usage_report(self, start_date: datetime,
                                  end_date: datetime) -> UsageReport:
        """Generate comprehensive usage report"""
        import uuid
        
        # Filter events for period
        period_events = [e for e in self.events
                        if start_date <= e.timestamp <= end_date]
        
        # Calculate metrics
        unique_users = set(e.user_id for e in period_events)
        total_users = len(unique_users)
        
        # Active users (had > 1 session)
        user_sessions = defaultdict(int)
        for session in self.sessions.values():
            if start_date <= session.start_time <= end_date:
                user_sessions[session.user_id] += 1
        active_users = len([u for u, count in user_sessions.items() if count > 1])
        
        # New vs returning users
        new_users = len([p for p in self.user_profiles.values()
                        if start_date <= p.first_seen <= end_date])
        returning_users = total_users - new_users
        
        # Feature usage
        feature_counts = Counter(e.feature_name for e in period_events
                                if e.feature_name)
        top_features = feature_counts.most_common(10)
        
        # User segments
        segment_counts = Counter(self.user_profiles[uid].segment
                               for uid in unique_users
                               if uid in self.user_profiles)
        
        # Calculate other metrics
        engagement_metrics = await self.get_engagement_metrics()
        feature_adoption = await self.get_feature_adoption()
        
        report = UsageReport(
            report_id=str(uuid.uuid4()),
            period_start=start_date,
            period_end=end_date,
            total_users=total_users,
            active_users=active_users,
            new_users=new_users,
            returning_users=returning_users,
            total_sessions=len([s for s in self.sessions.values()
                               if start_date <= s.start_time <= end_date]),
            total_events=len(period_events),
            average_session_duration=engagement_metrics.get("avg_session_duration", 0),
            bounce_rate=engagement_metrics.get("bounce_rate", 0),
            feature_adoption=feature_adoption,
            user_segments=dict(segment_counts),
            top_features=top_features,
            user_flow=[],  # Would be populated by flow analysis
            conversion_funnel={},  # Would be populated based on defined funnel
            engagement_metrics=engagement_metrics
        )
        
        self.usage_reports.append(report)
        return report
    
    def _calculate_user_segment(self, profile: UserProfile) -> UserSegment:
        """Calculate user segment based on behavior"""
        days_since_first = (datetime.now() - profile.first_seen).days
        days_since_last = (datetime.now() - profile.last_seen).days
        
        if days_since_first <= 7:
            return UserSegment.NEW_USER
        elif days_since_last > 30:
            return UserSegment.CHURNED_USER if days_since_last > 90 else UserSegment.DORMANT_USER
        elif profile.total_sessions > 20 and profile.average_session_duration > 600:
            return UserSegment.POWER_USER
        elif profile.total_sessions > 5:
            return UserSegment.ACTIVE_USER
        else:
            return UserSegment.CASUAL_USER
    
    def _categorize_feature(self, feature_name: str) -> FeatureCategory:
        """Categorize feature based on name"""
        feature_lower = feature_name.lower()
        
        if any(word in feature_lower for word in ["admin", "manage", "config"]):
            return FeatureCategory.ADMIN
        elif any(word in feature_lower for word in ["api", "integrate", "connect"]):
            return FeatureCategory.INTEGRATION
        elif any(word in feature_lower for word in ["auto", "schedule", "workflow"]):
            return FeatureCategory.AUTOMATION
        elif any(word in feature_lower for word in ["report", "dashboard", "metric"]):
            return FeatureCategory.ANALYTICS
        elif any(word in feature_lower for word in ["share", "collab", "team"]):
            return FeatureCategory.COLLABORATION
        elif any(word in feature_lower for word in ["security", "auth", "encrypt"]):
            return FeatureCategory.SECURITY
        elif any(word in feature_lower for word in ["beta", "preview", "experimental"]):
            return FeatureCategory.EXPERIMENTAL
        elif any(word in feature_lower for word in ["premium", "pro", "enterprise"]):
            return FeatureCategory.PREMIUM
        elif any(word in feature_lower for word in ["advanced", "complex", "power"]):
            return FeatureCategory.ADVANCED
        else:
            return FeatureCategory.CORE
    
    async def _flush_event_buffer(self):
        """Flush event buffer to storage"""
        if not self.event_buffer:
            return
        
        # Move events from buffer to main storage
        self.events.extend(self.event_buffer)
        
        # Keep only recent events (last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.events = [e for e in self.events if e.timestamp >= cutoff]
        
        # Clear buffer
        self.event_buffer.clear()
    
    async def _update_feature_metrics(self):
        """Update feature usage metrics"""
        for feature_name, usage in self.feature_usage.items():
            # Calculate unique users
            unique_users = len(set(
                e.user_id for e in self.events
                if e.feature_name == feature_name
            ))
            usage.unique_users = unique_users
            
            # Calculate retention rate
            week_ago = datetime.now() - timedelta(days=7)
            recent_users = set(
                e.user_id for e in self.events
                if e.feature_name == feature_name and e.timestamp >= week_ago
            )
            
            two_weeks_ago = datetime.now() - timedelta(days=14)
            previous_users = set(
                e.user_id for e in self.events
                if e.feature_name == feature_name and 
                   two_weeks_ago <= e.timestamp < week_ago
            )
            
            if previous_users:
                retained = len(recent_users & previous_users)
                usage.retention_rate = (retained / len(previous_users)) * 100
            
            # Determine usage trend
            recent_count = len([e for e in self.events
                              if e.feature_name == feature_name and
                                 e.timestamp >= week_ago])
            previous_count = len([e for e in self.events
                                if e.feature_name == feature_name and
                                   two_weeks_ago <= e.timestamp < week_ago])
            
            if previous_count > 0:
                growth = ((recent_count - previous_count) / previous_count) * 100
                if growth > 20:
                    usage.usage_trend = "growing"
                elif growth < -20:
                    usage.usage_trend = "declining"
                else:
                    usage.usage_trend = "stable"
    
    async def _calculate_engagement_scores(self):
        """Calculate user engagement scores"""
        for profile in self.user_profiles.values():
            # Factors for engagement score
            recency_score = self._calculate_recency_score(profile.last_seen)
            frequency_score = min(100, profile.total_sessions * 2)
            duration_score = min(100, profile.average_session_duration / 10)
            feature_score = min(100, len(profile.favorite_features) * 10)
            
            # Weighted average
            profile.engagement_score = (
                recency_score * 0.3 +
                frequency_score * 0.3 +
                duration_score * 0.2 +
                feature_score * 0.2
            )
    
    def _calculate_recency_score(self, last_seen: datetime) -> float:
        """Calculate recency score"""
        days_ago = (datetime.now() - last_seen).days
        
        if days_ago == 0:
            return 100
        elif days_ago <= 7:
            return 80
        elif days_ago <= 14:
            return 60
        elif days_ago <= 30:
            return 40
        elif days_ago <= 60:
            return 20
        else:
            return 0
    
    async def _predict_churn_risk(self):
        """Predict user churn risk"""
        for profile in self.user_profiles.values():
            # Simple churn risk calculation
            days_inactive = (datetime.now() - profile.last_seen).days
            
            if days_inactive > 30:
                profile.churn_risk = 90
            elif days_inactive > 14:
                profile.churn_risk = 70
            elif days_inactive > 7:
                profile.churn_risk = 50
            elif profile.engagement_score < 30:
                profile.churn_risk = 60
            elif profile.total_sessions < 3:
                profile.churn_risk = 40
            else:
                profile.churn_risk = max(0, 100 - profile.engagement_score)
    
    async def _analyze_user_flows(self):
        """Analyze common user flows"""
        # Would implement sequential pattern mining
        pass
    
    async def _identify_feature_correlations(self):
        """Identify correlations between feature usage"""
        # Would implement correlation analysis
        pass
    
    async def _detect_usage_anomalies(self):
        """Detect anomalies in usage patterns"""
        # Would implement anomaly detection
        pass