"""
Global Knowledge Network for Phase 9 Autonomous Development Teams

This module implements a distributed knowledge network that connects autonomous development teams
across different projects, organizations, and domains, enabling global learning and collaboration
while preserving privacy and maintaining competitive advantages.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from uuid import uuid4, UUID
import json
import hashlib
import hmac
from pathlib import Path
import aiohttp

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """Types of knowledge shared in the network."""
    PATTERN = "pattern"
    ANTI_PATTERN = "anti_pattern"
    BEST_PRACTICE = "best_practice"
    SOLUTION_TEMPLATE = "solution_template"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    TOOL_RECOMMENDATION = "tool_recommendation"
    ARCHITECTURE_PATTERN = "architecture_pattern"
    PROCESS_IMPROVEMENT = "process_improvement"


class PrivacyLevel(str, Enum):
    """Privacy levels for shared knowledge."""
    PUBLIC = "public"           # Fully public, no anonymization
    ANONYMIZED = "anonymized"   # Anonymized but detailed
    AGGREGATED = "aggregated"   # Statistical aggregates only
    ENCRYPTED = "encrypted"     # Encrypted for specific partners
    PRIVATE = "private"         # Not shared externally


class NetworkRole(str, Enum):
    """Roles within the knowledge network."""
    CONTRIBUTOR = "contributor"     # Shares knowledge and consumes
    CONSUMER = "consumer"          # Only consumes knowledge
    VALIDATOR = "validator"        # Validates and curates knowledge
    COORDINATOR = "coordinator"    # Manages network operations
    RESEARCHER = "researcher"      # Analyzes network-wide patterns


@dataclass
class KnowledgeItem:
    """Individual knowledge item in the global network."""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: KnowledgeType = KnowledgeType.PATTERN
    title: str = ""
    description: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    domain: str = ""           # e.g., "web_development", "mobile_apps"
    technologies: List[str] = field(default_factory=list)
    complexity_level: int = 5  # 1-10 scale
    success_rate: float = 0.0  # 0.0-1.0
    confidence_score: float = 0.0  # Statistical confidence
    
    # Privacy and attribution
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMIZED
    source_organization: Optional[str] = None
    contributor_id: str = ""
    anonymized_source: str = ""
    
    # Network metadata
    validation_score: float = 0.0  # Validation by network peers
    usage_count: int = 0
    success_feedback: int = 0
    failure_feedback: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    
    # Relationships
    related_items: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)


@dataclass
class NetworkNode:
    """A node (organization/team) in the knowledge network."""
    node_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    organization_type: str = "development_team"
    role: NetworkRole = NetworkRole.CONTRIBUTOR
    
    # Capabilities and interests
    domains: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    expertise_level: Dict[str, float] = field(default_factory=dict)  # Domain -> expertise (0-10)
    
    # Network participation
    knowledge_contributed: int = 0
    knowledge_consumed: int = 0
    validation_activity: int = 0
    reputation_score: float = 5.0  # Network reputation (0-10)
    trust_level: float = 5.0      # Trust level (0-10)
    
    # Connection preferences
    collaboration_openness: float = 0.5  # 0=closed, 1=fully open
    preferred_privacy_level: PrivacyLevel = PrivacyLevel.ANONYMIZED
    data_sharing_policy: Dict[str, Any] = field(default_factory=dict)
    
    # Contact and integration
    api_endpoint: Optional[str] = None
    webhook_url: Optional[str] = None
    encryption_key: Optional[str] = None
    
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkQuery:
    """Query for knowledge from the network."""
    query_id: str = field(default_factory=lambda: str(uuid4()))
    requester_id: str = ""
    
    # Query parameters
    domain: Optional[str] = None
    technologies: List[str] = field(default_factory=list)
    knowledge_types: List[KnowledgeType] = field(default_factory=list)
    complexity_range: Tuple[int, int] = (1, 10)
    min_success_rate: float = 0.0
    min_confidence: float = 0.0
    
    # Result preferences
    max_results: int = 50
    preferred_privacy_level: PrivacyLevel = PrivacyLevel.ANONYMIZED
    include_experimental: bool = False
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkAnalytics:
    """Analytics for the global knowledge network."""
    total_nodes: int = 0
    total_knowledge_items: int = 0
    active_contributors: int = 0
    knowledge_by_domain: Dict[str, int] = field(default_factory=dict)
    success_rate_by_domain: Dict[str, float] = field(default_factory=dict)
    
    # Network health metrics
    network_density: float = 0.0  # How connected the network is
    knowledge_diversity: float = 0.0  # How diverse the knowledge is
    validation_coverage: float = 0.0  # What percentage is validated
    update_frequency: float = 0.0  # How often knowledge is updated
    
    # Quality metrics
    average_validation_score: float = 0.0
    high_confidence_percentage: float = 0.0
    successful_application_rate: float = 0.0
    
    # Growth metrics
    new_nodes_per_month: float = 0.0
    new_knowledge_per_month: float = 0.0
    engagement_trend: float = 0.0  # Growing or shrinking engagement
    
    generated_at: datetime = field(default_factory=datetime.now)


class GlobalKnowledgeNetwork:
    """
    Global Knowledge Network for autonomous development teams.
    
    Facilitates knowledge sharing and collaboration across teams while
    preserving privacy and maintaining competitive advantages through
    intelligent anonymization and selective sharing.
    """
    
    def __init__(self, node_config: Dict[str, Any], storage_path: str = "data/global_network"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Network configuration
        self.local_node = NetworkNode(**node_config)
        self.network_nodes: Dict[str, NetworkNode] = {}
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.query_cache: Dict[str, Tuple[List[KnowledgeItem], datetime]] = {}
        
        # Privacy and security
        self.encryption_key = self._generate_encryption_key()
        self.anonymization_salt = self._generate_salt()
        
        # Network settings
        self.max_cache_age = timedelta(hours=1)
        self.validation_threshold = 0.7
        self.trust_decay_rate = 0.05  # Per month
        self.reputation_boost_factor = 0.1
        
        # Load existing data
        self._load_network_data()
        
        # HTTP session for network communication
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Background tasks
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key for secure communications."""
        key = Fernet.generate_key()
        return key.decode()
    
    def _generate_salt(self) -> str:
        """Generate salt for anonymization."""
        return base64.b64encode(hashlib.sha256(str(uuid4()).encode()).digest()).decode()
    
    def _load_network_data(self):
        """Load existing network data from storage."""
        try:
            # Load local knowledge
            knowledge_file = self.storage_path / "local_knowledge.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r') as f:
                    data = json.load(f)
                    self.knowledge_items = {
                        k: KnowledgeItem(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.knowledge_items)} knowledge items")
            
            # Load network nodes
            nodes_file = self.storage_path / "network_nodes.json"
            if nodes_file.exists():
                with open(nodes_file, 'r') as f:
                    data = json.load(f)
                    self.network_nodes = {
                        k: NetworkNode(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.network_nodes)} network nodes")
        
        except Exception as e:
            logger.error(f"Error loading network data: {e}")
    
    def _save_network_data(self):
        """Save network data to storage."""
        try:
            # Save knowledge items
            knowledge_file = self.storage_path / "local_knowledge.json"
            with open(knowledge_file, 'w') as f:
                data = {k: self._serialize_dataclass(v) for k, v in self.knowledge_items.items()}
                json.dump(data, f, indent=2, default=str)
            
            # Save network nodes
            nodes_file = self.storage_path / "network_nodes.json"
            with open(nodes_file, 'w') as f:
                data = {k: self._serialize_dataclass(v) for k, v in self.network_nodes.items()}
                json.dump(data, f, indent=2, default=str)
            
            logger.debug("Network data saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving network data: {e}")
    
    def _serialize_dataclass(self, obj) -> Dict[str, Any]:
        """Serialize dataclass to dictionary."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
            return result
        return obj
    
    def _anonymize_content(self, content: Dict[str, Any], privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Anonymize content based on privacy level."""
        if privacy_level == PrivacyLevel.PUBLIC:
            return content
        
        anonymized = content.copy()
        
        if privacy_level == PrivacyLevel.ANONYMIZED:
            # Remove or hash identifying information
            identifying_keys = ["organization", "project_name", "author", "contact", "url"]
            for key in identifying_keys:
                if key in anonymized:
                    # Hash the value with salt for consistent anonymization
                    hashed = hmac.new(
                        self.anonymization_salt.encode(),
                        str(anonymized[key]).encode(),
                        hashlib.sha256
                    ).hexdigest()[:8]
                    anonymized[key] = f"anon_{hashed}"
        
        elif privacy_level == PrivacyLevel.AGGREGATED:
            # Keep only statistical aggregates
            numerical_fields = {k: v for k, v in anonymized.items() 
                              if isinstance(v, (int, float))}
            anonymized = {
                "summary_stats": numerical_fields,
                "category_counts": len([k for k, v in content.items() if isinstance(v, str)])
            }
        
        elif privacy_level == PrivacyLevel.ENCRYPTED:
            # Encrypt the content
            fernet = Fernet(self.encryption_key.encode())
            encrypted_content = fernet.encrypt(json.dumps(content).encode())
            anonymized = {"encrypted_content": encrypted_content.decode()}
        
        return anonymized
    
    async def contribute_knowledge(
        self,
        knowledge_item: KnowledgeItem,
        share_globally: bool = True
    ) -> bool:
        """Contribute knowledge to the local store and optionally to the global network."""
        
        try:
            # Store locally
            self.knowledge_items[knowledge_item.id] = knowledge_item
            
            # Update local node stats
            self.local_node.knowledge_contributed += 1
            self.local_node.last_active = datetime.now()
            
            # Share globally if requested and node allows it
            if share_globally and self.local_node.collaboration_openness > 0.5:
                await self._share_knowledge_globally(knowledge_item)
            
            # Save to storage
            self._save_network_data()
            
            logger.info(f"Contributed knowledge: {knowledge_item.title}")
            return True
        
        except Exception as e:
            logger.error(f"Error contributing knowledge: {e}")
            return False
    
    async def _share_knowledge_globally(self, knowledge_item: KnowledgeItem):
        """Share knowledge item with the global network."""
        
        # Anonymize content based on privacy settings
        anonymized_content = self._anonymize_content(
            knowledge_item.content,
            knowledge_item.privacy_level
        )
        
        # Create network payload
        network_item = {
            "id": knowledge_item.id,
            "type": knowledge_item.type.value,
            "title": knowledge_item.title,
            "description": knowledge_item.description,
            "content": anonymized_content,
            "domain": knowledge_item.domain,
            "technologies": knowledge_item.technologies,
            "complexity_level": knowledge_item.complexity_level,
            "success_rate": knowledge_item.success_rate,
            "confidence_score": knowledge_item.confidence_score,
            "privacy_level": knowledge_item.privacy_level.value,
            "anonymized_source": self._generate_anonymous_id(),
            "created_at": knowledge_item.created_at.isoformat()
        }
        
        # Share with connected nodes
        for node in self.network_nodes.values():
            if node.api_endpoint and node.trust_level > self.validation_threshold:
                try:
                    await self._send_to_node(node.api_endpoint, "knowledge/share", network_item)
                    logger.debug(f"Shared knowledge with {node.name}")
                except Exception as e:
                    logger.warning(f"Failed to share with {node.name}: {e}")
    
    def _generate_anonymous_id(self) -> str:
        """Generate anonymous identifier for the local node."""
        node_info = f"{self.local_node.node_id}:{datetime.now().strftime('%Y-%m')}"
        return hmac.new(
            self.anonymization_salt.encode(),
            node_info.encode(),
            hashlib.sha256
        ).hexdigest()[:12]
    
    async def query_network(self, query: NetworkQuery) -> List[KnowledgeItem]:
        """Query the global knowledge network for relevant knowledge."""
        
        # Check cache first
        cache_key = self._generate_query_cache_key(query)
        if cache_key in self.query_cache:
            cached_results, cached_time = self.query_cache[cache_key]
            if datetime.now() - cached_time < self.max_cache_age:
                logger.debug("Returning cached query results")
                return cached_results
        
        # Query local knowledge first
        local_results = await self._query_local_knowledge(query)
        
        # Query network nodes
        network_results = await self._query_network_nodes(query)
        
        # Combine and deduplicate results
        all_results = local_results + network_results
        unique_results = self._deduplicate_results(all_results)
        
        # Rank results by relevance and quality
        ranked_results = self._rank_results(unique_results, query)
        
        # Limit results
        final_results = ranked_results[:query.max_results]
        
        # Cache results
        self.query_cache[cache_key] = (final_results, datetime.now())
        
        # Update local node stats
        self.local_node.knowledge_consumed += 1
        self.local_node.last_active = datetime.now()
        
        logger.info(f"Query returned {len(final_results)} results")
        return final_results
    
    def _generate_query_cache_key(self, query: NetworkQuery) -> str:
        """Generate cache key for query."""
        query_string = f"{query.domain}:{','.join(query.technologies)}:{','.join([kt.value for kt in query.knowledge_types])}"
        return hashlib.md5(query_string.encode()).hexdigest()
    
    async def _query_local_knowledge(self, query: NetworkQuery) -> List[KnowledgeItem]:
        """Query local knowledge store."""
        
        results = []
        
        for item in self.knowledge_items.values():
            if self._matches_query(item, query):
                results.append(item)
        
        return results
    
    async def _query_network_nodes(self, query: NetworkQuery) -> List[KnowledgeItem]:
        """Query network nodes for knowledge."""
        
        results = []
        
        # Convert query to network format
        network_query = {
            "domain": query.domain,
            "technologies": query.technologies,
            "knowledge_types": [kt.value for kt in query.knowledge_types],
            "complexity_range": query.complexity_range,
            "min_success_rate": query.min_success_rate,
            "min_confidence": query.min_confidence,
            "max_results": query.max_results // max(1, len(self.network_nodes)),
            "requester_id": self._generate_anonymous_id()
        }
        
        # Query each trusted node
        for node in self.network_nodes.values():
            if node.api_endpoint and node.trust_level > self.validation_threshold:
                try:
                    node_results = await self._query_node(node.api_endpoint, network_query)
                    results.extend(node_results)
                except Exception as e:
                    logger.warning(f"Failed to query {node.name}: {e}")
        
        return results
    
    def _matches_query(self, item: KnowledgeItem, query: NetworkQuery) -> bool:
        """Check if knowledge item matches query criteria."""
        
        # Domain match
        if query.domain and item.domain != query.domain:
            return False
        
        # Technology overlap
        if query.technologies:
            if not any(tech in item.technologies for tech in query.technologies):
                return False
        
        # Knowledge type match
        if query.knowledge_types:
            if item.type not in query.knowledge_types:
                return False
        
        # Complexity range
        if not (query.complexity_range[0] <= item.complexity_level <= query.complexity_range[1]):
            return False
        
        # Success rate threshold
        if item.success_rate < query.min_success_rate:
            return False
        
        # Confidence threshold
        if item.confidence_score < query.min_confidence:
            return False
        
        return True
    
    def _deduplicate_results(self, results: List[KnowledgeItem]) -> List[KnowledgeItem]:
        """Remove duplicate results based on content similarity."""
        
        unique_results = []
        seen_hashes = set()
        
        for item in results:
            # Create content hash for deduplication
            content_str = json.dumps(item.content, sort_keys=True)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                unique_results.append(item)
                seen_hashes.add(content_hash)
        
        return unique_results
    
    def _rank_results(self, results: List[KnowledgeItem], query: NetworkQuery) -> List[KnowledgeItem]:
        """Rank results by relevance and quality."""
        
        def calculate_score(item: KnowledgeItem) -> float:
            score = 0.0
            
            # Base quality score
            score += item.confidence_score * 0.3
            score += item.success_rate * 0.3
            score += item.validation_score * 0.2
            
            # Technology relevance
            if query.technologies:
                tech_overlap = len(set(query.technologies) & set(item.technologies))
                tech_relevance = tech_overlap / len(query.technologies)
                score += tech_relevance * 0.2
            
            # Usage and feedback
            if item.usage_count > 0:
                success_ratio = item.success_feedback / (item.success_feedback + item.failure_feedback + 1)
                score += success_ratio * 0.1
                
                # Popularity bonus
                popularity = min(item.usage_count / 100.0, 1.0)
                score += popularity * 0.05
            
            # Recency bonus
            age_days = (datetime.now() - item.created_at).days
            recency_factor = max(0, 1 - age_days / 365)  # Decay over 1 year
            score += recency_factor * 0.05
            
            return score
        
        # Sort by calculated score
        return sorted(results, key=calculate_score, reverse=True)
    
    async def validate_knowledge(self, knowledge_id: str, validation_score: float, feedback: str = "") -> bool:
        """Validate a knowledge item (for validator nodes)."""
        
        if knowledge_id not in self.knowledge_items:
            return False
        
        item = self.knowledge_items[knowledge_id]
        
        # Update validation score (weighted average)
        current_validations = item.usage_count or 1
        item.validation_score = (
            (item.validation_score * current_validations + validation_score) /
            (current_validations + 1)
        )
        
        item.last_validated = datetime.now()
        
        # Update local node validation activity
        self.local_node.validation_activity += 1
        self.local_node.reputation_score += self.reputation_boost_factor
        
        logger.info(f"Validated knowledge {knowledge_id} with score {validation_score}")
        return True
    
    async def provide_feedback(self, knowledge_id: str, success: bool, details: str = "") -> bool:
        """Provide feedback on knowledge usage."""
        
        if knowledge_id not in self.knowledge_items:
            # Try to find it in network results
            for cached_results, _ in self.query_cache.values():
                for item in cached_results:
                    if item.id == knowledge_id:
                        # Store feedback for network item
                        await self._send_feedback_to_network(knowledge_id, success, details)
                        return True
            return False
        
        item = self.knowledge_items[knowledge_id]
        
        if success:
            item.success_feedback += 1
        else:
            item.failure_feedback += 1
        
        item.usage_count += 1
        
        # Update success rate
        total_feedback = item.success_feedback + item.failure_feedback
        item.success_rate = item.success_feedback / total_feedback
        
        logger.info(f"Recorded {'positive' if success else 'negative'} feedback for {knowledge_id}")
        return True
    
    async def _send_feedback_to_network(self, knowledge_id: str, success: bool, details: str):
        """Send feedback for network knowledge item."""
        
        feedback_payload = {
            "knowledge_id": knowledge_id,
            "success": success,
            "details": details,
            "feedback_source": self._generate_anonymous_id(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connected nodes
        for node in self.network_nodes.values():
            if node.api_endpoint:
                try:
                    await self._send_to_node(node.api_endpoint, "knowledge/feedback", feedback_payload)
                except Exception as e:
                    logger.warning(f"Failed to send feedback to {node.name}: {e}")
    
    async def join_network(self, coordinator_endpoint: str, invitation_code: str = "") -> bool:
        """Join a global knowledge network."""
        
        try:
            # Send join request
            join_payload = {
                "node_info": self._serialize_dataclass(self.local_node),
                "invitation_code": invitation_code,
                "capabilities": {
                    "domains": self.local_node.domains,
                    "technologies": self.local_node.technologies,
                    "expertise": self.local_node.expertise_level
                }
            }
            
            response = await self._send_to_node(coordinator_endpoint, "network/join", join_payload)
            
            if response.get("success"):
                # Store network coordinator
                coordinator_node = NetworkNode(
                    name="Network Coordinator",
                    role=NetworkRole.COORDINATOR,
                    api_endpoint=coordinator_endpoint,
                    trust_level=10.0  # Full trust for coordinator
                )
                self.network_nodes["coordinator"] = coordinator_node
                
                # Receive initial network topology
                if "network_nodes" in response:
                    for node_data in response["network_nodes"]:
                        node = NetworkNode(**node_data)
                        self.network_nodes[node.node_id] = node
                
                logger.info(f"Successfully joined knowledge network with {len(self.network_nodes)} nodes")
                return True
            else:
                logger.error(f"Failed to join network: {response.get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"Error joining network: {e}")
            return False
    
    async def start_network_sync(self):
        """Start background network synchronization."""
        
        if self._running:
            return
        
        self._running = True
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Start background tasks
        self._sync_task = asyncio.create_task(self._network_sync_loop())
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        
        logger.info("Started network synchronization")
    
    async def stop_network_sync(self):
        """Stop network synchronization."""
        
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Save final state
        self._save_network_data()
        
        logger.info("Stopped network synchronization")
    
    async def _network_sync_loop(self):
        """Background loop for network synchronization."""
        
        while self._running:
            try:
                # Update node trust levels
                await self._update_trust_levels()
                
                # Synchronize with coordinator
                await self._sync_with_coordinator()
                
                # Clean expired cache entries
                self._clean_query_cache()
                
                # Save data periodically
                self._save_network_data()
                
                # Sleep until next sync
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in network sync loop: {e}", exc_info=True)
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _analytics_loop(self):
        """Background loop for network analytics."""
        
        while self._running:
            try:
                # Generate network analytics
                analytics = await self.generate_network_analytics()
                
                # Store analytics
                analytics_file = self.storage_path / f"analytics_{datetime.now().strftime('%Y%m%d')}.json"
                with open(analytics_file, 'w') as f:
                    json.dump(self._serialize_dataclass(analytics), f, indent=2, default=str)
                
                # Sleep until next analytics cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}", exc_info=True)
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _update_trust_levels(self):
        """Update trust levels for network nodes based on interaction history."""
        
        for node in self.network_nodes.values():
            # Decay trust over time (monthly decay)
            days_since_update = (datetime.now() - node.last_active).days
            if days_since_update > 30:
                months_inactive = days_since_update / 30
                decay = self.trust_decay_rate * months_inactive
                node.trust_level = max(0, node.trust_level - decay)
            
            # Boost trust for active, helpful nodes
            if node.validation_activity > 0:
                trust_boost = min(node.validation_activity * 0.1, 2.0)
                node.trust_level = min(10.0, node.trust_level + trust_boost)
    
    async def _sync_with_coordinator(self):
        """Synchronize with network coordinator."""
        
        coordinator = self.network_nodes.get("coordinator")
        if not coordinator or not coordinator.api_endpoint:
            return
        
        try:
            # Send status update
            status_payload = {
                "node_id": self.local_node.node_id,
                "last_active": self.local_node.last_active.isoformat(),
                "knowledge_contributed": self.local_node.knowledge_contributed,
                "knowledge_consumed": self.local_node.knowledge_consumed,
                "validation_activity": self.local_node.validation_activity
            }
            
            response = await self._send_to_node(
                coordinator.api_endpoint,
                "network/status",
                status_payload
            )
            
            # Process network updates
            if response.get("network_updates"):
                for update in response["network_updates"]:
                    if update["type"] == "new_node":
                        node = NetworkNode(**update["node_data"])
                        self.network_nodes[node.node_id] = node
                    elif update["type"] == "node_removed":
                        node_id = update["node_id"]
                        if node_id in self.network_nodes:
                            del self.network_nodes[node_id]
        
        except Exception as e:
            logger.warning(f"Failed to sync with coordinator: {e}")
    
    def _clean_query_cache(self):
        """Clean expired entries from query cache."""
        
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.query_cache.items()
            if now - timestamp > self.max_cache_age
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
    
    async def _send_to_node(self, endpoint: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request to a network node."""
        
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        url = f"{endpoint.rstrip('/')}/{path}"
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise aiohttp.ClientError(f"HTTP {response.status}: {await response.text()}")
    
    async def _query_node(self, endpoint: str, query: Dict[str, Any]) -> List[KnowledgeItem]:
        """Query a specific network node for knowledge."""
        
        response = await self._send_to_node(endpoint, "knowledge/query", query)
        
        results = []
        for item_data in response.get("results", []):
            # Convert back to KnowledgeItem
            item = KnowledgeItem(**item_data)
            results.append(item)
        
        return results
    
    async def generate_network_analytics(self) -> NetworkAnalytics:
        """Generate comprehensive network analytics."""
        
        analytics = NetworkAnalytics(
            total_nodes=len(self.network_nodes) + 1,  # +1 for local node
            total_knowledge_items=len(self.knowledge_items),
            active_contributors=len([n for n in self.network_nodes.values() 
                                   if n.knowledge_contributed > 0 and 
                                   (datetime.now() - n.last_active).days < 30])
        )
        
        # Analyze knowledge by domain
        domain_counts = Counter()
        domain_success_rates = defaultdict(list)
        
        for item in self.knowledge_items.values():
            domain_counts[item.domain] += 1
            if item.success_rate > 0:
                domain_success_rates[item.domain].append(item.success_rate)
        
        analytics.knowledge_by_domain = dict(domain_counts)
        analytics.success_rate_by_domain = {
            domain: np.mean(rates) for domain, rates in domain_success_rates.items()
        }
        
        # Calculate network health metrics
        total_possible_connections = len(self.network_nodes) * (len(self.network_nodes) - 1) / 2
        if total_possible_connections > 0:
            active_connections = len([n for n in self.network_nodes.values() if n.trust_level > 5.0])
            analytics.network_density = active_connections / total_possible_connections
        
        # Knowledge diversity
        unique_domains = len(set(item.domain for item in self.knowledge_items.values()))
        unique_techs = len(set(tech for item in self.knowledge_items.values() for tech in item.technologies))
        analytics.knowledge_diversity = (unique_domains + unique_techs) / 100.0  # Normalize to 0-1
        
        # Validation coverage
        validated_items = len([item for item in self.knowledge_items.values() 
                             if item.last_validated is not None])
        if self.knowledge_items:
            analytics.validation_coverage = validated_items / len(self.knowledge_items)
        
        # Quality metrics
        validation_scores = [item.validation_score for item in self.knowledge_items.values() 
                           if item.validation_score > 0]
        if validation_scores:
            analytics.average_validation_score = np.mean(validation_scores)
            analytics.high_confidence_percentage = len([s for s in validation_scores if s >= 8.0]) / len(validation_scores)
        
        return analytics
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status information."""
        
        return {
            "local_node": {
                "id": self.local_node.node_id,
                "name": self.local_node.name,
                "role": self.local_node.role.value,
                "reputation": round(self.local_node.reputation_score, 2),
                "trust_level": round(self.local_node.trust_level, 2),
                "knowledge_contributed": self.local_node.knowledge_contributed,
                "knowledge_consumed": self.local_node.knowledge_consumed
            },
            "network": {
                "connected_nodes": len(self.network_nodes),
                "trusted_nodes": len([n for n in self.network_nodes.values() if n.trust_level > 5.0]),
                "total_knowledge_items": len(self.knowledge_items),
                "cache_size": len(self.query_cache)
            },
            "activity": {
                "last_sync": self.local_node.last_active.isoformat(),
                "sync_running": self._running
            }
        }


async def main():
    """Test the global knowledge network."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create network configuration
    node_config = {
        "name": "Test Development Team",
        "organization_type": "development_team",
        "role": NetworkRole.CONTRIBUTOR,
        "domains": ["web_development", "mobile_apps"],
        "technologies": ["React", "Node.js", "Python", "PostgreSQL"],
        "expertise_level": {
            "web_development": 8.5,
            "mobile_apps": 7.0,
            "backend_development": 9.0
        }
    }
    
    network = GlobalKnowledgeNetwork(node_config)
    
    # Create test knowledge item
    knowledge_item = KnowledgeItem(
        type=KnowledgeType.BEST_PRACTICE,
        title="React Component Optimization",
        description="Best practices for optimizing React component performance",
        content={
            "technique": "Use React.memo for expensive components",
            "implementation": "const OptimizedComponent = React.memo(MyComponent)",
            "performance_gain": "30-50% render time reduction"
        },
        domain="web_development",
        technologies=["React", "JavaScript"],
        complexity_level=6,
        success_rate=0.85,
        confidence_score=8.5,
        privacy_level=PrivacyLevel.ANONYMIZED
    )
    
    # Contribute knowledge
    await network.contribute_knowledge(knowledge_item)
    
    # Create test query
    query = NetworkQuery(
        requester_id="test_user",
        domain="web_development",
        technologies=["React"],
        knowledge_types=[KnowledgeType.BEST_PRACTICE],
        min_success_rate=0.7
    )
    
    # Query network
    results = await network.query_network(query)
    print(f"Found {len(results)} knowledge items")
    
    # Get network status
    status = await network.get_network_status()
    print(f"Network status: {status}")
    
    # Generate analytics
    analytics = await network.generate_network_analytics()
    print(f"Network analytics: {analytics.total_knowledge_items} items, {analytics.network_density:.2f} density")


if __name__ == "__main__":
    asyncio.run(main())