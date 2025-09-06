"""
Knowledge Management System v3.0 - Multi-Layer Knowledge Storage and Evolution
Based on F-KMS-001, F-KMS-002, F-KMS-003 from PRD specifications

NLNH Protocol: Real knowledge storage with actual persistence and retrieval
DGTS Enforcement: No fake knowledge, actual confidence-based evolution
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import hashlib
import re

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Knowledge types as per F-KMS-001 storage architecture"""
    PATTERN = "pattern"
    DECISION = "decision"
    FAILURE = "failure"
    OPTIMIZATION = "optimization"
    RELATIONSHIP = "relationship"
    AGENT_MEMORY = "agent-memory"


class TransferType(Enum):
    """Knowledge transfer types as per F-KMS-002"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    INHERITANCE = "inheritance"
    CROSS_PROJECT = "cross_project"


@dataclass
class KnowledgeItem:
    """Knowledge item with all required fields as per TDD tests"""
    item_id: str
    item_type: str
    content: str
    confidence: float
    project_id: str
    agent_id: str
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    def __post_init__(self):
        """Validate confidence bounds"""
        self.confidence = max(0.1, min(0.99, self.confidence))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for knowledge item"""
        if self.usage_count == 0:
            return 0.5  # Neutral starting point
        return self.success_count / self.usage_count
    
    @property
    def is_promotable(self) -> bool:
        """Check if knowledge meets promotion threshold (0.8)"""
        return self.confidence >= 0.8
    
    @property
    def needs_review(self) -> bool:
        """Check if knowledge needs review (demotion threshold 0.3)"""
        return self.confidence <= 0.3
    
    def evolve_confidence(self, success: bool) -> float:
        """Evolve confidence based on success/failure as per F-KMS-003"""
        old_confidence = self.confidence
        
        if success:
            # Success: confidence * 1.1 (max 0.99)
            self.confidence = min(self.confidence * 1.1, 0.99)
            self.success_count += 1
        else:
            # Failure: confidence * 0.9 (min 0.1)  
            self.confidence = max(self.confidence * 0.9, 0.1)
            self.failure_count += 1
        
        self.usage_count += 1
        self.updated_at = datetime.now()
        
        logger.info(f"Knowledge {self.item_id} evolved: {old_confidence:.3f} -> {self.confidence:.3f} "
                   f"({'success' if success else 'failure'})")
        
        return self.confidence


@dataclass
class KnowledgeQuery:
    """Knowledge query with filtering capabilities"""
    query_text: str
    project_id: Optional[str] = None
    agent_type: Optional[str] = None
    knowledge_types: List[str] = None
    tags: List[str] = None
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    limit: int = 10
    include_cross_project: bool = False
    
    def __post_init__(self):
        if self.knowledge_types is None:
            self.knowledge_types = []
        if self.tags is None:
            self.tags = []


class KnowledgeStorage:
    """Multi-layer knowledge storage implementation"""
    
    def __init__(self, base_path: str = "/tmp/archon-knowledge"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_path / "knowledge.db"
        self._init_database()
        
        # Storage layer paths as per F-KMS-001
        self.storage_layers = {
            KnowledgeType.PATTERN: self.base_path / "patterns",
            KnowledgeType.DECISION: self.base_path / "decisions", 
            KnowledgeType.FAILURE: self.base_path / "failures",
            KnowledgeType.OPTIMIZATION: self.base_path / "optimizations",
            KnowledgeType.RELATIONSHIP: self.base_path / "relationships",
            KnowledgeType.AGENT_MEMORY: self.base_path / "agent-memory"
        }
        
        # Create storage layer directories
        for layer_path in self.storage_layers.values():
            layer_path.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for knowledge storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    item_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_project_confidence 
                ON knowledge_items(project_id, confidence)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_type_tags 
                ON knowledge_items(item_type, tags)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_updated 
                ON knowledge_items(agent_id, updated_at)
            ''')
    
    async def store_item(self, item: KnowledgeItem) -> bool:
        """Store knowledge item in database and file system"""
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO knowledge_items (
                        item_id, item_type, content, confidence, project_id, agent_id,
                        tags, metadata, created_at, updated_at, usage_count, 
                        success_count, failure_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item.item_id, item.item_type, item.content, item.confidence,
                    item.project_id, item.agent_id, json.dumps(item.tags),
                    json.dumps(item.metadata), item.created_at.isoformat(),
                    item.updated_at.isoformat(), item.usage_count,
                    item.success_count, item.failure_count
                ))
            
            # Store in appropriate file layer
            await self._store_in_layer(item)
            
            logger.debug(f"Stored knowledge item {item.item_id} ({item.item_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store knowledge item {item.item_id}: {e}")
            return False
    
    async def _store_in_layer(self, item: KnowledgeItem):
        """Store item in appropriate storage layer"""
        try:
            knowledge_type = KnowledgeType(item.item_type)
            layer_path = self.storage_layers[knowledge_type]
            
            # Create project subdirectory
            project_path = layer_path / item.project_id
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename based on content hash for deduplication
            content_hash = hashlib.md5(item.content.encode()).hexdigest()[:8]
            filename = f"{item.agent_id}_{content_hash}_{item.item_id}.json"
            file_path = project_path / filename
            
            # Store as JSON
            item_data = {
                "item_id": item.item_id,
                "item_type": item.item_type,
                "content": item.content,
                "confidence": item.confidence,
                "project_id": item.project_id,
                "agent_id": item.agent_id,
                "tags": item.tags,
                "metadata": item.metadata,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "usage_count": item.usage_count,
                "success_count": item.success_count,
                "failure_count": item.failure_count
            }
            
            with open(file_path, 'w') as f:
                json.dump(item_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to store item in layer: {e}")
    
    async def search_items(self, query: KnowledgeQuery) -> List[KnowledgeItem]:
        """Search knowledge items based on query"""
        try:
            conditions = ["1=1"]  # Always true base condition
            params = []
            
            # Project filtering
            if query.project_id:
                conditions.append("project_id = ?")
                params.append(query.project_id)
            
            # Confidence filtering
            if query.min_confidence > 0:
                conditions.append("confidence >= ?")
                params.append(query.min_confidence)
            
            if query.max_confidence < 1.0:
                conditions.append("confidence <= ?")
                params.append(query.max_confidence)
            
            # Knowledge type filtering
            if query.knowledge_types:
                placeholders = ','.join(['?' for _ in query.knowledge_types])
                conditions.append(f"item_type IN ({placeholders})")
                params.extend(query.knowledge_types)
            
            # Text search in content
            if query.query_text and query.query_text.strip():
                conditions.append("content LIKE ?")
                params.append(f"%{query.query_text}%")
            
            # Tag filtering (simple JSON search)
            if query.tags:
                for tag in query.tags:
                    conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
            
            # Build and execute query
            sql = f'''
                SELECT * FROM knowledge_items 
                WHERE {' AND '.join(conditions)}
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
            '''
            params.append(query.limit)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
            
            # Convert to KnowledgeItem objects
            items = []
            for row in rows:
                item = KnowledgeItem(
                    item_id=row['item_id'],
                    item_type=row['item_type'],
                    content=row['content'],
                    confidence=row['confidence'],
                    project_id=row['project_id'],
                    agent_id=row['agent_id'],
                    tags=json.loads(row['tags']),
                    metadata=json.loads(row['metadata']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    usage_count=row['usage_count'],
                    success_count=row['success_count'],
                    failure_count=row['failure_count']
                )
                items.append(item)
            
            logger.debug(f"Search returned {len(items)} items for query: {query.query_text}")
            return items
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def get_item_by_id(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve specific knowledge item by ID"""
        query = KnowledgeQuery(query_text="", limit=1)
        results = await self.search_items(query)
        
        for item in results:
            if item.item_id == item_id:
                return item
        return None
    
    async def update_confidence(self, item_id: str, success: bool) -> bool:
        """Update knowledge confidence based on usage outcome"""
        item = await self.get_item_by_id(item_id)
        if not item:
            return False
        
        item.evolve_confidence(success)
        return await self.store_item(item)


class KnowledgeTransferProtocol:
    """Knowledge transfer protocol implementation as per F-KMS-002"""
    
    def __init__(self, storage: KnowledgeStorage):
        self.storage = storage
        self.transfer_queue: List[Dict[str, Any]] = []
        self.broadcast_subscribers: Dict[str, List[str]] = {}  # agent_type -> [agent_ids]
    
    async def synchronous_transfer(self, item: KnowledgeItem, target_agent_id: str) -> bool:
        """Direct agent-to-agent transfer for critical knowledge"""
        try:
            # Mark as critical transfer
            transfer_metadata = {
                **item.metadata,
                "transfer_type": TransferType.SYNCHRONOUS.value,
                "target_agent_id": target_agent_id,
                "transferred_at": datetime.now().isoformat(),
                "priority": "immediate"
            }
            
            # Create transfer copy with new ID
            transfer_item = KnowledgeItem(
                item_id=str(uuid.uuid4()),
                item_type=item.item_type,
                content=item.content,
                confidence=item.confidence,
                project_id=item.project_id,
                agent_id=target_agent_id,  # Transfer ownership
                tags=item.tags + ["transferred", "critical"],
                metadata=transfer_metadata,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            success = await self.storage.store_item(transfer_item)
            
            if success:
                logger.info(f"Synchronous transfer completed: {item.item_id} -> {target_agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Synchronous transfer failed: {e}")
            return False
    
    async def asynchronous_broadcast(self, item: KnowledgeItem, agent_types: List[str]) -> int:
        """Broadcast discoveries to relevant agent types"""
        try:
            broadcast_count = 0
            
            for agent_type in agent_types:
                if agent_type in self.broadcast_subscribers:
                    for target_agent_id in self.broadcast_subscribers[agent_type]:
                        # Create broadcast copy
                        broadcast_metadata = {
                            **item.metadata,
                            "transfer_type": TransferType.ASYNCHRONOUS.value,
                            "broadcast_to": agent_type,
                            "broadcasted_at": datetime.now().isoformat()
                        }
                        
                        broadcast_item = KnowledgeItem(
                            item_id=str(uuid.uuid4()),
                            item_type=item.item_type,
                            content=item.content,
                            confidence=item.confidence * 0.9,  # Slight confidence reduction for broadcast
                            project_id=item.project_id,
                            agent_id=target_agent_id,
                            tags=item.tags + ["broadcast"],
                            metadata=broadcast_metadata,
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                        
                        if await self.storage.store_item(broadcast_item):
                            broadcast_count += 1
            
            logger.info(f"Asynchronous broadcast completed: {broadcast_count} recipients")
            return broadcast_count
            
        except Exception as e:
            logger.error(f"Asynchronous broadcast failed: {e}")
            return 0
    
    async def inherit_knowledge(self, parent_agent_id: str, child_agent_id: str, 
                               project_id: str, min_confidence: float = 0.7) -> int:
        """Transfer high-confidence knowledge to new agents"""
        try:
            # Search for high-confidence knowledge from parent
            inheritance_query = KnowledgeQuery(
                query_text="",
                project_id=project_id,
                min_confidence=min_confidence,
                limit=50  # Reasonable inheritance limit
            )
            
            parent_knowledge = await self.storage.search_items(inheritance_query)
            
            # Filter for parent agent's knowledge
            parent_items = [item for item in parent_knowledge if item.agent_id == parent_agent_id]
            
            inherited_count = 0
            
            for item in parent_items:
                # Create inherited copy
                inheritance_metadata = {
                    **item.metadata,
                    "transfer_type": TransferType.INHERITANCE.value,
                    "inherited_from": parent_agent_id,
                    "inherited_at": datetime.now().isoformat()
                }
                
                inherited_item = KnowledgeItem(
                    item_id=str(uuid.uuid4()),
                    item_type=item.item_type,
                    content=item.content,
                    confidence=item.confidence,  # Keep original confidence
                    project_id=item.project_id,
                    agent_id=child_agent_id,
                    tags=item.tags + ["inherited"],
                    metadata=inheritance_metadata,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                if await self.storage.store_item(inherited_item):
                    inherited_count += 1
            
            logger.info(f"Knowledge inheritance completed: {inherited_count} items from {parent_agent_id} to {child_agent_id}")
            return inherited_count
            
        except Exception as e:
            logger.error(f"Knowledge inheritance failed: {e}")
            return 0
    
    async def cross_project_sharing(self, source_project_id: str, target_project_id: str, 
                                  general_tags: List[str] = None) -> int:
        """Share general patterns across projects"""
        try:
            if general_tags is None:
                general_tags = ["general", "best-practice", "pattern"]
            
            # Search for shareable general patterns
            sharing_query = KnowledgeQuery(
                query_text="",
                project_id=source_project_id,
                tags=general_tags,
                min_confidence=0.8,  # Only high-confidence general patterns
                limit=20
            )
            
            source_knowledge = await self.storage.search_items(sharing_query)
            
            # Filter for shareable items
            shareable_items = [
                item for item in source_knowledge 
                if item.metadata.get("shareable", True)  # Default to shareable
            ]
            
            shared_count = 0
            
            for item in shareable_items:
                # Create cross-project copy
                sharing_metadata = {
                    **item.metadata,
                    "transfer_type": TransferType.CROSS_PROJECT.value,
                    "source_project_id": source_project_id,
                    "shared_at": datetime.now().isoformat(),
                    "shareable": True
                }
                
                shared_item = KnowledgeItem(
                    item_id=str(uuid.uuid4()),
                    item_type=item.item_type,
                    content=item.content,
                    confidence=item.confidence * 0.95,  # Slight reduction for cross-project
                    project_id=target_project_id,
                    agent_id=f"shared-{item.agent_id}",
                    tags=item.tags + ["cross-project-shared"],
                    metadata=sharing_metadata,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                if await self.storage.store_item(shared_item):
                    shared_count += 1
            
            logger.info(f"Cross-project sharing completed: {shared_count} patterns from {source_project_id} to {target_project_id}")
            return shared_count
            
        except Exception as e:
            logger.error(f"Cross-project sharing failed: {e}")
            return 0
    
    def subscribe_to_broadcasts(self, agent_id: str, agent_types: List[str]):
        """Subscribe agent to broadcasts for specific agent types"""
        for agent_type in agent_types:
            if agent_type not in self.broadcast_subscribers:
                self.broadcast_subscribers[agent_type] = []
            
            if agent_id not in self.broadcast_subscribers[agent_type]:
                self.broadcast_subscribers[agent_type].append(agent_id)
        
        logger.info(f"Agent {agent_id} subscribed to broadcasts for: {agent_types}")


class KnowledgeManager:
    """Main Knowledge Management System implementing F-KMS-001, F-KMS-002, F-KMS-003"""
    
    def __init__(self, base_path: str = "/tmp/archon-knowledge"):
        self.storage = KnowledgeStorage(base_path)
        self.transfer_protocol = KnowledgeTransferProtocol(self.storage)
        self.evolution_enabled = True
        
        # Knowledge promotion/demotion thresholds
        self.promotion_threshold = 0.8
        self.demotion_threshold = 0.3
    
    async def store_knowledge_item(self, item: KnowledgeItem) -> bool:
        """Store knowledge item in multi-layer storage"""
        return await self.storage.store_item(item)
    
    async def search_knowledge(self, query: KnowledgeQuery) -> List[KnowledgeItem]:
        """Search knowledge with filtering and ranking"""
        return await self.storage.search_items(query)
    
    async def record_usage_outcome(self, item_id: str, success: bool) -> bool:
        """Record usage outcome and evolve confidence"""
        if not self.evolution_enabled:
            return False
        
        return await self.storage.update_confidence(item_id, success)
    
    async def get_promotable_knowledge(self, project_id: Optional[str] = None) -> List[KnowledgeItem]:
        """Get knowledge items ready for promotion to stable patterns"""
        query = KnowledgeQuery(
            query_text="",
            project_id=project_id,
            min_confidence=self.promotion_threshold,
            limit=100
        )
        
        results = await self.search_knowledge(query)
        return [item for item in results if item.is_promotable]
    
    async def get_knowledge_needing_review(self, project_id: Optional[str] = None) -> List[KnowledgeItem]:
        """Get knowledge items that need review (low confidence)"""
        query = KnowledgeQuery(
            query_text="",
            project_id=project_id,
            max_confidence=self.demotion_threshold,
            limit=100
        )
        
        results = await self.search_knowledge(query)
        return [item for item in results if item.needs_review]
    
    async def transfer_critical_knowledge(self, item: KnowledgeItem, target_agent_id: str) -> bool:
        """Transfer critical knowledge immediately to target agent"""
        return await self.transfer_protocol.synchronous_transfer(item, target_agent_id)
    
    async def broadcast_discovery(self, item: KnowledgeItem, agent_types: List[str]) -> int:
        """Broadcast knowledge discovery to relevant agent types"""
        return await self.transfer_protocol.asynchronous_broadcast(item, agent_types)
    
    async def setup_knowledge_inheritance(self, parent_agent_id: str, child_agent_id: str, 
                                        project_id: str) -> int:
        """Set up knowledge inheritance from parent to child agent"""
        return await self.transfer_protocol.inherit_knowledge(parent_agent_id, child_agent_id, project_id)
    
    async def enable_cross_project_learning(self, source_project_id: str, target_project_id: str) -> int:
        """Enable cross-project pattern sharing"""
        return await self.transfer_protocol.cross_project_sharing(source_project_id, target_project_id)
    
    def subscribe_agent_to_broadcasts(self, agent_id: str, agent_types: List[str]):
        """Subscribe agent to knowledge broadcasts"""
        self.transfer_protocol.subscribe_to_broadcasts(agent_id, agent_types)
    
    async def get_knowledge_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive knowledge statistics"""
        try:
            # Get all knowledge for project or global
            all_query = KnowledgeQuery(
                query_text="",
                project_id=project_id,
                limit=10000  # Large limit for statistics
            )
            
            all_knowledge = await self.search_knowledge(all_query)
            
            if not all_knowledge:
                return {
                    "total_items": 0,
                    "by_type": {},
                    "confidence_distribution": {},
                    "usage_stats": {}
                }
            
            # Aggregate statistics
            stats = {
                "total_items": len(all_knowledge),
                "by_type": {},
                "confidence_distribution": {
                    "high (>0.8)": 0,
                    "medium (0.5-0.8)": 0, 
                    "low (<0.5)": 0
                },
                "usage_stats": {
                    "total_usage": sum(item.usage_count for item in all_knowledge),
                    "total_successes": sum(item.success_count for item in all_knowledge),
                    "total_failures": sum(item.failure_count for item in all_knowledge),
                    "average_confidence": sum(item.confidence for item in all_knowledge) / len(all_knowledge)
                },
                "promotable_items": 0,
                "items_needing_review": 0
            }
            
            # Type distribution
            for item in all_knowledge:
                stats["by_type"][item.item_type] = stats["by_type"].get(item.item_type, 0) + 1
                
                # Confidence distribution
                if item.confidence > 0.8:
                    stats["confidence_distribution"]["high (>0.8)"] += 1
                elif item.confidence >= 0.5:
                    stats["confidence_distribution"]["medium (0.5-0.8)"] += 1
                else:
                    stats["confidence_distribution"]["low (<0.5)"] += 1
                
                # Promotion/review counts
                if item.is_promotable:
                    stats["promotable_items"] += 1
                if item.needs_review:
                    stats["items_needing_review"] += 1
            
            # Calculate success rate
            total_usage = stats["usage_stats"]["total_usage"]
            if total_usage > 0:
                stats["usage_stats"]["success_rate"] = stats["usage_stats"]["total_successes"] / total_usage
            else:
                stats["usage_stats"]["success_rate"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate knowledge statistics: {e}")
            return {"error": str(e)}


# Utility functions
async def create_knowledge_item(item_type: str, content: str, confidence: float,
                              project_id: str, agent_id: str, tags: List[str] = None,
                              metadata: Dict[str, Any] = None) -> KnowledgeItem:
    """Utility function to create knowledge items"""
    return KnowledgeItem(
        item_id=str(uuid.uuid4()),
        item_type=item_type,
        content=content,
        confidence=confidence,
        project_id=project_id,
        agent_id=agent_id,
        tags=tags or [],
        metadata=metadata or {},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


# Example usage and testing
async def main():
    """Example usage of Knowledge Management System"""
    print("🧠 Archon v3.0 Knowledge Management System")
    print("=" * 50)
    
    # Initialize knowledge manager
    km = KnowledgeManager("/tmp/archon-knowledge-demo")
    
    # Create sample knowledge items
    pattern_item = await create_knowledge_item(
        item_type=KnowledgeType.PATTERN.value,
        content="Always use async/await for database operations to prevent blocking",
        confidence=0.8,
        project_id="proj-webapp",
        agent_id="agent-db-expert",
        tags=["database", "async", "best-practice"],
        metadata={"complexity": "medium", "domain": "backend"}
    )
    
    optimization_item = await create_knowledge_item(
        item_type=KnowledgeType.OPTIMIZATION.value,
        content="React.memo prevents unnecessary re-renders in list components",
        confidence=0.9,
        project_id="proj-webapp",
        agent_id="agent-react-optimizer",
        tags=["react", "performance", "optimization"],
        metadata={"impact": "high", "implementation_time": "5min"}
    )
    
    # Store knowledge items
    print("\n📝 Storing knowledge items...")
    await km.store_knowledge_item(pattern_item)
    await km.store_knowledge_item(optimization_item)
    
    # Search for knowledge
    print("\n🔍 Searching for React knowledge...")
    react_query = KnowledgeQuery(
        query_text="react",
        project_id="proj-webapp",
        min_confidence=0.8
    )
    
    react_results = await km.search_knowledge(react_query)
    for result in react_results:
        print(f"  ✅ {result.content} (confidence: {result.confidence:.2f})")
    
    # Test knowledge evolution
    print("\n📈 Testing knowledge evolution...")
    success_outcome = await km.record_usage_outcome(optimization_item.item_id, True)
    print(f"Success outcome recorded: {success_outcome}")
    
    # Get statistics
    print("\n📊 Knowledge Statistics:")
    stats = await km.get_knowledge_statistics("proj-webapp")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Average confidence: {stats['usage_stats']['average_confidence']:.2f}")
    print(f"  Promotable items: {stats['promotable_items']}")
    
    print("\n✅ Knowledge Management System demo completed!")


if __name__ == "__main__":
    asyncio.run(main())