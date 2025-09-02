#!/usr/bin/env python3
"""
Context Assembler for PRP-like Knowledge Packs - Archon+ Phase 4

Generates structured Markdown context packs with provenance tracking:
- Role-specific context prioritization
- Multi-source result fusion with provenance  
- Memory deduplication and relevance scoring
- Integration with MemoryService and GraphitiService
- PRP-like structured format for knowledge delivery
- Relevance scoring and coherence validation

Performance requirements: Context relevance >90% coherence
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import hashlib

# Import memory system components
from .memory_scopes import MemoryLayerType, AccessLevel, RoleBasedAccessControl
from ..graphiti.graphiti_service import GraphEntity, EntityType

logger = logging.getLogger(__name__)

@dataclass
class ContentSection:
    """Individual content section with metadata"""
    section_id: str
    section_type: str  # implementation_guide, patterns, examples, security, testing
    title: str
    content: str
    source: str  # provenance source
    relevance_score: float  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class Memory:
    """Universal Memory interface for context assembly"""
    memory_id: str
    content: Any
    memory_type: str  # "entry" or "entity"
    source: str
    relevance_score: float = 0.5
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    importance_weight: float = 0.5
    access_frequency: int = 0

@dataclass
class ContextPack:
    """Complete context pack with structured content"""
    pack_id: str
    role_context: str  # Target agent role
    task_context: str  # Task description
    content_sections: List[ContentSection]
    provenance: List[str]  # Source tracking
    relevance_score: float  # Overall pack relevance
    confidence: float = 1.0  # Overall pack confidence
    sources: List[str] = field(default_factory=list)  # Source references
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextAssembler:
    """
    Assembles structured context packs from multiple retrieval sources
    
    Creates PRP-like Markdown documents with:
    - Role-specific content prioritization
    - Multi-source provenance tracking
    - Structured sections (implementation, patterns, examples, etc.)
    - Relevance scoring and quality validation
    """
    
    def __init__(self, template_path: Optional[Path] = None, rbac: Optional[RoleBasedAccessControl] = None):
        """
        Initialize context assembler
        
        Args:
            template_path: Path to Markdown templates directory
            rbac: Role-based access control system
        """
        self.template_path = template_path or Path("python/src/agents/memory/templates")
        self.rbac = rbac or RoleBasedAccessControl()
        
        # Role-specific content priorities
        self.role_priorities = {
            "code-implementer": {
                "implementation_guide": 1.0,
                "examples": 0.9,
                "patterns": 0.8,
                "testing": 0.7,
                "security": 0.6,
                "architecture": 0.5
            },
            "system-architect": {
                "architecture": 1.0,
                "patterns": 0.9,
                "implementation_guide": 0.8,
                "security": 0.7,
                "examples": 0.6,
                "testing": 0.5
            },
            "security-auditor": {
                "security": 1.0,
                "patterns": 0.8,
                "testing": 0.7,
                "implementation_guide": 0.6,
                "examples": 0.5,
                "architecture": 0.4
            },
            "test-coverage-validator": {
                "testing": 1.0,
                "examples": 0.9,
                "implementation_guide": 0.8,
                "patterns": 0.6,
                "security": 0.5,
                "architecture": 0.4
            },
            "default": {  # Fallback for unspecified roles
                "implementation_guide": 0.8,
                "patterns": 0.8,
                "examples": 0.8,
                "testing": 0.7,
                "security": 0.7,
                "architecture": 0.6
            }
        }
        
        # Content type classification keywords
        self.content_classifiers = {
            "implementation_guide": ["implement", "code", "function", "class", "method", "algorithm"],
            "examples": ["example", "sample", "demo", "snippet", "template"],
            "patterns": ["pattern", "design", "architecture", "principle", "best practice"],
            "testing": ["test", "unit test", "integration", "coverage", "assert", "mock"],
            "security": ["security", "vulnerability", "authentication", "authorization", "encryption"],
            "architecture": ["architecture", "system", "component", "module", "design", "structure"]
        }
    
    def _classify_content_type(self, content: str) -> str:
        """
        Classify content into section types based on keywords
        
        Args:
            content: Content text to classify
            
        Returns:
            Content type (section name)
        """
        content_lower = content.lower()
        type_scores = {}
        
        for content_type, keywords in self.content_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                type_scores[content_type] = score
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return "implementation_guide"  # Default fallback
    
    def _calculate_memory_relevance(self, memory: Memory, query: str, 
                                  content_type: str, agent_role: str) -> float:
        """
        Calculate enhanced relevance score for memory
        
        Args:
            memory: Memory object with content and metadata
            query: Original query
            content_type: Type of content section
            agent_role: Target agent role
            
        Returns:
            Enhanced relevance score (0.0-1.0)
        """
        content = str(memory.content)
        
        # Base relevance from query terms
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        query_matches = sum(1 for term in query_terms if term in content_lower)
        base_relevance = min(1.0, query_matches / max(1, len(query_terms))) if query_terms else 0.5
        
        # Role-specific priority adjustment
        role_priorities = self.role_priorities.get(agent_role, self.role_priorities["default"])
        priority_weight = role_priorities.get(content_type, 0.5)
        
        # Content quality heuristics
        quality_score = 0.5
        if len(content) > 100:  # Substantial content
            quality_score += 0.2
        if any(keyword in content_lower for keyword in ["example", "implementation", "guide"]):
            quality_score += 0.2
        if content.count('\n') > 3:  # Multi-line content
            quality_score += 0.1
        
        # Memory-specific factors
        memory_factor = (
            memory.relevance_score * 0.3 +     # Existing relevance score
            memory.confidence * 0.2 +          # Confidence in memory
            memory.importance_weight * 0.2 +   # Importance weight
            min(1.0, memory.access_frequency / 10) * 0.1  # Access frequency (normalized)
        )
        
        # Tags relevance boost
        tag_boost = 0.0
        if memory.tags:
            query_in_tags = any(term in ' '.join(memory.tags).lower() for term in query_terms)
            if query_in_tags:
                tag_boost = 0.1
        
        # Combined relevance score with memory factors
        relevance = (
            base_relevance * 0.3 + 
            priority_weight * 0.25 + 
            quality_score * 0.15 + 
            memory_factor * 0.25 +
            tag_boost * 0.05
        )
        
        return min(1.0, max(0.0, relevance))
    
    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """
        Remove duplicate memories based on content similarity
        
        Args:
            memories: List of Memory objects
            
        Returns:
            Deduplicated list of memories
        """
        content_hash_to_memory = {}
        
        for memory in memories:
            # Create content hash for deduplication
            content_str = str(memory.content).strip().lower()
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            if content_hash not in content_hash_to_memory:
                # First time seeing this content
                content_hash_to_memory[content_hash] = memory
            else:
                # Merge with existing memory
                existing = content_hash_to_memory[content_hash]
                existing.relevance_score = max(existing.relevance_score, memory.relevance_score)
                existing.confidence = max(existing.confidence, memory.confidence)
                existing.access_frequency += memory.access_frequency
                existing.importance_weight = max(existing.importance_weight, memory.importance_weight)
                
                # Merge tags (avoid duplicates)
                existing.tags = list(set(existing.tags + memory.tags))
                
                # Merge metadata
                existing.metadata.update(memory.metadata)
        
        deduplicated = list(content_hash_to_memory.values())
        logger.debug(f"Deduplicated {len(memories)} memories to {len(deduplicated)}")
        return deduplicated
    
    def _extract_memory_provenance(self, memories: List[Memory]) -> List[str]:
        """
        Extract provenance information from memories
        
        Args:
            memories: List of Memory objects
            
        Returns:
            List of provenance sources with details
        """
        provenance = []
        
        for memory in memories:
            source = memory.source
            metadata = memory.metadata
            
            # Create detailed provenance entry
            if "url" in metadata:
                provenance.append(f"{source}: {metadata['url']}")
            elif "memory_layer" in metadata:
                provenance.append(f"{source}: {metadata['memory_layer']} layer")
            elif "entity_type" in metadata:
                provenance.append(f"{source}: {metadata['entity_type']} entity")
            elif "file_path" in metadata:
                provenance.append(f"{source}: {metadata['file_path']}")
            else:
                provenance.append(f"{source}: {memory.memory_type} memory")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_provenance = []
        for item in provenance:
            if item not in seen:
                seen.add(item)
                unique_provenance.append(item)
        
        return unique_provenance
    
    def assemble_context(self, query: str, memories: List[Memory], 
                        role: Optional[str] = None) -> ContextPack:
        """
        Assemble memories into structured markdown context with provenance
        
        Args:
            query: Original query for relevance calculation
            memories: List of Memory objects from different sources
            role: Target agent role for role-specific filtering
            
        Returns:
            Structured ContextPack with formatted context
        """
        # Apply role-based filtering if specified
        if role:
            memories = self.prioritize_for_role(memories, role)
        
        # Deduplicate memories
        memories = self._deduplicate_memories(memories)
        
        # Generate pack ID
        pack_id = hashlib.md5(f"{role or 'default'}_{query}_{time.time()}".encode()).hexdigest()[:16]
        
        # Extract provenance and sources
        provenance = self._extract_memory_provenance(memories)
        sources = list(set([m.source for m in memories]))
        
        # Process memories into content sections
        content_sections = []
        
        for i, memory in enumerate(memories):
            content = str(memory.content)
            if not content or len(content.strip()) < 5:  # Skip empty/minimal content
                continue
            
            # Classify content type
            content_type = self._classify_content_type(content)
            
            # Calculate final relevance score incorporating memory metadata
            relevance = self._calculate_memory_relevance(
                memory, query, content_type, role or "default"
            )
            
            # Create content section
            section = ContentSection(
                section_id=f"{memory.source}_{i}",
                section_type=content_type,
                title=f"{content_type.replace('_', ' ').title()} from {memory.source}",
                content=content,
                source=memory.source,
                relevance_score=relevance,
                metadata=memory.metadata,
                tags=memory.tags
            )
            
            content_sections.append(section)
        
        # Sort sections by relevance and group by type
        content_sections.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Calculate overall pack metrics
        if content_sections:
            overall_relevance = sum(s.relevance_score for s in content_sections) / len(content_sections)
            overall_confidence = sum(m.confidence for m in memories) / len(memories)
        else:
            overall_relevance = 0.0
            overall_confidence = 0.0
        
        # Create context pack
        context_pack = ContextPack(
            pack_id=pack_id,
            role_context=role or "default",
            task_context=query,
            content_sections=content_sections,
            provenance=provenance,
            sources=sources,
            relevance_score=overall_relevance,
            confidence=overall_confidence,
            metadata={
                "query": query,
                "memory_count": len(memories),
                "source_count": len(sources),
                "sections_created": len(content_sections),
                "avg_importance": sum(m.importance_weight for m in memories) / len(memories) if memories else 0.0
            }
        )
        
        logger.info(f"Assembled context pack {pack_id} with {len(content_sections)} sections, "
                   f"relevance: {overall_relevance:.3f}, confidence: {overall_confidence:.3f}")
        
        return context_pack
    
    def prioritize_for_role(self, memories: List[Memory], role: str) -> List[Memory]:
        """
        Rank and filter memories based on role relevance
        
        Args:
            memories: List of Memory objects
            role: Target agent role
            
        Returns:
            Ordered list of memories prioritized for role
        """
        return self._prioritize_memories_by_role(memories, role)
    
    def prioritize_by_role(self, memories: List[Memory], role: str) -> List[Memory]:
        """
        Prioritize memories based on role relevance for SCWT compatibility
        
        Args:
            memories: List of Memory objects
            role: Target agent role
            
        Returns:
            Ordered list of memories prioritized for role
        """
        return self._prioritize_memories_by_role(memories, role)
    
    def _prioritize_memories_by_role(self, memories: List[Memory], role: str) -> List[Memory]:
        """
        Internal method to prioritize memories by role
        
        Args:
            memories: List of Memory objects
            role: Target agent role
            
        Returns:
            Ordered list of memories prioritized for role
        """
        role_priorities = self.role_priorities.get(role, self.role_priorities["default"])
        
        # Adjust relevance scores based on role priorities
        for memory in memories:
            # Classify content and apply role priority
            content_type = self._classify_content_type(str(memory.content))
            priority_multiplier = role_priorities.get(content_type, 0.5)
            
            # Calculate role-adjusted score combining multiple factors
            role_adjusted_score = (
                memory.relevance_score * 0.4 +  # Base relevance
                priority_multiplier * 0.3 +     # Role priority
                memory.confidence * 0.2 +       # Confidence
                memory.importance_weight * 0.1  # Importance
            )
            
            # Store adjusted score in metadata for sorting
            memory.metadata["role_adjusted_score"] = role_adjusted_score
        
        # Sort by adjusted score
        memories.sort(key=lambda m: m.metadata.get("role_adjusted_score", 0), reverse=True)
        return memories
    
    def generate_markdown(self, context_pack: ContextPack) -> str:
        """
        Generate structured Markdown from context pack
        
        Args:
            context_pack: ContextPack to render
            
        Returns:
            Formatted Markdown string
        """
        # Header section
        markdown = f"""# Knowledge Pack: {context_pack.task_context}

**Pack ID**: `{context_pack.pack_id}`  
**Role**: {context_pack.role_context}  
**Generated**: {datetime.fromtimestamp(context_pack.created_at).strftime('%Y-%m-%d %H:%M:%S')}  
**Relevance Score**: {context_pack.relevance_score:.1%}

## [SUMMARY]

This knowledge pack contains {len(context_pack.content_sections)} sections compiled from {len(context_pack.provenance)} sources, optimized for the **{context_pack.role_context}** role.

"""
        
        # Group sections by type
        sections_by_type = {}
        for section in context_pack.content_sections:
            section_type = section.section_type
            if section_type not in sections_by_type:
                sections_by_type[section_type] = []
            sections_by_type[section_type].append(section)
        
        # Generate sections in priority order for the role
        role_priorities = self.role_priorities.get(context_pack.role_context, self.role_priorities["default"])
        sorted_types = sorted(sections_by_type.keys(), 
                            key=lambda x: role_priorities.get(x, 0), reverse=True)
        
        for section_type in sorted_types:
            sections = sections_by_type[section_type]
            
            # Section header with prefix
            prefix_map = {
                "implementation_guide": "[IMPL]",
                "examples": "[CODE]",
                "patterns": "[ARCH]",
                "testing": "[TEST]",
                "security": "[SEC]",
                "architecture": "[SYS]"
            }
            prefix = prefix_map.get(section_type, "[INFO]")
            
            markdown += f"\n## {prefix} {section_type.replace('_', ' ').title()}\n\n"
            
            # Add sections
            for section in sections:
                markdown += f"### {section.title}\n\n"
                markdown += f"**Relevance**: {section.relevance_score:.1%} | **Source**: {section.source}\n\n"
                
                # Format content
                content = section.content.strip()
                if len(content) > 1000:  # Truncate very long content
                    content = content[:1000] + "...\n\n*[Content truncated - see full source]*"
                
                markdown += f"{content}\n\n"
                
                # Add metadata if available
                if section.metadata:
                    interesting_metadata = {k: v for k, v in section.metadata.items() 
                                          if k not in ['content', 'raw_content'] and str(v)}
                    if interesting_metadata:
                        markdown += f"*Metadata*: {json.dumps(interesting_metadata, indent=None)}\n\n"
                
                markdown += "---\n\n"
        
        # Provenance section
        markdown += "## [SRC] Sources and Provenance\n\n"
        for i, source in enumerate(context_pack.provenance, 1):
            markdown += f"{i}. {source}\n"
        
        # Footer with pack metadata
        markdown += f"\n## [STATS] Pack Statistics\n\n"
        markdown += f"- **Total Sections**: {len(context_pack.content_sections)}\n"
        markdown += f"- **Source Strategies**: {context_pack.metadata.get('strategy_count', 0)}\n"
        markdown += f"- **Raw Results**: {context_pack.metadata.get('total_results', 0)}\n"
        markdown += f"- **Query**: `{context_pack.metadata.get('query', 'N/A')}`\n"
        markdown += f"- **Pack Generation Time**: {datetime.fromtimestamp(context_pack.created_at).isoformat()}\n"
        
        markdown += f"\n---\n*Generated by Archon+ Phase 4 Context Assembler*\n\n**Context Pack Metadata:**\n"
        markdown += f"- **Confidence**: {context_pack.confidence:.1%}\n"
        markdown += f"- **Memory Types**: {', '.join(set(s.source for s in context_pack.content_sections))}\n"
        
        return markdown
    
    def save_context_pack(self, context_pack: ContextPack, 
                         output_dir: Optional[Path] = None) -> Path:
        """
        Save context pack as JSON and Markdown files
        
        Args:
            context_pack: ContextPack to save
            output_dir: Output directory (defaults to memory storage)
            
        Returns:
            Path to saved Markdown file
        """
        if not output_dir:
            output_dir = Path("python/src/agents/memory/storage/context_packs")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_dir / f"{context_pack.pack_id}.json"
        json_data = {
            "pack_id": context_pack.pack_id,
            "role_context": context_pack.role_context,
            "task_context": context_pack.task_context,
            "relevance_score": context_pack.relevance_score,
            "created_at": context_pack.created_at,
            "metadata": context_pack.metadata,
            "provenance": context_pack.provenance,
            "content_sections": [
                {
                    "section_id": section.section_id,
                    "section_type": section.section_type,
                    "title": section.title,
                    "content": section.content,
                    "source": section.source,
                    "relevance_score": section.relevance_score,
                    "metadata": section.metadata,
                    "tags": section.tags
                }
                for section in context_pack.content_sections
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save as Markdown
        markdown_path = output_dir / f"{context_pack.pack_id}.md"
        markdown_content = self.generate_markdown(context_pack)
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Saved context pack to {markdown_path}")
        return markdown_path
    
    def to_markdown(self, context_pack: ContextPack) -> str:
        """
        Generate structured Markdown from context pack (alias for generate_markdown)
        
        Args:
            context_pack: ContextPack to render
            
        Returns:
            Formatted Markdown string
        """
        return self.generate_markdown(context_pack)
    
    def load_context_pack(self, pack_id: str, 
                         storage_dir: Optional[Path] = None) -> Optional[ContextPack]:
        """
        Load context pack from storage
        
        Args:
            pack_id: Pack ID to load
            storage_dir: Storage directory to search
            
        Returns:
            ContextPack if found, None otherwise
        """
        if not storage_dir:
            storage_dir = Path("python/src/agents/memory/storage/context_packs")
        
        json_path = storage_dir / f"{pack_id}.json"
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct content sections
            content_sections = []
            for section_data in data.get("content_sections", []):
                section = ContentSection(
                    section_id=section_data["section_id"],
                    section_type=section_data["section_type"],
                    title=section_data["title"],
                    content=section_data["content"],
                    source=section_data["source"],
                    relevance_score=section_data["relevance_score"],
                    metadata=section_data.get("metadata", {}),
                    tags=section_data.get("tags", [])
                )
                content_sections.append(section)
            
            # Reconstruct context pack
            context_pack = ContextPack(
                pack_id=data["pack_id"],
                role_context=data["role_context"],
                task_context=data["task_context"],
                content_sections=content_sections,
                provenance=data.get("provenance", []),
                relevance_score=data["relevance_score"],
                created_at=data["created_at"],
                metadata=data.get("metadata", {})
            )
            
            return context_pack
            
        except Exception as e:
            logger.error(f"Failed to load context pack {pack_id}: {e}")
            return None

    @classmethod
    def from_memory_entry(cls, entry) -> Memory:
        """
        Convert MemoryEntry to Memory object
        
        Args:
            entry: MemoryEntry from memory service
            
        Returns:
            Memory object for context assembly
        """
        return Memory(
            memory_id=entry.entry_id,
            content=entry.content,
            memory_type="entry",
            source=f"{entry.memory_layer.value}_layer",
            relevance_score=entry.importance_score,
            confidence=1.0,  # Default confidence for memory entries
            tags=entry.tags,
            metadata={
                **entry.metadata,
                "memory_layer": entry.memory_layer.value,
                "source_agent": entry.source_agent,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed
            },
            created_at=entry.created_at,
            importance_weight=entry.importance_score,
            access_frequency=entry.access_count
        )
    
    @classmethod
    def from_graph_entity(cls, entity: GraphEntity) -> Memory:
        """
        Convert GraphEntity to Memory object
        
        Args:
            entity: GraphEntity from Graphiti service
            
        Returns:
            Memory object for context assembly
        """
        return Memory(
            memory_id=entity.entity_id,
            content={
                "name": entity.name,
                "type": entity.entity_type.value,
                "attributes": entity.attributes
            },
            memory_type="entity",
            source="graphiti_graph",
            relevance_score=entity.importance_weight,
            confidence=entity.confidence_score,
            tags=entity.tags,
            metadata={
                **entity.attributes,
                "entity_type": entity.entity_type.value,
                "creation_time": entity.creation_time,
                "modification_time": entity.modification_time
            },
            created_at=entity.creation_time,
            importance_weight=entity.importance_weight,
            access_frequency=entity.access_frequency
        )
    
    def optimize_context_size(self, context_pack: ContextPack, max_sections: int = 20) -> ContextPack:
        """
        Optimize context pack size by limiting sections and content
        
        Args:
            context_pack: Original context pack
            max_sections: Maximum number of sections to keep
            
        Returns:
            Optimized context pack
        """
        if len(context_pack.content_sections) <= max_sections:
            return context_pack
        
        # Keep top sections by relevance score
        optimized_sections = context_pack.content_sections[:max_sections]
        
        # Recalculate overall metrics
        overall_relevance = sum(s.relevance_score for s in optimized_sections) / len(optimized_sections)
        
        # Create optimized pack
        optimized_pack = ContextPack(
            pack_id=context_pack.pack_id + "_opt",
            role_context=context_pack.role_context,
            task_context=context_pack.task_context,
            content_sections=optimized_sections,
            provenance=context_pack.provenance,
            sources=context_pack.sources,
            relevance_score=overall_relevance,
            confidence=context_pack.confidence,
            metadata={
                **context_pack.metadata,
                "optimized": True,
                "original_sections": len(context_pack.content_sections),
                "optimized_sections": len(optimized_sections)
            }
        )
        
        logger.info(f"Optimized context pack from {len(context_pack.content_sections)} to {len(optimized_sections)} sections")
        return optimized_pack

# Factory functions
def create_context_assembler() -> ContextAssembler:
    """Create a configured context assembler instance"""
    return ContextAssembler()

def create_memory_from_dict(data: Dict[str, Any]) -> Memory:
    """
    Create Memory object from dictionary data
    
    Args:
        data: Dictionary with memory data
        
    Returns:
        Memory object
    """
    return Memory(
        memory_id=data.get("id", str(time.time())),
        content=data.get("content", ""),
        memory_type=data.get("type", "unknown"),
        source=data.get("source", "unknown"),
        relevance_score=data.get("relevance", 0.5),
        confidence=data.get("confidence", 1.0),
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
        created_at=data.get("created_at", time.time()),
        importance_weight=data.get("importance", 0.5),
        access_frequency=data.get("access_count", 0)
    )