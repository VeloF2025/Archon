#!/usr/bin/env python3
"""
Integration Tests for Context Assembler - Archon+ Phase 4

Tests the updated ContextAssembler with Memory objects from MemoryService
and GraphEntity objects from GraphitiService.
"""

import pytest
import time
from pathlib import Path
from typing import List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.memory.context_assembler import (
    ContextAssembler, Memory, ContextPack, ContentSection,
    create_context_assembler, create_memory_from_dict
)
from agents.memory.memory_scopes import MemoryLayerType, AccessLevel, RoleBasedAccessControl
from agents.graphiti.graphiti_service import GraphEntity, EntityType


class TestMemoryObject:
    """Test Memory object creation and conversion"""
    
    def test_memory_creation(self):
        """Test creating Memory objects directly"""
        memory = Memory(
            memory_id="test_001",
            content="This is test content for implementation",
            memory_type="entry",
            source="test_source",
            relevance_score=0.8,
            confidence=0.9,
            tags=["implementation", "test"],
            metadata={"priority": "high"}
        )
        
        assert memory.memory_id == "test_001"
        assert memory.content == "This is test content for implementation"
        assert memory.memory_type == "entry"
        assert memory.source == "test_source"
        assert memory.relevance_score == 0.8
        assert memory.confidence == 0.9
        assert "implementation" in memory.tags
        assert memory.metadata["priority"] == "high"
    
    def test_create_memory_from_dict(self):
        """Test creating Memory from dictionary data"""
        data = {
            "id": "dict_001",
            "content": "Dictionary-based memory content",
            "type": "knowledge",
            "source": "knowledge_base",
            "relevance": 0.7,
            "confidence": 0.85,
            "tags": ["knowledge", "base"],
            "metadata": {"category": "technical"},
            "importance": 0.6,
            "access_count": 5
        }
        
        memory = create_memory_from_dict(data)
        
        assert memory.memory_id == "dict_001"
        assert memory.content == "Dictionary-based memory content"
        assert memory.memory_type == "knowledge"
        assert memory.relevance_score == 0.7
        assert memory.confidence == 0.85
        assert memory.importance_weight == 0.6
        assert memory.access_frequency == 5


class TestContextAssembler:
    """Test ContextAssembler functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.assembler = create_context_assembler()
        
        # Create test memories
        self.test_memories = [
            Memory(
                memory_id="mem_001",
                content="Implementation guide for authentication service using JWT tokens",
                memory_type="entry",
                source="global_layer",
                relevance_score=0.9,
                confidence=0.95,
                tags=["implementation", "authentication", "jwt"],
                metadata={"memory_layer": "global", "language": "python"},
                importance_weight=0.8,
                access_frequency=10
            ),
            Memory(
                memory_id="mem_002", 
                content="Example code for user login validation with proper error handling",
                memory_type="entry",
                source="project_layer",
                relevance_score=0.8,
                confidence=0.9,
                tags=["examples", "login", "validation"],
                metadata={"memory_layer": "project", "file_path": "/src/auth.py"},
                importance_weight=0.7,
                access_frequency=8
            ),
            Memory(
                memory_id="mem_003",
                content="Security patterns for preventing SQL injection attacks",
                memory_type="entry", 
                source="global_layer",
                relevance_score=0.85,
                confidence=0.92,
                tags=["security", "patterns", "sql"],
                metadata={"memory_layer": "global", "priority": "critical"},
                importance_weight=0.9,
                access_frequency=15
            ),
            Memory(
                memory_id="mem_004",
                content="Test cases for authentication module coverage",
                memory_type="entry",
                source="job_layer", 
                relevance_score=0.7,
                confidence=0.8,
                tags=["testing", "authentication", "coverage"],
                metadata={"memory_layer": "job", "test_type": "unit"},
                importance_weight=0.6,
                access_frequency=5
            )
        ]
    
    def test_assemble_context_basic(self):
        """Test basic context assembly"""
        query = "authentication implementation"
        
        context_pack = self.assembler.assemble_context(
            query=query,
            memories=self.test_memories,
            role="code-implementer"
        )
        
        # Verify context pack structure
        assert isinstance(context_pack, ContextPack)
        assert context_pack.role_context == "code-implementer"
        assert context_pack.task_context == query
        assert len(context_pack.content_sections) > 0
        assert context_pack.relevance_score > 0
        assert context_pack.confidence > 0
        assert len(context_pack.sources) > 0
        assert len(context_pack.provenance) > 0
    
    def test_role_based_prioritization(self):
        """Test role-based memory prioritization"""
        query = "security implementation"
        
        # Test with security-auditor role
        security_context = self.assembler.assemble_context(
            query=query,
            memories=self.test_memories,
            role="security-auditor"
        )
        
        # Test with code-implementer role
        implementer_context = self.assembler.assemble_context(
            query=query,
            memories=self.test_memories,
            role="code-implementer"
        )
        
        # Security auditor should prioritize security content higher
        security_sections = security_context.content_sections
        implementer_sections = implementer_context.content_sections
        
        # Find security-related sections in both contexts
        security_section_in_security = next(
            (s for s in security_sections if "security" in s.content.lower()), None
        )
        security_section_in_implementer = next(
            (s for s in implementer_sections if "security" in s.content.lower()), None
        )
        
        # Both should have security content, but security-auditor should rank it higher
        assert security_section_in_security is not None
        assert security_section_in_implementer is not None
        
        # Security auditor should rank security content more relevantly
        security_pos = next(
            i for i, s in enumerate(security_sections) 
            if "security" in s.content.lower()
        )
        implementer_pos = next(
            i for i, s in enumerate(implementer_sections) 
            if "security" in s.content.lower()
        )
        
        assert security_pos <= implementer_pos  # Security content ranked higher or equal for security-auditor
    
    def test_memory_deduplication(self):
        """Test memory deduplication functionality"""
        # Create duplicate memories
        duplicate_memories = self.test_memories + [
            Memory(
                memory_id="dup_001",
                content="Implementation guide for authentication service using JWT tokens",  # Duplicate content
                memory_type="entry",
                source="different_source",
                relevance_score=0.7,
                confidence=0.8,
                tags=["duplicate"],
                metadata={},
                access_frequency=3  # Add some access frequency to test merging
            )
        ]
        
        deduplicated = self.assembler._deduplicate_memories(duplicate_memories)
        
        # Should have removed the duplicate
        assert len(deduplicated) == len(self.test_memories)
        
        # Find memory with the original content (could have either ID after deduplication)
        auth_memory = next(
            m for m in deduplicated 
            if "Implementation guide for authentication service using JWT tokens" in str(m.content)
        )
        
        # Should have enhanced access frequency from merging duplicates (10 + 3 = 13)
        print(f"Original access frequency: 10, Duplicate: 3, Current: {auth_memory.access_frequency}")
        assert auth_memory.access_frequency == 13  # Should be 10 + 3 from merging
    
    def test_markdown_generation(self):
        """Test Markdown generation from context pack"""
        query = "authentication guide"
        
        context_pack = self.assembler.assemble_context(
            query=query,
            memories=self.test_memories,
            role="code-implementer"
        )
        
        markdown = self.assembler.generate_markdown(context_pack)
        
        # Verify markdown structure
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "# Knowledge Pack:" in markdown
        assert "Pack ID" in markdown
        assert "Role" in markdown
        assert "Relevance Score" in markdown
        assert "Confidence" in markdown
        assert "Sources and Provenance" in markdown
        assert "Pack Statistics" in markdown
        
        # Verify content sections are included
        for section in context_pack.content_sections:
            assert section.title in markdown
    
    def test_context_size_optimization(self):
        """Test context size optimization"""
        # Create many memories
        many_memories = []
        for i in range(30):
            many_memories.append(Memory(
                memory_id=f"mem_{i:03d}",
                content=f"Memory content number {i} with relevance scoring",
                memory_type="entry",
                source=f"source_{i % 3}",
                relevance_score=0.5 + (i % 10) * 0.05,  # Varying relevance
                confidence=0.8,
                tags=[f"tag_{i}"],
                metadata={}
            ))
        
        context_pack = self.assembler.assemble_context(
            query="memory content",
            memories=many_memories,
            role="code-implementer"
        )
        
        # Optimize to max 10 sections
        optimized_pack = self.assembler.optimize_context_size(context_pack, max_sections=10)
        
        assert len(optimized_pack.content_sections) == 10
        assert optimized_pack.metadata["optimized"] is True
        assert optimized_pack.metadata["original_sections"] == len(context_pack.content_sections)
        assert optimized_pack.metadata["optimized_sections"] == 10
        
        # Verify top relevance sections are kept
        original_top_10 = context_pack.content_sections[:10]
        optimized_sections = optimized_pack.content_sections
        
        for orig, opt in zip(original_top_10, optimized_sections):
            assert orig.section_id == opt.section_id
    
    def test_relevance_scoring(self):
        """Test enhanced relevance scoring"""
        query = "authentication security"
        
        # Memory with high relevance factors
        high_relevance_memory = Memory(
            memory_id="high_rel",
            content="Authentication security implementation guide with examples",
            memory_type="entry",
            source="expert_source",
            relevance_score=0.9,
            confidence=0.95,
            tags=["authentication", "security", "guide"],
            metadata={"priority": "high"},
            importance_weight=0.9,
            access_frequency=20
        )
        
        # Memory with low relevance factors  
        low_relevance_memory = Memory(
            memory_id="low_rel",
            content="Random unrelated content about databases",
            memory_type="entry",
            source="basic_source",
            relevance_score=0.3,
            confidence=0.6,
            tags=["database"],
            metadata={},
            importance_weight=0.3,
            access_frequency=1
        )
        
        memories = [high_relevance_memory, low_relevance_memory]
        
        # Test relevance calculation
        high_relevance = self.assembler._calculate_memory_relevance(
            high_relevance_memory, query, "implementation_guide", "code-implementer"
        )
        
        low_relevance = self.assembler._calculate_memory_relevance(
            low_relevance_memory, query, "implementation_guide", "code-implementer"
        )
        
        assert high_relevance > low_relevance
        assert high_relevance > 0.7  # Should be high due to matching content and metadata
        assert low_relevance < 0.5   # Should be low due to unrelated content
    
    def test_conversion_methods(self):
        """Test conversion from MemoryEntry and GraphEntity"""
        # Mock MemoryEntry-like object
        class MockMemoryEntry:
            def __init__(self):
                self.entry_id = "mock_entry_001"
                self.content = "Mock memory entry content"
                self.memory_layer = MemoryLayerType.PROJECT
                self.source_agent = "test-agent"
                self.tags = ["mock", "entry"]
                self.metadata = {"test": "value"}
                self.importance_score = 0.7
                self.created_at = time.time()
                self.last_accessed = time.time()
                self.access_count = 3
        
        mock_entry = MockMemoryEntry()
        memory = ContextAssembler.from_memory_entry(mock_entry)
        
        assert memory.memory_id == "mock_entry_001"
        assert memory.content == "Mock memory entry content"
        assert memory.memory_type == "entry"
        assert memory.source == "project_layer"
        assert memory.metadata["memory_layer"] == "project"
        assert memory.metadata["source_agent"] == "test-agent"
        
        # Mock GraphEntity
        graph_entity = GraphEntity(
            entity_id="entity_001",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            attributes={"language": "python", "complexity": "low"},
            confidence_score=0.85,
            importance_weight=0.75,
            tags=["function", "python"],
            access_frequency=12
        )
        
        entity_memory = ContextAssembler.from_graph_entity(graph_entity)
        
        assert entity_memory.memory_id == "entity_001"
        assert entity_memory.memory_type == "entity"
        assert entity_memory.source == "graphiti_graph"
        assert entity_memory.content["name"] == "test_function"
        assert entity_memory.content["type"] == "function"
        assert entity_memory.confidence == 0.85
        assert entity_memory.importance_weight == 0.75


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def test_mixed_memory_sources(self):
        """Test with memories from different sources"""
        assembler = create_context_assembler()
        
        # Mix of memory types
        mixed_memories = [
            Memory(
                memory_id="global_001",
                content="Global pattern for error handling in microservices",
                memory_type="entry",
                source="global_layer",
                tags=["patterns", "microservices", "errors"],
                relevance_score=0.8,
                confidence=0.9
            ),
            Memory(
                memory_id="entity_001", 
                content={"name": "AuthService", "type": "class", "methods": ["login", "logout"]},
                memory_type="entity",
                source="graphiti_graph",
                tags=["class", "authentication"],
                relevance_score=0.85,
                confidence=0.92
            ),
            Memory(
                memory_id="project_001",
                content="Project-specific authentication configuration",
                memory_type="entry", 
                source="project_layer",
                tags=["config", "auth"],
                relevance_score=0.7,
                confidence=0.85
            )
        ]
        
        context_pack = assembler.assemble_context(
            query="authentication service implementation",
            memories=mixed_memories,
            role="system-architect"
        )
        
        assert len(context_pack.content_sections) == len(mixed_memories)
        assert len(context_pack.sources) == 3  # Three different sources
        
        # Verify different memory types are handled
        entity_section = next(
            (s for s in context_pack.content_sections if "AuthService" in str(s.content)), 
            None
        )
        assert entity_section is not None
    
    def test_empty_memories_handling(self):
        """Test handling of empty or invalid memories"""
        assembler = create_context_assembler()
        
        empty_memories = [
            Memory(
                memory_id="empty_001",
                content="",  # Empty content
                memory_type="entry",
                source="test_source"
            ),
            Memory(
                memory_id="short_001", 
                content="hi",  # Too short
                memory_type="entry",
                source="test_source"
            ),
            Memory(
                memory_id="valid_001",
                content="This is valid content for testing purposes",
                memory_type="entry",
                source="test_source",
                relevance_score=0.8
            )
        ]
        
        context_pack = assembler.assemble_context(
            query="test content",
            memories=empty_memories,
            role="code-implementer"
        )
        
        # Should only include the valid memory
        assert len(context_pack.content_sections) == 1
        assert context_pack.content_sections[0].content == "This is valid content for testing purposes"


if __name__ == "__main__":
    # Run basic tests
    print("Running Context Assembler Integration Tests...")
    
    test_memory = TestMemoryObject()
    test_memory.test_memory_creation()
    test_memory.test_create_memory_from_dict()
    print("[PASS] Memory object tests passed")
    
    test_assembler = TestContextAssembler()
    test_assembler.setup_method()
    test_assembler.test_assemble_context_basic()
    test_assembler.test_role_based_prioritization()
    test_assembler.test_memory_deduplication()
    test_assembler.test_markdown_generation()
    test_assembler.test_context_size_optimization()
    test_assembler.test_relevance_scoring()
    test_assembler.test_conversion_methods()
    print("[PASS] Context assembler tests passed")
    
    test_integration = TestIntegrationScenarios()
    test_integration.test_mixed_memory_sources()
    test_integration.test_empty_memories_handling()
    print("[PASS] Integration scenario tests passed")
    
    print("\n[SUCCESS] All Context Assembler tests completed successfully!")
    print("\nFeatures validated:")
    print("- Memory object creation and conversion")
    print("- Context assembly from Memory objects")  
    print("- Role-based prioritization")
    print("- Memory deduplication")
    print("- Relevance scoring with memory factors")
    print("- Markdown generation")
    print("- Context size optimization")
    print("- Integration with MemoryEntry and GraphEntity")
    print("- Mixed memory source handling")
    print("- Error handling for invalid inputs")