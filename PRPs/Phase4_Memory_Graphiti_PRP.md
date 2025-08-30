# Product Requirements Prompt (PRP): Phase 4 - Memory Service and Graphiti Integration

## 📋 **Metadata**
- **PRP ID**: ARCH-P4-001
- **Phase**: 4 of 6
- **Project**: Archon+ Enhanced AI Coding System
- **Task ID**: memory-service-graphiti-001
- **Priority**: High
- **Estimated Duration**: 3-5 weeks
- **Dependencies**: Phase 3 validation system, existing RAG service

## 🎯 **Objective**
Implement layered memory service with role-specific scopes, adaptive retrieval with bandit weights, Graphiti temporal knowledge graphs using Kuzu database, and UI explorer for graph visualization - enabling ≥30% knowledge reuse and ≥85% retrieval precision.

## 🔄 **Context & Background**
Current Archon has basic RAG capabilities with vector search, hybrid search, and reranking. Phase 4 enhances this with:
- **Layered memory scopes** (global/project/job/runtime) with role-specific access
- **Adaptive retrieval** with bandit algorithms optimizing search strategies
- **Graphiti temporal graphs** for entity/relationship tracking over time
- **Context assembler** for PRP-like Markdown knowledge packs
- **UI integration** for graph exploration and temporal filtering

## 🏗️ **Implementation Requirements**

### **Memory Service Architecture**
```
Memory Layers (Role-Specific Access):
├── Global Memory (System-wide patterns, best practices)
├── Project Memory (Project-specific context, decisions)
├── Job Memory (Current session/task context)
└── Runtime Memory (Immediate execution context)

Access Control:
├── code-implementer: Job + Runtime + Project patterns
├── system-architect: Global + Project + Job decisions
├── security-auditor: Global security + Project vulnerabilities
└── [All 20+ agents with specific scope configurations]
```

### **Adaptive Retrieval System**
```
Retrieval Pipeline:
├── Strategy Selection (Bandit Algorithm)
│   ├── Vector Search (Embedding similarity)
│   ├── Hybrid Search (Vector + Keyword)
│   ├── Graphiti Search (Entity/relationship traversal)
│   └── Memory Scope Search (Role-specific contexts)
├── Result Fusion (Weighted combination)
├── Reranking (CrossEncoder + Temporal relevance)
└── Context Assembly (PRP-like Markdown packs)
```

### **Graphiti Temporal Knowledge Graphs**
```
Kuzu Database Schema:
├── Entities (code_functions, agents, projects, requirements)
├── Relationships (calls, implements, validates, references)
├── Temporal Tracking (creation_time, modification_time, access_frequency)
└── Attributes (confidence_scores, importance_weights, user_feedback)

Graph Operations:
├── Entity Ingestion (Auto-extraction from code/docs/interactions)
├── Relationship Discovery (Static analysis + runtime observation)  
├── Temporal Queries (Recent patterns, trending entities, decay models)
└── Confidence Propagation (Trust scores through relationship paths)
```

## 📋 **Acceptance Criteria & Test Specifications**

### **AC-001: Memory Service Layer Management**
**Given** an agent with role-specific memory access configured
**When** the agent requests context from different memory layers
**Then** it should only access scopes defined in its role configuration
**And** memory should be persisted across sessions for the appropriate layers

**Test Requirements:**
- Unit tests for memory scope validation and access control
- Integration tests for cross-layer memory retrieval
- Performance tests for memory query response times (<100ms)

### **AC-002: Adaptive Retrieval with Bandit Optimization**
**Given** multiple retrieval strategies available (vector, hybrid, graphiti, memory)
**When** a query is processed by the adaptive retriever
**Then** it should select the optimal strategy combination based on historical performance
**And** update strategy weights using bandit algorithm feedback

**Test Requirements:**
- Unit tests for bandit algorithm implementation and strategy selection
- Integration tests for multi-strategy result fusion and ranking
- Performance benchmarks showing ≥85% retrieval precision improvement

### **AC-003: Graphiti Temporal Knowledge Graph Operations**
**Given** a Kuzu database configured for temporal graph storage
**When** entities and relationships are ingested from code/docs/interactions
**Then** the system should maintain temporal tracking and confidence scores
**And** support queries for entity evolution, relationship patterns, and decay models

**Test Requirements:**
- Unit tests for entity/relationship extraction and ingestion
- Integration tests for temporal queries and confidence propagation
- Graph consistency tests ensuring referential integrity

### **AC-004: Context Assembler for PRP-like Knowledge Packs**
**Given** retrieval results from multiple sources (vector, graphiti, memory)
**When** context assembly is requested for a specific task/agent
**Then** it should generate structured Markdown packs with provenance
**And** include role-specific context prioritization and relevance scoring

**Test Requirements:**
- Unit tests for Markdown generation and provenance tracking
- Integration tests for multi-source context fusion and prioritization
- Quality tests ensuring context relevance and coherence

### **AC-005: UI Graphiti Explorer Integration**
**Given** the Graphiti temporal knowledge graph populated with data
**When** users access the UI explorer interface
**Then** they should see interactive graph visualization with temporal filtering
**And** be able to explore entity relationships, evolution, and confidence metrics

**Test Requirements:**
- UI component tests for graph visualization and interaction
- Integration tests for real-time graph updates and filtering
- Usability tests showing ≥10% CLI reduction in knowledge exploration

## 🔧 **Technical Implementation Details**

### **File Structure:**
```
python/src/agents/memory/
├── __init__.py
├── memory_service.py          # Layered memory management
├── memory_scopes.py           # Role-specific access control
├── adaptive_retriever.py      # Bandit-optimized strategy selection
├── context_assembler.py       # PRP-like Markdown pack generation
└── tests/
    ├── test_memory_service.py
    ├── test_adaptive_retriever.py
    └── test_context_assembler.py

python/src/agents/graphiti/
├── __init__.py
├── graphiti_service.py        # Main Kuzu graph operations
├── entity_extractor.py       # Auto-extraction from code/docs
├── temporal_queries.py       # Time-based graph queries
├── confidence_propagation.py # Trust score algorithms
└── tests/
    ├── test_graphiti_service.py
    ├── test_entity_extraction.py
    └── test_temporal_queries.py

archon-ui-main/src/components/graphiti/
├── GraphExplorer.tsx         # Main graph visualization
├── TemporalFilter.tsx        # Time-based filtering
├── EntityDetails.tsx         # Entity detail views
└── tests/
    ├── GraphExplorer.test.tsx
    └── TemporalFilter.test.tsx
```

### **Dependencies:**
- **Kuzu**: Graph database for temporal knowledge storage
- **networkx**: Graph algorithms and analysis
- **scikit-learn**: Bandit algorithms for adaptive retrieval
- **React Flow**: UI graph visualization library

### **Integration Points:**
- Existing RAG service enhancement with memory/graphiti retrieval
- Meta-agent integration for memory scope management
- Validator integration for knowledge quality assurance
- REF Tools integration for external documentation context

## 🧪 **Testing Strategy**

### **Unit Testing:**
- Memory service layer access control and persistence
- Adaptive retrieval strategy selection and optimization
- Graphiti entity extraction and relationship discovery
- Context assembler Markdown generation and provenance

### **Integration Testing:**
- End-to-end knowledge retrieval across all sources
- Memory service integration with existing agents
- Graphiti integration with RAG pipeline
- UI explorer real-time updates and interaction

### **Performance Testing:**
- Memory query response times (<100ms)
- Adaptive retrieval precision (≥85%)
- Graphiti query performance with temporal filtering
- Knowledge reuse measurement (≥30%)

### **SCWT Benchmark Testing:**
- Knowledge reuse: ≥30% in context packs
- Retrieval precision: ≥85% relevance scoring
- Graphiti impact: ≥10% precision boost over base RAG
- UI usability: ≥10% CLI reduction in knowledge tasks

## 🚀 **Deployment & Rollout**

### **Phase 4.1: Memory Service Foundation (Week 1-2)**
- Implement layered memory service with role-specific scopes
- Create memory persistence and access control systems
- Integrate with existing agent architecture

### **Phase 4.2: Adaptive Retrieval Implementation (Week 2-3)**
- Build bandit algorithm for strategy optimization
- Implement multi-strategy result fusion and ranking
- Integrate with existing RAG service pipeline

### **Phase 4.3: Graphiti Temporal Graphs (Week 3-4)**
- Set up Kuzu database and schema design
- Implement entity extraction and relationship discovery
- Build temporal query capabilities and confidence propagation

### **Phase 4.4: Context Assembly & UI (Week 4-5)**
- Develop PRP-like Markdown context assembler
- Create UI Graphiti explorer with graph visualization
- Implement temporal filtering and entity detail views

### **Phase 4.5: Integration & Optimization (Week 5)**
- Integrate all components with meta-agent and validator
- Performance optimization and SCWT benchmark testing
- Documentation and deployment preparation

## 📊 **Success Metrics**

### **Primary Metrics (SCWT Measured):**
- **Knowledge Reuse**: ≥30% of context from memory/graphiti/existing sources
- **Retrieval Precision**: ≥85% relevance in assembled context packs
- **Graphiti Impact**: ≥10% precision boost over baseline RAG
- **UI Usability**: ≥10% reduction in CLI usage for knowledge tasks

### **Secondary Metrics:**
- Memory query response time: <100ms average
- Adaptive retrieval optimization: >95% strategy selection accuracy
- Graph ingestion rate: >1000 entities/minute processing
- Context assembly quality: >90% coherence in generated packs

### **Quality Gates:**
- All unit tests passing with >95% coverage
- Integration tests showing seamless cross-component operation
- Performance benchmarks meeting or exceeding targets
- SCWT validation demonstrating measurable improvements

---

**PRP Status**: Ready for implementation with DGTS/NLNH enforcement  
**Next Step**: Create test specifications from these acceptance criteria before implementation  
**Documentation**: Tests must validate all AC requirements with measurable success criteria