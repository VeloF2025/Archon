"""
Knowledge Ingestion Pipeline
Processes and imports knowledge into the graph database
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib
from pydantic import BaseModel, Field
import asyncio
from collections import defaultdict

from .graph_client import Neo4jClient, GraphNode, GraphRelationship
from ...server.services.embeddings import create_embedding
from ...agents.pattern_recognition.pattern_detector import PatternDetector

logger = logging.getLogger(__name__)


class KnowledgeConcept(BaseModel):
    """Represents a knowledge concept to be ingested"""
    name: str
    category: str
    description: Optional[str] = None
    source: str  # Where this knowledge came from
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    relevance_score: float = 1.0
    confidence: float = 1.0


class KnowledgeRelation(BaseModel):
    """Represents a relationship between concepts"""
    source_concept: str
    target_concept: str
    relationship_type: str
    strength: float = 1.0
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeIngestionPipeline:
    """Pipeline for ingesting knowledge into the graph"""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph = graph_client
        self.pattern_detector = PatternDetector()
        self.concept_cache = {}
        self.ingestion_stats = defaultdict(int)
    
    async def ingest_from_code(
        self,
        code: str,
        language: str,
        source_file: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest knowledge from source code
        
        Args:
            code: Source code to analyze
            language: Programming language
            source_file: Source file path
            project_id: Associated project
            
        Returns:
            Ingestion results
        """
        try:
            logger.info(f"Ingesting knowledge from {source_file}")
            
            # Detect patterns in code
            patterns = await self.pattern_detector.detect_patterns(code, language)
            
            concepts = []
            relations = []
            
            # Create concept for the source file
            file_concept = KnowledgeConcept(
                name=source_file,
                category="SourceFile",
                description=f"{language} source file",
                source=source_file,
                metadata={
                    "language": language,
                    "project_id": project_id,
                    "line_count": len(code.split('\n'))
                }
            )
            concepts.append(file_concept)
            
            # Create concepts for detected patterns
            for pattern in patterns:
                pattern_concept = KnowledgeConcept(
                    name=pattern.name,
                    category="Pattern",
                    description=f"{pattern.category} pattern in {language}",
                    source=source_file,
                    metadata={
                        "pattern_id": pattern.id,
                        "language": language,
                        "confidence": pattern.confidence,
                        "is_antipattern": pattern.is_antipattern
                    },
                    confidence=pattern.confidence
                )
                concepts.append(pattern_concept)
                
                # Create relationship between file and pattern
                relations.append(KnowledgeRelation(
                    source_concept=source_file,
                    target_concept=pattern.name,
                    relationship_type="CONTAINS_PATTERN",
                    strength=pattern.confidence,
                    evidence=[f"Detected in {source_file}"]
                ))
            
            # Extract additional concepts from code
            code_concepts = await self._extract_code_concepts(code, language)
            concepts.extend(code_concepts)
            
            # Create relationships between patterns
            pattern_relations = self._analyze_pattern_relationships(patterns)
            relations.extend(pattern_relations)
            
            # Ingest all concepts and relationships
            result = await self._ingest_concepts_and_relations(concepts, relations)
            
            self.ingestion_stats["files_processed"] += 1
            self.ingestion_stats["patterns_detected"] += len(patterns)
            
            return {
                "success": True,
                "source_file": source_file,
                "concepts_created": result["concepts_created"],
                "relationships_created": result["relationships_created"],
                "patterns_detected": len(patterns),
                "stats": dict(self.ingestion_stats)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting from code: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def ingest_from_documentation(
        self,
        content: str,
        doc_type: str,
        source: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest knowledge from documentation
        
        Args:
            content: Documentation content
            doc_type: Type of documentation (README, API, etc.)
            source: Source identifier
            project_id: Associated project
            
        Returns:
            Ingestion results
        """
        try:
            logger.info(f"Ingesting knowledge from {doc_type}: {source}")
            
            concepts = []
            relations = []
            
            # Create concept for the document
            doc_concept = KnowledgeConcept(
                name=source,
                category="Documentation",
                description=f"{doc_type} documentation",
                source=source,
                metadata={
                    "doc_type": doc_type,
                    "project_id": project_id,
                    "content_length": len(content)
                }
            )
            concepts.append(doc_concept)
            
            # Extract concepts from content
            extracted_concepts = await self._extract_doc_concepts(content, doc_type)
            concepts.extend(extracted_concepts)
            
            # Create relationships
            for concept in extracted_concepts:
                relations.append(KnowledgeRelation(
                    source_concept=source,
                    target_concept=concept.name,
                    relationship_type="DOCUMENTS",
                    strength=concept.relevance_score,
                    evidence=[f"Mentioned in {source}"]
                ))
            
            # Ingest all concepts and relationships
            result = await self._ingest_concepts_and_relations(concepts, relations)
            
            self.ingestion_stats["docs_processed"] += 1
            
            return {
                "success": True,
                "source": source,
                "concepts_created": result["concepts_created"],
                "relationships_created": result["relationships_created"],
                "stats": dict(self.ingestion_stats)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting from documentation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def ingest_project_structure(
        self,
        project_id: str,
        project_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest project structure and relationships
        
        Args:
            project_id: Project identifier
            project_data: Project metadata and structure
            
        Returns:
            Ingestion results
        """
        try:
            logger.info(f"Ingesting project structure: {project_id}")
            
            concepts = []
            relations = []
            
            # Create project concept
            project_concept = KnowledgeConcept(
                name=project_data.get("name", project_id),
                category="Project",
                description=project_data.get("description", ""),
                source=f"project_{project_id}",
                metadata={
                    "project_id": project_id,
                    "created_at": project_data.get("created_at"),
                    "status": project_data.get("status", "active")
                }
            )
            concepts.append(project_concept)
            
            # Process project components
            if "components" in project_data:
                for component in project_data["components"]:
                    comp_concept = KnowledgeConcept(
                        name=component["name"],
                        category="Component",
                        description=component.get("description", ""),
                        source=f"project_{project_id}",
                        metadata=component
                    )
                    concepts.append(comp_concept)
                    
                    relations.append(KnowledgeRelation(
                        source_concept=project_concept.name,
                        target_concept=component["name"],
                        relationship_type="HAS_COMPONENT",
                        strength=1.0
                    ))
            
            # Process dependencies
            if "dependencies" in project_data:
                for dep in project_data["dependencies"]:
                    dep_concept = KnowledgeConcept(
                        name=dep["name"],
                        category="Dependency",
                        description=f"Version: {dep.get('version', 'latest')}",
                        source=f"project_{project_id}",
                        metadata=dep
                    )
                    concepts.append(dep_concept)
                    
                    relations.append(KnowledgeRelation(
                        source_concept=project_concept.name,
                        target_concept=dep["name"],
                        relationship_type="DEPENDS_ON",
                        strength=1.0
                    ))
            
            # Ingest all concepts and relationships
            result = await self._ingest_concepts_and_relations(concepts, relations)
            
            self.ingestion_stats["projects_processed"] += 1
            
            return {
                "success": True,
                "project_id": project_id,
                "concepts_created": result["concepts_created"],
                "relationships_created": result["relationships_created"],
                "stats": dict(self.ingestion_stats)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting project structure: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _extract_code_concepts(
        self,
        code: str,
        language: str
    ) -> List[KnowledgeConcept]:
        """Extract concepts from code"""
        concepts = []
        
        # Extract imports/dependencies
        if language == "python":
            import_lines = [line for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
            for line in import_lines[:10]:  # Limit to avoid too many
                module = line.split()[1].split('.')[0]
                concepts.append(KnowledgeConcept(
                    name=module,
                    category="Library",
                    description=f"Python module",
                    source="code_analysis",
                    metadata={"language": "python"}
                ))
        
        elif language in ["javascript", "typescript"]:
            import_lines = [line for line in code.split('\n') if 'import' in line or 'require' in line]
            for line in import_lines[:10]:
                if 'from' in line:
                    module = line.split('from')[1].strip().strip('"\'`;')
                elif 'require' in line:
                    module = line.split('require')[1].strip().strip('()"\';')
                else:
                    continue
                    
                concepts.append(KnowledgeConcept(
                    name=module,
                    category="Library",
                    description=f"JavaScript module",
                    source="code_analysis",
                    metadata={"language": language}
                ))
        
        return concepts
    
    async def _extract_doc_concepts(
        self,
        content: str,
        doc_type: str
    ) -> List[KnowledgeConcept]:
        """Extract concepts from documentation"""
        concepts = []
        
        # Simple keyword extraction (in production, use NLP)
        keywords = {
            "API": ["endpoint", "request", "response", "REST", "GraphQL"],
            "Architecture": ["microservice", "database", "cache", "queue", "service"],
            "Framework": ["React", "Vue", "Angular", "Django", "Flask", "Express"],
            "Technology": ["Docker", "Kubernetes", "AWS", "Azure", "GCP"],
            "Pattern": ["singleton", "factory", "observer", "MVC", "MVP"]
        }
        
        content_lower = content.lower()
        
        for category, terms in keywords.items():
            for term in terms:
                if term.lower() in content_lower:
                    concepts.append(KnowledgeConcept(
                        name=term,
                        category=category,
                        description=f"Found in {doc_type}",
                        source="doc_analysis",
                        relevance_score=0.7
                    ))
        
        return concepts[:20]  # Limit to avoid too many
    
    def _analyze_pattern_relationships(
        self,
        patterns: List[Any]
    ) -> List[KnowledgeRelation]:
        """Analyze relationships between patterns"""
        relations = []
        
        # Find complementary patterns
        pattern_categories = defaultdict(list)
        for pattern in patterns:
            pattern_categories[pattern.category].append(pattern)
        
        # Patterns in same category might be alternatives
        for category, category_patterns in pattern_categories.items():
            if len(category_patterns) > 1:
                for i, p1 in enumerate(category_patterns):
                    for p2 in category_patterns[i+1:]:
                        relations.append(KnowledgeRelation(
                            source_concept=p1.name,
                            target_concept=p2.name,
                            relationship_type="ALTERNATIVE_TO",
                            strength=0.6,
                            evidence=[f"Both are {category} patterns"]
                        ))
        
        # Antipatterns conflict with good patterns
        good_patterns = [p for p in patterns if not p.is_antipattern]
        antipatterns = [p for p in patterns if p.is_antipattern]
        
        for anti in antipatterns:
            for good in good_patterns:
                if anti.category == good.category:
                    relations.append(KnowledgeRelation(
                        source_concept=anti.name,
                        target_concept=good.name,
                        relationship_type="CONFLICTS_WITH",
                        strength=0.9,
                        evidence=["Antipattern conflicts with pattern"]
                    ))
        
        return relations
    
    async def _ingest_concepts_and_relations(
        self,
        concepts: List[KnowledgeConcept],
        relations: List[KnowledgeRelation]
    ) -> Dict[str, Any]:
        """Ingest concepts and relationships into the graph"""
        concepts_created = 0
        relationships_created = 0
        
        # Ingest concepts
        for concept in concepts:
            try:
                # Generate embedding if not present
                if not concept.embedding:
                    embedding_text = f"{concept.name} {concept.category} {concept.description or ''}"
                    concept.embedding = await create_embedding(embedding_text)
                
                # Check if concept exists
                existing = await self.graph.find_node(
                    labels=["Concept", concept.category],
                    properties={"name": concept.name}
                )
                
                if not existing:
                    # Create new concept node
                    node = await self.graph.create_node(
                        labels=["Concept", concept.category],
                        properties={
                            "id": self._generate_concept_id(concept.name),
                            "name": concept.name,
                            "description": concept.description,
                            "source": concept.source,
                            "embedding": concept.embedding,
                            "relevance_score": concept.relevance_score,
                            "confidence": concept.confidence,
                            "metadata": json.dumps(concept.metadata)
                        }
                    )
                    concepts_created += 1
                    self.concept_cache[concept.name] = node.id
                else:
                    # Update existing concept
                    await self.graph.update_node(
                        existing.id,
                        {
                            "relevance_score": max(existing.properties.get("relevance_score", 0), concept.relevance_score),
                            "confidence": max(existing.properties.get("confidence", 0), concept.confidence)
                        }
                    )
                    self.concept_cache[concept.name] = existing.id
                    
            except Exception as e:
                logger.error(f"Error ingesting concept {concept.name}: {e}")
        
        # Ingest relationships
        for relation in relations:
            try:
                # Get node IDs
                source_id = self.concept_cache.get(relation.source_concept)
                target_id = self.concept_cache.get(relation.target_concept)
                
                if source_id and target_id:
                    # Check if relationship exists
                    existing_rels = await self.graph.find_relationships(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=relation.relationship_type
                    )
                    
                    if not existing_rels:
                        # Create new relationship
                        await self.graph.create_relationship(
                            source_id=source_id,
                            target_id=target_id,
                            relationship_type=relation.relationship_type,
                            properties={
                                "strength": relation.strength,
                                "evidence": json.dumps(relation.evidence),
                                "metadata": json.dumps(relation.metadata)
                            }
                        )
                        relationships_created += 1
                        
            except Exception as e:
                logger.error(f"Error creating relationship: {e}")
        
        return {
            "concepts_created": concepts_created,
            "relationships_created": relationships_created
        }
    
    def _generate_concept_id(self, name: str) -> str:
        """Generate unique ID for concept"""
        return hashlib.md5(f"concept_{name}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
    
    async def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        graph_stats = await self.graph.get_statistics()
        
        return {
            "ingestion_stats": dict(self.ingestion_stats),
            "graph_stats": graph_stats,
            "concept_cache_size": len(self.concept_cache)
        }