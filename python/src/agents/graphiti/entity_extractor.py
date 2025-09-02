#!/usr/bin/env python3
"""
Entity Extractor for Graphiti Temporal Knowledge Graphs

Automatically extracts entities and relationships from:
- Code files (Python, TypeScript, JavaScript, etc.)
- Documentation (Markdown, text files)
- Agent interactions and outputs
- Project structures and configurations

Feeds extracted knowledge into the Graphiti temporal graph for pattern recognition.
"""

import ast
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import hashlib

from .graphiti_service import (
    GraphitiService, GraphEntity, GraphRelationship,
    EntityType, RelationshipType
)

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Result of entity extraction from source content"""
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]
    source_info: Dict[str, Any]
    extraction_time: float
    confidence: float

class CodeAnalyzer:
    """Analyzes code files to extract entities and relationships"""
    
    def __init__(self):
        self.supported_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx'}
    
    def analyze_python_file(self, file_path: Path, content: str) -> ExtractionResult:
        """Analyze Python file for entities and relationships"""
        entities = []
        relationships = []
        start_time = time.time()
        
        try:
            # Parse Python AST
            tree = ast.parse(content)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    entity_id = f"func_{file_path.stem}_{node.name}"
                    
                    # Extract function details
                    attributes = {
                        "file_path": str(file_path),
                        "line_number": node.lineno,
                        "docstring": ast.get_docstring(node),
                        "args": [arg.arg for arg in node.args.args],
                        "returns": self._extract_return_annotation(node)
                    }
                    
                    entity = GraphEntity(
                        entity_id=entity_id,
                        entity_type=EntityType.FUNCTION,
                        name=node.name,
                        attributes=attributes,
                        confidence_score=0.95,  # High confidence for parsed code
                        importance_weight=0.7,
                        tags=["python", "function", file_path.stem]
                    )
                    entities.append(entity)
                    
                    # Extract function calls (relationships)
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                            target_func = child.func.id
                            rel_id = f"call_{entity_id}_{target_func}"
                            
                            relationship = GraphRelationship(
                                relationship_id=rel_id,
                                source_id=entity_id,
                                target_id=f"func_unknown_{target_func}",  # May not exist yet
                                relationship_type=RelationshipType.CALLS,
                                confidence=0.8,
                                attributes={"call_line": child.lineno}
                            )
                            relationships.append(relationship)
                
                elif isinstance(node, ast.ClassDef):
                    entity_id = f"class_{file_path.stem}_{node.name}"
                    
                    attributes = {
                        "file_path": str(file_path),
                        "line_number": node.lineno,
                        "docstring": ast.get_docstring(node),
                        "bases": [self._extract_base_name(base) for base in node.bases],
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    }
                    
                    entity = GraphEntity(
                        entity_id=entity_id,
                        entity_type=EntityType.CLASS,
                        name=node.name,
                        attributes=attributes,
                        confidence_score=0.95,
                        importance_weight=0.8,
                        tags=["python", "class", file_path.stem]
                    )
                    entities.append(entity)
            
            # Extract imports (dependencies)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_id = f"module_{alias.name}"
                        
                        # Create relationship from file to imported module
                        rel_id = f"import_{file_path.stem}_{alias.name}"
                        relationship = GraphRelationship(
                            relationship_id=rel_id,
                            source_id=f"module_{file_path.stem}",
                            target_id=module_id,
                            relationship_type=RelationshipType.DEPENDS_ON,
                            confidence=0.9
                        )
                        relationships.append(relationship)
        
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")
        
        extraction_time = time.time() - start_time
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_info={"file_path": str(file_path), "language": "python"},
            extraction_time=extraction_time,
            confidence=0.9
        )
    
    def _extract_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation from function"""
        if node.returns:
            try:
                return ast.unparse(node.returns)
            except:
                return str(node.returns)
        return None
    
    def _extract_base_name(self, base: ast.expr) -> str:
        """Extract base class name"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return ast.unparse(base)
        else:
            return str(base)
    
    def analyze_typescript_file(self, file_path: Path, content: str) -> ExtractionResult:
        """Analyze TypeScript/JavaScript file (simplified pattern-based extraction)"""
        entities = []
        relationships = []
        start_time = time.time()
        
        try:
            # Function pattern
            func_pattern = r'(?:function\s+|const\s+|let\s+|var\s+)(\w+)\s*(?:=\s*)?(?:async\s+)?(?:\([^)]*\)|function\s*\([^)]*\))'
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                func_name = match.group(1)
                entity_id = f"func_{file_path.stem}_{func_name}"
                
                entity = GraphEntity(
                    entity_id=entity_id,
                    entity_type=EntityType.FUNCTION,
                    name=func_name,
                    attributes={
                        "file_path": str(file_path),
                        "language": "typescript" if file_path.suffix in {'.ts', '.tsx'} else "javascript"
                    },
                    confidence_score=0.8,  # Lower confidence for regex parsing
                    importance_weight=0.6,
                    tags=["javascript", "function", file_path.stem]
                )
                entities.append(entity)
            
            # Class pattern
            class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
            for match in re.finditer(class_pattern, content):
                class_name = match.group(1)
                entity_id = f"class_{file_path.stem}_{class_name}"
                
                attributes = {"file_path": str(file_path)}
                if match.group(2):  # Has base class
                    attributes["extends"] = match.group(2)
                
                entity = GraphEntity(
                    entity_id=entity_id,
                    entity_type=EntityType.CLASS,
                    name=class_name,
                    attributes=attributes,
                    confidence_score=0.85,
                    importance_weight=0.7,
                    tags=["javascript", "class", file_path.stem]
                )
                entities.append(entity)
            
            # Import patterns
            import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
            for match in re.finditer(import_pattern, content):
                imported_module = match.group(1)
                rel_id = f"import_{file_path.stem}_{hashlib.md5(imported_module.encode()).hexdigest()[:8]}"
                
                relationship = GraphRelationship(
                    relationship_id=rel_id,
                    source_id=f"module_{file_path.stem}",
                    target_id=f"module_{imported_module}",
                    relationship_type=RelationshipType.DEPENDS_ON,
                    confidence=0.9
                )
                relationships.append(relationship)
        
        except Exception as e:
            logger.error(f"Error analyzing TypeScript file {file_path}: {e}")
        
        extraction_time = time.time() - start_time
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_info={"file_path": str(file_path), "language": "typescript/javascript"},
            extraction_time=extraction_time,
            confidence=0.8
        )

class DocumentAnalyzer:
    """Analyzes documentation and text files for concepts and requirements"""
    
    def analyze_markdown_file(self, file_path: Path, content: str) -> ExtractionResult:
        """Extract concepts and requirements from Markdown documentation"""
        entities = []
        relationships = []
        start_time = time.time()
        
        try:
            # Extract headers as concepts
            header_pattern = r'^#+\s+(.+)$'
            for match in re.finditer(header_pattern, content, re.MULTILINE):
                concept_name = match.group(1).strip()
                entity_id = f"concept_{hashlib.md5(concept_name.encode()).hexdigest()[:12]}"
                
                entity = GraphEntity(
                    entity_id=entity_id,
                    entity_type=EntityType.CONCEPT,
                    name=concept_name,
                    attributes={
                        "file_path": str(file_path),
                        "document_type": "markdown",
                        "context": self._extract_context(content, match.start(), 200)
                    },
                    confidence_score=0.7,
                    importance_weight=0.6,
                    tags=["documentation", "concept", file_path.stem]
                )
                entities.append(entity)
            
            # Extract requirements (sentences with "must", "should", "shall")
            requirement_pattern = r'([^.]*(?:must|should|shall|required?)[^.]*\.)'
            for match in re.finditer(requirement_pattern, content, re.IGNORECASE):
                requirement_text = match.group(1).strip()
                if len(requirement_text) > 20:  # Filter out very short matches
                    entity_id = f"req_{hashlib.md5(requirement_text.encode()).hexdigest()[:12]}"
                    
                    entity = GraphEntity(
                        entity_id=entity_id,
                        entity_type=EntityType.REQUIREMENT,
                        name=requirement_text[:50] + "..." if len(requirement_text) > 50 else requirement_text,
                        attributes={
                            "file_path": str(file_path),
                            "full_text": requirement_text,
                            "document_type": "markdown"
                        },
                        confidence_score=0.8,
                        importance_weight=0.8,
                        tags=["documentation", "requirement", file_path.stem]
                    )
                    entities.append(entity)
        
        except Exception as e:
            logger.error(f"Error analyzing Markdown file {file_path}: {e}")
        
        extraction_time = time.time() - start_time
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_info={"file_path": str(file_path), "document_type": "markdown"},
            extraction_time=extraction_time,
            confidence=0.7
        )
    
    def _extract_context(self, content: str, position: int, length: int) -> str:
        """Extract context around a position in text"""
        start = max(0, position - length // 2)
        end = min(len(content), position + length // 2)
        return content[start:end].strip()

class EntityExtractor:
    """
    Main entity extractor that coordinates code and document analysis
    
    Automatically discovers entities and relationships from various source types
    and feeds them into the Graphiti temporal knowledge graph.
    """
    
    def __init__(self, graphiti_service: GraphitiService):
        """
        Initialize entity extractor
        
        Args:
            graphiti_service: GraphitiService instance for storing results
        """
        self.graphiti_service = graphiti_service
        self.code_analyzer = CodeAnalyzer()
        self.doc_analyzer = DocumentAnalyzer()
        
        # Track extraction statistics
        self.extraction_stats = {
            "files_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "total_time": 0.0
        }
    
    async def extract_from_file(self, file_path: Path) -> ExtractionResult:
        """
        Extract entities from a single file
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            ExtractionResult with extracted entities and relationships
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return ExtractionResult([], [], {}, 0.0, 0.0)
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding='latin-1')
            except Exception as e:
                logger.error(f"Cannot read file {file_path}: {e}")
                return ExtractionResult([], [], {}, 0.0, 0.0)
        
        # Route to appropriate analyzer
        if file_path.suffix == '.py':
            result = self.code_analyzer.analyze_python_file(file_path, content)
        elif file_path.suffix in {'.ts', '.tsx', '.js', '.jsx'}:
            result = self.code_analyzer.analyze_typescript_file(file_path, content)
        elif file_path.suffix in {'.md', '.txt'}:
            result = self.doc_analyzer.analyze_markdown_file(file_path, content)
        else:
            logger.debug(f"Unsupported file type: {file_path}")
            return ExtractionResult([], [], {}, 0.0, 0.0)
        
        # Store results in Graphiti
        await self._store_extraction_result(result)
        
        # Update statistics
        self.extraction_stats["files_processed"] += 1
        self.extraction_stats["entities_extracted"] += len(result.entities)
        self.extraction_stats["relationships_extracted"] += len(result.relationships)
        self.extraction_stats["total_time"] += result.extraction_time
        
        logger.info(f"Extracted {len(result.entities)} entities and {len(result.relationships)} "
                   f"relationships from {file_path}")
        
        return result
    
    async def extract_from_directory(self, directory: Path, 
                                   recursive: bool = True,
                                   file_patterns: Optional[List[str]] = None) -> List[ExtractionResult]:
        """
        Extract entities from all files in a directory
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            file_patterns: Glob patterns for files to include
            
        Returns:
            List of ExtractionResult objects
        """
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory not found: {directory}")
            return []
        
        results = []
        
        # Default patterns for supported file types
        if not file_patterns:
            file_patterns = ['*.py', '*.ts', '*.tsx', '*.js', '*.jsx', '*.md', '*.txt']
        
        # Collect files to process
        files_to_process = []
        for pattern in file_patterns:
            if recursive:
                files_to_process.extend(directory.rglob(pattern))
            else:
                files_to_process.extend(directory.glob(pattern))
        
        # Remove duplicates and sort
        files_to_process = sorted(set(files_to_process))
        
        logger.info(f"Processing {len(files_to_process)} files from {directory}")
        
        # Process files
        for file_path in files_to_process:
            try:
                result = await self.extract_from_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return results
    
    async def _store_extraction_result(self, result: ExtractionResult):
        """Store extraction result in Graphiti service"""
        try:
            # Store entities
            for entity in result.entities:
                await self.graphiti_service.add_entity(entity)
            
            # Store relationships
            for relationship in result.relationships:
                await self.graphiti_service.add_relationship(relationship)
                
        except Exception as e:
            logger.error(f"Failed to store extraction result: {e}")
    
    async def extract_from_agent_interaction(self, agent_name: str, 
                                           task_description: str,
                                           input_data: Dict[str, Any],
                                           output_data: Dict[str, Any]) -> ExtractionResult:
        """
        Extract knowledge from agent interactions
        
        Args:
            agent_name: Name of the agent
            task_description: Description of the task performed
            input_data: Input data provided to agent
            output_data: Output data produced by agent
            
        Returns:
            ExtractionResult with extracted interaction knowledge
        """
        entities = []
        relationships = []
        start_time = time.time()
        
        try:
            # Create agent entity
            agent_id = f"agent_{agent_name}"
            agent_entity = GraphEntity(
                entity_id=agent_id,
                entity_type=EntityType.AGENT,
                name=agent_name,
                attributes={
                    "last_task": task_description,
                    "interaction_count": 1  # Will be updated if entity exists
                },
                confidence_score=1.0,
                importance_weight=0.8,
                tags=["agent", "interaction"]
            )
            entities.append(agent_entity)
            
            # Create task/pattern entity
            task_id = f"pattern_{hashlib.md5(task_description.encode()).hexdigest()[:12]}"
            task_entity = GraphEntity(
                entity_id=task_id,
                entity_type=EntityType.PATTERN,
                name=task_description[:50] + "..." if len(task_description) > 50 else task_description,
                attributes={
                    "full_description": task_description,
                    "input_keys": list(input_data.keys()) if input_data else [],
                    "output_keys": list(output_data.keys()) if output_data else [],
                    "agent": agent_name
                },
                confidence_score=0.9,
                importance_weight=0.7,
                tags=["pattern", "task", agent_name]
            )
            entities.append(task_entity)
            
            # Create relationship
            rel_id = f"performs_{agent_id}_{task_id}"
            relationship = GraphRelationship(
                relationship_id=rel_id,
                source_id=agent_id,
                target_id=task_id,
                relationship_type=RelationshipType.IMPLEMENTS,
                confidence=0.95,
                temporal_data={
                    "task_timestamp": time.time(),
                    "success": bool(output_data.get("success", True))
                }
            )
            relationships.append(relationship)
            
        except Exception as e:
            logger.error(f"Error extracting from agent interaction: {e}")
        
        extraction_time = time.time() - start_time
        
        result = ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_info={"agent": agent_name, "task": task_description},
            extraction_time=extraction_time,
            confidence=0.9
        )
        
        # Store in Graphiti
        await self._store_extraction_result(result)
        
        return result
    
    def extract_from_content(self, content: str, language: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities from content string (SCWT compatible format)
        
        Args:
            content: Source content to analyze
            language: Programming language or content type ("python", "typescript", "markdown")
            metadata: Additional metadata about the content
            
        Returns:
            Dict with 'entities' key containing list of extracted entities
        """
        start_time = time.time()
        entities = []
        relationships = []
        
        try:
            if language.lower() == "python":
                # Use the same Python analysis logic from analyze_python_file
                try:
                    # Parse Python AST
                    tree = ast.parse(content)
                    
                    # Extract functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            entity_id = f"func_test_{node.name}"
                            
                            # Extract function details
                            attributes = {
                                "source": metadata.get("source", "inline"),
                                "line_number": node.lineno,
                                "docstring": ast.get_docstring(node),
                                "args": [arg.arg for arg in node.args.args],
                                "returns": self.code_analyzer._extract_return_annotation(node)
                            }
                            
                            entity = GraphEntity(
                                entity_id=entity_id,
                                entity_type=EntityType.FUNCTION,
                                name=node.name,
                                attributes=attributes,
                                confidence_score=0.95,  # High confidence for parsed code
                                importance_weight=0.7,
                                tags=["python", "function", "test"]
                            )
                            entities.append(entity)
                            
                        elif isinstance(node, ast.ClassDef):
                            entity_id = f"class_test_{node.name}"
                            
                            attributes = {
                                "source": metadata.get("source", "inline"),
                                "line_number": node.lineno,
                                "docstring": ast.get_docstring(node),
                                "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                            }
                            
                            entity = GraphEntity(
                                entity_id=entity_id,
                                entity_type=EntityType.CLASS,
                                name=node.name,
                                attributes=attributes,
                                confidence_score=0.95,
                                importance_weight=0.8,
                                tags=["python", "class", "test"]
                            )
                            entities.append(entity)
                            
                except SyntaxError as e:
                    logger.warning(f"Failed to parse Python content: {e}")
                    
            elif language.lower() in ["markdown", "md"]:
                # Extract concepts from markdown headers
                header_pattern = r'^#+\s+(.+)$'
                for match in re.finditer(header_pattern, content, re.MULTILINE):
                    concept_name = match.group(1).strip()
                    entity_id = f"concept_{hashlib.md5(concept_name.encode()).hexdigest()[:12]}"
                    
                    entity = GraphEntity(
                        entity_id=entity_id,
                        entity_type=EntityType.CONCEPT,
                        name=concept_name,
                        attributes={
                            "source": metadata.get("source", "inline"),
                            "document_type": "markdown",
                            "context": content[max(0, match.start()-100):match.end()+100]
                        },
                        confidence_score=0.7,
                        importance_weight=0.6,
                        tags=["documentation", "concept", "test"]
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.error(f"Error extracting from content: {e}")
        
        extraction_time = time.time() - start_time
        
        # Return SCWT-compatible format with 'entities' key containing actual GraphEntity objects
        return {
            'entities': entities,  # Keep as actual GraphEntity objects
            'relationships': relationships,  # Keep as actual GraphRelationship objects
            'extraction_metadata': {
                'language': language,
                'metadata': metadata,
                'extraction_time': extraction_time,
                'confidence': 0.8,
                'entities_count': len(entities),
                'relationships_count': len(relationships)
            }
        }
    
    def extract_from_content_full(self, content: str, language: str, metadata: Dict[str, Any]) -> ExtractionResult:
        """
        Extract entities from content string (returns full ExtractionResult for internal use)
        
        Args:
            content: Source content to analyze
            language: Programming language or content type ("python", "typescript", "markdown")
            metadata: Additional metadata about the content
            
        Returns:
            ExtractionResult with extracted entities and relationships
        """
        # Get the SCWT-compatible result
        scwt_result = self.extract_from_content(content, language, metadata)
        
        # Convert back to ExtractionResult format for internal use
        entities = []
        for entity_data in scwt_result['entities']:
            entity = GraphEntity(
                entity_id=entity_data['entity_id'],
                entity_type=EntityType(entity_data['entity_type']),
                name=entity_data['name'],
                attributes=entity_data['attributes'],
                confidence_score=entity_data['confidence_score'],
                importance_weight=entity_data['importance_weight'],
                tags=entity_data['tags']
            )
            entities.append(entity)
        
        relationships = []
        for rel_data in scwt_result['relationships']:
            relationship = GraphRelationship(
                relationship_id=rel_data['relationship_id'],
                source_id=rel_data['source_id'],
                target_id=rel_data['target_id'],
                relationship_type=RelationshipType(rel_data['relationship_type']),
                confidence=rel_data['confidence']
            )
            relationships.append(relationship)
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            source_info=scwt_result['extraction_metadata'],
            extraction_time=scwt_result['extraction_metadata']['extraction_time'],
            confidence=scwt_result['extraction_metadata']['confidence']
        )

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        stats = self.extraction_stats.copy()
        if stats["files_processed"] > 0:
            stats["avg_entities_per_file"] = stats["entities_extracted"] / stats["files_processed"]
            stats["avg_time_per_file"] = stats["total_time"] / stats["files_processed"]
        else:
            stats["avg_entities_per_file"] = 0
            stats["avg_time_per_file"] = 0
        
        return stats

# Factory function
def create_entity_extractor(graphiti_service: GraphitiService) -> EntityExtractor:
    """Create a configured entity extractor instance"""
    return EntityExtractor(graphiti_service)