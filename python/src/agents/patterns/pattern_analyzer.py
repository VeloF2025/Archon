"""
Pattern Analyzer for Project Architecture Recognition

Analyzes crawled projects and repositories to automatically detect
architectural patterns and extract reusable templates.
"""

import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import yaml

from .pattern_models import (
    Pattern, PatternMetadata, PatternType, PatternComplexity, PatternCategory,
    PatternTechnology, PatternComponent, PatternDetectionSource,
    PatternProvider, PatternWorkflow
)
import logging

logger = logging.getLogger(__name__)


class ProjectStructureAnalyzer:
    """Analyzes project structure to identify architectural patterns."""
    
    # Technology detection patterns
    TECHNOLOGY_PATTERNS = {
        'react': {
            'files': ['package.json'],
            'content': [r'"react":', r'import.*from ["\']react["\']'],
            'confidence': 0.9
        },
        'nextjs': {
            'files': ['next.config.js', 'next.config.ts', 'package.json'],
            'content': [r'"next":', r'import.*from ["\']next'],
            'confidence': 0.95
        },
        'vue': {
            'files': ['package.json', 'vue.config.js'],
            'content': [r'"vue":', r'<template>', r'<script.*vue'],
            'confidence': 0.9
        },
        'angular': {
            'files': ['angular.json', 'package.json'],
            'content': [r'"@angular/', r'import.*@angular'],
            'confidence': 0.95
        },
        'svelte': {
            'files': ['svelte.config.js', 'package.json'],
            'content': [r'"svelte":', r'<script.*svelte'],
            'confidence': 0.9
        },
        'fastapi': {
            'files': ['main.py', 'app.py', 'requirements.txt', 'pyproject.toml'],
            'content': [r'from fastapi import', r'FastAPI\(\)', r'"fastapi"'],
            'confidence': 0.95
        },
        'django': {
            'files': ['manage.py', 'settings.py', 'requirements.txt'],
            'content': [r'from django import', r'DJANGO_SETTINGS_MODULE'],
            'confidence': 0.95
        },
        'flask': {
            'files': ['app.py', 'requirements.txt'],
            'content': [r'from flask import', r'Flask\(__name__\)'],
            'confidence': 0.9
        },
        'express': {
            'files': ['package.json', 'server.js', 'app.js'],
            'content': [r'"express":', r'require\(["\']express["\']'],
            'confidence': 0.9
        },
        'nestjs': {
            'files': ['nest-cli.json', 'package.json'],
            'content': [r'"@nestjs/', r'import.*@nestjs'],
            'confidence': 0.95
        },
        'postgresql': {
            'files': ['docker-compose.yml', 'requirements.txt', 'package.json'],
            'content': [r'postgres:', r'postgresql://', r'"pg":', r'psycopg2'],
            'confidence': 0.8
        },
        'mongodb': {
            'files': ['docker-compose.yml', 'package.json'],
            'content': [r'mongo:', r'mongodb://', r'"mongoose":', r'pymongo'],
            'confidence': 0.8
        },
        'redis': {
            'files': ['docker-compose.yml', 'requirements.txt'],
            'content': [r'redis:', r'"redis":', r'import redis'],
            'confidence': 0.8
        },
        'docker': {
            'files': ['Dockerfile', 'docker-compose.yml', '.dockerignore'],
            'content': [r'FROM ', r'version:', r'services:'],
            'confidence': 0.95
        },
        'kubernetes': {
            'files': ['deployment.yaml', 'service.yaml', 'ingress.yaml'],
            'content': [r'apiVersion:', r'kind: Deployment', r'kind: Service'],
            'confidence': 0.95
        },
        'terraform': {
            'files': ['main.tf', 'variables.tf', 'outputs.tf'],
            'content': [r'resource "', r'provider "', r'variable "'],
            'confidence': 0.95
        }
    }
    
    # Pattern detection rules
    PATTERN_DETECTION_RULES = {
        PatternType.MICROSERVICES: {
            'indicators': [
                'multiple_services', 'docker_compose_services', 'api_gateway',
                'service_discovery', 'separate_databases'
            ],
            'min_confidence': 0.7
        },
        PatternType.MONOLITHIC: {
            'indicators': [
                'single_application', 'single_database', 'no_service_separation'
            ],
            'min_confidence': 0.6
        },
        PatternType.SERVERLESS: {
            'indicators': [
                'lambda_functions', 'vercel_functions', 'netlify_functions',
                'firebase_functions', 'cloudflare_workers'
            ],
            'min_confidence': 0.8
        },
        PatternType.JAMSTACK: {
            'indicators': [
                'static_site_generator', 'api_endpoints', 'markdown_content',
                'headless_cms'
            ],
            'min_confidence': 0.7
        },
        PatternType.MVC: {
            'indicators': [
                'models_directory', 'views_directory', 'controllers_directory',
                'mvc_framework'
            ],
            'min_confidence': 0.6
        }
    }
    
    def __init__(self):
        self.detected_technologies: Dict[str, float] = {}
        self.project_structure: Dict[str, Any] = {}
        self.pattern_indicators: Dict[str, float] = defaultdict(float)
    
    async def analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project structure and detect technologies."""
        try:
            logger.info(f"Analyzing project structure | path={project_path}")
            
            # Reset analysis state
            self.detected_technologies = {}
            self.project_structure = {}
            self.pattern_indicators = defaultdict(float)
            
            # Get all files in project
            all_files = self._get_all_files(project_path)
            
            # Detect technologies
            await self._detect_technologies(project_path, all_files)
            
            # Analyze directory structure
            self._analyze_directory_structure(project_path)
            
            # Detect architectural patterns
            detected_patterns = self._detect_patterns()
            
            # Build analysis result
            analysis_result = {
                'project_path': str(project_path),
                'detected_technologies': self.detected_technologies,
                'directory_structure': self.project_structure,
                'detected_patterns': detected_patterns,
                'pattern_indicators': dict(self.pattern_indicators),
                'file_count': len(all_files),
                'confidence_score': self._calculate_overall_confidence()
            }
            
            logger.info(f"Project analysis complete | technologies={len(self.detected_technologies)} | patterns={len(detected_patterns)}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Project analysis failed | path={project_path} | error={str(e)}")
            raise
    
    def _get_all_files(self, project_path: Path) -> List[Path]:
        """Get all files in project, excluding common ignore patterns."""
        ignore_patterns = {
            'node_modules', '.git', '__pycache__', '.venv', 'venv',
            '.next', 'dist', 'build', 'target', '.cache', '.temp'
        }
        
        all_files = []
        
        def should_ignore(path: Path) -> bool:
            return any(part in ignore_patterns for part in path.parts)
        
        try:
            for file_path in project_path.rglob('*'):
                if file_path.is_file() and not should_ignore(file_path):
                    all_files.append(file_path)
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not access some files in {project_path}: {e}")
        
        return all_files
    
    async def _detect_technologies(self, project_path: Path, all_files: List[Path]) -> None:
        """Detect technologies used in the project."""
        file_names = {f.name for f in all_files}
        
        for tech_name, patterns in self.TECHNOLOGY_PATTERNS.items():
            confidence = 0.0
            
            # Check for specific files
            file_matches = sum(1 for file in patterns['files'] if file in file_names)
            if file_matches > 0:
                confidence += (file_matches / len(patterns['files'])) * 0.5
            
            # Check file content for patterns
            content_matches = await self._search_content_patterns(
                all_files, patterns['content']
            )
            if content_matches > 0:
                confidence += min(content_matches * 0.2, 0.5)
            
            # Apply base confidence
            if confidence > 0:
                confidence *= patterns['confidence']
                self.detected_technologies[tech_name] = min(confidence, 1.0)
    
    async def _search_content_patterns(self, files: List[Path], patterns: List[str]) -> int:
        """Search for content patterns in files."""
        matches = 0
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # Limit search to reasonable file types and sizes
        searchable_extensions = {
            '.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.go', '.rb',
            '.php', '.cs', '.cpp', '.c', '.rs', '.kt', '.swift',
            '.json', '.yaml', '.yml', '.toml', '.xml', '.html', '.css',
            '.md', '.txt', '.cfg', '.ini', '.conf'
        }
        
        for file_path in files:
            if file_path.suffix.lower() not in searchable_extensions:
                continue
            
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                for pattern in compiled_patterns:
                    if pattern.search(content):
                        matches += 1
                        break  # Count each file only once per pattern search
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
        
        return matches
    
    def _analyze_directory_structure(self, project_path: Path) -> None:
        """Analyze directory structure to identify patterns."""
        
        directories = [d for d in project_path.rglob('*') if d.is_dir()]
        dir_names = {d.name.lower() for d in directories}
        
        # Common directory patterns
        structure_indicators = {
            'mvc_structure': ['models', 'views', 'controllers'],
            'clean_architecture': ['entities', 'usecases', 'adapters'],
            'domain_driven': ['domain', 'application', 'infrastructure'],
            'microservices': ['services', 'gateway', 'common'],
            'frontend_structure': ['components', 'pages', 'hooks', 'utils'],
            'backend_structure': ['routes', 'middleware', 'models', 'controllers'],
            'testing_structure': ['tests', 'test', '__tests__', 'spec']
        }
        
        detected_structures = {}
        for structure_name, required_dirs in structure_indicators.items():
            matches = sum(1 for dir_name in required_dirs if dir_name in dir_names)
            if matches > 0:
                confidence = matches / len(required_dirs)
                detected_structures[structure_name] = confidence
                
                # Add to pattern indicators
                if structure_name == 'mvc_structure':
                    self.pattern_indicators['mvc_framework'] += confidence
                elif structure_name == 'microservices':
                    self.pattern_indicators['multiple_services'] += confidence
        
        # Analyze service separation (microservices indicator)
        potential_services = [
            d for d in directories
            if d.name.lower() in {'api', 'backend', 'frontend', 'auth', 'user', 'order', 'payment'}
        ]
        
        if len(potential_services) > 2:
            self.pattern_indicators['multiple_services'] += 0.3
        
        # Check for Docker compose services
        compose_files = [f for f in project_path.glob('*compose*.yml')]
        if compose_files:
            self.pattern_indicators['docker_compose_services'] += 0.4
        
        self.project_structure = {
            'total_directories': len(directories),
            'detected_structures': detected_structures,
            'potential_services': len(potential_services),
            'has_docker_compose': len(compose_files) > 0
        }
    
    def _detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect architectural patterns based on collected indicators."""
        detected_patterns = []
        
        for pattern_type, rules in self.PATTERN_DETECTION_RULES.items():
            confidence = 0.0
            matched_indicators = []
            
            for indicator in rules['indicators']:
                if indicator in self.pattern_indicators:
                    indicator_confidence = self.pattern_indicators[indicator]
                    confidence += indicator_confidence
                    matched_indicators.append({
                        'indicator': indicator,
                        'confidence': indicator_confidence
                    })
            
            # Normalize confidence
            if matched_indicators:
                confidence = min(confidence / len(rules['indicators']), 1.0)
            
            if confidence >= rules['min_confidence']:
                detected_patterns.append({
                    'type': pattern_type.value,
                    'confidence': confidence,
                    'matched_indicators': matched_indicators
                })
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        return detected_patterns
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence in the analysis."""
        if not self.detected_technologies:
            return 0.0
        
        # Base confidence from technology detection
        tech_confidence = sum(self.detected_technologies.values()) / len(self.detected_technologies)
        
        # Pattern detection confidence
        pattern_confidence = sum(self.pattern_indicators.values()) / max(len(self.pattern_indicators), 1)
        
        # Weighted average
        return (tech_confidence * 0.6 + pattern_confidence * 0.4)


class PatternAnalyzer:
    """Main pattern analyzer that coordinates pattern detection and extraction."""
    
    def __init__(self):
        self.structure_analyzer = ProjectStructureAnalyzer()
    
    async def analyze_project_for_patterns(
        self, 
        project_path: str, 
        source_info: Optional[Dict[str, Any]] = None
    ) -> List[Pattern]:
        """
        Analyze a project and extract architectural patterns.
        
        Args:
            project_path: Path to the project to analyze
            source_info: Optional information about the source (URL, repo, etc.)
        
        Returns:
            List of detected patterns
        """
        try:
            logger.info(f"Starting pattern analysis | path={project_path}")
            
            project_path = Path(project_path)
            if not project_path.exists():
                raise ValueError(f"Project path does not exist: {project_path}")
            
            # Analyze project structure
            analysis_result = await self.structure_analyzer.analyze_project_structure(project_path)
            
            # Extract patterns from analysis
            patterns = await self._extract_patterns_from_analysis(analysis_result, source_info)
            
            logger.info(f"Pattern analysis complete | extracted_patterns={len(patterns)}")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern analysis failed | path={project_path} | error={str(e)}")
            raise
    
    async def _extract_patterns_from_analysis(
        self, 
        analysis: Dict[str, Any], 
        source_info: Optional[Dict[str, Any]] = None
    ) -> List[Pattern]:
        """Extract Pattern objects from analysis results."""
        patterns = []
        
        for pattern_data in analysis.get('detected_patterns', []):
            if pattern_data['confidence'] < 0.6:  # Skip low-confidence patterns
                continue
            
            try:
                pattern = await self._create_pattern_from_detection(
                    pattern_data, analysis, source_info
                )
                patterns.append(pattern)
            except Exception as e:
                logger.warning(f"Failed to create pattern from detection: {e}")
        
        return patterns
    
    async def _create_pattern_from_detection(
        self, 
        pattern_data: Dict[str, Any], 
        analysis: Dict[str, Any], 
        source_info: Optional[Dict[str, Any]] = None
    ) -> Pattern:
        """Create a Pattern object from detection data."""
        
        pattern_type = PatternType(pattern_data['type'])
        
        # Determine complexity based on technologies and structure
        complexity = self._determine_complexity(analysis)
        
        # Determine category
        category = self._determine_category(pattern_type, analysis)
        
        # Extract technologies
        technologies = self._extract_technologies(analysis)
        
        # Create detection source
        detection_source = PatternDetectionSource(
            source_url=source_info.get('url') if source_info else None,
            repository=source_info.get('repository') if source_info else None,
            project_name=source_info.get('project_name') if source_info else None,
            detection_confidence=pattern_data['confidence']
        )
        
        # Create metadata
        metadata = PatternMetadata(
            name=self._generate_pattern_name(pattern_type, technologies),
            description=self._generate_pattern_description(pattern_type, technologies),
            version="1.0.0",
            author="Archon Pattern Analyzer",
            type=pattern_type,
            category=category,
            complexity=complexity,
            technologies=technologies,
            tags=self._generate_tags(pattern_type, technologies),
            detection_source=detection_source
        )
        
        # Create components (simplified for detected patterns)
        components = self._extract_components(analysis)
        
        # Create workflow (basic implementation steps)
        workflows = self._generate_basic_workflow(pattern_type, technologies)
        
        return Pattern(
            metadata=metadata,
            components=components,
            workflows=workflows
        )
    
    def _determine_complexity(self, analysis: Dict[str, Any]) -> PatternComplexity:
        """Determine pattern complexity based on analysis."""
        tech_count = len(analysis.get('detected_technologies', {}))
        has_docker = 'docker' in analysis.get('detected_technologies', {})
        has_microservices = any(
            p['type'] == PatternType.MICROSERVICES.value 
            for p in analysis.get('detected_patterns', [])
        )
        
        if has_microservices or tech_count > 8:
            return PatternComplexity.EXPERT
        elif has_docker or tech_count > 5:
            return PatternComplexity.ADVANCED
        elif tech_count > 3:
            return PatternComplexity.INTERMEDIATE
        else:
            return PatternComplexity.BEGINNER
    
    def _determine_category(self, pattern_type: PatternType, analysis: Dict[str, Any]) -> PatternCategory:
        """Determine pattern category."""
        technologies = analysis.get('detected_technologies', {})
        
        # Map pattern types to categories
        type_category_map = {
            PatternType.MICROSERVICES: PatternCategory.ARCHITECTURE,
            PatternType.MONOLITHIC: PatternCategory.ARCHITECTURE,
            PatternType.SERVERLESS: PatternCategory.ARCHITECTURE,
            PatternType.JAMSTACK: PatternCategory.FRONTEND,
            PatternType.MVC: PatternCategory.ARCHITECTURE
        }
        
        if pattern_type in type_category_map:
            return type_category_map[pattern_type]
        
        # Determine based on technologies
        if any(tech in technologies for tech in ['react', 'vue', 'angular', 'svelte']):
            return PatternCategory.FRONTEND
        elif any(tech in technologies for tech in ['fastapi', 'django', 'flask', 'express']):
            return PatternCategory.BACKEND
        elif any(tech in technologies for tech in ['postgresql', 'mongodb', 'redis']):
            return PatternCategory.DATABASE
        elif any(tech in technologies for tech in ['docker', 'kubernetes', 'terraform']):
            return PatternCategory.DEVOPS
        else:
            return PatternCategory.ARCHITECTURE
    
    def _extract_technologies(self, analysis: Dict[str, Any]) -> List[PatternTechnology]:
        """Extract PatternTechnology objects from analysis."""
        technologies = []
        
        for tech_name, confidence in analysis.get('detected_technologies', {}).items():
            if confidence < 0.5:  # Skip low-confidence technologies
                continue
            
            # Categorize technology
            category = self._categorize_technology(tech_name)
            
            technologies.append(PatternTechnology(
                name=tech_name,
                category=category,
                required=confidence > 0.8
            ))
        
        return technologies
    
    def _categorize_technology(self, tech_name: str) -> str:
        """Categorize a technology."""
        categories = {
            'framework': ['react', 'vue', 'angular', 'svelte', 'nextjs', 'fastapi', 'django', 'flask', 'express', 'nestjs'],
            'database': ['postgresql', 'mongodb', 'redis', 'mysql', 'sqlite'],
            'infrastructure': ['docker', 'kubernetes', 'terraform'],
            'language': ['python', 'javascript', 'typescript', 'java', 'go'],
            'tool': ['webpack', 'vite', 'babel', 'eslint', 'prettier']
        }
        
        for category, techs in categories.items():
            if tech_name in techs:
                return category
        
        return 'other'
    
    def _extract_components(self, analysis: Dict[str, Any]) -> List[PatternComponent]:
        """Extract components from analysis."""
        components = []
        technologies = analysis.get('detected_technologies', {})
        
        # Create components based on detected technologies
        if 'react' in technologies or 'vue' in technologies or 'angular' in technologies:
            components.append(PatternComponent(
                name="frontend",
                type="frontend",
                description="Frontend application component",
                technologies=self._extract_technologies(analysis)
            ))
        
        if any(tech in technologies for tech in ['fastapi', 'django', 'flask', 'express']):
            components.append(PatternComponent(
                name="backend",
                type="backend", 
                description="Backend API service",
                technologies=self._extract_technologies(analysis)
            ))
        
        if any(tech in technologies for tech in ['postgresql', 'mongodb', 'redis']):
            components.append(PatternComponent(
                name="database",
                type="database",
                description="Data storage component",
                technologies=self._extract_technologies(analysis)
            ))
        
        return components
    
    def _generate_pattern_name(self, pattern_type: PatternType, technologies: List[PatternTechnology]) -> str:
        """Generate a descriptive name for the pattern."""
        tech_names = [tech.name.title() for tech in technologies[:3]]  # Top 3 techs
        tech_str = " + ".join(tech_names) if tech_names else "Multi-Tech"
        
        pattern_name_map = {
            PatternType.MICROSERVICES: "Microservices",
            PatternType.MONOLITHIC: "Monolithic",
            PatternType.SERVERLESS: "Serverless",
            PatternType.JAMSTACK: "JAMstack",
            PatternType.MVC: "MVC"
        }
        
        pattern_name = pattern_name_map.get(pattern_type, pattern_type.value.title())
        return f"{tech_str} {pattern_name} Pattern"
    
    def _generate_pattern_description(self, pattern_type: PatternType, technologies: List[PatternTechnology]) -> str:
        """Generate a description for the pattern."""
        tech_names = [tech.name for tech in technologies]
        tech_str = ", ".join(tech_names[:5])  # Top 5 techs
        
        descriptions = {
            PatternType.MICROSERVICES: f"Microservices architecture pattern using {tech_str}. Features independently deployable services with clear separation of concerns.",
            PatternType.MONOLITHIC: f"Monolithic application pattern using {tech_str}. Single deployable unit with integrated components.",
            PatternType.SERVERLESS: f"Serverless architecture pattern using {tech_str}. Event-driven, stateless functions with automatic scaling.",
            PatternType.JAMSTACK: f"JAMstack pattern using {tech_str}. Pre-built markup with dynamic functionality via JavaScript and APIs.",
            PatternType.MVC: f"Model-View-Controller pattern using {tech_str}. Clear separation between data, presentation, and business logic."
        }
        
        return descriptions.get(pattern_type, f"Architectural pattern using {tech_str}")
    
    def _generate_tags(self, pattern_type: PatternType, technologies: List[PatternTechnology]) -> List[str]:
        """Generate tags for the pattern."""
        tags = [pattern_type.value]
        tags.extend(tech.name for tech in technologies)
        
        # Add category tags
        if any(tech.name in ['react', 'vue', 'angular'] for tech in technologies):
            tags.append('frontend')
        if any(tech.name in ['fastapi', 'django', 'flask'] for tech in technologies):
            tags.append('backend')
        if any(tech.name in ['docker', 'kubernetes'] for tech in technologies):
            tags.append('containerized')
        
        return list(set(tags))  # Remove duplicates
    
    def _generate_basic_workflow(self, pattern_type: PatternType, technologies: List[PatternTechnology]) -> List[PatternWorkflow]:
        """Generate a basic workflow for the pattern."""
        workflows = [
            PatternWorkflow(
                step=1,
                name="Setup Project Structure",
                description="Create the basic project structure and configuration files",
                commands=["mkdir -p project", "cd project"]
            )
        ]
        
        # Add technology-specific steps
        tech_names = [tech.name for tech in technologies]
        
        if 'docker' in tech_names:
            workflows.append(PatternWorkflow(
                step=2,
                name="Setup Docker Configuration",
                description="Create Dockerfile and docker-compose.yml",
                commands=["touch Dockerfile", "touch docker-compose.yml"]
            ))
        
        if any(tech in tech_names for tech in ['react', 'vue', 'angular']):
            workflows.append(PatternWorkflow(
                step=3,
                name="Setup Frontend",
                description="Initialize and configure frontend application",
                commands=["npm install", "npm run build"]
            ))
        
        if any(tech in tech_names for tech in ['fastapi', 'django', 'flask']):
            workflows.append(PatternWorkflow(
                step=4,
                name="Setup Backend",
                description="Initialize and configure backend services",
                commands=["pip install -r requirements.txt", "python manage.py migrate"]
            ))
        
        return workflows