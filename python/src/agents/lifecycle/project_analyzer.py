"""
Project Analyzer v3.0 - Project Analysis Implementation
Based on Agent_Lifecycle_Management_PRP.md specifications

NLNH Protocol: Real project analysis with actual tech stack detection
DGTS Enforcement: No hardcoded tech stacks, actual file analysis
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re

from .pool_manager import AgentSpec

logger = logging.getLogger(__name__)


@dataclass
class ProjectAnalysis:
    """Project analysis result containing detected technologies and patterns"""
    project_id: str
    tech_stack: List[str] = field(default_factory=list)
    architecture_patterns: List[str] = field(default_factory=list)
    domain_requirements: List[str] = field(default_factory=list)
    complexity_indicators: Dict[str, Any] = field(default_factory=dict)
    file_count: int = 0
    languages_detected: List[str] = field(default_factory=list)
    frameworks_detected: List[str] = field(default_factory=list)


class ProjectAnalyzer:
    """
    Project Analyzer for Archon v3.0
    Implementation for test_project_technology_detection() and test_required_agents_determination()
    """
    
    def __init__(self):
        # Technology detection patterns
        self.tech_patterns = {
            # Frontend Technologies
            "react": ["package.json::react", "*.jsx", "*.tsx", "React."],
            "vue": ["package.json::vue", "*.vue", "Vue."],
            "angular": ["package.json::@angular", "angular.json", "*.component.ts"],
            "svelte": ["package.json::svelte", "*.svelte"],
            "typescript": ["package.json::typescript", "*.ts", "*.tsx", "tsconfig.json"],
            "javascript": ["*.js", "*.jsx", "package.json"],
            
            # Backend Technologies
            "fastapi": ["requirements.txt::fastapi", "from fastapi import", "FastAPI("],
            "django": ["requirements.txt::django", "from django import", "settings.py"],
            "flask": ["requirements.txt::flask", "from flask import", "Flask("],
            "express": ["package.json::express", "require('express')", "app.listen"],
            "pydantic": ["requirements.txt::pydantic", "from pydantic import", "BaseModel"],
            "python": ["*.py", "requirements.txt", "setup.py", "__init__.py"],
            "node": ["package.json", "node_modules/", "*.js"],
            
            # Databases
            "postgresql": ["docker-compose.yml::postgres", "requirements.txt::psycopg2", "DATABASE_URL"],
            "mysql": ["docker-compose.yml::mysql", "requirements.txt::pymysql", "MYSQL_"],
            "mongodb": ["docker-compose.yml::mongo", "requirements.txt::pymongo", "MongoDB"],
            "redis": ["docker-compose.yml::redis", "requirements.txt::redis", "REDIS_URL"],
            "sqlite": ["*.db", "*.sqlite", "requirements.txt::sqlite3"],
            
            # Infrastructure
            "docker": ["Dockerfile", "docker-compose.yml", ".dockerignore"],
            "kubernetes": ["*.yaml::apiVersion", "*.yml::kind:", "k8s/", "kustomization.yaml"],
            "nginx": ["nginx.conf", "docker-compose.yml::nginx"],
            
            # Cloud & DevOps
            "aws": ["*.yml::aws-", "requirements.txt::boto3", "AWS_"],
            "gcp": ["*.json::gcp", "requirements.txt::google-cloud"],
            "azure": ["*.yml::azure", "requirements.txt::azure"],
            
            # Domain-Specific
            "healthcare": ["HIPAA", "patient", "medical", "healthcare", "PHI"],
            "finance": ["payment", "transaction", "banking", "financial", "PCI"],
            "ecommerce": ["cart", "checkout", "product", "order", "payment"],
            "auth": ["authentication", "authorization", "jwt", "oauth", "login"],
            
            # Security & Compliance
            "hipaa-compliance": ["HIPAA", "patient.*data", "PHI", "healthcare.*compliant"],
            "pci-compliance": ["PCI", "payment.*card", "credit.*card", "financial.*data"],
            "gdpr-compliance": ["GDPR", "data.*protection", "privacy.*policy", "consent"],
            
            # Testing
            "pytest": ["requirements.txt::pytest", "test_*.py", "tests/"],
            "jest": ["package.json::jest", "*.test.js", "__tests__/"],
            "cypress": ["package.json::cypress", "cypress/"],
            
            # AI/ML
            "openai": ["requirements.txt::openai", "OPENAI_API_KEY", "from openai import"],
            "anthropic": ["requirements.txt::anthropic", "ANTHROPIC_API_KEY", "from anthropic import"],
            "langchain": ["requirements.txt::langchain", "from langchain import"],
            "tensorflow": ["requirements.txt::tensorflow", "from tensorflow import"],
            "pytorch": ["requirements.txt::torch", "import torch"]
        }
        
        # Architecture pattern detection
        self.architecture_patterns = {
            "microservices": ["docker-compose.yml", "services:", "multiple.*service", "api.*gateway"],
            "monolithic": ["single.*application", "all.*in.*one", "traditional.*app"],
            "serverless": ["lambda", "functions", "serverless.yml", "cloud.*function"],
            "event-driven": ["event", "queue", "pub.*sub", "message.*broker"],
            "rest-api": ["REST", "api", "endpoint", "/api/", "restful"],
            "graphql": ["graphql", "apollo", "schema", "resolver"],
            "spa": ["single.*page", "spa", "client.*side.*rendering"],
            "ssr": ["server.*side.*rendering", "next.js", "nuxt", "gatsby"],
            "frontend-backend": ["frontend", "backend", "client.*server", "api.*client"],
            "healthcare-compliant": ["HIPAA", "healthcare", "medical.*system", "patient.*management"],
            "patient-data-handling": ["patient.*data", "medical.*record", "health.*information"]
        }

    async def analyze_project(self, project_id: str, project_files: Optional[Dict[str, Any]] = None) -> ProjectAnalysis:
        """
        Analyze project to identify technology stack and architecture patterns
        Implementation for test_project_technology_detection()
        """
        analysis = ProjectAnalysis(project_id=project_id)
        
        if not project_files:
            logger.warning(f"No project files provided for analysis of project {project_id}")
            return analysis
        
        analysis.file_count = len(project_files)
        
        # Analyze files for technology patterns
        for file_path, content in project_files.items():
            await self._analyze_file(file_path, content, analysis)
        
        # Detect architecture patterns
        await self._detect_architecture_patterns(project_files, analysis)
        
        # Detect domain requirements
        await self._detect_domain_requirements(analysis)
        
        # Calculate complexity indicators
        await self._calculate_complexity_indicators(analysis)
        
        logger.info(f"Project {project_id} analysis complete: "
                   f"{len(analysis.tech_stack)} technologies, "
                   f"{len(analysis.architecture_patterns)} patterns detected")
        
        return analysis

    async def _analyze_file(self, file_path: str, content: Any, analysis: ProjectAnalysis) -> None:
        """Analyze individual file for technology indicators"""
        file_lower = file_path.lower()
        content_str = str(content) if content else ""
        content_lower = content_str.lower()
        
        # Check each technology pattern
        for tech, patterns in self.tech_patterns.items():
            if tech in analysis.tech_stack:
                continue  # Already detected
                
            for pattern in patterns:
                if "::" in pattern:
                    # File-specific content pattern (e.g., "package.json::react")
                    file_pattern, content_pattern = pattern.split("::", 1)
                    if file_pattern in file_lower and content_pattern.lower() in content_lower:
                        analysis.tech_stack.append(tech)
                        break
                elif pattern.startswith("*."):
                    # File extension pattern
                    ext = pattern[1:]  # Remove *
                    if file_path.endswith(ext):
                        analysis.tech_stack.append(tech)
                        break
                elif pattern in file_lower or pattern.lower() in content_lower:
                    # General pattern matching
                    analysis.tech_stack.append(tech)
                    break
        
        # Detect programming languages
        language_extensions = {
            ".py": "python", ".js": "javascript", ".ts": "typescript", 
            ".jsx": "react", ".tsx": "react", ".vue": "vue",
            ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp"
        }
        
        for ext, lang in language_extensions.items():
            if file_path.endswith(ext) and lang not in analysis.languages_detected:
                analysis.languages_detected.append(lang)

    async def _detect_architecture_patterns(self, project_files: Dict[str, Any], analysis: ProjectAnalysis) -> None:
        """Detect architecture patterns from project structure"""
        all_content = " ".join(str(content) for content in project_files.values()).lower()
        all_files = " ".join(project_files.keys()).lower()
        
        for pattern, indicators in self.architecture_patterns.items():
            for indicator in indicators:
                if indicator.lower() in all_content or indicator.lower() in all_files:
                    if pattern not in analysis.architecture_patterns:
                        analysis.architecture_patterns.append(pattern)
                    break

    async def _detect_domain_requirements(self, analysis: ProjectAnalysis) -> None:
        """Detect domain-specific requirements from technology stack"""
        domain_mapping = {
            "healthcare": ["hipaa-compliance", "patient-data-security", "medical-records"],
            "finance": ["pci-compliance", "financial-security", "transaction-processing"],
            "ecommerce": ["payment-processing", "inventory-management", "order-fulfillment"],
            "auth": ["user-management", "session-handling", "permission-system"]
        }
        
        for tech in analysis.tech_stack:
            if tech in domain_mapping:
                analysis.domain_requirements.extend(domain_mapping[tech])
                
        # Remove duplicates
        analysis.domain_requirements = list(set(analysis.domain_requirements))

    async def _calculate_complexity_indicators(self, analysis: ProjectAnalysis) -> None:
        """Calculate complexity indicators for the project"""
        analysis.complexity_indicators = {
            "tech_stack_size": len(analysis.tech_stack),
            "architecture_complexity": len(analysis.architecture_patterns),
            "domain_complexity": len(analysis.domain_requirements),
            "language_diversity": len(analysis.languages_detected),
            "overall_complexity": self._calculate_overall_complexity(analysis)
        }

    def _calculate_overall_complexity(self, analysis: ProjectAnalysis) -> float:
        """Calculate overall project complexity score (0.0-1.0)"""
        complexity_score = 0.0
        
        # Technology stack complexity
        tech_complexity = min(len(analysis.tech_stack) / 10, 0.3)  # Max 0.3 for tech stack
        
        # Architecture complexity
        arch_complexity = min(len(analysis.architecture_patterns) / 5, 0.25)  # Max 0.25 for architecture
        
        # Domain complexity
        domain_complexity = min(len(analysis.domain_requirements) / 8, 0.25)  # Max 0.25 for domain
        
        # Language diversity
        lang_complexity = min(len(analysis.languages_detected) / 4, 0.2)  # Max 0.2 for languages
        
        complexity_score = tech_complexity + arch_complexity + domain_complexity + lang_complexity
        
        return min(complexity_score, 1.0)

    async def determine_required_agents(self, analysis: ProjectAnalysis) -> List[AgentSpec]:
        """
        Generate appropriate agent specifications based on analysis
        Implementation for test_required_agents_determination()
        """
        required_agents = []
        
        # Determine agents based on detected technologies
        agent_rules = await self._get_agent_rules()
        
        for rule in agent_rules:
            if await self._rule_matches_analysis(rule, analysis):
                spec = AgentSpec(
                    agent_type=rule["agent_type"],
                    model_tier=rule["model_tier"],
                    specialization=rule.get("specialization")
                )
                required_agents.append(spec)
        
        # Ensure we have basic agents for any project
        if not any(agent.agent_type == "code-formatter" for agent in required_agents):
            required_agents.append(AgentSpec("code-formatter", "haiku"))
            
        if not any(agent.agent_type == "import-organizer" for agent in required_agents):
            required_agents.append(AgentSpec("import-organizer", "haiku"))
        
        logger.info(f"Generated {len(required_agents)} required agents for project {analysis.project_id}")
        
        return required_agents

    async def _get_agent_rules(self) -> List[Dict[str, Any]]:
        """Get agent generation rules with Sonnet-first preference"""
        return [
            # Healthcare-specific agents (from PRD example)
            {
                "agent_type": "healthcare-compliance-agent",
                "model_tier": "opus",  # Complex compliance requires Opus
                "tech_requirements": ["healthcare", "hipaa-compliance"],
                "specialization": "HIPAA compliance validation"
            },
            {
                "agent_type": "patient-data-handler", 
                "model_tier": "opus",  # Sensitive data handling requires Opus
                "tech_requirements": ["healthcare", "patient"],
                "specialization": "Patient data management"
            },
            {
                "agent_type": "appointment-scheduler",
                "model_tier": "sonnet",  # Business logic, perfect for Sonnet
                "tech_requirements": ["healthcare"],
                "specialization": "Healthcare scheduling systems"
            },
            {
                "agent_type": "form-validator",
                "model_tier": "sonnet",  # Validation logic, perfect for Sonnet
                "tech_requirements": ["healthcare", "forms"],
                "specialization": "Medical form validation"
            },
            
            # Architecture and Security (Opus tier)
            {
                "agent_type": "system-architect",
                "model_tier": "opus",  # Complex architecture requires Opus
                "tech_requirements": ["architecture", "system-design"],
                "specialization": "System architecture and design"
            },
            {
                "agent_type": "security-auditor",
                "model_tier": "opus",  # Security requires highest tier
                "tech_requirements": ["auth", "security"],
                "specialization": "Security analysis and auditing"
            },
            {
                "agent_type": "database-architect",
                "model_tier": "opus",  # Database design is complex
                "tech_requirements": ["postgresql", "mysql", "mongodb"],
                "specialization": "Database design and optimization"
            },
            
            # Development agents (Sonnet tier - DEFAULT)
            {
                "agent_type": "api-developer",
                "model_tier": "sonnet",  # API development, perfect for Sonnet
                "tech_requirements": ["fastapi", "rest-api", "graphql"],
                "specialization": "API development and documentation"
            },
            {
                "agent_type": "frontend-developer",
                "model_tier": "sonnet",  # UI development, perfect for Sonnet
                "tech_requirements": ["react", "vue", "angular"],
                "specialization": "Frontend development and UI components"
            },
            {
                "agent_type": "backend-developer",
                "model_tier": "sonnet",  # Backend development, perfect for Sonnet
                "tech_requirements": ["python", "node", "java"],
                "specialization": "Backend services and business logic"
            },
            {
                "agent_type": "fullstack-developer",
                "model_tier": "sonnet",  # Full stack development, perfect for Sonnet
                "tech_requirements": ["frontend", "backend"],
                "specialization": "Full stack application development"
            },
            {
                "agent_type": "devops-engineer",
                "model_tier": "sonnet",  # Infrastructure management, perfect for Sonnet
                "tech_requirements": ["docker", "kubernetes", "aws"],
                "specialization": "DevOps and infrastructure automation"
            },
            {
                "agent_type": "test-engineer", 
                "model_tier": "sonnet",  # Testing requires good analysis, perfect for Sonnet
                "tech_requirements": ["pytest", "jest", "cypress"],
                "specialization": "Test automation and quality assurance"
            },
            {
                "agent_type": "code-implementer",
                "model_tier": "sonnet",  # Code implementation, perfect for Sonnet
                "tech_requirements": [],  # General implementation
                "specialization": "Feature implementation and development"
            },
            {
                "agent_type": "code-quality-reviewer",
                "model_tier": "sonnet",  # Code review, perfect for Sonnet
                "tech_requirements": [],  # General review
                "specialization": "Code quality and review"
            },
            {
                "agent_type": "performance-optimizer",
                "model_tier": "sonnet",  # Performance work, good for Sonnet
                "tech_requirements": ["optimization", "performance"],
                "specialization": "Performance analysis and optimization"
            },
            {
                "agent_type": "documentation-writer",
                "model_tier": "sonnet",  # Documentation requires good writing, perfect for Sonnet
                "tech_requirements": [],  # Always needed
                "specialization": "Technical documentation and README files"
            },
            {
                "agent_type": "bug-fixer",
                "model_tier": "sonnet",  # Bug fixing often needs analysis, good for Sonnet
                "tech_requirements": [],  # General debugging
                "specialization": "Bug identification and resolution"
            },
            {
                "agent_type": "refactoring-specialist",
                "model_tier": "sonnet",  # Refactoring requires good analysis, perfect for Sonnet
                "tech_requirements": [],  # General refactoring
                "specialization": "Code refactoring and improvement"
            },
            
            # Basic task agents (Haiku tier - ONLY very basic tasks)
            {
                "agent_type": "code-formatter",
                "model_tier": "haiku",  # Pure formatting task
                "tech_requirements": ["formatting"],
                "specialization": "Code formatting and style consistency only"
            },
            {
                "agent_type": "import-organizer",
                "model_tier": "haiku",  # Pure import management
                "tech_requirements": ["imports"], 
                "specialization": "Import statement organization only"
            },
            {
                "agent_type": "typo-fixer",
                "model_tier": "haiku",  # Very basic text fixes
                "tech_requirements": ["typos", "spelling"],
                "specialization": "Typo and spelling corrections only"
            },
            {
                "agent_type": "comment-adder",
                "model_tier": "haiku",  # Basic comment addition
                "tech_requirements": ["comments"],
                "specialization": "Adding basic code comments only"
            }
        ]

    async def _rule_matches_analysis(self, rule: Dict[str, Any], analysis: ProjectAnalysis) -> bool:
        """Check if agent rule matches project analysis"""
        tech_requirements = rule.get("tech_requirements", [])
        
        if not tech_requirements:
            return True  # No specific requirements
        
        # Check if any required technologies are present
        all_detected = analysis.tech_stack + analysis.architecture_patterns + analysis.domain_requirements
        
        for req in tech_requirements:
            if any(req.lower() in detected.lower() for detected in all_detected):
                return True
                
        return False

    async def get_complexity_assessment(self, analysis: ProjectAnalysis) -> float:
        """Get overall complexity assessment for tier routing"""
        return analysis.complexity_indicators.get("overall_complexity", 0.5)