"""
Project-Specific Agent Factory v3.0 - Intelligent Agent Creation and Management
Based on F-PSA-001, F-PSA-002, F-PSA-003 from PRD specifications

NLNH Protocol: Real project analysis with actual technology detection
DGTS Enforcement: No fake agent creation, actual specialized agent spawning
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ProjectComplexity(Enum):
    """Project complexity levels for agent selection"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH = "high"
    ENTERPRISE = "enterprise"


class ModelTier(Enum):
    """Model tier assignments for different agent types"""
    OPUS = "opus"
    SONNET = "sonnet"
    HAIKU = "haiku"


@dataclass
class ProjectAnalysis:
    """Comprehensive project analysis results (F-PSA-001)"""
    project_id: str
    technology_stack: Dict[str, List[str]]
    architecture_patterns: List[str]
    domain_requirements: Dict[str, str]
    compliance_needs: List[str]
    performance_requirements: Dict[str, Any]
    complexity_score: float
    recommended_agents: List[Dict[str, Any]]
    analysis_timestamp: datetime
    confidence_score: float = 0.8
    
    def __post_init__(self):
        """Validate analysis data"""
        self.complexity_score = max(0.0, min(1.0, self.complexity_score))
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
    
    @property
    def complexity_level(self) -> ProjectComplexity:
        """Determine complexity level based on score"""
        if self.complexity_score >= 0.8:
            return ProjectComplexity.ENTERPRISE
        elif self.complexity_score >= 0.6:
            return ProjectComplexity.HIGH
        elif self.complexity_score >= 0.4:
            return ProjectComplexity.MEDIUM
        else:
            return ProjectComplexity.SIMPLE


@dataclass
class AgentSpec:
    """Specification for agent creation"""
    agent_type: str
    model_tier: str
    specialization: str
    priority: int
    estimated_cost: float
    required_knowledge: List[str]
    dependencies: List[str]
    min_confidence: float = 0.7
    max_parallel_tasks: int = 3
    
    def __post_init__(self):
        if self.required_knowledge is None:
            self.required_knowledge = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AgentHealthMetrics:
    """Agent performance and health metrics (F-PSA-003)"""
    agent_id: str
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    cost_per_task: float = 0.0
    knowledge_contributions: int = 0
    last_active: datetime = None
    needs_retraining: bool = False
    performance_trend: str = "stable"  # improving, stable, declining
    specialization_accuracy: float = 0.0
    
    def __post_init__(self):
        if self.last_active is None:
            self.last_active = datetime.now()
    
    def update_metrics(self, task_result: Dict[str, Any]):
        """Update metrics based on task result"""
        self.total_tasks += 1
        self.last_active = datetime.now()
        
        if task_result.get("success", False):
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.success_rate = self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
        
        # Update timing and cost
        if "execution_time" in task_result:
            # Simple moving average for now
            self.avg_execution_time = ((self.avg_execution_time * (self.total_tasks - 1)) + 
                                     task_result["execution_time"]) / self.total_tasks
        
        if "cost" in task_result:
            self.cost_per_task = ((self.cost_per_task * (self.total_tasks - 1)) + 
                                task_result["cost"]) / self.total_tasks
        
        # Check retraining criteria
        if self.total_tasks >= 10 and self.success_rate < 0.7:
            self.needs_retraining = True
        
        # Determine performance trend
        if self.total_tasks >= 5:
            if self.success_rate > 0.85:
                self.performance_trend = "improving"
            elif self.success_rate < 0.6:
                self.performance_trend = "declining"
            else:
                self.performance_trend = "stable"


class TechnologyStackAnalyzer:
    """Analyze project files to detect technology stack"""
    
    def __init__(self):
        self.language_patterns = {
            "python": [r"\.py$", r"requirements\.txt", r"setup\.py", r"pyproject\.toml"],
            "javascript": [r"\.js$", r"package\.json", r"\.ts$", r"\.tsx$"],
            "typescript": [r"\.ts$", r"\.tsx$", r"tsconfig\.json"],
            "java": [r"\.java$", r"pom\.xml", r"build\.gradle"],
            "csharp": [r"\.cs$", r"\.csproj$", r"\.sln$"],
            "go": [r"\.go$", r"go\.mod", r"go\.sum"],
            "rust": [r"\.rs$", r"Cargo\.toml", r"Cargo\.lock"],
            "php": [r"\.php$", r"composer\.json"],
            "ruby": [r"\.rb$", r"Gemfile", r"\.gemspec$"],
        }
        
        self.framework_patterns = {
            "react": [r"react", r"jsx", r"\.tsx$"],
            "vue": [r"\.vue$", r"vue\.config"],
            "angular": [r"angular\.json", r"@angular"],
            "django": [r"django", r"manage\.py"],
            "flask": [r"flask", r"app\.py"],
            "fastapi": [r"fastapi", r"uvicorn"],
            "express": [r"express", r"app\.js"],
            "spring": [r"springframework", r"@SpringBootApplication"],
            "laravel": [r"laravel", r"artisan"],
        }
        
        self.database_patterns = {
            "postgresql": [r"postgresql", r"psycopg2", r"pg"],
            "mysql": [r"mysql", r"pymysql"],
            "sqlite": [r"sqlite", r"\.db$", r"\.sqlite$"],
            "mongodb": [r"mongodb", r"pymongo", r"mongoose"],
            "redis": [r"redis", r"hiredis"],
            "elasticsearch": [r"elasticsearch", r"elastic"],
        }
    
    async def analyze_files(self, project_files: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze project files to detect technology stack"""
        stack = {
            "languages": [],
            "frameworks": [],
            "databases": [],
            "frontend": [],
            "backend": [],
            "tools": []
        }
        
        if not project_files:
            return stack
        
        # Analyze file paths and contents
        all_content = ""
        file_paths = []
        
        for file_path, content in project_files.items():
            file_paths.append(file_path)
            if isinstance(content, str):
                all_content += content.lower()
        
        # Detect languages
        for language, patterns in self.language_patterns.items():
            if any(re.search(pattern, file_path, re.IGNORECASE) for pattern in patterns for file_path in file_paths):
                stack["languages"].append(language)
            
            # Also check content for language indicators
            for pattern in patterns:
                if re.search(pattern.replace(r"\$", ""), all_content, re.IGNORECASE):
                    if language not in stack["languages"]:
                        stack["languages"].append(language)
        
        # Detect frameworks
        for framework, patterns in self.framework_patterns.items():
            for pattern in patterns:
                if (any(re.search(pattern, file_path, re.IGNORECASE) for file_path in file_paths) or
                    re.search(pattern, all_content, re.IGNORECASE)):
                    stack["frameworks"].append(framework)
        
        # Detect databases
        for database, patterns in self.database_patterns.items():
            for pattern in patterns:
                if (any(re.search(pattern, file_path, re.IGNORECASE) for file_path in file_paths) or
                    re.search(pattern, all_content, re.IGNORECASE)):
                    stack["databases"].append(database)
        
        # Categorize frontend/backend
        frontend_indicators = ["react", "vue", "angular", "html", "css", "javascript", "typescript"]
        backend_indicators = ["python", "java", "csharp", "go", "rust", "php", "ruby", "django", "flask", "express"]
        
        for tech in stack["languages"] + stack["frameworks"]:
            if tech in frontend_indicators:
                stack["frontend"].append(tech)
            elif tech in backend_indicators:
                stack["backend"].append(tech)
        
        # Remove duplicates
        for key in stack:
            stack[key] = list(set(stack[key]))
        
        return stack


class DomainAnalyzer:
    """Analyze project domain and specific requirements"""
    
    def __init__(self):
        self.domain_keywords = {
            "healthcare": [
                "patient", "medical", "hipaa", "hl7", "fhir", "healthcare",
                "hospital", "clinic", "doctor", "nurse", "medication", "diagnosis"
            ],
            "finance": [
                "bank", "financial", "payment", "transaction", "money", "credit",
                "debit", "loan", "investment", "trading", "portfolio", "insurance"
            ],
            "ecommerce": [
                "shop", "cart", "checkout", "product", "inventory", "order",
                "payment", "shipping", "customer", "retail", "marketplace"
            ],
            "education": [
                "student", "teacher", "course", "lesson", "grade", "school",
                "university", "learning", "curriculum", "assignment"
            ],
            "government": [
                "government", "public", "citizen", "official", "regulation",
                "compliance", "audit", "policy", "federal", "state", "municipal"
            ],
            "logistics": [
                "shipping", "delivery", "warehouse", "tracking", "transport",
                "logistics", "supply", "chain", "freight", "route"
            ]
        }
        
        self.compliance_mapping = {
            "healthcare": ["HIPAA", "HITECH", "FDA", "SOX"],
            "finance": ["PCI-DSS", "SOX", "GDPR", "PSD2", "BASEL"],
            "ecommerce": ["PCI-DSS", "GDPR", "CCPA", "COPPA"],
            "education": ["FERPA", "COPPA", "GDPR"],
            "government": ["FISMA", "FedRAMP", "NIST", "SOX"],
            "logistics": ["DOT", "FMCSA", "GDPR"]
        }
    
    async def analyze_domain(self, project_files: Dict[str, Any], project_id: str) -> Tuple[str, List[str]]:
        """Analyze project domain and determine compliance needs"""
        domain_scores = {}
        all_content = project_id.lower()
        
        # Add file content to analysis
        if project_files:
            for content in project_files.values():
                if isinstance(content, str):
                    all_content += " " + content.lower()
        
        # Score each domain based on keyword matches
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_content)
            if score > 0:
                domain_scores[domain] = score
        
        # Determine primary domain
        primary_domain = "general"
        if domain_scores:
            primary_domain = max(domain_scores.keys(), key=lambda x: domain_scores[x])
        
        # Get compliance requirements
        compliance_needs = self.compliance_mapping.get(primary_domain, ["GDPR"])
        
        return primary_domain, compliance_needs


class ArchitecturePatternDetector:
    """Detect architecture patterns from project structure"""
    
    def __init__(self):
        self.pattern_indicators = {
            "mvc": [
                r"models?/", r"views?/", r"controllers?/",
                r"model\.py", r"view\.py", r"controller\.py"
            ],
            "microservices": [
                r"services?/", r"api/", r"docker", r"kubernetes",
                r"service\.py", r"\.service\.", r"microservice"
            ],
            "spa": [
                r"single.*page", r"spa", r"router", r"routes?/",
                r"components?/", r"pages?/"
            ],
            "rest": [
                r"api/", r"rest", r"endpoints?", r"@app\.route",
                r"@api\.", r"RestController", r"APIView"
            ],
            "graphql": [
                r"graphql", r"schema", r"resolvers?", r"mutations?",
                r"queries", r"\.gql", r"\.graphql"
            ],
            "event-driven": [
                r"events?/", r"handlers?/", r"listeners?/",
                r"pubsub", r"queue", r"kafka", r"rabbitmq"
            ],
            "layered": [
                r"layers?/", r"business/", r"data/", r"presentation/",
                r"repository", r"service", r"dto"
            ]
        }
    
    async def detect_patterns(self, project_files: Dict[str, Any]) -> List[str]:
        """Detect architecture patterns from project structure"""
        patterns = []
        
        if not project_files:
            return ["unknown"]
        
        file_paths = list(project_files.keys())
        all_content = ""
        
        for content in project_files.values():
            if isinstance(content, str):
                all_content += content.lower()
        
        # Check each pattern
        for pattern_name, indicators in self.pattern_indicators.items():
            matches = 0
            
            for indicator in indicators:
                # Check file paths
                if any(re.search(indicator, file_path, re.IGNORECASE) for file_path in file_paths):
                    matches += 1
                
                # Check content
                if re.search(indicator, all_content, re.IGNORECASE):
                    matches += 1
            
            # If we have multiple matches, pattern is likely present
            if matches >= 2:
                patterns.append(pattern_name)
        
        return patterns if patterns else ["monolith"]


class ProjectAnalyzer:
    """Main project analyzer implementing F-PSA-001"""
    
    def __init__(self):
        self.tech_analyzer = TechnologyStackAnalyzer()
        self.domain_analyzer = DomainAnalyzer()
        self.pattern_detector = ArchitecturePatternDetector()
        self.analysis_cache = {}
    
    async def analyze_project(self, project_id: str, project_files: Dict[str, Any] = None) -> ProjectAnalysis:
        """Comprehensive project analysis"""
        # Check cache first
        cache_key = f"{project_id}_{hash(str(project_files))}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        logger.info(f"Analyzing project: {project_id}")
        
        # Technology stack detection
        tech_stack = await self.tech_analyzer.analyze_files(project_files or {})
        
        # Domain analysis
        domain, compliance_needs = await self.domain_analyzer.analyze_domain(project_files or {}, project_id)
        
        # Architecture pattern detection
        patterns = await self.pattern_detector.detect_patterns(project_files or {})
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(tech_stack, domain, patterns, compliance_needs)
        
        # Domain-specific requirements
        domain_requirements = self._extract_domain_requirements(domain, tech_stack)
        
        # Performance requirements based on complexity and domain
        performance_requirements = self._determine_performance_requirements(complexity_score, domain)
        
        # Create analysis
        analysis = ProjectAnalysis(
            project_id=project_id,
            technology_stack=tech_stack,
            architecture_patterns=patterns,
            domain_requirements=domain_requirements,
            compliance_needs=compliance_needs,
            performance_requirements=performance_requirements,
            complexity_score=complexity_score,
            recommended_agents=[],  # Will be populated by agent generator
            analysis_timestamp=datetime.now(),
            confidence_score=self._calculate_confidence(project_files)
        )
        
        # Cache result
        self.analysis_cache[cache_key] = analysis
        
        logger.info(f"Project analysis complete: complexity={complexity_score:.2f}, domain={domain}")
        return analysis
    
    def _calculate_complexity(self, tech_stack: Dict[str, List[str]], domain: str, 
                            patterns: List[str], compliance: List[str]) -> float:
        """Calculate project complexity score (0.0 - 1.0)"""
        complexity = 0.0
        
        # Base complexity from technology diversity
        total_techs = sum(len(stack) for stack in tech_stack.values())
        complexity += min(total_techs * 0.05, 0.3)  # Max 0.3 from tech diversity
        
        # Domain complexity
        domain_complexity = {
            "healthcare": 0.3, "finance": 0.25, "government": 0.25,
            "ecommerce": 0.2, "education": 0.15, "logistics": 0.15,
            "general": 0.1
        }
        complexity += domain_complexity.get(domain, 0.1)
        
        # Architecture pattern complexity
        pattern_complexity = {
            "microservices": 0.2, "event-driven": 0.15, "layered": 0.1,
            "rest": 0.05, "spa": 0.05, "mvc": 0.05, "monolith": 0.0
        }
        for pattern in patterns:
            complexity += pattern_complexity.get(pattern, 0.02)
        
        # Compliance complexity
        complexity += len(compliance) * 0.1
        
        return min(complexity, 1.0)
    
    def _extract_domain_requirements(self, domain: str, tech_stack: Dict[str, List[str]]) -> Dict[str, str]:
        """Extract domain-specific requirements"""
        requirements = {}
        
        if domain == "healthcare":
            requirements.update({
                "patient_data_encryption": "mandatory",
                "audit_trail": "required",
                "access_control": "role_based",
                "data_retention": "7_years",
                "backup_frequency": "daily"
            })
        elif domain == "finance":
            requirements.update({
                "transaction_encryption": "mandatory",
                "fraud_detection": "required",
                "audit_trail": "required",
                "data_retention": "7_years",
                "uptime_requirement": "99.9%"
            })
        elif domain == "ecommerce":
            requirements.update({
                "payment_processing": "required",
                "inventory_management": "required",
                "user_authentication": "required",
                "search_functionality": "required",
                "performance_target": "<2s_load_time"
            })
        else:
            requirements.update({
                "user_authentication": "recommended",
                "data_validation": "required",
                "error_handling": "required"
            })
        
        return requirements
    
    def _determine_performance_requirements(self, complexity: float, domain: str) -> Dict[str, Any]:
        """Determine performance requirements based on complexity and domain"""
        base_requirements = {
            "response_time": "2s",
            "throughput": "100_rps",
            "availability": "99.5%",
            "scalability": "horizontal"
        }
        
        # Adjust based on complexity
        if complexity > 0.8:
            base_requirements.update({
                "response_time": "500ms",
                "throughput": "1000_rps", 
                "availability": "99.9%",
                "scalability": "auto_scaling"
            })
        elif complexity > 0.6:
            base_requirements.update({
                "response_time": "1s",
                "throughput": "500_rps",
                "availability": "99.8%"
            })
        
        # Domain-specific adjustments
        if domain in ["finance", "healthcare"]:
            base_requirements["availability"] = "99.95%"
            base_requirements["data_consistency"] = "strong"
        elif domain == "ecommerce":
            base_requirements["response_time"] = "1s"
            base_requirements["peak_load_capacity"] = "10x_normal"
        
        return base_requirements
    
    def _calculate_confidence(self, project_files: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in analysis based on available data"""
        if not project_files:
            return 0.3  # Low confidence without files
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on file count and types
        file_count = len(project_files)
        confidence += min(file_count * 0.05, 0.3)  # Up to 0.3 bonus
        
        # Check for important files
        important_files = [
            "package.json", "requirements.txt", "pom.xml", "Cargo.toml",
            "go.mod", "composer.json", "setup.py", "pyproject.toml"
        ]
        
        found_important = sum(1 for file in project_files.keys() 
                            if any(important in file.lower() for important in important_files))
        confidence += found_important * 0.1
        
        return min(confidence, 0.95)  # Max 95% confidence


class AgentSwarmGenerator:
    """Agent swarm generation implementing F-PSA-002"""
    
    def __init__(self):
        self.agent_templates = self._load_agent_templates()
    
    def _load_agent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load agent templates for different project types and complexities"""
        return {
            # Core development agents
            "system-architect": {
                "base_tier": ModelTier.OPUS,
                "specializations": {
                    "enterprise": "enterprise-architecture",
                    "microservices": "microservices-architecture", 
                    "healthcare": "healthcare-systems-architecture",
                    "finance": "financial-systems-architecture"
                },
                "required_knowledge": ["architecture-patterns", "system-design"],
                "priority_base": 1
            },
            "code-implementer": {
                "base_tier": ModelTier.SONNET,
                "specializations": {
                    "full-stack": "full-stack-development",
                    "backend": "backend-api-development",
                    "frontend": "frontend-development",
                    "mobile": "mobile-development"
                },
                "required_knowledge": ["programming", "frameworks"],
                "priority_base": 2
            },
            "test-engineer": {
                "base_tier": ModelTier.SONNET,
                "specializations": {
                    "integration": "integration-testing",
                    "unit": "unit-testing",
                    "e2e": "end-to-end-testing",
                    "performance": "performance-testing"
                },
                "required_knowledge": ["testing-frameworks", "quality-assurance"],
                "priority_base": 2
            },
            "security-auditor": {
                "base_tier": ModelTier.OPUS,
                "specializations": {
                    "compliance": "compliance-auditing",
                    "penetration": "penetration-testing",
                    "code-security": "secure-code-review",
                    "infrastructure": "infrastructure-security"
                },
                "required_knowledge": ["security-standards", "compliance-requirements"],
                "priority_base": 1
            },
            "code-formatter": {
                "base_tier": ModelTier.HAIKU,
                "specializations": {
                    "linting": "code-linting",
                    "formatting": "code-formatting",
                    "style": "code-style-enforcement"
                },
                "required_knowledge": ["code-standards", "formatting-tools"],
                "priority_base": 3
            },
            
            # Specialized domain agents
            "compliance-auditor": {
                "base_tier": ModelTier.OPUS,
                "specializations": {
                    "healthcare": "hipaa-compliance",
                    "finance": "financial-compliance",
                    "government": "government-compliance"
                },
                "required_knowledge": ["regulatory-requirements", "audit-procedures"],
                "priority_base": 1
            },
            "performance-optimizer": {
                "base_tier": ModelTier.SONNET,
                "specializations": {
                    "database": "database-optimization",
                    "frontend": "frontend-performance",
                    "api": "api-performance"
                },
                "required_knowledge": ["performance-analysis", "optimization-techniques"],
                "priority_base": 2
            },
            "documentation-generator": {
                "base_tier": ModelTier.SONNET,
                "specializations": {
                    "api": "api-documentation",
                    "user": "user-documentation",
                    "technical": "technical-documentation"
                },
                "required_knowledge": ["documentation-standards", "technical-writing"],
                "priority_base": 3
            }
        }
    
    async def determine_required_agents(self, analysis: ProjectAnalysis) -> List[AgentSpec]:
        """Determine required agents based on project analysis"""
        specs = []
        complexity = analysis.complexity_level
        domain = self._determine_domain(analysis.domain_requirements)
        
        logger.info(f"Generating agent swarm for {complexity.value} complexity {domain} project")
        
        # Core agents based on complexity
        if complexity == ProjectComplexity.ENTERPRISE:
            specs.extend([
                self._create_agent_spec("system-architect", analysis, specialization="enterprise"),
                self._create_agent_spec("security-auditor", analysis, specialization="compliance"),
                self._create_agent_spec("code-implementer", analysis, specialization="backend"),
                self._create_agent_spec("code-implementer", analysis, specialization="frontend"),
                self._create_agent_spec("test-engineer", analysis, specialization="integration"),
                self._create_agent_spec("performance-optimizer", analysis, specialization="database"),
                self._create_agent_spec("documentation-generator", analysis, specialization="technical"),
                self._create_agent_spec("code-formatter", analysis, specialization="linting")
            ])
        elif complexity == ProjectComplexity.HIGH:
            specs.extend([
                self._create_agent_spec("system-architect", analysis, specialization="microservices"),
                self._create_agent_spec("security-auditor", analysis, specialization="code-security"),
                self._create_agent_spec("code-implementer", analysis, specialization="full-stack"),
                self._create_agent_spec("test-engineer", analysis, specialization="unit"),
                self._create_agent_spec("documentation-generator", analysis, specialization="api"),
                self._create_agent_spec("code-formatter", analysis, specialization="formatting")
            ])
        elif complexity == ProjectComplexity.MEDIUM:
            specs.extend([
                self._create_agent_spec("code-implementer", analysis, specialization="full-stack"),
                self._create_agent_spec("test-engineer", analysis, specialization="unit"),
                self._create_agent_spec("code-formatter", analysis, specialization="style")
            ])
        else:  # SIMPLE
            specs.extend([
                self._create_agent_spec("code-implementer", analysis, specialization="simple-features"),
                self._create_agent_spec("code-formatter", analysis, specialization="basic-formatting")
            ])
        
        # Add domain-specific agents
        specs.extend(self._add_domain_specific_agents(analysis, domain))
        
        # Add compliance agents
        specs.extend(self._add_compliance_agents(analysis))
        
        # Optimize for Sonnet preference (user requirement)
        specs = self._optimize_for_sonnet(specs, analysis)
        
        return specs
    
    def _create_agent_spec(self, agent_type: str, analysis: ProjectAnalysis, 
                          specialization: str = None) -> AgentSpec:
        """Create agent specification"""
        template = self.agent_templates.get(agent_type, {})
        
        # Determine model tier (prefer Sonnet as per user requirement)
        base_tier = template.get("base_tier", ModelTier.SONNET)
        
        # Override with Sonnet for most agents (except truly complex ones)
        if base_tier == ModelTier.HAIKU:
            model_tier = ModelTier.HAIKU.value  # Keep Haiku for simple tasks
        elif agent_type in ["system-architect"] and analysis.complexity_score > 0.8:
            model_tier = ModelTier.OPUS.value  # Keep Opus for complex architecture
        else:
            model_tier = ModelTier.SONNET.value  # Default to Sonnet
        
        # Determine specialization
        specializations = template.get("specializations", {})
        if not specialization and specializations:
            specialization = list(specializations.values())[0]  # Default specialization
        elif specialization in specializations:
            specialization = specializations[specialization]
        
        # Calculate priority
        priority = template.get("priority_base", 2)
        if analysis.complexity_score > 0.8:
            priority = max(1, priority - 1)  # Increase priority for complex projects
        
        # Estimate cost based on tier and complexity
        cost_multipliers = {
            ModelTier.OPUS.value: 1.0,
            ModelTier.SONNET.value: 0.3,
            ModelTier.HAIKU.value: 0.1
        }
        estimated_cost = cost_multipliers.get(model_tier, 0.3) * (1 + analysis.complexity_score)
        
        return AgentSpec(
            agent_type=agent_type,
            model_tier=model_tier,
            specialization=specialization or agent_type,
            priority=priority,
            estimated_cost=estimated_cost,
            required_knowledge=template.get("required_knowledge", []),
            dependencies=[]
        )
    
    def _determine_domain(self, domain_requirements: Dict[str, str]) -> str:
        """Determine primary domain from requirements"""
        if "patient_data_encryption" in domain_requirements:
            return "healthcare"
        elif "transaction_encryption" in domain_requirements:
            return "finance"
        elif "payment_processing" in domain_requirements:
            return "ecommerce"
        else:
            return "general"
    
    def _add_domain_specific_agents(self, analysis: ProjectAnalysis, domain: str) -> List[AgentSpec]:
        """Add domain-specific agents"""
        specs = []
        
        if domain == "healthcare" and analysis.complexity_score > 0.6:
            specs.append(self._create_agent_spec("compliance-auditor", analysis, "healthcare"))
        elif domain == "finance" and analysis.complexity_score > 0.6:
            specs.append(self._create_agent_spec("compliance-auditor", analysis, "finance"))
        elif domain == "ecommerce" and analysis.complexity_score > 0.5:
            specs.append(self._create_agent_spec("performance-optimizer", analysis, "api"))
        
        return specs
    
    def _add_compliance_agents(self, analysis: ProjectAnalysis) -> List[AgentSpec]:
        """Add compliance-specific agents"""
        specs = []
        
        high_compliance = ["HIPAA", "PCI-DSS", "SOX", "FISMA"]
        
        for compliance in analysis.compliance_needs:
            if compliance in high_compliance:
                # Add specialized security auditor for high-compliance requirements
                specs.append(AgentSpec(
                    agent_type="security-auditor",
                    model_tier=ModelTier.OPUS.value,  # Compliance always uses Opus
                    specialization=f"{compliance.lower()}-compliance",
                    priority=1,  # High priority
                    estimated_cost=1.0,
                    required_knowledge=[compliance.lower(), "compliance-frameworks"],
                    dependencies=[]
                ))
                break  # Only add one compliance auditor
        
        return specs
    
    def _optimize_for_sonnet(self, specs: List[AgentSpec], analysis: ProjectAnalysis) -> List[AgentSpec]:
        """Optimize agent specs to prefer Sonnet tier (user requirement)"""
        optimized = []
        
        for spec in specs:
            # Keep Haiku for truly basic tasks
            if spec.agent_type == "code-formatter" and spec.specialization in ["basic-formatting", "linting"]:
                pass  # Keep as is
            # Keep Opus only for critical, complex tasks
            elif (spec.agent_type in ["system-architect", "security-auditor"] and 
                  spec.specialization in ["enterprise-architecture", "compliance"] and
                  analysis.complexity_score > 0.8):
                pass  # Keep Opus for critical complex tasks
            # Everything else -> Sonnet
            else:
                spec.model_tier = ModelTier.SONNET.value
                # Adjust cost for Sonnet
                spec.estimated_cost = 0.3 * (1 + analysis.complexity_score)
            
            optimized.append(spec)
        
        return optimized


class AgentHealthMonitor:
    """Agent health monitoring implementing F-PSA-003"""
    
    def __init__(self):
        self.metrics: Dict[str, AgentHealthMetrics] = {}
        self.monitoring_active: bool = True
        self.alert_thresholds = {
            "min_success_rate": 0.7,
            "max_avg_execution_time": 300.0,  # 5 minutes
            "max_cost_per_task": 1.0,
            "min_tasks_for_analysis": 10
        }
    
    async def track_agent_performance(self, agent_id: str, task_result: Dict[str, Any]) -> AgentHealthMetrics:
        """Track agent performance metrics"""
        if not self.monitoring_active:
            return None
        
        if agent_id not in self.metrics:
            self.metrics[agent_id] = AgentHealthMetrics(agent_id)
        
        metrics = self.metrics[agent_id]
        metrics.update_metrics(task_result)
        
        # Check for alerts
        await self._check_health_alerts(metrics)
        
        logger.debug(f"Updated metrics for agent {agent_id}: "
                    f"success_rate={metrics.success_rate:.2f}, "
                    f"total_tasks={metrics.total_tasks}")
        
        return metrics
    
    async def _check_health_alerts(self, metrics: AgentHealthMetrics):
        """Check for health alerts and trigger actions"""
        if metrics.total_tasks < self.alert_thresholds["min_tasks_for_analysis"]:
            return
        
        alerts = []
        
        if metrics.success_rate < self.alert_thresholds["min_success_rate"]:
            alerts.append(f"Low success rate: {metrics.success_rate:.2f}")
        
        if metrics.avg_execution_time > self.alert_thresholds["max_avg_execution_time"]:
            alerts.append(f"High execution time: {metrics.avg_execution_time:.1f}s")
        
        if metrics.cost_per_task > self.alert_thresholds["max_cost_per_task"]:
            alerts.append(f"High cost per task: ${metrics.cost_per_task:.3f}")
        
        if alerts:
            logger.warning(f"Agent {metrics.agent_id} health alerts: {', '.join(alerts)}")
    
    async def get_agents_needing_retraining(self) -> List[str]:
        """Get list of agent IDs that need retraining"""
        return [agent_id for agent_id, metrics in self.metrics.items() 
                if metrics.needs_retraining]
    
    async def get_performance_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for specific agent or all agents"""
        if agent_id and agent_id in self.metrics:
            metrics = self.metrics[agent_id]
            return {
                "agent_id": agent_id,
                "success_rate": metrics.success_rate,
                "total_tasks": metrics.total_tasks,
                "avg_execution_time": metrics.avg_execution_time,
                "cost_per_task": metrics.cost_per_task,
                "needs_retraining": metrics.needs_retraining,
                "performance_trend": metrics.performance_trend,
                "last_active": metrics.last_active.isoformat()
            }
        else:
            # Summary for all agents
            all_metrics = list(self.metrics.values())
            if not all_metrics:
                return {"total_agents": 0}
            
            return {
                "total_agents": len(all_metrics),
                "avg_success_rate": sum(m.success_rate for m in all_metrics) / len(all_metrics),
                "total_tasks": sum(m.total_tasks for m in all_metrics),
                "agents_needing_retraining": sum(1 for m in all_metrics if m.needs_retraining),
                "performance_trends": {
                    "improving": sum(1 for m in all_metrics if m.performance_trend == "improving"),
                    "stable": sum(1 for m in all_metrics if m.performance_trend == "stable"),
                    "declining": sum(1 for m in all_metrics if m.performance_trend == "declining")
                }
            }


class ProjectAgentFactory:
    """Main factory for project-specific agent creation"""
    
    def __init__(self):
        self.project_analyzer = ProjectAnalyzer()
        self.agent_generator = AgentSwarmGenerator()
        self.health_monitor = AgentHealthMonitor()
        self.active_projects: Dict[str, ProjectAnalysis] = {}
    
    async def setup_project_agents(self, project_id: str, 
                                 project_files: Dict[str, Any] = None) -> Tuple[ProjectAnalysis, List[AgentSpec]]:
        """Complete project setup with analysis and agent generation"""
        logger.info(f"Setting up agents for project: {project_id}")
        
        # Step 1: Analyze project
        analysis = await self.project_analyzer.analyze_project(project_id, project_files)
        
        # Step 2: Generate agent swarm
        agent_specs = await self.agent_generator.determine_required_agents(analysis)
        
        # Step 3: Update analysis with agent recommendations
        analysis.recommended_agents = [asdict(spec) for spec in agent_specs]
        
        # Store project analysis
        self.active_projects[project_id] = analysis
        
        logger.info(f"Project setup complete: {len(agent_specs)} agents recommended for {project_id}")
        
        return analysis, agent_specs
    
    async def get_project_analysis(self, project_id: str) -> Optional[ProjectAnalysis]:
        """Get cached project analysis"""
        return self.active_projects.get(project_id)
    
    async def track_agent_task(self, agent_id: str, task_result: Dict[str, Any]) -> AgentHealthMetrics:
        """Track agent task performance"""
        return await self.health_monitor.track_agent_performance(agent_id, task_result)
    
    async def get_agent_health_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent health summary"""
        return await self.health_monitor.get_performance_summary(agent_id)
    
    async def get_project_health_dashboard(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive project health dashboard"""
        analysis = self.active_projects.get(project_id)
        if not analysis:
            return {"error": f"Project {project_id} not found"}
        
        # Get agent health summaries
        agent_health = await self.health_monitor.get_performance_summary()
        
        # Get agents needing retraining
        retraining_needed = await self.health_monitor.get_agents_needing_retraining()
        
        return {
            "project_id": project_id,
            "analysis": {
                "complexity_score": analysis.complexity_score,
                "complexity_level": analysis.complexity_level.value,
                "technology_stack": analysis.technology_stack,
                "compliance_needs": analysis.compliance_needs,
                "recommended_agents": len(analysis.recommended_agents)
            },
            "agent_health": agent_health,
            "agents_needing_retraining": retraining_needed,
            "last_updated": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of Project-Specific Agent Factory"""
    print("🏭 Archon v3.0 Project-Specific Agent Factory")
    print("=" * 60)
    
    # Initialize factory
    factory = ProjectAgentFactory()
    
    # Example project files
    healthcare_files = {
        "package.json": json.dumps({
            "name": "healthcare-portal",
            "dependencies": {"react": "^18.0.0", "typescript": "^4.5.0"}
        }),
        "src/models/Patient.py": "class Patient:\n    def __init__(self, patient_id, name):\n        self.patient_id = patient_id",
        "requirements.txt": "django\ndjango-rest-framework\npsycopg2\ncryptography",
        "README.md": "Healthcare Patient Portal - HIPAA compliant system for managing patient data"
    }
    
    # Setup project agents
    print("\n🔍 Setting up agents for healthcare project...")
    analysis, agent_specs = await factory.setup_project_agents("healthcare-portal-001", healthcare_files)
    
    print(f"\n📊 Project Analysis Results:")
    print(f"  Complexity: {analysis.complexity_score:.2f} ({analysis.complexity_level.value})")
    print(f"  Technology Stack: {analysis.technology_stack}")
    print(f"  Compliance Needs: {analysis.compliance_needs}")
    print(f"  Architecture Patterns: {analysis.architecture_patterns}")
    
    print(f"\n🤖 Recommended Agents ({len(agent_specs)}):")
    for spec in agent_specs:
        print(f"  • {spec.agent_type} ({spec.model_tier}) - {spec.specialization} [Priority: {spec.priority}]")
    
    # Simulate agent task tracking
    print(f"\n📈 Simulating agent performance tracking...")
    for i, spec in enumerate(agent_specs[:3]):  # Track first 3 agents
        agent_id = f"{spec.agent_type}-{uuid.uuid4().hex[:8]}"
        
        # Simulate 5 tasks per agent
        for task_num in range(5):
            task_result = {
                "success": task_num < 4,  # 80% success rate
                "execution_time": 60 + (task_num * 10),
                "cost": spec.estimated_cost,
                "knowledge_contributed": task_num % 2 == 0
            }
            
            await factory.track_agent_task(agent_id, task_result)
    
    # Get project health dashboard
    print(f"\n📋 Project Health Dashboard:")
    dashboard = await factory.get_project_health_dashboard("healthcare-portal-001")
    
    print(f"  Total Agents: {dashboard['agent_health']['total_agents']}")
    print(f"  Average Success Rate: {dashboard['agent_health']['avg_success_rate']:.2f}")
    print(f"  Total Tasks Completed: {dashboard['agent_health']['total_tasks']}")
    print(f"  Agents Needing Retraining: {len(dashboard['agents_needing_retraining'])}")
    
    print("\n✅ Project-Specific Agent Factory demo completed!")


if __name__ == "__main__":
    asyncio.run(main())