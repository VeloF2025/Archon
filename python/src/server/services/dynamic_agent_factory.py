"""
Dynamic Agent Factory for Archon
Creates project-specific agents on-demand based on intelligent analysis
This enables Claude Code to request ANY agent type and have it dynamically created
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DynamicAgentFactory:
    """
    Intelligent agent factory that dynamically creates project-specific agents
    based on the requested agent type and project context
    """
    
    def __init__(self):
        # Base agent capabilities mapping
        self.base_agents = {
            "devops_engineer": {
                "domains": ["docker", "kubernetes", "ci/cd", "deployment", "infrastructure"],
                "skills": ["containerization", "orchestration", "automation", "monitoring"],
                "tools": ["docker", "docker-compose", "k8s", "helm", "terraform", "ansible"]
            },
            "python_backend_coder": {
                "domains": ["backend", "api", "server", "database", "python"],
                "skills": ["fastapi", "django", "flask", "sqlalchemy", "async"],
                "tools": ["python", "pip", "poetry", "pytest", "black", "mypy"]
            },
            "typescript_frontend_agent": {
                "domains": ["frontend", "ui", "react", "vue", "angular", "typescript"],
                "skills": ["components", "state-management", "routing", "styling"],
                "tools": ["npm", "yarn", "webpack", "vite", "eslint", "prettier"]
            },
            "security_auditor": {
                "domains": ["security", "vulnerability", "audit", "compliance", "penetration"],
                "skills": ["owasp", "scanning", "encryption", "authentication", "authorization"],
                "tools": ["snyk", "sonarqube", "owasp-zap", "nmap", "metasploit"]
            },
            "test_generator": {
                "domains": ["testing", "test", "coverage", "quality", "validation"],
                "skills": ["unit-testing", "integration", "e2e", "mocking", "tdd"],
                "tools": ["jest", "pytest", "mocha", "cypress", "playwright", "selenium"]
            },
            "database_designer": {
                "domains": ["database", "sql", "nosql", "schema", "migration"],
                "skills": ["normalization", "indexing", "optimization", "replication"],
                "tools": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch"]
            },
            "api_integrator": {
                "domains": ["api", "rest", "graphql", "websocket", "integration"],
                "skills": ["openapi", "swagger", "authentication", "rate-limiting"],
                "tools": ["postman", "insomnia", "swagger", "graphql-playground"]
            },
            "system_architect": {
                "domains": ["architecture", "design", "system", "scalability", "planning"],
                "skills": ["patterns", "microservices", "monolith", "distributed"],
                "tools": ["draw.io", "lucidchart", "plantuml", "c4-model"]
            },
            "ui_ux_designer": {
                "domains": ["ui", "ux", "design", "frontend", "accessibility"],
                "skills": ["responsive", "mobile-first", "a11y", "user-testing"],
                "tools": ["figma", "sketch", "adobe-xd", "tailwind", "material-ui"]
            },
            "performance_optimizer": {
                "domains": ["performance", "optimization", "speed", "efficiency", "profiling"],
                "skills": ["caching", "lazy-loading", "bundling", "minification"],
                "tools": ["lighthouse", "webpack-analyzer", "chrome-devtools", "datadog"]
            },
            "documentation_writer": {
                "domains": ["documentation", "docs", "readme", "api-docs", "guides"],
                "skills": ["technical-writing", "markdown", "diagrams", "examples"],
                "tools": ["markdown", "swagger", "docusaurus", "mkdocs", "sphinx"]
            },
            "code_reviewer": {
                "domains": ["review", "quality", "standards", "refactoring", "best-practices"],
                "skills": ["code-analysis", "patterns", "solid", "dry", "kiss"],
                "tools": ["eslint", "pylint", "sonarqube", "codeclimate", "prettier"]
            },
            "refactoring_specialist": {
                "domains": ["refactoring", "cleanup", "optimization", "modernization"],
                "skills": ["patterns", "anti-patterns", "debt-reduction", "simplification"],
                "tools": ["refactoring-tools", "ast-parsers", "codemods", "linters"]
            },
            "deployment_coordinator": {
                "domains": ["deployment", "release", "rollout", "production", "staging"],
                "skills": ["blue-green", "canary", "rollback", "monitoring"],
                "tools": ["jenkins", "github-actions", "gitlab-ci", "argocd", "spinnaker"]
            },
            "monitoring_agent": {
                "domains": ["monitoring", "observability", "metrics", "logging", "alerting"],
                "skills": ["dashboards", "alerts", "tracing", "apm"],
                "tools": ["prometheus", "grafana", "elk", "datadog", "new-relic"]
            },
            "hrm_reasoning_agent": {
                "domains": ["general", "reasoning", "planning", "analysis", "decision"],
                "skills": ["problem-solving", "critical-thinking", "research", "synthesis"],
                "tools": ["general-purpose", "multi-domain", "adaptive"]
            }
        }
        
        # Technology-specific modifiers
        self.tech_modifiers = {
            "docker": ["containerization", "dockerfile", "docker-compose", "registry"],
            "kubernetes": ["k8s", "pods", "services", "deployments", "helm"],
            "react": ["jsx", "hooks", "components", "state", "props"],
            "vue": ["composition-api", "templates", "vuex", "pinia"],
            "angular": ["typescript", "rxjs", "dependency-injection", "modules"],
            "firebase": ["firestore", "auth", "functions", "hosting", "storage"],
            "aws": ["lambda", "s3", "ec2", "rds", "cloudformation"],
            "azure": ["functions", "blob", "cosmos", "app-service"],
            "nextjs": ["ssr", "ssg", "api-routes", "vercel"],
            "fastapi": ["pydantic", "async", "openapi", "uvicorn"],
            "django": ["orm", "admin", "middleware", "templates"],
            "postgresql": ["sql", "indexes", "triggers", "procedures"],
            "mongodb": ["nosql", "aggregation", "sharding", "replication"],
            "redis": ["caching", "pub-sub", "sessions", "queues"],
            "graphql": ["schema", "resolvers", "mutations", "subscriptions"],
            "terraform": ["iac", "providers", "modules", "state"],
            "velo": ["wix", "corvid", "frontend", "backend", "database"]
        }
        
        # Dynamic mapping cache
        self.dynamic_mappings: Dict[str, str] = {}
        
    def analyze_agent_request(self, agent_type: str, description: str = "", 
                             context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze the requested agent type and intelligently map it to a base agent
        with specialized context
        
        Returns:
            Tuple[str, Dict]: (base_agent_type, specialized_context)
        """
        agent_type_lower = agent_type.lower()
        desc_lower = description.lower() if description else ""
        
        # Check if we've already mapped this agent
        if agent_type_lower in self.dynamic_mappings:
            base_agent = self.dynamic_mappings[agent_type_lower]
            return base_agent, self._create_specialized_context(agent_type, base_agent, description, context)
        
        # Extract components from the agent type name
        components = self._extract_components(agent_type_lower)
        
        # Find the best matching base agent
        base_agent, confidence = self._find_best_match(components, desc_lower)
        
        # Cache the mapping for future use
        self.dynamic_mappings[agent_type_lower] = base_agent
        
        # Create specialized context for this agent
        specialized_context = self._create_specialized_context(
            agent_type, base_agent, description, context
        )
        
        logger.info(f"🤖 Dynamic Agent Creation: '{agent_type}' → '{base_agent}' (confidence: {confidence:.2f})")
        logger.info(f"   Specialized for: {', '.join(components[:3])}")
        
        return base_agent, specialized_context
    
    def _extract_components(self, agent_type: str) -> List[str]:
        """Extract meaningful components from the agent type name"""
        # Split by common separators
        parts = re.split(r'[-_\s]+', agent_type)
        
        # Add the full type as well
        components = [agent_type] + parts
        
        # Extract technology keywords
        for tech, keywords in self.tech_modifiers.items():
            if tech in agent_type:
                components.extend(keywords)
        
        # Remove common suffixes/prefixes
        cleaned = []
        for part in components:
            cleaned_part = part.replace('engineer', '').replace('specialist', '')
            cleaned_part = cleaned_part.replace('expert', '').replace('agent', '')
            if cleaned_part and cleaned_part not in cleaned:
                cleaned.append(cleaned_part)
        
        return cleaned
    
    def _find_best_match(self, components: List[str], description: str) -> Tuple[str, float]:
        """Find the best matching base agent for the given components"""
        scores = {}
        
        for base_agent, config in self.base_agents.items():
            score = 0.0
            
            # Check domains
            for domain in config["domains"]:
                if any(domain in comp or comp in domain for comp in components):
                    score += 2.0
                if domain in description:
                    score += 1.0
            
            # Check skills
            for skill in config["skills"]:
                if any(skill in comp or comp in skill for comp in components):
                    score += 1.5
                if skill in description:
                    score += 0.5
            
            # Check tools
            for tool in config["tools"]:
                if any(tool in comp or comp in tool for comp in components):
                    score += 1.0
                if tool in description:
                    score += 0.5
            
            scores[base_agent] = score
        
        # Find the best match
        if scores:
            best_agent = max(scores, key=scores.get)
            max_score = scores[best_agent]
            
            # Calculate confidence (0-1)
            confidence = min(max_score / 10.0, 1.0)
            
            # If confidence is too low, use general purpose agent
            if confidence < 0.2:
                return "hrm_reasoning_agent", 0.5
            
            return best_agent, confidence
        
        # Default fallback
        return "hrm_reasoning_agent", 0.3
    
    def _create_specialized_context(self, agent_type: str, base_agent: str,
                                   description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create specialized context for the dynamically mapped agent"""
        specialized_context = {
            "original_agent_type": agent_type,
            "base_agent": base_agent,
            "specialization": self._extract_specialization(agent_type),
            "dynamic_creation": True,
            "creation_reason": f"Dynamically created from '{agent_type}' request"
        }
        
        # Add technology-specific instructions
        tech_instructions = []
        for tech, keywords in self.tech_modifiers.items():
            if tech in agent_type.lower() or tech in description.lower():
                tech_instructions.append(f"Focus on {tech} best practices and patterns")
                specialized_context[f"{tech}_specialist"] = True
        
        if tech_instructions:
            specialized_context["specialized_instructions"] = tech_instructions
        
        # Merge with provided context
        if context:
            specialized_context.update(context)
        
        # Add project-specific hints
        if "docker" in agent_type.lower():
            specialized_context["container_focus"] = True
            specialized_context["tools_priority"] = ["docker", "docker-compose", "dockerfile"]
        
        if "velo" in agent_type.lower():
            specialized_context["platform"] = "Wix Velo"
            specialized_context["velo_specialist"] = True
            specialized_context["focus_areas"] = ["Wix APIs", "Corvid", "Velo backend"]
        
        return specialized_context
    
    def _extract_specialization(self, agent_type: str) -> str:
        """Extract the specialization from the agent type"""
        parts = re.split(r'[-_\s]+', agent_type.lower())
        
        # Remove generic terms
        specialized = [p for p in parts if p not in 
                      ['agent', 'engineer', 'specialist', 'expert', 'coder', 'developer']]
        
        if specialized:
            return " ".join(specialized).title()
        return "General Purpose"
    
    def get_agent_capabilities(self, agent_type: str) -> Dict[str, Any]:
        """Get the capabilities of a dynamically created agent"""
        base_agent, context = self.analyze_agent_request(agent_type)
        
        capabilities = self.base_agents.get(base_agent, {}).copy()
        capabilities["specialized_context"] = context
        capabilities["dynamic"] = True
        
        return capabilities
    
    def register_project_agents(self, project_path: str, agents: List[Dict[str, Any]]):
        """Register project-specific agents from the Archon Project Agent Factory"""
        for agent in agents:
            agent_id = agent.get("id", "").lower()
            if agent_id and agent_id not in self.dynamic_mappings:
                # Map to the most appropriate base agent
                base_agent, _ = self._find_best_match(
                    [agent_id] + agent.get("skills", []),
                    agent.get("description", "")
                )
                self.dynamic_mappings[agent_id] = base_agent
                logger.info(f"📝 Registered project agent: {agent_id} → {base_agent}")

# Global instance
_dynamic_factory = None

def get_dynamic_agent_factory() -> DynamicAgentFactory:
    """Get the global dynamic agent factory instance"""
    global _dynamic_factory
    if _dynamic_factory is None:
        _dynamic_factory = DynamicAgentFactory()
    return _dynamic_factory