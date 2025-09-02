"""
Phase 6 Specialized Agents Implementation
Provides all 22+ specialized agents for parallel execution
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .base_agent import BaseAgent
from .rag_agent import RagAgent

logger = logging.getLogger(__name__)

class SpecializedAgentDependencies(BaseModel):
    """Extended dependencies for specialized agents"""
    # Base fields from ArchonDependencies
    request_id: str | None = None
    user_id: str | None = None
    trace_id: str | None = None
    # Additional fields for specialized agents
    agent_role: str = Field(default="generic")
    task_description: str = Field(default="")
    context: Dict[str, Any] = Field(default_factory=dict)
    tool_permissions: List[str] = Field(default_factory=list)

# Python Backend Coder Agent  
class PythonBackendAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Python backend development specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="PythonBackendAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
        You are a Python backend development specialist.
        Focus on:
        - FastAPI and async programming
        - Database design with SQLAlchemy
        - API development and integration
        - Error handling and validation
        - Performance optimization
        """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Python backend agent error: {e}")
            return f"Error: {e}"

# TypeScript Frontend Agent  
class TypeScriptFrontendAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """TypeScript/React frontend development specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="TypeScriptFrontendAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
        You are a TypeScript/React frontend development specialist.
        Focus on:
        - React components and hooks
        - TypeScript type safety
        - State management
        - UI/UX best practices
        - Performance optimization
        """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"TypeScript frontend agent error: {e}")
            return f"Error: {e}"

# Security Auditor Agent
class SecurityAuditorAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Security analysis and vulnerability detection specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="SecurityAuditorAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
        You are a security auditing specialist.
        Focus on:
        - OWASP Top 10 vulnerabilities
        - Authentication and authorization
        - Input validation and sanitization
        - Encryption and data protection
        - Security best practices
        """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Security auditor agent error: {e}")
            return f"Error: {e}"

# Test Generator Agent
class TestGeneratorAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Test creation and coverage specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="TestGeneratorAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a test generation specialist.
            Focus on:
            - Unit test creation
            - Integration testing
            - Edge case identification
            - Test coverage optimization
            - Test-driven development
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"TestGeneratorAgent error: {e}")
            return f"Error: {e}"
class CodeReviewerAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Code quality and review specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="CodeReviewerAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a code review specialist.
            Focus on:
            - Code quality and readability
            - Design patterns and best practices
            - Performance issues
            - Security vulnerabilities
            - Maintainability
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"CodeReviewerAgent error: {e}")
            return f"Error: {e}"
class DocumentationWriterAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Documentation creation and maintenance specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="DocumentationWriterAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a documentation specialist.
            Focus on:
            - API documentation
            - User guides and tutorials
            - Code comments and docstrings
            - Architecture documentation
            - README files
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"DocumentationWriterAgent error: {e}")
            return f"Error: {e}"
class SystemArchitectAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """System design and architecture specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="SystemArchitectAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a system architecture specialist.
            Focus on:
            - System design and patterns
            - Microservices architecture
            - API design
            - Scalability and performance
            - Technology selection
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"SystemArchitectAgent error: {e}")
            return f"Error: {e}"
class DatabaseDesignerAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Database design and optimization specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="DatabaseDesignerAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a database design specialist.
            Focus on:
            - Schema design and normalization
            - Query optimization
            - Index strategy
            - Data migration
            - Performance tuning
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"DatabaseDesignerAgent error: {e}")
            return f"Error: {e}"
class DevOpsEngineerAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """DevOps and infrastructure specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="DevOpsEngineerAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a DevOps specialist.
            Focus on:
            - CI/CD pipelines
            - Docker and Kubernetes
            - Infrastructure as code
            - Monitoring and logging
            - Deployment automation
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"DevOpsEngineerAgent error: {e}")
            return f"Error: {e}"
class PerformanceOptimizerAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Performance analysis and optimization specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="PerformanceOptimizerAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a performance optimization specialist.
            Focus on:
            - Performance profiling
            - Code optimization
            - Caching strategies
            - Load balancing
            - Resource utilization
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"PerformanceOptimizerAgent error: {e}")
            return f"Error: {e}"
class APIIntegratorAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """API integration and design specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="APIIntegratorAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are an API integration specialist.
            Focus on:
            - RESTful API design
            - GraphQL implementation
            - API documentation
            - Authentication and security
            - Third-party integrations
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"APIIntegratorAgent error: {e}")
            return f"Error: {e}"
class UIUXDesignerAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """UI/UX design and optimization specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="UIUXDesignerAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a UI/UX design specialist.
            Focus on:
            - User interface design
            - User experience optimization
            - Accessibility standards
            - Responsive design
            - Design systems
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"UIUXDesignerAgent error: {e}")
            return f"Error: {e}"
class RefactoringSpecialistAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Code refactoring and improvement specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="RefactoringSpecialistAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a code refactoring specialist.
            Focus on:
            - Code smell detection
            - Design pattern application
            - Code simplification
            - Technical debt reduction
            - Maintainability improvement
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"RefactoringSpecialistAgent error: {e}")
            return f"Error: {e}"
class TechnicalWriterAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Technical documentation and communication specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="TechnicalWriterAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a technical writing specialist.
            Focus on:
            - Technical documentation
            - API reference guides
            - Tutorial creation
            - Knowledge base articles
            - Release notes
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"TechnicalWriterAgent error: {e}")
            return f"Error: {e}"
class IntegrationTesterAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Integration testing and validation specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="IntegrationTesterAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are an integration testing specialist.
            Focus on:
            - End-to-end testing
            - API testing
            - Service integration
            - Test automation
            - Regression testing
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"IntegrationTesterAgent error: {e}")
            return f"Error: {e}"
class DeploymentCoordinatorAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Deployment coordination and management specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="DeploymentCoordinatorAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a deployment coordination specialist.
            Focus on:
            - Release management
            - Deployment strategies
            - Rollback procedures
            - Environment management
            - Deployment automation
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"DeploymentCoordinatorAgent error: {e}")
            return f"Error: {e}"
class MonitoringAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """System monitoring and alerting specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="MonitoringAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a monitoring specialist.
            Focus on:
            - Application monitoring
            - Log analysis
            - Alert configuration
            - Performance metrics
            - Incident response
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"MonitoringAgent error: {e}")
            return f"Error: {e}"
class DataAnalystAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """Data analysis and insights specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="DataAnalystAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a data analysis specialist.
            Focus on:
            - Data analysis and visualization
            - Metrics and KPIs
            - Trend analysis
            - Report generation
            - Data-driven insights
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"DataAnalystAgent error: {e}")
            return f"Error: {e}"
class HRMReasoningAgent(BaseAgent[SpecializedAgentDependencies, str]):
    """High-level reasoning and decision support specialist"""
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="HRMReasoningAgent", model=model)
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent with MANIFEST compliance"""
        return Agent(
            self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """
            You are a high-level reasoning specialist.
            Focus on:
            - Strategic decision making
            - Complex problem solving
            - Trade-off analysis
            - Risk assessment
            - Solution synthesis
            """
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self._agent.run(prompt, deps=deps)
            # Use correct attribute for PydanticAI 0.8.x
            if hasattr(result, 'output'):
                return result.output
            elif hasattr(result, 'data'):
                return result.data  
            else:
                return str(result)
        except Exception as e:
            logger.error(f"HRMReasoningAgent error: {e}")
            return f"Error: {e}"
# Agent Registry
SPECIALIZED_AGENTS = {
    "python_backend_coder": PythonBackendAgent,
    "typescript_frontend_agent": TypeScriptFrontendAgent, 
    "security_auditor": SecurityAuditorAgent,
    "test_generator": TestGeneratorAgent,
    "code_reviewer": CodeReviewerAgent,
    "documentation_writer": DocumentationWriterAgent,
    "system_architect": SystemArchitectAgent,
    "database_designer": DatabaseDesignerAgent,
    "devops_engineer": DevOpsEngineerAgent,
    "performance_optimizer": PerformanceOptimizerAgent,
    "api_integrator": APIIntegratorAgent,
    "ui_ux_designer": UIUXDesignerAgent,
    "refactoring_specialist": RefactoringSpecialistAgent,
    "technical_writer": TechnicalWriterAgent,
    "integration_tester": IntegrationTesterAgent,
    "deployment_coordinator": DeploymentCoordinatorAgent,
    "monitoring_agent": MonitoringAgent,
    "data_analyst": DataAnalystAgent,
    "hrm_reasoning_agent": HRMReasoningAgent,
}

def get_specialized_agent(agent_role: str, model: str = "openai:gpt-4o-mini") -> Optional[BaseAgent]:
    """Get a specialized agent by role"""
    agent_class = SPECIALIZED_AGENTS.get(agent_role)
    if agent_class:
        return agent_class(model=model)
    return None

def list_specialized_agents() -> List[str]:
    """List all available specialized agent roles"""
    return list(SPECIALIZED_AGENTS.keys())