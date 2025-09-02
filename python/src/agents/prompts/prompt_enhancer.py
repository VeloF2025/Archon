#!/usr/bin/env python3
"""
Prompt Enhancement System for Archon+ Phase 3
Provides bidirectional prompt enhancement with context injection and validation
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Tuple
import httpx
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancementDirection(Enum):
    TO_SUB = "to-sub"      # User prompt → Agent prompt
    FROM_SUB = "from-sub"  # Agent response → User response

class EnhancementLevel(Enum):
    BASIC = "basic"                    # Minimal enhancement
    ENHANCED = "enhanced"              # Standard enhancement
    COMPREHENSIVE = "comprehensive"    # Maximum enhancement

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class PromptContext:
    """Context information for prompt enhancement"""
    agent_role: Optional[str] = None
    project_type: Optional[str] = None
    task_complexity: TaskComplexity = TaskComplexity.MEDIUM
    domain_knowledge: List[str] = field(default_factory=list)
    previous_interactions: List[str] = field(default_factory=list)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptEnhancementRequest:
    """Request for prompt enhancement"""
    original_prompt: str
    direction: EnhancementDirection
    context: PromptContext
    enhancement_level: EnhancementLevel = EnhancementLevel.ENHANCED
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

@dataclass
class ContextInjection:
    """Individual context injection"""
    type: str  # knowledge, pattern, template, validation
    content: str
    confidence: float  # 0.0-1.0
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptEnhancementResult:
    """Result of prompt enhancement"""
    enhanced_prompt: str
    enhancement_score: float  # 0.0-1.0 overall confidence
    context_injections: List[ContextInjection] = field(default_factory=list)
    validation_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    request_id: str = ""

class PromptTemplate:
    """Base class for prompt templates"""
    
    def __init__(self, name: str, template: str, variables: List[str]):
        self.name = name
        self.template = template
        self.variables = variables
    
    def apply(self, context: Dict[str, Any]) -> str:
        """Apply template with context"""
        try:
            return self.template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing template variable {e} for {self.name}")
            return self.template

class PRPTemplate(PromptTemplate):
    """Prompt Refinement Protocol template"""
    
    @classmethod
    def create_agent_enhancement_template(cls) -> 'PRPTemplate':
        """Create template for enhancing prompts sent to agents"""
        template = """ENHANCED PROMPT FOR {agent_role}:

PRIMARY OBJECTIVE: {original_objective}

CONTEXT:
- Project Type: {project_type}
- Task Complexity: {task_complexity}
- Quality Requirements: {quality_requirements}

SPECIFIC INSTRUCTIONS:
{enhanced_instructions}

CONSTRAINTS:
- Follow Archon quality standards (NLNH protocol)
- Maintain zero-tolerance policies
- Provide specific, actionable outputs
- Include confidence levels for uncertainty

VALIDATION CRITERIA:
{validation_criteria}

OUTPUT FORMAT:
{output_format}"""
        
        return cls(
            name="agent_enhancement",
            template=template,
            variables=[
                "agent_role", "original_objective", "project_type",
                "task_complexity", "quality_requirements", "enhanced_instructions",
                "validation_criteria", "output_format"
            ]
        )
    
    @classmethod
    def create_user_enhancement_template(cls) -> 'PRPTemplate':
        """Create template for enhancing responses to users"""
        template = """ENHANCED RESPONSE:

{enhanced_content}

SUMMARY:
{summary}

TECHNICAL DETAILS:
{technical_details}

NEXT STEPS:
{next_steps}

CONFIDENCE: {confidence_level}
VALIDATION STATUS: {validation_status}"""
        
        return cls(
            name="user_enhancement",
            template=template,
            variables=[
                "enhanced_content", "summary", "technical_details",
                "next_steps", "confidence_level", "validation_status"
            ]
        )

class ContextEnricher:
    """Enriches prompts with project and domain context"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")
    
    async def get_project_context(self, project_type: str) -> Dict[str, Any]:
        """Get project-specific context from knowledge base"""
        if not self.supabase_url or not self.supabase_key:
            return {"patterns": [], "best_practices": [], "common_issues": []}
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.supabase_url}/rest/v1/rpc/search_project_context",
                    headers={
                        "Authorization": f"Bearer {self.supabase_key}",
                        "apikey": self.supabase_key,
                        "Content-Type": "application/json"
                    },
                    json={"project_type": project_type}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to fetch project context: {response.status_code}")
                    return {"patterns": [], "best_practices": [], "common_issues": []}
        
        except Exception as e:
            logger.error(f"Error fetching project context: {e}")
            return {"patterns": [], "best_practices": [], "common_issues": []}
    
    async def get_domain_knowledge(self, keywords: List[str]) -> List[str]:
        """Get relevant domain knowledge from knowledge base"""
        if not keywords or not self.supabase_url or not self.supabase_key:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.supabase_url}/rest/v1/rpc/search_domain_knowledge",
                    headers={
                        "Authorization": f"Bearer {self.supabase_key}",
                        "apikey": self.supabase_key,
                        "Content-Type": "application/json"
                    },
                    json={"keywords": keywords, "limit": 10}
                )
                
                if response.status_code == 200:
                    results = response.json()
                    return [item.get("content", "") for item in results]
                else:
                    logger.warning(f"Failed to fetch domain knowledge: {response.status_code}")
                    return []
        
        except Exception as e:
            logger.error(f"Error fetching domain knowledge: {e}")
            return []

class ValidationEngine:
    """Validates enhanced prompts for quality and safety"""
    
    def __init__(self):
        self.quality_patterns = [
            r"(?i)\b(implement|create|build|develop)\b",  # Action words
            r"(?i)\b(test|validate|verify|check)\b",      # Quality words
            r"(?i)\b(specific|detailed|comprehensive)\b", # Precision words
        ]
        
        self.warning_patterns = [
            r"(?i)\b(maybe|might|possibly|unclear)\b",    # Uncertainty
            r"(?i)\b(hack|quick fix|temporary)\b",        # Quality issues
            r"(?i)\b(ignore|skip|bypass)\b",              # Safety issues
        ]
    
    def validate_enhancement(self, 
                           original: str, 
                           enhanced: str,
                           context: PromptContext) -> Tuple[float, List[str]]:
        """Validate enhancement quality and safety"""
        flags = []
        confidence = 1.0
        
        # Length validation
        if len(enhanced) < len(original) * 0.8:
            flags.append("enhancement_too_short")
            confidence -= 0.2
        
        if len(enhanced) > len(original) * 3:
            flags.append("enhancement_too_verbose")
            confidence -= 0.1
        
        # Quality pattern matching
        import re
        quality_matches = sum(1 for pattern in self.quality_patterns 
                            if re.search(pattern, enhanced))
        
        warning_matches = sum(1 for pattern in self.warning_patterns 
                            if re.search(pattern, enhanced))
        
        if quality_matches == 0:
            flags.append("lacks_action_words")
            confidence -= 0.3
        
        if warning_matches > 0:
            flags.append("contains_uncertainty_language")
            confidence -= 0.2 * warning_matches
        
        # Context relevance check
        if context.agent_role and context.agent_role not in enhanced.lower():
            flags.append("missing_role_context")
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, flags

class PromptEnhancer:
    """Main prompt enhancement engine"""
    
    def __init__(self):
        self.context_enricher = ContextEnricher()
        self.validation_engine = ValidationEngine()
        self.templates = {
            "agent_enhancement": PRPTemplate.create_agent_enhancement_template(),
            "user_enhancement": PRPTemplate.create_user_enhancement_template()
        }
        self.enhancement_cache: Dict[str, PromptEnhancementResult] = {}
        
    async def enhance_prompt(self, request: PromptEnhancementRequest) -> PromptEnhancementResult:
        """Main method to enhance a prompt"""
        start_time = time.time()
        
        try:
            logger.info(f"Enhancing prompt {request.request_id} - {request.direction.value}")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.enhancement_cache:
                cached_result = self.enhancement_cache[cache_key]
                cached_result.metadata["cache_hit"] = True
                return cached_result
            
            # Gather context
            context_injections = await self._gather_context(request)
            
            # Apply enhancement based on direction
            if request.direction == EnhancementDirection.TO_SUB:
                enhanced_prompt = await self._enhance_for_agent(request, context_injections)
            else:
                enhanced_prompt = await self._enhance_for_user(request, context_injections)
            
            # Validate enhancement
            confidence, validation_flags = self.validation_engine.validate_enhancement(
                request.original_prompt, enhanced_prompt, request.context
            )
            
            # Create result
            result = PromptEnhancementResult(
                enhanced_prompt=enhanced_prompt,
                enhancement_score=confidence,
                context_injections=context_injections,
                validation_flags=validation_flags,
                metadata={
                    "direction": request.direction.value,
                    "enhancement_level": request.enhancement_level.value,
                    "original_length": len(request.original_prompt),
                    "enhanced_length": len(enhanced_prompt),
                    "cache_hit": False
                },
                processing_time=time.time() - start_time,
                request_id=request.request_id
            )
            
            # Cache result
            self.enhancement_cache[cache_key] = result
            self._cleanup_cache()
            
            logger.info(f"Enhanced prompt {request.request_id} - Score: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing prompt {request.request_id}: {e}")
            return PromptEnhancementResult(
                enhanced_prompt=request.original_prompt,  # Fallback to original
                enhancement_score=0.0,
                validation_flags=["enhancement_failed"],
                metadata={"error": str(e)},
                processing_time=time.time() - start_time,
                request_id=request.request_id
            )
    
    async def _gather_context(self, request: PromptEnhancementRequest) -> List[ContextInjection]:
        """Gather relevant context for enhancement"""
        injections = []
        
        try:
            # Get project context
            if request.context.project_type:
                project_context = await self.context_enricher.get_project_context(
                    request.context.project_type
                )
                
                for pattern in project_context.get("patterns", []):
                    injections.append(ContextInjection(
                        type="pattern",
                        content=pattern,
                        confidence=0.8,
                        source="project_knowledge"
                    ))
            
            # Get domain knowledge
            if request.context.domain_knowledge:
                domain_knowledge = await self.context_enricher.get_domain_knowledge(
                    request.context.domain_knowledge
                )
                
                for knowledge in domain_knowledge:
                    injections.append(ContextInjection(
                        type="knowledge",
                        content=knowledge,
                        confidence=0.7,
                        source="domain_knowledge"
                    ))
            
            # Add quality requirements as context
            if request.context.quality_requirements:
                injections.append(ContextInjection(
                    type="validation",
                    content=f"Quality requirements: {json.dumps(request.context.quality_requirements)}",
                    confidence=0.9,
                    source="quality_standards"
                ))
            
        except Exception as e:
            logger.error(f"Error gathering context: {e}")
            # Add fallback context
            injections.append(ContextInjection(
                type="validation",
                content="Follow Archon quality standards and NLNH protocol",
                confidence=1.0,
                source="default_standards"
            ))
        
        return injections
    
    async def _enhance_for_agent(self, 
                                request: PromptEnhancementRequest,
                                context_injections: List[ContextInjection]) -> str:
        """Enhance prompt for agent consumption"""
        
        template = self.templates["agent_enhancement"]
        
        # Extract context information
        context_content = "\n".join([
            f"- {inj.type.title()}: {inj.content}" 
            for inj in context_injections[:5]  # Limit context
        ])
        
        # Build enhancement context
        template_context = {
            "agent_role": request.context.agent_role or "specialized_agent",
            "original_objective": request.original_prompt,
            "project_type": request.context.project_type or "unknown",
            "task_complexity": request.context.task_complexity.value,
            "quality_requirements": json.dumps(request.context.quality_requirements),
            "enhanced_instructions": self._generate_enhanced_instructions(request),
            "validation_criteria": self._generate_validation_criteria(request),
            "output_format": self._generate_output_format(request)
        }
        
        enhanced_prompt = template.apply(template_context)
        
        # Add context if comprehensive enhancement
        if request.enhancement_level == EnhancementLevel.COMPREHENSIVE:
            enhanced_prompt += f"\n\nADDITIONAL CONTEXT:\n{context_content}"
        
        return enhanced_prompt
    
    async def _enhance_for_user(self,
                              request: PromptEnhancementRequest,
                              context_injections: List[ContextInjection]) -> str:
        """Enhance response for user consumption"""
        
        template = self.templates["user_enhancement"]
        
        # For user-facing enhancement, we structure the original prompt/response
        template_context = {
            "enhanced_content": self._structure_user_content(request.original_prompt),
            "summary": self._generate_summary(request.original_prompt),
            "technical_details": self._extract_technical_details(request.original_prompt),
            "next_steps": self._suggest_next_steps(request),
            "confidence_level": "High" if len(context_injections) > 2 else "Medium",
            "validation_status": "Enhanced with project context"
        }
        
        return template.apply(template_context)
    
    def _generate_enhanced_instructions(self, request: PromptEnhancementRequest) -> str:
        """Generate enhanced instructions based on context"""
        base_instructions = request.original_prompt
        
        enhancements = []
        
        if request.context.task_complexity == TaskComplexity.COMPLEX:
            enhancements.append("Break down the task into smaller, manageable steps")
            enhancements.append("Provide progress updates at each major milestone")
        
        if request.context.agent_role:
            enhancements.append(f"Apply {request.context.agent_role} best practices and patterns")
        
        enhancements.append("Include confidence levels for any uncertain recommendations")
        enhancements.append("Validate outputs against Archon quality standards")
        
        enhanced = base_instructions
        if enhancements:
            enhanced += "\n\nENHANCED REQUIREMENTS:\n" + "\n".join(f"- {e}" for e in enhancements)
        
        return enhanced
    
    def _generate_validation_criteria(self, request: PromptEnhancementRequest) -> str:
        """Generate validation criteria for the task"""
        criteria = [
            "Zero TypeScript/ESLint errors",
            "Proper error handling implementation",
            "NLNH protocol compliance (honest reporting)",
            "Code follows established patterns"
        ]
        
        if request.context.quality_requirements:
            criteria.extend([
                f"Meets requirement: {req}" 
                for req in request.context.quality_requirements.keys()
            ])
        
        return "\n".join(f"- {c}" for c in criteria)
    
    def _generate_output_format(self, request: PromptEnhancementRequest) -> str:
        """Generate expected output format"""
        if request.context.agent_role and "code" in request.context.agent_role.lower():
            return """Provide:
1. Implementation code with full error handling
2. Test cases covering edge cases
3. Documentation and comments
4. Integration instructions
5. Confidence assessment (0.0-1.0)"""
        
        return """Provide:
1. Clear, actionable response
2. Specific recommendations with rationale
3. Potential risks or limitations
4. Next steps or follow-up actions
5. Confidence level and validation status"""
    
    def _structure_user_content(self, content: str) -> str:
        """Structure content for user readability"""
        # Simple structuring - in a real implementation, this would be more sophisticated
        lines = content.split('\n')
        structured = []
        
        for line in lines:
            line = line.strip()
            if line:
                if line.endswith(':'):
                    structured.append(f"\n**{line}**")
                elif line.startswith('-') or line.startswith('*'):
                    structured.append(f"  {line}")
                else:
                    structured.append(line)
        
        return '\n'.join(structured)
    
    def _generate_summary(self, content: str) -> str:
        """Generate a summary of the content"""
        words = content.split()
        if len(words) <= 20:
            return content
        
        # Simple summary - take first 20 words and add indication if truncated
        summary = ' '.join(words[:20])
        return f"{summary}..." if len(words) > 20 else summary
    
    def _extract_technical_details(self, content: str) -> str:
        """Extract technical details from content"""
        # Look for technical keywords and patterns
        technical_keywords = ['function', 'class', 'method', 'api', 'database', 'server', 'client']
        
        lines = content.split('\n')
        technical_lines = [
            line for line in lines
            if any(keyword in line.lower() for keyword in technical_keywords)
        ]
        
        if technical_lines:
            return '\n'.join(f"- {line.strip()}" for line in technical_lines[:3])
        
        return "No specific technical details identified"
    
    def _suggest_next_steps(self, request: PromptEnhancementRequest) -> str:
        """Suggest next steps based on context"""
        steps = []
        
        if request.context.task_complexity == TaskComplexity.COMPLEX:
            steps.append("Break down into smaller tasks")
            steps.append("Validate each component individually")
        
        if request.context.agent_role:
            steps.append(f"Consult {request.context.agent_role} best practices")
        
        steps.append("Run validation tests")
        steps.append("Review against quality standards")
        
        return '\n'.join(f"{i+1}. {step}" for i, step in enumerate(steps))
    
    def _generate_cache_key(self, request: PromptEnhancementRequest) -> str:
        """Generate cache key for the request"""
        import hashlib
        
        cache_data = {
            "prompt": request.original_prompt[:200],  # First 200 chars
            "direction": request.direction.value,
            "level": request.enhancement_level.value,
            "role": request.context.agent_role,
            "project": request.context.project_type,
            "complexity": request.context.task_complexity.value
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        if len(self.enhancement_cache) > 100:  # Keep cache size reasonable
            # Remove oldest 20 entries
            sorted_items = sorted(
                self.enhancement_cache.items(),
                key=lambda x: x[1].metadata.get("timestamp", 0)
            )
            
            for key, _ in sorted_items[:20]:
                del self.enhancement_cache[key]
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics"""
        if not self.enhancement_cache:
            return {"total_enhancements": 0}
        
        results = list(self.enhancement_cache.values())
        
        avg_score = sum(r.enhancement_score for r in results) / len(results)
        avg_processing_time = sum(r.processing_time for r in results) / len(results)
        
        direction_counts = {}
        for result in results:
            direction = result.metadata.get("direction", "unknown")
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        return {
            "total_enhancements": len(results),
            "average_score": avg_score,
            "average_processing_time": avg_processing_time,
            "direction_distribution": direction_counts,
            "cache_hits": sum(1 for r in results if r.metadata.get("cache_hit", False)),
            "recent_enhancements": results[-5:]  # Last 5 enhancements
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_enhancer():
        enhancer = PromptEnhancer()
        
        # Test agent-directed enhancement
        context = PromptContext(
            agent_role="code_implementer",
            project_type="react_typescript",
            task_complexity=TaskComplexity.MEDIUM,
            quality_requirements={"test_coverage": 0.9, "type_safety": True}
        )
        
        request = PromptEnhancementRequest(
            original_prompt="Create a login component with form validation",
            direction=EnhancementDirection.TO_SUB,
            context=context,
            enhancement_level=EnhancementLevel.ENHANCED
        )
        
        result = await enhancer.enhance_prompt(request)
        
        print("=== AGENT-DIRECTED ENHANCEMENT ===")
        print(f"Original: {request.original_prompt}")
        print(f"\nEnhanced:\n{result.enhanced_prompt}")
        print(f"\nScore: {result.enhancement_score:.2f}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Validation Flags: {result.validation_flags}")
        
        # Test user-directed enhancement
        user_request = PromptEnhancementRequest(
            original_prompt="The login component has been created with TypeScript interfaces, form validation using react-hook-form, and comprehensive error handling. All tests are passing.",
            direction=EnhancementDirection.FROM_SUB,
            context=context,
            enhancement_level=EnhancementLevel.BASIC
        )
        
        user_result = await enhancer.enhance_prompt(user_request)
        
        print("\n\n=== USER-DIRECTED ENHANCEMENT ===")
        print(f"Original: {user_request.original_prompt}")
        print(f"\nEnhanced:\n{user_result.enhanced_prompt}")
        print(f"\nScore: {user_result.enhancement_score:.2f}")
        
        # Print stats
        stats = enhancer.get_enhancement_stats()
        print(f"\n=== ENHANCEMENT STATS ===")
        print(json.dumps(stats, indent=2, default=str))
    
    asyncio.run(test_enhancer())