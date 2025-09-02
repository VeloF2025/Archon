"""
LLM client for External Validator
Supports DeepSeek, OpenAI, and Claude Code fallback
"""

import asyncio
import json
from typing import Dict, Any, Optional
import logging
import httpx
from openai import AsyncOpenAI

from .config import ValidatorConfig
from .claude_fallback import ClaudeFallbackValidator

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with external LLMs"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self.claude_fallback: Optional[ClaudeFallbackValidator] = None
        self.using_fallback = False
        self.total_tokens = 0
        self.total_requests = 0
    
    async def initialize(self):
        """Initialize LLM client with fallback support"""
        
        llm_config = self.config.get_llm_client_config()
        
        # Try to initialize external LLM first
        if llm_config.get("api_key"):
            try:
                provider = self.config.llm_config.provider
                
                if provider in ["deepseek", "groq", "mistral"]:
                    # These providers use OpenAI-compatible APIs
                    base_urls = {
                        "deepseek": "https://api.deepseek.com",
                        "groq": "https://api.groq.com/openai/v1",
                        "mistral": "https://api.mistral.ai/v1"
                    }
                    self.client = AsyncOpenAI(
                        api_key=llm_config["api_key"],
                        base_url=llm_config.get("base_url", base_urls.get(provider)),
                        default_headers={"Accept": "application/json"}
                    )
                    logger.info(f"Initialized {provider.title()} client")
                    self.using_fallback = False
                    
                elif provider == "openai":
                    self.client = AsyncOpenAI(
                        api_key=llm_config["api_key"]
                    )
                    logger.info("Initialized OpenAI client")
                    self.using_fallback = False
                    
                elif provider == "anthropic":
                    # Note: Would need anthropic SDK installed
                    logger.warning("Anthropic provider selected but SDK not installed, falling back")
                    self.client = None
                    
                elif provider == "google":
                    # Note: Would need google.generativeai SDK installed
                    logger.warning("Google provider selected but SDK not installed, falling back")
                    self.client = None
                    
                else:
                    logger.warning(f"Unknown provider: {provider}, attempting OpenAI-compatible API")
                    self.client = AsyncOpenAI(
                        api_key=llm_config["api_key"],
                        base_url=llm_config.get("base_url")
                    )
                    self.using_fallback = False
                    
            except Exception as e:
                logger.error(f"Failed to initialize external LLM client: {e}")
                self.client = None
        
        # If no external client available, use Claude fallback
        if not self.client:
            logger.warning("No external API key configured or initialization failed")
            logger.info("Initializing Claude Code fallback with strict guardrails")
            self.claude_fallback = ClaudeFallbackValidator()
            self.using_fallback = True
            logger.info("Claude fallback initialized with anti-bias guardrails:"
                       f"\n  - Minimum issues required: {self.claude_fallback.MIN_ISSUES_REQUIRED}"
                       f"\n  - Max confidence for self-work: {self.claude_fallback.MAX_CONFIDENCE_SELF_WORK}"
                       f"\n  - Skepticism level: {self.claude_fallback.SKEPTICISM_LEVEL}")
    
    async def cleanup(self):
        """Cleanup client resources"""
        
        if self.client:
            await self.client.close()
    
    async def check_connection(self) -> bool:
        """Check if LLM is accessible"""
        
        # Claude fallback is always available
        if self.using_fallback and self.claude_fallback:
            return True
        
        if not self.client:
            return False
        
        try:
            # Try a minimal completion
            response = await self.client.chat.completions.create(
                model=self.config.llm_config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0
            )
            return True
            
        except Exception as e:
            logger.error(f"LLM connection check failed: {e}")
            # Try to fall back to Claude
            if not self.using_fallback:
                logger.info("Switching to Claude fallback due to connection failure")
                self.claude_fallback = ClaudeFallbackValidator()
                self.using_fallback = True
                self.client = None
                return True  # Fallback is available
            return False
    
    async def validate_with_llm(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        validation_type: str = "general"
    ) -> Dict[str, Any]:
        """Perform validation using LLM or fallback"""
        
        # Use Claude fallback if enabled
        if self.using_fallback and self.claude_fallback:
            logger.info("Using Claude fallback with strict guardrails")
            
            # Track if this is Archon's own output
            if context and context.get("source") == "archon":
                self.claude_fallback.track_archon_output(prompt)
            
            # Perform validation with guardrails
            result = await self.claude_fallback.validate_with_guardrails(
                content=prompt,
                validation_type=validation_type,
                context=context
            )
            
            # Update metrics
            self.total_requests += 1
            
            return result
        
        # Use external LLM if available
        if not self.client:
            # Initialize fallback if not already done
            if not self.claude_fallback:
                logger.warning("No LLM client available, initializing Claude fallback")
                self.claude_fallback = ClaudeFallbackValidator()
                self.using_fallback = True
                return await self.validate_with_llm(prompt, context, temperature, validation_type)
            
            raise ValueError("No LLM client or fallback available")
        
        # Build validation prompt for external LLM
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(prompt, context)
        
        # Use temperature override or config default
        temp = temperature or self.config.llm_config.temperature
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.llm_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.llm_config.max_tokens,
                temperature=temp,
                response_format={"type": "json_object"}
            )
            
            # Update metrics
            self.total_requests += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens
            
            # Parse response
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                result["validator"] = self.config.llm_config.provider
                return result
            except json.JSONDecodeError:
                # Fallback to text parsing
                return self.parse_text_response(content)
                
        except Exception as e:
            logger.error(f"External LLM validation error: {e}", exc_info=True)
            
            # Try fallback on error
            if not self.using_fallback:
                logger.info("External LLM failed, switching to Claude fallback")
                self.claude_fallback = ClaudeFallbackValidator()
                self.using_fallback = True
                self.client = None
                return await self.validate_with_llm(prompt, context, temperature, validation_type)
            
            raise
    
    def build_system_prompt(self) -> str:
        """Build system prompt for validation"""
        
        return """You are an External Validator for the Archon system. Your role is to:

1. Detect hallucinations and false claims
2. Verify factual accuracy against provided context
3. Check for logical consistency
4. Identify potential errors or issues
5. Provide deterministic, fact-based validation

CRITICAL VALIDATION RULES:
- Only flag ACTUAL issues that will cause runtime errors or security vulnerabilities
- Do NOT flag Python attributes (self.x, this.x) as missing file references
- Do NOT flag standard library imports (os, sys, json, etc.) as missing
- Do NOT flag performance concerns (recursion, O(n²)) as errors unless infinite
- PASS code that works correctly even if not optimal
- Focus on correctness, not style or optimization

CODE CONTEXT AWARENESS:
- self.attribute refers to instance variables, NOT file paths
- module.function refers to Python modules/functions, NOT file paths
- Only paths with file extensions (.py, .js, .json, etc.) are file references
- Standard Python patterns (try/except, with statements) are valid constructs

You must respond in JSON format with the following structure:
{
    "valid": boolean,
    "confidence": float (0.0-1.0),
    "issues": [
        {
            "type": "hallucination|error|inconsistency|security|other",
            "description": "detailed description",
            "evidence": "supporting evidence",
            "severity": "critical|error|warning|info"
        }
    ],
    "verified_claims": ["list of verified factual claims"],
    "unverified_claims": ["list of unverifiable claims"],
    "suggestions": ["list of improvement suggestions"]
}

VALIDATION PRECISION TARGET: 92%+ accuracy with <8% false positives
Be strict about real issues, lenient about style and optimization.
Focus on factual accuracy and runtime correctness."""
    
    def build_user_prompt(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt for validation"""
        
        prompt_parts = ["Please validate the following content:\n"]
        
        # Add context if provided
        if context:
            prompt_parts.append("CONTEXT:")
            
            # Add PRP context if available
            if "prp" in context:
                prompt_parts.append(f"Project Requirements: {context['prp']}")
            
            # Add file context
            if "files" in context:
                prompt_parts.append(f"Related Files: {context['files']}")
            
            # Add entity context
            if "entities" in context:
                prompt_parts.append(f"Known Entities: {context['entities']}")
            
            # Add documentation context
            if "docs" in context:
                prompt_parts.append(f"Documentation: {context['docs']}")
            
            prompt_parts.append("")
        
        # Add content to validate
        prompt_parts.append("CONTENT TO VALIDATE:")
        prompt_parts.append(content)
        prompt_parts.append("")
        prompt_parts.append("Perform thorough validation and respond in JSON format.")
        
        return "\n".join(prompt_parts)
    
    def parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails"""
        
        # Default response structure
        result = {
            "valid": True,
            "confidence": 0.5,
            "issues": [],
            "verified_claims": [],
            "unverified_claims": [],
            "suggestions": []
        }
        
        # Simple heuristic parsing
        text_lower = text.lower()
        
        # Check for validation status
        if "invalid" in text_lower or "fail" in text_lower or "error" in text_lower:
            result["valid"] = False
            result["confidence"] = 0.3
        elif "valid" in text_lower or "pass" in text_lower or "correct" in text_lower:
            result["valid"] = True
            result["confidence"] = 0.8
        
        # Check for hallucination mentions
        if "hallucination" in text_lower:
            result["issues"].append({
                "type": "hallucination",
                "description": "Potential hallucination detected",
                "evidence": text[:200],
                "severity": "error"
            })
            result["confidence"] = min(result["confidence"], 0.4)
        
        return result
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get LLM client metrics"""
        
        base_metrics = {
            "provider": "claude_fallback" if self.using_fallback else self.config.llm_config.provider,
            "model": "claude-code-guardrails" if self.using_fallback else self.config.llm_config.model,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "average_tokens": self.total_tokens / max(self.total_requests, 1),
            "using_fallback": self.using_fallback
        }
        
        # Add fallback-specific metrics
        if self.using_fallback and self.claude_fallback:
            fallback_metrics = self.claude_fallback.get_metrics()
            base_metrics.update({
                "fallback_metrics": fallback_metrics,
                "guardrails_active": True,
                "skepticism_level": self.claude_fallback.SKEPTICISM_LEVEL,
                "min_issues_required": self.claude_fallback.MIN_ISSUES_REQUIRED,
                "max_confidence_self_work": self.claude_fallback.MAX_CONFIDENCE_SELF_WORK
            })
        
        return base_metrics