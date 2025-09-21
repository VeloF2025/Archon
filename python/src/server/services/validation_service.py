"""
Validation Service for Anti-Hallucination and Confidence Checking

This service integrates the enhanced anti-hallucination validator and 
confidence-based response system into Archon's main server.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime

from ...agents.validation.enhanced_antihall_validator import (
    EnhancedAntiHallValidator,
    CodeReference,
    ValidationReport,
    ValidationResult,
    RealTimeValidator,
    AgentValidationWrapper
)
from ...agents.validation.confidence_based_responses import (
    ConfidenceBasedResponseSystem,
    ConfidenceAssessment,
    ConfidenceLevel,
    UncertaintyHandler,
    AgentConfidenceWrapper,
    format_confidence_response
)

logger = logging.getLogger(__name__)

# Global instance
_validation_service: Optional['ValidationService'] = None

@dataclass
class ValidationConfig:
    """Configuration for validation service"""
    project_root: str
    min_confidence_threshold: float = 0.75
    enable_real_time_validation: bool = True
    enable_auto_fix: bool = True
    cache_validation_results: bool = True
    max_cache_size: int = 1000

class ValidationService:
    """
    Central validation service for all AI operations in Archon
    Enforces the 75% confidence rule and prevents hallucinations
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validator = EnhancedAntiHallValidator(config.project_root)
        self.confidence_system = ConfidenceBasedResponseSystem(config.min_confidence_threshold)
        self.uncertainty_handler = UncertaintyHandler()
        self.real_time_validator = RealTimeValidator(self.validator)
        
        # Validation cache for performance
        self.validation_cache: Dict[str, ValidationReport] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics tracking
        self.stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'low_confidence_blocks': 0,
            'auto_fixes_attempted': 0,
            'auto_fixes_successful': 0,
            'hallucinations_prevented': 0,
            'average_confidence': 0.0
        }
        
        logger.info(f"ValidationService initialized with project root: {config.project_root}")
        logger.info(f"Minimum confidence threshold: {config.min_confidence_threshold:.0%}")
        
    async def validate_code_reference(self, reference: str, reference_type: str = 'generic') -> ValidationReport:
        """
        Validate a single code reference
        """
        # Check cache first
        cache_key = f"{reference_type}:{reference}"
        if self.config.cache_validation_results and cache_key in self.validation_cache:
            self.cache_hits += 1
            return self.validation_cache[cache_key]
            
        self.cache_misses += 1
        
        # Create reference object
        code_ref = CodeReference(reference_type, reference)
        
        # Validate
        report = self.validator.validate_reference(code_ref)
        
        # Cache result
        if self.config.cache_validation_results:
            self.validation_cache[cache_key] = report
            
            # Limit cache size
            if len(self.validation_cache) > self.config.max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self.validation_cache.keys())[:100]
                for key in oldest_keys:
                    del self.validation_cache[key]
                    
        self.stats['total_validations'] += 1
        if report.result == ValidationResult.EXISTS:
            self.stats['successful_validations'] += 1
        else:
            self.stats['failed_validations'] += 1
            
        return report
        
    async def validate_code_snippet(self, 
                                   code: str, 
                                   language: str = 'python',
                                   min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate an entire code snippet and return detailed results
        """
        if min_confidence is None:
            min_confidence = self.config.min_confidence_threshold
            
        # Validate the code
        is_valid, validation_summary = self.validator.enforce_validation(code, language, min_confidence)
        
        # Track statistics
        if not is_valid:
            if validation_summary.get('confidence_too_low'):
                self.stats['low_confidence_blocks'] += 1
            else:
                self.stats['hallucinations_prevented'] += 1
                
        # Update average confidence
        if 'average_confidence' in validation_summary:
            total_validations = self.stats['total_validations']
            current_avg = self.stats['average_confidence']
            new_avg = ((current_avg * (total_validations - 1)) + validation_summary['average_confidence']) / total_validations
            self.stats['average_confidence'] = new_avg
            
        return {
            'valid': is_valid,
            'validation_summary': validation_summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def validate_with_confidence(self, 
                                      content: str,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate content with confidence assessment
        Returns appropriate response based on confidence level
        """
        # Assess confidence
        confidence = self.confidence_system.assess_confidence(context)
        
        # Check if confidence is too low
        if confidence.confidence_score < self.config.min_confidence_threshold:
            self.stats['low_confidence_blocks'] += 1
            
            response = self.confidence_system.generate_response(
                content,
                confidence,
                context.get('question_type', 'general')
            )
            
            return {
                'success': False,
                'confidence_too_low': True,
                'confidence_score': confidence.confidence_score,
                'confidence_level': confidence.confidence_level.value,
                'response': response,
                'suggestions': confidence.suggestions,
                'uncertainties': confidence.uncertainties
            }
            
        # Process with confidence-aware formatting
        response = self.confidence_system.generate_response(
            content,
            confidence,
            context.get('question_type', 'general')
        )
        
        return {
            'success': True,
            'confidence_score': confidence.confidence_score,
            'confidence_level': confidence.confidence_level.value,
            'response': response,
            'factors': confidence.factors
        }
        
    async def validate_agent_response(self, 
                                     agent_response: str,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an AI agent's response for hallucinations and confidence
        """
        # Check for uncertainty patterns
        has_uncertainty, patterns = self.uncertainty_handler.detect_uncertainty_in_response(agent_response)
        
        # Validate any code in the response
        validation_results = []
        if context.get('contains_code'):
            code_snippets = self._extract_code_snippets(agent_response)
            for snippet in code_snippets:
                result = await self.validate_code_snippet(
                    snippet['code'],
                    snippet.get('language', 'python')
                )
                validation_results.append(result)
                
        # Calculate overall confidence
        confidence_context = {
            'code_validation': {
                'all_references_valid': all(r['valid'] for r in validation_results),
                'validation_rate': sum(1 for r in validation_results if r['valid']) / len(validation_results) if validation_results else 1.0
            },
            'documentation_found': context.get('documentation_found', False),
            'tests_exist': context.get('tests_exist', False),
            'similar_patterns_found': context.get('similar_patterns_found', False),
            'recently_used': context.get('recently_used', False)
        }
        
        confidence_result = await self.validate_with_confidence(agent_response, confidence_context)
        
        # Rewrite response if needed
        if has_uncertainty and confidence_result['confidence_score'] < 0.9:
            agent_response = self.uncertainty_handler.rewrite_uncertain_response(
                agent_response,
                confidence_result['confidence_score']
            )
            
        return {
            'original_response': agent_response,
            'confidence_result': confidence_result,
            'validation_results': validation_results,
            'uncertainty_detected': has_uncertainty,
            'uncertainty_patterns': patterns if has_uncertainty else []
        }
        
    def _extract_code_snippets(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract code snippets from text (markdown code blocks)
        """
        import re
        
        snippets = []
        
        # Match markdown code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for language, code in matches:
            if not language:
                # Try to detect language
                if 'import' in code or 'def ' in code or 'class ' in code:
                    language = 'python'
                elif 'const ' in code or 'let ' in code or 'function ' in code:
                    language = 'javascript'
                elif 'interface ' in code or ': ' in code:
                    language = 'typescript'
                else:
                    language = 'unknown'
                    
            snippets.append({
                'language': language,
                'code': code
            })
            
        return snippets
        
    async def create_validated_agent(self, agent, agent_name: str) -> AgentValidationWrapper:
        """
        Wrap an agent with validation enforcement
        """
        logger.info(f"Creating validated agent wrapper for: {agent_name}")
        
        wrapped_agent = AgentValidationWrapper(agent, self.validator)
        
        # Also wrap with confidence checking
        confidence_wrapped = AgentConfidenceWrapper(
            wrapped_agent, 
            self.config.min_confidence_threshold
        )
        
        return confidence_wrapped
        
    async def perform_real_time_validation(self, 
                                          line: str,
                                          context: Dict[str, Any]) -> Optional[str]:
        """
        Perform real-time validation on a single line of code
        Returns error message if invalid, None if valid
        """
        if not self.config.enable_real_time_validation:
            return None
            
        return self.real_time_validator.validate_line(line, context)
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation service statistics
        """
        return {
            **self.stats,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'validation_success_rate': self.stats['successful_validations'] / self.stats['total_validations'] if self.stats['total_validations'] > 0 else 0,
            'confidence_block_rate': self.stats['low_confidence_blocks'] / self.stats['total_validations'] if self.stats['total_validations'] > 0 else 0,
            'auto_fix_success_rate': self.stats['auto_fixes_successful'] / self.stats['auto_fixes_attempted'] if self.stats['auto_fixes_attempted'] > 0 else 0
        }
        
    def clear_cache(self):
        """
        Clear the validation cache
        """
        self.validation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Validation cache cleared")
        
    async def shutdown(self):
        """
        Cleanup validation service
        """
        stats = self.get_statistics()
        logger.info(f"ValidationService shutting down. Final statistics: {stats}")
        self.clear_cache()


# Service initialization and management

async def initialize_validation_service(project_root: Optional[str] = None) -> ValidationService:
    """
    Initialize the global validation service
    """
    global _validation_service
    
    if _validation_service is not None:
        logger.warning("ValidationService already initialized")
        return _validation_service
        
    if project_root is None:
        # Try to detect project root
        project_root = os.getenv('ARCHON_PROJECT_ROOT', '/mnt/c/Jarvis/AI Workspace/Archon')
        
    config = ValidationConfig(
        project_root=project_root,
        min_confidence_threshold=float(os.getenv('ARCHON_MIN_CONFIDENCE', '0.75')),
        enable_real_time_validation=os.getenv('ARCHON_REAL_TIME_VALIDATION', 'true').lower() == 'true',
        enable_auto_fix=os.getenv('ARCHON_AUTO_FIX', 'true').lower() == 'true',
        cache_validation_results=os.getenv('ARCHON_CACHE_VALIDATION', 'true').lower() == 'true'
    )
    
    _validation_service = ValidationService(config)
    logger.info("ValidationService initialized successfully")
    
    # Build code index in background
    asyncio.create_task(_build_index_async())
    
    return _validation_service


async def _build_index_async():
    """
    Build code index asynchronously
    """
    global _validation_service
    if _validation_service:
        logger.info("Building code index in background...")
        # The index is built during validator initialization
        logger.info("Code index ready")


def get_validation_service() -> Optional[ValidationService]:
    """
    Get the global validation service instance
    """
    return _validation_service


async def cleanup_validation_service():
    """
    Cleanup the validation service
    """
    global _validation_service
    
    if _validation_service:
        await _validation_service.shutdown()
        _validation_service = None
        logger.info("ValidationService cleaned up")