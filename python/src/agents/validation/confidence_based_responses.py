"""
Confidence-Based Response System for Archon/Claude Code

This module ensures that AI agents only provide confident responses
and explicitly state uncertainty when confidence is below 75%
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for AI responses"""
    HIGH = "high"          # >90% - Very confident, proceed normally
    MODERATE = "moderate"  # 75-90% - Confident enough to proceed with caveats
    LOW = "low"           # 50-75% - Uncertain, need clarification
    VERY_LOW = "very_low" # <50% - Don't know, need human help

@dataclass
class ConfidenceAssessment:
    """Assessment of confidence for a response"""
    confidence_score: float
    confidence_level: ConfidenceLevel
    factors: List[str]  # What influenced the confidence
    uncertainties: List[str]  # Specific areas of uncertainty
    suggestions: List[str]  # How to improve confidence

class ConfidenceBasedResponseSystem:
    """
    System to ensure AI only confirms things when sufficiently confident
    Default threshold: 75% confidence required for confirmation
    """
    
    def __init__(self, min_confidence_threshold: float = 0.75):
        self.min_confidence_threshold = min_confidence_threshold
        self.confidence_factors = {
            'code_exists': 0.3,        # Weight for code existence validation
            'documentation_exists': 0.2, # Weight for documentation
            'test_coverage': 0.2,       # Weight for test coverage
            'similar_patterns': 0.15,   # Weight for similar code patterns
            'recent_usage': 0.15        # Weight for recent usage of the pattern
        }
        
    def assess_confidence(self, context: Dict[str, Any]) -> ConfidenceAssessment:
        """
        Assess confidence level for a given context
        """
        confidence_score = 0.0
        factors = []
        uncertainties = []
        
        # Check code existence
        if context.get('code_validation'):
            validation = context['code_validation']
            if validation.get('all_references_valid'):
                confidence_score += self.confidence_factors['code_exists']
                factors.append("All code references validated")
            else:
                uncertainties.append("Some code references could not be validated")
                confidence_score += self.confidence_factors['code_exists'] * validation.get('validation_rate', 0)
                
        # Check documentation
        if context.get('documentation_found'):
            confidence_score += self.confidence_factors['documentation_exists']
            factors.append("Documentation found")
        else:
            uncertainties.append("No documentation found for this feature")
            
        # Check test coverage
        if context.get('tests_exist'):
            confidence_score += self.confidence_factors['test_coverage']
            factors.append("Tests exist for this code")
        else:
            uncertainties.append("No tests found")
            
        # Check for similar patterns
        if context.get('similar_patterns_found'):
            confidence_score += self.confidence_factors['similar_patterns']
            factors.append("Similar patterns found in codebase")
        else:
            uncertainties.append("No similar patterns to reference")
            
        # Check recent usage
        if context.get('recently_used'):
            confidence_score += self.confidence_factors['recent_usage']
            factors.append("Pattern recently used successfully")
            
        # Determine confidence level
        if confidence_score >= 0.9:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= self.min_confidence_threshold:
            level = ConfidenceLevel.MODERATE
        elif confidence_score >= 0.5:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
            
        # Generate suggestions for improving confidence
        suggestions = self._generate_improvement_suggestions(uncertainties)
        
        return ConfidenceAssessment(
            confidence_score=confidence_score,
            confidence_level=level,
            factors=factors,
            uncertainties=uncertainties,
            suggestions=suggestions
        )
        
    def _generate_improvement_suggestions(self, uncertainties: List[str]) -> List[str]:
        """Generate suggestions to improve confidence"""
        suggestions = []
        
        if "Some code references could not be validated" in uncertainties:
            suggestions.append("Search the codebase for the exact method/class names")
            suggestions.append("Check if the feature has been renamed or moved")
            
        if "No documentation found for this feature" in uncertainties:
            suggestions.append("Look for README files or inline documentation")
            suggestions.append("Check commit messages for implementation details")
            
        if "No tests found" in uncertainties:
            suggestions.append("Search for test files related to this feature")
            suggestions.append("Check if tests use different naming conventions")
            
        if "No similar patterns to reference" in uncertainties:
            suggestions.append("Search for similar functionality in other modules")
            suggestions.append("Review architecture documentation for patterns")
            
        return suggestions
        
    def generate_response(self, 
                         content: str, 
                         confidence: ConfidenceAssessment,
                         question_type: str = "general") -> str:
        """
        Generate response based on confidence level
        """
        if confidence.confidence_level == ConfidenceLevel.HIGH:
            # High confidence - provide direct response
            return content
            
        elif confidence.confidence_level == ConfidenceLevel.MODERATE:
            # Moderate confidence - provide response with caveats
            caveats = self._format_caveats(confidence)
            return f"{content}\n\n{caveats}"
            
        elif confidence.confidence_level == ConfidenceLevel.LOW:
            # Low confidence - express uncertainty and ask for clarification
            return self._generate_uncertain_response(content, confidence, question_type)
            
        else:  # VERY_LOW
            # Very low confidence - admit not knowing
            return self._generate_dont_know_response(confidence, question_type)
            
    def _format_caveats(self, confidence: ConfidenceAssessment) -> str:
        """Format caveats for moderate confidence responses"""
        caveats = "**Note:** I'm moderately confident about this ("
        caveats += f"{confidence.confidence_score:.0%} confidence), but please verify:\n"
        
        for uncertainty in confidence.uncertainties[:3]:  # Show top 3 uncertainties
            caveats += f"- {uncertainty}\n"
            
        return caveats
        
    def _generate_uncertain_response(self, 
                                    content: str, 
                                    confidence: ConfidenceAssessment,
                                    question_type: str) -> str:
        """Generate response when uncertain"""
        response = "I'm not entirely certain about this ("
        response += f"{confidence.confidence_score:.0%} confidence). "
        
        if question_type == "code_implementation":
            response += "Let me clarify a few things before proceeding:\n\n"
        elif question_type == "bug_fix":
            response += "To provide an accurate fix, I need to verify:\n\n"
        elif question_type == "architecture":
            response += "To give you the best guidance, I should confirm:\n\n"
        else:
            response += "Here's what I think, but we should verify:\n\n"
            
        # Add partial content if available
        if content:
            response += f"**Preliminary thoughts:**\n{content}\n\n"
            
        # Add uncertainties
        response += "**I'm uncertain about:**\n"
        for uncertainty in confidence.uncertainties:
            response += f"- {uncertainty}\n"
            
        # Add suggestions
        if confidence.suggestions:
            response += "\n**To proceed with confidence, we could:**\n"
            for suggestion in confidence.suggestions:
                response += f"- {suggestion}\n"
                
        response += "\n**Would you like me to:**\n"
        response += "1. Search the codebase for more information?\n"
        response += "2. Show you what I found so far?\n"
        response += "3. Proceed with my best guess (with caveats)?\n"
        
        return response
        
    def _generate_dont_know_response(self, 
                                    confidence: ConfidenceAssessment,
                                    question_type: str) -> str:
        """Generate 'I don't know' response when confidence is too low"""
        response = "**I don't have enough information to answer this confidently** "
        response += f"(confidence: {confidence.confidence_score:.0%}).\n\n"
        
        if question_type == "code_implementation":
            response += "To implement this correctly, I need to understand:\n"
            response += "- The existing code structure and patterns\n"
            response += "- Which libraries and frameworks are available\n"
            response += "- The specific requirements and constraints\n"
            
        elif question_type == "bug_fix":
            response += "To diagnose and fix this issue, I need:\n"
            response += "- The full error message and stack trace\n"
            response += "- The relevant code context\n"
            response += "- Steps to reproduce the issue\n"
            
        elif question_type == "architecture":
            response += "To provide architectural guidance, I need to know:\n"
            response += "- The current system architecture\n"
            response += "- Performance and scalability requirements\n"
            response += "- Technology constraints and preferences\n"
            
        else:
            response += "To answer your question, I need more context about:\n"
            
        # Add specific uncertainties
        for uncertainty in confidence.uncertainties[:3]:
            response += f"- {uncertainty}\n"
            
        response += "\n**How can we figure this out together?**\n"
        
        # Add constructive suggestions
        if confidence.suggestions:
            for suggestion in confidence.suggestions:
                response += f"- {suggestion}\n"
        else:
            response += "- Could you provide more context or examples?\n"
            response += "- Should I search the codebase for related information?\n"
            response += "- Would you like to walk through this step by step?\n"
            
        return response


class UncertaintyHandler:
    """
    Handles uncertainty in AI responses and provides fallback strategies
    """
    
    def __init__(self):
        self.uncertainty_patterns = [
            "I think",
            "probably", 
            "might be",
            "could be",
            "possibly",
            "I believe",
            "it seems",
            "appears to be",
            "likely",
            "I assume"
        ]
        
    def detect_uncertainty_in_response(self, response: str) -> Tuple[bool, List[str]]:
        """
        Detect uncertainty markers in a response
        """
        found_patterns = []
        
        for pattern in self.uncertainty_patterns:
            if pattern.lower() in response.lower():
                found_patterns.append(pattern)
                
        # Also check for hedging phrases
        hedging_phrases = [
            "I'm not sure",
            "I don't know",
            "uncertain",
            "not certain",
            "need to verify",
            "should check",
            "would need to confirm"
        ]
        
        for phrase in hedging_phrases:
            if phrase.lower() in response.lower():
                found_patterns.append(phrase)
                
        return len(found_patterns) > 0, found_patterns
        
    def rewrite_uncertain_response(self, response: str, confidence: float) -> str:
        """
        Rewrite response to be explicit about uncertainty
        """
        if confidence >= 0.75:
            # Confidence is acceptable, return as is
            return response
            
        # Add uncertainty disclaimer
        disclaimer = f"⚠️ **Low Confidence ({confidence:.0%})**: "
        
        if confidence < 0.5:
            disclaimer += "I don't have enough information to provide a reliable answer. "
        else:
            disclaimer += "I'm uncertain about this response. "
            
        # Rewrite common uncertain phrases to be more explicit
        response = response.replace("I think ", "Based on limited information, ")
        response = response.replace("probably ", "it might ")
        response = response.replace("should work", "might work but needs verification")
        response = response.replace("This will ", "This might ")
        
        return f"{disclaimer}\n\n{response}\n\n**Please verify this information before using it.**"


class AgentConfidenceWrapper:
    """
    Wraps AI agents to enforce confidence-based responses
    """
    
    def __init__(self, agent, min_confidence: float = 0.75):
        self.agent = agent
        self.confidence_system = ConfidenceBasedResponseSystem(min_confidence)
        self.uncertainty_handler = UncertaintyHandler()
        
    async def process_request(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request with confidence checking
        """
        # Assess confidence based on context
        confidence = self.confidence_system.assess_confidence(context)
        
        # Log confidence assessment
        logger.info(f"Confidence assessment: {confidence.confidence_score:.0%} - {confidence.confidence_level.value}")
        
        # If confidence is too low, return uncertainty response immediately
        if confidence.confidence_score < self.confidence_system.min_confidence_threshold:
            return {
                'success': False,
                'confidence_too_low': True,
                'confidence_score': confidence.confidence_score,
                'response': self.confidence_system.generate_response(
                    "",
                    confidence,
                    context.get('question_type', 'general')
                ),
                'suggestions': confidence.suggestions
            }
            
        # Process with agent
        result = await self.agent.process_request(request, context)
        
        # Check for uncertainty in the response
        if 'response' in result:
            has_uncertainty, patterns = self.uncertainty_handler.detect_uncertainty_in_response(result['response'])
            
            if has_uncertainty and confidence.confidence_score < 0.9:
                # Rewrite to be explicit about uncertainty
                result['response'] = self.uncertainty_handler.rewrite_uncertain_response(
                    result['response'],
                    confidence.confidence_score
                )
                result['uncertainty_detected'] = True
                result['uncertainty_patterns'] = patterns
                
        # Add confidence information to result
        result['confidence_score'] = confidence.confidence_score
        result['confidence_level'] = confidence.confidence_level.value
        result['confidence_factors'] = confidence.factors
        
        # Generate final response based on confidence
        if 'response' in result:
            result['response'] = self.confidence_system.generate_response(
                result['response'],
                confidence,
                context.get('question_type', 'general')
            )
            
        return result


# Response templates for different confidence levels
RESPONSE_TEMPLATES = {
    'high_confidence': {
        'prefix': "",
        'suffix': ""
    },
    'moderate_confidence': {
        'prefix': "Based on my analysis (confidence: {confidence:.0%}):\n\n",
        'suffix': "\n\n*Note: Please verify this solution as some aspects could not be fully validated.*"
    },
    'low_confidence': {
        'prefix': "⚠️ **I'm not certain about this** (confidence: {confidence:.0%}), but here's what I found:\n\n",
        'suffix': "\n\n**Important:** This solution needs verification. Would you like me to:\n- Search for more information?\n- Try a different approach?\n- Walk through this step-by-step together?"
    },
    'no_confidence': {
        'response': """**I don't know the answer to this** (confidence: {confidence:.0%}).

To help you properly, I need:
{needs}

Let's figure this out together. Could you:
{suggestions}

Would you like me to:
1. Search the codebase for relevant information?
2. Look for similar examples?
3. Break down the problem into smaller parts?"""
    }
}


def format_confidence_response(content: str, confidence: float, context: Dict[str, Any]) -> str:
    """
    Format response based on confidence level
    """
    if confidence >= 0.9:
        template = RESPONSE_TEMPLATES['high_confidence']
        return f"{template['prefix']}{content}{template['suffix']}"
        
    elif confidence >= 0.75:
        template = RESPONSE_TEMPLATES['moderate_confidence']
        return template['prefix'].format(confidence=confidence) + content + template['suffix']
        
    elif confidence >= 0.5:
        template = RESPONSE_TEMPLATES['low_confidence']
        return template['prefix'].format(confidence=confidence) + content + template['suffix']
        
    else:
        # Don't know - need to construct helpful response
        needs = "\n".join(f"- {need}" for need in context.get('missing_information', ['More context about the problem']))
        suggestions = "\n".join(f"- {sug}" for sug in context.get('suggestions', ['Provide more details?']))
        
        return RESPONSE_TEMPLATES['no_confidence']['response'].format(
            confidence=confidence,
            needs=needs,
            suggestions=suggestions
        )