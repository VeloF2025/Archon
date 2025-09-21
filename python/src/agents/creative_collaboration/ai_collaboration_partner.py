"""
AI Collaboration Partner for Phase 10: Human-AI Pair Programming Evolution

This module implements personalized AI partners that adapt to individual developer styles,
provide predictive assistance, and enable seamless human-AI collaboration in creative
problem-solving and development workflows.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from uuid import uuid4, UUID
import json
import re
from collections import defaultdict, deque

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logger = logging.getLogger(__name__)


class DeveloperPersonality(str, Enum):
    """Developer personality types for collaboration adaptation."""
    ANALYTICAL = "analytical"          # Methodical, data-driven, systematic
    CREATIVE = "creative"              # Innovative, experimental, artistic
    PRAGMATIC = "pragmatic"            # Practical, solution-focused, efficient
    COLLABORATIVE = "collaborative"    # Team-oriented, communication-focused
    PERFECTIONIST = "perfectionist"    # Detail-oriented, quality-focused
    EXPLORER = "explorer"              # Curious, learning-oriented, adventurous


class CodingStyle(str, Enum):
    """Coding style preferences."""
    VERBOSE = "verbose"                # Detailed comments, explicit naming
    CONCISE = "concise"               # Minimal, elegant, compact
    FUNCTIONAL = "functional"         # Functional programming patterns
    OBJECT_ORIENTED = "object_oriented" # OOP patterns and structures
    EXPERIMENTAL = "experimental"     # Cutting-edge techniques
    TRADITIONAL = "traditional"       # Established patterns and practices


class AssistanceType(str, Enum):
    """Types of assistance provided by AI partner."""
    CODE_COMPLETION = "code_completion"
    REFACTORING_SUGGESTION = "refactoring_suggestion"
    ARCHITECTURE_GUIDANCE = "architecture_guidance"
    BUG_DETECTION = "bug_detection"
    OPTIMIZATION_TIP = "optimization_tip"
    LEARNING_RESOURCE = "learning_resource"
    CREATIVE_SUGGESTION = "creative_suggestion"
    PROBLEM_DECOMPOSITION = "problem_decomposition"


class InteractionMode(str, Enum):
    """Modes of AI-human interaction."""
    PASSIVE = "passive"               # Respond only when asked
    PROACTIVE = "proactive"           # Offer suggestions actively
    COLLABORATIVE = "collaborative"   # Full partnership mode
    MENTORING = "mentoring"           # Educational, guidance-focused
    CREATIVE = "creative"             # Brainstorming and innovation


@dataclass
class DeveloperProfile:
    """Comprehensive profile of a human developer."""
    developer_id: str
    name: str = ""
    
    # Personality and style
    personality_type: DeveloperPersonality = DeveloperPersonality.PRAGMATIC
    coding_style: CodingStyle = CodingStyle.TRADITIONAL
    experience_level: int = 5  # 1-10 scale
    
    # Technical preferences
    preferred_languages: List[str] = field(default_factory=list)
    favorite_frameworks: List[str] = field(default_factory=list)
    development_domains: List[str] = field(default_factory=list)
    
    # Working patterns
    typical_session_duration: float = 4.0  # hours
    preferred_work_times: List[str] = field(default_factory=list)  # "morning", "afternoon", "evening"
    collaboration_preference: float = 0.7  # 0=solo, 1=highly collaborative
    
    # Communication preferences
    feedback_style: str = "constructive"  # "direct", "gentle", "constructive"
    explanation_depth: str = "medium"     # "brief", "medium", "detailed"
    learning_pace: str = "moderate"       # "fast", "moderate", "gradual"
    
    # AI assistance preferences
    preferred_interaction_mode: InteractionMode = InteractionMode.PROACTIVE
    assistance_frequency: float = 0.6     # 0=minimal, 1=constant
    creativity_openness: float = 0.5      # Willingness to try creative suggestions
    
    # Performance metrics
    productivity_metrics: Dict[str, float] = field(default_factory=dict)
    satisfaction_scores: Dict[str, float] = field(default_factory=dict)
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Learning and adaptation
    skill_growth_areas: List[str] = field(default_factory=list)
    recent_interests: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CollaborationContext:
    """Context for a specific collaboration session."""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    developer_id: str = ""
    project_context: Dict[str, Any] = field(default_factory=dict)
    
    # Current task context
    current_task: str = ""
    task_complexity: int = 5  # 1-10 scale
    task_type: str = "development"  # "debugging", "design", "optimization", etc.
    deadline_pressure: int = 3  # 1-10 scale
    
    # Session state
    session_goals: List[str] = field(default_factory=list)
    progress_indicators: Dict[str, float] = field(default_factory=dict)
    challenges_encountered: List[str] = field(default_factory=list)
    
    # Code context
    current_file: str = ""
    programming_language: str = ""
    code_context: str = ""
    recent_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    started_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)


@dataclass
class AssistanceSuggestion:
    """A suggestion provided by the AI collaboration partner."""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: AssistanceType = AssistanceType.CODE_COMPLETION
    title: str = ""
    description: str = ""
    
    # Suggestion content
    code_snippet: Optional[str] = None
    explanation: str = ""
    benefits: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    
    # Targeting and adaptation
    confidence: float = 0.8  # AI confidence in suggestion
    relevance_score: float = 0.7  # How relevant to current context
    personalization_score: float = 0.5  # How well adapted to developer
    
    # Learning resources
    documentation_links: List[str] = field(default_factory=list)
    tutorial_suggestions: List[str] = field(default_factory=list)
    similar_examples: List[str] = field(default_factory=list)
    
    # Interaction tracking
    presented_at: datetime = field(default_factory=datetime.now)
    developer_response: Optional[str] = None  # "accepted", "rejected", "modified"
    feedback_score: Optional[float] = None  # 1-10 rating
    implementation_success: Optional[bool] = None
    
    # Context preservation
    session_context: Dict[str, Any] = field(default_factory=dict)
    code_context: str = ""


class AICollaborationPartner:
    """
    Personalized AI partner that adapts to individual developer styles
    and provides predictive, context-aware assistance.
    """
    
    def __init__(self, partner_id: str, specialized_domain: str = "general"):
        self.partner_id = partner_id
        self.specialized_domain = specialized_domain
        
        # Developer profiles and adaptation
        self.developer_profiles: Dict[str, DeveloperProfile] = {}
        self.active_sessions: Dict[str, CollaborationContext] = {}
        
        # Learning and prediction models
        self.interaction_patterns = defaultdict(list)
        self.suggestion_effectiveness = defaultdict(dict)
        self.code_pattern_recognizer = None
        self.preference_predictor = None
        
        # Knowledge base
        self.code_examples_db = {}
        self.best_practices_db = {}
        self.common_patterns_db = {}
        
        # Assistance configuration
        self.suggestion_algorithms = {
            AssistanceType.CODE_COMPLETION: self._generate_code_completion,
            AssistanceType.REFACTORING_SUGGESTION: self._generate_refactoring_suggestion,
            AssistanceType.ARCHITECTURE_GUIDANCE: self._generate_architecture_guidance,
            AssistanceType.BUG_DETECTION: self._generate_bug_detection,
            AssistanceType.OPTIMIZATION_TIP: self._generate_optimization_tip,
            AssistanceType.CREATIVE_SUGGESTION: self._generate_creative_suggestion,
        }
        
        # Personalization parameters
        self.adaptation_rate = 0.1  # How quickly to adapt to feedback
        self.suggestion_cooldown = 30  # Seconds between proactive suggestions
        self.min_confidence_threshold = 0.6
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models for adaptation."""
        
        # Initialize code pattern recognizer
        self.code_pattern_recognizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*'
        )
        
        # Initialize preference clustering
        self.preference_cluster_model = KMeans(n_clusters=5, random_state=42)
        
        logger.info(f"Initialized AI collaboration partner {self.partner_id}")
    
    async def create_developer_profile(
        self,
        developer_id: str,
        initial_data: Dict[str, Any]
    ) -> DeveloperProfile:
        """Create a new developer profile with initial data."""
        
        profile = DeveloperProfile(
            developer_id=developer_id,
            name=initial_data.get("name", ""),
            personality_type=DeveloperPersonality(initial_data.get("personality", "pragmatic")),
            coding_style=CodingStyle(initial_data.get("coding_style", "traditional")),
            experience_level=initial_data.get("experience_level", 5),
            preferred_languages=initial_data.get("languages", []),
            favorite_frameworks=initial_data.get("frameworks", []),
            development_domains=initial_data.get("domains", []),
            preferred_interaction_mode=InteractionMode(initial_data.get("interaction_mode", "proactive")),
            collaboration_preference=initial_data.get("collaboration_preference", 0.7)
        )
        
        self.developer_profiles[developer_id] = profile
        
        logger.info(f"Created profile for developer {developer_id}")
        return profile
    
    async def start_collaboration_session(
        self,
        developer_id: str,
        project_context: Dict[str, Any]
    ) -> str:
        """Start a new collaboration session with a developer."""
        
        if developer_id not in self.developer_profiles:
            # Create basic profile if not exists
            await self.create_developer_profile(developer_id, {"name": developer_id})
        
        context = CollaborationContext(
            developer_id=developer_id,
            project_context=project_context,
            current_task=project_context.get("task", "development"),
            programming_language=project_context.get("language", "python"),
            task_complexity=project_context.get("complexity", 5)
        )
        
        self.active_sessions[context.session_id] = context
        
        # Set session goals based on developer profile and context
        await self._set_session_goals(context)
        
        logger.info(f"Started collaboration session {context.session_id} for {developer_id}")
        return context.session_id
    
    async def _set_session_goals(self, context: CollaborationContext):
        """Set appropriate session goals based on context and developer profile."""
        
        profile = self.developer_profiles[context.developer_id]
        
        # Base goals for all sessions
        context.session_goals = ["Maintain code quality", "Enhance productivity"]
        
        # Add personalized goals based on developer profile
        if profile.personality_type == DeveloperPersonality.CREATIVE:
            context.session_goals.append("Explore innovative solutions")
        elif profile.personality_type == DeveloperPersonality.PERFECTIONIST:
            context.session_goals.append("Achieve optimal code quality")
        elif profile.personality_type == DeveloperPersonality.EXPLORER:
            context.session_goals.append("Learn new techniques and patterns")
        
        # Add goals based on task complexity
        if context.task_complexity >= 8:
            context.session_goals.append("Break down complex problems")
            context.session_goals.append("Provide architectural guidance")
        elif context.task_complexity <= 3:
            context.session_goals.append("Maintain engagement and learning")
    
    async def provide_proactive_assistance(self, session_id: str) -> Optional[AssistanceSuggestion]:
        """Provide proactive assistance based on current context."""
        
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        profile = self.developer_profiles[context.developer_id]
        
        # Check if developer wants proactive assistance
        if profile.preferred_interaction_mode == InteractionMode.PASSIVE:
            return None
        
        # Check cooldown period
        time_since_last = (datetime.now() - context.last_interaction).total_seconds()
        if time_since_last < self.suggestion_cooldown:
            return None
        
        # Analyze current context and predict needs
        predicted_needs = await self._predict_developer_needs(context)
        
        if not predicted_needs:
            return None
        
        # Generate suggestion for highest priority need
        top_need = predicted_needs[0]
        suggestion = await self._generate_suggestion(top_need, context)
        
        if suggestion and suggestion.confidence >= self.min_confidence_threshold:
            # Personalize the suggestion
            await self._personalize_suggestion(suggestion, profile)
            
            # Update interaction tracking
            context.last_interaction = datetime.now()
            
            return suggestion
        
        return None
    
    async def _predict_developer_needs(
        self,
        context: CollaborationContext
    ) -> List[Tuple[AssistanceType, float]]:
        """Predict what assistance the developer might need."""
        
        needs = []
        
        # Analyze code context for patterns
        if context.code_context:
            code_analysis = await self._analyze_code_context(context.code_context)
            
            # Predict based on code patterns
            if code_analysis.get("complexity_high"):
                needs.append((AssistanceType.REFACTORING_SUGGESTION, 0.8))
            
            if code_analysis.get("potential_bugs"):
                needs.append((AssistanceType.BUG_DETECTION, 0.9))
            
            if code_analysis.get("performance_issues"):
                needs.append((AssistanceType.OPTIMIZATION_TIP, 0.7))
            
            if code_analysis.get("incomplete_patterns"):
                needs.append((AssistanceType.CODE_COMPLETION, 0.6))
        
        # Predict based on developer behavior patterns
        developer_patterns = self.interaction_patterns.get(context.developer_id, [])
        if developer_patterns:
            recent_patterns = developer_patterns[-10:]  # Last 10 interactions
            
            # Analyze common assistance types requested
            assistance_frequency = defaultdict(int)
            for pattern in recent_patterns:
                if "assistance_type" in pattern:
                    assistance_frequency[pattern["assistance_type"]] += 1
            
            # Predict likely next need
            if assistance_frequency:
                most_common = max(assistance_frequency.items(), key=lambda x: x[1])
                needs.append((AssistanceType(most_common[0]), 0.5))
        
        # Predict based on task context
        if context.task_complexity >= 7:
            needs.append((AssistanceType.ARCHITECTURE_GUIDANCE, 0.6))
            needs.append((AssistanceType.PROBLEM_DECOMPOSITION, 0.7))
        
        if context.deadline_pressure >= 7:
            needs.append((AssistanceType.OPTIMIZATION_TIP, 0.8))
        
        # Sort by priority (confidence score)
        needs.sort(key=lambda x: x[1], reverse=True)
        
        return needs
    
    async def _analyze_code_context(self, code: str) -> Dict[str, bool]:
        """Analyze code context for patterns and issues."""
        
        analysis = {
            "complexity_high": False,
            "potential_bugs": False,
            "performance_issues": False,
            "incomplete_patterns": False
        }
        
        # Simple heuristic analysis (in production, would use AST parsing and ML)
        lines = code.split('\n')
        
        # Complexity analysis
        nested_level = 0
        max_nested = 0
        for line in lines:
            nested_level += line.count('{') - line.count('}')
            nested_level += line.count('if ') + line.count('for ') + line.count('while ')
            max_nested = max(max_nested, nested_level)
        
        analysis["complexity_high"] = max_nested > 4
        
        # Bug pattern detection
        bug_patterns = [
            r'==\s*null',  # Null comparison
            r'catch\s*\(\s*\)\s*{',  # Empty catch block
            r'console\.log',  # Debug statements left in
            r'TODO|FIXME|HACK'  # Developer comments indicating issues
        ]
        
        for pattern in bug_patterns:
            if re.search(pattern, code):
                analysis["potential_bugs"] = True
                break
        
        # Performance issue patterns
        perf_patterns = [
            r'for.*for.*for',  # Nested loops
            r'setTimeout\s*\(\s*\w+\s*,\s*0\s*\)',  # Unnecessary setTimeout
            r'document\.getElementById.*in.*for'  # DOM queries in loops
        ]
        
        for pattern in perf_patterns:
            if re.search(pattern, code):
                analysis["performance_issues"] = True
                break
        
        # Incomplete patterns
        incomplete_patterns = [
            r'function\s+\w+\s*\(\s*\)\s*{$',  # Empty function
            r'//\s*TODO',  # Incomplete implementation
            r'throw\s+new\s+Error\s*\(\s*["\']Not implemented'  # Placeholder errors
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, code):
                analysis["incomplete_patterns"] = True
                break
        
        return analysis
    
    async def _generate_suggestion(
        self,
        assistance_need: Tuple[AssistanceType, float],
        context: CollaborationContext
    ) -> Optional[AssistanceSuggestion]:
        """Generate a specific assistance suggestion."""
        
        assistance_type, confidence = assistance_need
        
        # Get the appropriate generation algorithm
        generator = self.suggestion_algorithms.get(assistance_type)
        if not generator:
            return None
        
        # Generate the suggestion
        suggestion = await generator(context)
        
        if suggestion:
            suggestion.type = assistance_type
            suggestion.confidence = min(confidence, suggestion.confidence)
            suggestion.session_context = {
                "session_id": context.session_id,
                "task_type": context.task_type,
                "complexity": context.task_complexity,
                "language": context.programming_language
            }
        
        return suggestion
    
    async def _generate_code_completion(self, context: CollaborationContext) -> AssistanceSuggestion:
        """Generate code completion suggestions."""
        
        suggestion = AssistanceSuggestion(
            title="Code Completion Suggestion",
            description="Complete the current code pattern based on context",
            confidence=0.7
        )
        
        # Analyze current code context for completion
        if context.code_context:
            # Simple pattern-based completion (in production, would use ML models)
            if "function" in context.code_context.lower() and "{" in context.code_context:
                suggestion.code_snippet = """
    // Add function body
    try {
        // Implementation here
        return result;
    } catch (error) {
        console.error('Error in function:', error);
        throw error;
    }"""
                suggestion.explanation = "Complete function with error handling pattern"
                suggestion.benefits = [
                    "Proper error handling",
                    "Consistent code structure",
                    "Better debugging capability"
                ]
            
            elif "class" in context.code_context.lower():
                suggestion.code_snippet = """
    constructor() {
        // Initialize properties
    }
    
    // Add public methods
    
    // Add private methods if needed"""
                suggestion.explanation = "Complete class structure with constructor and methods"
                suggestion.benefits = [
                    "Proper class structure",
                    "Clear method organization",
                    "Maintainable code design"
                ]
        
        return suggestion
    
    async def _generate_refactoring_suggestion(self, context: CollaborationContext) -> AssistanceSuggestion:
        """Generate refactoring suggestions."""
        
        suggestion = AssistanceSuggestion(
            title="Code Refactoring Opportunity",
            description="Improve code structure and maintainability",
            confidence=0.8
        )
        
        # Analyze for common refactoring opportunities
        refactoring_tips = [
            "Extract complex logic into separate functions",
            "Replace magic numbers with named constants",
            "Consider using more descriptive variable names",
            "Reduce function complexity by breaking into smaller parts"
        ]
        
        suggestion.implementation_steps = [
            "Identify code smells and complexity hotspots",
            "Extract reusable logic into functions",
            "Improve naming for clarity",
            "Add documentation for complex logic",
            "Test refactored code thoroughly"
        ]
        
        suggestion.benefits = [
            "Improved code readability",
            "Better maintainability",
            "Reduced complexity",
            "Enhanced testability"
        ]
        
        suggestion.explanation = f"Consider these refactoring opportunities: {', '.join(refactoring_tips[:2])}"
        
        return suggestion
    
    async def _generate_architecture_guidance(self, context: CollaborationContext) -> AssistanceSuggestion:
        """Generate architectural guidance suggestions."""
        
        suggestion = AssistanceSuggestion(
            title="Architecture Guidance",
            description="Structural recommendations for better system design",
            confidence=0.6
        )
        
        # Provide architecture guidance based on context
        if context.task_complexity >= 7:
            suggestion.explanation = "For complex features, consider implementing a layered architecture"
            suggestion.implementation_steps = [
                "Separate concerns into distinct layers (presentation, business, data)",
                "Define clear interfaces between layers",
                "Implement dependency injection where appropriate",
                "Consider using design patterns (Strategy, Observer, Factory)",
                "Plan for scalability and maintainability"
            ]
        else:
            suggestion.explanation = "Keep the architecture simple and focused for this task"
            suggestion.implementation_steps = [
                "Follow single responsibility principle",
                "Minimize dependencies",
                "Use clear and consistent naming",
                "Document key decisions"
            ]
        
        suggestion.benefits = [
            "Better code organization",
            "Improved maintainability",
            "Enhanced scalability",
            "Clearer team understanding"
        ]
        
        return suggestion
    
    async def _generate_bug_detection(self, context: CollaborationContext) -> AssistanceSuggestion:
        """Generate bug detection suggestions."""
        
        suggestion = AssistanceSuggestion(
            title="Potential Bug Detection",
            description="Identified potential issues in the code",
            confidence=0.9
        )
        
        # Common bug patterns to check for
        bug_checks = [
            "Check for null/undefined values before usage",
            "Ensure proper error handling in async operations",
            "Validate input parameters",
            "Check for memory leaks in event listeners",
            "Verify proper resource cleanup"
        ]
        
        suggestion.implementation_steps = [
            "Add null/undefined checks where needed",
            "Implement proper try-catch blocks",
            "Add input validation",
            "Review async/await usage",
            "Test edge cases thoroughly"
        ]
        
        suggestion.benefits = [
            "Prevent runtime errors",
            "Improve application stability",
            "Better user experience",
            "Easier debugging"
        ]
        
        suggestion.explanation = f"Review these potential issues: {', '.join(bug_checks[:3])}"
        
        return suggestion
    
    async def _generate_optimization_tip(self, context: CollaborationContext) -> AssistanceSuggestion:
        """Generate performance optimization suggestions."""
        
        suggestion = AssistanceSuggestion(
            title="Performance Optimization Tip",
            description="Improve application performance and efficiency",
            confidence=0.7
        )
        
        # Performance optimization strategies
        optimization_strategies = [
            "Implement caching for expensive operations",
            "Use lazy loading for large datasets",
            "Optimize database queries",
            "Minimize DOM manipulations",
            "Use efficient algorithms and data structures"
        ]
        
        suggestion.implementation_steps = [
            "Profile current performance bottlenecks",
            "Implement caching strategy",
            "Optimize critical path operations",
            "Measure and validate improvements",
            "Monitor performance in production"
        ]
        
        suggestion.benefits = [
            "Faster application response",
            "Better user experience",
            "Reduced resource usage",
            "Improved scalability"
        ]
        
        suggestion.explanation = f"Consider these optimizations: {', '.join(optimization_strategies[:2])}"
        
        return suggestion
    
    async def _generate_creative_suggestion(self, context: CollaborationContext) -> AssistanceSuggestion:
        """Generate creative problem-solving suggestions."""
        
        suggestion = AssistanceSuggestion(
            title="Creative Solution Approach",
            description="Alternative creative approaches to consider",
            confidence=0.6
        )
        
        # Creative approaches based on context
        creative_approaches = [
            "Consider a completely different approach to the problem",
            "Explore using emerging technologies or patterns",
            "Think about user experience from a different angle",
            "Consider breaking conventional patterns for innovation",
            "Explore cross-domain inspiration for solutions"
        ]
        
        suggestion.implementation_steps = [
            "Brainstorm alternative approaches",
            "Research innovative solutions in similar domains",
            "Prototype different concepts",
            "Get feedback on creative approaches",
            "Iterate based on user response"
        ]
        
        suggestion.benefits = [
            "Breakthrough solutions",
            "Competitive advantage",
            "Enhanced user experience",
            "Personal growth and learning"
        ]
        
        suggestion.explanation = f"Creative opportunity: {creative_approaches[0]}"
        
        return suggestion
    
    async def _personalize_suggestion(
        self,
        suggestion: AssistanceSuggestion,
        profile: DeveloperProfile
    ):
        """Personalize a suggestion based on developer profile."""
        
        # Adjust explanation depth based on preference
        if profile.explanation_depth == "brief":
            # Shorten explanation
            suggestion.explanation = suggestion.explanation.split('.')[0] + "."
            suggestion.implementation_steps = suggestion.implementation_steps[:3]
        elif profile.explanation_depth == "detailed":
            # Add more detail
            suggestion.explanation += " This approach aligns with best practices and will improve code quality."
            suggestion.documentation_links = [
                "https://example.com/best-practices",
                "https://example.com/advanced-techniques"
            ]
        
        # Adjust based on coding style
        if profile.coding_style == CodingStyle.VERBOSE and suggestion.code_snippet:
            # Add more comments
            suggestion.code_snippet = "// " + suggestion.title + "\n" + suggestion.code_snippet
        elif profile.coding_style == CodingStyle.CONCISE and suggestion.code_snippet:
            # Minimize comments, focus on essential code
            suggestion.code_snippet = re.sub(r'//.*\n', '', suggestion.code_snippet)
        
        # Adjust based on experience level
        if profile.experience_level <= 3:  # Beginner
            suggestion.tutorial_suggestions = [
                f"Learn more about {suggestion.type.value}",
                f"Practice with {profile.preferred_languages[0] if profile.preferred_languages else 'programming'} examples"
            ]
        elif profile.experience_level >= 8:  # Expert
            suggestion.benefits.append("Demonstrates advanced techniques")
        
        # Adjust relevance score based on personalization
        suggestion.personalization_score = 0.8  # High personalization applied
    
    async def handle_developer_feedback(
        self,
        session_id: str,
        suggestion_id: str,
        response: str,
        feedback_score: Optional[float] = None
    ):
        """Handle feedback from developer on suggestions."""
        
        if session_id not in self.active_sessions:
            return
        
        context = self.active_sessions[session_id]
        developer_id = context.developer_id
        
        # Record the interaction
        interaction = {
            "timestamp": datetime.now(),
            "session_id": session_id,
            "suggestion_id": suggestion_id,
            "response": response,
            "feedback_score": feedback_score,
            "context": context.current_task
        }
        
        self.interaction_patterns[developer_id].append(interaction)
        
        # Update suggestion effectiveness tracking
        if suggestion_id not in self.suggestion_effectiveness[developer_id]:
            self.suggestion_effectiveness[developer_id][suggestion_id] = []
        
        self.suggestion_effectiveness[developer_id][suggestion_id].append({
            "response": response,
            "score": feedback_score,
            "timestamp": datetime.now()
        })
        
        # Adapt future suggestions based on feedback
        await self._adapt_to_feedback(developer_id, response, feedback_score)
        
        logger.info(f"Processed feedback for developer {developer_id}: {response}")
    
    async def _adapt_to_feedback(
        self,
        developer_id: str,
        response: str,
        feedback_score: Optional[float]
    ):
        """Adapt AI behavior based on developer feedback."""
        
        profile = self.developer_profiles.get(developer_id)
        if not profile:
            return
        
        # Adjust interaction preferences based on response
        if response == "rejected":
            # Reduce proactive suggestions slightly
            profile.assistance_frequency = max(0.1, profile.assistance_frequency - self.adaptation_rate)
        elif response == "accepted":
            # Increase confidence in similar suggestions
            profile.assistance_frequency = min(1.0, profile.assistance_frequency + self.adaptation_rate)
        
        # Adjust based on feedback score
        if feedback_score is not None:
            if feedback_score >= 8.0:
                # High satisfaction - continue similar approach
                profile.satisfaction_scores["recent"] = feedback_score
            elif feedback_score <= 4.0:
                # Low satisfaction - adjust approach
                if profile.preferred_interaction_mode == InteractionMode.PROACTIVE:
                    profile.preferred_interaction_mode = InteractionMode.COLLABORATIVE
                profile.assistance_frequency *= 0.8
        
        # Update profile
        profile.last_updated = datetime.now()
    
    async def analyze_collaboration_effectiveness(
        self,
        developer_id: str
    ) -> Dict[str, Any]:
        """Analyze the effectiveness of collaboration with a specific developer."""
        
        if developer_id not in self.developer_profiles:
            return {"error": "Developer profile not found"}
        
        interactions = self.interaction_patterns.get(developer_id, [])
        if not interactions:
            return {"message": "No interaction data available"}
        
        # Calculate effectiveness metrics
        total_interactions = len(interactions)
        accepted_suggestions = len([i for i in interactions if i.get("response") == "accepted"])
        average_satisfaction = np.mean([
            i.get("feedback_score", 5.0) for i in interactions 
            if i.get("feedback_score") is not None
        ])
        
        # Analyze suggestion types effectiveness
        suggestion_type_effectiveness = defaultdict(list)
        for interaction in interactions:
            if "assistance_type" in interaction:
                suggestion_type_effectiveness[interaction["assistance_type"]].append(
                    interaction.get("feedback_score", 5.0)
                )
        
        # Calculate trends over time
        recent_interactions = interactions[-20:]  # Last 20 interactions
        recent_satisfaction = np.mean([
            i.get("feedback_score", 5.0) for i in recent_interactions 
            if i.get("feedback_score") is not None
        ]) if recent_interactions else 0
        
        return {
            "developer_id": developer_id,
            "total_interactions": total_interactions,
            "acceptance_rate": accepted_suggestions / total_interactions if total_interactions > 0 else 0,
            "average_satisfaction": round(average_satisfaction, 2),
            "recent_satisfaction": round(recent_satisfaction, 2),
            "suggestion_effectiveness": {
                stype: round(np.mean(scores), 2) 
                for stype, scores in suggestion_type_effectiveness.items()
            },
            "collaboration_trend": "improving" if recent_satisfaction > average_satisfaction else "declining",
            "personalization_level": self.developer_profiles[developer_id].assistance_frequency
        }
    
    async def get_collaboration_insights(self) -> Dict[str, Any]:
        """Get insights across all developer collaborations."""
        
        total_developers = len(self.developer_profiles)
        active_sessions = len(self.active_sessions)
        
        if total_developers == 0:
            return {"message": "No developers registered"}
        
        # Calculate overall metrics
        all_interactions = []
        for developer_interactions in self.interaction_patterns.values():
            all_interactions.extend(developer_interactions)
        
        if not all_interactions:
            return {"message": "No interaction data available"}
        
        # Overall satisfaction
        overall_satisfaction = np.mean([
            i.get("feedback_score", 5.0) for i in all_interactions 
            if i.get("feedback_score") is not None
        ])
        
        # Most effective suggestion types
        suggestion_scores = defaultdict(list)
        for interaction in all_interactions:
            if "assistance_type" in interaction and interaction.get("feedback_score"):
                suggestion_scores[interaction["assistance_type"]].append(
                    interaction["feedback_score"]
                )
        
        most_effective = max(
            suggestion_scores.items(),
            key=lambda x: np.mean(x[1])
        ) if suggestion_scores else None
        
        # Developer personality distribution
        personality_distribution = defaultdict(int)
        for profile in self.developer_profiles.values():
            personality_distribution[profile.personality_type.value] += 1
        
        return {
            "total_developers": total_developers,
            "active_sessions": active_sessions,
            "total_interactions": len(all_interactions),
            "overall_satisfaction": round(overall_satisfaction, 2),
            "most_effective_suggestion": most_effective[0] if most_effective else None,
            "personality_distribution": dict(personality_distribution),
            "collaboration_patterns": {
                "average_session_duration": "4.2 hours",  # Would calculate from actual data
                "peak_collaboration_times": ["morning", "afternoon"],
                "common_assistance_types": list(suggestion_scores.keys())[:5]
            }
        }


async def main():
    """Test the AI collaboration partner system."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create AI collaboration partner
    ai_partner = AICollaborationPartner("partner_001", "web_development")
    
    # Create developer profile
    developer_data = {
        "name": "Alice Developer",
        "personality": "creative",
        "coding_style": "functional",
        "experience_level": 7,
        "languages": ["JavaScript", "Python", "TypeScript"],
        "frameworks": ["React", "Node.js", "FastAPI"],
        "domains": ["web_development", "api_development"],
        "interaction_mode": "proactive"
    }
    
    profile = await ai_partner.create_developer_profile("dev_001", developer_data)
    print(f"Created profile for {profile.name}")
    
    # Start collaboration session
    project_context = {
        "task": "Implementing user authentication system",
        "language": "JavaScript",
        "complexity": 7,
        "framework": "React"
    }
    
    session_id = await ai_partner.start_collaboration_session("dev_001", project_context)
    print(f"Started collaboration session: {session_id}")
    
    # Simulate proactive assistance
    suggestion = await ai_partner.provide_proactive_assistance(session_id)
    if suggestion:
        print(f"AI Suggestion: {suggestion.title}")
        print(f"Description: {suggestion.description}")
        print(f"Confidence: {suggestion.confidence:.2f}")
        
        # Simulate developer feedback
        await ai_partner.handle_developer_feedback(
            session_id, suggestion.id, "accepted", 8.5
        )
        print("Feedback recorded: accepted with score 8.5")
    
    # Get collaboration insights
    effectiveness = await ai_partner.analyze_collaboration_effectiveness("dev_001")
    print(f"\nCollaboration Effectiveness: {effectiveness}")
    
    insights = await ai_partner.get_collaboration_insights()
    print(f"\nOverall Insights: {insights}")


if __name__ == "__main__":
    asyncio.run(main())