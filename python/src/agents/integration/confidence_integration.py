"""
Confidence Integration Module

Integrates DeepConf confidence scoring into agent execution workflows.
Provides seamless confidence assessment for all agent operations.

Author: Archon AI System
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from types import SimpleNamespace

from ..deepconf.engine import DeepConfEngine, ConfidenceScore
from ..deepconf.data_ingestion import ingest_agent_execution_data
from ..base_agent import ArchonDependencies, BaseAgentOutput

# Socket.IO integration for real-time updates
try:
    from ...server.api_routes.socketio_handlers import emit_confidence_update
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass 
class ConfidenceMetrics:
    """Enhanced metrics including confidence data"""
    confidence_score: ConfidenceScore
    execution_time: float
    agent_result: Any
    confidence_calculation_time: float

class ConfidenceIntegration:
    """
    Integrates DeepConf confidence scoring with agent execution
    
    Provides methods to wrap agent execution with confidence assessment,
    real-time confidence updates, and performance tracking.
    """
    
    def __init__(self, deepconf_engine: Optional[DeepConfEngine] = None):
        """Initialize confidence integration"""
        self._engine = deepconf_engine
        self._confidence_cache = {}
        self._active_streams = {}
        
        logger.info("Confidence integration initialized")
    
    @property
    def engine(self) -> DeepConfEngine:
        """Get or create DeepConf engine (lazy loading)"""
        if self._engine is None:
            try:
                self._engine = DeepConfEngine()
                logger.info("DeepConf engine lazy-loaded in confidence integration")
            except Exception as e:
                logger.error(f"Failed to initialize DeepConf engine: {e}")
                raise RuntimeError(f"DeepConf engine initialization failed: {e}")
        return self._engine
    
    async def execute_with_confidence(
        self,
        agent: Any,
        user_prompt: str,
        deps: ArchonDependencies,
        task_description: Optional[str] = None
    ) -> Tuple[Any, ConfidenceScore]:
        """
        Execute agent with confidence scoring
        
        Args:
            agent: Agent instance to execute
            user_prompt: User input prompt
            deps: Agent dependencies
            task_description: Optional task description for better confidence analysis
            
        Returns:
            Tuple[Any, ConfidenceScore]: Agent result and confidence score
        """
        start_time = time.time()
        
        try:
            # Create task object for confidence analysis
            task = self._create_task_object(user_prompt, deps, task_description)
            
            # Create context object
            context = self._create_context_object(deps)
            
            # Start confidence tracking if task_id available
            confidence_stream = None
            if hasattr(deps, 'trace_id') and deps.trace_id:
                try:
                    confidence_stream = self.engine.start_confidence_tracking(deps.trace_id)
                    self._active_streams[deps.trace_id] = confidence_stream
                except Exception as e:
                    logger.warning(f"Failed to start confidence tracking for {deps.trace_id}: {e}")
            
            # Calculate initial confidence
            confidence_calc_start = time.time()
            confidence_score = await self.engine.calculate_confidence(task, context)
            confidence_calc_time = time.time() - confidence_calc_start
            
            logger.info(f"Initial confidence for task {task.task_id}: {confidence_score.overall_confidence:.3f}")
            
            # Emit initial confidence update
            await self._emit_confidence_update(task.task_id, confidence_score, 0.0)
            
            # Execute the agent
            agent_result = await agent.run(user_prompt, deps)
            
            execution_time = time.time() - start_time
            
            # Update confidence based on execution results
            if confidence_stream and hasattr(agent_result, 'success'):
                try:
                    # Determine execution progress and success indicators
                    progress = 1.0 if getattr(agent_result, 'success', True) else 0.8
                    intermediate_result = str(agent_result) if agent_result else "completed"
                    
                    # Update real-time confidence
                    updated_confidence = await self.engine.update_confidence_realtime(
                        task.task_id,
                        {
                            'progress': progress,
                            'intermediate_result': intermediate_result,
                            'timestamp': time.time()
                        }
                    )
                    
                    logger.debug(f"Updated confidence for {task.task_id}: {updated_confidence.overall_confidence:.3f}")
                    confidence_score = updated_confidence
                    
                    # Emit Socket.IO update
                    await self._emit_confidence_update(task.task_id, confidence_score, progress)
                    
                except Exception as e:
                    logger.warning(f"Failed to update real-time confidence for {task.task_id}: {e}")
            
            # INGEST REAL EXECUTION DATA INTO DEEPCONF
            try:
                await ingest_agent_execution_data(
                    task_id=getattr(task, 'task_id', f"task_{int(time.time())}"),
                    agent_name=getattr(agent, 'name', str(agent.__class__.__name__)),
                    agent_type=getattr(agent, 'agent_type', 'unknown'),
                    user_prompt=user_prompt,
                    execution_start_time=start_time,
                    execution_end_time=time.time(),
                    success=getattr(agent_result, 'success', True),
                    confidence_score=confidence_score,
                    result_quality=self._assess_result_quality(agent_result),
                    complexity_assessment=getattr(task, 'complexity', 'moderate'),
                    domain=getattr(task, 'domain', 'general'),
                    phase=getattr(context, 'environment', 'production')
                )
                logger.debug(f"Successfully ingested execution data for {task.task_id}")
            except Exception as e:
                logger.warning(f"Failed to ingest execution data for {task.task_id}: {e}")
            
            # Log execution metrics with confidence
            logger.info(
                f"Agent execution completed",
                extra={
                    "agent_name": getattr(agent, 'name', str(agent.__class__.__name__)),
                    "task_id": getattr(task, 'task_id', 'unknown'),
                    "execution_time": execution_time,
                    "confidence_score": confidence_score.overall_confidence,
                    "confidence_calculation_time": confidence_calc_time,
                    "factual_confidence": confidence_score.factual_confidence,
                    "reasoning_confidence": confidence_score.reasoning_confidence,
                    "contextual_confidence": confidence_score.contextual_confidence,
                    "uncertainty_bounds": confidence_score.uncertainty_bounds,
                    "gaming_detection_score": confidence_score.gaming_detection_score
                }
            )
            
            return agent_result, confidence_score
            
        except Exception as e:
            logger.error(f"Confidence-integrated execution failed: {e}")
            
            # Return agent result without confidence if possible
            try:
                agent_result = await agent.run(user_prompt, deps)
                
                # Create fallback confidence score
                fallback_confidence = ConfidenceScore(
                    overall_confidence=0.5,
                    factual_confidence=0.5,
                    reasoning_confidence=0.5,
                    contextual_confidence=0.5,
                    epistemic_uncertainty=0.5,
                    aleatoric_uncertainty=0.5,
                    uncertainty_bounds=(0.3, 0.7),
                    confidence_factors={'error_fallback': 0.5},
                    primary_factors=['error_fallback'],
                    confidence_reasoning="Fallback confidence due to integration error",
                    model_source=getattr(agent, 'model', 'unknown'),
                    timestamp=time.time(),
                    task_id=getattr(deps, 'trace_id', 'unknown')
                )
                
                return agent_result, fallback_confidence
                
            except Exception as agent_error:
                logger.error(f"Both confidence integration and agent execution failed: {agent_error}")
                raise
    
    def _create_task_object(self, user_prompt: str, deps: ArchonDependencies, task_description: Optional[str] = None) -> Any:
        """Create task object for confidence analysis"""
        
        # Determine task complexity from prompt content
        complexity = self._infer_complexity(user_prompt, task_description)
        
        # Determine domain from agent role or task description
        domain = self._infer_domain(deps.agent_role, user_prompt, task_description)
        
        # Create task object
        task = SimpleNamespace(
            task_id=getattr(deps, 'trace_id', f"task_{int(time.time())}"),
            content=task_description or user_prompt,
            complexity=complexity,
            domain=domain,
            priority=1,  # Default high priority
            model_source='pydantic_ai',
            context_size=len(user_prompt) if user_prompt else 0
        )
        
        return task
    
    def _create_context_object(self, deps: ArchonDependencies) -> Any:
        """Create context object for confidence analysis"""
        
        context = SimpleNamespace(
            user_id=getattr(deps, 'user_id', None),
            session_id=getattr(deps, 'request_id', None),
            environment='production',  # Default to production
            model_history=None,  # TODO: Implement model history tracking
            performance_data=getattr(deps, 'context', {}),
            timestamp=time.time()
        )
        
        return context
    
    def _infer_complexity(self, user_prompt: str, task_description: Optional[str] = None) -> str:
        """Infer task complexity from prompt content"""
        content = f"{user_prompt} {task_description or ''}".lower()
        
        # Complex indicators
        complex_indicators = [
            'architect', 'design system', 'refactor', 'optimize',
            'integrate multiple', 'complex workflow', 'distributed',
            'scalable', 'microservice', 'algorithm', 'performance'
        ]
        
        # Simple indicators
        simple_indicators = [
            'fix bug', 'update text', 'add button', 'change color',
            'simple function', 'basic component', 'straightforward'
        ]
        
        # Count indicators
        complex_count = sum(1 for indicator in complex_indicators if indicator in content)
        simple_count = sum(1 for indicator in simple_indicators if indicator in content)
        
        if complex_count > simple_count and complex_count >= 2:
            return 'complex'
        elif simple_count > complex_count and simple_count >= 1:
            return 'simple' 
        elif len(content.split()) > 50:  # Long descriptions tend to be more complex
            return 'complex'
        else:
            return 'moderate'
    
    def _infer_domain(self, agent_role: str, user_prompt: str, task_description: Optional[str] = None) -> str:
        """Infer domain from agent role and task content"""
        
        # Domain mapping from agent roles
        role_domain_mapping = {
            'python_backend_coder': 'backend_development',
            'typescript_frontend_agent': 'frontend_development',
            'test_generator': 'testing',
            'security_auditor': 'security',
            'documentation_writer': 'technical_documentation',
            'code_reviewer': 'code_review',
            'refactoring_specialist': 'code_maintenance',
            'performance_optimizer': 'performance_optimization',
            'system_architect': 'system_architecture',
            'database_designer': 'database_design',
            'ui_ux_designer': 'ui_design',
            'devops_engineer': 'devops'
        }
        
        if agent_role in role_domain_mapping:
            return role_domain_mapping[agent_role]
        
        # Infer from content
        content = f"{user_prompt} {task_description or ''}".lower()
        
        domain_indicators = {
            'frontend_development': ['react', 'vue', 'angular', 'ui', 'component', 'css', 'html', 'javascript'],
            'backend_development': ['api', 'server', 'database', 'python', 'fastapi', 'django', 'flask'],
            'testing': ['test', 'unittest', 'pytest', 'coverage', 'mock', 'assertion'],
            'security': ['security', 'auth', 'permission', 'vulnerability', 'encryption', 'secure'],
            'system_architecture': ['architecture', 'design', 'microservice', 'scalability', 'pattern'],
            'database_design': ['database', 'sql', 'schema', 'migration', 'query', 'table']
        }
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in content for indicator in indicators):
                return domain
        
        return 'general_development'
    
    async def get_confidence_metrics(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get confidence metrics for a task"""
        try:
            if task_id in self.engine._active_tasks:
                tracking_info = self.engine._active_tasks[task_id]
                
                # Get latest confidence score
                if tracking_info['confidence_history']:
                    latest_confidence = tracking_info['confidence_history'][-1]
                    
                    return {
                        'task_id': task_id,
                        'latest_confidence': latest_confidence.overall_confidence,
                        'confidence_trend': [
                            score.overall_confidence 
                            for score in tracking_info['confidence_history']
                        ],
                        'uncertainty_bounds': latest_confidence.uncertainty_bounds,
                        'primary_factors': latest_confidence.primary_factors,
                        'confidence_reasoning': latest_confidence.confidence_reasoning,
                        'updates_count': len(tracking_info['confidence_history']),
                        'last_update': tracking_info['last_update']
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get confidence metrics for task {task_id}: {e}")
            return None
    
    async def _emit_confidence_update(self, task_id: str, confidence_score: ConfidenceScore, progress: Optional[float] = None) -> None:
        """Emit real-time confidence update via Socket.IO"""
        if not SOCKETIO_AVAILABLE:
            return
        
        try:
            confidence_data = {
                'task_id': task_id,
                'confidence_score': confidence_score.to_dict(),
                'timestamp': time.time()
            }
            
            if progress is not None:
                confidence_data['progress'] = progress
            
            # Emit via Socket.IO handler
            await emit_confidence_update(task_id, confidence_data)
            
        except Exception as e:
            logger.error(f"Failed to emit confidence update for {task_id}: {e}")
    
    def _assess_result_quality(self, agent_result: Any) -> float:
        """Assess quality of agent execution result (0.0-1.0)"""
        try:
            # Check if result has explicit success indicators
            if hasattr(agent_result, 'success'):
                base_quality = 0.9 if agent_result.success else 0.3
            elif hasattr(agent_result, 'error'):
                base_quality = 0.2 if agent_result.error else 0.8
            else:
                # Default quality based on result content
                base_quality = 0.7
            
            # Adjust based on result completeness
            if hasattr(agent_result, 'content') or hasattr(agent_result, 'output'):
                content = getattr(agent_result, 'content', None) or getattr(agent_result, 'output', None)
                if content and len(str(content)) > 50:
                    base_quality += 0.1
                elif not content:
                    base_quality -= 0.2
            
            return max(0.0, min(1.0, base_quality))
            
        except Exception as e:
            logger.debug(f"Error assessing result quality: {e}")
            return 0.5  # Default neutral quality
    
    def cleanup_tracking(self, task_id: str) -> None:
        """Clean up confidence tracking for completed task"""
        try:
            if task_id in self._active_streams:
                del self._active_streams[task_id]
                logger.debug(f"Cleaned up confidence tracking for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup confidence tracking for {task_id}: {e}")

# Global instance for easy access
_confidence_integration: Optional[ConfidenceIntegration] = None

def get_confidence_integration() -> ConfidenceIntegration:
    """Get or create global confidence integration instance"""
    global _confidence_integration
    
    if _confidence_integration is None:
        _confidence_integration = ConfidenceIntegration()
    
    return _confidence_integration

async def execute_agent_with_confidence(
    agent: Any,
    user_prompt: str,
    deps: ArchonDependencies,
    task_description: Optional[str] = None
) -> Tuple[Any, ConfidenceScore]:
    """
    Convenient function to execute any agent with confidence scoring
    
    Args:
        agent: Agent instance
        user_prompt: User input prompt
        deps: Agent dependencies
        task_description: Optional task description
        
    Returns:
        Tuple[Any, ConfidenceScore]: Agent result and confidence score
    """
    integration = get_confidence_integration()
    return await integration.execute_with_confidence(agent, user_prompt, deps, task_description)

# Export main classes and functions
__all__ = [
    'ConfidenceIntegration',
    'ConfidenceMetrics',
    'get_confidence_integration',
    'execute_agent_with_confidence'
]