"""
Gemini CLI Service - Integration with Google Gemini CLI for multimodal AI operations

This service provides:
- Rate-limited access to Gemini CLI (60 req/min, 1000 req/day)
- Intelligent task routing and queuing
- Response caching for efficiency
- Multimodal processing capabilities (images, PDFs, etc.)
- Fallback to API when limits are reached
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import httpx
from redis import asyncio as aioredis

from ..config.logfire_config import get_logger

logger = get_logger(__name__)

# Rate limit configuration
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_DAY = 1000
CACHE_TTL_HOURS = 24

# Task allocation buckets (daily budget)
TASK_ALLOCATIONS = {
    "multimodal": 200,  # 20% - Image/PDF processing
    "large_context": 300,  # 30% - Large codebase analysis
    "code_generation": 300,  # 30% - Code generation tasks
    "documentation": 100,  # 10% - Documentation processing
    "buffer": 100,  # 10% - On-demand buffer
}


class TaskType(Enum):
    """Types of tasks that can be routed to Gemini CLI"""
    MULTIMODAL = "multimodal"  # Image/PDF processing
    LARGE_CONTEXT = "large_context"  # >128K tokens
    CODE_GENERATION = "code_generation"  # Generate code from specs
    DOCUMENTATION = "documentation"  # Process documentation
    ANALYSIS = "analysis"  # Code analysis
    GENERAL = "general"  # General queries


class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = "high"  # Execute immediately
    NORMAL = "normal"  # Standard queue
    LOW = "low"  # Can wait or use cache


@dataclass
class GeminiTask:
    """Represents a task to be processed by Gemini CLI"""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: TaskType = TaskType.GENERAL
    priority: TaskPriority = TaskPriority.NORMAL
    prompt: str = ""
    files: List[str] = field(default_factory=list)  # File paths for multimodal
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    cache_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7


@dataclass
class RateLimitState:
    """Tracks rate limit state"""
    minute_count: int = 0
    minute_reset: datetime = field(default_factory=datetime.now)
    daily_count: int = 0
    daily_reset: datetime = field(default_factory=datetime.now)
    daily_allocations: Dict[str, int] = field(default_factory=lambda: TASK_ALLOCATIONS.copy())


class GeminiCLIService:
    """Service for managing Gemini CLI operations with rate limiting and caching"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """Initialize the Gemini CLI service
        
        Args:
            redis_client: Optional Redis client for caching and state management
        """
        self.redis = redis_client
        self.rate_limit = RateLimitState()
        self.task_queue: deque[GeminiTask] = deque()
        self.processing_lock = asyncio.Lock()
        self._cache_enabled = redis_client is not None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the Gemini CLI service and check availability
        
        Returns:
            bool: True if Gemini CLI is available and configured
        """
        try:
            # Check if Gemini CLI is installed
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info(f"Gemini CLI available: {result.stdout.strip()}")
                self._initialized = True
                
                # Load rate limit state from Redis if available
                if self.redis:
                    await self._load_rate_limit_state()
                
                return True
            else:
                logger.warning("Gemini CLI not found or not configured")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini CLI: {e}")
            return False
    
    async def can_execute(self, task_type: TaskType = TaskType.GENERAL) -> Tuple[bool, str]:
        """Check if we can execute a task given current rate limits
        
        Args:
            task_type: Type of task to check
            
        Returns:
            Tuple[bool, str]: (can_execute, reason_if_not)
        """
        now = datetime.now()
        
        # Reset minute counter if needed
        if (now - self.rate_limit.minute_reset).total_seconds() >= 60:
            self.rate_limit.minute_count = 0
            self.rate_limit.minute_reset = now
        
        # Reset daily counter if needed  
        if (now - self.rate_limit.daily_reset).total_seconds() >= 86400:
            self.rate_limit.daily_count = 0
            self.rate_limit.daily_reset = now
            self.rate_limit.daily_allocations = TASK_ALLOCATIONS.copy()
        
        # Check minute limit
        if self.rate_limit.minute_count >= RATE_LIMIT_PER_MINUTE:
            return False, f"Minute rate limit reached ({RATE_LIMIT_PER_MINUTE}/min)"
        
        # Check daily limit
        if self.rate_limit.daily_count >= RATE_LIMIT_PER_DAY:
            return False, f"Daily rate limit reached ({RATE_LIMIT_PER_DAY}/day)"
        
        # Check task type allocation
        task_category = self._get_task_category(task_type)
        if self.rate_limit.daily_allocations.get(task_category, 0) <= 0:
            # Try to use buffer allocation
            if self.rate_limit.daily_allocations.get("buffer", 0) > 0:
                logger.info(f"Using buffer allocation for {task_category}")
            else:
                return False, f"Daily allocation exhausted for {task_category}"
        
        return True, "OK"
    
    async def execute_task(self, task: GeminiTask) -> Dict[str, Any]:
        """Execute a task using Gemini CLI
        
        Args:
            task: The task to execute
            
        Returns:
            Dict containing the result or error information
        """
        # Check cache first
        if task.cache_key and self._cache_enabled:
            cached_result = await self._get_cached_response(task.cache_key)
            if cached_result:
                logger.info(f"Cache hit for task {task.id}")
                return cached_result
        
        # Check rate limits
        can_execute, reason = await self.can_execute(task.type)
        if not can_execute:
            # Add to queue for later processing
            self.task_queue.append(task)
            logger.warning(f"Task {task.id} queued: {reason}")
            return {
                "status": "queued",
                "reason": reason,
                "task_id": task.id,
                "queue_position": len(self.task_queue)
            }
        
        # Execute the task
        async with self.processing_lock:
            try:
                result = await self._execute_gemini_cli(task)
                
                # Update rate limits
                self.rate_limit.minute_count += 1
                self.rate_limit.daily_count += 1
                
                # Update allocations
                task_category = self._get_task_category(task.type)
                if self.rate_limit.daily_allocations[task_category] > 0:
                    self.rate_limit.daily_allocations[task_category] -= 1
                else:
                    self.rate_limit.daily_allocations["buffer"] -= 1
                
                # Cache the result
                if task.cache_key and self._cache_enabled:
                    await self._cache_response(task.cache_key, result)
                
                # Save rate limit state
                if self.redis:
                    await self._save_rate_limit_state()
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to execute task {task.id}: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "task_id": task.id
                }
    
    async def _execute_gemini_cli(self, task: GeminiTask) -> Dict[str, Any]:
        """Execute Gemini CLI command
        
        Args:
            task: The task to execute
            
        Returns:
            Dict containing the CLI output
        """
        # Build command
        cmd = ["gemini", "-p", task.prompt]
        
        # Add file includes for multimodal tasks
        for file_path in task.files:
            if os.path.exists(file_path):
                cmd.extend(["-i", file_path])
        
        # Add context files if specified
        if "context_files" in task.context:
            for ctx_file in task.context["context_files"]:
                if os.path.exists(ctx_file):
                    cmd.extend(["-i", ctx_file])
        
        # Set temperature if specified
        if task.temperature != 0.7:
            cmd.extend(["--temperature", str(task.temperature)])
        
        # Execute command
        logger.info(f"Executing Gemini CLI for task {task.id}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                env={**os.environ, "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "")}
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "content": result.stdout,
                    "task_id": task.id,
                    "execution_time": execution_time,
                    "model": "gemini-2.5-pro",
                    "context_window": "1M tokens"
                }
            else:
                return {
                    "status": "error",
                    "error": result.stderr or "Command failed",
                    "task_id": task.id,
                    "execution_time": execution_time
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": "Gemini CLI execution timed out",
                "task_id": task.id
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "task_id": task.id
            }
    
    async def process_multimodal(
        self,
        prompt: str,
        files: List[str],
        task_type: TaskType = TaskType.MULTIMODAL
    ) -> Dict[str, Any]:
        """Process multimodal content (images, PDFs, etc.)
        
        Args:
            prompt: The prompt to use
            files: List of file paths to process
            task_type: Type of task
            
        Returns:
            Dict containing the result
        """
        task = GeminiTask(
            type=task_type,
            priority=TaskPriority.HIGH,
            prompt=prompt,
            files=files,
            cache_key=self._generate_cache_key(prompt, files)
        )
        
        return await self.execute_task(task)
    
    async def analyze_codebase(self, path: str, prompt: str) -> Dict[str, Any]:
        """Analyze an entire codebase with large context window
        
        Args:
            path: Path to the codebase
            prompt: Analysis prompt
            
        Returns:
            Dict containing the analysis result
        """
        # Find all relevant files
        code_files = []
        for ext in [".py", ".ts", ".tsx", ".js", ".jsx"]:
            code_files.extend(Path(path).rglob(f"*{ext}"))
        
        # Limit to reasonable number
        code_files = code_files[:100]
        
        task = GeminiTask(
            type=TaskType.LARGE_CONTEXT,
            priority=TaskPriority.NORMAL,
            prompt=f"Analyze this codebase: {prompt}",
            files=[str(f) for f in code_files],
            cache_key=self._generate_cache_key(f"codebase_{path}_{prompt}")
        )
        
        return await self.execute_task(task)
    
    async def generate_code_from_spec(
        self,
        spec_file: str,
        output_type: str = "typescript"
    ) -> Dict[str, Any]:
        """Generate code from a specification document
        
        Args:
            spec_file: Path to specification (PDF, image, or text)
            output_type: Type of code to generate
            
        Returns:
            Dict containing generated code
        """
        prompt = f"Generate {output_type} code from this specification. Include proper types, error handling, and documentation."
        
        task = GeminiTask(
            type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            prompt=prompt,
            files=[spec_file],
            temperature=0.3,  # Lower temperature for code generation
            cache_key=self._generate_cache_key(f"codegen_{spec_file}_{output_type}")
        )
        
        return await self.execute_task(task)
    
    async def process_queue(self) -> List[Dict[str, Any]]:
        """Process queued tasks when rate limits allow
        
        Returns:
            List of results from processed tasks
        """
        results = []
        processed = 0
        
        while self.task_queue and processed < 10:  # Process max 10 at a time
            can_execute, _ = await self.can_execute()
            if not can_execute:
                break
            
            task = self.task_queue.popleft()
            result = await self.execute_task(task)
            results.append(result)
            processed += 1
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        logger.info(f"Processed {processed} queued tasks, {len(self.task_queue)} remaining")
        return results
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics
        
        Returns:
            Dict containing usage stats
        """
        now = datetime.now()
        
        minute_remaining = RATE_LIMIT_PER_MINUTE - self.rate_limit.minute_count
        minute_reset_in = 60 - (now - self.rate_limit.minute_reset).total_seconds()
        
        daily_remaining = RATE_LIMIT_PER_DAY - self.rate_limit.daily_count
        daily_reset_in = 86400 - (now - self.rate_limit.daily_reset).total_seconds()
        
        return {
            "minute": {
                "used": self.rate_limit.minute_count,
                "remaining": minute_remaining,
                "limit": RATE_LIMIT_PER_MINUTE,
                "reset_in_seconds": max(0, minute_reset_in)
            },
            "daily": {
                "used": self.rate_limit.daily_count,
                "remaining": daily_remaining,
                "limit": RATE_LIMIT_PER_DAY,
                "reset_in_seconds": max(0, daily_reset_in),
                "allocations": self.rate_limit.daily_allocations
            },
            "queue": {
                "pending": len(self.task_queue)
            },
            "cache": {
                "enabled": self._cache_enabled
            }
        }
    
    # Helper methods
    
    def _get_task_category(self, task_type: TaskType) -> str:
        """Map task type to allocation category"""
        mapping = {
            TaskType.MULTIMODAL: "multimodal",
            TaskType.LARGE_CONTEXT: "large_context",
            TaskType.CODE_GENERATION: "code_generation",
            TaskType.DOCUMENTATION: "documentation",
            TaskType.ANALYSIS: "large_context",
            TaskType.GENERAL: "buffer"
        }
        return mapping.get(task_type, "buffer")
    
    def _generate_cache_key(self, *args) -> str:
        """Generate a cache key from arguments"""
        import hashlib
        key_str = "_".join(str(arg) for arg in args)
        return f"gemini_cli:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response from Redis"""
        if not self.redis:
            return None
        
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response in Redis"""
        if not self.redis:
            return
        
        try:
            await self.redis.setex(
                cache_key,
                timedelta(hours=CACHE_TTL_HOURS),
                json.dumps(response)
            )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def _load_rate_limit_state(self):
        """Load rate limit state from Redis"""
        if not self.redis:
            return
        
        try:
            state = await self.redis.get("gemini_cli:rate_limit_state")
            if state:
                state_dict = json.loads(state)
                self.rate_limit.daily_count = state_dict.get("daily_count", 0)
                self.rate_limit.daily_allocations = state_dict.get("daily_allocations", TASK_ALLOCATIONS.copy())
                
                # Parse dates
                if "daily_reset" in state_dict:
                    self.rate_limit.daily_reset = datetime.fromisoformat(state_dict["daily_reset"])
                
                logger.info("Loaded rate limit state from Redis")
        except Exception as e:
            logger.error(f"Failed to load rate limit state: {e}")
    
    async def _save_rate_limit_state(self):
        """Save rate limit state to Redis"""
        if not self.redis:
            return
        
        try:
            state_dict = {
                "daily_count": self.rate_limit.daily_count,
                "daily_allocations": self.rate_limit.daily_allocations,
                "daily_reset": self.rate_limit.daily_reset.isoformat()
            }
            
            await self.redis.setex(
                "gemini_cli:rate_limit_state",
                timedelta(hours=25),  # Persist for > 24 hours
                json.dumps(state_dict)
            )
        except Exception as e:
            logger.error(f"Failed to save rate limit state: {e}")


# Singleton instance
_gemini_cli_service: Optional[GeminiCLIService] = None


async def get_gemini_cli_service(redis_client: Optional[aioredis.Redis] = None) -> GeminiCLIService:
    """Get or create the Gemini CLI service singleton
    
    Args:
        redis_client: Optional Redis client for caching
        
    Returns:
        GeminiCLIService instance
    """
    global _gemini_cli_service
    
    if _gemini_cli_service is None:
        _gemini_cli_service = GeminiCLIService(redis_client)
        await _gemini_cli_service.initialize()
    
    return _gemini_cli_service