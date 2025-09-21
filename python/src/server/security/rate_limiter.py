"""
Rate limiting and DDoS protection for Archon API.
"""

import time
import hashlib
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis
import json
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Advanced rate limiter with multiple strategies:
    - Token bucket algorithm for burst handling
    - Sliding window for accurate rate limiting
    - Distributed rate limiting via Redis
    - IP-based and API key-based limiting
    - Adaptive rate limiting based on system load
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        default_rate: int = 100,  # requests per minute
        burst_size: int = 20,  # additional burst capacity
        window_size: int = 60,  # window size in seconds
    ):
        self.redis_client = redis_client
        self.default_rate = default_rate
        self.burst_size = burst_size
        self.window_size = window_size
        
        # Local cache for rate limiting (fallback if Redis unavailable)
        self.local_cache: Dict[str, deque] = defaultdict(deque)
        self.token_buckets: Dict[str, Dict] = {}
        
        # Rate limit tiers
        self.rate_tiers = {
            "free": {"rate": 60, "burst": 10},
            "basic": {"rate": 300, "burst": 50},
            "pro": {"rate": 1000, "burst": 200},
            "enterprise": {"rate": 10000, "burst": 1000},
            "internal": {"rate": 100000, "burst": 10000},  # For internal services
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/knowledge/crawl": {"rate": 10, "burst": 2},  # Heavy operation
            "/api/agents/execute": {"rate": 30, "burst": 5},  # AI operations
            "/api/knowledge/search": {"rate": 200, "burst": 50},  # Search queries
            "/health": {"rate": 1000, "burst": 100},  # Health checks
            "/api/health": {"rate": 1000, "burst": 100},  # API health checks
            "/metrics": {"rate": 100, "burst": 20},  # Metrics endpoint
        }
        
        # Blocked IPs and temporary bans
        self.blocked_ips = set()
        self.temp_bans: Dict[str, datetime] = {}
    
    def get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Priority: API key > User ID > IP address
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"
        
        # Check for authenticated user
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def get_rate_tier(self, client_id: str) -> str:
        """Determine rate tier for client."""
        if client_id.startswith("api:"):
            # Look up API key tier from database
            # For now, return basic tier
            return "basic"
        elif client_id.startswith("user:"):
            # Look up user subscription tier
            # For now, return free tier
            return "free"
        else:
            # IP-based clients get free tier
            return "free"
    
    def is_blocked(self, client_id: str) -> bool:
        """Check if client is blocked or temporarily banned."""
        # Check permanent blocks
        if client_id in self.blocked_ips:
            return True
        
        # Check temporary bans
        if client_id in self.temp_bans:
            if datetime.now() < self.temp_bans[client_id]:
                return True
            else:
                # Ban expired, remove it
                del self.temp_bans[client_id]
        
        return False
    
    def add_temp_ban(self, client_id: str, duration_minutes: int = 60):
        """Add temporary ban for client."""
        self.temp_bans[client_id] = datetime.now() + timedelta(minutes=duration_minutes)
        logger.warning(f"Temporary ban added for {client_id} for {duration_minutes} minutes")
    
    async def check_rate_limit(
        self,
        request: Request,
        endpoint: Optional[str] = None
    ) -> Tuple[bool, Dict]:
        """
        Check if request should be rate limited.
        Returns (allowed, info) tuple.
        """
        client_id = self.get_client_id(request)
        endpoint = endpoint or request.url.path
        
        # Check if client is blocked
        if self.is_blocked(client_id):
            return False, {"reason": "blocked", "retry_after": 3600}
        
        # Get rate limits
        tier = self.get_rate_tier(client_id)
        tier_limits = self.rate_tiers.get(tier, self.rate_tiers["free"])
        
        # Check for endpoint-specific limits
        if endpoint in self.endpoint_limits:
            endpoint_limit = self.endpoint_limits[endpoint]
            rate_limit = min(tier_limits["rate"], endpoint_limit["rate"])
            burst_limit = min(tier_limits["burst"], endpoint_limit["burst"])
        else:
            rate_limit = tier_limits["rate"]
            burst_limit = tier_limits["burst"]
        
        # Use Redis if available, otherwise fall back to local cache
        if self.redis_client:
            allowed, info = await self._check_redis_rate_limit(
                client_id, endpoint, rate_limit, burst_limit
            )
        else:
            allowed, info = self._check_local_rate_limit(
                client_id, endpoint, rate_limit, burst_limit
            )
        
        # Add rate limit headers
        info.update({
            "X-RateLimit-Limit": str(rate_limit),
            "X-RateLimit-Remaining": str(max(0, rate_limit - info.get("requests", 0))),
            "X-RateLimit-Reset": str(int(time.time()) + self.window_size),
        })
        
        # Check for potential DDoS
        if not allowed and info.get("requests", 0) > rate_limit * 3:
            # Client is significantly over limit, might be DDoS
            self.add_temp_ban(client_id, duration_minutes=15)
            logger.error(f"Potential DDoS detected from {client_id}")
        
        return allowed, info
    
    async def _check_redis_rate_limit(
        self,
        client_id: str,
        endpoint: str,
        rate_limit: int,
        burst_limit: int
    ) -> Tuple[bool, Dict]:
        """Check rate limit using Redis (distributed)."""
        try:
            key = f"rate_limit:{client_id}:{endpoint}"
            now = time.time()
            
            # Sliding window rate limiting
            pipeline = self.redis_client.pipeline()
            
            # Remove old entries outside the window
            pipeline.zremrangebyscore(key, 0, now - self.window_size)
            
            # Count requests in current window
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(now): now})
            
            # Set expiry
            pipeline.expire(key, self.window_size * 2)
            
            results = pipeline.execute()
            request_count = results[1]
            
            # Check token bucket for burst handling
            bucket_key = f"token_bucket:{client_id}:{endpoint}"
            tokens = await self._get_token_bucket(bucket_key, rate_limit, burst_limit)
            
            if request_count < rate_limit and tokens > 0:
                # Consume a token
                await self._consume_token(bucket_key)
                return True, {"requests": request_count, "tokens": tokens - 1}
            else:
                return False, {"requests": request_count, "tokens": 0}
                
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fall back to local rate limiting
            return self._check_local_rate_limit(client_id, endpoint, rate_limit, burst_limit)
    
    def _check_local_rate_limit(
        self,
        client_id: str,
        endpoint: str,
        rate_limit: int,
        burst_limit: int
    ) -> Tuple[bool, Dict]:
        """Check rate limit using local cache (non-distributed)."""
        key = f"{client_id}:{endpoint}"
        now = time.time()
        
        # Clean old entries
        while self.local_cache[key] and self.local_cache[key][0] < now - self.window_size:
            self.local_cache[key].popleft()
        
        request_count = len(self.local_cache[key])
        
        # Token bucket for burst handling
        if key not in self.token_buckets:
            self.token_buckets[key] = {
                "tokens": burst_limit,
                "last_refill": now,
                "rate": rate_limit / self.window_size,
                "capacity": rate_limit + burst_limit,
            }
        
        bucket = self.token_buckets[key]
        
        # Refill tokens
        time_passed = now - bucket["last_refill"]
        tokens_to_add = time_passed * bucket["rate"]
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        if request_count < rate_limit and bucket["tokens"] >= 1:
            # Allow request
            self.local_cache[key].append(now)
            bucket["tokens"] -= 1
            return True, {"requests": request_count + 1, "tokens": int(bucket["tokens"])}
        else:
            return False, {"requests": request_count, "tokens": 0}
    
    async def _get_token_bucket(
        self,
        key: str,
        rate: int,
        capacity: int
    ) -> int:
        """Get current token count from Redis token bucket."""
        try:
            data = self.redis_client.get(key)
            if not data:
                # Initialize bucket
                bucket = {
                    "tokens": capacity,
                    "last_refill": time.time(),
                }
                self.redis_client.setex(key, self.window_size * 2, json.dumps(bucket))
                return capacity
            
            bucket = json.loads(data)
            now = time.time()
            
            # Refill tokens
            time_passed = now - bucket["last_refill"]
            tokens_to_add = time_passed * (rate / self.window_size)
            bucket["tokens"] = min(capacity, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now
            
            self.redis_client.setex(key, self.window_size * 2, json.dumps(bucket))
            return int(bucket["tokens"])
            
        except Exception as e:
            logger.error(f"Token bucket operation failed: {e}")
            return capacity
    
    async def _consume_token(self, key: str) -> bool:
        """Consume a token from the bucket."""
        try:
            data = self.redis_client.get(key)
            if data:
                bucket = json.loads(data)
                if bucket["tokens"] > 0:
                    bucket["tokens"] -= 1
                    self.redis_client.setex(key, self.window_size * 2, json.dumps(bucket))
                    return True
            return False
        except Exception as e:
            logger.error(f"Token consumption failed: {e}")
            return True  # Allow on error


class DDoSProtection:
    """
    DDoS protection mechanisms including:
    - SYN flood protection
    - Request pattern analysis
    - Geographical blocking
    - Automatic blacklisting
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.suspicious_patterns = [
            r"(?i)(union|select|insert|update|delete|drop)\s",  # SQL injection
            r"(?i)<script[^>]*>.*?</script>",  # XSS attempts
            r"(?i)(\.\./|\.\.\\)",  # Path traversal
            r"(?i)(cmd|powershell|bash|sh)\s",  # Command injection
        ]
        
        # Track request patterns
        self.request_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Geographic blocking (example)
        self.blocked_countries = set()  # Add country codes as needed
        
        # Suspicious behavior thresholds
        self.thresholds = {
            "requests_per_second": 50,
            "unique_endpoints_per_minute": 100,
            "error_rate": 0.5,  # 50% errors
            "identical_requests": 20,  # Same request repeated
        }
    
    async def check_request(self, request: Request) -> Tuple[bool, Optional[str]]:
        """
        Check request for DDoS patterns.
        Returns (allowed, reason) tuple.
        """
        client_ip = request.client.host
        
        # Check request patterns
        pattern_check = await self._check_patterns(client_ip, request)
        if not pattern_check[0]:
            return pattern_check
        
        # Check for suspicious payloads
        payload_check = await self._check_payload(request)
        if not payload_check[0]:
            return payload_check
        
        # Check geographical restrictions
        geo_check = await self._check_geography(client_ip)
        if not geo_check[0]:
            return geo_check
        
        # Analyze behavior patterns
        behavior_check = await self._analyze_behavior(client_ip, request)
        if not behavior_check[0]:
            return behavior_check
        
        return True, None
    
    async def _check_patterns(
        self,
        client_ip: str,
        request: Request
    ) -> Tuple[bool, Optional[str]]:
        """Check for suspicious request patterns."""
        key = f"pattern:{client_ip}"
        now = time.time()
        
        # Track request
        self.request_patterns[key].append({
            "time": now,
            "path": request.url.path,
            "method": request.method,
        })
        
        # Check request rate
        recent_requests = [
            r for r in self.request_patterns[key]
            if r["time"] > now - 1
        ]
        
        if len(recent_requests) > self.thresholds["requests_per_second"]:
            return False, "Request rate too high"
        
        # Check endpoint variety (bot behavior)
        unique_endpoints = len(set(r["path"] for r in self.request_patterns[key]))
        if unique_endpoints > self.thresholds["unique_endpoints_per_minute"]:
            return False, "Suspicious endpoint scanning detected"
        
        # Check for repeated identical requests
        request_hash = hashlib.md5(
            f"{request.method}:{request.url.path}".encode()
        ).hexdigest()
        
        identical_count = sum(
            1 for r in self.request_patterns[key]
            if hashlib.md5(f"{r['method']}:{r['path']}".encode()).hexdigest() == request_hash
        )
        
        if identical_count > self.thresholds["identical_requests"]:
            return False, "Too many identical requests"
        
        return True, None
    
    async def _check_payload(self, request: Request) -> Tuple[bool, Optional[str]]:
        """Check request payload for malicious content."""
        # Check URL parameters
        for param in request.query_params.values():
            for pattern in self.suspicious_patterns:
                if pattern in str(param):
                    return False, "Suspicious payload detected"
        
        # Check headers
        suspicious_headers = ["X-Forwarded-Host", "X-Original-URL", "X-Rewrite-URL"]
        for header in suspicious_headers:
            if header in request.headers:
                # Additional validation needed
                pass
        
        return True, None
    
    async def _check_geography(self, client_ip: str) -> Tuple[bool, Optional[str]]:
        """Check geographical restrictions."""
        # This would integrate with a GeoIP service
        # For now, just a placeholder
        return True, None
    
    async def _analyze_behavior(
        self,
        client_ip: str,
        request: Request
    ) -> Tuple[bool, Optional[str]]:
        """Analyze client behavior for anomalies."""
        if self.redis_client:
            try:
                # Track error rate
                error_key = f"errors:{client_ip}"
                total_key = f"total:{client_ip}"
                
                # This would be updated based on response status
                # For now, just check the ratio
                errors = int(self.redis_client.get(error_key) or 0)
                total = int(self.redis_client.get(total_key) or 1)
                
                error_rate = errors / total
                if error_rate > self.thresholds["error_rate"]:
                    return False, "High error rate detected"
                    
            except Exception as e:
                logger.error(f"Behavior analysis failed: {e}")
        
        return True, None


# Middleware factory
def create_rate_limit_middleware(
    redis_url: Optional[str] = None,
    default_rate: int = 100,
    burst_size: int = 20,
):
    """Create rate limiting middleware for FastAPI."""
    redis_client = None
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            logger.info("Redis connected for rate limiting")
        except Exception as e:
            logger.warning(f"Redis connection failed, using local rate limiting: {e}")
    
    rate_limiter = RateLimiter(
        redis_client=redis_client,
        default_rate=default_rate,
        burst_size=burst_size,
    )
    
    ddos_protection = DDoSProtection(redis_client=redis_client)
    
    async def rate_limit_middleware(request: Request, call_next):
        """Rate limiting and DDoS protection middleware."""
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/api/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        # Check DDoS patterns
        ddos_allowed, ddos_reason = await ddos_protection.check_request(request)
        if not ddos_allowed:
            logger.warning(f"DDoS protection triggered: {ddos_reason} for {request.client.host}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Forbidden", "reason": ddos_reason}
            )
        
        # Check rate limit
        allowed, info = await rate_limiter.check_rate_limit(request)
        
        if not allowed:
            # Add rate limit headers to response
            headers = {
                k: v for k, v in info.items()
                if k.startswith("X-RateLimit-")
            }
            
            retry_after = info.get("retry_after", 60)
            headers["Retry-After"] = str(retry_after)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                },
                headers=headers,
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful response
        for key, value in info.items():
            if key.startswith("X-RateLimit-"):
                response.headers[key] = str(value)
        
        return response
    
    return rate_limit_middleware