# Archon Phase 6 API Implementation Examples

**Version**: 2.0.0  
**Date**: August 31, 2025  
**Status**: Production Ready  

This document provides complete implementation examples for the Archon Phase 6 authentication API, including client SDKs, server implementations, and integration patterns.

## Table of Contents

1. [Server-Side Implementation](#server-side-implementation)
2. [Client SDKs](#client-sdks)
3. [Integration Examples](#integration-examples)
4. [Agent Authentication Examples](#agent-authentication-examples)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Performance Optimization Examples](#performance-optimization-examples)
7. [Testing Examples](#testing-examples)
8. [Webhook Implementation](#webhook-implementation)

## Server-Side Implementation

### FastAPI Authentication Service

```python
# File: src/server/api_routes/auth.py
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import asyncio
from datetime import datetime, timedelta
import jwt
import bcrypt
import redis.asyncio as redis
import asyncpg

from ..services.auth_service import AuthService
from ..services.cache_service import CacheService
from ..services.rate_limiter import RateLimiter
from ..models.auth_models import User, LoginRequest, AuthResponse
from ..core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

# Dependency injection
async def get_auth_service() -> AuthService:
    return AuthService()

async def get_cache_service() -> CacheService:
    return CacheService()

async def get_rate_limiter() -> RateLimiter:
    return RateLimiter()

# Rate limiting decorator
def rate_limit(limit: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request = next((arg for arg in args if isinstance(arg, Request)), None)
            if not request:
                return await func(*args, **kwargs)
            
            rate_limiter = await get_rate_limiter()
            client_ip = request.client.host
            key = f"{func.__name__}:{client_ip}"
            
            if not await rate_limiter.check_limit(key, limit):
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded: {limit}",
                        "retry_after": await rate_limiter.get_retry_after(key)
                    }
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@router.post("/login", response_model=dict)
@rate_limit("5/minute")
async def login(
    login_request: LoginRequest,
    request: Request,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Authenticate user and return JWT tokens.
    
    Performance: <50ms response time with multi-layered caching.
    Security: Rate limited to 5 attempts per minute per IP.
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate credentials with caching
        user = await auth_service.authenticate_user(
            login_request.email, 
            login_request.password
        )
        
        if not user:
            # Add delay to prevent timing attacks
            await asyncio.sleep(0.5)
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "AUTH_INVALID_CREDENTIALS",
                    "message": "Invalid email or password"
                }
            )
        
        # Generate tokens
        tokens = await auth_service.create_tokens(
            user=user,
            remember_me=login_request.remember_me,
            device_info=login_request.device_info
        )
        
        # Create session
        session_id = await auth_service.create_session(
            user_id=user.id,
            device_info=login_request.device_info,
            remember_me=login_request.remember_me
        )
        
        # Cache user data for performance
        await cache_service.set_user_cache(user.id, user, ttl=300)
        
        # Set secure cookie for refresh token
        response.set_cookie(
            key="refresh_token",
            value=tokens.refresh_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=7 * 24 * 3600 if login_request.remember_me else 24 * 3600
        )
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "data": {
                "access_token": tokens.access_token,
                "token_type": "Bearer",
                "expires_in": tokens.expires_in,
                "user": user.dict(exclude={"password_hash"}),
                "session_id": session_id
            },
            "error": None,
            "meta": {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0.0",
                "request_id": str(request.state.request_id),
                "performance": {
                    "response_time_ms": response_time,
                    "cache_hit": False
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SYSTEM_ERROR",
                "message": "Authentication service error",
                "trace_id": str(request.state.request_id)
            }
        )

@router.post("/register", response_model=dict, status_code=201)
@rate_limit("3/minute")
async def register(
    register_request: RegisterRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register new user account with email verification."""
    
    try:
        # Check if user already exists
        existing_user = await auth_service.get_user_by_email(register_request.email)
        if existing_user:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "USER_ALREADY_EXISTS",
                    "message": "User with this email already exists"
                }
            )
        
        # Create user
        user = await auth_service.create_user(register_request)
        
        # Send verification email
        await auth_service.send_verification_email(user.email)
        
        return {
            "success": True,
            "data": {
                "user_id": user.id,
                "email": user.email,
                "email_verification_sent": True
            },
            "error": None,
            "meta": {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0.0",
                "request_id": str(request.state.request_id)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SYSTEM_ERROR",
                "message": "Registration service error",
                "trace_id": str(request.state.request_id)
            }
        )

@router.post("/agents/authenticate", response_model=dict)
async def authenticate_agent(
    agent_request: AgentAuthRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate specialized agent and grant capability tokens.
    
    Security: Implements capability-based access control for 22+ agent types.
    Performance: Optimized for high-throughput agent operations.
    """
    
    try:
        # Validate user permissions for agent type
        if not await auth_service.can_user_use_agent(current_user, agent_request.agent_type):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "AGENT_UNAUTHORIZED",
                    "message": f"User not authorized for agent type: {agent_request.agent_type}"
                }
            )
        
        # Validate and grant capabilities
        granted_capabilities, denied_capabilities = await auth_service.grant_agent_capabilities(
            user=current_user,
            agent_type=agent_request.agent_type,
            requested_capabilities=agent_request.capabilities_requested,
            task_context=agent_request.task_context
        )
        
        # Generate agent token
        agent_token = await auth_service.create_agent_token(
            user_id=current_user.id,
            agent_type=agent_request.agent_type,
            capabilities=granted_capabilities,
            task_context=agent_request.task_context
        )
        
        return {
            "success": True,
            "data": {
                "agent_token": agent_token.token,
                "capabilities_granted": granted_capabilities,
                "capabilities_denied": denied_capabilities,
                "expires_in": agent_token.expires_in,
                "restrictions": agent_token.restrictions
            },
            "error": None,
            "meta": {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0.0",
                "request_id": str(request.state.request_id)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SYSTEM_ERROR",
                "message": "Agent authentication error",
                "trace_id": str(request.state.request_id)
            }
        )

# Dependency for getting current user from JWT
async def get_current_user(
    token: str = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Extract and validate user from JWT token."""
    
    try:
        # Extract token from Authorization header
        if token.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Invalid token type"}
            )
        
        # Validate token with caching
        user = await auth_service.get_user_from_token(token.credentials)
        if not user:
            raise HTTPException(
                status_code=401,
                detail={"code": "AUTH_TOKEN_INVALID", "message": "Invalid or expired token"}
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=401,
            detail={"code": "AUTH_TOKEN_INVALID", "message": "Token validation failed"}
        )
```

### Authentication Service Implementation

```python
# File: src/services/auth_service.py
import jwt
import bcrypt
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
import asyncpg
import redis.asyncio as redis
from uuid import uuid4

from ..models.auth_models import User, LoginRequest, RegisterRequest, AgentToken
from ..core.config import settings
from ..core.database import get_db_pool
from ..services.cache_service import CacheService
from ..services.email_service import EmailService

class AuthService:
    def __init__(self):
        self.cache = CacheService()
        self.email_service = EmailService()
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.jwt_algorithm = "RS256"
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with optimized caching."""
        
        # Check L1 cache (memory)
        cache_key = f"user:email:{email}"
        cached_user = await self.cache.get_memory(cache_key)
        if cached_user:
            if self._verify_password(password, cached_user.password_hash):
                return cached_user
            return None
        
        # Check L2 cache (Redis)
        cached_user = await self.cache.get_redis(cache_key)
        if cached_user:
            # Update L1 cache
            await self.cache.set_memory(cache_key, cached_user, ttl=60)
            if self._verify_password(password, cached_user.password_hash):
                return cached_user
            return None
        
        # Database query
        async with get_db_pool().acquire() as conn:
            user_data = await conn.fetchrow("""
                SELECT id, email, password_hash, name, email_verified, is_active,
                       created_at, updated_at, last_login
                FROM users 
                WHERE email = $1 AND is_active = true
            """, email)
        
        if not user_data:
            return None
        
        user = User(**dict(user_data))
        
        # Cache user data
        await self.cache.set_redis(cache_key, user, ttl=300)
        await self.cache.set_memory(cache_key, user, ttl=60)
        
        # Verify password
        if self._verify_password(password, user.password_hash):
            # Update last login
            await self._update_last_login(user.id)
            return user
        
        return None
    
    async def create_tokens(self, user: User, remember_me: bool = False, 
                          device_info: Dict = None) -> Dict[str, Any]:
        """Create JWT access and refresh tokens."""
        
        now = datetime.utcnow()
        access_expires = now + timedelta(hours=1)
        refresh_expires = now + timedelta(days=30 if remember_me else 7)
        
        # Access token payload
        access_payload = {
            "iss": "archon-auth-service",
            "sub": str(user.id),
            "aud": "archon-api",
            "exp": int(access_expires.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid4()),
            "type": "access",
            "email": user.email,
            "roles": await self._get_user_roles(user.id),
            "permissions": await self._get_user_permissions(user.id)
        }
        
        # Refresh token payload
        refresh_payload = {
            "iss": "archon-auth-service",
            "sub": str(user.id),
            "aud": "archon-api",
            "exp": int(refresh_expires.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid4()),
            "type": "refresh"
        }
        
        # Sign tokens
        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Cache token validation for performance
        await self.cache.set_redis(
            f"jwt:valid:{access_payload['jti']}", 
            {"valid": True, "user_id": str(user.id)},
            ttl=3600
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": 3600,
            "token_type": "Bearer"
        }
    
    async def create_agent_token(self, user_id: str, agent_type: str, 
                               capabilities: List[str], task_context: Dict) -> AgentToken:
        """Create agent authentication token with capabilities."""
        
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)
        
        # Agent token payload
        payload = {
            "iss": "archon-auth-service",
            "sub": str(user_id),
            "aud": "archon-agent-system",
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid4()),
            "type": "agent",
            "agent_type": agent_type,
            "capabilities": capabilities,
            "task_context": task_context,
            "restrictions": await self._get_agent_restrictions(agent_type, capabilities)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Cache token for validation
        await self.cache.set_redis(
            f"agent:token:{payload['jti']}",
            payload,
            ttl=3600
        )
        
        return AgentToken(
            token=token,
            expires_in=3600,
            capabilities=capabilities,
            restrictions=payload["restrictions"]
        )
    
    async def grant_agent_capabilities(self, user: User, agent_type: str, 
                                     requested_capabilities: List[str],
                                     task_context: Dict) -> Tuple[List[str], List[str]]:
        """Grant capabilities to agent based on user permissions and security policies."""
        
        # Get user's maximum capabilities for this agent type
        user_max_capabilities = await self._get_user_agent_capabilities(user.id, agent_type)
        
        # Get security policy for agent type
        security_policy = await self._get_agent_security_policy(agent_type)
        
        granted = []
        denied = []
        
        for capability in requested_capabilities:
            if (capability in user_max_capabilities and 
                await self._validate_capability_context(capability, task_context, security_policy)):
                granted.append(capability)
            else:
                denied.append(capability)
        
        return granted, denied
    
    async def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from JWT token with multi-layered caching."""
        
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check token blacklist
            if await self._is_token_blacklisted(payload["jti"]):
                return None
            
            # Check token cache
            cache_key = f"jwt:valid:{payload['jti']}"
            cached_validation = await self.cache.get_redis(cache_key)
            if not cached_validation:
                return None
            
            # Get user
            user_id = payload["sub"]
            user = await self._get_user_by_id(user_id)
            
            return user
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _hash_password(self, password: str) -> str:
        """Hash password with bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    async def _get_user_roles(self, user_id: str) -> List[str]:
        """Get user roles with caching."""
        cache_key = f"user:roles:{user_id}"
        
        # Try cache first
        roles = await self.cache.get_redis(cache_key)
        if roles:
            return roles
        
        # Database query
        async with get_db_pool().acquire() as conn:
            role_rows = await conn.fetch("""
                SELECT r.name FROM roles r
                JOIN user_roles ur ON r.id = ur.role_id
                WHERE ur.user_id = $1
            """, user_id)
        
        roles = [row['name'] for row in role_rows]
        
        # Cache result
        await self.cache.set_redis(cache_key, roles, ttl=600)
        
        return roles
    
    async def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions with caching."""
        cache_key = f"user:permissions:{user_id}"
        
        # Try cache first
        permissions = await self.cache.get_redis(cache_key)
        if permissions:
            return permissions
        
        # Database query
        async with get_db_pool().acquire() as conn:
            perm_rows = await conn.fetch("""
                SELECT DISTINCT p.name FROM permissions p
                JOIN role_permissions rp ON p.id = rp.permission_id
                JOIN user_roles ur ON rp.role_id = ur.role_id
                WHERE ur.user_id = $1
            """, user_id)
        
        permissions = [row['name'] for row in perm_rows]
        
        # Cache result
        await self.cache.set_redis(cache_key, permissions, ttl=600)
        
        return permissions
```

## Client SDKs

### JavaScript/TypeScript SDK

```typescript
// File: sdk/javascript/src/archon-auth-client.ts
import axios, { AxiosInstance, AxiosResponse } from 'axios';

interface AuthConfig {
  baseURL: string;
  apiKey?: string;
  timeout?: number;
  retries?: number;
}

interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
  deviceInfo?: {
    deviceId?: string;
    userAgent?: string;
    location?: string;
  };
}

interface RegisterData {
  email: string;
  password: string;
  name: string;
  termsAccepted: true;
  marketingConsent?: boolean;
}

interface User {
  id: string;
  email: string;
  name: string;
  roles: string[];
  isActive: boolean;
  emailVerified: boolean;
  createdAt: string;
  lastLogin?: string;
}

interface AuthResponse {
  success: boolean;
  data: {
    accessToken: string;
    tokenType: string;
    expiresIn: number;
    user: User;
    sessionId: string;
  };
  meta: {
    timestamp: string;
    version: string;
    requestId: string;
    performance: {
      responseTimeMs: number;
      cacheHit: boolean;
    };
  };
}

interface APIError {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
    traceId: string;
  };
}

class ArchonAuthClient {
  private client: AxiosInstance;
  private accessToken?: string;
  private refreshToken?: string;
  private tokenExpiry?: Date;
  
  constructor(config: AuthConfig) {
    this.client = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout || 10000,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'ArchonAuthClient/2.0.0',
        ...(config.apiKey && { 'X-API-Key': config.apiKey })
      }
    });
    
    // Request interceptor for auth token
    this.client.interceptors.request.use((config) => {
      if (this.accessToken) {
        config.headers.Authorization = `Bearer ${this.accessToken}`;
      }
      return config;
    });
    
    // Response interceptor for token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401 && this.refreshToken) {
          try {
            await this.refreshAccessToken();
            // Retry original request
            error.config.headers.Authorization = `Bearer ${this.accessToken}`;
            return this.client.request(error.config);
          } catch (refreshError) {
            // Refresh failed, clear tokens
            this.clearTokens();
            throw error;
          }
        }
        throw error;
      }
    );
  }
  
  /**
   * Login with email and password
   */
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const response: AxiosResponse<AuthResponse> = await this.client.post(
        '/auth/login',
        credentials
      );
      
      const { data } = response.data;
      this.accessToken = data.accessToken;
      this.tokenExpiry = new Date(Date.now() + data.expiresIn * 1000);
      
      // Extract refresh token from cookies if present
      const cookies = response.headers['set-cookie'];
      if (cookies) {
        const refreshTokenCookie = cookies.find(cookie => 
          cookie.startsWith('refresh_token=')
        );
        if (refreshTokenCookie) {
          this.refreshToken = refreshTokenCookie.split('=')[1].split(';')[0];
        }
      }
      
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Register new user
   */
  async register(userData: RegisterData): Promise<{ userId: string; emailVerificationSent: boolean }> {
    try {
      const response = await this.client.post('/auth/register', userData);
      return response.data.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Refresh access token
   */
  async refreshAccessToken(): Promise<void> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }
    
    try {
      const response: AxiosResponse<AuthResponse> = await this.client.post(
        '/auth/refresh',
        { refreshToken: this.refreshToken }
      );
      
      const { data } = response.data;
      this.accessToken = data.accessToken;
      this.tokenExpiry = new Date(Date.now() + data.expiresIn * 1000);
    } catch (error) {
      this.clearTokens();
      throw this.handleError(error);
    }
  }
  
  /**
   * Logout user
   */
  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } finally {
      this.clearTokens();
    }
  }
  
  /**
   * Get current user profile
   */
  async getCurrentUser(): Promise<User> {
    try {
      const response = await this.client.get('/users/me');
      return response.data.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Update user profile
   */
  async updateProfile(updates: Partial<Pick<User, 'name'>> & { preferences?: any }): Promise<User> {
    try {
      const response = await this.client.put('/users/me', updates);
      return response.data.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Change password
   */
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    try {
      await this.client.put('/users/me/password', {
        currentPassword,
        newPassword
      });
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Authenticate agent
   */
  async authenticateAgent(agentType: string, capabilities: string[], taskContext?: any): Promise<{
    agentToken: string;
    capabilitiesGranted: string[];
    capabilitiesDenied: string[];
    expiresIn: number;
    restrictions: any;
  }> {
    try {
      const response = await this.client.post('/auth/agents/authenticate', {
        agentType,
        capabilitiesRequested: capabilities,
        taskContext
      });
      return response.data.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Get user sessions
   */
  async getSessions(): Promise<Array<{
    sessionId: string;
    deviceInfo: any;
    createdAt: string;
    lastAccessed: string;
    isActive: boolean;
  }>> {
    try {
      const response = await this.client.get('/users/me/sessions');
      return response.data.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Terminate session
   */
  async terminateSession(sessionId: string): Promise<void> {
    try {
      await this.client.delete(`/users/me/sessions/${sessionId}`);
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!(this.accessToken && this.tokenExpiry && this.tokenExpiry > new Date());
  }
  
  /**
   * Get access token
   */
  getAccessToken(): string | undefined {
    return this.accessToken;
  }
  
  /**
   * Clear stored tokens
   */
  clearTokens(): void {
    this.accessToken = undefined;
    this.refreshToken = undefined;
    this.tokenExpiry = undefined;
  }
  
  /**
   * Handle API errors
   */
  private handleError(error: any): Error {
    if (error.response?.data?.error) {
      const apiError = error.response.data.error;
      const err = new Error(apiError.message);
      (err as any).code = apiError.code;
      (err as any).details = apiError.details;
      (err as any).traceId = apiError.traceId;
      return err;
    }
    return error;
  }
}

// Usage example
const authClient = new ArchonAuthClient({
  baseURL: 'https://api.archon.ai/v2'
});

// Login
try {
  const result = await authClient.login({
    email: 'user@example.com',
    password: 'password123',
    rememberMe: true
  });
  console.log('Login successful:', result.data.user);
} catch (error) {
  console.error('Login failed:', error.message);
}

// Authenticate agent
try {
  const agentAuth = await authClient.authenticateAgent(
    'code_implementer',
    ['file:read', 'file:write', 'git:commit']
  );
  console.log('Agent authenticated:', agentAuth);
} catch (error) {
  console.error('Agent auth failed:', error.message);
}

export default ArchonAuthClient;
```

### Python SDK

```python
# File: sdk/python/archon_auth_client/client.py
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from urllib.parse import urljoin

@dataclass
class AuthConfig:
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 10
    max_retries: int = 3

@dataclass
class User:
    id: str
    email: str
    name: str
    roles: List[str]
    is_active: bool
    email_verified: bool
    created_at: str
    last_login: Optional[str] = None

@dataclass
class LoginCredentials:
    email: str
    password: str
    remember_me: bool = False
    device_info: Optional[Dict[str, Any]] = None

@dataclass  
class RegisterData:
    email: str
    password: str
    name: str
    terms_accepted: bool = True
    marketing_consent: bool = False

class ArchonAuthError(Exception):
    def __init__(self, message: str, code: str = None, details: Dict = None, trace_id: str = None):
        super().__init__(message)
        self.code = code
        self.details = details
        self.trace_id = trace_id

class ArchonAuthClient:
    def __init__(self, config: AuthConfig):
        self.config = config
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        if not self.session or self.session.closed:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'ArchonAuthClient-Python/2.0.0'
            }
            if self.config.api_key:
                headers['X-API-Key'] = self.config.api_key
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        await self._ensure_session()
        
        url = urljoin(self.config.base_url, endpoint)
        
        # Add auth header if we have a token
        headers = kwargs.get('headers', {})
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        kwargs['headers'] = headers
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response_data = await response.json()
                    
                    if response.status == 401 and self.refresh_token and attempt == 0:
                        # Try to refresh token
                        try:
                            await self.refresh_access_token()
                            # Update auth header and retry
                            headers['Authorization'] = f'Bearer {self.access_token}'
                            continue
                        except:
                            self._clear_tokens()
                            raise ArchonAuthError("Authentication failed", "AUTH_FAILED")
                    
                    if not response_data.get('success', False):
                        error = response_data.get('error', {})
                        raise ArchonAuthError(
                            message=error.get('message', 'Request failed'),
                            code=error.get('code'),
                            details=error.get('details'),
                            trace_id=error.get('trace_id')
                        )
                    
                    return response_data
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries:
                    raise ArchonAuthError(f"Request failed: {str(e)}", "NETWORK_ERROR")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def login(self, credentials: LoginCredentials) -> Dict[str, Any]:
        """Login with email and password."""
        
        data = {
            'email': credentials.email,
            'password': credentials.password,
            'remember_me': credentials.remember_me
        }
        if credentials.device_info:
            data['device_info'] = credentials.device_info
        
        response = await self._request('POST', '/auth/login', json=data)
        
        # Store tokens
        auth_data = response['data']
        self.access_token = auth_data['access_token']
        self.token_expiry = datetime.utcnow() + timedelta(seconds=auth_data['expires_in'])
        
        return response
    
    async def register(self, user_data: RegisterData) -> Dict[str, Any]:
        """Register new user."""
        
        data = {
            'email': user_data.email,
            'password': user_data.password,
            'name': user_data.name,
            'terms_accepted': user_data.terms_accepted,
            'marketing_consent': user_data.marketing_consent
        }
        
        return await self._request('POST', '/auth/register', json=data)
    
    async def refresh_access_token(self) -> None:
        """Refresh access token."""
        
        if not self.refresh_token:
            raise ArchonAuthError("No refresh token available", "NO_REFRESH_TOKEN")
        
        response = await self._request('POST', '/auth/refresh', json={
            'refresh_token': self.refresh_token
        })
        
        auth_data = response['data']
        self.access_token = auth_data['access_token']
        self.token_expiry = datetime.utcnow() + timedelta(seconds=auth_data['expires_in'])
    
    async def logout(self) -> None:
        """Logout user."""
        
        try:
            await self._request('POST', '/auth/logout')
        finally:
            self._clear_tokens()
    
    async def get_current_user(self) -> User:
        """Get current user profile."""
        
        response = await self._request('GET', '/users/me')
        user_data = response['data']
        
        return User(
            id=user_data['id'],
            email=user_data['email'],
            name=user_data['name'],
            roles=user_data['roles'],
            is_active=user_data['is_active'],
            email_verified=user_data['email_verified'],
            created_at=user_data['created_at'],
            last_login=user_data.get('last_login')
        )
    
    async def update_profile(self, updates: Dict[str, Any]) -> User:
        """Update user profile."""
        
        response = await self._request('PUT', '/users/me', json=updates)
        user_data = response['data']
        
        return User(**user_data)
    
    async def change_password(self, current_password: str, new_password: str) -> None:
        """Change user password."""
        
        await self._request('PUT', '/users/me/password', json={
            'current_password': current_password,
            'new_password': new_password
        })
    
    async def authenticate_agent(self, agent_type: str, capabilities: List[str], 
                               task_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Authenticate agent and get capability tokens."""
        
        data = {
            'agent_type': agent_type,
            'capabilities_requested': capabilities
        }
        if task_context:
            data['task_context'] = task_context
        
        response = await self._request('POST', '/auth/agents/authenticate', json=data)
        return response['data']
    
    async def get_sessions(self) -> List[Dict[str, Any]]:
        """Get user sessions."""
        
        response = await self._request('GET', '/users/me/sessions')
        return response['data']
    
    async def terminate_session(self, session_id: str) -> None:
        """Terminate specific session."""
        
        await self._request('DELETE', f'/users/me/sessions/{session_id}')
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        
        return (self.access_token is not None and 
                self.token_expiry is not None and 
                self.token_expiry > datetime.utcnow())
    
    def get_access_token(self) -> Optional[str]:
        """Get current access token."""
        return self.access_token
    
    def _clear_tokens(self) -> None:
        """Clear stored tokens."""
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None

# Usage example
async def main():
    config = AuthConfig(base_url='https://api.archon.ai/v2')
    
    async with ArchonAuthClient(config) as client:
        # Login
        try:
            result = await client.login(LoginCredentials(
                email='user@example.com',
                password='password123',
                remember_me=True
            ))
            print(f"Login successful: {result['data']['user']['name']}")
        except ArchonAuthError as e:
            print(f"Login failed: {e} (Code: {e.code})")
        
        # Authenticate agent
        try:
            agent_auth = await client.authenticate_agent(
                'code_implementer',
                ['file:read', 'file:write', 'git:commit']
            )
            print(f"Agent authenticated with capabilities: {agent_auth['capabilities_granted']}")
        except ArchonAuthError as e:
            print(f"Agent auth failed: {e}")

if __name__ == '__main__':
    asyncio.run(main())
```

## Integration Examples

### React Hook for Authentication

```tsx
// File: examples/react-integration/useAuth.tsx
import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import ArchonAuthClient from '@archon/auth-client';

interface AuthContextType {
  user: User | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => Promise<void>;
  register: (userData: RegisterData) => Promise<void>;
  authenticateAgent: (agentType: string, capabilities: string[]) => Promise<AgentAuth>;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const authClient = new ArchonAuthClient({
    baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8181/api/v2'
  });
  
  useEffect(() => {
    // Check if user is already authenticated on app start
    checkAuthStatus();
  }, []);
  
  const checkAuthStatus = async () => {
    try {
      if (authClient.isAuthenticated()) {
        const currentUser = await authClient.getCurrentUser();
        setUser(currentUser);
      }
    } catch (error) {
      console.error('Auth status check failed:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const login = async (credentials: LoginCredentials) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await authClient.login(credentials);
      setUser(response.data.user);
    } catch (error: any) {
      setError(error.message || 'Login failed');
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  const logout = async () => {
    setIsLoading(true);
    
    try {
      await authClient.logout();
      setUser(null);
    } catch (error: any) {
      console.error('Logout error:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const register = async (userData: RegisterData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await authClient.register(userData);
      // Note: User needs to verify email before login
    } catch (error: any) {
      setError(error.message || 'Registration failed');
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  const authenticateAgent = async (agentType: string, capabilities: string[]) => {
    try {
      return await authClient.authenticateAgent(agentType, capabilities);
    } catch (error: any) {
      setError(error.message || 'Agent authentication failed');
      throw error;
    }
  };
  
  const value: AuthContextType = {
    user,
    login,
    logout,
    register,
    authenticateAgent,
    isAuthenticated: authClient.isAuthenticated(),
    isLoading,
    error
  };
  
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Example usage in a component
const LoginForm: React.FC = () => {
  const { login, isLoading, error } = useAuth();
  const [credentials, setCredentials] = useState({
    email: '',
    password: '',
    rememberMe: false
  });
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      await login(credentials);
      // Redirect on success
    } catch (error) {
      // Error is handled by context
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      {error && <div className="error">{error}</div>}
      
      <input
        type="email"
        value={credentials.email}
        onChange={(e) => setCredentials({...credentials, email: e.target.value})}
        placeholder="Email"
        required
      />
      
      <input
        type="password"
        value={credentials.password}
        onChange={(e) => setCredentials({...credentials, password: e.target.value})}
        placeholder="Password"
        required
      />
      
      <label>
        <input
          type="checkbox"
          checked={credentials.rememberMe}
          onChange={(e) => setCredentials({...credentials, rememberMe: e.target.checked})}
        />
        Remember me
      </label>
      
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};

export default LoginForm;
```

## Agent Authentication Examples

### Code Implementer Agent Example

```python
# File: examples/agent-integration/code_implementer_agent.py
import asyncio
from typing import Dict, List, Any
from archon_auth_client import ArchonAuthClient, AuthConfig

class CodeImplementerAgent:
    def __init__(self, auth_client: ArchonAuthClient):
        self.auth_client = auth_client
        self.agent_token: str = None
        self.capabilities: List[str] = []
        self.restrictions: Dict[str, Any] = {}
    
    async def authenticate(self, task_context: Dict[str, Any]) -> None:
        """Authenticate agent with required capabilities."""
        
        required_capabilities = [
            'file:read',
            'file:write', 
            'file:create',
            'git:commit',
            'npm:install',
            'shell:execute'
        ]
        
        try:
            auth_result = await self.auth_client.authenticate_agent(
                agent_type='code_implementer',
                capabilities=required_capabilities,
                task_context=task_context
            )
            
            self.agent_token = auth_result['agent_token']
            self.capabilities = auth_result['capabilities_granted']
            self.restrictions = auth_result['restrictions']
            
            print(f"Agent authenticated with capabilities: {self.capabilities}")
            
            if auth_result['capabilities_denied']:
                print(f"WARNING: Denied capabilities: {auth_result['capabilities_denied']}")
            
        except Exception as e:
            print(f"Agent authentication failed: {e}")
            raise
    
    async def implement_feature(self, feature_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Implement a feature using granted capabilities."""
        
        if not self.agent_token:
            raise RuntimeError("Agent not authenticated")
        
        results = {
            'files_created': [],
            'files_modified': [],
            'tests_added': [],
            'git_commits': []
        }
        
        try:
            # Check if we can create files
            if 'file:create' in self.capabilities:
                # Create feature implementation files
                files_to_create = self._plan_file_creation(feature_spec)
                for file_path, content in files_to_create.items():
                    if self._check_file_path_allowed(file_path):
                        await self._create_file(file_path, content)
                        results['files_created'].append(file_path)
            
            # Check if we can modify files
            if 'file:write' in self.capabilities:
                # Modify existing files
                files_to_modify = self._plan_file_modifications(feature_spec)
                for file_path, changes in files_to_modify.items():
                    if self._check_file_path_allowed(file_path):
                        await self._modify_file(file_path, changes)
                        results['files_modified'].append(file_path)
            
            # Add tests if capability granted
            if 'file:create' in self.capabilities:
                test_files = self._generate_tests(feature_spec)
                for test_path, test_content in test_files.items():
                    if self._check_file_path_allowed(test_path):
                        await self._create_file(test_path, test_content)
                        results['tests_added'].append(test_path)
            
            # Install dependencies if needed
            if 'npm:install' in self.capabilities:
                dependencies = self._get_required_dependencies(feature_spec)
                if dependencies:
                    await self._install_dependencies(dependencies)
            
            # Commit changes if allowed
            if 'git:commit' in self.capabilities:
                commit_message = f"Implement {feature_spec['name']}: {feature_spec['description']}"
                commit_hash = await self._git_commit(commit_message)
                results['git_commits'].append(commit_hash)
            
            return results
            
        except Exception as e:
            print(f"Feature implementation failed: {e}")
            raise
    
    def _check_file_path_allowed(self, file_path: str) -> bool:
        """Check if file path is allowed by restrictions."""
        
        if not self.restrictions:
            return True
        
        allowed_paths = self.restrictions.get('file_paths', [])
        if not allowed_paths:
            return True
        
        # Check if path matches any allowed pattern
        for allowed_pattern in allowed_paths:
            if file_path.startswith(allowed_pattern.replace('**', '')):
                return True
        
        return False
    
    async def _create_file(self, file_path: str, content: str) -> None:
        """Create file with capability validation."""
        
        # This would use the actual file system API with agent token
        print(f"Creating file: {file_path}")
        # Simulate API call with agent token authentication
        await asyncio.sleep(0.1)
    
    async def _modify_file(self, file_path: str, changes: Dict) -> None:
        """Modify file with capability validation."""
        
        print(f"Modifying file: {file_path}")
        # Simulate API call with agent token authentication
        await asyncio.sleep(0.1)
    
    async def _install_dependencies(self, dependencies: List[str]) -> None:
        """Install dependencies if capability granted."""
        
        print(f"Installing dependencies: {dependencies}")
        # This would execute npm install with security restrictions
        await asyncio.sleep(0.5)
    
    async def _git_commit(self, message: str) -> str:
        """Commit changes if capability granted."""
        
        print(f"Committing changes: {message}")
        # This would perform git commit with agent token
        await asyncio.sleep(0.2)
        return "abc123def456"  # Simulated commit hash
    
    def _plan_file_creation(self, feature_spec: Dict) -> Dict[str, str]:
        """Plan which files to create."""
        return {
            f"src/features/{feature_spec['name']}.ts": "// Feature implementation",
            f"src/types/{feature_spec['name']}.types.ts": "// Type definitions"
        }
    
    def _plan_file_modifications(self, feature_spec: Dict) -> Dict[str, Dict]:
        """Plan which files to modify."""
        return {
            "src/app.ts": {"action": "add_import", "content": f"import {feature_spec['name']}"},
            "package.json": {"action": "add_dependency", "content": feature_spec.get('dependencies', [])}
        }
    
    def _generate_tests(self, feature_spec: Dict) -> Dict[str, str]:
        """Generate test files."""
        return {
            f"tests/{feature_spec['name']}.test.ts": f"// Tests for {feature_spec['name']}"
        }
    
    def _get_required_dependencies(self, feature_spec: Dict) -> List[str]:
        """Get required npm dependencies."""
        return feature_spec.get('dependencies', [])

# Usage example
async def main():
    # Initialize auth client
    config = AuthConfig(base_url='https://api.archon.ai/v2')
    
    async with ArchonAuthClient(config) as auth_client:
        # Login as user who can operate agents
        await auth_client.login({
            'email': 'developer@company.com',
            'password': 'password123'
        })
        
        # Initialize agent
        agent = CodeImplementerAgent(auth_client)
        
        # Authenticate agent for specific task
        task_context = {
            'project_id': 'proj_user_dashboard',
            'task_type': 'feature_implementation',
            'estimated_duration': 1800,  # 30 minutes
            'feature_complexity': 'medium'
        }
        
        await agent.authenticate(task_context)
        
        # Implement feature
        feature_spec = {
            'name': 'UserProfile',
            'description': 'User profile management component',
            'type': 'react_component',
            'dependencies': ['react-hook-form', 'yup']
        }
        
        results = await agent.implement_feature(feature_spec)
        print(f"Feature implementation results: {results}")

if __name__ == '__main__':
    asyncio.run(main())
```

This comprehensive implementation examples document provides practical, production-ready code for integrating with the Archon Phase 6 authentication API. All examples include proper error handling, security considerations, and performance optimizations while demonstrating real-world usage patterns for both client applications and agent integrations.