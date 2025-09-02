# Archon Authentication Integration Guide

## Overview

This guide provides comprehensive instructions for integrating with the Archon Authentication API. It includes code examples, best practices, and common integration patterns for various client types.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication Flows](#authentication-flows)
3. [Client Libraries](#client-libraries)
4. [Framework Integrations](#framework-integrations)
5. [Mobile Applications](#mobile-applications)
6. [Single Page Applications (SPA)](#single-page-applications-spa)
7. [Server-to-Server](#server-to-server)
8. [Testing Integration](#testing-integration)
9. [Error Handling](#error-handling)
10. [Best Practices](#best-practices)

---

## Quick Start

### 1. Environment Setup

First, ensure you have access to the Archon Authentication API:

```bash
# Development environment
export ARCHON_AUTH_URL="http://localhost:8181"

# Production environment  
export ARCHON_AUTH_URL="https://api.archon.ai"
```

### 2. Basic Authentication Flow

```python
import httpx
import os

class ArchonAuth:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("ARCHON_AUTH_URL", "http://localhost:8181")
        self.client = httpx.AsyncClient(base_url=self.base_url)
        self.access_token = None
        self.refresh_token = None

    async def register(self, email: str, password: str, name: str):
        """Register a new user."""
        response = await self.client.post("/auth/register", json={
            "email": email,
            "password": password,
            "name": name
        })
        
        if response.status_code == 201:
            data = response.json()
            self.access_token = data["tokens"]["access_token"]
            self.refresh_token = data["tokens"]["refresh_token"]
            return data["user"]
        else:
            raise Exception(f"Registration failed: {response.text}")

    async def login(self, email: str, password: str):
        """Login with email and password."""
        response = await self.client.post("/auth/login", json={
            "email": email,
            "password": password,
            "remember_me": False
        })
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            return data
        else:
            raise Exception(f"Login failed: {response.text}")

    def get_auth_headers(self):
        """Get headers for authenticated requests."""
        return {"Authorization": f"Bearer {self.access_token}"}

# Usage example
async def main():
    auth = ArchonAuth()
    
    # Register new user
    user = await auth.register(
        email="user@example.com",
        password="SecurePass123!",
        name="John Doe"
    )
    print(f"Registered user: {user['email']}")
    
    # Make authenticated request
    headers = auth.get_auth_headers()
    response = await auth.client.get("/auth/me", headers=headers)
    profile = response.json()
    print(f"User profile: {profile}")
```

---

## Authentication Flows

### Standard Email/Password Flow

```python
class StandardAuthFlow:
    def __init__(self, auth_client):
        self.auth = auth_client
    
    async def authenticate_user(self, email: str, password: str):
        """Complete authentication flow with error handling."""
        try:
            # Attempt login
            tokens = await self.auth.login(email, password)
            
            # Store tokens securely (see security section)
            await self.store_tokens(tokens)
            
            # Get user profile
            profile = await self.get_user_profile()
            
            return {
                "success": True,
                "user": profile,
                "tokens": tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def store_tokens(self, tokens):
        """Store tokens securely (implement based on your needs)."""
        # For web apps: secure HTTP-only cookies
        # For mobile: secure keychain/keystore
        # For desktop: encrypted local storage
        pass
```

### OAuth2 Flow Implementation

```python
import secrets
from urllib.parse import urlencode

class OAuth2Flow:
    def __init__(self, auth_client, redirect_uri: str):
        self.auth = auth_client
        self.redirect_uri = redirect_uri
        self.state_store = {}  # In production, use Redis or database
    
    async def initiate_oauth(self, provider: str):
        """Start OAuth2 flow."""
        # Get authorization URL
        response = await self.auth.client.get(
            f"/auth/oauth/{provider}/authorize",
            params={"redirect_uri": self.redirect_uri}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store state for validation
            state = data["state"]
            self.state_store[state] = {
                "provider": provider,
                "timestamp": time.time()
            }
            
            return data["authorization_url"]
        else:
            raise Exception(f"OAuth initiation failed: {response.text}")
    
    async def handle_callback(self, provider: str, code: str, state: str):
        """Handle OAuth2 callback."""
        # Validate state
        if state not in self.state_store:
            raise Exception("Invalid state parameter")
        
        stored_state = self.state_store.pop(state)
        if stored_state["provider"] != provider:
            raise Exception("Provider mismatch")
        
        # Exchange code for tokens
        response = await self.auth.client.get(
            f"/auth/oauth/{provider}/callback",
            params={"code": code, "state": state}
        )
        
        if response.status_code == 200:
            tokens = response.json()
            await self.store_tokens(tokens)
            return tokens
        else:
            raise Exception(f"OAuth callback failed: {response.text}")

# Usage with Flask
from flask import Flask, request, redirect

app = Flask(__name__)
oauth = OAuth2Flow(auth_client, "http://localhost:5000/oauth/callback")

@app.route("/login/<provider>")
async def oauth_login(provider):
    auth_url = await oauth.initiate_oauth(provider)
    return redirect(auth_url)

@app.route("/oauth/callback")
async def oauth_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    provider = request.args.get("provider")  # You'll need to pass this
    
    try:
        tokens = await oauth.handle_callback(provider, code, state)
        # Store tokens and redirect to app
        return redirect("/dashboard")
    except Exception as e:
        return f"OAuth failed: {str(e)}", 400
```

### Token Refresh Flow

```python
class TokenManager:
    def __init__(self, auth_client):
        self.auth = auth_client
    
    async def refresh_access_token(self, refresh_token: str):
        """Refresh expired access token."""
        response = await self.auth.client.post("/auth/refresh", json={
            "refresh_token": refresh_token
        })
        
        if response.status_code == 200:
            tokens = response.json()
            await self.store_tokens(tokens)
            return tokens
        else:
            # Refresh failed - user needs to login again
            await self.clear_tokens()
            raise Exception("Refresh failed - please login again")
    
    async def make_authenticated_request(self, method: str, url: str, **kwargs):
        """Make request with automatic token refresh."""
        headers = kwargs.get("headers", {})
        headers.update(self.get_auth_headers())
        kwargs["headers"] = headers
        
        response = await self.auth.client.request(method, url, **kwargs)
        
        if response.status_code == 401:
            # Token expired, try to refresh
            try:
                await self.refresh_access_token(self.refresh_token)
                # Retry request with new token
                headers.update(self.get_auth_headers())
                kwargs["headers"] = headers
                response = await self.auth.client.request(method, url, **kwargs)
            except Exception:
                # Refresh failed, redirect to login
                raise Exception("Authentication required")
        
        return response
```

---

## Client Libraries

### Python Client Library

```python
# archon_auth_client.py
import asyncio
import httpx
import jwt
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

class ArchonAuthClient:
    """Comprehensive Python client for Archon Authentication API."""
    
    def __init__(
        self, 
        base_url: str,
        client_id: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout
        )
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    # Authentication Methods
    async def register(
        self,
        email: str,
        password: str,
        name: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Register a new user."""
        response = await self.client.post("/auth/register", json={
            "email": email,
            "password": password,
            "name": name,
            "metadata": metadata or {}
        })
        
        self._handle_response(response)
        data = response.json()
        self._store_tokens(data.get("tokens"))
        return data
    
    async def login(self, email: str, password: str, remember_me: bool = False):
        """Login with credentials."""
        response = await self.client.post("/auth/login", json={
            "email": email,
            "password": password,
            "remember_me": remember_me
        })
        
        self._handle_response(response)
        tokens = response.json()
        self._store_tokens(tokens)
        return tokens
    
    async def logout(self):
        """Logout and invalidate tokens."""
        if not self.access_token:
            return
        
        try:
            await self.client.post(
                "/auth/logout",
                headers=self._get_auth_headers()
            )
        finally:
            self._clear_tokens()
    
    async def refresh_tokens(self):
        """Refresh access token."""
        if not self.refresh_token:
            raise AuthException("No refresh token available")
        
        response = await self.client.post("/auth/refresh", json={
            "refresh_token": self.refresh_token
        })
        
        if response.status_code == 401:
            self._clear_tokens()
            raise AuthException("Refresh token expired")
        
        self._handle_response(response)
        tokens = response.json()
        self._store_tokens(tokens)
        return tokens
    
    # OAuth Methods
    async def get_oauth_providers(self):
        """Get available OAuth providers."""
        response = await self.client.get("/auth/oauth/providers")
        self._handle_response(response)
        return response.json()
    
    async def get_oauth_url(self, provider: str, redirect_uri: str):
        """Get OAuth authorization URL."""
        response = await self.client.get(
            f"/auth/oauth/{provider}/authorize",
            params={"redirect_uri": redirect_uri}
        )
        self._handle_response(response)
        return response.json()
    
    async def oauth_callback(self, provider: str, code: str, state: str):
        """Handle OAuth callback."""
        response = await self.client.get(
            f"/auth/oauth/{provider}/callback",
            params={"code": code, "state": state}
        )
        self._handle_response(response)
        tokens = response.json()
        self._store_tokens(tokens)
        return tokens
    
    # Password Management
    async def update_password(self, current_password: str, new_password: str):
        """Update user password."""
        response = await self._make_authenticated_request(
            "PUT", "/auth/password",
            json={
                "current_password": current_password,
                "new_password": new_password
            }
        )
        self._handle_response(response)
        return response.json()
    
    async def request_password_reset(self, email: str):
        """Request password reset."""
        response = await self.client.post("/auth/password/reset", json={
            "email": email
        })
        self._handle_response(response)
        return response.json()
    
    async def confirm_password_reset(self, token: str, new_password: str):
        """Confirm password reset."""
        response = await self.client.post("/auth/password/reset/confirm", json={
            "token": token,
            "new_password": new_password
        })
        self._handle_response(response)
        return response.json()
    
    # User Profile
    async def get_profile(self):
        """Get current user profile."""
        response = await self._make_authenticated_request("GET", "/auth/me")
        self._handle_response(response)
        return response.json()
    
    # Session Management
    async def get_sessions(self):
        """Get user sessions."""
        response = await self._make_authenticated_request("GET", "/auth/sessions")
        self._handle_response(response)
        return response.json()
    
    async def terminate_session(self, session_id: str):
        """Terminate specific session."""
        response = await self._make_authenticated_request(
            "DELETE", f"/auth/sessions/{session_id}"
        )
        self._handle_response(response)
        return response.json()
    
    async def terminate_all_sessions(self):
        """Terminate all sessions."""
        response = await self._make_authenticated_request(
            "DELETE", "/auth/sessions"
        )
        self._handle_response(response)
        return response.json()
    
    # Utility Methods
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return (
            self.access_token is not None and
            (self.token_expires_at is None or 
             datetime.now(timezone.utc) < self.token_expires_at)
        )
    
    def get_user_id(self) -> Optional[str]:
        """Get user ID from token."""
        if not self.access_token:
            return None
        
        try:
            # Decode without verification (just to read claims)
            payload = jwt.decode(
                self.access_token,
                options={"verify_signature": False}
            )
            return payload.get("sub")
        except:
            return None
    
    # Private Methods
    def _store_tokens(self, tokens: Dict[str, Any]):
        """Store tokens from response."""
        self.access_token = tokens.get("access_token")
        self.refresh_token = tokens.get("refresh_token")
        
        # Calculate expiry time
        expires_in = tokens.get("expires_in")
        if expires_in:
            self.token_expires_at = (
                datetime.now(timezone.utc) + 
                timedelta(seconds=expires_in - 60)  # 60s buffer
            )
    
    def _clear_tokens(self):
        """Clear stored tokens."""
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if not self.access_token:
            raise AuthException("No access token available")
        return {"Authorization": f"Bearer {self.access_token}"}
    
    async def _make_authenticated_request(self, method: str, url: str, **kwargs):
        """Make authenticated request with auto-refresh."""
        # Check if token needs refresh
        if (self.token_expires_at and 
            datetime.now(timezone.utc) >= self.token_expires_at):
            await self.refresh_tokens()
        
        headers = kwargs.get("headers", {})
        headers.update(self._get_auth_headers())
        kwargs["headers"] = headers
        
        response = await self.client.request(method, url, **kwargs)
        
        # Handle token expiry
        if response.status_code == 401 and self.refresh_token:
            try:
                await self.refresh_tokens()
                headers.update(self._get_auth_headers())
                kwargs["headers"] = headers
                response = await self.client.request(method, url, **kwargs)
            except AuthException:
                self._clear_tokens()
                raise
        
        return response
    
    def _handle_response(self, response: httpx.Response):
        """Handle API response and raise exceptions for errors."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise AuthException(
                    error_data.get("message", "API request failed"),
                    status_code=response.status_code,
                    error_code=error_data.get("error"),
                    details=error_data.get("details")
                )
            except ValueError:
                raise AuthException(
                    f"API request failed: {response.text}",
                    status_code=response.status_code
                )


class AuthException(Exception):
    """Authentication exception."""
    
    def __init__(
        self,
        message: str,
        status_code: int = None,
        error_code: str = None,
        details: Any = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.details = details


# Usage Example
async def example_usage():
    async with ArchonAuthClient("http://localhost:8181") as auth:
        try:
            # Register new user
            result = await auth.register(
                email="user@example.com",
                password="SecurePass123!",
                name="John Doe"
            )
            print(f"User registered: {result['user']['email']}")
            
            # Get profile
            profile = await auth.get_profile()
            print(f"Profile: {profile}")
            
            # Update password
            await auth.update_password("SecurePass123!", "NewPassword456!")
            print("Password updated")
            
        except AuthException as e:
            print(f"Auth error: {e} (status: {e.status_code})")
```

### JavaScript Client Library

```javascript
// archon-auth-client.js
class ArchonAuthClient {
    constructor(baseUrl, options = {}) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = options.timeout || 30000;
        this.accessToken = null;
        this.refreshToken = null;
        this.tokenExpiresAt = null;
        
        // Load tokens from storage on initialization
        this.loadTokens();
    }
    
    // Authentication Methods
    async register(email, password, name, metadata = {}) {
        const response = await this.fetch('/auth/register', {
            method: 'POST',
            body: JSON.stringify({
                email,
                password,
                name,
                metadata
            })
        });
        
        const data = await this.handleResponse(response);
        this.storeTokens(data.tokens);
        return data;
    }
    
    async login(email, password, rememberMe = false) {
        const response = await this.fetch('/auth/login', {
            method: 'POST',
            body: JSON.stringify({
                email,
                password,
                remember_me: rememberMe
            })
        });
        
        const tokens = await this.handleResponse(response);
        this.storeTokens(tokens);
        return tokens;
    }
    
    async logout() {
        if (this.accessToken) {
            try {
                await this.fetch('/auth/logout', {
                    method: 'POST',
                    headers: this.getAuthHeaders()
                });
            } catch (error) {
                // Continue with cleanup even if logout fails
                console.warn('Logout request failed:', error);
            }
        }
        this.clearTokens();
    }
    
    async refreshTokens() {
        if (!this.refreshToken) {
            throw new AuthError('No refresh token available');
        }
        
        const response = await this.fetch('/auth/refresh', {
            method: 'POST',
            body: JSON.stringify({
                refresh_token: this.refreshToken
            })
        });
        
        if (response.status === 401) {
            this.clearTokens();
            throw new AuthError('Refresh token expired');
        }
        
        const tokens = await this.handleResponse(response);
        this.storeTokens(tokens);
        return tokens;
    }
    
    // OAuth Methods
    async getOAuthProviders() {
        const response = await this.fetch('/auth/oauth/providers');
        return this.handleResponse(response);
    }
    
    async getOAuthUrl(provider, redirectUri) {
        const response = await this.fetch(
            `/auth/oauth/${provider}/authorize?redirect_uri=${encodeURIComponent(redirectUri)}`
        );
        return this.handleResponse(response);
    }
    
    async handleOAuthCallback(provider, code, state) {
        const response = await this.fetch(
            `/auth/oauth/${provider}/callback?code=${code}&state=${state}`
        );
        const tokens = await this.handleResponse(response);
        this.storeTokens(tokens);
        return tokens;
    }
    
    // Password Management
    async updatePassword(currentPassword, newPassword) {
        const response = await this.authenticatedFetch('/auth/password', {
            method: 'PUT',
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            })
        });
        return this.handleResponse(response);
    }
    
    async requestPasswordReset(email) {
        const response = await this.fetch('/auth/password/reset', {
            method: 'POST',
            body: JSON.stringify({ email })
        });
        return this.handleResponse(response);
    }
    
    // User Profile
    async getProfile() {
        const response = await this.authenticatedFetch('/auth/me');
        return this.handleResponse(response);
    }
    
    // Session Management
    async getSessions() {
        const response = await this.authenticatedFetch('/auth/sessions');
        return this.handleResponse(response);
    }
    
    async terminateSession(sessionId) {
        const response = await this.authenticatedFetch(`/auth/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        return this.handleResponse(response);
    }
    
    // Utility Methods
    isAuthenticated() {
        return (
            this.accessToken !== null &&
            (this.tokenExpiresAt === null || Date.now() < this.tokenExpiresAt)
        );
    }
    
    getUserId() {
        if (!this.accessToken) return null;
        
        try {
            const payload = JSON.parse(atob(this.accessToken.split('.')[1]));
            return payload.sub;
        } catch {
            return null;
        }
    }
    
    // Private Methods
    storeTokens(tokens) {
        this.accessToken = tokens.access_token;
        this.refreshToken = tokens.refresh_token;
        
        if (tokens.expires_in) {
            this.tokenExpiresAt = Date.now() + (tokens.expires_in - 60) * 1000; // 60s buffer
        }
        
        // Store in localStorage for persistence
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('archon_access_token', this.accessToken);
            localStorage.setItem('archon_refresh_token', this.refreshToken);
            if (this.tokenExpiresAt) {
                localStorage.setItem('archon_token_expires_at', this.tokenExpiresAt.toString());
            }
        }
    }
    
    loadTokens() {
        if (typeof localStorage === 'undefined') return;
        
        this.accessToken = localStorage.getItem('archon_access_token');
        this.refreshToken = localStorage.getItem('archon_refresh_token');
        
        const expiresAt = localStorage.getItem('archon_token_expires_at');
        if (expiresAt) {
            this.tokenExpiresAt = parseInt(expiresAt);
        }
    }
    
    clearTokens() {
        this.accessToken = null;
        this.refreshToken = null;
        this.tokenExpiresAt = null;
        
        if (typeof localStorage !== 'undefined') {
            localStorage.removeItem('archon_access_token');
            localStorage.removeItem('archon_refresh_token');
            localStorage.removeItem('archon_token_expires_at');
        }
    }
    
    getAuthHeaders() {
        if (!this.accessToken) {
            throw new AuthError('No access token available');
        }
        return { 'Authorization': `Bearer ${this.accessToken}` };
    }
    
    async fetch(url, options = {}) {
        const fullUrl = url.startsWith('http') ? url : this.baseUrl + url;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        const response = await fetch(fullUrl, defaultOptions);
        return response;
    }
    
    async authenticatedFetch(url, options = {}) {
        // Check if token needs refresh
        if (this.tokenExpiresAt && Date.now() >= this.tokenExpiresAt) {
            await this.refreshTokens();
        }
        
        const headers = {
            ...options.headers,
            ...this.getAuthHeaders()
        };
        
        let response = await this.fetch(url, { ...options, headers });
        
        // Handle token expiry
        if (response.status === 401 && this.refreshToken) {
            try {
                await this.refreshTokens();
                headers['Authorization'] = `Bearer ${this.accessToken}`;
                response = await this.fetch(url, { ...options, headers });
            } catch (error) {
                this.clearTokens();
                throw error;
            }
        }
        
        return response;
    }
    
    async handleResponse(response) {
        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch {
                throw new AuthError(`Request failed: ${response.statusText}`, response.status);
            }
            
            throw new AuthError(
                errorData.message || 'Request failed',
                response.status,
                errorData.error,
                errorData.details
            );
        }
        
        return response.json();
    }
}

class AuthError extends Error {
    constructor(message, statusCode = null, errorCode = null, details = null) {
        super(message);
        this.name = 'AuthError';
        this.statusCode = statusCode;
        this.errorCode = errorCode;
        this.details = details;
    }
}

// Usage Example
const auth = new ArchonAuthClient('http://localhost:8181');

// Register user
try {
    const result = await auth.register(
        'user@example.com',
        'SecurePass123!',
        'John Doe'
    );
    console.log('User registered:', result.user.email);
} catch (error) {
    console.error('Registration failed:', error.message);
}

// Login
try {
    const tokens = await auth.login('user@example.com', 'SecurePass123!');
    console.log('Login successful');
} catch (error) {
    console.error('Login failed:', error.message);
}

// Get profile
if (auth.isAuthenticated()) {
    try {
        const profile = await auth.getProfile();
        console.log('User profile:', profile);
    } catch (error) {
        console.error('Failed to get profile:', error.message);
    }
}
```

---

## Framework Integrations

### FastAPI Integration

```python
# auth_dependency.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
import httpx

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    """Validate JWT token and return user info."""
    try:
        # In production, verify token against public key
        payload = jwt.decode(
            token.credentials,
            "your-secret-key",
            algorithms=["RS256"],
            # Configure proper verification
        )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
            
        return {"user_id": user_id, "claims": payload}
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# main.py
from fastapi import FastAPI, Depends

app = FastAPI()

@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    return {"message": f"Hello {current_user['user_id']}!"}
```

### Express.js Integration

```javascript
// auth-middleware.js
const jwt = require('jsonwebtoken');
const axios = require('axios');

class AuthMiddleware {
    constructor(archonBaseUrl, publicKey) {
        this.archonBaseUrl = archonBaseUrl;
        this.publicKey = publicKey;
    }
    
    authenticate() {
        return async (req, res, next) => {
            try {
                const authHeader = req.headers.authorization;
                if (!authHeader?.startsWith('Bearer ')) {
                    return res.status(401).json({ error: 'No token provided' });
                }
                
                const token = authHeader.substring(7);
                
                // Verify token
                const decoded = jwt.verify(token, this.publicKey, {
                    algorithms: ['RS256']
                });
                
                req.user = {
                    id: decoded.sub,
                    email: decoded.email,
                    roles: decoded.roles || []
                };
                
                next();
                
            } catch (error) {
                if (error.name === 'TokenExpiredError') {
                    return res.status(401).json({ error: 'Token expired' });
                } else if (error.name === 'JsonWebTokenError') {
                    return res.status(401).json({ error: 'Invalid token' });
                }
                
                return res.status(500).json({ error: 'Authentication error' });
            }
        };
    }
    
    requireRoles(...roles) {
        return (req, res, next) => {
            if (!req.user) {
                return res.status(401).json({ error: 'Not authenticated' });
            }
            
            const userRoles = req.user.roles || [];
            const hasRole = roles.some(role => userRoles.includes(role));
            
            if (!hasRole) {
                return res.status(403).json({ error: 'Insufficient permissions' });
            }
            
            next();
        };
    }
}

// Usage
const express = require('express');
const fs = require('fs');

const app = express();
const publicKey = fs.readFileSync('path/to/public-key.pem');
const auth = new AuthMiddleware('http://localhost:8181', publicKey);

// Protected route
app.get('/api/profile', auth.authenticate(), (req, res) => {
    res.json({ user: req.user });
});

// Admin only route
app.get('/api/admin', 
    auth.authenticate(), 
    auth.requireRoles('admin'), 
    (req, res) => {
        res.json({ message: 'Admin access granted' });
    }
);
```

### React Integration

```jsx
// AuthProvider.jsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { ArchonAuthClient } from './archon-auth-client';

const AuthContext = createContext();

export const AuthProvider = ({ children, authUrl }) => {
    const [auth] = useState(() => new ArchonAuthClient(authUrl));
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        // Check if user is authenticated on mount
        if (auth.isAuthenticated()) {
            loadUser();
        } else {
            setLoading(false);
        }
    }, []);
    
    const loadUser = async () => {
        try {
            const profile = await auth.getProfile();
            setUser(profile);
        } catch (error) {
            console.error('Failed to load user:', error);
            auth.clearTokens();
        } finally {
            setLoading(false);
        }
    };
    
    const login = async (email, password) => {
        try {
            await auth.login(email, password);
            await loadUser();
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    };
    
    const logout = async () => {
        try {
            await auth.logout();
        } finally {
            setUser(null);
        }
    };
    
    const register = async (email, password, name) => {
        try {
            const result = await auth.register(email, password, name);
            setUser(result.user);
            return { success: true };
        } catch (error) {
            return { success: false, error: error.message };
        }
    };
    
    const value = {
        user,
        loading,
        login,
        logout,
        register,
        isAuthenticated: () => auth.isAuthenticated(),
        authClient: auth
    };
    
    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
};

// LoginForm.jsx
import React, { useState } from 'react';
import { useAuth } from './AuthProvider';

export const LoginForm = ({ onSuccess }) => {
    const { login } = useAuth();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        
        const result = await login(email, password);
        
        if (result.success) {
            onSuccess?.();
        } else {
            setError(result.error);
        }
        
        setLoading(false);
    };
    
    return (
        <form onSubmit={handleSubmit}>
            {error && <div className="error">{error}</div>}
            
            <div>
                <label htmlFor="email">Email:</label>
                <input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                />
            </div>
            
            <div>
                <label htmlFor="password">Password:</label>
                <input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                />
            </div>
            
            <button type="submit" disabled={loading}>
                {loading ? 'Logging in...' : 'Login'}
            </button>
        </form>
    );
};

// ProtectedRoute.jsx
import React from 'react';
import { useAuth } from './AuthProvider';

export const ProtectedRoute = ({ children, fallback }) => {
    const { user, loading } = useAuth();
    
    if (loading) {
        return <div>Loading...</div>;
    }
    
    if (!user) {
        return fallback || <div>Please log in to access this page.</div>;
    }
    
    return children;
};
```

---

## Best Practices

### Security Best Practices

1. **Token Storage**
   - Web apps: HTTP-only, secure cookies
   - Mobile apps: Secure keychain/keystore
   - Never store tokens in localStorage for sensitive apps

2. **Token Refresh**
   - Implement automatic refresh before expiry
   - Handle refresh failures gracefully
   - Use sliding sessions for better UX

3. **Error Handling**
   - Don't expose internal errors to users
   - Log security events for monitoring
   - Implement proper retry logic

4. **Network Security**
   - Always use HTTPS in production
   - Implement certificate pinning for mobile
   - Validate SSL certificates

### Performance Optimization

1. **Caching**
   - Cache public keys for token verification
   - Implement request caching where appropriate
   - Use connection pooling

2. **Lazy Loading**
   - Load user profile only when needed
   - Defer non-critical authentication data

3. **Error Recovery**
   - Implement exponential backoff
   - Queue requests during token refresh
   - Handle network failures gracefully

### User Experience

1. **Progressive Enhancement**
   - Work offline when possible
   - Provide clear loading states
   - Handle edge cases gracefully

2. **Error Messages**
   - Provide actionable error messages
   - Avoid technical jargon
   - Guide users to resolution

3. **Session Management**
   - Warn before session expiry
   - Provide "stay logged in" options
   - Handle concurrent sessions

---

**Next:** See [Security Best Practices](./04_SECURITY_BEST_PRACTICES.md) for detailed security guidance.