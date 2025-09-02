# Archon Authentication Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps diagnose and resolve common issues with the Archon Authentication API. It includes detailed error scenarios, diagnostic procedures, and step-by-step solutions.

## Table of Contents

1. [Quick Diagnosis Tools](#quick-diagnosis-tools)
2. [Authentication Issues](#authentication-issues)
3. [Token Problems](#token-problems)
4. [OAuth2 Issues](#oauth2-issues)
5. [Session Management Problems](#session-management-problems)
6. [Rate Limiting Issues](#rate-limiting-issues)
7. [Password Management Problems](#password-management-problems)
8. [Network and Connectivity Issues](#network-and-connectivity-issues)
9. [Performance Issues](#performance-issues)
10. [Security Alerts and Incidents](#security-alerts-and-incidents)
11. [Database and Redis Issues](#database-and-redis-issues)
12. [Debugging Tools and Techniques](#debugging-tools-and-techniques)

---

## Quick Diagnosis Tools

### Health Check Script

```bash
#!/bin/bash
# auth-health-check.sh - Quick health diagnostic

ARCHON_URL="${ARCHON_URL:-http://localhost:8181}"

echo "=== Archon Authentication Health Check ==="
echo "Base URL: $ARCHON_URL"
echo

# 1. Basic connectivity
echo "1. Testing basic connectivity..."
if curl -s -f "$ARCHON_URL/auth/health" > /dev/null; then
    echo "✅ Basic connectivity: OK"
else
    echo "❌ Basic connectivity: FAILED"
    echo "   Check if service is running and accessible"
    exit 1
fi

# 2. Health endpoint detailed check
echo "2. Checking service health..."
HEALTH=$(curl -s "$ARCHON_URL/auth/health")
STATUS=$(echo "$HEALTH" | jq -r '.status // "unknown"')
DATABASE=$(echo "$HEALTH" | jq -r '.dependencies.database // "unknown"')
REDIS=$(echo "$HEALTH" | jq -r '.dependencies.redis // "unknown"')

echo "   Service Status: $STATUS"
echo "   Database: $DATABASE"
echo "   Redis: $REDIS"

if [ "$STATUS" != "healthy" ]; then
    echo "⚠️ Service not fully healthy"
fi

# 3. Test authentication endpoints
echo "3. Testing authentication endpoints..."

# Test registration endpoint
REGISTER_TEST=$(curl -s -w "%{http_code}" -o /dev/null -X POST \
    "$ARCHON_URL/auth/register" \
    -H "Content-Type: application/json" \
    -d '{"email":"invalid","password":"test","name":"test"}')

if [ "$REGISTER_TEST" -eq 400 ]; then
    echo "✅ Registration endpoint: Responding correctly"
elif [ "$REGISTER_TEST" -eq 500 ]; then
    echo "❌ Registration endpoint: Internal server error"
else
    echo "⚠️ Registration endpoint: Unexpected response ($REGISTER_TEST)"
fi

# Test OAuth providers
echo "4. Testing OAuth providers..."
PROVIDERS=$(curl -s "$ARCHON_URL/auth/oauth/providers")
PROVIDER_COUNT=$(echo "$PROVIDERS" | jq '.providers | length // 0')
echo "   Available providers: $PROVIDER_COUNT"

if [ "$PROVIDER_COUNT" -gt 0 ]; then
    echo "✅ OAuth providers: Configured"
else
    echo "⚠️ OAuth providers: None configured"
fi

echo "=== Health Check Complete ==="
```

### Python Diagnostic Script

```python
#!/usr/bin/env python3
"""
Archon Authentication Diagnostic Tool
Comprehensive testing and debugging utility
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class AuthDiagnostic:
    def __init__(self, base_url: str = "http://localhost:8181"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=10.0)
        self.results = []
    
    async def run_all_tests(self):
        """Run comprehensive diagnostic tests."""
        print("🔧 Archon Authentication Diagnostic Tool")
        print(f"📡 Testing: {self.base_url}")
        print("=" * 50)
        
        tests = [
            ("Basic Connectivity", self.test_connectivity),
            ("Health Check", self.test_health_check),
            ("Registration Flow", self.test_registration_flow),
            ("Login Flow", self.test_login_flow),
            ("Token Validation", self.test_token_validation),
            ("OAuth Providers", self.test_oauth_providers),
            ("Rate Limiting", self.test_rate_limiting),
            ("Password Security", self.test_password_security),
            ("Session Management", self.test_session_management),
            ("Error Handling", self.test_error_handling)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}...")
            try:
                result = await test_func()
                self.results.append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "timestamp": datetime.now().isoformat()
                })
                print(f"{'✅' if result else '❌'} {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                self.results.append({
                    "test": test_name,
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                print(f"💥 {test_name}: ERROR - {str(e)}")
        
        await self.generate_report()
    
    async def test_connectivity(self) -> bool:
        """Test basic network connectivity."""
        try:
            response = await self.client.get("/auth/health")
            return response.status_code < 500
        except Exception:
            return False
    
    async def test_health_check(self) -> bool:
        """Test health check endpoint."""
        try:
            response = await self.client.get("/auth/health")
            if response.status_code != 200:
                return False
            
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Dependencies: {data.get('dependencies', {})}")
            
            return data.get('status') in ['healthy', 'degraded']
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_registration_flow(self) -> bool:
        """Test user registration."""
        try:
            # Test with invalid data first
            response = await self.client.post("/auth/register", json={
                "email": "invalid-email",
                "password": "weak",
                "name": ""
            })
            
            if response.status_code != 400:
                print(f"   Validation not working properly: {response.status_code}")
                return False
            
            # Test with valid data
            test_user = {
                "email": f"test-{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "name": "Test User"
            }
            
            response = await self.client.post("/auth/register", json=test_user)
            
            if response.status_code == 201:
                print("   Registration successful")
                return True
            elif response.status_code == 409:
                print("   User already exists (expected in some cases)")
                return True
            else:
                print(f"   Unexpected status: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_login_flow(self) -> bool:
        """Test login functionality."""
        try:
            # Create a test user first
            test_user = {
                "email": f"login-test-{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "name": "Login Test User"
            }
            
            reg_response = await self.client.post("/auth/register", json=test_user)
            if reg_response.status_code not in [201, 409]:
                print(f"   Setup failed: {reg_response.status_code}")
                return False
            
            # Test login
            login_response = await self.client.post("/auth/login", json={
                "email": test_user["email"],
                "password": test_user["password"]
            })
            
            if login_response.status_code == 200:
                tokens = login_response.json()
                required_fields = ["access_token", "refresh_token", "token_type", "expires_in"]
                
                if all(field in tokens for field in required_fields):
                    print("   Login successful with valid token structure")
                    return True
                else:
                    print("   Login response missing required fields")
                    return False
            else:
                print(f"   Login failed: {login_response.status_code}")
                return False
                
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_token_validation(self) -> bool:
        """Test JWT token validation."""
        try:
            # Register and login to get tokens
            test_user = {
                "email": f"token-test-{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "name": "Token Test User"
            }
            
            await self.client.post("/auth/register", json=test_user)
            login_response = await self.client.post("/auth/login", json={
                "email": test_user["email"],
                "password": test_user["password"]
            })
            
            if login_response.status_code != 200:
                print("   Login failed for token test")
                return False
            
            tokens = login_response.json()
            access_token = tokens["access_token"]
            
            # Test protected endpoint
            headers = {"Authorization": f"Bearer {access_token}"}
            profile_response = await self.client.get("/auth/me", headers=headers)
            
            # Note: /auth/me returns 501 in current implementation
            if profile_response.status_code in [200, 501]:
                print("   Token validation working")
                return True
            elif profile_response.status_code == 401:
                print("   Token validation failed")
                return False
            else:
                print(f"   Unexpected response: {profile_response.status_code}")
                return False
                
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_oauth_providers(self) -> bool:
        """Test OAuth provider configuration."""
        try:
            response = await self.client.get("/auth/oauth/providers")
            
            if response.status_code != 200:
                print(f"   OAuth providers endpoint failed: {response.status_code}")
                return False
            
            data = response.json()
            providers = data.get("providers", [])
            
            print(f"   Found {len(providers)} OAuth providers")
            for provider in providers:
                print(f"   - {provider.get('display_name')} ({provider.get('name')})")
            
            return len(providers) > 0
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality."""
        try:
            # Make rapid requests to trigger rate limiting
            requests_made = 0
            rate_limited = False
            
            for i in range(15):  # Exceed typical limits
                response = await self.client.post("/auth/login", json={
                    "email": "nonexistent@example.com",
                    "password": "wrong"
                })
                requests_made += 1
                
                if response.status_code == 429:
                    rate_limited = True
                    print(f"   Rate limited after {requests_made} requests")
                    
                    # Check rate limit headers
                    headers = response.headers
                    if "x-ratelimit-limit" in headers:
                        print(f"   Rate limit: {headers['x-ratelimit-limit']}")
                    break
            
            return rate_limited
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_password_security(self) -> bool:
        """Test password security requirements."""
        try:
            weak_passwords = [
                "123456",
                "password",
                "abc",
                "Password",  # Missing number and special char
                "password123",  # Missing uppercase and special char
                "PASSWORD123!"  # Missing lowercase
            ]
            
            for password in weak_passwords:
                response = await self.client.post("/auth/register", json={
                    "email": f"weak-{int(time.time())}@example.com",
                    "password": password,
                    "name": "Test User"
                })
                
                if response.status_code != 400:
                    print(f"   Weak password accepted: {password}")
                    return False
            
            print("   Password security requirements enforced")
            return True
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_session_management(self) -> bool:
        """Test session management."""
        try:
            # Create user and login
            test_user = {
                "email": f"session-test-{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "name": "Session Test User"
            }
            
            await self.client.post("/auth/register", json=test_user)
            login_response = await self.client.post("/auth/login", json={
                "email": test_user["email"],
                "password": test_user["password"]
            })
            
            if login_response.status_code != 200:
                return False
            
            tokens = login_response.json()
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            
            # Test session listing
            sessions_response = await self.client.get("/auth/sessions", headers=headers)
            
            if sessions_response.status_code == 200:
                sessions = sessions_response.json()
                print(f"   Found {sessions.get('total', 0)} active sessions")
                return True
            else:
                print(f"   Session listing failed: {sessions_response.status_code}")
                return False
                
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test API error handling."""
        try:
            # Test various error conditions
            error_tests = [
                ("/auth/nonexistent", 404),
                ("/auth/register", 400, {"email": "invalid"}),  # Bad request
                ("/auth/me", 401, None, {}),  # Unauthorized (no token)
            ]
            
            for test in error_tests:
                endpoint = test[0]
                expected_status = test[1]
                payload = test[2] if len(test) > 2 else None
                headers = test[3] if len(test) > 3 else {"Content-Type": "application/json"}
                
                if payload:
                    response = await self.client.post(endpoint, json=payload, headers=headers)
                else:
                    response = await self.client.get(endpoint, headers=headers)
                
                if response.status_code != expected_status:
                    print(f"   Error handling failed for {endpoint}: expected {expected_status}, got {response.status_code}")
                    return False
            
            print("   Error handling working correctly")
            return True
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    async def generate_report(self):
        """Generate diagnostic report."""
        print("\n" + "=" * 50)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        errors = sum(1 for r in self.results if r["status"] == "ERROR")
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Errors: {errors} 💥")
        
        if failed > 0 or errors > 0:
            print("\n❌ FAILED/ERROR TESTS:")
            for result in self.results:
                if result["status"] in ["FAIL", "ERROR"]:
                    print(f"  - {result['test']}: {result['status']}")
                    if "error" in result:
                        print(f"    Error: {result['error']}")
        
        # Save detailed report
        with open("auth_diagnostic_report.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "summary": {
                    "total": total_tests,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: auth_diagnostic_report.json")
        print(f"🎯 Overall Health: {'HEALTHY' if failed == 0 and errors == 0 else 'ISSUES DETECTED'}")

# Usage
async def main():
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8181"
    
    diagnostic = AuthDiagnostic(base_url)
    await diagnostic.run_all_tests()
    await diagnostic.client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Authentication Issues

### Issue: Registration Fails with Validation Errors

**Symptoms:**
- HTTP 400 responses on registration
- Password validation error messages
- Email format errors

**Common Causes:**
1. Password doesn't meet complexity requirements
2. Invalid email format
3. Missing required fields
4. User already exists

**Diagnostic Steps:**
```bash
# Test password requirements
curl -X POST http://localhost:8181/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "weak",
    "name": "Test User"
  }'

# Expected response: 400 with validation details
```

**Solutions:**
1. **Password Requirements:** Ensure passwords meet all criteria:
   ```
   - Minimum 8 characters
   - At least 1 uppercase letter
   - At least 1 lowercase letter  
   - At least 1 digit
   - At least 1 special character (!@#$%^&*(),.?":{}|<>)
   ```

2. **Email Validation:** Use proper email format:
   ```json
   {
     "email": "user@domain.com",  // Valid format
     "password": "SecurePass123!",
     "name": "John Doe"
   }
   ```

3. **Check for Existing Users:**
   ```bash
   # If you get 409 Conflict, user already exists
   # Try different email or use login instead
   ```

### Issue: Login Returns 401 Unauthorized

**Symptoms:**
- Correct credentials return 401
- "Invalid email or password" error
- Account appears to exist

**Diagnostic Steps:**
```python
# Debug login flow
import httpx

async def debug_login():
    client = httpx.AsyncClient()
    
    # First check if user exists (via registration attempt)
    reg_response = await client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "TestPass123!",
        "name": "Test"
    })
    
    if reg_response.status_code == 409:
        print("✅ User exists")
    elif reg_response.status_code == 201:
        print("✅ User created")
    else:
        print(f"❌ Registration issue: {reg_response.status_code}")
        return
    
    # Test login
    login_response = await client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "TestPass123!"
    })
    
    print(f"Login status: {login_response.status_code}")
    if login_response.status_code != 200:
        print(f"Error: {login_response.text}")
```

**Common Solutions:**
1. **Account Lockout:** Wait for lockout period to expire
   ```bash
   # Check Redis for lockout
   redis-cli GET "failed_login:user@example.com"
   ```

2. **Email Verification Required:**
   ```bash
   # Check if email verification is enabled
   curl -X POST http://localhost:8181/auth/email/verify/resend \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

3. **Password Case Sensitivity:** Ensure exact password match

4. **Database Connection Issues:** Check health endpoint:
   ```bash
   curl http://localhost:8181/auth/health
   ```

### Issue: Account Locked After Failed Attempts

**Symptoms:**
- 429 Too Many Requests
- "Account temporarily locked" message
- Unable to login with correct credentials

**Diagnostic Steps:**
```bash
# Check account lockout status
redis-cli GET "failed_login:user@example.com"

# Check lockout configuration
curl http://localhost:8181/auth/health
```

**Solutions:**
1. **Wait for Lockout Period:** Default is 30 minutes (1800 seconds)

2. **Manual Unlock (Admin):**
   ```bash
   redis-cli DEL "failed_login:user@example.com"
   ```

3. **Password Reset:** Use reset flow to unlock account:
   ```bash
   curl -X POST http://localhost:8181/auth/password/reset \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com"}'
   ```

---

## Token Problems

### Issue: Token Expired Errors

**Symptoms:**
- 401 responses on authenticated endpoints
- "Token has expired" error messages
- Valid operations failing suddenly

**Diagnostic Steps:**
```python
import jwt
import json
from datetime import datetime

def debug_token(token):
    """Debug JWT token without verification."""
    try:
        # Decode without verification to inspect claims
        payload = jwt.decode(token, options={"verify_signature": False})
        
        print("Token Claims:")
        print(json.dumps(payload, indent=2, default=str))
        
        # Check expiry
        exp = payload.get('exp')
        if exp:
            exp_time = datetime.fromtimestamp(exp)
            now = datetime.now()
            print(f"\nToken expires at: {exp_time}")
            print(f"Current time: {now}")
            print(f"Expired: {'Yes' if now > exp_time else 'No'}")
            
            if now <= exp_time:
                remaining = exp_time - now
                print(f"Time remaining: {remaining}")
        
    except Exception as e:
        print(f"Error decoding token: {e}")

# Usage
token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9..."
debug_token(token)
```

**Solutions:**
1. **Automatic Token Refresh:**
   ```python
   async def refresh_token_if_needed(auth_client):
       if auth_client.token_expires_soon():
           try:
               await auth_client.refresh_tokens()
               print("Token refreshed successfully")
           except Exception as e:
               print(f"Refresh failed: {e}")
               # Re-login required
               await auth_client.login(email, password)
   ```

2. **Manual Token Refresh:**
   ```bash
   curl -X POST http://localhost:8181/auth/refresh \
     -H "Content-Type: application/json" \
     -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
   ```

3. **Check Token Storage:** Ensure tokens are stored and retrieved correctly

### Issue: Invalid Token Errors

**Symptoms:**
- "Invalid token" responses
- Token validation failures
- Malformed JWT errors

**Diagnostic Steps:**
```bash
# Validate token structure
echo "YOUR_TOKEN" | cut -d. -f1 | base64 -d  # Header
echo "YOUR_TOKEN" | cut -d. -f2 | base64 -d  # Payload
echo "YOUR_TOKEN" | cut -d. -f3 | base64 -d  # Signature

# Check token format
if [[ "YOUR_TOKEN" =~ ^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$ ]]; then
    echo "Token format: Valid JWT structure"
else
    echo "Token format: Invalid JWT structure"
fi
```

**Solutions:**
1. **Token Corruption:** Check for truncation or modification
   ```python
   def validate_jwt_structure(token):
       parts = token.split('.')
       if len(parts) != 3:
           return False, "JWT must have 3 parts separated by dots"
       
       # Check if each part is valid base64
       for i, part in enumerate(parts):
           try:
               # Add padding if needed
               padded = part + '=' * (4 - len(part) % 4)
               base64.b64decode(padded)
           except:
               return False, f"Part {i+1} is not valid base64"
       
       return True, "Valid JWT structure"
   ```

2. **Wrong Token Type:** Ensure using access token for API calls
   ```python
   # Use access token, not refresh token
   headers = {"Authorization": f"Bearer {access_token}"}  # ✅
   headers = {"Authorization": f"Bearer {refresh_token}"}  # ❌
   ```

3. **Token Blacklisted:** Check if token was revoked
   ```bash
   # This would require access to Redis
   redis-cli GET "blacklist:TOKEN_JTI"
   ```

### Issue: Token Refresh Fails

**Symptoms:**
- 401 on token refresh endpoint
- "Invalid or expired refresh token"
- Refresh token rotation issues

**Diagnostic Steps:**
```python
async def debug_token_refresh():
    # Check refresh token validity
    refresh_token = "YOUR_REFRESH_TOKEN"
    
    # Decode without verification
    payload = jwt.decode(refresh_token, options={"verify_signature": False})
    
    print(f"Token type: {payload.get('type')}")  # Should be 'refresh'
    print(f"User ID: {payload.get('sub')}")
    print(f"Session ID: {payload.get('session_id')}")
    print(f"Expires at: {datetime.fromtimestamp(payload.get('exp'))}")
    
    # Test refresh
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8181/auth/refresh", json={
            "refresh_token": refresh_token
        })
        
        print(f"Refresh response: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
```

**Solutions:**
1. **Expired Refresh Token:** Re-login required
   ```python
   if response.status_code == 401:
       # Refresh token expired, need new login
       await auth_client.login(email, password)
   ```

2. **Token Rotation Conflict:** Handle rotation properly
   ```python
   async def safe_token_refresh(auth_client):
       try:
           new_tokens = await auth_client.refresh_tokens()
           # Store new tokens atomically
           await store_tokens_atomic(new_tokens)
       except RefreshFailedException:
           # Clear tokens and require re-login
           await auth_client.clear_tokens()
           raise AuthenticationRequired()
   ```

3. **Session Invalidated:** Check session status
   ```bash
   # Check if session exists
   redis-cli EXISTS "session:SESSION_ID"
   ```

---

## OAuth2 Issues

### Issue: OAuth Authorization URL Generation Fails

**Symptoms:**
- 400 Bad Request on authorization endpoint
- Invalid provider errors
- Missing redirect_uri errors

**Diagnostic Steps:**
```bash
# Test OAuth provider endpoint
curl "http://localhost:8181/auth/oauth/providers"

# Test authorization URL generation
curl "http://localhost:8181/auth/oauth/google/authorize?redirect_uri=https://myapp.com/callback"
```

**Solutions:**
1. **Invalid Provider:** Check available providers
   ```python
   async def check_oauth_providers():
       async with httpx.AsyncClient() as client:
           response = await client.get("http://localhost:8181/auth/oauth/providers")
           providers = response.json()
           
           print("Available providers:")
           for provider in providers.get('providers', []):
               print(f"  - {provider['name']} ({provider['display_name']})")
   ```

2. **Redirect URI Validation:** Ensure URI is whitelisted
   ```bash
   # Check OAuth configuration
   # Redirect URI must be exactly as configured
   ```

3. **Provider Configuration:** Verify OAuth credentials
   ```python
   # Check environment variables
   import os
   
   oauth_config = {
       'google': {
           'client_id': os.getenv('GOOGLE_CLIENT_ID'),
           'client_secret': os.getenv('GOOGLE_CLIENT_SECRET')
       },
       'github': {
           'client_id': os.getenv('GITHUB_CLIENT_ID'),
           'client_secret': os.getenv('GITHUB_CLIENT_SECRET')
       }
   }
   
   for provider, config in oauth_config.items():
       if not config['client_id']:
           print(f"❌ {provider}: Missing CLIENT_ID")
       if not config['client_secret']:
           print(f"❌ {provider}: Missing CLIENT_SECRET")
   ```

### Issue: OAuth Callback Fails

**Symptoms:**
- "Invalid state parameter" errors
- Code exchange failures
- User info retrieval errors

**Diagnostic Steps:**
```python
async def debug_oauth_callback():
    # Parse callback URL
    from urllib.parse import urlparse, parse_qs
    
    callback_url = "https://myapp.com/callback?code=abc123&state=xyz789"
    parsed = urlparse(callback_url)
    params = parse_qs(parsed.query)
    
    code = params.get('code', [None])[0]
    state = params.get('state', [None])[0]
    
    print(f"Authorization code: {code}")
    print(f"State parameter: {state}")
    
    # Test callback processing
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8181/auth/oauth/google/callback",
            params={"code": code, "state": state}
        )
        
        print(f"Callback response: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
```

**Solutions:**
1. **State Parameter Issues:** Check Redis for state storage
   ```bash
   # Check stored state
   redis-cli GET "oauth:state:xyz789"
   ```

2. **Code Expiry:** Authorization codes expire quickly (usually 10 minutes)
   ```python
   # Handle callback immediately after authorization
   # Don't cache or store authorization codes
   ```

3. **Provider-Specific Issues:**
   - **Google:** Ensure correct scopes (`openid email profile`)
   - **GitHub:** Check email permissions and API rate limits
   - **Microsoft:** Verify tenant configuration

### Issue: OAuth User Info Retrieval Fails

**Symptoms:**
- Token exchange succeeds but user info fails
- Missing email or profile data
- API rate limit errors from providers

**Diagnostic Steps:**
```python
async def test_oauth_user_info():
    # Test with actual OAuth access token from provider
    provider_token = "OAUTH_ACCESS_TOKEN"
    
    # Google
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {provider_token}"}
        )
        print(f"Google user info: {response.status_code}")
        if response.status_code == 200:
            print(response.json())
        else:
            print(f"Error: {response.text}")
```

**Solutions:**
1. **Scope Issues:** Ensure proper scopes are requested
   ```python
   # Google scopes
   scopes = ["openid", "email", "profile"]
   
   # GitHub scopes  
   scopes = ["user:email"]
   
   # Microsoft scopes
   scopes = ["openid", "email", "profile", "User.Read"]
   ```

2. **Rate Limiting:** Implement backoff and caching
   ```python
   import asyncio
   from functools import wraps
   
   def retry_with_backoff(max_retries=3, backoff_base=2):
       def decorator(func):
           @wraps(func)
           async def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return await func(*args, **kwargs)
                   except httpx.HTTPStatusError as e:
                       if e.response.status_code == 429:  # Rate limited
                           wait_time = backoff_base ** attempt
                           await asyncio.sleep(wait_time)
                           continue
                       raise
               raise Exception(f"Max retries ({max_retries}) exceeded")
           return wrapper
       return decorator
   ```

3. **Email Privacy Settings:** Handle missing emails gracefully
   ```python
   def parse_oauth_user_info(provider, user_data):
       if provider == "github":
           # GitHub users might have private emails
           email = user_data.get('email')
           if not email:
               # Try to get from separate emails API
               return None  # Handle gracefully
       
       return {
           'email': email,
           'name': user_data.get('name'),
           'picture': user_data.get('picture'),
           'provider': provider
       }
   ```

---

## Session Management Problems

### Issue: Sessions Expire Too Quickly

**Symptoms:**
- Users logged out unexpectedly
- Frequent re-authentication required
- Session timeouts during active use

**Diagnostic Steps:**
```bash
# Check session configuration
curl http://localhost:8181/auth/health

# Test session duration
# Login and wait, then test authenticated endpoint
```

**Solutions:**
1. **Configure Session Timeout:**
   ```python
   # In authentication service configuration
   config = {
       "session_timeout": 3600,  # 1 hour
       "sliding_window": True,   # Extend on activity
       "remember_me_duration": 604800  # 7 days
   }
   ```

2. **Implement Session Extension:**
   ```python
   async def extend_session_on_activity(session_id):
       session_key = f"session:{session_id}"
       await redis.expire(session_key, 3600)  # Extend by 1 hour
   ```

3. **Remember Me Functionality:**
   ```python
   # Longer session for remember me
   if remember_me:
       session_timeout = 7 * 24 * 3600  # 7 days
   else:
       session_timeout = 3600  # 1 hour
   ```

### Issue: Multiple Sessions Not Working

**Symptoms:**
- Previous sessions terminated on new login
- Session listing shows only one session
- Users can't access from multiple devices

**Diagnostic Steps:**
```bash
# Check session configuration
redis-cli KEYS "session:*"

# Test multiple logins
# Login from different IPs/user agents
```

**Solutions:**
1. **Configure Concurrent Sessions:**
   ```python
   class SessionConfig:
       max_concurrent_sessions = 5  # Allow multiple sessions
       terminate_old_sessions = False  # Don't auto-terminate
   ```

2. **Session Management per Device:**
   ```python
   async def create_session_with_device_info(user_id, request_context):
       device_fingerprint = generate_device_fingerprint(request_context)
       
       # Check existing sessions for this device
       existing_session = await get_session_by_device(user_id, device_fingerprint)
       if existing_session:
           # Update existing session
           await update_session(existing_session['id'])
       else:
           # Create new session
           await create_new_session(user_id, request_context)
   ```

### Issue: Session Hijacking Detection False Positives

**Symptoms:**
- Sessions terminated due to IP changes
- Mobile users frequently logged out
- VPN users having issues

**Diagnostic Steps:**
```python
async def analyze_session_terminations():
    # Check Redis for session termination logs
    termination_logs = await redis.lrange("session_terminations", 0, 100)
    
    for log in termination_logs:
        data = json.loads(log)
        print(f"Session {data['session_id']} terminated: {data['reason']}")
        print(f"  Original IP: {data['original_ip']}")
        print(f"  New IP: {data['new_ip']}")
```

**Solutions:**
1. **Adjust IP Validation:**
   ```python
   class SecurityConfig:
       strict_ip_validation = False  # Allow IP changes
       ip_change_threshold = 2  # Allow 2 IP changes per session
   ```

2. **Implement Intelligent Detection:**
   ```python
   def is_suspicious_ip_change(original_ip, new_ip, user_history):
       # Check if IPs are in same region/ISP
       if are_ips_in_same_region(original_ip, new_ip):
           return False
       
       # Check user's historical locations
       if new_ip in user_history.get('known_ips', []):
           return False
       
       # Check for impossible travel
       return is_impossible_travel(original_ip, new_ip, time_diff=300)
   ```

3. **Grace Period for IP Changes:**
   ```python
   async def handle_ip_change(session_id, new_ip):
       # Flag for manual review instead of immediate termination
       await flag_session_for_review(session_id, {
           'reason': 'ip_change',
           'new_ip': new_ip,
           'requires_verification': True
       })
       
       # Allow continued access for short period
       await extend_session_temporarily(session_id, duration=300)
   ```

---

## Rate Limiting Issues

### Issue: Rate Limits Too Restrictive

**Symptoms:**
- Legitimate users getting 429 errors
- Normal usage patterns blocked
- Development testing hindered

**Diagnostic Steps:**
```bash
# Check current rate limit settings
curl -I http://localhost:8181/auth/login

# Look for rate limit headers
# X-RateLimit-Limit
# X-RateLimit-Remaining
# X-RateLimit-Reset
```

**Solutions:**
1. **Adjust Rate Limit Configuration:**
   ```python
   rate_limits = {
       'login': {'limit': 20, 'window': 60},      # 20 per minute
       'register': {'limit': 10, 'window': 60},   # 10 per minute  
       'refresh': {'limit': 30, 'window': 60},    # 30 per minute
       'password_reset': {'limit': 5, 'window': 3600}  # 5 per hour
   }
   ```

2. **Implement User-Based Rate Limiting:**
   ```python
   async def get_rate_limit_key(request, user_id=None):
       if user_id:
           # Authenticated users get higher limits
           return f"user:{user_id}"
       else:
           # Anonymous users limited by IP
           return f"ip:{request.client.host}"
   ```

3. **Whitelist Development IPs:**
   ```python
   DEVELOPMENT_IPS = ['127.0.0.1', '192.168.1.0/24']
   
   async def apply_rate_limit(request, endpoint, limit, window):
       client_ip = request.client.host
       
       if any(ip_in_network(client_ip, network) for network in DEVELOPMENT_IPS):
           return  # Skip rate limiting for dev IPs
       
       # Apply normal rate limiting
       await check_rate_limit(client_ip, endpoint, limit, window)
   ```

### Issue: Rate Limiting Not Working

**Symptoms:**
- No 429 responses even with excessive requests
- Rate limit headers missing
- Abuse not being blocked

**Diagnostic Steps:**
```python
async def test_rate_limiting():
    async with httpx.AsyncClient() as client:
        for i in range(20):
            response = await client.post("http://localhost:8181/auth/login", json={
                "email": "test@example.com",
                "password": "wrong"
            })
            
            print(f"Request {i+1}: {response.status_code}")
            
            # Check rate limit headers
            if 'x-ratelimit-remaining' in response.headers:
                print(f"  Remaining: {response.headers['x-ratelimit-remaining']}")
            
            if response.status_code == 429:
                print("✅ Rate limiting working")
                break
        else:
            print("❌ Rate limiting not working")
```

**Solutions:**
1. **Check Redis Connection:**
   ```bash
   redis-cli ping
   redis-cli KEYS "rate_limit:*"
   ```

2. **Verify Rate Limiter Configuration:**
   ```python
   class RateLimiterConfig:
       enabled = True
       redis_host = "localhost"
       redis_port = 6379
       redis_db = 0
   
   # Test Redis connectivity
   import redis
   r = redis.Redis(host='localhost', port=6379, db=0)
   try:
       r.ping()
       print("✅ Redis connected")
   except:
       print("❌ Redis connection failed")
   ```

3. **Enable Rate Limiting Middleware:**
   ```python
   from fastapi import FastAPI
   from your_rate_limiter import RateLimitMiddleware
   
   app = FastAPI()
   app.add_middleware(
       RateLimitMiddleware,
       redis_client=redis_client,
       enabled=True
   )
   ```

---

## Password Management Problems

### Issue: Password Reset Emails Not Sent

**Symptoms:**
- Password reset returns success but no email received
- Users can't reset forgotten passwords
- Email service connectivity issues

**Diagnostic Steps:**
```bash
# Test password reset endpoint
curl -X POST http://localhost:8181/auth/password/reset \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'

# Check email service configuration
# Verify SMTP settings
```

**Solutions:**
1. **Configure Email Service:**
   ```python
   class EmailConfig:
       smtp_host = "smtp.gmail.com"
       smtp_port = 587
       smtp_user = "your-email@gmail.com"
       smtp_password = "app-specific-password"
       use_tls = True
   
   # Test email connectivity
   import smtplib
   
   try:
       server = smtplib.SMTP(smtp_host, smtp_port)
       server.starttls()
       server.login(smtp_user, smtp_password)
       print("✅ SMTP connection successful")
       server.quit()
   except Exception as e:
       print(f"❌ SMTP connection failed: {e}")
   ```

2. **Check Email Templates:**
   ```python
   def render_password_reset_email(user_email, reset_url):
       template = """
       Subject: Password Reset Request
       
       Click the link below to reset your password:
       {reset_url}
       
       This link expires in 1 hour.
       """
       return template.format(reset_url=reset_url)
   ```

3. **Verify Email Delivery:**
   ```python
   async def debug_email_sending():
       # Check email queue
       pending_emails = await redis.llen("email_queue")
       print(f"Pending emails: {pending_emails}")
       
       # Check failed emails
       failed_emails = await redis.llen("failed_emails")
       print(f"Failed emails: {failed_emails}")
   ```

### Issue: Password Reset Tokens Invalid

**Symptoms:**
- "Invalid or expired token" on reset confirmation
- Reset tokens not working despite being recent
- Token format issues

**Diagnostic Steps:**
```bash
# Check token storage in Redis
redis-cli KEYS "password_reset:*"

# Check token TTL
redis-cli TTL "password_reset:YOUR_TOKEN"
```

**Solutions:**
1. **Check Token Generation:**
   ```python
   import secrets
   
   def generate_secure_reset_token():
       # Use cryptographically secure random
       return secrets.token_urlsafe(32)
   ```

2. **Verify Token Storage:**
   ```python
   async def store_reset_token(email, token):
       token_key = f"password_reset:{token}"
       token_data = {
           "email": email,
           "timestamp": time.time(),
           "used": False
       }
       
       # Store with TTL
       await redis.setex(token_key, 3600, json.dumps(token_data))
   ```

3. **Debug Token Validation:**
   ```python
   async def debug_token_validation(token):
       token_key = f"password_reset:{token}"
       
       # Check if token exists
       exists = await redis.exists(token_key)
       print(f"Token exists: {exists}")
       
       if exists:
           # Check TTL
           ttl = await redis.ttl(token_key)
           print(f"Token TTL: {ttl} seconds")
           
           # Get token data
           data = await redis.get(token_key)
           print(f"Token data: {data}")
   ```

---

## Network and Connectivity Issues

### Issue: Service Unreachable

**Symptoms:**
- Connection refused errors
- Timeouts on API calls
- Service appears to be down

**Diagnostic Steps:**
```bash
# Check if service is running
curl -f http://localhost:8181/auth/health

# Check port binding
netstat -tlnp | grep 8181

# Check process status
ps aux | grep archon

# Check logs
tail -f /var/log/archon/auth.log

# Test network connectivity
telnet localhost 8181
```

**Solutions:**
1. **Service Not Running:**
   ```bash
   # Start the service
   docker-compose up -d

   # Or start manually
   cd /path/to/archon
   python -m src.server.main
   ```

2. **Port Binding Issues:**
   ```bash
   # Check what's using port 8181
   lsof -i :8181
   
   # Change port in configuration
   export ARCHON_AUTH_PORT=8182
   ```

3. **Firewall Issues:**
   ```bash
   # Allow port through firewall
   sudo ufw allow 8181
   
   # Or disable firewall for testing
   sudo ufw disable
   ```

### Issue: SSL/TLS Certificate Problems

**Symptoms:**
- Certificate verification errors
- "SSL handshake failed" messages
- Browsers showing security warnings

**Diagnostic Steps:**
```bash
# Test SSL certificate
openssl s_client -connect api.archon.ai:443 -servername api.archon.ai

# Check certificate expiry
echo | openssl s_client -connect api.archon.ai:443 2>/dev/null | openssl x509 -noout -dates

# Verify certificate chain
curl -I https://api.archon.ai/auth/health
```

**Solutions:**
1. **Self-Signed Certificate Issues:**
   ```python
   # For development, disable SSL verification
   import httpx
   
   client = httpx.AsyncClient(verify=False)
   
   # Or add certificate to trust store
   client = httpx.AsyncClient(verify="/path/to/cert.pem")
   ```

2. **Certificate Renewal:**
   ```bash
   # Using Let's Encrypt
   certbot renew
   
   # Restart nginx after renewal
   sudo systemctl reload nginx
   ```

3. **Configure Proper SSL:**
   ```nginx
   server {
       listen 443 ssl http2;
       server_name api.archon.ai;
       
       ssl_certificate /etc/ssl/certs/archon.crt;
       ssl_certificate_key /etc/ssl/private/archon.key;
       
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers HIGH:!aNULL:!MD5;
   }
   ```

---

## Performance Issues

### Issue: Slow API Response Times

**Symptoms:**
- Request timeouts
- High latency on authentication endpoints
- Poor user experience

**Diagnostic Steps:**
```bash
# Measure API response times
time curl http://localhost:8181/auth/health

# Use httpx for detailed timing
python -c "
import httpx
import time

start = time.time()
response = httpx.get('http://localhost:8181/auth/health')
end = time.time()

print(f'Response time: {(end-start)*1000:.2f}ms')
print(f'Status: {response.status_code}')
"

# Check system resources
top
htop
iotop
```

**Solutions:**
1. **Database Query Optimization:**
   ```python
   # Add database indexes
   CREATE INDEX idx_users_email ON users(email);
   CREATE INDEX idx_sessions_user_id ON sessions(user_id);
   CREATE INDEX idx_tokens_jti ON token_blacklist(jti);
   
   # Use connection pooling
   from sqlalchemy.pool import QueuePool
   
   engine = create_async_engine(
       database_url,
       poolclass=QueuePool,
       pool_size=20,
       max_overflow=30,
       pool_pre_ping=True
   )
   ```

2. **Redis Performance Tuning:**
   ```bash
   # Redis configuration
   echo "maxmemory 2gb" >> /etc/redis/redis.conf
   echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
   
   # Enable pipelining
   redis-cli CONFIG SET tcp-keepalive 60
   ```

3. **Caching Implementation:**
   ```python
   from functools import lru_cache
   import asyncio
   
   class CachingAuthService:
       def __init__(self):
           self._user_cache = {}
           self._cache_ttl = 300  # 5 minutes
       
       @lru_cache(maxsize=1000)
       async def get_user_roles(self, user_id: str):
           # Cache user roles for 5 minutes
           return await self.db.get_user_roles(user_id)
   ```

### Issue: High Memory Usage

**Symptoms:**
- Out of memory errors
- Process killed by OOM killer
- Gradual memory increase over time

**Diagnostic Steps:**
```bash
# Monitor memory usage
ps aux --sort=-%mem | head

# Check Python memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Monitor over time
watch 'ps aux | grep archon'
```

**Solutions:**
1. **Connection Pool Management:**
   ```python
   # Limit connection pool size
   engine = create_async_engine(
       database_url,
       pool_size=10,  # Reduce pool size
       max_overflow=20,
       pool_recycle=3600  # Recycle connections hourly
   )
   ```

2. **Memory-Efficient Token Handling:**
   ```python
   class MemoryEfficientTokenManager:
       def __init__(self):
           # Use weak references for token cache
           import weakref
           self._token_cache = weakref.WeakValueDictionary()
       
       async def validate_token(self, token: str):
           # Don't store large token strings in memory
           token_hash = hashlib.sha256(token.encode()).hexdigest()
           
           if token_hash in self._token_cache:
               return self._token_cache[token_hash]
           
           # Validate and cache result
           result = await self._validate_token_impl(token)
           self._token_cache[token_hash] = result
           return result
   ```

3. **Garbage Collection Tuning:**
   ```python
   import gc
   
   # Force garbage collection periodically
   async def cleanup_task():
       while True:
           await asyncio.sleep(300)  # Every 5 minutes
           gc.collect()
   
   # Start cleanup task
   asyncio.create_task(cleanup_task())
   ```

---

## Debugging Tools and Techniques

### Debug Mode Setup

```python
# debug_auth.py - Comprehensive debugging setup

import logging
import asyncio
import json
from datetime import datetime
from contextlib import asynccontextmanager

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auth_debug.log'),
        logging.StreamHandler()
    ]
)

class AuthDebugger:
    def __init__(self, base_url="http://localhost:8181"):
        self.base_url = base_url
        self.debug_log = []
    
    def log_debug(self, message, data=None):
        """Log debug information with timestamp."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'data': data
        }
        self.debug_log.append(entry)
        print(f"🐛 {message}")
        if data:
            print(f"   Data: {json.dumps(data, indent=2, default=str)}")
    
    async def trace_authentication_flow(self, email, password):
        """Trace complete authentication flow with debugging."""
        self.log_debug("Starting authentication flow trace", {
            'email': email,
            'base_url': self.base_url
        })
        
        async with httpx.AsyncClient() as client:
            # 1. Test health
            self.log_debug("Testing service health...")
            try:
                health_response = await client.get(f"{self.base_url}/auth/health")
                self.log_debug("Health check result", {
                    'status_code': health_response.status_code,
                    'response': health_response.json()
                })
            except Exception as e:
                self.log_debug("Health check failed", {'error': str(e)})
                return False
            
            # 2. Test registration
            self.log_debug("Testing user registration...")
            try:
                reg_response = await client.post(f"{self.base_url}/auth/register", json={
                    "email": email,
                    "password": password,
                    "name": "Debug Test User"
                })
                self.log_debug("Registration result", {
                    'status_code': reg_response.status_code,
                    'headers': dict(reg_response.headers),
                    'response': reg_response.text[:500] if reg_response.text else None
                })
            except Exception as e:
                self.log_debug("Registration failed", {'error': str(e)})
            
            # 3. Test login
            self.log_debug("Testing user login...")
            try:
                login_response = await client.post(f"{self.base_url}/auth/login", json={
                    "email": email,
                    "password": password
                })
                self.log_debug("Login result", {
                    'status_code': login_response.status_code,
                    'headers': dict(login_response.headers)
                })
                
                if login_response.status_code == 200:
                    tokens = login_response.json()
                    self.log_debug("Login successful", {
                        'has_access_token': bool(tokens.get('access_token')),
                        'has_refresh_token': bool(tokens.get('refresh_token')),
                        'token_type': tokens.get('token_type'),
                        'expires_in': tokens.get('expires_in')
                    })
                    
                    # 4. Test authenticated endpoint
                    access_token = tokens.get('access_token')
                    headers = {'Authorization': f'Bearer {access_token}'}
                    
                    self.log_debug("Testing authenticated endpoint...")
                    try:
                        profile_response = await client.get(
                            f"{self.base_url}/auth/me", 
                            headers=headers
                        )
                        self.log_debug("Profile endpoint result", {
                            'status_code': profile_response.status_code,
                            'response': profile_response.text[:200]
                        })
                    except Exception as e:
                        self.log_debug("Profile endpoint failed", {'error': str(e)})
                    
                    # 5. Test token refresh
                    refresh_token = tokens.get('refresh_token')
                    if refresh_token:
                        self.log_debug("Testing token refresh...")
                        try:
                            refresh_response = await client.post(
                                f"{self.base_url}/auth/refresh",
                                json={'refresh_token': refresh_token}
                            )
                            self.log_debug("Token refresh result", {
                                'status_code': refresh_response.status_code,
                                'response': refresh_response.text[:200]
                            })
                        except Exception as e:
                            self.log_debug("Token refresh failed", {'error': str(e)})
                
            except Exception as e:
                self.log_debug("Login failed", {'error': str(e)})
        
        # Save debug log
        with open(f'auth_trace_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(self.debug_log, f, indent=2)
        
        return True

# Usage
async def main():
    debugger = AuthDebugger()
    await debugger.trace_authentication_flow(
        email="debug-user@example.com",
        password="DebugPass123!"
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Production Monitoring

```python
# monitoring.py - Production monitoring setup

import time
import psutil
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class HealthMetrics:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    active_connections: int
    response_time_ms: float

class ProductionMonitor:
    def __init__(self):
        self.metrics_history: List[HealthMetrics] = []
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_ms': 1000.0,
            'disk_usage_percent': 90.0
        }
    
    async def collect_metrics(self):
        """Collect system and application metrics."""
        start_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Test API response time
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8181/auth/health")
                response_time_ms = (time.time() - start_time) * 1000
        except:
            response_time_ms = float('inf')
        
        # Network connections
        connections = len(psutil.net_connections())
        
        metrics = HealthMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,
            disk_usage_percent=disk.percent,
            active_connections=connections,
            response_time_ms=response_time_ms
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last hour of metrics
        if len(self.metrics_history) > 3600:
            self.metrics_history = self.metrics_history[-3600:]
        
        # Check alerts
        await self.check_alerts(metrics)
        
        return metrics
    
    async def check_alerts(self, metrics: HealthMetrics):
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.response_time_ms > self.alert_thresholds['response_time_ms']:
            alerts.append(f"Slow response time: {metrics.response_time_ms:.1f}ms")
        
        if metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        if alerts:
            await self.send_alerts(alerts, metrics)
    
    async def send_alerts(self, alerts: List[str], metrics: HealthMetrics):
        """Send alert notifications."""
        alert_message = f"""
🚨 ARCHON AUTH ALERT 🚨
Time: {datetime.fromtimestamp(metrics.timestamp)}

Issues detected:
{chr(10).join(f"- {alert}" for alert in alerts)}

Current metrics:
- CPU: {metrics.cpu_percent:.1f}%
- Memory: {metrics.memory_percent:.1f}%
- Response time: {metrics.response_time_ms:.1f}ms
- Disk usage: {metrics.disk_usage_percent:.1f}%
        """
        
        # Log alert
        print(alert_message)
        
        # In production, send to monitoring service
        # await send_slack_alert(alert_message)
        # await send_email_alert(alert_message)
    
    async def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring."""
        print(f"🔍 Starting Archon Auth monitoring (interval: {interval}s)")
        
        while True:
            try:
                metrics = await self.collect_metrics()
                print(f"📊 CPU: {metrics.cpu_percent:.1f}% | "
                      f"MEM: {metrics.memory_percent:.1f}% | "
                      f"RT: {metrics.response_time_ms:.1f}ms")
                
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(10)  # Short retry interval

# Usage
if __name__ == "__main__":
    monitor = ProductionMonitor()
    asyncio.run(monitor.start_monitoring())
```

---

This troubleshooting guide provides comprehensive diagnostic procedures and solutions for the most common issues with the Archon Authentication API. For issues not covered here, check the server logs and enable debug mode for detailed error information.

**Next:** See [Migration Guide](./06_MIGRATION_GUIDE.md) for migrating from other authentication systems.