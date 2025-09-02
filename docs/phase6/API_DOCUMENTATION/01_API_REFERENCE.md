# Archon Authentication API Reference

## Overview

The Archon Authentication API provides comprehensive user authentication, authorization, and session management capabilities. Built on FastAPI with modern security standards including JWT tokens, OAuth2 flows, rate limiting, and audit logging.

**Base URL:** `/auth`
**Version:** 1.0.0
**Authentication:** Bearer Token (JWT)

## Quick Start

```python
import httpx

# Register a new user
response = httpx.post("http://localhost:8181/auth/register", json={
    "email": "user@example.com",
    "password": "SecurePass123!",
    "name": "John Doe"
})

# Login and get tokens
response = httpx.post("http://localhost:8181/auth/login", json={
    "email": "user@example.com",
    "password": "SecurePass123!",
    "remember_me": False
})

tokens = response.json()
access_token = tokens["access_token"]

# Use token for authenticated requests
headers = {"Authorization": f"Bearer {access_token}"}
response = httpx.get("http://localhost:8181/auth/me", headers=headers)
```

## Authentication Flow

1. **Register** → Create user account
2. **Login** → Receive access + refresh tokens
3. **Access** → Use access token for authenticated requests
4. **Refresh** → Exchange refresh token for new access token
5. **Logout** → Invalidate tokens and session

## Rate Limits

All endpoints include rate limiting with the following default limits:

| Endpoint | Limit | Window |
|----------|-------|--------|
| Registration | 5 requests | 60 seconds |
| Login | 10 requests | 60 seconds |
| Token Refresh | 20 requests | 60 seconds |
| Password Reset | 3 requests | 3600 seconds |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Request limit for current window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Window reset timestamp
- `Retry-After`: Seconds until retry allowed (when rate limited)

## Common Response Formats

### Success Response
```json
{
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "email_verified": true,
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z",
    "last_login_at": "2024-01-01T12:00:00Z",
    "metadata": {}
  },
  "tokens": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
    "token_type": "Bearer",
    "expires_in": 900,
    "refresh_expires_in": 604800
  }
}
```

### Error Response
```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": [
    {
      "field": "password",
      "message": "Password must contain at least one uppercase letter",
      "code": "WEAK_PASSWORD"
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123"
}
```

---

## Endpoints

### Health Check

#### GET /auth/health
Check the health status of the authentication service and its dependencies.

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy"
  },
  "uptime_seconds": 86400
}
```

**Status Values:**
- `healthy`: All services operational
- `degraded`: Some non-critical services down
- `unhealthy`: Critical services unavailable

---

## User Registration & Authentication

### POST /auth/register
Register a new user account with email and password.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "name": "John Doe",
  "metadata": {
    "source": "web",
    "referrer": "google"
  }
}
```

**Validation Rules:**
- Email must be valid format
- Password minimum 8 characters with:
  - At least 1 uppercase letter
  - At least 1 lowercase letter
  - At least 1 digit
  - At least 1 special character
- Name 1-100 characters

**Response:** `201 Created`
```json
{
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "email_verified": false,
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z",
    "last_login_at": null,
    "metadata": {
      "source": "web",
      "referrer": "google"
    }
  },
  "tokens": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
    "token_type": "Bearer",
    "expires_in": 900,
    "refresh_expires_in": 604800
  }
}
```

**Errors:**
- `409 Conflict`: User already exists
- `400 Bad Request`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded

### POST /auth/login
Authenticate user with email and password.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "remember_me": false
}
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "token_type": "Bearer",
  "expires_in": 900,
  "refresh_expires_in": 604800
}
```

**Errors:**
- `401 Unauthorized`: Invalid credentials
- `403 Forbidden`: Email verification required or account disabled
- `429 Too Many Requests`: Rate limit or account locked

### POST /auth/refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9..."
}
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "token_type": "Bearer",
  "expires_in": 900,
  "refresh_expires_in": 604800
}
```

**Token Rotation:** When enabled, refresh tokens are rotated on each use for enhanced security.

**Errors:**
- `401 Unauthorized`: Invalid or expired refresh token

### POST /auth/logout
Logout user and invalidate session.

**Headers:** `Authorization: Bearer <access_token>`

**Response:** `204 No Content`

**Errors:**
- `401 Unauthorized`: Invalid or expired token

---

## OAuth2 Authentication

### GET /auth/oauth/providers
Get list of available OAuth providers.

**Response:** `200 OK`
```json
{
  "providers": [
    {
      "name": "google",
      "display_name": "Google",
      "icon_url": "https://developers.google.com/identity/images/g-logo.png",
      "enabled": true
    },
    {
      "name": "github",
      "display_name": "GitHub", 
      "icon_url": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
      "enabled": true
    },
    {
      "name": "microsoft",
      "display_name": "Microsoft",
      "icon_url": "https://docs.microsoft.com/favicon.ico",
      "enabled": true
    }
  ]
}
```

### GET /auth/oauth/{provider}/authorize
Get OAuth authorization URL for provider.

**Parameters:**
- `provider` (path): OAuth provider (`google`, `github`, `microsoft`)
- `redirect_uri` (query): Callback URL after authorization

**Example:**
```bash
GET /auth/oauth/google/authorize?redirect_uri=https://myapp.com/callback
```

**Response:** `200 OK`
```json
{
  "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?client_id=...",
  "state": "abc123"
}
```

**Security Features:**
- State parameter for CSRF protection
- PKCE code challenge for public clients
- Secure random state generation

### GET /auth/oauth/{provider}/callback
Handle OAuth callback and exchange code for tokens.

**Parameters:**
- `provider` (path): OAuth provider
- `code` (query): Authorization code from provider
- `state` (query): State parameter for verification

**Example:**
```bash
GET /auth/oauth/google/callback?code=abc123&state=xyz789
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "token_type": "Bearer",
  "expires_in": 900,
  "refresh_expires_in": 604800
}
```

**Provider-Specific Notes:**

**Google:**
- Scopes: `openid email profile`
- Returns verified email status
- Supports refresh tokens

**GitHub:**
- Scopes: `user:email`
- Requires separate API call for email
- Email verification status available

**Microsoft:**
- Scopes: `openid email profile User.Read`
- Uses Microsoft Graph API
- Emails always considered verified

---

## Password Management

### PUT /auth/password
Update user password (requires authentication).

**Headers:** `Authorization: Bearer <access_token>`

**Request Body:**
```json
{
  "current_password": "OldPass123!",
  "new_password": "NewSecurePass456!"
}
```

**Response:** `200 OK`
```json
{
  "message": "Password updated successfully",
  "success": true
}
```

**Security Features:**
- Verifies current password
- Validates new password strength
- Prevents password reuse (last 5 passwords)
- Invalidates all existing sessions
- Sends email notification

**Errors:**
- `401 Unauthorized`: Current password incorrect
- `400 Bad Request`: New password validation failed

### POST /auth/password/reset
Request password reset email.

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

**Response:** `200 OK`
```json
{
  "message": "Password reset email sent if account exists",
  "success": true
}
```

**Security Features:**
- Always returns success (prevents email enumeration)
- 5-minute reset token expiry
- Single-use tokens
- Secure random token generation

### POST /auth/password/reset/confirm
Confirm password reset with token.

**Request Body:**
```json
{
  "token": "secure_reset_token",
  "new_password": "NewSecurePass456!"
}
```

**Response:** `200 OK`
```json
{
  "message": "Password reset successfully",
  "success": true
}
```

**Security Features:**
- Token consumed after use
- Password validation applied
- All sessions invalidated
- Confirmation email sent

**Errors:**
- `400 Bad Request`: Invalid/expired token or weak password

---

## Email Verification

### POST /auth/email/verify
Verify email address with token.

**Request Body:**
```json
{
  "token": "verification_token"
}
```

**Response:** `200 OK`
```json
{
  "message": "Email verified successfully", 
  "success": true
}
```

**Features:**
- 24-hour token expiry
- Single-use tokens
- Welcome email sent after verification

### POST /auth/email/verify/resend
Resend email verification link.

**Headers:** `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
{
  "message": "Verification email sent",
  "success": true
}
```

**Rate Limits:** 3 requests per hour per user

---

## Session Management

### GET /auth/sessions
Get all active sessions for current user.

**Headers:** `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
{
  "sessions": [
    {
      "id": "session_123",
      "created_at": "2024-01-01T12:00:00Z",
      "last_accessed": "2024-01-01T12:30:00Z",
      "expires_at": "2024-01-08T12:00:00Z",
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0...",
      "active": true
    }
  ],
  "total": 1
}
```

### DELETE /auth/sessions/{session_id}
Terminate a specific session.

**Headers:** `Authorization: Bearer <access_token>`

**Parameters:**
- `session_id` (path): Session identifier to terminate

**Response:** `200 OK`
```json
{
  "message": "Session terminated successfully",
  "success": true
}
```

### DELETE /auth/sessions
Terminate all sessions except current.

**Headers:** `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
{
  "message": "All sessions terminated successfully",
  "success": true
}
```

**Rate Limits:** 5 requests per hour per user

---

## User Profile

### GET /auth/me
Get current user profile information.

**Headers:** `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
{
  "id": "user_123",
  "email": "user@example.com", 
  "name": "John Doe",
  "email_verified": true,
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "last_login_at": "2024-01-01T12:00:00Z",
  "metadata": {
    "profile_picture": "https://example.com/avatar.jpg",
    "locale": "en-US",
    "timezone": "America/New_York"
  }
}
```

**Status:** Currently returns `501 Not Implemented` - endpoint under development.

---

## Error Codes Reference

### HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request successful, no response body |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Authentication required or invalid |
| 403 | Forbidden | Permission denied |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource already exists |
| 422 | Validation Error | Request data validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 501 | Not Implemented | Feature not yet available |
| 503 | Service Unavailable | Dependent service unavailable |

### Application Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `WEAK_PASSWORD` | Password doesn't meet strength requirements | Use stronger password with mixed case, numbers, symbols |
| `EMAIL_EXISTS` | Email already registered | Use different email or login instead |
| `INVALID_CREDENTIALS` | Login failed | Check email and password |
| `EMAIL_NOT_VERIFIED` | Email verification required | Check email for verification link |
| `ACCOUNT_LOCKED` | Too many failed login attempts | Wait for lockout period or reset password |
| `TOKEN_EXPIRED` | JWT token has expired | Refresh token or login again |
| `TOKEN_INVALID` | JWT token is malformed | Provide valid token |
| `SESSION_EXPIRED` | User session has expired | Login again |
| `RATE_LIMITED` | Too many requests | Wait before retrying |

---

## Security Features

### Token Security
- **RS256 asymmetric signing** for token verification
- **15-minute access token expiry** limits exposure window
- **7-day refresh token expiry** balances security and UX
- **Token rotation** prevents token reuse attacks
- **Token blacklisting** for immediate revocation

### Password Security
- **Argon2 hashing** with configurable parameters
- **Password strength validation** enforces complex passwords  
- **Password history** prevents reuse of last 5 passwords
- **Secure random token generation** for resets

### Rate Limiting
- **Adaptive rate limiting** based on user behavior
- **IP-based and user-based** limiting strategies
- **Exponential backoff** for repeated violations
- **Configurable limits** per endpoint

### Session Security
- **Session hijacking detection** via IP/User-Agent validation
- **Concurrent session limits** prevent account sharing
- **Session rotation** on privilege changes
- **Secure cookie settings** with HttpOnly/Secure flags

### Audit & Monitoring
- **Comprehensive security event logging**
- **Failed login attempt tracking**
- **Suspicious activity detection**
- **Real-time security alerts**

---

## Development Tools

### Testing Endpoints
Use the health check endpoint to verify service status:
```bash
curl -X GET "http://localhost:8181/auth/health"
```

### Debugging
All endpoints include detailed error messages in development mode. Enable debug logging to see internal operations:
```python
import logging
logging.getLogger("auth").setLevel(logging.DEBUG)
```

### Monitoring
Monitor rate limits and authentication patterns via the audit logs and security events.

---

**Next:** See [OpenAPI Specification](./02_OPENAPI_SPEC.yaml) for machine-readable API documentation.