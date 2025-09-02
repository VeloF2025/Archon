# Archon Authentication Security Best Practices

## Overview

This document provides comprehensive security best practices for implementing and maintaining secure authentication using the Archon Authentication API. Security is paramount in authentication systems, and following these guidelines will help protect your users and applications from common security threats.

## Table of Contents

1. [Token Security](#token-security)
2. [Password Security](#password-security)
3. [Session Management](#session-management)
4. [OAuth2 Security](#oauth2-security)
5. [Network Security](#network-security)
6. [Client-Side Security](#client-side-security)
7. [Server-Side Security](#server-side-security)
8. [Monitoring & Incident Response](#monitoring--incident-response)
9. [Compliance & Auditing](#compliance--auditing)
10. [Security Checklist](#security-checklist)

---

## Token Security

### JWT Token Best Practices

#### 1. Token Storage

**✅ SECURE APPROACHES:**

```javascript
// Web Applications - HTTP-Only Cookies (Recommended)
// Server-side code to set secure cookies
app.post('/auth/login', async (req, res) => {
    const tokens = await authClient.login(email, password);
    
    // Set HTTP-only, secure cookies
    res.cookie('access_token', tokens.access_token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 15 * 60 * 1000 // 15 minutes
    });
    
    res.cookie('refresh_token', tokens.refresh_token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
    });
    
    res.json({ success: true });
});

// Mobile Applications - Secure Storage
// iOS - Keychain
import { Keychain } from 'react-native-keychain';

const storeTokens = async (accessToken, refreshToken) => {
    await Keychain.setInternetCredentials(
        'archon_tokens',
        'access_token',
        accessToken,
        {
            accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_CURRENT_SET,
            authenticationType: Keychain.AUTHENTICATION_TYPE.DEVICE_PASSCODE_OR_BIOMETRICS,
        }
    );
    
    await Keychain.setInternetCredentials(
        'archon_refresh_tokens',
        'refresh_token',
        refreshToken,
        {
            accessControl: Keychain.ACCESS_CONTROL.DEVICE_PASSCODE,
        }
    );
};

// Android - Encrypted SharedPreferences
import EncryptedStorage from 'react-native-encrypted-storage';

const storeTokens = async (accessToken, refreshToken) => {
    await EncryptedStorage.setItem('access_token', accessToken);
    await EncryptedStorage.setItem('refresh_token', refreshToken);
};
```

**❌ INSECURE APPROACHES:**
```javascript
// NEVER DO THIS - Vulnerable to XSS
localStorage.setItem('access_token', token);
sessionStorage.setItem('access_token', token);

// NEVER DO THIS - Vulnerable to CSRF
document.cookie = `access_token=${token}`;
```

#### 2. Token Transmission

```javascript
// Always use HTTPS in production
const authClient = new ArchonAuthClient('https://api.archon.ai', {
    // Enforce TLS certificate validation
    rejectUnauthorized: true,
    // Use TLS 1.2 minimum
    secureProtocol: 'TLSv1_2_method'
});

// Certificate pinning for mobile applications
const pinnedCert = 'sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=';
const authClient = new ArchonAuthClient('https://api.archon.ai', {
    certificatePinning: {
        hostname: 'api.archon.ai',
        hash: pinnedCert
    }
});
```

#### 3. Token Validation

```python
# Server-side token validation with proper verification
import jwt
from cryptography.hazmat.primitives import serialization

class TokenValidator:
    def __init__(self, public_key_path: str):
        with open(public_key_path, 'rb') as key_file:
            self.public_key = serialization.load_pem_public_key(
                key_file.read()
            )
    
    def validate_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=['RS256'],
                # Strict validation
                options={
                    'verify_signature': True,
                    'verify_exp': True,
                    'verify_nbf': True,
                    'verify_iat': True,
                    'verify_aud': True,
                    'verify_iss': True,
                    'require': ['exp', 'iat', 'nbf', 'sub', 'jti']
                },
                audience='archon-api',
                issuer='archon-auth'
            )
            
            # Additional custom validations
            if not self.is_token_active(payload.get('jti')):
                raise jwt.InvalidTokenError('Token has been revoked')
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthException('Token has expired')
        except jwt.InvalidTokenError as e:
            raise AuthException(f'Invalid token: {str(e)}')
    
    def is_token_active(self, jti: str) -> bool:
        # Check against blacklist/revocation list
        # This should check Redis or database
        return not self.is_token_blacklisted(jti)
```

#### 4. Token Rotation & Revocation

```python
# Implement secure token rotation
class SecureTokenManager:
    def __init__(self, auth_client, token_store):
        self.auth_client = auth_client
        self.token_store = token_store
        self.rotation_threshold = 300  # 5 minutes before expiry
    
    async def get_valid_token(self) -> str:
        """Get valid access token with automatic refresh."""
        current_token = await self.token_store.get_access_token()
        
        if not current_token:
            raise AuthException('No token available')
        
        # Check if token needs refresh
        if self.token_expires_soon(current_token):
            return await self.refresh_token()
        
        return current_token
    
    async def refresh_token(self) -> str:
        """Securely refresh token with rotation detection."""
        refresh_token = await self.token_store.get_refresh_token()
        
        if not refresh_token:
            raise AuthException('No refresh token available')
        
        try:
            new_tokens = await self.auth_client.refresh_tokens(refresh_token)
            
            # Atomic token update to prevent race conditions
            await self.token_store.update_tokens(
                new_tokens.access_token,
                new_tokens.refresh_token
            )
            
            return new_tokens.access_token
            
        except RefreshTokenExpiredError:
            # Clear tokens and require re-authentication
            await self.token_store.clear_tokens()
            raise AuthException('Session expired - please login again')
    
    def token_expires_soon(self, token: str) -> bool:
        """Check if token expires within threshold."""
        try:
            payload = jwt.decode(token, options={'verify_signature': False})
            exp_timestamp = payload.get('exp', 0)
            current_time = time.time()
            
            return (exp_timestamp - current_time) < self.rotation_threshold
            
        except (jwt.InvalidTokenError, KeyError):
            return True  # Assume expired if can't decode
```

---

## Password Security

### Password Requirements

```python
class PasswordValidator:
    """Comprehensive password validation."""
    
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    
    REQUIRED_PATTERNS = {
        'uppercase': r'[A-Z]',
        'lowercase': r'[a-z]',
        'digit': r'\d',
        'special': r'[!@#$%^&*(),.?":{}|<>]'
    }
    
    # Common password blacklist
    COMMON_PASSWORDS = {
        'password', '123456', 'qwerty', 'admin', 'letmein',
        'welcome', 'monkey', 'password123', '12345678'
    }
    
    def validate(self, password: str) -> list[str]:
        """Validate password and return list of errors."""
        errors = []
        
        # Length validation
        if len(password) < self.MIN_LENGTH:
            errors.append(f'Password must be at least {self.MIN_LENGTH} characters')
        
        if len(password) > self.MAX_LENGTH:
            errors.append(f'Password must not exceed {self.MAX_LENGTH} characters')
        
        # Pattern validation
        for name, pattern in self.REQUIRED_PATTERNS.items():
            if not re.search(pattern, password):
                errors.append(f'Password must contain at least one {name} character')
        
        # Common password check
        if password.lower() in self.COMMON_PASSWORDS:
            errors.append('Password is too common')
        
        # Sequential characters check
        if self.has_sequential_chars(password):
            errors.append('Password cannot contain sequential characters')
        
        # Repeated characters check
        if self.has_repeated_chars(password):
            errors.append('Password cannot have more than 2 consecutive identical characters')
        
        return errors
    
    def has_sequential_chars(self, password: str) -> bool:
        """Check for sequential characters like 'abc' or '123'."""
        for i in range(len(password) - 2):
            if (ord(password[i+1]) == ord(password[i]) + 1 and 
                ord(password[i+2]) == ord(password[i]) + 2):
                return True
        return False
    
    def has_repeated_chars(self, password: str) -> bool:
        """Check for repeated characters like 'aaa'."""
        for i in range(len(password) - 2):
            if password[i] == password[i+1] == password[i+2]:
                return True
        return False
    
    def estimate_strength(self, password: str) -> dict:
        """Estimate password strength."""
        score = 0
        feedback = []
        
        # Length scoring
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        
        # Character variety scoring
        for pattern in self.REQUIRED_PATTERNS.values():
            if re.search(pattern, password):
                score += 1
        
        # Length bonus
        if len(password) >= 16:
            score += 1
            feedback.append('Excellent length')
        
        # Determine strength level
        if score >= 7:
            strength = 'very_strong'
        elif score >= 5:
            strength = 'strong'
        elif score >= 3:
            strength = 'moderate'
        else:
            strength = 'weak'
            feedback.append('Consider using a longer password with mixed characters')
        
        return {
            'strength': strength,
            'score': score,
            'feedback': feedback
        }
```

### Password Hashing

```python
import argon2
import secrets

class SecurePasswordHasher:
    """Production-ready password hashing with Argon2."""
    
    def __init__(self):
        # Argon2id parameters tuned for security vs performance
        self.hasher = argon2.PasswordHasher(
            time_cost=3,        # Number of iterations
            memory_cost=65536,  # Memory usage in KB (64MB)
            parallelism=1,      # Number of parallel threads
            hash_len=32,        # Hash output length
            salt_len=16,        # Salt length
            type=argon2.Type.ID # Use Argon2id (hybrid)
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password with secure parameters."""
        return self.hasher.hash(password)
    
    def verify_password(self, hashed: str, password: str) -> bool:
        """Verify password against hash."""
        try:
            self.hasher.verify(hashed, password)
            return True
        except argon2.exceptions.VerifyMismatchError:
            return False
        except argon2.exceptions.VerificationError:
            # Hash corruption or invalid format
            return False
    
    def needs_rehash(self, hashed: str) -> bool:
        """Check if hash needs updating due to parameter changes."""
        return self.hasher.check_needs_rehash(hashed)
    
    def rehash_if_needed(self, hashed: str, password: str) -> str:
        """Rehash password if parameters have changed."""
        if self.needs_rehash(hashed):
            return self.hash_password(password)
        return hashed

# Password history to prevent reuse
class PasswordHistory:
    def __init__(self, redis_client, history_count: int = 5):
        self.redis = redis_client
        self.history_count = history_count
        self.hasher = SecurePasswordHasher()
    
    async def add_password(self, user_id: str, password_hash: str):
        """Add password hash to history."""
        key = f"pwd_history:{user_id}"
        
        # Add new hash to front of list
        await self.redis.lpush(key, password_hash)
        
        # Keep only configured number of passwords
        await self.redis.ltrim(key, 0, self.history_count - 1)
        
        # Set expiry (e.g., 1 year)
        await self.redis.expire(key, 365 * 24 * 3600)
    
    async def is_password_reused(self, user_id: str, new_password: str) -> bool:
        """Check if password was used recently."""
        key = f"pwd_history:{user_id}"
        history = await self.redis.lrange(key, 0, -1)
        
        for old_hash in history:
            if self.hasher.verify_password(old_hash, new_password):
                return True
        
        return False
```

---

## Session Management

### Secure Session Implementation

```python
class SecureSessionManager:
    """Secure session management with comprehensive security features."""
    
    def __init__(self, redis_client, config):
        self.redis = redis_client
        self.config = config
        self.max_sessions_per_user = config.max_concurrent_sessions
        self.session_timeout = config.session_timeout
        self.sliding_window = config.sliding_window
    
    async def create_session(
        self, 
        user_id: str, 
        request_context: dict
    ) -> str:
        """Create secure session with comprehensive metadata."""
        session_id = self.generate_secure_session_id()
        
        # Extract security-relevant information
        ip_address = request_context.get('ip_address', 'unknown')
        user_agent = request_context.get('user_agent', 'unknown')
        location = await self.geolocate_ip(ip_address)
        
        session_data = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'location': location,
            'device_fingerprint': self.generate_device_fingerprint(request_context),
            'security_flags': {
                'suspicious_location': await self.is_suspicious_location(user_id, location),
                'new_device': await self.is_new_device(user_id, request_context),
                'concurrent_session': await self.has_concurrent_sessions(user_id)
            }
        }
        
        # Store session
        session_key = f"session:{session_id}"
        await self.redis.hset(session_key, mapping=session_data)
        await self.redis.expire(session_key, self.session_timeout)
        
        # Manage concurrent sessions
        await self.manage_concurrent_sessions(user_id, session_id)
        
        # Log session creation
        await self.log_security_event('session_created', user_id, session_data)
        
        return session_id
    
    async def validate_session(
        self, 
        session_id: str, 
        request_context: dict
    ) -> dict:
        """Validate session with security checks."""
        session_key = f"session:{session_id}"
        session_data = await self.redis.hgetall(session_key)
        
        if not session_data:
            raise SessionException('Session not found')
        
        # Security validation
        current_ip = request_context.get('ip_address')
        current_ua = request_context.get('user_agent')
        
        # Check for session hijacking indicators
        if self.config.strict_ip_validation:
            if session_data.get('ip_address') != current_ip:
                await self.terminate_session(session_id, 'ip_change')
                raise SessionException('Session terminated due to IP change')
        
        if self.config.strict_ua_validation:
            if session_data.get('user_agent') != current_ua:
                await self.log_security_event(
                    'suspicious_activity',
                    session_data.get('user_id'),
                    {
                        'type': 'user_agent_change',
                        'original_ua': session_data.get('user_agent'),
                        'current_ua': current_ua
                    }
                )
        
        # Update last accessed time (sliding window)
        if self.sliding_window:
            await self.redis.hset(session_key, 'last_accessed', time.time())
            await self.redis.expire(session_key, self.session_timeout)
        
        return session_data
    
    async def terminate_session(self, session_id: str, reason: str):
        """Securely terminate session."""
        session_key = f"session:{session_id}"
        session_data = await self.redis.hgetall(session_key)
        
        if session_data:
            # Log termination
            await self.log_security_event(
                'session_terminated',
                session_data.get('user_id'),
                {'session_id': session_id, 'reason': reason}
            )
        
        # Remove session
        await self.redis.delete(session_key)
        
        # Blacklist associated tokens
        await self.blacklist_session_tokens(session_id)
    
    def generate_secure_session_id(self) -> str:
        """Generate cryptographically secure session ID."""
        # 256 bits of entropy
        return secrets.token_urlsafe(32)
    
    def generate_device_fingerprint(self, request_context: dict) -> str:
        """Generate device fingerprint for tracking."""
        components = [
            request_context.get('user_agent', ''),
            request_context.get('accept_language', ''),
            request_context.get('screen_resolution', ''),
            request_context.get('timezone', ''),
            request_context.get('platform', '')
        ]
        
        fingerprint_string = '|'.join(components)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
    
    async def detect_session_anomalies(self, user_id: str, session_data: dict):
        """Detect suspicious session patterns."""
        anomalies = []
        
        # Check for concurrent sessions from different locations
        active_sessions = await self.get_active_sessions(user_id)
        
        locations = [s.get('location', {}) for s in active_sessions]
        unique_countries = set(l.get('country') for l in locations if l.get('country'))
        
        if len(unique_countries) > 1:
            anomalies.append({
                'type': 'multiple_countries',
                'countries': list(unique_countries)
            })
        
        # Check for rapid session creation
        recent_sessions = await self.get_recent_sessions(user_id, hours=1)
        if len(recent_sessions) > 5:
            anomalies.append({
                'type': 'rapid_session_creation',
                'count': len(recent_sessions)
            })
        
        # Check for impossible travel
        if len(active_sessions) >= 2:
            for i, session1 in enumerate(active_sessions):
                for session2 in active_sessions[i+1:]:
                    if self.is_impossible_travel(session1, session2):
                        anomalies.append({
                            'type': 'impossible_travel',
                            'session1': session1['id'],
                            'session2': session2['id']
                        })
        
        if anomalies:
            await self.log_security_event(
                'session_anomalies',
                user_id,
                {'anomalies': anomalies}
            )
        
        return anomalies
```

---

## OAuth2 Security

### PKCE Implementation

```python
import base64
import hashlib
import secrets
from urllib.parse import urlencode

class PKCEFlow:
    """Secure OAuth2 PKCE implementation."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.code_challenge_method = 'S256'
        self.state_ttl = 300  # 5 minutes
    
    def generate_pkce_parameters(self) -> dict:
        """Generate PKCE code verifier and challenge."""
        # Generate code verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
        
        # Generate code challenge
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return {
            'code_verifier': code_verifier,
            'code_challenge': code_challenge,
            'code_challenge_method': self.code_challenge_method
        }
    
    async def initiate_oauth_flow(
        self, 
        provider: str, 
        redirect_uri: str,
        scopes: list = None
    ) -> dict:
        """Initiate secure OAuth2 flow with PKCE."""
        
        # Generate PKCE parameters
        pkce_params = self.generate_pkce_parameters()
        
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        
        # Store PKCE parameters and state
        state_key = f"oauth_state:{state}"
        await self.redis.setex(
            state_key,
            self.state_ttl,
            {
                'provider': provider,
                'redirect_uri': redirect_uri,
                'code_verifier': pkce_params['code_verifier'],
                'timestamp': time.time(),
                'scopes': scopes or []
            }
        )
        
        # Build authorization URL
        auth_params = {
            'client_id': self.get_client_id(provider),
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': ' '.join(scopes or []),
            'state': state,
            'code_challenge': pkce_params['code_challenge'],
            'code_challenge_method': pkce_params['code_challenge_method']
        }
        
        # Add provider-specific parameters
        auth_params.update(self.get_provider_params(provider))
        
        authorization_url = (
            f"{self.get_auth_endpoint(provider)}?"
            f"{urlencode(auth_params)}"
        )
        
        return {
            'authorization_url': authorization_url,
            'state': state
        }
    
    async def handle_oauth_callback(
        self, 
        provider: str, 
        code: str, 
        state: str
    ) -> dict:
        """Handle OAuth2 callback with security validation."""
        
        # Validate and retrieve state
        state_key = f"oauth_state:{state}"
        state_data = await self.redis.get(state_key)
        
        if not state_data:
            raise OAuthException('Invalid or expired state parameter')
        
        # Remove state to prevent replay attacks
        await self.redis.delete(state_key)
        
        # Validate provider matches
        if state_data['provider'] != provider:
            raise OAuthException('Provider mismatch')
        
        # Exchange authorization code for tokens
        token_response = await self.exchange_code(
            provider=provider,
            code=code,
            redirect_uri=state_data['redirect_uri'],
            code_verifier=state_data['code_verifier']
        )
        
        return token_response
    
    async def exchange_code(
        self, 
        provider: str, 
        code: str, 
        redirect_uri: str,
        code_verifier: str
    ) -> dict:
        """Exchange authorization code for tokens."""
        token_endpoint = self.get_token_endpoint(provider)
        
        token_params = {
            'client_id': self.get_client_id(provider),
            'client_secret': self.get_client_secret(provider),
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': redirect_uri,
            'code_verifier': code_verifier
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_endpoint,
                data=token_params,
                headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            )
            
            if response.status_code != 200:
                raise OAuthException(
                    f"Token exchange failed: {response.text}"
                )
            
            return response.json()
```

### OAuth2 Security Validation

```python
class OAuthSecurityValidator:
    """OAuth2 security validation and monitoring."""
    
    def __init__(self, audit_logger):
        self.audit_logger = audit_logger
        self.suspicious_patterns = {
            'rapid_requests': 10,  # Max requests per minute
            'failed_attempts': 5,  # Max failed attempts per hour
            'multiple_ips': 3      # Max IPs per user per hour
        }
    
    async def validate_oauth_request(
        self, 
        provider: str, 
        client_ip: str,
        user_agent: str,
        additional_context: dict = None
    ):
        """Validate OAuth request for suspicious patterns."""
        
        # Check rate limits
        await self.check_rate_limits(client_ip, user_agent)
        
        # Validate redirect URI
        await self.validate_redirect_uri(
            provider, 
            additional_context.get('redirect_uri')
        )
        
        # Check for suspicious patterns
        await self.detect_suspicious_activity(
            provider, 
            client_ip, 
            user_agent,
            additional_context
        )
    
    async def check_rate_limits(self, client_ip: str, user_agent: str):
        """Check OAuth-specific rate limits."""
        
        # Check IP-based rate limiting
        ip_key = f"oauth_rate:ip:{client_ip}"
        ip_requests = await self.redis.incr(ip_key)
        await self.redis.expire(ip_key, 60)  # 1-minute window
        
        if ip_requests > self.suspicious_patterns['rapid_requests']:
            await self.audit_logger.log_security_event(
                'oauth_rate_limit_exceeded',
                context={'ip': client_ip, 'requests': ip_requests}
            )
            raise RateLimitException('OAuth rate limit exceeded')
    
    async def validate_redirect_uri(self, provider: str, redirect_uri: str):
        """Validate redirect URI against allowlist."""
        allowed_uris = await self.get_allowed_redirect_uris(provider)
        
        if redirect_uri not in allowed_uris:
            await self.audit_logger.log_security_event(
                'invalid_redirect_uri',
                context={
                    'provider': provider,
                    'requested_uri': redirect_uri,
                    'allowed_uris': allowed_uris
                }
            )
            raise OAuthException('Invalid redirect URI')
    
    async def detect_suspicious_activity(
        self, 
        provider: str,
        client_ip: str, 
        user_agent: str,
        context: dict
    ):
        """Detect suspicious OAuth activity patterns."""
        
        suspicion_score = 0
        flags = []
        
        # Check for unusual geographic access
        location = await self.geolocate_ip(client_ip)
        if await self.is_high_risk_location(location):
            suspicion_score += 2
            flags.append('high_risk_location')
        
        # Check for bot-like user agents
        if self.is_bot_user_agent(user_agent):
            suspicion_score += 1
            flags.append('bot_user_agent')
        
        # Check for rapid provider switching
        if await self.has_rapid_provider_switching(client_ip):
            suspicion_score += 2
            flags.append('rapid_provider_switching')
        
        # Check for known malicious IPs
        if await self.is_malicious_ip(client_ip):
            suspicion_score += 3
            flags.append('malicious_ip')
        
        if suspicion_score >= 3:
            await self.audit_logger.log_security_event(
                'suspicious_oauth_activity',
                context={
                    'provider': provider,
                    'ip': client_ip,
                    'user_agent': user_agent,
                    'suspicion_score': suspicion_score,
                    'flags': flags,
                    'additional_context': context
                }
            )
            
            # Implement additional security measures
            if suspicion_score >= 5:
                # Block request for high suspicion
                raise SecurityException('Request blocked due to suspicious activity')
            else:
                # Require additional verification
                context['requires_mfa'] = True
```

---

## Network Security

### TLS/SSL Configuration

```nginx
# nginx.conf - Production TLS configuration
server {
    listen 443 ssl http2;
    server_name api.archon.ai;
    
    # TLS configuration
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # Use only secure protocols
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    
    # Secure cipher suites
    ssl_ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; object-src 'none';" always;
    
    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /path/to/chain.crt;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;
    
    location /auth/ {
        proxy_pass http://auth-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.archon.ai;
    return 301 https://$server_name$request_uri;
}
```

### API Security Headers

```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.archon.ai", "*.archon.ai"]
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.archon.ai"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["authorization", "content-type"],
    expose_headers=["x-ratelimit-limit", "x-ratelimit-remaining"]
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add comprehensive security headers."""
    response = await call_next(request)
    
    # Security headers
    security_headers = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none';",
        "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
        "X-Permitted-Cross-Domain-Policies": "none",
    }
    
    for header, value in security_headers.items():
        response.headers[header] = value
    
    # Remove server information
    response.headers.pop("server", None)
    
    return response
```

---

## Monitoring & Incident Response

### Security Event Monitoring

```python
class SecurityEventMonitor:
    """Comprehensive security event monitoring and alerting."""
    
    def __init__(self, config):
        self.config = config
        self.alert_thresholds = {
            'failed_logins': {'count': 10, 'window': 300},  # 10 in 5 minutes
            'token_attacks': {'count': 5, 'window': 60},    # 5 in 1 minute
            'session_anomalies': {'count': 3, 'window': 600}, # 3 in 10 minutes
            'oauth_abuse': {'count': 20, 'window': 300}     # 20 in 5 minutes
        }
    
    async def log_security_event(
        self, 
        event_type: str,
        user_id: str = None,
        ip_address: str = None,
        details: dict = None
    ):
        """Log security event with structured data."""
        
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details or {},
            'severity': self.calculate_severity(event_type, details),
            'source': 'archon-auth',
            'version': '1.0.0'
        }
        
        # Store in multiple locations for reliability
        await self.store_event_multiple_locations(event)
        
        # Check for alert conditions
        await self.check_alert_conditions(event)
        
        # Real-time threat detection
        await self.analyze_threat_patterns(event)
    
    async def check_alert_conditions(self, event: dict):
        """Check if event triggers alert thresholds."""
        event_type = event['event_type']
        
        if event_type in self.alert_thresholds:
            threshold = self.alert_thresholds[event_type]
            
            # Count recent events of same type
            recent_count = await self.count_recent_events(
                event_type, 
                threshold['window']
            )
            
            if recent_count >= threshold['count']:
                await self.trigger_security_alert(
                    event_type,
                    recent_count,
                    threshold,
                    event
                )
    
    async def trigger_security_alert(
        self, 
        event_type: str,
        count: int, 
        threshold: dict,
        triggering_event: dict
    ):
        """Trigger security alert and response actions."""
        
        alert = {
            'alert_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': 'HIGH' if count > threshold['count'] * 2 else 'MEDIUM',
            'count': count,
            'threshold': threshold,
            'triggering_event': triggering_event,
            'automated_actions': []
        }
        
        # Automated response actions
        if event_type == 'failed_logins':
            await self.auto_block_ip(triggering_event.get('ip_address'))
            alert['automated_actions'].append('ip_blocked')
        
        elif event_type == 'token_attacks':
            await self.invalidate_suspicious_tokens(triggering_event)
            alert['automated_actions'].append('tokens_invalidated')
        
        # Send notifications
        await self.send_alert_notifications(alert)
        
        # Store alert
        await self.store_alert(alert)
    
    async def analyze_threat_patterns(self, event: dict):
        """Analyze patterns for advanced threat detection."""
        
        # Implement ML-based anomaly detection
        anomaly_score = await self.calculate_anomaly_score(event)
        
        if anomaly_score > 0.8:  # High anomaly threshold
            await self.investigate_anomaly(event, anomaly_score)
    
    async def generate_security_report(self, timeframe: str = '24h'):
        """Generate comprehensive security report."""
        
        report = {
            'timeframe': timeframe,
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {},
            'top_threats': [],
            'affected_users': [],
            'geographic_analysis': {},
            'recommendations': []
        }
        
        # Event statistics
        report['summary'] = await self.get_event_statistics(timeframe)
        
        # Top threats analysis
        report['top_threats'] = await self.analyze_top_threats(timeframe)
        
        # Geographic analysis
        report['geographic_analysis'] = await self.analyze_geographic_patterns(timeframe)
        
        # Security recommendations
        report['recommendations'] = await self.generate_recommendations(report)
        
        return report

# Real-time alerting system
class RealTimeAlerting:
    def __init__(self, notification_channels):
        self.channels = notification_channels
    
    async def send_immediate_alert(self, alert: dict):
        """Send immediate high-priority alert."""
        
        if alert['severity'] == 'CRITICAL':
            # Page on-call engineer
            await self.channels['pager'].send(alert)
        
        # Send to security team
        await self.channels['slack'].send_security_alert(alert)
        
        # Log to SIEM
        await self.channels['siem'].ingest_alert(alert)
        
        # Email notifications
        if alert['severity'] in ['HIGH', 'CRITICAL']:
            await self.channels['email'].send_security_email(alert)
```

### Incident Response Procedures

```python
class IncidentResponse:
    """Automated incident response procedures."""
    
    def __init__(self, config):
        self.config = config
        self.response_procedures = {
            'brute_force_attack': self.handle_brute_force,
            'token_compromise': self.handle_token_compromise,
            'account_takeover': self.handle_account_takeover,
            'oauth_abuse': self.handle_oauth_abuse,
            'session_hijacking': self.handle_session_hijacking
        }
    
    async def handle_security_incident(self, incident_type: str, context: dict):
        """Handle security incident with appropriate response."""
        
        if incident_type in self.response_procedures:
            handler = self.response_procedures[incident_type]
            await handler(context)
        else:
            await self.handle_generic_incident(incident_type, context)
    
    async def handle_brute_force(self, context: dict):
        """Handle brute force attack."""
        
        # Immediate actions
        ip_address = context.get('ip_address')
        if ip_address:
            # Block IP at firewall level
            await self.block_ip_address(ip_address, duration=3600)
            
            # Rate limit related IP ranges
            await self.rate_limit_ip_range(ip_address, factor=10)
        
        # Account protection
        targeted_accounts = context.get('targeted_accounts', [])
        for account in targeted_accounts:
            # Temporarily lock account
            await self.temporarily_lock_account(account, duration=300)
            
            # Require MFA for next login
            await self.require_mfa_next_login(account)
            
            # Notify user
            await self.notify_user_of_attack(account, 'brute_force')
    
    async def handle_token_compromise(self, context: dict):
        """Handle potential token compromise."""
        
        # Immediate token revocation
        compromised_tokens = context.get('tokens', [])
        for token in compromised_tokens:
            await self.revoke_token(token)
        
        # Invalidate all user sessions
        user_id = context.get('user_id')
        if user_id:
            await self.invalidate_all_user_sessions(user_id)
            
            # Force password reset
            await self.force_password_reset(user_id)
            
            # Notify user immediately
            await self.send_urgent_security_notification(
                user_id, 
                'token_compromise'
            )
    
    async def handle_account_takeover(self, context: dict):
        """Handle account takeover attempt."""
        
        user_id = context.get('user_id')
        
        # Immediate account lockdown
        await self.lock_account_immediately(user_id)
        
        # Invalidate all authentication
        await self.invalidate_all_user_auth(user_id)
        
        # Alert user through all available channels
        await self.multi_channel_user_alert(user_id, 'account_takeover')
        
        # Create support ticket for manual review
        await self.create_security_support_ticket(user_id, context)
        
        # Alert security team
        await self.alert_security_team('account_takeover', context)
    
    async def generate_incident_report(self, incident_id: str):
        """Generate comprehensive incident report."""
        
        incident = await self.get_incident_details(incident_id)
        
        report = {
            'incident_id': incident_id,
            'timeline': await self.build_incident_timeline(incident),
            'impact_assessment': await self.assess_incident_impact(incident),
            'root_cause_analysis': await self.perform_root_cause_analysis(incident),
            'response_actions': await self.document_response_actions(incident),
            'lessons_learned': await self.extract_lessons_learned(incident),
            'preventive_measures': await self.recommend_preventive_measures(incident)
        }
        
        return report
```

---

## Security Checklist

### Pre-Production Security Checklist

**Authentication & Authorization:**
- [ ] JWT tokens use RS256 asymmetric signing
- [ ] Access tokens expire within 15 minutes
- [ ] Refresh tokens rotate on use
- [ ] Token blacklisting implemented
- [ ] Strong password requirements enforced
- [ ] Password hashing uses Argon2 with proper parameters
- [ ] Account lockout after failed attempts
- [ ] Password history prevents reuse

**Session Security:**
- [ ] Secure session ID generation (256-bit entropy)
- [ ] Session fixation protection
- [ ] Concurrent session limits enforced
- [ ] Session timeout properly configured
- [ ] Session hijacking detection
- [ ] Secure cookie settings (HTTPOnly, Secure, SameSite)

**OAuth2 Security:**
- [ ] PKCE implemented for public clients
- [ ] State parameter used for CSRF protection
- [ ] Redirect URI validation
- [ ] Authorization code single-use enforcement
- [ ] Scope validation and restriction

**Network Security:**
- [ ] TLS 1.2+ enforced
- [ ] Strong cipher suites only
- [ ] HSTS headers implemented
- [ ] Certificate pinning (mobile apps)
- [ ] Security headers configured
- [ ] CORS properly configured

**Input Validation:**
- [ ] All inputs validated server-side
- [ ] SQL injection prevention
- [ ] XSS protection implemented
- [ ] CSRF protection enabled
- [ ] File upload security (if applicable)

**Monitoring & Logging:**
- [ ] Security events logged
- [ ] Failed authentication attempts tracked
- [ ] Suspicious activity detection
- [ ] Real-time alerting configured
- [ ] Log integrity protection
- [ ] SIEM integration

**Data Protection:**
- [ ] PII encryption at rest
- [ ] Secure data transmission
- [ ] Key management procedures
- [ ] Data retention policies
- [ ] Right to be forgotten compliance
- [ ] Backup encryption

**Infrastructure Security:**
- [ ] WAF configured and tested
- [ ] DDoS protection enabled
- [ ] Firewall rules reviewed
- [ ] Intrusion detection system
- [ ] Regular security patches
- [ ] Vulnerability scanning

### Post-Deployment Monitoring Checklist

**Daily:**
- [ ] Review security alerts
- [ ] Check failed login patterns
- [ ] Monitor rate limiting metrics
- [ ] Verify backup integrity

**Weekly:**
- [ ] Analyze security event trends
- [ ] Review user access patterns
- [ ] Check SSL certificate status
- [ ] Update threat intelligence

**Monthly:**
- [ ] Security metrics review
- [ ] Incident response drill
- [ ] Access control audit
- [ ] Dependency vulnerability scan

**Quarterly:**
- [ ] Comprehensive security assessment
- [ ] Penetration testing
- [ ] Security policy review
- [ ] Compliance audit
- [ ] Business continuity testing

---

**Next:** See [Troubleshooting Guide](./05_TROUBLESHOOTING.md) for common issues and solutions.