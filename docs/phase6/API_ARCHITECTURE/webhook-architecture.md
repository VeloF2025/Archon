# Archon Phase 6 Webhook Architecture

**Version**: 2.0.0  
**Date**: August 31, 2025  
**Status**: Production Ready  

## Overview

This document defines the comprehensive webhook architecture for the Archon Phase 6 authentication system, enabling real-time event notifications for authentication events, agent activities, security incidents, and system status changes.

## Webhook Event Types

### Authentication Events

#### User Authentication Events
```yaml
user.login:
  description: User successfully logged in
  payload:
    user_id: string
    email: string
    login_method: enum [password, oauth, sso]
    device_info: object
    ip_address: string
    location: string
    session_id: string
    timestamp: datetime

user.login_failed:
  description: User login attempt failed
  payload:
    email: string
    failure_reason: enum [invalid_credentials, account_locked, rate_limited]
    ip_address: string
    attempts_remaining: integer
    lockout_duration: integer
    timestamp: datetime

user.logout:
  description: User logged out
  payload:
    user_id: string
    session_id: string
    logout_type: enum [user_initiated, session_expired, admin_terminated]
    session_duration: integer
    timestamp: datetime

user.registered:
  description: New user registration
  payload:
    user_id: string
    email: string
    name: string
    registration_source: string
    email_verification_required: boolean
    timestamp: datetime

user.email_verified:
  description: User email verified
  payload:
    user_id: string
    email: string
    verification_method: string
    timestamp: datetime

user.password_changed:
  description: User password changed
  payload:
    user_id: string
    change_method: enum [user_initiated, admin_reset, forced_reset]
    ip_address: string
    timestamp: datetime
```

#### Session Events
```yaml
session.created:
  description: New user session created
  payload:
    session_id: string
    user_id: string
    device_info: object
    ip_address: string
    remember_me: boolean
    expires_at: datetime
    timestamp: datetime

session.extended:
  description: Session extended/refreshed
  payload:
    session_id: string
    user_id: string
    old_expires_at: datetime
    new_expires_at: datetime
    timestamp: datetime

session.terminated:
  description: Session terminated
  payload:
    session_id: string
    user_id: string
    termination_reason: enum [logout, expiry, security, admin]
    timestamp: datetime
```

### Agent Events

#### Agent Authentication Events
```yaml
agent.authenticated:
  description: Agent successfully authenticated
  payload:
    agent_id: string
    agent_type: string
    user_id: string
    capabilities_requested: array[string]
    capabilities_granted: array[string]
    capabilities_denied: array[string]
    restrictions: object
    token_expires_at: datetime
    task_context: object
    timestamp: datetime

agent.authentication_failed:
  description: Agent authentication failed
  payload:
    agent_type: string
    user_id: string
    failure_reason: enum [insufficient_permissions, invalid_capabilities, security_policy_violation]
    requested_capabilities: array[string]
    timestamp: datetime

agent.capability_used:
  description: Agent used a specific capability
  payload:
    agent_id: string
    agent_type: string
    user_id: string
    capability: string
    resource: string
    action_details: object
    timestamp: datetime

agent.capability_denied:
  description: Agent capability access denied
  payload:
    agent_id: string
    agent_type: string
    user_id: string
    capability: string
    resource: string
    denial_reason: string
    timestamp: datetime

agent.token_expired:
  description: Agent token expired
  payload:
    agent_id: string
    agent_type: string
    user_id: string
    token_duration: integer
    renewal_required: boolean
    timestamp: datetime
```

### Security Events

#### Security Incidents
```yaml
security.suspicious_login:
  description: Suspicious login attempt detected
  payload:
    user_id: string
    email: string
    suspicious_indicators: array[string]
    ip_address: string
    device_info: object
    location: string
    risk_score: float
    action_taken: string
    timestamp: datetime

security.rate_limit_exceeded:
  description: Rate limit exceeded for endpoint
  payload:
    endpoint: string
    ip_address: string
    user_id: string
    limit_type: string
    current_rate: integer
    limit_threshold: integer
    window_duration: integer
    timestamp: datetime

security.account_locked:
  description: User account locked due to security
  payload:
    user_id: string
    email: string
    lock_reason: string
    failed_attempts: integer
    lockout_duration: integer
    unlock_method: string
    timestamp: datetime

security.password_breach_detected:
  description: Password found in breach database
  payload:
    user_id: string
    email: string
    breach_sources: array[string]
    force_reset_required: boolean
    timestamp: datetime

security.unauthorized_access_attempt:
  description: Unauthorized access attempt
  payload:
    resource: string
    attempted_action: string
    user_id: string
    agent_id: string
    denial_reason: string
    ip_address: string
    timestamp: datetime
```

### System Events

#### Performance Events
```yaml
system.performance_degraded:
  description: System performance below thresholds
  payload:
    service: string
    metric: string
    current_value: float
    threshold: float
    duration: integer
    affected_endpoints: array[string]
    timestamp: datetime

system.high_load:
  description: System experiencing high load
  payload:
    load_type: enum [cpu, memory, database, redis]
    current_utilization: float
    threshold: float
    affected_services: array[string]
    auto_scaling_triggered: boolean
    timestamp: datetime
```

#### Service Events
```yaml
system.service_unavailable:
  description: Service became unavailable
  payload:
    service: string
    error_message: string
    affected_endpoints: array[string]
    estimated_recovery_time: integer
    fallback_enabled: boolean
    timestamp: datetime

system.service_recovered:
  description: Service recovered from outage
  payload:
    service: string
    downtime_duration: integer
    recovery_method: string
    affected_users_count: integer
    timestamp: datetime
```

## Webhook Configuration

### Webhook Registration API

```python
# POST /webhooks
{
  "url": "https://your-app.com/webhooks/archon",
  "events": [
    "user.login",
    "user.logout", 
    "agent.authenticated",
    "security.suspicious_login"
  ],
  "secret": "your-webhook-secret-key",
  "active": true,
  "description": "Production webhook for user events",
  "headers": {
    "Authorization": "Bearer your-api-key",
    "X-Custom-Header": "custom-value"
  },
  "retry_policy": {
    "max_attempts": 3,
    "backoff_strategy": "exponential",
    "initial_delay": 1,
    "max_delay": 60
  },
  "filters": {
    "user_roles": ["admin", "user"],
    "agent_types": ["code_implementer", "security_auditor"]
  }
}
```

### Webhook Management

```yaml
# GET /webhooks - List webhooks
# GET /webhooks/{webhook_id} - Get webhook details
# PUT /webhooks/{webhook_id} - Update webhook
# DELETE /webhooks/{webhook_id} - Delete webhook
# POST /webhooks/{webhook_id}/test - Test webhook delivery
# GET /webhooks/{webhook_id}/deliveries - Get delivery history
# POST /webhooks/{webhook_id}/deliveries/{delivery_id}/redeliver - Redeliver webhook
```

## Webhook Payload Structure

### Standard Payload Format

```json
{
  "id": "evt_1234567890abcdef",
  "type": "user.login",
  "version": "2.0.0",
  "timestamp": "2025-08-31T10:00:00Z",
  "environment": "production",
  "source": {
    "service": "archon-auth-service",
    "version": "2.0.0",
    "instance_id": "auth-service-pod-123"
  },
  "data": {
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "email": "user@example.com",
    "login_method": "password",
    "device_info": {
      "device_id": "device_abc123",
      "user_agent": "Mozilla/5.0...",
      "platform": "web"
    },
    "ip_address": "192.168.1.100",
    "location": "San Francisco, CA, US",
    "session_id": "sess_1234567890",
    "timestamp": "2025-08-31T10:00:00Z"
  },
  "context": {
    "trace_id": "trace_abc123def456",
    "request_id": "req_1234567890",
    "user_agent": "ArchonAuthClient/2.0.0"
  }
}
```

### Security Headers

All webhook deliveries include security headers:

```
X-Archon-Signature: sha256=hash_of_payload_with_secret
X-Archon-Event-Type: user.login
X-Archon-Event-ID: evt_1234567890abcdef
X-Archon-Timestamp: 2025-08-31T10:00:00Z
X-Archon-Version: 2.0.0
```

## Webhook Security

### Signature Verification

```python
# Python webhook signature verification
import hashlib
import hmac
import json

def verify_webhook_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify webhook signature."""
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Remove 'sha256=' prefix if present
    if signature.startswith('sha256='):
        signature = signature[7:]
    
    return hmac.compare_digest(expected_signature, signature)

# Usage in webhook handler
@app.route('/webhooks/archon', methods=['POST'])
def handle_archon_webhook():
    payload = request.get_data(as_text=True)
    signature = request.headers.get('X-Archon-Signature')
    
    if not verify_webhook_signature(payload, signature, WEBHOOK_SECRET):
        return 'Invalid signature', 401
    
    event_data = json.loads(payload)
    process_webhook_event(event_data)
    
    return 'OK', 200
```

```javascript
// JavaScript webhook signature verification
const crypto = require('crypto');

function verifyWebhookSignature(payload, signature, secret) {
    const expectedSignature = crypto
        .createHmac('sha256', secret)
        .update(payload, 'utf8')
        .digest('hex');
    
    // Remove 'sha256=' prefix if present
    const actualSignature = signature.startsWith('sha256=') 
        ? signature.slice(7) 
        : signature;
    
    return crypto.timingSafeEqual(
        Buffer.from(expectedSignature, 'hex'),
        Buffer.from(actualSignature, 'hex')
    );
}

// Express.js webhook handler
app.post('/webhooks/archon', express.raw({type: 'application/json'}), (req, res) => {
    const payload = req.body.toString();
    const signature = req.headers['x-archon-signature'];
    
    if (!verifyWebhookSignature(payload, signature, WEBHOOK_SECRET)) {
        return res.status(401).send('Invalid signature');
    }
    
    const eventData = JSON.parse(payload);
    processWebhookEvent(eventData);
    
    res.status(200).send('OK');
});
```

### IP Whitelist

Configure IP whitelist for webhook deliveries:

```yaml
webhook_security:
  ip_whitelist:
    - "52.74.223.119"    # Archon webhook IP 1
    - "52.74.223.120"    # Archon webhook IP 2
    - "10.0.0.0/8"       # Internal network
  
  require_https: true
  signature_required: true
  timeout_seconds: 10
```

## Delivery Guarantees

### Retry Policy

```yaml
retry_policy:
  max_attempts: 3
  backoff_strategy: exponential  # linear, exponential, fixed
  initial_delay: 1               # seconds
  max_delay: 60                  # seconds
  retry_codes: [500, 502, 503, 504, 408, 429]
  
delivery_timeout: 30             # seconds
```

### Delivery Status Tracking

```json
{
  "delivery_id": "del_1234567890",
  "webhook_id": "whk_abcdef123456",
  "event_id": "evt_1234567890abcdef",
  "status": "delivered",  // pending, delivered, failed, retrying
  "attempts": [
    {
      "attempt": 1,
      "timestamp": "2025-08-31T10:00:00Z",
      "status_code": 200,
      "response_time_ms": 150,
      "response_headers": {
        "content-type": "application/json"
      }
    }
  ],
  "next_retry": null,  // timestamp for next retry if failed
  "created_at": "2025-08-31T10:00:00Z",
  "delivered_at": "2025-08-31T10:00:00Z"
}
```

## Webhook Implementation Examples

### Full Webhook Handler (Python/FastAPI)

```python
# File: webhook_handler.py
from fastapi import FastAPI, Request, HTTPException, Header
from typing import Optional, Dict, Any
import hmac
import hashlib
import json
import logging
from datetime import datetime

app = FastAPI()
logger = logging.getLogger(__name__)

WEBHOOK_SECRET = "your-webhook-secret-key"

class WebhookHandler:
    def __init__(self):
        self.event_handlers = {
            'user.login': self.handle_user_login,
            'user.logout': self.handle_user_logout,
            'user.registered': self.handle_user_registered,
            'agent.authenticated': self.handle_agent_authenticated,
            'security.suspicious_login': self.handle_suspicious_login,
            'security.rate_limit_exceeded': self.handle_rate_limit_exceeded,
            'system.performance_degraded': self.handle_performance_degraded
        }
    
    def verify_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature."""
        if not signature:
            return False
        
        expected_signature = hmac.new(
            WEBHOOK_SECRET.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Remove 'sha256=' prefix
        actual_signature = signature[7:] if signature.startswith('sha256=') else signature
        
        return hmac.compare_digest(expected_signature, actual_signature)
    
    async def process_event(self, event_data: Dict[str, Any]) -> bool:
        """Process webhook event."""
        event_type = event_data.get('type')
        
        if event_type not in self.event_handlers:
            logger.warning(f"Unknown event type: {event_type}")
            return True  # Return success to avoid retries
        
        try:
            handler = self.event_handlers[event_type]
            await handler(event_data)
            return True
        except Exception as e:
            logger.error(f"Error processing {event_type}: {e}")
            return False
    
    async def handle_user_login(self, event_data: Dict) -> None:
        """Handle user login event."""
        data = event_data['data']
        user_id = data['user_id']
        email = data['email']
        ip_address = data['ip_address']
        
        logger.info(f"User {email} logged in from {ip_address}")
        
        # Update user analytics
        await self.update_user_analytics(user_id, 'login')
        
        # Check for location-based alerts
        if await self.is_unusual_location(user_id, data.get('location')):
            await self.send_security_alert(user_id, 'unusual_location_login')
        
        # Update real-time dashboard
        await self.update_dashboard_metrics('active_users', 1)
    
    async def handle_user_logout(self, event_data: Dict) -> None:
        """Handle user logout event."""
        data = event_data['data']
        user_id = data['user_id']
        session_duration = data.get('session_duration', 0)
        
        logger.info(f"User {user_id} logged out after {session_duration}s")
        
        # Update session analytics
        await self.update_session_analytics(user_id, session_duration)
        
        # Update dashboard
        await self.update_dashboard_metrics('active_users', -1)
    
    async def handle_user_registered(self, event_data: Dict) -> None:
        """Handle user registration event."""
        data = event_data['data']
        user_id = data['user_id']
        email = data['email']
        
        logger.info(f"New user registered: {email}")
        
        # Send welcome email series
        await self.trigger_welcome_email_sequence(email)
        
        # Add to CRM system
        await self.add_to_crm(user_id, email, data['name'])
        
        # Update metrics
        await self.update_dashboard_metrics('total_users', 1)
    
    async def handle_agent_authenticated(self, event_data: Dict) -> None:
        """Handle agent authentication event."""
        data = event_data['data']
        agent_type = data['agent_type']
        user_id = data['user_id']
        capabilities = data['capabilities_granted']
        
        logger.info(f"Agent {agent_type} authenticated for user {user_id}")
        
        # Track agent usage
        await self.track_agent_usage(user_id, agent_type, capabilities)
        
        # Check for unusual agent usage patterns
        if await self.is_unusual_agent_usage(user_id, agent_type):
            await self.send_security_alert(user_id, 'unusual_agent_usage')
    
    async def handle_suspicious_login(self, event_data: Dict) -> None:
        """Handle suspicious login event."""
        data = event_data['data']
        user_id = data['user_id']
        email = data['email']
        risk_score = data['risk_score']
        indicators = data['suspicious_indicators']
        
        logger.warning(f"Suspicious login for {email}, risk score: {risk_score}")
        
        # Send security alert
        await self.send_immediate_security_alert(
            user_id, 
            'suspicious_login',
            {
                'risk_score': risk_score,
                'indicators': indicators,
                'ip_address': data['ip_address'],
                'location': data['location']
            }
        )
        
        # If high risk, temporarily lock account
        if risk_score > 0.8:
            await self.temporarily_lock_account(user_id, reason='high_risk_login')
    
    async def handle_rate_limit_exceeded(self, event_data: Dict) -> None:
        """Handle rate limit exceeded event."""
        data = event_data['data']
        endpoint = data['endpoint']
        ip_address = data['ip_address']
        current_rate = data['current_rate']
        
        logger.warning(f"Rate limit exceeded for {endpoint} from {ip_address}")
        
        # Check for potential DDoS
        if current_rate > 1000:  # requests per minute
            await self.trigger_ddos_protection(ip_address, endpoint)
        
        # Update security metrics
        await self.update_security_metrics('rate_limit_violations', 1)
    
    async def handle_performance_degraded(self, event_data: Dict) -> None:
        """Handle performance degradation event."""
        data = event_data['data']
        service = data['service']
        metric = data['metric']
        current_value = data['current_value']
        threshold = data['threshold']
        
        logger.error(f"Performance degraded: {service} {metric} = {current_value} (threshold: {threshold})")
        
        # Send alert to operations team
        await self.send_ops_alert('performance_degraded', data)
        
        # Check if auto-scaling is needed
        if metric in ['cpu', 'memory'] and current_value > threshold * 1.5:
            await self.trigger_auto_scaling(service)
    
    # Helper methods (implement based on your infrastructure)
    async def update_user_analytics(self, user_id: str, event: str) -> None:
        pass
    
    async def is_unusual_location(self, user_id: str, location: str) -> bool:
        # Check against user's typical locations
        return False
    
    async def send_security_alert(self, user_id: str, alert_type: str) -> None:
        pass
    
    async def update_dashboard_metrics(self, metric: str, delta: int) -> None:
        pass
    
    async def trigger_welcome_email_sequence(self, email: str) -> None:
        pass
    
    async def add_to_crm(self, user_id: str, email: str, name: str) -> None:
        pass
    
    async def track_agent_usage(self, user_id: str, agent_type: str, capabilities: list) -> None:
        pass

webhook_handler = WebhookHandler()

@app.post("/webhooks/archon")
async def handle_webhook(
    request: Request,
    x_archon_signature: Optional[str] = Header(None, alias='X-Archon-Signature'),
    x_archon_event_type: Optional[str] = Header(None, alias='X-Archon-Event-Type'),
    x_archon_event_id: Optional[str] = Header(None, alias='X-Archon-Event-ID')
):
    """Handle Archon webhooks."""
    
    # Get raw payload
    payload = await request.body()
    payload_str = payload.decode('utf-8')
    
    # Verify signature
    if not webhook_handler.verify_signature(payload_str, x_archon_signature):
        logger.error("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Parse event data
    try:
        event_data = json.loads(payload_str)
    except json.JSONDecodeError:
        logger.error("Invalid JSON payload")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    # Process event
    success = await webhook_handler.process_event(event_data)
    
    if success:
        return {"status": "success"}
    else:
        # Return 500 to trigger retry
        raise HTTPException(status_code=500, detail="Event processing failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Webhook Handler (Node.js/Express)

```javascript
// File: webhook_handler.js
const express = require('express');
const crypto = require('crypto');
const app = express();

const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET || 'your-webhook-secret-key';

class WebhookHandler {
    constructor() {
        this.eventHandlers = {
            'user.login': this.handleUserLogin.bind(this),
            'user.logout': this.handleUserLogout.bind(this),
            'user.registered': this.handleUserRegistered.bind(this),
            'agent.authenticated': this.handleAgentAuthenticated.bind(this),
            'security.suspicious_login': this.handleSuspiciousLogin.bind(this)
        };
    }

    verifySignature(payload, signature) {
        if (!signature) return false;

        const expectedSignature = crypto
            .createHmac('sha256', WEBHOOK_SECRET)
            .update(payload, 'utf8')
            .digest('hex');

        const actualSignature = signature.startsWith('sha256=') 
            ? signature.slice(7) 
            : signature;

        return crypto.timingSafeEqual(
            Buffer.from(expectedSignature, 'hex'),
            Buffer.from(actualSignature, 'hex')
        );
    }

    async processEvent(eventData) {
        const eventType = eventData.type;
        
        if (!this.eventHandlers[eventType]) {
            console.warn(`Unknown event type: ${eventType}`);
            return true;
        }

        try {
            await this.eventHandlers[eventType](eventData);
            return true;
        } catch (error) {
            console.error(`Error processing ${eventType}:`, error);
            return false;
        }
    }

    async handleUserLogin(eventData) {
        const { data } = eventData;
        console.log(`User ${data.email} logged in from ${data.ip_address}`);
        
        // Your business logic here
        await this.updateUserAnalytics(data.user_id, 'login');
        await this.checkUnusualLocation(data.user_id, data.location);
    }

    async handleUserLogout(eventData) {
        const { data } = eventData;
        console.log(`User ${data.user_id} logged out`);
        
        await this.updateSessionAnalytics(data.user_id, data.session_duration);
    }

    async handleUserRegistered(eventData) {
        const { data } = eventData;
        console.log(`New user registered: ${data.email}`);
        
        await this.sendWelcomeEmail(data.email);
        await this.addToCRM(data.user_id, data.email, data.name);
    }

    async handleAgentAuthenticated(eventData) {
        const { data } = eventData;
        console.log(`Agent ${data.agent_type} authenticated for user ${data.user_id}`);
        
        await this.trackAgentUsage(data.user_id, data.agent_type, data.capabilities_granted);
    }

    async handleSuspiciousLogin(eventData) {
        const { data } = eventData;
        console.warn(`Suspicious login for ${data.email}, risk score: ${data.risk_score}`);
        
        await this.sendSecurityAlert(data.user_id, 'suspicious_login', data);
        
        if (data.risk_score > 0.8) {
            await this.temporarilyLockAccount(data.user_id);
        }
    }

    // Implement these methods based on your infrastructure
    async updateUserAnalytics(userId, event) {
        // Implementation
    }

    async checkUnusualLocation(userId, location) {
        // Implementation
    }

    async sendWelcomeEmail(email) {
        // Implementation
    }

    async addToCRM(userId, email, name) {
        // Implementation
    }

    async trackAgentUsage(userId, agentType, capabilities) {
        // Implementation
    }

    async sendSecurityAlert(userId, alertType, data) {
        // Implementation
    }

    async temporarilyLockAccount(userId) {
        // Implementation
    }
}

const webhookHandler = new WebhookHandler();

// Middleware to capture raw body for signature verification
app.use('/webhooks/archon', express.raw({ type: 'application/json' }));

app.post('/webhooks/archon', async (req, res) => {
    const payload = req.body.toString();
    const signature = req.headers['x-archon-signature'];
    const eventType = req.headers['x-archon-event-type'];
    const eventId = req.headers['x-archon-event-id'];

    // Verify signature
    if (!webhookHandler.verifySignature(payload, signature)) {
        console.error('Invalid webhook signature');
        return res.status(401).json({ error: 'Invalid signature' });
    }

    // Parse event data
    let eventData;
    try {
        eventData = JSON.parse(payload);
    } catch (error) {
        console.error('Invalid JSON payload:', error);
        return res.status(400).json({ error: 'Invalid JSON' });
    }

    // Process event
    const success = await webhookHandler.processEvent(eventData);

    if (success) {
        res.json({ status: 'success' });
    } else {
        // Return 500 to trigger retry
        res.status(500).json({ error: 'Event processing failed' });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log(`Webhook handler listening on port ${PORT}`);
});

module.exports = app;
```

## Monitoring & Debugging

### Webhook Delivery Dashboard

```yaml
metrics:
  - webhook_deliveries_total
  - webhook_delivery_success_rate
  - webhook_delivery_latency_p95
  - webhook_retry_attempts_total
  - webhook_permanent_failures_total

alerts:
  - condition: webhook_delivery_success_rate < 95%
    severity: warning
  - condition: webhook_delivery_success_rate < 90%  
    severity: critical
  - condition: webhook_delivery_latency_p95 > 5s
    severity: warning
```

### Debug Tools

```python
# Webhook testing tool
import requests
import json

def test_webhook(webhook_url: str, event_type: str, test_data: dict):
    """Test webhook delivery."""
    
    payload = {
        "id": "test_event_123",
        "type": event_type,
        "version": "2.0.0",
        "timestamp": "2025-08-31T10:00:00Z",
        "data": test_data
    }
    
    headers = {
        'Content-Type': 'application/json',
        'X-Archon-Event-Type': event_type,
        'X-Archon-Event-ID': 'test_event_123'
    }
    
    response = requests.post(
        webhook_url,
        data=json.dumps(payload),
        headers=headers,
        timeout=30
    )
    
    return {
        'status_code': response.status_code,
        'response_time': response.elapsed.total_seconds(),
        'headers': dict(response.headers),
        'body': response.text
    }

# Usage
result = test_webhook(
    'https://your-app.com/webhooks/archon',
    'user.login',
    {
        'user_id': 'test_user_123',
        'email': 'test@example.com',
        'ip_address': '192.168.1.1'
    }
)
print(json.dumps(result, indent=2))
```

This comprehensive webhook architecture provides real-time event notifications with enterprise-grade reliability, security, and monitoring capabilities for the Archon Phase 6 authentication system.