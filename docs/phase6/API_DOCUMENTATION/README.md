# Archon Authentication API Documentation

## Overview

Welcome to the comprehensive documentation for Archon Phase 6 Authentication API. This documentation provides everything you need to understand, integrate, and maintain the authentication system.

## 🚀 Quick Start

1. **New to Archon Auth?** Start with the [API Reference](./01_API_REFERENCE.md)
2. **Ready to integrate?** Check out the [Integration Guide](./03_INTEGRATION_GUIDE.md)
3. **Need help?** See the [Troubleshooting Guide](./05_TROUBLESHOOTING.md)

## 📚 Documentation Structure

### [1. API Reference](./01_API_REFERENCE.md)
Complete API endpoint documentation with examples, request/response formats, and authentication flows.

**Key Features:**
- User registration and authentication
- OAuth2 flows (Google, GitHub, Microsoft)
- JWT token management
- Session management
- Password management
- Email verification
- Rate limiting

### [2. OpenAPI Specification](./02_OPENAPI_SPEC.yaml)
Machine-readable API specification in OpenAPI 3.0 format for code generation and testing.

**Use Cases:**
- Generate client SDKs
- Import into Postman/Insomnia
- API testing automation
- Documentation rendering

### [3. Integration Guide](./03_INTEGRATION_GUIDE.md)
Comprehensive guide for integrating Archon Authentication into your applications.

**Covers:**
- Client library implementations (Python, JavaScript)
- Framework integrations (FastAPI, Express, React)
- Mobile app integration
- Authentication flow examples
- Best practices

### [4. Security Best Practices](./04_SECURITY_BEST_PRACTICES.md)
Essential security guidelines for secure authentication implementation.

**Topics:**
- Token security and storage
- Password security requirements
- Session management security
- OAuth2 security practices
- Network security configuration
- Monitoring and incident response

### [5. Troubleshooting Guide](./05_TROUBLESHOOTING.md)
Detailed troubleshooting procedures for common issues and problems.

**Includes:**
- Diagnostic tools and scripts
- Common error scenarios
- Performance optimization
- Debug procedures
- Production monitoring

### [6. Migration Guide](./06_MIGRATION_GUIDE.md)
Step-by-step instructions for migrating from other authentication systems.

**Supported Migrations:**
- Auth0 to Archon
- Firebase Auth to Archon
- AWS Cognito to Archon
- Custom JWT systems
- Session-based authentication

## 🏗️ Architecture Overview

Archon Authentication is built as a microservices-based system with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (Port 3737)   │    │   (Port 8181)   │    │   (Supabase)    │
│                 │    │                 │    │                 │
│ React + TS      │◄──►│ FastAPI +       │◄──►│ PostgreSQL +    │
│ + TailwindCSS   │    │ Socket.IO       │    │ pgvector        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │   (Caching &    │
                       │   Sessions)     │
                       └─────────────────┘
```

## 🔑 Key Features

### Authentication Methods
- **Email/Password** - Traditional username/password authentication
- **OAuth2** - Google, GitHub, Microsoft integration
- **JWT Tokens** - RS256 signed tokens with rotation
- **Session Management** - Secure session handling with Redis

### Security Features
- **Rate Limiting** - Configurable rate limits per endpoint
- **Account Lockout** - Brute force protection
- **Token Blacklisting** - Immediate token revocation
- **Password Security** - Argon2 hashing with complexity requirements
- **Audit Logging** - Comprehensive security event logging

### Advanced Features
- **Token Rotation** - Automatic refresh token rotation
- **Session Analytics** - Device and location tracking
- **Email Verification** - Configurable email verification flows
- **Password History** - Prevent password reuse
- **Multi-Factor Authentication** - Ready for MFA implementation

## 🚦 Getting Started

### Prerequisites
- Python 3.12+
- PostgreSQL 12+
- Redis 6+
- Node.js 18+ (for frontend)

### Environment Setup

1. **Clone and Install Dependencies:**
   ```bash
   git clone https://github.com/your-org/archon.git
   cd archon
   
   # Backend dependencies
   cd python
   uv sync
   
   # Frontend dependencies
   cd ../archon-ui-main
   npm install
   ```

2. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Services:**
   ```bash
   # Using Docker Compose (Recommended)
   docker-compose up --build -d
   
   # Or start individually
   # Backend
   cd python
   uv run python -m src.server.main
   
   # Frontend
   cd archon-ui-main
   npm run dev
   ```

4. **Verify Installation:**
   ```bash
   # Check health endpoint
   curl http://localhost:8181/auth/health
   
   # Should return:
   # {"status":"healthy","version":"1.0.0", ...}
   ```

### First API Call

```bash
# Register a new user
curl -X POST http://localhost:8181/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "name": "John Doe"
  }'

# Login
curl -X POST http://localhost:8181/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'
```

## 📊 API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/health` | GET | Service health check |
| `/auth/register` | POST | User registration |
| `/auth/login` | POST | User authentication |
| `/auth/logout` | POST | User logout |
| `/auth/refresh` | POST | Token refresh |
| `/auth/oauth/{provider}/authorize` | GET | OAuth authorization |
| `/auth/oauth/{provider}/callback` | GET | OAuth callback |
| `/auth/password` | PUT | Update password |
| `/auth/password/reset` | POST | Request password reset |
| `/auth/password/reset/confirm` | POST | Confirm password reset |
| `/auth/email/verify` | POST | Verify email |
| `/auth/sessions` | GET | List user sessions |
| `/auth/sessions/{id}` | DELETE | Terminate session |
| `/auth/me` | GET | Get user profile |

## 🔧 Configuration

### Core Settings
```env
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key

# JWT Configuration
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# OAuth Providers
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Security
RATE_LIMIT_ENABLED=true
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION=1800

# Email Service
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## 🧪 Testing

### Running Tests
```bash
# Backend tests
cd python
uv run pytest tests/ -v

# Frontend tests
cd archon-ui-main
npm run test

# Integration tests
uv run pytest tests/test_service_integration.py -v
```

### Test Coverage
```bash
# Backend coverage
uv run pytest --cov=src tests/

# Frontend coverage
npm run test:coverage
```

## 📈 Monitoring

### Health Monitoring
```bash
# Service health
curl http://localhost:8181/auth/health

# Metrics endpoint (if enabled)
curl http://localhost:8181/metrics

# Database health
curl http://localhost:8181/auth/health | jq '.dependencies.database'
```

### Logging
```bash
# View logs
docker-compose logs archon-backend
docker-compose logs archon-frontend

# Follow logs
docker-compose logs -f archon-backend
```

## 🆘 Support

### Getting Help

1. **Documentation Issues?** Check the [Troubleshooting Guide](./05_TROUBLESHOOTING.md)
2. **Integration Problems?** See the [Integration Guide](./03_INTEGRATION_GUIDE.md)
3. **Security Questions?** Review [Security Best Practices](./04_SECURITY_BEST_PRACTICES.md)
4. **Migration Help?** Use the [Migration Guide](./06_MIGRATION_GUIDE.md)

### Common Issues

| Issue | Solution |
|-------|----------|
| Service won't start | Check environment variables and database connection |
| Authentication fails | Verify user exists and password meets requirements |
| OAuth not working | Check OAuth provider configuration and redirect URIs |
| Slow performance | Review database indexes and Redis configuration |
| Rate limiting issues | Adjust rate limit settings or check IP whitelisting |

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug output
uv run python -m src.server.main --debug

# Or use the diagnostic script
python docs/phase6/API_DOCUMENTATION/debug_auth.py
```

## 🔗 Related Resources

- [Archon Main Documentation](../../README.md)
- [Phase 6 Implementation Notes](../README.md)
- [API Change Log](./CHANGELOG.md)
- [Security Audit Reports](./security/)
- [Performance Benchmarks](./benchmarks/)

## 📝 Contributing

### Documentation Updates
1. Update the relevant documentation file
2. Test all code examples
3. Update the API specification if needed
4. Submit a pull request with clear description

### Code Changes
1. Update documentation for any API changes
2. Add/update tests as needed
3. Update OpenAPI specification
4. Follow security guidelines

## 📅 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-01 | Initial release |
| 1.1.0 | 2024-02-01 | OAuth2 improvements |
| 1.2.0 | 2024-03-01 | Enhanced security features |

---

**🛡️ Security Notice:** Always use HTTPS in production, keep dependencies updated, and follow security best practices outlined in this documentation.

**⚡ Performance Tip:** Enable Redis caching and use connection pooling for optimal performance.

**🔄 Updates:** This documentation is continuously updated. Check the repository for the latest version.

---

*Last Updated: 2024-12-31*
*Documentation Version: 1.0.0*