# Security Auditor PRP

## Context
- **Project**: {project_name}
- **Scope**: {audit_scope}
- **Files**: {file_paths}
- **Technologies**: {tech_stack}
- **Compliance**: {compliance_requirements}

## Task Requirements
{requirements}

## Security Standards & Guidelines

### OWASP Top 10 Compliance
- **A01: Broken Access Control** - Verify proper authorization
- **A02: Cryptographic Failures** - Check encryption implementation
- **A03: Injection** - Validate input sanitization
- **A04: Insecure Design** - Review security architecture
- **A05: Security Misconfiguration** - Audit configurations
- **A06: Vulnerable Components** - Check dependency security
- **A07: Authentication Failures** - Test auth mechanisms
- **A08: Software/Data Integrity** - Verify integrity checks
- **A09: Logging/Monitoring** - Review security logging
- **A10: Server-Side Request Forgery** - Check SSRF prevention

### Code Security Patterns
- Input validation at all entry points
- Output encoding for XSS prevention
- Parameterized queries for SQL injection prevention
- Proper session management
- Secure cryptographic implementations
- Access control enforcement
- Error handling without information leakage

## Security Audit Checklist

### Authentication & Authorization
```python
# Check for proper password hashing
import bcrypt
import hashlib

def audit_password_security(code_files):
    issues = []
    
    # Bad patterns to detect
    bad_patterns = [
        r'hashlib\.md5\(',          # MD5 usage
        r'hashlib\.sha1\(',         # SHA1 usage
        r'password.*==.*password',   # Plain text comparison
        r'\.encode\(.*password',     # Simple encoding
    ]
    
    for file_path in code_files:
        with open(file_path, 'r') as f:
            content = f.read()
            
        for pattern in bad_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'file': file_path,
                    'issue': f'Potentially insecure pattern: {pattern}',
                    'severity': 'HIGH'
                })
    
    return issues

# Example secure password verification
def secure_password_check(plain_password: str, hashed_password: str) -> bool:
    """Secure password verification using bcrypt."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False
```

### SQL Injection Prevention
```python
# SECURE: Using parameterized queries
async def get_user_secure(db: AsyncSession, user_id: int, email: str):
    query = select(User).where(
        and_(
            User.id == user_id,
            User.email == email
        )
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()

# VULNERABLE: String concatenation (DO NOT USE)
def get_user_vulnerable(db, user_id, email):
    # This is vulnerable to SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id} AND email = '{email}'"
    return db.execute(query)

# Audit function to detect SQL injection vulnerabilities
def audit_sql_injection(code_files):
    vulnerable_patterns = [
        r'f".*SELECT.*{.*}"',              # f-string in SQL
        r'"SELECT.*"\s*\+',                # String concatenation
        r'\.execute\s*\(\s*["\'].*%s.*["\']', # Old style formatting
        r'\.format\s*\(',                  # .format() in queries
    ]
    
    issues = []
    for file_path in code_files:
        # Check each file for vulnerable patterns
        pass
    return issues
```

### XSS Prevention
```typescript
// Frontend XSS prevention examples
import DOMPurify from 'dompurify';

// SECURE: Proper HTML sanitization
const sanitizeHTML = (dirty: string): string => {
    return DOMPurify.sanitize(dirty, {
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li'],
        ALLOWED_ATTR: []
    });
};

// SECURE: Using React's built-in XSS protection
const SafeComponent: React.FC<{content: string}> = ({content}) => {
    return <div>{content}</div>; // React automatically escapes
};

// VULNERABLE: Using dangerouslySetInnerHTML without sanitization
const VulnerableComponent: React.FC<{html: string}> = ({html}) => {
    return <div dangerouslySetInnerHTML={{__html: html}} />; // DANGEROUS!
};

// Audit function for XSS vulnerabilities
function auditXSS(codeFiles: string[]): SecurityIssue[] {
    const vulnerablePatterns = [
        /dangerouslySetInnerHTML.*\{.*\}/g,
        /innerHTML\s*=\s*[^;]+;/g,
        /document\.write\(/g,
        /eval\(/g
    ];
    
    const issues: SecurityIssue[] = [];
    // Implementation details...
    return issues;
}
```

### API Security
```python
# SECURE: API rate limiting and validation
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

security = HTTPBearer()

@app.post("/api/users")
@limiter.limit("5/minute")  # Rate limiting
async def create_user(
    request: Request,
    user_data: UserCreate,  # Pydantic validation
    credentials: HTTPAuthorizationCredentials = Depends(security)  # Auth required
):
    # Validate token
    user = await validate_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Additional validation
    if not user.has_permission("create_user"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Process request...
```

## Audit Report Template

```markdown
# Security Audit Report
**Project**: {project_name}
**Date**: {audit_date}
**Auditor**: Security Auditor Agent
**Scope**: {audit_scope}

## Executive Summary
- **Total Issues Found**: {total_issues}
- **Critical**: {critical_count}
- **High**: {high_count}
- **Medium**: {medium_count}
- **Low**: {low_count}

## Critical Findings

### 1. SQL Injection Vulnerability
**File**: `{file_path}`
**Line**: {line_number}
**Description**: Direct string concatenation in SQL query
**Risk**: Complete database compromise
**Recommendation**: Use parameterized queries

### 2. Weak Password Hashing
**File**: `{file_path}`
**Line**: {line_number}
**Description**: Using MD5 for password hashing
**Risk**: Password cracking via rainbow tables
**Recommendation**: Implement bcrypt with salt

## Recommendations

### Immediate Actions (Critical/High)
1. Fix SQL injection vulnerabilities
2. Implement proper password hashing
3. Add input validation for all endpoints
4. Enable HTTPS for all communications

### Short-term Improvements (Medium)
1. Add rate limiting to APIs
2. Implement proper session management
3. Add security headers
4. Update vulnerable dependencies

### Long-term Enhancements (Low)
1. Implement security monitoring
2. Add automated security testing
3. Regular dependency audits
4. Security awareness training

## Compliance Status
- **OWASP Top 10**: {compliance_percentage}% compliant
- **GDPR**: {gdpr_status}
- **SOC 2**: {soc2_status}
```

## Output Format

### Required Deliverables
1. **Security Audit Report** (Markdown format)
   - Executive summary with risk levels
   - Detailed findings with evidence
   - Remediation recommendations
   - Compliance assessment

2. **Vulnerability Scan Results** (JSON format)
   - Automated scan results
   - OWASP categorization
   - CVSS scores where applicable
   - False positive filtering

3. **Security Checklist** (Markdown format)
   - Implementation verification
   - Best practices compliance
   - Configuration security
   - Code review checklist

### Tools Integration
```bash
# Dependency vulnerability scanning
pip install safety
safety check --json > vulnerability_report.json

# Static code analysis
bandit -r . -f json -o security_analysis.json

# Container security (if applicable)
docker run --rm -v $(pwd):/app clair:latest scan /app
```

## Quality Checklist

- [ ] All input validation points audited
- [ ] Authentication mechanisms tested
- [ ] Authorization controls verified
- [ ] Data encryption reviewed
- [ ] Error handling examined
- [ ] Logging security assessed
- [ ] Dependencies scanned for vulnerabilities
- [ ] Configuration security verified
- [ ] API security tested
- [ ] Frontend XSS prevention confirmed

## Compliance Frameworks

### OWASP Testing Guide
- Authentication Testing
- Session Management Testing
- Input Validation Testing
- Error Handling Testing
- Cryptography Testing
- Business Logic Testing
- Client Side Testing

### Security Headers Checklist
```
Content-Security-Policy: default-src 'self'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Referrer-Policy: strict-origin-when-cross-origin
```