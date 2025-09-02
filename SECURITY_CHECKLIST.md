# Security Checklist for Archon Contributors

## 🚨 NEVER COMMIT THESE FILES/PATTERNS

### ❌ Absolute No-Commit List
```bash
.env                    # Real environment variables
.env.local              # Local environment overrides  
.env.production         # Production environment variables
*.env                   # Any environment file
.credentials            # Credential files
secrets.yml             # Secret configuration files
*api*key*              # Files containing API keys
*secret*               # Files containing secrets  
*token*                # Files containing tokens
*password*             # Files containing passwords
```

### ❌ Sensitive Patterns to Avoid
```regex
sk-[a-zA-Z0-9\-_]{20,}                    # API keys (OpenAI, DeepSeek, etc.)
eyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+   # JWT tokens
https://[a-z0-9\-_]+\.supabase\.co        # Database URLs
postgresql://[^@]+@[^/]+/[^?\s]+          # Database connection strings
/mnt/c/[Jj]arvis                          # Personal paths
/home/username                            # User directories
/Users/username                           # macOS user directories
password\s*[:=]\s*["\'][^"\']{8,}         # Password assignments
```

## ✅ BEFORE EVERY COMMIT

### 1. Run Security Validation ✅
```bash
# MANDATORY: Run before every commit
python3 scripts/security-validation.py

# Should output: "✅ SECURITY VALIDATION PASSED"
```

### 2. Check Environment Files ✅
```bash
# Verify .env is properly ignored
git status | grep -v ".env"

# Ensure .env.example has only placeholders
grep -v "your-.*-here" .env.example && echo "❌ Found real values!" || echo "✅ Only placeholders"
```

### 3. Verify .gitignore Protection ✅
```bash
# Test that sensitive files are ignored
git check-ignore .env || echo "❌ .env not ignored!"
git check-ignore python/venv_archon/ || echo "❌ venv not ignored!"
```

## ✅ BEFORE PULL REQUEST

### 1. Security Scan ✅
```bash
# Run comprehensive security scan
python3 scripts/security-validation.py --fail-on-violations

# Must exit with code 0 (success)
echo $? # Should print: 0
```

### 2. Manual Review ✅
```bash
# Check for any remaining sensitive patterns
grep -r "sk-" --exclude-dir=.git --exclude-dir=node_modules . || echo "✅ No API keys found"
grep -r "eyJhbGc" --exclude-dir=.git --exclude-dir=node_modules . || echo "✅ No JWT tokens found"
grep -r "\.supabase\.co" --exclude-dir=.git --exclude-dir=node_modules . || echo "✅ No database URLs found"
```

### 3. Documentation Review ✅
- [ ] All example API keys use format: `your-api-key-here`
- [ ] No real database URLs in documentation
- [ ] No personal paths or system information
- [ ] All screenshots redacted of sensitive information

## ✅ INSTALLATION & SETUP

### 1. Install Pre-commit Hook ✅
```bash
# Copy pre-commit security hook
cp scripts/pre-commit-security.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Test the hook
echo "test" > test-commit.txt
git add test-commit.txt
git commit -m "test" # Should run security validation
```

### 2. Verify .env Configuration ✅
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with YOUR credentials (never commit this file)
nano .env

# Verify .env is ignored by git
git status | grep ".env" && echo "❌ .env is tracked!" || echo "✅ .env is ignored"
```

## 🚫 COMMON MISTAKES TO AVOID

### ❌ Never Do This
```bash
# DON'T commit real .env file
git add .env
git commit -m "add environment"

# DON'T use real credentials in examples  
OPENAI_API_KEY=sk-proj-abcd1234... # In documentation

# DON'T hardcode secrets in code
const apiKey = "sk-proj-real-key-here"; // In source files

# DON'T commit personal paths
/mnt/c/Jarvis/AI Workspace/... # In configuration files
```

### ✅ Always Do This Instead
```bash
# Use environment variables
const apiKey = process.env.OPENAI_API_KEY;

# Use placeholders in examples
OPENAI_API_KEY=your-openai-api-key-here

# Use relative paths
./scripts/security-validation.py

# Test before committing
python3 scripts/security-validation.py
```

## 🚨 EMERGENCY PROCEDURES

### If You Accidentally Commit Credentials

#### 1. Immediate Response
```bash
# DON'T PANIC - but act quickly

# 1. Remove from current commit (if not pushed)
git reset --soft HEAD~1
git checkout -- file-with-credentials

# 2. If already pushed, force update (DANGEROUS - coordinate with team)
git reset --hard HEAD~1
git push --force-with-lease origin branch-name
```

#### 2. Credential Rotation (MANDATORY)
- **Supabase**: Generate new project keys immediately
- **OpenAI**: Revoke and create new API key  
- **DeepSeek**: Regenerate API key
- **Any other exposed credentials**: Rotate immediately

#### 3. Incident Documentation
```bash
# Create incident report
echo "$(date): Accidental credential commit - rotated [SERVICE] keys" >> SECURITY_INCIDENTS.md
```

## 🛡️ SECURITY CONTACTS

### Internal Team
- **Security Lead**: Create GitHub issue with `security` label
- **Repository Maintainer**: @mentions in PR review  

### External Security Issues  
- **GitHub Security Advisory**: Use GitHub's private security reporting
- **Email**: security@archon-project.com (if available)

## 📚 TRAINING RESOURCES

### Required Reading
1. [GitHub Security Best Practices](https://docs.github.com/en/security)
2. [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
3. [Environment Variables Security](https://www.twilio.com/blog/working-with-environment-variables-in-node-js)

### Security Tools
- `scripts/security-validation.py` - Our custom security scanner
- `pre-commit-security.sh` - Automated pre-commit validation  
- `.gitignore` - Comprehensive ignore patterns

---

## Quick Reference Commands

```bash
# Daily security check
python3 scripts/security-validation.py

# Before commit
git add . && python3 scripts/security-validation.py && git commit -m "message"

# Emergency credential check
grep -r "sk-" --exclude-dir=.git .
grep -r "eyJhbGc" --exclude-dir=.git .

# Verify protection
git check-ignore .env python/venv_archon/ node_modules/
```

**Remember**: When in doubt, ask! It's better to check with the team than accidentally expose credentials. 🛡️