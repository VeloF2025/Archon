# Migration Guide to Archon Authentication

## Overview

This guide provides comprehensive instructions for migrating from other authentication systems to Archon Authentication. It covers migration strategies, data transformation, and integration patterns for popular authentication providers.

## Table of Contents

1. [Migration Strategy](#migration-strategy)
2. [Pre-Migration Assessment](#pre-migration-assessment)
3. [Migrating from Auth0](#migrating-from-auth0)
4. [Migrating from Firebase Auth](#migrating-from-firebase-auth)
5. [Migrating from AWS Cognito](#migrating-from-aws-cognito)
6. [Migrating from Okta](#migrating-from-okta)
7. [Migrating from Custom JWT Systems](#migrating-from-custom-jwt-systems)
8. [Migrating from Session-Based Auth](#migrating-from-session-based-auth)
9. [User Data Migration](#user-data-migration)
10. [Testing and Validation](#testing-and-validation)
11. [Rollback Procedures](#rollback-procedures)

---

## Migration Strategy

### Migration Approaches

#### 1. Big Bang Migration
Complete cutover in a single deployment.

**Pros:**
- Simple to coordinate
- No dual-system complexity
- Immediate benefits

**Cons:**
- High risk
- Potential downtime
- Difficult rollback

**Best For:**
- Small applications
- Dev/staging environments
- Systems with simple auth requirements

#### 2. Gradual Migration (Recommended)
Phased approach with coexistence period.

**Pros:**
- Lower risk
- Can test with subset of users
- Easy rollback
- Maintains system availability

**Cons:**
- More complex
- Temporary dual-system overhead
- Longer migration period

**Best For:**
- Production systems
- Large user bases
- Critical applications

#### 3. Parallel Run
Run both systems simultaneously.

**Pros:**
- Zero downtime
- Comprehensive testing
- Gradual user migration

**Cons:**
- Resource intensive
- Complex synchronization
- Extended transition period

**Best For:**
- High-availability requirements
- Complex migration scenarios
- Risk-averse organizations

### Migration Timeline Template

```
Phase 1 (Weeks 1-2): Assessment and Planning
- Inventory current auth system
- Assess user data
- Plan migration strategy
- Set up Archon environment

Phase 2 (Weeks 3-4): Data Migration Preparation
- Export user data
- Transform data format
- Set up migration scripts
- Prepare test environment

Phase 3 (Weeks 5-6): Pilot Migration
- Migrate dev/staging environments
- Test functionality
- Migrate subset of users
- Gather feedback

Phase 4 (Weeks 7-8): Production Migration
- Execute migration plan
- Monitor system performance
- Handle user support issues
- Complete migration

Phase 5 (Weeks 9-10): Cleanup
- Remove old auth system
- Clean up migration tools
- Document lessons learned
- Post-migration optimization
```

---

## Pre-Migration Assessment

### Migration Checklist

```python
# migration_assessment.py
class MigrationAssessment:
    def __init__(self):
        self.findings = []
        self.compatibility_issues = []
        self.migration_complexity = "low"  # low, medium, high
    
    def assess_current_system(self):
        """Assess current authentication system."""
        
        # User data assessment
        user_count = self.count_users()
        user_attributes = self.analyze_user_attributes()
        
        # Authentication methods
        auth_methods = self.identify_auth_methods()
        
        # Integration points
        integrations = self.find_integrations()
        
        # Custom implementations
        customizations = self.identify_customizations()
        
        assessment = {
            "user_count": user_count,
            "user_attributes": user_attributes,
            "auth_methods": auth_methods,
            "integrations": integrations,
            "customizations": customizations,
            "estimated_effort": self.estimate_effort(),
            "risks": self.identify_risks()
        }
        
        return assessment
    
    def estimate_effort(self):
        """Estimate migration effort in person-days."""
        base_effort = 5  # Base setup
        
        # Add effort based on complexity factors
        if self.user_count > 100000:
            base_effort += 10
        elif self.user_count > 10000:
            base_effort += 5
        
        if len(self.auth_methods) > 2:
            base_effort += 3
        
        if len(self.integrations) > 5:
            base_effort += 5
        
        if len(self.customizations) > 0:
            base_effort += 10
        
        return base_effort
    
    def generate_migration_plan(self):
        """Generate detailed migration plan."""
        return {
            "recommended_approach": self.recommend_approach(),
            "data_migration_strategy": self.plan_data_migration(),
            "integration_updates": self.plan_integration_updates(),
            "testing_strategy": self.plan_testing(),
            "rollback_plan": self.plan_rollback()
        }
```

### Current System Inventory

**Authentication Methods:**
- [ ] Username/Password
- [ ] OAuth2 (Google, GitHub, Microsoft, etc.)
- [ ] SAML
- [ ] Multi-Factor Authentication (MFA)
- [ ] Social logins
- [ ] API keys
- [ ] Custom authentication

**User Data Fields:**
- [ ] Basic profile (name, email)
- [ ] Custom attributes
- [ ] Role/permission data
- [ ] Preferences
- [ ] Activity logs
- [ ] External IDs

**Integration Points:**
- [ ] Frontend applications
- [ ] Mobile applications
- [ ] API endpoints
- [ ] Microservices
- [ ] Third-party services
- [ ] Webhooks

---

## Migrating from Auth0

### Auth0 to Archon Migration

#### 1. Export User Data from Auth0

```python
import requests
from auth0.authentication import GetToken
from auth0.management import Auth0

# Auth0 Management API setup
def get_auth0_management_client():
    domain = "your-domain.auth0.com"
    client_id = "YOUR_MANAGEMENT_CLIENT_ID"
    client_secret = "YOUR_MANAGEMENT_CLIENT_SECRET"
    
    get_token = GetToken(domain, client_id, client_secret)
    token = get_token.client_credentials(f"https://{domain}/api/v2/")
    
    return Auth0(domain, token['access_token'])

def export_auth0_users():
    """Export all users from Auth0."""
    mgmt = get_auth0_management_client()
    
    all_users = []
    page = 0
    per_page = 100
    
    while True:
        users = mgmt.users.list(
            page=page,
            per_page=per_page,
            include_totals=True,
            search_engine='v3'
        )
        
        all_users.extend(users['users'])
        
        if len(users['users']) < per_page:
            break
        
        page += 1
    
    return all_users

def transform_auth0_user_to_archon(auth0_user):
    """Transform Auth0 user format to Archon format."""
    
    # Extract basic info
    email = auth0_user.get('email', '')
    name = auth0_user.get('name', '')
    email_verified = auth0_user.get('email_verified', False)
    
    # Handle Auth0 identities (social logins)
    oauth_provider = None
    oauth_provider_id = None
    
    if 'identities' in auth0_user:
        primary_identity = next(
            (id for id in auth0_user['identities'] if id.get('isPrimary')),
            auth0_user['identities'][0] if auth0_user['identities'] else None
        )
        
        if primary_identity and primary_identity['provider'] != 'auth0':
            provider_map = {
                'google-oauth2': 'google',
                'github': 'github',
                'windowslive': 'microsoft'
            }
            oauth_provider = provider_map.get(primary_identity['provider'])
            oauth_provider_id = primary_identity['user_id']
    
    # Extract custom metadata
    app_metadata = auth0_user.get('app_metadata', {})
    user_metadata = auth0_user.get('user_metadata', {})
    
    # Merge metadata
    metadata = {**user_metadata, **app_metadata}
    
    # Add Auth0 specific data for reference
    metadata['auth0_user_id'] = auth0_user['user_id']
    metadata['auth0_created_at'] = auth0_user.get('created_at')
    metadata['auth0_last_login'] = auth0_user.get('last_login')
    
    return {
        'email': email,
        'name': name,
        'email_verified': email_verified,
        'oauth_provider': oauth_provider,
        'oauth_provider_id': oauth_provider_id,
        'metadata': metadata,
        'original_auth0_data': auth0_user  # Keep for troubleshooting
    }
```

#### 2. Password Migration Strategy

Auth0 doesn't allow password export, so we need a migration strategy:

```python
class Auth0PasswordMigration:
    def __init__(self, archon_client, auth0_domain):
        self.archon = archon_client
        self.auth0_domain = auth0_domain
    
    async def setup_lazy_migration(self):
        """Set up lazy password migration on first login."""
        
        # This requires custom middleware in your login flow
        pass
    
    async def handle_login_attempt(self, email, password):
        """Handle login with lazy migration."""
        
        # Try Archon login first
        try:
            return await self.archon.login(email, password)
        except AuthenticationError:
            # If Archon login fails, try Auth0
            try:
                auth0_result = await self.verify_with_auth0(email, password)
                if auth0_result:
                    # Migrate user password to Archon
                    await self.migrate_user_password(email, password)
                    return await self.archon.login(email, password)
            except Exception:
                pass
        
        raise AuthenticationError("Invalid credentials")
    
    async def verify_with_auth0(self, email, password):
        """Verify credentials with Auth0."""
        import httpx
        
        token_url = f"https://{self.auth0_domain}/oauth/token"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, json={
                "grant_type": "password",
                "username": email,
                "password": password,
                "client_id": "YOUR_AUTH0_CLIENT_ID",
                "client_secret": "YOUR_AUTH0_CLIENT_SECRET",
                "scope": "openid profile email"
            })
            
            return response.status_code == 200
    
    async def migrate_user_password(self, email, password):
        """Migrate user password to Archon."""
        try:
            # Update user password in Archon
            user = await self.archon.get_user_by_email(email)
            if user:
                await self.archon.update_user_password(user['id'], password)
        except Exception as e:
            print(f"Failed to migrate password for {email}: {e}")
```

#### 3. Auth0 Rules Migration

Convert Auth0 Rules to Archon middleware:

```python
# Auth0 Rule: Add custom claims
# function (user, context, callback) {
#   const namespace = 'https://myapp.com/';
#   context.idToken[namespace + 'role'] = user.app_metadata.role;
#   callback(null, user, context);
# }

# Archon equivalent:
class CustomClaimsMiddleware:
    async def process_token_generation(self, user, claims):
        """Add custom claims during token generation."""
        
        # Add role from user metadata
        if 'role' in user.metadata:
            claims['https://myapp.com/role'] = user.metadata['role']
        
        # Add other custom claims
        if 'department' in user.metadata:
            claims['https://myapp.com/department'] = user.metadata['department']
        
        return claims
```

### Auth0 Migration Script

```python
import asyncio
import json
from datetime import datetime

class Auth0MigrationScript:
    def __init__(self, archon_client):
        self.archon = archon_client
        self.migration_log = []
    
    async def migrate_all_users(self):
        """Complete Auth0 to Archon user migration."""
        
        print("🚀 Starting Auth0 to Archon migration...")
        
        # 1. Export Auth0 users
        print("📤 Exporting users from Auth0...")
        auth0_users = export_auth0_users()
        print(f"   Found {len(auth0_users)} users")
        
        # 2. Transform and migrate users
        migrated_count = 0
        failed_count = 0
        
        for auth0_user in auth0_users:
            try:
                # Transform user data
                archon_user = transform_auth0_user_to_archon(auth0_user)
                
                # Create user in Archon
                await self.create_archon_user(archon_user)
                
                migrated_count += 1
                print(f"✅ Migrated: {archon_user['email']}")
                
            except Exception as e:
                failed_count += 1
                error_info = {
                    'auth0_user_id': auth0_user['user_id'],
                    'email': auth0_user.get('email'),
                    'error': str(e)
                }
                self.migration_log.append(error_info)
                print(f"❌ Failed: {auth0_user.get('email')} - {e}")
        
        print(f"\n📊 Migration Summary:")
        print(f"   Total users: {len(auth0_users)}")
        print(f"   Migrated: {migrated_count}")
        print(f"   Failed: {failed_count}")
        
        # Save migration log
        with open(f'auth0_migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(self.migration_log, f, indent=2)
    
    async def create_archon_user(self, user_data):
        """Create user in Archon system."""
        
        # Check if user already exists
        existing_user = await self.archon.get_user_by_email(user_data['email'])
        if existing_user:
            print(f"   User {user_data['email']} already exists, updating...")
            await self.archon.update_user(existing_user['id'], user_data)
        else:
            # Create new user
            await self.archon.create_user(user_data)

# Usage
async def main():
    archon_client = ArchonAuthClient("http://localhost:8181")
    migration = Auth0MigrationScript(archon_client)
    await migration.migrate_all_users()

asyncio.run(main())
```

---

## Migrating from Firebase Auth

### Firebase to Archon Migration

#### 1. Export Firebase Users

```python
import firebase_admin
from firebase_admin import credentials, auth
import json

def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    cred = credentials.Certificate("path/to/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

def export_firebase_users():
    """Export all users from Firebase Auth."""
    
    all_users = []
    page_token = None
    
    while True:
        # List users in batches of 1000
        list_users_result = auth.list_users(page_token=page_token, max_results=1000)
        
        for user in list_users_result.users:
            all_users.append({
                'uid': user.uid,
                'email': user.email,
                'email_verified': user.email_verified,
                'display_name': user.display_name,
                'photo_url': user.photo_url,
                'disabled': user.disabled,
                'provider_data': [
                    {
                        'provider_id': provider.provider_id,
                        'uid': provider.uid,
                        'email': provider.email,
                        'display_name': provider.display_name,
                        'photo_url': provider.photo_url
                    }
                    for provider in user.provider_data
                ],
                'custom_claims': user.custom_claims,
                'creation_timestamp': user.user_metadata.creation_timestamp,
                'last_sign_in_timestamp': user.user_metadata.last_sign_in_timestamp
            })
        
        page_token = list_users_result.next_page_token
        if not page_token:
            break
    
    return all_users

def transform_firebase_user_to_archon(firebase_user):
    """Transform Firebase user to Archon format."""
    
    # Basic info
    email = firebase_user.get('email', '')
    name = firebase_user.get('display_name', email.split('@')[0] if email else 'Unknown')
    email_verified = firebase_user.get('email_verified', False)
    
    # OAuth provider info
    oauth_provider = None
    oauth_provider_id = None
    
    provider_data = firebase_user.get('provider_data', [])
    for provider in provider_data:
        provider_id = provider.get('provider_id')
        if provider_id != 'password':
            provider_map = {
                'google.com': 'google',
                'github.com': 'github',
                'microsoft.com': 'microsoft'
            }
            oauth_provider = provider_map.get(provider_id)
            oauth_provider_id = provider.get('uid')
            break
    
    # Metadata
    metadata = {
        'firebase_uid': firebase_user['uid'],
        'photo_url': firebase_user.get('photo_url'),
        'creation_timestamp': firebase_user.get('creation_timestamp'),
        'last_sign_in_timestamp': firebase_user.get('last_sign_in_timestamp'),
        'custom_claims': firebase_user.get('custom_claims', {}),
        'disabled': firebase_user.get('disabled', False)
    }
    
    return {
        'email': email,
        'name': name,
        'email_verified': email_verified,
        'oauth_provider': oauth_provider,
        'oauth_provider_id': oauth_provider_id,
        'metadata': metadata
    }
```

#### 2. Firebase Custom Claims Migration

```python
def migrate_firebase_custom_claims():
    """Migrate Firebase custom claims to Archon."""
    
    # Firebase custom claims example:
    # {
    #   "role": "admin",
    #   "permissions": ["read", "write", "delete"],
    #   "organization": "acme-corp"
    # }
    
    # Archon equivalent in user metadata:
    archon_metadata = {
        "role": "admin",
        "permissions": ["read", "write", "delete"],
        "organization": "acme-corp",
        # Add to JWT claims during token generation
        "custom_claims": {
            "role": "admin",
            "permissions": ["read", "write", "delete"],
            "organization": "acme-corp"
        }
    }
    
    return archon_metadata
```

#### 3. Firebase Security Rules to Archon Authorization

```javascript
// Firebase Security Rules
// service cloud.firestore {
//   match /databases/{database}/documents {
//     match /users/{userId} {
//       allow read, write: if request.auth.uid == userId;
//     }
//     match /admin/{document=**} {
//       allow read, write: if request.auth.token.role == "admin";
//     }
//   }
// }

// Archon equivalent middleware
class FirebaseRulesEquivalent:
    async def check_user_document_access(self, user_id, requested_user_id, action):
        """Equivalent to Firebase rule: allow if request.auth.uid == userId"""
        return user_id == requested_user_id
    
    async def check_admin_access(self, user_claims, action):
        """Equivalent to Firebase rule: allow if token.role == "admin" """
        return user_claims.get('role') == 'admin'
```

---

## Migrating from AWS Cognito

### Cognito to Archon Migration

#### 1. Export Cognito Users

```python
import boto3
import json
from botocore.exceptions import ClientError

class CognitoMigration:
    def __init__(self, user_pool_id, region='us-east-1'):
        self.cognito = boto3.client('cognito-idp', region_name=region)
        self.user_pool_id = user_pool_id
    
    def export_cognito_users(self):
        """Export all users from Cognito User Pool."""
        
        all_users = []
        pagination_token = None
        
        while True:
            kwargs = {
                'UserPoolId': self.user_pool_id,
                'Limit': 60  # Max allowed by Cognito
            }
            
            if pagination_token:
                kwargs['PaginationToken'] = pagination_token
            
            try:
                response = self.cognito.list_users(**kwargs)
                
                for user in response['Users']:
                    # Get user attributes
                    attributes = {attr['Name']: attr['Value'] for attr in user['Attributes']}
                    
                    user_data = {
                        'username': user['Username'],
                        'user_status': user['UserStatus'],
                        'enabled': user['Enabled'],
                        'user_create_date': user['UserCreateDate'],
                        'user_last_modified_date': user['UserLastModifiedDate'],
                        'attributes': attributes
                    }
                    
                    all_users.append(user_data)
                
                pagination_token = response.get('PaginationToken')
                if not pagination_token:
                    break
                    
            except ClientError as e:
                print(f"Error listing users: {e}")
                break
        
        return all_users
    
    def transform_cognito_user_to_archon(self, cognito_user):
        """Transform Cognito user to Archon format."""
        
        attributes = cognito_user['attributes']
        
        # Basic info
        email = attributes.get('email', '')
        name = attributes.get('name', attributes.get('given_name', ''))
        email_verified = attributes.get('email_verified', 'false').lower() == 'true'
        
        # Handle federated identities
        oauth_provider = None
        oauth_provider_id = None
        
        identity_providers = attributes.get('identities')
        if identity_providers:
            import json
            try:
                identities = json.loads(identity_providers)
                if identities:
                    provider = identities[0]
                    provider_map = {
                        'Google': 'google',
                        'LoginWithAmazon': 'amazon',
                        'Facebook': 'facebook'
                    }
                    oauth_provider = provider_map.get(provider.get('providerName'))
                    oauth_provider_id = provider.get('userId')
            except:
                pass
        
        # Metadata
        metadata = {
            'cognito_username': cognito_user['username'],
            'cognito_user_status': cognito_user['user_status'],
            'cognito_enabled': cognito_user['enabled'],
            'cognito_created': cognito_user['user_create_date'].isoformat(),
            'phone_number': attributes.get('phone_number'),
            'phone_verified': attributes.get('phone_number_verified', 'false').lower() == 'true',
            'custom_attributes': {
                k: v for k, v in attributes.items() 
                if k.startswith('custom:')
            }
        }
        
        return {
            'email': email,
            'name': name,
            'email_verified': email_verified,
            'oauth_provider': oauth_provider,
            'oauth_provider_id': oauth_provider_id,
            'metadata': metadata
        }
```

#### 2. Cognito Lambda Triggers Migration

```python
# Cognito Pre-Authentication Lambda
# def lambda_handler(event, context):
#     if event['request']['userAttributes']['email'] == 'blocked@example.com':
#         raise Exception("This user is blocked")
#     return event

# Archon equivalent middleware
class CognitoTriggersEquivalent:
    async def pre_authentication_check(self, email, password):
        """Equivalent to Cognito Pre-Authentication trigger."""
        
        blocked_emails = ['blocked@example.com']
        if email in blocked_emails:
            raise AuthenticationError("This user is blocked")
        
        return True
    
    async def post_authentication_hook(self, user_id, login_context):
        """Equivalent to Cognito Post-Authentication trigger."""
        
        # Log successful login
        await self.log_user_login(user_id, login_context)
        
        # Update last login timestamp
        await self.update_last_login(user_id)
        
        return True
```

---

## Migrating from Custom JWT Systems

### Custom JWT to Archon Migration

#### 1. JWT Token Analysis

```python
import jwt
import json
from datetime import datetime

def analyze_existing_jwt_system():
    """Analyze current JWT implementation."""
    
    # Sample existing token
    existing_token = "YOUR_EXISTING_JWT_TOKEN"
    
    try:
        # Decode without verification to analyze structure
        payload = jwt.decode(existing_token, options={"verify_signature": False})
        
        analysis = {
            'algorithm': jwt.get_unverified_header(existing_token).get('alg'),
            'claims': list(payload.keys()),
            'custom_claims': [k for k in payload.keys() if k not in ['iss', 'sub', 'aud', 'exp', 'nbf', 'iat', 'jti']],
            'expiry_pattern': 'short' if payload.get('exp', 0) - payload.get('iat', 0) < 3600 else 'long',
            'has_refresh_pattern': 'refresh_token' in payload or 'token_type' in payload
        }
        
        print("🔍 Current JWT Analysis:")
        print(json.dumps(analysis, indent=2))
        
        return analysis
        
    except Exception as e:
        print(f"❌ Failed to analyze JWT: {e}")
        return None

def create_claim_mapping():
    """Create mapping between existing claims and Archon claims."""
    
    # Example mapping
    claim_mapping = {
        # Existing claim -> Archon claim
        'user_id': 'sub',
        'username': 'email',
        'roles': 'roles',
        'permissions': 'permissions',
        'tenant': 'tenant_id',
        'custom_field': 'custom_field'  # Keep custom fields
    }
    
    return claim_mapping
```

#### 2. Token Validation Migration

```python
class CustomJWTMigration:
    def __init__(self, old_secret, new_archon_client):
        self.old_secret = old_secret
        self.archon = new_archon_client
        self.claim_mapping = create_claim_mapping()
    
    async def validate_token_with_fallback(self, token):
        """Validate token with fallback to old system."""
        
        try:
            # Try Archon validation first
            return await self.archon.validate_token(token)
        
        except TokenError:
            # Fallback to old JWT validation
            try:
                payload = jwt.decode(
                    token,
                    self.old_secret,
                    algorithms=['HS256']  # Your old algorithm
                )
                
                # Convert old token format to Archon format
                return self.convert_old_payload_to_archon(payload)
                
            except jwt.InvalidTokenError:
                raise TokenError("Invalid token")
    
    def convert_old_payload_to_archon(self, old_payload):
        """Convert old JWT payload to Archon format."""
        
        archon_payload = {}
        
        # Map standard claims
        for old_claim, archon_claim in self.claim_mapping.items():
            if old_claim in old_payload:
                archon_payload[archon_claim] = old_payload[old_claim]
        
        # Ensure required claims
        if 'sub' not in archon_payload and 'user_id' in old_payload:
            archon_payload['sub'] = str(old_payload['user_id'])
        
        if 'email' not in archon_payload and 'username' in old_payload:
            archon_payload['email'] = old_payload['username']
        
        # Add conversion metadata
        archon_payload['migrated_from'] = 'custom_jwt'
        archon_payload['migration_timestamp'] = time.time()
        
        return archon_payload
    
    async def migrate_user_sessions(self):
        """Migrate active user sessions."""
        
        # This would depend on how you store sessions
        # Example for Redis-based sessions
        import redis
        
        redis_client = redis.Redis()
        
        # Find all active sessions
        session_keys = redis_client.keys("session:*")
        
        migrated_count = 0
        
        for session_key in session_keys:
            try:
                session_data = redis_client.get(session_key)
                if session_data:
                    # Convert session data to Archon format
                    old_session = json.loads(session_data)
                    
                    # Create new Archon session
                    new_session = await self.create_archon_session(old_session)
                    
                    # Store new session
                    redis_client.set(
                        f"archon_session:{new_session['id']}", 
                        json.dumps(new_session),
                        ex=new_session['expires_in']
                    )
                    
                    migrated_count += 1
                    
            except Exception as e:
                print(f"Failed to migrate session {session_key}: {e}")
        
        print(f"Migrated {migrated_count} sessions")
```

---

## User Data Migration

### Database Schema Migration

```python
class UserDataMigration:
    def __init__(self, source_db, target_db):
        self.source_db = source_db
        self.target_db = target_db
    
    async def migrate_user_table(self):
        """Migrate user table structure and data."""
        
        # 1. Create Archon user table structure
        await self.create_archon_user_table()
        
        # 2. Extract data from source
        source_users = await self.extract_source_users()
        
        # 3. Transform and load
        for source_user in source_users:
            try:
                archon_user = self.transform_user_data(source_user)
                await self.insert_archon_user(archon_user)
            except Exception as e:
                print(f"Failed to migrate user {source_user.get('id')}: {e}")
    
    async def create_archon_user_table(self):
        """Create Archon-compatible user table."""
        
        create_table_sql = """
        CREATE TABLE users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255),
            name VARCHAR(255) NOT NULL,
            email_verified BOOLEAN DEFAULT FALSE,
            is_active BOOLEAN DEFAULT TRUE,
            oauth_provider VARCHAR(50),
            oauth_provider_id VARCHAR(255),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_login_at TIMESTAMP WITH TIME ZONE
        );
        
        CREATE INDEX idx_users_email ON users(email);
        CREATE INDEX idx_users_oauth ON users(oauth_provider, oauth_provider_id);
        CREATE INDEX idx_users_active ON users(is_active);
        """
        
        await self.target_db.execute(create_table_sql)
    
    def transform_user_data(self, source_user):
        """Transform source user data to Archon format."""
        
        # This will vary based on your source schema
        return {
            'email': source_user['email'],
            'name': source_user.get('display_name') or source_user['email'].split('@')[0],
            'email_verified': source_user.get('email_verified', False),
            'is_active': source_user.get('active', True),
            'oauth_provider': source_user.get('provider'),
            'oauth_provider_id': source_user.get('provider_id'),
            'metadata': {
                'migrated_from': 'legacy_system',
                'original_id': source_user['id'],
                'migration_date': datetime.now().isoformat(),
                **source_user.get('custom_fields', {})
            },
            'created_at': source_user.get('created_at'),
            'last_login_at': source_user.get('last_login')
        }
```

### Password Hash Migration

```python
class PasswordMigration:
    def __init__(self):
        self.hash_mappings = {
            'bcrypt': self.migrate_bcrypt,
            'pbkdf2': self.migrate_pbkdf2,
            'scrypt': self.migrate_scrypt,
            'argon2': self.migrate_argon2,
            'md5': self.migrate_weak_hash,
            'sha1': self.migrate_weak_hash
        }
    
    async def migrate_password_hashes(self, users):
        """Migrate password hashes from various formats."""
        
        migrated_count = 0
        failed_count = 0
        
        for user in users:
            try:
                original_hash = user.get('password_hash')
                hash_algorithm = user.get('hash_algorithm', 'bcrypt')
                
                if hash_algorithm in self.hash_mappings:
                    migrator = self.hash_mappings[hash_algorithm]
                    new_hash = await migrator(original_hash, user)
                    
                    # Update user record
                    await self.update_user_password_hash(user['id'], new_hash)
                    migrated_count += 1
                else:
                    # Force password reset for unknown algorithms
                    await self.force_password_reset(user['id'])
                    failed_count += 1
                    
            except Exception as e:
                print(f"Failed to migrate password for user {user['id']}: {e}")
                failed_count += 1
        
        print(f"Password migration: {migrated_count} migrated, {failed_count} require reset")
    
    async def migrate_bcrypt(self, bcrypt_hash, user):
        """Migrate bcrypt hashes (can be kept as-is in many cases)."""
        # Bcrypt hashes are generally compatible
        # Just need to verify format
        if bcrypt_hash.startswith('$2b$') or bcrypt_hash.startswith('$2a$'):
            return bcrypt_hash
        else:
            # Invalid bcrypt format, force reset
            await self.force_password_reset(user['id'])
            return None
    
    async def migrate_weak_hash(self, weak_hash, user):
        """Handle weak hash algorithms (MD5, SHA1)."""
        # Cannot migrate weak hashes securely
        # Force password reset
        await self.force_password_reset(user['id'])
        return None
    
    async def force_password_reset(self, user_id):
        """Force user to reset password on next login."""
        await self.target_db.execute(
            "UPDATE users SET password_reset_required = TRUE WHERE id = ?",
            (user_id,)
        )
        
        # Send password reset email
        user = await self.get_user(user_id)
        if user:
            await self.send_password_reset_email(user['email'])
```

---

## Testing and Validation

### Migration Testing Framework

```python
class MigrationValidator:
    def __init__(self, source_system, target_system):
        self.source = source_system
        self.target = target_system
        self.validation_results = []
    
    async def validate_user_migration(self):
        """Validate user migration completeness and accuracy."""
        
        print("🧪 Starting user migration validation...")
        
        # 1. Count validation
        source_count = await self.source.count_users()
        target_count = await self.target.count_users()
        
        self.validation_results.append({
            'test': 'user_count',
            'source_count': source_count,
            'target_count': target_count,
            'passed': source_count == target_count
        })
        
        # 2. Data integrity validation
        source_users = await self.source.get_sample_users(100)
        
        for source_user in source_users:
            target_user = await self.target.get_user_by_email(source_user['email'])
            
            if target_user:
                integrity_check = self.validate_user_data_integrity(source_user, target_user)
                self.validation_results.append(integrity_check)
            else:
                self.validation_results.append({
                    'test': 'user_exists',
                    'email': source_user['email'],
                    'passed': False,
                    'error': 'User not found in target system'
                })
        
        # 3. Authentication validation
        await self.validate_authentication()
        
        # Generate validation report
        self.generate_validation_report()
    
    def validate_user_data_integrity(self, source_user, target_user):
        """Validate data integrity between source and target."""
        
        checks = []
        
        # Email match
        checks.append({
            'field': 'email',
            'passed': source_user['email'] == target_user['email']
        })
        
        # Name match (with flexibility for transformations)
        checks.append({
            'field': 'name',
            'passed': self.names_equivalent(source_user.get('name'), target_user.get('name'))
        })
        
        # Email verification status
        checks.append({
            'field': 'email_verified',
            'passed': source_user.get('email_verified') == target_user.get('email_verified')
        })
        
        return {
            'test': 'data_integrity',
            'email': source_user['email'],
            'checks': checks,
            'passed': all(check['passed'] for check in checks)
        }
    
    async def validate_authentication(self):
        """Validate authentication functionality."""
        
        test_credentials = [
            {'email': 'test1@example.com', 'password': 'TestPass123!'},
            {'email': 'test2@example.com', 'password': 'AnotherPass456!'}
        ]
        
        for creds in test_credentials:
            try:
                # Test login
                login_result = await self.target.login(creds['email'], creds['password'])
                
                # Test token validation
                token_valid = await self.target.validate_token(login_result['access_token'])
                
                # Test logout
                await self.target.logout(login_result['access_token'])
                
                self.validation_results.append({
                    'test': 'authentication_flow',
                    'email': creds['email'],
                    'passed': True
                })
                
            except Exception as e:
                self.validation_results.append({
                    'test': 'authentication_flow',
                    'email': creds['email'],
                    'passed': False,
                    'error': str(e)
                })
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result['passed'])
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'detailed_results': self.validation_results,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        with open(f'migration_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Validation Report:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success rate: {report['validation_summary']['success_rate']:.2%}")
        
        return report
```

---

## Rollback Procedures

### Automated Rollback System

```python
class MigrationRollback:
    def __init__(self):
        self.rollback_log = []
        self.backup_location = "/backup/migration"
    
    async def create_rollback_checkpoint(self):
        """Create rollback checkpoint before migration."""
        
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'database_backup': await self.backup_database(),
            'configuration_backup': await self.backup_configuration(),
            'user_data_snapshot': await self.create_user_snapshot()
        }
        
        # Save checkpoint metadata
        with open(f'{self.backup_location}/checkpoint_{int(time.time())}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        return checkpoint
    
    async def execute_rollback(self, checkpoint_id):
        """Execute rollback to specific checkpoint."""
        
        print(f"🔄 Starting rollback to checkpoint {checkpoint_id}...")
        
        try:
            # 1. Load checkpoint metadata
            checkpoint = await self.load_checkpoint(checkpoint_id)
            
            # 2. Stop Archon services
            await self.stop_archon_services()
            
            # 3. Restore database
            await self.restore_database(checkpoint['database_backup'])
            
            # 4. Restore configuration
            await self.restore_configuration(checkpoint['configuration_backup'])
            
            # 5. Restart original auth services
            await self.start_original_auth_services()
            
            # 6. Validate rollback
            rollback_valid = await self.validate_rollback()
            
            if rollback_valid:
                print("✅ Rollback completed successfully")
                return True
            else:
                print("❌ Rollback validation failed")
                return False
                
        except Exception as e:
            print(f"💥 Rollback failed: {e}")
            # Emergency procedures
            await self.emergency_rollback_procedures()
            return False
    
    async def validate_rollback(self):
        """Validate successful rollback."""
        
        validation_tests = [
            self.test_original_auth_endpoints(),
            self.test_user_data_integrity(),
            self.test_application_connectivity()
        ]
        
        results = await asyncio.gather(*validation_tests, return_exceptions=True)
        return all(result is True for result in results)
    
    async def emergency_rollback_procedures(self):
        """Emergency procedures if rollback fails."""
        
        # 1. Alert operations team
        await self.send_emergency_alert("Migration rollback failed - manual intervention required")
        
        # 2. Enable maintenance mode
        await self.enable_maintenance_mode()
        
        # 3. Document current state
        await self.document_emergency_state()
        
        # 4. Provide manual recovery instructions
        recovery_instructions = """
        EMERGENCY RECOVERY INSTRUCTIONS:
        
        1. Check system logs: tail -f /var/log/archon/*.log
        2. Verify database connection: psql -h localhost -U archon -d archon
        3. Check service status: systemctl status archon-*
        4. Restore from backup manually if needed
        5. Contact senior engineer for assistance
        
        Backup location: {self.backup_location}
        """
        
        print(recovery_instructions)
        
        with open('/tmp/emergency_recovery_instructions.txt', 'w') as f:
            f.write(recovery_instructions)
```

---

## Post-Migration Checklist

### Final Validation Checklist

- [ ] **User Data Migration**
  - [ ] All users successfully migrated
  - [ ] User data integrity verified
  - [ ] Custom attributes preserved
  - [ ] Password migration strategy implemented

- [ ] **Authentication Functionality**
  - [ ] Login/logout working
  - [ ] Token generation/validation working
  - [ ] OAuth flows functional
  - [ ] Password reset working
  - [ ] Session management working

- [ ] **Integration Updates**
  - [ ] All applications updated to use Archon
  - [ ] API endpoints updated
  - [ ] Mobile apps updated
  - [ ] Third-party integrations updated

- [ ] **Security Verification**
  - [ ] SSL certificates configured
  - [ ] Rate limiting enabled
  - [ ] Security headers configured
  - [ ] Audit logging enabled
  - [ ] Monitoring alerts configured

- [ ] **Performance Validation**
  - [ ] Response times acceptable
  - [ ] System resources adequate
  - [ ] Database performance optimal
  - [ ] Caching configured

- [ ] **Rollback Preparedness**
  - [ ] Rollback procedures tested
  - [ ] Backup systems verified
  - [ ] Emergency contacts identified
  - [ ] Documentation complete

- [ ] **User Communication**
  - [ ] Users notified of changes
  - [ ] Support documentation updated
  - [ ] Training materials prepared
  - [ ] Support team briefed

### Migration Success Criteria

A migration is considered successful when:

1. **100% user data migrated** without loss
2. **All authentication flows working** as expected
3. **Zero critical security issues** identified
4. **Performance meets or exceeds** previous system
5. **Rollback procedures validated** and ready
6. **User acceptance testing passed**
7. **Production monitoring** shows healthy metrics
8. **Support tickets** remain at normal levels

---

**Migration Support:**
- Create detailed migration logs
- Monitor system metrics closely
- Have rollback procedures ready
- Maintain communication channels
- Document lessons learned

This migration guide provides a comprehensive framework for moving to Archon Authentication from various systems. Adapt the specific procedures based on your source system and requirements.