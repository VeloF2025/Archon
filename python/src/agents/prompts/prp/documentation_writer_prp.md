# Documentation Writer PRP

## Context
- **Project**: {project_name}
- **Documentation Type**: {doc_type}
- **Target Files**: {file_paths}
- **Audience**: {target_audience}
- **Format**: {output_format}

## Task Requirements
{requirements}

## Documentation Standards

### Writing Guidelines
- **Clear and Concise**: Use simple, direct language
- **User-Focused**: Write for the intended audience
- **Consistent Style**: Follow established tone and formatting
- **Actionable**: Include concrete steps and examples
- **Maintainable**: Structure for easy updates

### Content Structure
- **Introduction**: Purpose and overview
- **Quick Start**: Get users running immediately
- **Detailed Guide**: Comprehensive instructions
- **Examples**: Real-world usage scenarios
- **Reference**: Complete API/feature documentation
- **Troubleshooting**: Common issues and solutions

## Examples

### API Documentation Template
````markdown
# User Management API

## Overview
The User Management API provides endpoints for creating, retrieving, updating, and deleting user accounts in the system.

**Base URL**: `https://api.example.com/v1`
**Authentication**: Bearer token required for all endpoints

## Quick Start

### 1. Get API Token
```bash
curl -X POST https://api.example.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'
```

### 2. Create a User
```bash
curl -X POST https://api.example.com/v1/users \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "newuser@example.com", "full_name": "New User"}'
```

## Endpoints

### Create User
Creates a new user account in the system.

**Endpoint**: `POST /users`
**Authentication**: Required

#### Request Body
```json
{
  "email": "string (required)",
  "password": "string (required, min 8 chars)",
  "full_name": "string (optional)"
}
```

#### Response
**Success (201 Created)**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "User Name",
  "is_active": true,
  "created_at": "2023-01-01T00:00:00Z"
}
```

**Error (400 Bad Request)**
```json
{
  "error": "validation_error",
  "message": "Invalid email format",
  "details": {
    "field": "email",
    "code": "invalid_format"
  }
}
```

#### Example
```bash
# Request
curl -X POST https://api.example.com/v1/users \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "jane.doe@example.com",
    "password": "securepassword123",
    "full_name": "Jane Doe"
  }'

# Response
{
  "id": 42,
  "email": "jane.doe@example.com",
  "full_name": "Jane Doe",
  "is_active": true,
  "created_at": "2023-12-01T10:30:00Z"
}
```

#### Validation Rules
- **email**: Must be valid email format, unique across system
- **password**: Minimum 8 characters, must contain letters and numbers
- **full_name**: Optional, maximum 100 characters

#### Error Codes
| Code | Description | Resolution |
|------|-------------|------------|
| `email_exists` | Email already registered | Use different email or login instead |
| `invalid_email` | Email format invalid | Check email format |
| `weak_password` | Password doesn't meet requirements | Use stronger password |

### Get User
Retrieves a specific user by ID.

**Endpoint**: `GET /users/{id}`
**Authentication**: Required

#### Path Parameters
- `id` (integer, required): User ID

#### Response
**Success (200 OK)**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "User Name",
  "is_active": true,
  "created_at": "2023-01-01T00:00:00Z",
  "last_login": "2023-12-01T10:00:00Z"
}
```

**Error (404 Not Found)**
```json
{
  "error": "not_found",
  "message": "User not found"
}
```

## Rate Limiting
All API endpoints are rate limited to prevent abuse:
- **Authenticated requests**: 1000 requests per hour
- **Unauthenticated requests**: 100 requests per hour

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Error Handling
All errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human readable message",
  "details": {
    "field": "specific_field",
    "code": "validation_code"
  }
}
```

### Common Error Codes
- `validation_error`: Input validation failed
- `authentication_required`: Valid token required
- `permission_denied`: Insufficient permissions
- `not_found`: Resource doesn't exist
- `rate_limit_exceeded`: Too many requests

## SDK Examples

### JavaScript
```javascript
import { UserAPI } from '@company/api-client';

const api = new UserAPI({
  baseURL: 'https://api.example.com/v1',
  token: 'your-api-token'
});

// Create user
const newUser = await api.users.create({
  email: 'user@example.com',
  password: 'securepassword',
  full_name: 'New User'
});

// Get user
const user = await api.users.get(42);
```

### Python
```python
from company_api import UserAPI

api = UserAPI(
    base_url='https://api.example.com/v1',
    token='your-api-token'
)

# Create user
new_user = api.users.create(
    email='user@example.com',
    password='securepassword',
    full_name='New User'
)

# Get user
user = api.users.get(42)
```
````

### README Template
```markdown
# Project Name

Brief description of what this project does and who it's for.

## Features

- 🚀 Feature 1 with brief description
- 📊 Feature 2 with brief description
- 🔒 Feature 3 with brief description

## Quick Start

### Prerequisites
- Node.js 16+ or Python 3.8+
- PostgreSQL 13+
- Redis (optional, for caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/company/project-name.git
   cd project-name
   ```

2. **Install dependencies**
   ```bash
   # Backend
   pip install -r requirements.txt
   
   # Frontend
   npm install
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

5. **Start the application**
   ```bash
   # Backend
   uvicorn app.main:app --reload
   
   # Frontend (new terminal)
   npm run dev
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/dbname

# Authentication
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### Database Setup
```bash
# Create database
createdb project_name

# Run migrations
alembic upgrade head

# Seed data (optional)
python scripts/seed_data.py
```

## Usage

### Basic Example
```python
from app.services.user_service import UserService

# Create user service
user_service = UserService()

# Create a user
user = await user_service.create_user({
    "email": "user@example.com",
    "password": "securepassword",
    "full_name": "John Doe"
})

print(f"Created user: {user.email}")
```

### Frontend Example
```typescript
import { useUserService } from './hooks/useUserService';

function UserForm() {
  const { createUser, isLoading } = useUserService();
  
  const handleSubmit = async (userData) => {
    try {
      const user = await createUser(userData);
      console.log('User created:', user);
    } catch (error) {
      console.error('Error:', error.message);
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      {/* Form fields */}
    </form>
  );
}
```

## API Reference

Full API documentation is available at `/docs` when running the server.

### Key Endpoints
- `POST /auth/login` - Authenticate user
- `POST /users` - Create user
- `GET /users/{id}` - Get user by ID
- `PUT /users/{id}` - Update user
- `DELETE /users/{id}` - Delete user

## Development

### Project Structure
```
project-name/
├── backend/
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── models/       # Database models
│   │   ├── services/     # Business logic
│   │   └── utils/        # Utilities
│   ├── tests/            # Backend tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── hooks/        # Custom hooks
│   │   ├── services/     # API clients
│   │   └── types/        # TypeScript types
│   ├── tests/            # Frontend tests
│   └── package.json
└── docs/                 # Documentation
```

### Running Tests
```bash
# Backend tests
pytest --cov=app tests/

# Frontend tests
npm test

# E2E tests
npm run test:e2e
```

### Code Quality
```bash
# Linting
flake8 app/
eslint src/

# Formatting
black app/
prettier --write src/

# Type checking
mypy app/
tsc --noEmit
```

## Deployment

### Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Production build
docker build -t project-name .
docker run -p 8000:8000 project-name
```

### Manual Deployment
1. Set up production environment variables
2. Install dependencies: `pip install -r requirements.txt`
3. Run migrations: `alembic upgrade head`
4. Start with gunicorn: `gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker`

## Troubleshooting

### Common Issues

#### Database Connection Error
```
sqlalchemy.exc.OperationalError: could not connect to server
```
**Solution**: Check PostgreSQL is running and DATABASE_URL is correct.

#### Import Errors
```
ModuleNotFoundError: No module named 'app'
```
**Solution**: Make sure you're in the project directory and virtual environment is activated.

#### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill the process using the port or use a different port.
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `npm test` and `pytest`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@company.com
- 💬 Discord: https://discord.gg/project
- 📖 Documentation: https://docs.project.com
- 🐛 Issues: https://github.com/company/project/issues
```

## Output Format

### Required Deliverables
1. **API Documentation** (Markdown/OpenAPI)
   - Complete endpoint documentation
   - Request/response examples
   - Authentication details
   - Error handling guide

2. **User Guides** (Markdown)
   - Installation instructions
   - Configuration guide
   - Usage examples
   - Troubleshooting section

3. **Code Documentation**
   - README files
   - Inline code comments
   - Architecture decision records
   - Setup guides

### Documentation Structure
```
docs/
├── api/
│   ├── authentication.md
│   ├── users.md
│   └── errors.md
├── guides/
│   ├── installation.md
│   ├── configuration.md
│   └── deployment.md
├── examples/
│   ├── basic-usage.md
│   └── advanced-scenarios.md
└── README.md
```

## Quality Checklist

- [ ] Clear, concise language used
- [ ] All code examples tested and working
- [ ] Screenshots/diagrams included where helpful
- [ ] Links and references verified
- [ ] Consistent formatting and style
- [ ] Up-to-date with current codebase
- [ ] Audience-appropriate level of detail
- [ ] Search-friendly headers and structure
- [ ] Mobile-friendly formatting
- [ ] Version information included