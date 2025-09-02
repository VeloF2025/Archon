# Python Backend Development PRP

## Context
- **Project**: {project_name}
- **Files**: {file_paths}
- **Dependencies**: {dependencies}
- **Database**: {database_type}
- **Framework**: FastAPI/Flask/Django

## Task Requirements
{requirements}

## Standards & Guidelines

### Code Quality
- Follow PEP 8 style guidelines
- Use type hints for all functions and variables
- Implement comprehensive error handling
- Include docstrings for all functions and classes
- Maintain test coverage >90%

### Architecture Patterns
- Use dependency injection for services
- Implement repository pattern for data access
- Apply SOLID principles
- Use async/await for I/O operations
- Implement proper logging with structured formats

### Security Requirements
- Validate all input parameters
- Use parameterized queries for database operations
- Implement proper authentication/authorization
- Sanitize output data
- Use environment variables for sensitive config

## Examples

### FastAPI Endpoint Example
```python
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, validator
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """Create a new user account."""
    try:
        logger.info(f"Creating user with email: {user_data.email}")
        
        # Check if user exists
        existing_user = await user_service.get_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        # Create user
        new_user = await user_service.create_user(user_data)
        logger.info(f"User created successfully: {new_user.id}")
        
        return UserResponse.from_orm(new_user)
        
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

### Service Layer Example
```python
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from .models import User
from .repositories import UserRepository
import bcrypt
import logging

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.user_repo = UserRepository(db_session)
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user with hashed password."""
        try:
            # Hash password
            password_hash = bcrypt.hashpw(
                user_data.password.encode('utf-8'), 
                bcrypt.gensalt()
            )
            
            # Create user object
            user = User(
                email=user_data.email,
                password_hash=password_hash.decode('utf-8'),
                full_name=user_data.full_name
            )
            
            # Save to database
            created_user = await self.user_repo.create(user)
            await self.db_session.commit()
            
            logger.info(f"User service created user: {created_user.id}")
            return created_user
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"User service error: {str(e)}")
            raise
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        return await self.user_repo.get_by_email(email)
```

### Database Model Example
```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"
```

## Output Format

### Required Deliverables
1. **Implementation Files**
   - Main application code with proper structure
   - Database models and migrations
   - API routes and handlers
   - Service layer classes
   - Configuration files

2. **Tests** (≥90% coverage)
   - Unit tests for all functions
   - Integration tests for API endpoints  
   - Database tests with test fixtures
   - Mock external dependencies

3. **Documentation**
   - API documentation (OpenAPI/Swagger)
   - Code documentation (docstrings)
   - Setup and deployment instructions
   - Configuration guide

### File Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── user_repository.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── user_service.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── users.py
│   └── schemas/
│       ├── __init__.py
│       └── user.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_services/
│   └── test_api/
├── alembic/
│   └── versions/
├── requirements.txt
└── README.md
```

## Quality Checklist

- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Error handling implemented everywhere
- [ ] Input validation on all endpoints
- [ ] Database operations use transactions
- [ ] Logging implemented throughout
- [ ] Tests cover all code paths
- [ ] Security best practices followed
- [ ] Configuration externalized
- [ ] API documentation generated

## Performance Targets
- API response time: <200ms (95th percentile)
- Database queries: <50ms average
- Memory usage: <512MB per worker
- CPU usage: <70% under normal load

## Dependencies to Install
```bash
pip install fastapi uvicorn sqlalchemy alembic psycopg2-binary
pip install pytest pytest-asyncio pytest-cov
pip install bcrypt python-jose[cryptography]
pip install pydantic[email] python-multipart
```

## Environment Variables Required
```env
DATABASE_URL=postgresql://user:password@localhost/dbname
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```