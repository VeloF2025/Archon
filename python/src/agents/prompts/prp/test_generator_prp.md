# Unit Test Generator PRP

## Context
- **Project**: {project_name}
- **Testing Framework**: {test_framework}
- **Files to Test**: {file_paths}
- **Coverage Target**: {coverage_target}%
- **Test Types**: {test_types}

## Task Requirements
{requirements}

## Testing Standards & Guidelines

### Test Quality Principles
- **Comprehensive Coverage**: >90% line coverage, >85% branch coverage
- **Test Isolation**: Each test independent and repeatable
- **Clear Naming**: Test names describe the scenario being tested
- **Arrange-Act-Assert**: Clear test structure
- **Meaningful Assertions**: Test behavior, not implementation
- **Fast Execution**: Unit tests run in <5 seconds total

### Test Categories
1. **Unit Tests**: Individual functions and methods
2. **Integration Tests**: Component interactions
3. **End-to-End Tests**: Full user workflows
4. **Performance Tests**: Response times and load handling
5. **Security Tests**: Input validation and access controls

## Examples

### Python Unit Tests (pytest)
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio
from datetime import datetime, timedelta

from app.services.user_service import UserService
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse
from app.main import app

class TestUserService:
    """Test suite for UserService"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return AsyncMock()
    
    @pytest.fixture
    def mock_user_repo(self):
        """Mock user repository"""
        return Mock()
    
    @pytest.fixture
    def user_service(self, mock_db_session, mock_user_repo):
        """UserService instance with mocked dependencies"""
        service = UserService(mock_db_session)
        service.user_repo = mock_user_repo
        return service
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user creation data"""
        return UserCreate(
            email="test@example.com",
            password="securepassword123",
            full_name="Test User"
        )
    
    @pytest.fixture
    def sample_user(self):
        """Sample user model instance"""
        return User(
            id=1,
            email="test@example.com",
            password_hash="$2b$12$hashed_password",
            full_name="Test User",
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, user_service, sample_user_data, sample_user, mock_db_session):
        """Test successful user creation"""
        # Arrange
        user_service.user_repo.create = AsyncMock(return_value=sample_user)
        user_service.user_repo.get_by_email = AsyncMock(return_value=None)
        
        # Act
        result = await user_service.create_user(sample_user_data)
        
        # Assert
        assert result.id == sample_user.id
        assert result.email == sample_user.email
        assert result.full_name == sample_user.full_name
        
        # Verify repository calls
        user_service.user_repo.get_by_email.assert_called_once_with("test@example.com")
        user_service.user_repo.create.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, user_service, sample_user_data, sample_user):
        """Test user creation with duplicate email fails"""
        # Arrange
        user_service.user_repo.get_by_email = AsyncMock(return_value=sample_user)
        
        # Act & Assert
        with pytest.raises(ValueError, match="User with email .* already exists"):
            await user_service.create_user(sample_user_data)
        
        # Verify no creation attempt
        user_service.user_repo.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_user_database_error(self, user_service, sample_user_data, mock_db_session):
        """Test user creation handles database errors"""
        # Arrange
        user_service.user_repo.get_by_email = AsyncMock(return_value=None)
        user_service.user_repo.create = AsyncMock(side_effect=Exception("Database error"))
        
        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await user_service.create_user(sample_user_data)
        
        # Verify rollback called
        mock_db_session.rollback.assert_called_once()
    
    def test_password_hashing(self, user_service):
        """Test password is properly hashed"""
        # Arrange
        password = "plaintext_password"
        
        # Act
        hashed = user_service._hash_password(password)
        
        # Assert
        assert hashed != password
        assert hashed.startswith("$2b$")
        assert len(hashed) == 60  # bcrypt hash length
        
        # Verify hash validation works
        assert user_service._verify_password(password, hashed)
        assert not user_service._verify_password("wrong_password", hashed)

class TestUserAPI:
    """Test suite for User API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client for API testing"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for protected endpoints"""
        return {"Authorization": "Bearer test_token"}
    
    def test_create_user_endpoint_success(self, client):
        """Test POST /users endpoint with valid data"""
        # Arrange
        user_data = {
            "email": "newuser@example.com",
            "password": "securepassword123",
            "full_name": "New User"
        }
        
        with patch('app.services.user_service.UserService.create_user') as mock_create:
            mock_user = User(
                id=1,
                email=user_data["email"],
                full_name=user_data["full_name"],
                is_active=True
            )
            mock_create.return_value = mock_user
            
            # Act
            response = client.post("/api/v1/users", json=user_data)
            
            # Assert
            assert response.status_code == 201
            assert response.json()["email"] == user_data["email"]
            assert response.json()["full_name"] == user_data["full_name"]
            assert "password" not in response.json()  # Password not exposed
    
    def test_create_user_endpoint_validation_error(self, client):
        """Test POST /users endpoint with invalid data"""
        # Arrange
        invalid_data = {
            "email": "invalid_email",  # Missing @ symbol
            "password": "123",         # Too short
            # Missing required full_name
        }
        
        # Act
        response = client.post("/api/v1/users", json=invalid_data)
        
        # Assert
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("email" in str(error) for error in errors)
        assert any("password" in str(error) for error in errors)
    
    def test_get_user_endpoint_authorized(self, client, auth_headers):
        """Test GET /users/{id} with valid authorization"""
        # Arrange
        user_id = 1
        
        with patch('app.services.user_service.UserService.get_user') as mock_get:
            mock_user = User(id=user_id, email="test@example.com", full_name="Test User")
            mock_get.return_value = mock_user
            
            # Act
            response = client.get(f"/api/v1/users/{user_id}", headers=auth_headers)
            
            # Assert
            assert response.status_code == 200
            assert response.json()["id"] == user_id
    
    def test_get_user_endpoint_unauthorized(self, client):
        """Test GET /users/{id} without authorization"""
        # Act
        response = client.get("/api/v1/users/1")
        
        # Assert
        assert response.status_code == 401
```

### TypeScript/React Tests (Jest + Testing Library)
```typescript
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { UserForm } from '../components/forms/UserForm';
import { User } from '../types/api';

// Mock server for API calls
const server = setupServer(
  rest.post('/api/v1/users', (req, res, ctx) => {
    const { email, password, full_name } = req.body as any;
    
    if (email === 'existing@example.com') {
      return res(
        ctx.status(409),
        ctx.json({ message: 'User with this email already exists' })
      );
    }
    
    return res(
      ctx.status(201),
      ctx.json({
        id: 1,
        email,
        full_name,
        is_active: true,
        created_at: new Date().toISOString()
      })
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('UserForm Component', () => {
  const mockOnUserCreated = jest.fn();
  
  beforeEach(() => {
    mockOnUserCreated.mockClear();
  });
  
  it('renders all form fields', () => {
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/full name/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create user/i })).toBeInTheDocument();
  });
  
  it('validates required fields', async () => {
    const user = userEvent.setup();
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    const submitButton = screen.getByRole('button', { name: /create user/i });
    
    // Try to submit without filling fields
    await user.click(submitButton);
    
    expect(screen.getByText(/email is required/i)).toBeInTheDocument();
    expect(screen.getByText(/password is required/i)).toBeInTheDocument();
    expect(screen.getByText(/full name is required/i)).toBeInTheDocument();
  });
  
  it('validates email format', async () => {
    const user = userEvent.setup();
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    const emailInput = screen.getByLabelText(/email/i);
    const submitButton = screen.getByRole('button', { name: /create user/i });
    
    // Enter invalid email
    await user.type(emailInput, 'invalid-email');
    await user.click(submitButton);
    
    expect(screen.getByText(/please enter a valid email/i)).toBeInTheDocument();
  });
  
  it('validates password length', async () => {
    const user = userEvent.setup();
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    const passwordInput = screen.getByLabelText(/password/i);
    const submitButton = screen.getByRole('button', { name: /create user/i });
    
    // Enter short password
    await user.type(passwordInput, '123');
    await user.click(submitButton);
    
    expect(screen.getByText(/password must be at least 8 characters/i)).toBeInTheDocument();
  });
  
  it('successfully creates user with valid data', async () => {
    const user = userEvent.setup();
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    // Fill out form
    await user.type(screen.getByLabelText(/email/i), 'newuser@example.com');
    await user.type(screen.getByLabelText(/password/i), 'securepassword123');
    await user.type(screen.getByLabelText(/full name/i), 'New User');
    
    // Submit form
    await user.click(screen.getByRole('button', { name: /create user/i }));
    
    // Wait for success
    await waitFor(() => {
      expect(mockOnUserCreated).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 1,
          email: 'newuser@example.com',
          full_name: 'New User'
        })
      );
    });
    
    // Form should be reset
    expect(screen.getByLabelText(/email/i)).toHaveValue('');
    expect(screen.getByLabelText(/password/i)).toHaveValue('');
    expect(screen.getByLabelText(/full name/i)).toHaveValue('');
  });
  
  it('handles API error responses', async () => {
    const user = userEvent.setup();
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    // Fill out form with existing email
    await user.type(screen.getByLabelText(/email/i), 'existing@example.com');
    await user.type(screen.getByLabelText(/password/i), 'securepassword123');
    await user.type(screen.getByLabelText(/full name/i), 'Existing User');
    
    // Submit form
    await user.click(screen.getByRole('button', { name: /create user/i }));
    
    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/user with this email already exists/i)).toBeInTheDocument();
    });
    
    // onUserCreated should not be called
    expect(mockOnUserCreated).not.toHaveBeenCalled();
  });
  
  it('shows loading state during submission', async () => {
    const user = userEvent.setup();
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    // Fill out form
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/password/i), 'securepassword123');
    await user.type(screen.getByLabelText(/full name/i), 'Test User');
    
    // Submit form
    const submitButton = screen.getByRole('button', { name: /create user/i });
    await user.click(submitButton);
    
    // Check loading state (button should be disabled)
    expect(submitButton).toBeDisabled();
  });
  
  it('clears field errors when user starts typing', async () => {
    const user = userEvent.setup();
    render(<UserForm onUserCreated={mockOnUserCreated} />);
    
    const emailInput = screen.getByLabelText(/email/i);
    const submitButton = screen.getByRole('button', { name: /create user/i });
    
    // Trigger validation error
    await user.click(submitButton);
    expect(screen.getByText(/email is required/i)).toBeInTheDocument();
    
    // Start typing in email field
    await user.type(emailInput, 'test');
    
    // Error should be cleared
    expect(screen.queryByText(/email is required/i)).not.toBeInTheDocument();
  });
});
```

## Output Format

### Required Deliverables
1. **Test Files**
   - Unit tests for all functions/methods
   - Integration tests for component interactions
   - API endpoint tests
   - Mock configurations and fixtures

2. **Test Configuration**
   - Test runner configuration (pytest.ini, jest.config.js)
   - Coverage reporting setup
   - CI/CD test integration
   - Test data factories/fixtures

3. **Coverage Report**
   - Line coverage metrics
   - Branch coverage analysis
   - Uncovered code identification
   - Coverage trend tracking

### Test Structure
```
tests/
├── unit/
│   ├── services/
│   │   └── test_user_service.py
│   ├── models/
│   │   └── test_user_model.py
│   └── utils/
│       └── test_validation.py
├── integration/
│   ├── api/
│   │   └── test_user_endpoints.py
│   └── database/
│       └── test_user_repository.py
├── e2e/
│   └── test_user_workflow.py
├── fixtures/
│   ├── users.py
│   └── database.py
├── conftest.py
└── pytest.ini
```

## Quality Checklist

- [ ] All public functions have tests
- [ ] Edge cases covered (empty inputs, null values, boundaries)
- [ ] Error conditions tested
- [ ] Mocks used appropriately
- [ ] Tests are independent and isolated
- [ ] Test names are descriptive
- [ ] Assertions are meaningful
- [ ] Setup and teardown implemented
- [ ] Test data is realistic
- [ ] Performance tests for critical paths

## Coverage Targets
- **Unit Tests**: >90% line coverage, >85% branch coverage
- **Integration Tests**: All API endpoints covered
- **E2E Tests**: Critical user journeys covered
- **Performance Tests**: Response time thresholds verified

## Test Configuration Examples

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
addopts = 
    --cov=app
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
    --strict-markers
    -v
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
```

### jest.config.js
```javascript
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  testMatch: ['**/__tests__/**/*.(ts|tsx|js)', '**/*.(test|spec).(ts|tsx|js)'],
  collectCoverageFrom: [
    'src/**/*.(ts|tsx)',
    '!src/**/*.d.ts',
    '!src/index.tsx',
    '!src/reportWebVitals.ts'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 85,
      statements: 85
    }
  },
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest'
  },
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1'
  }
};
```