# TypeScript Frontend Development PRP

## Context
- **Project**: {project_name}
- **Framework**: {framework} (React/Next.js/Vue)
- **Files**: {file_paths}
- **Dependencies**: {dependencies}
- **UI Library**: {ui_library}
- **State Management**: {state_management}

## Task Requirements
{requirements}

## Standards & Guidelines

### Code Quality
- Use TypeScript strict mode
- Implement proper component typing
- Use ESLint and Prettier configurations
- Follow React/Vue best practices
- Maintain component purity where possible

### Architecture Patterns
- Use composition over inheritance
- Implement custom hooks for logic reuse
- Use proper state management patterns
- Apply separation of concerns
- Implement proper error boundaries

### Performance Requirements
- Implement code splitting for large components
- Use React.memo/Vue computed for optimization
- Minimize re-renders with proper dependency arrays
- Implement proper loading states
- Use virtual scrolling for large lists

## Examples

### React Component with TypeScript
```typescript
import React, { useState, useCallback, useEffect } from 'react';
import { User, CreateUserRequest, ApiError } from '../types/api';
import { useUserService } from '../hooks/useUserService';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Alert } from '../components/ui/Alert';

interface UserFormProps {
  onUserCreated: (user: User) => void;
  className?: string;
  initialData?: Partial<CreateUserRequest>;
}

interface FormData {
  email: string;
  password: string;
  fullName: string;
}

interface FormErrors {
  email?: string;
  password?: string;
  fullName?: string;
  general?: string;
}

export const UserForm: React.FC<UserFormProps> = ({
  onUserCreated,
  className = '',
  initialData = {}
}) => {
  const [formData, setFormData] = useState<FormData>({
    email: initialData.email || '',
    password: initialData.password || '',
    fullName: initialData.fullName || ''
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { createUser, isLoading, error } = useUserService();

  const validateForm = useCallback((data: FormData): FormErrors => {
    const newErrors: FormErrors = {};

    if (!data.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email)) {
      newErrors.email = 'Please enter a valid email';
    }

    if (!data.password) {
      newErrors.password = 'Password is required';
    } else if (data.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    if (!data.fullName.trim()) {
      newErrors.fullName = 'Full name is required';
    }

    return newErrors;
  }, []);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    const formErrors = validateForm(formData);
    if (Object.keys(formErrors).length > 0) {
      setErrors(formErrors);
      return;
    }

    setIsSubmitting(true);
    setErrors({});

    try {
      const user = await createUser({
        email: formData.email,
        password: formData.password,
        full_name: formData.fullName
      });

      onUserCreated(user);

      // Reset form
      setFormData({
        email: '',
        password: '',
        fullName: ''
      });

    } catch (err) {
      const apiError = err as ApiError;
      setErrors({
        general: apiError.message || 'Failed to create user'
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [formData, validateForm, createUser, onUserCreated]);

  const handleInputChange = useCallback((field: keyof FormData) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData(prev => ({
      ...prev,
      [field]: e.target.value
    }));

    // Clear field error when user starts typing
    if (errors[field]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
  }, [errors]);

  return (
    <form onSubmit={handleSubmit} className={`space-y-4 ${className}`}>
      {errors.general && (
        <Alert variant="error">
          {errors.general}
        </Alert>
      )}

      <Input
        label="Email"
        type="email"
        value={formData.email}
        onChange={handleInputChange('email')}
        error={errors.email}
        required
        autoComplete="email"
      />

      <Input
        label="Password"
        type="password"
        value={formData.password}
        onChange={handleInputChange('password')}
        error={errors.password}
        required
        autoComplete="new-password"
      />

      <Input
        label="Full Name"
        type="text"
        value={formData.fullName}
        onChange={handleInputChange('fullName')}
        error={errors.fullName}
        required
        autoComplete="name"
      />

      <Button
        type="submit"
        disabled={isSubmitting || isLoading}
        loading={isSubmitting || isLoading}
        className="w-full"
      >
        Create User
      </Button>
    </form>
  );
};
```

### Custom Hook Example
```typescript
import { useState, useCallback, useRef, useEffect } from 'react';
import { User, CreateUserRequest, ApiError } from '../types/api';
import { userApi } from '../services/userApi';

interface UseUserServiceReturn {
  createUser: (userData: CreateUserRequest) => Promise<User>;
  getUser: (id: number) => Promise<User | null>;
  isLoading: boolean;
  error: ApiError | null;
  clearError: () => void;
}

export const useUserService = (): UseUserServiceReturn => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const createUser = useCallback(async (userData: CreateUserRequest): Promise<User> => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setIsLoading(true);
    setError(null);

    try {
      const user = await userApi.createUser(userData, {
        signal: abortControllerRef.current.signal
      });

      return user;
    } catch (err) {
      if (err.name !== 'AbortError') {
        const apiError = err as ApiError;
        setError(apiError);
        throw apiError;
      }
      throw err;
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, []);

  const getUser = useCallback(async (id: number): Promise<User | null> => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setIsLoading(true);
    setError(null);

    try {
      const user = await userApi.getUser(id, {
        signal: abortControllerRef.current.signal
      });

      return user;
    } catch (err) {
      if (err.name !== 'AbortError') {
        const apiError = err as ApiError;
        setError(apiError);
        return null;
      }
      return null;
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    createUser,
    getUser,
    isLoading,
    error,
    clearError
  };
};
```

### Type Definitions Example
```typescript
// types/api.ts
export interface User {
  id: number;
  email: string;
  fullName: string | null;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface CreateUserRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface ApiError {
  message: string;
  status: number;
  details?: Record<string, any>;
}

export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

// types/components.ts
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface FormFieldProps extends BaseComponentProps {
  label: string;
  error?: string;
  required?: boolean;
}
```

## Output Format

### Required Deliverables
1. **Component Files**
   - Properly typed React/Vue components
   - Custom hooks for business logic
   - Utility functions and helpers
   - Type definitions

2. **Tests** (≥85% coverage)
   - Component unit tests
   - Hook tests
   - Integration tests
   - E2E tests for critical paths

3. **Documentation**
   - Component API documentation
   - Storybook stories (if applicable)
   - Usage examples
   - Setup instructions

### File Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   └── Alert.tsx
│   │   └── forms/
│   │       └── UserForm.tsx
│   ├── hooks/
│   │   └── useUserService.ts
│   ├── services/
│   │   └── userApi.ts
│   ├── types/
│   │   ├── api.ts
│   │   └── components.ts
│   ├── utils/
│   │   └── validation.ts
│   └── styles/
│       └── globals.css
├── tests/
│   ├── components/
│   ├── hooks/
│   └── utils/
├── package.json
└── README.md
```

## Quality Checklist

- [ ] All components properly typed
- [ ] Props interfaces defined
- [ ] Error handling implemented
- [ ] Loading states handled
- [ ] Accessibility attributes added
- [ ] Performance optimizations applied
- [ ] Tests cover all scenarios
- [ ] ESLint/Prettier rules followed
- [ ] Mobile responsiveness verified
- [ ] Browser compatibility tested

## Performance Targets
- First Contentful Paint: <1.5s
- Largest Contentful Paint: <2.5s
- Cumulative Layout Shift: <0.1
- First Input Delay: <100ms
- Bundle size: <500KB (main chunk)

## Dependencies to Install
```bash
# Core dependencies
npm install react react-dom typescript
npm install @types/react @types/react-dom

# Development dependencies
npm install --save-dev @typescript-eslint/eslint-plugin
npm install --save-dev @typescript-eslint/parser
npm install --save-dev eslint-plugin-react-hooks
npm install --save-dev prettier eslint-config-prettier

# Testing
npm install --save-dev @testing-library/react
npm install --save-dev @testing-library/jest-dom
npm install --save-dev @testing-library/user-event

# Optional UI libraries
npm install tailwindcss @headlessui/react
# or
npm install @mui/material @emotion/react @emotion/styled
```

## Configuration Files Required

### tsconfig.json
```json
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
```