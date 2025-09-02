# API Design Guidelines

## Authentication Endpoints

### POST /auth/login
- Input: username, password
- Output: JWT token, user info
- Errors: 401 for invalid credentials

### POST /auth/logout
- Input: JWT token (header)
- Output: success confirmation
- Errors: 401 for invalid token

## Security Considerations
- Rate limiting on login attempts
- Token expiration handling
- Refresh token mechanism
