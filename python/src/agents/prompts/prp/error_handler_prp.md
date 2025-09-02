# Error Handling Agent PRP

## Context
- **Project**: {project_name}
- **Error Context**: {input_data}
- **Agent Role**: {agent_role}
- **Task**: {task_description}

## Task Requirements
{requirements}

## Error Handling Standards

### Error Categories
- **System Errors**: Infrastructure, database, external service failures
- **Application Errors**: Logic errors, validation failures
- **User Errors**: Invalid input, permission issues
- **Security Errors**: Authentication, authorization failures
- **Performance Errors**: Timeouts, resource exhaustion

### Error Handling Strategy
- **Graceful Degradation**: Fallback mechanisms
- **User Experience**: Meaningful error messages
- **Logging**: Comprehensive error tracking
- **Recovery**: Automatic retry mechanisms
- **Monitoring**: Real-time error detection

### Implementation Patterns
- **Try-Catch Blocks**: Proper exception handling
- **Error Boundaries**: Component-level error isolation
- **Circuit Breakers**: Prevent cascade failures
- **Timeout Handling**: Prevent hanging operations
- **Input Validation**: Prevent invalid data processing

### Error Response Format
```json
{
  "error": true,
  "code": "ERROR_CODE",
  "message": "User-friendly message",
  "details": "Technical details for debugging",
  "timestamp": "ISO string",
  "requestId": "unique identifier"
}
```

## Skills Applied
{skills}

## Input Data
```json
{input_data}
```

Implement robust error handling that maintains system stability and provides excellent user experience.