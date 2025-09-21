"""
Advanced Access Control Manager
Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC) implementation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import re
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class AccessDecision(Enum):
    """Access control decisions"""
    PERMIT = "permit"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"


class PermissionType(Enum):
    """Types of permissions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    CREATE = "create"
    UPDATE = "update"
    ADMIN = "admin"
    APPROVE = "approve"
    AUDIT = "audit"
    CONFIGURE = "configure"


class ResourceType(Enum):
    """Types of resources"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    SERVICE = "service"
    SYSTEM = "system"
    USER = "user"
    ROLE = "role"
    POLICY = "policy"
    CONFIGURATION = "configuration"
    REPORT = "report"


@dataclass
class Permission:
    """Individual permission"""
    permission_id: str
    name: str
    description: str
    permission_type: PermissionType
    resource_pattern: str  # Regex or glob pattern
    conditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches_resource(self, resource: str) -> bool:
        """Check if permission applies to resource"""
        return bool(re.match(self.resource_pattern, resource))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "permission_id": self.permission_id,
            "name": self.name,
            "description": self.description,
            "permission_type": self.permission_type.value,
            "resource_pattern": self.resource_pattern,
            "conditions": self.conditions,
            "metadata": self.metadata
        }


@dataclass
class Role:
    """Role containing permissions"""
    role_id: str
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)  # Permission IDs
    parent_roles: Set[str] = field(default_factory=set)  # Parent role IDs
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    enabled: bool = True
    
    def is_expired(self) -> bool:
        return self.expires_at and datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "description": self.description,
            "permissions": list(self.permissions),
            "parent_roles": list(self.parent_roles),
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "enabled": self.enabled,
            "expired": self.is_expired()
        }


@dataclass
class Subject:
    """Subject (user/service) in access control"""
    subject_id: str
    subject_type: str  # user, service, system
    roles: Set[str] = field(default_factory=set)  # Role IDs
    direct_permissions: Set[str] = field(default_factory=set)  # Permission IDs
    attributes: Dict[str, Any] = field(default_factory=dict)
    groups: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_access: Optional[datetime] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "roles": list(self.roles),
            "direct_permissions": list(self.direct_permissions),
            "attributes": self.attributes,
            "groups": list(self.groups),
            "created_at": self.created_at.isoformat(),
            "last_access": self.last_access.isoformat() if self.last_access else None,
            "enabled": self.enabled
        }


@dataclass
class AccessRequest:
    """Access control request"""
    request_id: str
    subject_id: str
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "subject_id": self.subject_id,
            "resource": self.resource,
            "action": self.action,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AccessResponse:
    """Access control response"""
    request_id: str
    decision: AccessDecision
    reason: str
    applicable_policies: List[str] = field(default_factory=list)
    evaluated_permissions: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "applicable_policies": self.applicable_policies,
            "evaluated_permissions": self.evaluated_permissions,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


@dataclass
class Policy:
    """Access control policy"""
    policy_id: str
    name: str
    description: str
    target: Dict[str, Any]  # Target conditions (subject, resource, action)
    rule: str  # Policy rule in expression language
    effect: AccessDecision  # PERMIT or DENY
    priority: int = 0
    conditions: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "target": self.target,
            "rule": self.rule,
            "effect": self.effect.value,
            "priority": self.priority,
            "conditions": self.conditions,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


class PolicyEvaluator:
    """Evaluates access control policies"""
    
    def __init__(self):
        self.functions = {
            'equals': lambda a, b: a == b,
            'contains': lambda a, b: b in a,
            'starts_with': lambda a, b: a.startswith(b),
            'ends_with': lambda a, b: a.endswith(b),
            'matches_regex': lambda a, b: bool(re.match(b, a)),
            'in_list': lambda a, b: a in b,
            'greater_than': lambda a, b: a > b,
            'less_than': lambda a, b: a < b,
            'time_between': self._time_between,
            'has_attribute': self._has_attribute,
            'has_role': self._has_role,
            'has_permission': self._has_permission
        }
    
    def evaluate_policy(self, policy: Policy, request: AccessRequest, 
                       subject: Subject, context: Dict[str, Any]) -> bool:
        """Evaluate if policy applies and grants/denies access"""
        if not policy.enabled:
            return False
            
        # Check if policy target matches request
        if not self._matches_target(policy.target, request, subject, context):
            return False
            
        # Evaluate policy rule
        return self._evaluate_rule(policy.rule, request, subject, context)
    
    def _matches_target(self, target: Dict[str, Any], request: AccessRequest,
                       subject: Subject, context: Dict[str, Any]) -> bool:
        """Check if policy target matches the request"""
        # Subject matching
        if 'subjects' in target:
            subject_match = False
            for subject_pattern in target['subjects']:
                if self._matches_pattern(subject.subject_id, subject_pattern):
                    subject_match = True
                    break
            if not subject_match:
                return False
        
        # Resource matching
        if 'resources' in target:
            resource_match = False
            for resource_pattern in target['resources']:
                if self._matches_pattern(request.resource, resource_pattern):
                    resource_match = True
                    break
            if not resource_match:
                return False
        
        # Action matching
        if 'actions' in target:
            action_match = False
            for action_pattern in target['actions']:
                if self._matches_pattern(request.action, action_pattern):
                    action_match = True
                    break
            if not action_match:
                return False
        
        return True
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if value matches pattern (supports wildcards and regex)"""
        if pattern == "*":
            return True
        if pattern.startswith("regex:"):
            return bool(re.match(pattern[6:], value))
        return value == pattern
    
    def _evaluate_rule(self, rule: str, request: AccessRequest,
                      subject: Subject, context: Dict[str, Any]) -> bool:
        """Evaluate policy rule expression"""
        # Simple rule evaluation - in production would use proper expression parser
        try:
            # Replace variables with values
            rule = rule.replace('${subject.id}', f'"{subject.subject_id}"')
            rule = rule.replace('${subject.type}', f'"{subject.subject_type}"')
            rule = rule.replace('${resource}', f'"{request.resource}"')
            rule = rule.replace('${action}', f'"{request.action}"')
            
            # Replace context variables
            for key, value in context.items():
                rule = rule.replace(f'${{context.{key}}}', f'"{value}"')
            
            # Replace attribute variables
            for key, value in subject.attributes.items():
                rule = rule.replace(f'${{subject.{key}}}', f'"{value}"')
            
            # Evaluate simple expressions
            if 'and' in rule:
                parts = rule.split(' and ')
                return all(self._evaluate_simple_expression(part.strip()) for part in parts)
            elif 'or' in rule:
                parts = rule.split(' or ')
                return any(self._evaluate_simple_expression(part.strip()) for part in parts)
            else:
                return self._evaluate_simple_expression(rule)
                
        except Exception as e:
            logger.error(f"Error evaluating policy rule '{rule}': {str(e)}")
            return False
    
    def _evaluate_simple_expression(self, expression: str) -> bool:
        """Evaluate simple comparison expression"""
        # Handle function calls
        for func_name, func in self.functions.items():
            if expression.startswith(func_name + '('):
                return self._evaluate_function_call(expression)
        
        # Handle simple comparisons
        operators = ['==', '!=', '>', '<', '>=', '<=']
        for op in operators:
            if op in expression:
                left, right = expression.split(op, 1)
                left = left.strip().strip('"')
                right = right.strip().strip('"')
                
                if op == '==':
                    return left == right
                elif op == '!=':
                    return left != right
                # Add other operators as needed
        
        return True  # Default to true for simple expressions
    
    def _evaluate_function_call(self, expression: str) -> bool:
        """Evaluate function call expression"""
        # Extract function name and arguments
        func_name = expression.split('(')[0]
        args_str = expression[len(func_name)+1:-1]  # Remove function name and parentheses
        
        # Parse arguments (simplified)
        args = [arg.strip().strip('"') for arg in args_str.split(',')]
        
        if func_name in self.functions:
            func = self.functions[func_name]
            try:
                return func(*args)
            except Exception as e:
                logger.error(f"Error calling function {func_name}: {str(e)}")
                return False
        
        return False
    
    def _time_between(self, current_time: str, start_time: str, end_time: str) -> bool:
        """Check if current time is between start and end times"""
        try:
            current = datetime.fromisoformat(current_time).time()
            start = datetime.strptime(start_time, "%H:%M").time()
            end = datetime.strptime(end_time, "%H:%M").time()
            return start <= current <= end
        except ValueError:
            return False
    
    def _has_attribute(self, subject_attrs: str, attribute: str, value: str) -> bool:
        """Check if subject has specific attribute value"""
        # This would be implemented with actual subject data
        return True
    
    def _has_role(self, subject_roles: str, role: str) -> bool:
        """Check if subject has specific role"""
        # This would be implemented with actual subject data
        return True
    
    def _has_permission(self, subject_perms: str, permission: str) -> bool:
        """Check if subject has specific permission"""
        # This would be implemented with actual subject data
        return True


class AccessControlManager:
    """Main access control manager implementing RBAC and ABAC"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.subjects: Dict[str, Subject] = {}
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        self.policies: Dict[str, Policy] = {}
        self.policy_evaluator = PolicyEvaluator()
        
        # Caching
        self.permission_cache: Dict[str, Set[str]] = {}  # subject_id -> permissions
        self.role_hierarchy_cache: Dict[str, Set[str]] = {}  # role_id -> all inherited roles
        self.cache_ttl = timedelta(minutes=15)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Metrics
        self.access_metrics: Dict[str, int] = defaultdict(int)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize default permissions and roles
        self._initialize_defaults()
    
    def _initialize_defaults(self) -> None:
        """Initialize default permissions and roles"""
        # Create basic permissions
        basic_permissions = [
            ("read_users", "Read Users", PermissionType.READ, "users/.*"),
            ("write_users", "Write Users", PermissionType.WRITE, "users/.*"),
            ("read_system", "Read System", PermissionType.READ, "system/.*"),
            ("admin_system", "Admin System", PermissionType.ADMIN, "system/.*"),
            ("read_reports", "Read Reports", PermissionType.READ, "reports/.*"),
            ("write_reports", "Write Reports", PermissionType.WRITE, "reports/.*"),
        ]
        
        for perm_id, name, perm_type, pattern in basic_permissions:
            permission = Permission(
                permission_id=perm_id,
                name=name,
                description=f"Permission to {perm_type.value} {pattern}",
                permission_type=perm_type,
                resource_pattern=pattern
            )
            self.permissions[perm_id] = permission
        
        # Create basic roles
        user_role = Role(
            role_id="user",
            name="User",
            description="Basic user role",
            permissions={"read_users", "read_reports"}
        )
        self.roles["user"] = user_role
        
        admin_role = Role(
            role_id="admin",
            name="Administrator",
            description="Administrator role with full access",
            permissions={"read_users", "write_users", "read_system", "admin_system", "read_reports", "write_reports"},
            parent_roles={"user"}
        )
        self.roles["admin"] = admin_role
        
        # Create default policies
        admin_policy = Policy(
            policy_id="admin_full_access",
            name="Admin Full Access",
            description="Administrators have full access to all resources",
            target={
                "subjects": ["*"],
                "resources": ["*"],
                "actions": ["*"]
            },
            rule="has_role(subject.roles, 'admin')",
            effect=AccessDecision.PERMIT,
            priority=100
        )
        self.policies["admin_full_access"] = admin_policy
        
        business_hours_policy = Policy(
            policy_id="business_hours_only",
            name="Business Hours Only",
            description="Allow access only during business hours",
            target={
                "subjects": ["*"],
                "resources": ["system/*"],
                "actions": ["*"]
            },
            rule="time_between(${context.current_time}, '09:00', '17:00')",
            effect=AccessDecision.PERMIT,
            priority=50
        )
        self.policies["business_hours_only"] = business_hours_policy
        
        logger.info("Initialized default permissions, roles, and policies")
    
    def create_subject(self, subject_id: str, subject_type: str = "user",
                      roles: Set[str] = None, attributes: Dict[str, Any] = None) -> Subject:
        """Create new subject"""
        subject = Subject(
            subject_id=subject_id,
            subject_type=subject_type,
            roles=roles or set(),
            attributes=attributes or {}
        )
        
        self.subjects[subject_id] = subject
        self._invalidate_cache(subject_id)
        
        logger.info(f"Created subject {subject_id}")
        return subject
    
    def create_role(self, role_id: str, name: str, description: str = "",
                   permissions: Set[str] = None, parent_roles: Set[str] = None) -> Role:
        """Create new role"""
        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=permissions or set(),
            parent_roles=parent_roles or set()
        )
        
        self.roles[role_id] = role
        self._invalidate_all_caches()
        
        logger.info(f"Created role {role_id}")
        return role
    
    def create_permission(self, permission_id: str, name: str, description: str,
                         permission_type: PermissionType, resource_pattern: str) -> Permission:
        """Create new permission"""
        permission = Permission(
            permission_id=permission_id,
            name=name,
            description=description,
            permission_type=permission_type,
            resource_pattern=resource_pattern
        )
        
        self.permissions[permission_id] = permission
        self._invalidate_all_caches()
        
        logger.info(f"Created permission {permission_id}")
        return permission
    
    def create_policy(self, policy_id: str, name: str, description: str,
                     target: Dict[str, Any], rule: str, effect: AccessDecision,
                     priority: int = 0) -> Policy:
        """Create new policy"""
        policy = Policy(
            policy_id=policy_id,
            name=name,
            description=description,
            target=target,
            rule=rule,
            effect=effect,
            priority=priority
        )
        
        self.policies[policy_id] = policy
        
        logger.info(f"Created policy {policy_id}")
        return policy
    
    def assign_role_to_subject(self, subject_id: str, role_id: str) -> bool:
        """Assign role to subject"""
        if subject_id not in self.subjects:
            return False
        if role_id not in self.roles:
            return False
        
        self.subjects[subject_id].roles.add(role_id)
        self._invalidate_cache(subject_id)
        
        logger.info(f"Assigned role {role_id} to subject {subject_id}")
        return True
    
    def revoke_role_from_subject(self, subject_id: str, role_id: str) -> bool:
        """Revoke role from subject"""
        if subject_id not in self.subjects:
            return False
        
        self.subjects[subject_id].roles.discard(role_id)
        self._invalidate_cache(subject_id)
        
        logger.info(f"Revoked role {role_id} from subject {subject_id}")
        return True
    
    def assign_permission_to_role(self, role_id: str, permission_id: str) -> bool:
        """Assign permission to role"""
        if role_id not in self.roles:
            return False
        if permission_id not in self.permissions:
            return False
        
        self.roles[role_id].permissions.add(permission_id)
        self._invalidate_all_caches()
        
        logger.info(f"Assigned permission {permission_id} to role {role_id}")
        return True
    
    def revoke_permission_from_role(self, role_id: str, permission_id: str) -> bool:
        """Revoke permission from role"""
        if role_id not in self.roles:
            return False
        
        self.roles[role_id].permissions.discard(permission_id)
        self._invalidate_all_caches()
        
        logger.info(f"Revoked permission {permission_id} from role {role_id}")
        return True
    
    async def check_access(self, subject_id: str, resource: str, action: str,
                          context: Dict[str, Any] = None) -> AccessResponse:
        """Main access control check"""
        start_time = time.time()
        
        request = AccessRequest(
            request_id=str(uuid.uuid4()),
            subject_id=subject_id,
            resource=resource,
            action=action,
            context=context or {}
        )
        
        # Add current time to context
        request.context["current_time"] = datetime.now().isoformat()
        
        # Get subject
        subject = self.subjects.get(subject_id)
        if not subject or not subject.enabled:
            return AccessResponse(
                request_id=request.request_id,
                decision=AccessDecision.DENY,
                reason="Subject not found or disabled",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Update last access time
        subject.last_access = datetime.now()
        
        # Get applicable policies
        applicable_policies = self._get_applicable_policies(request, subject)
        
        # Sort policies by priority (highest first)
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)
        
        # Evaluate policies
        decision = AccessDecision.NOT_APPLICABLE
        reason = "No applicable policies"
        policy_names = []
        
        for policy in applicable_policies:
            policy_names.append(policy.name)
            
            if self.policy_evaluator.evaluate_policy(policy, request, subject, request.context):
                decision = policy.effect
                reason = f"Policy '{policy.name}' {policy.effect.value}"
                break
        
        # If no policies matched, check direct permissions
        if decision == AccessDecision.NOT_APPLICABLE:
            if await self._check_direct_permissions(subject, resource, action):
                decision = AccessDecision.PERMIT
                reason = "Direct permissions allow access"
            else:
                decision = AccessDecision.DENY
                reason = "No permissions found for resource"
        
        # Get effective permissions for response
        effective_permissions = await self._get_effective_permissions(subject_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        self.access_metrics[f"decision_{decision.value}"] += 1
        self.access_metrics["total_requests"] += 1
        self.performance_metrics["processing_time"].append(processing_time)
        
        response = AccessResponse(
            request_id=request.request_id,
            decision=decision,
            reason=reason,
            applicable_policies=policy_names,
            evaluated_permissions=list(effective_permissions),
            processing_time_ms=processing_time
        )
        
        logger.debug(f"Access check: {subject_id} -> {resource}:{action} = {decision.value}")
        
        return response
    
    def _get_applicable_policies(self, request: AccessRequest, subject: Subject) -> List[Policy]:
        """Get policies applicable to the request"""
        applicable = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            # Check if policy target matches
            if self.policy_evaluator._matches_target(policy.target, request, subject, request.context):
                applicable.append(policy)
        
        return applicable
    
    async def _check_direct_permissions(self, subject: Subject, resource: str, action: str) -> bool:
        """Check if subject has direct permissions for resource and action"""
        effective_permissions = await self._get_effective_permissions(subject.subject_id)
        
        for perm_id in effective_permissions:
            permission = self.permissions.get(perm_id)
            if not permission:
                continue
            
            # Check if permission matches resource
            if permission.matches_resource(resource):
                # Check if permission type allows action
                if self._permission_allows_action(permission.permission_type, action):
                    return True
        
        return False
    
    def _permission_allows_action(self, permission_type: PermissionType, action: str) -> bool:
        """Check if permission type allows the action"""
        action_lower = action.lower()
        
        if permission_type == PermissionType.READ and action_lower in ["read", "get", "list", "view"]:
            return True
        elif permission_type == PermissionType.WRITE and action_lower in ["write", "create", "update", "modify"]:
            return True
        elif permission_type == PermissionType.DELETE and action_lower in ["delete", "remove"]:
            return True
        elif permission_type == PermissionType.EXECUTE and action_lower in ["execute", "run", "invoke"]:
            return True
        elif permission_type == PermissionType.ADMIN:
            return True  # Admin allows all actions
        
        return False
    
    async def _get_effective_permissions(self, subject_id: str) -> Set[str]:
        """Get all effective permissions for subject (including inherited from roles)"""
        # Check cache
        cache_key = f"permissions_{subject_id}"
        if cache_key in self.permission_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_ttl:
                return self.permission_cache[cache_key]
        
        subject = self.subjects.get(subject_id)
        if not subject:
            return set()
        
        effective_permissions = set(subject.direct_permissions)
        
        # Get permissions from roles
        all_roles = await self._get_all_roles(subject.roles)
        for role_id in all_roles:
            role = self.roles.get(role_id)
            if role and role.enabled and not role.is_expired():
                effective_permissions.update(role.permissions)
        
        # Cache result
        self.permission_cache[cache_key] = effective_permissions
        self.cache_timestamps[cache_key] = datetime.now()
        
        return effective_permissions
    
    async def _get_all_roles(self, role_ids: Set[str]) -> Set[str]:
        """Get all roles including inherited roles from hierarchy"""
        all_roles = set(role_ids)
        
        for role_id in role_ids:
            inherited_roles = await self._get_inherited_roles(role_id)
            all_roles.update(inherited_roles)
        
        return all_roles
    
    async def _get_inherited_roles(self, role_id: str) -> Set[str]:
        """Get all inherited roles from role hierarchy"""
        # Check cache
        cache_key = f"hierarchy_{role_id}"
        if cache_key in self.role_hierarchy_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_ttl:
                return self.role_hierarchy_cache[cache_key]
        
        inherited = set()
        visited = set()
        
        def traverse_hierarchy(current_role_id: str):
            if current_role_id in visited:
                return  # Prevent infinite loops
            visited.add(current_role_id)
            
            role = self.roles.get(current_role_id)
            if role:
                for parent_id in role.parent_roles:
                    inherited.add(parent_id)
                    traverse_hierarchy(parent_id)
        
        traverse_hierarchy(role_id)
        
        # Cache result
        self.role_hierarchy_cache[cache_key] = inherited
        self.cache_timestamps[cache_key] = datetime.now()
        
        return inherited
    
    def _invalidate_cache(self, subject_id: str) -> None:
        """Invalidate cache for specific subject"""
        cache_key = f"permissions_{subject_id}"
        self.permission_cache.pop(cache_key, None)
        self.cache_timestamps.pop(cache_key, None)
    
    def _invalidate_all_caches(self) -> None:
        """Invalidate all caches"""
        self.permission_cache.clear()
        self.role_hierarchy_cache.clear()
        self.cache_timestamps.clear()
    
    def get_subject(self, subject_id: str) -> Optional[Subject]:
        """Get subject by ID"""
        return self.subjects.get(subject_id)
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        return self.roles.get(role_id)
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID"""
        return self.permissions.get(permission_id)
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get policy by ID"""
        return self.policies.get(policy_id)
    
    def list_subjects(self, subject_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all subjects"""
        subjects = []
        for subject in self.subjects.values():
            if not subject_type or subject.subject_type == subject_type:
                subjects.append(subject.to_dict())
        return subjects
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles"""
        return [role.to_dict() for role in self.roles.values()]
    
    def list_permissions(self) -> List[Dict[str, Any]]:
        """List all permissions"""
        return [permission.to_dict() for permission in self.permissions.values()]
    
    def list_policies(self) -> List[Dict[str, Any]]:
        """List all policies"""
        return [policy.to_dict() for policy in self.policies.values()]
    
    def get_access_control_metrics(self) -> Dict[str, Any]:
        """Get access control metrics"""
        avg_processing_time = 0.0
        if self.performance_metrics["processing_time"]:
            avg_processing_time = sum(self.performance_metrics["processing_time"]) / len(self.performance_metrics["processing_time"])
        
        return {
            "total_subjects": len(self.subjects),
            "total_roles": len(self.roles),
            "total_permissions": len(self.permissions),
            "total_policies": len(self.policies),
            "total_requests": self.access_metrics.get("total_requests", 0),
            "decisions": {
                "permit": self.access_metrics.get("decision_permit", 0),
                "deny": self.access_metrics.get("decision_deny", 0),
                "not_applicable": self.access_metrics.get("decision_not_applicable", 0)
            },
            "average_processing_time_ms": avg_processing_time,
            "cache_size": len(self.permission_cache)
        }
    
    async def export_configuration(self) -> Dict[str, Any]:
        """Export access control configuration"""
        return {
            "subjects": [subject.to_dict() for subject in self.subjects.values()],
            "roles": [role.to_dict() for role in self.roles.values()],
            "permissions": [permission.to_dict() for permission in self.permissions.values()],
            "policies": [policy.to_dict() for policy in self.policies.values()],
            "exported_at": datetime.now().isoformat()
        }
    
    async def import_configuration(self, config: Dict[str, Any]) -> None:
        """Import access control configuration"""
        # Import permissions first
        for perm_data in config.get("permissions", []):
            permission = Permission(
                permission_id=perm_data["permission_id"],
                name=perm_data["name"],
                description=perm_data["description"],
                permission_type=PermissionType(perm_data["permission_type"]),
                resource_pattern=perm_data["resource_pattern"],
                conditions=perm_data.get("conditions", []),
                metadata=perm_data.get("metadata", {})
            )
            self.permissions[permission.permission_id] = permission
        
        # Import roles
        for role_data in config.get("roles", []):
            role = Role(
                role_id=role_data["role_id"],
                name=role_data["name"],
                description=role_data["description"],
                permissions=set(role_data.get("permissions", [])),
                parent_roles=set(role_data.get("parent_roles", [])),
                attributes=role_data.get("attributes", {}),
                enabled=role_data.get("enabled", True)
            )
            self.roles[role.role_id] = role
        
        # Import subjects
        for subject_data in config.get("subjects", []):
            subject = Subject(
                subject_id=subject_data["subject_id"],
                subject_type=subject_data["subject_type"],
                roles=set(subject_data.get("roles", [])),
                direct_permissions=set(subject_data.get("direct_permissions", [])),
                attributes=subject_data.get("attributes", {}),
                groups=set(subject_data.get("groups", [])),
                enabled=subject_data.get("enabled", True)
            )
            self.subjects[subject.subject_id] = subject
        
        # Import policies
        for policy_data in config.get("policies", []):
            policy = Policy(
                policy_id=policy_data["policy_id"],
                name=policy_data["name"],
                description=policy_data["description"],
                target=policy_data["target"],
                rule=policy_data["rule"],
                effect=AccessDecision(policy_data["effect"]),
                priority=policy_data.get("priority", 0),
                conditions=policy_data.get("conditions", []),
                enabled=policy_data.get("enabled", True)
            )
            self.policies[policy.policy_id] = policy
        
        # Clear caches
        self._invalidate_all_caches()
        
        logger.info("Imported access control configuration")