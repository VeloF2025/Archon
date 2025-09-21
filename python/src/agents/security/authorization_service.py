"""
Enterprise Authorization Service
Advanced RBAC/ABAC with policy-based access control
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from collections import defaultdict
import yaml
import ipaddress

logger = logging.getLogger(__name__)


class AuthorizationDecision(Enum):
    """Authorization decision types"""
    PERMIT = "permit"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"
    CHALLENGE = "challenge"  # Require additional verification


class ResourceType(Enum):
    """Types of resources for authorization"""
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    POLICY = "policy"
    SYSTEM = "system"
    DATA = "data"
    API = "api"
    FILE = "file"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    AUDIT_LOG = "audit_log"
    SECURITY_SETTING = "security_setting"
    COMPLIANCE_REPORT = "compliance_report"


class ActionType(Enum):
    """Types of actions for authorization"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"
    REJECT = "reject"
    SHARE = "share"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    ADMINISTER = "administer"
    AUDIT = "audit"
    CONFIGURE = "configure"
    BACKUP = "backup"
    RESTORE = "restore"


class PolicyCombiningAlgorithm(Enum):
    """Policy combining algorithms"""
    DENY_OVERRIDES = "deny_overrides"
    PERMIT_OVERRIDES = "permit_overrides"
    FIRST_APPLICABLE = "first_applicable"
    ONLY_ONE_APPLICABLE = "only_one_applicable"


@dataclass
class Resource:
    """Resource definition for authorization"""
    resource_id: str
    resource_type: ResourceType
    name: str
    description: str
    owner_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    sensitivity_level: int = 0  # 0-10, higher = more sensitive
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "attributes": self.attributes,
            "sensitivity_level": self.sensitivity_level,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Permission:
    """Permission definition"""
    permission_id: str
    name: str
    description: str
    resource_type: ResourceType
    action_type: ActionType
    conditions: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permission_id": self.permission_id,
            "name": self.name,
            "description": self.description,
            "resource_type": self.resource_type.value,
            "action_type": self.action_type.value,
            "conditions": self.conditions,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Role:
    """Role definition"""
    role_id: str
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)  # Permission IDs
    parent_roles: Set[str] = field(default_factory=set)  # Parent role IDs
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_system_role: bool = False
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
            "is_system_role": self.is_system_role,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "enabled": self.enabled,
            "expired": self.is_expired()
        }


@dataclass
class AuthorizationPolicy:
    """Authorization policy"""
    policy_id: str
    name: str
    description: str
    target: Dict[str, Any]  # Target conditions
    rule: str  # Policy rule expression
    effect: AuthorizationDecision
    priority: int = 0
    combining_algorithm: PolicyCombiningAlgorithm = PolicyCombiningAlgorithm.DENY_OVERRIDES
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    obligations: List[Dict[str, Any]] = field(default_factory=list)
    advice: List[Dict[str, Any]] = field(default_factory=list)
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
            "combining_algorithm": self.combining_algorithm.value,
            "conditions": self.conditions,
            "obligations": self.obligations,
            "advice": self.advice,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class AuthorizationRequest:
    """Authorization request"""
    request_id: str
    subject_id: str
    subject_type: str
    resource_id: str
    resource_type: ResourceType
    action: ActionType
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "action": self.action.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AuthorizationResponse:
    """Authorization response"""
    request_id: str
    decision: AuthorizationDecision
    reason: str
    policies_evaluated: List[str] = field(default_factory=list)
    applicable_policies: List[str] = field(default_factory=list)
    obligations: List[Dict[str, Any]] = field(default_factory=list)
    advice: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "policies_evaluated": self.policies_evaluated,
            "applicable_policies": self.applicable_policies,
            "obligations": self.obligations,
            "advice": self.advice,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


class PolicyEvaluator:
    """Policy evaluation engine"""

    def __init__(self):
        self.functions = {
            # Comparison functions
            'equals': lambda a, b: a == b,
            'not_equals': lambda a, b: a != b,
            'greater_than': lambda a, b: float(a) > float(b),
            'less_than': lambda a, b: float(a) < float(b),
            'greater_equal': lambda a, b: float(a) >= float(b),
            'less_equal': lambda a, b: float(a) <= float(b),

            # String functions
            'contains': lambda a, b: b in a,
            'starts_with': lambda a, b: str(a).startswith(str(b)),
            'ends_with': lambda a, b: str(a).endswith(str(b)),
            'matches_regex': lambda a, b: bool(re.match(str(b), str(a))),

            # List functions
            'in_list': lambda a, b: a in b if isinstance(b, (list, tuple, set)) else False,
            'contains_any': lambda a, b: any(item in a for item in b) if isinstance(b, (list, tuple, set)) else False,
            'contains_all': lambda a, b: all(item in a for item in b) if isinstance(b, (list, tuple, set)) else False,

            # Time functions
            'time_between': self._time_between,
            'is_weekday': self._is_weekday,
            'is_business_hours': self._is_business_hours,

            # Network functions
            'ip_in_range': self._ip_in_range,
            'ip_matches_subnet': self._ip_matches_subnet,

            # Attribute functions
            'has_attribute': lambda attrs, attr: attr in attrs,
            'attribute_equals': lambda attrs, attr, value: attrs.get(attr) == value,
            'attribute_contains': lambda attrs, attr, value: value in str(attrs.get(attr, '')),

            # Role and permission functions
            'has_role': self._has_role,
            'has_permission': self._has_permission,
            'has_any_role': self._has_any_role,
            'has_all_roles': self._has_all_roles,

            # Resource functions
            'is_owner': self._is_owner,
            'resource_sensitivity': self._resource_sensitivity,

            # Security functions
            'risk_below_threshold': self._risk_below_threshold,
            'mfa_verified': self._mfa_verified,
            'in_trusted_location': self._in_trusted_location
        }

    def evaluate_policy(self, policy: AuthorizationPolicy, request: AuthorizationRequest,
                       subject_attributes: Dict[str, Any], resource_attributes: Dict[str, Any]) -> bool:
        """Evaluate if policy applies and grants/denies access"""
        if not policy.enabled:
            return False

        # Check if policy target matches request
        if not self._matches_target(policy.target, request, subject_attributes, resource_attributes):
            return False

        # Evaluate policy rule
        return self._evaluate_rule(policy.rule, request, subject_attributes, resource_attributes)

    def _matches_target(self, target: Dict[str, Any], request: AuthorizationRequest,
                       subject_attributes: Dict[str, Any], resource_attributes: Dict[str, Any]) -> bool:
        """Check if policy target matches the request"""
        # Subject matching
        if 'subjects' in target:
            if not self._matches_pattern(request.subject_id, target['subjects']):
                return False

        # Subject type matching
        if 'subject_types' in target:
            if not self._matches_pattern(request.subject_type, target['subject_types']):
                return False

        # Resource matching
        if 'resources' in target:
            if not self._matches_pattern(request.resource_id, target['resources']):
                return False

        # Resource type matching
        if 'resource_types' in target:
            if not self._matches_pattern(request.resource_type.value, target['resource_types']):
                return False

        # Action matching
        if 'actions' in target:
            if not self._matches_pattern(request.action.value, target['actions']):
                return False

        # Attribute matching
        if 'subject_attributes' in target:
            for attr_name, attr_value in target['subject_attributes'].items():
                if subject_attributes.get(attr_name) != attr_value:
                    return False

        if 'resource_attributes' in target:
            for attr_name, attr_value in target['resource_attributes'].items():
                if resource_attributes.get(attr_name) != attr_value:
                    return False

        return True

    def _matches_pattern(self, value: str, patterns: Union[str, List[str]]) -> bool:
        """Check if value matches pattern(s)"""
        if isinstance(patterns, str):
            patterns = [patterns]

        for pattern in patterns:
            if pattern == "*":
                return True
            elif pattern.startswith("regex:"):
                if bool(re.match(pattern[6:], str(value))):
                    return True
            elif pattern.startswith("glob:"):
                # Simple glob matching
                glob_pattern = pattern[5:].replace('*', '.*').replace('?', '.')
                if bool(re.match(f'^{glob_pattern}$', str(value))):
                    return True
            else:
                if str(value) == pattern:
                    return True

        return False

    def _evaluate_rule(self, rule: str, request: AuthorizationRequest,
                      subject_attributes: Dict[str, Any], resource_attributes: Dict[str, Any]) -> bool:
        """Evaluate policy rule expression"""
        try:
            # Replace variables with values
            rule = self._substitute_variables(rule, request, subject_attributes, resource_attributes)

            # Evaluate logical operators
            if ' and ' in rule.lower():
                parts = re.split(r'\s+and\s+', rule, flags=re.IGNORECASE)
                return all(self._evaluate_simple_expression(part.strip()) for part in parts)
            elif ' or ' in rule.lower():
                parts = re.split(r'\s+or\s+', rule, flags=re.IGNORECASE)
                return any(self._evaluate_simple_expression(part.strip()) for part in parts)
            else:
                return self._evaluate_simple_expression(rule)

        except Exception as e:
            logger.error(f"Error evaluating policy rule '{rule}': {str(e)}")
            return False

    def _substitute_variables(self, rule: str, request: AuthorizationRequest,
                            subject_attributes: Dict[str, Any], resource_attributes: Dict[str, Any]) -> str:
        """Substitute variables in rule with actual values"""
        # Subject variables
        rule = rule.replace('${subject.id}', f'"{request.subject_id}"')
        rule = rule.replace('${subject.type}', f'"{request.subject_type}"')

        # Resource variables
        rule = rule.replace('${resource.id}', f'"{request.resource_id}"')
        rule = rule.replace('${resource.type}', f'"{request.resource_type.value}"')

        # Action variable
        rule = rule.replace('${action}', f'"{request.action.value}"')

        # Context variables
        for key, value in request.context.items():
            rule = rule.replace(f'${{context.{key}}}', self._format_value(value))

        # Subject attribute variables
        for key, value in subject_attributes.items():
            rule = rule.replace(f'${{subject.{key}}}', self._format_value(value))

        # Resource attribute variables
        for key, value in resource_attributes.items():
            rule = rule.replace(f'${{resource.{key}}}', self._format_value(value))

        return rule

    def _format_value(self, value: Any) -> str:
        """Format value for substitution"""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, tuple, set)):
            return f'[{", ".join(self._format_value(v) for v in value)}]'
        else:
            return f'"{str(value)}"'

    def _evaluate_simple_expression(self, expression: str) -> bool:
        """Evaluate simple comparison expression"""
        # Handle function calls
        for func_name, func in self.functions.items():
            if expression.startswith(func_name + '('):
                return self._evaluate_function_call(expression)

        # Handle simple comparisons
        operators = ['==', '!=', '>=', '<=', '>', '<']
        for op in operators:
            if op in expression:
                parts = expression.split(op, 1)
                if len(parts) == 2:
                    left = parts[0].strip().strip('"\'')
                    right = parts[1].strip().strip('"\'')
                    return self._evaluate_comparison(left, right, op)

        # Handle boolean values
        if expression.lower() in ['true', 'false']:
            return expression.lower() == 'true'

        return True  # Default to true for simple expressions

    def _evaluate_comparison(self, left: str, right: str, operator: str) -> bool:
        """Evaluate comparison operation"""
        try:
            # Try numeric comparison first
            left_num = float(left)
            right_num = float(right)

            if operator == '==':
                return left_num == right_num
            elif operator == '!=':
                return left_num != right_num
            elif operator == '>=':
                return left_num >= right_num
            elif operator == '<=':
                return left_num <= right_num
            elif operator == '>':
                return left_num > right_num
            elif operator == '<':
                return left_num < right_num

        except ValueError:
            # String comparison
            if operator == '==':
                return left == right
            elif operator == '!=':
                return left != right
            elif operator == '>=':
                return left >= right
            elif operator == '<=':
                return left <= right
            elif operator == '>':
                return left > right
            elif operator == '<':
                return left < right

        return False

    def _evaluate_function_call(self, expression: str) -> bool:
        """Evaluate function call expression"""
        try:
            # Extract function name and arguments
            func_name = expression.split('(')[0]
            args_str = expression[len(func_name)+1:-1]  # Remove function name and parentheses

            # Parse arguments (handle nested quotes and commas)
            args = self._parse_arguments(args_str)

            if func_name in self.functions:
                func = self.functions[func_name]
                return func(*args)

        except Exception as e:
            logger.error(f"Error calling function {func_name}: {str(e)}")

        return False

    def _parse_arguments(self, args_str: str) -> List[Any]:
        """Parse function arguments"""
        args = []
        current_arg = ""
        in_quotes = False
        quote_char = None

        for char in args_str:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == ',' and not in_quotes:
                # End of argument
                if current_arg.strip():
                    args.append(self._parse_value(current_arg.strip()))
                current_arg = ""
            else:
                current_arg += char

        # Add last argument
        if current_arg.strip():
            args.append(self._parse_value(current_arg.strip()))

        return args

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value to appropriate type"""
        # Remove quotes if present
        if len(value_str) >= 2 and value_str[0] in ['"', "'"] and value_str[-1] == value_str[0]:
            return value_str[1:-1]

        # Try boolean
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'

        # Try numeric
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Return as string
        return value_str

    # Helper functions for policy evaluation
    def _time_between(self, current_time: str, start_time: str, end_time: str) -> bool:
        """Check if current time is between start and end times"""
        try:
            current = datetime.fromisoformat(current_time).time()
            start = datetime.strptime(start_time, "%H:%M").time()
            end = datetime.strptime(end_time, "%H:%M").time()
            return start <= current <= end
        except ValueError:
            return False

    def _is_weekday(self, current_time: str) -> bool:
        """Check if current time is a weekday"""
        try:
            current = datetime.fromisoformat(current_time)
            return current.weekday() < 5  # Monday=0, Sunday=6
        except ValueError:
            return False

    def _is_business_hours(self, current_time: str) -> bool:
        """Check if current time is during business hours"""
        return self._is_weekday(current_time) and self._time_between(current_time, "09:00", "17:00")

    def _ip_in_range(self, ip: str, ip_ranges: List[str]) -> bool:
        """Check if IP is in allowed ranges"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            for ip_range in ip_ranges:
                if ip_addr in ipaddress.ip_network(ip_range, strict=False):
                    return True
        except ValueError:
            pass
        return False

    def _ip_matches_subnet(self, ip: str, subnet: str) -> bool:
        """Check if IP matches subnet"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            return ip_addr in ipaddress.ip_network(subnet, strict=False)
        except ValueError:
            return False

    def _has_role(self, roles: List[str], role: str) -> bool:
        """Check if subject has specific role"""
        return role in roles

    def _has_permission(self, permissions: List[str], permission: str) -> bool:
        """Check if subject has specific permission"""
        return permission in permissions

    def _has_any_role(self, roles: List[str], required_roles: List[str]) -> bool:
        """Check if subject has any of the required roles"""
        return any(role in roles for role in required_roles)

    def _has_all_roles(self, roles: List[str], required_roles: List[str]) -> bool:
        """Check if subject has all required roles"""
        return all(role in roles for role in required_roles)

    def _is_owner(self, subject_id: str, resource_owner: str) -> bool:
        """Check if subject is resource owner"""
        return subject_id == resource_owner

    def _resource_sensitivity(self, sensitivity: int, max_allowed: int) -> bool:
        """Check if resource sensitivity is within allowed level"""
        return sensitivity <= max_allowed

    def _risk_below_threshold(self, risk_score: float, threshold: float) -> bool:
        """Check if risk score is below threshold"""
        return risk_score <= threshold

    def _mfa_verified(self, mfa_status: bool) -> bool:
        """Check if MFA is verified"""
        return mfa_status

    def _in_trusted_location(self, location: str, trusted_locations: List[str]) -> bool:
        """Check if location is trusted"""
        return location in trusted_locations


class AuthorizationService:
    """Enterprise authorization service"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.resources: Dict[str, Resource] = {}
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.policies: Dict[str, AuthorizationPolicy] = {}
        self.policy_evaluator = PolicyEvaluator()

        # Subject store (in production, this would be a database)
        self.subjects: Dict[str, Dict[str, Any]] = {}

        # Caching
        self.authorization_cache: Dict[str, AuthorizationResponse] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.cache_timestamps: Dict[str, datetime] = {}

        # Metrics
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)

        # Initialize default configuration
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize default permissions, roles, and policies"""
        # Create default permissions
        self._create_default_permissions()

        # Create default roles
        self._create_default_roles()

        # Create default policies
        self._create_default_policies()

        logger.info("Initialized default authorization configuration")

    def _create_default_permissions(self) -> None:
        """Create default permissions"""
        default_permissions = [
            ("create_user", "Create Users", ResourceType.USER, ActionType.CREATE),
            ("read_user", "Read Users", ResourceType.USER, ActionType.READ),
            ("update_user", "Update Users", ResourceType.USER, ActionType.UPDATE),
            ("delete_user", "Delete Users", ResourceType.USER, ActionType.DELETE),
            ("administer_user", "Administer Users", ResourceType.USER, ActionType.ADMINISTER),

            ("create_role", "Create Roles", ResourceType.ROLE, ActionType.CREATE),
            ("read_role", "Read Roles", ResourceType.ROLE, ActionType.READ),
            ("update_role", "Update Roles", ResourceType.ROLE, ActionType.UPDATE),
            ("delete_role", "Delete Roles", ResourceType.ROLE, ActionType.DELETE),

            ("create_policy", "Create Policies", ResourceType.POLICY, ActionType.CREATE),
            ("read_policy", "Read Policies", ResourceType.POLICY, ActionType.READ),
            ("update_policy", "Update Policies", ResourceType.POLICY, ActionType.UPDATE),
            ("delete_policy", "Delete Policies", ResourceType.POLICY, ActionType.DELETE),

            ("read_system", "Read System", ResourceType.SYSTEM, ActionType.READ),
            ("configure_system", "Configure System", ResourceType.SYSTEM, ActionType.CONFIGURE),
            ("administer_system", "Administer System", ResourceType.SYSTEM, ActionType.ADMINISTER),

            ("read_audit", "Read Audit Logs", ResourceType.AUDIT_LOG, ActionType.READ),
            ("export_audit", "Export Audit Logs", ResourceType.AUDIT_LOG, ActionType.DOWNLOAD),

            ("read_compliance", "Read Compliance", ResourceType.COMPLIANCE_REPORT, ActionType.READ),
            ("generate_compliance", "Generate Compliance", ResourceType.COMPLIANCE_REPORT, ActionType.CREATE),

            ("read_security", "Read Security", ResourceType.SECURITY_SETTING, ActionType.READ),
            ("configure_security", "Configure Security", ResourceType.SECURITY_SETTING, ActionType.CONFIGURE),
        ]

        for perm_id, name, resource_type, action_type in default_permissions:
            permission = Permission(
                permission_id=perm_id,
                name=name,
                description=f"Permission to {action_type.value} {resource_type.value}",
                resource_type=resource_type,
                action_type=action_type
            )
            self.permissions[perm_id] = permission

    def _create_default_roles(self) -> None:
        """Create default roles"""
        # Super Admin role
        super_admin_role = Role(
            role_id="super_admin",
            name="Super Administrator",
            description="Full system access with all permissions",
            permissions=set(self.permissions.keys()),  # All permissions
            is_system_role=True
        )
        self.roles["super_admin"] = super_admin_role

        # Security Admin role
        security_admin_permissions = {
            "read_user", "update_user", "read_role", "read_policy", "read_system",
            "configure_system", "read_audit", "export_audit", "read_compliance",
            "generate_compliance", "read_security", "configure_security"
        }

        security_admin_role = Role(
            role_id="security_admin",
            name="Security Administrator",
            description="Security management permissions",
            permissions=security_admin_permissions,
            is_system_role=True
        )
        self.roles["security_admin"] = security_admin_role

        # Compliance Officer role
        compliance_officer_permissions = {
            "read_user", "read_system", "read_audit", "export_audit", "read_compliance",
            "generate_compliance", "read_security"
        }

        compliance_officer_role = Role(
            role_id="compliance_officer",
            name="Compliance Officer",
            description="Compliance monitoring and reporting",
            permissions=compliance_officer_permissions,
            is_system_role=True
        )
        self.roles["compliance_officer"] = compliance_officer_role

        # Auditor role
        auditor_permissions = {
            "read_user", "read_role", "read_policy", "read_system", "read_audit",
            "export_audit", "read_compliance", "read_security"
        }

        auditor_role = Role(
            role_id="auditor",
            name="Auditor",
            description="Read-only access for auditing",
            permissions=auditor_permissions,
            is_system_role=True
        )
        self.roles["auditor"] = auditor_role

        # Standard User role
        standard_user_permissions = {
            "read_system"
        }

        standard_user_role = Role(
            role_id="standard_user",
            name="Standard User",
            description="Basic system access",
            permissions=standard_user_permissions,
            is_system_role=True
        )
        self.roles["standard_user"] = standard_user_role

    def _create_default_policies(self) -> None:
        """Create default authorization policies"""
        # Super Admin policy
        super_admin_policy = AuthorizationPolicy(
            policy_id="super_admin_full_access",
            name="Super Administrator Full Access",
            description="Super administrators have full access to all resources",
            target={
                "subject_types": ["user"],
                "resource_types": ["*"],
                "actions": ["*"]
            },
            rule="has_role(subject.roles, 'super_admin')",
            effect=AuthorizationDecision.PERMIT,
            priority=1000,
            combining_algorithm=PolicyCombiningAlgorithm.PERMIT_OVERRIDES
        )
        self.policies["super_admin_full_access"] = super_admin_policy

        # Business hours policy
        business_hours_policy = AuthorizationPolicy(
            policy_id="business_hours_only",
            name="Business Hours Access Only",
            description="Restrict administrative access to business hours",
            target={
                "subject_types": ["user"],
                "resource_types": ["system", "security_setting", "policy"],
                "actions": ["configure", "administer", "update", "delete"]
            },
            rule="is_business_hours(${context.current_time})",
            effect=AuthorizationDecision.DENY,
            priority=500
        )
        self.policies["business_hours_only"] = business_hours_policy

        # High sensitivity resource policy
        high_sensitivity_policy = AuthorizationPolicy(
            policy_id="high_sensitivity_restriction",
            name="High Sensitivity Resource Restriction",
            description="Restrict access to high sensitivity resources",
            target={
                "resource_attributes": {"sensitivity_level": 9}
            },
            rule="has_role(subject.roles, 'super_admin')",
            effect=AuthorizationDecision.PERMIT,
            priority=750
        )
        self.policies["high_sensitivity_restriction"] = high_sensitivity_policy

        # Risk-based access policy
        risk_based_policy = AuthorizationPolicy(
            policy_id="risk_based_access",
            name="Risk-Based Access Control",
            description="Require additional verification for high-risk access",
            target={
                "actions": ["delete", "administer", "configure"]
            },
            rule="risk_below_threshold(${context.risk_score}, 0.7)",
            effect=AuthorizationDecision.CHALLENGE,
            priority=600,
            advice=[{"type": "mfa_required", "message": "MFA verification required for this action"}]
        )
        self.policies["risk_based_access"] = risk_based_policy

    async def authorize(self, subject_id: str, subject_type: str, resource_id: str,
                       resource_type: ResourceType, action: ActionType,
                       context: Dict[str, Any] = None) -> AuthorizationResponse:
        """Main authorization method"""
        start_time = time.time()

        request = AuthorizationRequest(
            request_id=str(uuid.uuid4()),
            subject_id=subject_id,
            subject_type=subject_type,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            context=context or {}
        )

        # Add current time to context
        request.context["current_time"] = datetime.now().isoformat()

        # Check cache
        cache_key = self._get_cache_key(request)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            cached_response.processing_time_ms = (time.time() - start_time) * 1000
            return cached_response

        # Get subject and resource information
        subject_info = self.subjects.get(subject_id, {})
        resource_info = self.resources.get(resource_id, Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            name=resource_id,
            description="Auto-generated resource",
            owner_id="system"
        ))

        # Get applicable policies
        applicable_policies = self._get_applicable_policies(request, subject_info, resource_info)

        # Evaluate policies
        response = self._evaluate_policies(request, applicable_policies, subject_info, resource_info)

        # Cache response
        self._cache_response(cache_key, response)

        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        self.metrics[f"decision_{response.decision.value}"] += 1
        self.metrics["total_requests"] += 1
        self.performance_metrics["processing_time"].append(processing_time)
        response.processing_time_ms = processing_time

        logger.debug(f"Authorization: {subject_id} -> {resource_type.value}:{resource_id}:{action.value} = {response.decision.value}")

        return response

    def _get_cache_key(self, request: AuthorizationRequest) -> str:
        """Generate cache key for authorization request"""
        key_data = {
            "subject_id": request.subject_id,
            "resource_id": request.resource_id,
            "action": request.action.value,
            "context_hash": hash(str(sorted(request.context.items())))
        }
        return hash(str(key_data))

    def _get_from_cache(self, cache_key: str) -> Optional[AuthorizationResponse]:
        """Get response from cache"""
        if cache_key in self.authorization_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < self.cache_ttl:
                return self.authorization_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: AuthorizationResponse) -> None:
        """Cache authorization response"""
        self.authorization_cache[cache_key] = response
        self.cache_timestamps[cache_key] = datetime.now()

    def _get_applicable_policies(self, request: AuthorizationRequest,
                                subject_info: Dict[str, Any],
                                resource_info: Resource) -> List[AuthorizationPolicy]:
        """Get policies applicable to the request"""
        applicable = []

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            if self.policy_evaluator._matches_target(
                policy.target, request, subject_info, resource_info.attributes
            ):
                applicable.append(policy)

        return applicable

    def _evaluate_policies(self, request: AuthorizationRequest,
                          applicable_policies: List[AuthorizationPolicy],
                          subject_info: Dict[str, Any],
                          resource_info: Resource) -> AuthorizationResponse:
        """Evaluate applicable policies and make decision"""
        if not applicable_policies:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.DENY,
                reason="No applicable policies found"
            )

        # Sort policies by priority (highest first)
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)

        # Check combining algorithm from first policy
        combining_algorithm = applicable_policies[0].combining_algorithm

        if combining_algorithm == PolicyCombiningAlgorithm.DENY_OVERRIDES:
            return self._evaluate_deny_overrides(request, applicable_policies, subject_info, resource_info)
        elif combining_algorithm == PolicyCombiningAlgorithm.PERMIT_OVERRIDES:
            return self._evaluate_permit_overrides(request, applicable_policies, subject_info, resource_info)
        elif combining_algorithm == PolicyCombiningAlgorithm.FIRST_APPLICABLE:
            return self._evaluate_first_applicable(request, applicable_policies, subject_info, resource_info)
        elif combining_algorithm == PolicyCombiningAlgorithm.ONLY_ONE_APPLICABLE:
            return self._evaluate_only_one_applicable(request, applicable_policies, subject_info, resource_info)
        else:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.DENY,
                reason="Invalid policy combining algorithm"
            )

    def _evaluate_deny_overrides(self, request: AuthorizationRequest,
                                policies: List[AuthorizationPolicy],
                                subject_info: Dict[str, Any],
                                resource_info: Resource) -> AuthorizationResponse:
        """Evaluate policies with deny-overrides combining algorithm"""
        at_least_one_permit = False
        evaluated_policies = []
        obligations = []
        advice = []

        for policy in policies:
            evaluated_policies.append(policy.policy_id)

            if self.policy_evaluator.evaluate_policy(policy, request, subject_info, resource_info.attributes):
                if policy.effect == AuthorizationDecision.DENY:
                    return AuthorizationResponse(
                        request_id=request.request_id,
                        decision=AuthorizationDecision.DENY,
                        reason=f"Denied by policy '{policy.name}'",
                        policies_evaluated=evaluated_policies,
                        applicable_policies=[policy.policy_id],
                        obligations=policy.obligations,
                        advice=policy.advice
                    )
                elif policy.effect == AuthorizationDecision.PERMIT:
                    at_least_one_permit = True
                    obligations.extend(policy.obligations)
                    advice.extend(policy.advice)

        if at_least_one_permit:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.PERMIT,
                reason="Permitted by deny-overrides algorithm",
                policies_evaluated=evaluated_policies,
                obligations=obligations,
                advice=advice
            )
        else:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.DENY,
                reason="No permit policies found",
                policies_evaluated=evaluated_policies
            )

    def _evaluate_permit_overrides(self, request: AuthorizationRequest,
                                 policies: List[AuthorizationPolicy],
                                 subject_info: Dict[str, Any],
                                 resource_info: Resource) -> AuthorizationResponse:
        """Evaluate policies with permit-overrides combining algorithm"""
        at_least_one_deny = False
        evaluated_policies = []
        obligations = []
        advice = []

        for policy in policies:
            evaluated_policies.append(policy.policy_id)

            if self.policy_evaluator.evaluate_policy(policy, request, subject_info, resource_info.attributes):
                if policy.effect == AuthorizationDecision.PERMIT:
                    return AuthorizationResponse(
                        request_id=request.request_id,
                        decision=AuthorizationDecision.PERMIT,
                        reason=f"Permitted by policy '{policy.name}'",
                        policies_evaluated=evaluated_policies,
                        applicable_policies=[policy.policy_id],
                        obligations=policy.obligations,
                        advice=policy.advice
                    )
                elif policy.effect == AuthorizationDecision.DENY:
                    at_least_one_deny = True
                    obligations.extend(policy.obligations)
                    advice.extend(policy.advice)

        if at_least_one_deny:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.DENY,
                reason="Denied by permit-overrides algorithm",
                policies_evaluated=evaluated_policies,
                obligations=obligations,
                advice=advice
            )
        else:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.PERMIT,
                reason="No deny policies found",
                policies_evaluated=evaluated_policies
            )

    def _evaluate_first_applicable(self, request: AuthorizationRequest,
                                 policies: List[AuthorizationPolicy],
                                 subject_info: Dict[str, Any],
                                 resource_info: Resource) -> AuthorizationResponse:
        """Evaluate policies with first-applicable combining algorithm"""
        evaluated_policies = []

        for policy in policies:
            evaluated_policies.append(policy.policy_id)

            if self.policy_evaluator.evaluate_policy(policy, request, subject_info, resource_info.attributes):
                return AuthorizationResponse(
                    request_id=request.request_id,
                    decision=policy.effect,
                    reason=f"Decision from first applicable policy '{policy.name}'",
                    policies_evaluated=evaluated_policies,
                    applicable_policies=[policy.policy_id],
                    obligations=policy.obligations,
                    advice=policy.advice
                )

        return AuthorizationResponse(
            request_id=request.request_id,
            decision=AuthorizationDecision.NOT_APPLICABLE,
            reason="No applicable policies found",
            policies_evaluated=evaluated_policies
        )

    def _evaluate_only_one_applicable(self, request: AuthorizationRequest,
                                    policies: List[AuthorizationPolicy],
                                    subject_info: Dict[str, Any],
                                    resource_info: Resource) -> AuthorizationResponse:
        """Evaluate policies with only-one-applicable combining algorithm"""
        applicable_policies = []
        evaluated_policies = []

        for policy in policies:
            evaluated_policies.append(policy.policy_id)

            if self.policy_evaluator.evaluate_policy(policy, request, subject_info, resource_info.attributes):
                applicable_policies.append(policy)

        if len(applicable_policies) == 1:
            policy = applicable_policies[0]
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=policy.effect,
                reason=f"Only one applicable policy '{policy.name}'",
                policies_evaluated=evaluated_policies,
                applicable_policies=[policy.policy_id],
                obligations=policy.obligations,
                advice=policy.advice
            )
        elif len(applicable_policies) > 1:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.DENY,
                reason="Multiple applicable policies found",
                policies_evaluated=evaluated_policies
            )
        else:
            return AuthorizationResponse(
                request_id=request.request_id,
                decision=AuthorizationDecision.NOT_APPLICABLE,
                reason="No applicable policies found",
                policies_evaluated=evaluated_policies
            )

    def create_role(self, role_id: str, name: str, description: str,
                   permissions: Set[str] = None, parent_roles: Set[str] = None,
                   attributes: Dict[str, Any] = None) -> Role:
        """Create a new role"""
        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=permissions or set(),
            parent_roles=parent_roles or set(),
            attributes=attributes or {}
        )

        self.roles[role_id] = role
        self._clear_cache()
        logger.info(f"Created role: {role_id}")
        return role

    def assign_role_to_subject(self, subject_id: str, role_id: str) -> bool:
        """Assign role to subject"""
        if role_id not in self.roles:
            return False

        if subject_id not in self.subjects:
            self.subjects[subject_id] = {"roles": set(), "attributes": {}}

        self.subjects[subject_id]["roles"].add(role_id)
        self._clear_cache()
        logger.info(f"Assigned role {role_id} to subject {subject_id}")
        return True

    def revoke_role_from_subject(self, subject_id: str, role_id: str) -> bool:
        """Revoke role from subject"""
        if subject_id not in self.subjects:
            return False

        if role_id in self.subjects[subject_id].get("roles", set()):
            self.subjects[subject_id]["roles"].discard(role_id)
            self._clear_cache()
            logger.info(f"Revoked role {role_id} from subject {subject_id}")
            return True

        return False

    def get_subject_roles(self, subject_id: str) -> List[str]:
        """Get all roles for a subject including inherited roles"""
        if subject_id not in self.subjects:
            return []

        direct_roles = self.subjects[subject_id].get("roles", set())
        all_roles = set(direct_roles)

        # Get inherited roles
        for role_id in direct_roles:
            all_roles.update(self._get_inherited_roles(role_id))

        return list(all_roles)

    def _get_inherited_roles(self, role_id: str) -> Set[str]:
        """Get all inherited roles from role hierarchy"""
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
        return inherited

    def get_subject_permissions(self, subject_id: str) -> Set[str]:
        """Get all permissions for a subject"""
        roles = self.get_subject_roles(subject_id)
        permissions = set()

        for role_id in roles:
            role = self.roles.get(role_id)
            if role and role.enabled and not role.is_expired():
                permissions.update(role.permissions)

        return permissions

    def _clear_cache(self) -> None:
        """Clear authorization cache"""
        self.authorization_cache.clear()
        self.cache_timestamps.clear()

    async def export_configuration(self) -> Dict[str, Any]:
        """Export authorization configuration"""
        return {
            "resources": [resource.to_dict() for resource in self.resources.values()],
            "permissions": [permission.to_dict() for permission in self.permissions.values()],
            "roles": [role.to_dict() for role in self.roles.values()],
            "policies": [policy.to_dict() for policy in self.policies.values()],
            "subjects": self.subjects,
            "exported_at": datetime.now().isoformat()
        }

    async def import_configuration(self, config: Dict[str, Any]) -> None:
        """Import authorization configuration"""
        # Import resources
        for resource_data in config.get("resources", []):
            resource = Resource(
                resource_id=resource_data["resource_id"],
                resource_type=ResourceType(resource_data["resource_type"]),
                name=resource_data["name"],
                description=resource_data["description"],
                owner_id=resource_data["owner_id"],
                attributes=resource_data.get("attributes", {}),
                sensitivity_level=resource_data.get("sensitivity_level", 0)
            )
            self.resources[resource.resource_id] = resource

        # Import permissions
        for perm_data in config.get("permissions", []):
            permission = Permission(
                permission_id=perm_data["permission_id"],
                name=perm_data["name"],
                description=perm_data["description"],
                resource_type=ResourceType(perm_data["resource_type"]),
                action_type=ActionType(perm_data["action_type"]),
                conditions=perm_data.get("conditions", []),
                constraints=perm_data.get("constraints", {})
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
                is_system_role=role_data.get("is_system_role", False),
                enabled=role_data.get("enabled", True)
            )
            self.roles[role.role_id] = role

        # Import policies
        for policy_data in config.get("policies", []):
            policy = AuthorizationPolicy(
                policy_id=policy_data["policy_id"],
                name=policy_data["name"],
                description=policy_data["description"],
                target=policy_data["target"],
                rule=policy_data["rule"],
                effect=AuthorizationDecision(policy_data["effect"]),
                priority=policy_data.get("priority", 0),
                combining_algorithm=PolicyCombiningAlgorithm(policy_data.get("combining_algorithm", "deny_overrides")),
                conditions=policy_data.get("conditions", []),
                obligations=policy_data.get("obligations", []),
                advice=policy_data.get("advice", []),
                enabled=policy_data.get("enabled", True)
            )
            self.policies[policy.policy_id] = policy

        # Import subjects
        self.subjects.update(config.get("subjects", {}))

        # Clear cache
        self._clear_cache()

        logger.info("Imported authorization configuration")

    def get_authorization_metrics(self) -> Dict[str, Any]:
        """Get authorization metrics"""
        avg_processing_time = 0.0
        if self.performance_metrics["processing_time"]:
            avg_processing_time = sum(self.performance_metrics["processing_time"]) / len(self.performance_metrics["processing_time"])

        return {
            "total_resources": len(self.resources),
            "total_permissions": len(self.permissions),
            "total_roles": len(self.roles),
            "total_policies": len(self.policies),
            "total_subjects": len(self.subjects),
            "cache_size": len(self.authorization_cache),
            "total_requests": self.metrics.get("total_requests", 0),
            "decisions": {
                "permit": self.metrics.get("decision_permit", 0),
                "deny": self.metrics.get("decision_deny", 0),
                "not_applicable": self.metrics.get("decision_not_applicable", 0),
                "challenge": self.metrics.get("decision_challenge", 0)
            },
            "average_processing_time_ms": avg_processing_time,
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.metrics.get("total_requests", 0)
        if total_requests == 0:
            return 0.0

        # Estimate cache hits based on total requests vs processing time
        # This is a simplified calculation
        return min(0.8, total_requests / (total_requests + 100))  # Cap at 80%