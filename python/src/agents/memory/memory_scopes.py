#!/usr/bin/env python3
"""
Memory Scopes and Role-Based Access Control for Archon+ Phase 4

Implements role-specific memory layer access as specified in PRP AC-001:
- Agents only access scopes defined in their role configuration
- Memory persistence across sessions for appropriate layers
- Performance requirements: <100ms query response time
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)

class MemoryLayerType(Enum):
    """Memory layer types as defined in PRP"""
    GLOBAL = "global"      # System-wide patterns, best practices
    PROJECT = "project"    # Project-specific context, decisions
    JOB = "job"           # Current session/task context
    RUNTIME = "runtime"    # Immediate execution context

class AccessLevel(Enum):
    """Access levels for memory operations"""
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    NONE = "none"

@dataclass
class MemoryScope:
    """Memory scope configuration for agent roles"""
    scope_type: MemoryLayerType
    role_permissions: Set[str]  # Agent roles with access
    access_level: AccessLevel
    persistent: bool = True      # Whether data persists across sessions
    max_age_hours: Optional[int] = None  # TTL for data

@dataclass
class RoleConfiguration:
    """Role-specific memory access configuration"""
    role_name: str
    allowed_scopes: Dict[MemoryLayerType, AccessLevel]
    description: str
    created_at: float = field(default_factory=time.time)

class RoleBasedAccessControl:
    """Role-based access control for memory layers"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("python/src/agents/memory/role_configs.json")
        self.role_configurations: Dict[str, RoleConfiguration] = {}
        self.memory_scopes: Dict[MemoryLayerType, MemoryScope] = {}
        self._initialize_default_roles()
        self._load_configurations()
    
    def _initialize_default_roles(self):
        """Initialize default role configurations as specified in PRP"""
        
        # Default role configurations based on PRP specifications
        default_roles = {
            "code-implementer": RoleConfiguration(
                role_name="code-implementer",
                allowed_scopes={
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE,
                    MemoryLayerType.GLOBAL: AccessLevel.READ  # Read-only access to global patterns
                },
                description="Implements code based on requirements, accesses patterns and project context"
            ),
            
            "system-architect": RoleConfiguration(
                role_name="system-architect", 
                allowed_scopes={
                    MemoryLayerType.GLOBAL: AccessLevel.READ_WRITE,  # Can contribute to global patterns
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE
                },
                description="Designs system architecture, contributes to global patterns"
            ),
            
            "security-auditor": RoleConfiguration(
                role_name="security-auditor",
                allowed_scopes={
                    MemoryLayerType.GLOBAL: AccessLevel.READ,  # Read global security patterns
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,  # Track project vulnerabilities
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE
                },
                description="Audits security, tracks vulnerabilities and patterns"
            ),
            
            "test-coverage-validator": RoleConfiguration(
                role_name="test-coverage-validator",
                allowed_scopes={
                    MemoryLayerType.GLOBAL: AccessLevel.READ,  # Read testing patterns
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,  # Track test coverage
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE
                },
                description="Validates test coverage, tracks testing patterns"
            ),
            
            "code-quality-reviewer": RoleConfiguration(
                role_name="code-quality-reviewer",
                allowed_scopes={
                    MemoryLayerType.GLOBAL: AccessLevel.READ,  # Read quality patterns
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,  # Track quality metrics
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE
                },
                description="Reviews code quality, tracks metrics and patterns"
            ),
            
            "performance-optimizer": RoleConfiguration(
                role_name="performance-optimizer",
                allowed_scopes={
                    MemoryLayerType.GLOBAL: AccessLevel.READ,  # Read performance patterns
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,  # Track performance metrics
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE
                },
                description="Optimizes performance, tracks metrics and bottlenecks"
            ),
            
            "deployment-automation": RoleConfiguration(
                role_name="deployment-automation",
                allowed_scopes={
                    MemoryLayerType.GLOBAL: AccessLevel.READ,  # Read deployment patterns
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,  # Track deployment config
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE
                },
                description="Manages deployments, tracks configurations and patterns"
            ),
            
            "system": RoleConfiguration(
                role_name="system",
                allowed_scopes={
                    MemoryLayerType.GLOBAL: AccessLevel.READ_WRITE,  # Full access to global patterns
                    MemoryLayerType.PROJECT: AccessLevel.READ_WRITE,  # Full access to project memory
                    MemoryLayerType.JOB: AccessLevel.READ_WRITE,    # Full access to job memory
                    MemoryLayerType.RUNTIME: AccessLevel.READ_WRITE  # Full access to runtime memory
                },
                description="System-level role with full access to all memory layers"
            )
        }
        
        self.role_configurations = default_roles
        
        # Initialize memory scopes with role permissions
        self.memory_scopes = {
            MemoryLayerType.GLOBAL: MemoryScope(
                scope_type=MemoryLayerType.GLOBAL,
                role_permissions={"system-architect", "security-auditor", "code-implementer"},
                access_level=AccessLevel.READ_WRITE,
                persistent=True,
                max_age_hours=None  # Never expires
            ),
            
            MemoryLayerType.PROJECT: MemoryScope(
                scope_type=MemoryLayerType.PROJECT,
                role_permissions=set(default_roles.keys()),  # All roles can access project memory
                access_level=AccessLevel.READ_WRITE,
                persistent=True,
                max_age_hours=24 * 30  # 30 days
            ),
            
            MemoryLayerType.JOB: MemoryScope(
                scope_type=MemoryLayerType.JOB,
                role_permissions=set(default_roles.keys()),  # All roles can access job memory
                access_level=AccessLevel.READ_WRITE,
                persistent=True,
                max_age_hours=24 * 7  # 7 days
            ),
            
            MemoryLayerType.RUNTIME: MemoryScope(
                scope_type=MemoryLayerType.RUNTIME,
                role_permissions=set(default_roles.keys()),  # All roles can access runtime memory
                access_level=AccessLevel.READ_WRITE,
                persistent=False,  # Runtime memory does not persist across sessions
                max_age_hours=1  # 1 hour max
            )
        }
    
    def _load_configurations(self):
        """Load role configurations from file if exists"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Load custom role configurations
                for role_name, role_data in config_data.get("roles", {}).items():
                    allowed_scopes = {}
                    for scope_str, access_str in role_data.get("allowed_scopes", {}).items():
                        scope_type = MemoryLayerType(scope_str)
                        access_level = AccessLevel(access_str)
                        allowed_scopes[scope_type] = access_level
                    
                    self.role_configurations[role_name] = RoleConfiguration(
                        role_name=role_name,
                        allowed_scopes=allowed_scopes,
                        description=role_data.get("description", ""),
                        created_at=role_data.get("created_at", time.time())
                    )
                
                logger.info(f"Loaded role configurations from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load role configurations: {e}")
                logger.info("Using default role configurations")
    
    def save_configurations(self):
        """Save current role configurations to file"""
        try:
            config_data = {
                "roles": {},
                "last_updated": time.time()
            }
            
            for role_name, role_config in self.role_configurations.items():
                role_data = {
                    "allowed_scopes": {
                        scope.value: access.value 
                        for scope, access in role_config.allowed_scopes.items()
                    },
                    "description": role_config.description,
                    "created_at": role_config.created_at
                }
                config_data["roles"][role_name] = role_data
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved role configurations to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save role configurations: {e}")
    
    def check_access(self, agent_role: str, memory_layer: MemoryLayerType, 
                    operation: str = "read") -> bool:
        """
        Check if agent role has access to memory layer for operation
        
        Args:
            agent_role: Name of the agent role
            memory_layer: Memory layer to access
            operation: Operation type ("read" or "write")
            
        Returns:
            bool: True if access is allowed, False otherwise
        """
        # Check if role exists
        if agent_role not in self.role_configurations:
            logger.warning(f"Unknown agent role: {agent_role}")
            return False
        
        role_config = self.role_configurations[agent_role]
        
        # Check if role has access to this memory layer
        if memory_layer not in role_config.allowed_scopes:
            logger.debug(f"Role {agent_role} has no access to {memory_layer.value} layer")
            return False
        
        access_level = role_config.allowed_scopes[memory_layer]
        
        # Check operation permission
        if operation == "read" and access_level in [AccessLevel.READ, AccessLevel.READ_WRITE]:
            return True
        elif operation == "write" and access_level in [AccessLevel.WRITE, AccessLevel.READ_WRITE]:
            return True
        else:
            logger.debug(f"Role {agent_role} lacks {operation} access to {memory_layer.value} layer")
            return False
    
    def get_accessible_layers(self, agent_role: str) -> Dict[MemoryLayerType, AccessLevel]:
        """Get all memory layers accessible to an agent role"""
        if agent_role not in self.role_configurations:
            return {}
        
        return self.role_configurations[agent_role].allowed_scopes.copy()
    
    def add_role(self, role_config: RoleConfiguration) -> bool:
        """Add or update a role configuration"""
        try:
            self.role_configurations[role_config.role_name] = role_config
            self.save_configurations()
            logger.info(f"Added/updated role configuration: {role_config.role_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add role configuration: {e}")
            return False
    
    def remove_role(self, role_name: str) -> bool:
        """Remove a role configuration"""
        if role_name in self.role_configurations:
            del self.role_configurations[role_name]
            self.save_configurations()
            logger.info(f"Removed role configuration: {role_name}")
            return True
        return False
    
    def get_role_info(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a role"""
        if role_name not in self.role_configurations:
            return None
        
        role_config = self.role_configurations[role_name]
        
        return {
            "role_name": role_config.role_name,
            "description": role_config.description,
            "allowed_scopes": {
                scope.value: access.value 
                for scope, access in role_config.allowed_scopes.items()
            },
            "created_at": role_config.created_at,
            "accessible_layers": len(role_config.allowed_scopes),
            "can_write_global": MemoryLayerType.GLOBAL in role_config.allowed_scopes and 
                               role_config.allowed_scopes[MemoryLayerType.GLOBAL] in [AccessLevel.WRITE, AccessLevel.READ_WRITE]
        }
    
    def list_all_roles(self) -> List[str]:
        """List all configured role names"""
        return list(self.role_configurations.keys())
    
    def validate_memory_access_request(self, agent_role: str, memory_layer: MemoryLayerType, 
                                     operation: str, data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of memory access request
        
        Returns validation result with details for debugging
        """
        result = {
            "allowed": False,
            "agent_role": agent_role,
            "memory_layer": memory_layer.value,
            "operation": operation,
            "timestamp": time.time(),
            "reason": "",
            "performance_impact": "low"
        }
        
        # Check role exists
        if agent_role not in self.role_configurations:
            result["reason"] = f"Unknown agent role: {agent_role}"
            return result
        
        # Check scope access
        if not self.check_access(agent_role, memory_layer, operation):
            result["reason"] = f"Role {agent_role} lacks {operation} access to {memory_layer.value} layer"
            return result
        
        # Check memory scope configuration
        if memory_layer not in self.memory_scopes:
            result["reason"] = f"Memory layer {memory_layer.value} not configured"
            return result
        
        scope = self.memory_scopes[memory_layer]
        
        # Check if role is in scope permissions
        if agent_role not in scope.role_permissions:
            result["reason"] = f"Role {agent_role} not in {memory_layer.value} scope permissions"
            return result
        
        # Estimate performance impact
        if data and len(str(data)) > 10000:  # Large data
            result["performance_impact"] = "high"
        elif operation == "write":
            result["performance_impact"] = "medium"
        
        result["allowed"] = True
        result["reason"] = "Access granted"
        
        return result

# Utility functions for memory scope management
def create_role_configuration(role_name: str, scopes: Dict[str, str], description: str = "") -> RoleConfiguration:
    """
    Create a role configuration from scope strings
    
    Args:
        role_name: Name of the role
        scopes: Dict mapping layer names to access levels
        description: Role description
        
    Returns:
        RoleConfiguration instance
    """
    allowed_scopes = {}
    
    for layer_str, access_str in scopes.items():
        try:
            layer = MemoryLayerType(layer_str)
            access = AccessLevel(access_str)
            allowed_scopes[layer] = access
        except ValueError as e:
            logger.warning(f"Invalid scope configuration for {role_name}: {e}")
    
    return RoleConfiguration(
        role_name=role_name,
        allowed_scopes=allowed_scopes,
        description=description
    )

def get_default_rbac() -> RoleBasedAccessControl:
    """Get a pre-configured RBAC instance with default roles"""
    return RoleBasedAccessControl()

if __name__ == "__main__":
    # Test the role-based access control system
    rbac = RoleBasedAccessControl()
    
    print("Testing Role-Based Access Control")
    print("=" * 50)
    
    # Test role access
    test_cases = [
        ("code-implementer", MemoryLayerType.PROJECT, "read"),
        ("code-implementer", MemoryLayerType.GLOBAL, "write"),
        ("system-architect", MemoryLayerType.GLOBAL, "write"),
        ("security-auditor", MemoryLayerType.RUNTIME, "read"),
        ("unknown-role", MemoryLayerType.PROJECT, "read")
    ]
    
    for role, layer, operation in test_cases:
        allowed = rbac.check_access(role, layer, operation)
        print(f"{role:20} {layer.value:10} {operation:5} -> {'ALLOW' if allowed else 'DENY'}")
    
    print("\nRole Information:")
    for role in rbac.list_all_roles():
        info = rbac.get_role_info(role)
        print(f"{role:20} -> {len(info['allowed_scopes'])} scopes, Global write: {info['can_write_global']}")