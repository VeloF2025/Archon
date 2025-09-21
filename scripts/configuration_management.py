#!/usr/bin/env python3
"""
Agency Swarm Configuration Management System

Comprehensive configuration management for the Agency Swarm deployment
including environment-specific configs, secrets management, and configuration validation.

Features:
- Environment-specific configuration management
- Secrets encryption and management
- Configuration validation and templating
- Dynamic configuration updates
- Configuration versioning and rollback
- Integration with Kubernetes ConfigMaps and Secrets
"""

import os
import sys
import json
import yaml
import base64
import hashlib
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jinja2

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('config_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    name: str
    username: str
    password: str
    ssl_mode: str = "require"
    pool_size: int = 20
    max_overflow: int = 30

@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str
    port: int
    db: int = 0
    password: Optional[str] = None
    ssl: bool = True
    max_connections: int = 50

@dataclass
class MCPConfig:
    """MCP server configuration."""
    host: str
    port: int
    endpoint: str
    tools: List[str]
    timeout: int = 30
    max_retries: int = 3

@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret: str
    jwt_expiry: int
    session_timeout: int
    rate_limit: int
    cors_origins: List[str]
    ssl_enabled: bool = True
    encryption_key: str

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool
    prometheus_port: int
    grafana_port: int
    log_level: str
    metrics_interval: int = 15

@dataclass
class AgencyConfig:
    """Agency Swarm specific configuration."""
    max_agents: int
    max_workflows: int
    thread_timeout: int
    message_timeout: int
    enable_persistence: bool = True
    enable_streaming: bool = True
    auto_scaling: bool = True

@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""
    environment: str
    version: str
    database: DatabaseConfig
    redis: RedisConfig
    mcp: MCPConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    agency: AgencyConfig
    kubernetes_namespace: str
    replicas: int
    docker_registry: str

class ConfigurationManager:
    """Agency Swarm configuration management system."""

    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Initialize encryption key
        self.encryption_key = self._get_encryption_key()
        self.cipher = Fernet(self.encryption_key)

        # Template engine
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.config_dir / "templates")),
            autoescape=True
        )

        # Configuration cache
        self.config_cache: Dict[str, EnvironmentConfig] = {}
        self.config_history: List[Dict[str, Any]] = []

    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key."""
        key_file = self.config_dir / ".encryption_key"

        if key_file.exists():
            return key_file.read_bytes()

        # Generate new key
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        key_file.chmod(0o600)  # Restrict permissions

        return key

    def generate_config_template(self, environment: str) -> str:
        """Generate configuration template for environment."""
        template_content = """
# Agency Swarm Configuration - {{ environment }}

# Environment settings
environment: "{{ environment }}"
version: "{{ version }}"
kubernetes_namespace: "archon-{{ environment }}"
replicas: {{ replicas }}
docker_registry: "{{ docker_registry }}"

# Database configuration
database:
  host: "{{ database.host }}"
  port: {{ database.port }}
  name: "{{ database.name }}"
  username: "{{ database.username }}"
  password: "{{ database.password }}"
  ssl_mode: "{{ database.ssl_mode }}"
  pool_size: {{ database.pool_size }}
  max_overflow: {{ database.max_overflow }}

# Redis configuration
redis:
  host: "{{ redis.host }}"
  port: {{ redis.port }}
  db: {{ redis.db }}
  password: "{{ redis.password }}"
  ssl: {{ redis.ssl }}
  max_connections: {{ redis.max_connections }}

# MCP server configuration
mcp:
  host: "{{ mcp.host }}"
  port: {{ mcp.port }}
  endpoint: "{{ mcp.endpoint }}"
  tools: {{ mcp.tools | tojson }}
  timeout: {{ mcp.timeout }}
  max_retries: {{ mcp.max_retries }}

# Security configuration
security:
  jwt_secret: "{{ security.jwt_secret }}"
  jwt_expiry: {{ security.jwt_expiry }}
  session_timeout: {{ security.session_timeout }}
  rate_limit: {{ security.rate_limit }}
  cors_origins: {{ security.cors_origins | tojson }}
  ssl_enabled: {{ security.ssl_enabled }}
  encryption_key: "{{ security.encryption_key }}"

# Monitoring configuration
monitoring:
  enabled: {{ monitoring.enabled }}
  prometheus_port: {{ monitoring.prometheus_port }}
  grafana_port: {{ monitoring.grafana_port }}
  log_level: "{{ monitoring.log_level }}"
  metrics_interval: {{ monitoring.metrics_interval }}

# Agency Swarm configuration
agency:
  max_agents: {{ agency.max_agents }}
  max_workflows: {{ agency.max_workflows }}
  thread_timeout: {{ agency.thread_timeout }}
  message_timeout: {{ agency.message_timeout }}
  enable_persistence: {{ agency.enable_persistence }}
  enable_streaming: {{ agency.enable_streaming }}
  auto_scaling: {{ agency.auto_scaling }}
"""

        return template_content

    def create_environment_config(
        self,
        environment: str,
        base_config: Optional[Dict[str, Any]] = None
    ) -> EnvironmentConfig:
        """Create environment-specific configuration."""
        logger.info(f"Creating configuration for environment: {environment}")

        # Default values
        defaults = self._get_environment_defaults(environment)

        # Merge with provided config
        if base_config:
            defaults.update(base_config)

        try:
            # Create configuration objects
            database_config = DatabaseConfig(**defaults['database'])
            redis_config = RedisConfig(**defaults['redis'])
            mcp_config = MCPConfig(**defaults['mcp'])
            security_config = SecurityConfig(**defaults['security'])
            monitoring_config = MonitoringConfig(**defaults['monitoring'])
            agency_config = AgencyConfig(**defaults['agency'])

            env_config = EnvironmentConfig(
                environment=environment,
                version=defaults['version'],
                database=database_config,
                redis=redis_config,
                mcp=mcp_config,
                security=security_config,
                monitoring=monitoring_config,
                agency=agency_config,
                kubernetes_namespace=defaults['kubernetes_namespace'],
                replicas=defaults['replicas'],
                docker_registry=defaults['docker_registry']
            )

            # Validate configuration
            self._validate_config(env_config)

            # Cache configuration
            self.config_cache[environment] = env_config

            logger.info(f"Configuration created successfully for {environment}")
            return env_config

        except Exception as e:
            logger.error(f"Failed to create configuration for {environment}: {e}")
            raise

    def _get_environment_defaults(self, environment: str) -> Dict[str, Any]:
        """Get default configuration for environment."""
        base_defaults = {
            'version': '3.0.0',
            'database': {
                'host': f'postgres-{environment}.example.com',
                'port': 5432,
                'name': f'archon_{environment}',
                'username': f'archon_{environment}_user',
                'password': self._generate_secure_password(),
                'ssl_mode': 'require',
                'pool_size': 20,
                'max_overflow': 30
            },
            'redis': {
                'host': f'redis-{environment}.example.com',
                'port': 6379,
                'db': 0,
                'password': self._generate_secure_password(),
                'ssl': True,
                'max_connections': 50
            },
            'mcp': {
                'host': 'mcp-service',
                'port': 8051,
                'endpoint': '/api/mcp',
                'tools': ['archon_perform_rag_query', 'archon_manage_project'],
                'timeout': 30,
                'max_retries': 3
            },
            'security': {
                'jwt_secret': self._generate_secure_key(),
                'jwt_expiry': 3600,
                'session_timeout': 1800,
                'rate_limit': 1000,
                'cors_origins': ['http://localhost:3738'],
                'ssl_enabled': True,
                'encryption_key': self._generate_secure_key()
            },
            'monitoring': {
                'enabled': True,
                'prometheus_port': 9090,
                'grafana_port': 3000,
                'log_level': 'INFO',
                'metrics_interval': 15
            },
            'agency': {
                'max_agents': 50,
                'max_workflows': 100,
                'thread_timeout': 3600,
                'message_timeout': 300,
                'enable_persistence': True,
                'enable_streaming': True,
                'auto_scaling': True
            }
        }

        # Environment-specific overrides
        env_specific = {
            'development': {
                'kubernetes_namespace': 'archon-dev',
                'replicas': 1,
                'docker_registry': 'localhost:5000',
                'database': {
                    'host': 'localhost',
                    'ssl_mode': 'disable'
                },
                'redis': {
                    'host': 'localhost',
                    'ssl': False
                },
                'monitoring': {
                    'log_level': 'DEBUG'
                }
            },
            'staging': {
                'kubernetes_namespace': 'archon-staging',
                'replicas': 2,
                'docker_registry': 'registry.example.com',
                'monitoring': {
                    'log_level': 'INFO'
                }
            },
            'production': {
                'kubernetes_namespace': 'archon-prod',
                'replicas': 3,
                'docker_registry': 'registry.example.com',
                'agency': {
                    'max_agents': 100,
                    'max_workflows': 200
                },
                'monitoring': {
                    'log_level': 'WARNING'
                }
            }
        }

        # Merge environment-specific overrides
        if environment in env_specific:
            self._deep_merge(base_defaults, env_specific[environment])

        return base_defaults

    def _generate_secure_password(self, length: int = 32) -> str:
        """Generate secure random password."""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def _generate_secure_key(self, length: int = 64) -> str:
        """Generate secure random key."""
        import secrets
        return ''.join(secrets.hex() for _ in range(length))

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _validate_config(self, config: EnvironmentConfig) -> None:
        """Validate configuration."""
        errors = []

        # Validate required fields
        if not config.version:
            errors.append("Version is required")

        if not config.database.host:
            errors.append("Database host is required")

        if not config.security.jwt_secret:
            errors.append("JWT secret is required")

        # Validate numeric ranges
        if config.database.port <= 0 or config.database.port > 65535:
            errors.append("Database port must be between 1 and 65535")

        if config.security.jwt_expiry <= 0:
            errors.append("JWT expiry must be positive")

        if config.agency.max_agents <= 0:
            errors.append("Max agents must be positive")

        # Validate URLs
        if config.security.cors_origins:
            for origin in config.security.cors_origins:
                if not origin.startswith(('http://', 'https://')):
                    errors.append(f"Invalid CORS origin: {origin}")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def save_config(self, config: EnvironmentConfig, encrypt_secrets: bool = True) -> str:
        """Save configuration to file."""
        config_data = asdict(config)

        if encrypt_secrets:
            config_data = self._encrypt_secrets(config_data)

        filename = f"config_{config.environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        filepath = self.config_dir / filename

        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        # Add to history
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'environment': config.environment,
            'filename': filename,
            'checksum': self._calculate_checksum(str(config_data))
        })

        logger.info(f"Configuration saved to {filename}")
        return str(filepath)

    def _encrypt_secrets(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in configuration."""
        sensitive_fields = [
            ('database', 'password'),
            ('redis', 'password'),
            ('security', 'jwt_secret'),
            ('security', 'encryption_key')
        ]

        encrypted_data = config_data.copy()

        for section, field in sensitive_fields:
            if section in encrypted_data and field in encrypted_data[section]:
                value = encrypted_data[section][field]
                if value:
                    encrypted_value = self.cipher.encrypt(value.encode()).decode()
                    encrypted_data[section][field] = f"ENC:{encrypted_value}"

        return encrypted_data

    def _decrypt_secrets(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in configuration."""
        decrypted_data = config_data.copy()

        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                return {k: decrypt_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('ENC:'):
                encrypted_value = obj[4:]  # Remove 'ENC:' prefix
                return self.cipher.decrypt(encrypted_value.encode()).decode()
            else:
                return obj

        return decrypt_recursive(decrypted_data)

    def load_config(self, filepath: str) -> EnvironmentConfig:
        """Load configuration from file."""
        config_path = Path(filepath)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Decrypt secrets if needed
        config_data = self._decrypt_secrets(config_data)

        # Create configuration object
        config = EnvironmentConfig(**config_data)

        # Cache configuration
        self.config_cache[config.environment] = config

        logger.info(f"Configuration loaded from {filepath}")
        return config

    def generate_kubernetes_manifests(self, config: EnvironmentConfig) -> Dict[str, str]:
        """Generate Kubernetes manifests from configuration."""
        manifests = {}

        # Generate ConfigMap
        config_map_data = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f'agency-swarm-config-{config.environment}',
                'namespace': config.kubernetes_namespace
            },
            'data': {
                'environment.yaml': yaml.dump(asdict(config), default_flow_style=False)
            }
        }
        manifests['configmap'] = yaml.dump(config_map_data, default_flow_style=False)

        # Generate Secret
        secret_data = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f'agency-swarm-secrets-{config.environment}',
                'namespace': config.kubernetes_namespace
            },
            'type': 'Opaque',
            'data': {
                'database-password': base64.b64encode(config.database.password.encode()).decode(),
                'redis-password': base64.b64encode(config.redis.password.encode()).decode(),
                'jwt-secret': base64.b64encode(config.security.jwt_secret.encode()).decode(),
                'encryption-key': base64.b64encode(config.security.encryption_key.encode()).decode()
            }
        }
        manifests['secret'] = yaml.dump(secret_data, default_flow_style=False)

        return manifests

    def apply_kubernetes_config(self, config: EnvironmentConfig) -> None:
        """Apply configuration to Kubernetes cluster."""
        manifests = self.generate_kubernetes_manifests(config)

        for manifest_type, manifest_content in manifests.items():
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(manifest_content)
                temp_path = f.name

            try:
                # Apply to Kubernetes
                subprocess.run(['kubectl', 'apply', '-f', temp_path], check=True)
                logger.info(f"Applied {manifest_type} to Kubernetes")

            finally:
                os.unlink(temp_path)

    def update_config(
        self,
        environment: str,
        updates: Dict[str, Any],
        dry_run: bool = False
    ) -> EnvironmentConfig:
        """Update configuration for environment."""
        logger.info(f"Updating configuration for {environment}")

        # Load current configuration
        current_config = self.config_cache.get(environment)
        if not current_config:
            raise ValueError(f"No configuration found for environment: {environment}")

        # Create backup
        backup_path = self.save_config(current_config, encrypt_secrets=False)

        # Apply updates
        current_dict = asdict(current_config)
        self._deep_merge(current_dict, updates)

        # Create new configuration
        new_config = EnvironmentConfig(**current_dict)

        # Validate updated configuration
        self._validate_config(new_config)

        if not dry_run:
            # Save new configuration
            self.save_config(new_config)
            self.config_cache[environment] = new_config

            # Apply to Kubernetes if needed
            self.apply_kubernetes_config(new_config)

            logger.info(f"Configuration updated for {environment}")
        else:
            logger.info("Dry run: configuration not actually updated")

        return new_config

    def rollback_config(self, environment: str, version: Optional[str] = None) -> EnvironmentConfig:
        """Rollback configuration to previous version."""
        logger.info(f"Rolling back configuration for {environment}")

        # Find previous version in history
        previous_configs = [
            entry for entry in self.config_history
            if entry['environment'] == environment
        ]

        if not previous_configs:
            raise ValueError(f"No previous configuration found for {environment}")

        # Load previous configuration
        previous_config_path = self.config_dir / previous_configs[-1]['filename']
        previous_config = self.load_config(str(previous_config_path))

        # Apply rollback
        self.config_cache[environment] = previous_config
        self.apply_kubernetes_config(previous_config)

        logger.info(f"Configuration rolled back for {environment}")
        return previous_config

    def validate_config_compliance(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Validate configuration against compliance requirements."""
        compliance_report = {
            'passed': True,
            'checks': [],
            'recommendations': []
        }

        # Security compliance checks
        security_checks = [
            ('password_strength', len(config.database.password) >= 16),
            ('jwt_key_strength', len(config.security.jwt_secret) >= 32),
            ('rate_limit_reasonable', config.security.rate_limit > 0),
            ('ssl_enabled', config.security.ssl_enabled),
            ('cors_restricted', len(config.security.cors_origins) <= 10)
        ]

        for check_name, passed in security_checks:
            status = 'PASS' if passed else 'FAIL'
            compliance_report['checks'].append({
                'category': 'security',
                'check': check_name,
                'status': status
            })

            if not passed:
                compliance_report['passed'] = False
                compliance_report['recommendations'].append(f"Fix security issue: {check_name}")

        # Performance compliance checks
        performance_checks = [
            ('database_pool_size', config.database.pool_size >= 10),
            ('redis_max_connections', config.redis.max_connections >= 20),
            ('agency_limits_reasonable', config.agency.max_agents <= 200),
            ('monitoring_enabled', config.monitoring.enabled)
        ]

        for check_name, passed in performance_checks:
            status = 'PASS' if passed else 'FAIL'
            compliance_report['checks'].append({
                'category': 'performance',
                'check': check_name,
                'status': status
            })

            if not passed:
                compliance_report['recommendations'].append(f"Optimize performance: {check_name}")

        return compliance_report

    def _calculate_checksum(self, data: str) -> str:
        """Calculate checksum of configuration data."""
        return hashlib.sha256(data.encode()).hexdigest()

    def get_config_history(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get configuration history."""
        if environment:
            return [entry for entry in self.config_history if entry['environment'] == environment]
        return self.config_history

    def cleanup_old_configs(self, keep_count: int = 10) -> None:
        """Clean up old configuration files."""
        config_files = sorted(self.config_dir.glob("config_*.yaml"), key=lambda x: x.stat().st_mtime)

        if len(config_files) > keep_count:
            for old_file in config_files[:-keep_count]:
                old_file.unlink()
                logger.info(f"Cleaned up old configuration file: {old_file}")

# CLI Interface
def main():
    """Command line interface for configuration management."""
    import argparse

    parser = argparse.ArgumentParser(description='Agency Swarm Configuration Management')
    parser.add_argument('action', choices=['create', 'update', 'load', 'validate', 'rollback', 'manifests'])
    parser.add_argument('--environment', '-e', choices=['development', 'staging', 'production'],
                        required=True, help='Environment name')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without actual changes')
    parser.add_argument('--update-file', help='JSON file with configuration updates')

    args = parser.parse_args()

    try:
        manager = ConfigurationManager()

        if args.action == 'create':
            config = manager.create_environment_config(args.environment)
            if args.output:
                manager.save_config(config)
                print(f"Configuration saved to {args.output}")
            else:
                print(yaml.dump(asdict(config), default_flow_style=False))

        elif args.action == 'update':
            if not args.update_file:
                print("Error: Update file is required for update action")
                sys.exit(1)

            with open(args.update_file, 'r') as f:
                updates = json.load(f)

            config = manager.update_config(args.environment, updates, args.dry_run)
            print(f"Configuration updated for {args.environment}")

        elif args.action == 'load':
            if not args.config:
                print("Error: Configuration file is required for load action")
                sys.exit(1)

            config = manager.load_config(args.config)
            print(f"Configuration loaded for {config.environment}")

        elif args.action == 'validate':
            config = manager.config_cache.get(args.environment)
            if not config:
                print("Error: No configuration found for environment")
                sys.exit(1)

            report = manager.validate_config_compliance(config)
            print(f"Compliance Report: {'PASS' if report['passed'] else 'FAIL'}")
            for check in report['checks']:
                print(f"  {check['category']}.{check['check']}: {check['status']}")

            if report['recommendations']:
                print("\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  - {rec}")

        elif args.action == 'rollback':
            config = manager.rollback_config(args.environment)
            print(f"Configuration rolled back for {args.environment}")

        elif args.action == 'manifests':
            config = manager.config_cache.get(args.environment)
            if not config:
                print("Error: No configuration found for environment")
                sys.exit(1)

            manifests = manager.generate_kubernetes_manifests(config)
            for manifest_type, manifest_content in manifests.items():
                if args.output:
                    filename = f"{args.output}_{manifest_type}.yaml"
                    with open(filename, 'w') as f:
                        f.write(manifest_content)
                    print(f"Manifest saved to {filename}")
                else:
                    print(f"--- {manifest_type.upper()} ---")
                    print(manifest_content)

    except Exception as e:
        logger.error(f"Configuration management failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()