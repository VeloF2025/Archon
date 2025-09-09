"""
Template Validation System

Provides comprehensive validation for templates including:
- Schema validation
- Variable dependency checking
- File path validation
- Security checks
"""

import re
import os
import yaml
from typing import List, Dict, Set, Any
from pathlib import Path

from .template_models import (
    Template, TemplateVariable, TemplateFile, TemplateValidationResult,
    VariableType
)
import logging

logger = logging.getLogger(__name__)


class TemplateValidator:
    """Validates templates for correctness and security."""
    
    # Security: Dangerous file patterns
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'/etc/',  # System directories
        r'/var/',
        r'/usr/',
        r'~/\.',   # Hidden files in home
        r'\.ssh/',
        r'\.env$',  # Environment files (but allow .env.example, .env.template, etc.)
    ]
    
    # Security: Dangerous commands in hooks
    DANGEROUS_COMMANDS = [
        'rm -rf',
        'sudo',
        'chmod 777',
        'eval',
        'exec',
        '> /dev/null',  # Suspicious redirection
        'curl | bash',  # Pipe to shell
        'wget | bash',
    ]
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
    
    def validate_template(self, template: Template) -> TemplateValidationResult:
        """Validate a complete template."""
        self._reset()
        
        try:
            # Basic validation
            self._validate_metadata(template)
            self._validate_variables(template.variables)
            self._validate_files(template.files, template.variables)
            self._validate_hooks(template.pre_generate_hooks + template.post_generate_hooks)
            self._validate_directory_structure(template.directory_structure)
            self._validate_security(template)
            
            # Cross-validation
            self._validate_variable_dependencies(template)
            
            return TemplateValidationResult(
                valid=len(self.errors) == 0,
                template_id=template.id,
                errors=self.errors,
                warnings=self.warnings,
                suggestions=self.suggestions
            )
            
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            self.errors.append(f"Validation error: {str(e)}")
            
            return TemplateValidationResult(
                valid=False,
                template_id=template.id,
                errors=self.errors,
                warnings=self.warnings,
                suggestions=self.suggestions
            )
    
    def validate_template_file(self, file_path: str) -> TemplateValidationResult:
        """Validate a template from a .archon-template.yaml file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            
            # Convert to Template model for validation
            template = Template(**template_data)
            return self.validate_template(template)
            
        except FileNotFoundError:
            return TemplateValidationResult(
                valid=False,
                template_id="unknown",
                errors=[f"Template file not found: {file_path}"]
            )
        except yaml.YAMLError as e:
            return TemplateValidationResult(
                valid=False,
                template_id="unknown",
                errors=[f"Invalid YAML syntax: {str(e)}"]
            )
        except Exception as e:
            return TemplateValidationResult(
                valid=False,
                template_id="unknown",
                errors=[f"Template validation failed: {str(e)}"]
            )
    
    def _reset(self):
        """Reset validation state."""
        self.errors.clear()
        self.warnings.clear()
        self.suggestions.clear()
    
    def _validate_metadata(self, template: Template):
        """Validate template metadata."""
        metadata = template.metadata
        
        # Name validation
        if not metadata.name or len(metadata.name.strip()) < 3:
            self.errors.append("Template name must be at least 3 characters long")
        
        # Description validation
        if not metadata.description or len(metadata.description.strip()) < 10:
            self.warnings.append("Template description should be at least 10 characters long")
        
        # Version validation
        version_pattern = r'^\d+\.\d+\.\d+(-[\w\d\-]+)?$'
        if not re.match(version_pattern, metadata.version):
            self.errors.append(f"Invalid version format: {metadata.version}. Expected: X.Y.Z or X.Y.Z-suffix")
        
        # Author validation
        if not metadata.author or len(metadata.author.strip()) < 2:
            self.warnings.append("Template should have an author specified")
        
        # Rating validation
        if metadata.rating < 0 or metadata.rating > 5:
            self.errors.append(f"Rating must be between 0 and 5, got: {metadata.rating}")
    
    def _validate_variables(self, variables: List[TemplateVariable]):
        """Validate template variables."""
        variable_names = set()
        
        for var in variables:
            # Check for duplicate names
            if var.name in variable_names:
                self.errors.append(f"Duplicate variable name: {var.name}")
            variable_names.add(var.name)
            
            # Validate variable name format
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var.name):
                self.errors.append(f"Invalid variable name format: {var.name}")
            
            # Validate SELECT/MULTISELECT options
            if var.type in [VariableType.SELECT, VariableType.MULTISELECT]:
                if not var.options or len(var.options) < 2:
                    self.errors.append(f"Variable {var.name} requires at least 2 options")
            
            # Validate regex if provided
            if var.validation_regex:
                try:
                    re.compile(var.validation_regex)
                except re.error as e:
                    self.errors.append(f"Invalid regex for variable {var.name}: {str(e)}")
            
            # Check default value compatibility
            if var.default is not None:
                if var.type == VariableType.BOOLEAN and not isinstance(var.default, bool):
                    self.warnings.append(f"Variable {var.name} has non-boolean default for boolean type")
                elif var.type == VariableType.NUMBER and not isinstance(var.default, (int, float)):
                    self.warnings.append(f"Variable {var.name} has non-numeric default for number type")
    
    def _validate_files(self, files: List[TemplateFile], variables: List[TemplateVariable]):
        """Validate template files."""
        file_paths = set()
        variable_names = {var.name for var in variables}
        
        for file in files:
            # Check for duplicate paths
            if file.path in file_paths:
                self.errors.append(f"Duplicate file path: {file.path}")
            file_paths.add(file.path)
            
            # Validate path format
            if file.path.startswith('/') or '..' in file.path:
                self.errors.append(f"Invalid file path (absolute or contains '..'): {file.path}")
            
            # Check for dangerous patterns
            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, file.path, re.IGNORECASE):
                    self.errors.append(f"Potentially dangerous file path: {file.path}")
                    break
            
            # Validate variables in file content
            self._validate_file_variables(file, variable_names)
            
            # Check file extension consistency
            if file.path.endswith(('.exe', '.dll', '.so', '.dylib')) and not file.is_binary:
                self.warnings.append(f"File {file.path} appears binary but is_binary=False")
    
    def _validate_file_variables(self, file: TemplateFile, variable_names: Set[str]):
        """Validate variables used in file content."""
        # Find all variable references in content
        variable_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
        found_vars = set(re.findall(variable_pattern, file.content))
        
        # Check for undefined variables
        undefined_vars = found_vars - variable_names
        if undefined_vars:
            self.errors.append(f"File {file.path} references undefined variables: {', '.join(undefined_vars)}")
        
        # Check for malformed variable syntax
        malformed_pattern = r'\{\{[^}]*\}\}'
        malformed_vars = set(re.findall(malformed_pattern, file.content)) - {f"{{{{{var}}}}}" for var in found_vars}
        if malformed_vars:
            self.warnings.append(f"File {file.path} contains potentially malformed variables: {', '.join(malformed_vars)}")
    
    def _validate_hooks(self, hooks: List):
        """Validate template hooks."""
        for hook in hooks:
            # Check for dangerous commands
            for dangerous_cmd in self.DANGEROUS_COMMANDS:
                if dangerous_cmd.lower() in hook.command.lower():
                    self.errors.append(f"Hook '{hook.name}' contains potentially dangerous command: {dangerous_cmd}")
            
            # Validate timeout
            if hook.timeout < 1 or hook.timeout > 3600:  # 1 second to 1 hour
                self.warnings.append(f"Hook '{hook.name}' has unusual timeout: {hook.timeout}s")
            
            # Check command format
            if not hook.command.strip():
                self.errors.append(f"Hook '{hook.name}' has empty command")
    
    def _validate_directory_structure(self, directories: List[str]):
        """Validate directory structure."""
        for directory in directories:
            # Check for dangerous patterns
            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, directory, re.IGNORECASE):
                    self.errors.append(f"Potentially dangerous directory path: {directory}")
                    break
            
            # Validate path format
            if directory.startswith('/') or '..' in directory:
                self.errors.append(f"Invalid directory path (absolute or contains '..'): {directory}")
    
    def _validate_security(self, template: Template):
        """Perform security validation."""
        # Check for suspicious patterns in file content
        suspicious_patterns = [
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls  
            r'subprocess\.call',  # subprocess calls
            r'os\.system',  # os.system calls
            r'__import__',  # dynamic imports
        ]
        
        for file in template.files:
            for pattern in suspicious_patterns:
                if re.search(pattern, file.content, re.IGNORECASE):
                    self.warnings.append(f"File {file.path} contains potentially dangerous code pattern: {pattern}")
        
        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        
        for file in template.files:
            for pattern in secret_patterns:
                if re.search(pattern, file.content, re.IGNORECASE):
                    self.warnings.append(f"File {file.path} may contain hardcoded secrets")
    
    def _validate_variable_dependencies(self, template: Template):
        """Validate variable dependencies and usage."""
        # Check that all required variables are used
        variable_names = {var.name for var in template.variables if var.required}
        used_variables = set()
        
        # Find variables used in files
        for file in template.files:
            variable_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
            used_variables.update(re.findall(variable_pattern, file.content))
        
        # Find variables used in hooks
        for hook in template.pre_generate_hooks + template.post_generate_hooks:
            variable_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
            used_variables.update(re.findall(variable_pattern, hook.command))
        
        # Warn about unused required variables
        unused_required = variable_names - used_variables
        if unused_required:
            self.warnings.append(f"Required variables not used in template: {', '.join(unused_required)}")
        
        # Suggest making unused variables optional
        for var_name in unused_required:
            self.suggestions.append(f"Consider making variable '{var_name}' optional if it's not essential")