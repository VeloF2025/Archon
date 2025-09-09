"""
Template Engine with Variable Substitution

Handles template processing including:
- Variable substitution with Jinja2-like syntax
- File generation and directory creation
- Hook execution
- Project initialization workflows
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import yaml
import json

from .template_models import (
    Template, TemplateFile, TemplateGenerationRequest, 
    TemplateGenerationResult, TemplateHook, VariableType
)
from .template_validator import TemplateValidator
import logging

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Processes templates and generates projects with variable substitution."""
    
    def __init__(self):
        self.validator = TemplateValidator()
    
    async def generate_project(
        self, 
        template: Template, 
        request: TemplateGenerationRequest,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> TemplateGenerationResult:
        """Generate a project from a template."""
        start_time = time.time()
        result = TemplateGenerationResult(
            success=False,
            template_id=template.id,
            output_directory=request.output_directory
        )
        
        try:
            logger.info(f"Starting template generation | template={template.id} | output={request.output_directory}")
            
            if progress_callback:
                await progress_callback("Validating template...", 10)
            
            # Validate template
            validation_result = self.validator.validate_template(template)
            if not validation_result.valid:
                result.errors.extend(validation_result.errors)
                result.warnings.extend(validation_result.warnings)
                return result
            
            if progress_callback:
                await progress_callback("Preparing variables...", 20)
            
            # Process and validate variables
            processed_variables = self._process_variables(template.variables, request.variables)
            validation_errors = self._validate_variable_values(template.variables, processed_variables)
            if validation_errors:
                result.errors.extend(validation_errors)
                return result
            
            # Check output directory
            output_path = Path(request.output_directory)
            if output_path.exists() and not request.overwrite_existing:
                result.errors.append(f"Output directory already exists: {request.output_directory}")
                return result
            
            if request.dry_run:
                logger.info("Dry run mode - no files will be created")
                result.success = True
                result.files_created = [self._substitute_variables(f.path, processed_variables) for f in template.files]
                result.directories_created = [self._substitute_variables(d, processed_variables) for d in template.directory_structure]
                return result
            
            if progress_callback:
                await progress_callback("Creating directory structure...", 30)
            
            # Create directory structure
            created_dirs = await self._create_directories(template, processed_variables, output_path)
            result.directories_created = created_dirs
            
            if progress_callback:
                await progress_callback("Executing pre-generation hooks...", 40)
            
            # Execute pre-generation hooks
            pre_hook_results = await self._execute_hooks(
                template.pre_generate_hooks, 
                processed_variables, 
                output_path
            )
            result.hooks_executed.extend(pre_hook_results.get('executed', []))
            result.errors.extend(pre_hook_results.get('errors', []))
            result.warnings.extend(pre_hook_results.get('warnings', []))
            
            if progress_callback:
                await progress_callback("Generating files...", 60)
            
            # Generate files
            created_files = await self._generate_files(template.files, processed_variables, output_path)
            result.files_created = created_files
            
            if progress_callback:
                await progress_callback("Executing post-generation hooks...", 80)
            
            # Execute post-generation hooks
            post_hook_results = await self._execute_hooks(
                template.post_generate_hooks, 
                processed_variables, 
                output_path
            )
            result.hooks_executed.extend(post_hook_results.get('executed', []))
            result.errors.extend(post_hook_results.get('errors', []))
            result.warnings.extend(post_hook_results.get('warnings', []))
            
            if progress_callback:
                await progress_callback("Finalizing project...", 90)
            
            # Create .archon-project.yaml metadata file
            await self._create_project_metadata(template, processed_variables, output_path)
            
            result.success = len(result.errors) == 0
            result.generation_time = time.time() - start_time
            
            if progress_callback:
                await progress_callback("Project generation completed!", 100)
            
            logger.info(f"Template generation completed | success={result.success} | time={result.generation_time:.2f}s | files={len(result.files_created)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Template generation failed | template={template.id} | error={str(e)}")
            result.errors.append(f"Generation failed: {str(e)}")
            result.generation_time = time.time() - start_time
            return result
    
    def _process_variables(self, template_vars: List, provided_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Process and merge template variables with provided values."""
        processed = {}
        
        for var in template_vars:
            if var.name in provided_vars:
                processed[var.name] = provided_vars[var.name]
            elif var.default is not None:
                processed[var.name] = var.default
            elif var.required:
                # This will be caught by validation
                continue
            else:
                processed[var.name] = ""
        
        return processed
    
    def _validate_variable_values(self, template_vars: List, values: Dict[str, Any]) -> List[str]:
        """Validate provided variable values against template definitions."""
        errors = []
        
        for var in template_vars:
            if var.required and var.name not in values:
                errors.append(f"Required variable missing: {var.name}")
                continue
            
            if var.name not in values:
                continue
            
            value = values[var.name]
            
            # Type validation
            if var.type == VariableType.NUMBER and not isinstance(value, (int, float)):
                errors.append(f"Variable {var.name} must be a number, got: {type(value).__name__}")
            elif var.type == VariableType.BOOLEAN and not isinstance(value, bool):
                errors.append(f"Variable {var.name} must be a boolean, got: {type(value).__name__}")
            elif var.type == VariableType.ARRAY and not isinstance(value, list):
                errors.append(f"Variable {var.name} must be an array, got: {type(value).__name__}")
            elif var.type == VariableType.OBJECT and not isinstance(value, dict):
                errors.append(f"Variable {var.name} must be an object, got: {type(value).__name__}")
            
            # Options validation for SELECT/MULTISELECT
            if var.type == VariableType.SELECT and var.options:
                if value not in var.options:
                    errors.append(f"Variable {var.name} must be one of: {', '.join(var.options)}")
            elif var.type == VariableType.MULTISELECT and var.options:
                if not isinstance(value, list) or not all(v in var.options for v in value):
                    errors.append(f"Variable {var.name} must be a list of values from: {', '.join(var.options)}")
            
            # Regex validation
            if var.validation_regex and isinstance(value, str):
                if not re.match(var.validation_regex, value):
                    errors.append(f"Variable {var.name} does not match required format: {var.validation_regex}")
        
        return errors
    
    def _substitute_variables(self, content: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in content using {{variable}} syntax."""
        def replace_var(match):
            var_name = match.group(1).strip()
            if var_name in variables:
                value = variables[var_name]
                # Handle different types
                if isinstance(value, bool):
                    return str(value).lower()
                elif isinstance(value, (list, dict)):
                    return json.dumps(value)
                else:
                    return str(value)
            else:
                # Return original if variable not found
                return match.group(0)
        
        # Replace {{variable}} patterns
        pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
        return re.sub(pattern, replace_var, content)
    
    async def _create_directories(self, template: Template, variables: Dict[str, Any], base_path: Path) -> List[str]:
        """Create directory structure for the template."""
        created_dirs = []
        
        # Create base directory
        base_path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(base_path))
        
        # Create template-defined directories
        for directory in template.directory_structure:
            dir_path = self._substitute_variables(directory, variables)
            full_path = base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(full_path))
        
        # Create directories for files that don't exist
        for file in template.files:
            file_path = self._substitute_variables(file.path, variables)
            full_file_path = base_path / file_path
            dir_path = full_file_path.parent
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                if str(dir_path) not in created_dirs:
                    created_dirs.append(str(dir_path))
        
        return created_dirs
    
    async def _generate_files(self, files: List[TemplateFile], variables: Dict[str, Any], base_path: Path) -> List[str]:
        """Generate files from templates."""
        created_files = []
        
        for file in files:
            try:
                # Substitute variables in path and content
                file_path = self._substitute_variables(file.path, variables)
                content = self._substitute_variables(file.content, variables)
                
                full_path = base_path / file_path
                
                # Check if we should overwrite
                if full_path.exists() and not file.overwrite:
                    logger.warning(f"Skipping existing file (overwrite=False): {full_path}")
                    continue
                
                # Create parent directories if needed
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file
                if file.is_binary:
                    # For binary files, content should be base64 encoded
                    import base64
                    binary_content = base64.b64decode(content)
                    full_path.write_bytes(binary_content)
                else:
                    full_path.write_text(content, encoding='utf-8')
                
                # Set executable permission if needed
                if file.executable:
                    full_path.chmod(0o755)
                
                created_files.append(str(full_path))
                logger.info(f"Created file: {full_path}")
                
            except Exception as e:
                logger.error(f"Failed to create file {file.path}: {e}")
                raise
        
        return created_files
    
    async def _execute_hooks(self, hooks: List[TemplateHook], variables: Dict[str, Any], base_path: Path) -> Dict[str, List[str]]:
        """Execute template hooks."""
        result = {
            'executed': [],
            'errors': [],
            'warnings': []
        }
        
        for hook in hooks:
            try:
                # Substitute variables in command
                command = self._substitute_variables(hook.command, variables)
                
                # Determine working directory
                if hook.working_directory:
                    work_dir = base_path / self._substitute_variables(hook.working_directory, variables)
                else:
                    work_dir = base_path
                
                logger.info(f"Executing hook '{hook.name}': {command}")
                
                # Execute command
                process = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(work_dir),
                    timeout=hook.timeout,
                    capture_output=True,
                    text=True
                )
                
                if process.returncode == 0:
                    result['executed'].append(f"{hook.name}: {command}")
                    if process.stdout:
                        logger.info(f"Hook '{hook.name}' output: {process.stdout}")
                else:
                    error_msg = f"Hook '{hook.name}' failed (exit {process.returncode}): {process.stderr or process.stdout}"
                    
                    if hook.failure_mode == "fail":
                        result['errors'].append(error_msg)
                    elif hook.failure_mode == "warn":
                        result['warnings'].append(error_msg)
                    else:  # continue
                        logger.warning(error_msg)
                
            except subprocess.TimeoutExpired:
                error_msg = f"Hook '{hook.name}' timed out after {hook.timeout} seconds"
                if hook.failure_mode == "fail":
                    result['errors'].append(error_msg)
                else:
                    result['warnings'].append(error_msg)
                    
            except Exception as e:
                error_msg = f"Hook '{hook.name}' execution error: {str(e)}"
                if hook.failure_mode == "fail":
                    result['errors'].append(error_msg)
                else:
                    result['warnings'].append(error_msg)
        
        return result
    
    async def _create_project_metadata(self, template: Template, variables: Dict[str, Any], base_path: Path):
        """Create .archon-project.yaml metadata file."""
        metadata = {
            'archon_project': {
                'version': '1.0',
                'generated_from_template': {
                    'template_id': template.id,
                    'template_name': template.metadata.name,
                    'template_version': template.metadata.version,
                    'generated_at': time.time(),
                    'variables_used': variables
                },
                'template_metadata': {
                    'name': template.metadata.name,
                    'description': template.metadata.description,
                    'author': template.metadata.author,
                    'type': template.metadata.type,
                    'category': template.metadata.category,
                    'tags': template.metadata.tags
                }
            }
        }
        
        metadata_path = base_path / '.archon-project.yaml'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created project metadata: {metadata_path}")
    
    def extract_variables_from_content(self, content: str) -> List[str]:
        """Extract variable names from content."""
        pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
        return list(set(re.findall(pattern, content)))
    
    def validate_template_syntax(self, content: str) -> List[str]:
        """Validate template syntax and return any errors."""
        errors = []
        
        # Check for unmatched braces
        brace_pattern = r'\{\{[^}]*\}\}'
        matches = re.findall(brace_pattern, content)
        
        for match in matches:
            # Check if it's a valid variable reference
            var_pattern = r'^\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\}$'
            if not re.match(var_pattern, match):
                errors.append(f"Invalid variable syntax: {match}")
        
        return errors