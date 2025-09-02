#!/usr/bin/env python3
"""
PRP Manager for Archon+ Agent System
Manages Product Requirements Prompt templates with variable interpolation
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from string import Template
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TemplateComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PRPTemplate:
    """PRP Template representation"""
    name: str
    file_path: str
    description: str
    variables: List[str]
    output_formats: List[str]
    complexity: TemplateComplexity
    estimated_time: str
    content: str = ""
    
class PRPManager:
    """
    Manages PRP (Product Requirements Prompt) templates for specialized agents
    Handles template loading, variable interpolation, and validation
    """
    
    def __init__(self, templates_path: str = "python/src/agents/prompts/prp"):
        self.templates_path = Path(templates_path)
        self.registry_path = self.templates_path / "template_registry.json"
        
        # Template storage
        self.templates: Dict[str, PRPTemplate] = {}
        self.registry: Dict[str, Any] = {}
        
        # Load registry and templates
        self._load_registry()
        self._load_templates()
        
        logger.info(f"PRPManager initialized with {len(self.templates)} templates")
    
    def _load_registry(self):
        """Load template registry configuration"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded template registry with {len(self.registry.get('templates', {}))} entries")
            else:
                logger.error(f"Registry file not found: {self.registry_path}")
                self.registry = {"templates": {}, "template_categories": {}}
                
        except Exception as e:
            logger.error(f"Failed to load template registry: {e}")
            self.registry = {"templates": {}, "template_categories": {}}
    
    def _load_templates(self):
        """Load all PRP template files"""
        for template_name, template_config in self.registry.get("templates", {}).items():
            try:
                template_file = self.templates_path / template_config["file"]
                
                if template_file.exists():
                    with open(template_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    template = PRPTemplate(
                        name=template_name,
                        file_path=str(template_file),
                        description=template_config["description"],
                        variables=template_config["variables"],
                        output_formats=template_config["output_formats"],
                        complexity=TemplateComplexity(template_config["complexity"]),
                        estimated_time=template_config["estimated_time"],
                        content=content
                    )
                    
                    self.templates[template_name] = template
                    logger.debug(f"Loaded template: {template_name}")
                    
                else:
                    logger.warning(f"Template file not found: {template_file}")
                    
            except Exception as e:
                logger.error(f"Failed to load template {template_name}: {e}")
        
        logger.info(f"Successfully loaded {len(self.templates)} PRP templates")
    
    def get_template(self, template_name: str) -> Optional[PRPTemplate]:
        """Get a specific template by name"""
        return self.templates.get(template_name)
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """List available templates, optionally filtered by category"""
        if category:
            category_templates = self.registry.get("template_categories", {}).get(category, [])
            return [name for name in category_templates if name in self.templates]
        
        return list(self.templates.keys())
    
    def get_categories(self) -> List[str]:
        """Get list of template categories"""
        return list(self.registry.get("template_categories", {}).keys())
    
    def get_templates_by_category(self, category: str) -> List[PRPTemplate]:
        """Get all templates in a specific category"""
        template_names = self.registry.get("template_categories", {}).get(category, [])
        return [self.templates[name] for name in template_names if name in self.templates]
    
    def interpolate_template(self, 
                           template_name: str, 
                           variables: Dict[str, Any],
                           validate_required: bool = True) -> Optional[str]:
        """
        Interpolate template with provided variables
        
        Args:
            template_name: Name of the template to use
            variables: Dictionary of variable values
            validate_required: Whether to validate required variables
        
        Returns:
            Interpolated template content or None if failed
        """
        template = self.get_template(template_name)
        if not template:
            logger.error(f"Template not found: {template_name}")
            return None
        
        # Validate required variables
        if validate_required:
            missing_vars = self._validate_variables(template, variables)
            if missing_vars:
                logger.error(f"Missing required variables for {template_name}: {missing_vars}")
                return None
        
        try:
            # Use string.Template for safe interpolation
            template_obj = Template(template.content)
            
            # Convert all variables to strings and handle missing ones
            safe_variables = {}
            for var in template.variables:
                if var in variables:
                    value = variables[var]
                    if isinstance(value, (list, dict)):
                        # Convert complex types to formatted strings
                        safe_variables[var] = self._format_complex_variable(value)
                    else:
                        safe_variables[var] = str(value)
                else:
                    # Provide placeholder for missing optional variables
                    safe_variables[var] = f"[{var}]"
            
            # Perform substitution
            interpolated = template_obj.safe_substitute(safe_variables)
            
            logger.info(f"Successfully interpolated template: {template_name}")
            return interpolated
            
        except Exception as e:
            logger.error(f"Failed to interpolate template {template_name}: {e}")
            return None
    
    def _validate_variables(self, template: PRPTemplate, variables: Dict[str, Any]) -> List[str]:
        """Validate that required variables are provided"""
        variable_defs = self.registry.get("variable_definitions", {})
        missing_vars = []
        
        for var in template.variables:
            var_def = variable_defs.get(var, {})
            is_required = var_def.get("required", False)
            
            if is_required and var not in variables:
                missing_vars.append(var)
        
        return missing_vars
    
    def _format_complex_variable(self, value: Any) -> str:
        """Format complex variables (lists, dicts) for template interpolation"""
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                # Simple string list
                return ", ".join(value)
            else:
                # Complex list - format as bullet points
                return "\n".join(f"- {str(item)}" for item in value)
        
        elif isinstance(value, dict):
            # Format dict as key-value pairs
            return "\n".join(f"- **{k}**: {v}" for k, v in value.items())
        
        else:
            return str(value)
    
    def generate_agent_prompt(self,
                            agent_role: str,
                            task_description: str,
                            context_variables: Dict[str, Any],
                            custom_variables: Dict[str, Any] = None) -> Optional[str]:
        """
        Generate a complete prompt for a specific agent role
        
        Args:
            agent_role: The role of the agent (must match template name)
            task_description: Description of the task to perform
            context_variables: Context variables (project, files, etc.)
            custom_variables: Additional custom variables
        
        Returns:
            Complete PRP prompt or None if failed
        """
        template = self.get_template(agent_role)
        if not template:
            logger.error(f"No template found for agent role: {agent_role}")
            return None
        
        # Combine all variables
        all_variables = {
            "requirements": task_description,
            **context_variables
        }
        
        if custom_variables:
            all_variables.update(custom_variables)
        
        # Interpolate template
        return self.interpolate_template(agent_role, all_variables)
    
    def get_template_dependencies(self, template_name: str) -> List[str]:
        """Get templates that this template depends on"""
        dependencies = self.registry.get("template_relationships", {}).get("dependencies", {})
        return dependencies.get(template_name, [])
    
    def get_complementary_templates(self, template_name: str) -> List[str]:
        """Get templates that work well with this template"""
        complementary = self.registry.get("template_relationships", {}).get("complementary", {})
        return complementary.get(template_name, [])
    
    def validate_template_content(self, template_name: str) -> Dict[str, Any]:
        """
        Validate template content against registry rules
        
        Returns:
            Validation results with issues and suggestions
        """
        template = self.get_template(template_name)
        if not template:
            return {"valid": False, "error": "Template not found"}
        
        validation_rules = self.registry.get("template_validation_rules", {})
        results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check required sections
        required_sections = validation_rules.get("required_sections", [])
        for section in required_sections:
            if f"## {section}" not in template.content and f"# {section}" not in template.content:
                results["issues"].append(f"Missing required section: {section}")
                results["valid"] = False
        
        # Check variable interpolation markers
        if validation_rules.get("variable_interpolation", True):
            for var in template.variables:
                if f"{{{var}}}" not in template.content:
                    results["warnings"].append(f"Variable '{var}' not found in template content")
        
        # Check for example code
        if validation_rules.get("example_code_validation", True):
            code_blocks = re.findall(r'```[\w]*\n.*?\n```', template.content, re.DOTALL)
            if not code_blocks:
                results["warnings"].append("No code examples found in template")
        
        return results
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get template usage statistics from registry"""
        return self.registry.get("usage_statistics", {})
    
    def suggest_templates_for_workflow(self, primary_template: str) -> List[Dict[str, str]]:
        """
        Suggest additional templates for a complete workflow
        
        Returns:
            List of suggested templates with reasons
        """
        suggestions = []
        
        # Get complementary templates
        complementary = self.get_complementary_templates(primary_template)
        for template_name in complementary:
            if template_name in self.templates:
                suggestions.append({
                    "template": template_name,
                    "reason": "Complementary functionality",
                    "description": self.templates[template_name].description
                })
        
        # Get category peers (templates in same category that aren't already included)
        primary_template_obj = self.get_template(primary_template)
        if primary_template_obj:
            for category, template_names in self.registry.get("template_categories", {}).items():
                if primary_template in template_names:
                    for peer_name in template_names:
                        if (peer_name != primary_template and 
                            peer_name in self.templates and
                            peer_name not in complementary):
                            suggestions.append({
                                "template": peer_name,
                                "reason": f"Same category: {category}",
                                "description": self.templates[peer_name].description
                            })
        
        return suggestions
    
    def export_template_summary(self) -> Dict[str, Any]:
        """Export summary of all templates for external use"""
        summary = {
            "total_templates": len(self.templates),
            "categories": self.get_categories(),
            "templates": {}
        }
        
        for name, template in self.templates.items():
            summary["templates"][name] = {
                "description": template.description,
                "complexity": template.complexity.value,
                "estimated_time": template.estimated_time,
                "variables_count": len(template.variables),
                "output_formats": template.output_formats
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    def main():
        # Initialize PRP Manager
        prp_manager = PRPManager()
        
        # List available templates
        print("Available Templates:")
        for template_name in prp_manager.list_templates():
            template = prp_manager.get_template(template_name)
            print(f"- {template_name}: {template.description}")
        
        # Generate a prompt for Python backend development
        context = {
            "project_name": "User Management API",
            "file_paths": ["app/models/user.py", "app/api/users.py"],
            "dependencies": ["fastapi", "sqlalchemy", "pydantic"],
            "database_type": "PostgreSQL"
        }
        
        task_description = """
        Implement a complete user management system with the following features:
        1. User registration with email validation
        2. User authentication with JWT tokens
        3. User profile management (CRUD operations)
        4. Password hashing with bcrypt
        5. Input validation and error handling
        6. Comprehensive test coverage
        """
        
        prompt = prp_manager.generate_agent_prompt(
            agent_role="python_backend_coder",
            task_description=task_description,
            context_variables=context
        )
        
        if prompt:
            print(f"\nGenerated prompt for python_backend_coder:")
            print("=" * 60)
            print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        
        # Get workflow suggestions
        suggestions = prp_manager.suggest_templates_for_workflow("python_backend_coder")
        print(f"\nSuggested complementary templates:")
        for suggestion in suggestions[:3]:  # Show top 3
            print(f"- {suggestion['template']}: {suggestion['reason']}")
        
        # Export summary
        summary = prp_manager.export_template_summary()
        print(f"\nTemplate Summary: {summary['total_templates']} templates in {len(summary['categories'])} categories")
    
    main()