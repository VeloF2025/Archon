"""
Dynamic Template Management System

This module provides comprehensive template management for accelerating project setup:
- Template registry with validation
- Variable substitution engine
- Template marketplace integration
- Battle-tested project patterns
"""

from .template_registry import TemplateRegistry
from .template_engine import TemplateEngine
from .template_validator import TemplateValidator
from .template_models import (
    Template, TemplateVariable, TemplateMetadata, TemplateSearchRequest, 
    TemplateGenerationRequest, TemplateType, TemplateCategory
)

__all__ = [
    'TemplateRegistry',
    'TemplateEngine', 
    'TemplateValidator',
    'Template',
    'TemplateVariable',
    'TemplateMetadata',
    'TemplateSearchRequest',
    'TemplateGenerationRequest', 
    'TemplateType',
    'TemplateCategory'
]