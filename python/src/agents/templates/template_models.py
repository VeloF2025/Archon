"""
Template Data Models

Defines the core data structures for the template management system.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class TemplateType(str, Enum):
    """Supported template types."""
    PROJECT = "project"
    COMPONENT = "component"
    API = "api"
    DEPLOYMENT = "deployment"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


class TemplateCategory(str, Enum):
    """Template categories for organization."""
    FRONTEND = "frontend"
    BACKEND = "backend"
    FULLSTACK = "fullstack"
    MOBILE = "mobile"
    DEVOPS = "devops"
    AI_ML = "ai_ml"
    DATA = "data"
    BLOCKCHAIN = "blockchain"
    IOT = "iot"


class VariableType(str, Enum):
    """Variable types for template substitution."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    SELECT = "select"  # Dropdown with predefined options
    MULTISELECT = "multiselect"


class TemplateVariable(BaseModel):
    """Template variable definition."""
    name: str = Field(..., description="Variable name for substitution")
    type: VariableType = Field(default=VariableType.STRING)
    description: str = Field(..., description="Human-readable description")
    default: Optional[Any] = Field(None, description="Default value")
    required: bool = Field(default=True)
    options: Optional[List[str]] = Field(None, description="Options for SELECT/MULTISELECT")
    validation_regex: Optional[str] = Field(None, description="Regex for validation")
    example: Optional[str] = Field(None, description="Example value")
    
    @validator('options')
    def validate_options(cls, v, values):
        """Validate options are provided for SELECT/MULTISELECT types."""
        if values.get('type') in [VariableType.SELECT, VariableType.MULTISELECT]:
            if not v or len(v) < 2:
                raise ValueError(f"SELECT/MULTISELECT variables must have at least 2 options")
        return v


class TemplateFile(BaseModel):
    """Template file definition."""
    path: str = Field(..., description="Relative path in template")
    content: str = Field(..., description="File content with variables")
    is_binary: bool = Field(default=False)
    executable: bool = Field(default=False)
    overwrite: bool = Field(default=True, description="Whether to overwrite existing files")


class TemplateMetadata(BaseModel):
    """Template metadata and configuration."""
    name: str = Field(..., description="Template display name")
    description: str = Field(..., description="Detailed template description")
    version: str = Field(default="1.0.0")
    author: str = Field(..., description="Template author")
    license: str = Field(default="MIT")
    tags: List[str] = Field(default_factory=list)
    
    # Classification
    type: TemplateType = Field(...)
    category: TemplateCategory = Field(...)
    
    # Requirements
    min_archon_version: Optional[str] = Field(None)
    dependencies: List[str] = Field(default_factory=list)
    
    # Usage stats
    downloads: int = Field(default=0)
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TemplateHook(BaseModel):
    """Template lifecycle hooks."""
    name: str = Field(..., description="Hook name")
    command: str = Field(..., description="Command to execute")
    working_directory: Optional[str] = Field(None)
    timeout: int = Field(default=300)  # 5 minutes
    failure_mode: str = Field(default="continue", pattern=r"^(continue|fail|warn)$")


class Template(BaseModel):
    """Complete template definition."""
    id: str = Field(..., description="Unique template identifier")
    metadata: TemplateMetadata
    variables: List[TemplateVariable] = Field(default_factory=list)
    files: List[TemplateFile] = Field(default_factory=list)
    
    # Lifecycle hooks
    pre_generate_hooks: List[TemplateHook] = Field(default_factory=list)
    post_generate_hooks: List[TemplateHook] = Field(default_factory=list)
    
    # Directory structure
    directory_structure: List[str] = Field(default_factory=list, description="Directories to create")
    
    # Template-specific configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('id')
    def validate_id(cls, v):
        """Validate template ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Template ID cannot be empty")
        if not v.replace('-', '').replace('_', '').replace('.', '').isalnum():
            raise ValueError("Template ID must contain only alphanumeric characters, hyphens, underscores, and dots")
        return v.strip().lower()
    
    @validator('files')
    def validate_files(cls, v):
        """Validate template files."""
        if not v:
            raise ValueError("Template must contain at least one file")
        
        paths = [f.path for f in v]
        if len(paths) != len(set(paths)):
            raise ValueError("Template files must have unique paths")
        
        return v


class TemplateGenerationRequest(BaseModel):
    """Request to generate a project from template."""
    template_id: str
    output_directory: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    overwrite_existing: bool = Field(default=False)
    dry_run: bool = Field(default=False)


class TemplateGenerationResult(BaseModel):
    """Result of template generation."""
    success: bool
    template_id: str
    output_directory: str
    files_created: List[str] = Field(default_factory=list)
    directories_created: List[str] = Field(default_factory=list)
    hooks_executed: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    generation_time: float = Field(default=0.0)


class TemplateSearchRequest(BaseModel):
    """Template search/filter request."""
    query: Optional[str] = Field(None, description="Search query")
    type: Optional[TemplateType] = Field(None)
    category: Optional[TemplateCategory] = Field(None)
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = Field(None)
    min_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="rating", pattern=r"^(rating|downloads|created_at|updated_at|name)$")
    sort_order: str = Field(default="desc", pattern=r"^(asc|desc)$")


class TemplateValidationResult(BaseModel):
    """Template validation result."""
    valid: bool
    template_id: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)