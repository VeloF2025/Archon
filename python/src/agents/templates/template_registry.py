"""
Template Registry System

Manages template storage, discovery, and retrieval including:
- Local template storage and indexing
- Template marketplace integration
- Search and filtering capabilities
- Version management
"""

import os
import json
import yaml
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .template_models import (
    Template, TemplateSearchRequest, TemplateMetadata,
    TemplateType, TemplateCategory
)
from .template_validator import TemplateValidator
import logging

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """Manages template storage and discovery."""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path or os.path.expanduser("~/.archon/templates"))
        self.index_file = self.registry_path / "index.json"
        self.validator = TemplateValidator()
        
        # Ensure registry directory exists
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize index if it doesn't exist
        if not self.index_file.exists():
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create an empty template index."""
        index = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "templates": {},
            "categories": {category.value: [] for category in TemplateCategory},
            "types": {type_.value: [] for type_ in TemplateType},
            "authors": {},
            "tags": {}
        }
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, default=str)
        
        logger.info(f"Created template registry index: {self.index_file}")
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the template index."""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load template index: {e}")
            return {}
    
    def _save_index(self, index: Dict[str, Any]):
        """Save the template index."""
        try:
            index["updated_at"] = datetime.utcnow().isoformat()
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save template index: {e}")
    
    def _get_template_path(self, template_id: str) -> Path:
        """Get the path for a template."""
        return self.registry_path / template_id
    
    def _get_template_file_path(self, template_id: str) -> Path:
        """Get the path for a template's main file."""
        return self._get_template_path(template_id) / ".archon-template.yaml"
    
    def register_template(self, template: Template) -> bool:
        """Register a template in the registry."""
        try:
            logger.info(f"Registering template | id={template.id} | name={template.metadata.name}")
            
            # Validate template
            validation_result = self.validator.validate_template(template)
            if not validation_result.valid:
                logger.error(f"Template validation failed | template={template.id} | errors={validation_result.errors}")
                return False
            
            # Create template directory
            template_path = self._get_template_path(template.id)
            template_path.mkdir(parents=True, exist_ok=True)
            
            # Save template definition
            template_file = self._get_template_file_path(template.id)
            template_dict = template.dict()
            
            with open(template_file, 'w', encoding='utf-8') as f:
                yaml.dump(template_dict, f, default_flow_style=False, sort_keys=False)
            
            # Update index
            self._update_index_for_template(template)
            
            logger.info(f"Template registered successfully | id={template.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register template | id={template.id} | error={str(e)}")
            return False
    
    def _update_index_for_template(self, template: Template):
        """Update the index with template information."""
        index = self._load_index()
        
        # Add/update template entry
        index["templates"][template.id] = {
            "id": template.id,
            "name": template.metadata.name,
            "description": template.metadata.description,
            "version": template.metadata.version,
            "author": template.metadata.author,
            "type": template.metadata.type.value,
            "category": template.metadata.category.value,
            "tags": template.metadata.tags,
            "downloads": template.metadata.downloads,
            "rating": template.metadata.rating,
            "created_at": template.metadata.created_at.isoformat(),
            "updated_at": template.metadata.updated_at.isoformat(),
            "file_path": str(self._get_template_file_path(template.id))
        }
        
        # Update category index
        category_key = template.metadata.category.value
        if template.id not in index["categories"][category_key]:
            index["categories"][category_key].append(template.id)
        
        # Update type index
        type_key = template.metadata.type.value
        if template.id not in index["types"][type_key]:
            index["types"][type_key].append(template.id)
        
        # Update author index
        author = template.metadata.author
        if author not in index["authors"]:
            index["authors"][author] = []
        if template.id not in index["authors"][author]:
            index["authors"][author].append(template.id)
        
        # Update tag index
        for tag in template.metadata.tags:
            if tag not in index["tags"]:
                index["tags"][tag] = []
            if template.id not in index["tags"][tag]:
                index["tags"][tag].append(template.id)
        
        self._save_index(index)
    
    def get_template(self, template_id: str) -> Optional[Template]:
        """Retrieve a template by ID."""
        try:
            template_file = self._get_template_file_path(template_id)
            if not template_file.exists():
                return None
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            
            return Template(**template_data)
            
        except Exception as e:
            logger.error(f"Failed to load template {template_id}: {e}")
            return None
    
    def search_templates(self, request: TemplateSearchRequest) -> List[Dict[str, Any]]:
        """Search templates based on criteria."""
        index = self._load_index()
        templates = list(index["templates"].values())
        
        # Filter by query
        if request.query:
            query_lower = request.query.lower()
            templates = [
                t for t in templates
                if (query_lower in t["name"].lower() or 
                    query_lower in t["description"].lower() or
                    any(query_lower in tag.lower() for tag in t["tags"]))
            ]
        
        # Filter by type
        if request.type:
            templates = [t for t in templates if t["type"] == request.type.value]
        
        # Filter by category
        if request.category:
            templates = [t for t in templates if t["category"] == request.category.value]
        
        # Filter by tags
        if request.tags:
            templates = [
                t for t in templates
                if any(tag in t["tags"] for tag in request.tags)
            ]
        
        # Filter by author
        if request.author:
            templates = [t for t in templates if t["author"] == request.author]
        
        # Filter by minimum rating
        if request.min_rating is not None:
            templates = [t for t in templates if t["rating"] >= request.min_rating]
        
        # Sort results
        if request.sort_by == "rating":
            templates.sort(key=lambda t: t["rating"], reverse=(request.sort_order == "desc"))
        elif request.sort_by == "downloads":
            templates.sort(key=lambda t: t["downloads"], reverse=(request.sort_order == "desc"))
        elif request.sort_by == "created_at":
            templates.sort(key=lambda t: t["created_at"], reverse=(request.sort_order == "desc"))
        elif request.sort_by == "updated_at":
            templates.sort(key=lambda t: t["updated_at"], reverse=(request.sort_order == "desc"))
        elif request.sort_by == "name":
            templates.sort(key=lambda t: t["name"], reverse=(request.sort_order == "desc"))
        
        # Apply pagination
        total = len(templates)
        start_idx = request.offset
        end_idx = start_idx + request.limit
        templates = templates[start_idx:end_idx]
        
        return {
            "templates": templates,
            "total": total,
            "offset": request.offset,
            "limit": request.limit
        }
    
    def list_templates(self, category: Optional[TemplateCategory] = None, type_: Optional[TemplateType] = None) -> List[Dict[str, Any]]:
        """List all templates or by category/type."""
        index = self._load_index()
        templates = list(index["templates"].values())
        
        if category:
            templates = [t for t in templates if t["category"] == category.value]
        
        if type_:
            templates = [t for t in templates if t["type"] == type_.value]
        
        return templates
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template from the registry."""
        try:
            # Remove template directory
            template_path = self._get_template_path(template_id)
            if template_path.exists():
                shutil.rmtree(template_path)
            
            # Update index
            index = self._load_index()
            
            if template_id in index["templates"]:
                template_data = index["templates"][template_id]
                
                # Remove from templates
                del index["templates"][template_id]
                
                # Remove from category index
                category = template_data["category"]
                if template_id in index["categories"][category]:
                    index["categories"][category].remove(template_id)
                
                # Remove from type index
                type_ = template_data["type"]
                if template_id in index["types"][type_]:
                    index["types"][type_].remove(template_id)
                
                # Remove from author index
                author = template_data["author"]
                if author in index["authors"] and template_id in index["authors"][author]:
                    index["authors"][author].remove(template_id)
                    if not index["authors"][author]:  # Remove empty author entry
                        del index["authors"][author]
                
                # Remove from tag index
                for tag in template_data["tags"]:
                    if tag in index["tags"] and template_id in index["tags"][tag]:
                        index["tags"][tag].remove(template_id)
                        if not index["tags"][tag]:  # Remove empty tag entry
                            del index["tags"][tag]
                
                self._save_index(index)
            
            logger.info(f"Template deleted | id={template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete template | id={template_id} | error={str(e)}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        index = self._load_index()
        
        return {
            "total_templates": len(index["templates"]),
            "categories": {k: len(v) for k, v in index["categories"].items()},
            "types": {k: len(v) for k, v in index["types"].items()},
            "total_authors": len(index["authors"]),
            "total_tags": len(index["tags"]),
            "average_rating": sum(t["rating"] for t in index["templates"].values()) / max(len(index["templates"]), 1),
            "total_downloads": sum(t["downloads"] for t in index["templates"].values()),
            "created_at": index.get("created_at"),
            "updated_at": index.get("updated_at")
        }
    
    def increment_download_count(self, template_id: str):
        """Increment download count for a template."""
        index = self._load_index()
        if template_id in index["templates"]:
            index["templates"][template_id]["downloads"] += 1
            self._save_index(index)
    
    def update_template_rating(self, template_id: str, rating: float):
        """Update template rating."""
        if not (0.0 <= rating <= 5.0):
            raise ValueError("Rating must be between 0.0 and 5.0")
        
        index = self._load_index()
        if template_id in index["templates"]:
            index["templates"][template_id]["rating"] = rating
            self._save_index(index)
    
    def export_template(self, template_id: str, export_path: str) -> bool:
        """Export a template to a directory or archive."""
        try:
            template_path = self._get_template_path(template_id)
            if not template_path.exists():
                return False
            
            export_path = Path(export_path)
            
            if export_path.suffix in ['.tar', '.tar.gz', '.zip']:
                # Create archive
                if export_path.suffix == '.zip':
                    import zipfile
                    with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for file_path in template_path.rglob('*'):
                            if file_path.is_file():
                                zf.write(file_path, file_path.relative_to(template_path))
                else:
                    import tarfile
                    mode = 'w:gz' if export_path.suffix == '.tar.gz' else 'w'
                    with tarfile.open(export_path, mode) as tf:
                        tf.add(template_path, arcname=template_id)
            else:
                # Copy directory
                if export_path.exists():
                    shutil.rmtree(export_path)
                shutil.copytree(template_path, export_path)
            
            logger.info(f"Template exported | id={template_id} | path={export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export template | id={template_id} | error={str(e)}")
            return False
    
    def import_template_from_directory(self, directory_path: str) -> Optional[str]:
        """Import a template from a directory."""
        try:
            directory_path = Path(directory_path)
            template_file = directory_path / ".archon-template.yaml"
            
            if not template_file.exists():
                logger.error(f"No .archon-template.yaml found in {directory_path}")
                return None
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            
            template = Template(**template_data)
            
            # Copy template files to registry
            template_path = self._get_template_path(template.id)
            if template_path.exists():
                shutil.rmtree(template_path)
            
            shutil.copytree(directory_path, template_path)
            
            # Register in index
            self._update_index_for_template(template)
            
            logger.info(f"Template imported | id={template.id} | from={directory_path}")
            return template.id
            
        except Exception as e:
            logger.error(f"Failed to import template | path={directory_path} | error={str(e)}")
            return None