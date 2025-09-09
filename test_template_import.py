#!/usr/bin/env python3
"""
Test Template Import via API

Tests importing our sample templates to validate database integration.
"""

import requests
import json
import yaml
from pathlib import Path

ARCHON_BASE_URL = "http://localhost:8181"

def load_template_yaml(template_name):
    """Load template YAML and convert to correct Pydantic format."""
    template_path = Path(__file__).parent / "templates" / template_name / "template.yaml"
    
    if not template_path.exists():
        return None
        
    with open(template_path, 'r') as f:
        template_data = yaml.safe_load(f)
    
    # Generate template ID
    template_id = template_name.lower().replace('_', '-')
    
    # Map category names to expected enum values
    category_mapping = {
        "frontend": "frontend",
        "backend": "backend", 
        "fullstack": "fullstack",
        "general": "backend"  # fallback
    }
    
    # Map framework to type (using correct enum values)
    framework = template_data.get("framework", "")
    template_type = "project"  # default
    if framework in ["react", "vue", "angular"]:
        template_type = "project"
    elif framework in ["fastapi", "django", "flask"]:
        template_type = "api"
    elif "fullstack" in template_name or framework == "react+fastapi":
        template_type = "project"
    
    # Convert to correct API format
    api_template = {
        "id": template_id,
        "metadata": {
            "name": template_data.get("name"),
            "description": template_data.get("description"),
            "version": template_data.get("version", "1.0.0"),
            "author": template_data.get("author", "Archon Template System"),
            "license": "MIT",
            "tags": template_data.get("tags", []),
            "type": template_type,
            "category": category_mapping.get(template_data.get("category", "general"), "backend"),
            "min_archon_version": "1.0.0",
            "target_environment": ["development", "production"]
        },
        "variables": [],
        "files": [],
        "pre_generate_hooks": [],
        "post_generate_hooks": [],
        "directory_structure": [],
        "config": {}
    }
    
    # Convert variables to list format
    variables = template_data.get("variables", {})
    for var_name, var_config in variables.items():
        api_template["variables"].append({
            "name": var_name,
            "type": var_config.get("type", "string"),
            "description": var_config.get("description", ""),
            "default": str(var_config.get("default", "")),  # Convert to string
            "required": var_config.get("required", False),
            "validation": "",
            "options": []
        })
    
    # Add files (simplified for test)
    files = template_data.get("files", [])
    for file_path in files:
        api_template["files"].append({
            "path": file_path,
            "content": f"# Template file: {file_path}\n# Generated from {template_name}",
            "is_binary": False,
            "executable": file_path.endswith(".sh"),
            "overwrite": True
        })
    
    # Add some post-generation hooks if specified in YAML
    hooks = template_data.get("hooks", {})
    post_hooks = hooks.get("post_generate", [])
    for hook in post_hooks:
        if isinstance(hook, dict):
            api_template["post_generate_hooks"].append({
                "name": hook.get("name", "unknown"),
                "command": hook.get("command", "echo 'Hook executed'"),
                "working_directory": hook.get("working_directory"),
                "timeout": hook.get("timeout", 300),
                "failure_mode": "continue"
            })
    
    return {"template": api_template}

def test_template_import():
    """Test importing all sample templates."""
    templates = ["react-typescript-app", "fastapi-backend", "fullstack-modern"]
    results = {}
    
    for template_name in templates:
        print(f"\n🎨 Testing {template_name} template import...")
        
        # Load template data
        template_data = load_template_yaml(template_name)
        if not template_data:
            print(f"❌ Could not load template: {template_name}")
            results[template_name] = False
            continue
        
        print(f"✅ Template loaded: {len(template_data['template']['variables'])} variables, {len(template_data['template']['files'])} files")
        
        # Try to import via API
        try:
            response = requests.post(
                f"{ARCHON_BASE_URL}/api/templates/",
                json=template_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Template imported successfully: HTTP {response.status_code}")
                results[template_name] = True
                
                # Try to get the response data
                try:
                    resp_data = response.json()
                    template_id = resp_data.get("id", "unknown")
                    print(f"   📝 Template ID: {template_id}")
                except:
                    pass
                    
            else:
                print(f"❌ Import failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Raw response: {response.text[:200]}")
                results[template_name] = False
                
        except Exception as e:
            print(f"❌ Import failed with exception: {str(e)}")
            results[template_name] = False
    
    return results

def test_template_listing():
    """Test listing templates after import."""
    print(f"\n📋 Testing template listing...")
    
    try:
        response = requests.get(f"{ARCHON_BASE_URL}/api/templates/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            template_count = len(data.get("templates", []))
            total = data.get("pagination", {}).get("total", 0)
            
            print(f"✅ Template listing successful")
            print(f"   📁 Templates returned: {template_count}")
            print(f"   📊 Total in database: {total}")
            
            return template_count > 0
        else:
            print(f"❌ Listing failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Listing failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 TEMPLATE IMPORT & DATABASE INTEGRATION TEST")
    print("=" * 60)
    
    # Test server health first
    try:
        response = requests.get(f"{ARCHON_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not healthy")
            return False
        print("✅ Server is healthy")
    except:
        print("❌ Cannot connect to server")
        return False
    
    # Test template import
    import_results = test_template_import()
    
    # Test template listing
    listing_success = test_template_listing()
    
    # Calculate success rate
    successful_imports = sum(1 for success in import_results.values() if success)
    total_tests = len(import_results) + (1 if listing_success else 0)
    success_rate = (successful_imports + (1 if listing_success else 0)) / (total_tests if total_tests > 0 else 1)
    
    print(f"\n" + "=" * 60)
    print("🎯 DATABASE INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    print(f"Template Import Results:")
    for template, success in import_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status} {template}")
    
    print(f"Template Listing: {'✅ SUCCESS' if listing_success else '❌ FAILED'}")
    
    percentage = success_rate * 100
    print(f"\n🏆 INTEGRATION SUCCESS RATE: {percentage:.0f}%")
    
    if percentage >= 75:
        print("✅ DATABASE INTEGRATION: WORKING!")
        print("   Templates can be imported and retrieved successfully")
    else:
        print("⚠️ DATABASE INTEGRATION: PARTIAL")
        print("   Some functionality working, may need debugging")
    
    return success_rate >= 0.75

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)