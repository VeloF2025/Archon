#!/usr/bin/env python3
"""
Test script for template system functionality
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python', 'src'))

def test_template_validation():
    """Test template file validation"""
    templates_dir = Path("templates")
    
    if not templates_dir.exists():
        print("❌ Templates directory not found")
        return False
    
    template_dirs = [d for d in templates_dir.iterdir() if d.is_dir()]
    print(f"📁 Found {len(template_dirs)} template directories")
    
    valid_templates = 0
    for template_dir in template_dirs:
        template_file = template_dir / ".archon-template.yaml"
        
        if not template_file.exists():
            print(f"❌ {template_dir.name}: Missing .archon-template.yaml")
            continue
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            
            # Basic validation
            required_fields = ['id', 'metadata', 'variables', 'files']
            missing_fields = [field for field in required_fields if field not in template_data]
            
            if missing_fields:
                print(f"❌ {template_dir.name}: Missing fields: {missing_fields}")
                continue
            
            # Validate metadata
            metadata = template_data['metadata']
            required_metadata = ['name', 'description', 'version', 'author', 'type', 'category']
            missing_metadata = [field for field in required_metadata if field not in metadata]
            
            if missing_metadata:
                print(f"❌ {template_dir.name}: Missing metadata: {missing_metadata}")
                continue
            
            # Count files
            file_count = len(template_data.get('files', []))
            variable_count = len(template_data.get('variables', []))
            
            print(f"✅ {template_dir.name}: Valid template ({file_count} files, {variable_count} variables)")
            valid_templates += 1
            
        except yaml.YAMLError as e:
            print(f"❌ {template_dir.name}: YAML parse error: {e}")
        except Exception as e:
            print(f"❌ {template_dir.name}: Validation error: {e}")
    
    print(f"\n📊 Template Validation Results: {valid_templates}/{len(template_dirs)} valid")
    return valid_templates == len(template_dirs)

def test_template_structure():
    """Test template directory structure"""
    templates_dir = Path("templates")
    
    expected_templates = {
        'react-typescript-app': {
            'type': 'frontend',
            'min_files': 10,
            'key_files': ['package.json', 'src/main.tsx', 'src/App.tsx']
        },
        'fastapi-backend': {
            'type': 'backend', 
            'min_files': 8,
            'key_files': ['pyproject.toml', 'app/main.py', 'app/core/config.py']
        },
        'fullstack-modern': {
            'type': 'fullstack',
            'min_files': 15,
            'key_files': ['docker-compose.yml', 'frontend/package.json', 'backend/pyproject.toml']
        }
    }
    
    print("\n🔍 Testing Template Structure...")
    
    all_passed = True
    for template_name, expected in expected_templates.items():
        template_dir = templates_dir / template_name
        template_file = template_dir / ".archon-template.yaml"
        
        if not template_file.exists():
            print(f"❌ {template_name}: Template file not found")
            all_passed = False
            continue
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
            
            # Check file count
            files = template_data.get('files', [])
            if len(files) < expected['min_files']:
                print(f"❌ {template_name}: Too few files ({len(files)} < {expected['min_files']})")
                all_passed = False
                continue
            
            # Check key files exist
            file_paths = {file['path'] for file in files}
            missing_key_files = [kf for kf in expected['key_files'] if kf not in file_paths]
            
            if missing_key_files:
                print(f"❌ {template_name}: Missing key files: {missing_key_files}")
                all_passed = False
                continue
            
            print(f"✅ {template_name}: Structure valid ({len(files)} files)")
            
        except Exception as e:
            print(f"❌ {template_name}: Structure test failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main test function"""
    print("🧪 Testing Archon Template System")
    print("=" * 50)
    
    # Test 1: Template validation
    validation_passed = test_template_validation()
    
    # Test 2: Template structure
    structure_passed = test_template_structure()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Template Validation: {'✅ PASS' if validation_passed else '❌ FAIL'}")
    print(f"   Template Structure:  {'✅ PASS' if structure_passed else '❌ FAIL'}")
    
    if validation_passed and structure_passed:
        print("\n🎉 All tests passed! Template system is ready.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)