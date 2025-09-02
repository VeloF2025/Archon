#!/usr/bin/env python3
"""
UNIVERSAL RULES CHECKER - MANDATORY SESSION START VALIDATION
Comprehensive rules validation system for all Claude Code sessions

USAGE:
    python "C:/Jarvis/AI Workspace/Archon/UNIVERSAL_RULES_CHECKER.py" --path "."

This script performs mandatory validation before any Archon operation:
1. Global rules validation (CLAUDE.md, RULES.md)
2. Project rules validation (local CLAUDE.md, RULES.md, PLANNING.md)
3. Configuration validation (package.json, tsconfig.json, etc.)
4. Critical requirements verification
5. Environment validation
6. Pre-commit validation (when --pre-commit flag used)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UniversalRulesChecker:
    """
    Comprehensive rules validation system
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.archon_root = Path(__file__).parent
        self.validation_results = {
            "success": True,
            "critical_failures": [],
            "warnings": [],
            "rules_loaded": [],
            "configurations_validated": [],
            "missing_requirements": []
        }
        
    def run_comprehensive_validation(self, pre_commit: bool = False) -> Dict[str, Any]:
        """
        Run complete validation sequence
        """
        logger.info("🔍 UNIVERSAL RULES CHECKER - VALIDATION INITIATED")
        logger.info(f"📁 Project Path: {self.project_path}")
        
        try:
            # Phase 1: Global Rules Validation
            logger.info("📋 Phase 1: Global Rules Validation")
            self.validate_global_rules()
            
            # Phase 2: Project Rules Validation
            logger.info("📝 Phase 2: Project Rules Validation")
            self.validate_project_rules()
            
            # Phase 3: Configuration Validation
            logger.info("⚙️ Phase 3: Configuration Validation")
            self.validate_configurations()
            
            # Phase 4: Critical Requirements
            logger.info("🔧 Phase 4: Critical Requirements Verification")
            self.validate_critical_requirements()
            
            # Phase 5: Environment Validation
            logger.info("🌍 Phase 5: Environment Validation")
            self.validate_environment()
            
            # Phase 6: Pre-commit Validation (if requested)
            if pre_commit:
                logger.info("✨ Phase 6: Pre-commit Validation")
                self.validate_pre_commit_requirements()
            
            # Final Assessment
            self.generate_final_assessment()
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"❌ Validation failed with error: {e}")
            self.validation_results["success"] = False
            self.validation_results["critical_failures"].append(f"Validation system error: {e}")
            return self.validation_results
    
    def validate_global_rules(self):
        """
        Validate global rules files (CLAUDE.md, RULES.md, PLAYWRIGHT_TESTING_PROTOCOL.md)
        """
        global_rules = [
            ("CLAUDE.md", True),  # (filename, required)
            ("RULES.md", False),
            ("MANIFEST.md", True),
            ("PLAYWRIGHT_TESTING_PROTOCOL.md", False)
        ]
        
        # Check Jarvis root directory
        jarvis_root = Path("/mnt/c/Jarvis")
        archon_root = self.archon_root
        
        for rule_file, required in global_rules:
            found = False
            
            # Check multiple locations
            locations = [jarvis_root / rule_file, archon_root / rule_file]
            
            for location in locations:
                if location.exists():
                    self.validation_results["rules_loaded"].append(f"✅ Global {rule_file} ({location.parent.name})")
                    found = True
                    
                    # Validate content for critical files
                    if rule_file == "MANIFEST.md":
                        self.validate_manifest_content(location)
                    elif rule_file == "CLAUDE.md":
                        self.validate_claude_md_content(location)
                    
                    break
            
            if not found and required:
                self.validation_results["critical_failures"].append(f"❌ Required global {rule_file} not found")
                self.validation_results["success"] = False
            elif not found:
                self.validation_results["warnings"].append(f"⚠️ Optional global {rule_file} not found")
    
    def validate_project_rules(self):
        """
        Validate project-specific rules files
        """
        project_rules = [
            ("RULES.md", False),
            ("CLAUDE.md", False), 
            ("PLANNING.md", False),
            ("TASK.md", False),
            ("UI_UX_DESIGN_SYSTEM.md", False),
            ("README.md", False)
        ]
        
        for rule_file, required in project_rules:
            rule_path = self.project_path / rule_file
            
            if rule_path.exists():
                self.validation_results["rules_loaded"].append(f"✅ Project {rule_file}")
                
                # Validate critical project rules
                if rule_file == "RULES.md":
                    self.validate_project_rules_content(rule_path)
            elif required:
                self.validation_results["critical_failures"].append(f"❌ Required project {rule_file} not found")
                self.validation_results["success"] = False
            else:
                self.validation_results["warnings"].append(f"⚠️ Optional project {rule_file} not found")
    
    def validate_configurations(self):
        """
        Validate project configurations
        """
        config_files = {
            "package.json": self.validate_package_json,
            "tsconfig.json": self.validate_typescript_config,
            "eslint.config.js": self.validate_eslint_config,
            ".eslintrc.json": self.validate_eslint_config,
            "prettier.config.js": self.validate_prettier_config,
            ".prettierrc": self.validate_prettier_config,
            "requirements.txt": self.validate_python_requirements,
            "pyproject.toml": self.validate_python_project,
            "Cargo.toml": self.validate_rust_config,
            "go.mod": self.validate_go_config
        }
        
        for config_file, validator in config_files.items():
            config_path = self.project_path / config_file
            if config_path.exists():
                try:
                    validator(config_path)
                    self.validation_results["configurations_validated"].append(f"✅ {config_file}")
                except Exception as e:
                    self.validation_results["warnings"].append(f"⚠️ {config_file} validation error: {e}")
    
    def validate_critical_requirements(self):
        """
        Validate critical requirements for the project
        """
        # Check for testing framework
        test_indicators = [
            self.project_path / "test",
            self.project_path / "tests", 
            self.project_path / "__tests__",
            self.project_path / "spec"
        ]
        
        has_tests = any(indicator.exists() for indicator in test_indicators)
        if has_tests:
            self.validation_results["configurations_validated"].append("✅ Test directory found")
        else:
            self.validation_results["missing_requirements"].append("❌ No test directory found")
        
        # Check for build scripts (if applicable)
        package_json = self.project_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    scripts = data.get("scripts", {})
                    
                    required_scripts = ["test", "build"]
                    for script in required_scripts:
                        if script in scripts:
                            self.validation_results["configurations_validated"].append(f"✅ {script} script defined")
                        else:
                            self.validation_results["warnings"].append(f"⚠️ {script} script not defined")
            except Exception as e:
                self.validation_results["warnings"].append(f"⚠️ package.json script validation error: {e}")
    
    def validate_environment(self):
        """
        Validate environment variables and setup
        """
        # Check for environment files
        env_files = [".env", ".env.example", ".env.local"]
        
        for env_file in env_files:
            env_path = self.project_path / env_file
            if env_path.exists():
                self.validation_results["configurations_validated"].append(f"✅ {env_file} found")
                
                if env_file == ".env":
                    self.validate_env_file_content(env_path)
    
    def validate_pre_commit_requirements(self):
        """
        Enhanced pre-commit validation with zero tolerance checks
        """
        logger.info("🔍 Running enhanced pre-commit validation...")
        
        # Check for zero tolerance script
        zero_tolerance_script = self.project_path / "scripts" / "zero-tolerance-check.js"
        if zero_tolerance_script.exists():
            try:
                result = subprocess.run(
                    ["node", str(zero_tolerance_script)],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.validation_results["configurations_validated"].append("✅ Zero tolerance validation passed")
                else:
                    self.validation_results["critical_failures"].append(f"❌ Zero tolerance validation failed: {result.stderr}")
                    self.validation_results["success"] = False
            except subprocess.TimeoutExpired:
                self.validation_results["critical_failures"].append("❌ Zero tolerance validation timed out")
                self.validation_results["success"] = False
            except Exception as e:
                self.validation_results["warnings"].append(f"⚠️ Zero tolerance validation error: {e}")
        else:
            # Run basic pre-commit checks
            self.run_basic_pre_commit_checks()
    
    def run_basic_pre_commit_checks(self):
        """
        Run basic pre-commit validation checks
        """
        # TypeScript compilation check
        if (self.project_path / "tsconfig.json").exists():
            try:
                result = subprocess.run(
                    ["npx", "tsc", "--noEmit"],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.validation_results["configurations_validated"].append("✅ TypeScript compilation check passed")
                else:
                    self.validation_results["critical_failures"].append(f"❌ TypeScript compilation failed: {result.stderr}")
                    self.validation_results["success"] = False
            except Exception as e:
                self.validation_results["warnings"].append(f"⚠️ TypeScript check error: {e}")
        
        # ESLint check
        eslint_configs = [".eslintrc.json", "eslint.config.js", ".eslintrc.js"]
        has_eslint = any((self.project_path / config).exists() for config in eslint_configs)
        
        if has_eslint:
            try:
                result = subprocess.run(
                    ["npx", "eslint", ".", "--ext", ".ts,.tsx,.js,.jsx"],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.validation_results["configurations_validated"].append("✅ ESLint validation passed")
                else:
                    self.validation_results["critical_failures"].append(f"❌ ESLint validation failed: {result.stderr}")
                    self.validation_results["success"] = False
            except Exception as e:
                self.validation_results["warnings"].append(f"⚠️ ESLint check error: {e}")
    
    def validate_manifest_content(self, manifest_path: Path):
        """
        Validate MANIFEST.md content
        """
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                required_sections = [
                    "ARCHON OPERATIONAL MANIFEST",
                    "MANDATORY COMPLIANCE",
                    "CLAUDE CODE → ARCHON PROCESS FLOW",
                    "QUALITY GATES & ENFORCEMENT"
                ]
                
                for section in required_sections:
                    if section not in content:
                        self.validation_results["warnings"].append(f"⚠️ MANIFEST.md missing section: {section}")
                        
        except Exception as e:
            self.validation_results["warnings"].append(f"⚠️ MANIFEST.md content validation error: {e}")
    
    def validate_claude_md_content(self, claude_path: Path):
        """
        Validate CLAUDE.md content
        """
        try:
            with open(claude_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for critical sections
                if "@Archon" in content or "ARCHON" in content:
                    self.validation_results["configurations_validated"].append("✅ CLAUDE.md has Archon integration")
                else:
                    self.validation_results["warnings"].append("⚠️ CLAUDE.md missing Archon integration")
                        
        except Exception as e:
            self.validation_results["warnings"].append(f"⚠️ CLAUDE.md content validation error: {e}")
    
    def validate_project_rules_content(self, rules_path: Path):
        """
        Validate project RULES.md content
        """
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Basic validation - file has content
                if len(content.strip()) > 100:  # At least 100 chars
                    self.validation_results["configurations_validated"].append("✅ Project RULES.md has substantial content")
                else:
                    self.validation_results["warnings"].append("⚠️ Project RULES.md appears incomplete")
                        
        except Exception as e:
            self.validation_results["warnings"].append(f"⚠️ Project RULES.md validation error: {e}")
    
    def validate_package_json(self, package_path: Path):
        """
        Validate package.json configuration
        """
        with open(package_path) as f:
            data = json.load(f)
            
            # Check for essential fields
            essential_fields = ["name", "version", "scripts"]
            for field in essential_fields:
                if field not in data:
                    self.validation_results["warnings"].append(f"⚠️ package.json missing {field}")
            
            # Check for testing setup
            scripts = data.get("scripts", {})
            if "test" not in scripts:
                self.validation_results["warnings"].append("⚠️ package.json missing test script")
            
            # Check for linting setup
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "eslint" not in deps and "@eslint/js" not in deps:
                self.validation_results["warnings"].append("⚠️ package.json missing ESLint dependency")
    
    def validate_typescript_config(self, tsconfig_path: Path):
        """
        Validate TypeScript configuration
        """
        with open(tsconfig_path) as f:
            # Basic JSON validation
            json.load(f)
            # TypeScript config is valid JSON
    
    def validate_eslint_config(self, eslint_path: Path):
        """
        Validate ESLint configuration
        """
        if eslint_path.suffix == '.json':
            with open(eslint_path) as f:
                json.load(f)
        # For JS configs, just check they exist
    
    def validate_prettier_config(self, prettier_path: Path):
        """
        Validate Prettier configuration
        """
        if prettier_path.suffix == '.json':
            with open(prettier_path) as f:
                json.load(f)
    
    def validate_python_requirements(self, requirements_path: Path):
        """
        Validate Python requirements.txt
        """
        with open(requirements_path) as f:
            lines = f.readlines()
            if len(lines) == 0:
                self.validation_results["warnings"].append("⚠️ requirements.txt is empty")
    
    def validate_python_project(self, pyproject_path: Path):
        """
        Validate Python pyproject.toml
        """
        try:
            import tomllib
            with open(pyproject_path, 'rb') as f:
                tomllib.load(f)
        except ImportError:
            # Python < 3.11
            pass
    
    def validate_rust_config(self, cargo_path: Path):
        """
        Validate Rust Cargo.toml
        """
        try:
            import tomllib
            with open(cargo_path, 'rb') as f:
                tomllib.load(f)
        except ImportError:
            pass
    
    def validate_go_config(self, go_mod_path: Path):
        """
        Validate Go module file
        """
        with open(go_mod_path) as f:
            content = f.read()
            if not content.startswith("module "):
                self.validation_results["warnings"].append("⚠️ go.mod missing module declaration")
    
    def validate_env_file_content(self, env_path: Path):
        """
        Validate .env file content
        """
        try:
            with open(env_path) as f:
                lines = f.readlines()
                
                # Check for common patterns
                has_database_url = any("DATABASE_URL" in line or "SUPABASE_URL" in line for line in lines)
                has_api_keys = any("API_KEY" in line for line in lines)
                
                if has_database_url:
                    self.validation_results["configurations_validated"].append("✅ Database configuration detected")
                if has_api_keys:
                    self.validation_results["configurations_validated"].append("✅ API key configuration detected")
                        
        except Exception as e:
            self.validation_results["warnings"].append(f"⚠️ .env validation error: {e}")
    
    def generate_final_assessment(self):
        """
        Generate final validation assessment
        """
        total_checks = (
            len(self.validation_results["rules_loaded"]) + 
            len(self.validation_results["configurations_validated"])
        )
        total_issues = (
            len(self.validation_results["critical_failures"]) +
            len(self.validation_results["warnings"]) +
            len(self.validation_results["missing_requirements"])
        )
        
        if len(self.validation_results["critical_failures"]) > 0:
            self.validation_results["status"] = "FAIL"
            self.validation_results["success"] = False
        elif total_issues == 0:
            self.validation_results["status"] = "EXCELLENT"
        elif total_issues <= 3:
            self.validation_results["status"] = "GOOD"  
        elif total_issues <= 6:
            self.validation_results["status"] = "FAIR"
        else:
            self.validation_results["status"] = "NEEDS_IMPROVEMENT"
        
        self.validation_results["summary"] = {
            "total_checks": total_checks,
            "total_issues": total_issues,
            "critical_failures": len(self.validation_results["critical_failures"]),
            "warnings": len(self.validation_results["warnings"]),
            "missing_requirements": len(self.validation_results["missing_requirements"])
        }


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description="Universal Rules Checker for Claude Code sessions")
    parser.add_argument("--path", default=".", help="Project path to validate")
    parser.add_argument("--pre-commit", action="store_true", help="Run pre-commit validation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🔍 UNIVERSAL RULES CHECKER")
    print("=" * 50)
    
    # Initialize checker
    checker = UniversalRulesChecker(args.path)
    
    # Run validation
    results = checker.run_comprehensive_validation(pre_commit=args.pre_commit)
    
    # Display results
    print(f"\n📊 VALIDATION RESULTS")
    print(f"Status: {results['status']}")
    print(f"Success: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
    
    if results.get("summary"):
        summary = results["summary"]
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Total Issues: {summary['total_issues']}")
    
    if results["rules_loaded"]:
        print(f"\n📋 RULES LOADED ({len(results['rules_loaded'])}):")
        for rule in results["rules_loaded"]:
            print(f"  {rule}")
    
    if results["configurations_validated"]:
        print(f"\n⚙️ CONFIGURATIONS VALIDATED ({len(results['configurations_validated'])}):")
        for config in results["configurations_validated"]:
            print(f"  {config}")
    
    if results["critical_failures"]:
        print(f"\n❌ CRITICAL FAILURES ({len(results['critical_failures'])}):")
        for failure in results["critical_failures"]:
            print(f"  {failure}")
    
    if results["warnings"]:
        print(f"\n⚠️ WARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"  {warning}")
    
    if results["missing_requirements"]:
        print(f"\n📋 MISSING REQUIREMENTS ({len(results['missing_requirements'])}):")
        for requirement in results["missing_requirements"]:
            print(f"  {requirement}")
    
    if results["success"]:
        print(f"\n🎉 VALIDATION PASSED - Ready for Archon activation!")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED - Address critical issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())