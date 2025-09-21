#!/usr/bin/env python3
"""
Enhanced Development Workflow Integration
Ensures Spec Kit enhanced CLI is used for all development activities
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional

class EnhancedDevelopmentWorkflow:
    """Integrates Spec Kit enhanced CLI into all development activities"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.spec_cli = project_root / "archon-spec-simple.py"
        self.config_file = project_root / ".enhanced-development-config.json"

    def validate_setup(self) -> bool:
        """Validate that enhanced development setup is ready"""
        if not self.spec_cli.exists():
            print("[ERROR] Enhanced Spec CLI not found. Run setup first.")
            return False

        config = self.load_config()
        if not config.get("enabled", False):
            print("[ERROR] Enhanced development workflow not enabled.")
            return False

        return True

    def load_config(self) -> Dict:
        """Load enhanced development configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {"enabled": False, "version": "1.0"}

    def save_config(self, config: Dict):
        """Save enhanced development configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def start_new_feature(self, feature_name: str, description: str) -> bool:
        """Start new feature development using enhanced spec process"""
        if not self.validate_setup():
            return False

        print(f"[ENHANCED WORKFLOW] Starting new feature: {feature_name}")

        # Step 1: Create enhanced specification
        cmd = [sys.executable, str(self.spec_cli), "specify", feature_name, "--input", description]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

        if result.returncode != 0:
            print(f"[ERROR] Failed to create specification: {result.stderr}")
            return False

        print(result.stdout)

        # Step 2: Generate implementation plan
        spec_file = self.project_root / f"specs/feat-{feature_name.lower().replace(' ', '-')}-spec/spec.md"
        if spec_file.exists():
            cmd = [sys.executable, str(self.spec_cli), "plan", str(spec_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                print(result.stdout)
                print(f"[SUCCESS] Enhanced specification workflow started for: {feature_name}")
                return True

        return False

    def validate_before_development(self, feature_name: str) -> bool:
        """Validate that proper spec process was followed before development"""
        if not self.validate_setup():
            return False

        # Check if specification exists
        spec_dir = self.project_root / f"specs/feat-{feature_name.lower().replace(' ', '-')}-spec"
        if not spec_dir.exists():
            print(f"[ERROR] No specification found for feature: {feature_name}")
            print(f"        Run: python archon-spec-simple.py specify {feature_name} --input 'description'")
            return False

        # Check if plan exists
        plan_files = list(self.project_root.glob("**/plan.md")) + list(self.project_root.glob("spec*/plan.md"))
        if not plan_files:
            print(f"[ERROR] No implementation plan found for feature: {feature_name}")
            print(f"        Run: python archon-spec-simple.py plan specs/feat-{feature_name}-spec/spec.md")
            return False

        # Check if tasks exist
        task_files = list(self.project_root.glob("**/tasks.md")) + list(self.project_root.glob("tasks.md"))
        if not task_files:
            print(f"[ERROR] No task list found for feature: {feature_name}")
            print(f"        Run: python archon-spec-simple.py tasks spec-plan")
            return False

        # Validate specification quality
        cmd = [sys.executable, str(self.spec_cli), "validate", str(spec_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

        if result.returncode != 0:
            print(f"[ERROR] Specification validation failed:")
            print(result.stdout)
            return False

        print(f"[OK] Enhanced specification process validated for: {feature_name}")
        return True

    def create_enhancement_request(self, enhancement_name: str, description: str) -> bool:
        """Create enhancement specification using enhanced process"""
        if not self.validate_setup():
            return False

        print(f"[ENHANCED WORKFLOW] Starting enhancement: {enhancement_name}")

        # Create specification for enhancement
        cmd = [sys.executable, str(self.spec_cli), "specify", enhancement_name, "--input", description]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

        if result.returncode == 0:
            print(result.stdout)
            print(f"[SUCCESS] Enhancement specification created: {enhancement_name}")
            return True

        return False

    def handle_bug_fix(self, bug_description: str) -> bool:
        """Handle bug fixes using enhanced specification process"""
        if not self.validate_setup():
            return False

        print(f"[ENHANCED WORKFLOW] Processing bug fix: {bug_description[:50]}...")

        # Create specification for bug fix
        bug_name = f"bug-fix-{hash(bug_description) % 10000}"
        cmd = [sys.executable, str(self.spec_cli), "specify", bug_name, "--input", f"Bug Fix: {bug_description}"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

        if result.returncode == 0:
            print(result.stdout)
            print(f"[SUCCESS] Bug fix specification created")
            return True

        return False

    def check_workflow_compliance(self) -> Dict:
        """Check overall workflow compliance"""
        compliance = {
            "spec_cli_exists": self.spec_cli.exists(),
            "config_exists": self.config_file.exists(),
            "enabled": self.load_config().get("enabled", False),
            "recent_specs": len(list(self.project_root.glob("specs/**/spec.md"))),
            "recent_plans": len(list(self.project_root.glob("**/plan.md")) + list(self.project_root.glob("spec*/plan.md"))),
            "recent_tasks": len(list(self.project_root.glob("**/tasks.md")) + list(self.project_root.glob("tasks.md")))
        }

        compliance["compliant"] = all([
            compliance["spec_cli_exists"],
            compliance["enabled"],
            compliance["recent_specs"] > 0
        ])

        return compliance

    def setup_enhanced_workflow(self) -> bool:
        """Setup enhanced development workflow"""
        print("[SETUP] Configuring enhanced development workflow...")

        # Create configuration
        config = {
            "enabled": True,
            "version": "1.0",
            "require_spec_for_all_development": True,
            "validate_before_commit": True,
            "auto_generate_tasks": True,
            "integration_points": [
                "pre-commit hooks",
                "agent workflows",
                "CI/CD pipeline",
                "development guidelines"
            ]
        }

        self.save_config(config)

        # Create workflow scripts
        self._create_workflow_scripts()

        print("[SUCCESS] Enhanced development workflow configured")
        return True

    def _create_workflow_scripts(self):
        """Create workflow integration scripts"""
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Create pre-development check script
        pre_dev_script = scripts_dir / "pre-development-check.py"
        pre_dev_content = '''#!/usr/bin/env python3
"""
Pre-development check script
Ensures enhanced spec process is followed before any development
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python pre-development-check.py <feature-name>")
        sys.exit(1)

    feature_name = sys.argv[1]
    workflow = EnhancedDevelopmentWorkflow(Path.cwd())

    if not workflow.validate_before_development(feature_name):
        print("[BLOCKED] Development cannot proceed without proper specification")
        sys.exit(1)

    print("[OK] Development approved")

if __name__ == "__main__":
    main()
'''

        with open(pre_dev_script, 'w') as f:
            f.write(pre_dev_content)

    def enforce_workflow(self) -> bool:
        """Enforce enhanced workflow for all development activities"""
        compliance = self.check_workflow_compliance()

        if not compliance["compliant"]:
            print("[ENFORCEMENT] Enhanced development workflow not properly configured")
            print("            Run: python scripts/enhanced-development-workflow.py --setup")
            return False

        print("[ENFORCED] Enhanced development workflow active")
        return True

def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent
    workflow = EnhancedDevelopmentWorkflow(project_root)

    if len(sys.argv) < 2:
        print("Enhanced Development Workflow Manager")
        print("Usage:")
        print("  python scripts/enhanced-development-workflow.py --setup")
        print("  python scripts/enhanced-development-workflow.py --new-feature <name> <description>")
        print("  python scripts/enhanced-development-workflow.py --enhancement <name> <description>")
        print("  python scripts/enhanced-development-workflow.py --bug-fix <description>")
        print("  python scripts/enhanced-development-workflow.py --validate <feature-name>")
        print("  python scripts/enhanced-development-workflow.py --compliance")
        print("  python scripts/enhanced-development-workflow.py --enforce")
        return

    command = sys.argv[1]

    if command == "--setup":
        workflow.setup_enhanced_workflow()
    elif command == "--new-feature" and len(sys.argv) >= 4:
        workflow.start_new_feature(sys.argv[2], sys.argv[3])
    elif command == "--enhancement" and len(sys.argv) >= 4:
        workflow.create_enhancement_request(sys.argv[2], sys.argv[3])
    elif command == "--bug-fix" and len(sys.argv) >= 3:
        workflow.handle_bug_fix(sys.argv[2])
    elif command == "--validate" and len(sys.argv) >= 3:
        workflow.validate_before_development(sys.argv[2])
    elif command == "--compliance":
        compliance = workflow.check_workflow_compliance()
        print("Workflow Compliance:")
        for key, value in compliance.items():
            print(f"  {key}: {value}")
    elif command == "--enforce":
        workflow.enforce_workflow()
    else:
        print("Invalid command. Use --help for usage.")

if __name__ == "__main__":
    main()