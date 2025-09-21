#!/usr/bin/env python3
"""
Pre-commit validation script for enhanced development workflow
Ensures all development follows Spec Kit enhanced process
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional

class PreCommitEnhancedValidation:
    """Validates that all development follows enhanced spec process"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.spec_cli = project_root / "archon-spec-simple.py"
        self.workflow_script = project_root / "scripts" / "enhanced-development-workflow.py"

    def validate_setup(self) -> bool:
        """Validate that enhanced development setup is ready"""
        if not self.spec_cli.exists():
            print("[ERROR] Enhanced Spec CLI not found")
            return False

        if not self.workflow_script.exists():
            print("[ERROR] Enhanced workflow script not found")
            return False

        return True

    def check_commit_message(self, commit_msg: str) -> Dict:
        """Check if commit message indicates feature development"""
        patterns = {
            "feature": r"feat(?:ure)?[:\-]\s*(.+)",
            "enhancement": r"enhance(?:ment)?[:\-]\s*(.+)",
            "bug": r"(?:bug|fix)[:\-]\s*(.+)",
            "new": r"new\s+(.+)",
            "add": r"add\s+(.+)"
        }

        detected_features = []
        for category, pattern in patterns.items():
            matches = re.findall(pattern, commit_msg, re.IGNORECASE)
            detected_features.extend([(category, match) for match in matches])

        return {
            "has_feature_indicators": len(detected_features) > 0,
            "detected_features": detected_features,
            "commit_message": commit_msg
        }

    def validate_feature_specification(self, feature_name: str) -> bool:
        """Validate that feature has proper specification"""
        # Check for specification directory
        spec_patterns = [
            f"specs/feat-{feature_name.lower().replace(' ', '-')}-spec",
            f"specs/{feature_name.lower().replace(' ', '-')}-spec",
            f"specs/*{feature_name.lower().replace(' ', '-')}*"
        ]

        spec_found = False
        for pattern in spec_patterns:
            if list(self.project_root.glob(pattern)):
                spec_found = True
                break

        if not spec_found:
            print(f"[ERROR] No specification found for feature: {feature_name}")
            print(f"        Create spec: python archon-spec-simple.py specify {feature_name} --input 'description'")
            return False

        # Check for plan and tasks
        plans = list(self.project_root.glob("**/plan.md")) + list(self.project_root.glob("spec*/plan.md"))
        tasks = list(self.project_root.glob("**/tasks.md")) + list(self.project_root.glob("tasks.md"))

        if not plans:
            print(f"[ERROR] No implementation plan found for feature: {feature_name}")
            return False

        if not tasks:
            print(f"[ERROR] No task list found for feature: {feature_name}")
            return False

        return True

    def check_modified_files(self, modified_files: List[str]) -> List[str]:
        """Check modified files for feature indicators"""
        feature_indicators = []

        for file_path in modified_files:
            if not os.path.exists(file_path):
                continue

            # Check file content for new features
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Look for TODO, FIXME, new functions, etc.
                    if "TODO:" in content or "FIXME:" in content:
                        feature_indicators.append(f"TODO/FIXME in {file_path}")

                    # Look for new function/class definitions
                    if re.search(r"(?:def|class)\s+\w+.*?(?=\n\ndef|\n\nclass|\Z)", content, re.DOTALL):
                        feature_indicators.append(f"New functions/classes in {file_path}")

            except Exception as e:
                print(f"[WARNING] Could not read {file_path}: {e}")

        return feature_indicators

    def validate_commit_files(self, commit_files: List[str]) -> bool:
        """Validate that all commit files have proper specifications"""
        issues = []

        # Check for common development patterns
        dev_patterns = [
            r".*\.py$",  # Python files
            r".*\.ts$",  # TypeScript files
            r".*\.tsx$", # TSX files
            r".*\.js$",  # JavaScript files
            r".*\.jsx$", # JSX files
        ]

        dev_files = [f for f in commit_files if any(re.match(pattern, f) for pattern in dev_patterns)]

        if dev_files:
            print(f"[INFO] Development files detected: {len(dev_files)} files")

            # Check for new feature indicators
            feature_indicators = self.check_modified_files(dev_files)

            if feature_indicators:
                print("[WARNING] Feature development detected without specification")
                for indicator in feature_indicators:
                    print(f"  - {indicator}")

                # Try to extract feature name from files
                feature_name = self._extract_feature_name(dev_files)
                if feature_name:
                    if not self.validate_feature_specification(feature_name):
                        issues.append(f"Missing specification for feature: {feature_name}")

        return len(issues) == 0, issues

    def _extract_feature_name(self, files: List[str]) -> Optional[str]:
        """Extract potential feature name from modified files"""
        # Look for common naming patterns
        for file_path in files:
            filename = Path(file_path).stem.lower()

            # Extract feature name from common patterns
            patterns = [
                r"(.*)_component",
                r"(.*)_service",
                r"(.*)_controller",
                r"(.*)_model",
                r"(.*)_page",
                r"(.*)_hook"
            ]

            for pattern in patterns:
                match = re.match(pattern, filename)
                if match:
                    return match.group(1).replace('_', ' ')

        return None

    def run_full_validation(self, commit_msg: str = "", commit_files: List[str] = None) -> bool:
        """Run complete pre-commit validation"""
        if not self.validate_setup():
            return False

        print("[ENHANCED VALIDATION] Running pre-commit checks...")
        all_issues = []

        # Validate commit message
        if commit_msg:
            msg_analysis = self.check_commit_message(commit_msg)
            if msg_analysis["has_feature_indicators"]:
                print("[INFO] Feature development detected in commit message")
                for category, feature in msg_analysis["detected_features"]:
                    if not self.validate_feature_specification(feature):
                        all_issues.append(f"Missing specification for {category}: {feature}")

        # Validate commit files
        if commit_files:
            files_valid, file_issues = self.validate_commit_files(commit_files)
            all_issues.extend(file_issues)

        # Check overall workflow compliance
        try:
            result = subprocess.run([
                sys.executable, str(self.workflow_script), "--compliance"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                all_issues.append("Enhanced workflow compliance check failed")
        except Exception as e:
            all_issues.append(f"Could not check workflow compliance: {e}")

        if all_issues:
            print("[ENHANCED VALIDATION FAILED]")
            print("Issues found:")
            for issue in all_issues:
                print(f"  - {issue}")
            print("\nPlease address these issues before committing.")
            print("To create a specification:")
            print("  python archon-spec-simple.py specify <feature-name> --input 'description'")
            return False

        print("[OK] Enhanced development validation passed")
        return True

def main():
    """Main entry point for git pre-commit hook"""
    project_root = Path(__file__).parent.parent
    validator = PreCommitEnhancedValidation(project_root)

    # Get commit message from stdin if available
    commit_msg = ""
    if not sys.stdin.isatty():
        commit_msg = sys.stdin.read().strip()

    # Get staged files (simplified - in real git hook, use git command)
    staged_files = []
    try:
        result = subprocess.run(["git", "diff", "--cached", "--name-only"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            staged_files = result.stdout.strip().split('\n')
    except:
        pass

    success = validator.run_full_validation(commit_msg, staged_files)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()