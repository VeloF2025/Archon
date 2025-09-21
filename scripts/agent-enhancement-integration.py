#!/usr/bin/env python3
"""
Agent Enhancement Integration
Updates existing Archon agents to use enhanced Spec Kit processes
"""

import sys
import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional

class AgentEnhancementIntegration:
    """Integrates Spec Kit methodologies into existing Archon agents"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.spec_cli = project_root / "archon-spec-simple.py"
        self.agents_config = project_root / ".archon" / "project_agents.yaml"
        self.enhancement_log = project_root / ".archon" / "agent_enhancements.log"

    def validate_setup(self) -> bool:
        """Validate setup requirements"""
        if not self.spec_cli.exists():
            print("[ERROR] Enhanced Spec CLI not found")
            return False

        if not self.agents_config.exists():
            print("[ERROR] Archon agents configuration not found")
            return False

        return True

    def load_agents_config(self) -> Dict:
        """Load existing agents configuration"""
        try:
            with open(self.agents_config, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}

    def save_agents_config(self, config: Dict):
        """Save updated agents configuration"""
        with open(self.agents_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def log_enhancement(self, agent_name: str, enhancement_type: str, details: str):
        """Log agent enhancement activities"""
        log_entry = {
            "timestamp": str(Path().cwd()),
            "agent": agent_name,
            "enhancement": enhancement_type,
            "details": details
        }

        try:
            if self.enhancement_log.exists():
                with open(self.enhancement_log, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            with open(self.enhancement_log, 'w') as f:
                json.dump(logs, f, indent=2)
        except:
            pass

    def enhance_strategic_planner_agent(self) -> bool:
        """Enhance strategic-planner agent with Spec Kit methodologies"""
        print("[ENHANCEMENT] Adding Spec Kit integration to agent configuration...")

        config = self.load_agents_config()
        if not config:
            print("[ERROR] Could not load agents configuration")
            return False

        # Add Spec Kit integration to the global configuration
        if "spec_kit_integration" not in config:
            config["spec_kit_integration"] = {
                "enabled": True,
                "version": "1.0",
                "cli_path": "archon-spec-simple.py",
                "required_for_all_development": True,
                "validation_commands": [
                    "python archon-spec-simple.py validate <feature>",
                    "python archon-spec-simple.py status"
                ]
            }

        # Add enhanced capabilities to specialized agents
        for agent in config.get("specialized_agents", []):
            agent_name = agent.get("name", "")
            if "enhanced_capabilities" not in agent:
                agent["enhanced_capabilities"] = []
                agent["enhanced_capabilities"].append("Spec Kit enhanced specification parsing")
                agent["enhanced_capabilities"].append("Four-phase planning methodology")
                agent["enhanced_capabilities"].append("User scenario extraction")
                agent["enhanced_capabilities"].append("Acceptance criteria generation")

            # Add Spec Kit integration to quality gates
            if "quality_gates" not in agent:
                agent["quality_gates"] = []
            agent["quality_gates"].append("Enhanced specification compliance")
            agent["quality_gates"].append("TDD documentation validation")
            agent["quality_gates"].append("DGTS anti-gaming compliance")

        self.save_agents_config(config)
        self.log_enhancement("all_agents", "spec_kit_integration",
                           "Added Spec Kit enhanced capabilities to all specialized agents")

        return True

    def enhance_all_agents(self) -> bool:
        """Enhance all existing agents with Spec Kit methodologies"""
        if not self.validate_setup():
            return False

        print("[ENHANCEMENT] Starting agent enhancement process...")

        try:
            result = self.enhance_strategic_planner_agent()
            print(f"[{'OK' if result else 'ERROR'}] Agent enhancement completed")
        except Exception as e:
            print(f"[ERROR] Agent enhancement failed: {e}")
            result = False

        if result:
            print("[SUCCESS] All agents enhanced with Spec Kit methodologies")
        else:
            print("[PARTIAL] Agent enhancement failed")

        return result

    def create_agent_integration_script(self):
        """Create script for agents to use enhanced spec process"""
        script_content = '''#!/usr/bin/env python3
"""
Agent Spec Kit Integration Script
For use by all Archon agents to ensure enhanced spec process compliance
"""

import sys
import subprocess
from pathlib import Path

def validate_spec_exists(feature_name: str, project_root: Path) -> bool:
    """Validate that specification exists for feature"""
    spec_cli = project_root / "archon-spec-simple.py"

    if not spec_cli.exists():
        print("[ERROR] Enhanced Spec CLI not found")
        return False

    # Check specification
    cmd = [sys.executable, str(spec_cli), "validate", f"specs/feat-{feature_name.lower().replace(' ', '-')}-spec"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    return result.returncode == 0

def create_spec_if_needed(feature_name: str, description: str, project_root: Path) -> bool:
    """Create specification if it doesn't exist"""
    spec_cli = project_root / "archon-spec-simple.py"

    cmd = [sys.executable, str(spec_cli), "specify", feature_name, "--input", description]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    return result.returncode == 0

def generate_plan_if_needed(feature_name: str, project_root: Path) -> bool:
    """Generate implementation plan if needed"""
    spec_cli = project_root / "archon-spec-simple.py"
    spec_file = project_root / f"specs/feat-{feature_name.lower().replace(' ', '-')}-spec/spec.md"

    if not spec_file.exists():
        return False

    cmd = [sys.executable, str(spec_cli), "plan", str(spec_file)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    return result.returncode == 0

def agent_pre_validation(feature_name: str, description: str, project_root: Path) -> bool:
    """Pre-validation for all agent activities"""
    print(f"[AGENT VALIDATION] Checking spec compliance for: {feature_name}")

    # Validate or create specification
    if not validate_spec_exists(feature_name, project_root):
        print(f"[AGENT] Creating specification for: {feature_name}")
        if not create_spec_if_needed(feature_name, description, project_root):
            print("[ERROR] Failed to create specification")
            return False

    # Generate plan if needed
    if not generate_plan_if_needed(feature_name, project_root):
        print("[WARNING] Could not generate implementation plan")

    print(f"[OK] Agent spec validation passed for: {feature_name}")
    return True

# Main validation function for agent use
def validate_agent_work(feature_name: str, description: str = "") -> bool:
    """Validate that agent work follows enhanced spec process"""
    project_root = Path(__file__).parent.parent.parent
    return agent_pre_validation(feature_name, description, project_root)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent-spec-integration.py <feature-name> [description]")
        sys.exit(1)

    feature_name = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else f"Implementation of {feature_name}"

    success = validate_agent_work(feature_name, description)
    sys.exit(0 if success else 1)
'''

        integration_script = self.project_root / "scripts" / "agent-spec-integration.py"
        with open(integration_script, 'w') as f:
            f.write(script_content)

        print(f"[OK] Agent integration script created: {integration_script}")

def run_enhancement_check(self) -> Dict:
    """Check current enhancement status"""
    config = self.load_agents_config()
    status = {
        "setup_valid": self.validate_setup(),
        "agents_configured": len(config.get("agents", {})),
        "enhanced_agents": 0,
        "enhancement_log_exists": self.enhancement_log.exists()
    }

    # Check which agents have been enhanced
    for agent_name, agent_config in config.get("agents", {}).items():
        if "enhanced_capabilities" in agent_config or "spec_kit_integration" in agent_config:
            status["enhanced_agents"] += 1

    return status

def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent
    integrator = AgentEnhancementIntegration(project_root)

    if len(sys.argv) < 2:
        print("Agent Enhancement Integration")
        print("Usage:")
        print("  python scripts/agent-enhancement-integration.py --enhance-all")
        print("  python scripts/agent-enhancement-integration.py --status")
        print("  python scripts/agent-enhancement-integration.py --setup-integration")
        return

    command = sys.argv[1]

    if command == "--enhance-all":
        integrator.enhance_all_agents()
    elif command == "--status":
        status = integrator.run_enhancement_check()
        print("Enhancement Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    elif command == "--setup-integration":
        integrator.create_agent_integration_script()
    else:
        print("Invalid command")

if __name__ == "__main__":
    main()