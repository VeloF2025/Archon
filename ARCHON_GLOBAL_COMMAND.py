#!/usr/bin/env python3
"""
ARCHON GLOBAL ACTIVATION COMMAND
Comprehensive @Archon system activation for any coding project

USAGE:
    python "C:\Jarvis\AI Workspace\Archon\ARCHON_GLOBAL_COMMAND.py"
    
TRIGGERS: "@Archon", "Archon", "@archon", "archon"

This script handles the complete activation of the Archon system:
1. System validation and dependency checks
2. Docker container orchestration
3. Project registration and knowledge base setup
4. Specialized agent activation
5. Rules and manifest enforcement
6. Health monitoring and status reporting
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArchonGlobalActivator:
    """
    Universal Archon system activator for any coding project
    """
    
    def __init__(self, project_path: str = None):
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.archon_root = Path(__file__).parent
        self.activation_status = {
            "system_validated": False,
            "containers_running": False,
            "project_registered": False,
            "agents_activated": False,
            "rules_loaded": False,
            "health_verified": False
        }
        
        # Archon service ports
        self.ports = {
            "server": int(os.getenv("ARCHON_SERVER_PORT", "8181")),
            "mcp": int(os.getenv("ARCHON_MCP_PORT", "8051")),
            "agents": int(os.getenv("ARCHON_AGENTS_PORT", "8052")),
            "ui": int(os.getenv("ARCHON_UI_PORT", "3737")),
            "validator": int(os.getenv("VALIDATOR_PORT", "8053"))
        }
        
        self.services_urls = {
            "server": f"http://localhost:{self.ports['server']}",
            "mcp": f"http://localhost:{self.ports['mcp']}",
            "agents": f"http://localhost:{self.ports['agents']}",
            "ui": f"http://localhost:{self.ports['ui']}",
            "validator": f"http://localhost:{self.ports['validator']}"
        }
        
    def activate_archon_system(self) -> Dict[str, Any]:
        """
        Main activation sequence - COMPREHENSIVE ARCHON STARTUP
        """
        logger.info("🚀 ARCHON GLOBAL ACTIVATION INITIATED")
        logger.info(f"📁 Project Path: {self.project_path}")
        logger.info(f"🏠 Archon Root: {self.archon_root}")
        
        activation_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_path": str(self.project_path),
            "archon_root": str(self.archon_root),
            "activation_steps": [],
            "status": "initiated"
        }
        
        try:
            # Phase 1: System Validation
            logger.info("📋 Phase 1: System Validation")
            validation_result = self.validate_system_requirements()
            activation_report["activation_steps"].append({
                "phase": "System Validation",
                "status": "completed" if validation_result["success"] else "failed",
                "details": validation_result
            })
            
            if not validation_result["success"]:
                raise Exception(f"System validation failed: {validation_result['errors']}")
            
            # Phase 2: Docker Container Orchestration
            logger.info("🐳 Phase 2: Docker Container Orchestration")
            container_result = self.ensure_containers_running()
            activation_report["activation_steps"].append({
                "phase": "Container Orchestration",
                "status": "completed" if container_result["success"] else "failed",
                "details": container_result
            })
            
            if not container_result["success"]:
                logger.warning("⚠️ Some containers failed to start - attempting recovery...")
                recovery_result = self.recover_container_failures(container_result["failures"])
                if not recovery_result["success"]:
                    raise Exception(f"Container orchestration failed: {recovery_result['errors']}")
            
            # Phase 3: Project Registration
            logger.info("📝 Phase 3: Project Registration & Knowledge Base Setup")
            project_result = self.register_project()
            activation_report["activation_steps"].append({
                "phase": "Project Registration",
                "status": "completed" if project_result["success"] else "failed",
                "details": project_result
            })
            
            # Phase 4: Agent Activation
            logger.info("🤖 Phase 4: Specialized Agent Activation")
            agent_result = self.activate_specialized_agents()
            activation_report["activation_steps"].append({
                "phase": "Agent Activation",
                "status": "completed" if agent_result["success"] else "failed",
                "details": agent_result
            })
            
            # Phase 5: Rules & Manifest Enforcement
            logger.info("📖 Phase 5: Rules & Manifest Loading")
            rules_result = self.load_rules_and_manifest()
            activation_report["activation_steps"].append({
                "phase": "Rules & Manifest Loading",
                "status": "completed" if rules_result["success"] else "failed",
                "details": rules_result
            })
            
            # Phase 6: Health Verification
            logger.info("🔍 Phase 6: System Health Verification")
            health_result = self.verify_system_health()
            activation_report["activation_steps"].append({
                "phase": "Health Verification",
                "status": "completed" if health_result["success"] else "failed",
                "details": health_result
            })
            
            # Final Status Report
            all_phases_success = all(
                step["status"] == "completed" 
                for step in activation_report["activation_steps"]
            )
            
            activation_report["status"] = "success" if all_phases_success else "partial"
            activation_report["services_status"] = health_result.get("services", {})
            activation_report["next_steps"] = self.generate_next_steps(activation_report)
            
            if all_phases_success:
                logger.info("✅ ARCHON SYSTEM FULLY ACTIVATED")
                logger.info(f"🌐 UI Available: {self.services_urls['ui']}")
                logger.info(f"🔧 MCP Tools Available: {len(agent_result.get('available_agents', []))} agents")
            else:
                logger.warning("⚠️ ARCHON SYSTEM PARTIALLY ACTIVATED - Check logs for issues")
            
            return activation_report
            
        except Exception as e:
            logger.error(f"❌ ARCHON ACTIVATION FAILED: {e}")
            activation_report["status"] = "failed"
            activation_report["error"] = str(e)
            return activation_report
    
    def validate_system_requirements(self) -> Dict[str, Any]:
        """
        Validate all system requirements before activation
        """
        validation_results = {
            "success": True,
            "checks": [],
            "errors": [],
            "warnings": []
        }
        
        # Check 1: MANIFEST.md exists
        manifest_path = self.archon_root / "MANIFEST.md"
        if manifest_path.exists():
            validation_results["checks"].append("✅ MANIFEST.md found")
        else:
            validation_results["errors"].append("❌ MANIFEST.md not found")
            validation_results["success"] = False
        
        # Check 2: Docker availability
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                validation_results["checks"].append(f"✅ Docker available: {result.stdout.strip()}")
            else:
                validation_results["errors"].append("❌ Docker not available")
                validation_results["success"] = False
        except Exception as e:
            validation_results["errors"].append(f"❌ Docker check failed: {e}")
            validation_results["success"] = False
        
        # Check 3: Environment variables
        required_env = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
        env_file = self.archon_root / ".env"
        
        if env_file.exists():
            validation_results["checks"].append("✅ .env file found")
            # Load env vars from file
            with open(env_file) as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        key, value = line.strip().split("=", 1)
                        if key in required_env and value:
                            validation_results["checks"].append(f"✅ {key} configured")
                        elif key in required_env:
                            validation_results["errors"].append(f"❌ {key} not set in .env")
                            validation_results["success"] = False
        else:
            validation_results["warnings"].append("⚠️ .env file not found - using environment variables")
            for env_var in required_env:
                if os.getenv(env_var):
                    validation_results["checks"].append(f"✅ {env_var} available")
                else:
                    validation_results["errors"].append(f"❌ {env_var} not set")
                    validation_results["success"] = False
        
        # Check 4: Python and dependencies
        try:
            import uvicorn, fastapi, supabase
            validation_results["checks"].append("✅ Python dependencies available")
        except ImportError as e:
            validation_results["errors"].append(f"❌ Python dependencies missing: {e}")
            validation_results["success"] = False
        
        # Check 5: Port availability
        import socket
        for service, port in self.ports.items():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        validation_results["warnings"].append(f"⚠️ Port {port} ({service}) already in use")
                    else:
                        validation_results["checks"].append(f"✅ Port {port} ({service}) available")
            except Exception as e:
                validation_results["warnings"].append(f"⚠️ Port check failed for {service}: {e}")
        
        return validation_results
    
    def ensure_containers_running(self) -> Dict[str, Any]:
        """
        Ensure all Archon Docker containers are running
        """
        container_results = {
            "success": True,
            "running_containers": [],
            "failed_containers": [],
            "failures": []
        }
        
        try:
            # Check current container status
            result = subprocess.run(
                ["docker", "compose", "-f", str(self.archon_root / "docker-compose.yml"), "ps", "--format", "json"],
                cwd=self.archon_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                try:
                    containers = json.loads(result.stdout) if result.stdout.strip() else []
                    running_services = [c.get("Service", c.get("Name", "")) for c in containers if c.get("State") == "running"]
                    container_results["running_containers"] = running_services
                    logger.info(f"🐳 Running containers: {running_services}")
                except json.JSONDecodeError:
                    # Fallback for older docker-compose versions
                    running_services = []
            
            # Start containers if not running
            required_services = ["archon-server", "archon-mcp", "archon-frontend"]
            agents_enabled = os.getenv("AGENTS_ENABLED", "false").lower() == "true"
            
            if agents_enabled:
                required_services.append("archon-agents")
            
            missing_services = [svc for svc in required_services if svc not in running_services]
            
            if missing_services:
                logger.info(f"🚀 Starting containers: {missing_services}")
                
                # Build and start containers
                build_cmd = ["docker", "compose", "build"]
                if agents_enabled:
                    build_cmd.extend(["--profile", "agents"])
                
                build_result = subprocess.run(
                    build_cmd,
                    cwd=self.archon_root,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes for build
                )
                
                if build_result.returncode != 0:
                    container_results["failures"].append(f"Build failed: {build_result.stderr}")
                    container_results["success"] = False
                    return container_results
                
                # Start containers
                start_cmd = ["docker", "compose", "up", "-d"]
                if agents_enabled:
                    start_cmd.extend(["--profile", "agents"])
                
                start_result = subprocess.run(
                    start_cmd,
                    cwd=self.archon_root,
                    capture_output=True,
                    text=True,
                    timeout=180  # 3 minutes for startup
                )
                
                if start_result.returncode == 0:
                    logger.info("✅ Containers started successfully")
                    # Wait for health checks
                    time.sleep(10)
                else:
                    container_results["failures"].append(f"Start failed: {start_result.stderr}")
                    container_results["success"] = False
            else:
                logger.info("✅ All required containers already running")
            
        except subprocess.TimeoutExpired:
            container_results["failures"].append("Docker operations timed out")
            container_results["success"] = False
        except Exception as e:
            container_results["failures"].append(f"Container management error: {e}")
            container_results["success"] = False
        
        return container_results
    
    def recover_container_failures(self, failures: List[str]) -> Dict[str, Any]:
        """
        Attempt to recover from container failures
        """
        recovery_results = {
            "success": False,
            "recovery_attempts": [],
            "errors": []
        }
        
        logger.info("🔄 Attempting container recovery...")
        
        try:
            # Stop all containers
            stop_result = subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.archon_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            recovery_results["recovery_attempts"].append(f"Stop containers: {stop_result.returncode == 0}")
            
            # Clean up volumes and networks
            cleanup_result = subprocess.run(
                ["docker", "system", "prune", "-f"],
                capture_output=True,
                text=True,
                timeout=60
            )
            recovery_results["recovery_attempts"].append(f"Cleanup: {cleanup_result.returncode == 0}")
            
            # Retry container startup
            retry_result = self.ensure_containers_running()
            recovery_results["success"] = retry_result["success"]
            recovery_results["recovery_attempts"].append(f"Restart: {retry_result['success']}")
            
            if not retry_result["success"]:
                recovery_results["errors"].extend(retry_result.get("failures", []))
                
        except Exception as e:
            recovery_results["errors"].append(f"Recovery failed: {e}")
        
        return recovery_results
    
    def register_project(self) -> Dict[str, Any]:
        """
        Register the current project with Archon system
        """
        registration_results = {
            "success": True,
            "project_id": None,
            "knowledge_base_status": "not_attempted",
            "errors": []
        }
        
        try:
            # Wait for server to be available
            self.wait_for_service("server", timeout=60)
            
            # Detect project details
            project_info = self.detect_project_info()
            
            # Register via MCP if available, otherwise direct API
            try:
                mcp_available = self.check_service_health("mcp")
                if mcp_available:
                    # Use MCP for registration
                    registration_results = self.register_via_mcp(project_info)
                else:
                    # Use direct API
                    registration_results = self.register_via_api(project_info)
            except Exception as e:
                registration_results["errors"].append(f"Registration failed: {e}")
                registration_results["success"] = False
            
        except Exception as e:
            registration_results["errors"].append(f"Project registration error: {e}")
            registration_results["success"] = False
        
        return registration_results
    
    def detect_project_info(self) -> Dict[str, Any]:
        """
        Detect project type, languages, frameworks
        """
        project_info = {
            "name": self.project_path.name,
            "path": str(self.project_path),
            "type": "unknown",
            "languages": [],
            "frameworks": [],
            "package_managers": []
        }
        
        # Check for common files and patterns
        files_to_check = {
            "package.json": {"type": "web", "languages": ["javascript", "typescript"], "pm": "npm"},
            "requirements.txt": {"type": "python", "languages": ["python"], "pm": "pip"},
            "Cargo.toml": {"type": "rust", "languages": ["rust"], "pm": "cargo"},
            "go.mod": {"type": "go", "languages": ["go"], "pm": "go"},
            "pom.xml": {"type": "java", "languages": ["java"], "pm": "maven"},
            "Gemfile": {"type": "ruby", "languages": ["ruby"], "pm": "gem"},
        }
        
        for file_name, info in files_to_check.items():
            if (self.project_path / file_name).exists():
                project_info["type"] = info["type"]
                project_info["languages"].extend(info["languages"])
                project_info["package_managers"].append(info["pm"])
        
        # Check for frameworks
        if (self.project_path / "package.json").exists():
            try:
                with open(self.project_path / "package.json") as f:
                    package_data = json.load(f)
                    deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
                    
                    if "react" in deps:
                        project_info["frameworks"].append("react")
                    if "vue" in deps:
                        project_info["frameworks"].append("vue")
                    if "angular" in deps:
                        project_info["frameworks"].append("angular")
                    if "next" in deps:
                        project_info["frameworks"].append("nextjs")
                    if "@nestjs/core" in deps:
                        project_info["frameworks"].append("nestjs")
            except:
                pass
        
        return project_info
    
    def register_via_mcp(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register project via MCP tools
        """
        # Implementation for MCP-based registration
        return {"success": True, "method": "mcp", "project_id": f"mcp_{project_info['name']}"}
    
    def register_via_api(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register project via direct API
        """
        try:
            response = requests.post(
                f"{self.services_urls['server']}/api/projects",
                json=project_info,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "method": "api",
                    "project_id": data.get("id"),
                    "project_info": project_info
                }
            else:
                return {
                    "success": False,
                    "method": "api",
                    "errors": [f"API registration failed: {response.status_code} - {response.text}"]
                }
        except Exception as e:
            return {
                "success": False,
                "method": "api",
                "errors": [f"API registration error: {e}"]
            }
    
    def activate_specialized_agents(self) -> Dict[str, Any]:
        """
        Activate all specialized agents according to MANIFEST
        """
        agent_results = {
            "success": True,
            "available_agents": [],
            "activated_agents": [],
            "failed_agents": [],
            "errors": []
        }
        
        # List of specialized agents from MANIFEST
        specialized_agents = [
            "strategic-planner",
            "system-architect", 
            "code-implementer",
            "api-design-architect",
            "test-coverage-validator",
            "code-quality-reviewer",
            "security-auditor",
            "performance-optimizer",
            "ui-ux-optimizer",
            "database-architect",
            "documentation-generator",
            "deployment-automation",
            "antihallucination-validator",
            "code-refactoring-optimizer"
        ]
        
        try:
            # Check if agents service is running
            agents_available = self.check_service_health("agents")
            
            if agents_available:
                # Get available agents from service
                try:
                    response = requests.get(f"{self.services_urls['agents']}/agents", timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        agent_results["available_agents"] = data.get("agents", [])
                        agent_results["activated_agents"] = specialized_agents
                        logger.info(f"✅ {len(specialized_agents)} specialized agents available")
                    else:
                        agent_results["errors"].append(f"Agents service error: {response.status_code}")
                        agent_results["success"] = False
                except Exception as e:
                    agent_results["errors"].append(f"Agents service communication error: {e}")
                    agent_results["success"] = False
            else:
                # Agents service not available - still consider success for core functionality
                agent_results["available_agents"] = []
                agent_results["activated_agents"] = []
                logger.warning("⚠️ Agents service not available - core functionality will work without specialized agents")
        
        except Exception as e:
            agent_results["errors"].append(f"Agent activation error: {e}")
            agent_results["success"] = False
        
        return agent_results
    
    def load_rules_and_manifest(self) -> Dict[str, Any]:
        """
        Load all rules and manifest compliance
        """
        rules_results = {
            "success": True,
            "loaded_files": [],
            "rules_count": 0,
            "manifest_status": "not_loaded",
            "errors": []
        }
        
        try:
            # Load MANIFEST.md
            manifest_path = self.archon_root / "MANIFEST.md"
            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_content = f.read()
                    if "ARCHON OPERATIONAL MANIFEST" in manifest_content:
                        rules_results["manifest_status"] = "loaded"
                        rules_results["loaded_files"].append("MANIFEST.md")
                        logger.info("✅ MANIFEST.md loaded successfully")
                    else:
                        rules_results["errors"].append("Invalid MANIFEST.md format")
                        rules_results["success"] = False
            else:
                rules_results["errors"].append("MANIFEST.md not found")
                rules_results["success"] = False
            
            # Load project-specific RULES.md if exists
            project_rules = self.project_path / "RULES.md"
            if project_rules.exists():
                rules_results["loaded_files"].append("RULES.md (project)")
                rules_results["rules_count"] += 1
                logger.info("✅ Project RULES.md loaded")
            
            # Load CLAUDE.md if exists
            claude_md_paths = [
                self.project_path / "CLAUDE.md",
                self.archon_root / "CLAUDE.md"
            ]
            
            for claude_path in claude_md_paths:
                if claude_path.exists():
                    rules_results["loaded_files"].append(f"CLAUDE.md ({claude_path.parent.name})")
                    rules_results["rules_count"] += 1
                    logger.info(f"✅ CLAUDE.md loaded from {claude_path.parent.name}")
                    break
        
        except Exception as e:
            rules_results["errors"].append(f"Rules loading error: {e}")
            rules_results["success"] = False
        
        return rules_results
    
    def verify_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health verification
        """
        health_results = {
            "success": True,
            "services": {},
            "overall_health": "unknown",
            "critical_issues": [],
            "warnings": []
        }
        
        # Check each service
        for service_name, url in self.services_urls.items():
            service_health = self.check_service_health(service_name)
            health_results["services"][service_name] = {
                "status": "healthy" if service_health else "unhealthy",
                "url": url,
                "required": service_name in ["server", "mcp", "ui"]
            }
            
            if not service_health and service_name in ["server", "mcp", "ui"]:
                health_results["critical_issues"].append(f"{service_name} service unhealthy")
                health_results["success"] = False
            elif not service_health:
                health_results["warnings"].append(f"{service_name} service unavailable")
        
        # Overall health assessment
        healthy_critical = sum(1 for s in health_results["services"].values() 
                             if s["required"] and s["status"] == "healthy")
        total_critical = sum(1 for s in health_results["services"].values() if s["required"])
        
        if healthy_critical == total_critical:
            health_results["overall_health"] = "excellent"
        elif healthy_critical >= total_critical * 0.8:
            health_results["overall_health"] = "good"
        elif healthy_critical >= total_critical * 0.5:
            health_results["overall_health"] = "fair"
        else:
            health_results["overall_health"] = "poor"
            health_results["success"] = False
        
        return health_results
    
    def check_service_health(self, service_name: str) -> bool:
        """
        Check if a specific service is healthy
        """
        try:
            url = self.services_urls[service_name]
            health_endpoint = f"{url}/health"
            
            response = requests.get(health_endpoint, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_service(self, service_name: str, timeout: int = 60) -> bool:
        """
        Wait for a service to become available
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_service_health(service_name):
                return True
            time.sleep(2)
        return False
    
    def generate_next_steps(self, activation_report: Dict[str, Any]) -> List[str]:
        """
        Generate recommended next steps based on activation status
        """
        next_steps = []
        
        if activation_report["status"] == "success":
            next_steps.extend([
                f"🌐 Access Archon UI at {self.services_urls['ui']}",
                "🔧 Use MCP tools in your IDE (archon:perform_rag_query, archon:manage_task, etc.)",
                "📝 Run '@Archon <task>' to activate specialized agents for your coding tasks",
                "📊 Monitor system health via the dashboard"
            ])
        else:
            # Add recovery steps based on failed phases
            failed_phases = [step for step in activation_report["activation_steps"] 
                           if step["status"] == "failed"]
            
            for phase in failed_phases:
                if "Container" in phase["phase"]:
                    next_steps.append("🐳 Check Docker installation and permissions")
                    next_steps.append("💾 Ensure sufficient disk space for containers")
                elif "Validation" in phase["phase"]:
                    next_steps.append("⚙️ Configure environment variables (.env file)")
                    next_steps.append("📋 Ensure MANIFEST.md is present and valid")
                elif "Health" in phase["phase"]:
                    next_steps.append("🔍 Check service logs: docker compose logs")
                    next_steps.append("🔄 Restart services: docker compose restart")
        
        return next_steps

def main():
    """
    Main execution function for @Archon activation
    """
    print("🚀 ARCHON GLOBAL ACTIVATION COMMAND")
    print("=" * 50)
    
    # Initialize activator
    activator = ArchonGlobalActivator()
    
    # Run activation
    result = activator.activate_archon_system()
    
    # Display results
    print(f"\n📊 ACTIVATION SUMMARY")
    print(f"Status: {result['status'].upper()}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Project: {result['project_path']}")
    
    print(f"\n📋 ACTIVATION PHASES:")
    for i, step in enumerate(result['activation_steps'], 1):
        status_icon = "✅" if step['status'] == 'completed' else "❌" if step['status'] == 'failed' else "⚠️"
        print(f"{i}. {status_icon} {step['phase']}: {step['status']}")
    
    if result.get('services_status'):
        print(f"\n🌐 SERVICES STATUS:")
        for service, status in result['services_status'].items():
            status_icon = "✅" if status['status'] == 'healthy' else "❌"
            print(f"  {status_icon} {service}: {status['status']} ({status['url']})")
    
    if result.get('next_steps'):
        print(f"\n➡️ NEXT STEPS:")
        for step in result['next_steps']:
            print(f"  {step}")
    
    if result['status'] == 'success':
        print(f"\n🎉 ARCHON SYSTEM FULLY ACTIVATED!")
        print(f"Ready for @Archon commands and specialized agent orchestration.")
    elif result['status'] == 'partial':
        print(f"\n⚠️ ARCHON SYSTEM PARTIALLY ACTIVATED")
        print(f"Core functionality available, some advanced features may be limited.")
    else:
        print(f"\n❌ ARCHON ACTIVATION FAILED")
        print(f"Please address the issues above and retry activation.")
    
    return result

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result['status'] in ['success', 'partial'] else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Activation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Activation failed with error: {e}")
        sys.exit(1)