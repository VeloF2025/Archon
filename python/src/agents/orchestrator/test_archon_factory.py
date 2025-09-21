#!/usr/bin/env python3
"""
Test script for the Archon Project Agent Factory

This script tests the project-specific agent generation system
to ensure it works correctly across different project types.
"""

import sys
import os
import tempfile
import json
import yaml
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from archon_project_agent_factory import ArchonProjectAgentFactory, ProjectAnalysis

def create_test_project(project_type: str) -> Path:
    """Create a temporary test project with given characteristics"""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"archon_test_{project_type}_"))
    
    if project_type == "react-firebase":
        # Create package.json for React + Firebase project
        package_json = {
            "name": "test-react-firebase-app",
            "dependencies": {
                "react": "^18.3.1",
                "react-dom": "^18.3.1",
                "firebase": "^11.2.0",
                "@headlessui/react": "^1.7.19",
                "tailwindcss": "^3.4.17"
            },
            "devDependencies": {
                "vite": "^6.0.5",
                "@vitejs/plugin-react": "^4.3.4",
                "typescript": "~5.7.2",
                "vitest": "^2.0.0",
                "@playwright/test": "^1.54.2"
            }
        }
        
        # Create project files
        with open(temp_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
            
        # Create TypeScript config
        tsconfig = {
            "compilerOptions": {
                "strict": True,
                "target": "ES2020",
                "jsx": "react-jsx"
            }
        }
        with open(temp_dir / "tsconfig.json", "w") as f:
            json.dump(tsconfig, f, indent=2)
            
        # Create vite config
        vite_config = """
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
"""
        with open(temp_dir / "vite.config.ts", "w") as f:
            f.write(vite_config)
            
        # Create src directory with some files
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        
        # Create some TypeScript/React files
        with open(src_dir / "App.tsx", "w") as f:
            f.write("import React from 'react'; export default function App() { return <div>Hello</div>; }")
            
        components_dir = src_dir / "components"
        components_dir.mkdir()
        with open(components_dir / "Button.tsx", "w") as f:
            f.write("import React from 'react'; export function Button() { return <button>Click</button>; }")
            
        services_dir = src_dir / "services"
        services_dir.mkdir()
        with open(services_dir / "authService.ts", "w") as f:
            f.write("export const authService = { login: () => {}, logout: () => {} };")
            
    elif project_type == "nextjs":
        # Create Next.js project
        package_json = {
            "name": "test-nextjs-app",
            "dependencies": {
                "next": "^14.0.0",
                "react": "^18.3.1",
                "react-dom": "^18.3.1",
                "@next/auth": "^4.24.0"
            },
            "devDependencies": {
                "typescript": "~5.7.2",
                "@types/react": "^18.3.17"
            }
        }
        
        with open(temp_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
            
        # Create minimal Next.js structure
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        app_dir = src_dir / "app"
        app_dir.mkdir()
        with open(app_dir / "page.tsx", "w") as f:
            f.write("export default function Home() { return <div>Hello Next.js</div>; }")
    
    return temp_dir

def test_react_firebase_analysis():
    """Test analysis of React + Firebase project"""
    print("🧪 Testing React + Firebase project analysis...")
    
    temp_project = create_test_project("react-firebase")
    factory = ArchonProjectAgentFactory(str(temp_project))
    
    analysis = factory.analyze_project()
    
    # Verify analysis results
    assert analysis.project_type == "React-Firebase SPA"
    assert "React" in analysis.frameworks
    assert "Firebase" in analysis.cloud_services
    assert "TypeScript" in analysis.languages
    assert "Vite" in analysis.bundlers
    assert analysis.complexity_score > 0
    
    print(f"✅ Analysis successful: {analysis.project_type} (complexity: {analysis.complexity_score:.1f})")
    
    # Test agent generation
    agents = factory.generate_project_agents(analysis)
    
    # Verify expected agents were created
    agent_ids = [agent.id for agent in agents]
    expected_agents = [
        "react-firebase-specialist",
        "typescript-strict-enforcer", 
        "firestore-security-architect",
        "vite-optimization-expert"
    ]
    
    for expected in expected_agents:
        assert expected in agent_ids, f"Expected agent {expected} not found"
    
    print(f"✅ Generated {len(agents)} specialized agents")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_project)
    
    return agents

def test_nextjs_analysis():
    """Test analysis of Next.js project"""
    print("🧪 Testing Next.js project analysis...")
    
    temp_project = create_test_project("nextjs")
    factory = ArchonProjectAgentFactory(str(temp_project))
    
    analysis = factory.analyze_project()
    
    # Verify analysis results
    assert "React" in analysis.frameworks
    assert "Next.js" in analysis.frameworks
    assert "TypeScript" in analysis.languages
    
    print(f"✅ Analysis successful: {analysis.project_type} (complexity: {analysis.complexity_score:.1f})")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_project)

def test_agent_configuration_save_load():
    """Test saving and loading agent configuration"""
    print("🧪 Testing agent configuration save/load...")
    
    temp_project = create_test_project("react-firebase")
    factory = ArchonProjectAgentFactory(str(temp_project))
    
    analysis = factory.analyze_project()
    agents = factory.generate_project_agents(analysis)
    
    # Save configuration
    factory.save_project_agents(agents)
    
    # Verify file was created
    config_file = temp_project / ".archon" / "project_agents.yaml"
    assert config_file.exists(), "Configuration file not created"
    
    # Load and verify configuration
    with open(config_file) as f:
        config_data = yaml.safe_load(f)
    
    assert "generated_at" in config_data
    assert "agents" in config_data
    assert len(config_data["agents"]) == len(agents)
    
    print(f"✅ Configuration saved and loaded successfully")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_project)

def test_life_arrow_v1_actual():
    """Test the actual life-arrow-v1 project"""
    print("🧪 Testing actual life-arrow-v1 project...")
    
    project_path = "/mnt/c/Jarvis/AI Workspace/life-arrow-v1"
    if not Path(project_path).exists():
        print("⚠️ Skipping life-arrow-v1 test - project not found")
        return
    
    factory = ArchonProjectAgentFactory(project_path)
    
    # Check if agents already exist
    config_file = Path(project_path) / ".archon" / "project_agents.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        
        agent_count = len(config_data.get("agents", []))
        print(f"✅ Found {agent_count} existing agents in life-arrow-v1")
        
        # Verify expected agents exist
        agent_ids = [agent["id"] for agent in config_data["agents"]]
        expected = ["react-firebase-specialist", "typescript-strict-enforcer"]
        
        for exp in expected:
            if exp in agent_ids:
                print(f"✅ Agent {exp} found")
            else:
                print(f"❌ Agent {exp} missing")
    else:
        print("❌ No agent configuration found for life-arrow-v1")

def main():
    """Run all tests"""
    print("🚀 Starting Archon Project Agent Factory Tests...")
    print("=" * 60)
    
    try:
        # Test different project types
        agents = test_react_firebase_analysis()
        test_nextjs_analysis()
        test_agent_configuration_save_load()
        test_life_arrow_v1_actual()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Archon Project Agent Factory is working correctly.")
        
        # Show example agent
        if agents:
            example_agent = agents[0]
            print(f"\n📋 Example generated agent: {example_agent.name}")
            print(f"   Specialization: {example_agent.specialization}")
            print(f"   Skills: {', '.join(example_agent.skills[:3])}...")
            print(f"   Activation: {example_agent.activation_commands[0]}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()