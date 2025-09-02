"""
Fix all specialized agents to use proper generic types
"""

import re

# Read the file
with open("python/src/agents/specialized_agents.py", "r") as f:
    content = f.read()

# Pattern to find and fix agent class declarations
# Replace all class definitions to include generic types
agent_classes = [
    "TypeScriptFrontendAgent",
    "SecurityAuditorAgent", 
    "TestGeneratorAgent",
    "CodeReviewerAgent",
    "DocumentationWriterAgent",
    "SystemArchitectAgent",
    "DatabaseDesignerAgent",
    "DevOpsEngineerAgent",
    "PerformanceOptimizerAgent",
    "APIIntegratorAgent",
    "UIUXDesignerAgent",
    "RefactoringSpecialistAgent",
    "TechnicalWriterAgent",
    "IntegrationTesterAgent",
    "DeploymentCoordinatorAgent",
    "MonitoringAgent",
    "DataAnalystAgent",
    "HRMReasoningAgent"
]

for agent_class in agent_classes:
    # Update class declaration to include generic types
    content = re.sub(
        f'class {agent_class}\\(BaseAgent\\):',
        f'class {agent_class}(BaseAgent[SpecializedAgentDependencies, str]):',
        content
    )

# Write back
with open("python/src/agents/specialized_agents.py", "w") as f:
    f.write(content)

print("Fixed all agent classes to use proper generic types")