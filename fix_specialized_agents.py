"""
Fix specialized agents to implement abstract methods
"""

import re

# Read the file
with open("python/src/agents/specialized_agents.py", "r") as f:
    content = f.read()

# Pattern to find agent classes that need fixing
pattern = r'class (\w+Agent)\(BaseAgent\):(.*?)(?=class \w+Agent\(BaseAgent\)|# Agent Registry|$)'

def fix_agent_class(match):
    class_name = match.group(1)
    class_content = match.group(2)
    
    # Skip if already fixed (has _create_agent)
    if '_create_agent' in class_content:
        return match.group(0)
    
    # Extract the docstring
    docstring_match = re.search(r'"""(.*?)"""', class_content, re.DOTALL)
    docstring = docstring_match.group(0) if docstring_match else '""""""'
    
    # Extract the system prompt from the old __init__
    system_prompt_match = re.search(r'system_prompt="""(.*?)"""', class_content, re.DOTALL)
    system_prompt = system_prompt_match.group(1) if system_prompt_match else "Specialized agent"
    
    # Create the fixed class
    fixed_class = f'''class {class_name}(BaseAgent):
    {docstring}
    
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        super().__init__(name="{class_name}", model=model)
        self.agent = self._create_agent()
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent"""
        return Agent(
            self.model,
            system_prompt=self.get_system_prompt(),
            deps_type=SpecializedAgentDependencies
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        return """{system_prompt}"""
    
    async def run(self, prompt: str, deps: SpecializedAgentDependencies) -> str:
        try:
            result = await self.agent.run(prompt, deps=deps)
            return result.data
        except Exception as e:
            logger.error(f"{class_name} error: {{e}}")
            return f"Error: {{e}}"
'''
    
    return fixed_class

# Fix all remaining agent classes
content = re.sub(pattern, fix_agent_class, content, flags=re.DOTALL)

# Write the fixed content back
with open("python/src/agents/specialized_agents.py", "w") as f:
    f.write(content)

print("Fixed all specialized agents to implement abstract methods")