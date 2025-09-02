"""
Fix specialized agents to work with BaseAgent's _agent pattern
"""

import re

# Read the file
with open("python/src/agents/specialized_agents.py", "r") as f:
    content = f.read()

# Fix the __init__ methods to not set self.agent
content = re.sub(
    r'(def __init__\(self, model: str = "openai:gpt-4o-mini"\):.*?\n\s+super\(\).__init__\(.*?\))\n\s+self\.agent = self\._create_agent\(\)',
    r'\1',
    content,
    flags=re.DOTALL
)

# Also fix the run method to use self._agent instead of self.agent
content = re.sub(
    r'result = await self\.agent\.run\(prompt, deps=deps\)',
    r'result = await self._agent.run(prompt, deps=deps)',
    content
)

# Write the fixed content back
with open("python/src/agents/specialized_agents.py", "w") as f:
    f.write(content)

print("Fixed all specialized agents to use BaseAgent's _agent pattern")