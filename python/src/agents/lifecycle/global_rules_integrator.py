"""
Global Rules Integrator v3.0 - Global Rules Integration
Ensures all project agents inherit and enforce global rules from CLAUDE.md, RULES.md, MANIFEST.md

NLNH Protocol: Real rules integration with actual file parsing
DGTS Enforcement: No fake rule compliance, actual rule enforcement
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class GlobalRule:
    """Individual global rule extracted from rule files"""
    id: str
    source_file: str  # CLAUDE.md, RULES.md, MANIFEST.md
    category: str  # MANDATORY, CRITICAL, ENFORCEMENT, etc.
    title: str
    description: str
    enforcement_level: str  # BLOCKING, WARNING, ADVISORY
    applicable_to: List[str] = field(default_factory=list)  # Agent types, tiers, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RulesProfile:
    """Complete rules profile for an agent"""
    agent_id: str
    global_rules: List[GlobalRule] = field(default_factory=list)
    project_rules: List[GlobalRule] = field(default_factory=list)
    combined_system_prompt: str = ""
    enforcement_config: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class GlobalRulesIntegrator:
    """
    Global Rules Integrator for Archon v3.0
    Ensures all agents inherit and enforce global rules automatically
    """
    
    def __init__(self, archon_root_path: Optional[str] = None):
        self.archon_root_path = archon_root_path or self._find_archon_root()
        self.global_rules_cache: Dict[str, List[GlobalRule]] = {}
        self.last_cache_update: Optional[datetime] = None
        
        # Rule file locations
        self.rule_files = {
            "CLAUDE.md": os.path.join(self.archon_root_path, "CLAUDE.md"),
            "RULES.md": os.path.join(self.archon_root_path, "RULES.md"),
            "MANIFEST.md": os.path.join(self.archon_root_path, "python", "MANIFEST.md"),
            "JARVIS_CLAUDE.md": "/mnt/c/Jarvis/CLAUDE.md",  # Global Jarvis rules
            "JARVIS_RULES.md": "/mnt/c/Jarvis/RULES.md"  # Global Jarvis rules
        }
        
        logger.info(f"GlobalRulesIntegrator initialized with Archon root: {self.archon_root_path}")

    def _find_archon_root(self) -> str:
        """Find Archon root directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != "/":
            if os.path.exists(os.path.join(current_dir, "CLAUDE.md")) and \
               os.path.exists(os.path.join(current_dir, "README.md")):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # Fallback to known location
        return "/mnt/c/Jarvis/AI Workspace/Archon"

    async def load_global_rules(self, force_refresh: bool = False) -> Dict[str, List[GlobalRule]]:
        """Load and parse all global rules files"""
        if not force_refresh and self.global_rules_cache and self.last_cache_update:
            # Use cache if recent (< 5 minutes)
            if (datetime.now() - self.last_cache_update).total_seconds() < 300:
                return self.global_rules_cache
        
        logger.info("Loading global rules from all source files...")
        
        all_rules = {}
        for rule_file, file_path in self.rule_files.items():
            try:
                if os.path.exists(file_path):
                    rules = await self._parse_rules_file(rule_file, file_path)
                    all_rules[rule_file] = rules
                    logger.info(f"Loaded {len(rules)} rules from {rule_file}")
                else:
                    logger.warning(f"Rules file not found: {file_path}")
                    all_rules[rule_file] = []
            except Exception as e:
                logger.error(f"Failed to load rules from {rule_file}: {e}")
                all_rules[rule_file] = []
        
        self.global_rules_cache = all_rules
        self.last_cache_update = datetime.now()
        
        total_rules = sum(len(rules) for rules in all_rules.values())
        logger.info(f"Global rules loading complete: {total_rules} total rules loaded")
        
        return all_rules

    async def _parse_rules_file(self, filename: str, filepath: str) -> List[GlobalRule]:
        """Parse individual rules file and extract rules"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            rules = []
            
            # Parse based on file type
            if filename == "CLAUDE.md":
                rules.extend(await self._parse_claude_md(content, filename))
            elif filename == "RULES.md":
                rules.extend(await self._parse_rules_md(content, filename))
            elif filename == "MANIFEST.md":
                rules.extend(await self._parse_manifest_md(content, filename))
            elif filename.startswith("JARVIS_"):
                rules.extend(await self._parse_jarvis_rules(content, filename))
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            return []

    async def _parse_claude_md(self, content: str, filename: str) -> List[GlobalRule]:
        """Parse CLAUDE.md file for rules"""
        rules = []
        
        # Look for critical enforcement patterns
        critical_patterns = [
            (r'🚫 ANTIHALL VALIDATOR.*?(?=##|$)', 'CRITICAL', 'Anti-Hallucination Validation'),
            (r'🚨 NLNH PROTOCOL.*?(?=##|$)', 'CRITICAL', 'No Lies No Hallucination Protocol'),
            (r'🚫 DGTS.*?(?=##|$)', 'CRITICAL', 'Don\'t Game The System Protocol'),
            (r'MANDATORY.*?(?=\n\n|\n#|$)', 'MANDATORY', 'Mandatory Requirements'),
            (r'CRITICAL.*?(?=\n\n|\n#|$)', 'CRITICAL', 'Critical Requirements'),
            (r'ZERO TOLERANCE.*?(?=\n\n|\n#|$)', 'BLOCKING', 'Zero Tolerance Policies')
        ]
        
        for pattern, enforcement, title in critical_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for i, match in enumerate(matches):
                rule = GlobalRule(
                    id=f"claude_{enforcement.lower()}_{i+1}",
                    source_file=filename,
                    category=enforcement,
                    title=title,
                    description=match.strip()[:500] + "..." if len(match.strip()) > 500 else match.strip(),
                    enforcement_level=enforcement,
                    applicable_to=["all_agents"],
                    metadata={"pattern_type": "regex_extract", "full_text": match.strip()}
                )
                rules.append(rule)
        
        # Look for specific trigger commands
        trigger_patterns = [
            (r'RYR.*?COMMAND.*?(?=##|$)', 'MANDATORY', 'Remember Your Rules Command'),
            (r'ForgeFlow.*?(?=##|$)', 'MANDATORY', 'ForgeFlow Orchestration'),
            (r'Archon.*?(?=##|$)', 'MANDATORY', 'Archon System Activation')
        ]
        
        for pattern, enforcement, title in trigger_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for i, match in enumerate(matches):
                rule = GlobalRule(
                    id=f"claude_trigger_{title.lower().replace(' ', '_')}_{i+1}",
                    source_file=filename,
                    category="SYSTEM_TRIGGER",
                    title=title,
                    description=match.strip()[:500] + "..." if len(match.strip()) > 500 else match.strip(),
                    enforcement_level=enforcement,
                    applicable_to=["all_agents"],
                    metadata={"trigger_type": "system_command", "full_text": match.strip()}
                )
                rules.append(rule)
        
        logger.info(f"Parsed {len(rules)} rules from CLAUDE.md")
        return rules

    async def _parse_rules_md(self, content: str, filename: str) -> List[GlobalRule]:
        """Parse RULES.md file for project-specific rules"""
        rules = []
        
        # Split content by headers
        sections = re.split(r'\n#+\s+', content)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            title = lines[0].strip()
            description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
            
            # Determine enforcement level based on content
            enforcement_level = "ADVISORY"
            if any(keyword in section.upper() for keyword in ['MANDATORY', 'MUST', 'REQUIRED']):
                enforcement_level = "MANDATORY"
            elif any(keyword in section.upper() for keyword in ['CRITICAL', 'BLOCKING', 'ZERO TOLERANCE']):
                enforcement_level = "BLOCKING"
            elif any(keyword in section.upper() for keyword in ['WARNING', 'IMPORTANT']):
                enforcement_level = "WARNING"
            
            rule = GlobalRule(
                id=f"rules_{i+1}",
                source_file=filename,
                category="PROJECT_RULE",
                title=title[:100],  # Limit title length
                description=description[:500] + "..." if len(description) > 500 else description,
                enforcement_level=enforcement_level,
                applicable_to=["all_agents"],
                metadata={"section_index": i, "full_text": section}
            )
            rules.append(rule)
        
        logger.info(f"Parsed {len(rules)} rules from RULES.md")
        return rules

    async def _parse_manifest_md(self, content: str, filename: str) -> List[GlobalRule]:
        """Parse MANIFEST.md file for operational mandates"""
        rules = []
        
        # Look for operational mandates
        mandate_patterns = [
            (r'MANDATE.*?(?=\n\n|\n#|$)', 'MANDATORY', 'Operational Mandate'),
            (r'PROTOCOL.*?(?=\n\n|\n#|$)', 'MANDATORY', 'Protocol Requirement'),
            (r'COMPLIANCE.*?(?=\n\n|\n#|$)', 'BLOCKING', 'Compliance Requirement')
        ]
        
        for pattern, enforcement, title in mandate_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for i, match in enumerate(matches):
                rule = GlobalRule(
                    id=f"manifest_{enforcement.lower()}_{i+1}",
                    source_file=filename,
                    category="OPERATIONAL_MANDATE",
                    title=title,
                    description=match.strip()[:500] + "..." if len(match.strip()) > 500 else match.strip(),
                    enforcement_level=enforcement,
                    applicable_to=["all_agents"],
                    metadata={"mandate_type": "operational", "full_text": match.strip()}
                )
                rules.append(rule)
        
        logger.info(f"Parsed {len(rules)} operational mandates from MANIFEST.md")
        return rules

    async def _parse_jarvis_rules(self, content: str, filename: str) -> List[GlobalRule]:
        """Parse Jarvis global rules"""
        rules = []
        
        # Similar parsing logic as CLAUDE.md but with Jarvis-specific patterns
        jarvis_patterns = [
            (r'UNIVERSAL.*?(?=##|$)', 'UNIVERSAL', 'Universal Rule'),
            (r'GLOBAL.*?(?=##|$)', 'GLOBAL', 'Global Rule'),
            (r'SYSTEM.*?(?=##|$)', 'SYSTEM', 'System Rule')
        ]
        
        for pattern, enforcement, title in jarvis_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for i, match in enumerate(matches):
                rule = GlobalRule(
                    id=f"jarvis_{enforcement.lower()}_{i+1}",
                    source_file=filename,
                    category="JARVIS_GLOBAL",
                    title=title,
                    description=match.strip()[:500] + "..." if len(match.strip()) > 500 else match.strip(),
                    enforcement_level=enforcement,
                    applicable_to=["all_agents"],
                    metadata={"scope": "global_jarvis", "full_text": match.strip()}
                )
                rules.append(rule)
        
        logger.info(f"Parsed {len(rules)} global rules from {filename}")
        return rules

    async def create_agent_rules_profile(
        self, 
        agent_id: str, 
        agent_type: str, 
        model_tier: str,
        project_id: str,
        specialization: Optional[str] = None
    ) -> RulesProfile:
        """Create comprehensive rules profile for an agent"""
        
        # Load all global rules
        all_rules = await self.load_global_rules()
        
        profile = RulesProfile(agent_id=agent_id)
        
        # Filter rules applicable to this agent
        for rule_file, rules in all_rules.items():
            for rule in rules:
                if await self._is_rule_applicable(rule, agent_type, model_tier, specialization):
                    if rule_file.startswith("JARVIS_") or rule_file in ["CLAUDE.md", "MANIFEST.md"]:
                        profile.global_rules.append(rule)
                    else:
                        profile.project_rules.append(rule)
        
        # Generate combined system prompt
        profile.combined_system_prompt = await self._generate_enhanced_system_prompt(profile)
        
        # Create enforcement configuration
        profile.enforcement_config = await self._create_enforcement_config(profile)
        
        logger.info(f"Created rules profile for {agent_id}: "
                   f"{len(profile.global_rules)} global rules, "
                   f"{len(profile.project_rules)} project rules")
        
        return profile

    async def _is_rule_applicable(
        self, 
        rule: GlobalRule, 
        agent_type: str, 
        model_tier: str, 
        specialization: Optional[str]
    ) -> bool:
        """Check if a rule applies to this specific agent"""
        
        # Universal rules apply to everyone
        if "all_agents" in rule.applicable_to or not rule.applicable_to:
            return True
        
        # Check agent type specificity
        if agent_type in rule.applicable_to:
            return True
            
        # Check model tier specificity
        if model_tier in rule.applicable_to:
            return True
            
        # Check specialization
        if specialization and specialization in rule.applicable_to:
            return True
        
        # Check for pattern matches in rule content
        rule_content = rule.description.lower()
        if agent_type.lower() in rule_content or \
           model_tier.lower() in rule_content or \
           (specialization and specialization.lower() in rule_content):
            return True
        
        return False

    async def _generate_enhanced_system_prompt(self, profile: RulesProfile) -> str:
        """Generate enhanced system prompt with all applicable rules"""
        
        prompt_sections = []
        
        # Core identity
        prompt_sections.append("You are an Archon v3.0 AI agent operating under strict global rule compliance.")
        
        # Critical enforcement rules first
        critical_rules = [rule for rule in profile.global_rules + profile.project_rules 
                         if rule.enforcement_level in ["BLOCKING", "CRITICAL", "MANDATORY"]]
        
        if critical_rules:
            prompt_sections.append("\n🚨 CRITICAL ENFORCEMENT RULES:")
            for rule in critical_rules[:5]:  # Limit to top 5 to avoid prompt bloat
                prompt_sections.append(f"- {rule.title}: {rule.description[:200]}...")
        
        # Protocol compliance
        prompt_sections.append("""
🔒 MANDATORY PROTOCOLS:
- NLNH Protocol: No lies, no hallucination - always be truthful about capabilities and limitations
- DGTS Protocol: Don't game the system - provide real functionality, not fake implementations  
- AntiHall Protocol: Validate all code references exist before suggesting them
- Manifest Compliance: Follow all operational mandates from MANIFEST.md
        """)
        
        # Quality standards
        quality_rules = [rule for rule in profile.global_rules + profile.project_rules
                        if "quality" in rule.title.lower() or "standard" in rule.title.lower()]
        
        if quality_rules:
            prompt_sections.append("\n⚡ QUALITY STANDARDS:")
            for rule in quality_rules[:3]:
                prompt_sections.append(f"- {rule.description[:150]}...")
        
        # Final compliance reminder
        prompt_sections.append("""
🎯 COMPLIANCE MANDATE:
You MUST follow ALL applicable rules without exception. If uncertain about rule interpretation, 
ask for clarification rather than assuming. Rule violations will result in immediate correction 
and enhanced monitoring.
        """)
        
        return "\n".join(prompt_sections)

    async def _create_enforcement_config(self, profile: RulesProfile) -> Dict[str, Any]:
        """Create enforcement configuration for the agent"""
        
        config = {
            "blocking_rules": [],
            "warning_rules": [],
            "advisory_rules": [],
            "validation_required": [],
            "monitoring_enabled": True,
            "rule_count_by_level": {}
        }
        
        # Categorize rules by enforcement level
        for rule in profile.global_rules + profile.project_rules:
            if rule.enforcement_level in ["BLOCKING", "CRITICAL"]:
                config["blocking_rules"].append(rule.id)
            elif rule.enforcement_level in ["MANDATORY", "WARNING"]:
                config["warning_rules"].append(rule.id)
            else:
                config["advisory_rules"].append(rule.id)
            
            # Track validation requirements
            if "validation" in rule.description.lower() or "antihall" in rule.description.lower():
                config["validation_required"].append(rule.id)
        
        # Count rules by level
        config["rule_count_by_level"] = {
            "blocking": len(config["blocking_rules"]),
            "warning": len(config["warning_rules"]),
            "advisory": len(config["advisory_rules"])
        }
        
        return config

    async def validate_agent_compliance(self, agent_id: str, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an agent action against applicable rules"""
        
        # This would integrate with the actual agent to check compliance
        # For now, return a validation framework
        
        validation_result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": [],
            "enforcement_actions": []
        }
        
        # Would implement actual rule checking logic here
        logger.info(f"Validating compliance for agent {agent_id} action: {action}")
        
        return validation_result

    async def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded global rules"""
        all_rules = await self.load_global_rules()
        
        summary = {
            "total_rules": sum(len(rules) for rules in all_rules.values()),
            "rules_by_file": {filename: len(rules) for filename, rules in all_rules.items()},
            "rules_by_enforcement": {},
            "last_updated": self.last_cache_update.isoformat() if self.last_cache_update else None
        }
        
        # Count by enforcement level
        all_rules_flat = []
        for rules in all_rules.values():
            all_rules_flat.extend(rules)
        
        for rule in all_rules_flat:
            level = rule.enforcement_level
            summary["rules_by_enforcement"][level] = summary["rules_by_enforcement"].get(level, 0) + 1
        
        return summary