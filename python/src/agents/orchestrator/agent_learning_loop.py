#!/usr/bin/env python3
"""
Agent Learning Loop - Continuous Improvement System

This module implements the learning loop that enables agents to improve over time
by analyzing patterns across projects and updating agent templates based on
successful implementations.

CRITICAL: Core component of the Archon project-specific agent learning system.
"""

import json
import asyncio
import yaml
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentLearningLoop:
    """
    Manages the continuous learning cycle for project-specific agents.
    Aggregates patterns, identifies improvements, and updates templates.
    """
    
    def __init__(self,
                 archon_path: str = "/mnt/c/Jarvis/AI Workspace/Archon",
                 knowledge_api: str = "http://localhost:3737",
                 update_interval_hours: int = 24):
        """
        Initialize the agent learning loop.
        
        Args:
            archon_path: Path to Archon system
            knowledge_api: Knowledge base API endpoint
            update_interval_hours: Hours between learning cycles
        """
        self.archon_path = Path(archon_path)
        self.knowledge_api = knowledge_api
        self.update_interval = timedelta(hours=update_interval_hours)
        
        # Paths to agent templates and configurations
        self.templates_path = self.archon_path / "python/src/agents/orchestrator/project_agent_templates.yaml"
        self.global_patterns_path = self.archon_path / "python/src/agents/patterns/global_patterns.json"
        self.learning_history_path = self.archon_path / "python/src/agents/learning/history.json"
        
        # Learning thresholds
        self.pattern_threshold = 3  # Minimum occurrences to consider a pattern
        self.quality_threshold = 0.7  # Minimum quality score for pattern adoption
        self.success_rate_threshold = 0.8  # Minimum success rate for pattern
        
        # Pattern categories for analysis
        self.pattern_categories = [
            'code_patterns',
            'error_solutions',
            'performance_optimizations',
            'architecture_decisions',
            'test_strategies',
            'security_fixes'
        ]
        
        logger.info(f"Agent Learning Loop initialized with {update_interval_hours}h update cycle")
    
    async def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Execute a complete learning cycle.
        
        Returns:
            Learning cycle report with statistics and updates
        """
        logger.info("Starting agent learning cycle...")
        
        cycle_report = {
            'timestamp': datetime.now().isoformat(),
            'patterns_analyzed': 0,
            'patterns_adopted': 0,
            'templates_updated': 0,
            'agents_improved': [],
            'insights': []
        }
        
        try:
            # Step 1: Collect patterns from knowledge base
            patterns = await self._collect_patterns()
            cycle_report['patterns_analyzed'] = len(patterns)
            
            # Step 2: Analyze patterns for quality and frequency
            valuable_patterns = self._analyze_patterns(patterns)
            
            # Step 3: Group patterns by technology stack
            tech_grouped_patterns = self._group_by_tech_stack(valuable_patterns)
            
            # Step 4: Generate insights from patterns
            insights = self._generate_insights(tech_grouped_patterns)
            cycle_report['insights'] = insights
            
            # Step 5: Update agent templates with new patterns
            updated_templates = await self._update_agent_templates(tech_grouped_patterns)
            cycle_report['templates_updated'] = len(updated_templates)
            cycle_report['agents_improved'] = updated_templates
            
            # Step 6: Propagate updates to active projects
            await self._propagate_updates(updated_templates)
            
            # Step 7: Store learning history
            self._store_learning_history(cycle_report)
            
            # Step 8: Calculate adopted patterns
            cycle_report['patterns_adopted'] = len(valuable_patterns)
            
            logger.info(f"Learning cycle complete: {cycle_report['patterns_adopted']} patterns adopted")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {str(e)}")
            cycle_report['error'] = str(e)
        
        return cycle_report
    
    async def _collect_patterns(self) -> List[Dict[str, Any]]:
        """
        Collect patterns from the knowledge base.
        
        Returns:
            List of patterns with metadata
        """
        patterns = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for category in self.pattern_categories:
                    response = await client.get(
                        f"{self.knowledge_api}/api/knowledge/patterns",
                        params={
                            'category': category,
                            'min_quality': self.quality_threshold,
                            'since': (datetime.now() - self.update_interval).isoformat()
                        }
                    )
                    
                    if response.status_code == 200:
                        category_patterns = response.json().get('patterns', [])
                        patterns.extend(category_patterns)
                        logger.info(f"Collected {len(category_patterns)} patterns for {category}")
        
        except Exception as e:
            logger.error(f"Error collecting patterns: {str(e)}")
        
        return patterns
    
    def _analyze_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """
        Analyze patterns for quality and frequency.
        
        Args:
            patterns: Raw patterns from knowledge base
        
        Returns:
            List of valuable patterns worth adopting
        """
        pattern_stats = defaultdict(lambda: {
            'occurrences': 0,
            'total_quality': 0,
            'success_count': 0,
            'failure_count': 0,
            'projects': set(),
            'agents': set(),
            'examples': []
        })
        
        # Aggregate pattern statistics
        for pattern in patterns:
            pattern_key = self._generate_pattern_key(pattern)
            stats = pattern_stats[pattern_key]
            
            stats['occurrences'] += 1
            stats['total_quality'] += pattern.get('metadata', {}).get('quality_score', 0)
            
            if pattern.get('metadata', {}).get('success', True):
                stats['success_count'] += 1
            else:
                stats['failure_count'] += 1
            
            stats['projects'].add(pattern.get('project_id', 'unknown'))
            stats['agents'].add(pattern.get('agent_id', 'unknown'))
            stats['examples'].append(pattern)
        
        # Filter valuable patterns
        valuable_patterns = []
        for pattern_key, stats in pattern_stats.items():
            # Calculate metrics
            avg_quality = stats['total_quality'] / stats['occurrences']
            success_rate = stats['success_count'] / (stats['success_count'] + stats['failure_count'])
            
            # Check thresholds
            if (stats['occurrences'] >= self.pattern_threshold and
                avg_quality >= self.quality_threshold and
                success_rate >= self.success_rate_threshold):
                
                valuable_patterns.append({
                    'key': pattern_key,
                    'stats': {
                        'occurrences': stats['occurrences'],
                        'avg_quality': avg_quality,
                        'success_rate': success_rate,
                        'project_count': len(stats['projects']),
                        'agent_count': len(stats['agents'])
                    },
                    'examples': stats['examples'][:5]  # Keep top 5 examples
                })
        
        logger.info(f"Found {len(valuable_patterns)} valuable patterns out of {len(patterns)}")
        return valuable_patterns
    
    def _group_by_tech_stack(self, patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group patterns by technology stack.
        
        Args:
            patterns: Valuable patterns to group
        
        Returns:
            Dictionary of tech_stack -> patterns
        """
        tech_grouped = defaultdict(list)
        
        for pattern in patterns:
            # Extract tech stack from examples
            tech_stacks = set()
            for example in pattern.get('examples', []):
                tech_stack = example.get('metadata', {}).get('tech_stack', {})
                if tech_stack:
                    # Create a tech stack key
                    frameworks = tuple(sorted(tech_stack.get('frameworks', [])))
                    languages = tuple(sorted(tech_stack.get('languages', [])))
                    tech_key = f"{','.join(languages)}:{','.join(frameworks)}"
                    tech_stacks.add(tech_key)
            
            # Add pattern to each relevant tech stack
            for tech_key in tech_stacks:
                tech_grouped[tech_key].append(pattern)
            
            # Also add to 'universal' for patterns that work across stacks
            if len(tech_stacks) > 2:
                tech_grouped['universal'].append(pattern)
        
        logger.info(f"Grouped patterns into {len(tech_grouped)} technology stacks")
        return dict(tech_grouped)
    
    def _generate_insights(self, tech_patterns: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Generate insights from pattern analysis.
        
        Args:
            tech_patterns: Patterns grouped by technology stack
        
        Returns:
            List of insights
        """
        insights = []
        
        for tech_stack, patterns in tech_patterns.items():
            if not patterns:
                continue
            
            # Calculate statistics
            total_occurrences = sum(p['stats']['occurrences'] for p in patterns)
            avg_success_rate = sum(p['stats']['success_rate'] for p in patterns) / len(patterns)
            
            # Generate insight
            insight = {
                'tech_stack': tech_stack,
                'pattern_count': len(patterns),
                'total_occurrences': total_occurrences,
                'avg_success_rate': avg_success_rate,
                'top_categories': self._get_top_categories(patterns),
                'recommendation': self._generate_recommendation(tech_stack, patterns)
            }
            insights.append(insight)
        
        # Sort by impact (occurrences * success_rate)
        insights.sort(key=lambda x: x['total_occurrences'] * x['avg_success_rate'], reverse=True)
        
        return insights[:10]  # Return top 10 insights
    
    async def _update_agent_templates(self, tech_patterns: Dict[str, List[Dict]]) -> List[str]:
        """
        Update agent templates with learned patterns.
        
        Args:
            tech_patterns: Patterns grouped by technology stack
        
        Returns:
            List of updated agent template IDs
        """
        updated_templates = []
        
        try:
            # Load current templates
            templates = {}
            if self.templates_path.exists():
                with open(self.templates_path, 'r') as f:
                    templates = yaml.safe_load(f) or {}
            
            # Update templates with new patterns
            for tech_stack, patterns in tech_patterns.items():
                if not patterns:
                    continue
                
                # Find or create template for this tech stack
                template_id = self._get_template_id(tech_stack)
                
                if template_id not in templates:
                    templates[template_id] = self._create_template(tech_stack)
                
                template = templates[template_id]
                
                # Add learned patterns to template
                if 'learned_patterns' not in template:
                    template['learned_patterns'] = []
                
                for pattern in patterns[:5]:  # Add top 5 patterns
                    pattern_entry = {
                        'key': pattern['key'],
                        'stats': pattern['stats'],
                        'examples': [self._sanitize_example(e) for e in pattern['examples'][:2]],
                        'added_date': datetime.now().isoformat()
                    }
                    
                    # Check if pattern already exists
                    existing = False
                    for existing_pattern in template['learned_patterns']:
                        if existing_pattern['key'] == pattern['key']:
                            # Update existing pattern stats
                            existing_pattern['stats'] = pattern['stats']
                            existing_pattern['updated_date'] = datetime.now().isoformat()
                            existing = True
                            break
                    
                    if not existing:
                        template['learned_patterns'].append(pattern_entry)
                
                updated_templates.append(template_id)
            
            # Save updated templates
            if updated_templates:
                with open(self.templates_path, 'w') as f:
                    yaml.dump(templates, f, default_flow_style=False)
                
                logger.info(f"Updated {len(updated_templates)} agent templates")
        
        except Exception as e:
            logger.error(f"Error updating templates: {str(e)}")
        
        return updated_templates
    
    async def _propagate_updates(self, updated_templates: List[str]):
        """
        Propagate template updates to active projects.
        
        Args:
            updated_templates: List of updated template IDs
        """
        if not updated_templates:
            return
        
        try:
            # Find all projects using these templates
            projects_to_update = await self._find_projects_using_templates(updated_templates)
            
            for project_path in projects_to_update:
                # Update project agent configuration
                config_path = Path(project_path) / '.archon' / 'project_agents.yaml'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                    
                    # Mark for update on next activation
                    config['templates_updated'] = datetime.now().isoformat()
                    config['pending_updates'] = updated_templates
                    
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    logger.info(f"Marked project for update: {project_path}")
        
        except Exception as e:
            logger.error(f"Error propagating updates: {str(e)}")
    
    def _generate_pattern_key(self, pattern: Dict) -> str:
        """Generate a unique key for a pattern."""
        category = pattern.get('category', 'unknown')
        title = pattern.get('title', '')
        # Create a normalized key
        return f"{category}:{title.lower().replace(' ', '_')}"
    
    def _get_top_categories(self, patterns: List[Dict]) -> List[str]:
        """Get top pattern categories."""
        category_counts = defaultdict(int)
        for pattern in patterns:
            for example in pattern.get('examples', []):
                category = example.get('category', 'unknown')
                category_counts[category] += 1
        
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in sorted_categories[:3]]
    
    def _generate_recommendation(self, tech_stack: str, patterns: List[Dict]) -> str:
        """Generate recommendation based on patterns."""
        if tech_stack == 'universal':
            return "These patterns work across multiple technology stacks and should be adopted globally."
        
        pattern_count = len(patterns)
        avg_quality = sum(p['stats']['avg_quality'] for p in patterns) / pattern_count
        
        if avg_quality > 0.9:
            return f"High-quality patterns for {tech_stack}. Strong recommendation for adoption."
        elif avg_quality > 0.8:
            return f"Good patterns for {tech_stack}. Recommended for most projects."
        else:
            return f"Useful patterns for {tech_stack}. Consider for specific use cases."
    
    def _get_template_id(self, tech_stack: str) -> str:
        """Get template ID for a technology stack."""
        if tech_stack == 'universal':
            return 'universal_patterns'
        
        # Normalize tech stack to template ID
        return tech_stack.replace(':', '_').replace(',', '_').replace(' ', '').lower()
    
    def _create_template(self, tech_stack: str) -> Dict:
        """Create a new template for a technology stack."""
        return {
            'id': self._get_template_id(tech_stack),
            'name': f"Learned patterns for {tech_stack}",
            'tech_stack': tech_stack,
            'created_date': datetime.now().isoformat(),
            'learned_patterns': [],
            'auto_generated': True
        }
    
    def _sanitize_example(self, example: Dict) -> Dict:
        """Sanitize example for storage in template."""
        # Remove sensitive or large data
        sanitized = {
            'title': example.get('title', ''),
            'category': example.get('category', ''),
            'summary': example.get('content', '')[:500] if 'content' in example else '',
            'quality_score': example.get('metadata', {}).get('quality_score', 0)
        }
        return sanitized
    
    async def _find_projects_using_templates(self, template_ids: List[str]) -> List[str]:
        """Find projects using specific templates."""
        projects = []
        
        try:
            # Search for projects in workspace
            workspace_path = Path("/mnt/c/Jarvis/AI Workspace")
            for project_dir in workspace_path.iterdir():
                if project_dir.is_dir():
                    config_path = project_dir / '.archon' / 'project_agents.yaml'
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f) or {}
                        
                        # Check if project uses any of the updated templates
                        project_templates = config.get('templates', [])
                        if any(t in template_ids for t in project_templates):
                            projects.append(str(project_dir))
        
        except Exception as e:
            logger.error(f"Error finding projects: {str(e)}")
        
        return projects
    
    def _store_learning_history(self, cycle_report: Dict):
        """Store learning cycle history."""
        try:
            # Load existing history
            history = []
            if self.learning_history_path.exists():
                with open(self.learning_history_path, 'r') as f:
                    history = json.load(f)
            
            # Add new cycle report
            history.append(cycle_report)
            
            # Keep only last 100 cycles
            history = history[-100:]
            
            # Save updated history
            self.learning_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error storing learning history: {str(e)}")
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """
        Get current learning loop status.
        
        Returns:
            Status report with statistics
        """
        status = {
            'active': True,
            'update_interval_hours': self.update_interval.total_seconds() / 3600,
            'pattern_categories': self.pattern_categories,
            'thresholds': {
                'pattern_threshold': self.pattern_threshold,
                'quality_threshold': self.quality_threshold,
                'success_rate_threshold': self.success_rate_threshold
            }
        }
        
        # Load last cycle report
        if self.learning_history_path.exists():
            with open(self.learning_history_path, 'r') as f:
                history = json.load(f)
                if history:
                    status['last_cycle'] = history[-1]
                    status['total_cycles'] = len(history)
        
        return status


# CLI Interface for testing
async def main():
    """CLI interface for testing agent learning loop."""
    import sys
    
    loop = AgentLearningLoop()
    
    if len(sys.argv) < 2:
        print("Usage: python agent_learning_loop.py <command>")
        print("Commands:")
        print("  run     - Run a learning cycle")
        print("  status  - Show learning loop status")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "run":
        print("Running learning cycle...")
        report = await loop.run_learning_cycle()
        print(f"Learning cycle report:")
        print(json.dumps(report, indent=2))
    
    elif command == "status":
        status = await loop.get_learning_status()
        print(f"Learning loop status:")
        print(json.dumps(status, indent=2))
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    asyncio.run(main())