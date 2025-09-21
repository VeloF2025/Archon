#!/usr/bin/env python3
"""
Archon Project Agent Factory - Dynamic Agent Generation System

This factory creates project-specific agents based on codebase analysis,
tailoring global agent templates to the exact technology stack and patterns
found in each project.

CRITICAL: This is a hardcoded part of the Archon system and cannot be bypassed.
Every project gets its own specialized agents based on its unique characteristics.
"""

import os
import json
import yaml
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add knowledge feedback service to path
sys.path.append(str(Path(__file__).parent))
from knowledge_feedback_service import KnowledgeFeedbackService

@dataclass
class ProjectAnalysis:
    """Analysis results for a project's technology stack and patterns"""
    languages: Dict[str, float]  # language -> percentage
    frameworks: List[str]
    build_tools: List[str]
    testing_frameworks: List[str]
    databases: List[str]
    cloud_services: List[str]
    ui_libraries: List[str]
    bundlers: List[str]
    linters: List[str]
    package_managers: List[str]
    architecture_patterns: List[str]
    project_type: str
    complexity_score: float
    file_count: int
    loc_count: int

@dataclass
class ProjectAgent:
    """Definition of a project-specific agent"""
    id: str
    name: str
    description: str
    specialization: str
    skills: List[str]
    tools: List[str]
    templates: List[str]  # Global agent templates this inherits from
    project_specific_config: Dict[str, Any]
    activation_commands: List[str]
    quality_gates: List[str]
    performance_targets: Dict[str, Any]

class ArchonProjectAgentFactory:
    """Factory for creating project-specific agents based on codebase analysis"""
    
    def __init__(self, project_path: str, archon_path: str = "/mnt/c/Jarvis/AI Workspace/Archon"):
        self.project_path = Path(project_path)
        self.archon_path = Path(archon_path)
        self.templates_path = self.archon_path / "python/src/agents/orchestrator/project_agent_templates.yaml"
        self.global_agents_path = self.archon_path / "python/src/agents"
        
    def analyze_project(self) -> ProjectAnalysis:
        """Perform comprehensive analysis of project codebase"""
        print("🔍 Analyzing project codebase...")
        
        # Analyze package.json for Node.js projects
        package_json = self.project_path / "package.json"
        analysis_data = {
            'languages': {},
            'frameworks': [],
            'build_tools': [],
            'testing_frameworks': [],
            'databases': [],
            'cloud_services': [],
            'ui_libraries': [],
            'bundlers': [],
            'linters': [],
            'package_managers': [],
            'architecture_patterns': [],
            'project_type': 'unknown',
            'complexity_score': 0.0,
            'file_count': 0,
            'loc_count': 0
        }
        
        if package_json.exists():
            with open(package_json) as f:
                pkg_data = json.load(f)
                self._analyze_package_json(pkg_data, analysis_data)
        
        # Analyze source code structure
        self._analyze_source_structure(analysis_data)
        
        # Analyze configuration files
        self._analyze_config_files(analysis_data)
        
        # Calculate complexity score
        analysis_data['complexity_score'] = self._calculate_complexity_score(analysis_data)
        
        return ProjectAnalysis(**analysis_data)
    
    def _analyze_package_json(self, pkg_data: Dict, analysis_data: Dict):
        """Analyze package.json for technology stack"""
        deps = {**pkg_data.get('dependencies', {}), **pkg_data.get('devDependencies', {})}
        
        # Detect frameworks
        if 'react' in deps:
            analysis_data['frameworks'].append('React')
            analysis_data['languages']['TypeScript'] = 80.0  # Assume React + TS
            analysis_data['languages']['JavaScript'] = 20.0
        if 'next' in deps:
            analysis_data['frameworks'].append('Next.js')
        if 'vue' in deps:
            analysis_data['frameworks'].append('Vue.js')
        if 'angular' in deps:
            analysis_data['frameworks'].append('Angular')
        
        # Detect build tools
        if 'vite' in deps:
            analysis_data['bundlers'].append('Vite')
            analysis_data['build_tools'].append('Vite')
        if 'webpack' in deps:
            analysis_data['bundlers'].append('Webpack')
        if 'rollup' in deps:
            analysis_data['bundlers'].append('Rollup')
        
        # Detect testing frameworks
        if 'vitest' in deps:
            analysis_data['testing_frameworks'].append('Vitest')
        if 'jest' in deps:
            analysis_data['testing_frameworks'].append('Jest')
        if '@playwright/test' in deps:
            analysis_data['testing_frameworks'].append('Playwright')
        
        # Detect databases
        if 'firebase' in deps:
            analysis_data['databases'].append('Firestore')
            analysis_data['cloud_services'].append('Firebase')
        if 'mongodb' in deps or 'mongoose' in deps:
            analysis_data['databases'].append('MongoDB')
        if 'postgresql' in deps or 'pg' in deps:
            analysis_data['databases'].append('PostgreSQL')
        
        # Detect UI libraries
        if 'tailwindcss' in deps:
            analysis_data['ui_libraries'].append('Tailwind CSS')
        if '@headlessui/react' in deps:
            analysis_data['ui_libraries'].append('Headless UI')
        if '@mui/material' in deps:
            analysis_data['ui_libraries'].append('Material-UI')
        
        # Detect linters
        if 'eslint' in deps:
            analysis_data['linters'].append('ESLint')
        if 'prettier' in deps:
            analysis_data['linters'].append('Prettier')
        
        # Package managers
        if (self.project_path / "package-lock.json").exists():
            analysis_data['package_managers'].append('npm')
        elif (self.project_path / "yarn.lock").exists():
            analysis_data['package_managers'].append('yarn')
        elif (self.project_path / "pnpm-lock.yaml").exists():
            analysis_data['package_managers'].append('pnpm')
        
        # Determine project type
        if 'react' in deps:
            if 'firebase' in deps:
                analysis_data['project_type'] = 'React-Firebase SPA'
            else:
                analysis_data['project_type'] = 'React SPA'
    
    def _analyze_source_structure(self, analysis_data: Dict):
        """Analyze source code structure and patterns"""
        src_path = self.project_path / "src"
        if not src_path.exists():
            return
        
        # Count files and estimate lines of code
        ts_files = list(src_path.glob("**/*.ts"))
        tsx_files = list(src_path.glob("**/*.tsx"))
        js_files = list(src_path.glob("**/*.js"))
        jsx_files = list(src_path.glob("**/*.jsx"))
        
        analysis_data['file_count'] = len(ts_files) + len(tsx_files) + len(js_files) + len(jsx_files)
        
        # Estimate LOC (rough calculation)
        total_loc = 0
        for file_list in [ts_files, tsx_files, js_files, jsx_files]:
            for file_path in file_list:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_loc += sum(1 for line in f if line.strip())
                except:
                    continue
        analysis_data['loc_count'] = total_loc
        
        # Detect architecture patterns
        if (src_path / "components").exists():
            analysis_data['architecture_patterns'].append('Component-based')
        if (src_path / "hooks").exists():
            analysis_data['architecture_patterns'].append('Custom Hooks')
        if (src_path / "services").exists():
            analysis_data['architecture_patterns'].append('Service Layer')
        if (src_path / "store" or src_path / "stores").exists():
            analysis_data['architecture_patterns'].append('State Management')
        if (src_path / "utils").exists():
            analysis_data['architecture_patterns'].append('Utility Functions')
        if (src_path / "types").exists():
            analysis_data['architecture_patterns'].append('Type Definitions')
        if (src_path / "features").exists():
            analysis_data['architecture_patterns'].append('Feature-based')
        if (src_path / "pages").exists():
            analysis_data['architecture_patterns'].append('Page-based Routing')
    
    def _analyze_config_files(self, analysis_data: Dict):
        """Analyze configuration files for additional insights"""
        # TypeScript configuration
        if (self.project_path / "tsconfig.json").exists():
            analysis_data['languages']['TypeScript'] = max(
                analysis_data['languages'].get('TypeScript', 0), 70.0
            )
        
        # Tailwind config
        if (self.project_path / "tailwind.config.js").exists() or (self.project_path / "tailwind.config.ts").exists():
            if 'Tailwind CSS' not in analysis_data['ui_libraries']:
                analysis_data['ui_libraries'].append('Tailwind CSS')
        
        # Firebase config
        if (self.project_path / "firebase.json").exists():
            if 'Firebase' not in analysis_data['cloud_services']:
                analysis_data['cloud_services'].append('Firebase')
        
        # Playwright config
        if (self.project_path / "playwright.config.ts").exists():
            if 'Playwright' not in analysis_data['testing_frameworks']:
                analysis_data['testing_frameworks'].append('Playwright')
    
    def _calculate_complexity_score(self, analysis_data: Dict) -> float:
        """Calculate project complexity score (0-10)"""
        score = 0.0
        
        # Base complexity from file count
        file_count = analysis_data['file_count']
        if file_count > 100:
            score += 2.0
        elif file_count > 50:
            score += 1.0
        
        # Framework complexity
        framework_count = len(analysis_data['frameworks'])
        score += min(framework_count * 0.5, 2.0)
        
        # Database complexity
        db_count = len(analysis_data['databases'])
        score += min(db_count * 0.5, 1.5)
        
        # Testing complexity
        test_count = len(analysis_data['testing_frameworks'])
        score += min(test_count * 0.3, 1.0)
        
        # Architecture patterns
        pattern_count = len(analysis_data['architecture_patterns'])
        score += min(pattern_count * 0.2, 1.5)
        
        # Build tools complexity
        build_count = len(analysis_data['build_tools'])
        score += min(build_count * 0.3, 1.0)
        
        # Cloud services
        cloud_count = len(analysis_data['cloud_services'])
        score += min(cloud_count * 0.4, 1.0)
        
        return min(score, 10.0)
    
    def load_agent_templates(self) -> Dict[str, Any]:
        """Load global agent templates"""
        if not self.templates_path.exists():
            # Create default templates if not exist
            self._create_default_templates()
        
        with open(self.templates_path) as f:
            return yaml.safe_load(f)
    
    def _create_default_templates(self):
        """Create default agent templates if they don't exist"""
        os.makedirs(self.templates_path.parent, exist_ok=True)
        
        default_templates = {
            'global_agents': {
                'system-architect': {
                    'skills': ['architecture_design', 'system_planning', 'scalability'],
                    'tools': ['diagrams', 'documentation', 'design_patterns'],
                    'quality_gates': ['architecture_review', 'scalability_check']
                },
                'code-implementer': {
                    'skills': ['coding', 'implementation', 'debugging'],
                    'tools': ['ide', 'compiler', 'debugger'],
                    'quality_gates': ['code_review', 'compilation_check']
                },
                'test-coverage-validator': {
                    'skills': ['testing', 'validation', 'coverage_analysis'],
                    'tools': ['test_runners', 'coverage_tools', 'mocking'],
                    'quality_gates': ['coverage_check', 'test_quality']
                },
                'security-auditor': {
                    'skills': ['security_analysis', 'vulnerability_scanning', 'penetration_testing'],
                    'tools': ['security_scanners', 'audit_tools', 'compliance'],
                    'quality_gates': ['security_scan', 'vulnerability_check']
                },
                'performance-optimizer': {
                    'skills': ['performance_analysis', 'optimization', 'profiling'],
                    'tools': ['profilers', 'benchmarking', 'monitoring'],
                    'quality_gates': ['performance_check', 'load_testing']
                }
            },
            'technology_mappings': {
                'React': {
                    'required_skills': ['react_patterns', 'jsx', 'component_lifecycle'],
                    'tools': ['react_devtools', 'component_testing'],
                    'quality_gates': ['react_best_practices', 'performance_check']
                },
                'Firebase': {
                    'required_skills': ['firestore', 'auth', 'cloud_functions'],
                    'tools': ['firebase_cli', 'emulators', 'security_rules'],
                    'quality_gates': ['security_rules_check', 'performance_monitoring']
                },
                'TypeScript': {
                    'required_skills': ['type_safety', 'generic_programming', 'strict_mode'],
                    'tools': ['tsc', 'type_checking', 'eslint'],
                    'quality_gates': ['type_coverage', 'strict_compilation']
                },
                'Vite': {
                    'required_skills': ['build_optimization', 'dev_server', 'bundling'],
                    'tools': ['vite_cli', 'build_analysis', 'optimization'],
                    'quality_gates': ['bundle_size_check', 'build_performance']
                }
            }
        }
        
        with open(self.templates_path, 'w') as f:
            yaml.dump(default_templates, f, default_flow_style=False, indent=2)
    
    def generate_project_agents(self, analysis: ProjectAnalysis) -> List[ProjectAgent]:
        """Generate project-specific agents based on analysis"""
        templates = self.load_agent_templates()
        agents = []
        
        print(f"🤖 Generating project-specific agents for {analysis.project_type}...")
        
        # Generate React + Firebase specialist
        if 'React' in analysis.frameworks and 'Firebase' in analysis.cloud_services:
            agents.append(self._create_react_firebase_specialist(analysis, templates))
        
        # Generate TypeScript strict enforcer
        if 'TypeScript' in analysis.languages:
            agents.append(self._create_typescript_strict_enforcer(analysis, templates))
        
        # Generate Firestore security architect
        if 'Firestore' in analysis.databases:
            agents.append(self._create_firestore_security_architect(analysis, templates))
        
        # Generate Vite optimization expert
        if 'Vite' in analysis.bundlers:
            agents.append(self._create_vite_optimization_expert(analysis, templates))
        
        # Generate Tailwind UI designer
        if 'Tailwind CSS' in analysis.ui_libraries:
            agents.append(self._create_tailwind_ui_designer(analysis, templates))
        
        # Generate Testing specialist if multiple frameworks
        if len(analysis.testing_frameworks) >= 2:
            agents.append(self._create_testing_specialist(analysis, templates))
        
        # Generate Performance monitor for complex projects
        if analysis.complexity_score >= 5.0:
            agents.append(self._create_performance_monitor(analysis, templates))
        
        return agents
    
    def _create_react_firebase_specialist(self, analysis: ProjectAnalysis, templates: Dict) -> ProjectAgent:
        """Create React + Firebase integration specialist"""
        return ProjectAgent(
            id='react-firebase-specialist',
            name='React Firebase Specialist',
            description='Specialized agent for React + Firebase integration patterns',
            specialization='React + Firebase Integration',
            skills=[
                'react_hooks_with_firebase',
                'firestore_real_time_updates',
                'firebase_auth_integration',
                'offline_first_patterns',
                'firebase_storage_with_react',
                'cloud_functions_integration',
                'firebase_security_rules',
                'react_query_with_firebase'
            ],
            tools=[
                'firebase_cli',
                'react_devtools',
                'firebase_emulators',
                'firestore_debug',
                'auth_debug_tools'
            ],
            templates=['code-implementer', 'system-architect'],
            project_specific_config={
                'firebase_project': 'detected',
                'react_version': '18.3+',
                'auth_providers': ['google', 'email'],
                'firestore_collections': 'auto_detect',
                'offline_support': True,
                'real_time_features': True
            },
            activation_commands=[
                '@react-firebase-specialist <task>',
                'ff2 assign <issue> react-firebase-specialist'
            ],
            quality_gates=[
                'firebase_security_rules_validation',
                'react_firebase_performance_check',
                'offline_functionality_test',
                'auth_flow_validation',
                'real_time_sync_test'
            ],
            performance_targets={
                'initial_load_time': '< 1.5s',
                'auth_response_time': '< 500ms',
                'firestore_query_time': '< 200ms',
                'offline_sync_time': '< 1s'
            }
        )
    
    def _create_typescript_strict_enforcer(self, analysis: ProjectAnalysis, templates: Dict) -> ProjectAgent:
        """Create TypeScript strict mode enforcer"""
        return ProjectAgent(
            id='typescript-strict-enforcer',
            name='TypeScript Strict Enforcer',
            description='Ensures 100% type safety and strict TypeScript compliance',
            specialization='TypeScript Strict Mode',
            skills=[
                'strict_type_checking',
                'zero_any_types',
                'generic_programming',
                'type_guards',
                'utility_types',
                'conditional_types',
                'template_literal_types',
                'type_inference_optimization'
            ],
            tools=[
                'typescript_compiler',
                'eslint_typescript',
                'type_coverage',
                'ts_node',
                'type_fest'
            ],
            templates=['code-quality-reviewer', 'code-implementer'],
            project_specific_config={
                'strict_mode': True,
                'no_implicit_any': True,
                'no_unused_locals': True,
                'no_unused_parameters': True,
                'no_fallthrough_cases': True,
                'target': 'ES2020',
                'module': 'ESNext'
            },
            activation_commands=[
                '@typescript-strict-enforcer <task>',
                'ff2 assign <issue> typescript-strict-enforcer'
            ],
            quality_gates=[
                'zero_typescript_errors',
                'zero_any_types',
                '100_percent_type_coverage',
                'strict_compilation_check',
                'type_safety_validation'
            ],
            performance_targets={
                'compilation_time': '< 10s',
                'type_checking_time': '< 5s',
                'incremental_build_time': '< 2s'
            }
        )
    
    def _create_firestore_security_architect(self, analysis: ProjectAnalysis, templates: Dict) -> ProjectAgent:
        """Create Firestore security rules architect"""
        return ProjectAgent(
            id='firestore-security-architect',
            name='Firestore Security Architect',
            description='Designs and validates Firestore security rules and data access patterns',
            specialization='Firestore Security & Architecture',
            skills=[
                'firestore_security_rules',
                'data_access_patterns',
                'role_based_access_control',
                'field_level_security',
                'audit_trail_design',
                'data_validation_rules',
                'compound_index_optimization',
                'security_testing'
            ],
            tools=[
                'firebase_emulators',
                'security_rules_testing',
                'firestore_cli',
                'security_audit_tools',
                'performance_monitoring'
            ],
            templates=['security-auditor', 'system-architect'],
            project_specific_config={
                'rbac_enabled': True,
                'audit_logging': True,
                'field_validation': True,
                'index_optimization': True,
                'security_testing': True
            },
            activation_commands=[
                '@firestore-security-architect <task>',
                'ff2 assign <issue> firestore-security-architect'
            ],
            quality_gates=[
                'security_rules_validation',
                'rbac_compliance_check',
                'data_access_audit',
                'performance_index_check',
                'security_penetration_test'
            ],
            performance_targets={
                'query_response_time': '< 200ms',
                'security_rule_evaluation': '< 50ms',
                'index_utilization': '> 95%'
            }
        )
    
    def _create_vite_optimization_expert(self, analysis: ProjectAnalysis, templates: Dict) -> ProjectAgent:
        """Create Vite build optimization expert"""
        return ProjectAgent(
            id='vite-optimization-expert',
            name='Vite Optimization Expert',
            description='Optimizes Vite build configuration for maximum performance',
            specialization='Vite Build Optimization',
            skills=[
                'vite_configuration',
                'bundle_splitting',
                'code_splitting',
                'tree_shaking',
                'asset_optimization',
                'dev_server_optimization',
                'build_performance',
                'chunk_optimization'
            ],
            tools=[
                'vite_cli',
                'bundle_analyzer',
                'performance_profiler',
                'build_optimizer',
                'dev_server_tools'
            ],
            templates=['performance-optimizer', 'code-implementer'],
            project_specific_config={
                'target': 'es2018',
                'minify': 'terser',
                'sourcemap': True,
                'chunk_size_limit': 300,
                'manual_chunks': True,
                'code_splitting': True
            },
            activation_commands=[
                '@vite-optimization-expert <task>',
                'ff2 assign <issue> vite-optimization-expert'
            ],
            quality_gates=[
                'bundle_size_check',
                'build_performance_test',
                'chunk_optimization_validation',
                'dev_server_performance_check',
                'asset_optimization_audit'
            ],
            performance_targets={
                'build_time': '< 30s',
                'dev_server_startup': '< 3s',
                'hmr_update_time': '< 100ms',
                'bundle_size': '< 500KB per chunk'
            }
        )
    
    def _create_tailwind_ui_designer(self, analysis: ProjectAnalysis, templates: Dict) -> ProjectAgent:
        """Create Tailwind CSS UI design specialist"""
        return ProjectAgent(
            id='tailwind-ui-designer',
            name='Tailwind UI Designer',
            description='Specialized in Tailwind CSS patterns and responsive design',
            specialization='Tailwind CSS & Responsive Design',
            skills=[
                'tailwind_css_utilities',
                'responsive_design',
                'component_composition',
                'design_system_creation',
                'dark_mode_implementation',
                'accessibility_patterns',
                'custom_component_design',
                'utility_optimization'
            ],
            tools=[
                'tailwind_cli',
                'design_tokens',
                'accessibility_checker',
                'responsive_tester',
                'ui_component_library'
            ],
            templates=['ui-ux-optimizer', 'code-implementer'],
            project_specific_config={
                'dark_mode': 'class',
                'responsive_breakpoints': ['sm', 'md', 'lg', 'xl', '2xl'],
                'custom_colors': True,
                'component_library': True,
                'accessibility_compliance': 'WCAG 2.1'
            },
            activation_commands=[
                '@tailwind-ui-designer <task>',
                'ff2 assign <issue> tailwind-ui-designer'
            ],
            quality_gates=[
                'responsive_design_validation',
                'accessibility_compliance_check',
                'dark_mode_compatibility',
                'component_consistency_audit',
                'performance_css_check'
            ],
            performance_targets={
                'css_bundle_size': '< 50KB',
                'unused_css': '< 5%',
                'accessibility_score': '> 95',
                'responsive_performance': 'A+ rating'
            }
        )
    
    def _create_testing_specialist(self, analysis: ProjectAnalysis, templates: Dict) -> ProjectAgent:
        """Create comprehensive testing specialist"""
        frameworks = ', '.join(analysis.testing_frameworks)
        return ProjectAgent(
            id='testing-specialist',
            name='Testing Specialist',
            description=f'Comprehensive testing with {frameworks}',
            specialization='Multi-Framework Testing',
            skills=[
                'unit_testing',
                'integration_testing',
                'e2e_testing',
                'visual_regression_testing',
                'accessibility_testing',
                'performance_testing',
                'security_testing',
                'test_automation'
            ],
            tools=analysis.testing_frameworks + ['coverage_tools', 'test_reporters'],
            templates=['test-coverage-validator', 'code-quality-reviewer'],
            project_specific_config={
                'frameworks': analysis.testing_frameworks,
                'coverage_threshold': 95,
                'e2e_browsers': ['chromium', 'firefox', 'webkit'],
                'visual_testing': True,
                'accessibility_testing': True
            },
            activation_commands=[
                '@testing-specialist <task>',
                'ff2 assign <issue> testing-specialist'
            ],
            quality_gates=[
                'unit_test_coverage_95_percent',
                'integration_test_validation',
                'e2e_test_completion',
                'accessibility_test_pass',
                'performance_test_validation'
            ],
            performance_targets={
                'test_execution_time': '< 120s',
                'coverage_generation': '< 30s',
                'e2e_test_time': '< 300s per suite'
            }
        )
    
    def _create_performance_monitor(self, analysis: ProjectAnalysis, templates: Dict) -> ProjectAgent:
        """Create performance monitoring specialist for complex projects"""
        return ProjectAgent(
            id='performance-monitor',
            name='Performance Monitor',
            description='Continuous performance monitoring and optimization',
            specialization='Performance Monitoring & Optimization',
            skills=[
                'performance_monitoring',
                'lighthouse_optimization',
                'bundle_analysis',
                'runtime_performance',
                'memory_profiling',
                'network_optimization',
                'caching_strategies',
                'performance_budgets'
            ],
            tools=[
                'lighthouse',
                'web_vitals',
                'bundle_analyzer',
                'performance_profiler',
                'monitoring_tools'
            ],
            templates=['performance-optimizer', 'system-architect'],
            project_specific_config={
                'performance_budget': {
                    'bundle_size': '500KB',
                    'first_contentful_paint': '1.5s',
                    'largest_contentful_paint': '2.5s',
                    'cumulative_layout_shift': '0.1',
                    'time_to_interactive': '3.8s'
                },
                'monitoring_frequency': 'continuous',
                'alert_thresholds': True
            },
            activation_commands=[
                '@performance-monitor <task>',
                'ff2 assign <issue> performance-monitor'
            ],
            quality_gates=[
                'lighthouse_score_90_plus',
                'web_vitals_validation',
                'bundle_size_compliance',
                'runtime_performance_check',
                'memory_leak_detection'
            ],
            performance_targets={
                'lighthouse_performance': '> 90',
                'lighthouse_accessibility': '> 95',
                'lighthouse_best_practices': '> 90',
                'lighthouse_seo': '> 90'
            }
        )
    
    def save_project_agents(self, agents: List[ProjectAgent]):
        """Save project-specific agents to configuration file with knowledge integration"""
        project_archon_dir = self.project_path / ".archon"
        project_archon_dir.mkdir(exist_ok=True)
        
        config_file = project_archon_dir / "project_agents.yaml"
        
        # Initialize knowledge feedback service
        knowledge_service = KnowledgeFeedbackService(str(self.project_path))
        
        agents_data = {
            'generated_at': datetime.now().isoformat(),
            'project_path': str(self.project_path),
            'project_id': self.project_path.name,
            'knowledge_integration': {
                'enabled': True,
                'sync_interval_hours': 24,
                'auto_learning': True,
                'knowledge_api': 'http://localhost:3737'
            },
            'agents': []
        }
        
        for agent in agents:
            agents_data['agents'].append({
                'id': agent.id,
                'name': agent.name,
                'description': agent.description,
                'specialization': agent.specialization,
                'skills': agent.skills,
                'tools': agent.tools,
                'templates': agent.templates,
                'project_specific_config': agent.project_specific_config,
                'activation_commands': agent.activation_commands,
                'quality_gates': agent.quality_gates,
                'performance_targets': agent.performance_targets,
                'knowledge_hooks': {
                    'before_execution': 'query_relevant_patterns',
                    'after_success': 'submit_successful_pattern',
                    'after_failure': 'submit_error_solution',
                    'periodic_sync': 'sync_with_global_knowledge'
                }
            })
        
        with open(config_file, 'w') as f:
            yaml.dump(agents_data, f, default_flow_style=False, indent=2)
        
        print(f"✅ Saved {len(agents)} project-specific agents to {config_file}")
    
    def execute_workflow(self) -> List[ProjectAgent]:
        """Execute the complete project agent creation workflow with knowledge integration"""
        print("🚀 Starting Archon Project Agent Factory Workflow...")
        
        # Step 1: Analyze project
        analysis = self.analyze_project()
        print(f"📊 Analysis complete: {analysis.project_type} ({analysis.complexity_score:.1f}/10 complexity)")
        print(f"   Languages: {', '.join(f'{k} ({v:.0f}%)' for k, v in analysis.languages.items())}")
        print(f"   Frameworks: {', '.join(analysis.frameworks)}")
        print(f"   Files: {analysis.file_count}, LOC: {analysis.loc_count}")
        
        # Step 2: Generate agents
        agents = self.generate_project_agents(analysis)
        print(f"🤖 Generated {len(agents)} specialized agents:")
        for agent in agents:
            print(f"   - {agent.name}: {agent.specialization}")
        
        # Step 3: Initialize knowledge integration
        self._initialize_knowledge_integration(agents, analysis)
        
        # Step 4: Save configuration
        self.save_project_agents(agents)
        
        return agents
    
    def _initialize_knowledge_integration(self, agents: List[ProjectAgent], analysis: ProjectAnalysis):
        """Initialize knowledge integration for project agents"""
        print("🧠 Initializing knowledge integration...")
        
        # Create knowledge sync configuration
        knowledge_sync_config = {
            'project_id': self.project_path.name,
            'tech_stack': {
                'languages': list(analysis.languages.keys()),
                'frameworks': analysis.frameworks,
                'databases': analysis.databases,
                'testing': analysis.testing_frameworks
            },
            'agents': [agent.id for agent in agents],
            'sync_enabled': True,
            'auto_learning': True,
            'quality_threshold': 0.7,
            'pattern_threshold': 3
        }
        
        # Save knowledge sync configuration
        sync_config_path = self.project_path / '.archon' / 'knowledge_sync.yaml'
        sync_config_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(sync_config_path, 'w') as f:
            yaml.dump(knowledge_sync_config, f, default_flow_style=False, indent=2)
        
        print(f"   ✅ Knowledge sync configured for {len(agents)} agents")
        print(f"   📁 Configuration saved to {sync_config_path}")

def main():
    """Main entry point for command-line usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python archon_project_agent_factory.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    factory = ArchonProjectAgentFactory(project_path)
    agents = factory.execute_workflow()
    
    print(f"\n🎯 Project-specific agents ready for activation!")
    print("   Use @Archon to activate these agents for your project")

if __name__ == "__main__":
    main()