#!/usr/bin/env python3
"""
Phase 6: Agent System Integration - Comprehensive Unit Tests
Tests the core agent system integration with Claude Code Task tool
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import components under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.orchestration.orchestrator import ArchonOrchestrator
from agents.orchestration.agent_pool import AgentPool, AgentState, AgentInstance
from agents.orchestration.parallel_executor import ParallelExecutor, AgentTask
from agents.orchestration.meta_agent import MetaAgent
from agents.triggers.trigger_engine import TriggerEngine
from agents.triggers.file_watcher import FileWatcher

class TestAgentSystemIntegration:
    """Test suite for Phase 6 Agent System Integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config_path = "python/src/agents/configs"
        self.orchestrator = ArchonOrchestrator(
            config_path=self.config_path,
            max_concurrent_tasks=8,
            max_total_agents=15
        )
        self.meta_agent = MetaAgent()
        
    def test_all_22_agents_loaded(self):
        """Test that all 22 specialized agents are loaded and available"""
        expected_agents = {
            'python_backend_coder', 'typescript_frontend_agent', 'api_integrator', 
            'database_designer', 'security_auditor', 'test_generator', 
            'code_reviewer', 'quality_assurance', 'integration_tester',
            'documentation_writer', 'technical_writer', 'devops_engineer',
            'deployment_coordinator', 'monitoring_agent', 'configuration_manager',
            'performance_optimizer', 'refactoring_specialist', 'error_handler',
            'ui_ux_designer', 'system_architect', 'data_analyst', 'hrm_reasoning_agent'
        }
        
        loaded_agents = set(self.orchestrator.executor.agent_configs.keys())
        
        assert len(loaded_agents) >= 22, f"Expected 22+ agents, got {len(loaded_agents)}"
        missing_agents = expected_agents - loaded_agents
        assert not missing_agents, f"Missing agents: {missing_agents}"
        
        # Verify each agent has proper configuration
        for agent_name in expected_agents:
            config = self.orchestrator.executor.agent_configs[agent_name]
            assert 'role' in config
            assert 'capabilities' in config
            assert 'execution_context' in config
            
    def test_agent_pool_initialization(self):
        """Test agent pool properly initializes with minimum agents"""
        pool = self.orchestrator.agent_pool
        
        # Should have minimum critical agents spawned
        critical_agents = ['security_auditor', 'deployment_coordinator', 'system_architect']
        
        for agent_role in critical_agents:
            instances = pool.get_agents_by_role(agent_role)
            assert len(instances) > 0, f"Critical agent {agent_role} not spawned"
            
    def test_meta_agent_task_distribution(self):
        """Test meta-agent correctly distributes tasks to specialized agents"""
        # Create test tasks for different agent types
        test_tasks = [
            {
                'description': 'Review Python security vulnerabilities',
                'expected_agent': 'security_auditor',
                'file_pattern': '*.py'
            },
            {
                'description': 'Generate unit tests for API endpoints',
                'expected_agent': 'test_generator', 
                'file_pattern': 'api/*.py'
            },
            {
                'description': 'Design database schema for user management',
                'expected_agent': 'database_designer',
                'file_pattern': '*.sql'
            }
        ]
        
        for task_spec in test_tasks:
            selected_agent = self.meta_agent.select_agent_for_task(
                task_spec['description'],
                context={'files': [task_spec['file_pattern']]}
            )
            
            assert selected_agent == task_spec['expected_agent'], \
                f"Expected {task_spec['expected_agent']}, got {selected_agent}"
                
    def test_tool_access_control(self):
        """Test that agents have proper scoped tool access"""
        # Test security_auditor has limited scope
        security_tools = self.orchestrator.get_agent_tools('security_auditor')
        expected_security_tools = {'Read', 'Grep', 'Bash', 'Write'}
        
        assert set(security_tools.keys()) <= expected_security_tools, \
            f"Security auditor has unauthorized tools: {set(security_tools.keys()) - expected_security_tools}"
            
        # Test that security_auditor cannot access UI-specific operations
        assert 'ui_design' not in security_tools
        assert 'component_generation' not in security_tools
        
        # Test ui_ux_designer has appropriate tools
        ui_tools = self.orchestrator.get_agent_tools('ui_ux_designer')
        expected_ui_tools = {'Read', 'Write', 'Edit', 'Bash'}
        
        assert 'component_analysis' in ui_tools or set(ui_tools.keys()) <= expected_ui_tools
        
    def test_agent_isolation(self):
        """Test that agents requiring isolation are properly contained"""
        isolation_required = [
            'python_backend_coder', 'typescript_frontend_agent', 
            'api_integrator', 'database_designer'
        ]
        
        for agent_role in isolation_required:
            task = self.orchestrator.create_task(
                agent_role=agent_role,
                description=f"Test isolation for {agent_role}",
                input_data={'test': True}
            )
            
            assert task.requires_isolation, f"{agent_role} should require isolation"
            
    def test_parallel_execution_limits(self):
        """Test that parallel execution respects defined limits"""
        config = self.orchestrator.executor.agent_configs['security_auditor']
        registry = json.loads(Path(f"{self.config_path}/agent_registry.json").read_text())
        
        # Test category limits
        max_quality_agents = registry['max_parallel_execution']['category_limits']['quality']
        assert max_quality_agents == 6, "Quality agents should have limit of 6"
        
        # Test total limit
        total_max = registry['max_parallel_execution']['total_max']
        assert total_max == 15, "Total agents should have limit of 15"
        
    def test_agent_dependency_resolution(self):
        """Test that agent dependencies are properly resolved"""
        registry = json.loads(Path(f"{self.config_path}/agent_registry.json").read_text())
        dependencies = registry['dependency_graph']
        
        # Test python_backend_coder dependencies
        python_deps = dependencies.get('python_backend_coder', [])
        assert 'security_auditor' in python_deps
        assert 'test_generator' in python_deps
        
        # Test that dependent agents can be spawned
        for primary_agent, deps in dependencies.items():
            for dep_agent in deps:
                assert dep_agent in self.orchestrator.executor.agent_configs, \
                    f"Dependency {dep_agent} for {primary_agent} not available"
                    
    def test_file_trigger_system(self):
        """Test file change triggers spawn correct agents"""
        trigger_engine = TriggerEngine(self.orchestrator)
        
        # Test Python file triggers
        triggered_agents = trigger_engine.get_agents_for_file_change('src/api/auth.py')
        expected_agents = {'security_auditor', 'test_generator', 'python_backend_coder'}
        
        assert expected_agents.issubset(set(triggered_agents)), \
            f"Python file should trigger {expected_agents}, got {triggered_agents}"
            
        # Test UI file triggers  
        ui_triggered = trigger_engine.get_agents_for_file_change('src/components/Login.tsx')
        expected_ui_agents = {'ui_ux_designer', 'test_generator', 'typescript_frontend_agent'}
        
        assert expected_ui_agents.issubset(set(ui_triggered)), \
            f"UI file should trigger {expected_ui_agents}, got {ui_triggered}"
            
    def test_agent_communication_protocol(self):
        """Test inter-agent communication works correctly"""
        # Create a workflow requiring agent communication
        workflow = {
            'primary': 'python_backend_coder',
            'dependencies': ['security_auditor', 'test_generator'],
            'task': 'Create secure API endpoint with tests'
        }
        
        # Test that primary agent can request work from dependencies
        primary_agent = self.orchestrator.agent_pool.get_agent('python_backend_coder')
        
        # Mock the communication
        with patch.object(primary_agent, 'request_dependency_work') as mock_request:
            mock_request.return_value = {'status': 'completed', 'results': []}
            
            result = primary_agent.execute_with_dependencies(workflow)
            
            assert mock_request.called
            assert result is not None
            
    def test_claude_code_integration_bridge(self):
        """Test seamless integration with Claude Code Task tool"""
        # This would test the actual bridge to Claude Code
        # For now, test the interface structure
        
        bridge = self.orchestrator.get_claude_code_bridge()
        
        # Test required methods exist
        assert hasattr(bridge, 'spawn_agent')
        assert hasattr(bridge, 'get_agent_result') 
        assert hasattr(bridge, 'escalate_to_claude')
        assert hasattr(bridge, 'register_task_completion')
        
    def test_agent_performance_monitoring(self):
        """Test that agent performance is properly monitored"""
        # Create and execute a task
        task = self.orchestrator.create_task(
            agent_role='security_auditor',
            description='Test security analysis',
            input_data={'files': ['test.py']}
        )
        
        # Mock execution
        with patch.object(self.orchestrator.executor, 'execute_task') as mock_execute:
            mock_execute.return_value = {
                'status': 'completed',
                'execution_time': 2.5,
                'memory_usage': 45.2
            }
            
            result = self.orchestrator.execute_single_task(task)
            
            # Verify metrics are captured
            assert 'execution_time' in result
            assert 'memory_usage' in result
            assert result['execution_time'] > 0
            
    def test_agent_error_handling_and_recovery(self):
        """Test agent error handling and recovery mechanisms"""
        # Test agent failure recovery
        failing_task = self.orchestrator.create_task(
            agent_role='test_generator',
            description='Task designed to fail',
            input_data={'invalid': True}
        )
        
        with patch.object(self.orchestrator.executor, 'execute_task') as mock_execute:
            mock_execute.side_effect = Exception("Agent execution failed")
            
            # Should not crash orchestrator
            result = self.orchestrator.execute_single_task(failing_task)
            
            assert result['status'] == 'failed'
            assert 'error' in result
            
        # Test agent pool recovery - spawn replacement agent
        original_count = len(self.orchestrator.agent_pool.get_agents_by_role('test_generator'))
        
        # Simulate agent failure
        self.orchestrator.agent_pool.mark_agent_failed('test_generator', "Test failure")
        
        # Should auto-spawn replacement
        new_count = len(self.orchestrator.agent_pool.get_agents_by_role('test_generator'))
        assert new_count >= original_count, "Failed agent should be replaced"
        
    def test_resource_management(self):
        """Test proper resource management and cleanup"""
        initial_memory = self.orchestrator.get_system_memory_usage()
        
        # Create multiple tasks to stress test
        tasks = []
        for i in range(10):
            task = self.orchestrator.create_task(
                agent_role='security_auditor',
                description=f'Test task {i}',
                input_data={'task_id': i}
            )
            tasks.append(task)
            
        # Execute tasks
        for task in tasks:
            with patch.object(self.orchestrator.executor, 'execute_task') as mock_execute:
                mock_execute.return_value = {'status': 'completed'}
                self.orchestrator.execute_single_task(task)
                
        # Memory should not have increased significantly
        final_memory = self.orchestrator.get_system_memory_usage()
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100, f"Memory leak detected: {memory_increase}MB increase"

class TestAgentWorkflows:
    """Test specific agent workflows and scenarios"""
    
    def setup_method(self):
        """Set up test environment"""
        self.orchestrator = ArchonOrchestrator()
        
    def test_full_development_workflow(self):
        """Test complete development workflow with multiple agents"""
        # Simulate: User wants to add authentication API
        workflow_steps = [
            {
                'step': 'Architecture Design',
                'agent': 'system_architect',
                'input': {'feature': 'authentication API'},
                'expected_output': ['design_document', 'api_specification']
            },
            {
                'step': 'Security Review',
                'agent': 'security_auditor', 
                'input': {'design': 'auth_design.md'},
                'expected_output': ['security_analysis', 'recommendations']
            },
            {
                'step': 'Backend Implementation',
                'agent': 'python_backend_coder',
                'input': {'specification': 'auth_api_spec.json'},
                'expected_output': ['auth_endpoints.py', 'middleware.py']
            },
            {
                'step': 'Test Generation',
                'agent': 'test_generator',
                'input': {'source_files': ['auth_endpoints.py']},
                'expected_output': ['test_auth.py', 'integration_tests.py']
            }
        ]
        
        results = []
        for step in workflow_steps:
            with patch.object(self.orchestrator.executor, 'execute_task') as mock_execute:
                mock_execute.return_value = {
                    'status': 'completed',
                    'outputs': step['expected_output']
                }
                
                task = self.orchestrator.create_task(
                    agent_role=step['agent'],
                    description=step['step'],
                    input_data=step['input']
                )
                
                result = self.orchestrator.execute_single_task(task)
                results.append(result)
                
        # Verify all steps completed
        assert all(r['status'] == 'completed' for r in results)
        assert len(results) == 4
        
    def test_concurrent_agent_execution(self):
        """Test multiple agents working concurrently"""
        # Create tasks that can run in parallel
        parallel_tasks = [
            ('security_auditor', 'Audit security vulnerabilities'),
            ('performance_optimizer', 'Analyze performance bottlenecks'), 
            ('documentation_writer', 'Update API documentation'),
            ('code_reviewer', 'Review recent code changes')
        ]
        
        tasks = []
        for agent_role, description in parallel_tasks:
            task = self.orchestrator.create_task(
                agent_role=agent_role,
                description=description,
                input_data={'concurrent': True}
            )
            tasks.append(task)
            
        # Execute all tasks concurrently
        with patch.object(self.orchestrator.executor, 'execute_tasks_parallel') as mock_parallel:
            mock_parallel.return_value = [
                {'status': 'completed', 'agent': task.agent_role} 
                for task in tasks
            ]
            
            results = self.orchestrator.execute_parallel_tasks(tasks)
            
            assert len(results) == 4
            assert all(r['status'] == 'completed' for r in results)

# NLNH/DGTS Anti-Gaming Tests
class TestAntiGaming:
    """Tests to prevent gaming and ensure genuine implementation"""
    
    def test_no_mock_agent_execution(self):
        """DGTS: Ensure agents actually execute, not just return fake results"""
        orchestrator = ArchonOrchestrator()
        
        task = orchestrator.create_task(
            agent_role='security_auditor',
            description='Real security audit',
            input_data={'files': ['auth.py']}
        )
        
        # This should fail if agents are just returning mock data
        result = orchestrator.execute_single_task(task)
        
        # DGTS Check: Verify result has real execution traces
        assert 'execution_time' in result
        assert result['execution_time'] > 0
        assert 'agent_id' in result
        assert result['agent_id'] is not None
        
    def test_agent_configs_not_empty(self):
        """DGTS: Verify agent configurations are real, not placeholder"""
        orchestrator = ArchonOrchestrator()
        
        for agent_name, config in orchestrator.executor.agent_configs.items():
            # DGTS: Each config must have real content
            assert config['role'] != 'placeholder'
            assert len(config['capabilities']) > 0
            assert 'execution_context' in config
            
            # DGTS: No TODO or incomplete markers
            config_str = json.dumps(config)
            assert 'TODO' not in config_str
            assert 'placeholder' not in config_str.lower()
            assert 'mock' not in config_str.lower()
            
    def test_tool_access_actually_restricted(self):
        """DGTS: Verify tool restrictions are enforced, not just documented"""
        orchestrator = ArchonOrchestrator()
        
        # Try to give security_auditor unauthorized tools
        with pytest.raises(Exception):
            orchestrator.assign_tool_to_agent('security_auditor', 'ui_design_tool')
            
        # Try to access files outside scope
        with pytest.raises(Exception):
            orchestrator.execute_agent_tool('security_auditor', 'Read', {'file': '/etc/passwd'})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])