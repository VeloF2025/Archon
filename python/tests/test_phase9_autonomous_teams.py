"""
Comprehensive tests for Phase 9 Autonomous Development Teams

This test suite validates all components of the autonomous development teams system,
including team assembly, workflow orchestration, performance tracking, and
global knowledge network functionality.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Import the modules being tested
from src.agents.autonomous_teams.team_assembly import (
    TeamAssemblyEngine, ProjectRequirements, TeamComposition, AgentCapability
)
from src.agents.autonomous_teams.workflow_orchestrator import (
    WorkflowOrchestrator, WorkflowExecution, WorkflowTask, WorkflowPhase, TaskStatus
)
from src.agents.autonomous_teams.team_performance_tracker import (
    TeamPerformanceTracker, PerformanceDataPoint, PerformanceMetric, TeamPerformanceProfile
)
from src.agents.autonomous_teams.cross_project_knowledge import (
    CrossProjectKnowledgeEngine, ProjectMetrics, ProjectOutcome, IdentifiedPattern, PatternType
)
from src.agents.autonomous_teams.global_knowledge_network import (
    GlobalKnowledgeNetwork, KnowledgeItem, KnowledgeType, NetworkQuery, NetworkRole, PrivacyLevel
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
async def team_assembly_engine():
    """Create team assembly engine for testing."""
    engine = TeamAssemblyEngine()
    return engine


@pytest.fixture
async def workflow_orchestrator():
    """Create workflow orchestrator for testing."""
    orchestrator = WorkflowOrchestrator()
    return orchestrator


@pytest.fixture
async def performance_tracker(temp_storage):
    """Create performance tracker for testing."""
    tracker = TeamPerformanceTracker(storage_path=temp_storage)
    return tracker


@pytest.fixture
async def knowledge_engine(temp_storage):
    """Create cross-project knowledge engine for testing."""
    engine = CrossProjectKnowledgeEngine(storage_path=temp_storage)
    return engine


@pytest.fixture
async def knowledge_network(temp_storage):
    """Create global knowledge network for testing."""
    node_config = {
        "name": "Test Development Team",
        "organization_type": "development_team",
        "role": NetworkRole.CONTRIBUTOR,
        "domains": ["web_development", "mobile_apps"],
        "technologies": ["React", "Node.js", "Python"],
        "expertise_level": {
            "web_development": 8.5,
            "mobile_apps": 7.0
        }
    }
    network = GlobalKnowledgeNetwork(node_config, storage_path=temp_storage)
    return network


class TestTeamAssembly:
    """Test team assembly functionality."""
    
    @pytest.mark.asyncio
    async def test_create_project_requirements(self):
        """Test creating project requirements."""
        requirements = ProjectRequirements(
            project_name="E-commerce Platform",
            project_description="Modern e-commerce platform with React frontend",
            technical_requirements={"frontend": "React", "backend": "Node.js"},
            timeline_weeks=12
        )
        
        assert requirements.project_name == "E-commerce Platform"
        assert requirements.timeline_weeks == 12
        assert "frontend" in requirements.technical_requirements
    
    @pytest.mark.asyncio
    async def test_assemble_team_basic(self, team_assembly_engine):
        """Test basic team assembly."""
        requirements = ProjectRequirements(
            project_name="Test Project",
            project_description="A test project for team assembly",
            technical_requirements={"language": "Python", "framework": "FastAPI"},
            complexity_score=7.5,
            timeline_weeks=8
        )
        
        team_composition = await team_assembly_engine.assemble_team(requirements)
        
        assert team_composition is not None
        assert team_composition.project_id == requirements.project_id
        assert len(team_composition.agents) > 0
        assert team_composition.total_cost > 0
        assert team_composition.estimated_efficiency > 0
    
    @pytest.mark.asyncio
    async def test_team_composition_has_required_roles(self, team_assembly_engine):
        """Test that team composition includes required roles."""
        requirements = ProjectRequirements(
            project_name="Web Application",
            project_description="Full-stack web application",
            technical_requirements={"frontend": "React", "backend": "Node.js", "database": "PostgreSQL"},
            complexity_score=8.0,
            timeline_weeks=16
        )
        
        team_composition = await team_assembly_engine.assemble_team(requirements)
        
        # Check that we have essential roles
        agent_roles = [agent.primary_role for agent in team_composition.agents]
        
        assert "architect" in agent_roles or "system_architect" in agent_roles
        assert any("developer" in role for role in agent_roles)
        assert any("tester" in role or "qa" in role for role in agent_roles)
    
    @pytest.mark.asyncio
    async def test_team_optimization(self, team_assembly_engine):
        """Test team composition optimization."""
        requirements = ProjectRequirements(
            project_name="Complex System",
            project_description="High-complexity system with multiple components",
            technical_requirements={
                "microservices": True,
                "database": "PostgreSQL",
                "cache": "Redis",
                "frontend": "React"
            },
            complexity_score=9.0,
            timeline_weeks=24
        )
        
        # Test with different preferences
        preferences1 = {"team_size_preference": "small", "cost_priority": 0.8}
        preferences2 = {"team_size_preference": "large", "efficiency_priority": 0.9}
        
        team1 = await team_assembly_engine.assemble_team(requirements, preferences1)
        team2 = await team_assembly_engine.assemble_team(requirements, preferences2)
        
        # Small team should have fewer agents and lower cost
        assert len(team1.agents) <= len(team2.agents)
        assert team1.total_cost <= team2.total_cost


class TestWorkflowOrchestration:
    """Test workflow orchestration functionality."""
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, workflow_orchestrator):
        """Test workflow creation."""
        workflow = await workflow_orchestrator.create_workflow(
            project_id="test-project",
            workflow_name="Test Workflow",
            template_name="web_application"
        )
        
        assert workflow.project_id == "test-project"
        assert workflow.name == "Test Workflow"
        assert len(workflow.tasks) > 0
        assert workflow.estimated_duration_hours > 0
    
    @pytest.mark.asyncio
    async def test_workflow_task_dependencies(self, workflow_orchestrator):
        """Test that workflow tasks have proper dependencies."""
        workflow = await workflow_orchestrator.create_workflow(
            project_id="dep-test",
            workflow_name="Dependency Test",
            template_name="api_service"
        )
        
        # Check that implementation tasks depend on architecture tasks
        implementation_tasks = [task for task in workflow.tasks if task.phase == WorkflowPhase.IMPLEMENTATION]
        architecture_tasks = [task for task in workflow.tasks if task.phase == WorkflowPhase.ARCHITECTURE_DESIGN]
        
        if implementation_tasks and architecture_tasks:
            impl_task = implementation_tasks[0]
            arch_task_ids = [task.id for task in architecture_tasks]
            
            # Implementation should depend on architecture
            assert any(dep_id in arch_task_ids for dep_id in impl_task.dependencies)
    
    @pytest.mark.asyncio
    async def test_workflow_execution_start(self, workflow_orchestrator):
        """Test starting workflow execution."""
        workflow = await workflow_orchestrator.create_workflow(
            project_id="exec-test",
            workflow_name="Execution Test",
            template_name="web_application"
        )
        
        success = await workflow_orchestrator.start_workflow(workflow.id)
        assert success
        
        status = await workflow_orchestrator.get_workflow_status(workflow.id)
        assert status is not None
        assert status["status"] == "in_progress"
    
    @pytest.mark.asyncio
    async def test_workflow_ready_tasks_identification(self, workflow_orchestrator):
        """Test identification of ready-to-execute tasks."""
        workflow = await workflow_orchestrator.create_workflow(
            project_id="ready-test",
            workflow_name="Ready Tasks Test",
            template_name="api_service"
        )
        
        # Find tasks without dependencies (should be ready)
        no_dep_tasks = [task for task in workflow.tasks if not task.dependencies]
        assert len(no_dep_tasks) > 0
        
        # These should be identified as ready
        ready_tasks = workflow_orchestrator._find_ready_tasks(workflow)
        ready_task_ids = [task.id for task in ready_tasks]
        
        for task in no_dep_tasks:
            assert task.id in ready_task_ids or task.status == TaskStatus.READY


class TestPerformanceTracking:
    """Test performance tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_create_team_profile(self, performance_tracker):
        """Test creating a team performance profile."""
        profile = await performance_tracker.create_team_profile(
            team_id="test-team-001",
            team_name="Test Development Team",
            team_members=["agent-1", "agent-2", "agent-3"]
        )
        
        assert profile.team_id == "test-team-001"
        assert profile.team_name == "Test Development Team"
        assert len(profile.team_members) == 3
        assert len(profile.current_metrics) == len(PerformanceMetric)
    
    @pytest.mark.asyncio
    async def test_record_performance_data(self, performance_tracker):
        """Test recording performance data."""
        # Create team profile first
        await performance_tracker.create_team_profile(
            team_id="perf-team",
            team_name="Performance Test Team",
            team_members=["agent-1", "agent-2"]
        )
        
        # Record performance data
        data_point = PerformanceDataPoint(
            team_id="perf-team",
            metric=PerformanceMetric.TASK_COMPLETION_RATE,
            value=0.85,
            project_id="test-project"
        )
        
        await performance_tracker.record_performance_data(data_point)
        
        # Check that data was recorded
        profile = performance_tracker.team_profiles["perf-team"]
        assert profile.current_metrics[PerformanceMetric.TASK_COMPLETION_RATE] == 0.85
        assert len(profile.historical_data[PerformanceMetric.TASK_COMPLETION_RATE]) == 1
    
    @pytest.mark.asyncio
    async def test_performance_score_calculation(self, performance_tracker):
        """Test performance score calculation."""
        # Create team and record multiple metrics
        await performance_tracker.create_team_profile(
            team_id="score-team",
            team_name="Score Test Team",
            team_members=["agent-1"]
        )
        
        metrics_data = [
            (PerformanceMetric.TASK_COMPLETION_RATE, 0.90),
            (PerformanceMetric.QUALITY_SCORE, 8.5),
            (PerformanceMetric.COLLABORATION_EFFICIENCY, 7.8),
            (PerformanceMetric.CLIENT_SATISFACTION, 8.2)
        ]
        
        for metric, value in metrics_data:
            data_point = PerformanceDataPoint(
                team_id="score-team",
                metric=metric,
                value=value
            )
            await performance_tracker.record_performance_data(data_point)
        
        profile = performance_tracker.team_profiles["score-team"]
        assert profile.performance_score > 0
        assert profile.performance_score <= 10
    
    @pytest.mark.asyncio
    async def test_performance_analytics(self, performance_tracker):
        """Test performance analytics generation."""
        # Create team with historical data
        team_id = "analytics-team"
        await performance_tracker.create_team_profile(
            team_id=team_id,
            team_name="Analytics Test Team",
            team_members=["agent-1", "agent-2"]
        )
        
        # Add multiple data points over time
        base_time = datetime.now() - timedelta(days=30)
        for i in range(10):
            data_point = PerformanceDataPoint(
                timestamp=base_time + timedelta(days=i * 3),
                team_id=team_id,
                metric=PerformanceMetric.TASK_COMPLETION_RATE,
                value=0.7 + (i * 0.02)  # Improving trend
            )
            await performance_tracker.record_performance_data(data_point)
        
        analytics = await performance_tracker.analyze_team_performance(team_id)
        
        assert analytics is not None
        assert analytics.team_id == team_id
        assert len(analytics.performance_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, performance_tracker):
        """Test generation of optimization recommendations."""
        # Create team with suboptimal performance
        team_id = "opt-team"
        await performance_tracker.create_team_profile(
            team_id=team_id,
            team_name="Optimization Test Team",
            team_members=["agent-1", "agent-2"]
        )
        
        # Record poor performance metrics
        poor_metrics = [
            (PerformanceMetric.TASK_COMPLETION_RATE, 0.65),  # Below threshold
            (PerformanceMetric.QUALITY_SCORE, 6.5),         # Below threshold
            (PerformanceMetric.COLLABORATION_EFFICIENCY, 5.8)  # Below threshold
        ]
        
        for metric, value in poor_metrics:
            data_point = PerformanceDataPoint(
                team_id=team_id,
                metric=metric,
                value=value
            )
            await performance_tracker.record_performance_data(data_point)
        
        optimizations = await performance_tracker.generate_optimization_recommendations(team_id)
        
        assert len(optimizations) > 0
        assert all(opt.expected_impact > 0 for opt in optimizations)
        assert all(opt.priority >= 1 and opt.priority <= 10 for opt in optimizations)


class TestCrossProjectKnowledge:
    """Test cross-project knowledge synthesis."""
    
    @pytest.mark.asyncio
    async def test_add_project_metrics(self, knowledge_engine):
        """Test adding project metrics to knowledge engine."""
        metrics = ProjectMetrics(
            project_id="test-proj-001",
            name="Test E-commerce Platform",
            outcome=ProjectOutcome.SUCCESS,
            completion_time_hours=320,
            estimated_time_hours=300,
            team_size=5,
            technologies_used=["React", "Node.js", "PostgreSQL"],
            architectural_patterns=["Microservices", "Event-Driven"],
            test_coverage=0.85,
            performance_score=8.5,
            security_score=9.0
        )
        
        success = await knowledge_engine.add_project_metrics(metrics)
        assert success
        assert "test-proj-001" in knowledge_engine.project_metrics
    
    @pytest.mark.asyncio
    async def test_pattern_identification(self, knowledge_engine):
        """Test identification of patterns from project data."""
        # Add multiple successful projects with common patterns
        successful_projects = [
            ProjectMetrics(
                project_id=f"success-{i}",
                name=f"Successful Project {i}",
                outcome=ProjectOutcome.SUCCESS,
                team_size=5,
                technologies_used=["React", "Node.js", "PostgreSQL"],
                architectural_patterns=["Microservices"],
                test_coverage=0.85 + (i * 0.02),
                performance_score=8.0 + i,
                completion_time_hours=300 + (i * 10),
                estimated_time_hours=300
            )
            for i in range(3)
        ]
        
        for metrics in successful_projects:
            await knowledge_engine.add_project_metrics(metrics)
        
        # Patterns should be identified
        assert len(knowledge_engine.identified_patterns) > 0
        
        # Check for technology patterns
        tech_patterns = [p for p in knowledge_engine.identified_patterns.values() 
                        if "React" in p.name or "Node.js" in p.name]
        assert len(tech_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_anti_pattern_detection(self, knowledge_engine):
        """Test detection of anti-patterns from failed projects."""
        # Add failed projects with common failure patterns
        failed_projects = [
            ProjectMetrics(
                project_id=f"failed-{i}",
                name=f"Failed Project {i}",
                outcome=ProjectOutcome.FAILURE,
                team_size=2,  # Too small
                technologies_used=["OldFramework", "LegacyDB"],
                test_coverage=0.35,  # Poor coverage
                performance_score=4.0,
                completion_time_hours=800,  # Major overrun
                estimated_time_hours=300
            )
            for i in range(3)
        ]
        
        for metrics in failed_projects:
            await knowledge_engine.add_project_metrics(metrics)
        
        # Anti-patterns should be identified
        anti_patterns = [p for p in knowledge_engine.identified_patterns.values() 
                        if p.type == PatternType.ANTI_PATTERN]
        assert len(anti_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_project_recommendations(self, knowledge_engine):
        """Test getting recommendations for new projects."""
        # Add some successful projects first
        await knowledge_engine.add_project_metrics(ProjectMetrics(
            project_id="rec-test-1",
            name="Successful Web App",
            outcome=ProjectOutcome.SUCCESS,
            technologies_used=["React", "Node.js"],
            team_size=5,
            test_coverage=0.90,
            performance_score=9.0
        ))
        
        recommendations = await knowledge_engine.get_recommendations_for_project(
            technologies=["React", "Node.js"],
            team_size=5,
            estimated_duration=350.0
        )
        
        assert len(recommendations) > 0
        assert all("type" in rec for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_knowledge_synthesis_generation(self, knowledge_engine):
        """Test generation of comprehensive knowledge synthesis."""
        # Add diverse project data
        projects_data = [
            ("web-1", ProjectOutcome.SUCCESS, ["React", "Node.js"], 5),
            ("web-2", ProjectOutcome.SUCCESS, ["React", "Python"], 4),
            ("mobile-1", ProjectOutcome.FAILURE, ["ReactNative", "Firebase"], 3),
            ("api-1", ProjectOutcome.SUCCESS, ["Python", "FastAPI"], 3)
        ]
        
        for proj_id, outcome, techs, team_size in projects_data:
            await knowledge_engine.add_project_metrics(ProjectMetrics(
                project_id=proj_id,
                name=f"Project {proj_id}",
                outcome=outcome,
                technologies_used=techs,
                team_size=team_size
            ))
        
        synthesis = await knowledge_engine.generate_knowledge_synthesis()
        
        assert synthesis.projects_analyzed == len(projects_data)
        assert len(synthesis.key_insights) > 0
        assert len(synthesis.recommendations) > 0


class TestGlobalKnowledgeNetwork:
    """Test global knowledge network functionality."""
    
    @pytest.mark.asyncio
    async def test_contribute_knowledge(self, knowledge_network):
        """Test contributing knowledge to the network."""
        knowledge_item = KnowledgeItem(
            type=KnowledgeType.BEST_PRACTICE,
            title="React Performance Optimization",
            description="Best practices for React component optimization",
            content={
                "technique": "Use React.memo for expensive components",
                "example": "const OptimizedComponent = React.memo(MyComponent)"
            },
            domain="web_development",
            technologies=["React", "JavaScript"],
            complexity_level=6,
            success_rate=0.85,
            confidence_score=8.5,
            privacy_level=PrivacyLevel.ANONYMIZED
        )
        
        success = await knowledge_network.contribute_knowledge(knowledge_item, share_globally=False)
        assert success
        assert knowledge_item.id in knowledge_network.knowledge_items
    
    @pytest.mark.asyncio
    async def test_query_local_knowledge(self, knowledge_network):
        """Test querying local knowledge store."""
        # Add knowledge first
        knowledge_item = KnowledgeItem(
            type=KnowledgeType.SOLUTION_TEMPLATE,
            title="Authentication Template",
            description="JWT authentication implementation template",
            content={"implementation": "auth_code_here"},
            domain="web_development",
            technologies=["Node.js", "JWT"],
            complexity_level=5
        )
        
        await knowledge_network.contribute_knowledge(knowledge_item, share_globally=False)
        
        # Query for it
        query = NetworkQuery(
            domain="web_development",
            technologies=["Node.js"],
            knowledge_types=[KnowledgeType.SOLUTION_TEMPLATE],
            max_results=10
        )
        
        results = await knowledge_network.query_network(query)
        assert len(results) > 0
        assert any(item.title == "Authentication Template" for item in results)
    
    @pytest.mark.asyncio
    async def test_knowledge_feedback(self, knowledge_network):
        """Test providing feedback on knowledge usage."""
        # Add knowledge
        knowledge_item = KnowledgeItem(
            type=KnowledgeType.BEST_PRACTICE,
            title="Test Knowledge",
            description="Test knowledge for feedback",
            content={"test": "content"}
        )
        
        await knowledge_network.contribute_knowledge(knowledge_item, share_globally=False)
        
        # Provide positive feedback
        success = await knowledge_network.provide_feedback(
            knowledge_id=knowledge_item.id,
            success=True,
            details="Worked great!"
        )
        
        assert success
        
        # Check that feedback was recorded
        item = knowledge_network.knowledge_items[knowledge_item.id]
        assert item.success_feedback == 1
        assert item.failure_feedback == 0
        assert item.usage_count == 1
    
    @pytest.mark.asyncio
    async def test_content_anonymization(self, knowledge_network):
        """Test content anonymization based on privacy levels."""
        content = {
            "organization": "Acme Corp",
            "project_name": "Secret Project",
            "author": "John Doe",
            "technical_details": "Implementation details"
        }
        
        # Test different privacy levels
        public_content = knowledge_network._anonymize_content(content, PrivacyLevel.PUBLIC)
        assert public_content == content  # No changes for public
        
        anonymized_content = knowledge_network._anonymize_content(content, PrivacyLevel.ANONYMIZED)
        assert "anon_" in str(anonymized_content.get("organization", ""))
        assert "technical_details" in anonymized_content  # Technical content preserved
        
        aggregated_content = knowledge_network._anonymize_content(content, PrivacyLevel.AGGREGATED)
        assert "summary_stats" in aggregated_content or "category_counts" in aggregated_content
    
    @pytest.mark.asyncio
    async def test_network_analytics(self, knowledge_network):
        """Test generation of network analytics."""
        # Add some knowledge items first
        for i in range(3):
            knowledge_item = KnowledgeItem(
                title=f"Knowledge Item {i}",
                domain="web_development",
                technologies=["React"],
                success_rate=0.8 + (i * 0.05)
            )
            await knowledge_network.contribute_knowledge(knowledge_item, share_globally=False)
        
        analytics = await knowledge_network.generate_network_analytics()
        
        assert analytics.total_knowledge_items >= 3
        assert "web_development" in analytics.knowledge_by_domain
        assert analytics.knowledge_by_domain["web_development"] >= 3


class TestIntegration:
    """Integration tests for autonomous teams components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_team_workflow(self, team_assembly_engine, workflow_orchestrator, performance_tracker):
        """Test complete end-to-end workflow from team assembly to execution."""
        
        # Step 1: Define project requirements
        requirements = ProjectRequirements(
            project_name="Integration Test Project",
            project_description="End-to-end integration test",
            technical_requirements={"frontend": "React", "backend": "FastAPI"},
            timeline_weeks=10
        )
        
        # Step 2: Assemble team
        team_composition = await team_assembly_engine.assemble_team(requirements)
        assert len(team_composition.agents) > 0
        
        # Step 3: Create performance profile
        profile = await performance_tracker.create_team_profile(
            team_id=team_composition.team_id,
            team_name=f"{requirements.project_name} Team",
            team_members=[agent.id for agent in team_composition.agents]
        )
        assert profile.team_id == team_composition.team_id
        
        # Step 4: Create and start workflow
        workflow = await workflow_orchestrator.create_workflow(
            project_id=requirements.project_id,
            workflow_name=f"{requirements.project_name} Workflow",
            template_name="web_application"
        )
        assert workflow.project_id == requirements.project_id
        
        # Step 5: Record some performance data
        data_point = PerformanceDataPoint(
            team_id=team_composition.team_id,
            metric=PerformanceMetric.TASK_COMPLETION_RATE,
            value=0.80,
            project_id=requirements.project_id
        )
        await performance_tracker.record_performance_data(data_point)
        
        # Verify integration
        dashboard_data = await performance_tracker.get_team_dashboard_data(team_composition.team_id)
        assert dashboard_data is not None
        assert dashboard_data["team_info"]["id"] == team_composition.team_id
    
    @pytest.mark.asyncio
    async def test_knowledge_flow_between_components(self, knowledge_engine, knowledge_network):
        """Test knowledge flow between cross-project engine and global network."""
        
        # Step 1: Add project metrics to knowledge engine
        project_metrics = ProjectMetrics(
            project_id="knowledge-flow-test",
            name="Knowledge Flow Test Project",
            outcome=ProjectOutcome.SUCCESS,
            technologies_used=["Python", "FastAPI", "PostgreSQL"],
            architectural_patterns=["Microservices"],
            test_coverage=0.90,
            performance_score=8.8
        )
        
        await knowledge_engine.add_project_metrics(project_metrics)
        
        # Step 2: Extract patterns should generate knowledge
        patterns = [p for p in knowledge_engine.identified_patterns.values() 
                   if PatternType.SUCCESS_PATTERN == p.type]
        
        # Step 3: Convert pattern to knowledge item and contribute to network
        if patterns:
            pattern = patterns[0]
            knowledge_item = KnowledgeItem(
                type=KnowledgeType.PATTERN,
                title=pattern.name,
                description=pattern.description,
                content={
                    "pattern_type": pattern.type.value,
                    "evidence_projects": pattern.evidence_projects,
                    "success_correlation": pattern.success_correlation,
                    "implementation_guide": pattern.implementation_guide
                },
                domain="software_development",
                technologies=["Python", "FastAPI"],
                complexity_level=6,
                success_rate=pattern.success_correlation if pattern.success_correlation > 0 else 0.8,
                confidence_score=8.0
            )
            
            success = await knowledge_network.contribute_knowledge(knowledge_item)
            assert success
            
            # Step 4: Query network should return the knowledge
            query = NetworkQuery(
                technologies=["Python"],
                knowledge_types=[KnowledgeType.PATTERN],
                max_results=10
            )
            
            results = await knowledge_network.query_network(query)
            assert len(results) > 0
            assert any(item.title == pattern.name for item in results)


# Test utilities and fixtures

@pytest.fixture
def sample_project_requirements():
    """Sample project requirements for testing."""
    return ProjectRequirements(
        project_name="Sample Web Application",
        project_description="A sample web application for testing purposes",
        technical_requirements={
            "frontend": "React",
            "backend": "Node.js",
            "database": "PostgreSQL",
            "authentication": "JWT"
        },
        functional_requirements={
            "user_management": True,
            "data_persistence": True,
            "api_endpoints": True
        },
        non_functional_requirements={
            "performance": "high",
            "security": "standard",
            "scalability": "medium"
        },
        timeline_weeks=12
    )


@pytest.fixture
def sample_team_composition():
    """Sample team composition for testing."""
    agents = [
        AgentCapability(
            id="arch-001",
            name="Senior Architect",
            primary_role="system_architect",
            skills={"system_design": 9, "architecture_patterns": 8},
            cost_per_hour=150.0,
            availability=0.8
        ),
        AgentCapability(
            id="dev-001",
            name="Full Stack Developer",
            primary_role="full_stack_developer",
            skills={"react": 8, "nodejs": 8, "postgresql": 7},
            cost_per_hour=120.0,
            availability=1.0
        ),
        AgentCapability(
            id="test-001",
            name="QA Engineer",
            primary_role="tester",
            skills={"test_automation": 8, "manual_testing": 9},
            cost_per_hour=100.0,
            availability=0.9
        )
    ]
    
    from src.agents.autonomous_teams.team_assembly import TeamComposition
    return TeamComposition(
        project_id="sample-project",
        team_id="sample-team",
        agents=agents,
        total_cost=sum(agent.cost_per_hour for agent in agents),
        estimated_efficiency=8.5,
        confidence_score=0.85
    )


# Parametrized tests for different scenarios

@pytest.mark.parametrize("project_type,expected_agents", [
    ("web_application", 4),  # Expect at least 4 agents for web app
    ("api_service", 3),      # Expect at least 3 agents for API service
    ("mobile_app", 4),       # Expect at least 4 agents for mobile app
])
@pytest.mark.asyncio
async def test_team_assembly_by_project_type(team_assembly_engine, project_type, expected_agents):
    """Test team assembly for different project types."""
    requirements = ProjectRequirements(
        project_name=f"Test {project_type}",
        project_description=f"A test {project_type} project",
        technical_requirements={"type": project_type},
        timeline_weeks=8
    )
    
    team_composition = await team_assembly_engine.assemble_team(requirements)
    assert len(team_composition.agents) >= expected_agents


@pytest.mark.parametrize("complexity,expected_team_size_min", [
    (3, 2),   # Simple project, small team
    (7, 4),   # Medium project, medium team
    (9, 6),   # Complex project, larger team
])
@pytest.mark.asyncio
async def test_team_size_by_complexity(team_assembly_engine, complexity, expected_team_size_min):
    """Test that team size scales with project complexity."""
    requirements = ProjectRequirements(
        project_name="Complexity Test",
        project_description="Testing complexity scaling",
        complexity_score=complexity,
        timeline_weeks=12
    )
    
    team_composition = await team_assembly_engine.assemble_team(requirements)
    assert len(team_composition.agents) >= expected_team_size_min


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "--tb=short"])