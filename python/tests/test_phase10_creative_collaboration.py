"""
Comprehensive test suite for Phase 10: Creative AI Collaboration

Tests all components of the creative collaboration system including:
- Creative Agent System with 8 specialized roles
- AI Collaboration Partner with personalized adaptation
- Collaborative Design Platform with real-time co-design
- Innovation Acceleration Engine with breakthrough detection
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import json
import uuid

# Test imports
from src.agents.creative_collaboration.creative_agent_system import (
    CreativeAgentSystem, CreativeProblem, CreativeSession, CreativeAgent,
    CreativePhase, SessionParticipant, AgentRole, SessionStatus
)
from src.agents.creative_collaboration.ai_collaboration_partner import (
    AICollaborationPartner, DeveloperProfile, CollaborationContext,
    DeveloperPersonality, CodingStyle, CollaborationHistory
)
from src.agents.creative_collaboration.collaborative_design_platform import (
    CollaborativeDesignPlatform, DesignCanvas, DesignElement, DesignSession,
    DesignPhase, AIDesignAgent, DesignElementType
)
from src.agents.creative_collaboration.innovation_acceleration_engine import (
    InnovationAccelerationEngine, CreativeProblem as InnovativeProblem,
    SolutionConcept, CrossDomainInsight, BreakthroughIndicator,
    InnovationMetrics, InnovationType, ProblemComplexity, SolutionStatus
)

# API test imports
from src.server.api_routes.creative_collaboration_api import (
    ProblemRequest, SessionRequest, DeveloperProfileRequest,
    CollaborationRequest, DesignSessionRequest, DesignElementRequest,
    InnovationProblemRequest
)

class TestCreativeAgentSystem:
    """Test suite for Creative Agent System with specialized AI roles"""
    
    @pytest.fixture
    def creative_system(self):
        return CreativeAgentSystem()
    
    @pytest.fixture
    def sample_problem(self):
        return CreativeProblem(
            title="Redesign User Onboarding Experience",
            description="Create an engaging and efficient user onboarding flow",
            domain="ux_design",
            success_criteria=["Reduce drop-off rate", "Increase user engagement"],
            constraints=["Mobile-first design", "Accessibility compliance"],
            stakeholders=["Product team", "Users", "Support team"]
        )
    
    @pytest.mark.asyncio
    async def test_problem_registration(self, creative_system, sample_problem):
        """Test registering a creative problem"""
        problem_id = await creative_system.register_problem(sample_problem)
        
        assert problem_id is not None
        assert len(problem_id) > 0
        
        # Verify problem was stored
        problems = await creative_system.get_problems()
        assert len(problems) == 1
        assert problems[0].title == sample_problem.title
        
    @pytest.mark.asyncio
    async def test_creative_session_startup(self, creative_system, sample_problem):
        """Test starting a creative collaboration session"""
        problem_id = await creative_system.register_problem(sample_problem)
        
        session_id = await creative_system.start_creative_session(
            problem_id=problem_id,
            human_participants=["alice@example.com", "bob@example.com"],
            session_type="brainstorming",
            duration_minutes=90
        )
        
        assert session_id is not None
        
        # Verify session details
        session = await creative_system.get_session(session_id)
        assert session is not None
        assert session.problem_id == problem_id
        assert session.status == SessionStatus.ACTIVE
        assert session.current_phase == CreativePhase.IDEATION
        assert len(session.participants) == 2  # Human participants
        
    @pytest.mark.asyncio
    async def test_agent_specialization(self, creative_system):
        """Test that different agent roles provide specialized contributions"""
        # Create agents for different roles
        ideation_agent = CreativeAgent(
            agent_id="ideation_001",
            role=AgentRole.IDEATION,
            specialization="brainstorming",
            capabilities=["divergent_thinking", "idea_generation"]
        )
        
        design_agent = CreativeAgent(
            agent_id="design_001", 
            role=AgentRole.DESIGN,
            specialization="ui_design",
            capabilities=["visual_design", "prototyping"]
        )
        
        # Test role-specific contributions
        ideation_contribution = await creative_system.generate_agent_contribution(
            ideation_agent, "How can we make onboarding more engaging?"
        )
        
        design_contribution = await creative_system.generate_agent_contribution(
            design_agent, "Create visual concepts for onboarding flow"
        )
        
        # Verify contributions reflect agent specializations
        assert "ideas" in ideation_contribution.get("content", "").lower()
        assert "visual" in design_contribution.get("content", "").lower()
        
    @pytest.mark.asyncio
    async def test_session_phase_progression(self, creative_system, sample_problem):
        """Test progression through creative session phases"""
        problem_id = await creative_system.register_problem(sample_problem)
        session_id = await creative_system.start_creative_session(problem_id=problem_id)
        
        # Start in ideation phase
        session = await creative_system.get_session(session_id)
        assert session.current_phase == CreativePhase.IDEATION
        
        # Advance through phases
        result = await creative_system.advance_phase(session_id)
        assert result["phase"] == CreativePhase.CONCEPT_DEVELOPMENT.value
        
        result = await creative_system.advance_phase(session_id)
        assert result["phase"] == CreativePhase.PROTOTYPING.value
        
        result = await creative_system.advance_phase(session_id)
        assert result["phase"] == CreativePhase.VALIDATION.value
        
    @pytest.mark.asyncio
    async def test_collaborative_contributions(self, creative_system, sample_problem):
        """Test adding and processing collaborative contributions"""
        problem_id = await creative_system.register_problem(sample_problem)
        session_id = await creative_system.start_creative_session(problem_id=problem_id)
        
        # Add human contribution
        human_contribution = await creative_system.add_contribution(
            session_id=session_id,
            contributor="alice@example.com",
            content="What if we used gamification elements?",
            contribution_type="idea"
        )
        
        assert human_contribution["contribution_id"] is not None
        assert "ai_response" in human_contribution
        
        # Verify session progress updated
        progress = await creative_system.get_session_progress(session_id)
        assert progress["total_contributions"] >= 1
        
    @pytest.mark.asyncio
    async def test_session_insights_generation(self, creative_system, sample_problem):
        """Test generation of session insights and analytics"""
        problem_id = await creative_system.register_problem(sample_problem)
        session_id = await creative_system.start_creative_session(problem_id=problem_id)
        
        # Add several contributions
        for i in range(5):
            await creative_system.add_contribution(
                session_id=session_id,
                contributor=f"user_{i}",
                content=f"Idea number {i+1}",
                contribution_type="idea"
            )
        
        insights = await creative_system.get_session_insights(session_id)
        
        assert "idea_categories" in insights
        assert "participation_balance" in insights
        assert "creative_momentum" in insights
        assert insights["total_ideas"] >= 5


class TestAICollaborationPartner:
    """Test suite for AI Collaboration Partner with personalized human-AI pairing"""
    
    @pytest.fixture
    def ai_partner(self):
        return AICollaborationPartner()
    
    @pytest.fixture
    def sample_developer(self):
        return DeveloperProfile(
            developer_id="dev_alice",
            personality_type=DeveloperPersonality.ANALYTICAL,
            coding_style=CodingStyle.FUNCTIONAL,
            experience_level="senior",
            preferred_languages=["Python", "TypeScript"],
            interests=["AI/ML", "system_architecture"]
        )
    
    @pytest.mark.asyncio
    async def test_developer_profile_creation(self, ai_partner, sample_developer):
        """Test creating and retrieving developer profiles"""
        await ai_partner.create_or_update_profile(sample_developer)
        
        retrieved_profile = await ai_partner.get_developer_profile("dev_alice")
        
        assert retrieved_profile is not None
        assert retrieved_profile.developer_id == "dev_alice"
        assert retrieved_profile.personality_type == DeveloperPersonality.ANALYTICAL
        assert retrieved_profile.coding_style == CodingStyle.FUNCTIONAL
        assert "Python" in retrieved_profile.preferred_languages
        
    @pytest.mark.asyncio
    async def test_personalized_ai_adaptation(self, ai_partner, sample_developer):
        """Test AI personality adaptation based on developer profile"""
        await ai_partner.create_or_update_profile(sample_developer)
        
        # Test adaptation for analytical personality
        adapted_personality = await ai_partner.get_adapted_personality("dev_alice")
        
        assert "analytical" in adapted_personality["communication_style"].lower()
        assert "structured" in adapted_personality["approach"].lower()
        assert "details" in adapted_personality["focus_areas"]
        
    @pytest.mark.asyncio
    async def test_collaboration_session_startup(self, ai_partner, sample_developer):
        """Test starting AI-assisted collaboration session"""
        await ai_partner.create_or_update_profile(sample_developer)
        
        context = CollaborationContext(
            developer_id="dev_alice",
            task_description="Implementing user authentication system",
            code_context="FastAPI backend with JWT tokens",
            session_goals=["Security best practices", "Clean code"],
            timestamp=datetime.utcnow()
        )
        
        session_id = await ai_partner.start_collaboration_session(context)
        
        assert session_id is not None
        
        # Get initial suggestions
        suggestions = await ai_partner.get_initial_suggestions(session_id)
        assert len(suggestions) > 0
        assert any("security" in s.lower() for s in suggestions)
        
    @pytest.mark.asyncio
    async def test_interaction_processing(self, ai_partner, sample_developer):
        """Test processing developer interactions with AI partner"""
        await ai_partner.create_or_update_profile(sample_developer)
        
        context = CollaborationContext(
            developer_id="dev_alice",
            task_description="Code review assistance",
            session_goals=["Code quality", "Performance optimization"]
        )
        
        session_id = await ai_partner.start_collaboration_session(context)
        
        # Test different interaction types
        question_response = await ai_partner.process_interaction(
            session_id=session_id,
            interaction_type="question",
            content="How can I optimize this database query?",
            context={"language": "Python", "framework": "SQLAlchemy"}
        )
        
        assert question_response["type"] == "answer"
        assert "optimization" in question_response["content"].lower()
        assert len(question_response["suggestions"]) > 0
        
    @pytest.mark.asyncio
    async def test_learning_adaptation(self, ai_partner, sample_developer):
        """Test AI learning from developer interactions over time"""
        await ai_partner.create_or_update_profile(sample_developer)
        
        # Simulate multiple collaboration sessions
        for i in range(3):
            context = CollaborationContext(
                developer_id="dev_alice",
                task_description=f"Task {i+1}",
                session_goals=["Learning test"]
            )
            session_id = await ai_partner.start_collaboration_session(context)
            
            # Add interaction history
            await ai_partner.process_interaction(
                session_id=session_id,
                interaction_type="feedback",
                content="This approach worked well",
                context={"satisfaction": "high"}
            )
        
        # Check learning adaptation
        stats = await ai_partner.get_collaboration_stats("dev_alice")
        assert stats["total_sessions"] == 3
        assert "learning_insights" in stats
        
    @pytest.mark.asyncio
    async def test_session_insights_analytics(self, ai_partner, sample_developer):
        """Test collaboration session insights and productivity analytics"""
        await ai_partner.create_or_update_profile(sample_developer)
        
        context = CollaborationContext(
            developer_id="dev_alice",
            task_description="Performance optimization project"
        )
        session_id = await ai_partner.start_collaboration_session(context)
        
        # Simulate productive collaboration
        for i in range(5):
            await ai_partner.process_interaction(
                session_id=session_id,
                interaction_type="question",
                content=f"Question {i+1}",
                context={"productivity": "high"}
            )
        
        insights = await ai_partner.get_session_insights(session_id)
        
        assert "productivity" in insights
        assert "learning" in insights
        assert "quality" in insights
        assert "recommendations" in insights
        assert insights["productivity"]["interaction_count"] == 5


class TestCollaborativeDesignPlatform:
    """Test suite for Collaborative Design Platform with real-time co-design"""
    
    @pytest.fixture
    def design_platform(self):
        return CollaborativeDesignPlatform()
    
    @pytest.mark.asyncio
    async def test_design_session_creation(self, design_platform):
        """Test creating collaborative design sessions"""
        session_id = await design_platform.create_design_session(
            project_name="Mobile App Redesign",
            design_brief="Create modern, accessible mobile interface",
            target_audience="Young professionals",
            design_goals=["Improve usability", "Modern aesthetics"],
            constraints=["iOS and Android", "Accessibility compliance"],
            participants=["designer@company.com", "pm@company.com"]
        )
        
        assert session_id is not None
        
        session = await design_platform.get_design_session(session_id)
        assert session is not None
        assert session.project_name == "Mobile App Redesign"
        assert session.current_phase == DesignPhase.RESEARCH
        assert len(session.participants) == 2
        
    @pytest.mark.asyncio
    async def test_canvas_element_management(self, design_platform):
        """Test adding, updating, and managing design elements on canvas"""
        session_id = await design_platform.create_design_session(
            project_name="Test Design",
            design_brief="Test brief",
            target_audience="Test users",
            design_goals=["Test goal"]
        )
        
        # Add design element
        element_id = await design_platform.add_canvas_element(
            session_id=session_id,
            element_type="button",
            content={"text": "Submit", "color": "#007bff", "size": "large"},
            position={"x": 100, "y": 200}
        )
        
        assert element_id is not None
        
        # Update element
        await design_platform.update_canvas_element(
            session_id=session_id,
            element_id=element_id,
            updates={"content": {"text": "Save Changes", "color": "#28a745"}}
        )
        
        # Verify canvas state
        canvas_state = await design_platform.get_canvas_state(session_id)
        assert len(canvas_state["elements"]) == 1
        assert canvas_state["elements"][0]["content"]["text"] == "Save Changes"
        
    @pytest.mark.asyncio
    async def test_ai_design_feedback(self, design_platform):
        """Test AI-powered design feedback and suggestions"""
        session_id = await design_platform.create_design_session(
            project_name="UI Design",
            design_brief="Create user dashboard",
            target_audience="Business users",
            design_goals=["Clear information hierarchy"]
        )
        
        # Add design element
        element_id = await design_platform.add_canvas_element(
            session_id=session_id,
            element_type="text",
            content={"text": "Dashboard Title", "size": "small", "color": "#999"},
            position={"x": 50, "y": 50}
        )
        
        # Get AI feedback
        feedback = await design_platform.get_ai_element_feedback(session_id, element_id)
        
        assert feedback is not None
        assert "suggestions" in feedback
        assert len(feedback["suggestions"]) > 0
        assert "accessibility" in feedback or "usability" in feedback
        
    @pytest.mark.asyncio
    async def test_design_phase_progression(self, design_platform):
        """Test advancing through design phases with AI guidance"""
        session_id = await design_platform.create_design_session(
            project_name="Phase Test",
            design_brief="Test phase progression",
            target_audience="Test users",
            design_goals=["Test progression"]
        )
        
        # Start in research phase
        session = await design_platform.get_design_session(session_id)
        assert session.current_phase == DesignPhase.RESEARCH
        
        # Advance through phases
        result = await design_platform.advance_design_phase(session_id)
        assert result["phase"] == DesignPhase.IDEATION.value
        assert "requirements" in result
        assert "guidance" in result
        
        result = await design_platform.advance_design_phase(session_id)
        assert result["phase"] == DesignPhase.WIREFRAMING.value
        
    @pytest.mark.asyncio
    async def test_ai_design_suggestions(self, design_platform):
        """Test AI-generated design suggestions based on context"""
        session_id = await design_platform.create_design_session(
            project_name="Suggestion Test",
            design_brief="E-commerce checkout flow",
            target_audience="Online shoppers",
            design_goals=["Reduce cart abandonment"]
        )
        
        suggestions = await design_platform.get_ai_design_suggestions(
            session_id=session_id,
            suggestion_type="layout",
            focus_area="checkout_flow",
            constraints=["mobile_first", "single_page"]
        )
        
        assert len(suggestions) > 0
        assert any("checkout" in s["description"].lower() for s in suggestions)
        assert any("mobile" in s["rationale"].lower() for s in suggestions)
        
    @pytest.mark.asyncio
    async def test_design_asset_export(self, design_platform):
        """Test exporting design assets in various formats"""
        session_id = await design_platform.create_design_session(
            project_name="Export Test",
            design_brief="Test asset export",
            target_audience="Developers",
            design_goals=["Asset generation"]
        )
        
        # Add some elements
        for i in range(3):
            await design_platform.add_canvas_element(
                session_id=session_id,
                element_type="rectangle",
                content={"width": 100, "height": 50, "color": f"#00{i}{i}00"},
                position={"x": i * 120, "y": 100}
            )
        
        # Export assets
        assets = await design_platform.export_design_assets(session_id, "svg")
        
        assert assets is not None
        assert "svg_data" in assets or "assets" in assets
        
        # Test download URLs generation
        urls = await design_platform.get_asset_download_urls(session_id)
        assert len(urls) > 0


class TestInnovationAccelerationEngine:
    """Test suite for Innovation Acceleration Engine with breakthrough detection"""
    
    @pytest.fixture
    def innovation_engine(self):
        return InnovationAccelerationEngine()
    
    @pytest.fixture
    def complex_problem(self):
        return InnovativeProblem(
            id="prob_test_001",
            title="Sustainable Urban Transportation",
            description="Design transportation system for carbon-neutral urban mobility",
            domain="transportation",
            complexity=ProblemComplexity.HIGHLY_COMPLEX,
            dimensions=[],  # Will be populated during decomposition
            success_criteria=["Zero emissions", "High capacity", "Cost effective"],
            constraints=["Existing infrastructure", "Budget limitations"],
            stakeholders=["City government", "Citizens", "Transport operators"]
        )
    
    @pytest.mark.asyncio
    async def test_problem_decomposition(self, innovation_engine, complex_problem):
        """Test systematic problem decomposition into manageable dimensions"""
        decomposition = await innovation_engine.decompose_problem(complex_problem)
        
        assert "problem_id" in decomposition
        assert "complexity_analysis" in decomposition
        assert "core_dimensions" in decomposition
        assert "constraint_map" in decomposition
        assert "recommended_approaches" in decomposition
        
        # Verify dimensions were identified
        assert len(decomposition["core_dimensions"]) >= 2
        
        # Verify problem was stored with dimensions
        assert complex_problem.id in innovation_engine.problems
        stored_problem = innovation_engine.problems[complex_problem.id]
        assert len(stored_problem.dimensions) > 0
        
    @pytest.mark.asyncio
    async def test_solution_space_exploration(self, innovation_engine, complex_problem):
        """Test systematic exploration of solution space with multiple approaches"""
        # First decompose the problem
        await innovation_engine.decompose_problem(complex_problem)
        
        # Explore solution space
        solutions = await innovation_engine.explore_solution_space(complex_problem.id, depth=2)
        
        assert len(solutions) > 0
        assert len(solutions) >= 6  # Should generate solutions from 6 different approaches
        
        # Verify solution diversity
        approaches = set(sol.approach for sol in solutions)
        assert len(approaches) >= 3  # Multiple different approaches
        
        # Verify solutions have proper scores
        for solution in solutions:
            assert 0 <= solution.innovation_score <= 1
            assert 0 <= solution.feasibility_score <= 1
            assert 0 <= solution.potential_impact <= 1
        
        # Verify solutions are ranked by potential
        for i in range(len(solutions) - 1):
            current_score = (solutions[i].innovation_score * 
                           solutions[i].potential_impact * 
                           solutions[i].feasibility_score)
            next_score = (solutions[i+1].innovation_score * 
                         solutions[i+1].potential_impact * 
                         solutions[i+1].feasibility_score)
            assert current_score >= next_score
        
    @pytest.mark.asyncio
    async def test_cross_domain_inspiration(self, innovation_engine, complex_problem):
        """Test generation of insights from other domains"""
        insights = await innovation_engine.generate_cross_domain_inspiration(complex_problem)
        
        assert len(insights) > 0
        
        # Verify insights from different domains
        domains = set(insight.source_domain for insight in insights)
        assert len(domains) >= 3  # Should draw from multiple domains
        assert "transportation" not in domains  # Shouldn't include same domain
        
        # Verify insight quality
        for insight in insights:
            assert insight.applicability_score > 0.3  # Only relevant insights
            assert len(insight.principle) > 0
            assert len(insight.adaptation_notes) > 0
        
        # Verify insights are sorted by applicability
        for i in range(len(insights) - 1):
            assert insights[i].applicability_score >= insights[i+1].applicability_score
        
    @pytest.mark.asyncio
    async def test_breakthrough_detection(self, innovation_engine):
        """Test detection of breakthrough innovation potential"""
        # Create a high-potential solution
        breakthrough_solution = SolutionConcept(
            id="sol_breakthrough",
            problem_id="prob_test",
            title="Revolutionary Transport System",
            description="Paradigm-shifting approach to urban mobility",
            approach="contrarian_thinking",
            innovation_type=InnovationType.DISRUPTIVE,
            status=SolutionStatus.CONCEPT,
            feasibility_score=0.7,
            innovation_score=0.9,
            risk_score=0.6,
            potential_impact=0.95,
            development_effort=0.8,
            inspiration_domains=["biology", "physics", "technology"],
            key_insights=["Revolutionary concept", "Paradigm shift"],
            implementation_steps=["Research", "Prototype", "Test"],
            success_metrics=["Performance leap", "Market disruption"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        indicators = await innovation_engine.detect_breakthrough_potential(breakthrough_solution)
        
        assert len(indicators) > 0
        
        # Should detect multiple breakthrough indicators
        indicator_types = [ind.indicator_type for ind in indicators]
        assert "performance_leap" in indicator_types
        assert "paradigm_shift" in indicator_types
        
        # Check for overall breakthrough potential
        overall_indicators = [ind for ind in indicators if ind.indicator_type == "overall_breakthrough_potential"]
        assert len(overall_indicators) == 1
        assert overall_indicators[0].confidence > 0.7
        
    @pytest.mark.asyncio
    async def test_innovation_metrics_calculation(self, innovation_engine):
        """Test comprehensive innovation metrics calculation"""
        test_solution = SolutionConcept(
            id="sol_metrics_test",
            problem_id="prob_test",
            title="Test Solution",
            description="Solution for metrics testing",
            approach="analytical",
            innovation_type=InnovationType.INCREMENTAL,
            status=SolutionStatus.CONCEPT,
            feasibility_score=0.8,
            innovation_score=0.7,
            risk_score=0.3,
            potential_impact=0.75,
            development_effort=0.5,
            inspiration_domains=["technology"],
            key_insights=["Practical approach"],
            implementation_steps=["Design", "Build", "Test"],
            success_metrics=["User adoption"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        metrics = await innovation_engine.calculate_innovation_metrics(test_solution)
        
        assert metrics.solution_id == "sol_metrics_test"
        
        # Verify all metric components are calculated
        assert 0 <= metrics.novelty_score <= 1
        assert 0 <= metrics.usefulness_score <= 1
        assert 0 <= metrics.elegance_score <= 1
        assert 0 <= metrics.scalability_score <= 1
        assert 0 <= metrics.sustainability_score <= 1
        assert 0 <= metrics.market_potential <= 1
        assert 0 <= metrics.technical_feasibility <= 1
        assert 0 <= metrics.overall_innovation_score <= 1
        
        # Overall score should be reasonable weighted average
        expected_range = (0.5, 0.9)  # Based on input values
        assert expected_range[0] <= metrics.overall_innovation_score <= expected_range[1]
        
    @pytest.mark.asyncio
    async def test_innovation_pipeline_optimization(self, innovation_engine, complex_problem):
        """Test optimization of innovation pipeline across multiple problems"""
        # Create and decompose multiple problems
        problems = [complex_problem]
        for i in range(2):
            problem = InnovativeProblem(
                id=f"prob_test_{i+2}",
                title=f"Test Problem {i+2}",
                description=f"Description {i+2}",
                domain="test_domain",
                complexity=ProblemComplexity.MODERATE,
                dimensions=[],
                success_criteria=[f"Criteria {i+1}"],
                constraints=[f"Constraint {i+1}"],
                stakeholders=[f"Stakeholder {i+1}"]
            )
            problems.append(problem)
            await innovation_engine.decompose_problem(problem)
        
        await innovation_engine.decompose_problem(complex_problem)
        
        # Generate solutions for problems
        for problem in problems:
            await innovation_engine.explore_solution_space(problem.id, depth=1)
        
        # Optimize pipeline
        problem_ids = [p.id for p in problems]
        optimization = await innovation_engine.optimize_innovation_pipeline(problem_ids)
        
        assert "total_problems" in optimization
        assert "total_solutions" in optimization
        assert "development_sequence" in optimization
        assert "resource_allocation" in optimization
        assert "expected_outcomes" in optimization
        
        assert optimization["total_problems"] == len(problems)
        assert optimization["total_solutions"] > 0
        
        # Verify sequence optimization
        sequence = optimization["development_sequence"]
        assert len(sequence) > 0
        assert all("priority" in item for item in sequence)
        assert all("solution_id" in item for item in sequence)


class TestCreativeCollaborationAPI:
    """Test suite for Creative Collaboration API endpoints"""
    
    @pytest.fixture
    def api_client(self):
        """Mock API client for testing"""
        from fastapi.testclient import TestClient
        from src.server.main import app
        return TestClient(app)
    
    def test_create_creative_problem_endpoint(self, api_client):
        """Test API endpoint for creating creative problems"""
        problem_data = {
            "title": "API Test Problem",
            "description": "Testing problem creation via API",
            "domain": "api_testing",
            "success_criteria": ["API works", "Data persists"],
            "constraints": ["REST API", "JSON format"],
            "stakeholders": ["API users", "Developers"]
        }
        
        response = api_client.post("/api/creative/problems", json=problem_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "problem_id" in result
        assert result["status"] == "created"
        assert "next_steps" in result
        
    def test_create_developer_profile_endpoint(self, api_client):
        """Test API endpoint for creating developer profiles"""
        profile_data = {
            "developer_id": "api_test_dev",
            "personality_type": "analytical",
            "coding_style": "functional",
            "experience_level": "senior",
            "preferred_languages": ["Python", "JavaScript"],
            "interests": ["API design", "testing"]
        }
        
        response = api_client.post("/api/creative/developers/profile", json=profile_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["developer_id"] == "api_test_dev"
        assert result["status"] == "profile_created"
        
    def test_create_design_session_endpoint(self, api_client):
        """Test API endpoint for creating design sessions"""
        session_data = {
            "project_name": "API Design Test",
            "design_brief": "Test collaborative design via API",
            "target_audience": "API consumers", 
            "design_goals": ["Intuitive API", "Good documentation"],
            "constraints": ["RESTful design", "OpenAPI spec"],
            "participants": ["designer@test.com"]
        }
        
        response = api_client.post("/api/creative/design/sessions", json=session_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "session_id" in result
        assert result["project_name"] == "API Design Test"
        assert result["status"] == "created"
        
    def test_create_innovation_problem_endpoint(self, api_client):
        """Test API endpoint for innovation problems"""
        innovation_data = {
            "title": "API Innovation Challenge",
            "description": "Innovate better API design patterns",
            "domain": "software_architecture",
            "complexity": "complex",
            "success_criteria": ["Developer experience", "Performance"],
            "constraints": ["Backward compatibility", "Security"],
            "stakeholders": ["API team", "Client developers"]
        }
        
        response = api_client.post("/api/creative/innovation/problems", json=innovation_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "problem_id" in result
        assert result["status"] == "created"
        assert "dimensions_identified" in result
        
    def test_health_check_endpoint(self, api_client):
        """Test creative collaboration health check endpoint"""
        response = api_client.get("/api/creative/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert "services" in result
        assert "creative_system" in result["services"]
        assert "ai_partner" in result["services"] 
        assert "design_platform" in result["services"]
        assert "innovation_engine" in result["services"]


class TestCreativeCollaborationIntegration:
    """Integration tests for complete creative collaboration workflows"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_creative_workflow(self):
        """Test complete creative collaboration workflow from problem to solution"""
        # Initialize systems
        creative_system = CreativeAgentSystem()
        innovation_engine = InnovationAccelerationEngine()
        
        # 1. Register creative problem
        problem = CreativeProblem(
            title="Next-Gen User Interface",
            description="Design revolutionary user interface paradigm",
            domain="interface_design",
            success_criteria=["Intuitive interaction", "Accessibility", "Performance"],
            constraints=["Cross-platform", "Responsive design"],
            stakeholders=["Users", "Designers", "Developers"]
        )
        
        problem_id = await creative_system.register_problem(problem)
        
        # 2. Start collaborative session
        session_id = await creative_system.start_creative_session(
            problem_id=problem_id,
            human_participants=["designer@company.com"],
            session_type="innovation_workshop",
            duration_minutes=120
        )
        
        # 3. Add contributions and advance through phases
        for phase in range(3):  # Simulate 3 phases
            await creative_system.add_contribution(
                session_id=session_id,
                contributor="designer@company.com",
                content=f"Phase {phase+1} contribution",
                contribution_type="concept"
            )
            
            if phase < 2:  # Don't advance after last phase
                await creative_system.advance_phase(session_id)
        
        # 4. Generate innovation insights
        innovation_problem = InnovativeProblem(
            id=f"innov_{problem_id}",
            title=problem.title,
            description=problem.description,
            domain=problem.domain,
            complexity=ProblemComplexity.COMPLEX,
            dimensions=[],
            success_criteria=problem.success_criteria,
            constraints=problem.constraints,
            stakeholders=problem.stakeholders
        )
        
        await innovation_engine.decompose_problem(innovation_problem)
        solutions = await innovation_engine.explore_solution_space(innovation_problem.id, depth=2)
        
        # 5. Analyze breakthrough potential
        best_solution = max(solutions, key=lambda s: s.innovation_score)
        indicators = await innovation_engine.detect_breakthrough_potential(best_solution)
        
        # 6. Verify end-to-end workflow
        session = await creative_system.get_session(session_id)
        assert session.status == SessionStatus.ACTIVE
        assert len(solutions) > 0
        assert best_solution.innovation_score > 0.5
        assert len(indicators) > 0
        
        # Export session results
        results = await creative_system.export_session(session_id, "json")
        assert results is not None
        assert "session_summary" in results or "contributions" in results
        
    @pytest.mark.asyncio  
    async def test_multi_agent_creative_collaboration(self):
        """Test collaboration between multiple AI agents and human participants"""
        creative_system = CreativeAgentSystem()
        ai_partner = AICollaborationPartner()
        
        # Create developer profile
        developer = DeveloperProfile(
            developer_id="collab_dev",
            personality_type=DeveloperPersonality.CREATIVE,
            coding_style=CodingStyle.AGILE,
            experience_level="mid",
            preferred_languages=["TypeScript", "Python"],
            interests=["UI/UX", "creative_coding"]
        )
        await ai_partner.create_or_update_profile(developer)
        
        # Create problem requiring both creative and technical input
        problem = CreativeProblem(
            title="Interactive Data Visualization",
            description="Create engaging, interactive way to explore complex datasets",
            domain="data_visualization", 
            success_criteria=["User engagement", "Data clarity", "Performance"],
            constraints=["Web-based", "Real-time updates"],
            stakeholders=["Data analysts", "End users", "Developers"]
        )
        
        problem_id = await creative_system.register_problem(problem)
        
        # Start creative session
        creative_session_id = await creative_system.start_creative_session(
            problem_id=problem_id,
            human_participants=["analyst@company.com"],
            session_type="design_thinking"
        )
        
        # Start AI collaboration session
        collab_context = CollaborationContext(
            developer_id="collab_dev",
            task_description="Implement interactive data visualization",
            session_goals=["Creative coding", "Performance optimization"]
        )
        ai_session_id = await ai_partner.start_collaboration_session(collab_context)
        
        # Simulate multi-agent collaboration
        for i in range(3):
            # Creative contribution
            await creative_system.add_contribution(
                session_id=creative_session_id,
                contributor="ideation_agent",
                content=f"Creative idea {i+1} for visualization",
                contribution_type="idea"
            )
            
            # Technical collaboration
            await ai_partner.process_interaction(
                session_id=ai_session_id,
                interaction_type="technical_discussion",
                content=f"How to implement idea {i+1}?",
                context={"focus": "performance"}
            )
        
        # Verify collaborative outcomes
        creative_insights = await creative_system.get_session_insights(creative_session_id)
        collab_insights = await ai_partner.get_session_insights(ai_session_id)
        
        assert creative_insights["total_ideas"] >= 3
        assert collab_insights["productivity"]["interaction_count"] >= 3
        assert creative_insights["creative_momentum"] > 0


# Performance and stress tests
class TestCreativeCollaborationPerformance:
    """Performance tests for creative collaboration systems"""
    
    @pytest.mark.asyncio
    async def test_concurrent_creative_sessions(self):
        """Test handling multiple concurrent creative sessions"""
        creative_system = CreativeAgentSystem()
        
        # Create multiple problems
        problems = []
        for i in range(5):
            problem = CreativeProblem(
                title=f"Concurrent Problem {i+1}",
                description=f"Testing concurrent session {i+1}",
                domain="performance_testing",
                success_criteria=[f"Criterion {i+1}"],
                constraints=[f"Constraint {i+1}"],
                stakeholders=[f"Stakeholder {i+1}"]
            )
            problem_id = await creative_system.register_problem(problem)
            problems.append(problem_id)
        
        # Start concurrent sessions
        sessions = []
        for problem_id in problems:
            session_id = await creative_system.start_creative_session(
                problem_id=problem_id,
                human_participants=[f"user_{problem_id}@test.com"]
            )
            sessions.append(session_id)
        
        # Add contributions concurrently
        tasks = []
        for session_id in sessions:
            for i in range(3):
                task = creative_system.add_contribution(
                    session_id=session_id,
                    contributor=f"user_{session_id}",
                    content=f"Concurrent contribution {i+1}",
                    contribution_type="idea"
                )
                tasks.append(task)
        
        # Wait for all contributions to complete
        await asyncio.gather(*tasks)
        
        # Verify all sessions handled contributions properly
        for session_id in sessions:
            progress = await creative_system.get_session_progress(session_id)
            assert progress["total_contributions"] >= 3
        
    @pytest.mark.asyncio
    async def test_large_solution_space_exploration(self):
        """Test performance with large-scale solution exploration"""
        innovation_engine = InnovationAccelerationEngine()
        
        # Create complex problem
        problem = InnovativeProblem(
            id="perf_test_problem",
            title="Complex Performance Test Problem",
            description="Problem requiring extensive solution exploration",
            domain="performance_testing",
            complexity=ProblemComplexity.HIGHLY_COMPLEX,
            dimensions=[],
            success_criteria=["Scalability", "Performance", "Reliability"],
            constraints=["Resource limitations", "Time constraints"],
            stakeholders=["Performance team", "Users", "Operations"]
        )
        
        # Decompose problem
        await innovation_engine.decompose_problem(problem)
        
        # Explore with high depth
        start_time = datetime.utcnow()
        solutions = await innovation_engine.explore_solution_space(problem.id, depth=4)
        end_time = datetime.utcnow()
        
        duration = (end_time - start_time).total_seconds()
        
        # Verify performance
        assert len(solutions) > 10  # Should generate many solutions
        assert duration < 30  # Should complete within 30 seconds
        
        # Verify solution quality despite scale
        high_quality_solutions = [s for s in solutions if s.innovation_score > 0.6]
        assert len(high_quality_solutions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])