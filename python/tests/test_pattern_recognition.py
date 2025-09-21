"""
Comprehensive tests for Pattern Recognition Engine
Tests pattern detection, storage, analysis, and recommendations
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.agents.pattern_recognition.pattern_detector import PatternDetector, CodePattern
from src.agents.pattern_recognition.pattern_storage import PatternStorage
from src.agents.pattern_recognition.pattern_analyzer import PatternAnalyzer
from src.agents.pattern_recognition.pattern_recommender import PatternRecommender


class TestPatternDetector:
    """Test the pattern detection functionality"""
    
    @pytest.fixture
    def detector(self):
        """Create a pattern detector instance"""
        return PatternDetector()
    
    @pytest.mark.asyncio
    async def test_detect_singleton_pattern(self, detector):
        """Test detection of Singleton pattern"""
        code = '''
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self):
        return "Connected to database"
'''
        patterns = await detector.detect_patterns(code, "python")
        
        # Should detect Singleton pattern
        assert len(patterns) > 0
        singleton_patterns = [p for p in patterns if p.name == "Singleton"]
        assert len(singleton_patterns) == 1
        
        pattern = singleton_patterns[0]
        assert pattern.category == "Creational"
        assert pattern.confidence >= 0.8
        assert not pattern.is_antipattern
    
    @pytest.mark.asyncio
    async def test_detect_factory_pattern(self, detector):
        """Test detection of Factory pattern"""
        code = '''
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")
'''
        patterns = await detector.detect_patterns(code, "python")
        
        factory_patterns = [p for p in patterns if p.name == "Factory"]
        assert len(factory_patterns) == 1
        assert factory_patterns[0].category == "Creational"
    
    @pytest.mark.asyncio
    async def test_detect_god_class_antipattern(self, detector):
        """Test detection of God Class anti-pattern"""
        # Create a large class with many methods
        code = f'''
class GodClass:
    def __init__(self):
        self.data = []
    
    {''.join([f"    def method_{i}(self): pass\n" for i in range(25)])}
'''
        patterns = await detector.detect_patterns(code, "python")
        
        god_class_patterns = [p for p in patterns if p.name == "God Class"]
        assert len(god_class_patterns) == 1
        assert god_class_patterns[0].is_antipattern
        assert god_class_patterns[0].severity == "high"
    
    @pytest.mark.asyncio
    async def test_detect_long_method_antipattern(self, detector):
        """Test detection of Long Method anti-pattern"""
        code = f'''
def long_method():
    {"    x = 1\n" * 60}
    return x
'''
        patterns = await detector.detect_patterns(code, "python")
        
        long_method_patterns = [p for p in patterns if p.name == "Long Method"]
        assert len(long_method_patterns) == 1
        assert long_method_patterns[0].is_antipattern
    
    @pytest.mark.asyncio
    async def test_javascript_pattern_detection(self, detector):
        """Test pattern detection in JavaScript code"""
        code = '''
class Singleton {
    constructor() {
        if (Singleton.instance) {
            return Singleton.instance;
        }
        Singleton.instance = this;
    }
}
'''
        patterns = await detector.detect_patterns(code, "javascript")
        
        # Should detect patterns in JavaScript
        assert len(patterns) > 0
        assert any(p.language == "javascript" for p in patterns)
    
    @pytest.mark.asyncio
    async def test_empty_code_handling(self, detector):
        """Test handling of empty code"""
        patterns = await detector.detect_patterns("", "python")
        assert patterns == []
    
    @pytest.mark.asyncio
    async def test_invalid_syntax_handling(self, detector):
        """Test handling of invalid syntax"""
        code = "def broken_function(:\n    pass"
        patterns = await detector.detect_patterns(code, "python")
        # Should handle gracefully without crashing
        assert isinstance(patterns, list)


class TestPatternStorage:
    """Test pattern storage functionality"""
    
    @pytest.fixture
    async def storage(self):
        """Create a pattern storage instance with mocked Supabase"""
        with patch('src.agents.pattern_recognition.pattern_storage.get_supabase_client'):
            storage = PatternStorage()
            storage.supabase = MagicMock()
            
            # Mock embedding function
            with patch('src.agents.pattern_recognition.pattern_storage.create_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 1536
                yield storage
    
    @pytest.mark.asyncio
    async def test_store_pattern(self, storage):
        """Test storing a pattern"""
        pattern = CodePattern(
            id="test-pattern-1",
            name="TestPattern",
            category="Behavioral",
            description="Test pattern",
            code_example="def test(): pass",
            language="python",
            confidence=0.9,
            effectiveness_score=0.85,
            is_antipattern=False,
            metadata={"test": True}
        )
        
        # Mock the insert response
        storage.supabase.table().insert().execute.return_value.data = [pattern.dict()]
        
        result = await storage.store_pattern(pattern)
        assert result is True
        
        # Verify Supabase was called
        storage.supabase.table.assert_called_with("code_patterns")
    
    @pytest.mark.asyncio
    async def test_search_patterns_by_similarity(self, storage):
        """Test searching patterns by similarity"""
        # Mock the RPC response
        mock_patterns = [
            {
                "id": "pattern-1",
                "name": "Singleton",
                "category": "Creational",
                "similarity": 0.95
            }
        ]
        storage.supabase.rpc().execute.return_value.data = mock_patterns
        
        results = await storage.search_patterns("singleton pattern", limit=5)
        
        assert len(results) > 0
        storage.supabase.rpc.assert_called_with(
            "search_patterns_by_embedding",
            {"query_embedding": [0.1] * 1536, "match_count": 5}
        )
    
    @pytest.mark.asyncio
    async def test_get_pattern_by_id(self, storage):
        """Test retrieving a pattern by ID"""
        mock_pattern = {
            "id": "test-id",
            "name": "TestPattern",
            "category": "Structural"
        }
        storage.supabase.table().select().eq().execute.return_value.data = [mock_pattern]
        
        pattern = await storage.get_pattern("test-id")
        
        assert pattern is not None
        assert pattern.id == "test-id"
        assert pattern.name == "TestPattern"
    
    @pytest.mark.asyncio
    async def test_update_pattern_effectiveness(self, storage):
        """Test updating pattern effectiveness score"""
        storage.supabase.table().update().eq().execute.return_value.data = [{"id": "test-id"}]
        
        result = await storage.update_pattern_effectiveness("test-id", 0.92)
        
        assert result is True
        storage.supabase.table().update.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_patterns_by_language(self, storage):
        """Test getting patterns by language"""
        mock_patterns = [
            {"id": "p1", "name": "Pattern1", "language": "python"},
            {"id": "p2", "name": "Pattern2", "language": "python"}
        ]
        storage.supabase.table().select().eq().execute.return_value.data = mock_patterns
        
        patterns = await storage.get_patterns_by_language("python", limit=10)
        
        assert len(patterns) == 2
        assert all(p.language == "python" for p in patterns)


class TestPatternAnalyzer:
    """Test pattern analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a pattern analyzer instance"""
        storage = MagicMock()
        return PatternAnalyzer(storage)
    
    @pytest.mark.asyncio
    async def test_analyze_pattern_effectiveness(self, analyzer):
        """Test analyzing pattern effectiveness"""
        patterns = [
            CodePattern(
                id=f"p{i}",
                name=f"Pattern{i}",
                category="Behavioral",
                effectiveness_score=0.5 + i * 0.1,
                usage_count=10 * i
            ) for i in range(5)
        ]
        
        # Mock storage methods
        analyzer.pattern_storage.get_all_patterns = AsyncMock(return_value=patterns)
        
        analysis = await analyzer.analyze_pattern_effectiveness()
        
        assert "most_effective" in analysis
        assert "least_effective" in analysis
        assert "average_effectiveness" in analysis
        assert analysis["total_patterns"] == 5
    
    @pytest.mark.asyncio
    async def test_detect_pattern_conflicts(self, analyzer):
        """Test detecting conflicting patterns"""
        patterns = [
            CodePattern(id="p1", name="Singleton", category="Creational"),
            CodePattern(id="p2", name="Multiton", category="Creational")
        ]
        
        conflicts = await analyzer.detect_pattern_conflicts(patterns)
        
        # Singleton and Multiton should conflict
        assert len(conflicts) > 0
        assert any("Singleton" in str(c) and "Multiton" in str(c) for c in conflicts)
    
    @pytest.mark.asyncio
    async def test_calculate_code_quality_score(self, analyzer):
        """Test calculating code quality score"""
        patterns = [
            CodePattern(
                id="p1",
                name="Factory",
                is_antipattern=False,
                effectiveness_score=0.8,
                confidence=0.9
            ),
            CodePattern(
                id="p2",
                name="God Class",
                is_antipattern=True,
                severity="high",
                confidence=0.85
            )
        ]
        
        score = await analyzer.calculate_code_quality_score(patterns)
        
        assert 0 <= score <= 100
        # Should be lower due to anti-pattern
        assert score < 70
    
    @pytest.mark.asyncio
    async def test_generate_pattern_insights(self, analyzer):
        """Test generating pattern insights"""
        patterns = [
            CodePattern(id=f"p{i}", name=f"Pattern{i}", category="Behavioral")
            for i in range(10)
        ]
        
        analyzer.pattern_storage.get_all_patterns = AsyncMock(return_value=patterns)
        
        insights = await analyzer.generate_pattern_insights(patterns)
        
        assert "summary" in insights
        assert "recommendations" in insights
        assert "statistics" in insights


class TestPatternRecommender:
    """Test pattern recommendation functionality"""
    
    @pytest.fixture
    def recommender(self):
        """Create a pattern recommender instance"""
        storage = MagicMock()
        analyzer = MagicMock()
        return PatternRecommender(storage, analyzer)
    
    @pytest.mark.asyncio
    async def test_recommend_patterns_for_context(self, recommender):
        """Test recommending patterns for code context"""
        context = {
            "intent": "create_single_instance",
            "existing_patterns": [],
            "language": "python"
        }
        
        # Mock storage search
        mock_patterns = [
            CodePattern(id="p1", name="Singleton", effectiveness_score=0.9),
            CodePattern(id="p2", name="Factory", effectiveness_score=0.8)
        ]
        recommender.pattern_storage.search_patterns = AsyncMock(return_value=mock_patterns)
        
        recommendations = await recommender.recommend_patterns(context)
        
        assert len(recommendations) > 0
        assert recommendations[0].pattern.name == "Singleton"  # Highest score
    
    @pytest.mark.asyncio
    async def test_suggest_refactoring(self, recommender):
        """Test suggesting refactoring for anti-patterns"""
        antipatterns = [
            CodePattern(
                id="ap1",
                name="God Class",
                is_antipattern=True,
                severity="high"
            )
        ]
        
        refactorings = await recommender.suggest_refactoring(antipatterns)
        
        assert len(refactorings) > 0
        assert any("God Class" in r.description for r in refactorings)
        assert any(r.priority == "high" for r in refactorings)
    
    @pytest.mark.asyncio
    async def test_get_pattern_alternatives(self, recommender):
        """Test getting alternative patterns"""
        pattern = CodePattern(
            id="p1",
            name="Singleton",
            category="Creational"
        )
        
        mock_alternatives = [
            CodePattern(id="p2", name="Dependency Injection"),
            CodePattern(id="p3", name="Factory")
        ]
        recommender.pattern_storage.get_patterns_by_category = AsyncMock(
            return_value=mock_alternatives
        )
        
        alternatives = await recommender.get_pattern_alternatives(pattern)
        
        assert len(alternatives) > 0
        assert all(a.id != pattern.id for a in alternatives)


@pytest.mark.integration
class TestPatternRecognitionIntegration:
    """Integration tests for the complete pattern recognition system"""
    
    @pytest.mark.asyncio
    async def test_full_pattern_pipeline(self):
        """Test the complete pattern recognition pipeline"""
        # This would test the full flow from detection to recommendation
        detector = PatternDetector()
        
        code = '''
class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def log(self, message):
        print(f"[LOG] {message}")
'''
        
        # Detect patterns
        patterns = await detector.detect_patterns(code, "python")
        assert len(patterns) > 0
        
        # Store patterns (would need real storage in integration test)
        # Analyze patterns
        # Generate recommendations
        
        # Verify the complete flow works
        assert any(p.name == "Singleton" for p in patterns)