#!/usr/bin/env python3
"""
Test script to validate confidence propagation async/sync behavior
"""

import sys
import tempfile
import shutil
from pathlib import Path
import logging
import inspect

# Add src to path
sys.path.append('src')

from agents.graphiti.graphiti_service import GraphitiService, GraphEntity, GraphRelationship, EntityType, RelationshipType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_confidence_propagation_sync_behavior():
    """Test that propagate_confidence method is synchronous and returns float"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize service
        service = GraphitiService(db_path=temp_dir)
        logger.info("[OK] Service initialized successfully")
        
        # Check method properties
        method = service.propagate_confidence
        logger.info(f"Method type: {type(method)}")
        logger.info(f"Is coroutine function: {inspect.iscoroutinefunction(method)}")
        logger.info(f"Method signature: {inspect.signature(method)}")
        
        # Create test entities
        source_entity = GraphEntity(
            entity_id="test_source",
            entity_type=EntityType.FUNCTION,
            name="source_function",
            attributes={},
            confidence_score=0.9,
            creation_time=1234567890,
            modification_time=1234567890,
            access_frequency=1
        )
        
        target_entity = GraphEntity(
            entity_id="test_target", 
            entity_type=EntityType.FUNCTION,
            name="target_function",
            attributes={},
            confidence_score=0.5,
            creation_time=1234567890,
            modification_time=1234567890,
            access_frequency=1
        )
        
        relationship = GraphRelationship(
            relationship_id="test_rel",
            source_id="test_source",
            target_id="test_target",
            relationship_type=RelationshipType.CALLS,
            confidence=0.8,
            creation_time=1234567890,
            modification_time=1234567890,
            access_frequency=1
        )
        
        # Test the method call
        logger.info("[TEST] Testing propagate_confidence method...")
        result = service.propagate_confidence(source_entity, target_entity, relationship)
        
        # Validate result
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result value: {result}")
        
        # Check that result is a float
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert 0.0 <= result <= 1.0, f"Confidence should be between 0.0 and 1.0, got {result}"
        
        # Check that it's not a coroutine
        import types
        assert not isinstance(result, types.CoroutineType), "Result should not be a coroutine"
        
        logger.info("[OK] All validation checks passed!")
        logger.info(f"[OK] Confidence propagated from {source_entity.confidence_score:.3f} to {result:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if hasattr(service, 'close'):
                service.close()
        except:
            pass
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        logger.info("[CLEANUP] Cleanup completed")

def test_method_not_awaitable():
    """Test that the method doesn't need to be awaited"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        service = GraphitiService(db_path=temp_dir)
        
        # Create minimal test data
        source = GraphEntity("src", EntityType.FUNCTION, "src")
        target = GraphEntity("tgt", EntityType.FUNCTION, "tgt") 
        rel = GraphRelationship("rel", "src", "tgt", RelationshipType.CALLS)
        
        # This should work (sync call)
        try:
            result = service.propagate_confidence(source, target, rel)
            logger.info("[OK] Synchronous call works correctly")
        except Exception as e:
            logger.error(f"[FAIL] Synchronous call failed: {e}")
            return False
        
        # This should fail (trying to await a non-async function)
        try:
            import asyncio
            
            async def test_await():
                # This would fail if the method was sync but being awaited
                result = await service.propagate_confidence(source, target, rel)
                return result
            
            # If this doesn't raise an error, the method might be incorrectly async
            result = asyncio.run(test_await())
            logger.error("[FAIL] Method incorrectly accepts await - this suggests async/sync mismatch")
            return False
            
        except TypeError as e:
            if "object is not awaitable" in str(e):
                logger.info("[OK] Method correctly rejects await (as expected for sync method)")
                return True
            else:
                logger.error(f"[FAIL] Unexpected TypeError: {e}")
                return False
                
    except Exception as e:
        logger.error(f"[FAIL] Test setup failed: {e}")
        return False
        
    finally:
        try:
            if hasattr(service, 'close'):
                service.close()
        except:
            pass
        shutil.rmtree(str(temp_dir), ignore_errors=True)

if __name__ == "__main__":
    print("[TESTING] Confidence Propagation Async/Sync Behavior")
    print("=" * 60)
    
    success1 = test_confidence_propagation_sync_behavior()
    success2 = test_method_not_awaitable()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("[SUCCESS] ALL TESTS PASSED - No async/sync issues found!")
        print("[OK] propagate_confidence method is correctly synchronous")
        print("[OK] Method returns float values as expected")
        print("[OK] No coroutine vs float comparison issues detected")
    else:
        print("[FAIL] TESTS FAILED - Async/sync issues may exist")
        sys.exit(1)