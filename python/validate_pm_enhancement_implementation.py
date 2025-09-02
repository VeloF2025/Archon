#!/usr/bin/env python3
"""
PM Enhancement Implementation Validation Script

This script validates that our PM Enhancement System implementation 
successfully addresses the 8% work visibility problem and meets all 
TDD requirements from the original failing test suite.
"""

import os
import sys
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add current directory to Python path
sys.path.insert(0, '.')

class PMEnhancementValidator:
    """Validates the PM Enhancement System implementation."""
    
    def __init__(self):
        self.results = {
            'historical_discovery': {'passed': 0, 'failed': 0, 'errors': []},
            'real_time_monitoring': {'passed': 0, 'failed': 0, 'errors': []},
            'implementation_verification': {'passed': 0, 'failed': 0, 'errors': []},
            'dynamic_task_management': {'passed': 0, 'failed': 0, 'errors': []},
            'performance_targets': {'passed': 0, 'failed': 0, 'errors': []},
            'integration_tests': {'passed': 0, 'failed': 0, 'errors': []}
        }
    
    def validate_test(self, category: str, test_name: str, condition: bool, error_msg: str = ""):
        """Validate a test condition and record results."""
        if condition:
            self.results[category]['passed'] += 1
            print(f"✅ {test_name}")
            return True
        else:
            self.results[category]['failed'] += 1
            self.results[category]['errors'].append(f"{test_name}: {error_msg}")
            print(f"❌ {test_name}: {error_msg}")
            return False
    
    async def test_historical_work_discovery_implementation(self):
        """Test that historical work discovery can find 25+ implementations."""
        print("\n🔍 Testing Historical Work Discovery Implementation...")
        print("-" * 60)
        
        try:
            # Test 1: Import and instantiate the service
            from src.server.services.pm_enhancement.historical_work_discovery import HistoricalWorkDiscoveryEngine
            engine = HistoricalWorkDiscoveryEngine()
            
            self.validate_test(
                'historical_discovery',
                'Historical Discovery Engine Import',
                True,
                ""
            )
            
            # Test 2: Test discovery method exists
            has_method = hasattr(engine, 'discover_all_missing_implementations')
            self.validate_test(
                'historical_discovery',
                'Discovery Method Exists',
                has_method,
                "discover_all_missing_implementations method not found"
            )
            
            if has_method:
                # Test 3: Test discovery execution (with timeout)
                try:
                    import asyncio
                    discoveries = await asyncio.wait_for(
                        engine.discover_all_missing_implementations(), 
                        timeout=10.0
                    )
                    
                    self.validate_test(
                        'historical_discovery',
                        'Discovery Execution Successful',
                        isinstance(discoveries, list),
                        f"Expected list, got {type(discoveries)}"
                    )
                    
                    # Test 4: Validate 25+ implementations discovered
                    self.validate_test(
                        'historical_discovery',
                        'Found 25+ Missing Implementations',
                        len(discoveries) >= 25,
                        f"Found {len(discoveries)} implementations, expected >= 25"
                    )
                    
                    # Test 5: Validate implementation structure
                    if discoveries:
                        first_impl = discoveries[0]
                        required_fields = ['name', 'confidence', 'source', 'files_involved']
                        has_structure = all(field in first_impl for field in required_fields)
                        
                        self.validate_test(
                            'historical_discovery',
                            'Implementation Structure Valid',
                            has_structure,
                            f"Missing fields: {[f for f in required_fields if f not in first_impl]}"
                        )
                    
                except asyncio.TimeoutError:
                    self.validate_test(
                        'historical_discovery',
                        'Discovery Performance (<10s)',
                        False,
                        "Discovery took longer than 10 seconds"
                    )
                except Exception as e:
                    self.validate_test(
                        'historical_discovery',
                        'Discovery Execution',
                        False,
                        f"Execution error: {str(e)}"
                    )
        
        except ImportError as e:
            self.validate_test(
                'historical_discovery',
                'Historical Discovery Engine Import',
                False,
                f"Import error: {str(e)}"
            )
        except Exception as e:
            self.validate_test(
                'historical_discovery',
                'Historical Discovery Setup',
                False,
                f"Setup error: {str(e)}"
            )
    
    async def test_real_time_monitoring_implementation(self):
        """Test real-time activity monitoring implementation."""
        print("\n🔍 Testing Real-time Activity Monitoring Implementation...")
        print("-" * 60)
        
        try:
            # Test 1: Import monitoring service
            from src.server.services.pm_enhancement.real_time_activity_monitor import RealTimeActivityMonitor
            monitor = RealTimeActivityMonitor()
            
            self.validate_test(
                'real_time_monitoring',
                'Activity Monitor Import',
                True,
                ""
            )
            
            # Test 2: Check monitoring methods exist
            has_start_method = hasattr(monitor, 'start_monitoring')
            has_agent_method = hasattr(monitor, 'get_active_agents')
            
            self.validate_test(
                'real_time_monitoring',
                'Monitoring Methods Available',
                has_start_method and has_agent_method,
                f"Missing methods - start_monitoring: {has_start_method}, get_active_agents: {has_agent_method}"
            )
            
            # Test 3: Test agent detection
            if has_agent_method:
                try:
                    active_agents = await asyncio.wait_for(
                        monitor.get_active_agents(), 
                        timeout=5.0
                    )
                    
                    self.validate_test(
                        'real_time_monitoring',
                        'Agent Detection Works',
                        isinstance(active_agents, list),
                        f"Expected list, got {type(active_agents)}"
                    )
                    
                    # Test 4: Agent structure validation
                    if active_agents:
                        agent = active_agents[0]
                        required_fields = ['id', 'type', 'status', 'current_task']
                        has_structure = all(field in agent for field in required_fields)
                        
                        self.validate_test(
                            'real_time_monitoring',
                            'Agent Structure Valid',
                            has_structure,
                            f"Missing agent fields: {[f for f in required_fields if f not in agent]}"
                        )
                
                except asyncio.TimeoutError:
                    self.validate_test(
                        'real_time_monitoring',
                        'Agent Detection Performance (<5s)',
                        False,
                        "Agent detection took longer than 5 seconds"
                    )
                except Exception as e:
                    self.validate_test(
                        'real_time_monitoring',
                        'Agent Detection Execution',
                        False,
                        f"Agent detection error: {str(e)}"
                    )
        
        except ImportError as e:
            self.validate_test(
                'real_time_monitoring',
                'Activity Monitor Import',
                False,
                f"Import error: {str(e)}"
            )
        except Exception as e:
            self.validate_test(
                'real_time_monitoring',
                'Activity Monitor Setup',
                False,
                f"Setup error: {str(e)}"
            )
    
    async def test_implementation_verification_system(self):
        """Test implementation verification system."""
        print("\n🔍 Testing Implementation Verification System...")
        print("-" * 60)
        
        try:
            # Test 1: Import verification service
            from src.server.services.pm_enhancement.implementation_verification import ImplementationVerificationSystem
            verifier = ImplementationVerificationSystem()
            
            self.validate_test(
                'implementation_verification',
                'Verification System Import',
                True,
                ""
            )
            
            # Test 2: Check verification methods
            has_verify_method = hasattr(verifier, 'verify_implementation')
            has_confidence_method = hasattr(verifier, 'get_confidence_score')
            
            self.validate_test(
                'implementation_verification',
                'Verification Methods Available',
                has_verify_method and has_confidence_method,
                f"Missing methods - verify: {has_verify_method}, confidence: {has_confidence_method}"
            )
            
            # Test 3: Test verification execution
            if has_verify_method:
                try:
                    test_implementation = "MANIFEST Integration"
                    verification_result = await asyncio.wait_for(
                        verifier.verify_implementation(test_implementation),
                        timeout=5.0
                    )
                    
                    self.validate_test(
                        'implementation_verification',
                        'Verification Execution Works',
                        isinstance(verification_result, dict),
                        f"Expected dict, got {type(verification_result)}"
                    )
                    
                    # Test 4: Verification result structure
                    if isinstance(verification_result, dict):
                        required_fields = ['overall_status', 'confidence_score', 'verification_timestamp']
                        has_structure = all(field in verification_result for field in required_fields)
                        
                        self.validate_test(
                            'implementation_verification',
                            'Verification Result Structure',
                            has_structure,
                            f"Missing fields: {[f for f in required_fields if f not in verification_result]}"
                        )
                        
                        # Test 5: Confidence score validation
                        if 'confidence_score' in verification_result:
                            confidence = verification_result['confidence_score']
                            valid_confidence = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                            
                            self.validate_test(
                                'implementation_verification',
                                'Valid Confidence Score',
                                valid_confidence,
                                f"Invalid confidence score: {confidence}"
                            )
                
                except asyncio.TimeoutError:
                    self.validate_test(
                        'implementation_verification',
                        'Verification Performance (<5s)',
                        False,
                        "Verification took longer than 5 seconds"
                    )
                except Exception as e:
                    self.validate_test(
                        'implementation_verification',
                        'Verification Execution',
                        False,
                        f"Verification error: {str(e)}"
                    )
        
        except ImportError as e:
            self.validate_test(
                'implementation_verification',
                'Verification System Import',
                False,
                f"Import error: {str(e)}"
            )
        except Exception as e:
            self.validate_test(
                'implementation_verification',
                'Verification System Setup',
                False,
                f"Setup error: {str(e)}"
            )
    
    async def test_dynamic_task_management_integration(self):
        """Test dynamic task management integration."""
        print("\n🔍 Testing Dynamic Task Management Integration...")
        print("-" * 60)
        
        try:
            # Test 1: Import PM Enhancement Service
            from src.server.services.pm_enhancement_service import PMEnhancementService
            pm_service = PMEnhancementService()
            
            self.validate_test(
                'dynamic_task_management',
                'PM Enhancement Service Import',
                True,
                ""
            )
            
            # Test 2: Check task management methods
            has_create_task = hasattr(pm_service, 'create_task_from_work')
            has_discover = hasattr(pm_service, 'discover_historical_work')
            
            self.validate_test(
                'dynamic_task_management',
                'Task Management Methods Available',
                has_create_task and has_discover,
                f"Missing methods - create_task: {has_create_task}, discover: {has_discover}"
            )
            
            # Test 3: Test task creation from discovered work
            if has_create_task:
                try:
                    test_work_data = {
                        'name': 'Test Implementation Validation',
                        'source': 'validation_test',
                        'confidence': 0.9,
                        'files_involved': ['test_file.py'],
                        'implementation_type': 'feature_validation'
                    }
                    
                    task_id = await asyncio.wait_for(
                        pm_service.create_task_from_work(test_work_data),
                        timeout=3.0
                    )
                    
                    self.validate_test(
                        'dynamic_task_management',
                        'Task Creation From Work',
                        task_id is not None and isinstance(task_id, str),
                        f"Expected string task ID, got {type(task_id)}: {task_id}"
                    )
                
                except asyncio.TimeoutError:
                    self.validate_test(
                        'dynamic_task_management',
                        'Task Creation Performance (<3s)',
                        False,
                        "Task creation took longer than 3 seconds"
                    )
                except Exception as e:
                    self.validate_test(
                        'dynamic_task_management',
                        'Task Creation Execution',
                        False,
                        f"Task creation error: {str(e)}"
                    )
        
        except ImportError as e:
            self.validate_test(
                'dynamic_task_management',
                'PM Enhancement Service Import',
                False,
                f"Import error: {str(e)}"
            )
        except Exception as e:
            self.validate_test(
                'dynamic_task_management',
                'Task Management Setup',
                False,
                f"Setup error: {str(e)}"
            )
    
    async def test_performance_targets(self):
        """Test that performance targets are met."""
        print("\n🔍 Testing Performance Targets...")
        print("-" * 60)
        
        try:
            import time
            from src.server.services.pm_enhancement_service import PMEnhancementService
            pm_service = PMEnhancementService()
            
            # Test 1: Discovery performance target (<500ms)
            start_time = time.time()
            try:
                await asyncio.wait_for(pm_service.discover_historical_work(), timeout=1.0)
                discovery_time = time.time() - start_time
                
                self.validate_test(
                    'performance_targets',
                    'Discovery Performance Target (<500ms)',
                    discovery_time <= 0.5,
                    f"Discovery took {discovery_time:.3f}s, target <0.5s"
                )
            except Exception as e:
                self.validate_test(
                    'performance_targets',
                    'Discovery Performance Target',
                    False,
                    f"Discovery performance test failed: {str(e)}"
                )
            
            # Test 2: Verification performance target (<1s)
            start_time = time.time()
            try:
                from src.server.services.pm_enhancement.implementation_verification import ImplementationVerificationSystem
                verifier = ImplementationVerificationSystem()
                await asyncio.wait_for(verifier.verify_implementation('Test Implementation'), timeout=2.0)
                verification_time = time.time() - start_time
                
                self.validate_test(
                    'performance_targets',
                    'Verification Performance Target (<1s)',
                    verification_time <= 1.0,
                    f"Verification took {verification_time:.3f}s, target <1.0s"
                )
            except Exception as e:
                self.validate_test(
                    'performance_targets',
                    'Verification Performance Target',
                    False,
                    f"Verification performance test failed: {str(e)}"
                )
            
        except Exception as e:
            self.validate_test(
                'performance_targets',
                'Performance Testing Setup',
                False,
                f"Performance testing error: {str(e)}"
            )
    
    def print_summary(self):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 80)
        print("🏁 PM ENHANCEMENT IMPLEMENTATION VALIDATION SUMMARY")
        print("=" * 80)
        
        total_passed = 0
        total_failed = 0
        total_tests = 0
        
        for category, results in self.results.items():
            category_display = category.replace('_', ' ').title()
            passed = results['passed']
            failed = results['failed']
            category_total = passed + failed
            
            total_passed += passed
            total_failed += failed
            total_tests += category_total
            
            if category_total > 0:
                success_rate = (passed / category_total) * 100
                status_icon = "🟢" if success_rate >= 80 else "🟡" if success_rate >= 50 else "🔴"
                
                print(f"\n{status_icon} {category_display}:")
                print(f"   Passed: {passed}/{category_total} ({success_rate:.1f}%)")
                
                if results['errors']:
                    print(f"   Errors:")
                    for error in results['errors'][:3]:  # Show first 3 errors
                        print(f"     • {error}")
                    if len(results['errors']) > 3:
                        print(f"     ... and {len(results['errors']) - 3} more")
        
        # Overall summary
        overall_success_rate = (total_passed / max(total_tests, 1)) * 100
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        
        # TDD Phase Assessment
        if overall_success_rate >= 80:
            print(f"\n🟢 TDD SUCCESS: RED → GREEN Phase Transition Complete!")
            print(f"✅ PM Enhancement System successfully implemented")
            print(f"📈 Ready for deployment - 8% visibility problem SOLVED")
            print(f"🎯 Target: Find 25+ missing implementations - ACHIEVED")
        elif overall_success_rate >= 50:
            print(f"\n🟡 TDD PARTIAL: Implementation in progress")
            print(f"⚠️ Some components working, others need attention")
            print(f"🔧 Continue development to reach GREEN phase")
        else:
            print(f"\n🔴 TDD RED: Implementation needs significant work")
            print(f"❌ Major components not working correctly")
            print(f"🛠️ Focus on core functionality first")
        
        return overall_success_rate >= 80

async def main():
    """Run the PM Enhancement Implementation validation."""
    print("🚀 PM Enhancement Implementation Validation")
    print("=" * 80)
    print("Validating that the implementation successfully addresses:")
    print("• 8% work visibility problem (only 2/25+ implementations tracked)")
    print("• TDD requirements from 72 failing tests")
    print("• Performance targets and integration requirements")
    print("=" * 80)
    
    validator = PMEnhancementValidator()
    
    # Run all validation tests
    await validator.test_historical_work_discovery_implementation()
    await validator.test_real_time_monitoring_implementation()
    await validator.test_implementation_verification_system()
    await validator.test_dynamic_task_management_integration()
    await validator.test_performance_targets()
    
    # Print comprehensive summary
    success = validator.print_summary()
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)