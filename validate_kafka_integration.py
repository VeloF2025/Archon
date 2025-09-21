#!/usr/bin/env python3
"""
Kafka Integration Validation Script
Validates that all Kafka integration components are properly implemented and accessible
"""

import sys
import os
import importlib
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_file_exists(file_path: str) -> bool:
    """Validate that a file exists"""
    path = Path(file_path)
    exists = path.exists()
    status = "✅" if exists else "❌"
    logger.info(f"{status} {file_path} - {'EXISTS' if exists else 'MISSING'}")
    return exists

def validate_import(module_path: str, class_or_function: str = None) -> bool:
    """Validate that a module can be imported"""
    try:
        module = importlib.import_module(module_path)
        if class_or_function:
            if hasattr(module, class_or_function):
                logger.info(f"✅ {module_path}.{class_or_function} - IMPORTABLE")
                return True
            else:
                logger.error(f"❌ {module_path}.{class_or_function} - MISSING")
                return False
        else:
            logger.info(f"✅ {module_path} - IMPORTABLE")
            return True
    except ImportError as e:
        logger.error(f"❌ {module_path} - IMPORT ERROR: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ {module_path} - ERROR: {e}")
        return False

def validate_kafka_integration():
    """Validate all Kafka integration components"""
    logger.info("🔍 VALIDATING KAFKA INTEGRATION IMPLEMENTATION")
    logger.info("=" * 60)
    
    validation_results = {}
    
    # 1. Validate core files exist
    logger.info("📁 VALIDATING FILE EXISTENCE")
    logger.info("-" * 40)
    
    files_to_check = [
        "python/src/agents/messaging/distributed_messaging_system.py",
        "python/src/server/services/kafka_integration_service.py",
        "python/src/server/services/real_time_streaming_service.py",
        "python/src/server/api_routes/kafka_api.py",
        "python/src/server/api_routes/streaming_api.py",
        "python/src/agents/analytics/streaming_analytics.py",
        "python/src/agents/base_agent.py"
    ]
    
    file_validation = []
    for file_path in files_to_check:
        file_validation.append(validate_file_exists(file_path))
    
    validation_results["files_exist"] = all(file_validation)
    
    # 2. Validate imports
    logger.info("\n🔧 VALIDATING MODULE IMPORTS")
    logger.info("-" * 40)
    
    import_checks = [
        ("python.src.agents.messaging.distributed_messaging_system", "DistributedMessagingSystem"),
        ("python.src.agents.messaging.distributed_messaging_system", "KafkaMessagingBackend"),
        ("python.src.server.services.kafka_integration_service", "KafkaIntegrationService"),
        ("python.src.server.services.kafka_integration_service", "get_kafka_service"),
        ("python.src.server.services.real_time_streaming_service", "RealTimeStreamingService"),
        ("python.src.server.services.real_time_streaming_service", "get_streaming_service"),
        ("python.src.agents.analytics.streaming_analytics", "StreamingAnalytics"),
        ("python.src.agents.analytics.streaming_analytics", "StreamType"),
        ("python.src.server.api_routes.kafka_api", "router"),
        ("python.src.server.api_routes.streaming_api", "router")
    ]
    
    import_validation = []
    for module_path, class_name in import_checks:
        import_validation.append(validate_import(module_path, class_name))
    
    validation_results["imports_work"] = all(import_validation)
    
    # 3. Validate base agent Kafka integration
    logger.info("\n🤖 VALIDATING BASE AGENT KAFKA INTEGRATION")
    logger.info("-" * 40)
    
    base_agent_methods = [
        "publish_agent_event",
        "send_agent_command", 
        "publish_analytics",
        "register_command_handler",
        "get_kafka_status"
    ]
    
    base_agent_validation = []
    try:
        from python.src.agents.base_agent import BaseAgent
        
        for method_name in base_agent_methods:
            if hasattr(BaseAgent, method_name):
                logger.info(f"✅ BaseAgent.{method_name} - METHOD EXISTS")
                base_agent_validation.append(True)
            else:
                logger.error(f"❌ BaseAgent.{method_name} - METHOD MISSING")
                base_agent_validation.append(False)
                
    except Exception as e:
        logger.error(f"❌ BaseAgent validation failed: {e}")
        base_agent_validation = [False] * len(base_agent_methods)
    
    validation_results["base_agent_methods"] = all(base_agent_validation)
    
    # 4. Validate dependency availability
    logger.info("\n📦 VALIDATING DEPENDENCIES")
    logger.info("-" * 40)
    
    dependencies = [
        "aiokafka",
        "numpy", 
        "asyncio",
        "datetime",
        "uuid"
    ]
    
    dependency_validation = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            logger.info(f"✅ {dep} - AVAILABLE")
            dependency_validation.append(True)
        except ImportError:
            logger.error(f"❌ {dep} - MISSING")
            dependency_validation.append(False)
    
    validation_results["dependencies"] = all(dependency_validation)
    
    # 5. Validate main server integration
    logger.info("\n🖥️  VALIDATING MAIN SERVER INTEGRATION")
    logger.info("-" * 40)
    
    try:
        # Check if routers are imported in main.py
        main_py_content = Path("python/src/server/main.py").read_text()
        
        main_integrations = [
            ("kafka_router import", "from .api_routes.kafka_api import router as kafka_router" in main_py_content),
            ("streaming_router import", "from .api_routes.streaming_api import router as streaming_router" in main_py_content),
            ("kafka_router include", "app.include_router(kafka_router)" in main_py_content),
            ("streaming_router include", "app.include_router(streaming_router)" in main_py_content),
            ("streaming service init", "initialize_streaming_service" in main_py_content)
        ]
        
        main_validation = []
        for check_name, check_result in main_integrations:
            status = "✅" if check_result else "❌"
            logger.info(f"{status} {check_name} - {'INTEGRATED' if check_result else 'MISSING'}")
            main_validation.append(check_result)
        
        validation_results["main_integration"] = all(main_validation)
        
    except Exception as e:
        logger.error(f"❌ Main server integration validation failed: {e}")
        validation_results["main_integration"] = False
    
    # Summary
    logger.info("\n📊 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed_categories = 0
    total_categories = len(validation_results)
    
    for category, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {category.replace('_', ' ').title()}")
        if result:
            passed_categories += 1
    
    logger.info("-" * 40)
    logger.info(f"📈 Results: {passed_categories}/{total_categories} categories passed")
    
    if passed_categories == total_categories:
        logger.info("🎉 KAFKA INTEGRATION VALIDATION SUCCESSFUL!")
        logger.info("✅ All components are properly implemented and integrated.")
        return True
    else:
        logger.error(f"💥 {total_categories - passed_categories} validation categories failed.")
        logger.error("❌ Kafka integration needs attention before testing.")
        return False

def main():
    """Main validation runner"""
    try:
        # Change to the correct directory
        os.chdir(Path(__file__).parent)
        
        success = validate_kafka_integration()
        
        logger.info("=" * 60)
        if success:
            logger.info("✅ VALIDATION COMPLETE - READY FOR TESTING")
            logger.info("You can now run: python test_kafka_integration_end_to_end.py")
        else:
            logger.error("❌ VALIDATION FAILED - ISSUES NEED FIXING")
            logger.error("Fix the issues above before proceeding to testing.")
        logger.info("=" * 60)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"💥 Validation runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)