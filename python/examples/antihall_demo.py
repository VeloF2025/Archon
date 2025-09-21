"""
Anti-Hallucination System Demo

This script demonstrates how the 75% confidence rule and anti-hallucination
validation prevents AI agents from generating incorrect code.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.server.services.validation_service import (
    ValidationService,
    ValidationConfig,
    initialize_validation_service
)
from src.agents.validation.enhanced_antihall_validator import (
    EnhancedAntiHallValidator,
    CodeReference,
    ValidationResult
)
from src.agents.validation.confidence_based_responses import (
    ConfidenceBasedResponseSystem,
    ConfidenceLevel
)

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

async def demo_code_validation():
    """Demonstrate code validation preventing hallucinations"""
    print_header("DEMO 1: Code Validation (Preventing Hallucinations)")
    
    # Initialize validator with Archon project
    project_root = Path(__file__).parent.parent
    validator = EnhancedAntiHallValidator(str(project_root))
    
    # Example 1: Valid code references
    print(f"{Colors.BOLD}Testing VALID code:{Colors.ENDC}")
    valid_code = """
from pathlib import Path
import asyncio

async def main():
    path = Path(__file__)
    print(f"Current file: {path}")
    await asyncio.sleep(1)
"""
    
    reports = validator.validate_code_snippet(valid_code, "python")
    is_valid, summary = validator.enforce_validation(valid_code, "python")
    
    if is_valid:
        print_success(f"Code validation PASSED (confidence: {summary['average_confidence']:.0%})")
        print_info(f"Total references: {summary['total_references']}, Valid: {summary['valid_references']}")
    else:
        print_error("Code validation FAILED")
        
    # Example 2: Code with hallucinations
    print(f"\n{Colors.BOLD}Testing HALLUCINATED code:{Colors.ENDC}")
    hallucinated_code = """
from fake_module import NonExistentClass

def process_data():
    obj = NonExistentClass()
    result = obj.do_something_generic()
    return magic_function_that_doesnt_exist(result)
"""
    
    is_valid, summary = validator.enforce_validation(hallucinated_code, "python")
    
    if not is_valid:
        print_error(f"Code validation FAILED (as expected)")
        print_warning(f"Invalid references found: {summary['invalid_references']}")
        print_info("Critical errors:")
        for error in summary['critical_errors']:
            print(f"  {error}")
        if summary['suggestions']:
            print_info("Suggestions:")
            for suggestion in summary['suggestions']:
                print(f"  💡 {suggestion}")
    
    # Example 3: Code with typos that get suggestions
    print(f"\n{Colors.BOLD}Testing code with TYPOS:{Colors.ENDC}")
    typo_code = """
from pathib import Path  # Typo: should be 'pathlib'
import asyncoi  # Typo: should be 'asyncio'
"""
    
    reports = validator.validate_code_snippet(typo_code, "python")
    
    for report in reports:
        if report.result == ValidationResult.NOT_FOUND and report.similar_matches:
            print_warning(f"'{report.reference.name}' not found")
            if report.suggestion:
                print_info(f"  {report.suggestion}")

async def demo_confidence_checking():
    """Demonstrate 75% confidence rule enforcement"""
    print_header("DEMO 2: 75% Confidence Rule Enforcement")
    
    confidence_system = ConfidenceBasedResponseSystem(min_confidence_threshold=0.75)
    
    # Test different confidence levels
    test_cases = [
        {
            "name": "High Confidence (95%)",
            "context": {
                "code_validation": {"all_references_valid": True, "validation_rate": 1.0},
                "documentation_found": True,
                "tests_exist": True,
                "similar_patterns_found": True,
                "recently_used": True
            },
            "content": "Here's the solution to your problem..."
        },
        {
            "name": "Moderate Confidence (80%)",
            "context": {
                "code_validation": {"all_references_valid": True, "validation_rate": 0.9},
                "documentation_found": True,
                "tests_exist": False,
                "similar_patterns_found": True,
                "recently_used": False
            },
            "content": "Based on my analysis, this should work..."
        },
        {
            "name": "Low Confidence (60%)",
            "context": {
                "code_validation": {"all_references_valid": False, "validation_rate": 0.6},
                "documentation_found": False,
                "tests_exist": False,
                "similar_patterns_found": True,
                "recently_used": False
            },
            "content": "I found a possible solution..."
        },
        {
            "name": "Very Low Confidence (30%)",
            "context": {
                "code_validation": {"all_references_valid": False, "validation_rate": 0.3},
                "documentation_found": False,
                "tests_exist": False,
                "similar_patterns_found": False,
                "recently_used": False
            },
            "content": "Maybe this could work..."
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{Colors.BOLD}{test_case['name']}:{Colors.ENDC}")
        
        assessment = confidence_system.assess_confidence(test_case['context'])
        response = confidence_system.generate_response(
            test_case['content'],
            assessment,
            "code_implementation"
        )
        
        # Print confidence assessment
        confidence_pct = assessment.confidence_score * 100
        if assessment.confidence_level == ConfidenceLevel.HIGH:
            print_success(f"Confidence: {confidence_pct:.0f}% - {assessment.confidence_level.value}")
        elif assessment.confidence_level == ConfidenceLevel.MODERATE:
            print_info(f"Confidence: {confidence_pct:.0f}% - {assessment.confidence_level.value}")
        else:
            print_warning(f"Confidence: {confidence_pct:.0f}% - {assessment.confidence_level.value}")
        
        # Show if 75% rule blocks it
        if assessment.confidence_score < 0.75:
            print_error("❌ BLOCKED by 75% rule - Would say 'I don't know'")
        else:
            print_success("✅ PASSES 75% threshold - Can provide answer")
        
        # Show response preview
        print(f"\n{Colors.OKCYAN}Response preview:{Colors.ENDC}")
        preview = response[:200] + "..." if len(response) > 200 else response
        print(f"  {preview}")
        
        # Show uncertainties if any
        if assessment.uncertainties:
            print(f"\n{Colors.WARNING}Uncertainties:{Colors.ENDC}")
            for uncertainty in assessment.uncertainties[:3]:
                print(f"  • {uncertainty}")

async def demo_realtime_validation():
    """Demonstrate real-time validation as code is typed"""
    print_header("DEMO 3: Real-Time Validation (As You Type)")
    
    # Initialize service
    config = ValidationConfig(
        project_root=str(Path(__file__).parent.parent),
        min_confidence_threshold=0.75,
        enable_real_time_validation=True
    )
    service = ValidationService(config)
    
    # Simulate typing code line by line
    code_lines = [
        "import os  # Valid",
        "from pathlib import Path  # Valid",
        "result = fake_function()  # Will be flagged",
        "obj.do_something()  # Suspicious generic method",
        "data = process_data()  # Might exist",
        "# TODO: implement this later  # Incomplete code flag",
    ]
    
    print("Simulating real-time code entry:\n")
    
    for line in code_lines:
        # Strip comment for display
        display_line = line.split("#")[0].strip()
        comment = line.split("#")[1].strip() if "#" in line else ""
        
        print(f"Typing: {Colors.BOLD}{display_line}{Colors.ENDC}")
        
        # Validate in real-time
        error = await service.perform_real_time_validation(display_line, {})
        
        if error:
            print_error(f"  Real-time validation: {error}")
        else:
            print_success(f"  Real-time validation: OK")
        
        if comment:
            print_info(f"  Expected: {comment}")
        
        print()
        await asyncio.sleep(0.5)  # Simulate typing delay

async def demo_agent_response_validation():
    """Demonstrate validation of AI agent responses"""
    print_header("DEMO 4: AI Agent Response Validation")
    
    # Initialize service
    service = await initialize_validation_service(str(Path(__file__).parent.parent))
    
    # Example agent responses to validate
    responses = [
        {
            "name": "Confident but wrong response",
            "response": """Here's the solution:

```python
def process_data(input_data):
    processor = DataProcessor()  # This class doesn't exist
    result = processor.transform(input_data)
    return optimize_result(result)  # This function doesn't exist
```

This will definitely work!""",
            "contains_code": True
        },
        {
            "name": "Uncertain response with hedging",
            "response": """I think this might work, but I'm not entirely sure:

```python
import os
path = os.getcwd()
```

This should probably give you what you need, though I could be wrong.""",
            "contains_code": True
        },
        {
            "name": "High confidence valid response",
            "response": """Here's the solution using standard library functions:

```python
from pathlib import Path
import json

def load_config(config_path):
    path = Path(config_path)
    with open(path, 'r') as f:
        return json.load(f)
```

This uses well-established Python standard library modules.""",
            "contains_code": True
        }
    ]
    
    for example in responses:
        print(f"\n{Colors.BOLD}{example['name']}:{Colors.ENDC}")
        print(f"Response: '{example['response'][:100]}...'\n")
        
        result = await service.validate_agent_response(
            example['response'],
            {"contains_code": example['contains_code']}
        )
        
        # Show confidence
        confidence = result['confidence_result']['confidence_score']
        if confidence >= 0.75:
            print_success(f"Confidence: {confidence:.0%}")
        else:
            print_error(f"Confidence: {confidence:.0%} - TOO LOW!")
        
        # Show validation results
        if result['validation_results']:
            valid_count = sum(1 for v in result['validation_results'] if v['valid'])
            total_count = len(result['validation_results'])
            if valid_count == total_count:
                print_success(f"Code validation: {valid_count}/{total_count} references valid")
            else:
                print_error(f"Code validation: {valid_count}/{total_count} references valid")
        
        # Show uncertainty detection
        if result['uncertainty_detected']:
            print_warning(f"Uncertainty detected: {', '.join(result['uncertainty_patterns'][:3])}")
        
        # Show final verdict
        if result['confidence_result'].get('confidence_too_low'):
            print_error("❌ BLOCKED: Would be rewritten to say 'I don't know'")
        elif not all(v['valid'] for v in result['validation_results'] if result['validation_results']):
            print_error("❌ BLOCKED: Contains hallucinated code references")
        else:
            print_success("✅ APPROVED: Response can be used")

async def demo_statistics():
    """Show validation service statistics"""
    print_header("DEMO 5: Validation Service Statistics")
    
    # Initialize and use service
    service = await initialize_validation_service(str(Path(__file__).parent.parent))
    
    # Perform various validations to generate stats
    print("Performing test validations...\n")
    
    # Valid references
    for _ in range(5):
        await service.validate_code_reference("Path", "class")
    
    # Invalid references
    for _ in range(3):
        await service.validate_code_reference("FakeClass", "class")
    
    # Low confidence checks
    for _ in range(2):
        await service.validate_with_confidence(
            "Test content",
            {"code_validation": {"all_references_valid": False, "validation_rate": 0.4}}
        )
    
    # Get statistics
    stats = service.get_statistics()
    
    print(f"{Colors.BOLD}Validation Service Statistics:{Colors.ENDC}\n")
    
    print(f"Total Validations: {stats['total_validations']}")
    print(f"Successful: {stats['successful_validations']}")
    print(f"Failed: {stats['failed_validations']}")
    print(f"Hallucinations Prevented: {stats['hallucinations_prevented']}")
    print(f"Low Confidence Blocks: {stats['low_confidence_blocks']}")
    
    print(f"\n{Colors.BOLD}Performance Metrics:{Colors.ENDC}\n")
    
    if stats['validation_success_rate'] > 0.7:
        print_success(f"Validation Success Rate: {stats['validation_success_rate']:.1%}")
    else:
        print_warning(f"Validation Success Rate: {stats['validation_success_rate']:.1%}")
    
    print_info(f"Average Confidence: {stats['average_confidence']:.1%}")
    print_info(f"Confidence Block Rate: {stats['confidence_block_rate']:.1%}")
    print_info(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")

async def main():
    """Run all demos"""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ARCHON ANTI-HALLUCINATION SYSTEM DEMONSTRATION      ║")
    print("║           75% Confidence Rule Enforcement               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    demos = [
        ("Code Validation", demo_code_validation),
        ("Confidence Checking", demo_confidence_checking),
        ("Real-Time Validation", demo_realtime_validation),
        ("Agent Response Validation", demo_agent_response_validation),
        ("Statistics", demo_statistics)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        if i > 1:
            print(f"\n{Colors.OKCYAN}Press Enter to continue to Demo {i}...{Colors.ENDC}")
            input()
        
        await demo_func()
    
    print_header("DEMONSTRATION COMPLETE")
    print_success("The Anti-Hallucination System is protecting your code!")
    print_success("75% Confidence Rule is enforced on all AI responses!")
    print_info("When in doubt, Archon will say 'I don't know' and work with you.")

if __name__ == "__main__":
    asyncio.run(main()