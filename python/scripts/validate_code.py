#!/usr/bin/env python3
"""
Command-line tool for validating code using Archon's Anti-Hallucination System

Usage:
    python validate_code.py <file_or_code> [options]
    
Examples:
    python validate_code.py myfile.py
    python validate_code.py "print(fake_function())"
    python validate_code.py --check-confidence "solution code" --context file.json
    python validate_code.py --real-time
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.server.services.validation_service import (
    ValidationService,
    ValidationConfig,
    initialize_validation_service
)
from src.agents.validation.enhanced_antihall_validator import (
    EnhancedAntiHallValidator,
    ValidationResult
)

# Color codes for terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}", file=sys.stderr)

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.ENDC}")

def print_info(msg: str):
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.ENDC}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{msg}{Colors.ENDC}")

async def validate_file(file_path: Path, language: Optional[str] = None) -> bool:
    """Validate a code file"""
    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        return False
    
    # Detect language from extension if not provided
    if not language:
        ext = file_path.suffix.lower()
        language_map = {
            '.py': 'python',
            '.ts': 'typescript',
            '.js': 'javascript',
            '.tsx': 'typescript',
            '.jsx': 'javascript'
        }
        language = language_map.get(ext, 'python')
    
    print_header(f"Validating {file_path} ({language})")
    
    # Read file content
    code = file_path.read_text()
    
    # Initialize validator
    project_root = Path.cwd()
    validator = EnhancedAntiHallValidator(str(project_root))
    
    # Validate
    is_valid, summary = validator.enforce_validation(code, language)
    
    # Print results
    print(f"\nValidation Results:")
    print(f"  Total references: {summary['total_references']}")
    print(f"  Valid references: {summary['valid_references']}")
    print(f"  Invalid references: {summary['invalid_references']}")
    
    if 'average_confidence' in summary:
        confidence_pct = summary['average_confidence'] * 100
        if confidence_pct >= 75:
            print_success(f"Average confidence: {confidence_pct:.0f}%")
        else:
            print_error(f"Average confidence: {confidence_pct:.0f}% (below 75% threshold)")
    
    if is_valid:
        print_success(f"\nValidation PASSED - Code is safe to use")
    else:
        print_error(f"\nValidation FAILED - Code contains issues")
        
        if summary['critical_errors']:
            print_error("\nCritical Errors:")
            for error in summary['critical_errors']:
                print(f"  {error}")
        
        if summary['suggestions']:
            print_info("\nSuggestions:")
            for suggestion in summary['suggestions']:
                print(f"  💡 {suggestion}")
    
    return is_valid

async def validate_snippet(code: str, language: str = 'python') -> bool:
    """Validate a code snippet"""
    print_header(f"Validating code snippet ({language})")
    
    # Initialize validator
    project_root = Path.cwd()
    validator = EnhancedAntiHallValidator(str(project_root))
    
    # Validate
    is_valid, summary = validator.enforce_validation(code, language)
    
    # Print results
    if is_valid:
        print_success("Code validation PASSED")
    else:
        print_error("Code validation FAILED")
        
        for error in summary.get('critical_errors', []):
            print_error(f"  {error}")
    
    return is_valid

async def check_confidence(content: str, context_file: Optional[Path] = None) -> bool:
    """Check confidence level for content"""
    print_header("Checking Confidence Level")
    
    # Load context if provided
    context = {}
    if context_file and context_file.exists():
        context = json.loads(context_file.read_text())
        print_info(f"Loaded context from {context_file}")
    
    # Initialize service
    service = await initialize_validation_service(str(Path.cwd()))
    
    # Check confidence
    result = await service.validate_with_confidence(content, context)
    
    # Print results
    confidence = result.get('confidence_score', 0)
    confidence_pct = confidence * 100
    
    print(f"\nContent: {content[:100]}..." if len(content) > 100 else f"\nContent: {content}")
    
    if confidence >= 0.75:
        print_success(f"Confidence: {confidence_pct:.0f}% - PASSES 75% threshold")
        print_info(f"Level: {result.get('confidence_level', 'unknown')}")
    else:
        print_error(f"Confidence: {confidence_pct:.0f}% - BELOW 75% threshold")
        print_warning("AI would say: 'I don't know, let's figure this out together'")
    
    if result.get('uncertainties'):
        print_warning("\nUncertainties:")
        for uncertainty in result['uncertainties']:
            print(f"  • {uncertainty}")
    
    if result.get('suggestions'):
        print_info("\nSuggestions to improve confidence:")
        for suggestion in result['suggestions']:
            print(f"  • {suggestion}")
    
    return confidence >= 0.75

async def real_time_mode():
    """Interactive real-time validation mode"""
    print_header("Real-Time Validation Mode")
    print_info("Type code lines for instant validation (type 'exit' to quit)")
    print_info("Suspicious patterns will be flagged immediately\n")
    
    # Initialize service
    config = ValidationConfig(
        project_root=str(Path.cwd()),
        enable_real_time_validation=True
    )
    service = ValidationService(config)
    
    while True:
        try:
            line = input(f"{Colors.BOLD}>>> {Colors.ENDC}")
            
            if line.lower() in ['exit', 'quit', 'q']:
                break
            
            if not line.strip():
                continue
            
            # Validate line
            error = await service.perform_real_time_validation(line, {})
            
            if error:
                print_error(f"Validation: {error}")
            else:
                print_success("Validation: OK")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print_error(f"Error: {e}")
    
    print_info("\nExiting real-time mode")

async def show_statistics():
    """Show validation service statistics"""
    print_header("Validation Service Statistics")
    
    # Initialize service
    service = await initialize_validation_service(str(Path.cwd()))
    
    # Get stats
    stats = service.get_statistics()
    
    print("\nValidation Metrics:")
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  Successful: {stats['successful_validations']}")
    print(f"  Failed: {stats['failed_validations']}")
    print(f"  Hallucinations prevented: {stats['hallucinations_prevented']}")
    print(f"  Low confidence blocks: {stats['low_confidence_blocks']}")
    
    print("\nPerformance:")
    print(f"  Validation success rate: {stats['validation_success_rate']:.1%}")
    print(f"  Average confidence: {stats['average_confidence']:.1%}")
    print(f"  Confidence block rate: {stats['confidence_block_rate']:.1%}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate code using Archon's Anti-Hallucination System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s myfile.py                    # Validate a Python file
  %(prog)s script.ts --language typescript  # Validate TypeScript
  %(prog)s "print(fake())" --snippet    # Validate code snippet
  %(prog)s --check-confidence "answer" --context ctx.json
  %(prog)s --real-time                  # Interactive mode
  %(prog)s --stats                      # Show statistics
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='File path or code snippet to validate'
    )
    
    parser.add_argument(
        '--language', '-l',
        default='python',
        choices=['python', 'typescript', 'javascript'],
        help='Programming language (default: python)'
    )
    
    parser.add_argument(
        '--snippet', '-s',
        action='store_true',
        help='Treat input as code snippet instead of file path'
    )
    
    parser.add_argument(
        '--check-confidence', '-c',
        metavar='CONTENT',
        help='Check confidence level for content'
    )
    
    parser.add_argument(
        '--context',
        type=Path,
        help='JSON file with context for confidence checking'
    )
    
    parser.add_argument(
        '--real-time', '-r',
        action='store_true',
        help='Enter real-time validation mode'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show validation service statistics'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.75,
        help='Minimum confidence threshold (default: 0.75)'
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    success = True
    
    try:
        if args.stats:
            await show_statistics()
        elif args.real_time:
            await real_time_mode()
        elif args.check_confidence:
            success = await check_confidence(args.check_confidence, args.context)
        elif args.input:
            if args.snippet:
                success = await validate_snippet(args.input, args.language)
            else:
                file_path = Path(args.input)
                success = await validate_file(file_path, args.language)
        else:
            parser.print_help()
            return 0
    
    except KeyboardInterrupt:
        print_info("\nInterrupted by user")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)