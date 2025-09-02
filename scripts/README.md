# Archon Development Scripts

This directory contains all development, testing, and utility scripts organized by purpose.

## 📁 Directory Structure

### `/development/`
Scripts for project lifecycle management:
- `create_*.py` - Project creation and initialization
- `update_*.py` - Update and modification scripts  
- `finalize_*.py` - Project finalization scripts
- `mark_*.py` - Phase marking and completion scripts

### `/testing/`
Test execution and validation scripts:
- `test_*.py` - Component and integration tests
- `quick_*.py` - Quick validation scripts
- Testing framework utilities

### `/debugging/`  
Debugging and investigation tools:
- `debug_*.py` - Debug analysis scripts
- `investigate_*.py` - Investigation utilities
- `monitor_*.py` - Monitoring scripts

### `/fixes/`
Issue resolution and correction scripts:
- `fix_*.py` - Bug fix automation
- `execute_*.py` - Fix execution scripts
- Emergency repair utilities

### `/validation/`
Validation and verification tools:
- `*validation*.py` - Data validation scripts
- `verify_*.py` - System verification
- `check_*.py` - Health check utilities

### `/benchmarks/`
Performance and metrics scripts:
- `*benchmark*.py` - Performance benchmarking
- `*scwt*.py` - SCWT metrics evaluation
- Performance analysis tools

### `/utilities/`
General utility and helper scripts:
- `dynamic_task_tracker.py` - Task management
- `dgts_workflow_demo.py` - Workflow demonstrations
- `claude_code_phase6_executor.py` - Phase execution
- `get-pip.py` - Package installer

## 🚀 Usage

All scripts maintain their original functionality and command-line interfaces. Simply reference them by their new organized paths:

```bash
# Before
python debug_deepconf.py

# After  
python scripts/debugging/debug_deepconf.py
```

## 📋 Best Practices

- Scripts are categorized by primary function
- Original functionality preserved
- Use relative paths when calling between scripts
- Maintain backwards compatibility where possible

## 🔧 Production Safety

**Important**: These scripts are development tools and should not be used in production environments without proper testing and validation.

---
*Organized: September 2025*  
*Total Scripts: 50+ development utilities*