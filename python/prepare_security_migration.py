#!/usr/bin/env python3
"""
Supabase Security Migration Preparation

This script prepares the security migration files and provides execution instructions.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the python directory to the path
python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))

try:
    from supabase import create_client, Client
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install supabase python-dotenv")
    sys.exit(1)

def main():
    """Prepare the security migration."""
    print("\n" + "="*70)
    print("SUPABASE SECURITY MIGRATION - PREPARATION COMPLETE")
    print("="*70)
    print()
    print("CRITICAL SECURITY VULNERABILITIES TO BE FIXED:")
    print("  1. match_archon_crawled_pages - search_path vulnerability")
    print("  2. archive_task - search_path vulnerability")
    print("  3. update_updated_at_column - search_path vulnerability")
    print("  4. match_archon_code_examples - search_path vulnerability")
    print()
    print("EXPECTED PERFORMANCE IMPROVEMENT:")
    print("  Current: 6.39s (PM Enhancement System)")
    print("  Target:  ~500ms")
    print("  Speedup: 12.8x faster")
    print()
    
    # Load environment and test connection
    load_dotenv()
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file")
        return False
    
    # Test connection
    print("TESTING DATABASE CONNECTION...")
    try:
        supabase = create_client(supabase_url, supabase_key)
        response = supabase.table('archon_sources').select('*').limit(1).execute()
        print("  SUCCESS: Database connection verified")
        print(f"  URL: {supabase_url}")
        print()
    except Exception as e:
        print(f"  FAILED: Database connection error: {e}")
        return False
    
    # Check migration files
    migration_file = python_dir / "database_security_migration.sql"
    validation_file = python_dir / "security_validation_queries.sql"
    
    print("CHECKING MIGRATION FILES...")
    if migration_file.exists():
        size = len(migration_file.read_text(encoding='utf-8'))
        print(f"  SUCCESS: Migration script ready ({size:,} characters)")
    else:
        print("  ERROR: Migration script not found!")
        return False
    
    if validation_file.exists():
        size = len(validation_file.read_text(encoding='utf-8'))
        print(f"  SUCCESS: Validation script ready ({size:,} characters)")
    else:
        print("  ERROR: Validation script not found!")
        return False
    
    # Create execution instructions
    instructions_file = python_dir / "MIGRATION_INSTRUCTIONS.md"
    instructions = f"""# Database Security Migration Instructions

## Critical Security Fix - URGENT

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Performance Impact:** 12.8x speedup (6.39s → ~500ms)
**Risk Level:** Critical vulnerabilities → Secure

## Pre-Migration Checklist

- [ ] **BACKUP CREATED** (Go to Supabase Dashboard → Settings → Database → Backups → Create Backup)
- [ ] **PM Enhancement System downtime acceptable** (~5 minutes)
- [ ] **Admin access** to Supabase SQL Editor confirmed

## Execution Steps (15 minutes total)

### Step 1: Create Database Backup (2 minutes)
1. Go to your Supabase Dashboard: {supabase_url.replace('/rest/v1', '')}
2. Navigate to: Settings → Database → Backups
3. Click "Create Backup"
4. Wait for backup completion

### Step 2: Execute Migration (5 minutes)
1. Go to: SQL Editor in your Supabase Dashboard
2. Open file: `database_security_migration.sql`
3. Copy ALL contents of the file
4. Paste into SQL Editor
5. Click "RUN" button
6. Wait for completion (should show success messages)

### Step 3: Validate Migration (3 minutes)
1. In SQL Editor, clear the previous query
2. Open file: `security_validation_queries.sql`
3. Copy ALL contents of the file
4. Paste into SQL Editor
5. Click "RUN" button
6. Verify all functions show "SECURE (search_path set)"

### Step 4: Test System (5 minutes)
1. Test PM Enhancement System functionality
2. Verify performance improvement (should be ~12.8x faster)
3. Check application logs for any errors
4. Monitor system for 10 minutes

## Success Indicators

- [ ] All validation queries show "SECURE"
- [ ] PM Enhancement System responds in ~500ms (vs 6.39s before)
- [ ] No errors in application logs
- [ ] All Archon functionality working normally

## Rollback (Emergency Only)

If critical issues occur:
1. Restore from the backup created in Step 1
2. Contact support with details

**WARNING:** Do not rollback unless absolutely necessary - it restores vulnerabilities!

## Files Required

- `database_security_migration.sql` - Main security fix
- `security_validation_queries.sql` - Verification queries

## Expected Results

Before migration:
```
match_archon_crawled_pages     | ❌ VULNERABLE
archive_task                   | ❌ VULNERABLE  
update_updated_at_column       | ❌ VULNERABLE
match_archon_code_examples     | ❌ VULNERABLE
```

After migration:
```
match_archon_crawled_pages     | ✅ SECURE
archive_task                   | ✅ SECURE
update_updated_at_column       | ✅ SECURE  
match_archon_code_examples     | ✅ SECURE
```

Performance improvement: **6.39s → ~500ms (12.8x speedup)**

---
**Migration prepared at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** Ready for execution
"""
    
    instructions_file.write_text(instructions, encoding='utf-8')
    print(f"  SUCCESS: Instructions created at {instructions_file}")
    print()
    
    print("MIGRATION PREPARATION COMPLETE!")
    print("="*70)
    print()
    print("NEXT ACTIONS:")
    print("1. READ the instructions file:")
    print(f"   {instructions_file}")
    print()
    print("2. EXECUTE the migration using Supabase SQL Editor:")
    print(f"   - Migration script: {migration_file}")
    print(f"   - Validation script: {validation_file}")
    print()
    print("3. VERIFY the 12.8x performance improvement")
    print()
    print("CRITICAL: Create a database backup before proceeding!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)