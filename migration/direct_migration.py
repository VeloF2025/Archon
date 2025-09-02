#!/usr/bin/env python3
"""
Direct Phase 7 DeepConf Database Migration
==========================================
Executes migration using direct PostgreSQL connection (psycopg2).
This approach bypasses Supabase client dependencies.
"""

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

def load_env_vars():
    """Load environment variables from .env file."""
    env_file = Path('.env')
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"\'')
    
    # Also check system environment
    env_vars.update({
        'SUPABASE_URL': env_vars.get('SUPABASE_URL') or os.getenv('SUPABASE_URL'),
        'SUPABASE_SERVICE_KEY': env_vars.get('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_SERVICE_KEY')
    })
    
    return env_vars

def parse_supabase_url(supabase_url, service_key):
    """Parse Supabase URL to PostgreSQL connection parameters."""
    # Supabase URLs are like: https://project.supabase.co
    # We need to convert to PostgreSQL connection format
    parsed = urlparse(supabase_url)
    project_id = parsed.hostname.split('.')[0]
    
    return {
        'host': f"db.{project_id}.supabase.co",
        'port': 5432,
        'database': 'postgres',
        'user': 'postgres',
        'password': service_key
    }

def execute_sql_file(cursor, file_path, description):
    """Execute an SQL file."""
    try:
        print(f"\n🔄 {description}...")
        
        with open(file_path, 'r') as f:
            sql_content = f.read()
        
        # Execute the entire content
        cursor.execute(sql_content)
        print(f"✓ {description} completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ {description} failed: {str(e)}")
        return False

def validate_migration(cursor):
    """Validate that the migration was successful."""
    validation_results = {
        'tables': 0,
        'indexes': 0,
        'views': 0,
        'functions': 0,
        'success': False,
        'errors': []
    }
    
    try:
        print("\n🔍 Validating migration results...")
        
        # Check tables
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name IN ('archon_confidence_scores', 'archon_performance_metrics', 'archon_confidence_calibration')
            AND table_schema = 'public'
        """)
        validation_results['tables'] = cursor.fetchone()[0]
        print(f"  Tables: {validation_results['tables']}/3")
        
        # Check indexes
        cursor.execute("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%'
        """)
        validation_results['indexes'] = cursor.fetchone()[0]
        print(f"  Indexes: {validation_results['indexes']} (expected 15+)")
        
        # Check views
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.views 
            WHERE table_name LIKE 'archon_%confidence%' OR table_name LIKE 'archon_%performance%'
        """)
        validation_results['views'] = cursor.fetchone()[0]
        print(f"  Views: {validation_results['views']}/3")
        
        # Check functions
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.routines 
            WHERE (routine_name LIKE '%confidence%' OR routine_name LIKE '%performance%')
            AND routine_type = 'FUNCTION'
        """)
        validation_results['functions'] = cursor.fetchone()[0]
        print(f"  Functions: {validation_results['functions']} (expected 2+)")
        
        # Determine overall success
        validation_results['success'] = (
            validation_results['tables'] == 3 and
            validation_results['views'] >= 2 and
            validation_results['functions'] >= 2
        )
        
        return validation_results
        
    except Exception as e:
        validation_results['errors'].append(f"Validation process failed: {e}")
        return validation_results

def main():
    """Main migration execution function."""
    print("🚀 Phase 7 DeepConf Database Migration (Direct Connection)")
    print("=" * 60)
    
    # Load environment
    env_vars = load_env_vars()
    supabase_url = env_vars.get('SUPABASE_URL')
    supabase_key = env_vars.get('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("✗ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables")
        sys.exit(1)
    
    print(f"📍 Target: {supabase_url[:50]}...")
    
    # Parse connection parameters
    conn_params = parse_supabase_url(supabase_url, supabase_key)
    print(f"📡 Connecting to: {conn_params['host']}")
    
    # Try to connect using psycopg2
    try:
        import psycopg2
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True  # Enable autocommit for DDL
        cursor = conn.cursor()
        print("✓ PostgreSQL connection successful")
    except ImportError:
        print("✗ psycopg2 package not found. Cannot proceed with direct connection.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        sys.exit(1)
    
    try:
        # Test connection
        cursor.execute("SELECT version(), current_database(), current_user")
        version, database, user = cursor.fetchone()
        print(f"  Database: {database}")
        print(f"  User: {user}")
        print(f"  Version: {version[:50]}...")
        
        # Define migration files
        migration_dir = Path('migration')
        prerequisite_file = migration_dir / 'prerequisite_functions.sql'
        main_migration_file = migration_dir / 'phase7_deepconf_schema.sql'
        
        # Check files exist
        if not prerequisite_file.exists():
            print(f"✗ Prerequisite file not found: {prerequisite_file}")
            sys.exit(1)
        
        if not main_migration_file.exists():
            print(f"✗ Main migration file not found: {main_migration_file}")
            sys.exit(1)
        
        print("✓ All migration files found")
        
        # Execute prerequisite functions
        if not execute_sql_file(cursor, prerequisite_file, "Installing prerequisite functions"):
            print("✗ Migration aborted due to prerequisite failure")
            sys.exit(1)
        
        # Execute main migration
        if not execute_sql_file(cursor, main_migration_file, "Executing Phase 7 DeepConf migration"):
            print("✗ Main migration failed")
            sys.exit(1)
        
        # Validate migration
        validation = validate_migration(cursor)
        
        print("\n📊 Migration Summary:")
        print(f"  Tables created: {validation['tables']}/3")
        print(f"  Indexes created: {validation['indexes']}")
        print(f"  Views created: {validation['views']}/3")
        print(f"  Functions created: {validation['functions']}")
        
        if validation['errors']:
            print("\n⚠ Validation Errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        if validation['success']:
            print("\n🎉 Phase 7 DeepConf migration completed successfully!")
            print("\nNext steps:")
            print("  1. Test basic CRUD operations")
            print("  2. Verify RLS policies")
            print("  3. Update storage.py to use new tables")
            return True
        else:
            print("\n⚠ Migration completed with issues - manual verification required")
            return False
            
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)