#!/usr/bin/env python3
"""
Execute Phase 7 DeepConf Database Migration
Connects to Supabase and runs the Phase 7 DeepConf schema migration
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Check if asyncpg is available
try:
    import asyncpg
except ImportError:
    print("❌ asyncpg not found. Please install it with: pip install asyncpg")
    exit(1)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

# Parse database URL for asyncpg
# Convert Supabase URL to PostgreSQL connection format
if SUPABASE_URL:
    # Extract project reference from URL
    project_ref = SUPABASE_URL.split('https://')[1].split('.supabase.co')[0]
    
    # Construct PostgreSQL connection URL
    db_url = f"postgresql://postgres:{SUPABASE_SERVICE_KEY}@db.{project_ref}.supabase.co:5432/postgres"
else:
    print("❌ SUPABASE_URL not found in .env file")
    exit(1)

async def validate_migration(conn):
    """Validate that the migration was successful."""
    print("\n🔍 Validating migration results...")
    
    validation_results = {
        'tables': 0,
        'indexes': 0,
        'views': 0,
        'functions': 0,
        'success': False
    }
    
    try:
        # Check tables
        tables_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name IN ('archon_confidence_scores', 'archon_performance_metrics', 'archon_confidence_calibration')
            AND table_schema = 'public'
        """)
        validation_results['tables'] = tables_count
        print(f"  Tables: {validation_results['tables']}/3")
        
        # Check indexes
        indexes_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%'
        """)
        validation_results['indexes'] = indexes_count
        print(f"  Indexes: {validation_results['indexes']} (expected 15+)")
        
        # Check views
        views_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.views 
            WHERE table_name LIKE 'archon_%confidence%' OR table_name LIKE 'archon_%performance%'
        """)
        validation_results['views'] = views_count
        print(f"  Views: {validation_results['views']}/3")
        
        # Check functions
        functions_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM information_schema.routines 
            WHERE (routine_name LIKE '%confidence%' OR routine_name LIKE '%performance%')
            AND routine_type = 'FUNCTION'
            AND routine_schema = 'public'
        """)
        validation_results['functions'] = functions_count
        print(f"  Functions: {validation_results['functions']} (expected 2+)")
        
        # Check RLS policies
        try:
            rls_count = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM pg_policies 
                WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%'
            """)
            print(f"  RLS Policies: {rls_count} (expected 3+)")
        except Exception as e:
            print(f"  RLS Policies: Could not check ({e})")
        
        # Determine overall success
        validation_results['success'] = (
            validation_results['tables'] == 3 and
            validation_results['views'] >= 2 and
            validation_results['functions'] >= 2
        )
        
        return validation_results
        
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return validation_results

async def test_basic_operations(conn):
    """Test basic CRUD operations on the confidence tables."""
    print("\n🧪 Testing basic CRUD operations...")
    
    try:
        # Test insert into confidence scores
        test_id = await conn.fetchval("""
            INSERT INTO archon_confidence_scores (
                request_id, factual_confidence, reasoning_confidence, contextual_relevance,
                uncertainty_lower, uncertainty_upper, model_consensus, request_type
            ) VALUES (
                gen_random_uuid(), 0.8500, 0.7200, 0.9100,
                0.0500, 0.1200, '{"model_1": 0.85, "model_2": 0.82}', 'test_migration'
            ) RETURNING id
        """)
        print(f"  ✅ Insert test successful (ID: {str(test_id)[:8]}...)")
        
        # Test select with computed column
        result = await conn.fetchrow("""
            SELECT overall_confidence, factual_confidence, reasoning_confidence, contextual_relevance
            FROM archon_confidence_scores 
            WHERE id = $1
        """, test_id)
        
        expected_overall = (0.8500 + 0.7200 + 0.9100) / 3.0
        actual_overall = float(result['overall_confidence'])
        
        if abs(actual_overall - expected_overall) < 0.0001:
            print(f"  ✅ Computed column test successful (overall: {actual_overall:.4f})")
        else:
            print(f"  ⚠️ Computed column mismatch: expected {expected_overall:.4f}, got {actual_overall:.4f}")
        
        # Test performance metrics insert
        perf_id = await conn.fetchval("""
            SELECT insert_performance_metric(
                0.8500, 250, 0.7800, 0.0200, 0.6500, 
                'test_migration', 'test-model-v1', '/api/test'
            )
        """)
        print(f"  ✅ Performance metric insert successful (ID: {str(perf_id)[:8]}...)")
        
        # Test confidence statistics function
        stats = await conn.fetchrow("""
            SELECT * FROM calculate_confidence_stats(
                NOW() - INTERVAL '1 hour',
                NOW(),
                'test_migration'
            )
        """)
        
        if stats and stats['total_requests'] >= 1:
            print(f"  ✅ Statistics function test successful ({stats['total_requests']} requests)")
        else:
            print(f"  ⚠️ Statistics function returned unexpected results")
        
        # Clean up test data
        await conn.execute("DELETE FROM archon_confidence_scores WHERE request_type = 'test_migration'")
        await conn.execute("DELETE FROM archon_performance_metrics WHERE request_type = 'test_migration'")
        print(f"  🧹 Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CRUD operations test failed: {e}")
        return False

async def execute_phase7_migration():
    """Execute the Phase 7 DeepConf migration"""
    
    print("🚀 Phase 7 DeepConf Database Migration")
    print("=" * 45)
    
    try:
        # Check if migration files exist
        prerequisite_file = Path("migration/prerequisite_functions.sql")
        main_migration_file = Path("migration/phase7_deepconf_schema.sql")
        
        if not prerequisite_file.exists():
            print(f"❌ Prerequisite file not found: {prerequisite_file}")
            return False
        
        if not main_migration_file.exists():
            print(f"❌ Main migration file not found: {main_migration_file}")
            return False
        
        print("📄 Migration files found")
        
        # Read the migration scripts
        with open(prerequisite_file, 'r', encoding='utf-8') as f:
            prerequisite_sql = f.read()
        
        with open(main_migration_file, 'r', encoding='utf-8') as f:
            main_migration_sql = f.read()
        
        print(f"📄 Loaded prerequisite script ({len(prerequisite_sql)} characters)")
        print(f"📄 Loaded main migration script ({len(main_migration_sql)} characters)")
        
        # Connect to Supabase PostgreSQL
        print("\n🔌 Connecting to Supabase database...")
        
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to Supabase successfully")
        
        # Check database version and extensions
        version = await conn.fetchval("SELECT version()")
        print(f"📊 Database: {version[:50]}...")
        
        # Execute prerequisite functions
        print("\n🔧 Installing prerequisite functions...")
        try:
            await conn.execute(prerequisite_sql)
            print("✅ Prerequisite functions installed successfully")
        except Exception as e:
            print(f"⚠️ Prerequisite installation warning: {e}")
            print("   Continuing with main migration...")
        
        # Execute the main migration
        print("\n🚀 Executing Phase 7 DeepConf migration...")
        print("⏱️  This will take approximately 1-2 minutes...")
        print()
        
        # Execute the migration script
        await conn.execute(main_migration_sql)
        
        print()
        print("✅ Phase 7 DeepConf migration executed successfully!")
        
        # Validate migration
        validation = await validate_migration(conn)
        
        if validation['success']:
            print("\n🎉 Migration validation passed!")
            
            # Test basic operations
            if await test_basic_operations(conn):
                print("\n🏆 All tests passed - migration fully successful!")
            else:
                print("\n⚠️ Migration completed but some tests failed")
        else:
            print("\n⚠️ Migration completed but validation had issues")
        
        await conn.close()
        print("\n📊 Migration Summary:")
        print(f"  Tables created: {validation['tables']}/3")
        print(f"  Indexes created: {validation['indexes']}")
        print(f"  Views created: {validation['views']}/3") 
        print(f"  Functions created: {validation['functions']}")
        
        return validation['success']
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(execute_phase7_migration())
    
    if success:
        print("\n" + "=" * 45)
        print("🚀 NEXT STEPS:")
        print("1. Update storage.py to use new tables")
        print("2. Test DeepConf integration")
        print("3. Monitor confidence scoring in dashboard")
        print("=" * 45)
    else:
        print("\n❌ MIGRATION FAILED - Please check logs and try again")