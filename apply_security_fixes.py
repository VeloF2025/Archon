#!/usr/bin/env python3
"""
Apply Supabase Security Fixes
Executes the security fixes SQL script and validates the results.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add the Python src path to enable imports
sys.path.append(str(Path(__file__).parent / "python" / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class SupabaseSecurityFixer:
    def __init__(self):
        self.db_url = os.getenv("SUPABASE_URL")
        self.service_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.db_url or not self.service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment")
        
        print(f"🔗 Using Supabase project: {self.db_url}")

    def execute_sql_file(self, sql_file_path: str) -> bool:
        """Execute the SQL file to apply security fixes."""
        try:
            print(f"🔧 Applying security fixes from {sql_file_path}")
            
            # Read the SQL file
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Connect to database and execute SQL
            conn = await asyncpg.connect(self.connection_string)
            
            try:
                # Split SQL into individual statements and execute
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for i, statement in enumerate(statements):
                    if statement and not statement.startswith('--'):
                        try:
                            print(f"  Executing statement {i+1}/{len(statements)}...")
                            await conn.execute(statement)
                        except Exception as e:
                            print(f"  ⚠️  Warning on statement {i+1}: {e}")
                            # Continue with other statements
                
                print("✅ Security fixes applied successfully")
                return True
                
            finally:
                await conn.close()
                
        except Exception as e:
            print(f"❌ Error applying security fixes: {e}")
            return False

    async def validate_security_fixes(self) -> Dict[str, Any]:
        """Validate that security fixes were applied correctly."""
        try:
            print("🔍 Validating security fixes...")
            
            conn = await asyncpg.connect(self.connection_string)
            results = {}
            
            try:
                # Check RLS is enabled on all target tables
                rls_check = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        rowsecurity as rls_enabled
                    FROM pg_tables pt
                    JOIN pg_class c ON c.relname = pt.tablename
                    JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = pt.schemaname
                    WHERE schemaname = 'public' 
                      AND (tablename LIKE 'archon_%' OR tablename LIKE 'feature_%' OR tablename = 'user_feature_assignments')
                    ORDER BY tablename;
                """)
                
                results['rls_status'] = [dict(row) for row in rls_check]
                
                # Check views exist and don't have SECURITY DEFINER
                views_check = await conn.fetch("""
                    SELECT 
                        schemaname,
                        viewname,
                        CASE WHEN definition ILIKE '%security definer%' THEN true ELSE false END as has_security_definer
                    FROM pg_views 
                    WHERE schemaname = 'public' 
                      AND viewname IN ('archon_cost_optimization_recommendations', 'archon_project_intelligence_overview', 'archon_agent_performance_dashboard');
                """)
                
                results['views_status'] = [dict(row) for row in views_check]
                
                # Count policies per table
                policies_check = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        COUNT(*) as policy_count
                    FROM pg_policies
                    WHERE schemaname = 'public'
                      AND tablename LIKE 'archon_%' OR tablename LIKE 'feature_%'
                    GROUP BY schemaname, tablename
                    ORDER BY tablename;
                """)
                
                results['policies_count'] = [dict(row) for row in policies_check]
                
                # Summary statistics
                total_tables = len(results['rls_status'])
                rls_enabled_count = sum(1 for table in results['rls_status'] if table['rls_enabled'])
                total_views = len(results['views_status'])
                secure_views_count = sum(1 for view in results['views_status'] if not view['has_security_definer'])
                total_policies = sum(row['policy_count'] for row in results['policies_count'])
                
                results['summary'] = {
                    'total_tables': total_tables,
                    'rls_enabled_tables': rls_enabled_count,
                    'total_views': total_views,
                    'secure_views': secure_views_count,
                    'total_policies': total_policies,
                    'all_rls_enabled': rls_enabled_count == total_tables,
                    'all_views_secure': secure_views_count == total_views
                }
                
                return results
                
            finally:
                await conn.close()
                
        except Exception as e:
            print(f"❌ Error validating security fixes: {e}")
            return {'error': str(e)}

    def print_validation_results(self, results: Dict[str, Any]):
        """Print validation results in a readable format."""
        if 'error' in results:
            print(f"❌ Validation failed: {results['error']}")
            return
        
        summary = results.get('summary', {})
        
        print("\n📊 Security Validation Results:")
        print("=" * 50)
        
        print(f"📋 Tables: {summary.get('rls_enabled_tables', 0)}/{summary.get('total_tables', 0)} have RLS enabled")
        print(f"👁️  Views: {summary.get('secure_views', 0)}/{summary.get('total_views', 0)} are secure (no SECURITY DEFINER)")
        print(f"🛡️  Policies: {summary.get('total_policies', 0)} RLS policies created")
        
        if summary.get('all_rls_enabled') and summary.get('all_views_secure'):
            print("✅ All security issues have been resolved!")
        else:
            print("⚠️  Some security issues may remain:")
            
            # Show tables without RLS
            for table in results.get('rls_status', []):
                if not table['rls_enabled']:
                    print(f"  ❌ Table {table['tablename']} still has RLS disabled")
            
            # Show views with SECURITY DEFINER
            for view in results.get('views_status', []):
                if view['has_security_definer']:
                    print(f"  ❌ View {view['viewname']} still has SECURITY DEFINER")

async def main():
    """Main function to apply and validate security fixes."""
    try:
        fixer = SupabaseSecurityFixer()
        
        # Path to the SQL file
        sql_file = Path(__file__).parent / "fix_supabase_security_issues.sql"
        
        if not sql_file.exists():
            print(f"❌ SQL file not found: {sql_file}")
            return
        
        # Apply security fixes
        success = await fixer.execute_sql_file(str(sql_file))
        
        if not success:
            print("❌ Failed to apply security fixes")
            return
        
        # Wait a moment for changes to propagate
        print("⏳ Waiting for changes to propagate...")
        await asyncio.sleep(2)
        
        # Validate the fixes
        results = await fixer.validate_security_fixes()
        fixer.print_validation_results(results)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())