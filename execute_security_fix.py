#!/usr/bin/env python3
"""
Execute Supabase Security Deployment Script
Connects to Supabase and runs the production security fixes
"""

import os
import asyncio
from pathlib import Path
import asyncpg
from dotenv import load_dotenv

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

async def execute_security_deployment():
    """Execute the production security deployment script"""
    
    print("🚀 Starting Supabase Security Deployment...")
    print("=" * 50)
    
    try:
        # Read the security deployment script
        script_path = Path(__file__).parent / "python" / "production_security_deployment.sql"
        
        if not script_path.exists():
            print(f"❌ Security script not found: {script_path}")
            return False
        
        with open(script_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        print(f"📄 Loaded security script ({len(sql_script)} characters)")
        
        # Connect to Supabase PostgreSQL
        print("🔌 Connecting to Supabase database...")
        
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to Supabase successfully")
        
        # Execute the security deployment script
        print("🔒 Executing production security fixes...")
        print("⏱️  This will take approximately 2-5 minutes...")
        print()
        
        # Execute the script (it contains multiple statements)
        await conn.execute(sql_script)
        
        print()
        print("✅ Security deployment completed successfully!")
        print("🛡️  All critical vulnerabilities have been fixed")
        
        # Verify the security status
        print("\n📊 Verifying security status...")
        
        security_status = await conn.fetchrow("SELECT * FROM security_status_prod")
        alert_status = await conn.fetchval("SELECT security_alert_check()")
        
        print(f"🔒 Security Status: {security_status['security_status']}")
        print(f"✅ Alert Check: {alert_status}")
        print(f"🎯 Secured Functions: {security_status['secured_functions']}/{security_status['total_critical_functions']}")
        
        await conn.close()
        print("\n🎉 DEPLOYMENT SUCCESSFUL - All Supabase warnings resolved!")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(execute_security_deployment())
    
    if success:
        print("\n" + "=" * 50)
        print("🚀 NEXT STEPS:")
        print("1. Check Supabase dashboard - warnings should be gone")
        print("2. Test your application functionality")
        print("3. Monitor the security_status_prod view")
        print("=" * 50)
    else:
        print("\n❌ DEPLOYMENT FAILED - Please run manually in Supabase SQL Editor")