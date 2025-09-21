#!/usr/bin/env python3
"""
Apply Supabase Security Fixes - Simple Version
Uses Supabase REST API to execute SQL commands.
"""

import os
import sys
import requests
from pathlib import Path

# Load environment variables  
from dotenv import load_dotenv
load_dotenv()

def apply_security_fixes():
    """Apply security fixes using Supabase REST API."""
    
    # Get environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not service_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file")
        return False
    
    print(f"Using Supabase project: {supabase_url}")
    
    # Read the SQL file
    sql_file = Path(__file__).parent / "fix_supabase_security_issues.sql"
    
    if not sql_file.exists():
        print(f"Error: SQL file not found: {sql_file}")
        return False
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    print("Applying security fixes...")
    
    # Split SQL into individual statements
    statements = []
    current_statement = ""
    
    for line in sql_content.split('\n'):
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('--'):
            continue
            
        current_statement += line + " "
        
        # End of statement
        if line.endswith(';'):
            statements.append(current_statement.strip())
            current_statement = ""
    
    # Execute each statement via Supabase REST API
    headers = {
        'apikey': service_key,
        'Authorization': f'Bearer {service_key}',
        'Content-Type': 'application/json'
    }
    
    successful_statements = 0
    failed_statements = 0
    
    for i, statement in enumerate(statements, 1):
        if not statement:
            continue
            
        try:
            print(f"  Executing statement {i}/{len(statements)}...")
            
            # Use Supabase's RPC function to execute raw SQL
            response = requests.post(
                f"{supabase_url}/rest/v1/rpc/exec_sql",
                headers=headers,
                json={"sql": statement}
            )
            
            if response.status_code in [200, 201, 204]:
                successful_statements += 1
            else:
                # Some statements might not work via REST API but that's okay for our fixes
                print(f"    Warning: Statement {i} returned {response.status_code}: {response.text[:100]}")
                failed_statements += 1
                
        except Exception as e:
            print(f"    Error on statement {i}: {e}")
            failed_statements += 1
    
    print(f"\nExecution Summary:")
    print(f"Successful: {successful_statements}")
    print(f"Failed/Skipped: {failed_statements}")
    
    # Since many statements might fail via REST API, let's also provide manual instructions
    print("\nManual Application Instructions:")
    print("=" * 50)
    print("1. Go to your Supabase Dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Copy and paste the contents of 'fix_supabase_security_issues.sql'")
    print("4. Execute the script manually")
    print("5. This will ensure all security fixes are properly applied")
    
    return True

if __name__ == "__main__":
    apply_security_fixes()