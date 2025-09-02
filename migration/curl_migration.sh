#!/bin/bash

# Phase 7 DeepConf Database Migration using curl
# ==============================================

set -e  # Exit on any error

echo "🚀 Phase 7 DeepConf Database Migration (curl method)"
echo "===================================================="

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_KEY" ]; then
    echo "✗ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables"
    exit 1
fi

echo "📍 Target: ${SUPABASE_URL:0:50}..."

# Function to execute SQL via Supabase API
execute_sql() {
    local sql_content="$1"
    local description="$2"
    
    echo ""
    echo "🔄 $description..."
    
    # Escape SQL content for JSON
    local escaped_sql=$(echo "$sql_content" | jq -Rs .)
    
    # Create JSON payload
    local payload="{\"query\": $escaped_sql}"
    
    # Try multiple API endpoints that might work
    local endpoints=(
        "/rest/v1/rpc/sql"
        "/database/sql"
        "/sql"
    )
    
    local success=false
    
    for endpoint in "${endpoints[@]}"; do
        local full_url="${SUPABASE_URL}${endpoint}"
        
        # Execute SQL via API
        local response=$(curl -s -w "%{http_code}" \
            -X POST \
            -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
            -H "apikey: $SUPABASE_SERVICE_KEY" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "$full_url" 2>/dev/null)
        
        local http_code="${response: -3}"
        local body="${response%???}"
        
        if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
            echo "✓ $description completed successfully (endpoint: $endpoint)"
            success=true
            break
        elif [ "$http_code" = "404" ]; then
            continue  # Try next endpoint
        else
            echo "⚠ $description failed with HTTP $http_code at $endpoint"
            echo "   Response: $body"
        fi
    done
    
    if [ "$success" = false ]; then
        echo "✗ $description failed - all endpoints tried"
        return 1
    fi
    
    return 0
}

# Function to validate tables exist
validate_tables() {
    echo ""
    echo "🔍 Validating migration results..."
    
    local tables=("archon_confidence_scores" "archon_performance_metrics" "archon_confidence_calibration")
    local success_count=0
    
    for table in "${tables[@]}"; do
        local response=$(curl -s -w "%{http_code}" \
            -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
            -H "apikey: $SUPABASE_SERVICE_KEY" \
            "${SUPABASE_URL}/rest/v1/${table}?select=count" 2>/dev/null)
        
        local http_code="${response: -3}"
        
        if [ "$http_code" = "200" ]; then
            echo "  ✓ Table $table exists and is accessible"
            ((success_count++))
        else
            echo "  ✗ Table $table check failed (HTTP $http_code)"
        fi
    done
    
    echo "  Tables accessible: $success_count/${#tables[@]}"
    
    if [ $success_count -eq ${#tables[@]} ]; then
        return 0
    else
        return 1
    fi
}

# Test connection first
echo "Testing API connection..."
test_response=$(curl -s -w "%{http_code}" \
    -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
    -H "apikey: $SUPABASE_SERVICE_KEY" \
    "${SUPABASE_URL}/rest/v1/sources?select=count&limit=0" 2>/dev/null)

test_http_code="${test_response: -3}"

if [ "$test_http_code" = "200" ]; then
    echo "✓ Supabase API connection successful"
else
    echo "✗ Supabase API connection failed (HTTP $test_http_code)"
    echo "Response: ${test_response%???}"
    exit 1
fi

# Check if migration files exist
if [ ! -f "migration/prerequisite_functions.sql" ]; then
    echo "✗ Prerequisite file not found: migration/prerequisite_functions.sql"
    exit 1
fi

if [ ! -f "migration/phase7_deepconf_schema.sql" ]; then
    echo "✗ Main migration file not found: migration/phase7_deepconf_schema.sql"
    exit 1
fi

echo "✓ All migration files found"

# Read SQL files
prerequisite_sql=$(cat migration/prerequisite_functions.sql)
main_sql=$(cat migration/phase7_deepconf_schema.sql)

# Execute prerequisite functions
if execute_sql "$prerequisite_sql" "Installing prerequisite functions"; then
    echo "✓ Prerequisites installed"
else
    echo "⚠ Prerequisite installation failed, continuing anyway..."
fi

# Execute main migration
if execute_sql "$main_sql" "Executing Phase 7 DeepConf migration"; then
    echo "✓ Main migration executed"
else
    echo "⚠ Main migration had issues, continuing to validation..."
fi

# Validate migration
if validate_tables; then
    echo ""
    echo "🎉 Phase 7 DeepConf migration completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Test basic CRUD operations"
    echo "  2. Verify RLS policies"
    echo "  3. Update storage.py to use new tables"
    exit 0
else
    echo ""
    echo "⚠ Migration completed with issues"
    echo "  Please check the Supabase dashboard manually"
    exit 1
fi