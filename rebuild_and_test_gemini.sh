#!/bin/bash
#
# Script to rebuild Docker containers and test Gemini CLI integration
#

set -e  # Exit on error

echo "================================================"
echo "🔧 REBUILDING ARCHON WITH GEMINI CLI INTEGRATION"
echo "================================================"

# Set environment variables if not already set
export ARCHON_SERVER_PORT=${ARCHON_SERVER_PORT:-8181}
export ARCHON_MCP_PORT=${ARCHON_MCP_PORT:-8051}
export ARCHON_UI_PORT=${ARCHON_UI_PORT:-3737}

echo ""
echo "📦 Step 1: Stopping existing containers..."
docker-compose down || true

echo ""
echo "🏗️ Step 2: Rebuilding containers with Gemini CLI..."
docker-compose build --no-cache archon-server

echo ""
echo "🚀 Step 3: Starting services..."
docker-compose up -d

echo ""
echo "⏳ Step 4: Waiting for services to be healthy..."
sleep 10

# Check if server is running
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:${ARCHON_SERVER_PORT}/health > /dev/null 2>&1; then
        echo "✅ Server is healthy!"
        break
    fi
    echo "   Waiting for server... (attempt $((attempt+1))/$max_attempts)"
    sleep 2
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Server failed to start. Check logs with: docker-compose logs archon-server"
    exit 1
fi

echo ""
echo "📋 Step 5: Checking Gemini CLI installation..."
docker-compose exec archon-server sh -c "which gemini || npm list -g @google-gemini/gemini-cli"

echo ""
echo "📊 Step 6: Running integration tests..."
echo "================================================"

# Run the test script
python3 test_gemini_cli_integration.py

echo ""
echo "================================================"
echo "🎉 Rebuild and test complete!"
echo "================================================"
echo ""
echo "📝 Useful commands:"
echo "   View logs:        docker-compose logs -f archon-server"
echo "   Check Gemini:     docker-compose exec archon-server gemini --version"
echo "   Test endpoint:    curl http://localhost:${ARCHON_SERVER_PORT}/api/gemini/usage-stats"
echo "   Stop services:    docker-compose down"
echo ""