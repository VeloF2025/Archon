#!/bin/bash
set -e

# Ensure neo4j package is installed (fix for build timeout issues)
echo "Checking for neo4j package..."
if ! python -c "import neo4j" 2>/dev/null; then
    echo "Installing neo4j package..."
    pip install --no-cache-dir neo4j>=5.15.0 pytz
    echo "neo4j package installed successfully"
else
    echo "neo4j package already installed"
fi

# Start Redis in background
echo "Starting Redis server..."
redis-server /etc/redis/redis.conf --daemonize yes

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
for i in {1..30}; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "Redis is ready!"
        break
    fi
    echo "Waiting for Redis... ($i/30)"
    sleep 1
done

# Start the main application
echo "Starting Archon server..."
exec python -m uvicorn src.server.main:socket_app --host 0.0.0.0 --port ${ARCHON_SERVER_PORT:-8181} --workers 1