#!/bin/bash
# Fix neo4j installation issue

echo "Starting neo4j fix..."

# Step 1: Create a temporary Dockerfile that just installs neo4j
cat > /tmp/Dockerfile.neo4j << 'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir neo4j>=5.15.0 pytz
EOF

# Step 2: Build this simple image
echo "Building neo4j base image..."
docker build -t archon-neo4j-base:latest -f /tmp/Dockerfile.neo4j /tmp

# Step 3: Start the current server container
echo "Starting server container..."
cd "/mnt/c/Jarvis/AI Workspace/Archon"
docker-compose up -d archon-server || true

# Step 4: Install neo4j in the running container
echo "Installing neo4j in container..."
docker exec archon-server pip install --no-cache-dir neo4j>=5.15.0 pytz

# Step 5: Commit the changes to a new image
echo "Committing changes to new image..."
docker commit archon-server archon-server-with-neo4j:latest

# Step 6: Tag the new image
docker tag archon-server-with-neo4j:latest archon-archon-server:latest

echo "Neo4j fix complete! Restart the server to use the fixed image."