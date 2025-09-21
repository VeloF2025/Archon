# Neo4j Installation Issue - Docker Build Timeout

## Problem
The neo4j Python package installation causes Docker builds to timeout when building the archon-server image. This happens because the full requirements.server.txt file contains many dependencies that take a long time to install.

## Temporary Solution
The knowledge graph API routes that depend on neo4j have been temporarily disabled in `python/src/server/main.py`:
- Line 40: Import commented out
- Line 275: Router inclusion commented out

## Attempted Fixes
1. **Separate requirements file** - Created requirements.neo4j.txt but build still timed out
2. **Entrypoint script** - Added neo4j installation to entrypoint.sh but script wasn't executing properly
3. **Direct Dockerfile installation** - Added RUN pip install neo4j but build still timed out
4. **Manual container installation** - Works temporarily but doesn't persist

## Permanent Fix Options
1. **Multi-stage build optimization** - Split requirements into smaller chunks
2. **Pre-built base image** - Create a base image with all dependencies pre-installed
3. **Build cache optimization** - Use Docker buildkit cache mounts
4. **Separate neo4j service** - Run neo4j-dependent code in a separate container

## Current Status
- Server runs successfully without neo4j/knowledge graph features
- Other Phase 7 components are fully functional
- Phase 8 development can proceed

## To Re-enable Neo4j
1. Fix the Docker build timeout issue using one of the permanent fix options
2. Uncomment lines 40 and 275 in `python/src/server/main.py`
3. Rebuild and test the Docker image

## Commands for Testing
```bash
# Test if neo4j is installed in container
docker exec archon-server python -c "import neo4j; print('neo4j installed')"

# Manually install neo4j in running container (temporary)
docker exec archon-server pip install neo4j>=5.15.0

# Check server health
curl http://localhost:8181/health
```