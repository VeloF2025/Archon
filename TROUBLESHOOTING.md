# Agency Swarm Enhancement - Troubleshooting Guide

This guide provides comprehensive troubleshooting procedures for common issues encountered with the Agency Swarm enhancement.

## 📋 Table of Contents

- [Quick Reference](#quick-reference)
- [Common Issues](#common-issues)
- [Diagnostic Procedures](#diagnostic-procedures)
- [Performance Issues](#performance-issues)
- [Connectivity Issues](#connectivity-issues)
- [Database Issues](#database-issues)
- [Agent Issues](#agent-issues)
- [Security Issues](#security-issues)
- [Advanced Troubleshooting](#advanced-troubleshooting)
- [Contacting Support](#contacting-support)

## 🚀 Quick Reference

### Emergency Commands

```bash
# Check system status
kubectl get pods -n agency-swarm
kubectl get services -n agency-swarm
kubectl top pods -n agency-swarm

# Restart services
kubectl rollout restart deployment/agency-swarm-backend -n agency-swarm
kubectl rollout restart deployment/agency-swarm-frontend -n agency-swarm

# View logs
kubectl logs -f deployment/agency-swarm-backend -n agency-swarm
kubectl logs -f deployment/agency-swarm-frontend -n agency-swarm

# Check resource usage
kubectl top nodes
kubectl describe resourcequotas -n agency-swarm
```

### Health Check Endpoints

```bash
# Backend health
curl http://localhost:8181/health

# Frontend health
curl http://localhost:3737/health

# MCP service health
curl http://localhost:8051/health

# Agents service health
curl http://localhost:8052/health
```

## 🚨 Common Issues

### 1. Application Won't Start

#### Symptoms
- Pods in CrashLoopBackOff state
- Container image pull errors
- Resource limits exceeded

#### Solutions

**Pod CrashLoopBackOff**
```bash
# Check pod status
kubectl get pods -n agency-swarm

# Describe pod for events
kubectl describe pod <pod-name> -n agency-swarm

# View previous logs
kubectl logs <pod-name> -n agency-swarm --previous

# Check resource limits
kubectl describe deployment <deployment-name> -n agency-swarm | grep -A 10 -B 5 resources
```

**Image Pull Errors**
```bash
# Check image pull secrets
kubectl get secrets -n agency-swarm | grep image-pull

# Verify image exists in registry
docker pull <image-name>:<tag>

# Update image pull policy
kubectl set image deployment/<deployment-name> <container>=<image-name>:<tag> --image-pull-policy=Always -n agency-swarm
```

**Resource Limits**
```bash
# Check resource usage
kubectl top pods -n agency-swarm

# Update resource limits
kubectl edit deployment <deployment-name> -n agency-swarm

# Add more resources if needed
kubectl set resources deployment <deployment-name> --limits=cpu=2,memory=4Gi -n agency-swarm
```

### 2. Database Connection Issues

#### Symptoms
- Application logs show connection errors
- Database pods not starting
- Slow database queries

#### Solutions

**Connection Errors**
```bash
# Check database pod status
kubectl get pods -l app=postgresql -n agency-swarm

# Test database connectivity
kubectl exec -it <backend-pod> -n agency-swarm -- psql $DATABASE_URL -c "SELECT 1;"

# Check database logs
kubectl logs <postgresql-pod> -n agency-swarm

# Verify database configuration
kubectl get configmap database-config -n agency-swarm -o yaml
```

**Database Not Starting**
```bash
# Check persistent volume claims
kubectl get pvc -n agency-swarm

# Check persistent volumes
kubectl get pv -n agency-swarm

# Check storage class
kubectl get storageclass

# Restart database pod
kubectl delete pod <postgresql-pod> -n agency-swarm
```

### 3. Agent Communication Issues

#### Symptoms
- Agents not responding to messages
- Real-time collaboration not working
- Task handoffs failing

#### Solutions

**Agent Health**
```bash
# Check agent pod status
kubectl get pods -l app=agents -n agency-swarm

# Check agent logs
kubectl logs <agent-pod> -n agency-swarm

# Check agent service
kubectl get svc -n agency-swarm | grep agents

# Test agent connectivity
kubectl exec -it <backend-pod> -n agency-swarm -- curl http://agents-service:8052/health
```

**Message Routing**
```bash
# Check message queue status
kubectl exec -it <agent-pod> -n agency-swarm -- ps aux | grep message

# Check agent state
kubectl exec -it <agent-pod> -n agency-swarm -- curl http://localhost:8052/state

# Reset agent state
kubectl exec -it <agent-pod> -n agency-swarm -- curl -X POST http://localhost:8052/reset

# Restart agent
kubectl delete pod <agent-pod> -n agency-swarm
```

### 4. Performance Issues

#### Symptoms
- High CPU/Memory usage
- Slow API response times
- Agent timeouts

#### Solutions

**High CPU Usage**
```bash
# Identify high CPU pods
kubectl top pods -n agency-swarm --sort-by=cpu

# Scale deployment
kubectl scale deployment <deployment-name> --replicas=5 -n agency-swarm

# Set resource limits
kubectl set resources deployment <deployment-name> --limits=cpu=2,memory=4Gi --requests=cpu=1,memory=2Gi -n agency-swarm
```

**Memory Issues**
```bash
# Check memory usage
kubectl top pods -n agency-swarm --sort-by=memory

# Increase memory limits
kubectl set resources deployment <deployment-name> --limits=memory=8Gi --requests=memory=4Gi -n agency-swarm

# Add horizontal pod autoscaler
kubectl apply -f k8s/hpa.yaml
```

## 🔍 Diagnostic Procedures

### System Health Check

```bash
#!/bin/bash

# Agency Swarm System Health Check
echo "=== Agency Swarm System Health Check ==="
echo "Date: $(date)"
echo

# Check cluster connectivity
echo "1. Cluster Connectivity:"
kubectl cluster-info
echo

# Check namespace
echo "2. Namespace Status:"
kubectl get namespace agency-swarm
echo

# Check pods
echo "3. Pod Status:"
kubectl get pods -n agency-swarm -o wide
echo

# Check services
echo "4. Service Status:"
kubectl get svc -n agency-swarm
echo

# Check resource usage
echo "5. Resource Usage:"
kubectl top pods -n agency-swarm
echo

# Check persistent volumes
echo "6. Persistent Volume Claims:"
kubectl get pvc -n agency-swarm
echo

# Check events
echo "7. Recent Events:"
kubectl get events -n agency-swarm --sort-by=.metadata.creationTimestamp | tail -10
echo

echo "=== Health Check Complete ==="
```

### Application Health Check

```bash
#!/bin/bash

# Application Health Check
echo "=== Application Health Check ==="
echo "Date: $(date)"
echo

# Backend health
echo "1. Backend Health:"
curl -s http://localhost:8181/health | jq .
echo

# Frontend health
echo "2. Frontend Health:"
curl -s http://localhost:3737/health | jq .
echo

# MCP health
echo "3. MCP Health:"
curl -s http://localhost:8051/health | jq .
echo

# Agents health
echo "4. Agents Health:"
curl -s http://localhost:8052/health | jq .
echo

# Database health
echo "5. Database Health:"
kubectl exec -it <postgresql-pod> -n agency-swarm -- psql $DATABASE_URL -c "SELECT 1;"
echo

echo "=== Application Health Check Complete ==="
```

## 🐛 Common Issues & Solutions

### 1. Installation Issues

#### Docker Not Found / Permission Denied

**Symptoms**:
```
bash: docker: command not found
# OR
permission denied while trying to connect to Docker daemon
```

**Solutions**:
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# Linux: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker is running
docker --version
docker compose version
```

#### Port Already in Use

**Symptoms**:
```
Error: bind: address already in use
Port 3737 is already allocated
```

**Solutions**:
```bash
# Find what's using the port
lsof -i :3737
netstat -tulpn | grep :3737

# Kill the process
sudo kill -9 <PID>

# Or change ports in .env
ARCHON_UI_PORT=4000
ARCHON_SERVER_PORT=8282
```

#### Make Command Not Found

**Symptoms**:
```bash
bash: make: command not found
```

**Solutions**:
```bash
# Windows - Using Chocolatey
choco install make

# Windows - Using Scoop
scoop install make

# Windows - Using WSL2
wsl --install
# Then in WSL: sudo apt-get install make

# macOS (usually pre-installed)
brew install make

# Linux
sudo apt-get install make  # Ubuntu/Debian
sudo yum install make      # RHEL/CentOS
```

### 2. Configuration Issues

#### Supabase Connection Failed

**Symptoms**:
```
Failed to connect to Supabase
Authentication failed
Permission denied errors on save
```

**Solutions**:

1. **Verify Credentials**:
```bash
# Check .env file
cat .env | grep SUPABASE

# Ensure you're using SERVICE_ROLE key, not anon key
# Service role key is longer and contains "service_role" in JWT
```

2. **Test Connection**:
```bash
# Test Supabase connection
curl -H "Authorization: Bearer YOUR_SERVICE_KEY" \
     -H "apikey: YOUR_SERVICE_KEY" \
     "YOUR_SUPABASE_URL/rest/v1/"
```

3. **Check Database Setup**:
```sql
-- Run in Supabase SQL Editor
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name LIKE '%archon%';

-- Should show: sources, documents, projects, tasks, credentials, settings
```

#### API Keys Not Working

**Symptoms**:
```
OpenAI API error: Invalid API key
Model not found
Rate limit exceeded immediately
```

**Solutions**:
```bash
# Via UI (Recommended)
1. Go to http://localhost:3737/settings
2. Select API Keys tab
3. Choose provider (OpenAI/Gemini/Ollama)
4. Enter valid API key
5. Test connection

# Via Database (Advanced)
psql YOUR_SUPABASE_URL -c "
INSERT INTO credentials (key, value) 
VALUES ('OPENAI_API_KEY', 'sk-your-key-here') 
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
"
```

#### Environment Variables Not Loading

**Symptoms**:
```
Environment variable not found
Default ports being used despite .env configuration
Settings not persisting
```

**Solutions**:
```bash
# Verify .env file exists and is correct
ls -la .env
cat .env

# Check for typos and format
# Correct format (no spaces around =):
SUPABASE_URL=https://project.supabase.co
ARCHON_SERVER_PORT=8181

# Restart services after .env changes
docker compose down
docker compose --profile full up -d
```

### 3. Service Health Issues

#### Services Won't Start

**Symptoms**:
```
Container archon-server exited with code 1
Waiting for archon-server to become healthy
Health check failed
```

**Solutions**:
```bash
# Check specific service logs
docker compose logs archon-server
docker compose logs archon-mcp
docker compose logs archon-ui

# Common fixes:
1. Restart Docker Desktop
2. docker system prune -f
3. docker compose down --remove-orphans
4. docker compose --profile full up --build -d

# Check resource usage
docker stats
# Ensure adequate memory/CPU available
```

#### Health Checks Failing

**Symptoms**:
```
Health check timeout
Service marked as unhealthy
502 Bad Gateway errors
```

**Solutions**:
```bash
# Test health endpoints directly
curl -v http://localhost:8181/health
curl -v http://localhost:8051/health
curl -v http://localhost:3737

# Check container resource limits
docker inspect archon-server | grep -A 10 "Memory"

# Increase health check timeout in docker-compose.yml
healthcheck:
  timeout: 30s
  retries: 5
  start_period: 60s
```

#### Database Connection Issues

**Symptoms**:
```
Could not connect to database
SSL connection error
Connection timeout
```

**Solutions**:
```bash
# Test database connection
docker exec archon-server python -c "
import os
from supabase import create_client
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
print('Connection successful')
"

# Check firewall/network issues
ping your-project.supabase.co
nslookup your-project.supabase.co

# Verify SSL settings
curl -v https://your-project.supabase.co
```

### 4. UI/Frontend Issues

#### Frontend Won't Load

**Symptoms**:
```
This site can't be reached
Connection refused on port 3737
Blank white screen
```

**Solutions**:
```bash
# Check if UI service is running
docker compose ps | grep archon-ui

# Check UI logs
docker compose logs archon-ui

# Try hybrid development mode
make dev  # Frontend runs locally with hot reload

# Check port conflicts
lsof -i :3737
netstat -tulpn | grep :3737
```

#### API Connection Errors

**Symptoms**:
```
Network Error
Failed to fetch
CORS errors in browser console
```

**Solutions**:
```bash
# Check API server is accessible
curl http://localhost:8181/health

# Verify environment variables in UI
# Should point to correct API server
VITE_API_URL=http://localhost:8181

# Check browser network tab for failed requests
# Common fix: restart both UI and server
docker compose restart archon-ui archon-server
```

#### Real-time Updates Not Working

**Symptoms**:
```
Progress bars stuck at 0%
No live updates during crawling
WebSocket connection failed
```

**Solutions**:
```bash
# Check WebSocket connection in browser dev tools
# Network tab -> WS tab -> Should see socket.io connections

# Test WebSocket endpoint
curl -v http://localhost:8181/socket.io/

# Common fixes:
1. Disable browser ad blockers
2. Check firewall settings
3. Try different browser
4. Restart server: docker compose restart archon-server
```

### 5. Knowledge Base Issues

#### Crawling Fails

**Symptoms**:
```
Crawl stuck at 0%
SSL certificate errors
Robots.txt blocking
Timeout errors
```

**Solutions**:
```bash
# Test URL accessibility
curl -v https://target-site.com
wget --spider https://target-site.com

# Check crawler logs
docker compose logs archon-server | grep -i crawl

# Common fixes:
1. Use http:// instead of https:// if SSL issues
2. Check robots.txt: https://target-site.com/robots.txt
3. Increase timeout in settings
4. Try smaller batch sizes
```

#### Embeddings Generation Fails

**Symptoms**:
```
OpenAI embedding error
Vector dimension mismatch
Embedding service timeout
```

**Solutions**:
```bash
# Verify API key and model
curl -H \"Authorization: Bearer $OPENAI_API_KEY\" \\\n     -H \"Content-Type: application/json\" \\\n     -d '{\"input\":\"test\", \"model\":\"text-embedding-3-small\"}' \\\n     https://api.openai.com/v1/embeddings\n\n# Check embedding dimensions in settings\n# OpenAI text-embedding-3-small = 1536\n# Other models may vary\n\n# Reset embeddings if dimension mismatch\npsql YOUR_SUPABASE_URL -c \"DELETE FROM documents WHERE embedding IS NOT NULL;\"\n```\n\n#### Search Results Poor Quality\n\n**Symptoms**:\n```\nIrrelevant search results\nEmpty search results\nLow confidence scores\n```\n\n**Solutions**:\n```bash\n# Try different RAG strategies in Settings:\n1. Enable hybrid search\n2. Enable contextual embeddings\n3. Enable reranking (requires agents service)\n\n# Check if content was processed correctly\npsql YOUR_SUPABASE_URL -c \"SELECT COUNT(*) FROM documents;\"\npsql YOUR_SUPABASE_URL -c \"SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL;\"\n\n# Rebuild search index\n-- In Supabase SQL Editor:\nREINDEX INDEX documents_embedding_idx;\nREINDEX INDEX documents_content_search_idx;\n```\n\n### 6. MCP Integration Issues\n\n#### MCP Server Not Responding\n\n**Symptoms**:\n```\nMCP server connection failed\nTool execution timeout\nAI client can't connect\n```\n\n**Solutions**:\n```bash\n# Check MCP server status\ncurl http://localhost:8051/health\n\n# Test MCP tools\ncurl -X POST http://localhost:8051/tools/archon:perform_rag_query \\\n     -H \"Content-Type: application/json\" \\\n     -d '{\"query\": \"test\", \"match_count\": 5}'\n\n# Restart MCP server\ndocker compose restart archon-mcp\n```\n\n#### AI Client Configuration\n\n**Symptoms**:\n```\nClaude Code can't find Archon tools\nCursor MCP connection failed\nWindsurf integration not working\n```\n\n**Solutions**:\n```json\n// Claude Code config (~/.config/claude-code/mcp.json)\n{\n  \"mcpServers\": {\n    \"archon\": {\n      \"command\": \"curl\",\n      \"args\": [\n        \"-X\", \"POST\",\n        \"http://localhost:8051/tools/execute\",\n        \"-H\", \"Content-Type: application/json\"\n      ]\n    }\n  }\n}\n\n// Test MCP connection\ncurl -X GET http://localhost:8051/tools\n```\n\n### 7. Agent System Issues\n\n#### Agents Service Won't Start\n\n**Symptoms**:\n```\nContainer archon-agents not found\nAgents service health check failed\nPydanticAI import errors\n```\n\n**Solutions**:\n```bash\n# Agents service is opt-in, enable with:\ndocker compose --profile agents up -d\n\n# Or permanently enable\necho \"AGENTS_ENABLED=true\" >> .env\ndocker compose down\ndocker compose --profile full up -d\n\n# Check agents logs\ndocker compose logs archon-agents\n```\n\n#### DeepConf Scoring Issues\n\n**Symptoms**:\n```\nConfidence scores always 0.0\nDeepConf engine timeout\nUncertainty estimation failed\n```\n\n**Solutions**:\n```bash\n# Check if agents service is running\ncurl http://localhost:8052/health\n\n# Test DeepConf endpoint\ncurl -X POST http://localhost:8052/deepconf/score \\\n     -H \"Content-Type: application/json\" \\\n     -d '{\"content\": \"test code\", \"context\": \"test\"}'\n\n# Check DeepConf configuration\ndocker exec archon-agents python -c \"from src.agents.deepconf import engine; print('DeepConf loaded')\"\n```\n\n### 8. Performance Issues\n\n#### Slow Response Times\n\n**Symptoms**:\n```\nUI loading slowly\nAPI responses > 5 seconds\nSearch queries timing out\n```\n\n**Solutions**:\n```bash\n# Check resource usage\ndocker stats\nhtop  # Or top on macOS\n\n# Enable caching\necho \"CACHE_ENABLED=true\" >> .env\n\n# Optimize database\npsql YOUR_SUPABASE_URL -c \"VACUUM ANALYZE;\"\n\n# Check network latency to Supabase\nping your-project.supabase.co\n```\n\n#### High Memory Usage\n\n**Symptoms**:\n```\nSystem running out of memory\nDocker containers killed (OOMKilled)\nCrawling stops unexpectedly\n```\n\n**Solutions**:\n```bash\n# Monitor memory usage\ndocker stats --format \"table {{.Container}}\\t{{.MemUsage}}\\t{{.MemPerc}}\"\n\n# Reduce memory usage:\n1. Lower concurrent crawling in settings\n2. Reduce batch sizes\n3. Enable memory threshold limits\n4. Increase Docker memory limits\n\n# Docker Desktop: Settings > Resources > Memory\n# Increase to at least 8GB for full system\n```\n\n### 9. Development Issues\n\n#### Hot Reload Not Working\n\n**Symptoms**:\n```\nCode changes not reflected\nFrontend not updating\nBackend not restarting\n```\n\n**Solutions**:\n```bash\n# Use hybrid development mode\nmake dev  # Backend in Docker, frontend local\n\n# Check volume mounts in docker-compose.yml\n# Should include:\nvolumes:\n  - ./python/src:/app/src\n  - ./archon-ui-main/src:/app/src\n\n# For Windows/WSL issues:\n# Use WSL2 for development\n# Enable file watching in Docker Desktop settings\n```\n\n#### Build Failures\n\n**Symptoms**:\n```\nDocker build failed\nDependency installation errors\nTypeScript compilation errors\n```\n\n**Solutions**:\n```bash\n# Clean Docker cache\ndocker system prune -a -f\n\n# Rebuild without cache\ndocker compose build --no-cache\n\n# Check dependency versions\ncd archon-ui-main && npm audit fix\ncd python && uv sync\n\n# For TypeScript errors:\ncd archon-ui-main && npm run type-check\n```\n\n## 🔧 Advanced Diagnostics\n\n### System Information Collection\n\n```bash\n#!/bin/bash\n# Create diagnostic report\necho \"=== Archon Diagnostic Report ===\" > archon-diagnostic.txt\necho \"Date: $(date)\" >> archon-diagnostic.txt\necho \"\" >> archon-diagnostic.txt\n\necho \"=== System Info ===\" >> archon-diagnostic.txt\nuname -a >> archon-diagnostic.txt\ndocker --version >> archon-diagnostic.txt\ndocker compose version >> archon-diagnostic.txt\necho \"\" >> archon-diagnostic.txt\n\necho \"=== Container Status ===\" >> archon-diagnostic.txt\ndocker compose ps >> archon-diagnostic.txt\necho \"\" >> archon-diagnostic.txt\n\necho \"=== Service Logs (last 50 lines) ===\" >> archon-diagnostic.txt\ndocker compose logs --tail=50 >> archon-diagnostic.txt\n\necho \"=== Network Connectivity ===\" >> archon-diagnostic.txt\ncurl -I http://localhost:3737 >> archon-diagnostic.txt 2>&1\ncurl -I http://localhost:8181/health >> archon-diagnostic.txt 2>&1\necho \"\" >> archon-diagnostic.txt\n\necho \"Report saved to: archon-diagnostic.txt\"\n```\n\n### Database Diagnostics\n\n```sql\n-- Check database health\nSELECT \n  schemaname,\n  tablename,\n  n_tup_ins as inserts,\n  n_tup_upd as updates,\n  n_tup_del as deletes\nFROM pg_stat_user_tables \nWHERE schemaname = 'public';\n\n-- Check vector index health\nSELECT \n  schemaname,\n  tablename,\n  indexname,\n  idx_scan,\n  idx_tup_read,\n  idx_tup_fetch\nFROM pg_stat_user_indexes \nWHERE tablename = 'documents';\n\n-- Check storage usage\nSELECT \n  pg_size_pretty(pg_database_size(current_database())) as database_size,\n  pg_size_pretty(pg_total_relation_size('documents')) as documents_size,\n  pg_size_pretty(pg_total_relation_size('sources')) as sources_size;\n```\n\n### Performance Monitoring\n\n```bash\n# Monitor API performance\nwhile true; do\n  echo \"$(date): $(curl -w '%{time_total}s' -s -o /dev/null http://localhost:8181/health)\"\n  sleep 5\ndone\n\n# Monitor memory usage\nwatch \"docker stats --no-stream --format 'table {{.Container}}\\t{{.MemUsage}}\\t{{.MemPerc}}\\t{{.CPUPerc}}'\"\n\n# Monitor disk usage\nwatch \"df -h && echo '' && docker system df\"\n```\n\n## 🆘 Getting Help\n\n### Before Opening an Issue\n\n1. **Run diagnostics**: Use the diagnostic script above\n2. **Check logs**: Include relevant error messages\n3. **Search existing issues**: Your issue might already be reported\n4. **Try latest version**: Update to the latest release\n\n### Reporting Issues\n\nInclude this information:\n\n```markdown\n## Environment\n- OS: [Windows 11/macOS/Ubuntu 20.04]\n- Docker Version: [output of `docker --version`]\n- Archon Version: [git commit hash or release version]\n\n## Issue Description\n[Clear description of the problem]\n\n## Steps to Reproduce\n1. [Step 1]\n2. [Step 2]\n3. [Step 3]\n\n## Expected Behavior\n[What should happen]\n\n## Actual Behavior\n[What actually happens]\n\n## Logs\n```\n[Paste relevant logs here]\n```\n\n## Additional Context\n[Any other relevant information]\n```\n\n### Community Support\n\n- **GitHub Issues**: [Report bugs and feature requests](https://github.com/coleam00/archon/issues)\n- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/coleam00/archon/discussions)\n- **Discord**: [Real-time community support](https://discord.gg/archon) (if available)\n\n### Commercial Support\n\nFor enterprise deployments or priority support, contact the maintainers through the GitHub repository.\n\n---\n\n**Remember**: Most issues have simple solutions. Try the basic troubleshooting steps first, and don't hesitate to ask for help in our community!