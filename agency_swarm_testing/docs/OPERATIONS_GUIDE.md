# Agency Swarm Operations Guide

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Daily Operations](#daily-operations)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Performance Management](#performance-management)
- [Security Operations](#security-operations)
- [Incident Management](#incident-management)
- [Maintenance Procedures](#maintenance-procedures)
- [Scaling and Capacity Planning](#scaling-and-capacity-planning)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)
- [Troubleshooting](#troubleshooting)

## Overview

This operations guide provides comprehensive procedures for managing the Agency Swarm system in production environments. It covers daily operations, monitoring, maintenance, and troubleshooting procedures.

## System Architecture

### Component Overview
```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Application Gateway                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───┴────┐      ┌────┴────┐      ┌────┴────┐
│Frontend│      │   API    │      │   MCP    │
│Cluster │      │ Cluster  │      │ Cluster  │
└───┬────┘      └────┬────┘      └────┬────┘
    │                 │                 │
    └─────────────────┼─────────────────┘
                      │
            ┌───────────┴───────────┐
            │      Agents Cluster    │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │   Data & Services     │
            │  ┌─────────┬─────────┐ │
            │  │PostgreSQL│  Redis  │ │
            │  └─────────┴─────────┘ │
            └─────────────────────────┘
```

### Service Dependencies
- **Frontend** depends on **API** for data and authentication
- **API** depends on **Database**, **Redis**, and **MCP**
- **MCP** depends on **Database** and external services
- **Agents** depends on **API**, **Database**, and **Redis**
- **All services** depend on **Load Balancer** and **Monitoring**

## Daily Operations

### Morning Health Checks
```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Agency Swarm Daily Health Check ==="
echo "Time: $(date)"
echo ""

# Check service status
echo "1. Service Status:"
kubectl get pods -n archon --no-headers | while read pod status ready restarts age; do
  echo "  $pod: $status"
done

echo ""
echo "2. Resource Usage:"
kubectl top pods -n archon --sort-by=cpu

echo ""
echo "3. Database Health:"
kubectl exec -n archon postgres-0 -- pg_isready -U postgres

echo ""
echo "4. Redis Health:"
kubectl exec -n archon redis-xxxxx -- redis-cli ping

echo ""
echo "5. Application Endpoints:"
for endpoint in "http://archon-frontend-service:80/health" \
                "http://archon-api-service:8000/api/health" \
                "http://archon-mcp-service:8000/health" \
                "http://archon-agents-service:8000/health"; do
  response=$(curl -s -o /dev/null -w "%{http_code}" $endpoint 2>/dev/null)
  echo "  $endpoint: $response"
done
```

### Log Review
```bash
# Check error logs in the last 24 hours
kubectl logs -n archon -l app=archon-api --since=24h | grep ERROR

# Check for unusual patterns
kubectl logs -n archon -l app=archon-api --since=24h | grep -i "timeout\|error\|exception"

# Monitor authentication failures
kubectl logs -n archon -l app=archon-api --since=24h | grep -i "auth.*fail"
```

### Performance Monitoring
```bash
# Check response times
kubectl logs -n archon -l app=archon-api --since=1h | grep "response_time" | awk '{sum+=$NF; count++} END {print "Average response time:", sum/count "ms"}'

# Monitor database queries
kubectl logs -n archon -l app=archon-api --since=1h | grep "db_query" | awk '{print $NF}' | sort | uniq -c | sort -nr | head -10
```

## Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- **Response Time**: P50, P95, P99 percentiles
- **Error Rate**: HTTP 5xx errors, application exceptions
- **Throughput**: Requests per second
- **Memory Usage**: JVM heap, native memory
- **CPU Usage**: Application and system CPU

#### Database Metrics
- **Connection Pool**: Active connections, idle connections
- **Query Performance**: Slow queries, query execution time
- **Database Size**: Table sizes, index sizes
- **Replication Lag**: For read replicas

#### Infrastructure Metrics
- **Pod Status**: Running, pending, failed pods
- **Resource Usage**: CPU, memory, disk, network
- **Load Balancer**: Request count, response codes, backend health

### Grafana Dashboards

#### 1. System Overview Dashboard
- Total requests per minute
- Error rate percentage
- Average response time
- Active users
- System resource usage

#### 2. Application Performance Dashboard
- API response times by endpoint
- Database query performance
- Cache hit rates
- External API call performance

#### 3. Infrastructure Dashboard
- Pod resource usage
- Node resource usage
- Persistent volume usage
- Network traffic

### Alerting Rules

#### Critical Alerts
```yaml
# critical_alerts.yaml
groups:
- name: critical
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | printf "%.2f" }}% for 5 minutes"

  - alert: DatabaseDown
    expr: up{job="postgres"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database is down"
      description: "PostgreSQL database is not responding"

  - alert: HighMemoryUsage
    expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90% for 5 minutes"
```

#### Warning Alerts
```yaml
# warning_alerts.yaml
groups:
- name: warnings
  rules:
  - alert: SlowResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Slow response times detected"
      description: "95th percentile response time is {{ $value }} seconds"

  - alert: HighCPUsage
    expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is above 80% for 5 minutes"
```

## Performance Management

### Performance Tuning

#### Application Level
```bash
# Increase JVM heap size for API service
kubectl set env deployment/archon-api -n archon JAVA_OPTS="-Xmx2g -Xms2g"

# Adjust connection pool size
kubectl set env deployment/archon-api -n archon DB_MAX_CONNECTIONS=50

# Enable caching
kubectl set env deployment/archon-api -n archon CACHE_ENABLED=true
```

#### Database Level
```sql
-- Add indexes for frequently queried columns
CREATE INDEX CONCURRENTLY idx_knowledge_search_vector ON documents USING GIN(search_vector);

-- Optimize slow queries
EXPLAIN ANALYZE SELECT * FROM documents WHERE search_vector @@ to_tsquery('english', 'search term');

-- Update statistics
ANALYZE documents;
```

#### Infrastructure Level
```yaml
# Increase resource limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: archon-api
spec:
  template:
    spec:
      containers:
      - name: api
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
```

### Load Testing
```bash
# Run load test
python agency_swarm_testing/performance/load_testing.py --scenario knowledge_workload --users 100

# Monitor during load test
kubectl top pods -n archon --watch
kubectl logs -n archon -l app=archon-api -f
```

## Security Operations

### Security Monitoring
```bash
# Monitor authentication failures
kubectl logs -n archon -l app=archon-api --since=1h | grep "auth.*fail" | wc -l

# Check for unusual IP addresses
kubectl logs -n archon -l app=archon-api --since=1h | grep -oP '\d+\.\d+\.\d+\.\d+' | sort | uniq -c | sort -nr

# Monitor sensitive operations
kubectl logs -n archon -l app=archon-api --since=1h | grep -i "admin\|delete\|update.*password"
```

### Security Scans
```bash
# Run security vulnerability scan
trivy image --severity CRITICAL,HIGH ghcr.io/your-org/archon-api:latest

# Check for secrets in code
git-secrets --scan

# Run static code analysis
sonar-scanner -Dsonar.projectKey=agency-swarm -Dsonar.sources=python/src
```

### Access Control
```bash
# Review user permissions
kubectl get roles,rolebindings -n archon

# Audit admin access
kubectl auth can-i create deployments --as=system:serviceaccount:archon:admin -n archon

# Rotate secrets
kubectl create secret generic archon-secrets --from-env-file=.env.new -n archon --dry-run=client -o yaml | kubectl apply -f -
```

## Incident Management

### Incident Response Procedure

#### Phase 1: Detection
1. Alert triggers from monitoring system
2. Verify alert accuracy
3. Determine impact scope
4. Classify incident severity

#### Phase 2: Assessment
1. Gather diagnostic data
2. Identify root cause
3. Determine affected services
4. Estimate time to resolution

#### Phase 3: Resolution
1. Implement temporary fix if needed
2. Apply permanent solution
3. Verify resolution
4. Monitor for recurrence

#### Phase 4: Recovery
1. Restore normal operations
2. Monitor system stability
3. Communicate status updates
4. Document incident

### Incident Command System
```bash
# Create incident channel
curl -X POST -H 'Authorization: Bearer $SLACK_TOKEN' \
  -H 'Content-type: application/json' \
  --data '{"text":"🚨 Incident declared: API Service Outage\n\nJoin: #incident-api-outage"}' \
  https://slack.com/api/chat.postMessage

# Update incident status
curl -X POST -H 'Authorization: Bearer $SLACK_TOKEN' \
  -H 'Content-type: application/json' \
  --data '{"text":"📊 Status Update: Root cause identified, working on fix\n\nETA: 30 minutes"}' \
  https://slack.com/api/chat.postMessage
```

### Post-Incident Review
```markdown
# Incident Report: API Service Outage

## Summary
- **Date**: 2024-01-15 14:30-15:45 UTC
- **Duration**: 75 minutes
- **Impact**: All users unable to access API
- **Root Cause**: Database connection pool exhaustion

## Timeline
- 14:30: Alert triggered for high error rate
- 14:35: Incident declared
- 14:40: Root cause identified
- 15:00: Fix implemented
- 15:45: Service fully restored

## Root Cause
Database connection pool exhausted due to:
- Unbounded query execution
- Missing connection timeout
- Insufficient max connections

## Actions Taken
1. Increased max connections from 20 to 50
2. Added query timeout of 30 seconds
3. Implemented connection pool monitoring
4. Added circuit breaker pattern

## Prevention Measures
1. Implement connection pool monitoring
2. Add automated scaling based on connection usage
3. Improve query optimization
4. Add rate limiting
```

## Maintenance Procedures

### Rolling Updates
```bash
# Update API service with rolling update
kubectl set image deployment/archon-api -n archon api=ghcr.io/your-org/archon-api:v2.0.0

# Monitor rollout status
kubectl rollout status deployment/archon-api -n archon

# Rollback if issues
kubectl rollout undo deployment/archon-api -n archon
```

### Database Maintenance
```bash
# Database backup
kubectl exec -n archon postgres-0 -- pg_dump -U postgres archon > backup-$(date +%Y%m%d).sql

# Vacuum and analyze
kubectl exec -n archon postgres-0 -- vacuumdb -U postgres -z -v archon

# Update statistics
kubectl exec -n archon postgres-0 -- psql -U postgres -c "ANALYZE;"
```

### Certificate Rotation
```bash
# Generate new certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout tls.key -out tls.crt -subj "/CN=archon.yourdomain.com"

# Create Kubernetes secret
kubectl create secret tls archon-tls --cert=tls.crt --key=tls.key -n archon

# Update ingress
kubectl apply -f ingress.yaml
```

## Scaling and Capacity Planning

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: archon-api-hpa
  namespace: archon
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: archon-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: archon-api-vpa
  namespace: archon
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: archon-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: "api"
      minAllowed:
        cpu: "100m"
        memory: "256Mi"
      maxAllowed:
        cpu: "2"
        memory: "4Gi"
```

### Capacity Planning Metrics
- **User Growth**: Projected user count over 6-12 months
- **Data Growth**: Database size growth projections
- **API Volume**: Expected request volume increase
- **Storage Requirements**: Backup and log storage needs
- **Network Bandwidth**: Current and projected usage

## Backup and Disaster Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup_strategy.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
kubectl exec -n archon postgres-0 -- pg_dump -U postgres archon > $BACKUP_DIR/database.sql

# Configuration backup
kubectl get secrets,configmaps -n archon -o yaml > $BACKUP_DIR/config.yaml

# Persistent volume snapshots
kubectl get pv -o yaml > $BACKUP_DIR/persistent-volumes.yaml

# Upload to cloud storage
aws s3 cp $BACKUP_DIR s3://your-backup-bucket/agency-swarm/$DATE/ --recursive

echo "Backup completed: $BACKUP_DIR"
```

### Disaster Recovery Plan

#### Scenario 1: Single Service Failure
1. Identify failed service
2. Restart pods
3. Check logs for errors
4. Restore from backup if needed
5. Monitor for recurrence

#### Scenario 2: Database Failure
1. Switch to read replica
2. Restore primary from backup
3. Validate data consistency
4. Switch back to primary
5. Investigate root cause

#### Scenario 3: Complete Outage
1. Activate disaster recovery site
2. Restore from latest backup
3. Validate all services
4. Update DNS records
5. Monitor system performance

### Recovery Testing
```bash
# Simulate service failure
kubectl delete pod -l app=archon-api -n archon

# Verify automatic recovery
kubectl get pods -n archon -l app=archon-api

# Test backup restoration
kubectl exec -i -n archon postgres-0 -- psql -U postgres archon < backup-20240115.sql
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage
**Symptoms**: Pods being OOM killed, slow response times
**Solutions**:
```bash
# Check memory usage
kubectl top pods -n archon --sort-by=memory

# Increase memory limits
kubectl set resources deployment/archon-api -n archon --limits=memory=4Gi --requests=memory=2Gi

# Check for memory leaks
kubectl logs -n archon archon-api-xxxxx | grep -i "out.*of.*memory"
```

#### 2. Database Connection Issues
**Symptoms**: Connection timeouts, connection pool exhausted
**Solutions**:
```bash
# Check database connectivity
kubectl exec -n archon archon-api-xxxxx -- psql $DATABASE_URL -c "SELECT 1"

# Increase connection pool size
kubectl set env deployment/archon-api -n archon DB_MAX_CONNECTIONS=100

# Check slow queries
kubectl logs -n archon archon-api-xxxxx | grep "slow.*query"
```

#### 3. High CPU Usage
**Symptoms**: Slow response times, high load average
**Solutions**:
```bash
# Check CPU usage
kubectl top pods -n archon --sort-by=cpu

# Scale horizontally
kubectl scale deployment archon-api -n archon --replicas=5

# Check for CPU-intensive operations
kubectl logs -n archon archon-api-xxxxx | grep -i "cpu.*intensive"
```

#### 4. External API Failures
**Symptoms**: Integration failures, timeout errors
**Solutions**:
```bash
# Check external API status
curl -I https://api.openai.com/v1/models

# Implement circuit breaker
kubectl set env deployment/archon-api -n archon CIRCUIT_BREAKER_ENABLED=true

# Add retry logic
kubectl set env deployment/archon-api -n archon MAX_RETRIES=3
```

### Debug Commands
```bash
# Port forwarding for debugging
kubectl port-forward -n archon svc/archon-api-service 8181:8000

# Container shell access
kubectl exec -it -n archon archon-api-xxxxx -- /bin/bash

# Network connectivity test
kubectl exec -n archon archon-api-xxxxx -- curl -f http://archon-mcp-service:8000/health

# Resource usage monitoring
kubectl top pods -n archon --watch
```

### Performance Debugging
```bash
# Profile CPU usage
kubectl exec -n archon archon-api-xxxxx -- python -m cProfile -o profile.prof -m your_module

# Memory profiling
kubectl exec -n archon archon-api-xxxxx -- python -m memory_profiler your_module

# Thread dump
kubectl exec -n archon archon-api-xxxxx -- jstack <pid>
```

## Best Practices

### Monitoring
- Implement comprehensive monitoring with alerts
- Set up dashboards for key metrics
- Monitor both application and infrastructure
- Establish baseline metrics for comparison

### Security
- Regular security audits and vulnerability scanning
- Implement proper access controls
- Monitor for suspicious activities
- Keep systems patched and updated

### Performance
- Regular performance testing and optimization
- Monitor resource usage and scaling
- Optimize database queries and indexes
- Implement caching strategies

### Reliability
- Implement redundant systems where possible
- Regular backup and recovery testing
- Disaster recovery planning and testing
- Incident response procedures

### Documentation
- Maintain up-to-date documentation
- Document all changes and procedures
- Create runbooks for common issues
- Regular training for operations team

This operations guide provides a comprehensive reference for managing the Agency Swarm system in production environments. Regular review and updates to this guide are recommended to ensure it remains current and accurate.