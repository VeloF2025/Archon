# 🚨 Archon Disaster Recovery Runbook

## Overview
This runbook provides step-by-step procedures for recovering the Archon platform from various disaster scenarios.

## Recovery Time Objectives
- **RTO (Recovery Time Objective)**: < 1 hour
- **RPO (Recovery Point Objective)**: < 24 hours

## Disaster Scenarios

### 1. Database Failure

#### Symptoms
- API returns database connection errors
- Services unable to connect to PostgreSQL
- Data inconsistencies or corruption

#### Recovery Steps

##### 1.1 Immediate Response (5 minutes)
```bash
# Check database status
kubectl get pods -n archon-prod | grep postgres
kubectl logs -n archon-prod postgres-0 --tail=100

# Check Supabase status
curl https://status.supabase.com/api/v2/status.json
```

##### 1.2 Failover to Backup (15 minutes)
```bash
# Switch to read replica if available
kubectl patch service postgres-service -n archon-prod \
  -p '{"spec":{"selector":{"role":"replica"}}}'

# Update connection strings
kubectl set env deployment/archon-server \
  SUPABASE_URL=$BACKUP_DB_URL -n archon-prod
```

##### 1.3 Restore from Backup (30 minutes)
```bash
# List available backups
./scripts/backup/restore-database.sh -l -s

# Restore latest backup
./scripts/backup/restore-database.sh -s -t daily

# Verify restoration
psql $SUPABASE_DB_URL -c "SELECT COUNT(*) FROM sources;"
```

### 2. Service Outage

#### Symptoms
- One or more services unavailable
- Health checks failing
- Users unable to access application

#### Recovery Steps

##### 2.1 Identify Failed Services (5 minutes)
```bash
# Check all services
kubectl get pods -n archon-prod
kubectl get services -n archon-prod

# Check recent events
kubectl get events -n archon-prod --sort-by='.lastTimestamp'

# Check metrics
curl http://prometheus:9090/api/v1/query?query=up
```

##### 2.2 Restart Failed Services (10 minutes)
```bash
# Restart specific service
kubectl rollout restart deployment/archon-server -n archon-prod

# Scale up if needed
kubectl scale deployment/archon-server --replicas=5 -n archon-prod

# Force pod recreation
kubectl delete pod archon-server-xxx -n archon-prod
```

##### 2.3 Rollback if Necessary (15 minutes)
```bash
# Check deployment history
kubectl rollout history deployment/archon-server -n archon-prod

# Rollback to previous version
kubectl rollout undo deployment/archon-server -n archon-prod

# Rollback to specific revision
kubectl rollout undo deployment/archon-server --to-revision=2 -n archon-prod
```

### 3. Data Corruption

#### Symptoms
- Inconsistent data in UI
- API returning malformed responses
- Database constraint violations

#### Recovery Steps

##### 3.1 Isolate Corruption (10 minutes)
```bash
# Check database integrity
psql $SUPABASE_DB_URL << EOF
-- Check for orphaned records
SELECT * FROM tasks WHERE project_id NOT IN (SELECT id FROM projects);

-- Check for invalid embeddings
SELECT id FROM documents WHERE array_length(embedding, 1) != 1536;

-- Check for duplicate keys
SELECT id, COUNT(*) FROM sources GROUP BY id HAVING COUNT(*) > 1;
EOF
```

##### 3.2 Point-in-Time Recovery (30 minutes)
```bash
# Find backup before corruption
./scripts/backup/restore-database.sh -l -s

# Restore to specific point
./scripts/backup/restore-database.sh -d 20240115 -s -t daily

# Replay transactions if available
psql $SUPABASE_DB_URL < /var/log/postgresql/wal_archive/
```

##### 3.3 Data Repair (45 minutes)
```sql
-- Remove orphaned records
DELETE FROM tasks WHERE project_id NOT IN (SELECT id FROM projects);

-- Fix invalid embeddings
UPDATE documents SET embedding = NULL 
WHERE array_length(embedding, 1) != 1536;

-- Rebuild indexes
REINDEX DATABASE archon;

-- Update statistics
ANALYZE;
```

### 4. Complete Platform Failure

#### Symptoms
- All services down
- Infrastructure unavailable
- Total loss of primary region

#### Recovery Steps

##### 4.1 Activate DR Site (15 minutes)
```bash
# Switch DNS to DR region
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456 \
  --change-batch file://dr-dns-failover.json

# Deploy to backup cluster
kubectl config use-context dr-cluster
kubectl apply -f kubernetes/
```

##### 4.2 Restore Data (30 minutes)
```bash
# Download latest backup from S3
aws s3 cp s3://archon-backups/daily/latest.sql.gz /tmp/

# Restore to DR database
gunzip -c /tmp/latest.sql.gz | psql $DR_DATABASE_URL

# Verify data integrity
./scripts/verify-dr-data.sh
```

##### 4.3 Validate Services (15 minutes)
```bash
# Run smoke tests
npm run test:smoke

# Check all endpoints
for endpoint in health metrics api/knowledge api/projects; do
  curl -f https://dr.archon.example.com/$endpoint
done

# Monitor metrics
watch -n 5 'curl -s http://dr-prometheus:9090/api/v1/query?query=up | jq'
```

### 5. Security Breach

#### Symptoms
- Unauthorized access detected
- Data exfiltration alerts
- Suspicious API activity

#### Recovery Steps

##### 5.1 Immediate Containment (5 minutes)
```bash
# Rotate all API keys
kubectl delete secret archon-secrets -n archon-prod
kubectl create secret generic archon-secrets \
  --from-literal=supabase-key=$NEW_SUPABASE_KEY \
  --from-literal=openai-key=$NEW_OPENAI_KEY \
  -n archon-prod

# Block suspicious IPs
kubectl patch ingress archon-ingress -n archon-prod \
  --type='json' -p='[{"op": "add", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1whitelist-source-range", "value": "10.0.0.0/8"}]'

# Enable emergency mode
kubectl set env deployment/archon-server EMERGENCY_MODE=true -n archon-prod
```

##### 5.2 Audit and Investigation (20 minutes)
```bash
# Export audit logs
kubectl logs -n archon-prod -l app=archon-server --since=24h > audit.log

# Check for data access
grep -E "DELETE|UPDATE|INSERT" audit.log | grep -v "expected_user"

# Analyze access patterns
awk '/api\/knowledge/ {print $1}' audit.log | sort | uniq -c | sort -rn
```

##### 5.3 Recovery and Hardening (30 minutes)
```bash
# Reset all user sessions
redis-cli -h redis-service FLUSHALL

# Force re-authentication
kubectl set env deployment/archon-server FORCE_REAUTH=true -n archon-prod

# Deploy with enhanced security
kubectl apply -f kubernetes/security-enhanced/

# Enable additional monitoring
kubectl apply -f monitoring/security-alerts.yaml
```

## Monitoring and Alerts

### Critical Alerts Setup
```yaml
# Prometheus alerting rules
groups:
  - name: disaster_recovery
    rules:
      - alert: DatabaseDown
        expr: up{job="postgresql"} == 0
        for: 1m
        annotations:
          runbook: "https://docs.archon.io/runbooks/database-failure"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          runbook: "https://docs.archon.io/runbooks/service-outage"
      
      - alert: DataCorruption
        expr: data_integrity_check_failed > 0
        for: 1m
        annotations:
          runbook: "https://docs.archon.io/runbooks/data-corruption"
```

### Health Check Dashboard
```bash
# Quick health check script
#!/bin/bash
echo "=== Archon System Health Check ==="
echo "Database: $(psql $SUPABASE_DB_URL -c 'SELECT 1' &>/dev/null && echo 'OK' || echo 'FAIL')"
echo "API: $(curl -s -o /dev/null -w '%{http_code}' https://api.archon.io/health)"
echo "Frontend: $(curl -s -o /dev/null -w '%{http_code}' https://archon.io)"
echo "Redis: $(redis-cli ping 2>/dev/null || echo 'FAIL')"
echo "Backups: $(aws s3 ls s3://archon-backups/daily/ | tail -1)"
```

## Communication Plan

### Incident Response Team
1. **Incident Commander**: Overall coordination
2. **Technical Lead**: Technical resolution
3. **Communications Lead**: Stakeholder updates
4. **Operations Lead**: Infrastructure management

### Communication Channels
- **Primary**: Slack #incident-response
- **Backup**: incident@archon.io email list
- **Status Page**: https://status.archon.io
- **War Room**: Zoom meeting ID: 123-456-789

### Status Update Template
```markdown
**Incident Status Update**
Time: [TIMESTAMP]
Status: [Investigating/Identified/Monitoring/Resolved]
Impact: [Description of user impact]
Current Actions: [What we're doing]
Next Update: [Time of next update]
```

## Post-Incident Review

### Review Checklist
- [ ] Root cause identified
- [ ] Timeline documented
- [ ] Impact assessment completed
- [ ] Action items defined
- [ ] Runbook updated
- [ ] Monitoring improved
- [ ] Team debriefed

### Report Template
```markdown
# Incident Report: [INCIDENT_ID]

## Summary
- Date/Time: 
- Duration: 
- Impact: 
- Root Cause: 

## Timeline
- T+0: Incident detected
- T+X: [Action taken]
- T+Y: Service restored

## Lessons Learned
1. What went well
2. What could be improved
3. Action items

## Follow-up Actions
- [ ] Action item 1
- [ ] Action item 2
```

## Testing and Drills

### Monthly DR Test
```bash
# Test backup restoration
./scripts/backup/restore-database.sh --dry-run -d $(date -d "yesterday" +%Y%m%d)

# Test failover
kubectl apply -f kubernetes/dr-test.yaml

# Test monitoring
./scripts/test-alerts.sh
```

### Quarterly Full DR Drill
1. Simulate complete region failure
2. Execute full recovery procedure
3. Measure RTO/RPO
4. Document improvements

## Important Contacts

### Internal
- On-call Engineer: PagerDuty
- Engineering Manager: [Contact]
- CTO: [Contact]

### External
- Supabase Support: support@supabase.io
- AWS Support: [Account ID]
- CloudFlare Support: [Account ID]

## Recovery Tools

### Required Access
- [ ] Kubernetes cluster access
- [ ] AWS console access
- [ ] Database credentials
- [ ] Monitoring dashboards
- [ ] Communication channels

### Recovery Scripts Location
- Local: `/opt/archon/dr-tools/`
- S3: `s3://archon-dr-tools/`
- Git: `https://github.com/archon/dr-runbooks`

---

**Remember**: Stay calm, follow the runbook, communicate regularly, and document everything.