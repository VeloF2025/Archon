# Agency Swarm Enhancement - Administration Guide

This guide provides comprehensive instructions for system administrators to manage, monitor, and maintain the Agency Swarm enhancement infrastructure.

## 🎯 Overview

The Administration Guide covers system management, monitoring, security, maintenance, and operational procedures for the Agency Swarm enhancement.

## 📋 Table of Contents

- [System Architecture](#system-architecture)
- [Installation and Setup](#installation-and-setup)
- [System Monitoring](#system-monitoring)
- [Security Management](#security-management)
- [Performance Tuning](#performance-tuning)
- [Backup and Recovery](#backup-and-recovery)
- [Maintenance Procedures](#maintenance-procedures)
- [Troubleshooting](#troubleshooting)
- [Compliance and Auditing](#compliance-and-auditing)

## 🏗️ System Architecture

### Component Overview

The Agency Swarm system consists of the following components:

```mermaid
graph TB
    A[Load Balancer] --> B[Frontend Service]
    A --> C[Backend API Service]
    C --> D[Database Service]
    C --> E[MCP Service]
    C --> F[Agents Service]
    D --> G[(PostgreSQL)]
    E --> H[(Knowledge Base)]
    F --> I[Agent Pools]
    B --> J[WebSocket Service]
    J --> K[Real-time Updates]
```

### Service Dependencies

1. **Frontend Service** (Port 3737)
   - React + TypeScript application
   - Serves user interface
   - Handles WebSocket connections

2. **Backend API Service** (Port 8181)
   - FastAPI application
   - RESTful API endpoints
   - Authentication and authorization

3. **MCP Service** (Port 8051)
   - Model Context Protocol server
   - AI/ML operations
   - Knowledge management

4. **Agents Service** (Port 8052)
   - PydanticAI agents
   - Task processing
   - Agent lifecycle management

5. **Database Service**
   - PostgreSQL with pgvector
   - Data persistence
   - Vector embeddings storage

## 🚀 Installation and Setup

### Prerequisites

```bash
# System requirements
kubectl version --client # v1.25+
docker --version        # 20.10+
helm version            # v3.0+

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### Infrastructure Setup

1. **Create Namespace and Resources**
   ```bash
   # Create namespace
   kubectl create namespace agency-swarm

   # Create service account
   kubectl apply -f k8s/service-account.yaml

   # Set up RBAC
   kubectl apply -f k8s/rbac.yaml
   ```

2. **Configure Storage**
   ```bash
   # Create persistent volumes
   kubectl apply -f k8s/storage.yaml

   # Configure database storage
   kubectl apply -f k8s/database-storage.yaml
   ```

3. **Deploy Database**
   ```bash
   # Deploy PostgreSQL
   kubectl apply -f k8s/postgresql.yaml

   # Wait for database to be ready
   kubectl wait --for=condition=ready pod -l app=postgresql -n agency-swarm --timeout=300s

   # Initialize database schema
   kubectl apply -f k8s/database-init.yaml
   ```

4. **Deploy Application Services**
   ```bash
   # Deploy backend services
   kubectl apply -f k8s/backend.yaml

   # Deploy frontend
   kubectl apply -f k8s/frontend.yaml

   # Deploy monitoring
   kubectl apply -f k8s/monitoring.yaml
   ```

### Configuration Management

1. **Environment Variables**
   ```yaml
   # configmaps.yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: agency-swarm-config
     namespace: agency-swarm
   data:
     DATABASE_URL: "postgresql://user:pass@host:5432/dbname"
     REDIS_URL: "redis://redis:6379"
     LOG_LEVEL: "INFO"
     ENVIRONMENT: "production"
   ```

2. **Secrets Management**
   ```yaml
   # secrets.yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: agency-swarm-secrets
     namespace: agency-swarm
   type: Opaque
   data:
     SUPABASE_URL: <base64-encoded>
     SUPABASE_KEY: <base64-encoded>
     JWT_SECRET: <base64-encoded>
     ENCRYPTION_KEY: <base64-encoded>
   ```

## 📊 System Monitoring

### Monitoring Stack

1. **Prometheus** - Metrics collection
2. **Grafana** - Visualization and dashboards
3. **Alertmanager** - Alert management
4. **Loki** - Log aggregation

### Key Metrics to Monitor

```yaml
# System Metrics
cpu_usage: percentage
memory_usage: percentage
disk_usage: percentage
network_traffic: bytes/sec

# Application Metrics
api_response_time: milliseconds
database_query_time: milliseconds
websocket_connections: count
agent_performance: success_rate
task_completion_time: seconds

# Business Metrics
daily_active_users: count
tasks_completed: count
cost_per_task: currency
user_satisfaction: score
```

### Dashboard Setup

1. **Grafana Dashboards**
   ```bash
   # Import dashboards
   kubectl apply -f monitoring/dashboards/

   # Configure data sources
   kubectl apply -f monitoring/datasources/
   ```

2. **Alert Rules**
   ```yaml
   # alert-rules.yaml
   groups:
     - name: agency-swarm-alerts
       rules:
         - alert: HighCPUUsage
           expr: cpu_usage > 80
           for: 5m
           labels:
             severity: warning
           annotations:
             summary: "High CPU usage detected"
   ```

### Log Management

1. **Centralized Logging**
   ```bash
   # Deploy log aggregation
   kubectl apply -f monitoring/logging.yaml

   # Configure log shipping
   kubectl apply -f monitoring/log-shipping.yaml
   ```

2. **Log Analysis**
   ```bash
   # Access logs
   kubectl logs -f deployment/agency-swarm-backend -n agency-swarm

   # Filter logs by service
   kubectl logs -l app=agency-swarm-backend -n agency-swarm
   ```

## 🔒 Security Management

### Security Best Practices

1. **Network Security**
   ```yaml
   # Network policies
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: agency-swarm-network-policy
   spec:
     podSelector:
       matchLabels:
         app: agency-swarm
     policyTypes:
       - Ingress
       - Egress
     ingress:
       - from:
           - namespaceSelector:
               matchLabels:
                 name: agency-swarm
     egress:
       - to:
           - namespaceSelector:
               matchLabels:
                 name: agency-swarm
   ```

2. **Authentication and Authorization**
   ```python
   # JWT validation middleware
   async def validate_token(request: Request):
       token = request.headers.get("Authorization")
       if not token:
           raise HTTPException(status_code=401, detail="Token required")

       try:
           payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
           return payload
       except JWTError:
           raise HTTPException(status_code=401, detail="Invalid token")
   ```

3. **Data Encryption**
   ```python
   # Encryption utilities
   class DataEncryption:
       def __init__(self, key: str):
           self.cipher_suite = Fernet(key.encode())

       def encrypt(self, data: str) -> str:
           return self.cipher_suite.encrypt(data.encode()).decode()

       def decrypt(self, encrypted_data: str) -> str:
           return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
   ```

### Access Control

1. **Role-Based Access Control (RBAC)**
   ```yaml
   # rbac.yaml
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: agency-swarm-admin
     namespace: agency-swarm
   rules:
     - apiGroups: [""]
       resources: ["pods", "services", "configmaps"]
       verbs: ["get", "list", "watch", "create", "update", "delete"]
   ```

2. **User Permissions**
   ```python
   # Permission levels
   class PermissionLevel(Enum):
       READ_ONLY = "read_only"
       OPERATOR = "operator"
       ADMINISTRATOR = "administrator"

   # Permission check
   def check_permission(user_role: str, required_permission: PermissionLevel) -> bool:
       permission_hierarchy = {
           "read_only": [PermissionLevel.READ_ONLY],
           "operator": [PermissionLevel.READ_ONLY, PermissionLevel.OPERATOR],
           "administrator": [PermissionLevel.READ_ONLY, PermissionLevel.OPERATOR, PermissionLevel.ADMINISTRATOR]
       }
       return required_permission in permission_hierarchy.get(user_role, [])
   ```

### Security Auditing

1. **Audit Logging**
   ```python
   # Audit log configuration
   audit_logger = logging.getLogger("audit")

   def log_security_event(event_type: str, user: str, details: dict):
       audit_log = {
           "timestamp": datetime.utcnow().isoformat(),
           "event_type": event_type,
           "user": user,
           "details": details,
           "ip_address": request.client.host
       }
       audit_logger.info(json.dumps(audit_log))
   ```

2. **Compliance Monitoring**
   ```python
   # Compliance checks
   def check_compliance():
       checks = [
           check_data_encryption,
           check_access_controls,
           check_audit_logging,
           check_backup_procedures
       ]
       return all(check() for check in checks)
   ```

## ⚡ Performance Tuning

### Database Optimization

1. **Query Optimization**
   ```sql
   -- Add indexes for performance
   CREATE INDEX idx_agents_performance ON archon_agents_v3(model_tier, state, success_rate);
   CREATE INDEX idx_cost_tracking_date ON archon_cost_tracking(recorded_at);
   CREATE INDEX idx_knowledge_search ON archon_agent_knowledge USING gin(to_tsvector('english', content));

   -- Monitor query performance
   EXPLAIN ANALYZE SELECT * FROM archon_agents_v3 WHERE model_tier = 'SONNET';
   ```

2. **Connection Pooling**
   ```python
   # Database connection pool configuration
   database_config = {
       "database_url": settings.DATABASE_URL,
       "pool_size": 20,
       "max_overflow": 30,
       "pool_timeout": 30,
       "pool_recycle": 3600,
       "echo": settings.LOG_LEVEL == "DEBUG"
   }
   ```

### Application Performance

1. **Caching Strategy**
   ```python
   # Redis caching configuration
   redis_client = redis.Redis(
       host=settings.REDIS_HOST,
       port=settings.REDIS_PORT,
       db=0,
       decode_responses=True,
       socket_keepalive=True,
       socket_keepalive_options={}
   )

   # Cache decorator
   def cache_result(expiration: int = 3600):
       def decorator(func):
           async def wrapper(*args, **kwargs):
               cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
               cached_result = redis_client.get(cache_key)
               if cached_result:
                   return json.loads(cached_result)

               result = await func(*args, **kwargs)
               redis_client.setex(cache_key, expiration, json.dumps(result))
               return result
           return wrapper
       return decorator
   ```

2. **Load Balancing**
   ```yaml
   # Horizontal Pod Autoscaler
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: agency-swarm-hpa
     namespace: agency-swarm
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: agency-swarm-backend
     minReplicas: 3
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

### Resource Management

1. **Resource Limits**
   ```yaml
   # Resource quotas
   apiVersion: v1
   kind: ResourceQuota
   metadata:
     name: agency-swarm-quota
     namespace: agency-swarm
   spec:
     hard:
       requests.cpu: "10"
       requests.memory: 20Gi
       limits.cpu: "20"
       limits.memory: 40Gi
       persistentvolumeclaims: "10"
       requests.storage: "100Gi"
   ```

2. **Memory Management**
   ```python
   # Memory monitoring
   import psutil
   import gc

   def monitor_memory_usage():
       process = psutil.Process()
       memory_info = process.memory_info()

       if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
           gc.collect()
           log_warning("High memory usage detected, garbage collection triggered")
   ```

## 💾 Backup and Recovery

### Backup Strategy

1. **Database Backups**
   ```bash
   # Automated database backups
   kubectl apply -f backup/database-backup.yaml

   # Manual backup
   kubectl exec -it postgresql-pod -n agency-swarm -- pg_dump -U user dbname > backup.sql

   # Verify backup
   kubectl exec -it postgresql-pod -n agency-swarm -- psql -U user -d dbname -f backup.sql
   ```

2. **Configuration Backups**
   ```bash
   # Backup Kubernetes resources
   kubectl get all -n agency-swarm -o yaml > backup/cluster-backup.yaml
   kubectl get configmaps -n agency-swarm -o yaml > backup/configmaps.yaml
   kubectl get secrets -n agency-swarm -o yaml > backup/secrets.yaml
   ```

3. **File System Backups**
   ```bash
   # Backup persistent volumes
   kubectl cp agency-swarm/pod-name:/app/data backup/data/

   # Compress backups
   tar -czf backup/agency-swarm-backup-$(date +%Y%m%d).tar.gz backup/
   ```

### Recovery Procedures

1. **Database Recovery**
   ```bash
   # Restore from backup
   kubectl exec -i postgresql-pod -n agency-swarm -- psql -U user -d dbname < backup.sql

   # Point-in-time recovery
   kubectl exec -it postgresql-pod -n agency-swarm -- pg_restore -U user -d dbname backup.dump
   ```

2. **Service Recovery**
   ```bash
   # Restart services
   kubectl rollout restart deployment/agency-swarm-backend -n agency-swarm
   kubectl rollout restart deployment/agency-swarm-frontend -n agency-swarm

   # Check recovery status
   kubectl get pods -n agency-swarm -w
   ```

3. **Disaster Recovery**
   ```bash
   # Restore from off-site backup
   aws s3 cp s3://backup-bucket/agency-swarm-backup.tar.gz .
   tar -xzf agency-swarm-backup.tar.gz

   # Apply resources
   kubectl apply -f backup/cluster-backup.yaml
   ```

## 🔧 Maintenance Procedures

### Regular Maintenance Tasks

1. **Daily Tasks**
   ```bash
   # Check system health
   kubectl get pods -n agency-swarm
   kubectl top pods -n agency-swarm

   # Review logs
   kubectl logs --tail=100 -l app=agency-swarm-backend -n agency-swarm

   # Check alerts
   kubectl get prometheusrules -n monitoring
   ```

2. **Weekly Tasks**
   ```bash
   # Performance review
   kubectl top nodes
   kubectl describe hpa agency-swarm-hpa -n agency-swarm

   # Security audit
   kubectl auth can-i create deployments --namespace agency-swarm
   kubectl get networkpolicies -n agency-swarm

   # Backup verification
   kubectl exec -it postgresql-pod -n agency-swarm -- pg_dump --schema-only
   ```

3. **Monthly Tasks**
   ```bash
   # System updates
   kubectl set image deployment/agency-swarm-backend backend=new-image:tag -n agency-swarm

   # Performance tuning
   kubectl apply -f monitoring/performance-tuning.yaml

   # Capacity planning
   kubectl get nodes --show-labels
   kubectl describe resourcequotas -n agency-swarm
   ```

### System Updates

1. **Rolling Updates**
   ```bash
   # Update deployment
   kubectl set image deployment/agency-swarm-backend backend=agency-swarm/backend:v2.0.0 -n agency-swarm

   # Monitor update progress
   kubectl rollout status deployment/agency-swarm-backend -n agency-swarm

   # Rollback if needed
   kubectl rollout undo deployment/agency-swarm-backend -n agency-swarm
   ```

2. **Database Migrations**
   ```bash
   # Apply database migrations
   kubectl apply -f migrations/v2.0.0.yaml

   # Verify migration
   kubectl exec -it postgresql-pod -n agency-swarm -- psql -U user -d dbname -c "\dt"

   # Check data integrity
   kubectl exec -it postgresql-pod -n agency-swarm -- psql -U user -d dbname -c "SELECT COUNT(*) FROM archon_agents_v3;"
   ```

### Health Checks

1. **Liveness and Readiness Probes**
   ```yaml
   # Health check configuration
   livenessProbe:
     httpGet:
       path: /health
       port: 8181
     initialDelaySeconds: 30
     periodSeconds: 10

   readinessProbe:
     httpGet:
       path: /ready
       port: 8181
     initialDelaySeconds: 5
     periodSeconds: 5
   ```

2. **Custom Health Checks**
   ```python
   # Health check endpoint
   @app.get("/health")
   async def health_check():
       checks = {
           "database": await check_database(),
           "redis": await check_redis(),
           "mcp": await check_mcp(),
           "agents": await check_agents()
       }

       if all(checks.values()):
           return {"status": "healthy", "checks": checks}
       else:
           return {"status": "unhealthy", "checks": checks}, 500
   ```

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **Pod CrashLoopBackOff**
   ```bash
   # Check pod events
   kubectl describe pod <pod-name> -n agency-swarm

   # View previous logs
   kubectl logs <pod-name> -n agency-swarm --previous

   # Check resource limits
   kubectl describe deployment <deployment-name> -n agency-swarm
   ```

2. **Database Connection Issues**
   ```bash
   # Check database pod
   kubectl get pods -l app=postgresql -n agency-swarm

   # Test database connectivity
   kubectl exec -it <backend-pod> -n agency-swarm -- psql $DATABASE_URL -c "SELECT 1;"

   # Check database logs
   kubectl logs <postgresql-pod> -n agency-swarm
   ```

3. **High CPU Usage**
   ```bash
   # Identify high CPU pods
   kubectl top pods -n agency-swarm --sort-by=cpu

   # Check pod resource usage
   kubectl describe pod <pod-name> -n agency-swarm

   # Scale deployment
   kubectl scale deployment <deployment-name> --replicas=5 -n agency-swarm
   ```

4. **Memory Issues**
   ```bash
   # Check memory usage
   kubectl top pods -n agency-swarm --sort-by=memory

   # View memory limits
   kubectl get deployment <deployment-name> -o yaml -n agency-swarm | grep resources

   # Restart pod to clear memory
   kubectl delete pod <pod-name> -n agency-swarm
   ```

### Performance Debugging

1. **Slow Database Queries**
   ```bash
   # Enable query logging
   kubectl exec -it <postgresql-pod> -n agency-swarm -- psql -U user -d dbname -c "SET log_min_duration_statement = 1000;"

   # Monitor slow queries
   kubectl logs <postgresql-pod> -n agency-swarm -f | grep "duration:"

   # Analyze query plans
   kubectl exec -it <postgresql-pod> -n agency-swarm -- psql -U user -d dbname -c "EXPLAIN ANALYZE <query>;"
   ```

2. **API Performance Issues**
   ```bash
   # Check API response times
   kubectl exec -it <backend-pod> -n agency-swarm -- curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8181/health

   # Monitor API calls
   kubectl logs <backend-pod> -n agency-swarm -f | grep "API"
   ```

### Emergency Procedures

1. **System Outage Response**
   ```bash
   # 1. Assess the situation
   kubectl get pods -n agency-swarm

   # 2. Check critical services
   kubectl get services -n agency-swarm

   # 3. Restart critical services
   kubectl rollout restart deployment/agency-swarm-backend -n agency-swarm

   # 4. Monitor recovery
   kubectl get pods -n agency-swarm -w
   ```

2. **Security Incident Response**
   ```bash
   # 1. Isolate affected systems
   kubectl scale deployment <deployment-name> --replicas=0 -n agency-swarm

   # 2. Review audit logs
   kubectl logs <deployment-name> -n agency-swarm | grep "audit"

   # 3. Apply security patches
   kubectl set image deployment/<deployment-name> <container>=<patched-image> -n agency-swarm

   # 4. Restore services
   kubectl scale deployment <deployment-name> --replicas=3 -n agency-swarm
   ```

## 📋 Compliance and Auditing

### Compliance Requirements

1. **Data Protection**
   - GDPR compliance
   - Data encryption at rest and in transit
   - Regular security audits
   - Data retention policies

2. **System Auditing**
   - Access logs
   - Change management
   - Performance monitoring
   - Security incident tracking

### Audit Procedures

1. **Security Audits**
   ```bash
   # Run security scan
   kubectl apply -f security/security-scan.yaml

   # Review access controls
   kubectl get roles,rolebindings -n agency-swarm

   # Check network policies
   kubectl get networkpolicies -n agency-swarm
   ```

2. **Performance Audits**
   ```bash
   # Generate performance report
   kubectl top pods -n agency-swarm > performance-report.txt

   # Review resource usage
   kubectl describe resourcequotas -n agency-swarm

   # Check scaling events
   kubectl describe hpa -n agency-swarm
   ```

### Documentation Requirements

1. **System Documentation**
   - Architecture diagrams
   - Configuration details
   - Operational procedures
   - Contact information

2. **Compliance Documentation**
   - Security policies
   - Data handling procedures
   - Incident response plans
   - Audit trails

---

For additional support, refer to the [Deployment Guide](DEPLOYMENT_GUIDE.md) or contact the system administrator.