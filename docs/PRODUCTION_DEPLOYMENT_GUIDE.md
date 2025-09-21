# 🚀 Archon Production Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Deployment Process](#deployment-process)
4. [Configuration Management](#configuration-management)
5. [Monitoring Setup](#monitoring-setup)
6. [Security Hardening](#security-hardening)
7. [Performance Optimization](#performance-optimization)
8. [Operational Procedures](#operational-procedures)

## Prerequisites

### Required Tools
```bash
# Install required CLI tools
brew install kubectl helm terraform aws-cli
brew install --cask docker

# Verify installations
kubectl version --client
helm version
terraform version
aws --version
docker --version
```

### Access Requirements
- [ ] Kubernetes cluster access
- [ ] AWS account with appropriate permissions
- [ ] Supabase project credentials
- [ ] Domain name and DNS control
- [ ] SSL certificates (or cert-manager configured)

## Infrastructure Setup

### 1. Kubernetes Cluster Setup

#### Option A: AWS EKS
```bash
# Create EKS cluster using eksctl
eksctl create cluster \
  --name archon-prod \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5 \
  --managed
```

#### Option B: Self-Managed Kubernetes
```bash
# Initialize cluster with kubeadm
kubeadm init --pod-network-cidr=10.244.0.0/16

# Install network plugin (Calico)
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

# Join worker nodes
kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash <hash>
```

### 2. Storage Setup
```bash
# Install EBS CSI driver for AWS
kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"

# Create storage class
kubectl apply -f kubernetes/storage-class.yaml
```

### 3. Ingress Controller
```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.metrics.enabled=true \
  --set controller.podAnnotations."prometheus\.io/scrape"=true
```

### 4. Certificate Manager
```bash
# Install cert-manager for automatic SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create cluster issuer for Let's Encrypt
kubectl apply -f kubernetes/cert-issuer.yaml
```

## Deployment Process

### 1. Create Namespace and Secrets
```bash
# Create production namespace
kubectl apply -f kubernetes/archon-namespace.yaml

# Create secrets
kubectl create secret generic archon-secrets \
  --from-literal=supabase-url=$SUPABASE_URL \
  --from-literal=supabase-key=$SUPABASE_SERVICE_KEY \
  --from-literal=openai-key=$OPENAI_API_KEY \
  -n archon-prod

# Create Docker registry secret (if using private registry)
kubectl create secret docker-registry regcred \
  --docker-server=<registry-server> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n archon-prod
```

### 2. Deploy Database Components
```bash
# Deploy Redis
kubectl apply -f kubernetes/redis-deployment.yaml

# Verify Redis is running
kubectl get pods -n archon-prod -l app=redis
kubectl logs -n archon-prod redis-0
```

### 3. Deploy Application Services
```bash
# Deploy main server
kubectl apply -f kubernetes/archon-server-deployment.yaml

# Deploy MCP server
kubectl apply -f kubernetes/archon-mcp-deployment.yaml

# Deploy frontend
kubectl apply -f kubernetes/archon-frontend-deployment.yaml

# Verify all services are running
kubectl get pods -n archon-prod
kubectl get services -n archon-prod
```

### 4. Configure Ingress
```bash
# Update domain in ingress.yaml
sed -i 's/archon.example.com/your-domain.com/g' kubernetes/ingress.yaml

# Apply ingress configuration
kubectl apply -f kubernetes/ingress.yaml

# Get ingress IP
kubectl get ingress -n archon-prod
```

### 5. Database Migrations
```bash
# Run database migrations
kubectl run migration --rm -it --image=archon/server:latest \
  --env="SUPABASE_URL=$SUPABASE_URL" \
  --env="SUPABASE_SERVICE_KEY=$SUPABASE_SERVICE_KEY" \
  -- python -m alembic upgrade head
```

## Configuration Management

### Environment Variables
```yaml
# ConfigMap for non-sensitive configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: archon-config
  namespace: archon-prod
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  ARCHON_SERVER_PORT: "8181"
  ARCHON_MCP_PORT: "8051"
  REDIS_URL: "redis://redis-service:6379"
```

### Feature Flags
```bash
# Enable/disable features via API
curl -X POST https://api.archon.com/api/feature-flags \
  -H "Content-Type: application/json" \
  -d '{
    "flag": "new_feature",
    "enabled": true,
    "rollout_percentage": 10
  }'
```

## Monitoring Setup

### 1. Deploy Prometheus
```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml
```

### 2. Deploy Grafana
```bash
# Grafana is included with kube-prometheus-stack
# Get admin password
kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Import dashboards
curl -X POST http://admin:password@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana-dashboards.json
```

### 3. Configure Alerts
```bash
# Apply alerting rules
kubectl apply -f monitoring/prometheus-config.yaml

# Configure AlertManager
kubectl apply -f monitoring/alertmanager-config.yaml
```

## Security Hardening

### 1. Network Policies
```yaml
# Restrict pod-to-pod communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: archon-network-policy
  namespace: archon-prod
spec:
  podSelector:
    matchLabels:
      app: archon-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: archon-frontend
    - podSelector:
        matchLabels:
          app: ingress-nginx
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
```

### 2. Pod Security Policies
```yaml
# Enforce security standards
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: archon-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 3. Secrets Management
```bash
# Use Sealed Secrets for GitOps
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Encrypt secrets
kubeseal --format yaml < secret.yaml > sealed-secret.yaml
```

### 4. RBAC Configuration
```yaml
# Create service account with minimal permissions
apiVersion: v1
kind: ServiceAccount
metadata:
  name: archon-sa
  namespace: archon-prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: archon-role
  namespace: archon-prod
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get"]
```

## Performance Optimization

### 1. Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: archon-server-hpa
  namespace: archon-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: archon-server
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

### 2. Vertical Pod Autoscaling
```bash
# Install VPA
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/latest/download/vertical-pod-autoscaler.yaml

# Create VPA for optimization recommendations
kubectl apply -f kubernetes/vpa.yaml
```

### 3. CDN Configuration
```bash
# Configure CloudFlare
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/pagerules" \
  -H "X-Auth-Email: email@example.com" \
  -H "X-Auth-Key: api_key" \
  -H "Content-Type: application/json" \
  --data '{
    "targets": [
      {
        "target": "url",
        "constraint": {
          "operator": "matches",
          "value": "*archon.com/static/*"
        }
      }
    ],
    "actions": [
      {
        "id": "cache_level",
        "value": "cache_everything"
      }
    ]
  }'
```

## Operational Procedures

### Daily Operations
```bash
# Check cluster health
kubectl get nodes
kubectl top nodes
kubectl get pods --all-namespaces | grep -v Running

# Check application logs
kubectl logs -n archon-prod -l app=archon-server --tail=100

# Check metrics
curl https://api.archon.com/metrics | grep -E "http_request|error_rate"
```

### Deployment Procedure
```bash
# 1. Build and push new image
docker build -t archon/server:v8.1.0 .
docker push archon/server:v8.1.0

# 2. Update deployment
kubectl set image deployment/archon-server \
  archon-server=archon/server:v8.1.0 \
  -n archon-prod

# 3. Monitor rollout
kubectl rollout status deployment/archon-server -n archon-prod

# 4. Verify deployment
kubectl get pods -n archon-prod -l app=archon-server
curl https://api.archon.com/health
```

### Backup Procedure
```bash
# Manual backup
./scripts/backup/backup-database.sh

# Verify backup
./scripts/backup/restore-database.sh --dry-run -d $(date +%Y%m%d)

# Schedule automated backups
kubectl apply -f kubernetes/backup-cronjob.yaml
```

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment/archon-server --replicas=5 -n archon-prod

# Check autoscaling status
kubectl get hpa -n archon-prod

# Adjust autoscaling parameters
kubectl edit hpa archon-server-hpa -n archon-prod
```

### Troubleshooting
```bash
# Debug pod issues
kubectl describe pod <pod-name> -n archon-prod
kubectl logs <pod-name> -n archon-prod --previous

# Execute commands in pod
kubectl exec -it <pod-name> -n archon-prod -- /bin/bash

# Check events
kubectl get events -n archon-prod --sort-by='.lastTimestamp'

# Port forward for debugging
kubectl port-forward -n archon-prod pod/<pod-name> 8181:8181
```

## Production Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Backup verified
- [ ] Rollback plan prepared

### Deployment
- [ ] Database migrations successful
- [ ] All pods running
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs aggregating properly
- [ ] SSL certificates valid

### Post-Deployment
- [ ] Smoke tests passing
- [ ] Performance metrics normal
- [ ] Error rates acceptable
- [ ] User acceptance verified
- [ ] Monitoring alerts configured
- [ ] Team notified

## Support and Maintenance

### Regular Maintenance
- **Daily**: Check logs, metrics, and alerts
- **Weekly**: Review performance trends, update dependencies
- **Monthly**: Security patches, backup tests, capacity planning
- **Quarterly**: Disaster recovery drill, architecture review

### Contact Information
- **On-Call**: PagerDuty rotation
- **Escalation**: Engineering Manager → CTO
- **External Support**: Supabase, AWS, CloudFlare

## Appendix

### Useful Commands
```bash
# Get all resources in namespace
kubectl get all -n archon-prod

# Describe all pods
kubectl describe pods -n archon-prod

# Get pod logs with timestamps
kubectl logs -n archon-prod <pod> --timestamps=true

# Watch pod status
kubectl get pods -n archon-prod -w

# Get resource usage
kubectl top pods -n archon-prod

# Export current configuration
kubectl get deploy,svc,ingress,cm,secret -n archon-prod -o yaml > backup.yaml
```

### Environment-Specific Configurations
- **Development**: Single node, minimal resources, debug logging
- **Staging**: Multi-node, production-like, feature flags enabled
- **Production**: Full HA, monitoring, backups, security hardened

---

**Remember**: Always test in staging before deploying to production!