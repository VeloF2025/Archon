# Archon Phase 6 Authentication Infrastructure

Production-ready CI/CD pipeline, containerization, and infrastructure as code for the Archon authentication system.

## 🏗️ Infrastructure Overview

### Architecture Components

- **GitHub Actions**: Comprehensive CI/CD pipelines with security scanning
- **Docker**: Multi-stage production builds with security hardening
- **Kubernetes**: Auto-scaling deployments with security policies
- **Terraform**: Infrastructure as Code for AWS EKS clusters
- **Prometheus/Grafana**: Full observability stack with SLO monitoring
- **Blue-Green Deployments**: Zero-downtime deployments with automated rollback

### Security Features

- **Multi-layer scanning**: SAST, dependency, container, and IaC security scans
- **Runtime security**: Falco monitoring with real-time threat detection
- **Secret management**: External Secrets Operator with HashiCorp Vault
- **Network policies**: Kubernetes network segmentation and isolation
- **RBAC**: Principle of least privilege access controls

## 🚀 Quick Start

### Prerequisites

```bash
# Install required tools
kubectl version --client
docker --version
helm version
terraform --version
aws --version
```

### Local Development

```bash
# Start development environment
docker-compose up -d

# Run with agents enabled
docker-compose --profile agents up -d

# Run production configuration
docker-compose -f docker-compose.production.yml up -d
```

### Production Deployment

```bash
# Deploy to staging
ENVIRONMENT=staging IMAGE_TAG=v1.0.0 ./infrastructure/scripts/deployment-automation.sh deploy

# Deploy to production with blue-green strategy
ENVIRONMENT=production IMAGE_TAG=v1.0.0 DEPLOYMENT_STRATEGY=blue-green ./infrastructure/scripts/deployment-automation.sh deploy

# Emergency rollback
./infrastructure/scripts/rollback-procedures.sh rollback immediate "High error rate detected"
```

## 📁 Directory Structure

```
infrastructure/
├── terraform/                 # Infrastructure as Code
│   ├── main.tf                # Main Terraform configuration
│   ├── variables.tf           # Input variables
│   └── helm.tf               # Helm charts and add-ons
├── kubernetes/               # Kubernetes manifests
│   ├── manifests/           # Raw Kubernetes YAML
│   └── helm/               # Helm chart templates
├── monitoring/              # Observability configuration
│   ├── prometheus.yml      # Prometheus configuration
│   ├── alert-rules.yml     # Alerting rules
│   └── grafana/           # Grafana dashboards
├── scripts/                # Automation scripts
│   ├── deployment-automation.sh      # Main deployment orchestrator
│   ├── blue-green-deploy.sh         # Blue-green deployment
│   ├── rollback-procedures.sh       # Emergency rollback
│   ├── generate-security-report.py  # Security report generator
│   └── check-security-thresholds.py # Security gate checker
└── README.md               # This file
```

## 🔐 Security Pipeline

### Automated Security Scans

The security pipeline runs on every commit and includes:

1. **SAST (Static Application Security Testing)**
   - Python: Bandit, Safety, Semgrep
   - TypeScript: ESLint security rules, npm audit

2. **Container Security**
   - Trivy vulnerability scanning
   - Grype container analysis
   - SBOM generation with Syft

3. **Infrastructure Security**
   - Terraform: TFSec analysis
   - Kubernetes: Checkov policy validation
   - Docker: Hadolint best practices

4. **Dependency Scanning**
   - Python: Safety, pip-audit
   - Node.js: npm audit, Snyk (if configured)

5. **Secrets Detection**
   - GitLeaks historical scan
   - TruffleHog verified secrets

### Security Thresholds

| Severity | Threshold | Action |
|----------|-----------|--------|
| Critical | 0 | Block deployment |
| High | ≤ 5 | Allow with approval |
| Medium | ≤ 20 | Allow with warning |
| Low | ≤ 50 | Allow (informational) |

### Security Commands

```bash
# Run full security scan
.github/workflows/security-scan.yml

# Generate security report
./infrastructure/scripts/generate-security-report.py --input-dir security-reports/ --output security-summary

# Check security thresholds
./infrastructure/scripts/check-security-thresholds.py --report security-summary.json --fail-on-critical --max-high 5
```

## 🐳 Container Strategy

### Multi-Stage Docker Builds

Our production Dockerfiles implement security best practices:

- **Minimal base images**: Alpine and distroless images
- **Non-root execution**: All containers run as unprivileged users
- **Layer optimization**: Efficient caching and minimal image size
- **Security scanning**: Built-in vulnerability detection
- **Health checks**: Comprehensive container health monitoring

### Container Security Features

- Read-only root filesystems
- Capability dropping (remove ALL capabilities)
- Security contexts with runAsNonRoot
- Resource limits and requests
- Network policies for isolation

## ☸️ Kubernetes Deployment

### Production Features

- **High Availability**: Multi-replica deployments with pod anti-affinity
- **Auto-scaling**: HPA based on CPU/memory with custom metrics
- **Zero Downtime**: Rolling updates and blue-green deployments
- **Security**: Network policies, RBAC, pod security standards
- **Monitoring**: Prometheus metrics and Grafana dashboards

### Deployment Strategies

#### Blue-Green Deployment (Default for Production)

```bash
# Automated blue-green deployment
./infrastructure/scripts/blue-green-deploy.sh deploy

# Check deployment status
./infrastructure/scripts/blue-green-deploy.sh status

# Emergency rollback
./infrastructure/scripts/blue-green-deploy.sh rollback
```

#### Rolling Deployment (Staging)

```bash
# Rolling update deployment
DEPLOYMENT_STRATEGY=rolling ./infrastructure/scripts/deployment-automation.sh deploy
```

### Rollback Procedures

#### Immediate Rollback (< 30 seconds)
```bash
./infrastructure/scripts/rollback-procedures.sh rollback immediate "High error rate"
```

#### Gradual Rollback (Canary rollback)
```bash
./infrastructure/scripts/rollback-procedures.sh rollback gradual "Performance issues"
```

#### Emergency Circuit Breaker
```bash
./infrastructure/scripts/rollback-procedures.sh circuit-breaker
```

## 🏗️ Infrastructure as Code

### Terraform Configuration

The Terraform setup provisions:

- **EKS Cluster**: Managed Kubernetes with auto-scaling node groups
- **VPC**: Secure networking with public/private subnets
- **RDS PostgreSQL**: High-availability database with pgvector
- **ElastiCache Redis**: Distributed caching layer
- **Application Load Balancer**: SSL termination and traffic routing
- **IAM Roles**: Least-privilege access controls
- **KMS**: Encryption key management
- **S3 Buckets**: Log storage and backups

### Environment Management

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan infrastructure changes
terraform plan -var="environment=staging"

# Apply infrastructure
terraform apply -var="environment=production" -var="db_instance_class=db.r6g.large"

# Destroy infrastructure (staging only)
terraform destroy -var="environment=staging"
```

### Terraform Variables

Key variables for environment configuration:

```hcl
# Required variables
environment = "production"  # production, staging, development
aws_region = "us-west-2"

# Database configuration
db_instance_class = "db.t3.medium"
db_allocated_storage = 100

# Redis configuration
redis_node_type = "cache.t4g.medium"
redis_num_cache_clusters = 3

# Feature flags
enable_monitoring = true
enable_security_scanning = true
auto_scaling_enabled = true
```

## 📊 Monitoring and Alerting

### Metrics Collection

- **Application Metrics**: Request rates, error rates, latency (RED metrics)
- **Infrastructure Metrics**: CPU, memory, disk, network usage
- **Business Metrics**: Authentication rates, user registrations
- **Security Metrics**: Failed login attempts, suspicious activities

### Alert Rules

Critical alerts that trigger immediate response:

- **Service Availability**: Any service down > 30 seconds
- **High Error Rate**: >5% errors over 2 minutes
- **Critical Latency**: >3s response time (95th percentile)
- **Security Events**: Brute force attacks, suspicious patterns
- **Infrastructure**: Node failures, resource exhaustion

### SLO Monitoring

Service Level Objectives with error budget tracking:

- **Availability**: 99.9% uptime (43.2 minutes downtime/month)
- **Latency**: 95th percentile < 1.5 seconds
- **Error Rate**: < 0.1% of all requests
- **Security**: < 0.01% authentication failures

### Dashboard Access

- **Grafana**: https://admin.archon.com/grafana
- **Prometheus**: https://admin.archon.com/prometheus
- **AlertManager**: Integrated with Slack/Teams notifications

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### Main CI/CD Pipeline (`ci-cd-auth.yml`)

Triggered on: Push to main/develop, Pull Requests

Stages:
1. **Security Scan**: SAST, dependency, container scanning
2. **Frontend Tests**: TypeScript, ESLint, Prettier, unit tests
3. **Backend Tests**: Python, MyPy, Ruff, pytest with coverage
4. **Integration Tests**: Docker Compose with health checks
5. **Build Images**: Multi-stage Docker builds with caching
6. **Deploy Staging**: Automated staging deployment
7. **Deploy Production**: Blue-green production deployment with approval

#### Security Pipeline (`security-scan.yml`)

Triggered on: Push, PR, Daily schedule (2 AM UTC)

Scans:
- **SAST**: Bandit, Semgrep, ESLint security rules
- **Dependencies**: Safety, npm audit, Snyk
- **Containers**: Trivy, Grype with SBOM generation
- **Infrastructure**: Checkov, TFSec, OPA policy validation
- **Secrets**: GitLeaks, TruffleHog
- **Compliance**: License scanning, policy validation

### Pipeline Commands

```bash
# Trigger deployment manually
gh workflow run ci-cd-auth.yml -f environment=staging -f image_tag=v1.2.3

# Trigger security scan
gh workflow run security-scan.yml

# Monitor pipeline status
gh run list --workflow=ci-cd-auth.yml

# Download artifacts
gh run download <run-id> --name security-reports
```

## 🛠️ Operations

### Daily Operations

```bash
# Check system health
kubectl get pods -n archon-production
./infrastructure/scripts/blue-green-deploy.sh status

# View logs
kubectl logs -f deployment/archon-auth-server -n archon-production

# Check metrics
curl https://api.archon.com/metrics
```

### Maintenance

```bash
# Update dependencies
cd archon-ui-main && npm update
cd python && uv sync --upgrade

# Security updates
./infrastructure/scripts/deployment-automation.sh security-scan

# Database maintenance
kubectl exec -n archon-production deployment/postgres -- pg_dump archon > backup.sql
```

### Troubleshooting

```bash
# Check deployment status
./infrastructure/scripts/blue-green-deploy.sh status

# View recent events
kubectl get events -n archon-production --sort-by='.lastTimestamp'

# Debug pod issues
kubectl describe pod <pod-name> -n archon-production

# Check resource usage
kubectl top pods -n archon-production
```

## 📈 Performance Optimization

### Resource Limits

Production resource allocation:

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|-------------|-----------|----------------|--------------|
| Server | 500m | 2000m | 1Gi | 4Gi |
| MCP | 250m | 1000m | 256Mi | 1Gi |
| Agents | 500m | 2000m | 1Gi | 4Gi |
| Validator | 250m | 1000m | 512Mi | 2Gi |
| Frontend | 100m | 500m | 128Mi | 512Mi |

### Auto-scaling Configuration

- **HPA**: CPU (70%) and Memory (80%) thresholds
- **Cluster Autoscaler**: Node scaling based on pod resource requests
- **Vertical Pod Autoscaler**: Automatic resource limit recommendations

## 🔒 Security Hardening

### Network Security

- **Network Policies**: Restrict pod-to-pod communication
- **Ingress Security**: Rate limiting, WAF integration
- **TLS Everywhere**: End-to-end encryption with cert-manager
- **VPC Security**: Private subnets with NAT gateways

### Access Control

- **RBAC**: Kubernetes role-based access control
- **Service Accounts**: Minimal permission service accounts
- **AWS IAM**: Instance profiles and cross-service access
- **Secret Management**: External Secrets Operator with rotation

### Compliance

- **PCI DSS**: Payment card industry standards
- **SOC 2**: Security, availability, and confidentiality
- **GDPR**: Data protection and privacy compliance
- **HIPAA**: Healthcare information protection (if applicable)

## 📋 Runbooks

### Incident Response

1. **Alert Received**: Check Grafana dashboards and Prometheus alerts
2. **Initial Assessment**: Determine impact and severity
3. **Immediate Actions**: Scale up, rollback, or circuit breaker
4. **Investigation**: Analyze logs, metrics, and traces
5. **Resolution**: Apply fixes and verify resolution
6. **Post-mortem**: Document learnings and preventive measures

### Common Issues

#### High Error Rate
```bash
# Check error metrics
kubectl exec -n archon-monitoring deployment/prometheus-server -- \
  promtool query instant 'rate(archon_http_requests_total{status=~"5.."}[5m])'

# Immediate rollback if critical
./infrastructure/scripts/rollback-procedures.sh rollback immediate "High error rate"
```

#### Database Issues
```bash
# Check database connectivity
kubectl exec -n archon-production deployment/archon-auth-server -- \
  python -c "import psycopg2; print('DB OK')"

# Check connection pool
kubectl exec -n archon-production deployment/postgres -- \
  psql -U archon -c "SELECT count(*) FROM pg_stat_activity;"
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n archon-production

# Scale up if needed
kubectl scale deployment archon-auth-server --replicas=5 -n archon-production

# Check latency metrics
curl -s https://api.archon.com/metrics | grep http_request_duration
```

## 📞 Support and Contacts

### Team Contacts

- **DevOps Team**: devops@archon.com
- **Security Team**: security@archon.com
- **On-call Engineer**: oncall@archon.com

### Emergency Procedures

1. **Critical Production Issue**: Execute immediate rollback
2. **Security Incident**: Activate circuit breaker, notify security team
3. **Data Loss**: Stop all writes, activate disaster recovery
4. **Service Unavailable**: Check monitoring, scale resources, rollback if needed

### Monitoring URLs

- **Production Dashboard**: https://admin.archon.com/grafana
- **Staging Dashboard**: https://staging-admin.archon.com/grafana
- **Alert Manager**: https://admin.archon.com/alertmanager
- **Prometheus**: https://admin.archon.com/prometheus

---

## 🔧 Development Workflow

### Feature Development

1. **Create Feature Branch**: `git checkout -b feature/auth-enhancement`
2. **Local Testing**: `docker-compose up -d` for development
3. **Security Scan**: Automated on PR creation
4. **Staging Deploy**: Automatic on merge to develop
5. **Production Deploy**: Manual approval required

### Hotfix Workflow

1. **Create Hotfix Branch**: `git checkout -b hotfix/critical-auth-fix`
2. **Emergency Testing**: Fast-track testing with reduced test suite
3. **Security Validation**: Critical security scan only
4. **Production Deploy**: Immediate blue-green deployment
5. **Monitoring**: Enhanced monitoring for 24 hours

### Database Migrations

```bash
# Development
cd python && uv run alembic upgrade head

# Production (with backup)
kubectl exec -n archon-production deployment/postgres -- \
  pg_dump -U archon archon > backup-$(date +%Y%m%d).sql

kubectl exec -n archon-production deployment/archon-auth-server -- \
  python -m alembic upgrade head
```

---

*This infrastructure is designed for production-scale authentication systems with enterprise-grade security, monitoring, and operational excellence.*