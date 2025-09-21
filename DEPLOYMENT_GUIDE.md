# Agency Swarm Enhancement - Comprehensive Deployment Guide

## 🎯 Overview

This guide provides comprehensive instructions for deploying the Agency Swarm enhancement across different environments, from development to production. This is the final phase to ensure everything works together seamlessly.

## 📋 Table of Contents

- [Deployment Prerequisites](#deployment-prerequisites)
- [Environment Setup](#environment-setup)
- [Quick Start Deployment](#quick-start-deployment)
- [Production Deployment](#production-deployment)
- [Blue-Green Deployment Strategy](#blue-green-deployment-strategy)
- [Post-Deployment Validation](#post-deployment-validation)
- [Troubleshooting](#troubleshooting)

## 🏗️ System Components

### ✅ **Agency Swarm Enhancement Components**
1. **Dynamic Agent Communication** - Real-time messaging and collaboration
2. **Intelligent Task Routing** - Smart escalation between agent tiers
3. **Knowledge Sharing System** - Distributed learning and experience transfer
4. **Real-Time Collaboration** - Pub/sub messaging with shared contexts
5. **Cost Optimization Engine** - Budget tracking with ROI analysis
6. **Enhanced Security** - Encryption and compliance validation
7. **Performance Monitoring** - Real-time metrics and analytics
8. **MCP Integration** - Model Context Protocol for seamless AI integration
9. **Deployment Automation** - Kubernetes-ready with blue-green strategy
10. **Configuration Management** - Environment-specific configs with encryption
11. **E2E Test Suite** - Comprehensive testing across all components
12. **CI/CD Integration** - Automated pipelines with validation

## 🔒 Deployment Prerequisites

### System Requirements

- **Kubernetes Cluster**: v1.25+ with at least 8GB RAM and 4 vCPUs
- **Database**: Supabase/PostgreSQL with pgvector extension
- **Storage**: Persistent storage for embeddings and knowledge base
- **Networking**: Load balancer with support for WebSocket connections
- **Security**: TLS certificates, network policies, and RBAC configured

### Required Tools

```bash
# Install required tools
brew install kubectl helm
npm install -g @playwright/test

# Kubernetes cluster access
kubectl config current-context

# Verify connectivity
kubectl cluster-info
```

### Configuration Files

Prepare the following configuration files:

1. `.env.production` - Environment variables
2. `k8s/` - Kubernetes manifests
3. `config/` - Configuration management files
4. `scripts/` - Deployment automation scripts

## 🔧 Environment Setup

### 1. Environment Configuration

Create environment-specific configuration:

```bash
# Clone configuration management
cp config/production.example.yaml config/production.yaml

# Update configuration
vim config/production.yaml
```

Key configuration parameters:

```yaml
environment: production
database:
  url: "postgresql://user:pass@host:5432/dbname"
  pool_size: 20
  ssl_enabled: true

security:
  encryption_key: "your-256-bit-encryption-key"
  jwt_secret: "your-jwt-secret"
  allowed_origins: ["https://your-domain.com"]

scaling:
  min_replicas: 3
  max_replicas: 10
  target_cpu_utilization: 70

monitoring:
  metrics_enabled: true
  logging_level: INFO
  tracing_enabled: true
```

### 2. Secrets Management

Generate and configure secrets:

```bash
# Generate secrets
python scripts/generate_secrets.py --environment production

# Apply Kubernetes secrets
kubectl apply -f k8s/secrets/

# Verify secrets
kubectl get secrets -n agency-swarm
```

### 3. Database Setup

Configure database schema:

```bash
# Run database migrations
python scripts/database/migrate.py --env production

# Verify schema
psql $DATABASE_URL -c "\dt"
```

## 🚀 Quick Start Deployment

### Development Deployment

For rapid development and testing:

```bash
# 1. Start local development environment
docker-compose -f docker-compose.dev.yml up -d

# 2. Install dependencies
npm install
cd python && uv sync

# 3. Run tests
npm run test:coverage
cd python && uv run pytest

# 4. Start services
npm run dev &
cd python && uv run python -m src.server.main &
```

### Staging Deployment

For pre-production validation:

```bash
# 1. Configure staging environment
python scripts/configure_environment.py staging

# 2. Build and push images
python scripts/build_images.py --environment staging

# 3. Deploy to staging
python scripts/deployment_scripts.py deploy --environment staging

# 4. Validate deployment
python scripts/validation/health_check.py --environment staging
```

## 🏭 Production Deployment

### Step 1: Pre-Deployment Checks

```bash
# Run comprehensive validation
python scripts/validation/pre_deployment_check.py --environment production

# Verify all dependencies
python scripts/validation/dependencies_check.py

# Check database connectivity
python scripts/validation/database_check.py
```

### Step 2: Build and Push Images

```bash
# Build Docker images
docker build -t agency-swarm-frontend:latest ./archon-ui-main
docker build -t agency-swarm-backend:latest ./python

# Tag images
docker tag agency-swarm-frontend:latest registry.example.com/agency-swarm-frontend:$VERSION
docker tag agency-swarm-backend:latest registry.example.com/agency-swarm-backend:$VERSION

# Push to registry
docker push registry.example.com/agency-swarm-frontend:$VERSION
docker push registry.example.com/agency-swarm-backend:$VERSION
```

### Step 3: Deploy Using Automation

```bash
# Execute production deployment
python scripts/deployment_scripts.py deploy \
  --environment production \
  --strategy blue-green \
  --version $VERSION \
  --health-check-interval 30s \
  --rollback-on-failure
```

### Step 4: Monitor Deployment

```bash
# Watch deployment progress
kubectl get pods -n agency-swarm -w

# Check deployment status
kubectl describe deployment agency-swarm -n agency-swarm

# View logs
kubectl logs -f deployment/agency-swarm -n agency-swarm
```

## 🔄 Blue-Green Deployment Strategy

### Overview

The blue-green deployment strategy ensures zero-downtime deployments by maintaining two identical production environments:

- **Blue**: Current production version
- **Green**: New version being deployed

### Deployment Process

1. **Prepare Green Environment**
   ```bash
   python scripts/deployment_scripts.py deploy --environment green
   ```

2. **Validate Green Environment**
   ```bash
   python scripts/validation/production_validation.py --environment green
   ```

3. **Traffic Switch**
   ```bash
   python scripts/deployment_scripts.py switch-traffic --to green
   ```

4. **Monitor and Cleanup**
   ```bash
   python scripts/monitoring/post_deployment.py --environment green
   ```

### Rollback Procedure

```bash
# Emergency rollback
python scripts/deployment_scripts.py rollback \
  --from green \
  --to blue \
  --immediate
```

## ✅ Post-Deployment Validation

### Health Checks

```bash
# Run comprehensive health checks
python scripts/validation/health_check.py --environment production

# API endpoint validation
python scripts/validation/api_validation.py

# Performance validation
python scripts/validation/performance_check.py
```

### Integration Tests

```bash
# Run E2E tests
npm run test:e2e

# Run integration tests
python scripts/validation/integration_tests.py
```

### Security Validation

```bash
# Security compliance check
python scripts/validation/security_compliance.py

# Vulnerability scan
python scripts/validation/vulnerability_scan.py
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database connectivity
python scripts/validation/database_check.py --verbose

# Verify database configuration
kubectl get configmap database-config -n agency-swarm -o yaml
```

#### 2. Pod Crashing
```bash
# Check pod status
kubectl get pods -n agency-swarm

# View pod logs
kubectl logs <pod-name> -n agency-swarm --previous

# Describe pod for events
kubectl describe pod <pod-name> -n agency-swarm
```

#### 3. Performance Issues
```bash
# Check resource usage
kubectl top pods -n agency-swarm

# View resource limits
kubectl get deployment agency-swarm -n agency-swarm -o yaml | grep resources
```

#### 4. WebSocket Connection Issues
```bash
# Check WebSocket connectivity
python scripts/validation/websocket_check.py

# Verify ingress configuration
kubectl get ingress -n agency-swarm -o yaml
```

### Emergency Procedures

#### Immediate Rollback
```bash
# Quick rollback to previous version
python scripts/deployment_scripts.py rollback --immediate
```

#### Service Restart
```bash
# Restart all services
kubectl rollout restart deployment agency-swarm -n agency-swarm
```

#### Scale Down
```bash
# Emergency scale down
kubectl scale deployment agency-swarm --replicas=0 -n agency-swarm
```

### Support and Monitoring

For production support:

1. **Monitoring Dashboard**: Access Grafana at `https://monitoring.example.com`
2. **Alerting**: Check PagerDuty for critical alerts
3. **Logs**: View logs in ELK stack at `https://logs.example.com`
4. **Support**: Contact infrastructure team for assistance

## ✅ Deployment Checklist

- [ ] Environment configuration completed
- [ ] Secrets generated and applied
- [ ] Database migrations completed
- [ ] Docker images built and pushed
- [ ] Kubernetes manifests validated
- [ ] Pre-deployment checks passed
- [ ] Blue-green deployment executed
- [ ] Health checks passing
- [ ] Integration tests passing
- [ ] Security validation completed
- [ ] Performance metrics within SLA
- [ ] Monitoring and alerting configured
- [ ] Documentation updated
- [ ] Backup verification completed

## 🎯 Best Practices

1. **Always test in staging** before deploying to production
2. **Monitor deployment progress** and respond to alerts promptly
3. **Document all changes** and maintain deployment logs
4. **Follow the rollback procedure** if issues arise
5. **Regular security audits** and compliance checks
6. **Performance monitoring** and optimization
7. **Disaster recovery testing** and validation

## 🚀 Next Steps

After successful deployment:

1. **Monitor performance** and user experience
2. **Optimize configuration** based on real-world usage
3. **Plan for scaling** based on demand
4. **Schedule regular maintenance** windows
5. **Update documentation** with deployment insights

---

For additional support, refer to the [Troubleshooting Guide](TROUBLESHOOTING.md) or contact the infrastructure team.