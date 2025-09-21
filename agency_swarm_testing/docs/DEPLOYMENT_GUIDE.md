# Agency Swarm Deployment Guide

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Architecture](#deployment-architecture)
- [Environment Setup](#environment-setup)
- [Deployment Methods](#deployment-methods)
- [Configuration Management](#configuration-management)
- [Health Monitoring](#health-monitoring)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)
- [Production Checklist](#production-checklist)

## Overview

The Agency Swarm system is designed for enterprise-grade deployment with multiple deployment options including Docker, Kubernetes, and cloud-native solutions. This guide provides comprehensive instructions for deploying the Agency Swarm system in production environments.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: Minimum 4 cores, recommended 8+ cores for production
- **Memory**: Minimum 8GB RAM, recommended 16GB+ for production
- **Storage**: Minimum 50GB available space, recommended 100GB+ for production
- **Network**: Stable internet connection with minimum 100Mbps bandwidth

### Software Dependencies
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher
- **Kubernetes**: Version 1.24+ (for Kubernetes deployment)
- **kubectl**: Version 1.24+ (for Kubernetes deployment)
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18 or higher
- **Git**: Version 2.30 or higher

### External Services
- **Supabase**: PostgreSQL database with pgvector extension
- **Redis**: For caching and session management
- **OpenAI API Key**: For AI/ML capabilities
- **Container Registry**: For storing Docker images (GitHub Container Registry recommended)

## Deployment Architecture

### Component Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │      API        │    │      MCP        │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (HTTP)        │
│   Port: 3737    │    │   Port: 8181    │    │   Port: 8051    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Agents      │
                    │   (Python)      │
                    │   Port: 8052    │
                    └─────────────────┘
```

### Data Flow
1. **Frontend** serves the React application and handles user interactions
2. **API Server** manages business logic, authentication, and data processing
3. **MCP Server** provides tool execution and external service integration
4. **Agents Service** handles AI agent execution and orchestration

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-org/agency-swarm.git
cd agency-swarm
```

### 2. Environment Variables
Create `.env` file:
```bash
# Database Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here

# AI Services
OPENAI_API_KEY=your-openai-api-key

# Monitoring
GRAFANA_PASSWORD=your-secure-password

# Security
POSTGRES_PASSWORD=your-secure-db-password

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 3. Initialize Dependencies
```bash
# Python dependencies
cd python
python -m pip install --upgrade pip
pip install -r requirements.server.txt

# Node.js dependencies
cd ../archon-ui-main
npm install
```

## Deployment Methods

### Method 1: Docker Compose (Recommended for Development)

#### Quick Start
```bash
# Build and start all services
docker-compose -f agency_swarm_testing/deployment/docker-compose.yml up -d

# Check service status
docker-compose -f agency_swarm_testing/deployment/docker-compose.yml ps

# View logs
docker-compose -f agency_swarm_testing/deployment/docker-compose.yml logs -f
```

#### Service Management
```bash
# Stop services
docker-compose -f agency_swarm_testing/deployment/docker-compose.yml down

# Restart services
docker-compose -f agency_swarm_testing/deployment/docker-compose.yml restart

# Scale services
docker-compose -f agency_swarm_testing/deployment/docker-compose.yml up -d --scale archon-api=3
```

### Method 2: Kubernetes (Production Recommended)

#### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install kubectl plugin for neat output
kubectl krew install neat
```

#### Deployment Steps
```bash
# Apply Kubernetes manifests
kubectl apply -f agency_swarm_testing/deployment/kubernetes/

# Check deployment status
kubectl get pods -n archon
kubectl get services -n archon

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l app=archon -n archon --timeout=300s
```

#### Access Services
```bash
# Port forward to local machine
kubectl port-forward -n archon svc/archon-frontend-service 3737:80
kubectl port-forward -n archon svc/archon-api-service 8181:8000
kubectl port-forward -n archon svc/archon-mcp-service 8051:8000
kubectl port-forward -n archon svc/archon-agents-service 8052:8000
```

### Method 3: Automated Deployment Script

#### Using the Deployment Manager
```bash
# Deploy to development
python agency_swarm_testing/deployment/scripts/deploy.py --environment development --action deploy

# Deploy to production
python agency_swarm_testing/deployment/scripts/deploy.py --environment production --action deploy

# Scale services
python agency_swarm_testing/deployment/scripts/deploy.py --environment production --action scale --service archon-api --replicas 5

# Check health
python agency_swarm_testing/deployment/scripts/deploy.py --environment production --action health
```

## Configuration Management

### Environment-Specific Configurations

#### Development Environment
```yaml
# agency_swarm_testing/deployment/config/dev-config.yaml
environment: development
replicas:
  frontend: 1
  api: 1
  mcp: 1
  agents: 1

resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```

#### Production Environment
```yaml
# agency_swarm_testing/deployment/config/prod-config.yaml
environment: production
replicas:
  frontend: 3
  api: 3
  mcp: 2
  agents: 2

resources:
  requests:
    cpu: "200m"
    memory: "512Mi"
  limits:
    cpu: "1000m"
    memory: "1Gi"

autoscaling:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### Configuration Updates
```bash
# Update ConfigMap
kubectl create configmap archon-config --from-env-file=.env -n archon --dry-run=client -o yaml | kubectl apply -f -

# Update Secrets
kubectl create secret generic archon-secrets --from-env-file=.env -n archon --dry-run=client -o yaml | kubectl apply -f -

# Restart deployments to apply changes
kubectl rollout restart deployment -n archon
```

## Health Monitoring

### Built-in Health Checks
Each service includes health check endpoints:

- **Frontend**: `GET /health`
- **API**: `GET /api/health`
- **MCP**: `GET /health`
- **Agents**: `GET /health`

### Monitoring Stack
The deployment includes a comprehensive monitoring stack:

#### Prometheus
- Access: `http://localhost:9090`
- Metrics collection from all services
- Alerting rules configuration

#### Grafana
- Access: `http://localhost:3000`
- Default credentials: `admin` / password from environment
- Pre-configured dashboards for Agency Swarm metrics

#### ELK Stack
- **Elasticsearch**: `http://localhost:9200`
- **Kibana**: `http://localhost:5601`
- Centralized logging and log analysis

### Custom Health Checks
```bash
# Check all services health
for service in frontend api mcp agents; do
  echo "Checking $service health..."
  curl -f "http://localhost:${service_ports[$service]}/health" || echo "$service health check failed"
done
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
kubectl exec -n archon postgres-0 -- pg_dump -U postgres archon > backup.sql

# Restore backup
kubectl exec -i -n archon postgres-0 -- psql -U postgres archon < backup.sql
```

### Configuration Backup
```bash
# Backup all Kubernetes resources
kubectl get all -n archon -o yaml > archon-backup.yaml

# Backup secrets and configmaps
kubectl get secrets,configmaps -n archon -o yaml > archon-config-backup.yaml
```

### Automated Backup Script
```bash
#!/bin/bash
# agency_swarm_testing/deployment/scripts/backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"
mkdir -p $BACKUP_DIR

# Database backup
kubectl exec -n archon postgres-0 -- pg_dump -U postgres archon > $BACKUP_DIR/database.sql

# Kubernetes resources backup
kubectl get all -n archon -o yaml > $BACKUP_DIR/kubernetes.yaml
kubectl get secrets,configmaps -n archon -o yaml > $BACKUP_DIR/config.yaml

echo "Backup completed: $BACKUP_DIR"
```

## Troubleshooting

### Common Issues

#### 1. Service Not Starting
```bash
# Check pod status
kubectl get pods -n archon

# Check pod logs
kubectl logs -n archon pod-name

# Describe pod for detailed information
kubectl describe pod -n archon pod-name
```

#### 2. Database Connection Issues
```bash
# Check database pod status
kubectl get pods -n archon -l app=postgres

# Test database connectivity
kubectl exec -n archon archon-api-xxxxx -- psql $DATABASE_URL -c "SELECT 1"

# Check database logs
kubectl logs -n archon postgres-0
```

#### 3. High Memory Usage
```bash
# Monitor resource usage
kubectl top pods -n archon

# Check memory limits
kubectl get deployment -n archon -o yaml | grep resources

# Scale down if needed
kubectl scale deployment archon-api -n archon --replicas=1
```

#### 4. Performance Issues
```bash
# Check response times
kubectl logs -n archon archon-api-xxxxx | grep "response_time"

# Monitor CPU usage
kubectl top pods -n archon --sort-by=cpu

# Check for errors
kubectl logs -n archon archon-api-xxxxx | grep ERROR
```

### Debug Commands
```bash
# Port forward for local debugging
kubectl port-forward -n archon svc/archon-api-service 8181:8000

# Execute shell in container
kubectl exec -it -n archon archon-api-xxxxx -- /bin/bash

# Check network connectivity
kubectl exec -n archon archon-api-xxxxx -- curl -f http://archon-mcp-service:8000/health
```

## Production Checklist

### Pre-Deployment Checklist
- [ ] Environment variables configured correctly
- [ ] Database schema initialized
- [ ] SSL/TLS certificates configured
- [ ] Backup procedures tested
- [ ] Monitoring and alerting configured
- [ ] Security scanning completed
- [ ] Performance testing completed
- [ ] Disaster recovery plan documented

### Post-Deployment Checklist
- [ ] All services running and healthy
- [ ] Database connections working
- [ ] External APIs accessible
- [ ] Authentication functioning
- [ ] Monitoring dashboards accessible
- [ ] Logs being collected
- [ ] Backup processes running
- [ ] Performance metrics within thresholds

### Ongoing Maintenance
- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Log review and analysis
- [ ] Backup verification
- [ ] Capacity planning
- [ ] Disaster recovery drills
- [ ] Security audits
- [ ] Performance optimization

## Support

For deployment issues:
1. Check the troubleshooting section above
2. Review logs for error messages
3. Consult the monitoring dashboards
4. Contact the support team with detailed error information

## Next Steps

After successful deployment:
1. Configure users and permissions
2. Set up data sources and knowledge bases
3. Configure agent templates and patterns
4. Set up automated testing
5. Configure CI/CD pipelines
6. Establish monitoring alerting rules