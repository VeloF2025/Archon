# 🚀 Deployment Guide

This directory contains comprehensive deployment guides for Archon across different platforms and environments.

## 📋 Quick Deployment Options

| Platform | Complexity | Time to Deploy | Recommended For |
|----------|------------|----------------|-----------------|
| [Docker Compose](docker-compose.md) | ⭐ Easy | 10 minutes | Local development, small teams |
| [AWS](aws.md) | ⭐⭐ Medium | 30 minutes | Production, scalable workloads |
| [Google Cloud](gcp.md) | ⭐⭐ Medium | 30 minutes | Production, AI/ML workloads |
| [Azure](azure.md) | ⭐⭐ Medium | 30 minutes | Enterprise, Microsoft ecosystem |
| [DigitalOcean](digitalocean.md) | ⭐ Easy | 20 minutes | Startups, cost-effective |
| [Railway](railway.md) | ⭐ Easy | 15 minutes | Quick prototypes, demos |
| [Render](render.md) | ⭐ Easy | 15 minutes | Static sites, simple apps |

## 🏗️ Architecture Considerations

### Single Server Deployment
**Best for**: Development, small teams, proof of concepts

```
┌─────────────────────────┐
│    Single Server        │
│                         │
│ ┌─────────────────────┐ │
│ │   Archon Stack      │ │
│ │                     │ │
│ │ • UI (Port 3737)    │ │
│ │ • API (Port 8181)   │ │
│ │ • MCP (Port 8051)   │ │
│ │ • Agents (8052)     │ │
│ └─────────────────────┘ │
│                         │
│ ┌─────────────────────┐ │
│ │   Supabase          │ │
│ │   (External SaaS)   │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

### Multi-Container Deployment
**Best for**: Production, high availability, scaling

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    UI       │  │    API      │  │   Agents    │
│  Container  │  │  Container  │  │  Container  │
│             │  │             │  │             │
│ Port: 3737  │  │ Port: 8181  │  │ Port: 8052  │
└─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
           ┌─────────────▼─────────────┐
           │      Load Balancer       │
           │     (Nginx/Traefik)      │
           └──────────────────────────┘
                        │
           ┌─────────────▼─────────────┐
           │      Database Layer      │
           │                          │
           │ • Supabase (Primary)     │
           │ • Redis (Cache)          │
           │ • File Storage (S3)      │
           └──────────────────────────┘
```

### Kubernetes Deployment
**Best for**: Enterprise, auto-scaling, multi-region

```
┌─────────────────────────────────────────┐
│             Kubernetes Cluster          │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │          Ingress Controller         │ │
│ │        (NGINX/Traefik)              │ │
│ └─────────────────────────────────────┘ │
│                    │                    │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │   UI     │ │   API    │ │  Agents  │ │
│ │   Pods   │ │   Pods   │ │   Pods   │ │
│ │  (3x)    │ │  (5x)    │ │  (3x)    │ │
│ └──────────┘ └──────────┘ └──────────┘ │
│                    │                    │
│ ┌─────────────────────────────────────┐ │
│ │         Services & Storage          │ │
│ │                                     │ │
│ │ • ConfigMaps & Secrets              │ │
│ │ • Persistent Volumes                │ │
│ │ • Redis (StatefulSet)               │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                    │
    ┌───────────────▼───────────────┐
    │      External Services        │
    │                               │
    │ • Supabase (Database)         │
    │ • OpenAI/Gemini (LLM APIs)    │
    │ • Monitoring (Logfire)        │
    └───────────────────────────────┘
```

## 🔧 Environment-Specific Configurations

### Development Environment
```bash
# .env.development
PROD=false
LOG_LEVEL=DEBUG
AGENTS_ENABLED=true
HOT_RELOAD=true
BACKUP_ENABLED=false
```

### Staging Environment
```bash
# .env.staging
PROD=false
LOG_LEVEL=INFO
AGENTS_ENABLED=true
BACKUP_ENABLED=true
MONITORING_ENABLED=true
CACHE_TTL=300
```

### Production Environment
```bash
# .env.production
PROD=true
LOG_LEVEL=WARNING
SECURITY_ENHANCED=true
BACKUP_ENABLED=true
MONITORING_ENABLED=true
CACHE_TTL=3600
SSL_REDIRECT=true
```

## 📊 Resource Requirements

### Minimum Requirements (Development)
- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Storage**: 10GB disk space
- **Network**: Broadband internet

### Recommended Requirements (Production)
- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ SSD
- **Network**: High-speed internet with low latency

### Enterprise Requirements (High Scale)
- **CPU**: 8+ cores per service
- **Memory**: 16GB+ RAM per service
- **Storage**: 100GB+ SSD with backups
- **Network**: Dedicated bandwidth, CDN
- **Database**: Dedicated database server/cluster

## 🔐 Security Checklist

### Pre-Deployment Security
- [ ] Update all default passwords and API keys
- [ ] Enable HTTPS/TLS for all communications
- [ ] Configure firewall rules (only required ports open)
- [ ] Set up VPN access for administrative tasks
- [ ] Enable database encryption at rest
- [ ] Configure secure backup storage
- [ ] Set up monitoring and alerting
- [ ] Review and apply security patches

### Network Security
- [ ] Use private networks where possible
- [ ] Implement network segmentation
- [ ] Configure load balancer SSL termination
- [ ] Set up DDoS protection
- [ ] Enable rate limiting on public endpoints
- [ ] Configure CORS policies appropriately

### Application Security
- [ ] Enable Row Level Security (RLS) in Supabase
- [ ] Use encrypted credentials storage
- [ ] Implement proper authentication flows
- [ ] Set up API key rotation
- [ ] Configure session management
- [ ] Enable audit logging

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build and Deploy
        run: |
          docker compose -f docker-compose.prod.yml build
          docker compose -f docker-compose.prod.yml up -d
          
      - name: Health Check
        run: |
          curl -f http://your-domain.com/health
```

### GitLab CI Example
```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

deploy_production:
  stage: deploy
  script:
    - docker compose -f docker-compose.prod.yml up -d
  only:
    - main
```

## 📈 Monitoring and Observability

### Essential Metrics to Monitor
- **System Health**: CPU, memory, disk usage
- **Application Performance**: Response times, error rates
- **Database Performance**: Query times, connection counts
- **User Activity**: Active users, feature usage
- **Business Metrics**: Projects created, tasks completed

### Recommended Monitoring Stack
- **Metrics**: Prometheus + Grafana
- **Logs**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **APM**: Logfire (built-in support)
- **Uptime**: UptimeRobot or Pingdom
- **Alerts**: PagerDuty or Slack integration

## 🔧 Maintenance and Operations

### Regular Maintenance Tasks
- **Daily**: Check system health and logs
- **Weekly**: Review performance metrics and alerts
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Backup testing and disaster recovery drills

### Backup Strategy
```bash
# Database backup (Supabase)
pg_dump $SUPABASE_URL > backup-$(date +%Y%m%d).sql

# Application data backup
tar -czf archon-data-$(date +%Y%m%d).tar.gz ./data

# Upload to cloud storage
aws s3 cp backup-$(date +%Y%m%d).sql s3://your-backup-bucket/
```

### Disaster Recovery
1. **Recovery Time Objective (RTO)**: < 4 hours
2. **Recovery Point Objective (RPO)**: < 1 hour
3. **Backup Frequency**: Daily automated backups
4. **Backup Retention**: 30 days local, 1 year cloud
5. **Testing**: Monthly recovery drills

## 🚀 Scaling Strategies

### Horizontal Scaling
- Add more service instances
- Use container orchestration (Kubernetes)
- Implement service mesh for communication
- Use external cache layer (Redis)

### Vertical Scaling
- Increase CPU/memory for existing instances
- Upgrade to faster storage (NVMe SSD)
- Optimize database queries and indexing
- Enable connection pooling

### Database Scaling
- Use read replicas for read-heavy workloads
- Implement database sharding for large datasets
- Use connection pooling (PgBouncer)
- Optimize queries and add proper indexes

## 📋 Deployment Checklists

### Pre-Deployment Checklist
- [ ] Environment configuration validated
- [ ] Database migrations applied
- [ ] Security configurations in place
- [ ] Monitoring and alerting configured
- [ ] Backup systems operational
- [ ] DNS and SSL certificates configured
- [ ] Load balancing configured (if applicable)
- [ ] Health checks implemented

### Post-Deployment Checklist
- [ ] All services responding to health checks
- [ ] Database connections stable
- [ ] External API integrations working
- [ ] Monitoring dashboards showing green
- [ ] Backup jobs running successfully
- [ ] SSL certificates valid
- [ ] Performance within acceptable limits
- [ ] Error rates within normal ranges

## 🆘 Troubleshooting Deployments

### Common Deployment Issues
1. **Container fails to start**: Check logs and resource limits
2. **Database connection errors**: Verify credentials and network access
3. **SSL/TLS issues**: Check certificate validity and configuration
4. **Performance problems**: Monitor resource usage and optimize
5. **Health check failures**: Verify endpoint accessibility

### Emergency Procedures
```bash
# Quick rollback
docker compose down
git checkout previous-stable-tag
docker compose --profile full up -d

# Emergency maintenance mode
# Create maintenance.html and serve via nginx
```

---

**Choose your deployment platform:**
- [🐳 Docker Compose](docker-compose.md) - Quick local deployment
- [☁️ Cloud Providers](aws.md) - Scalable production deployments
- [⚡ Platform as a Service](railway.md) - Managed deployments

**Need help?** Check our [troubleshooting guide](../TROUBLESHOOTING.md) or join our [community discussions](https://github.com/coleam00/archon/discussions).