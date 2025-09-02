# 🐳 Docker Compose Deployment

This guide covers deploying Archon using Docker Compose for local development, staging, and production environments.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/coleam00/archon.git
cd archon

# Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials

# Start all services
docker compose --profile full up -d

# Access Archon
open http://localhost:3737
```

## 🏗️ Architecture Overview

Docker Compose deployment includes these services:

```yaml
services:
  archon-ui:        # React frontend (Port 3737)
  archon-server:    # FastAPI backend (Port 8181)
  archon-mcp:       # MCP server (Port 8051)
  archon-agents:    # AI agents (Port 8052) - Optional
  archon-validator: # External validator (Port 8053) - Optional
```

## 📋 Configuration Options

### Service Profiles

Docker Compose uses profiles to control which services start:

```bash
# Minimal setup (UI + Server + MCP)
docker compose up -d

# Full setup with agents
docker compose --profile full up -d

# Agents only (requires server running)
docker compose --profile agents up -d

# Validator service
docker compose --profile validator up -d

# Development with all services
docker compose --profile full --profile validator up -d
```

### Environment Configuration

Create different environment files for different deployments:

```bash
# Development
cp .env.example .env.development

# Staging
cp .env.example .env.staging

# Production
cp .env.example .env.production
```

## 🔧 Development Deployment

### Hybrid Development Mode (Recommended)

Run backend in Docker, frontend locally for fast development:

```bash
# Start backend services in Docker
make dev
# OR manually:
docker compose --profile backend up -d
cd archon-ui-main && npm run dev
```

**Benefits**:
- Instant frontend hot reload
- Isolated backend services
- Easy debugging of UI components

### Full Docker Development

Run everything in Docker containers:

```bash
make dev-docker
# OR manually:
docker compose --profile full up -d
```

**Benefits**:
- Consistent environment across team
- Easier integration testing
- No local Node.js setup required

### Development Configuration

```bash
# .env.development
PROD=false
LOG_LEVEL=DEBUG
AGENTS_ENABLED=true

# Development-specific settings
HOT_RELOAD=true
DEBUG_MODE=true
CORS_ALLOW_ALL=true
```

## 🏭 Production Deployment

### Production Environment Setup

```bash
# Create production environment file
cat > .env.production << EOF
PROD=true
LOG_LEVEL=WARNING

# Production database
SUPABASE_URL=https://your-prod-project.supabase.co
SUPABASE_SERVICE_KEY=your-production-service-key

# Security settings
HOST=yourdomain.com
SSL_REDIRECT=true
SECURE_HEADERS=true

# Performance optimization
CACHE_ENABLED=true
COMPRESSION_ENABLED=true

# Monitoring
LOGFIRE_TOKEN=your-production-logfire-token
MONITORING_ENABLED=true

# Backup configuration
BACKUP_ENABLED=true
BACKUP_RETENTION_DAYS=30
EOF
```

### Production Docker Compose

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  archon-server:
    restart: always
    environment:
      - PROD=true
      - LOG_LEVEL=WARNING
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  archon-ui:
    restart: always
    environment:
      - PROD=true
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  archon-mcp:
    restart: always
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  archon-agents:
    restart: always
    profiles:
      - agents
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # Production additions
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.production.conf:/etc/nginx/conf.d/default.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - archon-ui
      - archon-server
    restart: always

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

### Nginx Configuration

Create `nginx.production.conf`:

```nginx
upstream archon_ui {
    server archon-ui:3737;
}

upstream archon_api {
    server archon-server:8181;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Main application
    location / {
        proxy_pass http://archon_ui;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # API endpoints
    location /api/ {
        proxy_pass http://archon_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # WebSocket support for real-time features
    location /socket.io/ {
        proxy_pass http://archon_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # Static assets caching
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Production Deployment Commands

```bash
# Deploy to production
docker compose -f docker-compose.yml -f docker-compose.production.yml --profile full up -d

# Update production deployment
git pull origin main
docker compose -f docker-compose.yml -f docker-compose.production.yml --profile full up -d --build

# View production logs
docker compose -f docker-compose.yml -f docker-compose.production.yml logs -f

# Backup production data
docker exec archon-server python scripts/backup.py
```

## 🔒 Security Configuration

### SSL/TLS Setup

#### Option 1: Let's Encrypt (Recommended)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal cron job
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

#### Option 2: Custom SSL Certificates

```bash
# Create SSL directory
mkdir -p ssl

# Copy your certificates
cp your-certificate.pem ssl/fullchain.pem
cp your-private-key.pem ssl/privkey.pem

# Set proper permissions
chmod 600 ssl/privkey.pem
chmod 644 ssl/fullchain.pem
```

### Environment Security

```bash
# Secure environment files
chmod 600 .env.production
chown root:root .env.production

# Use Docker secrets (Docker Swarm)
echo "your-supabase-key" | docker secret create supabase_key -
```

### Network Security

```yaml
# docker-compose.production.yml - Network isolation
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

services:
  archon-ui:
    networks:
      - frontend
      
  archon-server:
    networks:
      - frontend
      - backend
      
  redis:
    networks:
      - backend
```

## 📊 Monitoring and Logging

### Log Management

```yaml
# docker-compose.production.yml - Logging configuration
x-logging: &logging
  logging:
    driver: "json-file"
    options:
      max-size: "10m"
      max-file: "3"

services:
  archon-server:
    <<: *logging
    
  archon-ui:
    <<: *logging
```

### Health Monitoring

```bash
# Check service health
docker compose ps
docker compose logs --tail=50

# Automated health checks
#!/bin/bash
# health-check.sh
SERVICES=("archon-ui" "archon-server" "archon-mcp")

for service in "${SERVICES[@]}"; do
    if ! docker compose ps $service | grep -q "running"; then
        echo "ALERT: $service is not running"
        # Send notification (email, Slack, etc.)
    fi
done
```

### Metrics Collection

```yaml
# Add Prometheus monitoring
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  prometheus_data:
  grafana_data:
```

## 🔄 Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh - Automated backup script

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/archon"
mkdir -p $BACKUP_DIR

# Database backup (if using local database)
docker exec archon-postgres pg_dump -U archon archon > $BACKUP_DIR/db_$DATE.sql

# Configuration backup
tar -czf $BACKUP_DIR/config_$DATE.tar.gz .env* docker-compose*.yml nginx*.conf

# Application data backup
docker run --rm -v archon_data:/data -v $BACKUP_DIR:/backup alpine tar -czf /backup/data_$DATE.tar.gz -C /data .

# Upload to cloud storage (optional)
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/archon/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Recovery Procedure

```bash
#!/bin/bash
# restore.sh - Recovery script

BACKUP_DATE=$1
BACKUP_DIR="/backups/archon"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 YYYYMMDD_HHMMSS"
    exit 1
fi

# Stop services
docker compose down

# Restore database
docker exec -i archon-postgres psql -U archon archon < $BACKUP_DIR/db_$BACKUP_DATE.sql

# Restore configuration
tar -xzf $BACKUP_DIR/config_$BACKUP_DATE.tar.gz

# Restore application data
docker run --rm -v archon_data:/data -v $BACKUP_DIR:/backup alpine tar -xzf /backup/data_$BACKUP_DATE.tar.gz -C /data

# Start services
docker compose --profile full up -d
```

## 🚀 Scaling Docker Compose

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  archon-server:
    deploy:
      replicas: 3
      
  archon-agents:
    deploy:
      replicas: 2

  # Load balancer
  nginx:
    depends_on:
      - archon-server
    environment:
      - NGINX_UPSTREAM_SERVERS=archon-server:8181
```

### Resource Optimization

```yaml
# Resource limits and reservations
services:
  archon-server:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    
  archon-agents:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

## 🔧 Maintenance

### Regular Maintenance Tasks

```bash
# Update containers
docker compose pull
docker compose --profile full up -d

# Clean up unused resources
docker system prune -f
docker volume prune -f

# Check disk usage
docker system df

# Rotate logs
docker compose logs --since 24h > logs/archon-$(date +%Y%m%d).log
```

### Troubleshooting Commands

```bash
# Check service status
docker compose ps
docker compose logs -f archon-server

# Check resource usage
docker stats

# Restart specific service
docker compose restart archon-server

# Check network connectivity
docker compose exec archon-server curl -f http://archon-ui:3737
docker compose exec archon-ui curl -f http://archon-server:8181/health
```

## 🎯 Performance Optimization

### Container Optimization

```dockerfile
# Multi-stage build for smaller images
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine as runtime
WORKDIR /app
COPY --from=build /app/node_modules ./node_modules
COPY . .
CMD ["npm", "start"]
```

### Caching Strategy

```yaml
# Redis cache for improved performance
services:
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
```

### Database Connection Pooling

```yaml
# PgBouncer for connection pooling
services:
  pgbouncer:
    image: pgbouncer/pgbouncer:latest
    environment:
      - DATABASES_HOST=your-supabase-host
      - DATABASES_PORT=5432
      - DATABASES_USER=postgres
      - DATABASES_PASSWORD=your-password
      - POOL_MODE=transaction
      - MAX_CLIENT_CONN=100
      - DEFAULT_POOL_SIZE=20
```

---

## 📚 Next Steps

- [🏗️ Production Architecture](aws.md) - Scale to cloud platforms
- [🔍 Monitoring Setup](../monitoring/README.md) - Comprehensive monitoring
- [🔐 Security Hardening](../security/README.md) - Advanced security configuration
- [🔧 Troubleshooting](../../TROUBLESHOOTING.md) - Solve common issues

**Need help?** Join our [community discussions](https://github.com/coleam00/archon/discussions) or check our [troubleshooting guide](../../TROUBLESHOOTING.md).