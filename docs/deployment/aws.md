# ☁️ AWS Deployment Guide

This guide covers deploying Archon to Amazon Web Services (AWS) for production workloads.

## 🎯 Deployment Options

| Option | Complexity | Cost | Best For |
|--------|------------|------|----------|
| **[EC2 + Docker](#ec2--docker-compose)** | ⭐ Easy | $ Low | Small deployments, learning |
| **[ECS Fargate](#ecs-fargate)** | ⭐⭐ Medium | $$ Medium | Production, auto-scaling |
| **[EKS (Kubernetes)](#eks-kubernetes)** | ⭐⭐⭐ Hard | $$$ High | Enterprise, complex workloads |
| **[Lambda + RDS](#serverless-lambda)** | ⭐⭐ Medium | $ Variable | Event-driven, cost optimization |

## 🚀 EC2 + Docker Compose

### Quick Setup

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --count 1 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678

# 2. Connect and setup
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install Docker
sudo apt update
sudo apt install docker.io docker-compose-plugin -y
sudo usermod -aG docker ubuntu
sudo systemctl enable docker
sudo systemctl start docker

# 4. Deploy Archon
git clone https://github.com/coleam00/archon.git
cd archon
cp .env.example .env
# Edit .env with your configuration
docker compose --profile full up -d
```

### EC2 Configuration

#### Instance Requirements

```bash
# Recommended EC2 instance types
t3.large   # 2 vCPU, 8GB RAM - Development/Small production
t3.xlarge  # 4 vCPU, 16GB RAM - Production
m5.xlarge  # 4 vCPU, 16GB RAM - Compute optimized
m5.2xlarge # 8 vCPU, 32GB RAM - High performance
```

#### Security Groups

```bash
# Create security group
aws ec2 create-security-group \
  --group-name archon-sg \
  --description "Archon security group"

# Allow HTTP/HTTPS
aws ec2 authorize-security-group-ingress \
  --group-name archon-sg \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-name archon-sg \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Allow SSH (restrict to your IP)
aws ec2 authorize-security-group-ingress \
  --group-name archon-sg \
  --protocol tcp \
  --port 22 \
  --cidr YOUR_IP/32
```

#### Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name archon-alb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678

# Create target group
aws elbv2 create-target-group \
  --name archon-targets \
  --protocol HTTP \
  --port 3737 \
  --vpc-id vpc-12345678 \
  --health-check-path /health

# Register EC2 instance
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=i-1234567890abcdef0,Port=3737
```

### Production Environment File

```bash
# .env.production on EC2
PROD=true
HOST=your-domain.com
LOG_LEVEL=WARNING

# Supabase (recommended external database)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key

# AWS-specific settings
AWS_REGION=us-west-2
AWS_S3_BUCKET=archon-backups
AWS_CLOUDWATCH_LOG_GROUP=archon-logs

# Security
SSL_REDIRECT=true
SECURE_HEADERS=true
RATE_LIMITING=true

# Performance
CACHE_ENABLED=true
COMPRESSION_ENABLED=true
```

## 🐋 ECS Fargate

### Task Definitions

```json
{
  "family": "archon-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "archon-server",
      "image": "your-account.dkr.ecr.us-west-2.amazonaws.com/archon-server:latest",
      "portMappings": [
        {
          "containerPort": 8181,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "SUPABASE_URL",
          "value": "https://your-project.supabase.co"
        }
      ],
      "secrets": [
        {
          "name": "SUPABASE_SERVICE_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:123456789012:secret:archon/supabase-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/archon-server",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8181/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### ECS Service

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name archon-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster archon-cluster \
  --service-name archon-server \
  --task-definition archon-server:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678,subnet-87654321],securityGroups=[sg-12345678],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=archon-server,containerPort=8181"
```

### Auto Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/archon-cluster/archon-server \
  --min-capacity 1 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/archon-cluster/archon-server \
  --policy-name archon-cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

## ☸️ EKS (Kubernetes)

### Cluster Creation

```bash
# Create EKS cluster
eksctl create cluster \
  --name archon-cluster \
  --version 1.27 \
  --region us-west-2 \
  --nodegroup-name workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed
```

### Kubernetes Manifests

```yaml
# archon-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: archon
---
# archon-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: archon-config
  namespace: archon
data:
  PROD: "true"
  LOG_LEVEL: "INFO"
  ARCHON_SERVER_PORT: "8181"
  ARCHON_UI_PORT: "3737"
---
# archon-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: archon-secrets
  namespace: archon
type: Opaque
stringData:
  SUPABASE_URL: "https://your-project.supabase.co"
  SUPABASE_SERVICE_KEY: "your-service-key"
  OPENAI_API_KEY: "your-openai-key"
---
# archon-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: archon-server
  namespace: archon
spec:
  replicas: 3
  selector:
    matchLabels:
      app: archon-server
  template:
    metadata:
      labels:
        app: archon-server
    spec:
      containers:
      - name: archon-server
        image: your-account.dkr.ecr.us-west-2.amazonaws.com/archon-server:latest
        ports:
        - containerPort: 8181
        env:
        - name: SUPABASE_URL
          valueFrom:
            secretKeyRef:
              name: archon-secrets
              key: SUPABASE_URL
        - name: SUPABASE_SERVICE_KEY
          valueFrom:
            secretKeyRef:
              name: archon-secrets
              key: SUPABASE_SERVICE_KEY
        envFrom:
        - configMapRef:
            name: archon-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8181
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8181
          initialDelaySeconds: 5
          periodSeconds: 5
---
# archon-server-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: archon-server-service
  namespace: archon
spec:
  selector:
    app: archon-server
  ports:
  - name: http
    port: 8181
    targetPort: 8181
  type: ClusterIP
---
# archon-ui-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: archon-ui
  namespace: archon
spec:
  replicas: 2
  selector:
    matchLabels:
      app: archon-ui
  template:
    metadata:
      labels:
        app: archon-ui
    spec:
      containers:
      - name: archon-ui
        image: your-account.dkr.ecr.us-west-2.amazonaws.com/archon-ui:latest
        ports:
        - containerPort: 3737
        env:
        - name: VITE_API_URL
          value: "http://archon-server-service:8181"
        envFrom:
        - configMapRef:
            name: archon-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
# archon-ui-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: archon-ui-service
  namespace: archon
spec:
  selector:
    app: archon-ui
  ports:
  - name: http
    port: 3737
    targetPort: 3737
  type: ClusterIP
---
# archon-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: archon-ingress
  namespace: archon
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012
spec:
  rules:
  - host: archon.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: archon-server-service
            port:
              number: 8181
      - path: /
        pathType: Prefix
        backend:
          service:
            name: archon-ui-service
            port:
              number: 3737
```

### Deployment Commands

```bash
# Apply all manifests
kubectl apply -f archon-namespace.yaml
kubectl apply -f archon-configmap.yaml
kubectl apply -f archon-secret.yaml
kubectl apply -f archon-server-deployment.yaml
kubectl apply -f archon-ui-deployment.yaml
kubectl apply -f archon-ingress.yaml

# Check deployment status
kubectl get pods -n archon
kubectl get services -n archon
kubectl get ingress -n archon

# View logs
kubectl logs -f deployment/archon-server -n archon
kubectl logs -f deployment/archon-ui -n archon
```

## 🔐 Security Best Practices

### AWS IAM Roles

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:archon/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::archon-backups/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### Secrets Management

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
  --name archon/supabase-key \
  --description "Supabase service key for Archon" \
  --secret-string "your-service-key-here"

aws secretsmanager create-secret \
  --name archon/openai-key \
  --description "OpenAI API key for Archon" \
  --secret-string "your-openai-key-here"
```

### VPC Configuration

```bash
# Create VPC for Archon
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=archon-vpc}]'

# Create subnets
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.1.0/24 --availability-zone us-west-2a
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.2.0/24 --availability-zone us-west-2b

# Create and attach internet gateway
aws ec2 create-internet-gateway
aws ec2 attach-internet-gateway --internet-gateway-id igw-12345678 --vpc-id vpc-12345678
```

## 📊 Monitoring and Logging

### CloudWatch Setup

```bash
# Create log groups
aws logs create-log-group --log-group-name /aws/ecs/archon-server
aws logs create-log-group --log-group-name /aws/ecs/archon-ui

# Create custom metrics
aws cloudwatch put-metric-data \
  --namespace "Archon/Application" \
  --metric-data MetricName=ActiveUsers,Value=100,Unit=Count
```

### CloudWatch Alarms

```bash
# High CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "Archon-High-CPU" \
  --alarm-description "Archon high CPU utilization" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2

# Error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "Archon-High-Errors" \
  --alarm-description "Archon high error rate" \
  --metric-name 4XXError \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

## 💾 Backup and Disaster Recovery

### RDS Backup (if using RDS)

```bash
# Create RDS instance for local database needs
aws rds create-db-instance \
  --db-instance-identifier archon-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --allocated-storage 20 \
  --master-username archonuser \
  --master-user-password your-password \
  --vpc-security-group-ids sg-12345678 \
  --backup-retention-period 7 \
  --multi-az \
  --storage-encrypted
```

### S3 Backup Strategy

```bash
# Create S3 bucket for backups
aws s3 mb s3://archon-backups-unique-name

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket archon-backups-unique-name \
  --versioning-configuration Status=Enabled

# Set lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket archon-backups-unique-name \
  --lifecycle-configuration file://lifecycle.json
```

### Automated Backup Script

```bash
#!/bin/bash
# backup-to-s3.sh
DATE=$(date +%Y%m%d_%H%M%S)
BUCKET="archon-backups-unique-name"

# Backup application data
docker run --rm \
  -v archon_data:/data \
  -v $(pwd):/backup \
  alpine tar -czf /backup/archon-data-$DATE.tar.gz -C /data .

# Upload to S3
aws s3 cp archon-data-$DATE.tar.gz s3://$BUCKET/backups/

# Cleanup local backup
rm archon-data-$DATE.tar.gz

# Upload logs
aws s3 sync /var/log/archon s3://$BUCKET/logs/$(date +%Y/%m/%d)/
```

## 🚀 CI/CD Pipeline

### GitHub Actions for AWS

```yaml
# .github/workflows/deploy-aws.yml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Login to Amazon ECR
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build and push Docker images
        run: |
          # Build server image
          docker build -f python/Dockerfile.server -t archon-server .
          docker tag archon-server:latest $ECR_REPOSITORY/archon-server:latest
          docker push $ECR_REPOSITORY/archon-server:latest
          
          # Build UI image
          docker build -f archon-ui-main/Dockerfile -t archon-ui .
          docker tag archon-ui:latest $ECR_REPOSITORY/archon-ui:latest
          docker push $ECR_REPOSITORY/archon-ui:latest

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster archon-cluster \
            --service archon-server \
            --force-new-deployment
          
          aws ecs update-service \
            --cluster archon-cluster \
            --service archon-ui \
            --force-new-deployment
```

## 💰 Cost Optimization

### Cost Estimates (Monthly)

| Service | Configuration | Estimated Cost |
|---------|---------------|----------------|
| **EC2 t3.large** | Single instance | $60-80 |
| **ECS Fargate** | 2 tasks, 1 vCPU each | $80-120 |
| **EKS** | 3 m5.large nodes | $150-200 |
| **ALB** | Standard load balancer | $20-25 |
| **NAT Gateway** | Single AZ | $45-50 |
| **S3** | 100GB storage | $2-3 |
| **CloudWatch** | Standard monitoring | $10-15 |

### Cost Optimization Tips

1. **Use Spot Instances**: Save 50-70% on EC2 costs
2. **Reserved Instances**: 1-3 year commitment for 30-60% savings  
3. **Auto Scaling**: Scale down during low usage
4. **S3 Intelligent Tiering**: Automatic storage optimization
5. **CloudWatch Costs**: Use log retention policies
6. **Data Transfer**: Minimize cross-AZ traffic

### Spot Instance Configuration

```bash
# Launch spot instance
aws ec2 request-spot-instances \
  --launch-specification '{
    "ImageId": "ami-0abcdef1234567890",
    "InstanceType": "t3.large",
    "KeyName": "your-key-pair",
    "SecurityGroupIds": ["sg-12345678"],
    "SubnetId": "subnet-12345678"
  }' \
  --instance-count 1 \
  --spot-price "0.05"
```

## 🔧 Maintenance and Operations

### Health Checks

```bash
# Application health check script
#!/bin/bash
ENDPOINTS=(
  "http://your-domain.com/health"
  "http://your-domain.com/api/health"
)

for endpoint in "${ENDPOINTS[@]}"; do
  if curl -f -s "$endpoint" > /dev/null; then
    echo "✅ $endpoint is healthy"
  else
    echo "❌ $endpoint is unhealthy"
    # Send alert notification
  fi
done
```

### Update Strategy

```bash
# Blue-green deployment script
#!/bin/bash
NEW_TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json file://new-task-definition.json \
  --query 'taskDefinition.taskDefinitionArn' --output text)

aws ecs update-service \
  --cluster archon-cluster \
  --service archon-server \
  --task-definition $NEW_TASK_DEF_ARN

# Wait for deployment to complete
aws ecs wait services-stable \
  --cluster archon-cluster \
  --services archon-server
```

### Scaling Commands

```bash
# Manual scaling
aws ecs update-service \
  --cluster archon-cluster \
  --service archon-server \
  --desired-count 5

# Check scaling status
aws ecs describe-services \
  --cluster archon-cluster \
  --services archon-server \
  --query 'services[0].{Running:runningCount,Desired:desiredCount}'
```

---

## 📚 Next Steps

- **[Monitoring Setup](../monitoring/)** - Set up comprehensive monitoring
- **[Security Hardening](../security/)** - Advanced security configuration  
- **[Performance Tuning](../performance/)** - Optimize for high performance
- **[Multi-Region Setup](../multi-region/)** - Global deployment strategy

**Need help?** Check our [troubleshooting guide](../../TROUBLESHOOTING.md) or join our [community discussions](https://github.com/coleam00/archon/discussions).