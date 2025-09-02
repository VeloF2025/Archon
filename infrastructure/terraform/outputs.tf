# Terraform Outputs for Archon Phase 6 Authentication Infrastructure
# Provides essential information for CI/CD pipeline integration

# ============================================================================
# EKS Cluster Outputs
# ============================================================================

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "The Kubernetes server version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = module.eks.cluster_platform_version
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_security_group_id" {
  description = "ID of the cluster security group"
  value       = module.eks.cluster_security_group_id
}

output "node_security_group_id" {
  description = "ID of the EKS node shared security group"
  value       = module.eks.node_security_group_id
}

# ============================================================================
# Network Infrastructure Outputs
# ============================================================================

output "vpc_id" {
  description = "ID of the VPC where the cluster is deployed"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "nat_gateway_ids" {
  description = "List of IDs of NAT Gateways"
  value       = module.vpc.natgw_ids
}

# ============================================================================
# Load Balancer Outputs
# ============================================================================

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.main.arn
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

# ============================================================================
# Database Outputs
# ============================================================================

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.identifier
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.main.arn
}

output "db_subnet_group_name" {
  description = "Name of the DB subnet group"
  value       = aws_db_subnet_group.main.name
}

output "db_security_group_id" {
  description = "ID of the database security group"
  value       = aws_security_group.rds.id
}

# ============================================================================
# Cache Outputs
# ============================================================================

output "redis_endpoint" {
  description = "ElastiCache Redis primary endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.main.port
}

output "redis_configuration_endpoint" {
  description = "ElastiCache Redis configuration endpoint"
  value       = aws_elasticache_replication_group.main.configuration_endpoint_address
}

output "redis_auth_token" {
  description = "ElastiCache Redis auth token"
  value       = random_password.redis_password.result
  sensitive   = true
}

# ============================================================================
# Security Outputs
# ============================================================================

output "kms_key_id" {
  description = "ID of the KMS key used for EKS encryption"
  value       = aws_kms_key.eks.key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key used for EKS encryption"
  value       = aws_kms_key.eks.arn
}

output "eks_admin_role_arn" {
  description = "ARN of the EKS admin role"
  value       = aws_iam_role.eks_admin.arn
}

# ============================================================================
# Monitoring Outputs
# ============================================================================

output "s3_logs_bucket" {
  description = "S3 bucket name for ALB logs"
  value       = aws_s3_bucket.alb_logs.bucket
}

output "loki_storage_bucket" {
  description = "S3 bucket name for Loki log storage"
  value       = var.enable_monitoring ? aws_s3_bucket.loki_storage[0].bucket : null
}

# ============================================================================
# Application Configuration Outputs
# ============================================================================

output "application_config" {
  description = "Application configuration for CI/CD pipeline"
  value = {
    environment = var.environment
    namespace   = "archon-${var.environment}"
    
    # Service endpoints
    endpoints = {
      api      = "https://${var.subdomain_prefix}api.${var.domain_name}"
      frontend = "https://${var.subdomain_prefix}app.${var.domain_name}"
      admin    = "https://${var.subdomain_prefix}admin.${var.domain_name}"
    }
    
    # Infrastructure details
    infrastructure = {
      cluster_name = module.eks.cluster_name
      vpc_id       = module.vpc.vpc_id
      region       = var.aws_region
    }
    
    # Feature flags
    features = var.feature_flags
  }
  sensitive = false
}

# ============================================================================
# CI/CD Integration Outputs
# ============================================================================

output "cicd_config" {
  description = "Configuration for CI/CD pipeline integration"
  value = {
    # Kubernetes connection
    kubernetes = {
      cluster_name     = module.eks.cluster_name
      cluster_endpoint = module.eks.cluster_endpoint
      region          = var.aws_region
      namespace       = "archon-${var.environment}"
    }
    
    # Container registry
    registry = {
      url_prefix = "ghcr.io/${var.environment}"
      images = {
        server    = "archon-server"
        mcp       = "archon-mcp"
        agents    = "archon-agents"
        validator = "archon-validator"
        frontend  = "archon-frontend"
      }
    }
    
    # Deployment configuration
    deployment = {
      strategy         = "blue-green"
      health_check_url = "https://${var.subdomain_prefix}api.${var.domain_name}/health"
      timeout_seconds  = 600
      rollback_enabled = true
    }
    
    # Monitoring endpoints
    monitoring = {
      prometheus_url = "https://${var.subdomain_prefix}admin.${var.domain_name}/prometheus"
      grafana_url    = "https://${var.subdomain_prefix}admin.${var.domain_name}/grafana"
      alerts_enabled = var.enable_monitoring
    }
  }
  sensitive = false
}

# ============================================================================
# Environment-Specific Outputs
# ============================================================================

output "environment_secrets" {
  description = "Environment-specific secrets for application configuration"
  value = {
    database_url = "postgresql://archon:${random_password.db_password.result}@${aws_db_instance.main.endpoint}:${aws_db_instance.main.port}/archon"
    redis_url    = "redis://:${random_password.redis_password.result}@${aws_elasticache_replication_group.main.primary_endpoint_address}:${aws_elasticache_replication_group.main.port}"
  }
  sensitive = true
}

# ============================================================================
# Cost Optimization Outputs
# ============================================================================

output "cost_optimization" {
  description = "Cost optimization recommendations and current resource allocation"
  value = {
    estimated_monthly_cost = {
      eks_cluster      = "~$73/month (control plane)"
      node_groups      = "~$200-600/month (depending on instance types and scaling)"
      rds_postgres     = "~$150-500/month (depending on instance class)"
      elasticache      = "~$100-300/month (depending on node type)"
      load_balancer    = "~$25/month"
      data_transfer    = "~$50-200/month (depending on traffic)"
      storage          = "~$50-150/month (EBS, S3, backups)"
      monitoring       = "~$50-100/month (CloudWatch, logs)"
    }
    
    recommendations = [
      "Use Spot instances for non-critical workloads",
      "Enable cluster autoscaler for cost optimization",
      "Set up automated scaling policies",
      "Monitor and optimize storage usage",
      "Use Reserved Instances for predictable workloads"
    ]
    
    cost_tags = {
      Environment = var.environment
      Project     = "Archon"
      Component   = "Authentication"
      Owner       = "DevOps Team"
    }
  }
}

# ============================================================================
# Disaster Recovery Outputs
# ============================================================================

output "disaster_recovery" {
  description = "Disaster recovery configuration and procedures"
  value = {
    backup_configuration = {
      rds_automated_backups    = "30 days retention"
      redis_snapshots         = "7 days retention"
      kubernetes_etcd_backups = "Managed by AWS"
      application_data_backup = "Daily automated backups"
    }
    
    recovery_procedures = {
      rto_target = "< 4 hours"  # Recovery Time Objective
      rpo_target = "< 1 hour"   # Recovery Point Objective
      
      backup_locations = [
        aws_s3_bucket.alb_logs.bucket,
        var.enable_monitoring ? aws_s3_bucket.loki_storage[0].bucket : "N/A"
      ]
      
      cross_region_backup = var.enable_cross_region_backup ? var.backup_region : "disabled"
    }
    
    runbook_urls = {
      disaster_recovery = "https://runbooks.archon.com/disaster-recovery"
      database_restore  = "https://runbooks.archon.com/database-restore"
      cluster_recovery  = "https://runbooks.archon.com/cluster-recovery"
    }
  }
}

# ============================================================================
# Security Compliance Outputs
# ============================================================================

output "security_compliance" {
  description = "Security compliance status and configurations"
  value = {
    encryption = {
      at_rest = {
        database = "AES-256 with AWS KMS"
        cache    = "AES-256 with ElastiCache encryption"
        storage  = "EBS encryption enabled"
      }
      in_transit = {
        api_calls     = "TLS 1.3"
        database      = "SSL/TLS enforced"
        cache         = "TLS encryption enabled"
        internal_mesh = "mTLS with service mesh"
      }
    }
    
    access_control = {
      kubernetes_rbac = "Enabled with least privilege"
      aws_iam         = "Service-specific roles with minimal permissions"
      network_policies = "Pod-to-pod communication restricted"
    }
    
    compliance_frameworks = [
      "SOC 2 Type II",
      "ISO 27001",
      "PCI DSS Level 1",
      "GDPR",
      "HIPAA (if healthcare data)"
    ]
    
    security_monitoring = {
      falco_runtime_security = "Enabled"
      prometheus_metrics     = "Security metrics collection"
      audit_logging         = "Kubernetes audit logs enabled"
      access_logging        = "ALB access logs to S3"
    }
  }
}