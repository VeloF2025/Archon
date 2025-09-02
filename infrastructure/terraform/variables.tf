# Terraform Variables for Archon Phase 6 Authentication Infrastructure

# ============================================================================
# Environment Configuration
# ============================================================================

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  
  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
  
  validation {
    condition = can(regex("^[a-z]{2}-[a-z]+-[0-9]{1}$", var.environment))
    error_message = "AWS region must be in valid format (e.g., us-west-2)."
  }
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access EKS API server"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

# ============================================================================
# Database Configuration
# ============================================================================

variable "db_instance_class" {
  description = "RDS instance class for PostgreSQL database"
  type        = string
  default     = "db.t3.medium"
  
  validation {
    condition = can(regex("^db\\.", var.db_instance_class))
    error_message = "Database instance class must be a valid RDS instance type."
  }
}

variable "db_allocated_storage" {
  description = "Initial storage allocation for RDS instance (GB)"
  type        = number
  default     = 100
  
  validation {
    condition     = var.db_allocated_storage >= 20 && var.db_allocated_storage <= 65536
    error_message = "Database allocated storage must be between 20 and 65536 GB."
  }
}

variable "db_max_allocated_storage" {
  description = "Maximum storage allocation for auto-scaling (GB)"
  type        = number
  default     = 1000
  
  validation {
    condition     = var.db_max_allocated_storage >= var.db_allocated_storage
    error_message = "Maximum allocated storage must be greater than or equal to allocated storage."
  }
}

variable "db_backup_retention_period" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 7
  
  validation {
    condition     = var.db_backup_retention_period >= 0 && var.db_backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

# ============================================================================
# Redis Configuration
# ============================================================================

variable "redis_node_type" {
  description = "ElastiCache node type for Redis cluster"
  type        = string
  default     = "cache.t4g.medium"
  
  validation {
    condition = can(regex("^cache\\.", var.redis_node_type))
    error_message = "Redis node type must be a valid ElastiCache instance type."
  }
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters in Redis replication group"
  type        = number
  default     = 2
  
  validation {
    condition     = var.redis_num_cache_clusters >= 2 && var.redis_num_cache_clusters <= 6
    error_message = "Number of Redis cache clusters must be between 2 and 6."
  }
}

# ============================================================================
# EKS Configuration
# ============================================================================

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.29"
  
  validation {
    condition = can(regex("^1\\.(2[8-9]|[3-9][0-9])$", var.kubernetes_version))
    error_message = "Kubernetes version must be 1.28 or higher."
  }
}

variable "node_groups" {
  description = "Configuration for EKS managed node groups"
  type = map(object({
    instance_types = list(string)
    min_size      = number
    max_size      = number
    desired_size  = number
    disk_size     = number
    disk_type     = string
    labels        = map(string)
    taints        = map(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  
  default = {
    auth_nodes = {
      instance_types = ["m6i.large", "m6i.xlarge"]
      min_size      = 2
      max_size      = 6
      desired_size  = 3
      disk_size     = 50
      disk_type     = "gp3"
      labels = {
        Purpose = "authentication"
      }
      taints = {
        dedicated = {
          key    = "archon.io/dedicated"
          value  = "auth"
          effect = "NO_SCHEDULE"
        }
      }
    }
    
    monitoring_nodes = {
      instance_types = ["m6i.large"]
      min_size      = 1
      max_size      = 3
      desired_size  = 2
      disk_size     = 100
      disk_type     = "gp3"
      labels = {
        Purpose = "monitoring"
      }
      taints = {}
    }
  }
}

# ============================================================================
# Monitoring Configuration
# ============================================================================

variable "enable_monitoring" {
  description = "Enable comprehensive monitoring stack (Prometheus, Grafana, AlertManager)"
  type        = bool
  default     = true
}

variable "monitoring_retention_days" {
  description = "Number of days to retain monitoring data"
  type        = number
  default     = 30
  
  validation {
    condition     = var.monitoring_retention_days >= 1 && var.monitoring_retention_days <= 365
    error_message = "Monitoring retention must be between 1 and 365 days."
  }
}

variable "alert_email" {
  description = "Email address for critical alerts"
  type        = string
  default     = "devops@archon.com"
  
  validation {
    condition = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
    error_message = "Alert email must be a valid email address."
  }
}

# ============================================================================
# Security Configuration
# ============================================================================

variable "enable_waf" {
  description = "Enable AWS WAF for application protection"
  type        = bool
  default     = true
}

variable "enable_security_scanning" {
  description = "Enable container image security scanning"
  type        = bool
  default     = true
}

variable "certificate_arn" {
  description = "ARN of SSL certificate for HTTPS (if using existing certificate)"
  type        = string
  default     = ""
}

# ============================================================================
# Cost Optimization
# ============================================================================

variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = false
}

variable "auto_scaling_enabled" {
  description = "Enable cluster auto-scaling"
  type        = bool
  default     = true
}

# ============================================================================
# Backup and Disaster Recovery
# ============================================================================

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup for disaster recovery"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Secondary AWS region for cross-region backups"
  type        = string
  default     = "us-east-1"
}

# ============================================================================
# Feature Flags
# ============================================================================

variable "feature_flags" {
  description = "Feature flags for Archon application"
  type        = map(bool)
  default = {
    agents_enabled     = true
    projects_enabled   = true
    validation_enabled = true
    monitoring_enabled = true
    security_scanning  = true
  }
}

# ============================================================================
# Domain Configuration
# ============================================================================

variable "domain_name" {
  description = "Domain name for Archon application"
  type        = string
  default     = "archon.com"
}

variable "subdomain_prefix" {
  description = "Subdomain prefix for environment"
  type        = string
  default     = ""
}

# ============================================================================
# Resource Tags
# ============================================================================

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}