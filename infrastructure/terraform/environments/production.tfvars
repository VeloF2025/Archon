# Production Environment Configuration for Archon Phase 6 Authentication
# Optimized for high availability, security, and performance

# ============================================================================
# Environment Configuration
# ============================================================================

environment = "production"
aws_region  = "us-west-2"

# Restrict API access to known IP ranges
allowed_cidr_blocks = [
  "10.0.0.0/8",      # Internal corporate network
  "172.16.0.0/12",   # VPN ranges
  "192.168.0.0/16"   # Development networks
]

# ============================================================================
# Database Configuration (High Performance)
# ============================================================================

db_instance_class           = "db.r6g.xlarge"    # Memory-optimized for performance
db_allocated_storage        = 200                # 200GB initial storage
db_max_allocated_storage    = 2000               # Auto-scale up to 2TB
db_backup_retention_period  = 30                 # 30 days backup retention

# ============================================================================
# Redis Configuration (High Availability)
# ============================================================================

redis_node_type           = "cache.r6g.large"    # Memory-optimized Redis nodes
redis_num_cache_clusters  = 3                    # 3-node cluster for HA

# ============================================================================
# EKS Configuration (Production Grade)
# ============================================================================

kubernetes_version = "1.29"

# Production node groups with dedicated instance types
node_groups = {
  # Authentication service nodes (dedicated)
  auth_nodes = {
    instance_types = ["m6i.xlarge", "m6i.2xlarge"]
    min_size      = 3
    max_size      = 10
    desired_size  = 5
    disk_size     = 100
    disk_type     = "gp3"
    labels = {
      Purpose     = "authentication"
      Environment = "production"
      CriticalService = "true"
    }
    taints = {
      dedicated = {
        key    = "archon.io/dedicated"
        value  = "auth"
        effect = "NO_SCHEDULE"
      }
    }
  }
  
  # Monitoring and observability nodes
  monitoring_nodes = {
    instance_types = ["m6i.large", "m6i.xlarge"]
    min_size      = 2
    max_size      = 5
    desired_size  = 3
    disk_size     = 200  # More storage for metrics and logs
    disk_type     = "gp3"
    labels = {
      Purpose     = "monitoring"
      Environment = "production"
    }
    taints = {}
  }
  
  # General workload nodes
  general_nodes = {
    instance_types = ["m6i.large", "m6i.xlarge", "c6i.large"]
    min_size      = 2
    max_size      = 8
    desired_size  = 4
    disk_size     = 80
    disk_type     = "gp3"
    labels = {
      Purpose     = "general"
      Environment = "production"
    }
    taints = {}
  }
}

# ============================================================================
# Monitoring Configuration (Full Stack)
# ============================================================================

enable_monitoring           = true
monitoring_retention_days   = 90    # 90 days retention for compliance
alert_email                = "devops@archon.com"

# ============================================================================
# Security Configuration (Maximum Security)
# ============================================================================

enable_waf                 = true
enable_security_scanning   = true
enable_cross_region_backup = true
backup_region             = "us-east-1"

# ============================================================================
# Feature Flags (Production Features)
# ============================================================================

feature_flags = {
  agents_enabled     = true
  projects_enabled   = true
  validation_enabled = true
  monitoring_enabled = true
  security_scanning  = true
  audit_logging     = true
  performance_monitoring = true
  error_tracking    = true
  distributed_tracing = true
}

# ============================================================================
# Auto-scaling Configuration (Production Scaling)
# ============================================================================

auto_scaling_enabled = true
enable_spot_instances = false  # No spot instances in production

# ============================================================================
# Domain Configuration
# ============================================================================

domain_name      = "archon.com"
subdomain_prefix = ""  # Production uses root domain

# ============================================================================
# Additional Tags (Production Compliance)
# ============================================================================

additional_tags = {
  Environment     = "production"
  CostCenter     = "engineering"
  Owner          = "devops-team"
  Project        = "archon-authentication"
  Compliance     = "sox-pci-gdpr"
  BackupRequired = "true"
  MonitoringRequired = "true"
  SecurityLevel  = "high"
  DataClassification = "confidential"
  BusinessCriticality = "high"
  SLA            = "99.9"
  MaintenanceWindow = "sun-02:00-04:00-utc"
  
  # Compliance and audit tags
  SOXCompliance   = "required"
  PCICompliance   = "required"
  GDPRCompliance  = "required"
  AuditRequired   = "true"
  
  # Operational tags
  AlertingEnabled = "true"
  LoggingEnabled  = "true"
  BackupEnabled   = "true"
  
  # Security tags
  EncryptionRequired = "true"
  SecurityScanning   = "enabled"
  VulnerabilityManagement = "required"
}