# Staging Environment Configuration for Archon Phase 6 Authentication
# Cost-optimized for testing and validation

# ============================================================================
# Environment Configuration
# ============================================================================

environment = "staging"
aws_region  = "us-west-2"

# More open access for staging testing
allowed_cidr_blocks = [
  "0.0.0.0/0"  # Open access for staging
]

# ============================================================================
# Database Configuration (Cost-Optimized)
# ============================================================================

db_instance_class           = "db.t3.medium"     # Smaller instance for staging
db_allocated_storage        = 50                 # 50GB storage
db_max_allocated_storage    = 200                # Auto-scale up to 200GB
db_backup_retention_period  = 7                  # 7 days backup retention

# ============================================================================
# Redis Configuration (Minimal HA)
# ============================================================================

redis_node_type           = "cache.t4g.medium"   # Smaller Redis nodes
redis_num_cache_clusters  = 2                    # 2-node cluster

# ============================================================================
# EKS Configuration (Development-Friendly)
# ============================================================================

kubernetes_version = "1.29"

# Staging node groups (smaller, cost-optimized)
node_groups = {
  # General purpose nodes for all workloads
  general_nodes = {
    instance_types = ["t3.medium", "t3.large"]
    min_size      = 1
    max_size      = 4
    desired_size  = 2
    disk_size     = 50
    disk_type     = "gp3"
    labels = {
      Purpose     = "general"
      Environment = "staging"
    }
    taints = {}
  }
  
  # Monitoring nodes (minimal)
  monitoring_nodes = {
    instance_types = ["t3.medium"]
    min_size      = 1
    max_size      = 2
    desired_size  = 1
    disk_size     = 50
    disk_type     = "gp3"
    labels = {
      Purpose     = "monitoring"
      Environment = "staging"
    }
    taints = {}
  }
}

# ============================================================================
# Monitoring Configuration (Basic)
# ============================================================================

enable_monitoring           = true
monitoring_retention_days   = 14    # 14 days retention for staging
alert_email                = "staging-alerts@archon.com"

# ============================================================================
# Security Configuration (Relaxed for Testing)
# ============================================================================

enable_waf                 = false  # Disable WAF for staging
enable_security_scanning   = true   # Keep security scanning enabled
enable_cross_region_backup = false  # No cross-region backup for staging
backup_region             = "us-east-1"

# ============================================================================
# Feature Flags (All Features Enabled for Testing)
# ============================================================================

feature_flags = {
  agents_enabled     = true
  projects_enabled   = true
  validation_enabled = true
  monitoring_enabled = true
  security_scanning  = true
  audit_logging     = false  # Disable audit logging in staging
  performance_monitoring = true
  error_tracking    = true
  distributed_tracing = true
  debug_mode        = true   # Enable debug features
  experimental_features = true
}

# ============================================================================
# Auto-scaling Configuration (Cost-Conscious)
# ============================================================================

auto_scaling_enabled = true
enable_spot_instances = true  # Use spot instances for cost savings

# ============================================================================
# Domain Configuration
# ============================================================================

domain_name      = "archon.com"
subdomain_prefix = "staging-"  # staging-api.archon.com

# ============================================================================
# Additional Tags (Staging Environment)
# ============================================================================

additional_tags = {
  Environment     = "staging"
  CostCenter     = "engineering"
  Owner          = "devops-team"
  Project        = "archon-authentication"
  Purpose        = "testing-validation"
  AutoShutdown   = "enabled"  # Enable auto-shutdown for cost savings
  
  # Development tags
  DeveloperAccess = "enabled"
  TestingEnabled  = "true"
  DebuggingEnabled = "true"
  
  # Cost optimization tags
  CostOptimization = "aggressive"
  SpotInstancesEnabled = "true"
  AutoScalingEnabled = "true"
  
  # Lifecycle tags
  ScheduledShutdown = "weekends"  # Shut down on weekends
  BusinessHours     = "mon-fri-9-17-pst"
  
  # Security tags (relaxed for staging)
  SecurityLevel      = "medium"
  DataClassification = "test-data"
  ComplianceRequired = "false"
}