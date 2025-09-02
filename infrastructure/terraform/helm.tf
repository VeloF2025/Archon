# Helm Charts and Kubernetes Add-ons for Archon Authentication System
# Production-ready monitoring, security, and operational tools

# ============================================================================
# NGINX Ingress Controller
# ============================================================================

resource "helm_release" "nginx_ingress" {
  name             = "nginx-ingress"
  repository       = "https://kubernetes.github.io/ingress-nginx"
  chart            = "ingress-nginx"
  version          = "4.8.0"
  namespace        = "ingress-nginx"
  create_namespace = true

  values = [
    yamlencode({
      controller = {
        replicaCount = var.environment == "production" ? 3 : 2
        
        service = {
          type = "LoadBalancer"
          annotations = {
            "service.beta.kubernetes.io/aws-load-balancer-type" = "nlb"
            "service.beta.kubernetes.io/aws-load-balancer-backend-protocol" = "tcp"
            "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled" = "true"
          }
        }
        
        config = {
          "use-forwarded-headers" = "true"
          "compute-full-forwarded-for" = "true"
          "use-proxy-protocol" = "false"
          "server-tokens" = "false"
          "ssl-protocols" = "TLSv1.2 TLSv1.3"
          "ssl-ciphers" = "ECDHE-ECDSA-AES128-GCM-SHA256,ECDHE-RSA-AES128-GCM-SHA256,ECDHE-ECDSA-AES256-GCM-SHA384,ECDHE-RSA-AES256-GCM-SHA384"
          "enable-real-ip" = "true"
        }
        
        metrics = {
          enabled = true
          service = {
            annotations = {
              "prometheus.io/scrape" = "true"
              "prometheus.io/port" = "10254"
            }
          }
        }
        
        resources = {
          requests = {
            cpu = "100m"
            memory = "128Mi"
          }
          limits = {
            cpu = "1000m"
            memory = "512Mi"
          }
        }
        
        autoscaling = {
          enabled = true
          minReplicas = var.environment == "production" ? 3 : 2
          maxReplicas = var.environment == "production" ? 10 : 5
          targetCPUUtilizationPercentage = 80
          targetMemoryUtilizationPercentage = 80
        }
      }
    })
  ]

  depends_on = [module.eks]
}

# ============================================================================
# Cert-Manager for SSL Certificate Management
# ============================================================================

resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  version          = "v1.13.0"
  namespace        = "cert-manager"
  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }

  set {
    name  = "global.leaderElection.namespace"
    value = "cert-manager"
  }

  values = [
    yamlencode({
      prometheus = {
        enabled = true
      }
      
      resources = {
        requests = {
          cpu = "50m"
          memory = "64Mi"
        }
        limits = {
          cpu = "200m"
          memory = "256Mi"
        }
      }
      
      securityContext = {
        runAsNonRoot = true
        seccompProfile = {
          type = "RuntimeDefault"
        }
      }
    })
  ]

  depends_on = [module.eks]
}

# ============================================================================
# Prometheus Stack (kube-prometheus-stack)
# ============================================================================

resource "helm_release" "prometheus_stack" {
  count = var.enable_monitoring ? 1 : 0
  
  name             = "prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "51.0.0"
  namespace        = "archon-monitoring"
  create_namespace = true

  values = [
    yamlencode({
      # Prometheus configuration
      prometheus = {
        prometheusSpec = {
          retention = "${var.monitoring_retention_days}d"
          retentionSize = "10GB"
          
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = "gp3"
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "50Gi"
                  }
                }
              }
            }
          }
          
          resources = {
            requests = {
              cpu = "500m"
              memory = "2Gi"
            }
            limits = {
              cpu = "2000m"
              memory = "8Gi"
            }
          }
          
          # Add scrape configs for Archon services
          additionalScrapeConfigs = [
            {
              job_name = "archon-services"
              kubernetes_sd_configs = [
                {
                  role = "pod"
                  namespaces = {
                    names = ["archon-production"]
                  }
                }
              ]
              relabel_configs = [
                {
                  source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"]
                  action = "keep"
                  regex = "true"
                },
                {
                  source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_path"]
                  action = "replace"
                  target_label = "__metrics_path__"
                  regex = "(.+)"
                }
              ]
            }
          ]
        }
      }
      
      # Grafana configuration
      grafana = {
        enabled = true
        
        adminPassword = "admin-change-me"  # Use secret in production
        
        persistence = {
          enabled = true
          storageClassName = "gp3"
          size = "10Gi"
        }
        
        resources = {
          requests = {
            cpu = "100m"
            memory = "256Mi"
          }
          limits = {
            cpu = "500m"
            memory = "1Gi"
          }
        }
        
        # Pre-configured dashboards
        dashboardProviders = {
          "dashboardproviders.yaml" = {
            apiVersion = 1
            providers = [
              {
                name = "archon-dashboards"
                orgId = 1
                folder = "Archon"
                type = "file"
                disableDeletion = false
                updateIntervalSeconds = 30
                allowUiUpdates = true
                options = {
                  path = "/var/lib/grafana/dashboards/archon"
                }
              }
            ]
          }
        }
        
        # Archon-specific dashboards
        dashboards = {
          archon = {
            "archon-overview" = {
              url = "https://raw.githubusercontent.com/archon/dashboards/main/overview.json"
              datasource = "Prometheus"
            }
            "archon-authentication" = {
              url = "https://raw.githubusercontent.com/archon/dashboards/main/authentication.json"
              datasource = "Prometheus"
            }
          }
        }
        
        # Data sources
        datasources = {
          "datasources.yaml" = {
            apiVersion = 1
            datasources = [
              {
                name = "Prometheus"
                type = "prometheus"
                url = "http://prometheus-stack-kube-prom-prometheus:9090"
                access = "proxy"
                isDefault = true
              },
              {
                name = "Loki"
                type = "loki"
                url = "http://loki:3100"
                access = "proxy"
              }
            ]
          }
        }
      }
      
      # AlertManager configuration
      alertmanager = {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = "gp3"
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
              }
            }
          }
          
          resources = {
            requests = {
              cpu = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu = "500m"
              memory = "1Gi"
            }
          }
        }
        
        config = {
          global = {
            smtp_smarthost = "smtp.company.com:587"
            smtp_from = "alerts@archon.com"
          }
          
          route = {
            group_by = ["alertname", "cluster", "service"]
            group_wait = "10s"
            group_interval = "10s"
            repeat_interval = "1h"
            receiver = "web.hook"
            routes = [
              {
                match = {
                  alertname = "Watchdog"
                }
                receiver = "null"
              },
              {
                match = {
                  severity = "critical"
                }
                receiver = "critical-alerts"
                group_wait = "0s"
                repeat_interval = "5m"
              }
            ]
          }
          
          receivers = [
            {
              name = "null"
            },
            {
              name = "web.hook"
              slack_configs = [
                {
                  api_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
                  channel = "#alerts"
                  title = "Archon Alert - {{ .GroupLabels.alertname }}"
                  text = "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                }
              ]
            },
            {
              name = "critical-alerts"
              email_configs = [
                {
                  to = var.alert_email
                  subject = "[CRITICAL] Archon Authentication System Alert"
                  body = "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                }
              ]
              slack_configs = [
                {
                  api_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
                  channel = "#critical-alerts"
                  title = "🚨 CRITICAL: {{ .GroupLabels.alertname }}"
                  text = "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                }
              ]
            }
          ]
        }
      }
    })
  ]

  depends_on = [module.eks]
}

# ============================================================================
# Loki for Log Aggregation
# ============================================================================

resource "helm_release" "loki" {
  count = var.enable_monitoring ? 1 : 0
  
  name             = "loki"
  repository       = "https://grafana.github.io/helm-charts"
  chart            = "loki"
  version          = "5.36.0"
  namespace        = "archon-monitoring"
  create_namespace = true

  values = [
    yamlencode({
      deploymentMode = "SimpleScalable"
      
      loki = {
        auth_enabled = false
        
        storage = {
          type = "s3"
          s3 = {
            s3 = "s3://us-west-2"
            bucketnames = aws_s3_bucket.loki_storage[0].bucket
            region = var.aws_region
            access_key_id = aws_iam_access_key.loki[0].id
            secret_access_key = aws_iam_access_key.loki[0].secret
          }
        }
        
        limits_config = {
          retention_period = "${var.monitoring_retention_days * 24}h"
          max_query_series = 100000
        }
        
        chunk_store_config = {
          max_look_back_period = "${var.monitoring_retention_days * 24}h"
        }
      }
      
      write = {
        replicas = var.environment == "production" ? 3 : 2
        
        resources = {
          requests = {
            cpu = "300m"
            memory = "512Mi"
          }
          limits = {
            cpu = "1000m"
            memory = "2Gi"
          }
        }
      }
      
      read = {
        replicas = var.environment == "production" ? 3 : 2
        
        resources = {
          requests = {
            cpu = "300m"
            memory = "512Mi"
          }
          limits = {
            cpu = "1000m"
            memory = "2Gi"
          }
        }
      }
      
      backend = {
        replicas = var.environment == "production" ? 3 : 2
        
        resources = {
          requests = {
            cpu = "300m"
            memory = "1Gi"
          }
          limits = {
            cpu = "1000m"
            memory = "4Gi"
          }
        }
      }
    })
  ]

  depends_on = [module.eks, aws_s3_bucket.loki_storage]
}

# S3 bucket for Loki storage
resource "aws_s3_bucket" "loki_storage" {
  count = var.enable_monitoring ? 1 : 0
  
  bucket = "${local.name}-loki-storage-${random_string.bucket_suffix.result}"

  tags = local.tags
}

resource "aws_s3_bucket_versioning" "loki_storage" {
  count = var.enable_monitoring ? 1 : 0
  
  bucket = aws_s3_bucket.loki_storage[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "loki_storage" {
  count = var.enable_monitoring ? 1 : 0
  
  bucket = aws_s3_bucket.loki_storage[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM for Loki S3 access
resource "aws_iam_user" "loki" {
  count = var.enable_monitoring ? 1 : 0
  
  name = "${local.name}-loki"
  path = "/"

  tags = local.tags
}

resource "aws_iam_access_key" "loki" {
  count = var.enable_monitoring ? 1 : 0
  
  user = aws_iam_user.loki[0].name
}

resource "aws_iam_user_policy" "loki" {
  count = var.enable_monitoring ? 1 : 0
  
  name = "${local.name}-loki-policy"
  user = aws_iam_user.loki[0].name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.loki_storage[0].arn,
          "${aws_s3_bucket.loki_storage[0].arn}/*"
        ]
      }
    ]
  })
}

# ============================================================================
# External Secrets Operator
# ============================================================================

resource "helm_release" "external_secrets" {
  name             = "external-secrets"
  repository       = "https://charts.external-secrets.io"
  chart            = "external-secrets"
  version          = "0.9.0"
  namespace        = "external-secrets-system"
  create_namespace = true

  values = [
    yamlencode({
      installCRDs = true
      
      replicaCount = var.environment == "production" ? 2 : 1
      
      resources = {
        requests = {
          cpu = "50m"
          memory = "64Mi"
        }
        limits = {
          cpu = "200m"
          memory = "256Mi"
        }
      }
      
      securityContext = {
        runAsNonRoot = true
        runAsUser = 65534
        seccompProfile = {
          type = "RuntimeDefault"
        }
      }
      
      webhook = {
        resources = {
          requests = {
            cpu = "50m"
            memory = "64Mi"
          }
          limits = {
            cpu = "200m"
            memory = "256Mi"
          }
        }
      }
      
      certController = {
        resources = {
          requests = {
            cpu = "50m"
            memory = "64Mi"
          }
          limits = {
            cpu = "200m"
            memory = "256Mi"
          }
        }
      }
    })
  ]

  depends_on = [module.eks]
}

# ============================================================================
# Cluster Autoscaler
# ============================================================================

resource "helm_release" "cluster_autoscaler" {
  count = var.auto_scaling_enabled ? 1 : 0
  
  name             = "cluster-autoscaler"
  repository       = "https://kubernetes.github.io/autoscaler"
  chart            = "cluster-autoscaler"
  version          = "9.29.0"
  namespace        = "kube-system"

  set {
    name  = "autoDiscovery.clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "awsRegion"
    value = var.aws_region
  }

  values = [
    yamlencode({
      rbac = {
        serviceAccount = {
          annotations = {
            "eks.amazonaws.com/role-arn" = aws_iam_role.cluster_autoscaler[0].arn
          }
        }
      }
      
      resources = {
        requests = {
          cpu = "100m"
          memory = "300Mi"
        }
        limits = {
          cpu = "500m"
          memory = "600Mi"
        }
      }
      
      extraArgs = {
        logtostderr = true
        stderrthreshold = "info"
        v = 4
        "cluster-name" = module.eks.cluster_name
        "balance-similar-node-groups" = true
        "skip-nodes-with-system-pods" = false
        "scale-down-enabled" = true
        "scale-down-delay-after-add" = "10m"
        "scale-down-utilization-threshold" = "0.5"
      }
      
      # Node selector for system nodes
      nodeSelector = {
        "kubernetes.io/arch" = "amd64"
      }
      
      # Security context
      securityContext = {
        runAsNonRoot = true
        runAsUser = 65534
        runAsGroup = 65534
        seccompProfile = {
          type = "RuntimeDefault"
        }
      }
    })
  ]

  depends_on = [module.eks]
}

# IAM role for cluster autoscaler
resource "aws_iam_role" "cluster_autoscaler" {
  count = var.auto_scaling_enabled ? 1 : 0
  
  name = "${local.name}-cluster-autoscaler"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:kube-system:cluster-autoscaler"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "cluster_autoscaler" {
  count = var.auto_scaling_enabled ? 1 : 0
  
  name = "${local.name}-cluster-autoscaler"
  role = aws_iam_role.cluster_autoscaler[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeTags",
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeLaunchTemplateVersions"
        ]
        Resource = "*"
      }
    ]
  })
}