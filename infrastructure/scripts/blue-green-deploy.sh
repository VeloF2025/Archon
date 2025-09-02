#!/bin/bash

# Blue-Green Deployment Script for Archon Phase 6 Authentication System
# Automated zero-downtime deployment with health checks and rollback

set -euo pipefail

# ============================================================================
# Configuration and Variables
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-archon-production}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
TRAFFIC_SHIFT_TIMEOUT="${TRAFFIC_SHIFT_TIMEOUT:-600}"
ROLLBACK_TIMEOUT="${ROLLBACK_TIMEOUT:-180}"

# Slack webhook for notifications
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

notify_slack() {
    local message="$1"
    local color="${2:-good}"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"$color\",\"text\":\"$message\"}]}" \
            "$SLACK_WEBHOOK" || true
    fi
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    if ! kubectl auth can-i create deployments --namespace="$NAMESPACE" &> /dev/null; then
        error "Insufficient permissions to deploy to namespace $NAMESPACE"
    fi
    
    success "Prerequisites check passed"
}

get_current_environment() {
    local current_env
    current_env=$(kubectl get configmap blue-green-deployment-config \
        -n "$NAMESPACE" \
        -o jsonpath='{.data.active-environment}' 2>/dev/null || echo "blue")
    echo "$current_env"
}

get_target_environment() {
    local current_env="$1"
    if [ "$current_env" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# ============================================================================
# Deployment Functions
# ============================================================================

deploy_green_environment() {
    local target_env="$1"
    
    log "Deploying to $target_env environment..."
    
    # Scale up target environment
    kubectl patch deployment "archon-auth-server-$target_env" \
        -n "$NAMESPACE" \
        -p '{"spec":{"replicas":3}}'
    
    # Update image tag
    kubectl set image deployment/"archon-auth-server-$target_env" \
        archon-server="ghcr.io/archon/archon-server:$IMAGE_TAG" \
        -n "$NAMESPACE"
    
    log "Waiting for $target_env deployment to be ready..."
    
    if ! kubectl rollout status deployment/"archon-auth-server-$target_env" \
        -n "$NAMESPACE" \
        --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        error "Deployment to $target_env environment failed"
    fi
    
    success "$target_env environment deployed successfully"
}

run_health_checks() {
    local target_env="$1"
    
    log "Running comprehensive health checks for $target_env environment..."
    
    # Create health check job
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: health-check-$target_env-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: health-checker
        image: curlimages/curl:latest
        command: ["/bin/sh"]
        args:
        - -c
        - |
          set -e
          SERVICE_URL="http://archon-server-$target_env:8181"
          
          echo "Testing basic health endpoint..."
          curl -f "\$SERVICE_URL/health"
          
          echo "Testing readiness endpoint..."
          curl -f "\$SERVICE_URL/ready"
          
          echo "Testing authentication status..."
          curl -f "\$SERVICE_URL/api/auth/status"
          
          echo "Testing MCP connectivity..."
          curl -f "\$SERVICE_URL/api/mcp/health"
          
          echo "All health checks passed!"
        env:
        - name: TARGET_ENVIRONMENT
          value: "$target_env"
EOF
    
    # Wait for health check job to complete
    local job_name
    job_name=$(kubectl get jobs -n "$NAMESPACE" -l app=archon-server,component=health-check --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
    
    if ! kubectl wait --for=condition=complete job/"$job_name" \
        -n "$NAMESPACE" \
        --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        
        # Show job logs for debugging
        kubectl logs job/"$job_name" -n "$NAMESPACE" || true
        error "Health checks failed for $target_env environment"
    fi
    
    # Clean up health check job
    kubectl delete job "$job_name" -n "$NAMESPACE" --ignore-not-found=true
    
    success "Health checks passed for $target_env environment"
}

perform_canary_deployment() {
    local target_env="$1"
    
    log "Starting canary deployment to $target_env environment..."
    
    # Create canary ingress rule (10% traffic to new environment)
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: archon-canary-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"
    nginx.ingress.kubernetes.io/canary-by-header: "X-Canary"
    nginx.ingress.kubernetes.io/canary-by-header-value: "true"
spec:
  rules:
  - host: api.archon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: archon-server-$target_env
            port:
              number: 8181
EOF
    
    log "Canary deployment active - 10% traffic to $target_env"
    
    # Monitor canary for 5 minutes
    log "Monitoring canary deployment for 5 minutes..."
    sleep 300
    
    # Check error rates during canary
    local error_rate
    error_rate=$(kubectl exec -n archon-monitoring deployment/prometheus-server -- \
        promtool query instant \
        'rate(archon_http_requests_total{status=~"5..",environment="'$target_env'"}[5m]) / rate(archon_http_requests_total{environment="'$target_env'"}[5m]) * 100' \
        | grep -o '[0-9.]*' | head -1 || echo "0")
    
    if (( $(echo "$error_rate > 1.0" | bc -l) )); then
        warning "High error rate detected during canary: $error_rate%"
        return 1
    fi
    
    success "Canary deployment successful - error rate: $error_rate%"
    return 0
}

switch_traffic() {
    local target_env="$1"
    local current_env="$2"
    
    log "Switching traffic from $current_env to $target_env..."
    
    # Update active service selector
    kubectl patch service archon-server-active \
        -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"environment\":\"$target_env\"}}}"
    
    # Update ingress to point to target environment
    kubectl patch ingress archon-ingress \
        -n "$NAMESPACE" \
        -p "{\"spec\":{\"rules\":[{\"host\":\"api.archon.com\",\"http\":{\"paths\":[{\"path\":\"/\",\"pathType\":\"Prefix\",\"backend\":{\"service\":{\"name\":\"archon-server-$target_env\",\"port\":{\"number\":8181}}}}]}}]}}"
    
    # Remove canary ingress
    kubectl delete ingress archon-canary-ingress -n "$NAMESPACE" --ignore-not-found=true
    
    # Update deployment config
    kubectl patch configmap blue-green-deployment-config \
        -n "$NAMESPACE" \
        -p "{\"data\":{\"active-environment\":\"$target_env\"}}"
    
    log "Traffic switched to $target_env environment"
    
    # Wait for traffic to stabilize
    sleep 30
    
    # Verify the switch worked
    if ! curl -f "https://api.archon.com/health" &> /dev/null; then
        error "Traffic switch verification failed"
    fi
    
    success "Traffic switch completed and verified"
}

scale_down_old_environment() {
    local old_env="$1"
    
    log "Scaling down $old_env environment..."
    
    # Scale down old environment
    kubectl patch deployment "archon-auth-server-$old_env" \
        -n "$NAMESPACE" \
        -p '{"spec":{"replicas":0}}'
    
    # Wait for scale down
    kubectl rollout status deployment/"archon-auth-server-$old_env" \
        -n "$NAMESPACE" \
        --timeout="60s"
    
    success "$old_env environment scaled down"
}

perform_rollback() {
    local rollback_env="$1"
    
    warning "Performing rollback to $rollback_env environment..."
    
    # Immediately switch traffic back
    kubectl patch service archon-server-active \
        -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"environment\":\"$rollback_env\"}}}"
    
    # Update ingress
    kubectl patch ingress archon-ingress \
        -n "$NAMESPACE" \
        -p "{\"spec\":{\"rules\":[{\"host\":\"api.archon.com\",\"http\":{\"paths\":[{\"path\":\"/\",\"pathType\":\"Prefix\",\"backend\":{\"service\":{\"name\":\"archon-server-$rollback_env\",\"port\":{\"number\":8181}}}}]}}]}}"
    
    # Ensure rollback environment is scaled up
    kubectl patch deployment "archon-auth-server-$rollback_env" \
        -n "$NAMESPACE" \
        -p '{"spec":{"replicas":3}}'
    
    # Wait for rollback environment to be ready
    kubectl rollout status deployment/"archon-auth-server-$rollback_env" \
        -n "$NAMESPACE" \
        --timeout="${ROLLBACK_TIMEOUT}s"
    
    # Update config
    kubectl patch configmap blue-green-deployment-config \
        -n "$NAMESPACE" \
        -p "{\"data\":{\"active-environment\":\"$rollback_env\"}}"
    
    success "Rollback to $rollback_env completed"
    
    notify_slack "🔄 Archon rollback to $rollback_env environment completed" "warning"
}

# ============================================================================
# Main Deployment Function
# ============================================================================

main() {
    local action="${1:-deploy}"
    
    case "$action" in
        "deploy")
            perform_blue_green_deployment
            ;;
        "rollback")
            perform_emergency_rollback
            ;;
        "status")
            show_deployment_status
            ;;
        "health-check")
            run_standalone_health_check "${2:-green}"
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status|health-check [environment]}"
            echo ""
            echo "Commands:"
            echo "  deploy        - Perform blue-green deployment"
            echo "  rollback      - Emergency rollback to previous environment"
            echo "  status        - Show current deployment status"
            echo "  health-check  - Run health check for specified environment"
            exit 1
            ;;
    esac
}

perform_blue_green_deployment() {
    log "Starting Blue-Green Deployment for Archon Phase 6 Authentication..."
    
    check_prerequisites
    
    local current_env
    local target_env
    
    current_env=$(get_current_environment)
    target_env=$(get_target_environment "$current_env")
    
    log "Current environment: $current_env"
    log "Target environment: $target_env"
    log "Image tag: $IMAGE_TAG"
    
    notify_slack "🚀 Starting Blue-Green deployment to $target_env environment (tag: $IMAGE_TAG)"
    
    # Step 1: Deploy to target environment
    deploy_green_environment "$target_env"
    
    # Step 2: Run comprehensive health checks
    run_health_checks "$target_env"
    
    # Step 3: Perform canary deployment
    if perform_canary_deployment "$target_env"; then
        log "Canary deployment successful, proceeding with full traffic switch"
    else
        warning "Canary deployment failed, rolling back..."
        perform_rollback "$current_env"
        error "Deployment failed during canary phase"
    fi
    
    # Step 4: Switch all traffic
    switch_traffic "$target_env" "$current_env"
    
    # Step 5: Final verification
    log "Running final verification checks..."
    sleep 60
    
    # Check error rates post-deployment
    local post_deploy_error_rate
    post_deploy_error_rate=$(kubectl exec -n archon-monitoring deployment/prometheus-server -- \
        promtool query instant \
        'rate(archon_http_requests_total{status=~"5.."}[5m]) / rate(archon_http_requests_total[5m]) * 100' \
        | grep -o '[0-9.]*' | head -1 || echo "0")
    
    if (( $(echo "$post_deploy_error_rate > 1.0" | bc -l) )); then
        warning "High error rate detected post-deployment: $post_deploy_error_rate%"
        perform_rollback "$current_env"
        error "Deployment failed post-deployment verification"
    fi
    
    # Step 6: Scale down old environment
    scale_down_old_environment "$current_env"
    
    success "Blue-Green deployment completed successfully!"
    success "Active environment: $target_env"
    success "Error rate: $post_deploy_error_rate%"
    
    notify_slack "✅ Blue-Green deployment completed successfully! Active: $target_env (error rate: $post_deploy_error_rate%)" "good"
}

perform_emergency_rollback() {
    log "Starting emergency rollback..."
    
    check_prerequisites
    
    local current_env
    local rollback_env
    
    current_env=$(get_current_environment)
    rollback_env=$(get_target_environment "$current_env")
    
    warning "Emergency rollback from $current_env to $rollback_env"
    
    notify_slack "🚨 Emergency rollback initiated: $current_env → $rollback_env" "danger"
    
    perform_rollback "$rollback_env"
    
    success "Emergency rollback completed"
}

show_deployment_status() {
    log "Archon Blue-Green Deployment Status"
    echo "=================================="
    
    local current_env
    current_env=$(get_current_environment)
    
    echo "Active Environment: $current_env"
    echo "Namespace: $NAMESPACE"
    echo ""
    
    echo "Blue Environment:"
    kubectl get deployment archon-auth-server-blue -n "$NAMESPACE" -o wide 2>/dev/null || echo "  Not found"
    echo ""
    
    echo "Green Environment:"
    kubectl get deployment archon-auth-server-green -n "$NAMESPACE" -o wide 2>/dev/null || echo "  Not found"
    echo ""
    
    echo "Active Service:"
    kubectl get service archon-server-active -n "$NAMESPACE" -o wide
    echo ""
    
    echo "Ingress Status:"
    kubectl get ingress archon-ingress -n "$NAMESPACE" -o wide
    echo ""
    
    echo "Recent Events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

run_standalone_health_check() {
    local target_env="$1"
    
    log "Running standalone health check for $target_env environment..."
    
    run_health_checks "$target_env"
    
    success "Standalone health check completed for $target_env"
}

# ============================================================================
# Signal Handlers
# ============================================================================

cleanup() {
    log "Cleaning up..."
    
    # Remove any temporary jobs
    kubectl delete jobs -n "$NAMESPACE" -l component=health-check --ignore-not-found=true
    
    # Remove canary ingress if it exists
    kubectl delete ingress archon-canary-ingress -n "$NAMESPACE" --ignore-not-found=true
}

trap cleanup EXIT

# Handle interrupt signals for emergency rollback
handle_interrupt() {
    warning "Deployment interrupted! Initiating emergency rollback..."
    
    local current_env
    current_env=$(get_current_environment)
    local rollback_env
    rollback_env=$(get_target_environment "$current_env")
    
    perform_rollback "$rollback_env"
    exit 1
}

trap handle_interrupt SIGINT SIGTERM

# ============================================================================
# Script Execution
# ============================================================================

# Validate environment
if [ -z "${IMAGE_TAG:-}" ] && [ "${1:-}" = "deploy" ]; then
    error "IMAGE_TAG environment variable is required for deployment"
fi

# Run main function with all arguments
main "$@"