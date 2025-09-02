#!/bin/bash

# Emergency Rollback Procedures for Archon Phase 6 Authentication System
# Comprehensive rollback strategies with automated incident response

set -euo pipefail

# ============================================================================
# Configuration and Variables
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-archon-production}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
INCIDENT_WEBHOOK="${INCIDENT_WEBHOOK:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
}

critical() {
    echo -e "${RED}[CRITICAL] $1${NC}"
    notify_incident "$1"
    exit 1
}

notify_slack() {
    local message="$1"
    local color="${2:-warning}"
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"$color\",\"text\":\"🚨 ROLLBACK: $message\"}]}" \
            "$SLACK_WEBHOOK" || true
    fi
}

notify_incident() {
    local message="$1"
    
    if [ -n "$INCIDENT_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"incident\": {
                    \"title\": \"Archon Authentication System Rollback\",
                    \"description\": \"$message\",
                    \"severity\": \"critical\",
                    \"service\": \"archon-authentication\",
                    \"source\": \"deployment-script\"
                }
            }" \
            "$INCIDENT_WEBHOOK" || true
    fi
}

get_current_environment() {
    kubectl get configmap blue-green-deployment-config \
        -n "$NAMESPACE" \
        -o jsonpath='{.data.active-environment}' 2>/dev/null || echo "blue"
}

get_previous_environment() {
    local current_env="$1"
    if [ "$current_env" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# ============================================================================
# Health Check Functions
# ============================================================================

check_service_health() {
    local environment="$1"
    local service_url="http://archon-server-$environment:8181"
    
    log "Checking health of $environment environment..."
    
    # Port forward for testing
    kubectl port-forward -n "$NAMESPACE" "service/archon-server-$environment" 8181:8181 &
    local pf_pid=$!
    
    sleep 5
    
    local health_status=0
    
    # Basic health check
    if ! curl -f -s "http://localhost:8181/health" > /dev/null; then
        warning "$environment environment health check failed"
        health_status=1
    fi
    
    # Readiness check
    if ! curl -f -s "http://localhost:8181/ready" > /dev/null; then
        warning "$environment environment readiness check failed"
        health_status=1
    fi
    
    # Authentication endpoint check
    if ! curl -f -s "http://localhost:8181/api/auth/status" > /dev/null; then
        warning "$environment environment auth endpoint check failed"
        health_status=1
    fi
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
    
    return $health_status
}

get_error_metrics() {
    local environment="$1"
    
    # Query Prometheus for error rate
    local error_rate
    error_rate=$(kubectl exec -n archon-monitoring deployment/prometheus-server -- \
        promtool query instant \
        "rate(archon_http_requests_total{status=~\"5..\",environment=\"$environment\"}[5m]) / rate(archon_http_requests_total{environment=\"$environment\"}[5m]) * 100" \
        2>/dev/null | grep -o '[0-9.]*' | head -1 || echo "unknown")
    
    echo "$error_rate"
}

# ============================================================================
# Rollback Strategies
# ============================================================================

immediate_rollback() {
    local reason="$1"
    
    critical "IMMEDIATE ROLLBACK TRIGGERED: $reason"
    
    local current_env
    local rollback_env
    
    current_env=$(get_current_environment)
    rollback_env=$(get_previous_environment "$current_env")
    
    log "Rolling back from $current_env to $rollback_env"
    
    notify_slack "IMMEDIATE ROLLBACK: $reason (from $current_env to $rollback_env)" "danger"
    
    # Step 1: Ensure rollback environment is available and healthy
    if ! check_service_health "$rollback_env"; then
        # Emergency: Scale up rollback environment
        warning "Rollback environment unhealthy, scaling up..."
        kubectl patch deployment "archon-auth-server-$rollback_env" \
            -n "$NAMESPACE" \
            -p '{"spec":{"replicas":3}}'
        
        # Wait for rollback environment
        if ! kubectl rollout status deployment/"archon-auth-server-$rollback_env" \
            -n "$NAMESPACE" \
            --timeout="180s"; then
            critical "Rollback environment failed to become ready - MANUAL INTERVENTION REQUIRED"
        fi
    fi
    
    # Step 2: Immediate traffic switch
    log "Switching traffic immediately..."
    
    kubectl patch service archon-server-active \
        -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"environment\":\"$rollback_env\"}}}"
    
    kubectl patch ingress archon-ingress \
        -n "$NAMESPACE" \
        -p "{\"spec\":{\"rules\":[{\"host\":\"api.archon.com\",\"http\":{\"paths\":[{\"path\":\"/\",\"pathType\":\"Prefix\",\"backend\":{\"service\":{\"name\":\"archon-server-$rollback_env\",\"port\":{\"number\":8181}}}}]}}]}}"
    
    # Step 3: Update configuration
    kubectl patch configmap blue-green-deployment-config \
        -n "$NAMESPACE" \
        -p "{\"data\":{\"active-environment\":\"$rollback_env\"}}"
    
    # Step 4: Verify rollback
    sleep 30
    
    if check_service_health "$rollback_env"; then
        success "Immediate rollback completed successfully"
        notify_slack "✅ Immediate rollback successful - service restored" "good"
    else
        critical "Rollback verification failed - MANUAL INTERVENTION REQUIRED"
    fi
    
    # Step 5: Scale down problematic environment
    log "Scaling down problematic $current_env environment..."
    kubectl patch deployment "archon-auth-server-$current_env" \
        -n "$NAMESPACE" \
        -p '{"spec":{"replicas":0}}'
    
    success "Immediate rollback procedure completed"
}

gradual_rollback() {
    local reason="$1"
    
    warning "GRADUAL ROLLBACK TRIGGERED: $reason"
    
    local current_env
    local rollback_env
    
    current_env=$(get_current_environment)
    rollback_env=$(get_previous_environment "$current_env")
    
    notify_slack "Gradual rollback initiated: $reason (from $current_env to $rollback_env)" "warning"
    
    # Step 1: Ensure rollback environment is ready
    kubectl patch deployment "archon-auth-server-$rollback_env" \
        -n "$NAMESPACE" \
        -p '{"spec":{"replicas":3}}'
    
    kubectl rollout status deployment/"archon-auth-server-$rollback_env" \
        -n "$NAMESPACE" \
        --timeout="180s"
    
    # Step 2: Canary rollback (shift 10% traffic back)
    log "Starting canary rollback - 10% traffic to $rollback_env"
    
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: archon-rollback-canary-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"
spec:
  rules:
  - host: api.archon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: archon-server-$rollback_env
            port:
              number: 8181
EOF
    
    sleep 120  # Monitor canary for 2 minutes
    
    # Step 3: Check if canary rollback is improving metrics
    local canary_error_rate
    canary_error_rate=$(get_error_metrics "$rollback_env")
    
    log "Rollback canary error rate: $canary_error_rate%"
    
    if (( $(echo "$canary_error_rate < 1.0" | bc -l) )); then
        log "Canary rollback showing improvement, proceeding with full rollback"
        
        # Remove canary and switch all traffic
        kubectl delete ingress archon-rollback-canary-ingress -n "$NAMESPACE"
        
        kubectl patch service archon-server-active \
            -n "$NAMESPACE" \
            -p "{\"spec\":{\"selector\":{\"environment\":\"$rollback_env\"}}}"
        
        kubectl patch ingress archon-ingress \
            -n "$NAMESPACE" \
            -p "{\"spec\":{\"rules\":[{\"host\":\"api.archon.com\",\"http\":{\"paths\":[{\"path\":\"/\",\"pathType\":\"Prefix\",\"backend\":{\"service\":{\"name\":\"archon-server-$rollback_env\",\"port\":{\"number\":8181}}}}]}}]}}"
        
        success "Gradual rollback completed successfully"
        notify_slack "✅ Gradual rollback completed - service restored" "good"
    else
        warning "Rollback canary not improving metrics, may need manual intervention"
        notify_slack "⚠️ Rollback canary not improving - needs investigation" "warning"
    fi
    
    # Update configuration
    kubectl patch configmap blue-green-deployment-config \
        -n "$NAMESPACE" \
        -p "{\"data\":{\"active-environment\":\"$rollback_env\"}}"
}

database_rollback() {
    warning "DATABASE ROLLBACK PROCEDURE - USE WITH EXTREME CAUTION"
    
    read -p "Are you sure you want to perform a database rollback? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Database rollback cancelled"
        return 0
    fi
    
    log "Creating database backup before rollback..."
    
    # Create immediate backup
    local backup_name="rollback-backup-$(date +%Y%m%d-%H%M%S)"
    
    kubectl exec -n "$NAMESPACE" deployment/postgres -- \
        pg_dump -U archon archon > "/tmp/$backup_name.sql"
    
    log "Database backup created: $backup_name.sql"
    
    # Apply rollback migration (if exists)
    if [ -f "migrations/rollback-$(date +%Y%m%d).sql" ]; then
        warning "Applying rollback migration..."
        kubectl exec -n "$NAMESPACE" deployment/postgres -- \
            psql -U archon -d archon -f "migrations/rollback-$(date +%Y%m%d).sql"
        success "Database rollback migration applied"
    else
        warning "No rollback migration found for today"
    fi
    
    notify_slack "🗄️ Database rollback completed - backup: $backup_name" "warning"
}

# ============================================================================
# Automated Monitoring and Auto-Rollback
# ============================================================================

monitor_and_auto_rollback() {
    local monitoring_duration="${1:-300}"  # Default 5 minutes
    local error_threshold="${2:-5.0}"      # Default 5% error rate
    
    log "Starting automated monitoring for $monitoring_duration seconds..."
    log "Auto-rollback will trigger if error rate exceeds $error_threshold%"
    
    local current_env
    current_env=$(get_current_environment)
    
    local start_time
    start_time=$(date +%s)
    
    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $monitoring_duration ]; then
            success "Monitoring period completed without issues"
            break
        fi
        
        # Check current error rate
        local error_rate
        error_rate=$(get_error_metrics "$current_env")
        
        if [ "$error_rate" != "unknown" ] && (( $(echo "$error_rate > $error_threshold" | bc -l) )); then
            critical "Auto-rollback triggered: Error rate $error_rate% exceeds threshold $error_threshold%"
            immediate_rollback "Automated rollback due to high error rate: $error_rate%"
            break
        fi
        
        # Check service availability
        if ! curl -f -s "https://api.archon.com/health" > /dev/null; then
            critical "Auto-rollback triggered: Service health check failed"
            immediate_rollback "Automated rollback due to service unavailability"
            break
        fi
        
        log "Monitoring... Error rate: $error_rate%, Elapsed: ${elapsed}s"
        sleep 30
    done
}

# ============================================================================
# Incident Response Integration
# ============================================================================

create_incident() {
    local title="$1"
    local description="$2"
    local severity="${3:-high}"
    
    log "Creating incident: $title"
    
    if [ -n "$INCIDENT_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"incident\": {
                    \"title\": \"$title\",
                    \"description\": \"$description\",
                    \"severity\": \"$severity\",
                    \"service\": \"archon-authentication\",
                    \"status\": \"investigating\",
                    \"created_by\": \"deployment-automation\",
                    \"tags\": [\"rollback\", \"authentication\", \"production\"]
                }
            }" \
            "$INCIDENT_WEBHOOK" || warning "Failed to create incident"
    fi
}

generate_rollback_report() {
    local rollback_reason="$1"
    local rollback_time="$2"
    
    log "Generating rollback report..."
    
    local current_env
    current_env=$(get_current_environment)
    
    local report_file="/tmp/rollback-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" <<EOF
{
  "rollback_timestamp": "$rollback_time",
  "reason": "$rollback_reason",
  "previous_environment": "$(get_previous_environment "$current_env")",
  "current_environment": "$current_env",
  "namespace": "$NAMESPACE",
  "system_status": {
    "error_rate": "$(get_error_metrics "$current_env")",
    "health_check": "$(check_service_health "$current_env" && echo "passed" || echo "failed")"
  },
  "kubernetes_events": [
$(kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' -o json | jq '.items[-10:] | map({time: .firstTimestamp, reason: .reason, message: .message})' | sed 's/^/    /')
  ],
  "deployment_status": {
    "blue_replicas": $(kubectl get deployment archon-auth-server-blue -n "$NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo 0),
    "green_replicas": $(kubectl get deployment archon-auth-server-green -n "$NAMESPACE" -o jsonpath='{.status.replicas}' 2>/dev/null || echo 0),
    "blue_ready": $(kubectl get deployment archon-auth-server-blue -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0),
    "green_ready": $(kubectl get deployment archon-auth-server-green -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
  }
}
EOF
    
    log "Rollback report generated: $report_file"
    
    # Upload to S3 or log aggregation system
    if command -v aws &> /dev/null; then
        aws s3 cp "$report_file" "s3://archon-incident-reports/rollback-reports/" || true
    fi
    
    echo "$report_file"
}

# ============================================================================
# Main Rollback Functions
# ============================================================================

perform_rollback() {
    local rollback_type="${1:-immediate}"
    local reason="${2:-Manual rollback requested}"
    
    log "Performing $rollback_type rollback..."
    log "Reason: $reason"
    
    local rollback_start_time
    rollback_start_time=$(date -Iseconds)
    
    case "$rollback_type" in
        "immediate")
            immediate_rollback "$reason"
            ;;
        "gradual")
            gradual_rollback "$reason"
            ;;
        "database")
            database_rollback
            ;;
        *)
            error "Unknown rollback type: $rollback_type"
            ;;
    esac
    
    # Generate post-rollback report
    local report_file
    report_file=$(generate_rollback_report "$reason" "$rollback_start_time")
    
    success "Rollback completed. Report: $report_file"
}

emergency_circuit_breaker() {
    critical "EMERGENCY CIRCUIT BREAKER ACTIVATED"
    
    # Immediately scale down all deployments
    kubectl patch deployment archon-auth-server-blue -n "$NAMESPACE" -p '{"spec":{"replicas":0}}'
    kubectl patch deployment archon-auth-server-green -n "$NAMESPACE" -p '{"spec":{"replicas":0}}'
    
    # Deploy emergency maintenance page
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: archon-maintenance-page
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: maintenance-page
  template:
    metadata:
      labels:
        app: maintenance-page
    spec:
      containers:
      - name: maintenance
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: maintenance-content
          mountPath: /usr/share/nginx/html
      volumes:
      - name: maintenance-content
        configMap:
          name: maintenance-page-content
---
apiVersion: v1
kind: Service
metadata:
  name: maintenance-page
  namespace: $NAMESPACE
spec:
  selector:
    app: maintenance-page
  ports:
  - port: 80
    targetPort: 80
EOF
    
    # Update ingress to point to maintenance page
    kubectl patch ingress archon-ingress \
        -n "$NAMESPACE" \
        -p '{"spec":{"rules":[{"host":"api.archon.com","http":{"paths":[{"path":"/","pathType":"Prefix","backend":{"service":{"name":"maintenance-page","port":{"number":80}}}}]}}]}}'
    
    create_incident "Emergency Circuit Breaker Activated" "All Archon services have been taken offline due to critical issues. Maintenance page is now active." "critical"
    
    notify_slack "🔴 EMERGENCY CIRCUIT BREAKER ACTIVATED - All services offline, maintenance page active" "danger"
    
    critical "Emergency circuit breaker completed - ALL SERVICES OFFLINE"
}

# ============================================================================
# Status and Diagnostics
# ============================================================================

show_rollback_status() {
    log "Archon Rollback Status and Diagnostics"
    echo "======================================"
    
    local current_env
    current_env=$(get_current_environment)
    
    echo "Current Active Environment: $current_env"
    echo "Namespace: $NAMESPACE"
    echo ""
    
    echo "Environment Health Status:"
    echo "-------------------------"
    
    if check_service_health "blue"; then
        echo "🟢 Blue Environment: Healthy"
    else
        echo "🔴 Blue Environment: Unhealthy"
    fi
    
    if check_service_health "green"; then
        echo "🟢 Green Environment: Healthy"
    else
        echo "🔴 Green Environment: Unhealthy"
    fi
    
    echo ""
    echo "Error Rates:"
    echo "-----------"
    echo "Blue Error Rate: $(get_error_metrics "blue")%"
    echo "Green Error Rate: $(get_error_metrics "green")%"
    echo ""
    
    echo "Deployment Status:"
    echo "-----------------"
    kubectl get deployments -n "$NAMESPACE" -l app=archon-server -o wide
    echo ""
    
    echo "Service Status:"
    echo "--------------"
    kubectl get services -n "$NAMESPACE" -l app=archon-server -o wide
    echo ""
    
    echo "Recent Events (Last 10):"
    echo "------------------------"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# ============================================================================
# Main Script Logic
# ============================================================================

main() {
    local action="${1:-status}"
    local type="${2:-immediate}"
    local reason="${3:-Manual execution}"
    
    case "$action" in
        "rollback")
            perform_rollback "$type" "$reason"
            ;;
        "monitor")
            monitor_and_auto_rollback "${2:-300}" "${3:-5.0}"
            ;;
        "circuit-breaker")
            emergency_circuit_breaker
            ;;
        "status")
            show_rollback_status
            ;;
        *)
            echo "Archon Emergency Rollback Procedures"
            echo "==================================="
            echo ""
            echo "Usage: $0 <action> [options]"
            echo ""
            echo "Actions:"
            echo "  rollback <type> <reason>   - Perform rollback"
            echo "    Types: immediate, gradual, database"
            echo "  monitor <duration> <threshold> - Auto-monitor and rollback"
            echo "  circuit-breaker            - Emergency shutdown"
            echo "  status                     - Show current status"
            echo ""
            echo "Examples:"
            echo "  $0 rollback immediate 'High error rate detected'"
            echo "  $0 rollback gradual 'Performance degradation'"
            echo "  $0 monitor 600 3.0"
            echo "  $0 circuit-breaker"
            echo "  $0 status"
            echo ""
            echo "Environment Variables:"
            echo "  NAMESPACE           - Kubernetes namespace (default: archon-production)"
            echo "  SLACK_WEBHOOK       - Slack webhook URL for notifications"
            echo "  INCIDENT_WEBHOOK    - Incident management webhook URL"
            exit 1
            ;;
    esac
}

# ============================================================================
# Script Execution
# ============================================================================

# Ensure we're in the right context
if ! kubectl config current-context | grep -q archon; then
    warning "Current kubectl context doesn't appear to be Archon cluster"
    read -p "Continue anyway? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        exit 1
    fi
fi

# Run main function
main "$@"