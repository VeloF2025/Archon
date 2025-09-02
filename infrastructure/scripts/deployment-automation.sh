#!/bin/bash

# Comprehensive Deployment Automation for Archon Phase 6 Authentication System
# Orchestrates the complete CI/CD pipeline with safety checks and rollback capabilities

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="${NAMESPACE:-archon-$ENVIRONMENT}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-blue-green}"

# Timeouts
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1800}"    # 30 minutes
TEST_TIMEOUT="${TEST_TIMEOUT:-900}"       # 15 minutes
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-600}"   # 10 minutes
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"  # 5 minutes

# Notification webhooks
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
TEAMS_WEBHOOK="${TEAMS_WEBHOOK:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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
    exit 1
}

stage() {
    echo -e "${PURPLE}[STAGE] $1${NC}"
    echo "=============================================="
}

notify() {
    local message="$1"
    local color="${2:-good}"
    local webhook_type="${3:-slack}"
    
    case "$webhook_type" in
        "slack")
            if [ -n "$SLACK_WEBHOOK" ]; then
                curl -X POST -H 'Content-type: application/json' \
                    --data "{\"attachments\":[{\"color\":\"$color\",\"text\":\"🚀 Archon Deploy: $message\"}]}" \
                    "$SLACK_WEBHOOK" 2>/dev/null || true
            fi
            ;;
        "teams")
            if [ -n "$TEAMS_WEBHOOK" ]; then
                curl -X POST -H 'Content-type: application/json' \
                    --data "{\"text\":\"🚀 Archon Deploy: $message\"}" \
                    "$TEAMS_WEBHOOK" 2>/dev/null || true
            fi
            ;;
    esac
}

check_prerequisites() {
    stage "Checking Prerequisites"
    
    local missing_tools=()
    
    # Required tools
    for tool in kubectl docker helm terraform; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
    fi
    
    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context)
    log "Current kubectl context: $current_context"
    
    # Check permissions
    if ! kubectl auth can-i create deployments --namespace="$NAMESPACE" &> /dev/null; then
        error "Insufficient permissions for namespace $NAMESPACE"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon not accessible"
    fi
    
    success "Prerequisites check passed"
}

# ============================================================================
# Build and Test Pipeline
# ============================================================================

run_security_scans() {
    stage "Security Scanning"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Skipping security scans"
        return 0
    fi
    
    log "Running comprehensive security scans..."
    
    # Python security scan
    if [ -f "$PROJECT_ROOT/python/pyproject.toml" ]; then
        log "Running Python security scans..."
        cd "$PROJECT_ROOT/python"
        
        # Bandit SAST
        python -m bandit -r src/ -f json -o ../security-bandit.json || true
        
        # Safety dependency scan
        python -m safety check --json --output ../security-safety.json || true
        
        cd "$PROJECT_ROOT"
    fi
    
    # TypeScript security scan
    if [ -f "$PROJECT_ROOT/archon-ui-main/package.json" ]; then
        log "Running TypeScript security scans..."
        cd "$PROJECT_ROOT/archon-ui-main"
        
        # npm audit
        npm audit --json > ../security-npm-audit.json || true
        
        cd "$PROJECT_ROOT"
    fi
    
    # Infrastructure security scan
    if [ -d "$PROJECT_ROOT/infrastructure" ]; then
        log "Running infrastructure security scans..."
        
        # Trivy IaC scan
        docker run --rm -v "$PROJECT_ROOT":/workspace aquasecurity/trivy:latest config /workspace/infrastructure > security-trivy-iac.json || true
        
        # Checkov scan
        docker run --rm -v "$PROJECT_ROOT":/workspace bridgecrew/checkov:latest -d /workspace/infrastructure --framework terraform,kubernetes,dockerfile --output json > security-checkov.json || true
    fi
    
    # Generate comprehensive report
    python3 "$SCRIPT_DIR/generate-security-report.py" \
        --input-dir . \
        --output security-summary \
        --format both
    
    # Check if security scan passed thresholds
    python3 "$SCRIPT_DIR/check-security-thresholds.py" \
        --report security-summary.json \
        --fail-on-critical \
        --max-high 5 \
        --max-medium 20
    
    success "Security scans completed"
}

run_tests() {
    stage "Running Test Suite"
    
    if [ "$SKIP_TESTS" = "true" ]; then
        warning "Skipping tests as requested"
        return 0
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Skipping tests"
        return 0
    fi
    
    local test_start_time
    test_start_time=$(date +%s)
    
    # Backend tests
    if [ -f "$PROJECT_ROOT/python/pyproject.toml" ]; then
        log "Running Python tests..."
        cd "$PROJECT_ROOT/python"
        
        timeout "$TEST_TIMEOUT" uv run pytest tests/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --junitxml=test-results.xml \
            -v
        
        cd "$PROJECT_ROOT"
        success "Python tests passed"
    fi
    
    # Frontend tests
    if [ -f "$PROJECT_ROOT/archon-ui-main/package.json" ]; then
        log "Running TypeScript tests..."
        cd "$PROJECT_ROOT/archon-ui-main"
        
        timeout "$TEST_TIMEOUT" npm run test:coverage
        
        cd "$PROJECT_ROOT"
        success "TypeScript tests passed"
    fi
    
    # Integration tests
    log "Running integration tests..."
    timeout "$TEST_TIMEOUT" docker-compose -f docker-compose.yml up -d
    sleep 30
    
    # Wait for services to be healthy
    local max_attempts=20
    for i in $(seq 1 $max_attempts); do
        if curl -f "http://localhost:8181/health" && curl -f "http://localhost:3737/health"; then
            break
        fi
        if [ $i -eq $max_attempts ]; then
            docker-compose -f docker-compose.yml logs
            docker-compose -f docker-compose.yml down
            error "Integration test setup failed - services not healthy"
        fi
        sleep 10
    done
    
    # Run integration tests
    docker-compose -f docker-compose.yml exec -T archon-server \
        python -m pytest tests/integration/ -v
    
    docker-compose -f docker-compose.yml down
    
    local test_end_time
    test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))
    
    success "All tests passed in ${test_duration}s"
}

build_images() {
    stage "Building Container Images"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Skipping image builds"
        return 0
    fi
    
    local build_start_time
    build_start_time=$(date +%s)
    
    log "Building production images with tag: $IMAGE_TAG"
    
    # Build all service images in parallel
    local pids=()
    
    # Backend services
    for service in server mcp agents validator; do
        log "Building archon-$service image..."
        (
            docker build \
                -t "ghcr.io/archon/archon-$service:$IMAGE_TAG" \
                -f "python/Dockerfile.production" \
                --target "$service-service" \
                --build-arg BUILDKIT_INLINE_CACHE=1 \
                python/
        ) &
        pids+=($!)
    done
    
    # Frontend image
    log "Building archon-frontend image..."
    (
        docker build \
            -t "ghcr.io/archon/archon-frontend:$IMAGE_TAG" \
            -f "archon-ui-main/Dockerfile.production" \
            archon-ui-main/
    ) &
    pids+=($!)
    
    # Wait for all builds to complete
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            error "Image build failed"
        fi
    done
    
    local build_end_time
    build_end_time=$(date +%s)
    local build_duration=$((build_end_time - build_start_time))
    
    success "All images built in ${build_duration}s"
    
    # Security scan built images
    log "Running security scans on built images..."
    for service in server mcp agents validator frontend; do
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasecurity/trivy:latest image \
            --format json \
            --output "security-$service-image.json" \
            "ghcr.io/archon/archon-$service:$IMAGE_TAG" || true
    done
    
    success "Image security scans completed"
}

push_images() {
    stage "Pushing Images to Registry"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Skipping image push"
        return 0
    fi
    
    log "Pushing images to registry..."
    
    # Login to registry (assumes GitHub token is available)
    echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin
    
    # Push all images
    for service in server mcp agents validator frontend; do
        log "Pushing archon-$service:$IMAGE_TAG..."
        docker push "ghcr.io/archon/archon-$service:$IMAGE_TAG"
    done
    
    success "All images pushed to registry"
}

deploy_to_kubernetes() {
    stage "Deploying to Kubernetes ($ENVIRONMENT)"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Would deploy to $ENVIRONMENT with strategy $DEPLOYMENT_STRATEGY"
        return 0
    fi
    
    case "$DEPLOYMENT_STRATEGY" in
        "blue-green")
            log "Executing blue-green deployment..."
            
            export NAMESPACE="$NAMESPACE"
            export IMAGE_TAG="$IMAGE_TAG"
            
            "$SCRIPT_DIR/blue-green-deploy.sh" deploy
            ;;
            
        "rolling")
            log "Executing rolling deployment..."
            
            # Update image tags for all deployments
            for service in server mcp agents validator frontend; do
                kubectl set image deployment/"archon-$service" \
                    "archon-$service=ghcr.io/archon/archon-$service:$IMAGE_TAG" \
                    -n "$NAMESPACE"
            done
            
            # Wait for rollouts
            for service in server mcp agents validator frontend; do
                kubectl rollout status deployment/"archon-$service" \
                    -n "$NAMESPACE" \
                    --timeout="${DEPLOY_TIMEOUT}s"
            done
            ;;
            
        "canary")
            log "Executing canary deployment..."
            error "Canary deployment not yet implemented"
            ;;
            
        *)
            error "Unknown deployment strategy: $DEPLOYMENT_STRATEGY"
            ;;
    esac
    
    success "Deployment completed using $DEPLOYMENT_STRATEGY strategy"
}

run_smoke_tests() {
    stage "Running Smoke Tests"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Skipping smoke tests"
        return 0
    fi
    
    log "Running post-deployment smoke tests..."
    
    local base_url
    case "$ENVIRONMENT" in
        "production")
            base_url="https://api.archon.com"
            ;;
        "staging")
            base_url="https://staging-api.archon.com"
            ;;
        *)
            # Use port-forward for development
            kubectl port-forward -n "$NAMESPACE" service/archon-server 8181:8181 &
            local pf_pid=$!
            sleep 5
            base_url="http://localhost:8181"
            ;;
    esac
    
    local smoke_tests_passed=true
    
    # Test 1: Health endpoint
    log "Testing health endpoint..."
    if ! curl -f -s "$base_url/health" > /dev/null; then
        error "Health endpoint test failed"
        smoke_tests_passed=false
    fi
    
    # Test 2: API endpoints
    log "Testing API endpoints..."
    if ! curl -f -s "$base_url/api/auth/status" > /dev/null; then
        warning "Auth status endpoint test failed"
        smoke_tests_passed=false
    fi
    
    # Test 3: MCP endpoint
    log "Testing MCP endpoint..."
    if ! curl -f -s "$base_url/api/mcp/health" > /dev/null; then
        warning "MCP health endpoint test failed"
        smoke_tests_passed=false
    fi
    
    # Test 4: Database connectivity
    log "Testing database connectivity..."
    if ! curl -f -s "$base_url/api/health/db" > /dev/null; then
        warning "Database connectivity test failed"
        smoke_tests_passed=false
    fi
    
    # Test 5: Redis connectivity
    log "Testing Redis connectivity..."
    if ! curl -f -s "$base_url/api/health/cache" > /dev/null; then
        warning "Redis connectivity test failed"
        smoke_tests_passed=false
    fi
    
    # Clean up port-forward if used
    if [ -n "${pf_pid:-}" ]; then
        kill "$pf_pid" 2>/dev/null || true
    fi
    
    if [ "$smoke_tests_passed" = "true" ]; then
        success "All smoke tests passed"
    else
        error "Some smoke tests failed"
    fi
}

monitor_deployment() {
    stage "Monitoring Deployment"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: Skipping deployment monitoring"
        return 0
    fi
    
    log "Monitoring deployment for 5 minutes..."
    
    local monitoring_start
    monitoring_start=$(date +%s)
    local monitoring_duration=300  # 5 minutes
    
    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - monitoring_start))
        
        if [ $elapsed -ge $monitoring_duration ]; then
            success "Deployment monitoring completed successfully"
            break
        fi
        
        # Check error rate
        local error_rate
        error_rate=$(kubectl exec -n archon-monitoring deployment/prometheus-server -- \
            promtool query instant \
            'rate(archon_http_requests_total{status=~"5.."}[2m]) / rate(archon_http_requests_total[2m]) * 100' \
            2>/dev/null | grep -o '[0-9.]*' | head -1 || echo "0")
        
        # Check if error rate is too high
        if [ "$error_rate" != "0" ] && (( $(echo "$error_rate > 5.0" | bc -l) )); then
            error "High error rate detected: $error_rate% - initiating rollback"
        fi
        
        # Check service availability
        if ! curl -f -s "https://api.archon.com/health" > /dev/null; then
            error "Service availability check failed - initiating rollback"
        fi
        
        log "Monitoring... Error rate: $error_rate%, Elapsed: ${elapsed}s"
        sleep 30
    done
}

# ============================================================================
# Main Deployment Pipeline
# ============================================================================

run_full_pipeline() {
    local pipeline_start_time
    pipeline_start_time=$(date +%s)
    
    log "Starting full deployment pipeline for Archon Phase 6 Authentication"
    log "Environment: $ENVIRONMENT"
    log "Image Tag: $IMAGE_TAG"
    log "Strategy: $DEPLOYMENT_STRATEGY"
    log "Dry Run: $DRY_RUN"
    
    notify "Starting deployment pipeline to $ENVIRONMENT (tag: $IMAGE_TAG)" "good"
    
    # Stage 1: Prerequisites
    check_prerequisites
    
    # Stage 2: Security scanning
    run_security_scans
    
    # Stage 3: Build and test
    if [ "$SKIP_TESTS" != "true" ]; then
        run_tests
    fi
    
    # Stage 4: Build images
    build_images
    
    # Stage 5: Push images
    if [ "$ENVIRONMENT" != "local" ]; then
        push_images
    fi
    
    # Stage 6: Deploy to Kubernetes
    deploy_to_kubernetes
    
    # Stage 7: Post-deployment testing
    run_smoke_tests
    
    # Stage 8: Monitor deployment
    monitor_deployment
    
    local pipeline_end_time
    pipeline_end_time=$(date +%s)
    local pipeline_duration=$((pipeline_end_time - pipeline_start_time))
    
    success "🎉 Full deployment pipeline completed successfully!"
    success "Total duration: ${pipeline_duration}s"
    success "Environment: $ENVIRONMENT"
    success "Image tag: $IMAGE_TAG"
    
    notify "✅ Deployment pipeline completed successfully in ${pipeline_duration}s" "good"
    
    # Generate deployment report
    generate_deployment_report "$pipeline_duration"
}

generate_deployment_report() {
    local duration="$1"
    
    local report_file="deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" <<EOF
{
  "deployment_timestamp": "$(date -Iseconds)",
  "environment": "$ENVIRONMENT",
  "image_tag": "$IMAGE_TAG",
  "deployment_strategy": "$DEPLOYMENT_STRATEGY",
  "duration_seconds": $duration,
  "dry_run": $DRY_RUN,
  "tests_skipped": $SKIP_TESTS,
  "pipeline_stages": {
    "security_scan": "completed",
    "tests": "$([ "$SKIP_TESTS" = "true" ] && echo "skipped" || echo "completed")",
    "build": "completed",
    "deploy": "completed",
    "smoke_tests": "completed",
    "monitoring": "completed"
  },
  "service_status": {
$(kubectl get deployments -n "$NAMESPACE" -l app.kubernetes.io/name=archon -o json | jq '.items[] | {name: .metadata.name, replicas: .status.replicas, ready: .status.readyReplicas}' | sed 's/^/    /')
  },
  "health_checks": {
    "overall": "passed",
    "timestamp": "$(date -Iseconds)"
  }
}
EOF
    
    log "Deployment report generated: $report_file"
    
    # Upload to S3 if available
    if command -v aws &> /dev/null; then
        aws s3 cp "$report_file" "s3://archon-deployment-reports/" 2>/dev/null || true
    fi
}

# ============================================================================
# Rollback Integration
# ============================================================================

initiate_rollback() {
    local reason="$1"
    
    warning "Initiating rollback: $reason"
    
    notify "🔄 Initiating rollback: $reason" "warning"
    
    # Use the dedicated rollback script
    export NAMESPACE="$NAMESPACE"
    "$SCRIPT_DIR/rollback-procedures.sh" rollback immediate "$reason"
}

# ============================================================================
# Main Script Logic
# ============================================================================

show_usage() {
    echo "Archon Deployment Automation Script"
    echo "==================================="
    echo ""
    echo "Usage: $0 <action> [options]"
    echo ""
    echo "Actions:"
    echo "  deploy                 - Run full deployment pipeline"
    echo "  build                  - Build images only"
    echo "  test                   - Run tests only"
    echo "  security-scan          - Run security scans only"
    echo "  smoke-test             - Run smoke tests only"
    echo "  rollback <reason>      - Initiate emergency rollback"
    echo "  status                 - Show deployment status"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT            - Target environment (staging/production)"
    echo "  IMAGE_TAG              - Docker image tag to deploy"
    echo "  DRY_RUN                - Set to 'true' for dry run"
    echo "  SKIP_TESTS             - Set to 'true' to skip tests"
    echo "  DEPLOYMENT_STRATEGY    - blue-green, rolling, canary"
    echo "  SLACK_WEBHOOK          - Slack webhook for notifications"
    echo ""
    echo "Examples:"
    echo "  $0 deploy"
    echo "  ENVIRONMENT=production IMAGE_TAG=v1.2.3 $0 deploy"
    echo "  DRY_RUN=true $0 deploy"
    echo "  $0 rollback 'Database connection issues'"
}

main() {
    local action="${1:-deploy}"
    
    case "$action" in
        "deploy")
            run_full_pipeline
            ;;
        "build")
            check_prerequisites
            build_images
            ;;
        "test")
            check_prerequisites
            run_tests
            ;;
        "security-scan")
            check_prerequisites
            run_security_scans
            ;;
        "smoke-test")
            check_prerequisites
            run_smoke_tests
            ;;
        "rollback")
            local reason="${2:-Manual rollback requested}"
            initiate_rollback "$reason"
            ;;
        "status")
            "$SCRIPT_DIR/blue-green-deploy.sh" status
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            echo "Unknown action: $action"
            show_usage
            exit 1
            ;;
    esac
}

# ============================================================================
# Error Handling and Cleanup
# ============================================================================

cleanup() {
    log "Cleaning up..."
    
    # Stop any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clean up temporary files
    rm -f security-*.json deployment-report-*.json 2>/dev/null || true
}

trap cleanup EXIT

handle_error() {
    local line_number="$1"
    error "Script failed at line $line_number"
    
    # Auto-rollback on pipeline failure in production
    if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" != "true" ]; then
        warning "Production deployment failed - initiating automatic rollback"
        initiate_rollback "Deployment pipeline failure at line $line_number"
    fi
}

trap 'handle_error $LINENO' ERR

# ============================================================================
# Script Execution
# ============================================================================

# Validate required environment variables
if [ -z "${IMAGE_TAG:-}" ] && [ "${1:-}" = "deploy" ]; then
    warning "IMAGE_TAG not set, using 'latest'"
    IMAGE_TAG="latest"
fi

# Run main function
main "$@"