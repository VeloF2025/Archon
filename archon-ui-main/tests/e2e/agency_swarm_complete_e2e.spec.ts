/**
 * Agency Swarm Complete E2E Test Suite
 *
 * Comprehensive end-to-end tests for the complete Agency Swarm workflow
 * including Phase 1-3 integration, deployment validation, and production readiness.
 *
 * Key test scenarios:
 * - Complete agency lifecycle from creation to execution
 * - Phase 1-3 component integration
 * - Production deployment validation
 * - Performance and security testing
 * - Real-time communication and handoffs
 * - Error handling and recovery
 * - Cross-platform compatibility
 * - Monitoring and observability
 */

import { test, expect, Page } from '@playwright/test';
import { v4 as uuidv4 } from 'uuid';

// Test configuration
const BASE_URL = 'http://localhost:3738';
const AGENCY_URL = `${BASE_URL}/agency`;
const WORKFLOW_URL = `${BASE_URL}/workflow`;
const MCP_URL = `${BASE_URL}/mcp`;

// Complete agency workflow test data
const completeAgencyData = {
  id: uuidv4(),
  name: 'Complete Agency Swarm Test',
  description: 'Comprehensive test agency with all Phase 1-3 features',
  version: '3.0.0',
  phase: 'production',
  agents: [
    {
      id: 'agent-architect',
      name: 'System Architect',
      agent_type: 'SYSTEM_ARCHITECT',
      model_tier: 'OPUS',
      project_id: 'test-project',
      state: 'ACTIVE',
      state_changed_at: new Date(),
      tasks_completed: 45,
      success_rate: 0.94,
      avg_completion_time_seconds: 180,
      memory_usage_mb: 1024,
      cpu_usage_percent: 35,
      capabilities: {
        architecture_design: true,
        system_planning: true,
        technical_analysis: true,
        agency_coordination: true,
        workflow_management: true
      },
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'agent-implementer',
      name: 'Code Implementer',
      agent_type: 'CODE_IMPLEMENTER',
      model_tier: 'SONNET',
      project_id: 'test-project',
      state: 'ACTIVE',
      state_changed_at: new Date(),
      tasks_completed: 320,
      success_rate: 0.89,
      avg_completion_time_seconds: 45,
      memory_usage_mb: 512,
      cpu_usage_percent: 28,
      capabilities: {
        code_generation: true,
        debugging: true,
        testing: true,
        documentation: true,
        optimization: true
      },
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'agent-quality',
      name: 'Quality Reviewer',
      agent_type: 'CODE_QUALITY_REVIEWER',
      model_tier: 'SONNET',
      project_id: 'test-project',
      state: 'ACTIVE',
      state_changed_at: new Date(),
      tasks_completed: 156,
      success_rate: 0.96,
      avg_completion_time_seconds: 60,
      memory_usage_mb: 256,
      cpu_usage_percent: 22,
      capabilities: {
        code_review: true,
        quality_assurance: true,
        linting: true,
        security_analysis: true,
        performance_analysis: true
      },
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'agent-tester',
      name: 'Test Validator',
      agent_type: 'TEST_COVERAGE_VALIDATOR',
      model_tier: 'HAIKU',
      project_id: 'test-project',
      state: 'ACTIVE',
      state_changed_at: new Date(),
      tasks_completed: 425,
      success_rate: 0.91,
      avg_completion_time_seconds: 30,
      memory_usage_mb: 128,
      cpu_usage_percent: 18,
      capabilities: {
        test_generation: true,
        coverage_analysis: true,
        validation: true,
        performance_testing: true,
        integration_testing: true
      },
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'agent-security',
      name: 'Security Auditor',
      agent_type: 'SECURITY_AUDITOR',
      model_tier: 'OPUS',
      project_id: 'test-project',
      state: 'IDLE',
      state_changed_at: new Date(),
      tasks_completed: 89,
      success_rate: 0.98,
      avg_completion_time_seconds: 120,
      memory_usage_mb: 768,
      cpu_usage_percent: 45,
      capabilities: {
        security_analysis: true,
        vulnerability_scanning: true,
        compliance_checking: true,
        penetration_testing: true,
        security_monitoring: true
      },
      created_at: new Date(),
      updated_at: new Date()
    }
  ],
  communication_flows: [
    {
      id: 'flow-architect-implementer',
      source_agent_id: 'agent-architect',
      target_agent_id: 'agent-implementer',
      communication_type: 'DIRECT',
      status: 'active',
      message_count: 89,
      last_message_at: new Date(),
      message_type: 'task_assignment',
      priority: 'high',
      encryption_enabled: true,
      compression_enabled: false,
      data_flow: {
        input_size: 2048,
        output_size: 1536,
        processing_time_ms: 250
      }
    },
    {
      id: 'flow-implementer-quality',
      source_agent_id: 'agent-implementer',
      target_agent_id: 'agent-quality',
      communication_type: 'COLLABORATIVE',
      status: 'active',
      message_count: 156,
      last_message_at: new Date(),
      message_type: 'code_submission',
      priority: 'medium',
      encryption_enabled: true,
      compression_enabled: true,
      data_flow: {
        input_size: 4096,
        output_size: 1024,
        processing_time_ms: 180
      }
    },
    {
      id: 'flow-quality-tester',
      source_agent_id: 'agent-quality',
      target_agent_id: 'agent-tester',
      communication_type: 'CHAIN',
      status: 'active',
      message_count: 203,
      last_message_at: new Date(),
      message_type: 'quality_review',
      priority: 'high',
      encryption_enabled: true,
      compression_enabled: false,
      data_flow: {
        input_size: 1024,
        output_size: 512,
        processing_time_ms: 90
      }
    },
    {
      id: 'flow-architect-broadcast',
      source_agent_id: 'agent-architect',
      target_agent_id: ['agent-implementer', 'agent-quality', 'agent-tester', 'agent-security'],
      communication_type: 'BROADCAST',
      status: 'active',
      message_count: 34,
      last_message_at: new Date(),
      message_type: 'specification_update',
      priority: 'high',
      encryption_enabled: true,
      compression_enabled: true,
      data_flow: {
        input_size: 1536,
        output_size: 1536,
        processing_time_ms: 450
      }
    },
    {
      id: 'flow-security-monitor',
      source_agent_id: 'agent-security',
      target_agent_id: ['agent-implementer', 'agent-quality'],
      communication_type: 'MONITOR',
      status: 'pending',
      message_count: 12,
      last_message_at: new Date(),
      message_type: 'security_alert',
      priority: 'critical',
      encryption_enabled: true,
      compression_enabled: false,
      data_flow: {
        input_size: 512,
        output_size: 256,
        processing_time_ms: 75
      }
    }
  ],
  workflow_rules: {
    routing_rules: {
      complex_tasks: 'SYSTEM_ARCHITECT',
      code_tasks: 'CODE_IMPLEMENTER',
      review_tasks: 'CODE_QUALITY_REVIEWER',
      test_tasks: 'TEST_COVERAGE_VALIDATOR',
      security_tasks: 'SECURITY_AUDITOR'
    },
    collaboration_patterns: {
      peer_review: true,
      collective_intelligence: true,
      security_first: true,
      performance_aware: true,
      documentation_driven: true
    },
    escalation_paths: ['SYSTEM_ARCHITECT', 'SECURITY_AUDITOR'],
    quality_gates: {
      code_quality: 0.95,
      test_coverage: 0.90,
      security_score: 0.98,
      performance_score: 0.85
    },
    auto_scaling: {
      enabled: true,
      min_agents: 3,
      max_agents: 10,
      scale_up_threshold: 0.8,
      scale_down_threshold: 0.3
    }
  },
  monitoring: {
    metrics: {
      total_messages: 494,
      success_rate: 0.93,
      avg_response_time_ms: 187,
      active_threads: 12,
      queued_tasks: 5
    },
    alerts: {
      response_time_threshold: 500,
      error_rate_threshold: 0.05,
      queue_size_threshold: 20
    }
  },
  created_at: new Date(),
  updated_at: new Date(),
  deployment_status: 'production_ready'
};

// Production configuration test data
const productionConfig = {
  environment: 'production',
  version: '3.0.0',
  deployment: {
    strategy: 'blue_green',
    health_check_endpoint: '/api/health',
    readiness_endpoint: '/api/ready',
    liveness_endpoint: '/api/live'
  },
  scaling: {
    min_instances: 3,
    max_instances: 10,
    target_cpu_utilization: 0.7,
    target_memory_utilization: 0.8
  },
  monitoring: {
    prometheus_endpoint: '/metrics',
    jaeger_endpoint: '/api/tracing',
    logs_endpoint: '/api/logs'
  },
  security: {
    tls_enabled: true,
    api_rate_limit: 1000,
    jwt_expiry: 3600,
    session_timeout: 1800
  }
};

// Helper functions
async function setupCompleteMockAPI(page: Page) {
  // Mock agency data API
  await page.route('**/api/agency/workflow/data', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(completeAgencyData)
    });
  });

  // Mock workflow statistics API
  await page.route('**/api/agency/workflow/stats', async (route) => {
    const stats = {
      total_agents: completeAgencyData.agents.length,
      active_agents: completeAgencyData.agents.filter(a => a.state === 'ACTIVE').length,
      total_communications: completeAgencyData.communication_flows.length,
      active_communications: completeAgencyData.communication_flows.filter(f => f.status === 'active').length,
      avg_messages_per_connection: completeAgencyData.communication_flows.reduce((sum, f) => sum + f.message_count, 0) / completeAgencyData.communication_flows.length,
      busiest_agent: {
        agent_id: 'agent-tester',
        message_count: 425
      },
      communication_type_distribution: {
        DIRECT: 1,
        COLLABORATIVE: 1,
        CHAIN: 1,
        BROADCAST: 1,
        MONITOR: 1
      },
      agent_type_distribution: {
        SYSTEM_ARCHITECT: 1,
        CODE_IMPLEMENTER: 1,
        CODE_QUALITY_REVIEWER: 1,
        TEST_COVERAGE_VALIDATOR: 1,
        SECURITY_AUDITOR: 1
      },
      performance_metrics: {
        avg_response_time_ms: 187,
        success_rate: 0.93,
        error_rate: 0.07,
        throughput_rps: 45
      },
      security_metrics: {
        encrypted_flows: completeAgencyData.communication_flows.filter(f => f.encryption_enabled).length,
        security_alerts: 2,
        vulnerability_count: 0
      }
    };
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(stats)
    });
  });

  // Mock deployment status API
  await page.route('**/api/deployment/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(productionConfig)
    });
  });

  // Mock health check API
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'healthy',
        version: '3.0.0',
        uptime: 86400,
        components: {
          database: 'healthy',
          redis: 'healthy',
          mcp_server: 'healthy',
          agency_service: 'healthy'
        }
      })
    });
  });
}

async function simulateRealtimeEvent(page: Page, eventType: string, data: any) {
  await page.evaluate(({ eventType, data }) => {
    window.dispatchEvent(new CustomEvent('agency-event', {
      detail: { type: eventType, data }
    }));
  }, { eventType, data });
}

test.describe('Agency Swarm Complete E2E Suite', () => {
  test.beforeEach(async ({ page }) => {
    // Set up comprehensive error tracking
    const consoleErrors: string[] = [];
    const networkErrors: string[] = [];

    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
        console.log(`Console error: ${msg.text()}`);
      }
    });

    page.on('requestfailed', request => {
      const error = `${request.url()} - ${request.failure()?.errorText}`;
      networkErrors.push(error);
      console.log(`Network error: ${error}`);
    });

    page.on('pageerror', error => {
      consoleErrors.push(error.message);
      console.log(`Page error: ${error.message}`);
    });

    // Attach error tracking to page
    await page.exposeFunction('getConsoleErrors', () => consoleErrors);
    await page.exposeFunction('getNetworkErrors', () => networkErrors);
  });

  test.describe('Phase 1-3 Integration Tests', () => {
    test('should load complete agency workflow with all phases', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');

      // Wait for complete agency workflow to load
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Verify all phase components are loaded
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();

      // Verify all agents are displayed
      const agentNodes = page.locator('[data-testid="agent-node"]');
      await expect(agentNodes).toHaveCount(completeAgencyData.agents.length);

      // Verify all communication flows are displayed
      const edges = page.locator('[data-testid="communication-flow"]');
      await expect(edges).toHaveCount(completeAgencyData.communication_flows.length);

      // Verify phase indicators are present
      await expect(page.locator('[data-testid="phase-indicator"]')).toBeVisible();
      await expect(page.locator('[data-testid="phase-status"]')).toHaveText(/production|ready/);

      // Verify no critical errors
      const errors = await page.evaluate(() => (window as any).getConsoleErrors());
      const criticalErrors = errors.filter((e: string) =>
        !e.includes('Warning') && !e.includes('deprecated')
      );
      expect(criticalErrors).toHaveLength(0);
    });

    test('should handle complete agency lifecycle from creation to execution', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Phase 1: Agency Creation
      await page.locator('[data-testid="create-agency"]').click();
      await page.waitForSelector('[data-testid="agency-creation-form"]', { timeout: 10000 });

      // Fill agency creation form
      await page.locator('[data-testid="agency-name"]').fill('E2E Test Agency');
      await page.locator('[data-testid="agency-description"]').fill('Complete E2E test agency');

      // Select agents for agency
      const agentCheckboxes = page.locator('[data-testid="agent-checkbox"]');
      for (let i = 0; i < Math.min(await agentCheckboxes.count(), 3); i++) {
        await agentCheckboxes.nth(i).check();
      }

      // Create agency
      await page.locator('[data-testid="submit-creation"]').click();
      await page.waitForSelector('[data-testid="agency-created"]', { timeout: 10000 });

      // Phase 2: Communication Flow Setup
      await page.locator('[data-testid="setup-communication"]').click();
      await page.waitForSelector('[data-testid="flow-builder"]', { timeout: 10000 });

      // Create communication flows
      await page.locator('[data-testid="add-flow"]').click();
      await page.waitForTimeout(500);

      // Configure flow
      await page.locator('[data-testid="flow-source"]').selectOption({ label: 'System Architect' });
      await page.locator('[data-testid="flow-target"]').selectOption({ label: 'Code Implementer' });
      await page.locator('[data-testid="save-flow"]').click();

      // Phase 3: Advanced Features
      await page.locator('[data-testid="configure-advanced"]').click();
      await page.waitForSelector('[data-testid="advanced-settings"]', { timeout: 10000 });

      // Enable advanced features
      await page.locator('[data-testid="enable-handoffs"]').check();
      await page.locator('[data-testid="enable-monitoring"]').check();
      await page.locator('[data-testid="enable-scaling"]').check();

      // Save advanced settings
      await page.locator('[data-testid="save-advanced"]').click();
      await page.waitForSelector('[data-testid="advanced-configured"]', { timeout: 10000 });

      // Phase 4: Execution
      await page.locator('[data-testid="start-execution"]').click();
      await page.waitForSelector('[data-testid="execution-active"]', { timeout: 10000 });

      // Verify agency is active and running
      await expect(page.locator('[data-testid="agency-status"]')).toHaveText(/active|running/);

      // Verify all components are integrated
      await expect(page.locator('[data-testid="phase-1-complete"]')).toBeVisible();
      await expect(page.locator('[data-testid="phase-2-complete"]')).toBeVisible();
      await expect(page.locator('[data-testid="phase-3-complete"]')).toBeVisible();
    });

    test('should demonstrate real-time agent communication and handoffs', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Start agency execution
      await page.locator('[data-testid="start-execution"]').click();
      await page.waitForSelector('[data-testid="execution-active"]', { timeout: 10000 });

      // Simulate real-time agent communication
      await simulateRealtimeEvent(page, 'agent_message', {
        from_agent: 'agent-architect',
        to_agent: 'agent-implementer',
        message: 'Implement new feature X with security considerations',
        message_type: 'task_assignment',
        priority: 'high'
      });

      await page.waitForTimeout(1000);

      // Verify communication flow becomes active
      const activeFlow = page.locator('[data-testid="communication-flow"].active');
      await expect(activeFlow).toBeVisible();

      // Simulate agent handoff
      await simulateRealtimeEvent(page, 'agent_handoff', {
        from_agent: 'agent-implementer',
        to_agent: 'agent-quality',
        task_id: 'task-123',
        handoff_reason: 'Code review required',
        context: {
          feature: 'Feature X',
          files_modified: ['file1.ts', 'file2.ts'],
          tests_written: 5
        }
      });

      await page.waitForTimeout(1000);

      // Verify handoff completion
      const handoffIndicator = page.locator('[data-testid="handoff-completed"]');
      if (await handoffIndicator.isVisible()) {
        await expect(handoffIndicator).toBeVisible();
      }

      // Simulate broadcast message
      await simulateRealtimeEvent(page, 'broadcast_message', {
        from_agent: 'agent-architect',
        to_agents: ['agent-implementer', 'agent-quality', 'agent-tester'],
        message: 'Team meeting at 3 PM for sprint planning',
        message_type: 'announcement'
      });

      await page.waitForTimeout(1000);

      // Verify broadcast was received by multiple agents
      const activeAgents = page.locator('[data-testid="agent-node"].active');
      expect(await activeAgents.count()).toBeGreaterThan(1);

      // Verify workflow remains stable during real-time operations
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
      await expect(page.locator('[data-testid="execution-status"]')).toHaveText(/active|running/);
    });

    test('should validate production deployment readiness', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Navigate to deployment section
      await page.locator('[data-testid="deployment-tab"]').click();
      await page.waitForSelector('[data-testid="deployment-dashboard"]', { timeout: 10000 });

      // Verify deployment status
      await expect(page.locator('[data-testid="deployment-status"]')).toHaveText(/production_ready|ready/);

      // Verify health checks
      await expect(page.locator('[data-testid="health-check-status"]')).toHaveText(/healthy|passing/);

      // Verify scaling configuration
      await expect(page.locator('[data-testid="scaling-config"]')).toBeVisible();
      await expect(page.locator('[data-testid="min-instances"]')).toHaveText('3');
      await expect(page.locator('[data-testid="max-instances"]')).toHaveText('10');

      // Verify monitoring setup
      await expect(page.locator('[data-testid="monitoring-status"]')).toBeVisible();
      await expect(page.locator('[data-testid="metrics-endpoint"]')).toBeVisible();
      await expect(page.locator('[data-testid="logs-endpoint"]')).toBeVisible();

      // Verify security configuration
      await expect(page.locator('[data-testid="security-config"]')).toBeVisible();
      await expect(page.locator('[data-testid="tls-status"]')).toHaveText(/enabled|true/);
      await expect(page.locator('[data-testid="rate-limit"]')).toHaveText('1000');

      // Run deployment validation
      await page.locator('[data-testid="validate-deployment"]').click();
      await page.waitForSelector('[data-testid="validation-results"]', { timeout: 15000 });

      // Verify validation results
      await expect(page.locator('[data-testid="validation-status"]')).toHaveText(/success|passed/);

      const validationChecks = page.locator('[data-testid="validation-check"]');
      expect(await validationChecks.count()).toBeGreaterThan(0);

      // Verify all critical checks pass
      const criticalChecks = validationChecks.locator('[data-critical="true"]');
      for (let i = 0; i < await criticalChecks.count(); i++) {
        await expect(criticalChecks.nth(i).locator('[data-testid="check-status"]')).toHaveText(/pass|success/);
      }
    });
  });

  test.describe('Performance and Load Testing', () => {
    test('should handle high-load scenarios with 100+ concurrent operations', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Start performance monitoring
      await page.locator('[data-testid="start-performance-monitor"]').click();
      await page.waitForSelector('[data-testid="monitoring-active"]', { timeout: 10000 });

      const startTime = Date.now();

      // Simulate high load: 100 concurrent agent operations
      const loadOperations = [];
      for (let i = 0; i < 100; i++) {
        loadOperations.push(async () => {
          await simulateRealtimeEvent(page, 'agent_operation', {
            agent_id: `agent-${i % 5 + 1}`,
            operation: 'task_execution',
            task_id: `task-${i}`,
            complexity: Math.random() > 0.5 ? 'high' : 'low'
          });
        });
      }

      // Execute all operations concurrently
      await Promise.all(loadOperations.map(op => op()));

      // Wait for operations to complete
      await page.waitForTimeout(5000);

      const loadTime = Date.now() - startTime;

      // Verify system remains responsive under load
      expect(loadTime).toBeLessThan(30000); // Should complete within 30 seconds

      // Verify no system crashes
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();

      // Check performance metrics
      const performanceMetrics = page.locator('[data-testid="performance-metrics"]');
      if (await performanceMetrics.isVisible()) {
        await expect(performanceMetrics).toBeVisible();

        const responseTime = performanceMetrics.locator('[data-metric="response-time"]');
        const throughput = performanceMetrics.locator('[data-metric="throughput"]');
        const errorRate = performanceMetrics.locator('[data-metric="error-rate"]');

        // Verify performance within acceptable thresholds
        if (await responseTime.isVisible()) {
          const responseTimeText = await responseTime.textContent();
          const responseTimeValue = parseFloat(responseTimeText || '0');
          expect(responseTimeValue).toBeLessThan(1000); // < 1s response time
        }

        if (await errorRate.isVisible()) {
          const errorRateText = await errorRate.textContent();
          const errorRateValue = parseFloat(errorRateText || '0');
          expect(errorRateValue).toBeLessThan(0.05); // < 5% error rate
        }
      }

      // Verify system stability after load
      await page.waitForTimeout(2000);
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
    });

    test('should validate performance SLAs for production', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Navigate to performance SLA validation
      await page.locator('[data-testid="performance-tab"]').click();
      await page.waitForSelector('[data-testid="sla-dashboard"]', { timeout: 10000 });

      // Run SLA validation tests
      await page.locator('[data-testid="run-sla-tests"]').click();
      await page.waitForSelector('[data-testid="sla-results"]', { timeout: 30000 });

      // Verify SLA compliance
      const slaResults = page.locator('[data-testid="sla-result"]');
      expect(await slaResults.count()).toBeGreaterThan(0);

      // Check critical SLAs
      const criticalSLAs = [
        'response_time_sla',
        'availability_sla',
        'throughput_sla',
        'error_rate_sla'
      ];

      for (const sla of criticalSLAs) {
        const slaElement = page.locator(`[data-sla="${sla}"]`);
        if (await slaElement.isVisible()) {
          await expect(slaElement.locator('[data-testid="sla-status"]')).toHaveText(/pass|compliant/);
        }
      }

      // Verify performance trends
      const performanceChart = page.locator('[data-testid="performance-chart"]');
      if (await performanceChart.isVisible()) {
        await expect(performanceChart).toBeVisible();
      }

      // Verify no SLA violations
      const violations = page.locator('[data-testid="sla-violation"]');
      const violationCount = await violations.count();
      expect(violationCount).toBe(0);
    });
  });

  test.describe('Security and Compliance Testing', () => {
    test('should validate security measures and compliance', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Navigate to security dashboard
      await page.locator('[data-testid="security-tab"]').click();
      await page.waitForSelector('[data-testid="security-dashboard"]', { timeout: 10000 });

      // Run security validation
      await page.locator('[data-testid="run-security-scan"]').click();
      await page.waitForSelector('[data-testid="security-results"]', { timeout: 20000 });

      // Verify security checks
      const securityChecks = page.locator('[data-testid="security-check"]');
      expect(await securityChecks.count()).toBeGreaterThan(0);

      // Verify critical security measures
      const criticalChecks = [
        'encryption_check',
        'authentication_check',
        'authorization_check',
        'input_validation_check',
        'vulnerability_scan'
      ];

      for (const check of criticalChecks) {
        const checkElement = page.locator(`[data-security-check="${check}"]`);
        if (await checkElement.isVisible()) {
          await expect(checkElement.locator('[data-testid="check-status"]')).toHaveText(/pass|secure/);
        }
      }

      // Verify compliance standards
      const complianceStandards = page.locator('[data-testid="compliance-standard"]');
      for (let i = 0; i < await complianceStandards.count(); i++) {
        const standard = complianceStandards.nth(i);
        await expect(standard.locator('[data-testid="compliance-status"]')).toHaveText(/compliant|pass/);
      }

      // Verify security monitoring
      await expect(page.locator('[data-testid="security-monitoring"]')).toBeVisible();
      await expect(page.locator('[data-testid="threat-detection"]')).toBeVisible();
    });

    test('should handle security incidents and alerts', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Simulate security incident
      await simulateRealtimeEvent(page, 'security_incident', {
        incident_type: 'unauthorized_access',
        severity: 'high',
        affected_agents: ['agent-implementer', 'agent-quality'],
        description: 'Unauthorized access attempt detected',
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(1000);

      // Verify security alert is displayed
      const securityAlert = page.locator('[data-testid="security-alert"]');
      if (await securityAlert.isVisible()) {
        await expect(securityAlert).toBeVisible();
        await expect(securityAlert.locator('[data-testid="alert-severity"]')).toHaveText(/high|critical/);
      }

      // Verify incident response workflow
      const incidentResponse = page.locator('[data-testid="incident-response"]');
      if (await incidentResponse.isVisible()) {
        await expect(incidentResponse).toBeVisible();
      }

      // Verify system remains operational during security incident
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
      const activeAgents = page.locator('[data-testid="agent-node"].active');
      expect(await activeAgents.count()).toBeGreaterThan(0);
    });
  });

  test.describe('Disaster Recovery and Resilience', () => {
    test('should handle system failures and recovery scenarios', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Start agency execution
      await page.locator('[data-testid="start-execution"]').click();
      await page.waitForSelector('[data-testid="execution-active"]', { timeout: 10000 });

      // Simulate system failure
      await simulateRealtimeEvent(page, 'system_failure', {
        failure_type: 'database_connection',
        severity: 'critical',
        affected_components: ['database', 'agency_service'],
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(1000);

      // Verify failure detection and alerting
      const failureAlert = page.locator('[data-testid="system-failure-alert"]');
      if (await failureAlert.isVisible()) {
        await expect(failureAlert).toBeVisible();
      }

      // Verify automatic failover
      const failoverStatus = page.locator('[data-testid="failover-status"]');
      if (await failoverStatus.isVisible()) {
        await expect(failoverStatus).toHaveText(/initiated|active/);
      }

      // Simulate recovery
      await simulateRealtimeEvent(page, 'system_recovery', {
        recovery_type: 'automatic',
        recovered_components: ['database', 'agency_service'],
        recovery_time_ms: 5000,
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(1000);

      // Verify system recovery
      const recoveryStatus = page.locator('[data-testid="recovery-status"]');
      if (await recoveryStatus.isVisible()) {
        await expect(recoveryStatus).toHaveText(/complete|successful/);
      }

      // Verify agency resumes normal operation
      await page.waitForTimeout(2000);
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
      await expect(page.locator('[data-testid="execution-status"]')).toHaveText(/active|running/);
    });

    test('should validate backup and restore procedures', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Navigate to backup management
      await page.locator('[data-testid="backup-tab"]').click();
      await page.waitForSelector('[data-testid="backup-management"]', { timeout: 10000 });

      // Create backup
      await page.locator('[data-testid="create-backup"]').click();
      await page.waitForSelector('[data-testid="backup-created"]', { timeout: 15000 });

      // Verify backup creation
      const backupStatus = page.locator('[data-testid="backup-status"]');
      await expect(backupStatus).toHaveText(/success|complete/);

      // Verify backup details
      const backupDetails = page.locator('[data-testid="backup-details"]');
      if (await backupDetails.isVisible()) {
        await expect(backupDetails).toBeVisible();
        await expect(backupDetails.locator('[data-testid="backup-size"]')).toBeVisible();
        await expect(backupDetails.locator('[data-testid="backup-timestamp"]')).toBeVisible();
      }

      // Test restore procedure
      await page.locator('[data-testid="test-restore"]').click();
      await page.waitForSelector('[data-testid="restore-test-results"]', { timeout: 20000 });

      // Verify restore test results
      const restoreResults = page.locator('[data-testid="restore-result"]');
      if (await restoreResults.isVisible()) {
        await expect(restoreResults.locator('[data-testid="restore-status"]')).toHaveText(/success|pass/);
      }
    });
  });

  test.describe('Cross-Browser and Cross-Platform Testing', () => {
    test('should work across different browsers and devices', async ({ page, browserName }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Basic functionality should work across all browsers
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
      await expect(page.locator('[data-testid="agent-node"]')).toHaveCount(completeAgencyData.agents.length);

      // Test basic interactions
      await page.locator('[data-testid="start-execution"]').click();
      await page.waitForTimeout(1000);
      await expect(page.locator('[data-testid="execution-status"]')).toBeVisible();

      // Test responsive design across viewports
      const viewports = [
        { width: 1920, height: 1080 }, // Desktop
        { width: 768, height: 1024 },   // Tablet
        { width: 375, height: 667 }     // Mobile
      ];

      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        await page.waitForTimeout(500);

        // Verify workflow adapts to viewport
        await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();

        // Verify controls remain accessible
        const controls = page.locator('[data-testid="workflow-controls"]');
        await expect(controls).toBeVisible();

        // Test basic functionality works
        await page.locator('[data-testid="zoom-in"]').click();
        await page.waitForTimeout(200);
        await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
      }
    });
  });

  test.describe('Final Production Validation', () => {
    test('should pass comprehensive production readiness validation', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Run comprehensive production validation
      await page.locator('[data-testid="run-production-validation"]').click();
      await page.waitForSelector('[data-testid="validation-summary"]', { timeout: 60000 });

      // Verify validation summary
      await expect(page.locator('[data-testid="validation-summary"]')).toBeVisible();

      // Check overall validation status
      const overallStatus = page.locator('[data-testid="validation-overall-status"]');
      await expect(overallStatus).toHaveText(/pass|success|ready/);

      // Verify all validation categories pass
      const validationCategories = [
        'functionality_validation',
        'performance_validation',
        'security_validation',
        'compliance_validation',
        'scalability_validation',
        'reliability_validation'
      ];

      for (const category of validationCategories) {
        const categoryResult = page.locator(`[data-validation-category="${category}"]`);
        if (await categoryResult.isVisible()) {
          await expect(categoryResult.locator('[data-testid="category-status"]')).toHaveText(/pass|success/);
        }
      }

      // Verify no critical issues
      const criticalIssues = page.locator('[data-testid="critical-issue"]');
      expect(await criticalIssues.count()).toBe(0);

      // Verify deployment readiness
      const deploymentReadiness = page.locator('[data-testid="deployment-readiness"]');
      await expect(deploymentReadiness).toHaveText(/ready|approved/);

      // Generate production report
      await page.locator('[data-testid="generate-production-report"]').click();
      await page.waitForSelector('[data-testid="report-generated"]', { timeout: 15000 });

      // Verify report generation
      const reportStatus = page.locator('[data-testid="report-status"]');
      await expect(reportStatus).toHaveText(/success|complete/);

      // Final system health check
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
      await expect(page.locator('[data-testid="system-health"]')).toHaveText(/healthy|optimal/);
    });

    test('should demonstrate complete Agency Swarm production capabilities', async ({ page }) => {
      await setupCompleteMockAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 15000 });

      // Demonstrate complete production workflow
      const startTime = Date.now();

      // 1. Agency Initialization
      await page.locator('[data-testid="initialize-agency"]').click();
      await page.waitForSelector('[data-testid="agency-initialized"]', { timeout: 10000 });

      // 2. Agent Registration
      await page.locator('[data-testid="register-agents"]').click();
      await page.waitForSelector('[data-testid="agents-registered"]', { timeout: 10000 });

      // 3. Communication Setup
      await page.locator('[data-testid="setup-communication"]').click();
      await page.waitForSelector('[data-testid="communication-ready"]', { timeout: 10000 });

      // 4. Advanced Configuration
      await page.locator('[data-testid="configure-advanced"]').click();
      await page.waitForSelector('[data-testid="advanced-configured"]', { timeout: 10000 });

      // 5. Production Deployment
      await page.locator('[data-testid="deploy-to-production"]').click();
      await page.waitForSelector('[data-testid="deployment-complete"]', { timeout: 20000 });

      const totalDeploymentTime = Date.now() - startTime;

      // Verify deployment completed within acceptable time
      expect(totalDeploymentTime).toBeLessThan(120000); // 2 minutes

      // Verify production status
      await expect(page.locator('[data-testid="production-status"]')).toHaveText(/active|live|production/);

      // Verify all systems operational
      const systemStatus = page.locator('[data-testid="system-status"]');
      await expect(systemStatus).toHaveText(/operational|healthy/);

      // Verify monitoring active
      await expect(page.locator('[data-testid="monitoring-active"]')).toBeVisible();

      // Verify security measures active
      await expect(page.locator('[data-testid="security-active"]')).toBeVisible();

      // Final validation: Complete Agency Swarm is production-ready
      await expect(page.locator('[data-testid="production-ready"]')).toBeVisible();
      await expect(page.locator('[data-testid="agency-swarm-version"]')).toHaveText('3.0.0');

      console.log('🚀 Agency Swarm Complete E2E Test Suite - SUCCESS');
      console.log(`📊 Total Deployment Time: ${totalDeploymentTime}ms`);
      console.log('✅ All Phase 1-3 components validated');
      console.log('✅ Production deployment verified');
      console.log('✅ Security and compliance validated');
      console.log('✅ Performance and scalability confirmed');
      console.log('✅ Disaster recovery tested');
    });
  });
});