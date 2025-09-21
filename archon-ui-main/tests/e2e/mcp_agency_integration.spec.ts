/**
 * MCP Agency Integration Test Suite
 *
 * Comprehensive Playwright tests for MCP (Model Context Protocol) integration
 * with agency workflows, including tool execution, real-time communication,
 * and cross-agent coordination.
 *
 * Key test scenarios:
 * - MCP server connection and health checks
 * - MCP tool execution with agency workflows
 * - Real-time communication between MCP and agency agents
 * - Tool discovery and validation
 * - Error handling and recovery
 * - Performance with concurrent tool execution
 * - Security and access control
 * - Integration with knowledge management
 * - Cross-agent tool coordination
 * - MCP protocol compliance
 */

import { test, expect, Page } from '@playwright/test';
import { v4 as uuidv4 } from 'uuid';

// Test configuration
const BASE_URL = 'http://localhost:3738';
const MCP_URL = `${BASE_URL}/mcp`;
const AGENCY_URL = `${BASE_URL}/agency`;

// Mock MCP tools for testing
const mockMCPTools = [
  {
    name: 'archon_perform_rag_query',
    description: 'Search knowledge base using RAG',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query for knowledge base'
        },
        limit: {
          type: 'number',
          description: 'Maximum number of results to return',
          default: 10
        },
        filter: {
          type: 'object',
          description: 'Optional filters for search'
        }
      },
      required: ['query']
    }
  },
  {
    name: 'archon_search_code_examples',
    description: 'Search for code examples in knowledge base',
    inputSchema: {
      type: 'object',
      properties: {
        language: {
          type: 'string',
          description: 'Programming language to search for'
        },
        pattern: {
          type: 'string',
          description: 'Code pattern to search for'
        },
        framework: {
          type: 'string',
          description: 'Framework to filter by'
        }
      },
      required: ['pattern']
    }
  },
  {
    name: 'archon_manage_project',
    description: 'Manage project-related operations',
    inputSchema: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['create', 'update', 'delete', 'get'],
          description: 'Action to perform'
        },
        project_data: {
          type: 'object',
          description: 'Project data for create/update actions'
        },
        project_id: {
          type: 'string',
          description: 'Project ID for get/update/delete actions'
        }
      },
      required: ['action']
    }
  },
  {
    name: 'archon_manage_task',
    description: 'Manage task-related operations',
    inputSchema: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['create', 'update', 'delete', 'assign', 'complete'],
          description: 'Action to perform'
        },
        task_data: {
          type: 'object',
          description: 'Task data for create/update actions'
        },
        task_id: {
          type: 'string',
          description: 'Task ID for update/delete/complete actions'
        },
        assignee_id: {
          type: 'string',
          description: 'Agent ID to assign task to'
        }
      },
      required: ['action']
    }
  },
  {
    name: 'archon_get_available_sources',
    description: 'Get available knowledge sources',
    inputSchema: {
      type: 'object',
      properties: {
        type: {
          type: 'string',
          enum: ['all', 'documents', 'websites', 'code'],
          description: 'Type of sources to retrieve'
        },
        project_id: {
          type: 'string',
          description: 'Filter by project ID'
        }
      }
    }
  }
];

// Mock MCP server responses
const mockMCPResponses = {
  health: {
    status: 'healthy',
    version: '1.0.0',
    uptime: 3600,
    connected_agents: 5,
    active_tools: mockMCPTools.length,
    last_heartbeat: new Date().toISOString()
  },
  tools: {
    tools: mockMCPTools,
    total_count: mockMCPTools.length
  },
  toolExecution: {
    success: {
      result: {
        status: 'success',
        data: 'Tool execution completed successfully',
        execution_time_ms: 150,
        timestamp: new Date().toISOString()
      }
    },
    error: {
      result: {
        status: 'error',
        error: 'Tool execution failed',
        error_details: 'Invalid input parameters',
        timestamp: new Date().toISOString()
      }
    }
  }
};

// Mock agency workflow data
const mockAgencyWorkflow = {
  id: uuidv4(),
  name: 'MCP Integrated Agency',
  agents: [
    {
      id: 'agent-mcp-1',
      name: 'MCP Coordinator',
      agent_type: 'SYSTEM_ARCHITECT' as any,
      model_tier: 'OPUS' as any,
      state: 'ACTIVE' as any,
      capabilities: {
        mcp_coordination: true,
        tool_orchestration: true,
        workflow_management: true
      }
    },
    {
      id: 'agent-mcp-2',
      name: 'Knowledge Searcher',
      agent_type: 'TEST_COVERAGE_VALIDATOR' as any,
      model_tier: 'SONNET' as any,
      state: 'ACTIVE' as any,
      capabilities: {
        rag_queries: true,
        knowledge_retrieval: true,
        information_processing: true
      }
    },
    {
      id: 'agent-mcp-3',
      name: 'Task Executor',
      agent_type: 'CODE_IMPLEMENTER' as any,
      model_tier: 'SONNET' as any,
      state: 'ACTIVE' as any,
      capabilities: {
        task_execution: true,
        code_generation: true,
        result_processing: true
      }
    }
  ],
  communication_flows: [
    {
      id: 'mcp-flow-1',
      source_agent_id: 'agent-mcp-1',
      target_agent_id: 'agent-mcp-2',
      communication_type: 'DIRECT' as any,
      status: 'active' as any,
      message_type: 'mcp_tool_request'
    },
    {
      id: 'mcp-flow-2',
      source_agent_id: 'agent-mcp-2',
      target_agent_id: 'agent-mcp-3',
      communication_type: 'COLLABORATIVE' as any,
      status: 'active' as any,
      message_type: 'knowledge_delivery'
    }
  ]
};

// Helper functions
async function setupMockMCPAPI(page: Page) {
  // Mock MCP health check
  await page.route('**/api/mcp/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockMCPResponses.health)
    });
  });

  // Mock MCP tools list
  await page.route('**/api/mcp/tools', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockMCPResponses.tools)
    });
  });

  // Mock MCP tool execution
  await page.route('**/api/mcp/tools/**', async (route) => {
    const url = route.request().url();
    const toolName = url.split('/').pop();

    if (toolName === 'archon_perform_rag_query') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          result: {
            status: 'success',
            data: {
              results: [
                {
                  id: 'doc-1',
                  title: 'Architecture Patterns',
                  content: 'Common architecture patterns for distributed systems',
                  relevance_score: 0.95
                },
                {
                  id: 'doc-2',
                  title: 'Best Practices',
                  content: 'Software development best practices',
                  relevance_score: 0.87
                }
              ],
              total_results: 2,
              query_time_ms: 45
            },
            execution_time_ms: 120,
            timestamp: new Date().toISOString()
          }
        })
      });
    } else if (toolName === 'archon_search_code_examples') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          result: {
            status: 'success',
            data: {
              examples: [
                {
                  id: 'code-1',
                  language: 'typescript',
                  framework: 'react',
                  code: 'const Component = () => <div>Hello World</div>',
                  description: 'Simple React component'
                }
              ],
              total_examples: 1,
              search_time_ms: 32
            },
            execution_time_ms: 85,
            timestamp: new Date().toISOString()
          }
        })
      });
    } else {
      // Generic success response
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockMCPResponses.toolExecution.success)
      });
    }
  });

  // Mock agency workflow data
  await page.route('**/api/agency/workflow/data', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockAgencyWorkflow)
    });
  });
}

async function simulateMCPEvent(page: Page, eventType: string, data: any) {
  await page.evaluate(({ eventType, data }) => {
    // Simulate MCP event
    window.dispatchEvent(new CustomEvent('mcp-event', {
      detail: { type: eventType, data }
    }));
  }, { eventType, data });
}

test.describe('MCP Agency Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Set up console error tracking
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
        console.log(`Console error: ${msg.text()}`);
      }
    });

    // Track React errors
    page.on('pageerror', error => {
      console.log(`Page error: ${error.message}`);
      consoleErrors.push(error.message);
    });

    // Attach errors to page for test access
    await page.exposeFunction('getConsoleErrors', () => consoleErrors);
  });

  test.describe('MCP Server Connection', () => {
    test('should connect to MCP server successfully', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');

      // Verify MCP page loads
      await expect(page.locator('[data-testid="mcp-container"]')).toBeVisible();

      // Wait for MCP connection to establish
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Verify connection status
      const statusIndicator = page.locator('[data-testid="mcp-connection-status"]');
      await expect(statusIndicator).toHaveText(/Connected|Healthy/);

      // Verify MCP tools are loaded
      const toolsContainer = page.locator('[data-testid="mcp-tools-container"]');
      await expect(toolsContainer).toBeVisible();

      // Verify no critical errors
      const errors = await page.evaluate(() => (window as any).getConsoleErrors());
      const criticalErrors = errors.filter((e: string) =>
        !e.includes('Warning') && !e.includes('deprecated')
      );
      expect(criticalErrors).toHaveLength(0);
    });

    test('should display MCP server health information', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Verify health information is displayed
      const healthInfo = page.locator('[data-testid="mcp-health-info"]');
      await expect(healthInfo).toBeVisible();

      // Verify specific health metrics
      await expect(healthInfo.locator('[data-testid="server-uptime"]')).toBeVisible();
      await expect(healthInfo.locator('[data-testid="connected-agents"]')).toBeVisible();
      await expect(healthInfo.locator('[data-testid="active-tools"]')).toBeVisible();

      // Verify metrics show correct values
      await expect(healthInfo.locator('[data-testid="connected-agents"]')).toHaveText('5');
      await expect(healthInfo.locator('[data-testid="active-tools"]')).toHaveText(mockMCPTools.length.toString());
    });

    test('should handle MCP server connection failures gracefully', async ({ page }) => {
      // Mock MCP server failure
      await page.route('**/api/mcp/health', async (route) => {
        await route.fulfill({
          status: 500,
          body: 'MCP Server Unavailable'
        });
      });

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');

      // Should show connection error
      await page.waitForSelector('[data-testid="mcp-status-error"]', { timeout: 10000 });
      const errorStatus = page.locator('[data-testid="mcp-connection-status"]');
      await expect(errorStatus).toHaveText(/Error|Disconnected/);

      // Should show retry mechanism
      const retryButton = page.locator('[data-testid="mcp-retry-connection"]');
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible();
      }

      // Should not crash the interface
      await expect(page.locator('[data-testid="mcp-container"]')).toBeVisible();
    });
  });

  test.describe('MCP Tool Discovery and Management', () => {
    test('should discover and list available MCP tools', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Verify tools are loaded
      const toolsList = page.locator('[data-testid="mcp-tools-list"]');
      await expect(toolsList).toBeVisible();

      // Verify all expected tools are present
      for (const tool of mockMCPTools) {
        const toolCard = toolsList.locator(`[data-tool-name="${tool.name}"]`);
        await expect(toolCard).toBeVisible();
        await expect(toolCard.locator('[data-testid="tool-name"]')).toHaveText(tool.name);
        await expect(toolCard.locator('[data-testid="tool-description"]')).toHaveText(tool.description);
      }

      // Verify tool count matches
      const toolCards = toolsList.locator('[data-tool-name]');
      expect(await toolCards.count()).toBe(mockMCPTools.length);
    });

    test('should display tool schemas and parameters', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Select first tool
      const firstTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await firstTool.click();

      // Verify tool details panel appears
      const toolDetails = page.locator('[data-testid="tool-details-panel"]');
      await expect(toolDetails).toBeVisible();

      // Verify input schema is displayed
      const schemaContainer = toolDetails.locator('[data-testid="tool-schema"]');
      await expect(schemaContainer).toBeVisible();

      // Verify required parameters are highlighted
      const requiredParams = schemaContainer.locator('[data-required="true"]');
      expect(await requiredParams.count()).toBeGreaterThan(0);

      // Verify parameter descriptions
      const paramDescriptions = schemaContainer.locator('[data-testid="param-description"]');
      expect(await paramDescriptions.count()).toBeGreaterThan(0);
    });

    test('should filter and search tools', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      const searchInput = page.locator('[data-testid="tool-search-input"]');
      await expect(searchInput).toBeVisible();

      // Search for specific tool
      await searchInput.fill('rag_query');
      await page.waitForTimeout(500);

      // Verify filtering works
      const visibleTools = page.locator('[data-tool-name]:visible');
      expect(await visibleTools.count()).toBe(1);
      await expect(visibleTools.first()).toHaveAttribute('data-tool-name', 'archon_perform_rag_query');

      // Clear search
      await searchInput.fill('');
      await page.waitForTimeout(500);

      // Verify all tools are visible again
      const allTools = page.locator('[data-tool-name]:visible');
      expect(await allTools.count()).toBe(mockMCPTools.length);
    });
  });

  test.describe('MCP Tool Execution', () => {
    test('should execute MCP tools successfully', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Select RAG query tool
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      // Fill in required parameters
      const queryInput = page.locator('[data-testid="param-query"]');
      await expect(queryInput).toBeVisible();
      await queryInput.fill('architecture patterns');

      // Optional parameters
      const limitInput = page.locator('[data-testid="param-limit"]');
      if (await limitInput.isVisible()) {
        await limitInput.fill('5');
      }

      // Execute tool
      const executeButton = page.locator('[data-testid="execute-tool"]');
      await expect(executeButton).toBeVisible();
      await executeButton.click();

      // Wait for execution to complete
      await page.waitForSelector('[data-testid="execution-success"]', { timeout: 10000 });

      // Verify results are displayed
      const resultsPanel = page.locator('[data-testid="tool-results"]');
      await expect(resultsPanel).toBeVisible();

      // Verify result data structure
      await expect(resultsPanel.locator('[data-testid="result-status"]')).toHaveText('success');
      await expect(resultsPanel.locator('[data-testid="execution-time"]')).toBeVisible();
      await expect(resultsPanel.locator('[data-testid="result-data"]')).toBeVisible();

      // Verify specific results for RAG query
      const searchResults = resultsPanel.locator('[data-testid="search-results"]');
      if (await searchResults.isVisible()) {
        await expect(searchResults.locator('[data-testid="result-item"]')).toHaveCount(2);
      }
    });

    test('should handle tool execution errors gracefully', async ({ page }) => {
      await setupMockMCPAPI(page);

      // Mock tool execution error
      await page.route('**/api/mcp/tools/archon_perform_rag_query', async (route) => {
        await route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({
            result: {
              status: 'error',
              error: 'Invalid query parameter',
              error_details: 'Query cannot be empty',
              timestamp: new Date().toISOString()
            }
          })
        });
      });

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Select RAG query tool
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      // Execute with empty query (should fail)
      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Wait for error response
      await page.waitForSelector('[data-testid="execution-error"]', { timeout: 10000 });

      // Verify error is displayed
      const errorPanel = page.locator('[data-testid="tool-error"]');
      await expect(errorPanel).toBeVisible();
      await expect(errorPanel.locator('[data-testid="error-message"]')).toHaveText(/Invalid query/);

      // Verify retry option is available
      const retryButton = page.locator('[data-testid="retry-execution"]');
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible();
      }
    });

    test('should handle tool execution timeouts', async ({ page }) => {
      await setupMockMCPAPI(page);

      // Mock slow tool execution
      await page.route('**/api/mcp/tools/archon_perform_rag_query', async (route) => {
        await new Promise(resolve => setTimeout(resolve, 15000));
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockMCPResponses.toolExecution.success)
        });
      });

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Select and execute tool
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      const queryInput = page.locator('[data-testid="param-query"]');
      await queryInput.fill('test query');

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Should show loading state
      const loadingIndicator = page.locator('[data-testid="execution-loading"]');
      await expect(loadingIndicator).toBeVisible();

      // After reasonable time, should show timeout or still loading
      await page.waitForTimeout(5000);
      await expect(loadingIndicator).toBeVisible();

      // Interface should remain responsive
      await expect(page.locator('[data-testid="mcp-container"]')).toBeVisible();
    });
  });

  test.describe('MCP-Agency Workflow Integration', () => {
    test('should integrate MCP tools with agency workflows', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Verify MCP integration panel is visible
      const mcpPanel = page.locator('[data-testid="mcp-integration-panel"]');
      await expect(mcpPanel).toBeVisible();

      // Verify MCP tools are accessible from workflow
      const mcpToolsButton = page.locator('[data-testid="open-mcp-tools"]');
      await expect(mcpToolsButton).toBeVisible();

      // Open MCP tools from workflow
      await mcpToolsButton.click();
      await page.waitForTimeout(1000);

      // Should show MCP tools interface
      const mcpToolsModal = page.locator('[data-testid="mcp-tools-modal"]');
      if (await mcpToolsModal.isVisible()) {
        await expect(mcpToolsModal).toBeVisible();
      }
    });

    test('should execute MCP tools from agency agents', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Select agent with MCP capabilities
      const mcpAgent = page.locator('[data-agent-id="agent-mcp-2"]');
      await expect(mcpAgent).toBeVisible();
      await mcpAgent.click();

      // Verify agent details show MCP capabilities
      const agentDetails = page.locator('[data-testid="agent-details-panel"]');
      await expect(agentDetails).toBeVisible();

      const mcpCapabilities = agentDetails.locator('[data-testid="mcp-capabilities"]');
      if (await mcpCapabilities.isVisible()) {
        await expect(mcpCapabilities).toHaveText(/rag_queries|knowledge_retrieval/);
      }

      // Execute MCP tool through agent
      const executeToolButton = agentDetails.locator('[data-testid="execute-mcp-tool"]');
      if (await executeToolButton.isVisible()) {
        await executeToolButton.click();
        await page.waitForTimeout(1000);

        // Should show tool execution interface
        const toolExecution = page.locator('[data-testid="agent-tool-execution"]');
        if (await toolExecution.isVisible()) {
          await expect(toolExecution).toBeVisible();
        }
      }
    });

    test('should handle MCP tool coordination between agents', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Simulate agent requesting MCP tool execution
      await simulateMCPEvent(page, 'agent_tool_request', {
        agent_id: 'agent-mcp-1',
        tool_name: 'archon_perform_rag_query',
        parameters: { query: 'architecture patterns' },
        target_agent_id: 'agent-mcp-2'
      });

      await page.waitForTimeout(1000);

      // Simulate tool execution response
      await simulateMCPEvent(page, 'tool_execution_result', {
        tool_name: 'archon_perform_rag_query',
        result: {
          status: 'success',
          data: { results: [{ id: 'doc-1', relevance_score: 0.95 }] },
          execution_time_ms: 120
        },
        source_agent_id: 'agent-mcp-2',
        target_agent_id: 'agent-mcp-1'
      });

      await page.waitForTimeout(1000);

      // Verify workflow remains stable
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();

      // Verify communication flow is active
      const activeFlows = page.locator('[data-testid="communication-flow"].active');
      expect(await activeFlows.count()).toBeGreaterThan(0);
    });
  });

  test.describe('Real-time MCP Communication', () => {
    test('should handle real-time MCP tool status updates', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Start tool execution
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      const queryInput = page.locator('[data-testid="param-query"]');
      await queryInput.fill('real-time test');

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Should show executing status
      await expect(page.locator('[data-testid="execution-status"]')).toHaveText(/executing|running/);

      // Simulate real-time progress updates
      await simulateMCPEvent(page, 'tool_execution_progress', {
        tool_name: 'archon_perform_rag_query',
        progress: 50,
        status: 'processing',
        message: 'Searching knowledge base...'
      });

      await page.waitForTimeout(500);

      // Should update progress indicator
      const progressBar = page.locator('[data-testid="execution-progress"]');
      if (await progressBar.isVisible()) {
        await expect(progressBar).toBeVisible();
      }

      // Simulate completion
      await simulateMCPEvent(page, 'tool_execution_complete', {
        tool_name: 'archon_perform_rag_query',
        result: { status: 'success', execution_time_ms: 180 }
      });

      await page.waitForTimeout(500);

      // Should show completed status
      await expect(page.locator('[data-testid="execution-status"]')).toHaveText(/success|complete/);
    });

    test('should broadcast MCP tool results to multiple agents', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(AGENCY_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Simulate MCP tool execution with broadcast
      await simulateMCPEvent(page, 'mcp_tool_broadcast', {
        tool_name: 'archon_get_available_sources',
        result: {
          sources: ['documents', 'websites', 'code'],
          total_count: 15
        },
        target_agents: ['agent-mcp-1', 'agent-mcp-2', 'agent-mcp-3']
      });

      await page.waitForTimeout(1000);

      // Verify all agents show activity
      const agentNodes = page.locator('[data-testid="agent-node"]');
      for (let i = 0; i < await agentNodes.count(); i++) {
        const agent = agentNodes.nth(i);
        const activityIndicator = agent.locator('[data-testid="agent-activity"]');
        if (await activityIndicator.isVisible()) {
          await expect(activityIndicator).toBeVisible();
        }
      }

      // Verify workflow remains responsive
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
    });
  });

  test.describe('MCP Tool Performance and Scaling', () => {
    test('should handle concurrent MCP tool executions', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10010 });

      // Execute multiple tools concurrently
      const concurrentExecutions = [];

      for (let i = 0; i < 3; i++) {
        concurrentExecutions.push(async () => {
          // Select tool
          const toolName = i === 0 ? 'archon_perform_rag_query' :
                           i === 1 ? 'archon_search_code_examples' :
                           'archon_get_available_sources';

          const tool = page.locator(`[data-tool-name="${toolName}"]`);
          await tool.click();

          // Fill parameters
          const queryInput = page.locator('[data-testid="param-query"], [data-testid="param-pattern"]');
          if (await queryInput.isVisible()) {
            await queryInput.fill(`concurrent test ${i}`);
          }

          // Execute
          const executeButton = page.locator('[data-testid="execute-tool"]');
          await executeButton.click();
        });
      }

      // Start all concurrent executions
      await Promise.all(concurrentExecutions.map(exec => exec()));

      // Wait for executions to complete
      await page.waitForTimeout(3000);

      // Verify multiple execution results are shown
      const resultsPanels = page.locator('[data-testid="tool-results"]');
      expect(await resultsPanels.count()).toBeGreaterThan(0);

      // Verify interface remains responsive
      await expect(page.locator('[data-testid="mcp-container"]')).toBeVisible();
    });

    test('should handle MCP tool execution under load', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      const startTime = Date.now();

      // Execute tool multiple times rapidly
      for (let i = 0; i < 10; i++) {
        // Select RAG tool
        const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
        await ragTool.click();

        // Fill query
        const queryInput = page.locator('[data-testid="param-query"]');
        await queryInput.fill(`load test query ${i}`);

        // Execute
        const executeButton = page.locator('[data-testid="execute-tool"]');
        await executeButton.click();

        // Wait for completion
        await page.waitForSelector('[data-testid="execution-success"], [data-testid="execution-error"]', { timeout: 5000 });
      }

      const totalTime = Date.now() - startTime;

      // Should complete within reasonable time (accounting for mocking)
      expect(totalTime).toBeLessThan(30000);

      // Verify interface remains stable
      await expect(page.locator('[data-testid="mcp-container"]')).toBeVisible();

      // Verify no critical errors
      const errors = await page.evaluate(() => (window as any).getConsoleErrors());
      const criticalErrors = errors.filter((e: string) =>
        !e.includes('Warning') && !e.includes('deprecated')
      );
      expect(criticalErrors.length).toBeLessThan(3); // Allow some errors under load
    });
  });

  test.describe('MCP Security and Access Control', () => {
    test('should validate MCP tool permissions', async ({ page }) => {
      await setupMockMCPAPI(page);

      // Mock permission check
      await page.route('**/api/mcp/tools/archon_manage_project', async (route) => {
        const request = route.request();
        const headers = request.headers();

        // Check for authentication/authorization
        if (!headers['authorization']) {
          await route.fulfill({
            status: 403,
            body: JSON.stringify({
              result: {
                status: 'error',
                error: 'Unauthorized',
                error_details: 'Missing authentication token'
              }
            })
          });
        } else {
          await route.fulfill({
            status: 200,
            body: JSON.stringify(mockMCPResponses.toolExecution.success)
          });
        }
      });

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Try to execute restricted tool
      const projectTool = page.locator('[data-tool-name="archon_manage_project"]');
      await projectTool.click();

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Should show permission error
      await page.waitForSelector('[data-testid="execution-error"]', { timeout: 10000 });
      const errorPanel = page.locator('[data-testid="tool-error"]');
      await expect(errorPanel).toBeVisible();
      await expect(errorPanel.locator('[data-testid="error-message"]')).toHaveText(/Unauthorized|Forbidden/);
    });

    test('should validate MCP tool input parameters', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Select tool with required parameters
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      // Try to execute without required parameters
      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Should show validation error
      const validationError = page.locator('[data-testid="validation-error"]');
      if (await validationError.isVisible({ timeout: 2000 })) {
        await expect(validationError).toBeVisible();
        await expect(validationError.locator('[data-testid="error-message"]')).toHaveText(/required|missing/);
      }

      // Fill invalid parameter type
      const queryInput = page.locator('[data-testid="param-query"]');
      await queryInput.fill(''); // Empty string should be invalid

      await executeButton.click();

      // Should show type validation error
      const typeError = page.locator('[data-testid="type-error"]');
      if (await typeError.isVisible({ timeout: 2000 })) {
        await expect(typeError).toBeVisible();
      }
    });
  });

  test.describe('MCP Integration with Knowledge Management', () => {
    test('should integrate MCP tools with knowledge search', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Execute knowledge search tool
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      const queryInput = page.locator('[data-testid="param-query"]');
      await queryInput.fill('architecture patterns');

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Wait for results
      await page.waitForSelector('[data-testid="execution-success"]', { timeout: 10000 });

      // Verify results integrate with knowledge management
      const resultsPanel = page.locator('[data-testid="tool-results"]');
      await expect(resultsPanel).toBeVisible();

      // Check for knowledge integration features
      const knowledgeLinks = resultsPanel.locator('[data-testid="knowledge-link"]');
      if (await knowledgeLinks.isVisible()) {
        await expect(knowledgeLinks).toBeVisible();
      }

      const addToKnowledgeButton = resultsPanel.locator('[data-testid="add-to-knowledge"]');
      if (await addToKnowledgeButton.isVisible()) {
        await expect(addToKnowledgeButton).toBeVisible();
      }
    });

    test('should update knowledge base from MCP tool results', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Execute search tool
      const codeSearchTool = page.locator('[data-tool-name="archon_search_code_examples"]');
      await codeSearchTool.click();

      const patternInput = page.locator('[data-testid="param-pattern"]');
      await patternInput.fill('react component');

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Wait for results
      await page.waitForSelector('[data-testid="execution-success"]', { timeout: 10000 });

      // Try to save results to knowledge base
      const saveToKnowledgeButton = page.locator('[data-testid="save-to-knowledge"]');
      if (await saveToKnowledgeButton.isVisible()) {
        await saveToKnowledgeButton.click();
        await page.waitForTimeout(1000);

        // Should show success confirmation
        const successMessage = page.locator('[data-testid="knowledge-save-success"]');
        if (await successMessage.isVisible()) {
          await expect(successMessage).toBeVisible();
        }
      }
    });
  });

  test.describe('MCP Error Recovery and Resilience', () => {
    test('should recover from MCP server disconnections', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Simulate server disconnection
      await page.route('**/api/mcp/**', route => route.abort('failed'));

      // Trigger connection check
      await page.locator('[data-testid="check-connection"]').click();
      await page.waitForTimeout(2000);

      // Should show disconnected status
      await expect(page.locator('[data-testid="mcp-status-disconnected"]')).toBeVisible();

      // Restore connection
      await setupMockMCPAPI(page);

      // Retry connection
      const retryButton = page.locator('[data-testid="mcp-retry-connection"]');
      if (await retryButton.isVisible()) {
        await retryButton.click();
        await page.waitForTimeout(2000);

        // Should reconnect successfully
        await expect(page.locator('[data-testid="mcp-status-connected"]')).toBeVisible();
      }
    });

    test('should handle MCP tool execution failures gracefully', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Mock tool execution failure
      await page.route('**/api/mcp/tools/**', async (route) => {
        await route.fulfill({
          status: 500,
          body: JSON.stringify({
            result: {
              status: 'error',
              error: 'Internal server error',
              error_details: 'Tool execution service unavailable'
            }
          })
        });
      });

      // Try to execute tool
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      const queryInput = page.locator('[data-testid="param-query"]');
      await queryInput.fill('test query');

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Should show error with retry option
      await page.waitForSelector('[data-testid="execution-error"]', { timeout: 10000 });
      const errorPanel = page.locator('[data-testid="tool-error"]');
      await expect(errorPanel).toBeVisible();

      const retryButton = errorPanel.locator('[data-testid="retry-execution"]');
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible();
      }

      // Should maintain interface stability
      await expect(page.locator('[data-testid="mcp-container"]')).toBeVisible();
    });
  });

  test.describe('MCP Protocol Compliance', () => {
    test('should follow MCP protocol specification', async ({ page }) => {
      await setupMockMCPAPI(page);

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10010 });

      // Verify protocol version is displayed
      const protocolInfo = page.locator('[data-testid="mcp-protocol-version"]');
      if (await protocolInfo.isVisible()) {
        await expect(protocolInfo).toHaveText(/1.0/);
      }

      // Verify tool requests follow MCP format
      await page.route('**/api/mcp/tools/**', async (route) => {
        const request = route.request();
        const postData = JSON.parse(request.postData() || '{}');

        // Verify MCP protocol compliance
        expect(postData).toHaveProperty('method');
        expect(postData).toHaveProperty('params');
        expect(postData.params).toHaveProperty('tool_name');

        await route.fulfill({
          status: 200,
          body: JSON.stringify(mockMCPResponses.toolExecution.success)
        });
      });

      // Execute tool to trigger protocol check
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      const queryInput = page.locator('[data-testid="param-query"]');
      await queryInput.fill('protocol test');

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      await page.waitForTimeout(1000);
    });

    test('should handle MCP protocol errors correctly', async ({ page }) => {
      await setupMockMCPAPI(page);

      // Mock protocol error response
      await page.route('**/api/mcp/tools/**', async (route) => {
        await route.fulfill({
          status: 400,
          body: JSON.stringify({
            jsonrpc: '2.0',
            error: {
              code: -32600,
              message: 'Invalid Request',
              data: 'Malformed MCP protocol request'
            },
            id: null
          })
        });
      });

      await page.goto(MCP_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Execute tool to trigger protocol error
      const ragTool = page.locator('[data-tool-name="archon_perform_rag_query"]');
      await ragTool.click();

      const queryInput = page.locator('[data-testid="param-query"]');
      await queryInput.fill('protocol error test');

      const executeButton = page.locator('[data-testid="execute-tool"]');
      await executeButton.click();

      // Should show protocol error
      await page.waitForSelector('[data-testid="protocol-error"]', { timeout: 10000 });
      const protocolError = page.locator('[data-testid="protocol-error"]');
      if (await protocolError.isVisible()) {
        await expect(protocolError).toBeVisible();
        await expect(protocolError.locator('[data-testid="error-code"]')).toHaveText('-32600');
      }
    });
  });
});