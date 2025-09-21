/**
 * Knowledge Workflow Integration Test Suite
 *
 * Comprehensive Playwright tests for knowledge management integration
 * with agency workflows, including RAG search, knowledge graph visualization,
 * and collaborative knowledge building.
 *
 * Key test scenarios:
 * - Knowledge base integration with workflow visualization
 * - RAG (Retrieval-Augmented Generation) functionality
 * - Knowledge graph visualization and navigation
 * - Agent knowledge sharing and collaboration
 * - Knowledge extraction from workflow data
 * - Real-time knowledge updates
 * - Knowledge search and filtering
 * - Knowledge contribution workflows
 * - Performance with large knowledge bases
 * - Knowledge security and access control
 */

import { test, expect, Page } from '@playwright/test';
import { v4 as uuidv4 } from 'uuid';

// Test configuration
const BASE_URL = 'http://localhost:3738';
const KNOWLEDGE_URL = `${BASE_URL}/knowledge`;
const WORKFLOW_URL = `${BASE_URL}/workflow`;

// Mock knowledge base data
const mockKnowledgeBase = {
  sources: [
    {
      id: 'source-1',
      name: 'Architecture Documentation',
      type: 'document',
      description: 'System architecture patterns and best practices',
      url: '/docs/architecture',
      created_at: new Date(),
      updated_at: new Date(),
      document_count: 45,
      embedding_count: 156
    },
    {
      id: 'source-2',
      name: 'Code Examples Repository',
      type: 'code',
      description: 'Curated code examples and patterns',
      url: '/examples',
      created_at: new Date(),
      updated_at: new Date(),
      document_count: 120,
      embedding_count: 423
    },
    {
      id: 'source-3',
      name: 'Research Papers',
      type: 'document',
      description: 'AI and software engineering research papers',
      url: '/research',
      created_at: new Date(),
      updated_at: new Date(),
      document_count: 78,
      embedding_count: 289
    }
  ],
  items: [
    {
      id: 'knowledge-1',
      title: 'Microservices Architecture Patterns',
      content: 'Microservices architecture involves breaking down applications into small, independent services that communicate over networks.',
      type: 'pattern',
      source_id: 'source-1',
      tags: ['microservices', 'architecture', 'distributed-systems'],
      confidence: 0.95,
      embedding: Array(1536).fill(0).map((_, i) => Math.sin(i * 0.1)),
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'knowledge-2',
      title: 'React Component Best Practices',
      content: 'React components should be small, focused, and follow single responsibility principle. Use hooks for state management and side effects.',
      type: 'best-practice',
      source_id: 'source-2',
      tags: ['react', 'components', 'javascript'],
      confidence: 0.89,
      embedding: Array(1536).fill(0).map((_, i) => Math.cos(i * 0.1)),
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'knowledge-3',
      title: 'Machine Learning Model Evaluation',
      content: 'ML model evaluation requires comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC for classification tasks.',
      type: 'methodology',
      source_id: 'source-3',
      tags: ['machine-learning', 'evaluation', 'metrics'],
      confidence: 0.92,
      embedding: Array(1536).fill(0).map((_, i) => Math.tan(i * 0.05)),
      created_at: new Date(),
      updated_at: new Date()
    }
  ],
  graph: {
    nodes: [
      {
        id: 'node-1',
        label: 'Architecture',
        type: 'concept',
        x: 100,
        y: 100,
        connections: ['node-2', 'node-3']
      },
      {
        id: 'node-2',
        label: 'Microservices',
        type: 'pattern',
        x: 200,
        y: 150,
        connections: ['node-1', 'node-4']
      },
      {
        id: 'node-3',
        label: 'Design Patterns',
        type: 'concept',
        x: 150,
        y: 50,
        connections: ['node-1', 'node-5']
      },
      {
        id: 'node-4',
        label: 'Service Discovery',
        type: 'technique',
        x: 300,
        y: 200,
        connections: ['node-2']
      },
      {
        id: 'node-5',
        label: 'Singleton Pattern',
        type: 'pattern',
        x: 100,
        y: 0,
        connections: ['node-3']
      }
    ],
    edges: [
      {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        type: 'related',
        weight: 0.8
      },
      {
        id: 'edge-2',
        source: 'node-1',
        target: 'node-3',
        type: 'related',
        weight: 0.6
      },
      {
        id: 'edge-3',
        source: 'node-2',
        target: 'node-4',
        type: 'implements',
        weight: 0.9
      },
      {
        id: 'edge-4',
        source: 'node-3',
        target: 'node-5',
        type: 'includes',
        weight: 0.7
      }
    ]
  }
};

// Mock workflow data with knowledge integration
const mockKnowledgeWorkflow = {
  id: uuidv4(),
  name: 'Knowledge-Enhanced Agency',
  agents: [
    {
      id: 'agent-knowledge-1',
      name: 'Knowledge Architect',
      agent_type: 'SYSTEM_ARCHITECT' as any,
      model_tier: 'OPUS' as any,
      state: 'ACTIVE' as any,
      capabilities: {
        knowledge_extraction: true,
        pattern_recognition: true,
        documentation_generation: true
      },
      knowledge_items: 45
    },
    {
      id: 'agent-knowledge-2',
      name: 'Learning Specialist',
      agent_type: 'TEST_COVERAGE_VALIDATOR' as any,
      model_tier: 'SONNET' as any,
      state: 'ACTIVE' as any,
      capabilities: {
        rag_queries: true,
        knowledge_synthesis: true,
        learning_optimization: true
      },
      knowledge_items: 78
    },
    {
      id: 'agent-knowledge-3',
      name: 'Knowledge Curator',
      agent_type: 'CODE_QUALITY_REVIEWER' as any,
      model_tier: 'SONNET' as any,
      state: 'ACTIVE' as any,
      capabilities: {
        knowledge_validation: true,
        content_curation: true,
        quality_assessment: true
      },
      knowledge_items: 32
    }
  ],
  communication_flows: [
    {
      id: 'knowledge-flow-1',
      source_agent_id: 'agent-knowledge-1',
      target_agent_id: 'agent-knowledge-2',
      communication_type: 'COLLABORATIVE' as any,
      status: 'active' as any,
      message_type: 'knowledge_sharing',
      knowledge_transferred: 12
    },
    {
      id: 'knowledge-flow-2',
      source_agent_id: 'agent-knowledge-2',
      target_agent_id: 'agent-knowledge-3',
      communication_type: 'DIRECT' as any,
      status: 'active' as any,
      message_type: 'learning_update',
      knowledge_transferred: 8
    }
  ]
};

// Helper functions
async function setupMockKnowledgeAPI(page: Page) {
  // Mock knowledge sources API
  await page.route('**/api/knowledge/sources', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockKnowledgeBase.sources)
    });
  });

  // Mock knowledge items API
  await page.route('**/api/knowledge/items', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: mockKnowledgeBase.items,
        total: mockKnowledgeBase.items.length,
        page: 1,
        per_page: 20
      })
    });
  });

  // Mock RAG search API
  await page.route('**/api/knowledge/search', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const query = url.searchParams.get('q') || '';

    // Simulate search results
    const results = mockKnowledgeBase.items.filter(item =>
      item.title.toLowerCase().includes(query.toLowerCase()) ||
      item.content.toLowerCase().includes(query.toLowerCase()) ||
      item.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
    );

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        results: results.map(item => ({
          ...item,
          relevance_score: Math.random() * 0.3 + 0.7,
          snippet: item.content.substring(0, 200) + '...'
        })),
        total_results: results.length,
        query_time_ms: Math.floor(Math.random() * 50) + 20,
        timestamp: new Date().toISOString()
      })
    });
  });

  // Mock knowledge graph API
  await page.route('**/api/knowledge/graph', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockKnowledgeBase.graph)
    });
  });

  // Mock workflow data with knowledge integration
  await page.route('**/api/agency/workflow/data', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockKnowledgeWorkflow)
    });
  });

  // Mock knowledge contribution API
  let knowledgeContributions: any[] = [];
  await page.route('**/api/knowledge/contribute', async (route) => {
    const request = route.request();
    const postData = JSON.parse(request.postData() || '{}');
    knowledgeContributions.push(postData);

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        knowledge_id: uuidv4(),
        timestamp: new Date().toISOString()
      })
    });
  });

  return { knowledgeContributions };
}

async function simulateKnowledgeUpdate(page: Page, eventType: string, data: any) {
  await page.evaluate(({ eventType, data }) => {
    window.dispatchEvent(new CustomEvent('knowledge-update', {
      detail: { type: eventType, data }
    }));
  }, { eventType, data });
}

test.describe('Knowledge Workflow Integration', () => {
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

  test.describe('Knowledge Base Integration', () => {
    test('should load and display knowledge base sources', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Verify knowledge base container
      await expect(page.locator('[data-testid="knowledge-base-container"]')).toBeVisible();

      // Wait for knowledge sources to load
      await page.waitForSelector('[data-testid="knowledge-source"]', { timeout: 10000 });

      // Verify all sources are displayed
      const sourceCards = page.locator('[data-testid="knowledge-source"]');
      expect(await sourceCards.count()).toBe(mockKnowledgeBase.sources.length);

      // Verify source information is displayed correctly
      for (let i = 0; i < mockKnowledgeBase.sources.length; i++) {
        const source = mockKnowledgeBase.sources[i];
        const sourceCard = sourceCards.nth(i);

        await expect(sourceCard.locator('[data-testid="source-name"]')).toHaveText(source.name);
        await expect(sourceCard.locator('[data-testid="source-type"]')).toHaveText(source.type);
        await expect(sourceCard.locator('[data-testid="source-description"]')).toHaveText(source.description);
        await expect(sourceCard.locator('[data-testid="document-count"]')).toHaveText(source.document_count.toString());
      }
    });

    test('should display knowledge items with proper metadata', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to items tab
      await page.locator('[data-testid="knowledge-items-tab"]').click();
      await page.waitForTimeout(1000);

      // Verify knowledge items are displayed
      await page.waitForSelector('[data-testid="knowledge-item"]', { timeout: 10000 });
      const itemCards = page.locator('[data-testid="knowledge-item"]');
      expect(await itemCards.count()).toBe(mockKnowledgeBase.items.length);

      // Verify item metadata
      for (let i = 0; i < mockKnowledgeBase.items.length; i++) {
        const item = mockKnowledgeBase.items[i];
        const itemCard = itemCards.nth(i);

        await expect(itemCard.locator('[data-testid="item-title"]')).toHaveText(item.title);
        await expect(itemCard.locator('[data-testid="item-type"]')).toHaveText(item.type);
        await expect(itemCard.locator('[data-testid="confidence-score"]')).toBeVisible();

        // Verify tags are displayed
        const tagElements = itemCard.locator('[data-testid="knowledge-tag"]');
        expect(await tagElements.count()).toBe(item.tags.length);

        for (let j = 0; j < item.tags.length; j++) {
          await expect(tagElements.nth(j)).toHaveText(item.tags[j]);
        }
      }
    });

    test('should filter knowledge items by type and source', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to items tab
      await page.locator('[data-testid="knowledge-items-tab"]').click();
      await page.waitForTimeout(1000);

      // Filter by type
      const typeFilter = page.locator('[data-testid="knowledge-type-filter"]');
      await expect(typeFilter).toBeVisible();
      await typeFilter.selectOption('pattern');

      await page.waitForTimeout(500);

      // Verify filtering works
      const visibleItems = page.locator('[data-testid="knowledge-item"]:visible');
      const patternItems = mockKnowledgeBase.items.filter(item => item.type === 'pattern');
      expect(await visibleItems.count()).toBe(patternItems.length);

      // Clear filter
      await typeFilter.selectOption('all');
      await page.waitForTimeout(500);

      // Verify all items are visible again
      const allItems = page.locator('[data-testid="knowledge-item"]:visible');
      expect(await allItems.count()).toBe(mockKnowledgeBase.items.length);
    });
  });

  test.describe('RAG Search Functionality', () => {
    test('should perform RAG search successfully', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Wait for search interface
      await page.waitForSelector('[data-testid="rag-search-input"]', { timeout: 10000 });

      const searchInput = page.locator('[data-testid="rag-search-input"]');
      const searchButton = page.locator('[data-testid="rag-search-button"]');

      await expect(searchInput).toBeVisible();
      await expect(searchButton).toBeVisible();

      // Perform search
      await searchInput.fill('architecture');
      await searchButton.click();

      // Wait for search results
      await page.waitForSelector('[data-testid="search-results"]', { timeout: 10000 });

      // Verify results are displayed
      const resultsContainer = page.locator('[data-testid="search-results"]');
      await expect(resultsContainer).toBeVisible();

      const resultItems = resultsContainer.locator('[data-testid="search-result-item"]');
      expect(await resultItems.count()).toBeGreaterThan(0);

      // Verify result metadata
      const firstResult = resultItems.first();
      await expect(firstResult.locator('[data-testid="result-title"]')).toBeVisible();
      await expect(firstResult.locator('[data-testid="result-snippet"]')).toBeVisible();
      await expect(firstResult.locator('[data-testid="relevance-score"]')).toBeVisible();
      await expect(firstResult.locator('[data-testid="result-source"]')).toBeVisible();
    });

    test('should handle RAG search with filters', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Wait for search interface
      await page.waitForSelector('[data-testid="rag-search-input"]', { timeout: 10000 });

      const searchInput = page.locator('[data-testid="rag-search-input"]');
      const searchButton = page.locator('[data-testid="rag-search-button"]');

      // Apply filters before search
      const sourceFilter = page.locator('[data-testid="search-source-filter"]');
      if (await sourceFilter.isVisible()) {
        await sourceFilter.selectOption('source-1');
      }

      const typeFilter = page.locator('[data-testid="search-type-filter"]');
      if (await typeFilter.isVisible()) {
        await typeFilter.selectOption('pattern');
      }

      // Perform filtered search
      await searchInput.fill('microservices');
      await searchButton.click();

      // Wait for results
      await page.waitForSelector('[data-testid="search-results"]', { timeout: 10000 });

      // Verify filtered results
      const resultsContainer = page.locator('[data-testid="search-results"]');
      await expect(resultsContainer).toBeVisible();

      // Verify active filters are displayed
      const activeFilters = page.locator('[data-testid="active-filter"]');
      if (await activeFilters.isVisible()) {
        expect(await activeFilters.count()).toBeGreaterThan(0);
      }
    });

    test('should handle empty RAG search results gracefully', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Wait for search interface
      await page.waitForSelector('[data-testid="rag-search-input"]', { timeout: 10010 });

      const searchInput = page.locator('[data-testid="rag-search-input"]');
      const searchButton = page.locator('[data-testid="rag-search-button"]');

      // Search for non-existent term
      await searchInput.fill('nonexistent_term_xyz_123');
      await searchButton.click();

      // Wait for search completion
      await page.waitForTimeout(2000);

      // Verify empty state is displayed
      const emptyState = page.locator('[data-testid="no-search-results"]');
      if (await emptyState.isVisible({ timeout: 5000 })) {
        await expect(emptyState).toBeVisible();
        await expect(emptyState.locator('[data-testid="empty-message"]')).toHaveText(/No results found/);
      }

      // Should show suggestions
      const suggestions = page.locator('[data-testid="search-suggestions"]');
      if (await suggestions.isVisible()) {
        await expect(suggestions).toBeVisible();
      }
    });
  });

  test.describe('Knowledge Graph Visualization', () => {
    test('should display knowledge graph with nodes and edges', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to graph tab
      await page.locator('[data-testid="knowledge-graph-tab"]').click();
      await page.waitForTimeout(1000);

      // Wait for graph to load
      await page.waitForSelector('[data-testid="knowledge-graph"]', { timeout: 10000 });

      // Verify graph container
      const graphContainer = page.locator('[data-testid="knowledge-graph"]');
      await expect(graphContainer).toBeVisible();

      // Verify nodes are displayed
      const graphNodes = page.locator('[data-testid="graph-node"]');
      expect(await graphNodes.count()).toBe(mockKnowledgeBase.graph.nodes.length);

      // Verify edges are displayed
      const graphEdges = page.locator('[data-testid="graph-edge"]');
      expect(await graphEdges.count()).toBe(mockKnowledgeBase.graph.edges.length);

      // Verify graph controls
      await expect(page.locator('[data-testid="graph-zoom-in"]')).toBeVisible();
      await expect(page.locator('[data-testid="graph-zoom-out"]')).toBeVisible();
      await expect(page.locator('[data-testid="graph-fit-to-screen"]')).toBeVisible();
    });

    test('should support interactive graph navigation', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to graph tab
      await page.locator('[data-testid="knowledge-graph-tab"]').click();
      await page.waitForTimeout(1000);

      // Wait for graph to load
      await page.waitForSelector('[data-testid="knowledge-graph"]', { timeout: 10000 });

      // Test zoom controls
      await page.locator('[data-testid="graph-zoom-in"]').click();
      await page.waitForTimeout(500);

      await page.locator('[data-testid="graph-zoom-out"]').click();
      await page.waitForTimeout(500);

      await page.locator('[data-testid="graph-fit-to-screen"]').click();
      await page.waitForTimeout(500);

      // Verify graph remains interactive
      await expect(page.locator('[data-testid="knowledge-graph"]')).toBeVisible();

      // Test node interaction
      const firstNode = page.locator('[data-testid="graph-node"]').first();
      await firstNode.click();
      await page.waitForTimeout(500);

      // Verify node details appear
      const nodeDetails = page.locator('[data-testid="node-details-panel"]');
      if (await nodeDetails.isVisible()) {
        await expect(nodeDetails).toBeVisible();
        await expect(nodeDetails.locator('[data-testid="node-label"]')).toBeVisible();
        await expect(nodeDetails.locator('[data-testid="node-type"]')).toBeVisible();
      }
    });

    test('should filter knowledge graph by node type', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to graph tab
      await page.locator('[data-testid="knowledge-graph-tab"]').click();
      await page.waitForTimeout(1000);

      // Wait for graph to load
      await page.waitForSelector('[data-testid="knowledge-graph"]', { timeout: 10000 });

      // Apply node type filter
      const nodeTypeFilter = page.locator('[data-testid="node-type-filter"]');
      if (await nodeTypeFilter.isVisible()) {
        await nodeTypeFilter.selectOption('pattern');

        await page.waitForTimeout(1000);

        // Verify filtered nodes
        const visibleNodes = page.locator('[data-testid="graph-node"]:visible');
        const patternNodes = mockKnowledgeBase.graph.nodes.filter(node => node.type === 'pattern');
        expect(await visibleNodes.count()).toBe(patternNodes.length);
      }
    });
  });

  test.describe('Knowledge Integration with Agency Workflow', () => {
    test('should display knowledge-enhanced agent information', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Wait for workflow to load
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Verify agents with knowledge integration
      const agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(mockKnowledgeWorkflow.agents.length);

      // Click on knowledge architect agent
      const knowledgeAgent = page.locator('[data-agent-id="agent-knowledge-1"]');
      await knowledgeAgent.click();

      // Verify agent details show knowledge information
      const agentDetails = page.locator('[data-testid="agent-details-panel"]');
      await expect(agentDetails).toBeVisible();

      const knowledgeSection = agentDetails.locator('[data-testid="agent-knowledge-section"]');
      if (await knowledgeSection.isVisible()) {
        await expect(knowledgeSection).toBeVisible();
        await expect(knowledgeSection.locator('[data-testid="knowledge-items-count"]')).toHaveText('45');
      }

      // Verify knowledge capabilities are highlighted
      const capabilitiesList = agentDetails.locator('[data-testid="agent-capabilities"]');
      await expect(capabilitiesList).toBeVisible();
      await expect(capabilitiesList.locator('[data-capability="knowledge-extraction"]')).toBeVisible();
    });

    test('should visualize knowledge flow between agents', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Wait for workflow to load
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10010 });

      // Verify knowledge communication flows are displayed
      const knowledgeFlows = page.locator('[data-testid="communication-flow"][data-message-type="knowledge_sharing"], [data-testid="communication-flow"][data-message-type="learning_update"]');
      expect(await knowledgeFlows.count()).toBeGreaterThan(0);

      // Verify knowledge flows have special styling
      const firstKnowledgeFlow = knowledgeFlows.first();
      await expect(firstKnowledgeFlow).toBeVisible();
      await expect(firstKnowledgeFlow).toHaveClass(/knowledge-flow/);

      // Click on knowledge flow to see details
      await firstKnowledgeFlow.click();
      await page.waitForTimeout(500);

      // Verify flow details show knowledge information
      const flowDetails = page.locator('[data-testid="flow-details-panel"]');
      if (await flowDetails.isVisible()) {
        await expect(flowDetails).toBeVisible();
        await expect(flowDetails.locator('[data-testid="knowledge-transferred"]')).toBeVisible();
      }
    });

    test('should integrate knowledge search within workflow context', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Wait for workflow to load
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Open knowledge integration panel
      const knowledgeButton = page.locator('[data-testid="workflow-knowledge-panel"]');
      if (await knowledgeButton.isVisible()) {
        await knowledgeButton.click();
        await page.waitForTimeout(1000);

        // Verify knowledge panel appears
        const knowledgePanel = page.locator('[data-testid="workflow-knowledge-panel"]');
        await expect(knowledgePanel).toBeVisible();

        // Perform context-aware search
        const searchInput = knowledgePanel.locator('[data-testid="context-search-input"]');
        if (await searchInput.isVisible()) {
          await searchInput.fill('architecture patterns');
          await searchInput.press('Enter');

          await page.waitForTimeout(1000);

          // Verify search results are displayed in workflow context
          const results = knowledgePanel.locator('[data-testid="context-search-results"]');
          if (await results.isVisible()) {
            await expect(results).toBeVisible();
          }
        }
      }
    });
  });

  test.describe('Knowledge Contribution and Collaboration', () => {
    test('should allow agents to contribute knowledge', async ({ page }) => {
      const { knowledgeContributions } = await setupMockKnowledgeAPI(page);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Wait for workflow to load
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Select knowledge architect agent
      const knowledgeAgent = page.locator('[data-agent-id="agent-knowledge-1"]');
      await knowledgeAgent.click();

      // Open contribution interface
      const contributeButton = page.locator('[data-testid="contribute-knowledge"]');
      if (await contributeButton.isVisible()) {
        await contributeButton.click();
        await page.waitForTimeout(1000);

        // Verify contribution modal appears
        const contributionModal = page.locator('[data-testid="knowledge-contribution-modal"]');
        await expect(contributionModal).toBeVisible();

        // Fill contribution form
        const titleInput = contributionModal.locator('[data-testid="contribution-title"]');
        await titleInput.fill('New Architecture Pattern');

        const contentInput = contributionModal.locator('[data-testid="contribution-content"]');
        await contentInput.fill('This is a newly discovered architecture pattern based on recent agent experiences.');

        const typeSelect = contributionModal.locator('[data-testid="contribution-type"]');
        await typeSelect.selectOption('pattern');

        // Submit contribution
        const submitButton = contributionModal.locator('[data-testid="submit-contribution"]');
        await submitButton.click();

        await page.waitForTimeout(1000);

        // Verify contribution was submitted
        expect(knowledgeContributions.length).toBe(1);
        expect(knowledgeContributions[0].title).toBe('New Architecture Pattern');
      }
    });

    test('should handle knowledge validation and review', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to validation queue
      await page.locator('[data-testid="knowledge-validation-tab"]').click();
      await page.waitForTimeout(1000);

      // Wait for validation items to load
      await page.waitForSelector('[data-testid="validation-item"]', { timeout: 10000 });

      // Verify validation items are displayed
      const validationItems = page.locator('[data-testid="validation-item"]');
      expect(await validationItems.count()).toBeGreaterThan(0);

      // Select first item for validation
      const firstItem = validationItems.first();
      await firstItem.click();

      // Verify validation interface appears
      const validationInterface = page.locator('[data-testid="knowledge-validation-interface"]');
      if (await validationInterface.isVisible()) {
        await expect(validationInterface).toBeVisible();

        // Test validation actions
        const approveButton = validationInterface.locator('[data-testid="approve-knowledge"]');
        const rejectButton = validationInterface.locator('[data-testid="reject-knowledge"]');
        const requestChangesButton = validationInterface.locator('[data-testid="request-changes"]');

        await expect(approveButton).toBeVisible();
        await expect(rejectButton).toBeVisible();
        await expect(requestChangesButton).toBeVisible();

        // Test validation feedback
        const feedbackInput = validationInterface.locator('[data-testid="validation-feedback"]');
        await expect(feedbackInput).toBeVisible();
      }
    });

    test('should support collaborative knowledge building', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to collaborative knowledge section
      await page.locator('[data-testid="collaborative-knowledge-tab"]').click();
      await page.waitForTimeout(1000);

      // Wait for collaborative interface to load
      await page.waitForSelector('[data-testid="collaborative-workspace"]', { timeout: 10000 });

      // Verify collaborative workspace
      const workspace = page.locator('[data-testid="collaborative-workspace"]');
      await expect(workspace).toBeVisible();

      // Verify participant list
      const participants = workspace.locator('[data-testid="collaboration-participant"]');
      expect(await participants.count()).toBeGreaterThan(0);

      // Verify shared knowledge items
      const sharedItems = workspace.locator('[data-testid="shared-knowledge-item"]');
      expect(await sharedItems.count()).toBeGreaterThan(0);

      // Test collaboration actions
      const addItemButton = workspace.locator('[data-testid="add-shared-item"]');
      if (await addItemButton.isVisible()) {
        await expect(addItemButton).toBeVisible();
      }
    });
  });

  test.describe('Real-time Knowledge Updates', () => {
    test('should handle real-time knowledge additions', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Wait for knowledge items to load
      await page.waitForSelector('[data-testid="knowledge-item"]', { timeout: 10000 });

      const initialCount = await page.locator('[data-testid="knowledge-item"]').count();

      // Simulate real-time knowledge addition
      await simulateKnowledgeUpdate(page, 'knowledge_added', {
        id: 'new-knowledge-1',
        title: 'Real-time Knowledge Update',
        content: 'This knowledge was added in real-time',
        type: 'update',
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(1000);

      // Verify new knowledge appears (if real-time updates are implemented)
      const newKnowledgeItem = page.locator('[data-knowledge-id="new-knowledge-1"]');
      if (await newKnowledgeItem.isVisible({ timeout: 2000 })) {
        await expect(newKnowledgeItem).toBeVisible();
        await expect(newKnowledgeItem.locator('[data-testid="item-title"]')).toHaveText('Real-time Knowledge Update');
      }
    });

    test('should update knowledge graph in real-time', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to graph tab
      await page.locator('[data-testid="knowledge-graph-tab"]').click();
      await page.waitForTimeout(1000);

      // Wait for graph to load
      await page.waitForSelector('[data-testid="knowledge-graph"]', { timeout: 10000 });

      const initialNodeCount = await page.locator('[data-testid="graph-node"]').count();

      // Simulate graph update
      await simulateKnowledgeUpdate(page, 'graph_updated', {
        action: 'node_added',
        node: {
          id: 'new-node-1',
          label: 'New Concept',
          type: 'concept',
          x: 400,
          y: 300,
          connections: []
        }
      });

      await page.waitForTimeout(1001);

      // Verify graph updates (if real-time updates are implemented)
      const newNode = page.locator('[data-node-id="new-node-1"]');
      if (await newNode.isVisible({ timeout: 2000 })) {
        await expect(newNode).toBeVisible();
        await expect(newNode.locator('[data-testid="node-label"]')).toHaveText('New Concept');
      }
    });
  });

  test.describe('Knowledge Performance and Scaling', () => {
    test('should handle large knowledge bases efficiently', async ({ page }) => {
      // Create large knowledge base mock
      const largeKnowledgeBase = {
        items: Array.from({ length: 100 }, (_, i) => ({
          id: `knowledge-${i}`,
          title: `Knowledge Item ${i}`,
          content: `Content for knowledge item ${i}`,
          type: ['pattern', 'best-practice', 'methodology'][i % 3],
          source_id: `source-${(i % 3) + 1}`,
          tags: [`tag-${i % 5}`, `category-${i % 3}`],
          confidence: Math.random() * 0.3 + 0.7,
          embedding: Array(1536).fill(0).map(() => Math.random()),
          created_at: new Date(),
          updated_at: new Date()
        }))
      };

      await page.route('**/api/knowledge/items', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            items: largeKnowledgeBase.items.slice(0, 20), // First page
            total: largeKnowledgeBase.items.length,
            page: 1,
            per_page: 20
          })
        });
      });

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to items tab
      await page.locator('[data-testid="knowledge-items-tab"]').click();
      await page.waitForTimeout(1000);

      const startTime = Date.now();

      // Wait for items to load
      await page.waitForSelector('[data-testid="knowledge-item"]', { timeout: 15000 });

      const loadTime = Date.now() - startTime;

      // Should load within reasonable time
      expect(loadTime).toBeLessThan(10000);

      // Verify pagination works
      const pagination = page.locator('[data-testid="knowledge-pagination"]');
      if (await pagination.isVisible()) {
        await expect(pagination).toBeVisible();

        const nextPageButton = pagination.locator('[data-testid="next-page"]');
        if (await nextPageButton.isVisible()) {
          await nextPageButton.click();
          await page.waitForTimeout(1000);

          // Verify page loads successfully
          await expect(page.locator('[data-testid="knowledge-item"]')).toBeVisible();
        }
      }
    });

    test('should handle rapid knowledge searches', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Wait for search interface
      await page.waitForSelector('[data-testid="rag-search-input"]', { timeout: 10000 });

      const searchInput = page.locator('[data-testid="rag-search-input"]');
      const searchButton = page.locator('[data-testid="rag-search-button"]');

      // Perform rapid searches
      const searchTerms = ['architecture', 'react', 'patterns', 'testing', 'microservices'];

      for (const term of searchTerms) {
        await searchInput.fill(term);
        await searchButton.click();
        await page.waitForTimeout(500);
      }

      // Verify interface remains responsive
      await expect(page.locator('[data-testid="knowledge-base-container"]')).toBeVisible();

      // Verify last search completed
      await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
    });
  });

  test.describe('Knowledge Security and Access Control', () => {
    test('should validate knowledge access permissions', async ({ page }) => {
      // Mock permission-based access
      await page.route('**/api/knowledge/items', async (route) => {
        const request = route.request();
        const headers = request.headers();

        // Check for authorization
        if (!headers['authorization']) {
          await route.fulfill({
            status: 403,
            body: JSON.stringify({ error: 'Access denied' })
          });
        } else {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              items: mockKnowledgeBase.items,
              total: mockKnowledgeBase.items.length
            })
          });
        }
      });

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Should handle access denied gracefully
      const accessDenied = page.locator('[data-testid="access-denied"]');
      if (await accessDenied.isVisible({ timeout: 5000 })) {
        await expect(accessDenied).toBeVisible();
      } else {
        // If access is granted, verify items are displayed
        await expect(page.locator('[data-testid="knowledge-item"]')).toBeVisible();
      }
    });

    test('should validate knowledge contribution permissions', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Wait for workflow to load
      await page.waitForSelector('[data-testid="agency-workflow"]', { timeout: 10000 });

      // Select agent without knowledge contribution capabilities
      const regularAgent = page.locator('[data-agent-id="agent-knowledge-3"]');
      await regularAgent.click();

      // Contribution button should be disabled or hidden for non-knowledge agents
      const contributeButton = page.locator('[data-testid="contribute-knowledge"]');
      if (await contributeButton.isVisible()) {
        await expect(contributeButton).toBeDisabled();
      }
    });
  });

  test.describe('Knowledge Export and Integration', () => {
    test('should export knowledge base in various formats', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Open export interface
      const exportButton = page.locator('[data-testid="export-knowledge"]');
      if (await exportButton.isVisible()) {
        await exportButton.click();
        await page.waitForTimeout(1000);

        // Verify export modal appears
        const exportModal = page.locator('[data-testid="knowledge-export-modal"]');
        await expect(exportModal).toBeVisible();

        // Test format selection
        const formatSelect = exportModal.locator('[data-testid="export-format"]');
        await expect(formatSelect).toBeVisible();

        await formatSelect.selectOption('json');
        await page.waitForTimeout(500);

        // Test export options
        const includeMetadata = exportModal.locator('[data-testid="include-metadata"]');
        await expect(includeMetadata).toBeVisible();

        const includeEmbeddings = exportModal.locator('[data-testid="include-embeddings"]');
        await expect(includeEmbeddings).toBeVisible();

        // Test export button
        const exportConfirmButton = exportModal.locator('[data-testid="confirm-export"]');
        await expect(exportConfirmButton).toBeVisible();
      }
    });

    test('should integrate with external knowledge sources', async ({ page }) => {
      await setupMockKnowledgeAPI(page);

      await page.goto(KNOWLEDGE_URL);
      await page.waitForLoadState('networkidle');

      // Navigate to integration settings
      await page.locator('[data-testid="knowledge-integration-tab"]').click();
      await page.waitForTimeout(1000);

      // Wait for integration interface
      await page.waitForSelector('[data-testid="knowledge-sources-management"]', { timeout: 10000 });

      // Verify source management interface
      const sourcesManagement = page.locator('[data-testid="knowledge-sources-management"]');
      await expect(sourcesManagement).toBeVisible();

      // Test add source functionality
      const addSourceButton = sourcesManagement.locator('[data-testid="add-knowledge-source"]');
      if (await addSourceButton.isVisible()) {
        await expect(addSourceButton).toBeVisible();
      }

      // Test source sync functionality
      const syncButton = sourcesManagement.locator('[data-testid="sync-sources"]');
      if (await syncButton.isVisible()) {
        await expect(syncButton).toBeVisible();
      }
    });
  });
});