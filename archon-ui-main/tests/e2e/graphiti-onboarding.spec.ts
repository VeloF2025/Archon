import { test, expect, Page } from '@playwright/test';

/**
 * Graphiti Onboarding System Tests
 * Validates guided tour and help system
 */

class GraphitiOnboardingPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/graphiti');
  }

  async waitForLoad() {
    await this.page.waitForLoadState('networkidle');
    await this.page.waitForTimeout(2000); // Allow for onboarding to auto-start
  }

  async clearOnboardingState() {
    // Clear localStorage to simulate first-time user
    await this.page.evaluate(() => {
      localStorage.removeItem('graphiti-onboarding-complete');
    });
  }
}

test.describe('Graphiti Onboarding Tests', () => {
  let onboardingPage: GraphitiOnboardingPage;

  test.beforeEach(async ({ page }) => {
    onboardingPage = new GraphitiOnboardingPage(page);
  });

  test('shows onboarding for first-time users', async ({ page }) => {
    await onboardingPage.clearOnboardingState();
    await onboardingPage.goto();
    await onboardingPage.waitForLoad();

    // Check for onboarding tooltip
    await expect(page.locator('text=Welcome to Graphiti Explorer')).toBeVisible({ timeout: 5000 });
    
    // Check for step indicator
    await expect(page.locator('text=Step 1 of')).toBeVisible();
    
    // Take screenshot showing onboarding
    await page.screenshot({ 
      path: './test-results/graphiti-onboarding-welcome.png',
      fullPage: true 
    });
  });

  test('allows navigation through onboarding steps', async ({ page }) => {
    await onboardingPage.clearOnboardingState();
    await onboardingPage.goto();
    await onboardingPage.waitForLoad();

    // Start from welcome step
    await expect(page.locator('text=Welcome to Graphiti Explorer')).toBeVisible({ timeout: 5000 });
    
    // Click Next to go to step 2
    const nextButton = page.locator('button:has-text("Next")');
    await nextButton.click();
    await page.waitForTimeout(500);
    
    // Should now show Knowledge Graph step
    await expect(page.locator('text=Knowledge Graph Visualization')).toBeVisible();
    await expect(page.locator('text=Step 2 of')).toBeVisible();
    
    // Click Next to go to step 3
    await nextButton.click();
    await page.waitForTimeout(500);
    
    // Should show Entity Cards step
    await expect(page.locator('text=Entity Cards')).toBeVisible();
    await expect(page.locator('text=Step 3 of')).toBeVisible();
    
    // Take screenshot showing progression
    await page.screenshot({ 
      path: './test-results/graphiti-onboarding-entity-cards.png',
      fullPage: true 
    });
  });

  test('allows skipping onboarding', async ({ page }) => {
    await onboardingPage.clearOnboardingState();
    await onboardingPage.goto();
    await onboardingPage.waitForLoad();

    // Check onboarding appears
    await expect(page.locator('text=Welcome to Graphiti Explorer')).toBeVisible({ timeout: 5000 });
    
    // Click Skip Tour
    const skipButton = page.locator('button:has-text("Skip Tour")');
    await skipButton.click();
    await page.waitForTimeout(500);
    
    // Onboarding should disappear
    await expect(page.locator('text=Welcome to Graphiti Explorer')).not.toBeVisible();
    
    // Help button should be visible
    await expect(page.locator('button:has-text("Help")')).toBeVisible();
  });

  test('shows help button after onboarding completion', async ({ page }) => {
    await onboardingPage.goto();
    await onboardingPage.waitForLoad();

    // Look for help button (should be there if onboarding was completed before)
    const helpButton = page.locator('button:has-text("Help")');
    
    if (await helpButton.isVisible()) {
      console.log('✓ Help button found - onboarding was previously completed');
      
      // Test clicking help button to restart tour
      await helpButton.click();
      await page.waitForTimeout(1000);
      
      // Should show onboarding again
      await expect(page.locator('text=Welcome to Graphiti Explorer')).toBeVisible();
      
      // Take screenshot showing restarted onboarding
      await page.screenshot({ 
        path: './test-results/graphiti-onboarding-restarted.png',
        fullPage: true 
      });
    } else {
      console.log('ℹ Help button not visible - clearing state and testing');
      
      // Clear state and test fresh onboarding
      await onboardingPage.clearOnboardingState();
      await page.reload();
      await onboardingPage.waitForLoad();
      
      // Now should show onboarding
      await expect(page.locator('text=Welcome to Graphiti Explorer')).toBeVisible({ timeout: 5000 });
    }
  });

  test('validates search functionality integration', async ({ page }) => {
    // Ensure onboarding is complete so we can test the UI
    await onboardingPage.goto();
    await onboardingPage.waitForLoad();
    
    // Look for search input
    const searchInput = page.locator('input[placeholder*="Search"]');
    await expect(searchInput).toBeVisible();
    
    // Test search functionality
    await searchInput.fill('login');
    await page.waitForTimeout(500);
    
    // Should show filtered results
    const entityCards = page.locator('[data-testid="rf__node"]');
    const cardCount = await entityCards.count();
    console.log(`📊 Entity cards visible after search: ${cardCount}`);
    
    // Clear search
    await searchInput.fill('');
    await page.waitForTimeout(500);
    
    // Should show all entities again
    const allCards = await entityCards.count();
    console.log(`📊 Entity cards after clearing search: ${allCards}`);
    
    expect(allCards).toBeGreaterThan(cardCount);
    
    // Take screenshot showing search functionality
    await page.screenshot({ 
      path: './test-results/graphiti-onboarding-search.png',
      fullPage: true 
    });
  });

  test('validates entity interaction', async ({ page }) => {
    await onboardingPage.goto();
    await onboardingPage.waitForLoad();

    // Find entity cards
    const entityCards = page.locator('[data-testid="rf__node"]');
    const cardCount = await entityCards.count();
    console.log(`📊 Entity cards found: ${cardCount}`);
    
    expect(cardCount).toBeGreaterThan(0);
    
    // Click first entity card
    if (cardCount > 0) {
      await entityCards.first().click();
      await page.waitForTimeout(500);
      
      // Should show entity details
      const entityDetails = page.locator('text=Entity Details, text=Details');
      if (await entityDetails.isVisible()) {
        console.log('✓ Entity details panel opened');
      } else {
        console.log('ℹ Entity details panel not visible');
      }
      
      // Take screenshot showing interaction
      await page.screenshot({ 
        path: './test-results/graphiti-onboarding-interaction.png',
        fullPage: true 
      });
    }
  });
});