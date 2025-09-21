import { test, expect } from '@playwright/test';

test.describe('CardDescription Export Debug', () => {
  test('should check console errors and card component exports', async ({ page }) => {
    const consoleErrors: string[] = [];
    
    // Capture console errors
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
        console.log('Console Error:', msg.text());
      }
    });

    // Capture page errors
    page.on('pageerror', error => {
      console.log('Page Error:', error.message);
      consoleErrors.push(error.message);
    });

    // Navigate to the page
    await page.goto('http://localhost:3738');
    
    // Wait for page to load
    await page.waitForTimeout(5000);
    
    // Check if there are console errors related to CardDescription
    const cardErrors = consoleErrors.filter(error => 
      error.includes('CardDescription') || error.includes('card.tsx')
    );
    
    console.log('All console errors:', consoleErrors);
    console.log('Card-related errors:', cardErrors);
    
    // Try to evaluate the card module directly
    const moduleCheck = await page.evaluate(async () => {
      try {
        const module = await import('/src/components/ui/card.tsx');
        return {
          success: true,
          exports: Object.keys(module),
          hasCardDescription: 'CardDescription' in module
        };
      } catch (error) {
        return {
          success: false,
          error: error.message
        };
      }
    });
    
    console.log('Module check result:', moduleCheck);
    
    // Check if the page loaded successfully
    const title = await page.title();
    console.log('Page title:', title);
    
    // Take a screenshot for debugging
    await page.screenshot({ path: 'debug-card-export.png', fullPage: true });
    
    // Report findings
    expect(cardErrors.length).toBe(0);
  });
  
  test('should verify card component file content', async ({ page }) => {
    // Check the actual file content via browser fetch
    const fileContent = await page.evaluate(async () => {
      try {
        const response = await fetch('/src/components/ui/card.tsx');
        const text = await response.text();
        return {
          success: true,
          content: text,
          hasCardDescriptionExport: text.includes('export const CardDescription')
        };
      } catch (error) {
        return {
          success: false,
          error: error.message
        };
      }
    });
    
    console.log('File content check:', fileContent);
    
    if (fileContent.success) {
      console.log('File has CardDescription export:', fileContent.hasCardDescriptionExport);
      if (fileContent.hasCardDescriptionExport) {
        console.log('✅ CardDescription export found in file');
      } else {
        console.log('❌ CardDescription export NOT found in file');
        console.log('File content preview:', fileContent.content.substring(0, 500));
      }
    }
  });
});