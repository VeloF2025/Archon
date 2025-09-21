/**
 * Performance Test Runner Script
 *
 * Main entry point for running performance tests:
 * - Command-line interface for test execution
 * - Supports different test configurations
 * - Integrates with CI/CD pipelines
 */

import { PerformanceTestRunner, createQuickTestRunner, createFullTestRunner, createProductionTestRunner } from './PerformanceTestRunner';
import { performanceConfigs, performancePresets } from '@/performance';

// CLI Configuration
interface CLIConfig {
  mode: 'quick' | 'full' | 'production' | 'custom';
  environment: 'development' | 'staging' | 'production';
  outputFormat: 'console' | 'json' | 'html';
  saveResults: boolean;
  categories: {
    performance: boolean;
    regression: boolean;
    integration: boolean;
  };
  maxDuration: number;
}

// Parse command line arguments
function parseArgs(): CLIConfig {
  const args = process.argv.slice(2);
  const config: CLIConfig = {
    mode: 'quick',
    environment: 'development',
    outputFormat: 'console',
    saveResults: true,
    categories: {
      performance: true,
      regression: true,
      integration: true,
    },
    maxDuration: 300000,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case '--mode':
      case '-m':
        config.mode = args[i + 1] as CLIConfig['mode'];
        i++;
        break;
      case '--environment':
      case '-e':
        config.environment = args[i + 1] as CLIConfig['environment'];
        i++;
        break;
      case '--output':
      case '-o':
        config.outputFormat = args[i + 1] as CLIConfig['outputFormat'];
        i++;
        break;
      case '--no-save':
        config.saveResults = false;
        break;
      case '--performance-only':
        config.categories.performance = true;
        config.categories.regression = false;
        config.categories.integration = false;
        break;
      case '--regression-only':
        config.categories.performance = false;
        config.categories.regression = true;
        config.categories.integration = false;
        break;
      case '--integration-only':
        config.categories.performance = false;
        config.categories.regression = false;
        config.categories.integration = true;
        break;
      case '--duration':
        config.maxDuration = parseInt(args[i + 1]);
        i++;
        break;
      case '--help':
      case '-h':
        showHelp();
        process.exit(0);
        break;
    }
  }

  return config;
}

// Show help
function showHelp(): void {
  console.log(`
Performance Test Runner

Usage: npm run test:performance [options]

Options:
  --mode, -m <mode>          Test mode: quick, full, production, custom (default: quick)
  --environment, -e <env>     Environment: development, staging, production (default: development)
  --output, -o <format>      Output format: console, json, html (default: console)
  --no-save                  Don't save results to file
  --performance-only         Run only performance tests
  --regression-only          Run only regression tests
  --integration-only         Run only integration tests
  --duration <ms>            Maximum test duration in milliseconds (default: 300000)
  --help, -h                 Show this help message

Examples:
  npm run test:performance --mode quick
  npm run test:performance --mode full --output html
  npm run test:performance --mode production --environment production
  npm run test:performance --performance-only --duration 60000
  `);
}

// Create test runner based on configuration
function createTestRunner(config: CLIConfig): PerformanceTestRunner {
  switch (config.mode) {
    case 'quick':
      return createQuickTestRunner();
    case 'full':
      return createFullTestRunner();
    case 'production':
      return createProductionTestRunner();
    case 'custom':
      return new PerformanceTestRunner({
        testCategories: config.categories,
        environment: config.environment,
        performanceConfig: performanceConfigs[config.environment],
        baselineConfig: performanceConfigs[config.environment],
        outputFormat: config.outputFormat,
        saveResults: config.saveResults,
        maxDuration: config.maxDuration,
      });
    default:
      throw new Error(`Unknown mode: ${config.mode}`);
  }
}

// Main execution function
async function main(): Promise<void> {
  try {
    console.log('🚀 Starting Performance Test Runner...');

    // Parse configuration
    const config = parseArgs();
    console.log(`📋 Configuration: ${JSON.stringify(config, null, 2)}`);

    // Create test runner
    const runner = createTestRunner(config);
    console.log('✅ Test runner created');

    // Run comprehensive tests
    console.log('🏃 Running comprehensive tests...');
    const results = await runner.runComprehensiveTests();

    // Display summary
    console.log('\n🎉 Test execution completed!');
    console.log(`📊 Overall Score: ${results.summary.overallScore}/100`);
    console.log(`✅ Passed: ${results.summary.passedCount}/${results.summary.testCount} tests`);
    console.log(`⏱️  Duration: ${results.summary.duration.toFixed(2)}ms`);

    // Exit with appropriate code
    if (results.summary.failedCount > 0) {
      console.log(`\n❌ ${results.summary.failedCount} tests failed`);
      process.exit(1);
    } else {
      console.log('\n✅ All tests passed!');
      process.exit(0);
    }

  } catch (error) {
    console.error('❌ Performance test execution failed:', error);
    process.exit(1);
  }
}

// Export for programmatic usage
export { PerformanceTestRunner, createQuickTestRunner, createFullTestRunner, createProductionTestRunner };

// Run if called directly
if (require.main === module) {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}