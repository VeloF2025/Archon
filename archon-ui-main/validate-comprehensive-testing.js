/**
 * Comprehensive Test Validation Script
 * Runs all tests and generates coverage reports for Agency Swarm Phase 2
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Starting Comprehensive Test Validation for Agency Swarm Phase 2');
console.log('═'.repeat(80));

// Test configuration
const testConfig = {
  e2eTests: [
    'tests/e2e/agency_workflow_visualization.spec.ts',
    'tests/e2e/workflow_editor_integration.spec.ts',
    'tests/e2e/mcp_agency_integration.spec.ts',
    'tests/e2e/knowledge_workflow_integration.spec.ts',
    'tests/e2e/performance-accessibility.spec.ts',
    'tests/e2e/error-scenarios-cross-browser.spec.ts'
  ],
  componentTests: [
    'src/components/workflow/__tests__/CompleteWorkflowIntegration.test.tsx',
    'src/components/workflow/__tests__/WorkflowVisualizationIntegration.test.tsx',
    'src/components/workflow/__tests__/WorkflowEditorIntegration.test.tsx',
    'src/components/workflow/__tests__/KnowledgeIntegration.test.tsx'
  ],
  coverageThreshold: 80,
  performanceThreshold: {
    pageLoad: 3000,
    interactionTime: 500,
    memoryUsage: 50000000
  }
};

// Results tracking
const testResults = {
  e2e: { passed: 0, failed: 0, total: 0 },
  component: { passed: 0, failed: 0, total: 0 },
  performance: {},
  accessibility: {},
  coverage: {},
  errors: []
};

function runCommand(command, description, options = {}) {
  try {
    console.log(`\n📋 ${description}`);
    console.log('Command:', command);

    const result = execSync(command, {
      stdio: 'pipe',
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 10, // 10MB buffer
      ...options
    });

    console.log('✅ Command completed successfully');
    return { success: true, output: result };
  } catch (error) {
    console.log('❌ Command failed:', error.message);
    testResults.errors.push({
      type: 'command',
      command,
      error: error.message,
      timestamp: new Date().toISOString()
    });
    return { success: false, output: error.stdout || '', error: error.message };
  }
}

function analyzeCoverage() {
  console.log('\n📊 Analyzing Test Coverage...');

  const coveragePath = path.join(__dirname, 'public/test-results/coverage');

  if (!fs.existsSync(coveragePath)) {
    console.log('⚠️  Coverage directory not found');
    return;
  }

  try {
    // Read coverage summary
    const summaryPath = path.join(coveragePath, 'coverage-summary.json');
    if (fs.existsSync(summaryPath)) {
      const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
      testResults.coverage = summary;

      console.log('📈 Coverage Summary:');
      Object.entries(summary.total || {}).forEach(([key, value]) => {
        if (typeof value === 'object' && value.pct !== undefined) {
          console.log(`  ${key}: ${value.pct.toFixed(1)}%`);
        }
      });

      // Check coverage thresholds
      const overallCoverage = summary.total?.lines?.pct || 0;
      if (overallCoverage >= testConfig.coverageThreshold) {
        console.log(`✅ Coverage meets threshold (${testConfig.coverageThreshold}%)`);
      } else {
        console.log(`❌ Coverage below threshold (${overallCoverage.toFixed(1)}% < ${testConfig.coverageThreshold}%)`);
      }
    }
  } catch (error) {
    console.log('❌ Error analyzing coverage:', error.message);
  }
}

function analyzePerformance() {
  console.log('\n⚡ Analyzing Performance Metrics...');

  try {
    // Look for performance test results
    const performanceResultsPath = path.join(__dirname, 'public/test-results/performance.json');

    if (fs.existsSync(performanceResultsPath)) {
      const performanceData = JSON.parse(fs.readFileSync(performanceResultsPath, 'utf8'));
      testResults.performance = performanceData;

      console.log('🚀 Performance Results:');
      Object.entries(performanceData).forEach(([metric, value]) => {
        console.log(`  ${metric}: ${value}`);

        // Check against thresholds
        if (metric === 'pageLoad' && value > testConfig.performanceThreshold.pageLoad) {
          console.log(`    ⚠️  Exceeds threshold (${testConfig.performanceThreshold.pageLoad}ms)`);
        }
        if (metric === 'interactionTime' && value > testConfig.performanceThreshold.interactionTime) {
          console.log(`    ⚠️  Exceeds threshold (${testConfig.performanceThreshold.interactionTime}ms)`);
        }
      });
    } else {
      console.log('📝 Performance results file not found (may need to run performance tests first)');
    }
  } catch (error) {
    console.log('❌ Error analyzing performance:', error.message);
  }
}

function generateReport() {
  console.log('\n📋 Generating Comprehensive Test Report...');

  const report = {
    timestamp: new Date().toISOString(),
    project: 'Agency Swarm Phase 2',
    testResults,
    config: testConfig,
    summary: {
      totalE2ETests: testResults.e2e.total,
      passedE2ETests: testResults.e2e.passed,
      failedE2ETests: testResults.e2e.failed,
      totalComponentTests: testResults.component.total,
      passedComponentTests: testResults.component.passed,
      failedComponentTests: testResults.component.failed,
      totalTests: testResults.e2e.total + testResults.component.total,
      totalPassed: testResults.e2e.passed + testResults.component.passed,
      totalFailed: testResults.e2e.failed + testResults.component.failed,
      successRate: ((testResults.e2e.passed + testResults.component.passed) /
                    (testResults.e2e.total + testResults.component.total) * 100).toFixed(1) + '%',
      overallCoverage: testResults.coverage.total?.lines?.pct || 0,
      errorCount: testResults.errors.length
    }
  };

  // Save detailed report
  const reportPath = path.join(__dirname, 'public/test-results/comprehensive-test-report.json');
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  console.log('\n📊 Test Summary Report');
  console.log('═'.repeat(50));
  console.log(`E2E Tests: ${testResults.e2e.passed}/${testResults.e2e.total} passed`);
  console.log(`Component Tests: ${testResults.component.passed}/${testResults.component.total} passed`);
  console.log(`Overall Success Rate: ${report.summary.successRate}`);
  console.log(`Overall Coverage: ${report.summary.overallCoverage.toFixed(1)}%`);
  console.log(`Errors: ${testResults.errors.length}`);

  if (testResults.errors.length > 0) {
    console.log('\n❌ Errors Encountered:');
    testResults.errors.forEach((error, index) => {
      console.log(`  ${index + 1}. ${error.type}: ${error.error}`);
    });
  }

  console.log(`\n📄 Detailed report saved to: ${reportPath}`);

  return report;
}

async function runComponentTests() {
  console.log('\n🧪 Running Component Tests...');

  try {
    // Install test dependencies if needed
    runCommand('npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event',
                'Installing test dependencies');

    // Run component tests with coverage
    const result = runCommand('npm run test:coverage:stream', 'Running component tests with coverage');

    if (result.success) {
      // Parse test results
      const testOutput = result.output;
      const testMatches = testOutput.match(/Tests:\s*(\d+)\s*passed,\s*(\d+)\s*failed/);

      if (testMatches) {
        testResults.component.total = parseInt(testMatches[1]) + parseInt(testMatches[2]);
        testResults.component.passed = parseInt(testMatches[1]);
        testResults.component.failed = parseInt(testMatches[2]);
      }

      console.log(`✅ Component Tests: ${testResults.component.passed}/${testResults.component.total} passed`);
    }

    analyzeCoverage();
  } catch (error) {
    console.log('❌ Error running component tests:', error.message);
  }
}

async function runE2ETests() {
  console.log('\n🌐 Running E2E Tests...');

  try {
    // Install Playwright browsers if needed
    runCommand('npx playwright install', 'Installing Playwright browsers');

    // Run each E2E test file
    for (const testFile of testConfig.e2eTests) {
      console.log(`\n🎯 Running E2E test: ${testFile}`);

      const result = runCommand(`npx playwright test ${testFile} --reporter=json`,
                                `Running ${path.basename(testFile)}`);

      if (result.success) {
        try {
          const testOutput = JSON.parse(result.output);
          const stats = testOutput.stats || { tests: 0, passed: 0, failed: 0 };

          testResults.e2e.total += stats.tests;
          testResults.e2e.passed += stats.passed;
          testResults.e2e.failed += stats.failed;

          console.log(`✅ ${path.basename(testFile)}: ${stats.passed}/${stats.tests} passed`);
        } catch (parseError) {
          console.log('⚠️  Could not parse test results, but test completed');
        }
      }
    }

    console.log(`\n📊 E2E Tests Summary: ${testResults.e2e.passed}/${testResults.e2e.total} passed`);
  } catch (error) {
    console.log('❌ Error running E2E tests:', error.message);
  }
}

async function runAccessibilityTests() {
  console.log('\n♿ Running Accessibility Tests...');

  try {
    // Run accessibility-specific tests
    const result = runCommand('npx playwright test tests/e2e/performance-accessibility.spec.ts --grep="@a11y"',
                              'Running accessibility tests');

    if (result.success) {
      console.log('✅ Accessibility tests completed');
    }
  } catch (error) {
    console.log('❌ Error running accessibility tests:', error.message);
  }
}

async function validateTestFiles() {
  console.log('\n🔍 Validating Test Files Exist...');

  let allFilesExist = true;

  // Check E2E test files
  for (const testFile of testConfig.e2eTests) {
    const fullPath = path.join(__dirname, testFile);
    if (fs.existsSync(fullPath)) {
      console.log(`✅ E2E Test: ${testFile}`);
    } else {
      console.log(`❌ Missing E2E Test: ${testFile}`);
      allFilesExist = false;
    }
  }

  // Check component test files
  for (const testFile of testConfig.componentTests) {
    const fullPath = path.join(__dirname, testFile);
    if (fs.existsSync(fullPath)) {
      console.log(`✅ Component Test: ${testFile}`);
    } else {
      console.log(`❌ Missing Component Test: ${testFile}`);
      allFilesExist = false;
    }
  }

  if (!allFilesExist) {
    console.log('⚠️  Some test files are missing. Please ensure all test files are created.');
  }

  return allFilesExist;
}

async function main() {
  console.log('🎯 Agency Swarm Phase 2 - Comprehensive Test Validation');
  console.log('═'.repeat(80));

  try {
    // Validate test files exist
    await validateTestFiles();

    // Create test results directory
    const resultsDir = path.join(__dirname, 'public/test-results');
    fs.mkdirSync(resultsDir, { recursive: true });

    // Run all test suites
    await runComponentTests();
    await runE2ETests();
    await runAccessibilityTests();

    // Analyze results
    analyzePerformance();

    // Generate comprehensive report
    const report = generateReport();

    // Summary
    console.log('\n🎉 Test Validation Complete!');
    console.log('═'.repeat(50));

    const successRate = parseFloat(report.summary.successRate);
    const coverageRate = report.summary.overallCoverage;

    if (successRate >= 90 && coverageRate >= testConfig.coverageThreshold) {
      console.log('🟢 EXCELLENT: High test coverage and success rate');
    } else if (successRate >= 80 && coverageRate >= 70) {
      console.log('🟡 GOOD: Adequate test coverage and success rate');
    } else if (successRate >= 70 && coverageRate >= 60) {
      console.log('🟠 ACCEPTABLE: Test coverage needs improvement');
    } else {
      console.log('🔴 NEEDS IMPROVEMENT: Low test coverage or success rate');
    }

    console.log('\n📋 Next Steps:');
    if (testResults.errors.length > 0) {
      console.log('  • Fix encountered errors');
    }
    if (coverageRate < testConfig.coverageThreshold) {
      console.log('  • Increase test coverage to meet threshold');
    }
    if (successRate < 90) {
      console.log('  • Fix failing tests to improve success rate');
    }

    console.log(`\n📄 Full report available at: public/test-results/comprehensive-test-report.json`);

  } catch (error) {
    console.log('❌ Fatal error during test validation:', error);
    process.exit(1);
  }
}

// Run the validation
if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  runComponentTests,
  runE2ETests,
  validateTestFiles,
  generateReport,
  testConfig
};