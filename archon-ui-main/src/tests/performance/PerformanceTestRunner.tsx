/**
 * Performance Test Runner
 *
 * Main orchestrator for all performance tests:
 * - Coordinates PerformanceTestSuite, PerformanceRegressionTest, and IntegrationTest
 * - Provides unified testing interface
 * - Generates comprehensive reports
 * - Supports multiple test configurations
 */

import { PerformanceTestSuite } from './PerformanceTestSuite';
import { PerformanceRegressionTest } from './PerformanceRegressionTest';
import { IntegrationTest, defaultUserScenarios } from './IntegrationTest';
import { PerformanceOptimizer, performanceConfigs, performancePresets } from '@/performance';

// Test runner configuration
export interface TestRunnerConfig {
  testCategories: {
    performance: boolean;
    regression: boolean;
    integration: boolean;
  };
  environment: 'development' | 'staging' | 'production';
  performanceConfig: any;
  baselineConfig: any;
  userScenarios?: any[];
  outputFormat: 'json' | 'html' | 'console';
  saveResults: boolean;
  maxDuration: number;
}

// Comprehensive test results
export interface ComprehensiveTestResults {
  summary: {
    overallScore: number;
    testCount: number;
    passedCount: number;
    failedCount: number;
    duration: number;
    timestamp: number;
  };
  performanceResults: any[];
  regressionResults: any[];
  integrationResults: any[];
  optimizationValidation: any[];
  recommendations: string[];
  metadata: {
    environment: string;
    config: TestRunnerConfig;
    systemInfo: any;
  };
}

// Main Performance Test Runner
export class PerformanceTestRunner {
  private config: TestRunnerConfig;
  private performanceOptimizer: PerformanceOptimizer;
  private testSuite: PerformanceTestSuite;
  private regressionTest: PerformanceRegressionTest;
  private integrationTest: IntegrationTest;
  private results: ComprehensiveTestResults | null = null;

  constructor(config: Partial<TestRunnerConfig> = {}) {
    this.config = this.mergeConfig(config);
    this.performanceOptimizer = new PerformanceOptimizer(this.config.performanceConfig);
    this.testSuite = new PerformanceTestSuite();
    this.regressionTest = new PerformanceRegressionTest(this.config.baselineConfig);
    this.integrationTest = new IntegrationTest({
      testDuration: this.config.maxDuration,
      userScenarios: this.config.userScenarios || defaultUserScenarios,
      environment: this.config.environment,
      performanceConfig: this.config.performanceConfig,
      baselineConfig: this.config.baselineConfig,
    });
  }

  // Merge configuration with defaults
  private mergeConfig(config: Partial<TestRunnerConfig>): TestRunnerConfig {
    return {
      testCategories: {
        performance: true,
        regression: true,
        integration: true,
        ...config.testCategories,
      },
      environment: config.environment || 'development',
      performanceConfig: config.performanceConfig || performanceConfigs.development,
      baselineConfig: config.baselineConfig || performanceConfigs.development,
      userScenarios: config.userScenarios,
      outputFormat: config.outputFormat || 'console',
      saveResults: config.saveResults ?? true,
      maxDuration: config.maxDuration || 300000, // 5 minutes
    };
  }

  // Run comprehensive test suite
  async runComprehensiveTests(): Promise<ComprehensiveTestResults> {
    console.log('🚀 Starting Comprehensive Performance Test Suite...');
    const startTime = performance.now();

    // Initialize test monitoring
    this.performanceOptimizer.startPerformanceMeasurement('comprehensive-test-suite');

    let performanceResults: any[] = [];
    let regressionResults: any[] = [];
    let integrationResults: any[] = [];
    let optimizationValidation: any[] = [];

    try {
      // Run performance tests if enabled
      if (this.config.testCategories.performance) {
        console.log('📊 Running Performance Tests...');
        performanceResults = await this.testSuite.runAllTests();
        console.log(`✅ Performance Tests Complete: ${performanceResults.length} tests executed`);
      }

      // Run regression tests if enabled
      if (this.config.testCategories.regression) {
        console.log('🔍 Running Regression Tests...');
        regressionResults = await this.regressionTest.runRegressionTests();
        console.log(`✅ Regression Tests Complete: ${regressionResults.length} tests executed`);
      }

      // Run integration tests if enabled
      if (this.config.testCategories.integration) {
        console.log('🎭 Running Integration Tests...');
        const integrationData = await this.integrationTest.runIntegrationTest();
        integrationResults = integrationData.integrationResults;
        optimizationValidation = integrationData.optimizationValidation;
        console.log(`✅ Integration Tests Complete: ${integrationResults.length} scenarios executed`);
      }

      // Calculate comprehensive results
      const duration = performance.now() - startTime;
      const overallScore = this.calculateOverallScore(
        performanceResults,
        regressionResults,
        integrationResults,
        optimizationValidation
      );

      const testCount = performanceResults.length + regressionResults.length + integrationResults.length;
      const passedCount = this.countPassedTests(performanceResults, regressionResults, integrationResults);
      const failedCount = testCount - passedCount;

      const recommendations = this.generateComprehensiveRecommendations(
        performanceResults,
        regressionResults,
        integrationResults,
        optimizationValidation
      );

      this.results = {
        summary: {
          overallScore,
          testCount,
          passedCount,
          failedCount,
          duration,
          timestamp: Date.now(),
        },
        performanceResults,
        regressionResults,
        integrationResults,
        optimizationValidation,
        recommendations,
        metadata: {
          environment: this.config.environment,
          config: this.config,
          systemInfo: this.getSystemInfo(),
        },
      };

      // Output results
      await this.outputResults();

      // Save results if enabled
      if (this.config.saveResults) {
        await this.saveResults();
      }

      console.log('🎉 Comprehensive Test Suite Complete!');
      console.log(`📈 Overall Score: ${overallScore}/100`);
      console.log(`✅ Passed: ${passedCount}/${testCount} tests`);
      console.log(`⏱️  Duration: ${duration.toFixed(2)}ms`);

      return this.results;

    } catch (error) {
      console.error('❌ Comprehensive Test Suite Failed:', error);
      throw error;
    }
  }

  // Calculate overall score from all test results
  private calculateOverallScore(
    performanceResults: any[],
    regressionResults: any[],
    integrationResults: any[],
    optimizationValidation: any[]
  ): number {
    let score = 0;
    let weight = 0;

    // Performance tests (35%)
    if (performanceResults.length > 0) {
      const performanceScore = performanceResults.reduce((sum, r) => sum + r.score, 0) / performanceResults.length;
      score += performanceScore * 0.35;
      weight += 0.35;
    }

    // Regression tests (30%)
    if (regressionResults.length > 0) {
      const regressionScore = regressionResults.filter(r => r.passed).length / regressionResults.length * 100;
      score += regressionScore * 0.3;
      weight += 0.3;
    }

    // Integration tests (25%)
    if (integrationResults.length > 0) {
      const integrationScore = integrationResults.reduce((sum, r) => sum + r.score, 0) / integrationResults.length;
      score += integrationScore * 0.25;
      weight += 0.25;
    }

    // Optimization validation (10%)
    if (optimizationValidation.length > 0) {
      const optimizationScore = optimizationValidation.reduce((sum, v) => sum + v.effectiveness, 0) / optimizationValidation.length;
      score += optimizationScore * 0.1;
      weight += 0.1;
    }

    return weight > 0 ? Math.round(score / weight) : 0;
  }

  // Count passed tests across all categories
  private countPassedTests(
    performanceResults: any[],
    regressionResults: any[],
    integrationResults: any[]
  ): number {
    let passed = 0;

    // Count passed performance tests
    passed += performanceResults.filter(r => r.passed).length;

    // Count passed regression tests
    passed += regressionResults.filter(r => r.passed).length;

    // Count passed integration tests
    passed += integrationResults.filter(r => r.passed).length;

    return passed;
  }

  // Generate comprehensive recommendations
  private generateComprehensiveRecommendations(
    performanceResults: any[],
    regressionResults: any[],
    integrationResults: any[],
    optimizationValidation: any[]
  ): string[] {
    const recommendations: string[] = [];

    // Performance test recommendations
    const failingPerformance = performanceResults.filter(r => !r.passed);
    if (failingPerformance.length > 0) {
      recommendations.push(`Address failing performance tests: ${failingPerformance.map(t => t.testName).join(', ')}`);
    }

    // Regression test recommendations
    const failingRegression = regressionResults.filter(r => !r.passed);
    if (failingRegression.length > 0) {
      recommendations.push(`Fix performance regressions: ${failingRegression.map(r => r.testName).join(', ')}`);
    }

    // Integration test recommendations
    const failingIntegration = integrationResults.filter(r => !r.passed);
    if (failingIntegration.length > 0) {
      recommendations.push(`Resolve failing user scenarios: ${failingIntegration.map(s => s.scenarioName).join(', ')}`);
    }

    // Optimization recommendations
    const ineffectiveOptimizations = optimizationValidation.filter(v => v.effectiveness < 20);
    if (ineffectiveOptimizations.length > 0) {
      recommendations.push(`Improve ineffective optimizations: ${ineffectiveOptimizations.map(v => v.feature).join(', ')}`);
    }

    // High-level recommendations based on overall score
    if (this.results && this.results.summary.overallScore < 70) {
      recommendations.push('Overall performance score is below 70%. Consider comprehensive optimization review.');
    }

    if (this.results && this.results.summary.overallScore >= 90) {
      recommendations.push('Excellent performance! Consider reducing testing frequency or focusing on edge cases.');
    }

    return recommendations;
  }

  // Output results in specified format
  private async outputResults(): Promise<void> {
    if (!this.results) return;

    switch (this.config.outputFormat) {
      case 'console':
        this.outputToConsole();
        break;
      case 'json':
        this.outputToJSON();
        break;
      case 'html':
        this.outputToHTML();
        break;
    }
  }

  // Output results to console
  private outputToConsole(): void {
    if (!this.results) return;

    console.log('\n'.repeat(2));
    console.log('='.repeat(60));
    console.log('COMPREHENSIVE PERFORMANCE TEST RESULTS');
    console.log('='.repeat(60));

    // Summary
    console.log('\n📊 SUMMARY');
    console.log('-'.repeat(40));
    console.log(`Overall Score: ${this.results.summary.overallScore}/100`);
    console.log(`Tests: ${this.results.summary.passedCount}/${this.results.summary.testCount} passed`);
    console.log(`Duration: ${this.results.summary.duration.toFixed(2)}ms`);
    console.log(`Environment: ${this.results.metadata.environment}`);

    // Performance Results
    if (this.results.performanceResults.length > 0) {
      console.log('\n📈 PERFORMANCE TESTS');
      console.log('-'.repeat(40));
      this.results.performanceResults.forEach(result => {
        const status = result.passed ? '✅' : '❌';
        console.log(`${status} ${result.testName}: ${result.score}/100 (${result.duration.toFixed(2)}ms)`);
      });
    }

    // Regression Results
    if (this.results.regressionResults.length > 0) {
      console.log('\n🔍 REGRESSION TESTS');
      console.log('-'.repeat(40));
      this.results.regressionResults.forEach(result => {
        const status = result.passed ? '✅' : '❌';
        console.log(`${status} ${result.testName}: ${result.regressionScore.toFixed(2)}%`);
      });
    }

    // Integration Results
    if (this.results.integrationResults.length > 0) {
      console.log('\n🎭 INTEGRATION TESTS');
      console.log('-'.repeat(40));
      this.results.integrationResults.forEach(result => {
        const status = result.passed ? '✅' : '❌';
        console.log(`${status} ${result.scenarioName}: ${result.score}/100`);
      });
    }

    // Optimization Validation
    if (this.results.optimizationValidation.length > 0) {
      console.log('\n⚡ OPTIMIZATION VALIDATION');
      console.log('-'.repeat(40));
      this.results.optimizationValidation.forEach(validation => {
        console.log(`🔧 ${validation.feature}: ${validation.effectiveness.toFixed(1)}% effective`);
      });
    }

    // Recommendations
    if (this.results.recommendations.length > 0) {
      console.log('\n💡 RECOMMENDATIONS');
      console.log('-'.repeat(40));
      this.results.recommendations.forEach((recommendation, index) => {
        console.log(`${index + 1}. ${recommendation}`);
      });
    }

    console.log('\n' + '='.repeat(60));
  }

  // Output results to JSON
  private outputToJSON(): void {
    if (!this.results) return;

    const jsonString = JSON.stringify(this.results, null, 2);
    console.log(jsonString);
  }

  // Output results to HTML
  private outputToHTML(): void {
    if (!this.results) return;

    const html = this.generateHTMLReport(this.results);
    console.log(html);
  }

  // Generate HTML report
  private generateHTMLReport(results: ComprehensiveTestResults): string {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Performance Test Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .summary-item {
            text-align: center;
        }
        .summary-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .summary-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .section {
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #333;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }
        .test-item {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .test-item.passed {
            border-left: 4px solid #28a745;
        }
        .test-item.failed {
            border-left: 4px solid #dc3545;
        }
        .test-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .test-name {
            font-weight: bold;
            color: #333;
        }
        .test-score {
            font-size: 1.2em;
            font-weight: bold;
        }
        .test-score.good {
            color: #28a745;
        }
        .test-score.fair {
            color: #ffc107;
        }
        .test-score.poor {
            color: #dc3545;
        }
        .test-details {
            font-size: 0.9em;
            color: #666;
        }
        .recommendations {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 20px;
        }
        .recommendations h3 {
            margin-top: 0;
            color: #856404;
        }
        .recommendations ul {
            margin-bottom: 0;
        }
        .recommendations li {
            margin-bottom: 8px;
        }
        .optimization-item {
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .optimization-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .effectiveness {
            font-weight: bold;
            font-size: 1.1em;
        }
        .effectiveness.high {
            color: #28a745;
        }
        .effectiveness.medium {
            color: #ffc107;
        }
        .effectiveness.low {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Comprehensive Performance Test Report</h1>
            <p>Environment: ${results.metadata.environment}</p>
            <p>Generated: ${new Date(results.summary.timestamp).toLocaleString()}</p>
        </div>

        <div class="summary">
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-value">${results.summary.overallScore}</div>
                    <div class="summary-label">Overall Score</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${results.summary.passedCount}/${results.summary.testCount}</div>
                    <div class="summary-label">Tests Passed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${results.summary.duration.toFixed(0)}ms</div>
                    <div class="summary-label">Duration</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${Math.round((results.summary.passedCount / results.summary.testCount) * 100)}%</div>
                    <div class="summary-label">Pass Rate</div>
                </div>
            </div>
        </div>

        ${results.performanceResults.length > 0 ? `
        <div class="section">
            <h2 class="section-title">📈 Performance Tests</h2>
            ${results.performanceResults.map(result => `
                <div class="test-item ${result.passed ? 'passed' : 'failed'}">
                    <div class="test-header">
                        <div class="test-name">${result.testName}</div>
                        <div class="test-score ${result.score >= 80 ? 'good' : result.score >= 60 ? 'fair' : 'poor'}">${result.score}/100</div>
                    </div>
                    <div class="test-details">
                        Duration: ${result.duration.toFixed(2)}ms | Status: ${result.passed ? 'Passed' : 'Failed'}
                    </div>
                </div>
            `).join('')}
        </div>
        ` : ''}

        ${results.regressionResults.length > 0 ? `
        <div class="section">
            <h2 class="section-title">🔍 Regression Tests</h2>
            ${results.regressionResults.map(result => `
                <div class="test-item ${result.passed ? 'passed' : 'failed'}">
                    <div class="test-header">
                        <div class="test-name">${result.testName}</div>
                        <div class="test-score ${result.passed ? 'good' : 'poor'}">${result.regressionScore.toFixed(2)}%</div>
                    </div>
                    <div class="test-details">
                        Status: ${result.passed ? 'Passed' : 'Failed'} | Improvement: ${result.improvement.toFixed(2)}%
                    </div>
                </div>
            `).join('')}
        </div>
        ` : ''}

        ${results.integrationResults.length > 0 ? `
        <div class="section">
            <h2 class="section-title">🎭 Integration Tests</h2>
            ${results.integrationResults.map(result => `
                <div class="test-item ${result.passed ? 'passed' : 'failed'}">
                    <div class="test-header">
                        <div class="test-name">${result.scenarioName}</div>
                        <div class="test-score ${result.score >= 80 ? 'good' : result.score >= 60 ? 'fair' : 'poor'}">${result.score}/100</div>
                    </div>
                    <div class="test-details">
                        ${result.details} | Steps: ${result.steps.length}
                    </div>
                </div>
            `).join('')}
        </div>
        ` : ''}

        ${results.optimizationValidation.length > 0 ? `
        <div class="section">
            <h2 class="section-title">⚡ Optimization Validation</h2>
            ${results.optimizationValidation.map(validation => `
                <div class="optimization-item">
                    <div class="optimization-header">
                        <div class="test-name">${validation.feature}</div>
                        <div class="effectiveness ${validation.effectiveness >= 70 ? 'high' : validation.effectiveness >= 40 ? 'medium' : 'low'}">${validation.effectiveness.toFixed(1)}%</div>
                    </div>
                    <div class="test-details">
                        Improvement: ${validation.metrics.improvement.toFixed(2)}% | Status: ${validation.enabled ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
            `).join('')}
        </div>
        ` : ''}

        ${results.recommendations.length > 0 ? `
        <div class="section">
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                <ul>
                    ${results.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
        ` : ''}
    </div>
</body>
</html>
    `;
  }

  // Save results to file
  private async saveResults(): Promise<void> {
    if (!this.results) return;

    const timestamp = new Date(this.results.summary.timestamp).toISOString().replace(/[:.]/g, '-');
    const filename = `performance-test-results-${timestamp}.json`;

    try {
      // In a real implementation, this would save to a file
      console.log(`📁 Results saved to: ${filename}`);

      // For now, store in localStorage for demo purposes
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(filename, JSON.stringify(this.results, null, 2));
      }
    } catch (error) {
      console.error('Failed to save results:', error);
    }
  }

  // Get system information
  private getSystemInfo(): any {
    return {
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Unknown',
      platform: typeof navigator !== 'undefined' ? navigator.platform : 'Unknown',
      memory: typeof performance !== 'undefined' && 'memory' in performance ? {
        usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
        totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
        jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit,
      } : null,
      cores: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : 'Unknown',
      timestamp: Date.now(),
    };
  }

  // Get test results
  getResults(): ComprehensiveTestResults | null {
    return this.results;
  }

  // Clear results
  clearResults(): void {
    this.results = null;
  }

  // Get configuration
  getConfig(): TestRunnerConfig {
    return this.config;
  }

  // Update configuration
  updateConfig(config: Partial<TestRunnerConfig>): void {
    this.config = this.mergeConfig(config);
  }
}

// Factory functions for common test configurations
export function createQuickTestRunner(): PerformanceTestRunner {
  return new PerformanceTestRunner({
    testCategories: {
      performance: true,
      regression: false,
      integration: false,
    },
    outputFormat: 'console',
    maxDuration: 60000, // 1 minute
  });
}

export function createFullTestRunner(): PerformanceTestRunner {
  return new PerformanceTestRunner({
    testCategories: {
      performance: true,
      regression: true,
      integration: true,
    },
    outputFormat: 'html',
    maxDuration: 300000, // 5 minutes
  });
}

export function createProductionTestRunner(): PerformanceTestRunner {
  return new PerformanceTestRunner({
    testCategories: {
      performance: true,
      regression: true,
      integration: true,
    },
    environment: 'production',
    performanceConfig: performanceConfigs.production,
    baselineConfig: performanceConfigs.production,
    outputFormat: 'html',
    maxDuration: 600000, // 10 minutes
  });
}

// Default export
export default PerformanceTestRunner;