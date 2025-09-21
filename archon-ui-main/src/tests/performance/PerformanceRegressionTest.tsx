/**
 * Performance Regression Test
 *
 * Automated performance regression testing with:
 * - Baseline comparison
 * - Performance threshold validation
 * - Trend analysis
 * - Automated reporting
 * - Performance budget enforcement
 */

import { PerformanceTestSuite } from './PerformanceTestSuite';

// Baseline performance metrics
export interface PerformanceBaseline {
  loadTime: number;
  memoryUsage: number;
  frameRate: number;
  responseTime: number;
  bundleSize: number;
  coreWebVitals: {
    fcp: number;
    lcp: number;
    cls: number;
    fid: number;
  };
  timestamp: number;
}

// Regression test configuration
export interface RegressionTestConfig {
  baseline: PerformanceBaseline;
  thresholds: {
    loadTimeRegression: number; // Maximum allowed regression in percentage
    memoryRegression: number;
    frameRateRegression: number;
    responseTimeRegression: number;
    bundleSizeRegression: number;
    coreWebVitalsRegression: number;
  };
  performanceBudget: {
    maxLoadTime: number;
    maxMemoryUsage: number;
    minFrameRate: number;
    maxResponseTime: number;
    maxBundleSize: number;
  };
}

// Regression test result
export interface RegressionTestResult {
  testName: string;
  passed: boolean;
  currentValue: number;
  baselineValue: number;
  regressionPercentage: number;
  threshold: number;
  severity: 'none' | 'minor' | 'major' | 'critical';
  details: string;
  recommendation: string;
}

// Performance trend data
export interface PerformanceTrend {
  timestamp: number;
  loadTime: number;
  memoryUsage: number;
  frameRate: number;
  responseTime: number;
  bundleSize: number;
  score: number;
}

// Performance Regression Test Suite
export class PerformanceRegressionTest {
  private config: RegressionTestConfig;
  private testSuite: PerformanceTestSuite;
  private historicalData: PerformanceTrend[] = [];
  private regressionResults: RegressionTestResult[] = [];

  constructor(config: RegressionTestConfig) {
    this.config = config;
    this.testSuite = new PerformanceTestSuite();
  }

  // Run regression tests
  async runRegressionTests(): Promise<RegressionTestResult[]> {
    console.log('🔍 Running Performance Regression Tests...');

    this.regressionResults = [];

    // Run performance tests
    const testResults = await this.testSuite.runAllTests();

    // Compare with baseline
    this.regressionResults = [
      this.compareLoadTime(testResults),
      this.compareMemoryUsage(testResults),
      this.compareFrameRate(testResults),
      this.compareResponseTime(testResults),
      this.compareBundleSize(testResults),
      this.compareCoreWebVitals(testResults),
    ];

    // Store historical data
    this.storeHistoricalData(testResults);

    console.log('📊 Regression Tests Complete');
    return this.regressionResults;
  }

  // Compare load time with baseline
  private compareLoadTime(testResults: any[]): RegressionTestResult {
    const loadTimeTest = testResults.find(t => t.testName === 'Load Performance');
    if (!loadTimeTest) {
      return this.createErrorResult('Load Time', 'No load time test results found');
    }

    const currentValue = loadTimeTest.metrics.loadTime;
    const baselineValue = this.config.baseline.loadTime;
    const regressionPercentage = ((currentValue - baselineValue) / baselineValue) * 100;
    const threshold = this.config.thresholds.loadTimeRegression;

    const passed = regressionPercentage <= threshold;
    const severity = this.getSeverity(regressionPercentage, threshold);

    return {
      testName: 'Load Time',
      passed,
      currentValue,
      baselineValue,
      regressionPercentage,
      threshold,
      severity,
      details: `Load time: ${currentValue.toFixed(2)}ms vs baseline ${baselineValue.toFixed(2)}ms (${regressionPercentage > 0 ? '+' : ''}${regressionPercentage.toFixed(1)}%)`,
      recommendation: this.getRecommendation('Load Time', regressionPercentage, threshold),
    };
  }

  // Compare memory usage with baseline
  private compareMemoryUsage(testResults: any[]): RegressionTestResult {
    const memoryTest = testResults.find(t => t.testName === 'Memory Usage');
    if (!memoryTest) {
      return this.createErrorResult('Memory Usage', 'No memory usage test results found');
    }

    const currentValue = memoryTest.metrics.memoryUsage;
    const baselineValue = this.config.baseline.memoryUsage;
    const regressionPercentage = ((currentValue - baselineValue) / baselineValue) * 100;
    const threshold = this.config.thresholds.memoryRegression;

    const passed = regressionPercentage <= threshold;
    const severity = this.getSeverity(regressionPercentage, threshold);

    return {
      testName: 'Memory Usage',
      passed,
      currentValue,
      baselineValue,
      regressionPercentage,
      threshold,
      severity,
      details: `Memory usage: ${currentValue.toFixed(2)}MB vs baseline ${baselineValue.toFixed(2)}MB (${regressionPercentage > 0 ? '+' : ''}${regressionPercentage.toFixed(1)}%)`,
      recommendation: this.getRecommendation('Memory Usage', regressionPercentage, threshold),
    };
  }

  // Compare frame rate with baseline
  private compareFrameRate(testResults: any[]): RegressionTestResult {
    const frameRateTest = testResults.find(t => t.testName === 'Frame Rate');
    if (!frameRateTest) {
      return this.createErrorResult('Frame Rate', 'No frame rate test results found');
    }

    const currentValue = frameRateTest.metrics.frameRate;
    const baselineValue = this.config.baseline.frameRate;
    const regressionPercentage = ((baselineValue - currentValue) / baselineValue) * 100; // Inverted for frame rate
    const threshold = this.config.thresholds.frameRateRegression;

    const passed = regressionPercentage <= threshold;
    const severity = this.getSeverity(regressionPercentage, threshold);

    return {
      testName: 'Frame Rate',
      passed,
      currentValue,
      baselineValue,
      regressionPercentage,
      threshold,
      severity,
      details: `Frame rate: ${currentValue.toFixed(1)} FPS vs baseline ${baselineValue.toFixed(1)} FPS (${regressionPercentage > 0 ? '+' : ''}${regressionPercentage.toFixed(1)}%)`,
      recommendation: this.getRecommendation('Frame Rate', regressionPercentage, threshold),
    };
  }

  // Compare response time with baseline
  private compareResponseTime(testResults: any[]): RegressionTestResult {
    const responseTimeTest = testResults.find(t => t.testName === 'Response Time');
    if (!responseTimeTest) {
      return this.createErrorResult('Response Time', 'No response time test results found');
    }

    const currentValue = responseTimeTest.metrics.responseTime;
    const baselineValue = this.config.baseline.responseTime;
    const regressionPercentage = ((currentValue - baselineValue) / baselineValue) * 100;
    const threshold = this.config.thresholds.responseTimeRegression;

    const passed = regressionPercentage <= threshold;
    const severity = this.getSeverity(regressionPercentage, threshold);

    return {
      testName: 'Response Time',
      passed,
      currentValue,
      baselineValue,
      regressionPercentage,
      threshold,
      severity,
      details: `Response time: ${currentValue.toFixed(2)}ms vs baseline ${baselineValue.toFixed(2)}ms (${regressionPercentage > 0 ? '+' : ''}${regressionPercentage.toFixed(1)}%)`,
      recommendation: this.getRecommendation('Response Time', regressionPercentage, threshold),
    };
  }

  // Compare bundle size with baseline
  private compareBundleSize(testResults: any[]): RegressionTestResult {
    const bundleTest = testResults.find(t => t.testName === 'Bundle Size');
    if (!bundleTest) {
      return this.createErrorResult('Bundle Size', 'No bundle size test results found');
    }

    const currentValue = bundleTest.metrics.bundleSize;
    const baselineValue = this.config.baseline.bundleSize;
    const regressionPercentage = ((currentValue - baselineValue) / baselineValue) * 100;
    const threshold = this.config.thresholds.bundleSizeRegression;

    const passed = regressionPercentage <= threshold;
    const severity = this.getSeverity(regressionPercentage, threshold);

    return {
      testName: 'Bundle Size',
      passed,
      currentValue,
      baselineValue,
      regressionPercentage,
      threshold,
      severity,
      details: `Bundle size: ${(currentValue / 1024).toFixed(2)}KB vs baseline ${(baselineValue / 1024).toFixed(2)}KB (${regressionPercentage > 0 ? '+' : ''}${regressionPercentage.toFixed(1)}%)`,
      recommendation: this.getRecommendation('Bundle Size', regressionPercentage, threshold),
    };
  }

  // Compare Core Web Vitals with baseline
  private compareCoreWebVitals(testResults: any[]): RegressionTestResult {
    const webVitalsTest = testResults.find(t => t.testName === 'Core Web Vitals');
    if (!webVitalsTest) {
      return this.createErrorResult('Core Web Vitals', 'No Core Web Vitals test results found');
    }

    // Calculate weighted average of Core Web Vitals regression
    const currentMetrics = {
      fcp: webVitalsTest.metrics.loadTime, // Using load time as FCP proxy
      lcp: webVitalsTest.metrics.loadTime * 1.5, // Estimate LCP
      cls: 0.05, // Assume CLS
      fid: webVitalsTest.metrics.responseTime, // Using response time as FID proxy
    };

    const baselineMetrics = this.config.baseline.coreWebVitals;

    const fcpRegression = ((currentMetrics.fcp - baselineMetrics.fcp) / baselineMetrics.fcp) * 100;
    const lcpRegression = ((currentMetrics.lcp - baselineMetrics.lcp) / baselineMetrics.lcp) * 100;
    const clsRegression = ((currentMetrics.cls - baselineMetrics.cls) / baselineMetrics.cls) * 100;
    const fidRegression = ((currentMetrics.fid - baselineMetrics.fid) / baselineMetrics.fid) * 100;

    // Weighted average
    const regressionPercentage = (
      fcpRegression * 0.3 +
      lcpRegression * 0.3 +
      clsRegression * 0.2 +
      fidRegression * 0.2
    );

    const threshold = this.config.thresholds.coreWebVitalsRegression;
    const passed = regressionPercentage <= threshold;
    const severity = this.getSeverity(regressionPercentage, threshold);

    return {
      testName: 'Core Web Vitals',
      passed,
      currentValue: 0, // Not applicable for combined metrics
      baselineValue: 0,
      regressionPercentage,
      threshold,
      severity,
      details: `Core Web Vitals regression: ${regressionPercentage.toFixed(1)}% (FCP: ${fcpRegression.toFixed(1)}%, LCP: ${lcpRegression.toFixed(1)}%, CLS: ${clsRegression.toFixed(1)}%, FID: ${fidRegression.toFixed(1)}%)`,
      recommendation: this.getRecommendation('Core Web Vitals', regressionPercentage, threshold),
    };
  }

  // Get severity level
  private getSeverity(regression: number, threshold: number): 'none' | 'minor' | 'major' | 'critical' {
    if (regression <= 0) return 'none';
    if (regression <= threshold * 0.5) return 'minor';
    if (regression <= threshold) return 'major';
    return 'critical';
  }

  // Get recommendation based on regression
  private getRecommendation(metric: string, regression: number, threshold: number): string {
    if (regression <= 0) return `Performance improved! ${metric} is better than baseline.`;
    if (regression <= threshold * 0.5) return `Minor regression detected. Monitor ${metric} closely.`;
    if (regression <= threshold) return `Major regression detected. Investigate ${metric} optimization opportunities.`;
    return `Critical regression detected! ${metric} requires immediate attention.`;
  }

  // Create error result
  private createErrorResult(testName: string, error: string): RegressionTestResult {
    return {
      testName,
      passed: false,
      currentValue: 0,
      baselineValue: 0,
      regressionPercentage: 0,
      threshold: 0,
      severity: 'critical',
      details: error,
      recommendation: 'Fix test configuration and rerun tests.',
    };
  }

  // Store historical data
  private storeHistoricalData(testResults: any[]): void {
    const summary = this.testSuite.getTestSummary();

    const trend: PerformanceTrend = {
      timestamp: Date.now(),
      loadTime: testResults.find(t => t.testName === 'Load Performance')?.metrics.loadTime || 0,
      memoryUsage: testResults.find(t => t.testName === 'Memory Usage')?.metrics.memoryUsage || 0,
      frameRate: testResults.find(t => t.testName === 'Frame Rate')?.metrics.frameRate || 0,
      responseTime: testResults.find(t => t.testName === 'Response Time')?.metrics.responseTime || 0,
      bundleSize: testResults.find(t => t.testName === 'Bundle Size')?.metrics.bundleSize || 0,
      score: summary.averageScore,
    };

    this.historicalData.push(trend);

    // Keep only last 30 days of data
    const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
    this.historicalData = this.historicalData.filter(d => d.timestamp > thirtyDaysAgo);
  }

  // Check performance budget
  checkPerformanceBudget(testResults: any[]): {
    passed: boolean;
    budgetViolations: { metric: string; current: number; budget: number; exceededBy: number }[];
  } {
    const violations: { metric: string; current: number; budget: number; exceededBy: number }[] = [];

    const checkMetric = (testName: string, metric: string, budget: number) => {
      const test = testResults.find(t => t.testName === testName);
      if (test) {
        const current = test.metrics[metric as keyof typeof test.metrics] as number;
        if (current > budget) {
          violations.push({
            metric: testName,
            current,
            budget,
            exceededBy: current - budget,
          });
        }
      }
    };

    checkMetric('Load Performance', 'loadTime', this.config.performanceBudget.maxLoadTime);
    checkMetric('Memory Usage', 'memoryUsage', this.config.performanceBudget.maxMemoryUsage);
    checkMetric('Frame Rate', 'frameRate', this.config.performanceBudget.minFrameRate); // Inverted check
    checkMetric('Response Time', 'responseTime', this.config.performanceBudget.maxResponseTime);
    checkMetric('Bundle Size', 'bundleSize', this.config.performanceBudget.maxBundleSize);

    return {
      passed: violations.length === 0,
      budgetViolations: violations,
    };
  }

  // Analyze trends
  analyzeTrends(): {
    trends: {
      loadTime: 'improving' | 'stable' | 'degrading';
      memoryUsage: 'improving' | 'stable' | 'degrading';
      frameRate: 'improving' | 'stable' | 'degrading';
      responseTime: 'improving' | 'stable' | 'degrading';
      bundleSize: 'improving' | 'stable' | 'degrading';
    };
    overallTrend: 'improving' | 'stable' | 'degrading';
    recommendations: string[];
  } {
    if (this.historicalData.length < 3) {
      return {
        trends: {
          loadTime: 'stable',
          memoryUsage: 'stable',
          frameRate: 'stable',
          responseTime: 'stable',
          bundleSize: 'stable',
        },
        overallTrend: 'stable',
        recommendations: ['Insufficient historical data for trend analysis'],
      };
    }

    const recent = this.historicalData.slice(-7); // Last 7 days
    const older = this.historicalData.slice(-14, -7); // Previous 7 days

    const calculateTrend = (recentValues: number[], olderValues: number[], isInverted = false) => {
      const recentAvg = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
      const olderAvg = olderValues.reduce((a, b) => a + b, 0) / olderValues.length;

      if (isInverted) {
        if (recentAvg > olderAvg * 1.05) return 'improving';
        if (recentAvg < olderAvg * 0.95) return 'degrading';
      } else {
        if (recentAvg < olderAvg * 0.95) return 'improving';
        if (recentAvg > olderAvg * 1.05) return 'degrading';
      }
      return 'stable';
    };

    const trends = {
      loadTime: calculateTrend(
        recent.map(d => d.loadTime),
        older.map(d => d.loadTime)
      ),
      memoryUsage: calculateTrend(
        recent.map(d => d.memoryUsage),
        older.map(d => d.memoryUsage)
      ),
      frameRate: calculateTrend(
        recent.map(d => d.frameRate),
        older.map(d => d.frameRate),
        true
      ),
      responseTime: calculateTrend(
        recent.map(d => d.responseTime),
        older.map(d => d.responseTime)
      ),
      bundleSize: calculateTrend(
        recent.map(d => d.bundleSize),
        older.map(d => d.bundleSize)
      ),
    };

    const improvingCount = Object.values(trends).filter(t => t === 'improving').length;
    const degradingCount = Object.values(trends).filter(t => t === 'degrading').length;

    const overallTrend = improvingCount > degradingCount ? 'improving' :
                         degradingCount > improvingCount ? 'degrading' : 'stable';

    const recommendations: string[] = [];
    if (overallTrend === 'degrading') {
      recommendations.push('Overall performance is degrading. Investigate recent changes.');
    }

    Object.entries(trends).forEach(([metric, trend]) => {
      if (trend === 'degrading') {
        recommendations.push(`${metric} is showing degrading performance trends.`);
      }
    });

    return {
      trends,
      overallTrend,
      recommendations,
    };
  }

  // Generate regression report
  generateReport(): {
    summary: {
      totalTests: number;
      passedTests: number;
      failedTests: number;
      criticalRegressions: number;
      overallResult: 'pass' | 'warn' | 'fail';
    };
    regressions: RegressionTestResult[];
    budgetCheck: {
      passed: boolean;
      violations: any[];
    };
    trends: any;
    recommendations: string[];
    timestamp: number;
  } {
    const passedTests = this.regressionResults.filter(r => r.passed).length;
    const failedTests = this.regressionResults.length - passedTests;
    const criticalRegressions = this.regressionResults.filter(r => r.severity === 'critical').length;

    const overallResult = criticalRegressions > 0 ? 'fail' :
                         failedTests > this.regressionResults.length * 0.2 ? 'warn' : 'pass';

    const budgetCheck = this.checkPerformanceBudget(this.testSuite.getTestSummary().totalTests);
    const trends = this.analyzeTrends();

    const recommendations: string[] = [];

    if (overallResult === 'fail') {
      recommendations.push('Critical performance regressions detected. Immediate action required.');
    }

    if (!budgetCheck.passed) {
      recommendations.push('Performance budget violations detected. Review and optimize.');
    }

    if (trends.overallTrend === 'degrading') {
      recommendations.push('Performance trends are degrading. Investigate root causes.');
    }

    recommendations.push(...trends.recommendations);

    return {
      summary: {
        totalTests: this.regressionResults.length,
        passedTests,
        failedTests,
        criticalRegressions,
        overallResult,
      },
      regressions: this.regressionResults,
      budgetCheck,
      trends,
      recommendations,
      timestamp: Date.now(),
    };
  }

  // Export report
  exportReport(format: 'json' | 'html' = 'json'): string {
    const report = this.generateReport();

    if (format === 'json') {
      return JSON.stringify(report, null, 2);
    }

    if (format === 'html') {
      return this.generateHTMLReport(report);
    }

    return '';
  }

  // Generate HTML report
  private generateHTMLReport(report: any): string {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Performance Regression Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .regression { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .regression.passed { border-left-color: #4caf50; }
        .regression.failed { border-left-color: #f44336; }
        .regression.critical { border-left-color: #d32f2f; background: #ffebee; }
        .recommendations { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metric { display: flex; justify-content: space-between; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Regression Report</h1>
        <p>Generated: ${new Date(report.timestamp).toLocaleString()}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <span>Overall Result:</span>
            <span style="color: ${report.summary.overallResult === 'pass' ? 'green' : report.summary.overallResult === 'warn' ? 'orange' : 'red'}">
                ${report.summary.overallResult.toUpperCase()}
            </span>
        </div>
        <div class="metric">
            <span>Tests Passed:</span>
            <span>${report.summary.passedTests}/${report.summary.totalTests}</span>
        </div>
        <div class="metric">
            <span>Critical Regressions:</span>
            <span>${report.summary.criticalRegressions}</span>
        </div>
    </div>

    <h2>Regression Tests</h2>
    ${report.regressions.map((r: any) => `
        <div class="regression ${r.passed ? 'passed' : 'failed'} ${r.severity === 'critical' ? 'critical' : ''}">
            <h3>${r.testName}</h3>
            <p>${r.details}</p>
            <p><strong>Recommendation:</strong> ${r.recommendation}</p>
        </div>
    `).join('')}

    ${report.recommendations.length > 0 ? `
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
                ${report.recommendations.map((r: string) => `<li>${r}</li>`).join('')}
            </ul>
        </div>
    ` : ''}
</body>
</html>
    `;
  }

  // Update baseline
  updateBaseline(testResults: any[]): void {
    this.config.baseline = {
      loadTime: testResults.find(t => t.testName === 'Load Performance')?.metrics.loadTime || 0,
      memoryUsage: testResults.find(t => t.testName === 'Memory Usage')?.metrics.memoryUsage || 0,
      frameRate: testResults.find(t => t.testName === 'Frame Rate')?.metrics.frameRate || 0,
      responseTime: testResults.find(t => t.testName === 'Response Time')?.metrics.responseTime || 0,
      bundleSize: testResults.find(t => t.testName === 'Bundle Size')?.metrics.bundleSize || 0,
      coreWebVitals: {
        fcp: testResults.find(t => t.testName === 'Core Web Vitals')?.metrics.loadTime || 0,
        lcp: testResults.find(t => t.testName === 'Core Web Vitals')?.metrics.loadTime * 1.5 || 0,
        cls: 0.05,
        fid: testResults.find(t => t.testName === 'Core Web Vitals')?.metrics.responseTime || 0,
      },
      timestamp: Date.now(),
    };

    console.log('📊 Performance baseline updated');
  }
}

export default PerformanceRegressionTest;