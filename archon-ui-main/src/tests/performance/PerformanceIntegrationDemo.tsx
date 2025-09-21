/**
 * Performance Integration Demo
 *
 * Complete demonstration of the performance optimization system:
 * - Shows all components working together
 * - Demonstrates real-world usage scenarios
 * - Provides performance metrics and validation
 */

import React, { useState, useEffect, useCallback } from 'react';
import { PerformanceOptimizer, performanceConfigs, usePerformanceMonitor } from '@/performance';
import { VirtualList } from '@/performance/virtualization-manager';
import { useCache } from '@/performance/cache-manager';
import { LazyImage } from '@/performance/lazy-loading';
import { PerformanceDashboard } from '@/components/performance/PerformanceDashboard';
import { PerformanceTestRunner, createFullTestRunner } from './PerformanceTestRunner';

// Demo data generator
const generateDemoData = (count: number) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    name: `Demo Item ${i}`,
    description: `This is a detailed description for demo item ${i}`,
    category: `Category ${i % 5}`,
    price: Math.floor(Math.random() * 1000) + 10,
    rating: (Math.random() * 5).toFixed(1),
    image: `https://picsum.photos/seed/item-${i}/300/200.jpg`,
    timestamp: Date.now() - Math.random() * 86400000, // Random time in last 24h
  }));
};

// Demo component using performance optimizations
export const PerformanceIntegrationDemo: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState<any[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [testResults, setTestResults] = useState<any>(null);
  const [showDashboard, setShowDashboard] = useState(false);

  // Performance monitoring
  const { metrics, score, monitor } = usePerformanceMonitor();

  // Caching
  const [cachedData, setCachedData] = useCache('demo-items', null, 300);

  // Performance optimizer
  const [optimizer] = useState(() => new PerformanceOptimizer(performanceConfigs.development));

  // Load demo data
  useEffect(() => {
    const loadData = async () => {
      monitor.startMeasurement('load-demo-data');

      try {
        // Check cache first
        if (cachedData) {
          setItems(cachedData);
          console.log('✅ Data loaded from cache');
        } else {
          // Generate new data (simulating API call)
          const demoData = generateDemoData(10000);
          setItems(demoData);
          setCachedData(demoData);
          console.log('✅ Data generated and cached');
        }
      } catch (error) {
        console.error('❌ Failed to load data:', error);
      } finally {
        const duration = monitor.endMeasurement('load-demo-data');
        console.log(`📊 Data load time: ${duration?.toFixed(2) || 'N/A'}ms`);
        setLoading(false);
      }
    };

    loadData();
  }, [monitor, cachedData, setCachedData]);

  // Filter items by category
  const filteredItems = React.useMemo(() => {
    if (selectedCategory === 'all') return items;
    return items.filter(item => item.category === selectedCategory);
  }, [items, selectedCategory]);

  // Get unique categories
  const categories = React.useMemo(() => {
    const cats = Array.from(new Set(items.map(item => item.category)));
    return ['all', ...cats];
  }, [items]);

  // Render item for virtual list
  const renderItem = useCallback((item: any) => (
    <div className="demo-item bg-white rounded-lg shadow-md p-4 mb-2 border border-gray-200 hover:shadow-lg transition-shadow">
      <div className="flex items-center space-x-4">
        <LazyImage
          src={item.image}
          alt={item.name}
          className="w-20 h-20 rounded-lg object-cover"
          placeholder={<div className="w-20 h-20 bg-gray-200 rounded-lg animate-pulse" />}
        />
        <div className="flex-1">
          <h3 className="font-semibold text-lg text-gray-800">{item.name}</h3>
          <p className="text-gray-600 text-sm mb-1">{item.description}</p>
          <div className="flex items-center space-x-4 text-sm">
            <span className="text-blue-600 font-medium">{item.category}</span>
            <span className="text-green-600 font-bold">${item.price}</span>
            <span className="text-yellow-500">⭐ {item.rating}</span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500">
            {new Date(item.timestamp).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  ), []);

  // Run performance tests
  const runPerformanceTests = useCallback(async () => {
    try {
      monitor.startMeasurement('performance-tests');
      console.log('🚀 Running performance tests...');

      const runner = createFullTestRunner();
      const results = await runner.runComprehensiveTests();

      setTestResults(results);
      const duration = monitor.endMeasurement('performance-tests');
      console.log(`✅ Performance tests completed in ${duration?.toFixed(2) || 'N/A'}ms`);

      return results;
    } catch (error) {
      console.error('❌ Performance tests failed:', error);
      return null;
    }
  }, [monitor]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      if (items.length > 0) {
        // Simulate price updates
        setItems(prevItems =>
          prevItems.map(item => ({
            ...item,
            price: item.price + (Math.random() - 0.5) * 10,
          }))
        );
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [items.length]);

  // Performance metrics display
  const renderPerformanceMetrics = () => (
    <div className="bg-gray-50 rounded-lg p-4 mb-4 border border-gray-200">
      <h3 className="text-lg font-semibold mb-2 text-gray-800">Performance Metrics</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-white p-3 rounded border">
          <div className="text-gray-600">Overall Score</div>
          <div className={`text-xl font-bold ${score >= 80 ? 'text-green-600' : score >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
            {score}/100
          </div>
        </div>
        <div className="bg-white p-3 rounded border">
          <div className="text-gray-600">Items Loaded</div>
          <div className="text-xl font-bold text-blue-600">{items.length.toLocaleString()}</div>
        </div>
        <div className="bg-white p-3 rounded border">
          <div className="text-gray-600">Memory Usage</div>
          <div className="text-xl font-bold text-purple-600">
            {metrics.memoryUsage ? `${metrics.memoryUsage.toFixed(1)}MB` : 'N/A'}
          </div>
        </div>
        <div className="bg-white p-3 rounded border">
          <div className="text-gray-600">Frame Rate</div>
          <div className="text-xl font-bold text-green-600">
            {metrics.frameRate ? `${metrics.frameRate}fps` : 'N/A'}
          </div>
        </div>
      </div>
    </div>
  );

  // Test results display
  const renderTestResults = () => {
    if (!testResults) return null;

    return (
      <div className="bg-blue-50 rounded-lg p-4 mb-4 border border-blue-200">
        <h3 className="text-lg font-semibold mb-2 text-blue-800">Performance Test Results</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white p-3 rounded border border-blue-300">
            <div className="text-blue-600 text-sm">Overall Score</div>
            <div className="text-2xl font-bold text-blue-800">
              {testResults.summary.overallScore}/100
            </div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-300">
            <div className="text-blue-600 text-sm">Tests Passed</div>
            <div className="text-2xl font-bold text-green-600">
              {testResults.summary.passedCount}/{testResults.summary.testCount}
            </div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-300">
            <div className="text-blue-600 text-sm">Duration</div>
            <div className="text-2xl font-bold text-purple-600">
              {testResults.summary.duration.toFixed(0)}ms
            </div>
          </div>
        </div>

        {testResults.recommendations.length > 0 && (
          <div className="mt-4">
            <h4 className="font-semibold text-blue-800 mb-2">Recommendations:</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              {testResults.recommendations.slice(0, 3).map((rec: string, index: number) => (
                <li key={index} className="flex items-start">
                  <span className="mr-2">💡</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading performance optimization demo...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          🚀 Performance Optimization Integration Demo
        </h1>
        <p className="text-gray-600">
          Complete demonstration of virtualization, caching, lazy loading, and performance monitoring
        </p>
      </div>

      {/* Performance Metrics */}
      {renderPerformanceMetrics()}

      {/* Test Results */}
      {renderTestResults()}

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6 border border-gray-200">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Category:</label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="border border-gray-300 rounded px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => runPerformanceTests()}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              Run Performance Tests
            </button>

            <button
              onClick={() => setShowDashboard(!showDashboard)}
              className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-colors text-sm font-medium"
            >
              {showDashboard ? 'Hide Dashboard' : 'Show Dashboard'}
            </button>

            <button
              onClick={() => optimizer.clearCache()}
              className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 transition-colors text-sm font-medium"
            >
              Clear Cache
            </button>
          </div>
        </div>
      </div>

      {/* Performance Dashboard */}
      {showDashboard && (
        <div className="mb-6">
          <PerformanceDashboard />
        </div>
      )}

      {/* Virtual List Demo */}
      <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Virtual List Demo ({filteredItems.length.toLocaleString()} items)
        </h2>

        <div className="border border-gray-200 rounded-lg overflow-hidden">
          <VirtualList
            items={filteredItems}
            renderItem={renderItem}
            itemHeight={120}
            containerHeight={600}
            overscanCount={8}
            enableDynamicSizing={true}
          />
        </div>

        <div className="mt-4 text-sm text-gray-600">
          <p>✅ Features demonstrated:</p>
          <ul className="mt-2 space-y-1 ml-4">
            <li>• Virtual scrolling for 10,000+ items</li>
            <li>• Multi-level caching with 5-minute TTL</li>
            <li>• Lazy loading images with placeholders</li>
            <li>• Real-time performance monitoring</li>
            <li>• Dynamic content updates</li>
            <li>• Responsive design</li>
          </ul>
        </div>
      </div>

      {/* Optimization Summary */}
      <div className="mt-8 bg-green-50 rounded-lg p-6 border border-green-200">
        <h2 className="text-xl font-semibold mb-4 text-green-800">
          🎯 Performance Optimizations Applied
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-green-700 mb-2">Virtualization</h3>
            <ul className="text-sm text-green-600 space-y-1">
              <li>• Dynamic windowing for large datasets</li>
              <li>• Efficient DOM manipulation</li>
              <li>• Smooth scrolling performance</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-green-700 mb-2">Caching</h3>
            <ul className="text-sm text-green-600 space-y-1">
              <li>• Memory cache for fast access</li>
              <li>• Session storage persistence</li>
              <li>• Intelligent cache invalidation</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-green-700 mb-2">Resource Optimization</h3>
            <ul className="text-sm text-green-600 space-y-1">
              <li>• Lazy loading with Intersection Observer</li>
              <li>• Image optimization and placeholders</li>
              <li>• Efficient bundle loading</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-green-700 mb-2">Monitoring</h3>
            <ul className="text-sm text-green-600 space-y-1">
              <li>• Real-time performance metrics</li>
              <li>• Core Web Vitals tracking</li>
              <li>• Performance scoring system</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceIntegrationDemo;