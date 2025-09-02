/**
 * Vite Performance Optimization Configuration
 * 
 * Optimizations:
 * - Bundle splitting and code splitting
 * - Bundle size monitoring and validation
 * - Tree shaking optimization
 * - Dependency optimization
 * - Build performance monitoring
 */

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Bundle size limits (in KB)
const BUNDLE_LIMITS = {
  vendor: 400,      // React, React DOM, core libraries
  charts: 300,      // Chart libraries
  ui: 200,          // UI components
  deepconf: 150,    // DeepConf specific code
  main: 200,        // Main application code
} as const;

// Performance monitoring plugin
function performanceMonitoringPlugin() {
  let buildStartTime: number;
  
  return {
    name: 'performance-monitoring',
    buildStart() {
      buildStartTime = Date.now();
      console.log('🚀 Build started - Performance monitoring enabled');
    },
    
    async generateBundle(options: any, bundle: any) {
      // Analyze bundle sizes
      const chunks = Object.values(bundle).filter((chunk: any) => chunk.type === 'chunk');
      const assets = Object.values(bundle).filter((asset: any) => asset.type === 'asset');
      
      console.log('\n📊 Bundle Analysis:');
      console.log('==================');
      
      let totalSize = 0;
      const chunkSizes: Record<string, number> = {};
      
      for (const chunk of chunks) {
        const chunkData = chunk as any;
        const size = chunkData.code.length;
        totalSize += size;
        chunkSizes[chunkData.fileName] = size;
        
        const sizeKB = (size / 1024).toFixed(2);
        const status = this.checkSizeLimit(chunkData.name, size);
        console.log(`${status} ${chunkData.fileName}: ${sizeKB}KB`);
      }
      
      for (const asset of assets) {
        const assetData = asset as any;
        const size = assetData.source.length || 0;
        totalSize += size;
        
        const sizeKB = (size / 1024).toFixed(2);
        console.log(`📄 ${assetData.fileName}: ${sizeKB}KB`);
      }
      
      const totalSizeKB = (totalSize / 1024).toFixed(2);
      console.log(`\n📦 Total bundle size: ${totalSizeKB}KB`);
      
      // Check against limits
      this.validateBundleSizes(chunkSizes);
      
      // Generate bundle report
      await this.generateBundleReport({
        chunks: chunkSizes,
        totalSize: totalSize,
        timestamp: new Date().toISOString()
      });
    },
    
    closeBundle() {
      const buildTime = Date.now() - buildStartTime;
      console.log(`\n⚡ Build completed in ${buildTime}ms`);
    },
    
    checkSizeLimit(chunkName: string, size: number): string {
      const sizeKB = size / 1024;
      
      for (const [limitName, limitKB] of Object.entries(BUNDLE_LIMITS)) {
        if (chunkName?.includes(limitName)) {
          if (sizeKB > limitKB) {
            return `❌`;
          } else if (sizeKB > limitKB * 0.8) {
            return `⚠️ `;
          } else {
            return `✅`;
          }
        }
      }
      
      // Default check for main bundle
      if (sizeKB > 500) {
        return `❌`;
      } else if (sizeKB > 400) {
        return `⚠️ `;
      }
      return `✅`;
    },
    
    validateBundleSizes(chunkSizes: Record<string, number>) {
      let hasErrors = false;
      
      console.log('\n🔍 Bundle Size Validation:');
      console.log('=========================');
      
      for (const [fileName, size] of Object.entries(chunkSizes)) {
        const sizeKB = size / 1024;
        
        for (const [limitName, limitKB] of Object.entries(BUNDLE_LIMITS)) {
          if (fileName.includes(limitName) && sizeKB > limitKB) {
            console.log(`❌ ${fileName} exceeds limit: ${sizeKB.toFixed(2)}KB > ${limitKB}KB`);
            hasErrors = true;
          }
        }
      }
      
      if (!hasErrors) {
        console.log('✅ All bundles within size limits');
      }
    },
    
    async generateBundleReport(data: any) {
      const reportPath = path.join(process.cwd(), 'bundle-report.json');
      const fs = await import('fs/promises');
      
      try {
        await fs.writeFile(reportPath, JSON.stringify(data, null, 2));
        console.log(`\n📄 Bundle report saved to: ${reportPath}`);
      } catch (error) {
        console.warn('Failed to save bundle report:', error);
      }
    }
  };
}

// Tree shaking optimization plugin
function treeShakingPlugin() {
  return {
    name: 'tree-shaking-optimizer',
    config(config: any) {
      // Ensure proper tree shaking
      config.build = config.build || {};
      config.build.rollupOptions = config.build.rollupOptions || {};
      config.build.rollupOptions.treeshake = {
        moduleSideEffects: false,
        propertyReadSideEffects: false,
        unknownGlobalSideEffects: false
      };
    }
  };
}

// Bundle analyzer plugin (development)
function bundleAnalyzerPlugin() {
  return {
    name: 'bundle-analyzer',
    async closeBundle() {
      if (process.env.ANALYZE_BUNDLE === 'true') {
        try {
          console.log('\n📊 Generating bundle analysis...');
          await execAsync('npx vite-bundle-analyzer');
        } catch (error) {
          console.warn('Bundle analyzer failed:', error);
        }
      }
    }
  };
}

export default defineConfig({
  plugins: [
    react({
      // Optimize React for production
      babel: {
        plugins: [
          // Remove console.logs in production
          process.env.NODE_ENV === 'production' && [
            'transform-remove-console',
            { exclude: ['error', 'warn'] }
          ]
        ].filter(Boolean)
      }
    }),
    performanceMonitoringPlugin(),
    treeShakingPlugin(),
    bundleAnalyzerPlugin()
  ],
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  
  build: {
    // Target modern browsers for smaller bundles
    target: 'esnext',
    
    // Enable minification
    minify: 'esbuild',
    
    // Source maps for debugging (disable in production for smaller bundles)
    sourcemap: process.env.NODE_ENV !== 'production',
    
    // Optimize chunks
    rollupOptions: {
      output: {
        // Manual chunk splitting for optimal loading
        manualChunks: {
          // Core React libraries
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          
          // Chart and visualization libraries
          'vendor-charts': [
            'recharts',
            // Note: Remove duplicate react-flow libraries
          ],
          
          // UI and utility libraries
          'vendor-ui': [
            'lucide-react',
            'clsx',
            'tailwind-merge',
            'date-fns'
          ],
          
          // Socket.IO and real-time features
          'vendor-realtime': [
            'socket.io-client'
          ],
          
          // DeepConf specific components
          'deepconf': [
            './src/components/deepconf/index.ts',
            './src/services/deepconfService.ts',
            './src/hooks/useOptimizedDeepConf.ts'
          ],
          
          // Async components (only if large)
          'async-components': [
            './src/components/deepconf/SCWTDashboard.tsx',
            './src/components/deepconf/index.ts'
          ]
        },
        
        // Optimize chunk naming
        chunkFileNames: (chunkInfo) => {
          return `js/[name]-[hash].js`;
        },
        
        // Optimize asset naming
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name?.split('.') || [];
          const extType = info[info.length - 1];
          
          if (/\.(png|jpe?g|gif|svg|webp|avif)$/i.test(assetInfo.name || '')) {
            return `images/[name]-[hash].[ext]`;
          }
          
          if (/\.(woff2?|eot|ttf|otf)$/i.test(assetInfo.name || '')) {
            return `fonts/[name]-[hash].[ext]`;
          }
          
          return `assets/[name]-[hash].[ext]`;
        }
      },
      
      // External dependencies (CDN optimization)
      external: process.env.USE_CDN === 'true' ? [
        // Can externalize large libraries to CDN
        // 'react', 'react-dom' 
      ] : []
    },
    
    // Optimize for bundle size
    reportCompressedSize: true,
    chunkSizeWarningLimit: 500, // Warn for chunks > 500KB
    
    // Enable compression
    assetsInlineLimit: 4096, // Inline assets < 4KB
  },
  
  // Development optimizations
  server: {
    host: '0.0.0.0',
    port: parseInt(process.env.ARCHON_UI_PORT || '3737'),
    strictPort: true
  },
  
  // Optimize dependencies
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'socket.io-client',
      'date-fns',
      'clsx'
    ],
    exclude: [
      // Exclude large libraries that should be lazy loaded
      'framer-motion'
    ]
  },
  
  // Define environment variables
  define: {
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __PERFORMANCE_MODE__: JSON.stringify(process.env.NODE_ENV === 'production')
  }
});

// Export bundle size utilities
export const bundleUtils = {
  checkSizeLimit: (size: number, limit: number) => {
    const ratio = size / limit;
    if (ratio > 1) return 'error';
    if (ratio > 0.8) return 'warning';
    return 'ok';
  },
  
  formatSize: (bytes: number) => {
    const units = ['B', 'KB', 'MB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  }
};