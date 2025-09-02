#!/usr/bin/env node
/**
 * Archon Light Mode Setup Script
 * ==============================
 * Prepares the environment for quick, lightweight Archon usage
 */

const fs = require('fs');
const path = require('path');

console.log('🪶 Setting up Archon Light Mode...');

// Create necessary directories
const dirs = [
  'archon-light-data',
  'archon-light-data/knowledge',
  'archon-light-data/chats',
  'archon-light-data/logs'
];

dirs.forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`📁 Created directory: ${dir}`);
  }
});

// Check for .env file
if (!fs.existsSync('.env')) {
  if (fs.existsSync('.env.light')) {
    fs.copyFileSync('.env.light', '.env');
    console.log('📝 Copied .env.light to .env');
    console.log('⚠️  Please edit .env and add your API key!');
  } else {
    console.log('❌ No .env or .env.light file found');
    console.log('💡 Please create .env with your API key');
    process.exit(1);
  }
}

// Verify API key is set
const envContent = fs.readFileSync('.env', 'utf8');
const hasApiKey = envContent.includes('OPENAI_API_KEY=sk-') || 
                  envContent.includes('ANTHROPIC_API_KEY=sk-') ||
                  envContent.includes('GEMINI_API_KEY=');

if (!hasApiKey) {
  console.log('⚠️  API key not configured in .env file');
  console.log('📝 Please add one of the following to .env:');
  console.log('   OPENAI_API_KEY=your-key-here');
  console.log('   ANTHROPIC_API_KEY=your-key-here'); 
  console.log('   GEMINI_API_KEY=your-key-here');
}

// Create simple UI if it doesn't exist
const uiPackageJson = 'archon-ui-light/package.json';
if (!fs.existsSync(uiPackageJson)) {
  const uiPackage = {
    name: 'archon-ui-light',
    version: '1.0.0',
    private: true,
    dependencies: {
      'react': '^18.2.0',
      'react-dom': '^18.2.0',
      'react-scripts': '^5.0.1'
    },
    scripts: {
      'start': 'BROWSER=none react-scripts start',
      'build': 'react-scripts build'
    },
    browserslist: {
      production: ['>0.2%', 'not dead', 'not op_mini all'],
      development: ['last 1 chrome version', 'last 1 firefox version', 'last 1 safari version']
    }
  };
  
  fs.writeFileSync(uiPackageJson, JSON.stringify(uiPackage, null, 2));
  console.log('📦 Created light UI package.json');
}

console.log('✅ Archon Light Mode setup complete!');
console.log('🚀 Run: npm run light');