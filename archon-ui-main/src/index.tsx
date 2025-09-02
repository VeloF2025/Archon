import './index.css';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';

// CACHE BUSTING: Force module refresh - timestamp: 2025-09-01-20:55 - TRAP ADDED
console.log('🚀 Archon UI starting with cache-busted modules, timestamp:', Date.now());

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}