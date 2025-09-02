// Test Socket.IO with better timeout configuration
import { io } from 'socket.io-client';

console.log('Testing Socket.IO with optimized timeouts...');

const socket = io('http://localhost:8181', {
  // Increased timeout to match server configuration
  timeout: 120000,  // Match server pingTimeout (2 minutes)
  reconnection: true,
  reconnectionAttempts: 5,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  transports: ['polling', 'websocket']
});

const startTime = Date.now();
let timeoutCount = 0;
let errorCount = 0;
let connected = false;

socket.on('connect', () => {
  const connectTime = Date.now() - startTime;
  connected = true;
  console.log(`✅ Connected successfully in ${connectTime}ms`);
  console.log(`🔌 Socket ID: ${socket.id}`);
  
  // Test immediate message
  socket.emit('test_message', { content: 'Quick connection test!', timestamp: Date.now() });
});

socket.on('connect_error', (error) => {
  errorCount++;
  console.log(`❌ Connection error #${errorCount}: ${error.message}`);
  console.log(`❌ Error type: ${error.type || 'unknown'}`);
  
  if (error.message === 'timeout') {
    timeoutCount++;
    console.log(`⏱️ Timeout error count: ${timeoutCount}`);
  }
});

socket.on('disconnect', (reason) => {
  console.log(`🔌 Disconnected: ${reason}`);
  connected = false;
});

socket.on('reconnect', (attemptNumber) => {
  console.log(`🔄 Reconnected after ${attemptNumber} attempts`);
  connected = true;
});

socket.on('reconnect_attempt', (attemptNumber) => {
  console.log(`🔄 Reconnection attempt #${attemptNumber}`);
});

// Test for 30 seconds with better timeout
setTimeout(() => {
  console.log('\n=== TIMEOUT FIX TEST RESULTS ===');
  console.log(`Connection state: ${connected ? 'CONNECTED' : 'DISCONNECTED'}`);
  console.log(`Total errors: ${errorCount}`);
  console.log(`Timeout errors: ${timeoutCount}`);
  console.log(`Socket ID: ${socket.id || 'N/A'}`);
  
  if (timeoutCount === 0 && connected) {
    console.log('✅ SUCCESS: No timeout errors with increased timeout configuration!');
    console.log('✅ RECOMMENDATION: Update frontend timeout to 120000ms to match server');
  } else if (timeoutCount > 0) {
    console.log('⚠️ Still experiencing timeouts - may need server configuration adjustment');
  } else {
    console.log('❌ Connection failed completely');
  }
  
  socket.disconnect();
  process.exit(0);
}, 30000);

console.log('🔍 Testing optimized timeout configuration for 30 seconds...');