// Simple Socket.IO client test
import { io } from 'socket.io-client';

console.log('Testing Socket.IO connection to localhost:8181...');

const socket = io('http://localhost:8181', {
  timeout: 30000,
  reconnection: true,
  reconnectionAttempts: 3,
  reconnectionDelay: 1000,
  transports: ['polling', 'websocket']
});

const startTime = Date.now();
let timeoutCount = 0;
let errorCount = 0;

socket.on('connect', () => {
  const connectTime = Date.now() - startTime;
  console.log(`✅ Connected successfully in ${connectTime}ms`);
  console.log(`🔌 Socket ID: ${socket.id}`);
  
  // Test sending a message
  setTimeout(() => {
    console.log('📤 Sending test message...');
    socket.emit('test_message', { content: 'Hello from test client!', timestamp: Date.now() });
  }, 1000);
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
});

socket.on('reconnect', (attemptNumber) => {
  console.log(`🔄 Reconnected after ${attemptNumber} attempts`);
});

socket.on('reconnect_attempt', (attemptNumber) => {
  console.log(`🔄 Reconnection attempt #${attemptNumber}`);
});

socket.on('reconnect_error', (error) => {
  console.log(`❌ Reconnection error: ${error.message}`);
});

socket.on('reconnect_failed', () => {
  console.log('❌ Reconnection failed - giving up');
});

// Listen for any events from server
socket.onAny((eventName, ...args) => {
  console.log(`📨 Received event '${eventName}':`, args);
});

// Test for 60 seconds
setTimeout(() => {
  console.log('\n=== TEST SUMMARY ===');
  console.log(`Connection state: ${socket.connected ? 'CONNECTED' : 'DISCONNECTED'}`);
  console.log(`Total errors: ${errorCount}`);
  console.log(`Timeout errors: ${timeoutCount}`);
  console.log(`Socket ID: ${socket.id || 'N/A'}`);
  
  if (timeoutCount === 0) {
    console.log('✅ NO TIMEOUT ERRORS DETECTED - Socket.IO appears stable');
  } else {
    console.log('⚠️ TIMEOUT ERRORS DETECTED - Connection may be unstable');
  }
  
  socket.disconnect();
  process.exit(0);
}, 60000);

console.log('🔍 Monitoring Socket.IO connection for 60 seconds...');