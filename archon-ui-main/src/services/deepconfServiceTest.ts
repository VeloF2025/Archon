/**
 * EMERGENCY DEBUG TEST - DeepConf Service Method Verification
 * This file tests the Socket.IO service methods to debug the runtime error
 * CACHE BUSTING: 2025-09-01-17:00
 */

import { knowledgeSocketIO, WebSocketService } from './socketIOService';
import { deepconfServiceFixed as deepconfService, createDeepConfServiceFixed as createDeepConfService } from './deepconfService_fixed';

// Immediate diagnostic function that runs on import
(function immediateDeepConfDiagnostic() {
  console.log('=== DEEPCONF EMERGENCY DIAGNOSTIC START ===');
  console.log('Timestamp:', new Date().toISOString());
  
  // Test 1: Verify knowledgeSocketIO exists
  console.log('1. knowledgeSocketIO exists:', !!knowledgeSocketIO);
  console.log('1. knowledgeSocketIO type:', typeof knowledgeSocketIO);
  console.log('1. knowledgeSocketIO constructor:', knowledgeSocketIO?.constructor?.name);
  
  // Test 2: Verify it's a WebSocketService instance
  console.log('2. knowledgeSocketIO instanceof WebSocketService:', knowledgeSocketIO instanceof WebSocketService);
  
  // Test 3: Check all required methods
  const requiredMethods = ['addMessageHandler', 'addStateChangeHandler', 'isConnected', 'connect'];
  requiredMethods.forEach(method => {
    const exists = knowledgeSocketIO && typeof knowledgeSocketIO[method as keyof WebSocketService] === 'function';
    console.log(`3. Method ${method} exists:`, exists);
    if (!exists) {
      console.error(`CRITICAL: Method ${method} is missing!`);
    }
  });
  
  // Test 4: List all available methods
  if (knowledgeSocketIO) {
    console.log('4. Available methods on knowledgeSocketIO:');
    const proto = Object.getPrototypeOf(knowledgeSocketIO);
    const methods = Object.getOwnPropertyNames(proto).filter(name => 
      typeof proto[name] === 'function' && name !== 'constructor'
    );
    methods.forEach(method => console.log(`   - ${method}`));
  }
  
  // Test 5: Try calling isConnected safely
  try {
    const connected = knowledgeSocketIO?.isConnected?.();
    console.log('5. isConnected() call result:', connected);
  } catch (error) {
    console.error('5. isConnected() call failed:', error);
  }
  
  // Test 6: Test deepconfService creation
  try {
    console.log('6. deepconfService exists:', !!deepconfService);
    console.log('6. createDeepConfService function exists:', typeof createDeepConfService === 'function');
    
    // Try creating a new service instance
    const testService = createDeepConfService();
    console.log('6. New service instance created successfully:', !!testService);
  } catch (error) {
    console.error('6. Service creation failed:', error);
  }
  
  console.log('=== DEEPCONF EMERGENCY DIAGNOSTIC END ===');
})();

// Export a test function that can be called manually
export function runDeepConfDiagnostic(): void {
  console.log('=== MANUAL DEEPCONF DIAGNOSTIC ===');
  
  // Try to call all the problematic methods
  try {
    if (knowledgeSocketIO && typeof knowledgeSocketIO.addMessageHandler === 'function') {
      console.log('✅ addMessageHandler is available');
      
      // Test adding a handler
      knowledgeSocketIO.addMessageHandler('test_event', (message) => {
        console.log('Test handler received:', message);
      });
      console.log('✅ addMessageHandler call succeeded');
      
    } else {
      console.error('❌ addMessageHandler is NOT available');
      console.log('knowledgeSocketIO:', knowledgeSocketIO);
      console.log('addMessageHandler type:', typeof knowledgeSocketIO?.addMessageHandler);
    }
    
  } catch (error) {
    console.error('❌ addMessageHandler test failed:', error);
  }
  
  try {
    if (knowledgeSocketIO && typeof knowledgeSocketIO.isConnected === 'function') {
      const connected = knowledgeSocketIO.isConnected();
      console.log('✅ isConnected call succeeded, result:', connected);
    } else {
      console.error('❌ isConnected is NOT available');
    }
  } catch (error) {
    console.error('❌ isConnected test failed:', error);
  }
  
  console.log('=== MANUAL DIAGNOSTIC END ===');
}

// Export everything for testing
export { knowledgeSocketIO, WebSocketService };