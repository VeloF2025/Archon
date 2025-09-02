const { Stagehand } = require('@browserbase/stagehand');

async function testSocketIOConnection() {
  const stagehand = new Stagehand({
    env: 'LOCAL',
    verbose: 1,
    debugDom: true,
    headless: false
  });
  
  await stagehand.init();
  
  try {
    console.log('Navigating to localhost:3737...');
    await stagehand.page.goto('http://localhost:3737', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    console.log('Opening developer console...');
    await stagehand.page.evaluate(() => {
      console.log('=== SOCKET.IO MONITORING STARTED ===');
    });
    
    // Capture console messages
    const consoleLogs = [];
    stagehand.page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push({
        type: msg.type(),
        text: text,
        timestamp: new Date().toISOString()
      });
      console.log(`[${msg.type().toUpperCase()}] ${text}`);
    });
    
    // Wait for initial page load and Socket.IO connection
    console.log('Waiting for initial connection...');
    await stagehand.page.waitForTimeout(5000);
    
    // Check if Socket.IO is connected
    const socketStatus = await stagehand.page.evaluate(() => {
      return {
        socketIOAvailable: typeof window.io !== 'undefined',
        socketConnected: window.socket ? window.socket.connected : false,
        socketId: window.socket ? window.socket.id : null
      };
    });
    
    console.log('Socket Status:', JSON.stringify(socketStatus, null, 2));
    
    // Monitor for 30 seconds for any timeout errors
    console.log('Monitoring for Socket.IO errors for 30 seconds...');
    await stagehand.page.waitForTimeout(30000);
    
    // Try to navigate to knowledge base page to test interaction
    console.log('Testing knowledge base page...');
    try {
      await stagehand.page.click('a[href*="knowledge"], button:has-text("Knowledge"), nav a:has-text("Knowledge")');
    } catch (e) {
      console.log('Knowledge link not found, trying alternative navigation...');
    }
    
    await stagehand.page.waitForTimeout(3000);
    
    // Final status check
    const finalSocketStatus = await stagehand.page.evaluate(() => {
      return {
        socketConnected: window.socket ? window.socket.connected : false,
        socketId: window.socket ? window.socket.id : null,
        currentUrl: window.location.href
      };
    });
    
    console.log('Final Socket Status:', JSON.stringify(finalSocketStatus, null, 2));
    
    // Filter and analyze console logs for Socket.IO related messages
    const socketLogs = consoleLogs.filter(log => 
      log.text.toLowerCase().includes('socket') || 
      log.text.toLowerCase().includes('timeout') ||
      log.text.toLowerCase().includes('disconnect') ||
      log.text.toLowerCase().includes('connect')
    );
    
    console.log('=== SOCKET.IO RELATED LOGS ===');
    socketLogs.forEach(log => {
      console.log(`[${log.timestamp}] [${log.type}] ${log.text}`);
    });
    
    // Check for timeout errors specifically
    const timeoutErrors = consoleLogs.filter(log => 
      log.text.toLowerCase().includes('timeout') && 
      log.type === 'error'
    );
    
    console.log('=== TIMEOUT ERRORS DETECTED ===');
    console.log(`Found ${timeoutErrors.length} timeout errors`);
    timeoutErrors.forEach(error => {
      console.log(`[${error.timestamp}] ${error.text}`);
    });
    
    return {
      socketStatus: finalSocketStatus,
      totalLogs: consoleLogs.length,
      socketLogs: socketLogs.length,
      timeoutErrors: timeoutErrors.length,
      timeoutErrorDetails: timeoutErrors
    };
    
  } catch (error) {
    console.error('Test failed:', error);
    return { error: error.message };
  } finally {
    await stagehand.close();
  }
}

testSocketIOConnection().then(result => {
  console.log('=== TEST RESULTS ===');
  console.log(JSON.stringify(result, null, 2));
}).catch(console.error);