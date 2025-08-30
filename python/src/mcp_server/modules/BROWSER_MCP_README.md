# Browser MCP Module Documentation

The Browser MCP module provides comprehensive Playwright browser automation capabilities for the Archon system. It enables AI agents to perform web automation, visual testing, and UI/UX validation through MCP tools.

## Features

### Core Browser Automation
- **navigate_to**: Navigate to URLs with configurable wait conditions
- **click**: Click on elements with selector targeting
- **fill**: Fill form fields with text input
- **get_text**: Extract text content from elements
- **wait_for_element**: Wait for elements to appear or change state

### Visual Testing & Screenshots
- **screenshot**: Capture full page or element-specific screenshots
- **visual_compare**: Compare screenshots for visual regression testing

### Advanced Capabilities
- **evaluate_script**: Execute JavaScript in the browser context
- **get_accessibility_tree**: Extract accessibility information
- **get_performance_metrics**: Collect page performance data
- **browser_cleanup**: Clean up browser resources

## Supported Browsers

- **Chromium** (default) - Best for CI/CD and automation
- **Firefox** - Alternative engine for cross-browser testing
- **WebKit** - Safari engine for Apple ecosystem testing

## Viewport Presets

The module includes standard viewport configurations:
- **Desktop**: 1920x1080
- **Laptop**: 1366x768  
- **Tablet**: 768x1024
- **Mobile**: 375x667
- **iPhone**: 375x812
- **Android**: 360x640

## Installation Requirements

```bash
# Install Playwright dependency
pip install playwright>=1.40.0

# Install browser engines
playwright install
```

## Usage Examples

### Basic Navigation
```json
{
  "tool": "navigate_to",
  "params": {
    "url": "https://example.com",
    "browser_type": "chromium",
    "headless": true,
    "wait_until": "domcontentloaded"
  }
}
```

### Taking Screenshots
```json
{
  "tool": "screenshot",
  "params": {
    "selector": ".main-content",
    "format": "png",
    "quality": 90,
    "path": "/tmp/screenshot.png"
  }
}
```

### Form Automation
```json
{
  "tool": "fill",
  "params": {
    "selector": "#email-input",
    "text": "user@example.com",
    "clear_first": true
  }
}
```

### Performance Analysis
```json
{
  "tool": "get_performance_metrics",
  "params": {}
}
```

### Accessibility Testing
```json
{
  "tool": "get_accessibility_tree",
  "params": {
    "selector": "main",
    "interesting_only": true
  }
}
```

## Error Handling

All tools return consistent JSON responses with:
- `success`: Boolean indicating operation success
- `error`: Error message if operation failed
- Additional data specific to each tool

Example error response:
```json
{
  "success": false,
  "error": "Element not found: .missing-selector",
  "selector": ".missing-selector"
}
```

## Browser Management

- Browsers are automatically launched on first use
- Contexts are reused for efficiency
- Resources are automatically cleaned up
- Thread-safe operation with async locks

## Performance Considerations

- Browsers run in headless mode by default for better performance
- Single browser context is reused across operations
- Automatic cleanup prevents memory leaks
- Timeouts are configurable to prevent hanging operations

## Integration with UI/UX Agents

The Browser MCP module is specifically designed for UI/UX agents to:

1. **Visual Validation**: Take screenshots for design review
2. **Accessibility Auditing**: Extract accessibility tree data
3. **Performance Monitoring**: Measure page load times and metrics
4. **Cross-browser Testing**: Test across different browser engines
5. **Responsive Design Testing**: Test various viewport sizes
6. **Visual Regression Testing**: Compare screenshots over time

## Security Considerations

- Browsers run with security flags enabled
- HTTPS errors are ignored for testing environments
- No sensitive data is logged in screenshots
- Cleanup ensures no persistent browser processes

## Troubleshooting

### Common Issues

**Playwright Not Available**
```
⚠️ Browser tools module not available (Playwright not installed)
```
Solution: Install Playwright with `pip install playwright && playwright install`

**Element Not Found**
```json
{"success": false, "error": "Element not found: .selector"}
```
Solution: Verify selector exists and wait for element to load

**Navigation Timeout**
```json
{"success": false, "error": "Navigation timeout: Timeout 30000ms exceeded"}
```
Solution: Increase timeout or check network connectivity

### Debug Mode

For debugging, set `headless=false` to see browser actions:
```json
{
  "tool": "navigate_to",
  "params": {
    "url": "https://example.com",
    "headless": false
  }
}
```

## API Reference

### navigate_to(url, wait_until, timeout, browser_type, headless)
Navigate to a URL and wait for page load.

**Parameters:**
- `url` (str): URL to navigate to
- `wait_until` (str): load, domcontentloaded, networkidle (default: domcontentloaded)
- `timeout` (int): Maximum wait time in ms (default: 30000)
- `browser_type` (str): chromium, firefox, webkit (default: chromium)
- `headless` (bool): Run headless (default: true)

### screenshot(selector, full_page, quality, format, path)
Take a screenshot of the page or element.

**Parameters:**
- `selector` (str, optional): CSS selector for element
- `full_page` (bool): Capture full scrollable page (default: false)
- `quality` (int): JPEG quality 0-100 (default: 90)
- `format` (str): png or jpeg (default: png)
- `path` (str, optional): File path to save screenshot

### click(selector, timeout, force, wait_for_navigation)
Click on an element.

**Parameters:**
- `selector` (str): CSS selector for element
- `timeout` (int): Maximum wait time in ms (default: 30000)
- `force` (bool): Force click if not actionable (default: false)
- `wait_for_navigation` (bool): Wait for navigation after click (default: false)

### fill(selector, text, timeout, clear_first)
Fill a form field with text.

**Parameters:**
- `selector` (str): CSS selector for input field
- `text` (str): Text to enter
- `timeout` (int): Maximum wait time in ms (default: 30000)
- `clear_first` (bool): Clear existing text first (default: true)

### get_text(selector, timeout, inner_text)
Extract text from an element.

**Parameters:**
- `selector` (str): CSS selector for element
- `timeout` (int): Maximum wait time in ms (default: 30000)
- `inner_text` (bool): Use innerText vs textContent (default: true)

### wait_for_element(selector, state, timeout)
Wait for element to reach specific state.

**Parameters:**
- `selector` (str): CSS selector for element
- `state` (str): visible, hidden, attached, detached (default: visible)
- `timeout` (int): Maximum wait time in ms (default: 30000)

### evaluate_script(script, selector)
Execute JavaScript in browser.

**Parameters:**
- `script` (str): JavaScript code to execute
- `selector` (str, optional): CSS selector to execute on element

### get_accessibility_tree(selector, interesting_only)
Get accessibility information.

**Parameters:**
- `selector` (str, optional): CSS selector for element
- `interesting_only` (bool): Only nodes with accessibility info (default: true)

### visual_compare(baseline_path, threshold, max_diff_pixels)
Compare screenshot with baseline.

**Parameters:**
- `baseline_path` (str): Path to baseline image
- `threshold` (float): Pixel difference threshold 0.0-1.0 (default: 0.2)
- `max_diff_pixels` (int, optional): Max different pixels allowed

### get_performance_metrics()
Get page performance data including Core Web Vitals.

**Returns:**
- Navigation timing
- Paint metrics
- Memory usage
- Network request stats

### browser_cleanup()
Clean up browser resources and close all instances.

---

This module enables powerful browser automation capabilities for AI agents, supporting everything from basic navigation to advanced visual testing and accessibility auditing.