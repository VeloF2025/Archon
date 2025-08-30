# UI/UX Validation Agent

A comprehensive UI/UX validation agent that leverages the Browser MCP module to perform detailed analysis of web applications. This agent ensures high-quality user experiences through automated testing of visual design, accessibility, performance, and usability.

## 🌟 Features

### Core Validation Capabilities
- **Visual Design Consistency**: Colors, typography, spacing validation
- **Responsive Design Testing**: Multi-viewport and multi-device testing  
- **Accessibility Compliance**: WCAG 2.1 standards validation (A, AA, AAA levels)
- **Performance Analysis**: Core Web Vitals and load time metrics
- **Cross-Browser Compatibility**: Testing across Chromium, Firefox, WebKit
- **Form Usability**: Form validation, labeling, and interaction testing
- **Navigation Flow**: User journey and navigation pattern validation
- **Error State Handling**: Error message and loading state validation
- **Visual Regression Testing**: Screenshot comparison and change detection
- **Comprehensive Reporting**: Detailed reports with actionable recommendations

### Validation Modes

#### 🚀 Quick Scan (`quick_scan`)
- Fast validation focusing on critical issues
- Basic accessibility and visual checks
- Single viewport testing
- Ideal for: CI/CD pipelines, rapid feedback

#### 🔍 Comprehensive (`comprehensive`) 
- Full validation suite with all checks
- Multi-viewport and multi-browser testing
- Detailed accessibility audit
- Performance analysis with Core Web Vitals
- Ideal for: Release validation, thorough QA

#### 🎯 Focused (`focused`)
- Targeted validation of specific components
- Component-specific accessibility checks
- Detailed form and interaction testing
- Ideal for: Feature validation, component testing

#### 🔄 Continuous (`continuous`)
- Lightweight monitoring for ongoing validation
- Performance trend monitoring
- Critical functionality checks
- Ideal for: Production monitoring, performance tracking

## 🚀 Quick Start

### Installation

Ensure you have the required dependencies:

```bash
# Install Playwright for browser automation
pip install playwright>=1.40.0
playwright install

# Ensure MCP server is running
docker-compose up archon-mcp
```

### Basic Usage

```python
import asyncio
from src.agents.ui_ux_validator_agent import validate_ui_ux, ValidationMode

async def validate_website():
    # Quick validation scan
    result = await validate_ui_ux(
        target_url="https://example.com",
        mode=ValidationMode.QUICK_SCAN
    )
    
    print(f"Status: {result.pass_fail_status}")
    print(f"Issues: {sum(result.issues_count.values())}")
    
    # Show critical issues
    for issue in result.critical_issues:
        print(f"🚨 {issue.title}: {issue.description}")

# Run validation
asyncio.run(validate_website())
```

### Advanced Usage

```python
from src.agents.ui_ux_validator_agent import (
    UIUXValidatorAgent, 
    ValidationMode,
    ViewportConfig,
    BrowserType
)

async def comprehensive_validation():
    # Custom viewport configurations
    viewports = [
        ViewportConfig(name="Desktop", width=1920, height=1080, device_type="desktop"),
        ViewportConfig(name="Tablet", width=768, height=1024, device_type="tablet"),
        ViewportConfig(name="Mobile", width=375, height=667, device_type="mobile"),
    ]
    
    # Initialize agent
    agent = UIUXValidatorAgent()
    
    # Run comprehensive validation
    result = await agent.run_validation(
        target_url="https://example.com",
        validation_mode=ValidationMode.COMPREHENSIVE,
        viewports_to_test=viewports,
        browsers_to_test=[BrowserType.CHROMIUM, BrowserType.FIREFOX],
        wcag_level="AA",
        output_directory="./validation_results"
    )
    
    # Access detailed report
    if result.validation_report:
        report = result.validation_report
        print(f"Accessibility Score: {report.accessibility_score}/100")
        print(f"Performance Metrics: {report.performance_metrics}")
        
        # Show recommendations
        for rec in report.recommendations:
            print(f"💡 {rec}")

asyncio.run(comprehensive_validation())
```

## 📊 Validation Categories

### 1. Visual Design Consistency
- **Color scheme validation**: Consistent color usage across components
- **Typography analysis**: Font consistency, readability, hierarchy
- **Spacing and layout**: Consistent margins, padding, alignment
- **Brand compliance**: Logo placement, brand color accuracy

### 2. Responsive Design
- **Multi-viewport testing**: Desktop, tablet, mobile viewports
- **Breakpoint analysis**: CSS breakpoint effectiveness
- **Touch target sizing**: Minimum 44px touch targets for mobile
- **Content reflow**: Proper content adaptation across screen sizes
- **Horizontal scrolling detection**: Prevention of unwanted horizontal scroll

### 3. Accessibility (WCAG 2.1)

#### Level A Compliance
- Images with alt text
- Keyboard navigation support
- Focus indicators
- Basic color contrast

#### Level AA Compliance  
- Enhanced color contrast (4.5:1 for normal text)
- Resizable text up to 200%
- Audio/video captions
- Consistent navigation

#### Level AAA Compliance
- Superior color contrast (7:1 for normal text)
- Context-sensitive help
- Advanced keyboard shortcuts
- Enhanced error identification

### 4. Performance Analysis

#### Core Web Vitals
- **Largest Contentful Paint (LCP)**: Loading performance
- **First Input Delay (FID)**: Interactivity responsiveness  
- **Cumulative Layout Shift (CLS)**: Visual stability

#### Additional Metrics
- First Paint and First Contentful Paint
- Total page load time
- Network request analysis
- JavaScript heap usage
- Resource optimization opportunities

### 5. Form Usability
- **Label associations**: Proper input-label relationships
- **Required field indicators**: Visual indication of required fields
- **Validation messaging**: Clear, actionable error messages
- **Submission feedback**: Loading states and confirmation
- **Input accessibility**: ARIA labels and descriptions

### 6. Navigation and Flow
- **Navigation consistency**: Consistent navigation patterns
- **Breadcrumb implementation**: Clear user location indication
- **Link accessibility**: Descriptive link text
- **Focus management**: Logical tab order
- **Skip navigation**: Skip links for screen readers

## 🛠️ Browser MCP Integration

The agent uses the Browser MCP module to perform all browser automation:

### Available Browser Tools
- `navigate_to`: URL navigation with wait conditions
- `screenshot`: Full page and element screenshots
- `click`: Element interaction and clicking
- `fill`: Form field input and testing
- `get_text`: Text content extraction
- `wait_for_element`: Element state waiting
- `evaluate_script`: JavaScript execution
- `get_accessibility_tree`: Accessibility tree extraction
- `visual_compare`: Screenshot comparison for regression testing
- `get_performance_metrics`: Performance data collection

### Browser Support
- **Chromium**: Default, best for automation and CI/CD
- **Firefox**: Alternative engine for cross-browser testing
- **WebKit**: Safari engine for Apple ecosystem testing

## 📋 Validation Report Structure

```json
{
  "url": "https://example.com",
  "validation_mode": "comprehensive",
  "timestamp": "2024-08-30T10:30:00Z",
  "summary": {
    "total_issues": 15,
    "critical_issues": 2,
    "high_issues": 5,
    "medium_issues": 6,
    "low_issues": 2
  },
  "issues": [
    {
      "category": "accessibility",
      "severity": "critical",
      "title": "Missing Alt Text on Images",
      "description": "5 images found without alt text attributes",
      "element_selector": "img:not([alt])",
      "wcag_criteria": "1.1.1",
      "recommendations": [
        "Add descriptive alt text to all images",
        "Use empty alt='' for decorative images"
      ]
    }
  ],
  "screenshots": {
    "desktop": "base64_screenshot_data",
    "tablet": "base64_screenshot_data", 
    "mobile": "base64_screenshot_data"
  },
  "performance_metrics": {
    "lcp": 2.1,
    "fid": 120,
    "cls": 0.05,
    "total_load_time": 1800
  },
  "accessibility_score": 78.5,
  "recommendations": [
    "🚨 URGENT: Address 2 critical accessibility issues immediately",
    "⚠️ HIGH PRIORITY: Fix 5 high-priority accessibility issues",
    "💡 IMPROVE: Optimize images to improve load performance"
  ]
}
```

## 🎯 Use Cases

### 1. Development Workflow Integration
```python
# In your development pipeline
async def validate_feature_branch():
    result = await validate_ui_ux(
        target_url="https://staging.example.com",
        mode=ValidationMode.QUICK_SCAN
    )
    
    # Block deployment if critical issues found
    if result.critical_issues:
        raise Exception("Critical UI/UX issues found - blocking deployment")
    
    return result
```

### 2. Release Validation
```python
# Before production release
async def release_validation():
    result = await validate_ui_ux(
        target_url="https://staging.example.com",
        mode=ValidationMode.COMPREHENSIVE
    )
    
    # Generate comprehensive report
    if result.validation_report:
        # Save detailed report for review
        with open("release_validation_report.json", "w") as f:
            f.write(result.validation_report.json(indent=2))
    
    return result
```

### 3. Component Testing
```python
# Test specific UI components
async def validate_checkout_form():
    agent = UIUXValidatorAgent()
    
    result = await agent.run_validation(
        target_url="https://example.com/checkout",
        validation_mode=ValidationMode.FOCUSED,
        focus_component="checkout form"
    )
    
    # Focus on form-specific issues
    form_issues = [
        issue for issue in result.validation_report.issues
        if "form" in issue.category.lower()
    ]
    
    return form_issues
```

### 4. Performance Monitoring
```python
# Continuous performance monitoring
async def monitor_performance():
    result = await validate_ui_ux(
        target_url="https://production.example.com",
        mode=ValidationMode.CONTINUOUS
    )
    
    # Check performance thresholds
    if result.validation_report.performance_metrics:
        metrics = result.validation_report.performance_metrics
        if metrics.get("total_load_time", 0) > 3000:  # 3 second threshold
            # Alert slow performance
            send_performance_alert(metrics)
    
    return result
```

## 🔧 Configuration Options

### Viewport Configurations
```python
# Custom viewport testing
custom_viewports = [
    ViewportConfig(name="Large Desktop", width=2560, height=1440, device_type="desktop"),
    ViewportConfig(name="Standard Laptop", width=1366, height=768, device_type="desktop"),
    ViewportConfig(name="iPad", width=1024, height=768, device_type="tablet"),
    ViewportConfig(name="iPad Portrait", width=768, height=1024, device_type="tablet"),
    ViewportConfig(name="iPhone 12", width=390, height=844, device_type="mobile"),
    ViewportConfig(name="Samsung Galaxy", width=360, height=800, device_type="mobile"),
]
```

### WCAG Level Configuration
```python
# Different WCAG compliance levels
wcag_levels = {
    "A": "Basic accessibility requirements",
    "AA": "Standard compliance level (recommended)",  
    "AAA": "Highest accessibility standard"
}
```

### Browser Configuration
```python
# Multi-browser testing
browsers = [
    BrowserType.CHROMIUM,  # Google Chrome/Edge
    BrowserType.FIREFOX,   # Mozilla Firefox  
    BrowserType.WEBKIT,    # Safari
]
```

## 📁 Example Scripts

The `/examples` directory contains comprehensive examples:

- `ui_ux_validation_example.py`: Complete usage examples
- Run examples: `python examples/ui_ux_validation_example.py --examples`

## 🚨 Error Handling

The agent provides comprehensive error handling:

```python
try:
    result = await validate_ui_ux(target_url="https://invalid-url.com")
except Exception as e:
    if "navigation timeout" in str(e).lower():
        print("Website is too slow or unavailable")
    elif "element not found" in str(e).lower():
        print("Page structure has changed")
    else:
        print(f"Validation error: {e}")
```

## 🔗 Integration with Archon System

The agent integrates seamlessly with the Archon ecosystem:

- **MCP Protocol**: Uses standardized MCP tools for browser automation
- **Agent Framework**: Built on the Archon BaseAgent pattern
- **Service Discovery**: Automatically discovers MCP server endpoints
- **Dependency Injection**: Supports custom dependencies and configurations
- **Logging**: Integrated with Archon logging system

## 📈 Performance Considerations

### Optimization Tips
- Use `ValidationMode.QUICK_SCAN` for frequent checks
- Limit viewport testing for faster execution
- Use headless mode for better performance
- Implement caching for repeated validations
- Run browser cleanup after validation sessions

### Resource Management
```python
# Proper resource cleanup
async def validate_with_cleanup():
    agent = UIUXValidatorAgent()
    try:
        result = await agent.run_validation(url)
        return result
    finally:
        # Cleanup browser resources
        async with await get_mcp_client() as mcp:
            await mcp.call_tool("browser_cleanup")
```

## 🛡️ Security Considerations

- Browsers run with security flags enabled
- HTTPS errors ignored for testing environments
- No sensitive data logged in screenshots  
- Automatic cleanup prevents persistent processes
- Input validation on all parameters

## 🐛 Troubleshooting

### Common Issues

**Playwright Not Available**
```
Error: Playwright is not installed
Solution: pip install playwright && playwright install
```

**MCP Server Connection Failed**
```
Error: Failed to connect to MCP server
Solution: Ensure MCP server is running on port 8051
```

**Navigation Timeout**
```
Error: Navigation timeout exceeded
Solution: Increase timeout or check network connectivity
```

**Element Not Found**
```
Error: Element not found: .selector
Solution: Verify selector exists and wait for dynamic content
```

### Debug Mode

Enable debug mode for troubleshooting:
```python
# Run with visible browser for debugging
result = await agent.run_validation(
    target_url=url,
    headless=False  # Shows browser window
)
```

## 📚 API Reference

### Main Classes

#### `UIUXValidatorAgent`
Main agent class for UI/UX validation.

**Methods:**
- `run_validation(target_url, validation_mode, **kwargs)`: Run validation
- `_parse_validation_results(raw_results)`: Parse results into report
- `_calculate_accessibility_score(issues)`: Calculate accessibility score
- `_generate_recommendations(issues)`: Generate actionable recommendations

#### `ValidationReport` 
Comprehensive validation report data structure.

**Properties:**
- `url`: Target URL validated
- `validation_mode`: Mode used for validation
- `issues`: List of ValidationIssue objects
- `screenshots`: Viewport screenshots as base64
- `performance_metrics`: Performance data
- `accessibility_score`: Calculated accessibility score
- `recommendations`: Actionable improvement suggestions

#### `ValidationIssue`
Individual validation issue data structure.

**Properties:**
- `category`: Issue category (accessibility, performance, etc.)
- `severity`: Issue severity level (critical, high, medium, low, info)
- `title`: Short issue description
- `description`: Detailed issue explanation
- `recommendations`: List of fix suggestions
- `wcag_criteria`: Relevant WCAG guideline
- `element_selector`: CSS selector for affected element

### Convenience Functions

#### `validate_ui_ux(target_url, mode, output_dir)`
Quick validation function for simple use cases.

**Parameters:**
- `target_url` (str): URL to validate
- `mode` (ValidationMode): Validation mode 
- `output_dir` (str, optional): Output directory for results

**Returns:**
- `UIUXValidatorOutput`: Validation results

## 🤝 Contributing

To extend the UI/UX Validation Agent:

1. **Add new validation tools**: Extend browser tools in `_add_browser_tools()`
2. **Implement new checks**: Add validation logic in tool functions
3. **Enhance reporting**: Extend `ValidationReport` model
4. **Add new modes**: Implement additional validation modes

### Example Extension
```python
@agent.tool
async def validate_color_contrast() -> str:
    """Validate color contrast ratios across the page."""
    # Implementation for detailed contrast analysis
    pass
```

---

## 📄 License

This agent is part of the Archon system and follows the project's licensing terms.

---

**Need help?** Check the examples directory or refer to the Browser MCP documentation for additional details on browser automation capabilities.