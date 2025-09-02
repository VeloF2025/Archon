"""
UI/UX Validation Agent for Archon System

This agent provides comprehensive UI/UX validation capabilities using the Browser MCP module.
It performs visual testing, accessibility auditing, performance analysis, and responsive design
validation to ensure high-quality user experiences.

Features:
- Visual design consistency validation
- Responsive design testing across multiple viewports
- Accessibility compliance (WCAG standards)
- Performance metrics analysis (Core Web Vitals)
- Cross-browser compatibility testing
- Form usability validation
- Navigation flow testing
- Error state handling validation
- Visual regression testing
- Comprehensive reporting with screenshots
"""

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .base_agent import BaseAgent, ArchonDependencies, BaseAgentOutput
from .mcp_client import MCPClient, get_mcp_client

logger = logging.getLogger(__name__)


class ValidationMode(str, Enum):
    """UI/UX validation modes"""
    QUICK_SCAN = "quick_scan"           # Basic checks only
    COMPREHENSIVE = "comprehensive"      # Full validation suite
    FOCUSED = "focused"                 # Specific component validation
    CONTINUOUS = "continuous"           # Periodic monitoring checks


class ValidationSeverity(str, Enum):
    """Issue severity levels"""
    CRITICAL = "critical"    # Blocking issues
    HIGH = "high"           # Important fixes needed
    MEDIUM = "medium"       # Should be addressed
    LOW = "low"            # Minor improvements
    INFO = "info"          # Informational only


class BrowserType(str, Enum):
    """Supported browser types"""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class ViewportConfig(BaseModel):
    """Viewport configuration for responsive testing"""
    name: str
    width: int
    height: int
    device_type: str  # desktop, tablet, mobile


@dataclass
class UIUXValidationDependencies(ArchonDependencies):
    """Dependencies for UI/UX validation agent"""
    target_url: str
    validation_mode: ValidationMode = ValidationMode.COMPREHENSIVE
    browsers_to_test: List[BrowserType] = field(default_factory=lambda: [BrowserType.CHROMIUM])
    viewports_to_test: List[ViewportConfig] = field(default_factory=list)
    baseline_path: Optional[str] = None
    output_directory: Optional[str] = None
    wcag_level: str = "AA"  # A, AA, AAA
    progress_callback: Optional[callable] = None


class ValidationIssue(BaseModel):
    """Represents a single validation issue"""
    category: str
    severity: ValidationSeverity
    title: str
    description: str
    element_selector: Optional[str] = None
    screenshot_path: Optional[str] = None
    recommendations: List[str] = []
    wcag_criteria: Optional[str] = None
    browser: Optional[str] = None
    viewport: Optional[str] = None


class ValidationReport(BaseModel):
    """Comprehensive validation report"""
    url: str
    validation_mode: ValidationMode
    timestamp: datetime
    summary: Dict[str, Any]
    issues: List[ValidationIssue]
    screenshots: Dict[str, str]  # viewport -> base64 screenshot
    performance_metrics: Dict[str, Any]
    accessibility_score: Optional[float] = None
    recommendations: List[str]
    browsers_tested: List[str]
    viewports_tested: List[str]


class UIUXValidatorOutput(BaseAgentOutput):
    """Output model for UI/UX validation agent"""
    validation_report: Optional[ValidationReport] = None
    issues_count: Dict[str, int] = Field(default_factory=dict)
    pass_fail_status: str = "unknown"
    critical_issues: List[ValidationIssue] = Field(default_factory=list)


class UIUXValidatorAgent(BaseAgent[UIUXValidationDependencies, UIUXValidatorOutput]):
    """
    Comprehensive UI/UX validation agent using Browser MCP.
    
    This agent performs extensive UI/UX validation including visual design,
    accessibility, performance, and responsive design testing.
    """

    def __init__(self, **kwargs):
        super().__init__(
            model="openai:gpt-4o",
            name="UIUXValidator",
            **kwargs
        )
        
        # Default viewport configurations
        self.default_viewports = [
            ViewportConfig(name="Desktop", width=1920, height=1080, device_type="desktop"),
            ViewportConfig(name="Laptop", width=1366, height=768, device_type="desktop"), 
            ViewportConfig(name="Tablet", width=768, height=1024, device_type="tablet"),
            ViewportConfig(name="Mobile", width=375, height=667, device_type="mobile"),
            ViewportConfig(name="iPhone", width=375, height=812, device_type="mobile"),
        ]

    def _create_agent(self, **kwargs) -> Agent:
        """Create the PydanticAI agent with UI/UX validation tools."""
        agent = Agent(
            model=self.model,
            result_type=UIUXValidatorOutput,
            system_prompt=self.get_system_prompt(),
            **kwargs
        )
        
        # Add MCP browser tools
        self._add_browser_tools(agent)
        
        return agent

    def get_system_prompt(self) -> str:
        """Get the system prompt for UI/UX validation."""
        return """
        You are a comprehensive UI/UX validation agent that performs detailed analysis
        of web applications to ensure high-quality user experiences.

        Your responsibilities include:
        1. Visual design consistency validation (colors, typography, spacing)
        2. Responsive design testing across multiple viewports
        3. Accessibility compliance validation (WCAG standards)
        4. Performance metrics analysis (Core Web Vitals)
        5. Cross-browser compatibility testing
        6. Form usability and validation testing
        7. Navigation flow validation
        8. Error state and loading state validation
        9. Visual regression testing
        10. Comprehensive reporting with actionable recommendations

        Use the provided browser automation tools to:
        - Navigate to the target URL
        - Take screenshots across different viewports and browsers
        - Extract accessibility tree information
        - Measure performance metrics
        - Test form interactions and navigation flows
        - Analyze visual consistency and design patterns

        Always provide detailed, actionable feedback with specific recommendations
        for improvement. Include severity levels and WCAG criteria where applicable.
        Generate comprehensive reports with evidence (screenshots) to support findings.
        
        Focus on real-world usability issues that impact user experience.
        Prioritize accessibility and performance as critical quality gates.
        """

    def _add_browser_tools(self, agent: Agent) -> None:
        """Add browser automation tools to the agent."""
        
        @agent.tool
        async def navigate_and_capture(
            target_url: str, 
            browser_type: str = "chromium",
            viewport_name: str = "Desktop"
        ) -> str:
            """Navigate to URL and capture initial page state with screenshot."""
            try:
                async with await get_mcp_client() as mcp:
                    # Navigate to the target URL
                    nav_result = await mcp.call_tool(
                        "navigate_to",
                        url=target_url,
                        browser_type=browser_type,
                        headless=True,
                        wait_until="networkidle"
                    )
                    
                    # Take screenshot for visual validation
                    screenshot_result = await mcp.call_tool(
                        "screenshot",
                        full_page=True,
                        format="png",
                        quality=90
                    )
                    
                    return json.dumps({
                        "navigation": nav_result,
                        "screenshot": screenshot_result,
                        "viewport": viewport_name,
                        "browser": browser_type
                    })
            except Exception as e:
                logger.error(f"Navigation and capture error: {e}")
                return json.dumps({"error": str(e), "success": False})

        @agent.tool
        async def perform_accessibility_audit(selector: str = "body") -> str:
            """Perform comprehensive accessibility audit using accessibility tree."""
            try:
                async with await get_mcp_client() as mcp:
                    # Get accessibility tree
                    a11y_result = await mcp.call_tool(
                        "get_accessibility_tree",
                        selector=selector,
                        interesting_only=True
                    )
                    
                    # Check for common accessibility issues using JavaScript
                    a11y_checks = await mcp.call_tool(
                        "evaluate_script",
                        script="""
                        (() => {
                            const issues = [];
                            
                            // Check for images without alt text
                            const imgsWithoutAlt = document.querySelectorAll('img:not([alt])');
                            if (imgsWithoutAlt.length > 0) {
                                issues.push({
                                    type: 'missing_alt_text',
                                    count: imgsWithoutAlt.length,
                                    severity: 'critical',
                                    wcag: '1.1.1'
                                });
                            }
                            
                            // Check for form inputs without labels
                            const inputsWithoutLabels = document.querySelectorAll('input:not([aria-label]):not([aria-labelledby])');
                            const unlabeledInputs = Array.from(inputsWithoutLabels).filter(input => {
                                const id = input.id;
                                return !id || !document.querySelector(`label[for="${id}"]`);
                            });
                            if (unlabeledInputs.length > 0) {
                                issues.push({
                                    type: 'unlabeled_form_controls',
                                    count: unlabeledInputs.length,
                                    severity: 'critical',
                                    wcag: '3.3.2'
                                });
                            }
                            
                            // Check for missing headings hierarchy
                            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
                            if (headings.length === 0) {
                                issues.push({
                                    type: 'no_headings',
                                    count: 0,
                                    severity: 'high',
                                    wcag: '2.4.6'
                                });
                            }
                            
                            // Check for insufficient color contrast (simplified)
                            const colorContrastElements = document.querySelectorAll('p, span, div, a, button, label');
                            let lowContrastCount = 0;
                            
                            // Simplified contrast check - in reality would need more sophisticated analysis
                            Array.from(colorContrastElements).forEach(el => {
                                const style = getComputedStyle(el);
                                const color = style.color;
                                const bg = style.backgroundColor;
                                
                                // Basic check for very light colors that might have contrast issues
                                if (color.includes('rgb(200,') || color.includes('rgb(220,') || color.includes('rgb(240,')) {
                                    lowContrastCount++;
                                }
                            });
                            
                            if (lowContrastCount > 0) {
                                issues.push({
                                    type: 'potential_color_contrast',
                                    count: lowContrastCount,
                                    severity: 'medium',
                                    wcag: '1.4.3'
                                });
                            }
                            
                            // Check for keyboard navigation support
                            const interactiveElements = document.querySelectorAll('a, button, input, select, textarea, [tabindex]');
                            const nonKeyboardAccessible = Array.from(interactiveElements).filter(el => {
                                return el.tabIndex === -1 && !el.hasAttribute('aria-hidden');
                            });
                            
                            if (nonKeyboardAccessible.length > 0) {
                                issues.push({
                                    type: 'keyboard_inaccessible',
                                    count: nonKeyboardAccessible.length,
                                    severity: 'high',
                                    wcag: '2.1.1'
                                });
                            }
                            
                            return {
                                issues,
                                total_issues: issues.length,
                                elements_checked: {
                                    images: document.querySelectorAll('img').length,
                                    form_controls: document.querySelectorAll('input, select, textarea').length,
                                    interactive: interactiveElements.length,
                                    headings: headings.length
                                }
                            };
                        })()
                        """
                    )
                    
                    return json.dumps({
                        "accessibility_tree": a11y_result,
                        "accessibility_checks": a11y_checks
                    })
            except Exception as e:
                logger.error(f"Accessibility audit error: {e}")
                return json.dumps({"error": str(e), "success": False})

        @agent.tool
        async def analyze_performance_metrics() -> str:
            """Analyze page performance including Core Web Vitals."""
            try:
                async with await get_mcp_client() as mcp:
                    # Get performance metrics
                    perf_result = await mcp.call_tool("get_performance_metrics")
                    
                    # Get additional performance insights via JavaScript
                    additional_metrics = await mcp.call_tool(
                        "evaluate_script",
                        script="""
                        (() => {
                            const metrics = {};
                            
                            // Core Web Vitals calculation
                            const observer = new PerformanceObserver((list) => {
                                list.getEntries().forEach((entry) => {
                                    if (entry.entryType === 'largest-contentful-paint') {
                                        metrics.lcp = entry.startTime;
                                    }
                                    if (entry.entryType === 'first-input') {
                                        metrics.fid = entry.processingStart - entry.startTime;
                                    }
                                });
                            });
                            
                            // Layout shift calculation (simplified)
                            let cumulativeLayoutShift = 0;
                            const clsObserver = new PerformanceObserver((list) => {
                                list.getEntries().forEach((entry) => {
                                    if (!entry.hadRecentInput) {
                                        cumulativeLayoutShift += entry.value;
                                    }
                                });
                            });
                            
                            try {
                                observer.observe({entryTypes: ['largest-contentful-paint', 'first-input']});
                                clsObserver.observe({entryTypes: ['layout-shift']});
                            } catch (e) {
                                // Fallback for browsers that don't support these APIs
                            }
                            
                            // Calculate performance score
                            const perfData = performance.getEntriesByType('navigation')[0];
                            const loadTime = perfData ? perfData.loadEventEnd - perfData.loadEventStart : 0;
                            
                            return {
                                cls: cumulativeLayoutShift,
                                load_time: loadTime,
                                dom_nodes: document.querySelectorAll('*').length,
                                images: document.querySelectorAll('img').length,
                                external_scripts: document.querySelectorAll('script[src]').length,
                                external_stylesheets: document.querySelectorAll('link[rel="stylesheet"]').length,
                                performance_score: loadTime < 1500 ? 'good' : loadTime < 2500 ? 'needs_improvement' : 'poor'
                            };
                        })()
                        """
                    )
                    
                    return json.dumps({
                        "performance_metrics": perf_result,
                        "additional_metrics": additional_metrics
                    })
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                return json.dumps({"error": str(e), "success": False})

        @agent.tool  
        async def test_responsive_design(viewports: List[dict]) -> str:
            """Test responsive design across multiple viewport sizes."""
            results = []
            
            try:
                async with await get_mcp_client() as mcp:
                    for viewport in viewports:
                        # Navigate with specific viewport
                        nav_result = await mcp.call_tool(
                            "navigate_to",
                            url=target_url,  # This would need to be passed from context
                            browser_type="chromium",
                            headless=True
                        )
                        
                        # Take screenshot for this viewport
                        screenshot_result = await mcp.call_tool(
                            "screenshot",
                            full_page=True,
                            format="png"
                        )
                        
                        # Check for responsive design issues
                        responsive_check = await mcp.call_tool(
                            "evaluate_script",
                            script=f"""
                            (() => {{
                                const viewport = {{ width: {viewport['width']}, height: {viewport['height']} }};
                                const issues = [];
                                
                                // Check for horizontal scrollbars
                                if (document.documentElement.scrollWidth > viewport.width) {{
                                    issues.push({{
                                        type: 'horizontal_scroll',
                                        severity: 'medium',
                                        description: 'Horizontal scrollbar present'
                                    }});
                                }}
                                
                                // Check for fixed-width elements that might cause issues
                                const fixedWidthElements = document.querySelectorAll('[style*="width:"]');
                                const problematicElements = Array.from(fixedWidthElements).filter(el => {{
                                    const style = getComputedStyle(el);
                                    const width = parseInt(style.width);
                                    return width > viewport.width;
                                }});
                                
                                if (problematicElements.length > 0) {{
                                    issues.push({{
                                        type: 'fixed_width_overflow',
                                        count: problematicElements.length,
                                        severity: 'high',
                                        description: 'Elements with fixed width larger than viewport'
                                    }});
                                }}
                                
                                // Check for text that might be too small on mobile
                                if (viewport.width < 768) {{
                                    const smallTextElements = document.querySelectorAll('p, span, div');
                                    let smallTextCount = 0;
                                    
                                    Array.from(smallTextElements).forEach(el => {{
                                        const style = getComputedStyle(el);
                                        const fontSize = parseInt(style.fontSize);
                                        if (fontSize < 14) {{
                                            smallTextCount++;
                                        }}
                                    }});
                                    
                                    if (smallTextCount > 0) {{
                                        issues.push({{
                                            type: 'small_text_mobile',
                                            count: smallTextCount,
                                            severity: 'medium',
                                            description: 'Text smaller than 14px on mobile'
                                        }});
                                    }}
                                }}
                                
                                return {{
                                    viewport: viewport,
                                    issues: issues,
                                    page_width: document.documentElement.scrollWidth,
                                    page_height: document.documentElement.scrollHeight
                                }};
                            }})()
                            """
                        )
                        
                        results.append({
                            "viewport": viewport,
                            "navigation": nav_result,
                            "screenshot": screenshot_result,
                            "responsive_check": responsive_check
                        })
                        
                        # Small delay to prevent overwhelming the browser
                        await asyncio.sleep(0.5)
                
                return json.dumps({"responsive_test_results": results})
                
            except Exception as e:
                logger.error(f"Responsive design test error: {e}")
                return json.dumps({"error": str(e), "success": False})

        @agent.tool
        async def test_form_usability(form_selector: str = "form") -> str:
            """Test form usability including validation and error handling."""
            try:
                async with await get_mcp_client() as mcp:
                    # Analyze form structure and usability
                    form_analysis = await mcp.call_tool(
                        "evaluate_script",
                        script=f"""
                        (() => {{
                            const forms = document.querySelectorAll('{form_selector}');
                            const analysis = [];
                            
                            forms.forEach((form, index) => {{
                                const formAnalysis = {{
                                    form_index: index,
                                    issues: []
                                }};
                                
                                // Check for required fields without visual indicators
                                const requiredInputs = form.querySelectorAll('input[required], select[required], textarea[required]');
                                const missingIndicators = Array.from(requiredInputs).filter(input => {{
                                    const label = form.querySelector(`label[for="${{input.id}}"]`);
                                    const hasAsterisk = label && label.textContent.includes('*');
                                    const hasAriaRequired = input.getAttribute('aria-required') === 'true';
                                    return !hasAsterisk && !hasAriaRequired;
                                }});
                                
                                if (missingIndicators.length > 0) {{
                                    formAnalysis.issues.push({{
                                        type: 'required_field_indicators',
                                        count: missingIndicators.length,
                                        severity: 'medium',
                                        description: 'Required fields without visual indicators'
                                    }});
                                }}
                                
                                // Check for submit buttons
                                const submitButtons = form.querySelectorAll('button[type="submit"], input[type="submit"]');
                                if (submitButtons.length === 0) {{
                                    formAnalysis.issues.push({{
                                        type: 'no_submit_button',
                                        severity: 'critical',
                                        description: 'Form has no submit button'
                                    }});
                                }}
                                
                                // Check for form validation attributes
                                const inputs = form.querySelectorAll('input, select, textarea');
                                const validationChecks = {{
                                    email_validation: 0,
                                    pattern_validation: 0,
                                    length_validation: 0
                                }};
                                
                                Array.from(inputs).forEach(input => {{
                                    if (input.type === 'email') validationChecks.email_validation++;
                                    if (input.hasAttribute('pattern')) validationChecks.pattern_validation++;
                                    if (input.hasAttribute('minlength') || input.hasAttribute('maxlength')) {{
                                        validationChecks.length_validation++;
                                    }}
                                }});
                                
                                formAnalysis.validation_attributes = validationChecks;
                                formAnalysis.total_inputs = inputs.length;
                                formAnalysis.total_required = requiredInputs.length;
                                
                                analysis.push(formAnalysis);
                            }});
                            
                            return {{
                                forms_found: forms.length,
                                analysis: analysis
                            }};
                        }})()
                        """
                    )
                    
                    # Test form interaction if forms are found
                    interaction_result = None
                    form_analysis_data = json.loads(form_analysis).get("result", {})
                    
                    if form_analysis_data.get("forms_found", 0) > 0:
                        # Try to interact with the first form
                        interaction_result = await mcp.call_tool(
                            "evaluate_script",
                            script=f"""
                            (() => {{
                                const form = document.querySelector('{form_selector}');
                                if (!form) return {{ error: 'Form not found' }};
                                
                                const inputs = form.querySelectorAll('input, select, textarea');
                                const interactions = [];
                                
                                Array.from(inputs).forEach((input, index) => {{
                                    const interaction = {{
                                        type: input.type || input.tagName.toLowerCase(),
                                        placeholder: input.placeholder || '',
                                        label: '',
                                        accessible: false
                                    }};
                                    
                                    // Check for associated label
                                    if (input.id) {{
                                        const label = document.querySelector(`label[for="${{input.id}}"]`);
                                        if (label) {{
                                            interaction.label = label.textContent.trim();
                                            interaction.accessible = true;
                                        }}
                                    }}
                                    
                                    // Check for aria-label
                                    if (input.getAttribute('aria-label')) {{
                                        interaction.accessible = true;
                                    }}
                                    
                                    interactions.push(interaction);
                                }});
                                
                                return {{
                                    form_action: form.action || 'No action specified',
                                    form_method: form.method || 'GET',
                                    inputs: interactions
                                }};
                            }})()
                            """
                        )
                    
                    return json.dumps({
                        "form_analysis": form_analysis,
                        "interaction_test": interaction_result
                    })
                    
            except Exception as e:
                logger.error(f"Form usability test error: {e}")
                return json.dumps({"error": str(e), "success": False})

        @agent.tool
        async def visual_regression_comparison(baseline_path: str, current_screenshot_path: str = None) -> str:
            """Compare current page state with baseline for visual regression testing."""
            try:
                async with await get_mcp_client() as mcp:
                    # Take current screenshot if not provided
                    if not current_screenshot_path:
                        screenshot_result = await mcp.call_tool(
                            "screenshot",
                            full_page=True,
                            format="png"
                        )
                        current_screenshot_path = "/tmp/current_screenshot.png"
                    
                    # Perform visual comparison
                    comparison_result = await mcp.call_tool(
                        "visual_compare",
                        baseline_path=baseline_path,
                        threshold=0.1,  # 10% difference threshold
                        max_diff_pixels=1000
                    )
                    
                    return json.dumps({
                        "comparison_result": comparison_result,
                        "baseline_path": baseline_path,
                        "current_path": current_screenshot_path
                    })
                    
            except Exception as e:
                logger.error(f"Visual regression comparison error: {e}")
                return json.dumps({"error": str(e), "success": False})

    async def run_validation(
        self, 
        target_url: str,
        validation_mode: ValidationMode = ValidationMode.COMPREHENSIVE,
        **kwargs
    ) -> UIUXValidatorOutput:
        """
        Run comprehensive UI/UX validation on the target URL.
        
        Args:
            target_url: URL to validate
            validation_mode: Type of validation to perform
            **kwargs: Additional configuration options
        """
        # Create dependencies
        deps = UIUXValidationDependencies(
            target_url=target_url,
            validation_mode=validation_mode,
            **kwargs
        )
        
        # Use default viewports if none provided
        if not deps.viewports_to_test:
            deps.viewports_to_test = self.default_viewports
        
        # Create validation prompt based on mode
        if validation_mode == ValidationMode.QUICK_SCAN:
            prompt = f"Perform a quick UI/UX validation scan of {target_url}. Focus on critical accessibility and visual issues."
        elif validation_mode == ValidationMode.COMPREHENSIVE:
            prompt = f"Perform comprehensive UI/UX validation of {target_url}. Include all validation categories with detailed analysis."
        elif validation_mode == ValidationMode.FOCUSED:
            component = kwargs.get('focus_component', 'main content area')
            prompt = f"Perform focused UI/UX validation of the {component} on {target_url}."
        else:  # CONTINUOUS
            prompt = f"Perform continuous monitoring validation of {target_url}. Focus on performance and critical functionality."
        
        # Run the agent
        return await self.run(prompt, deps)

    async def _parse_validation_results(self, raw_results: str) -> ValidationReport:
        """Parse and structure validation results into a comprehensive report."""
        try:
            # This would parse the raw JSON results from the tools
            # and create a structured ValidationReport
            # Implementation would depend on the exact format of tool outputs
            
            report = ValidationReport(
                url="",  # Would be populated from results
                validation_mode=ValidationMode.COMPREHENSIVE,
                timestamp=datetime.now(),
                summary={},
                issues=[],
                screenshots={},
                performance_metrics={},
                recommendations=[],
                browsers_tested=[],
                viewports_tested=[]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error parsing validation results: {e}")
            raise

    def _calculate_accessibility_score(self, accessibility_issues: List[ValidationIssue]) -> float:
        """Calculate accessibility score based on found issues."""
        if not accessibility_issues:
            return 100.0
        
        # Simple scoring: deduct points based on severity
        score = 100.0
        for issue in accessibility_issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 20.0
            elif issue.severity == ValidationSeverity.HIGH:
                score -= 10.0
            elif issue.severity == ValidationSeverity.MEDIUM:
                score -= 5.0
            elif issue.severity == ValidationSeverity.LOW:
                score -= 2.0
        
        return max(0.0, score)

    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations based on found issues."""
        recommendations = []
        
        # Group issues by category
        issue_categories = {}
        for issue in issues:
            if issue.category not in issue_categories:
                issue_categories[issue.category] = []
            issue_categories[issue.category].append(issue)
        
        # Generate category-specific recommendations
        for category, category_issues in issue_categories.items():
            critical_count = len([i for i in category_issues if i.severity == ValidationSeverity.CRITICAL])
            high_count = len([i for i in category_issues if i.severity == ValidationSeverity.HIGH])
            
            if critical_count > 0:
                recommendations.append(
                    f"🚨 URGENT: Address {critical_count} critical {category} issues immediately"
                )
            elif high_count > 0:
                recommendations.append(
                    f"⚠️ HIGH PRIORITY: Fix {high_count} high-priority {category} issues"
                )
        
        return recommendations


# Convenience function for quick validation
async def validate_ui_ux(
    target_url: str,
    mode: ValidationMode = ValidationMode.COMPREHENSIVE,
    output_dir: Optional[str] = None
) -> UIUXValidatorOutput:
    """
    Convenience function to perform UI/UX validation.
    
    Args:
        target_url: URL to validate
        mode: Validation mode to use
        output_dir: Directory to save screenshots and reports
    
    Returns:
        Comprehensive validation results
    """
    agent = UIUXValidatorAgent()
    
    return await agent.run_validation(
        target_url=target_url,
        validation_mode=mode,
        output_directory=output_dir
    )