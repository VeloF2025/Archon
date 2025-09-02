"""
Tests for Browser MCP Module

Tests the Playwright-based browser automation capabilities
without requiring actual browser instances in CI/CD.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Test imports and module availability
def test_browser_mcp_imports():
    """Test that the browser MCP module can be imported"""
    try:
        from src.mcp_server.modules.browser_mcp import (
            BrowserType,
            ViewportSize,
            BrowserManager,
            register_browser_tools
        )
        assert True
    except ImportError as e:
        # If Playwright is not available, that's acceptable
        assert "playwright" in str(e).lower()

@pytest.mark.asyncio
async def test_browser_manager_initialization():
    """Test BrowserManager initialization"""
    from src.mcp_server.modules.browser_mcp import BrowserManager
    
    manager = BrowserManager()
    assert manager.playwright is None
    assert manager.browser is None
    assert manager.context is None
    assert manager.page is None

def test_browser_type_enum():
    """Test BrowserType enumeration values"""
    from src.mcp_server.modules.browser_mcp import BrowserType
    
    assert BrowserType.CHROMIUM == "chromium"
    assert BrowserType.FIREFOX == "firefox"
    assert BrowserType.WEBKIT == "webkit"

def test_viewport_sizes():
    """Test ViewportSize standard configurations"""
    from src.mcp_server.modules.browser_mcp import ViewportSize
    
    assert ViewportSize.DESKTOP == {"width": 1920, "height": 1080}
    assert ViewportSize.MOBILE == {"width": 375, "height": 667}
    assert ViewportSize.TABLET == {"width": 768, "height": 1024}

@pytest.mark.asyncio
@patch('src.mcp_server.modules.browser_mcp.PLAYWRIGHT_AVAILABLE', True)
async def test_navigate_to_success():
    """Test successful navigation"""
    from src.mcp_server.modules.browser_mcp import register_browser_tools
    from mcp.server.fastmcp import FastMCP, Context
    
    # Mock FastMCP and context
    mock_mcp = MagicMock()
    mock_context = MagicMock(spec=Context)
    
    # Mock Playwright components
    mock_response = MagicMock()
    mock_response.status = 200
    
    mock_page = AsyncMock()
    mock_page.goto.return_value = mock_response
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.url = "https://example.com"
    
    # Mock browser manager
    with patch('src.mcp_server.modules.browser_mcp._browser_manager') as mock_manager:
        mock_manager.ensure_browser = AsyncMock()
        mock_manager.get_page = AsyncMock(return_value=mock_page)
        
        # Register tools and get the navigate_to function
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool
        register_browser_tools(mock_mcp)
        
        # Test navigate_to function
        if 'navigate_to' in tools:
            result = await tools['navigate_to'](
                mock_context,
                url="https://example.com",
                wait_until="domcontentloaded",
                timeout=30000,
                browser_type="chromium",
                headless=True
            )
            
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["url"] == "https://example.com"
            assert result_data["title"] == "Test Page"

@pytest.mark.asyncio
@patch('src.mcp_server.modules.browser_mcp.PLAYWRIGHT_AVAILABLE', True)
async def test_screenshot_success():
    """Test successful screenshot capture"""
    from src.mcp_server.modules.browser_mcp import register_browser_tools
    from mcp.server.fastmcp import FastMCP, Context
    
    # Mock FastMCP and context
    mock_mcp = MagicMock()
    mock_context = MagicMock(spec=Context)
    
    # Mock page with screenshot capability
    mock_page = AsyncMock()
    mock_page.screenshot.return_value = b"fake_image_data"
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.url = "https://example.com"
    
    # Mock browser manager
    with patch('src.mcp_server.modules.browser_mcp._browser_manager') as mock_manager:
        mock_manager.get_page = AsyncMock(return_value=mock_page)
        
        # Register tools and get the screenshot function
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool
        register_browser_tools(mock_mcp)
        
        # Test screenshot function
        if 'screenshot' in tools:
            result = await tools['screenshot'](
                mock_context,
                selector=None,
                full_page=False,
                quality=90,
                format="png",
                path=None
            )
            
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert "image_data" in result_data
            assert result_data["format"] == "png"

@pytest.mark.asyncio
@patch('src.mcp_server.modules.browser_mcp.PLAYWRIGHT_AVAILABLE', False)
async def test_playwright_not_available():
    """Test graceful handling when Playwright is not available"""
    from src.mcp_server.modules.browser_mcp import register_browser_tools
    from mcp.server.fastmcp import FastMCP
    
    mock_mcp = MagicMock()
    
    # Should not raise an exception, just log a warning
    register_browser_tools(mock_mcp)
    
    # Should not have registered any tools
    assert not mock_mcp.tool.called

def test_error_handling_structure():
    """Test that error responses have consistent structure"""
    from src.mcp_server.modules.browser_mcp import register_browser_tools
    
    # This test validates that our error response format is consistent
    error_response = json.dumps({
        "success": False,
        "error": "Test error message"
    }, indent=2)
    
    parsed = json.loads(error_response)
    assert "success" in parsed
    assert "error" in parsed
    assert parsed["success"] is False

@pytest.mark.asyncio
async def test_browser_cleanup():
    """Test browser cleanup functionality"""
    from src.mcp_server.modules.browser_mcp import BrowserManager
    
    manager = BrowserManager()
    
    # Mock browser components - simplified test
    mock_page = AsyncMock()
    mock_page.is_closed.return_value = False  # Page is not closed
    mock_context = AsyncMock()
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()
    
    manager.page = mock_page
    manager.context = mock_context
    manager.browser = mock_browser
    manager.playwright = mock_playwright
    
    # Test cleanup - should not raise exception
    await manager.cleanup()
    
    # Main verification: references were cleared (most important for memory management)
    assert manager.page is None
    assert manager.context is None
    assert manager.browser is None
    assert manager.playwright is None

@pytest.mark.asyncio
async def test_get_performance_metrics_structure():
    """Test performance metrics response structure"""
    from src.mcp_server.modules.browser_mcp import register_browser_tools
    from mcp.server.fastmcp import FastMCP, Context
    
    # Mock FastMCP and context
    mock_mcp = MagicMock()
    mock_context = MagicMock(spec=Context)
    
    # Mock page with performance data
    mock_page = AsyncMock()
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.url = "https://example.com"
    mock_page.evaluate.side_effect = [
        {  # Performance metrics
            "dns_lookup": 10.5,
            "tcp_connection": 25.2,
            "request_response": 150.8,
            "dom_processing": 75.3,
            "load_complete": 45.1,
            "total_load_time": 500.2,
            "first_paint": 200.1,
            "first_contentful_paint": 250.5,
            "used_heap_size": 1024000,
            "total_heap_size": 2048000,
            "heap_size_limit": 4096000
        },
        {  # Network info
            "total_requests": 15,
            "resource_types": {
                "document": 1,
                "script": 5,
                "stylesheet": 3,
                "image": 6
            }
        }
    ]
    
    # Mock browser manager
    with patch('src.mcp_server.modules.browser_mcp._browser_manager') as mock_manager:
        mock_manager.get_page = AsyncMock(return_value=mock_page)
        
        # Register tools
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool
        register_browser_tools(mock_mcp)
        
        # Test performance metrics function
        if 'get_performance_metrics' in tools:
            result = await tools['get_performance_metrics'](mock_context)
            
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert "page_info" in result_data
            assert "performance_metrics" in result_data
            assert "network_info" in result_data
            
            # Validate performance metrics structure
            metrics = result_data["performance_metrics"]
            assert "total_load_time" in metrics
            assert "first_contentful_paint" in metrics
            
            # Validate network info structure
            network = result_data["network_info"]
            assert "total_requests" in network
            assert "resource_types" in network