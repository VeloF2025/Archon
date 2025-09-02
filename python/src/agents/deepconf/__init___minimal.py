"""
Minimal lazy loading test
"""

import threading

__version__ = "1.0.0"

# Lazy loading registry
_lazy_components = {}
_lazy_lock = threading.RLock()

class LazyDeepConfEngine:
    """Truly lazy loading wrapper"""
    
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        self._actual_instance = None
    
    def calculate_confidence(self, task, context):
        """Lazy load only when this method is called"""
        if self._actual_instance is None:
            with _lazy_lock:
                if self._actual_instance is None:
                    # Import happens here, not at module level
                    from .engine import DeepConfEngine as ActualEngine
                    self._actual_instance = ActualEngine(*self._init_args, **self._init_kwargs)
        return self._actual_instance.calculate_confidence(task, context)

DeepConfEngine = LazyDeepConfEngine

__all__ = ["DeepConfEngine"]