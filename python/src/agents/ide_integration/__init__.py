"""
IDE Integration Module
Provides comprehensive IDE plugin framework and integration capabilities
"""

from .ide_plugin_manager import IDEPluginManager
from .vscode_plugin import VSCodePlugin
from .intellij_plugin import IntelliJPlugin
from .vim_plugin import VimPlugin
from .emacs_plugin import EmacsPlugin
from .sublime_plugin import SublimeTextPlugin
from .atom_plugin import AtomPlugin
from .plugin_base import PluginBase, PluginCapability
from .ide_communication import IDECommunicationProtocol
from .code_completion_engine import CodeCompletionEngine
from .real_time_sync import RealTimeSyncEngine
from .plugin_installer import PluginInstaller

__all__ = [
    "IDEPluginManager",
    "VSCodePlugin", 
    "IntelliJPlugin",
    "VimPlugin",
    "EmacsPlugin", 
    "SublimeTextPlugin",
    "AtomPlugin",
    "PluginBase",
    "PluginCapability",
    "IDECommunicationProtocol",
    "CodeCompletionEngine",
    "RealTimeSyncEngine",
    "PluginInstaller"
]