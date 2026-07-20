"""Re-export ``app`` so ADK's loader finds it on the package itself."""

from . import agent
from .agent import app, root_agent

__all__ = ["agent", "app", "root_agent"]
